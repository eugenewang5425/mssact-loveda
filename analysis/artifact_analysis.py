"""
解码器伪影分析 (artifact_analysis.py)

动机
----
§5.6.3 指出解码器无跳连、仅凭 32² 特征上采样 8×。但"缺少细节"是**错误**的预期：
实测各模型预测的边缘密度**都高于**标签，MSSACT-Net 更达 **4.56×**。

本脚本量化两件事（逐图 2D 计算, 严禁跨图求差）：
  1. 边缘密度 (相邻有效像素对中类别不同的占比): 标签 vs 各模型预测
  2. 周期性伪影: 边缘密度按列索引 mod k 分组的起伏 (max/min)
     -> 堆叠的 stride-2 转置卷积会产生棋盘伪影 (Odena et al. 2016,
        "Deconvolution and Checkerboard Artifacts"), 其周期 = 总上采样倍数

判定依据
  标签几乎无周期性 (≈1.00)。若某模型在周期 k 上起伏显著, 且 k 等于其解码器
  总上采样倍数, 则该起伏可归因于转置卷积棋盘伪影, 而非真实地物边界。

用法: python artifact_analysis.py
"""

# --- 路径引导（本文件位于子目录 analysis/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os, sys, json
import numpy as np

sys.path.insert(0, _HERE)
import paths

CONS = paths.CONSENSUS
MODELS = {"full_all_v2": "MSSACT-light 6.1M", "nd_deeplab": "DeepLabV3+ 39.7M",
          "nd_fpn_seg": "FPN 27.2M", "nd_fcn": "FCN 23.8M",
          "nd_unet": "U-Net 2.4M", "nd_swin_unet": "Swin-Unet 32.7M",
          "nd_abl_bilinear": "MSSACT 双线性上采样"}
# 各模型解码器总上采样倍数 (用于判定伪影周期是否与上采样倍数吻合)
UPSAMPLE = {"full_all_v2": 8, "nd_abl_bilinear": 8, "nd_fpn_seg": 8,
            "nd_fcn": 32, "nd_unet": 16, "nd_deeplab": 4, "nd_swin_unet": 16}
PERIODS = (2, 4, 8, 16)


def edges_2d(A):
    """返回 (差异对数, 有效对总数) —— 逐图 2D, 不跨图"""
    v = A != 255
    tot = dif = 0
    for ax in (0, 1):
        a = np.take(A, range(0, A.shape[ax] - 1), axis=ax)
        b = np.take(A, range(1, A.shape[ax]), axis=ax)
        m = np.take(v, range(0, v.shape[ax] - 1), axis=ax) & \
            np.take(v, range(1, v.shape[ax]), axis=ax)
        dif += int(((a != b) & m).sum())
        tot += int(m.sum())
    return dif, tot


def period_power_ratio(A, P_OF_INTEREST=(4, 8, 16, 32), half_band=3):
    """沿列方向对水平边缘图做 FFT, 返回指定周期附近功率占总功率的百分比

    为何用 FFT 功率占比, 而不用"按列 mod k 分组后 max/min 或 CV":
      两者都是**分组统计量**, 分组数 k 本身会改变统计量的期望 ——
      max/min 随 k 单调上升 (顺序统计量偏差); CV 随 k 上升 (每组样本数 n/k 减少,
      均值抽样方差 ~k/n 增大)。实测这两者在**标签**上也都随 k 单调漂移,
      说明漂移来自方法而非数据, 无法据此判定伪影。

      FFT 功率占比不受此影响: 它直接度量"某周期成分占总能量多少",
      对**同一 k** 比较模型与标签即可, 无需跨 k 比较。
      判据: 标签作为参照; 若某模型在某个 k 上显著高于标签基线,
            且该 k 等于其解码器总上采样倍数, 则可归因于转置卷积棋盘伪影。
    """
    v = A != 255
    rows = v.all(1)                        # 只用整行无 no-data 的行
    if int(rows.sum()) < 20:
        return None
    E = (A[rows][:, 1:] != A[rows][:, :-1]).astype(np.float64)
    if E.sum() < 500:
        return None
    E = E - E.mean(1, keepdims=True)       # 去直流
    P = (np.abs(np.fft.rfft(E, axis=1)) ** 2).mean(0)
    W = E.shape[1]
    f = np.fft.rfftfreq(W, d=1.0)
    tot = P[1:].sum()
    if tot <= 0:
        return None
    out = {}
    for k in P_OF_INTEREST:
        i = int(np.argmin(np.abs(f - 1.0 / k)))
        lo, hi = max(1, i - half_band), i + half_band + 1
        out[k] = float(P[lo:hi].sum() / tot * 100.0)
    return out


def main():
    z = np.load(os.path.join(CONS, "labels.npz"), allow_pickle=True)
    L = z["labels"]

    # ---- 1. 标签边缘密度 (参照) ----
    ld = lt = 0
    for i in range(L.shape[0]):
        d, t = edges_2d(L[i]); ld += d; lt += t
    lab_ed = ld / lt
    lab_pows = [p for p in (period_power_ratio(L[i], PERIODS) for i in range(L.shape[0])) if p is not None]
    lab_period = {k: float(np.mean([p[k] for p in lab_pows])) for k in PERIODS}

    print(f"标签 边缘密度 = {lab_ed:.5f}")
    print("  周期起伏: " + "  ".join(
        f"k={k}:{lab_period[k]:.3f}" if lab_period[k] is not None else f"k={k}:-" for k in PERIODS))
    print()

    out = {"label_edge_density": round(lab_ed, 6),
           "label_period_power_pct": {str(k): round(v, 4) for k, v in lab_period.items()},
           "period_metric": "FFT 功率占比(%%) 在周期 k 附近 ±3 bin; 仅用整行无 no-data 的行; "
                            "报告的是 模型/标签 的超额比",
           "models": {}}

    print(f"{'模型':22s} {'边缘密度':>9s} {'/标签':>7s} {'超额比 k=2/4/8/16/32':>26s} {'上采样':>7s}")
    for tag, name in MODELS.items():
        p = os.path.join(CONS, f"pred_{tag}.npz")
        if not os.path.exists(p):
            print(f"{name:22s}   (无缓存, 跳过)")
            continue
        P = np.load(p)["pred"]
        pd_ = pt = 0
        for i in range(P.shape[0]):
            d, t = edges_2d(P[i]); pd_ += d; pt += t
        ed = pd_ / pt
        pows = [p for p in (period_power_ratio(P[i], PERIODS) for i in range(P.shape[0])) if p is not None]
        amps = {k: float(np.mean([p[k] for p in pows])) for k in PERIODS}
        ups = UPSAMPLE.get(tag)
        # 超额比 = 模型该周期功率占比 / 标签同周期功率占比
        exc = {k: amps[k] / lab_period[k] for k in PERIODS if lab_period.get(k)}
        peak = max(exc, key=exc.get) if exc else None
        out["models"][tag] = {
            "name": name, "edge_density": round(ed, 6),
            "edge_ratio_vs_label": round(ed / lab_ed, 3),
            "period_power_pct": {str(k): round(v, 4) for k, v in amps.items()},
            "excess_vs_label": {str(k): round(exc[k], 3) for k in PERIODS},
            "peak_excess_period": peak, "decoder_upsample": ups,
            "peak_matches_upsample": (peak == ups),
        }
        flag = f"{ups}x" + ("  <- 吻合" if peak == ups else "")
        cells = "/".join(f"{exc[k]:.2f}" for k in PERIODS)
        print(f"{name:22s} {ed:9.5f} {ed/lab_ed:7.3f}   {cells:26s} {flag:>9s}")

    dst = os.path.join(CONS, "artifact.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {os.path.basename(dst)}")
    print("\n判读: 标签周期起伏≈1.00 (无周期性)。若某模型的起伏峰值出现在")
    print("      周期 = 其解码器总上采样倍数处, 则该起伏可归因于转置卷积棋盘伪影。")


if __name__ == "__main__":
    main()
