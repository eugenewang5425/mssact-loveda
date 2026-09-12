"""
Kappa 归属分解 (headroom_analysis.py) —— 定量回答
"为什么模块差异在聚合指标上消失?"

方法 (纯数学, 基于同一批缓存预测)
--------------------------------
对模型 M 和标签 L, 在 test_clean 上构造两个"理想化"预测:

  oracle-boundary : 在边界像素 (距边界 <0.5px, 占 2.4%) 上作弊为真值, 其余保留 M
  oracle-interior : 在内部像素 (距边界 >8px,  占 78.4%) 上作弊为真值, 其余保留 M

比较三者 Kappa, 即可得出指标"被哪一部分像素支配":
  ΔKappa(oracle-boundary) = 完美解决边界所能获得的上限
  ΔKappa(oracle-interior) = 完美解决内部所能获得的上限

此外做 ΔKappa 的分带归因: mine vs DeepLab 的差距分布在哪些带上。

用法: python headroom_analysis.py
"""
import os, sys, json
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

CONS = paths.CONSENSUS
IGNORE, N_CLS = 255, 7
BANDS = [(0, 0.5, "0 (边界)"), (0.5, 2.5, "1-2"), (2.5, 4.5, "3-4"),
         (4.5, 8.5, "5-8"), (8.5, 1e9, ">8 (内部)")]
MODELS = {"full_all_v2": "MSSACT-light 6.1M", "nd_deeplab": "DeepLabV3+ 39.6M",
          "nd_fpn_seg": "FPN 27.2M", "nd_fcn": "FCN 23.7M",
          "nd_unet": "U-Net 2.4M", "nd_swin_unet": "Swin-Unet 32.6M"}


def edge_mask(lab2d):
    valid = lab2d != IGNORE
    e = np.zeros_like(valid, dtype=bool)
    for ax in (0, 1):
        a = np.take(lab2d, range(0, lab2d.shape[ax] - 1), axis=ax)
        b = np.take(lab2d, range(1, lab2d.shape[ax]), axis=ax)
        d = (a != b) & (a != IGNORE) & (b != IGNORE)
        s = [slice(None)] * 2; s[ax] = slice(0, -1); e[tuple(s)] |= d
        s2 = [slice(None)] * 2; s2[ax] = slice(1, None); e[tuple(s2)] |= d
    return e & valid, valid


def kappa_of(pred, lab, mask):
    p, l = pred[mask].astype(np.int64), lab[mask].astype(np.int64)
    n = p.size
    if n == 0:
        return np.nan
    cm = np.bincount(l * N_CLS + p, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main():
    z = np.load(os.path.join(CONS, "labels.npz"), allow_pickle=True)
    labels = z["labels"]
    n = labels.shape[0]

    # ---- 预加载全部预测 ----
    P = {}
    for m in MODELS:
        p = os.path.join(CONS, f"pred_{m}.npz")
        if os.path.exists(p):
            P[m] = np.load(p)["pred"]
    print(f"载入 {len(P)} 个模型的预测", flush=True)

    band_cm = {m: {nm: np.zeros((N_CLS, N_CLS), np.int64) for _, _, nm in BANDS}
               for m in P}
    orc = {(m, v): np.zeros((N_CLS, N_CLS), np.int64)
           for m in P for v in ("base", "oracle_boundary", "oracle_interior")}
    band_n = {nm: 0 for _, _, nm in BANDS}

    for i in range(n):
        lab = labels[i]
        e, valid = edge_mask(lab)
        dist = ndimage.distance_transform_edt(~e)
        for lo, hi, nm in BANDS:
            mm = valid & (dist >= lo) & (dist < hi)
            band_n[nm] += int(mm.sum())
        for m, pr in P.items():
            pred = pr[i]
            for lo, hi, nm in BANDS:
                mm = valid & (dist >= lo) & (dist < hi)
                if mm.any():
                    p_, l_ = pred[mm].astype(np.int64), lab[mm].astype(np.int64)
                    band_cm[m][nm] += np.bincount(l_ * N_CLS + p_,
                                                  minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
            # oracle 变体 (全图)
            for var, sel in (("base", np.zeros_like(valid)),
                             ("oracle_boundary", valid & (dist < 0.5)),
                             ("oracle_interior", valid & (dist > 8.0))):
                p2 = np.where(sel, lab, pred)
                mm = valid
                p_, l_ = p2[mm].astype(np.int64), lab[mm].astype(np.int64)
                orc[(m, var)] += np.bincount(l_ * N_CLS + p_,
                                             minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)

    def kap(cm):
        n_ = cm.sum()
        po = np.trace(cm) / n_
        pe = (cm.sum(0) * cm.sum(1)).sum() / (n_ * n_)
        return (po - pe) / (1 - pe) if pe < 1 else np.nan

    tot = sum(band_n.values())
    print(f"\n=== 1. 边界/内部 的 Kappa 归属上限 (test_clean {n} 张, 有效像素 {tot:,}) ===")
    print(f"{'模型':22s} {'K_base':>8s} {'K_边界作弊':>11s} {'Δ边界':>9s} "
          f"{'K_内部作弊':>11s} {'Δ内部':>9s}")
    rows = {}
    for m, name in MODELS.items():
        if m not in P:
            continue
        kb = kap(orc[(m, "base")])
        kbd = kap(orc[(m, "oracle_boundary")])
        kin = kap(orc[(m, "oracle_interior")])
        rows[m] = dict(kappa_base=round(kb, 4),
                       kappa_oracle_boundary=round(kbd, 4), d_boundary=round(kbd - kb, 4),
                       kappa_oracle_interior=round(kin, 4), d_interior=round(kin - kb, 4))
        print(f"{name:22s} {kb:8.4f} {kbd:11.4f} {kbd-kb:+9.4f} {kin:11.4f} {kin-kb:+9.4f}")

    print(f"\n=== 2. ΔKappa 的分带归因 (MSSACT-light vs DeepLabV3+) ===")
    a, b = "full_all_v2", "nd_deeplab"
    if a in P and b in P:
        ka = sum(kap(band_cm[a][nm]) * 0 for _, _, nm in BANDS)   # 占位, 下面按带算 OA 差
        print(f"{'band':>10s} {'占比':>7s} {'MSSACT OA':>10s} {'DeepLab OA':>11s} {'ΔOA':>8s} {'ΔOA×占比':>9s}")
        tot_contrib = 0.0
        for _, _, nm in BANDS:
            cma, cmb = band_cm[a][nm], band_cm[b][nm]
            na = cma.sum()
            oa_a = np.trace(cma) / na if na else np.nan
            oa_b = np.trace(cmb) / cmb.sum() if cmb.sum() else np.nan
            f = band_n[nm] / tot
            contrib = (oa_b - oa_a) * f
            tot_contrib += contrib
            print(f"{nm:>10s} {f:7.4f} {oa_a:10.4f} {oa_b:11.4f} {oa_b-oa_a:+8.4f} {contrib:+9.4f}")
        print(f"{'合计':>10s} {'1.0000':>7s} {'':>10s} {'':>11s} {'':>8s} {tot_contrib:+9.4f}")

    print(f"\n=== 3. 逐带 Kappa (每带单独计算, 看模块差异在哪个带最大) ===")
    print(f"{'band':>10s} " + " ".join(f"{MODELS[m][:11]:>12s}" for m in MODELS if m in P))
    for _, _, nm in BANDS:
        cells = " ".join(f"{kap(band_cm[m][nm]):12.4f}" for m in MODELS if m in P)
        print(f"{nm:>10s} {cells}")
    print("\n说明: 若各带内模型间差异同样微小 -> 模块在**所有尺度**上都无贡献;")
    print("      若仅边界带差异大 -> 聚合指标因边界仅占 2.4% 而系统性掩盖了模块价值。")

    json.dump(dict(n_images=n, n_valid_pixels=int(tot),
                   band_frac={nm: round(band_n[nm] / tot, 4) for _, _, nm in BANDS},
                   oracle_kappa=rows),
              open(os.path.join(CONS, "headroom.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n写入 headroom.json")


if __name__ == "__main__":
    main()
