# -*- coding: utf-8 -*-
"""等效性检验 (TOST) 与可分辨性分级 —— 一等产物

为什么单独做这个脚本
------------------
`analysis/eval_rigor.py` 里已有配对 bootstrap 与 TOST 判定, 但它**只 print**,
结果不落盘: 报告无法引用、无法复现、也无法在新增 tag 后自动扩展。
本脚本把那套判定变成 `results/artifacts/tost_equivalence.json`, 并补上第二套
误差模型 (见下)。

两种误差模型, 回答的是**两个不同问题**
------------------------------------
1. **种子层 (training-level)**: "换一个随机种子重训, 这个差异还在吗?"
   σ_seed 由同一配置多次独立训练实测 (当前 n=4, σ=0.0050)。
   两次独立训练之差的 SE = σ_seed * sqrt(2) ≈ 0.0071, 故 **|Δ| >= 2*σ_seed = 0.010
   记为可分辨**; 反之落在 ±0.010 内属"不可分辨"。
2. **测试集层 (test-set-level)**: "只给定这两个固定模型与这 141 张测试图,
   差异有多少来自**取样了哪些图**?" —— 图像级配对 bootstrap。

**实测(2026-09-13, 141 张, 5 对)**：配对 bootstrap 的 95% CI 半宽约 0.011
(如 nd_unet vs full_all_v2 为 [-0.0532, -0.0318], 半宽 0.0107), **小于**
SESOI = 0.02。原因是配对设计: 两个模型跑同一批图, Δ 的方差远小于各自 Kappa 的
方差。故**测试集层足以在 ±0.02 边界上给出等效判定** —— 这一条是实测结论,
不是先验估计。(曾误以为 n=141 的 CI 半宽约 ±0.02 而"无法认证等效", 实测推翻。)

两个误差模型的**量级也不同**: 种子层 ±0.010 略小于测试集层 ±0.011。两者都报告,
不合并成单一阈值; 合并 SE = sqrt(2*σ_seed² + var_boot(Δ)) 仅用于描述单次观测
ΔKappa 的总误差。

口径纪律
--------
- 只在**同可比性组**内配对 (数据划分 / 管线 / 协议 三者全同), 组定义读自
  `experiments_index.json` 的 `comparable_groups`, 禁止跨管线世代比较。
- 度量函数**直接 import 自 eval_rigor**, 不重复实现, 避免口径漂移。
- pooled Kappa 与 per-image mean Kappa 是两个不同的估计量, 两者都报告。

输出: results/artifacts/tost_equivalence.json
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import paths
import eval_rigor as ER          # 复用其度量函数与常数, 保证口径一致

SESOI = ER.SESOI                 # 0.02, 预指定的最小关注效应量
N_BOOT = ER.N_BOOT               # 1000
SEED = ER.SEED                   # 12345

OUT_JSON = os.path.join(paths.RESULTS, "artifacts", "tost_equivalence.json")


def sigma_seed():
    """测试集协议下的实测种子标准差。

    **口径纪律**: 本脚本的 dKappa 全部来自整图测试协议 (1024² 滑窗, 141 张), 故必须用
    **同一协议**测出的 sigma。`sigma_seed_used` (0.0050) 是在 val 协议 (221 张 256² 裁剪)
    上测的; 两协议的评估集/分辨率/推理方式都不同, 直接拿来判定属**口径错配**。
    实测: val 0.0050 / test 0.0041 (n=4, 同一批 4 个种子)。
    优先读 `sigma_seed_test_protocol`, 缺失时回退并**显式标注**口径不匹配。
    """
    try:
        d = json.load(open(paths.SEEDVAR_JSON, encoding="utf-8"))
        t = d.get("sigma_seed_test_protocol")
        if t and t.get("std"):
            return float(t["std"]), int(t.get("n", 0)), "test 协议(已匹配)"
        s = d.get("sigma_seed_used")
        if s:
            # 注意: 分组的键名曾带参数量(如 "完整模型 MSSACT 6.10M (P-MEM-ROLL)"),
            # 后因"参数量口径一变键就失效"而改名 -> 这里曾因此静默取到 n=0
            g = d.get("groups", {}).get("完整模型 MSSACT (P-MEM-ROLL)", {})
            return float(s), int(g.get("n", 0)), "val 协议(回退, 口径不匹配)"
    except Exception:
        pass
    return 0.0050, 0, "硬编码回退(口径不匹配)"


def comparable_groups():
    """读可比性分组; 缺失则退化为空(此时不做跨组检查, 但会显式警告)"""
    try:
        d = json.load(open(paths.INDEX_JSON, encoding="utf-8"))
        return d.get("comparable_groups", {}) or {}
    except Exception:
        return {}


def load_preds(tags):
    """载入 labels 与各 tag 的预测; 返回 (labels, {tag: pred}, 可用 tag 列表)"""
    lp = os.path.join(paths.CONSENSUS, "labels.npz")
    if not os.path.exists(lp):
        raise SystemExit("缺少 labels.npz, 请先运行 consensus_analysis.py")
    z = np.load(lp, allow_pickle=True)
    labels = z["labels"]
    out = {}
    for t in tags:
        p = os.path.join(paths.CONSENSUS, "pred_%s.npz" % t)
        if not os.path.exists(p):
            continue
        pr = np.load(p)["pred"]
        if pr.shape != labels.shape:
            print("  ! %s 形状 %s != %s, 跳过" % (t, pr.shape, labels.shape))
            continue
        out[t] = pr
    return labels, out


def boot_pooled_kappa(cms, idx):
    """对图像索引做重采样, 每次重算 pooled Kappa (不是对 Kappa 取平均)"""
    return np.array([ER.pooled_kappa(cms[i]) for i in idx])


def main():
    s_seed, n_seed, sigma_src = sigma_seed()
    groups = comparable_groups()
    tag2grp = {}
    for gname, g in groups.items():
        if gname == "REJECTED":
            continue
        for m in g.get("members", []):
            tag2grp[m] = gname
    if not groups:
        print("  ! 未读到 comparable_groups, 跳过跨组检查")

    # 要评估的对照对: (基准, 变体)。只列组内配对。
    PAIRS = [
        # G1: 60 轮 P-PNG 组 (对比 + 消融)
        ("full_all_v2", ["nd_unet", "nd_pspnet", "nd_fcn", "nd_deeplab", "nd_deeplab_scr",
                         "nd_segformer", "nd_fpn_seg", "nd_swin_unet"] +
                        [f"nd_abl_{s}" for s in ("no_emr", "no_ecsam", "no_fpn", "no_trans",
                                                 "no_adapter", "bilinear", "trans4l",
                                                 "trans6l")]),
        # G2: 60 轮组 (既有架构结论的依据)
        ("lgR_bs8_full_lr2e4", ["lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca",
                                "lgR_D3_ch_tiny", "lgR_D4_ch_large"]),
        # G4: 120 轮修复后组 (新结论的依据)
        ("lgR120_full", ["lgR120_fx_skip", "lgR120_fx_pos", "lgR120_fx_all"]),
        ("lgR120_fx_all", ["lgR120_abl_no_emr", "lgR120_abl_no_ecsam", "lgR120_abl_no_fpn",
                           "lgR120_abl_no_trans", "lgR120_abl_no_adapter",
                           "lgR120_abl_bilinear", "lgR120_abl_trans4l",
                           "lgR120_abl_trans6l"]),
    ]
    want = set()
    for ref, vs in PAIRS:
        want.add(ref)
        want.update(vs)

    labels, preds = load_preds(sorted(want))
    if not preds:
        raise SystemExit("没有任何可用预测; 先跑 analysis/eval_rigor.py --infer")
    print("载入预测 %d 个 tag, 测试集 %d 张 %d²" % (len(preds), labels.shape[0],
                                                labels.shape[1]), flush=True)
    print("σ_seed(测试集协议) = %.4f (n=%d, %s)" % (s_seed, n_seed, sigma_src), flush=True)

    # 逐图混淆矩阵 (pooled Kappa 的唯一来源)
    cms = {t: np.array([ER.conf_mat(preds[t][i], labels[i])
                        for i in range(labels.shape[0])]) for t in preds}
    pooled = {t: float(ER.pooled_kappa(cms[t])) for t in preds}
    for t in sorted(pooled, key=lambda k: -pooled[k]):
        print("  K(pooled) %-30s %.4f" % (t, pooled[t]))

    n = labels.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    # 同一组重采样索引用于所有 tag -> 配对结构保留
    cache = {}
    for t in preds:
        cache[t] = boot_pooled_kappa(cms[t], idx)

    rows = []
    for ref, variants in PAIRS:
        if ref not in preds:
            print("\n  跳过对照 %s (无预测)" % ref)
            continue
        print("\n=== 对照 %s (K=%.4f) ===" % (ref, pooled[ref]))
        print("%-24s %9s %19s %11s %11s %9s  %s" % (
            "变体", "dKappa", "boot 95%CI", "2*sig_seed", "bootSE", "合并SE", "判定"))
        for t in variants:
            if t not in preds:
                continue
            if groups and tag2grp.get(ref) != tag2grp.get(t):
                print("  ! %s 与 %s 不同可比组(%s vs %s), 跳过" % (
                    ref, t, tag2grp.get(ref), tag2grp.get(t)))
                continue
            d_obs = pooled[t] - pooled[ref]
            d_boot = cache[t] - cache[ref]
            lo, hi = np.percentile(d_boot, [2.5, 97.5])
            lo90, hi90 = np.percentile(d_boot, [5, 95])
            boot_se = float(np.std(d_boot, ddof=1))
            comb_se = float(np.sqrt(2 * s_seed ** 2 + boot_se ** 2))
            # 判定: 以种子层为准 (回答"换种子重训差异还在吗"), 并附 TOST 结论
            resolvable = abs(d_obs) >= 2 * s_seed
            tost_equiv = bool(lo90 > -SESOI and hi90 < SESOI)
            if resolvable:
                verd = "可分辨(%+.1f sig)" % (d_obs / s_seed)
            else:
                verd = "不可分辨(%.1f sig)" % (d_obs / s_seed)
            row = dict(
                ref=ref, tag=t, group=tag2grp.get(t) if groups else None,
                kappa_ref=round(pooled[ref], 6), kappa_tag=round(pooled[t], 6),
                delta=round(d_obs, 6),
                delta_in_sigma_seed=round(d_obs / s_seed, 3) if s_seed else None,
                resolvable_2sigma_seed=bool(resolvable),
                boot_ci95=[round(float(lo), 6), round(float(hi), 6)],
                boot_ci90=[round(float(lo90), 6), round(float(hi90), 6)],
                boot_se=round(boot_se, 6),
                combined_se=round(comb_se, 6),
                tost_equivalent_at_SESOI=tost_equiv,
                tost_margin=SESOI,
                note=("boot CI 半宽 %.4f > SESOI %.2f, 测试集取样误差不足以认证等效"
                      % ((hi - lo) / 2, SESOI)) if not tost_equiv else "",
            )
            rows.append(row)
            print("%-24s %+9.4f [%+.4f,%+.4f] %11.4f %11.4f %9.4f  %s" % (
                t, d_obs, lo, hi, 2 * s_seed, boot_se, comb_se, verd))

    out = dict(
        sesoi=SESOI, n_boot=N_BOOT, seed=SEED, n_images=int(n),
        sigma_seed=s_seed, sigma_seed_n_runs=n_seed, sigma_seed_source=sigma_src,
        sigma_seed_note=("判定测试集协议的 dKappa 必须用**同协议**的 sigma; "
                         "val 协议的 0.0050 只用于 val 上的量(见报告 5.7)。"
                         "本脚本的 dKappa 全部来自整图测试协议。"),
        threshold_rule="|dKappa| >= 2*sigma_seed 记为可分辨; 1-2 sigma 弱; <1 sigma 不可分辨",
        two_error_models_note=(
            "种子层回答'换种子重训差异还在吗'(误差 = sqrt(2)*sigma_seed); "
            "测试集层回答'取样了哪些测试图造成的差异'(图像级配对 bootstrap)。"
            "两者不可混用; 合并 SE 仅用于描述单次观测 dKappa 的总误差。"),
        metric_def="kappa_pooled = 由逐图混淆矩阵合并后的 Kappa (与报告口径一致)",
        rows=rows,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n写入 %s (%d 对)" % (OUT_JSON, len(rows)))
    if rows:
        nr = sum(1 for r in rows if not r["resolvable_2sigma_seed"])
        print("  其中 %d/%d 对在 2*sigma_seed 下不可分辨" % (nr, len(rows)))


if __name__ == "__main__":
    main()
