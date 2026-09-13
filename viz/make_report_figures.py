# -*- coding: utf-8 -*-
"""
报告图全量生成器 (viz/make_report_figures.py)

背景
----
2026-09-13 的图片审计发现：报告原先引用的 13 张图**没有一张与当前结论一致**——
其中 6 张无生成脚本（或仅 legacy 可生成、不可复现），其余 7 张的生成脚本读的是
**阶段 A/B 的旧数据**；且多张图含**已更正的数据**（93% 边界统计、预训练 DeepLab、
阶段 A 消融）。

故本脚本**从权威数据源重新生成报告所需的全部图**，不再复用任何旧图：

    FACTS.json                    结果汇总（唯一数据源）
    experiments_index.json        参数量（直读 checkpoint 张量）
    seed_variance.json            σ_seed
    results/artifacts/*           类别分布、收敛量、GPU 采样、伪影
    results/consensus_analysis/*  标签质量、归属、严格性
    results/checkpoints/*_history.json  学习曲线与每轮耗时

原则: **一图一数据源，可追溯**；旧图中仍有历史价值的（阶段 A/B）不再保留图片，
改由正文文字说明。

用法: python viz/make_report_figures.py
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for _f in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(x.name == _f for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

FIG = paths.FIGURES
os.makedirs(FIG, exist_ok=True)
F = json.load(open(paths.FACTS_JSON, encoding="utf-8"))
IX = json.load(open(paths.INDEX_JSON, encoding="utf-8"))
SV = json.load(open(paths.SEEDVAR_JSON, encoding="utf-8"))
CONS = paths.CONSENSUS
ART = os.path.join(paths.RESULTS, "artifacts")
SIGMA = SV.get("sigma_seed_used") or 0.0050
C_OURS, C_SCR, C_PRE, C_OTH = "#c0392b", "#2471a3", "#7f8c8d", "#95a5a6"


def _j(p, d=None):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else d


def save(name):
    plt.savefig(os.path.join(FIG, name)); plt.close()
    print(f"  ✓ {name}", flush=True)


# ---------------------------------------------------------------- §2 数据
def fig_class_distribution():
    d = _j(os.path.join(ART, "class_distribution.json"))
    if not d: return
    names = d["val"]["names"]
    splits = [("train", "训练集 1768"), ("val", "验证集 221"), ("test_clean", "干净测试 141")]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (k, lab) in enumerate(splits):
        if k not in d: continue
        ax.bar(x + (i - 1) * w, np.array(d[k]["frac"]) * 100, w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("像素占比 (%)"); ax.set_title("LoveDA 三划分的类别分布（本项目实际划分）")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    for i, (k, _) in enumerate(splits):
        if k in d:
            ax.text(len(names) - 0.5, 2 + i * 3.4, f"{k} ignore {d[k]['ignore_frac']*100:.2f}%",
                    fontsize=7, color="#555")
    save("fig_class_distribution.png")


# ---------------------------------------------------------------- §3 方法
def fig_param_allocation():
    """模块参数占比：直读 checkpoint 张量求和（与 build_index.py 同法）"""
    import torch
    p = os.path.join(paths.CKPT, "full_all_v2_best.pt")
    if not os.path.exists(p): return
    try:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return
    tot = sum(v.numel() for v in sd.values())
    grp = {}
    for k, v in sd.items():
        top = k.split(".")[0]
        grp[top] = grp.get(top, 0) + v.numel()
    order = ["transformer", "fpn", "encoder", "decoder", "ecsam", "stem"]
    label = {"transformer": "Transformer", "fpn": "FPN", "encoder": "EMR 编码器",
             "decoder": "解码器", "ecsam": "ECSAM(坐标注意力)", "stem": "Stem"}
    vals = [grp.get(k, 0) for k in order]
    dead = F.get("arch_defects", {}).get("D1_fpn_dead_branches", {}).get("dead_params", 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = ["#c0392b" if k in ("fpn", "transformer") else "#95a5a6" for k in order]
    b = a1.barh([label[k] for k in order][::-1], [v / 1e6 for v in vals][::-1],
                color=colors[::-1])
    a1.set_xlabel("参数量 (M)"); a1.set_title(f"模块参数占比（合计 {tot/1e6:.2f}M）")
    for r, v in zip(b, [v / 1e6 for v in vals][::-1]):
        a1.text(v + 0.03, r.get_y() + r.get_height() / 2, f"{v:.2f}M ({100*v*1e6/tot:.1f}%)",
                va="center", fontsize=8)
    a1.grid(axis="x", alpha=.3)
    a2.bar(["实际参与前向", "死参数\n(FPN 支路 0-2)"], [(tot - dead) / 1e6, dead / 1e6],
           color=["#27ae60", "#c0392b"])
    a2.set_ylabel("参数量 (M)"); a2.set_title("有效 vs 死参数")
    a2.text(0, (tot - dead) / 1e6 / 2, f"{100*(tot-dead)/tot:.2f}%", ha="center", color="w")
    a2.text(1, dead / 1e6 / 2, f"{100*dead/tot:.2f}%", ha="center", color="w")
    a2.grid(axis="y", alpha=.3)
    save("fig_param_allocation.png")


# ---------------------------------------------------------------- §4 结果
def fig_learning_curves():
    tags = [("full_all_v2", "MSSACT-Net light (6.10M) 从零", C_OURS),
            ("nd_deeplab_scr", "DeepLabV3+ (39.69M) 从零", C_SCR),
            ("nd_deeplab", "DeepLabV3+ 预训练主干（旁证）", C_PRE),
            ("nd_unet", "U-Net (2.45M)", C_OTH)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for t, lab, c in tags:
        h = _j(os.path.join(paths.CKPT, f"{t}_history.json"))
        if not h: continue
        ax.plot([r["epoch"] for r in h], [r["kappa"] for r in h], label=lab, color=c, lw=1.6)
    ax.axhline(0.5, ls=":", c="#888", lw=1)
    ax.text(1, 0.505, "Kappa = 0.5", fontsize=8, color="#666")
    ax.set_xlabel("训练轮次 (epoch)"); ax.set_ylabel("验证集 Kappa")
    ax.set_title("阶段 C 学习曲线（P-PNG 管线，同协议）")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save("fig_learning_curves.png")


def fig_comparison_bars():
    B = F["stage_C_1768_final"].get("baselines", {})
    M = F["stage_C_1768_final"].get("main", {})
    order = [("full_all_v2", "MSSACT-Net light\n(自研, 6.10M)", C_OURS),
             ("nd_deeplab_scr", "DeepLabV3+\n从零 (39.69M)", C_SCR),
             ("nd_deeplab", "DeepLabV3+\n预训练 (旁证)", C_PRE),
             ("nd_fpn_seg", "FPN-Seg\n27.20M", C_OTH), ("nd_fcn", "FCN\n23.78M", C_OTH),
             ("nd_unet", "U-Net\n2.45M", C_OTH), ("nd_swin_unet", "Swin-Unet\n32.67M", C_OTH),
             ("nd_pspnet", "PSPNet\n13.60M", C_OTH), ("nd_segformer", "SegFormer\n1.23M", C_OTH)]
    xs, ys, cs, ls = [], [], [], []
    for t, lab, c in order:
        r = M.get(t) or B.get(t)
        if not r: continue
        xs.append(lab); ys.append(r["kappa"]); cs.append(c); ls.append(t)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(xs)), ys, color=cs)
    ax.axhline(M.get("full_all_v2", {}).get("kappa", 0), ls="--", c=C_OURS, lw=1)
    ax.errorbar([0], [ys[0]], yerr=[3 * SIGMA], fmt="none", ecolor="k", capsize=4)
    ax.text(0.1, ys[0] + 3 * SIGMA + .004, "±3σ_seed", fontsize=8)
    for i, v in enumerate(ys):
        ax.text(i, v + .003, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=7.5)
    ax.set_ylabel("验证集 Kappa"); ax.set_ylim(0, max(ys) * 1.18)
    ax.set_title("阶段 C 对比实验（同协议；从零训练的公平对比）")
    ax.grid(axis="y", alpha=.3)
    save("fig_comparison_bars.png")


def fig_ablation_bars():
    A = F["stage_C_1768_final"].get("ablation", {})
    ref = F["stage_C_1768_final"]["main"]["full_all_v2"]["kappa"]
    items = sorted(A.items(), key=lambda kv: -(kv[1] or {}).get("kappa", 0))
    lab = {"nd_abl_bilinear": "Bilinear-Up", "nd_abl_trans4l": "Trans-4L",
           "nd_abl_no_emr": "w/o EMR", "nd_abl_no_adapter": "w/o Adapter",
           "nd_abl_trans6l": "Trans-6L", "nd_abl_no_fpn": "w/o FPN",
           "nd_abl_no_trans": "w/o Transformer", "nd_abl_no_ecsam": "w/o ECSAM"}
    xs = ["完整模型"] + [lab.get(k, k) for k, v in items if v]
    ys = [ref] + [v["kappa"] for k, v in items if v]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(ref - 2 * SIGMA, ref + 2 * SIGMA, color="#f1c40f", alpha=.18,
               label=f"±2σ_seed（σ={SIGMA:.4f}）")
    ax.bar(range(len(xs)), ys, color=[C_OURS] + ["#95a5a6"] * (len(xs) - 1))
    ax.axhline(ref, ls="--", c=C_OURS, lw=1)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("验证集 Kappa")
    ax.set_ylim(min(ys) - 2 * SIGMA, max(ys) + 2 * SIGMA)
    ax.set_title("阶段 C 消融实验：全部落在 ±2σ_seed 带内（不可分辨）")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    save("fig_ablation_bars.png")


def fig_confusion_and_perclass():
    """归一化混淆矩阵 + 逐类 IoU：用缓存的预测数组现算"""
    labs = os.path.join(CONS, "labels.npz")
    if not os.path.exists(labs): return
    L = np.load(labs, allow_pickle=True)["labels"]
    names = ["背景", "建筑", "道路", "水域", "裸地", "森林", "农田"]
    for tag, title, fn in (("full_all_v2", "MSSACT-Net light（自研）", "fig_confusion_ours.png"),
                           ("nd_deeplab", "DeepLabV3+（预训练主干，旁证）", "fig_confusion_deeplab.png")):
        p = os.path.join(CONS, f"pred_{tag}.npz")
        if not os.path.exists(p): continue
        P = np.load(p)["pred"]
        cm = np.zeros((7, 7), np.int64)
        for i in range(L.shape[0]):
            m = L[i] != 255
            a, b = P[i][m].astype(np.int64), L[i][m].astype(np.int64)
            cm += np.bincount(b * 7 + a, minlength=49).reshape(7, 7)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(7)); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("预测"); ax.set_ylabel("真值（行归一化）")
        ax.set_title(title, fontsize=9)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="w" if cmn[i, j] > .5 else "#333")
        plt.colorbar(im, fraction=.046)
        save(fn)
    # 逐类 IoU（从 rigor.json）
    R = (_j(os.path.join(CONS, "rigor.json")) or {}).get("results", {})
    if "full_all_v2" not in R: return
    a = R["full_all_v2"]["per_class_iou"]
    b = R.get("nd_deeplab", {}).get("per_class_iou")
    x = np.arange(7); w = .38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w/2, a, w, label=f"自研 (mIoU {R['full_all_v2']['miou_pooled']:.4f})", color=C_OURS)
    if b: ax.bar(x + w/2, b, w, label=f"DeepLabV3+ 预训练 (mIoU {R['nd_deeplab']['miou_pooled']:.4f})", color=C_PRE)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("IoU")
    ax.set_title("逐类 IoU（test_clean 141 张）"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    save("fig_perclass_iou.png")


# ---------------------------------------------------------------- §5 分析
def fig_data_ladder():
    D = (F.get("data_ladder") or {}).get("rows") or {}
    if not D: return
    series = [("full", "MSSACT-Net 完整 (6.10M)", C_OURS, "-o"),
              ("noecsam", "MSSACT-Net w/o ECSAM", "#e67e22", "-s"),
              ("unet", "U-Net (2.45M)", "#95a5a6", "-^"),
              ("deeplab_scr", "DeepLabV3+ 从零 (39.69M)", C_SCR, "-D"),
              ("deeplab", "DeepLabV3+ 预训练（旁证）", C_PRE, "--o")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, lab, c, st in series:
        ns = [250, 500, 1000]
        ys = [D.get(f"lgR_n{n}_{m}", {}).get("kappa") for n in ns]
        if not any(ys): continue
        ax.plot(ns, [y if y is not None else np.nan for y in ys], st, label=lab, color=c, lw=1.6)
    ax.set_xlabel("训练集规模 (张)"); ax.set_ylabel("验证集 Kappa")
    ax.set_xticks([250, 500, 1000])
    ax.set_title("数据量阶梯（30 轮固定预算 / memmap 子集 / 验证集固定 221 张）")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save("fig_data_ladder.png")


def fig_learning_amount():
    rows = _j(os.path.join(ART, "convergence_learning_amount.json")) or []
    rows = [r for r in rows if r.get("e50") and r["e50"][1] is not None]
    if not rows: return
    rows.sort(key=lambda r: r["e50"][1])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ys = [r["e50"][1] * 100 for r in rows]
    cols = [C_OURS if "MSSACT" in r["label"] else
            (C_SCR if "从零" in r["label"] else
             (C_PRE if "预训练" in r["label"] else C_OTH)) for r in rows]
    ax.barh(range(len(rows)), ys, color=cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvspan(50, 65, color="#f1c40f", alpha=.15)
    ax.text(51, len(rows) - 0.6, "多数模型集中区\n50–65%", fontsize=8, color="#8a6d00")
    for i, y in enumerate(ys):
        ax.text(y + 1, i, f"{y:.1f}%", va="center", fontsize=7.5)
    ax.set_xlabel("达到 Kappa 0.5 所需的累计学习量 S = sum(eta)（占全程比例）")
    ax.set_title("收敛速度归因：调度 vs 架构")
    ax.grid(axis="x", alpha=.3)
    save("fig_learning_amount.png")


def fig_boundary_and_headroom():
    B = _j(os.path.join(CONS, "boundary_analysis.json")) or {}
    H = _j(os.path.join(CONS, "headroom.json")) or {}
    if B:
        bf = B.get("band_frac") or {}
        order = ["0 (边界像素)", "1-2", "3-4", "5-8", ">8 (内部)"]
        fr = [bf.get(k, {}).get("frac", 0) * 100 for k in order if k in bf]
        nm = [k for k in order if k in bf]
        cons = B.get("consensus_vs_label_by_band") or {}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        a1.bar(nm, fr, color=["#c0392b"] + ["#e67e22"] * 3 + ["#27ae60"])
        for i, v in enumerate(fr):
            a1.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
        a1.set_ylabel("占有效像素 (%)"); a1.set_title("距类边界的像素分布（已更正）")
        a1.tick_params(axis="x", labelsize=8); a1.grid(axis="y", alpha=.3)
        cv = [cons.get(k) for k in nm]
        a2.plot(nm, cv, "-o", color="#2471a3")
        for i, v in enumerate(cv):
            if v is not None: a2.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=8)
        a2.set_ylabel("共识 vs 人工标签 一致率"); a2.set_ylim(0.4, 0.8)
        a2.set_title("准确率随距边界距离单调上升")
        a2.tick_params(axis="x", labelsize=8); a2.grid(alpha=.3)
        save("fig_boundary_bands.png")
    # Kappa 归属
    ok = H.get("oracle_kappa") or {}
    if ok:
        lab = {"full_all_v2": "MSSACT-Net", "nd_deeplab": "DeepLabV3+", "nd_fpn_seg": "FPN-Seg",
               "nd_fcn": "FCN", "nd_unet": "U-Net", "nd_swin_unet": "Swin-Unet"}
        ks = [k for k in ok if k in lab]
        x = np.arange(len(ks)); w = .38
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(x - w/2, [ok[k]["d_boundary"] for k in ks], w,
               label="完美解决边界（占 2.4% 像素）", color="#c0392b")
        ax.bar(x + w/2, [ok[k]["d_interior"] for k in ks], w,
               label="完美解决内部（占 78.4% 像素）", color="#27ae60")
        for i, k in enumerate(ks):
            ax.text(i - w/2, ok[k]["d_boundary"] + .006, f"+{ok[k]['d_boundary']:.3f}",
                    ha="center", fontsize=8)
            ax.text(i + w/2, ok[k]["d_interior"] + .006, f"+{ok[k]['d_interior']:.3f}",
                    ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in ks], fontsize=8)
        ax.set_ylabel("Kappa 增量上限"); ax.set_title("Kappa 归属：聚合指标几乎看不见边界质量")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        save("fig_headroom.png")


def fig_agreement():
    C = _j(os.path.join(CONS, "consistency.json")) or {}
    if not C: return
    ml = C.get("model_label_agreement", {})
    mm = C.get("model_model_agreement", {})
    mmv = list(mm.values())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(["模型–标签\n(平均)", "模型–模型\n(平均)", "共识–人工标签"],
           [np.mean(list(ml.values())), np.mean(mmv), C.get("consensus_vs_label", 0)],
           color=["#c0392b", "#2471a3", "#e67e22"])
    for i, v in enumerate([np.mean(list(ml.values())), np.mean(mmv), C.get("consensus_vs_label", 0)]):
        ax.text(i, v + .008, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("一致率"); ax.grid(axis="y", alpha=.3)
    ax.set_title("标签质量上限的证据（test_clean 141 张，6 个模型）")
    ax.text(1.0, 0.30, "模型互相认同 远高于 认同人工标签", ha="center", fontsize=9, color="#2471a3")
    save("fig_agreement.png")


def fig_overfit():
    O = F.get("overfit") or {}
    if not O: return
    lab = {"mssact_full60": "MSSACT (阶段A)", "v3_s256": "v3_s256 (阶段A)",
           "deeplab": "DeepLab (阶段A)", "fcn": "FCN (阶段A)", "pspnet": "PSPNet (阶段A)",
           "segformer": "SegFormer (阶段A)", "unet_matrix": "U-Net (阶段A)",
           "fpn_seg": "FPN-Seg (阶段A)", "swin_unet": "Swin-Unet (阶段A)"}
    ks = [k for k in O if isinstance(O[k], dict) and "H_gap" in O[k]]
    if not ks: return
    ks.sort(key=lambda k: O[k]["H_gap"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(range(len(ks)), [O[k]["H_gap"] for k in ks],
            color=["#c0392b" if abs(O[k]["H_gap"]) > .2 else "#95a5a6" for k in ks])
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([lab.get(k, k) for k in ks], fontsize=8)
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel("ΔH = H_train − H_val（负值=验证不确定度更高）")
    ax.set_title("过拟合诊断（阶段 A 历史结果；本报告仅作趋势参考）")
    ax.grid(axis="x", alpha=.3)
    save("fig_overfit_dh.png")


def fig_hardware():
    """GPU 利用率时序 + 每轮耗时 vs 参数量"""
    files = [("fig_gpu_duty_png", "PNG 管线（切换前）", C_PRE),
             ("fig_gpu_duty_memmap", "memmap 管线（切换后）", C_OURS)]
    png = os.path.join(ART, "gpu_duty_png.txt"); mem = os.path.join(ART, "gpu_duty_memmap.txt")
    txt = os.path.join(ART, "gpu_duty.txt")
    if os.path.exists(png) and os.path.exists(mem):
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        for f, lab, c in [(png, "PNG 管线（数据受限）", C_PRE), (mem, "memmap 管线（计算受限）", C_OURS)]:
            u = [int(l.split(",")[0]) for l in open(f) if l.strip()]
            ax.plot(np.arange(len(u)) * .25, u, color=c, lw=1.2, label=f"{lab}  均值 {np.mean(u):.1f}%")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("GPU 利用率 (%)"); ax.set_ylim(0, 105)
        ax.set_title("同一台机器的 GPU 利用率时序（同类负载，仅数据投递后端不同）")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        save("fig_gpu_duty.png")
    # 每轮耗时 vs 参数量（用 history 的 sec）
    # 每轮耗时优先取结果 artifacts/epoch_times.json（由队列日志与 history.sec 汇总）
    ET = _j(os.path.join(ART, "epoch_times.json")) or {}
    PX = {k: (v.get("params_M") or 0) for k, v in (IX.get("experiments") or {}).items()}
    rec = [("lgR_D3_ch_tiny", 1.547), ("lgR_D1_join_nofpn_notrans", 2.005),
           ("lgR_bs8_full_lr2e4", 6.102), ("lgR_D2_decoder_ca", 6.115),
           ("lgR_D4_ch_large", 24.242)]
    xs, ys, ls = [], [], []
    for t, pm in rec:
        sec = (ET.get(t) or {}).get("s_per_epoch")
        if sec is None:
            h = _j(os.path.join(paths.CKPT, f"{t}_history.json")) or []
            s = [r["sec"] for r in h if isinstance(r, dict) and "sec" in r]
            sec = float(np.median(s)) if s else None
        if sec is None: continue
        xs.append(PX.get(t) or pm); ys.append(float(sec)); ls.append(t)
    if xs:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(xs, ys, "-o", color=C_OURS)
        for x, y, t in zip(xs, ys, ls):
            ax.annotate(t.replace("lgR_", ""), (x, y), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("参数量 (M)"); ax.set_ylabel("每轮耗时 (s)")
        ax.set_title("memmap 管线：每轮耗时随参数量增长（计算受限）")
        ax.grid(alpha=.3)
        save("fig_epoch_time.png")


def fig_architecture_defects():
    """死参数 + 边缘密度/周期伪影"""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    A = _j(os.path.join(CONS, "artifact.json")) or {}
    M = A.get("models") or {}
    lab = {"full_all_v2": "MSSACT-Net", "nd_deeplab": "DeepLabV3+", "nd_fpn_seg": "FPN-Seg",
           "nd_fcn": "FCN", "nd_unet": "U-Net", "nd_swin_unet": "Swin-Unet"}
    base = A.get("label_edge_density")
    ks = [k for k in M if k in lab]
    if base and ks:
        ratio = [M[k]["edge_ratio_vs_label"] for k in ks]
        cols = [C_OURS if k == "full_all_v2" else C_OTH for k in ks]
        ax.bar([lab[k] for k in ks], ratio, color=cols)
        ax.axhline(1.0, ls="--", c="k", lw=1)
        ax.text(0.02, 1.05, "标签本身（1.00）", fontsize=8, transform=ax.get_yaxis_transform())
        for i, v in enumerate(ratio):
            ax.text(i, v + .06, f"{v:.2f}×", ha="center", fontsize=8)
        ax.set_ylabel("边缘密度 / 标签"); ax.set_ylim(0, max(ratio) * 1.25)
        ax.set_title("预测的过度碎裂程度（自研 4.56× 于真值）")
        ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=.3)
        save("fig_edge_density.png")


def fig_scratch_vs_pretrained():
    S = F.get("scratch_vs_pretrained") or {}
    rows = S.get("rows") or []
    m = S.get("main_from_scratch") or {}
    if not rows and not m: return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    if m:
        a1.bar(["MSSACT-Net\n6.10M 从零", "DeepLabV3+\n39.69M 从零"],
               [m["mssact_6p10M"], m["deeplab_39p69M_scratch"]], color=[C_OURS, C_SCR])
        for i, v in enumerate([m["mssact_6p10M"], m["deeplab_39p69M_scratch"]]):
            a1.text(i, v + .003, f"{v:.4f}", ha="center", fontsize=9)
        a1.errorbar([0, 1], [m["mssact_6p10M"], m["deeplab_39p69M_scratch"]],
                    yerr=[2 * SIGMA, 2 * SIGMA], fmt="none", ecolor="k", capsize=5)
        a1.set_ylabel("验证集 Kappa"); a1.set_ylim(0.60, 0.66)
        a1.set_title(f"主协议·均从零：Δ={m['delta']:+.4f}（{m['sigma_units']:.2f}σ，不可分辨）",
                     fontsize=9)
        a1.grid(axis="y", alpha=.3)
    if rows:
        x = np.arange(len(rows)); w = .36
        pre = [r["pretrained"] if r.get("pretrained") is not None else np.nan for r in rows]
        scr = [r["scratch"] for r in rows]
        a2.bar(x - w/2, pre, w, label="预训练主干", color=C_PRE)
        a2.bar(x + w/2, scr, w, label="从零训练", color=C_SCR)
        for i, r in enumerate(rows):
            if r.get("pretrain_gain") is not None:
                a2.text(i, max(pre[i], scr[i]) + .02, f"+{r['pretrain_gain']:.3f}",
                        ha="center", fontsize=8, color="#c0392b")
        a2.set_xticks(x); a2.set_xticklabels([r["setting"] for r in rows], fontsize=8)
        a2.set_ylabel("Kappa"); a2.set_ylim(0, .85)
        a2.set_title("预训练的独立贡献（同模型同管线，仅差初始化）", fontsize=9)
        a2.legend(fontsize=8); a2.grid(axis="y", alpha=.3)
    save("fig_scratch_vs_pretrained.png")


def fig_seed_variance():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    g = SV.get("groups", {}).get("完整模型 MSSACT 6.10M (P-MEM-ROLL)", {})
    mem = g.get("members") or {}
    if mem:
        ks = list(mem.values())
        a1.scatter(range(len(ks)), ks, s=60, color=C_OURS, zorder=3)
        a1.axhline(np.mean(ks), ls="--", c="#333", lw=1)
        a1.axhspan(np.mean(ks) - np.std(ks, ddof=1), np.mean(ks) + np.std(ks, ddof=1),
                   color="#f1c40f", alpha=.2, label=f"±1σ（σ={np.std(ks,ddof=1):.4f}）")
        a1.axhspan(np.mean(ks) - 2*np.std(ks, ddof=1), np.mean(ks) + 2*np.std(ks, ddof=1),
                   color="#f1c40f", alpha=.10)
        for i, (k, v) in enumerate(mem.items()):
            a1.text(i, v + .0012, f"{v:.4f}", ha="center", fontsize=8)
        a1.set_xticks(range(len(ks)))
        a1.set_xticklabels([k.split("_")[0] for k in mem], fontsize=8)
        a1.set_ylabel("验证集 Kappa")
        a1.set_title(f"同一配置 4 个随机种子（极差 {max(ks)-min(ks):.4f}）", fontsize=9)
        a1.legend(fontsize=8); a1.grid(axis="y", alpha=.3)
    d = SV.get("deltas_in_sigma") or []
    if d:
        d = sorted(d, key=lambda r: -abs(r["n_sigma"]))
        cols = ["#c0392b" if abs(r["n_sigma"]) >= 2 else "#95a5a6" for r in d]
        a2.barh(range(len(d)), [r["n_sigma"] for r in d], color=cols)
        a2.set_yticks(range(len(d))); a2.set_yticklabels([r["tag"] for r in d], fontsize=7)
        a2.invert_yaxis()
        a2.axvline(2, ls=":", c="#c0392b"); a2.axvline(-2, ls=":", c="#c0392b")
        a2.axvline(1, ls=":", c="#888"); a2.axvline(-1, ls=":", c="#888")
        a2.set_xlabel("ΔKappa（以 σ_seed 为单位）")
        a2.set_title("各变体相对基线：仅容量变体超过 2σ", fontsize=9)
        a2.grid(axis="x", alpha=.3)
    save("fig_seed_variance.png")


def fig_crop_strategy():
    C = (F.get("crop_strategy") or {}).get("rows") or {}
    if not C: return
    order = ["lgR_bs8_full_lr2e4", "lgR_c256_center", "lgR_c384_rand", "lgR_c512_rand"]
    lab = {"lgR_bs8_full_lr2e4": "256 随机\n(基准)", "lgR_c256_center": "256 固定中心",
           "lgR_c384_rand": "384 随机", "lgR_c512_rand": "512 随机"}
    rows = [C[t] for t in order if t in C]
    xs = [lab[t] for t in order if t in C]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    b = ax.bar(range(len(rows)), [r["kappa"] for r in rows],
               color=[C_OURS] + ["#95a5a6"] * (len(rows) - 1))
    base = rows[0]["kappa"]
    ax.axhline(base, ls="--", c=C_OURS, lw=1)
    for i, r in enumerate(rows):
        ax.text(i, r["kappa"] + .003, f"{r['kappa']:.4f}\nΔ{r['delta']:+.4f}",
                ha="center", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(range(len(rows)), [r["s_per_epoch"] for r in rows], "-o", color="#e67e22")
    for i, r in enumerate(rows):
        ax2.text(i, r["s_per_epoch"] + 5, f"{r['s_per_epoch']}s/轮", ha="center",
                 fontsize=8, color="#a04000")
    ax2.set_ylabel("每轮耗时 (s)", color="#a04000")
    ax.set_ylabel("验证集 Kappa"); ax.set_ylim(min(r["kappa"] for r in rows) - .02,
                                              max(r["kappa"] for r in rows) + .03)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(xs, fontsize=8.5)
    ax.set_title("裁剪策略：效应量 3.8–4.5σ（训练策略 ＞ 架构模块）")
    ax.grid(axis="y", alpha=.3)
    save("fig_crop_strategy.png")


def fig_budget_attack():
    AE = (F.get("arch_experiments") or {}).get("rows") or {}
    b = {k: v for k, v in AE.items() if k.startswith("lgR_b15_")}
    if not b: return
    lab = {"lgR_b15_full": "MSSACT 完整", "lgR_b15_noecsam": "w/o ECSAM",
           "lgR_b15_noemr": "w/o EMR", "lgR_b15_unet": "U-Net",
           "lgR_b15_deeplab": "DeepLabV3+ 预训练(旁证)",
           "lgR_b15_deeplab_scr": "DeepLabV3+ 从零"}
    ks = sorted(b, key=lambda k: -(b[k].get("kappa") or 0))
    ref = (b.get("lgR_b15_full") or {}).get("kappa")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = []
    for k in ks:
        cols.append(C_OURS if k == "lgR_b15_full" else
                    (C_SCR if k.endswith("_scr") else
                     (C_PRE if k == "lgR_b15_deeplab" else "#95a5a6")))
    ax.bar(range(len(ks)), [b[k]["kappa"] for k in ks], color=cols)
    ax.axhline(ref, ls="--", c=C_OURS, lw=1)
    for i, k in enumerate(ks):
        ax.text(i, b[k]["kappa"] + .006, f"{b[k]['kappa']:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([lab.get(k, k) for k in ks], fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("验证集 Kappa"); ax.set_ylim(0, max(b[k]["kappa"] for k in ks) * 1.2)
    ax.set_title("预算攻击（15 轮）：模块差异仍很小；从零 DeepLab 反低于自研")
    ax.grid(axis="y", alpha=.3)
    save("fig_budget_attack.png")


def fig_architecture_schematic():
    """架构数据流示意（纯 matplotlib 绘制）"""
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    stages = [("输入\n256²×3", "#d5dbdb"), ("Stem\n256²", "#aeb6bf"),
              ("EMR×4\n256²→32²", "#85c1e9"), ("FPN\n(仅最深层生效)", "#f5b7b1"),
              ("Transformer\n32²→1024 token", "#f9e79f"), ("解码器\n32²→256²\n无跳连", "#f5b7b1"),
              ("输出\n7 类", "#d5dbdb")]
    x = 0
    for i, (t, c) in enumerate(stages):
        ax.add_patch(plt.Rectangle((x, .25), 1.15, .5, fc=c, ec="#333", lw=1))
        ax.text(x + .575, .5, t, ha="center", va="center", fontsize=7.6)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.3, .5), xytext=(x + 1.15, .5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
        x += 1.32
    ax.text(0, .06, "红色 = 已数值证实的缺陷环节：FPN 的多尺度输出仅最深一层进入前向（29.96% 参数不参与），"
                    "解码器无跳连、由 32² 特征上采样 8 倍；Transformer 无位置编码",
            fontsize=7.6, color="#a93226")
    ax.set_xlim(-.05, x + .05); ax.set_ylim(0, 1)
    save("fig_architecture_schematic.png")


if __name__ == "__main__":
    print("生成报告图（全部来自权威数据源）...")
    for fn in (fig_architecture_schematic, fig_param_allocation, fig_class_distribution,
               fig_learning_curves, fig_comparison_bars, fig_ablation_bars,
               fig_confusion_and_perclass, fig_data_ladder, fig_learning_amount,
               fig_boundary_and_headroom, fig_agreement, fig_overfit, fig_hardware,
               fig_architecture_defects, fig_scratch_vs_pretrained, fig_seed_variance,
               fig_crop_strategy, fig_budget_attack):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} 失败: {type(e).__name__} {e}", flush=True)
    print("完成。")
