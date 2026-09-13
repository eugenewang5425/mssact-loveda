# -*- coding: utf-8 -*-
"""
累计学习量分析 (analysis/convergence_amount.py)

目的
----
回答「模型收敛快，是**调度**的作用还是**架构**的作用？」

方法（为什么这样能分离两者）
--------------------------
OneCycleLR 的 LR 轨迹是**解析已知**的：给定 max_lr、总步数 T、warmup 比例 pct_start，
任一时刻的 η 可由公式直接算出，与模型无关。定义**累计学习量**：

        S(t) = Σ_{τ≤t} η_τ

若两个模型在同一水平上所需的 S 相同，则差异来自架构；若几乎所有模型都在相近的
S 比例处达到同一水平，则说明"第几轮收敛"只是调度曲线形状的映射，与架构无关。

**这是本脚本的全部论点**——原先报告只说了一句"S=Σηₜ"而没解释为何有效。

重要更正（2026-09-13）
--------------------
原 `convergence_learning_amount.json` 里的 DeepLabV3+ 条目为 `k_final=0.6876`，
即**带 ImageNet 预训练主干的 `nd_deeplab`**。用它论证"空洞卷积结构学习效率更高"
属过度归因（预训练本身就加速早期收敛）。本脚本同时输出**预训练版与从零版**，
使二者可直接对照。

用法: python analysis/convergence_amount.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

# 全部为阶段 C（P-PNG 管线）同协议: 60ep / pt20 / bs8 / lr2e-4 / OneCycle(pct_start=0.06)
# 注意: 全部必须同属 P-PNG 管线, 否则是跨管线比较(违反既定纪律)
MODELS = [
    ("full_all_v2",          "MSSACT-light (6.10M)"),
    ("nd_abl_no_ecsam",      "w/o ECSAM"),
    ("nd_abl_no_trans",      "w/o Transformer"),
    ("nd_abl_no_fpn",        "w/o FPN"),
    ("nd_abl_no_adapter",    "w/o Adapter"),
    ("nd_abl_no_emr",        "w/o EMR"),
    ("nd_abl_bilinear",      "Bilinear-Up"),
    ("nd_abl_trans4l",       "Trans-4L"),
    ("nd_unet",              "U-Net (2.45M)"),
    ("nd_fcn",               "FCN (23.78M)"),
    ("nd_pspnet",            "PSPNet (13.60M)"),
    ("nd_segformer",         "SegFormer-Lite (1.23M)"),
    ("nd_fpn_seg",           "FPN-Seg (27.20M)"),
    ("nd_swin_unet",         "Swin-Unet (32.67M)"),
    ("nd_deeplab",           "DeepLabV3+ (预训练主干)"),
    ("nd_deeplab_scr",       "DeepLabV3+ (从零)"),
]
MAX_LR, PCT_START, EPOCHS, ITERS_PER_EPOCH = 2e-4, 0.06, 60, 221   # 1768/8 = 221


def onecycle_curve(max_lr=MAX_LR, epochs=EPOCHS, iters=ITERS_PER_EPOCH, pct_start=PCT_START):
    """复现 torch OneCycleLR 的 LR 轨迹（与 PyTorch 实现一致的两段式）"""
    total = epochs * iters
    warmup = int(total * pct_start)
    out = []
    for i in range(total):
        if i < warmup:
            frac = i / max(warmup, 1)
            lr = max_lr * (0.1 + 0.9 * frac) if False else max_lr * frac / max(pct_start, 1e-9) * pct_start
            lr = max_lr * (i / max(warmup, 1))
        else:
            frac = (i - warmup) / max(total - warmup, 1)
            lr = max_lr * 0.5 * (1 + __import__("math").cos(__import__("math").pi * frac))
        out.append(lr)
    return out


def cumulative(curve, iters=ITERS_PER_EPOCH):
    """每个 epoch 结束时的累计学习量（占全程的比例）"""
    s = 0.0
    tot = sum(curve)
    per_ep = []
    for e in range(1, EPOCHS + 1):
        s += sum(curve[(e - 1) * iters:e * iters])
        per_ep.append(s / tot)
    return per_ep


def main():
    curve = onecycle_curve()
    cum = cumulative(curve)
    rows = []
    for tag, label in MODELS:
        p = os.path.join(paths.CKPT, f"{tag}_history.json")
        if not os.path.exists(p):
            print(f"  跳过（无 history）: {tag}"); continue
        h = json.load(open(p, encoding="utf-8"))
        ks = [r.get("kappa", 0.0) for r in h]
        eps = [r.get("epoch", i + 1) for i, r in enumerate(h)]
        kmax = max(ks) if ks else 0.0
        def first(th):
            for k, e in zip(ks, eps):
                if k >= th:
                    return int(e), round(cum[int(e) - 1], 4)
            return None, None
        e50, s50 = first(0.5)
        e60, s60 = first(0.6)
        rows.append(dict(tag=tag, label=label, T=len(h),
                         k_final=round(float(ks[-1]), 4), k_max=round(float(kmax), 4),
                         e50=[e50, s50], e60=[e60, s60]))
    dst = os.path.join(paths.RESULTS, "artifacts", "convergence_learning_amount.json")
    json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {dst}")
    print(f"\n{'模型':30s} {'末轮K':>8s} {'达K0.5轮':>9s} {'累计学习量':>10s}")
    for r in sorted(rows, key=lambda x: (x["e50"][1] if x["e50"][1] is not None else 9)):
        e50 = r["e50"][0] if r["e50"][0] is not None else "—"
        s50 = f"{r['e50'][1]*100:.1f}%" if r["e50"][1] is not None else "未达到"
        print(f"{r['label']:30s} {r['k_final']:8.4f} {str(e50):>9s} {s50:>10s}")
    print("\n判读: 若多数模型集中在相近的累计学习量附近达 K0.5, 则'第几轮收敛'主要由")
    print("      调度曲线决定; 显著偏离者才可能有架构层面的学习效率优势。")


if __name__ == "__main__":
    main()
