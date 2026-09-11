# -*- coding: utf-8 -*-
"""统计图表集: 混淆矩阵(计数/归一化) + 精度-召回-F1 + 模型效率散点 + LR调度曲线"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = f"{BASE}/figures"; TT = f"{BASE}/tta_eval"; CKPT = f"{BASE}/checkpoints"
NAMES = ["背景","建筑","道路","水域","裸地","森林","农田"]

def load(tag):
    p = f"{TT}/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def cmaps():
    return LinearSegmentedColormap.from_list("wr", ["#ffffff","#ffcccc","#ff6666","#cc0000","#660000"])

def plot_cm(ax, cm, title, norm=False, vmax=None):
    cm = np.array(cm, dtype=np.float64)
    if norm:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cm, cmap=cmaps(), vmin=0, vmax=(1.0 if norm else vmax))
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(NAMES, fontsize=9)
    ax.set_xlabel("预测", fontsize=10); ax.set_ylabel("真实", fontsize=10)
    ax.set_title(title, fontsize=11)
    for i in range(7):
        for j in range(7):
            v = cm[i,j]
            txt = f"{v:.2f}" if norm else (f"{v/1000:.0f}k" if v>=1000 else f"{v:.0f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if (norm and v>0.5) or (not norm and v>=(vmax or 1)*0.5) else "#333")
    plt.colorbar(im, ax=ax, fraction=0.046)

def main():
    old = load("mssact_full60_cleantest")
    new = load("full_all_cleantest")
    final = load("post_all_cleantest_TTA") or load("post_all_cleantest")
    if not (old and new): print("缺少混淆矩阵数据"); return

    # ---- 图1: 混淆矩阵 计数 (旧 vs 新) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    vmax = max(np.array(old["confusion"]).max(), np.array(new["confusion"]).max())
    plot_cm(axes[0], old["confusion"], f"混淆矩阵 (计数) — 旧模型 957张\nKappa={old['kappa']:.4f}", vmax=vmax)
    plot_cm(axes[1], new["confusion"], f"混淆矩阵 (计数) — 全量模型 2257张\nKappa={new['kappa']:.4f}", vmax=vmax)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_confusion_count.png", dpi=150); plt.close()
    print("fig_confusion_count done")

    # ---- 图2: 混淆矩阵 行归一化 (召回视角) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    plot_cm(axes[0], old["confusion"], f"归一化混淆矩阵 (行=召回率) — 旧模型\nKappa={old['kappa']:.4f}", norm=True)
    plot_cm(axes[1], new["confusion"], f"归一化混淆矩阵 (行=召回率) — 全量模型\nKappa={new['kappa']:.4f}", norm=True)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_confusion_norm.png", dpi=150); plt.close()
    print("fig_confusion_norm done")

    # ---- 图3: 最终模型混淆矩阵 (单独大图) ----
    if final:
        fig, ax = plt.subplots(figsize=(8.5, 7))
        plot_cm(ax, final["confusion"], f"最终模型 (2257张+后训练+TTA)\nKappa={final['kappa']:.4f} OA={final['oa']:.4f}", norm=True)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig_confusion_final.png", dpi=150); plt.close()
        print("fig_confusion_final done")

    # ---- 图4: 每类 精度(UA)/召回(PA)/F1 三指标对比 ----
    def prf(conf):
        c = np.array(conf, dtype=np.float64)
        prec = np.diag(c)/np.maximum(c.sum(0),1)   # UA
        rec  = np.diag(c)/np.maximum(c.sum(1),1)   # PA
        f1   = 2*prec*rec/np.maximum(prec+rec,1e-9)
        return prec, rec, f1
    p_old, r_old, f_old = prf(old["confusion"])
    p_new, r_new, f_new = prf(new["confusion"])
    x = np.arange(7); w = 0.13
    plt.figure(figsize=(12.5,5.5))
    plt.bar(x-2.5*w, p_old, w, label="旧-精度UA", color="#bbbbbb")
    plt.bar(x-1.5*w, r_old, w, label="旧-召回PA", color="#888888")
    plt.bar(x-0.5*w, f_old, w, label="旧-F1", color="#555555")
    plt.bar(x+0.5*w, p_new, w, label="新-精度UA", color="#ff9999")
    plt.bar(x+1.5*w, r_new, w, label="新-召回PA", color="#ff4444")
    plt.bar(x+2.5*w, f_new, w, label="新-F1", color="#990000")
    plt.xticks(x, NAMES); plt.ylim(0,1); plt.ylabel("分数")
    plt.title("每类精度/召回/F1: 旧模型 vs 全量模型 (192张干净测试集)")
    plt.legend(ncol=2, fontsize=9); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_prf_perclass.png", dpi=150); plt.close()
    print("fig_prf_perclass done")

    # ---- 图5: 模型效率散点 (参数量 vs Kappa) ----
    try:
        eff = [("U-Net",2.44,0.0939),("SegFormer-Lite",1.01,0.1775),("PSPNet",13.59,0.2513),
               ("Swin-Unet",32.64,0.2682),("FCN",23.72,0.2837),("FPN",27.15,0.2982),
               ("DeepLabV3+",39.64,0.3806),
               ("MSSACT(旧957)",6.10,0.4065),("MSSACT(全量2257)",6.10,0.6155)]
        plt.figure(figsize=(10,6))
        for name, p, k in eff:
            c = "crimson" if "MSSACT" in name else "steelblue"
            mk = "*" if "MSSACT" in name else "o"
            sz = 300 if "MSSACT" in name else 90
            plt.scatter(p, k, c=c, marker=mk, s=sz, zorder=3)
            plt.annotate(name, (p,k), textcoords="offset points", xytext=(7,5), fontsize=8.5)
        plt.xlabel("参数量 (M)"); plt.ylabel("验证集 Kappa")
        plt.title("参数效率: Kappa vs 模型大小 (旧数据集957张)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{FIG}/fig_efficiency.png", dpi=150); plt.close()
        print("fig_efficiency done")
    except Exception as e:
        print("skip efficiency:", e)

    # ---- 图6: OneCycle 学习率调度曲线 ----
    import math
    def lr_frac(t, T, pct=0.06):
        tw = pct*T
        return t/tw if t<=tw else 0.5*(1+math.cos(math.pi*(t-tw)/(T-tw)))
    plt.figure(figsize=(10,5))
    for T, c, l in [(60,"tab:red","OneCycle-60 (全量模型)"),(120,"tab:gray","OneCycle-120 (v3_s256)")]:
        ts = np.arange(T)
        plt.plot(ts, [lr_frac(t,T) for t in ts], color=c, lw=1.8, label=l)
    plt.axhline(0.1, ls="--", c="k", alpha=0.4, lw=1)
    plt.text(2, 0.11, "LR=10% 峰值 (best epoch 通常在此附近)", fontsize=9, alpha=0.8)
    plt.xlabel("Epoch"); plt.ylabel("学习率 (相对峰值)")
    plt.title("OneCycle 学习率调度"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_lr_schedule.png", dpi=150); plt.close()
    print("fig_lr_schedule done")

    # ---- 数值表: 混淆矩阵关键错误流向 ----
    cn = np.array(new["confusion"], dtype=np.float64)
    rn = cn/np.maximum(cn.sum(1,keepdims=True),1)
    print("\n=== 新模型主要错误流向 (真实类 -> 最易误判为) ===")
    for i,n in enumerate(NAMES):
        off = [(NAMES[j], rn[i,j]) for j in range(7) if j!=i]
        off.sort(key=lambda t:-t[1])
        print(f"  {n:4s} (召回 {rn[i,i]:.3f}) 主要混淆: " + ", ".join(f"{a} {b:.2f}" for a,b in off[:2]))

if __name__ == "__main__":
    main()
