"""实验可视化: 训练曲线 / 指标柱状 / 每类F1 / 同patch多模型预测对比"""

# --- 路径引导（本文件位于子目录 viz/，仓库根为上一级）---
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
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = paths.REPO
CKPT = f"{BASE}/checkpoints"
FT = f"{BASE}/fulltile_eval"
ROOT = paths.DATA_ROOT
FIG = f"{BASE}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAMES = ["背景","建筑","道路","水域","裸地","森林","农田"]
IGNORE_COLOR = np.array([40,40,40], np.uint8)  # 未标注/ignore -> 深灰
CLASS_NAMES_SHOW = CLASS_NAMES
PALETTE = np.array([[128,128,128],[255,0,0],[255,255,0],[0,0,255],[160,82,45],[0,128,0],[255,165,0]], np.uint8)

def hist(tag):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return None
    try:
        h = json.load(open(p))
        return h if isinstance(h, list) and h else None
    except Exception:
        return None

# ---------- 1. 训练曲线: Kappa vs epoch ----------


# ---------- 补充: 策略探索训练曲线 ----------
def fig_strategy_curves():
    curves = [("strat_mssact_sgdr120","SGDR(热重启)120ep","#d62728"),
              ("strat_mssact_full30m_oc60","MSSACT 30.5M OneCycle60","#9467bd"),
              ("mssact_full60","MSSACT light OneCycle60","#7f7f7f"),
              ("strat_mssact_oc60_lr5e4","light lr=5e-4 OC60","#ff7f0e"),
              ("strat_no_emr_oc60_lr1e4","w/o EMR lr=1e-4","#8c564b"),
              ("strat_bilinear_oc60_lr1e4","bilinear lr=1e-4","#e377c2"),
              ("strat_deeplab_oc30","DeepLab 快退火30","#1f77b4"),
              ("strat_unet_oc12_lr1e3","U-Net 12ep lr1e-3","#bcbd22")]
    plt.figure(figsize=(10,6))
    for tag, label, c in curves:
        h = hist(tag)
        if not h: continue
        plt.plot([r["epoch"] for r in h],[r["kappa"] for r in h],label=label,color=c,lw=1.8)
    plt.xlabel("Epoch"); plt.ylabel("验证集 Kappa")
    plt.title("策略探索: 调度/学习率/架构 对收敛的影响")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG}/fig_strategy_curves.png", dpi=150); plt.close()
    print("fig_strategy_curves done")

def fig_curves():
    plt.figure(figsize=(10,6))
    curves = [("v3_s256","MSSACT-Net(120ep)","crimson"),
              ("mssact_full60","MSSACT-Net(60ep)","red"),
              ("abl120_no_adapter","w/o Adapter","tab:orange"),
              ("abl120_no_trans","w/o Transformer","tab:purple"),
              ("abl120_no_fpn","w/o FPN","tab:green"),
              ("abl120_no_emr","w/o EMR","tab:brown"),
              ("abl120_no_ecsam","w/o ECSAM","tab:blue")]
    for tag,label,c in curves:
        h = hist(tag)
        if not h: continue
        plt.plot([r["epoch"] for r in h],[r["kappa"] for r in h],label=label,color=c,lw=1.8)
    plt.xlabel("Epoch"); plt.ylabel("验证集 Kappa")
    plt.title("各模型验证 Kappa 收敛曲线 (统一协议, 256px, patience=15)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG}/fig_curves.png", dpi=150); plt.close()
    print("fig_curves done")

# ---------- 2. 全量滑窗评估柱状 ----------
def fig_fulltile_bars():
    p = f"{FT}/fulltile_results.json"
    res = {}
    import glob
    if os.path.exists(p):
        res = json.load(open(p))
    else:
        for cp in glob.glob(f"{FT}/*_conf.json"):
            tag = os.path.basename(cp).replace("_conf.json","")
            try:
                res[tag] = json.load(open(cp))
            except Exception: pass
    if not res: print("skip bars (no fulltile results yet)"); return
    order = ["mssact_full60","deeplab","fcn","pspnet","segformer","unet_matrix"]
    labels = ["MSSACT-Net","DeepLabV3+","FCN","PSPNet","SegFormer-Lite","U-Net"]
    metrics = [("kappa","Kappa"),("oa","OA"),("mf1","mF1"),("miou","mIoU")]
    x = np.arange(len(order)); w = 0.2
    plt.figure(figsize=(12,6))
    for i,(mkey,mlab) in enumerate(metrics):
        vals = [res[t][mkey] for t in order]
        plt.bar(x+(i-1.5)*w, vals, w, label=mlab)
        for xi,v in zip(x+(i-1.5)*w, vals):
            plt.text(xi, v+0.01, f"{v:.3f}", ha="center", fontsize=7)
    plt.xticks(x, labels); plt.ylim(0,0.72); plt.ylabel("分数")
    plt.title("全量滑窗推理评估 (299 张验证tile, 256/128 滑窗+高斯融合)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/fig_fulltile_bars.png", dpi=150); plt.close()
    print("fig_fulltile_bars done")

# ---------- 3. 消融 ΔKappa 柱状 ----------
def fig_ablation():
    rows = [("full60","完整模型"),("no_adapter","w/o Adapter"),("no_trans","w/o Transformer"),
            ("no_fpn","w/o FPN"),("no_ecsam","w/o ECSAM"),("no_emr","w/o EMR")]
    tags = [r[0] for r in rows]; labels=[r[1] for r in rows]
    vals = []
    for t in tags:
        p = f"{CKPT}/{'mssact_full60' if t=='full60' else 'ablate_'+t}_history.json"
        p = p.replace("ablate_full60","ablate_full60")
        if t == "full60": p = f"{CKPT}/mssact_full60_history.json"
        else: p = f"{CKPT}/ablate_{t}_history.json"
        if not os.path.exists(p): vals.append(None); continue
        h = json.load(open(p)); vals.append(max(r["kappa"] for r in h))
    full = vals[0]
    deltas = [0]+[v-full for v in vals[1:]]
    colors = ["crimson"]+["tab:blue"]*len(vals[1:]) if full else None
    plt.figure(figsize=(10,6))
    bars = plt.bar(labels, vals, color=["crimson"]+["steelblue"]*5)
    for b,v,d in zip(bars, vals, deltas):
        plt.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}\n({d:+.3f})", ha="center", fontsize=9)
    plt.ylabel("最优验证 Kappa (60轮预算)"); plt.ylim(0, 0.48)
    plt.title("MSSACT-Net 消融实验 (ΔKappa 相对完整模型)")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_ablation.png", dpi=150); plt.close()
    print("fig_ablation done")

# ---------- 4. 每类F1 (全量滑窗) ----------
def fig_perclass():
    import glob
    p = f"{FT}/fulltile_results.json"
    res = {}
    if os.path.exists(p):
        res = json.load(open(p))
    else:
        for cp in glob.glob(f"{FT}/*_conf.json"):
            tag = os.path.basename(cp).replace("_conf.json","")
            try: res[tag] = json.load(open(cp))
            except Exception: pass
    if not res: print("skip perclass"); return
    order = ["mssact_full60","deeplab","fcn","pspnet","segformer","unet_matrix"]
    labels = ["MSSACT-Net","DeepLabV3+","FCN","PSPNet","SegFormer","U-Net"]
    x = np.arange(len(CLASS_NAMES)); w = 0.15
    plt.figure(figsize=(13,6))
    for i,tag in enumerate(order):
        plt.bar(x+(i-2)*w, res[tag]["f1"], w, label=labels[i])
    plt.xticks(x, CLASS_NAMES); plt.ylim(0,1); plt.ylabel("F1")
    plt.title("每类 F1 (全量滑窗评估)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_perclass.png", dpi=150); plt.close()
    print("fig_perclass done")

# ---------- 5. 同patch多模型预测对比 ----------
def fig_showcase():
    from train_v3 import REMAP
    preds_root = FT
    model_dirs = [("mssact_full60","MSSACT-Net"),("deeplab","DeepLabV3+"),("fcn","FCN"),
                  ("pspnet","PSPNet"),("segformer","SegFormer"),("unet_matrix","U-Net")]
    avail = [m for m in model_dirs if os.path.isdir(f"{preds_root}/preds_{m[0]}")]
    if not avail: print("skip showcase (no preds yet)"); return
    # 选3个tile: 第一张/中间/最后 (必须同时存在于影像目录)
    tiles = sorted(set(os.listdir(f"{preds_root}/preds_{avail[0][0]}")) &
                   set(os.listdir(f"{ROOT}/val/images")) &
                   set(os.listdir(f"{ROOT}/val/masks")))
    if not tiles: print("skip showcase (no common tiles)"); return
    show_tiles = [tiles[0], tiles[len(tiles)//2], tiles[-1]]
    # 选择真正包含有效标注的 tile
    import random
    random.seed(0)
    cand = []
    for t in tiles:
        m = REMAP[np.array(Image.open(f"{ROOT}/val/masks/{t}"))]
        if m[m!=255].size > 1000: cand.append(t)
    if cand: show_tiles = [cand[0], cand[len(cand)//2], cand[-1]]
    for tile in show_tiles:
        img = Image.open(f"{ROOT}/val/images/{tile}").convert("RGB")
        gt = REMAP[np.array(Image.open(f"{ROOT}/val/masks/{tile}"))]
        gt_rgb = PALETTE[np.where(gt==255,0,gt)]
        gt_rgb = np.where(gt[...,None]==255, IGNORE_COLOR, gt_rgb)
        cols = [("影像", np.array(img)), ("真值", gt_rgb)] +                [(lbl, PALETTE[np.array(Image.open(f"{preds_root}/preds_{t}/{tile}"))]) for t,lbl in avail]
        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(3.2*n, 3.8))
        for ax,(lbl,arr) in zip(axes, cols):
            ax.imshow(arr); ax.set_title(lbl, fontsize=10); ax.axis("off")
        from matplotlib.patches import Patch
        handles = [Patch(color=PALETTE[i]/255, label=CLASS_NAMES_SHOW[i]) for i in range(7)]
        handles.append(Patch(color=IGNORE_COLOR/255, label="未标注(ignore)"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.suptitle(f"同tile多模型预测对比: {tile}", fontsize=12)
        plt.tight_layout(rect=[0,0.06,1,1]); plt.savefig(f"{FIG}/fig_showcase_{tile}", dpi=130); plt.close()
    print(f"fig_showcase done ({len(show_tiles)} tiles)")

if __name__ == "__main__":
    fig_curves(); fig_strategy_curves(); fig_ablation(); fig_fulltile_bars(); fig_perclass(); fig_showcase()
