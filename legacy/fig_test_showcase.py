"""测试集 patch 预测对比图: 影像 | 真值 | 旧模型(957) | 全量模型(2257) | 后训练模型
选 3 个测试集 tile (干净子集, 均未参与任何训练)
"""
import os, sys, json
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from train_v3 import REMAP

CLEAN = os.path.join(paths.DATA_ROOT, "newsplit/test_clean")
CKPT = paths.CKPT
FIG = os.path.join(paths.REPO, "figures")
NUM_CLASSES, IGNORE, S, STRIDE = 7, 255, 256, 128
MEAN = np.array([0.298,0.3224,0.2949], np.float32); STD = np.array([0.1713,0.1415,0.1328], np.float32)
PALETTE = np.array([[128,128,128],[255,0,0],[255,255,0],[0,0,255],[160,82,45],[0,128,0],[255,165,0]], np.uint8)
IGNORE_COLOR = np.array([40,40,40], np.uint8)
NAMES = ["背景","建筑","道路","水域","裸地","森林","农田"]

def gauss2d(s, sigma):
    ax = np.arange(s)-(s-1)/2.0; g = np.exp(-(ax**2)/(2*sigma**2)); return np.outer(g,g).astype(np.float32)

@torch.no_grad()
def infer_tile(model, img_u8, batch=4):
    H, W = img_u8.shape[:2]
    ph=(STRIDE-(H-S)%STRIDE)%STRIDE; pw=(STRIDE-(W-S)%STRIDE)%STRIDE
    x = np.pad(img_u8,((0,ph),(0,pw),(0,0)),mode="reflect").astype(np.float32)/255.0
    x = (x-MEAN)/STD
    xp = torch.from_numpy(x.transpose(2,0,1)[None])
    Hp, Wp = H+ph, W+pw
    prob = np.zeros((NUM_CLASSES,Hp,Wp), np.float32); wsum = np.zeros((Hp,Wp), np.float32)
    gw = gauss2d(S, S/4.0)
    wins = [(y,x) for y in range(0,Hp-S+1,STRIDE) for x in range(0,Wp-S+1,STRIDE)]
    for i in range(0,len(wins),batch):
        bs = [xp[:,:,y:y+S,x:x+S] for y,x in wins[i:i+batch]]
        bx = torch.cat(bs).to("cuda")
        with torch.amp.autocast("cuda"):
            p = torch.softmax(model(bx),1).float()
        p = p.cpu().numpy()
        for j,(y,x) in enumerate(wins[i:i+batch]):
            prob[:,y:y+S,x:x+S] += p[j]*gw; wsum[y:y+S,x:x+S] += gw
    wsum[wsum==0]=1
    return (prob/wsum).argmax(0).astype(np.uint8)

def build(ckpt):
    m = MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256],
                  transformer_layers=2, transformer_heads=4)
    m.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
    return m.to("cuda").eval()

names = sorted(set(os.listdir(f"{CLEAN}/images")) & set(os.listdir(f"{CLEAN}/masks")))
# 选3个含多类别的 tile
cand = []
for n in names:
    m = REMAP[np.array(Image.open(f"{CLEAN}/masks/{n}"))]
    v = m[m!=255]
    if len(np.unique(v)) >= 4 and v.size > 5000: cand.append(n)
    if len(cand) >= 12: break
show = [cand[0], cand[len(cand)//2], cand[-1]] if len(cand)>=3 else names[:3]
print("展示 tiles:", show, flush=True)

models = [("mssact_full60_best.pt","旧模型 957张"), ("full_all_best.pt","全量 2257张"),
          ("post_all_best.pt","后训练")]
preds = {}
for ckpt, label in models:
    p = f"{CKPT}/{ckpt}"
    if not os.path.exists(p): print("SKIP", ckpt); continue
    m = build(p)
    preds[label] = {}
    for n in show:
        img = np.array(Image.open(f"{CLEAN}/images/{n}").convert("RGB"))
        preds[label][n] = infer_tile(m, img)
    del m; torch.cuda.empty_cache()
    print(f"  {label} done", flush=True)

for n in show:
    img = np.array(Image.open(f"{CLEAN}/images/{n}").convert("RGB"))
    gt = REMAP[np.array(Image.open(f"{CLEAN}/masks/{n}"))]
    gt_rgb = PALETTE[np.where(gt==255,0,gt)]
    gt_rgb = np.where(gt[...,None]==255, IGNORE_COLOR, gt_rgb)
    cols = [("影像", img), ("真值", gt_rgb)] + [(lbl, PALETTE[preds[lbl][n]]) for lbl in preds]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4*len(cols), 4.1))
    for ax,(lbl,arr) in zip(axes, cols):
        ax.imshow(arr); ax.set_title(lbl, fontsize=10); ax.axis("off")
    handles = [Patch(color=PALETTE[i]/255, label=NAMES[i]) for i in range(7)]
    handles.append(Patch(color=IGNORE_COLOR/255, label="未标注"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"测试集预测对比 (未参与训练): {n}", fontsize=12)
    plt.tight_layout(rect=[0,0.07,1,1]); plt.savefig(f"{FIG}/fig_test_showcase_{n}", dpi=140); plt.close()
    print(f"  saved fig_test_showcase_{n}", flush=True)
print("DONE", flush=True)
