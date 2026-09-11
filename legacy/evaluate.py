"""评估 LoveDA 训练结果：OA / Kappa / 各类F1 / 混淆矩阵，输出 JSON + Markdown"""
import os, sys, json
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo-local models/
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from train import PatchDS, compute_stats, PATCH_DIR, NUM_CLASSES, IGNORE, DEVICE

CLASS_NAMES = ["背景", "建筑", "道路", "水域", "裸地", "森林", "农田"]
CKPT = os.path.join(paths.REPO, "checkpoints/mssact_v2_best.pt")
OUT = os.path.join(paths.REPO, "eval")

@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    mean, std = compute_stats()
    va = DataLoader(PatchDS("val", mean, std, 600), batch_size=4, shuffle=False, num_workers=0)
    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES, embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, lab in va:
        img, lab = img.to(DEVICE), lab.to(DEVICE)
        pred = model(img).argmax(1)
        valid = lab != IGNORE
        conf += torch.bincount((lab[valid]*NUM_CLASSES+pred[valid]).cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    conf = conf.numpy()
    oa = np.trace(conf)/conf.sum()
    pe = (conf.sum(1)*conf.sum(0)).sum()/conf.sum()**2
    kappa = (oa-pe)/(1-pe)
    rows = []
    for c in range(NUM_CLASSES):
        tp = conf[c,c]; fp = conf[:,c].sum()-tp; fn = conf[c,:].sum()-tp
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-9)
        pa, ua = r, p
        rows.append(dict(cls=CLASS_NAMES[c], pa=float(pa), ua=float(ua), f1=float(f1), support=int(conf[c].sum())))
    mf1 = float(np.mean([r["f1"] for r in rows]))
    res = dict(oa=float(oa), kappa=float(kappa), macro_f1=mf1, per_class=rows,
               confusion_matrix=conf.tolist(), classes=CLASS_NAMES)
    json.dump(res, open(f"{OUT}/results.json","w"), indent=2, ensure_ascii=False)
    md = ["# LoveDA 验证集评估结果\n",
          f"- **OA = {oa:.4f}**, **Kappa = {kappa:.4f}**, **mF1 = {mf1:.4f}**\n",
          "| 类别 | PA(召回) | UA(精度) | F1 | 样本像素 |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cls']} | {r['pa']:.4f} | {r['ua']:.4f} | {r['f1']:.4f} | {r['support']} |")
    md.append("\n## 混淆矩阵 (行=真实, 列=预测)\n")
    md.append("| | " + " | ".join(CLASS_NAMES) + " |")
    md.append("|---|" + "---|"*NUM_CLASSES)
    for i, name in enumerate(CLASS_NAMES):
        md.append(f"| {name} | " + " | ".join(str(int(v)) for v in conf[i]) + " |")
    open(f"{OUT}/results.md","w",encoding="utf-8").write("\n".join(md))
    print(f"OA={oa:.4f} Kappa={kappa:.4f} mF1={mf1:.4f}")
    print("per-class F1:", {r['cls']: round(r['f1'],3) for r in rows})

if __name__ == "__main__":
    main()
