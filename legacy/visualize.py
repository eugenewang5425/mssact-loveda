"""可视化验证集预测：原图 | 真值 | 预测 三联图"""
import os, sys
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo-local models/
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from train import PatchDS, compute_stats, NUM_CLASSES, IGNORE, DEVICE, PATCH_DIR

CLASS_NAMES = ["背景", "建筑", "道路", "水域", "裸地", "森林", "农田"]
# 调色板: 背景/建筑/道路/水/裸地/森林/农田
PALETTE = np.array([[128,128,128],[255,0,0],[255,255,0],[0,0,255],[160,82,45],[0,128,0],[255,165,0]], np.uint8)
CKPT = os.path.join(paths.REPO, "checkpoints/mssact_v2_best.pt")
OUT = os.path.join(paths.REPO, "vis")

@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    mean, std = compute_stats()
    va = DataLoader(PatchDS("val", mean, std, 12), batch_size=4, num_workers=0)
    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES, embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    n = 0
    for img, lab in va:
        pred = model(img.to(DEVICE)).argmax(1).cpu()
        for b in range(img.shape[0]):
            x = img[b].permute(1,2,0).numpy()*std.view(3,1,1).numpy().squeeze()[None,None,:] if False else None
            im = img[b].permute(1,2,0).numpy()
            im = im*np.array([0.1652,0.1433,0.1333]) + np.array([0.2822,0.3082,0.2812])
            im = np.clip(im,0,1)
            gt = PALETTE[np.where(lab[b].numpy()==255, 0, lab[b].numpy())]
            pr = PALETTE[pred[b].numpy()]
            panel = np.concatenate([im, gt/255.0, pr/255.0], axis=1)
            Image.fromarray((panel*255).astype(np.uint8)).save(f"{OUT}/val_{n:02d}.png")
            n += 1
    print(f"saved {n} panels to {OUT}")

if __name__ == "__main__":
    main()
