# -*- coding: utf-8 -*-
"""可比性验证: 新管线(memmap) 与 旧管线(PNG) 在同一模型/数据上的训练轨迹是否一致"""

# --- 路径引导（本文件位于子目录 verify/，仓库根为上一级）---
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
sys.path.insert(0, _HERE)
import paths
import torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, paths.REPO)
from train_v3 import LoveDADataset, compute_stats, REMAP
from models.msscactnet import MSSACTNet

ROOT = paths.DATA_NEWSPLIT2
DEVICE = "cuda"
BATCH, EPOCHS = 8, 3

def run(tag, fast, workers):
    mean, std = compute_stats(root=ROOT)
    kw = dict(batch_size=BATCH, shuffle=True, drop_last=True, pin_memory=True)
    kw.update(num_workers=workers, **({"persistent_workers": True, "prefetch_factor": 4} if workers else {}))
    tr = DataLoader(LoveDADataset("train", mean, std, train=True, root=ROOT, fast=fast), **kw)
    va = DataLoader(LoveDADataset("val", mean, std, train=False, n_val_patches=4, root=ROOT, fast=fast),
                    batch_size=BATCH, shuffle=False, num_workers=(min(4,workers) if workers else 0))
    torch.manual_seed(42)
    model = MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256],
                      transformer_layers=2, transformer_heads=4).to(DEVICE)
    cw = torch.ones(7, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=EPOCHS*len(tr), pct_start=0.06)
    scaler = torch.amp.GradScaler()
    out = []
    for ep in range(1, EPOCHS+1):
        model.train(); tr.dataset.epoch_seed = ep
        tot = 0.0
        for img, lab in tr:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.cross_entropy(model(img), lab, weight=cw, ignore_index=255)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        # 验证 (与 train_one 一致: 4角 patch, 无EMA, 直接模型)
        model.eval(); conf = torch.zeros(7,7,dtype=torch.long)
        with torch.no_grad():
            for img, lab in va:
                img, lab = img.to(DEVICE), lab.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    pred = model(img).argmax(1)
                v = lab != 255
                conf += torch.bincount((lab[v]*7+pred[v]).cpu(), minlength=49).reshape(7,7)
        c = conf.numpy().astype(float)
        oa = c.trace()/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
        k = (oa-pe)/(1-pe)
        out.append(dict(epoch=ep, loss=round(tot/len(tr),4), oa=round(float(oa),4), kappa=round(float(k),4)))
        print(f"  [{tag}] ep{ep}: loss={tot/len(tr):.4f} OA={oa:.4f} Kappa={k:.4f}", flush=True)
        model.train()
    del model; torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    print("=== 可比性验证 (同一模型/数据/种子, 3 轮) ===", flush=True)
    a = run("旧 PNG w=0", False, 0)
    b = run("新 memmap w=8", True, 8)
    json.dump({"png": a, "memmap": b}, open(os.path.join(paths.REPO,"artifacts","bench_parity.json"),"w"), indent=1)
    print("\n=== 对比 ===")
    print(f"{'轮':>3s} {'旧Kappa':>9s} {'新Kappa':>9s} {'差':>8s}")
    for x, y in zip(a, b):
        print(f"{x['epoch']:3d} {x['kappa']:9.4f} {y['kappa']:9.4f} {y['kappa']-x['kappa']:+8.4f}")
