# -*- coding: utf-8 -*-
"""数据管线基准测试: 旧(PNG, workers=0) vs 新(memmap, workers=8)
每个配置跑 2 轮 (同一轻量模型), 比较每轮耗时与 GPU 利用率
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, paths.REPO)
from train_v3 import LoveDADataset, compute_stats
from models.msscactnet import MSSACTNet

ROOT = paths.DATA_NEWSPLIT2
DEVICE = "cuda"
BATCH = 8
EPOCHS = 2

def run(tag, fast, workers):
    mean, std = compute_stats(root=ROOT)
    kw = dict(batch_size=BATCH, shuffle=True, drop_last=True, pin_memory=True)
    if workers > 0:
        kw.update(num_workers=workers, persistent_workers=True, prefetch_factor=4)
    else:
        kw.update(num_workers=0)
    tr = DataLoader(LoveDADataset("train", mean, std, train=True, root=ROOT, fast=fast), **kw)
    va = DataLoader(LoveDADataset("val", mean, std, train=False, n_val_patches=4, root=ROOT, fast=fast),
                    batch_size=BATCH, shuffle=False, num_workers=(min(4, workers) if workers else 0))
    torch.manual_seed(42)
    model = MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256],
                      transformer_layers=2, transformer_heads=4).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler()
    # 数据加载计时
    t_load = 0.0; n_batches = 0
    ep_times = []
    for ep in range(EPOCHS):
        tr.dataset.epoch_seed = ep
        model.train()
        t0 = time.time(); t_data = 0.0
        it = iter(tr)
        while True:
            td0 = time.time()
            try: img, lab = next(it)
            except StopIteration: break
            t_data += time.time() - td0
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(img), lab, ignore_index=255)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            n_batches += 1
        torch.cuda.synchronize()
        ep_times.append(time.time() - t0)
        t_load += t_data
    # GPU 峰值
    peak = torch.cuda.max_memory_allocated()/1e9
    del model; torch.cuda.empty_cache()
    avg_ep = sum(ep_times)/len(ep_times)
    print(f"  [{tag:22s}] 每轮 {avg_ep:6.1f}s | 数据等待 {t_load/EPOCHS:5.1f}s/轮 "
          f"({t_load/n_batches*1000:5.0f}ms/batch) | 显存峰值 {peak:.2f}GB", flush=True)
    return dict(tag=tag, per_epoch=round(avg_ep,1), data_wait_per_epoch=round(t_load/EPOCHS,1),
                ms_per_batch=round(t_load/n_batches*1000,1), peak_gb=round(peak,2), batches=n_batches)

if __name__ == "__main__":
    res = []
    print("=== 基准测试 (2 轮/配置, batch=8, 256²) ===", flush=True)
    res.append(run("旧: PNG + workers=0", False, 0))
    res.append(run("新: memmap + workers=8", True, 8))
    json.dump(res, open(os.path.join(paths.REPO, "bench_pipeline.json"), "w"), indent=1)
    a, b = res[0], res[1]
    print(f"\n加速比: {a['per_epoch']/b['per_epoch']:.2f}x "
          f"(每轮 {a['per_epoch']}s -> {b['per_epoch']}s, 省 {a['per_epoch']-b['per_epoch']:.0f}s/轮)")
    print(f"数据等待: {a['data_wait_per_epoch']}s -> {b['data_wait_per_epoch']}s")
