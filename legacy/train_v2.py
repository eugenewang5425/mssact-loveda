"""MSSACT-Net 稳定训练 v2：warmup + EMA + AMP NaN防护 + 类别加权CE"""
import os, sys, time, json, copy
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo-local models/
sys.path.insert(0, paths.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.msscactnet import MSSACTNet
from train import PatchDS, compute_stats, evaluate, compute_class_weights, \
    PATCH_DIR, CKPT_DIR, NUM_CLASSES, IGNORE, DEVICE, EPOCHS

BATCH = 4
LR, PEAK_EPOCH_WARMUP = 1.5e-4, 3
WD = 0.01
LIMIT_TRAIN, LIMIT_VAL = 3000, 600
EMA_DECAY = 0.999

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for es, ms in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if es.dtype.is_floating_point:
                es.mul_(d).add_(ms.detach(), alpha=1-d)
            else:
                es.copy_(ms)

def main():
    mean, std = compute_stats()
    tr = DataLoader(PatchDS("train", mean, std, LIMIT_TRAIN), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    va = DataLoader(PatchDS("val", mean, std, LIMIT_VAL), batch_size=BATCH, num_workers=0, pin_memory=True)
    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES,
                      embed_dims=[32, 64, 128, 256],
                      transformer_layers=2, transformer_heads=4).to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    cw = compute_class_weights()
    print("class weights:", [round(x,3) for x in cw.tolist()], flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS*steps_per_epoch,
                                                pct_start=PEAK_EPOCH_WARMUP/EPOCHS, anneal_strategy='cos')
    ema = EMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler()
    hist = []
    for ep in range(1, EPOCHS+1):
        model.train(); t0=time.time(); tot=0; skipped=0
        for i,(img,lab) in enumerate(tr):
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = nn.functional.cross_entropy(out, lab, weight=cw, ignore_index=IGNORE)
            if not torch.isfinite(loss):
                skipped += 1; opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            tot += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep} step {i}/{steps_per_epoch} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        vl, oa, k, f1s, conf = evaluate(ema.shadow, va)   # 用EMA权重评估
        print(f"[epoch {ep}/{EPOCHS}] train_loss={tot/max(len(tr)-skipped,1):.4f} val_loss={vl:.4f} OA={oa:.4f} Kappa={k:.4f} mF1={np.mean(f1s):.4f} skip={skipped} ({time.time()-t0:.0f}s)", flush=True)
        hist.append(dict(epoch=ep, train_loss=tot/len(tr), val_loss=vl, oa=oa, kappa=k, f1=f1s))
        if k >= max(h['kappa'] for h in hist[:-1] or [hist[-1]]):
            torch.save(ema.shadow.state_dict(), f"{CKPT_DIR}/mssact_v2_best.pt")
    json.dump(hist, open(f"{CKPT_DIR}/mssact_v2_history.json","w"), indent=2)
    print("best Kappa:", max(h['kappa'] for h in hist))

if __name__ == "__main__":
    main()
