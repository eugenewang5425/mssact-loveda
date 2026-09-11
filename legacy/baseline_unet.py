"""轻量 U-Net 基线（同一数据管线），用于判断 MSSACT-Net 训练问题是否模型特有"""
import os, sys, time, json
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo-local models/
sys.path.insert(0, paths.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PatchDS, compute_stats, evaluate, PATCH_DIR, NUM_CLASSES, IGNORE, DEVICE

CKPT_DIR = paths.CKPT
EPOCHS, BATCH, LR = 12, 8, 1e-3
LIMIT_TRAIN, LIMIT_VAL = 3000, 600

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=7, base=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.b1, self.b2, self.b3, self.b4 = blk(in_ch,base), blk(base,base*2), blk(base*2,base*4), blk(base*4,base*8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.d3, self.d2, self.d1 = blk(base*8,base*4), blk(base*4,base*2), blk(base*2,base)
        self.head = nn.Conv2d(base, num_classes, 1)
    def forward(self, x):
        x1 = self.b1(x); x2 = self.b2(self.pool(x1)); x3 = self.b3(self.pool(x2)); x4 = self.b4(self.pool(x3))
        y3 = self.d3(torch.cat([self.up3(x4), x3], 1))
        y2 = self.d2(torch.cat([self.up2(y3), x2], 1))
        y1 = self.d1(torch.cat([self.up1(y2), x1], 1))
        return self.head(y1)

@torch.no_grad()
def eval_conf(model, loader):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, lab in loader:
        pred = model(img.to(DEVICE)).argmax(1).cpu()
        v = lab != IGNORE
        conf += torch.bincount((lab[v]*NUM_CLASSES+pred[v]).cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    oa = (conf.diag().sum()/conf.sum()).item()
    pe = (conf.sum(1)*conf.sum(0)).sum()/max(conf.sum().item(),1)**2
    kappa = (oa-pe)/(1-pe)
    f1s = [2*conf[c,c].item()/max(2*conf[c,c].item()+conf[:,c].sum().item()+conf[c,:].sum().item()-2*conf[c,c].item(),1) for c in range(NUM_CLASSES)]
    return oa, kappa, float(np.mean(f1s))

def main():
    mean, std = compute_stats()
    tr = DataLoader(PatchDS("train", mean, std, LIMIT_TRAIN), batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(PatchDS("val", mean, std, LIMIT_VAL), batch_size=BATCH, num_workers=2, pin_memory=True)
    model = UNet().to(DEVICE)
    counts = torch.zeros(NUM_CLASSES)
    for f in sorted(os.listdir(f"{PATCH_DIR}/train"))[:LIMIT_TRAIN]:
        lab = np.load(f"{PATCH_DIR}/train/{f}")["label"]
        for c in range(NUM_CLASSES): counts[c] += (lab==c).sum()
    w = 1.0/torch.sqrt(counts/counts.sum()+1e-6); w = (w/w.mean()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = []
    for ep in range(1, EPOCHS+1):
        model.train(); t0=time.time(); tot=0
        for i,(img,lab) in enumerate(tr):
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(img), lab, weight=w, ignore_index=IGNORE)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step()
        oa, k, mf1 = eval_conf(model, va)
        print(f"[epoch {ep}/{EPOCHS}] loss={tot/len(tr):.4f} OA={oa:.4f} Kappa={k:.4f} mF1={mf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        hist.append(dict(epoch=ep, loss=tot/len(tr), oa=oa, kappa=k, mf1=mf1))
        torch.save(model.state_dict(), f"{CKPT_DIR}/unet_best.pt" if k>=max(h['kappa'] for h in hist) else f"{CKPT_DIR}/unet_last.pt")
    json.dump(hist, open(f"{CKPT_DIR}/unet_history.json","w"), indent=2)
    print("best Kappa:", max(h['kappa'] for h in hist))

if __name__ == "__main__":
    main()
