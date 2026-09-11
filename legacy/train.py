"""LoveDA 上训练 MSSACT-Net（真实数据、真实结果）"""
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo-local models/
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet

PATCH_DIR = os.path.join(paths.REPO, "patches")
CKPT_DIR = paths.CKPT
NUM_CLASSES, IGNORE = 7, 255
EPOCHS, BATCH, LR = 18, 8, 2e-4
LIMIT_TRAIN, LIMIT_VAL = 3000, 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.4145, 0.4223, 0.4101]).view(3,1,1)   # LoveDA 统计，近似值，训练前重新计算
STD  = torch.tensor([0.1992, 0.1876, 0.1884]).view(3,1,1)

class PatchDS(Dataset):
    def __init__(self, split, mean, std, limit=None):
        self.dir = f"{PATCH_DIR}/{split}"
        self.files = sorted(os.listdir(self.dir))[:limit]
        self.mean, self.std = mean, std
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(f"{self.dir}/{self.files[i]}")
        img = torch.from_numpy(d["img"]).float().permute(2,0,1) / 255.0
        img = (img - self.mean) / self.std
        lab = torch.from_numpy(d["label"].astype(np.int64))
        # 无插值几何增强（训练集）: 随机rot90 + 双向翻转
        if self.dir.endswith("train"):
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img, lab = torch.rot90(img, k, dims=[1,2]), torch.rot90(lab, k, dims=[0,1])
            if torch.rand(1).item() < 0.5:
                img, lab = torch.flip(img, dims=[2]), torch.flip(lab, dims=[1])
            if torch.rand(1).item() < 0.5:
                img, lab = torch.flip(img, dims=[1]), torch.flip(lab, dims=[0])
        return img.contiguous(), lab.contiguous()

def compute_stats():
    files = sorted(os.listdir(f"{PATCH_DIR}/train"))[:200]
    s = np.zeros(3); s2 = np.zeros(3); n = 0
    for f in files:
        d = np.load(f"{PATCH_DIR}/train/{f}")
        x = d["img"].astype(np.float64)/255.0
        s += x.sum((0,1)); s2 += (x**2).sum((0,1)); n += x.shape[0]*x.shape[1]
    mean = s/n; std = np.sqrt(s2/n - mean**2)
    return torch.tensor(mean).view(3,1,1).float(), torch.tensor(std).view(3,1,1).float()

@torch.no_grad()
def compute_class_weights():
    """按训练patch像素频率的逆平方根计算类别权重，防止多数类坍缩"""
    counts = torch.zeros(NUM_CLASSES)
    files = sorted(os.listdir(f"{PATCH_DIR}/train"))[:LIMIT_TRAIN]
    for f in files:
        lab = np.load(f"{PATCH_DIR}/train/{f}")["label"]
        for c in range(NUM_CLASSES):
            counts[c] += (lab == c).sum()
    w = 1.0 / torch.sqrt(counts / counts.sum() + 1e-6)
    return (w / w.mean()).to(DEVICE)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    loss_sum, n = 0.0, 0
    for img, lab in loader:
        img, lab = img.to(DEVICE), lab.to(DEVICE)
        with torch.amp.autocast("cuda"):
            out = model(img)
        loss = nn.functional.cross_entropy(out.float(), lab, ignore_index=IGNORE)
        loss_sum += loss.item()*lab.numel(); n += lab.numel()
        pred = out.argmax(1)
        valid = lab != IGNORE
        idx = lab[valid]*NUM_CLASSES + pred[valid]
        conf += torch.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES).cpu()
    oa = (conf.diag().sum()/conf.sum()).item()
    pe = (conf.sum(1)*conf.sum(0)).sum() / max(conf.sum().item(),1)**2
    kappa = (oa-pe)/(1-pe)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = conf[c,c].item(); fp = conf[:,c].sum().item()-tp; fn = conf[c,:].sum().item()-tp
        f1s.append(2*tp/max(tp*2+fp+fn,1))
    return loss_sum/n, oa, kappa, f1s, conf

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    mean, std = compute_stats()
    print("dataset mean/std:", mean.flatten().tolist(), std.flatten().tolist())
    tr = DataLoader(PatchDS("train", mean, std, LIMIT_TRAIN), batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(PatchDS("val", mean, std, LIMIT_VAL), batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    print(f"train {len(tr.dataset)} / val {len(va.dataset)} patches")

    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    cw = compute_class_weights()
    print("class weights:", cw.tolist())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler()
    hist = []
    for ep in range(1, EPOCHS+1):
        model.train(); t0=time.time(); tot=0
        for i,(img,lab) in enumerate(tr):
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = nn.functional.cross_entropy(out, lab, weight=cw, ignore_index=IGNORE)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item()
            if i % 25 == 0:
                print(f"  ep{ep} step {i}/{len(tr)} loss {loss.item():.4f} ({(i+1)/(time.time()-t0):.2f} it/s)", flush=True)
        sched.step()
        vl, oa, k, f1s, conf = evaluate(model, va)
        print(f"[epoch {ep}/{EPOCHS}] train_loss={tot/len(tr):.4f} val_loss={vl:.4f} OA={oa:.4f} Kappa={k:.4f} mF1={np.mean(f1s):.4f} ({time.time()-t0:.0f}s)")
        hist.append(dict(epoch=ep, train_loss=tot/len(tr), val_loss=vl, oa=oa, kappa=k, f1=f1s))
        torch.save(model.state_dict(), f"{CKPT_DIR}/best.pt" if k>=max(h['kappa'] for h in hist) else f"{CKPT_DIR}/last.pt")
    json.dump(hist, open(f"{CKPT_DIR}/history.json","w"), indent=2)
    print("best val OA:", max(h['oa'] for h in hist))

if __name__ == "__main__":
    main()
