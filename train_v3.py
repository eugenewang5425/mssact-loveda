"""LoveDA 最终训练 v3：
- 直接读原图，训练时每步随机 256 裁剪（不同 epoch 看到不同位置，等价强增强）
- 验证用每图4角固定 patch（位置稳定，可复现）
- 收敛协议: OneCycle(warmup+余弦) + EMA + 早停(patience=20, 监控EMA验证Kappa), 上限120轮
- 指标: OA / Kappa / mF1 / mIoU，历史正确序列化
"""
import os, sys, time, json, copy
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo-local models/
sys.path.insert(0, paths.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.msscactnet import MSSACTNet

ROOT = paths.DATA_ROOT
CKPT_DIR = paths.CKPT
NUM_CLASSES, IGNORE = 7, 255
CROP = int(os.environ.get("CROP", "256"))
GFMATCH = CROP == 128   # LoveDA整图降采样1024->128, GSD 2.4m 与GF-1一致
BATCH = {"128":16, "256":8, "512":2, "1024":1}[str(CROP)]
LR, PCT_WARMUP = 2e-4, 0.06
MAX_EPOCHS, PATIENCE = {"128":(120,20), "256":(120,20), "512":(80,20), "1024":(50,20)}[str(CROP)]
TAG = "s128gf" if GFMATCH else f"s{CROP}"
VAL_PATCHES_PER_IMG = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["背景", "建筑", "道路", "水域", "裸地", "森林", "农田"]

# 官方 LoveDA 标签约定: 1=背景 2=建筑 3=道路 4=水域 5=裸地 6=森林 7=农田; 0=no-data(必须忽略)
# 映射: 1..7 -> 0..6; 0 及一切其他值 -> 255 (ignore, 官方协议)
REMAP = np.full(256, 255, dtype=np.uint8)
for s in range(1, 8): REMAP[s] = s - 1          # 1..7 -> 0..6; 0/8+ -> 255

# [BUGFIX 2026-09-10] 旧实现为 REMAP=np.arange(256,dtype=np.uint8)*255, 其上 REMAP[0]=0,
# 导致 no-data(值0, 占2.6-3.1%) 被错误当作"背景"类参与训练与评估 (评估虚高约0.007-0.009 Kappa)

class LoveDADataset(Dataset):
    """训练: 每epoch随机裁剪1个patch/图; 验证: 4个固定角patch"""
    def __init__(self, split, mean, std, train, n_val_patches=4, epoch_seed=None, crop=256,
                 root=None, fast=False, fast_dir=None, center_crop=False):
        """fast=True: 使用预解码 memmap（fast_dataset/），消除 PNG 解码开销；
        数据内容与 PNG 管线完全一致（无插值），仅解码时机不同。
        center_crop=True: 训练时固定取中心 patch（默认 False = 每轮随机位置），
        用于"随机裁剪 vs 固定裁剪"的训练策略对比。"""
        self.crop = crop
        self.center_crop = center_crop
        self.fast = fast
        # 注意: fast 后端固定对应主数据集（newsplit2）; 若 root 指向其他目录
        # （如数据阶梯子集），调用方须显式传 fast=False
        if fast:
            fd = fast_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_dataset")
            self._fast_dir = fd
            self._split = split
            self._imgs = None      # 懒加载: worker 首次访问时才打开 memmap
            self._msks = None
            _n = json.load(open(os.path.join(fd, "index.json")))[split]
            self.names = list(range(_n))
        else:
            base = root if root is not None else ROOT
            self.img_dir = f"{base}/{split}/images"
            self.msk_dir = f"{base}/{split}/masks"
            self.names = sorted(set(os.listdir(self.img_dir)) & set(os.listdir(self.msk_dir)))
        self.mean, self.std, self.train = mean, std, train
        self.n_val = n_val_patches
        self.epoch_seed = epoch_seed
    def __len__(self):
        return len(self.names) * (1 if (self.train or GFMATCH) else self.n_val)
    def _load(self, i):
        if self.fast:
            if self._imgs is None:      # 每个 worker 独立打开只读 memmap
                self._imgs = np.load(os.path.join(self._fast_dir, f"{self._split}_images.npy"), mmap_mode="r")
                self._msks = np.load(os.path.join(self._fast_dir, f"{self._split}_masks.npy"), mmap_mode="r")
            return np.asarray(self._imgs[i]), np.asarray(self._msks[i])
        n = self.names[i]
        img = Image.open(f"{self.img_dir}/{n}").convert("RGB")
        msk = Image.open(f"{self.msk_dir}/{n}")
        if GFMATCH:   # 整图降采样到128, GSD 0.3m->2.4m 与GF-1匹配
            img = img.resize((128,128), Image.BILINEAR)
            msk = msk.resize((128,128), Image.NEAREST)
        img = np.array(img)
        msk = REMAP[np.array(msk)]
        return img, msk
    def _norm(self, img):
        # np.asarray(..., float32) 在 dtype 不同时创建可写副本, 避免 memmap 只读告警
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return (x - self.mean) / self.std
    @staticmethod
    def _geom(img, lab):
        """几何增强（回退版: 使用全局 RNG, 与原管线行为一致）"""
        k = int(torch.randint(0, 4, (1,)).item())
        if k: img, lab = torch.rot90(img, k, dims=[1,2]), torch.rot90(lab, k, dims=[0,1])
        if torch.rand(1) < .5: img, lab = torch.flip(img,[2]), torch.flip(lab,[1])
        if torch.rand(1) < .5: img, lab = torch.flip(img,[1]), torch.flip(lab,[0])
        return img.contiguous(), lab.contiguous()
    def __getitem__(self, idx):
        if self.train:
            i = idx
            g = torch.Generator().manual_seed(self.epoch_seed*100003 + idx) if self.epoch_seed is not None else None
            img, msk = self._load(i)
            H, W = msk.shape
            S = self.crop
            if S >= H:
                y = x = 0
            elif self.center_crop:
                y, x = (H - S) // 2, (W - S) // 2
            else:
                y = torch.randint(0, H-S+1, (1,), generator=g).item() if g else torch.randint(0, H-S+1, (1,)).item()
                x = torch.randint(0, W-S+1, (1,), generator=g).item() if g else torch.randint(0, W-S+1, (1,)).item()
            img, lab = img[y:y+S, x:x+S], msk[y:y+S, x:x+S]
            img_t, lab_t = self._geom(self._norm(img), torch.from_numpy(lab.astype(np.int64)))
            return img_t, lab_t
        else:
            i, p = divmod(idx, 1 if GFMATCH else self.n_val)
            img, msk = self._load(i)
            H, W = msk.shape
            S = self.crop
            if S >= H:
                y, x = 0, 0
            else:
                ys = [0, H-S]; xs = [0, W-S]
                y, x = ys[p//2], xs[p%2]
            return self._norm(img[y:y+S, x:x+S]), torch.from_numpy(msk[y:y+S, x:x+S].astype(np.int64))

@torch.no_grad()
def compute_stats(root=None):
    base = root if root is not None else ROOT
    files = sorted(os.listdir(f"{base}/train/images"))[:150]
    s = np.zeros(3); s2 = np.zeros(3); n = 0
    for f in files:
        x = np.array(Image.open(f"{base}/train/images/{f}").convert("RGB")).astype(np.float64)/255.0
        s += x.sum((0,1)); s2 += (x**2).sum((0,1)); n += x.shape[0]*x.shape[1]
    mean = s/n; std = np.sqrt(s2/n-mean**2)
    return torch.tensor(mean).view(3,1,1).float(), torch.tensor(std).view(3,1,1).float()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for es, ms in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if es.dtype.is_floating_point: es.mul_(self.decay).add_(ms.detach(), alpha=1-self.decay)
            else: es.copy_(ms)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, lab in loader:
        img, lab = img.to(DEVICE), lab.to(DEVICE)
        with torch.amp.autocast("cuda"):
            out = model(img)
        pred = out.argmax(1)
        v = lab != IGNORE
        conf += torch.bincount((lab[v]*NUM_CLASSES+pred[v]).cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    c = conf.cpu().numpy().astype(np.float64)
    oa = np.trace(c)/c.sum()
    pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
    kappa = (oa-pe)/(1-pe)
    f1s, ious = [], []
    for k in range(NUM_CLASSES):
        tp = c[k,k]; fp = c[:,k].sum()-tp; fn = c[k,:].sum()-tp
        f1s.append(2*tp/max(2*tp+fp+fn,1))
        ious.append(tp/max(tp+fp+fn,1))
    return float(oa), float(kappa), float(np.mean(f1s)), float(np.mean(ious)), [float(x) for x in f1s], [float(x) for x in ious]

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    mean, std = compute_stats()
    print("mean/std:", [round(x,4) for x in mean.flatten().tolist()], [round(x,4) for x in std.flatten().tolist()], flush=True)
    tr = DataLoader(LoveDADataset("train", mean, std, train=True), batch_size=BATCH, shuffle=True,
                    num_workers=0, pin_memory=True, drop_last=True)
    va = DataLoader(LoveDADataset("val", mean, std, train=False, n_val_patches=VAL_PATCHES_PER_IMG),
                    batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    print(f"train imgs {len(tr.dataset)} / val patches {len(va.dataset)}", flush=True)

    torch.manual_seed(42)
    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES,
                      embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4).to(DEVICE)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    ema = EMA(model)

    # 类别权重（由抽样统计）
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in sorted(os.listdir(f"{ROOT}/train/masks"))[::5]:
        m = REMAP[np.array(Image.open(f"{ROOT}/train/masks/{f}"))]
        counts += np.bincount(m[m!=255], minlength=NUM_CLASSES)
    w = 1.0/np.sqrt(counts/counts.sum()+1e-6); w = w/w.mean()
    cw = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    print("class weights:", [round(x,3) for x in w.tolist()], flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    total_steps = MAX_EPOCHS*len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps, pct_start=PCT_WARMUP)
    scaler = torch.amp.GradScaler()

    hist, best_k, bad = [], -1.0, 0
    t_start = time.time()
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        tr_ds = tr.dataset; tr_ds.epoch_seed = ep
        t0 = time.time(); tot = 0
        for img, lab in tr:
            img, lab = img.to(DEVICE, non_blocking=True), lab.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.cross_entropy(model(img), lab, weight=cw, ignore_index=IGNORE)
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model); tot += loss.item()
        oa, k, mf1, miou, f1s, ious = evaluate(ema.shadow, va)
        hist.append(dict(epoch=ep, loss=tot/len(tr), oa=oa, kappa=k, mf1=mf1, miou=miou, f1=f1s, iou=ious,
                         seconds=round(time.time()-t0,1)))
        flag = ""
        if k > best_k:
            best_k, bad = k, 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/v3_{TAG}_best_raw.pt")
            torch.save(ema.shadow.state_dict(), f"{CKPT_DIR}/v3_{TAG}_best.pt")
            flag = " *"
        else:
            bad += 1
        print(f"[ep {ep:3d}] loss={tot/len(tr):.4f} OA={oa:.4f} Kappa={k:.4f} mF1={mf1:.4f} mIoU={miou:.4f} "
              f"best_k={best_k:.4f} bad={bad} ({time.time()-t0:.0f}s, total {(time.time()-t_start)/60:.0f}min){flag}", flush=True)
        json.dump(hist, open(f"{CKPT_DIR}/v3_{TAG}_history.json","w"), indent=1)  # 每轮落盘, 全为纯float
        if bad >= PATIENCE:
            print(f"early stop at epoch {ep} (no val Kappa improvement for {PATIENCE} epochs)", flush=True)
            break
    print(f"BEST Kappa={best_k:.4f}; curves at {CKPT_DIR}/v3_history.json", flush=True)

if __name__ == "__main__":
    main()
