"""实验矩阵：对比实验(基线) + 消融实验(MSSACT-Net)
对比: U-Net / PSPNet / FCN / DeepLabV3+ / SegFormer(轻量)  vs  MSSACT-Net (light, s256)
消融: MSSACT-Net 各模块开关 (EMR / ECSAM / FPN / Transformer)

统一训练协议(train_v3.py 的 s256 设置 + EMA + 早停), 固定 256px @ 0.3m, 验证集 299×4 patch
"""
import os, sys, time, copy, json, random
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, paths.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v3 import LoveDADataset, compute_stats
from models.msscactnet import MSSACTNet

NUM_CLASSES, IGNORE = 7, 255
DEVICE = "cuda"
CKPT = paths.CKPT

# ============ 基线模型 ============

class UNet(nn.Module):
    def __init__(self, in_ch=3, nc=7, base=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.b1, self.b2, self.b3, self.b4 = blk(in_ch,base), blk(base,base*2), blk(base*2,base*4), blk(base*4,base*8)
        self.pool = nn.MaxPool2d(2)
        self.u3, self.u2, self.u1 = nn.ConvTranspose2d(base*8,base*4,4,2,1), nn.ConvTranspose2d(base*4,base*2,4,2,1), nn.ConvTranspose2d(base*2,base,4,2,1)
        self.d3, self.d2, self.d1 = blk(base*8,base*4), blk(base*4,base*2), blk(base*2,base)
        self.head = nn.Conv2d(base, nc, 1)
    def forward(self, x):
        x1=self.b1(x); x2=self.b2(self.pool(x1)); x3=self.b3(self.pool(x2)); x4=self.b4(self.pool(x3))
        y3=self.d3(torch.cat([self.u3(x4),x3],1)); y2=self.d2(torch.cat([self.u2(y3),x2],1))
        y1=self.d1(torch.cat([self.u1(y2),x1],1)); return self.head(y1)

class PSPNet(nn.Module):
    """精简PSPNet: ResNet-18 主干 + 4级金字塔池化"""
    def __init__(self, in_ch=3, nc=7, base=64):
        super().__init__()
        import torchvision.models as tvm
        m = tvm.resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.psp = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(512,64,1,bias=False), nn.GroupNorm(8,64), nn.ReLU(True)) for s in (1,2,3,6)])
        self.fuse = nn.Sequential(nn.Conv2d(512+64*4,256,3,padding=1,bias=False), nn.GroupNorm(8,256), nn.ReLU(True),
                                  nn.Conv2d(256,128,3,padding=1,bias=False), nn.GroupNorm(8,128), nn.ReLU(True))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1,bias=False), nn.GroupNorm(8,64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1,bias=False), nn.GroupNorm(8,32), nn.ReLU(True),
            nn.ConvTranspose2d(32,32,4,2,1,bias=False), nn.GroupNorm(8,32), nn.ReLU(True),
            nn.ConvTranspose2d(32,32,4,2,1,bias=False), nn.GroupNorm(8,32), nn.ReLU(True),
            nn.ConvTranspose2d(32,32,4,2,1,bias=False), nn.GroupNorm(8,32), nn.ReLU(True),
            nn.Conv2d(32,nc,1))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        H, W = x.shape[-2:]
        feats = [x] + [nn.functional.interpolate(p(x), size=(H,W), mode='bilinear', align_corners=False) for p in self.psp]
        x = self.fuse(torch.cat(feats, 1))
        return self.up(x)

class FCN(nn.Module):
    """FCN-32s (ResNet50 backbone)"""
    def __init__(self, in_ch=3, nc=7):
        super().__init__()
        import torchvision.models as tvm
        m = tvm.resnet50(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.head = nn.Conv2d(2048, nc, 1)
        self.up = nn.ConvTranspose2d(nc, nc, 64, 32, 16, bias=False)
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.up(x)

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ (ResNet50 backbone, torch.hub hub)"""
    def __init__(self, in_ch=3, nc=7):
        super().__init__()
        import torchvision.models as tvm
        m = tvm.segmentation.deeplabv3_resnet50(weights=None, num_classes=nc)
        m.backbone.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.m = m
    def forward(self, x): return self.m(x)['out']

class SegFormerLite(nn.Module):
    """SegFormer 简化实现: MiT-B2 风格 (light)"""
    def __init__(self, in_ch=3, nc=7, dims=(32,64,160,256), heads=(1,2,5,8)):
        super().__init__()
        # 简化编码器: 4阶段 ConvNeXt-like blocks + patch merge
        self.patch1 = nn.Conv2d(in_ch, dims[0], 7, 4, 2)
        self.block1 = nn.Sequential(*[nn.Conv2d(dims[0], dims[0], 3, 1, 1, groups=dims[0], bias=False),
                                       nn.BatchNorm2d(dims[0]), nn.GELU(),
                                       nn.Conv2d(dims[0], dims[0], 1), nn.GELU()] * 2)
        self.down2 = nn.Conv2d(dims[0], dims[1], 3, 2, 1)
        self.block2 = nn.Sequential(*[nn.Conv2d(dims[1], dims[1], 3, 1, 1, groups=dims[1], bias=False),
                                       nn.BatchNorm2d(dims[1]), nn.GELU(),
                                       nn.Conv2d(dims[1], dims[1], 1), nn.GELU()] * 2)
        self.down3 = nn.Conv2d(dims[1], dims[2], 3, 2, 1)
        self.block3 = nn.Sequential(*[nn.Conv2d(dims[2], dims[2], 3, 1, 1, groups=dims[2], bias=False),
                                       nn.BatchNorm2d(dims[2]), nn.GELU(),
                                       nn.Conv2d(dims[2], dims[2], 1), nn.GELU()] * 6)
        self.down4 = nn.Conv2d(dims[2], dims[3], 3, 2, 1)
        self.block4 = nn.Sequential(*[nn.Conv2d(dims[3], dims[3], 3, 1, 1, groups=dims[3], bias=False),
                                       nn.BatchNorm2d(dims[3]), nn.GELU(),
                                       nn.Conv2d(dims[3], dims[3], 1), nn.GELU()] * 2)
        # 简化解码: 全 MLP 融合 (SegFormer 核心)
        self.decode = nn.Sequential(
            nn.Conv2d(sum(dims), 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, nc, 1))
    def forward(self, x):
        x1 = self.block1(self.patch1(x))
        x2 = self.block2(self.down2(x1))
        x3 = self.block3(self.down3(x2))
        x4 = self.block4(self.down4(x3))
        H, W = x1.shape[-2:]
        feats = [x1, nn.functional.interpolate(x2, size=(H,W), mode='bilinear', align_corners=False),
                 nn.functional.interpolate(x3, size=(H,W), mode='bilinear', align_corners=False),
                 nn.functional.interpolate(x4, size=(H,W), mode='bilinear', align_corners=False)]
        return self.decode(torch.cat(feats, 1))


# ============ 训练协议 ============

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        """遍历 parameters()/buffers() 而非 state_dict() —— 省去每步构建字典的开销"""
        d = self.decay
        for es, ms in zip(self.shadow.parameters(), model.parameters()):
            es.mul_(d).add_(ms.detach(), alpha=1 - d)
        for eb, mb in zip(self.shadow.buffers(), model.buffers()):
            eb.copy_(mb)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
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
    f1s = []
    for k in range(NUM_CLASSES):
        tp = c[k,k]; fp = c[:,k].sum()-tp; fn = c[k,:].sum()-tp
        f1s.append(2*tp/max(2*tp+fp+fn,1))
    return float(oa), float(kappa), float(np.mean(f1s)), [float(x) for x in f1s]


def train_one(model_fn, name, max_epochs=120, patience=20, batch=8, lr=2e-4, sched_kind="onecycle", root=None, init_ckpt=None, val_root=None, fast_data=True, num_workers=0, seed=42, crop=256, center_crop=False):   # num_workers 回退默认 0（同原管线）; seed/crop 默认值与原行为一致
    """通用训练入口: 复用 train_v3 的数据/EMA/早停协议, 模型/head/调度可替换
    root: 数据根目录 (None=旧957张; 传入 newsplit 路径=官方全量2257张)
    seed: 控制 模型初始化 / DataLoader 打乱顺序 / _geom 增强抽样
          （裁剪位置由 epoch_seed 决定, 与 seed 无关）
    crop: 训练裁剪边长 (原图 1024²; 默认 256)
    center_crop: True 时固定取中心 patch（用于"随机裁剪 vs 固定裁剪"对比）"""
    mean, std = compute_stats(root=root)
    while True:
        try:
            tr = DataLoader(LoveDADataset("train", mean, std, train=True, root=root,
                                          fast=fast_data, crop=crop,
                                          center_crop=center_crop),
                             batch_size=batch, shuffle=True, num_workers=num_workers,
                             pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
                             prefetch_factor=(4 if num_workers > 0 else None))
            va = DataLoader(LoveDADataset("val", mean, std, train=False, n_val_patches=4,
                                              root=(val_root or root), fast=fast_data,
                                              crop=crop),
                             batch_size=batch, shuffle=False,
                             num_workers=min(4, num_workers), pin_memory=True,
                             persistent_workers=(num_workers > 0))
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = model_fn().to(DEVICE)
            if init_ckpt and os.path.exists(init_ckpt):
                _sd = torch.load(init_ckpt, map_location=DEVICE)
                _r = model.load_state_dict(_sd, strict=False)
                print(f"[init] {name} <- {os.path.basename(init_ckpt)} miss={len(_r.missing_keys)} unexp={len(_r.unexpected_keys)}", flush=True)
            # dry-run 前向验证输出形状 (eval模式, 避开 train 模式下 1×1 BN 限制)
            model.eval()
            with torch.no_grad():
                _probe = model(torch.zeros(1, 3, 256, 256, device=DEVICE))
                assert _probe.shape[-2:] == (256, 256), f"模型输出形状 {_probe.shape} 与 label 不匹配"
            nparam = sum(p.numel() for p in model.parameters())
            print(f"\n=== {name} | {nparam/1e6:.2f}M params | batch={batch} ===", flush=True)
            break
        except (RuntimeError, AssertionError) as e:
            if "out of memory" in str(e).lower() or "shape" in str(e).lower():
                print(f"[retry] {name} OOM/shape: {str(e)[:200]}, dropping batch", flush=True)
                batch = max(2, batch // 2)
                torch.cuda.empty_cache()
                if batch < 2:
                    raise
                continue
            raise

    ema = EMA(model)
    # 类别权重
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    from train_v3 import REMAP
    _cw_base = root if root is not None else paths.DATA_ROOT
    for f in sorted(os.listdir(rf"{_cw_base}\train\masks"))[::5]:
        m = REMAP[np.array(Image.open(rf"{_cw_base}\train\masks\{f}"))]
        counts += np.bincount(m[m!=255], minlength=NUM_CLASSES)
    w = 1.0/np.sqrt(counts/counts.sum()+1e-6); w = w/w.mean()
    cw = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    if sched_kind == "constant":
        sched = None
        step_per_batch = False
    elif sched_kind == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max_epochs*len(tr), pct_start=0.06)
        step_per_batch = True
    elif sched_kind == "constant":  # 恒定LR (不做退火, 攻击实验用)
        sched = None
        step_per_batch = False
    elif sched_kind == "sgdr30":   # 每30轮热重启的余弦退火
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=1)
        step_per_batch = False
    else:
        raise ValueError(sched_kind)
    scaler = torch.amp.GradScaler()
    best_k, bad = -1.0, 0; hist = []
    for ep in range(1, max_epochs+1):
        model.train()
        tr.dataset.epoch_seed = ep
        t0 = time.time(); tot = 0
        tot_t = torch.zeros((), device=DEVICE)   # 张量累积, 避免每步同步
        for img, lab in tr:
            img = img.to(DEVICE, non_blocking=True)
            lab = lab.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda"):
                    out = model(img)
                    # 防御性上采样到 label 形状（PSPNet/DeepLab 形状不一致时兜底）
                    if out.shape[-2:] != lab.shape[-2:]:
                        out = nn.functional.interpolate(out, size=lab.shape[-2:], mode='bilinear', align_corners=False)
                    loss = nn.functional.cross_entropy(out, lab, weight=cw, ignore_index=IGNORE)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[ep {ep}] OOM, halving batch and reloading from best ckpt", flush=True)
                    torch.cuda.empty_cache()
                    sd = torch.load(f"{CKPT}/{name}_best.pt", map_location=DEVICE) if os.path.exists(f"{CKPT}/{name}_best.pt") else None
                    if sd: model.load_state_dict(sd)
                    break
                raise
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if step_per_batch and sched is not None: sched.step()
            ema.update(model); tot_t += loss.detach().float()
        tot = float(tot_t.item())
        if not step_per_batch and sched is not None: sched.step()
        else:  # only if inner loop didn't break (no OOM)
            oa, k, mf1, f1s = evaluate(ema.shadow, va)
            flag = ""
            if k > best_k:
                best_k, bad = k, 0
                torch.save(ema.shadow.state_dict(), f"{CKPT}/{name}_best.pt")
                flag = " *"
            else:
                bad += 1
            print(f"[ep {ep:3d}] {name} loss={tot/len(tr):.4f} OA={oa:.4f} Kappa={k:.4f} mF1={mf1:.4f} best_k={best_k:.4f} bad={bad} ({time.time()-t0:.0f}s){flag}", flush=True)
            hist.append(dict(epoch=ep, loss=tot/len(tr), oa=oa, kappa=k, mf1=mf1, f1=f1s))
            json.dump(hist, open(f"{CKPT}/{name}_history.json","w"), indent=1)
            if bad >= patience:
                print(f"  early stop ep {ep}", flush=True); break
            continue
        # OOM 路径: 减 batch 并重新初始化
        torch.cuda.empty_cache(); return train_one(model_fn, name, max_epochs, patience, batch=max(2,batch//2))

    print(f"=== {name} DONE best Kappa {best_k:.4f} ===\n", flush=True)
    return best_k


# ============ 实验矩阵 ============
def mssact_light(**flags):
    return MSSACTNet(in_channels=3, num_classes=NUM_CLASSES,
                     embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4, **flags)

MATRIX = [
    # --- 基线对比 ---
    ("unet",      lambda: UNet(),                                            8),
    ("pspnet",    lambda: PSPNet(),                                          8),
    ("fcn",       lambda: FCN(),                                             8),
    ("deeplab",   lambda: DeepLabV3Plus(),                                   8),
    ("segformer", lambda: SegFormerLite(),                                   8),
    # --- 消融 ---
    ("ablate_no_emr",     lambda: mssact_light(use_emr=False),               8),
    ("ablate_no_ecsam",   lambda: mssact_light(use_ecsam=False),             8),
    ("ablate_no_fpn",     lambda: mssact_light(use_fpn=False),               8),
    ("ablate_no_trans",   lambda: mssact_light(use_transformer=False),       8),
    ("ablate_no_adapter", lambda: mssact_light(use_adapter=False),           8),
]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=str, default=None, help="只跑一个实验 (矩阵中的 name)")
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None, help="覆盖矩阵中默认 batch")
    args = ap.parse_args()
    selected = MATRIX if args.single is None else [m for m in MATRIX if m[0] == args.single]
    if not selected:
        print(f"ERROR: --single {args.single} not in {[m[0] for m in MATRIX]}", flush=True); raise SystemExit(1)
    print(f"matrix ({len(selected)}):", [m[0] for m in selected], flush=True)
    for name, fn, default_batch in selected:
        batch = args.batch if args.batch is not None else default_batch
        try:
            train_one(fn, name, max_epochs=args.max_epochs, patience=args.patience, batch=batch)
        except Exception as e:
            print(f"!!! {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue