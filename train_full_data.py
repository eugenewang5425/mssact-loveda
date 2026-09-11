"""全量 LoveDA 训练 + 真·后训练 + TTA 评估
流程:
  A. 全量数据重训 light MSSACT (官方 train ~9900张): full10k
  B. 后训练: 加载全量模型权重 + 全量数据继续训练 (微调, lr=5e-5): post10k
  C. TTA 推理评估 (8x 翻转/旋转平均) vs 无TTA
数据: 官方 loveda_train.zip 解压进入 data/loveda/train/{images,masks}
"""
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from train_v3 import LoveDADataset, compute_stats, REMAP

ROOT = paths.DATA_NEWSPLIT2  # 筛选后(no-data<=10%) 1768 train
ROOT_OLD = paths.DATA_NEWSPLIT  # 旧划分(未筛选, 历史用)
CKPT = paths.CKPT
NUM_CLASSES, IGNORE = 7, 255
DEVICE = "cuda"

def run_train(tag, init_ckpt=None, max_epochs=60, patience=20, batch=8, lr=2e-4, seed=42):
    """通用训练(全量数据); init_ckpt 非空则用于后训练"""
    import torch.optim as optim
    mean, std = compute_stats()
    tr = DataLoader(LoveDADataset("train", mean, std, train=True, root=ROOT), batch_size=batch, shuffle=True,
                    num_workers=0, pin_memory=True, drop_last=True)
    va = DataLoader(LoveDADataset("val", mean, std, train=False, n_val_patches=4, root=ROOT),
                    batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    print(f"train imgs {len(tr.dataset)} / val patches {len(va.dataset)}", flush=True)
    torch.manual_seed(seed)
    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES, embed_dims=[32,64,128,256],
                      transformer_layers=2, transformer_heads=4).to(DEVICE)
    if init_ckpt and os.path.exists(init_ckpt):
        sd = torch.load(init_ckpt, map_location=DEVICE)
        load_res = model.load_state_dict(sd, strict=False)
        print(f"post-train init from {init_ckpt}: miss={len(load_res.missing_keys)} unexp={len(load_res.unexpected_keys)}", flush=True)
    # EMA
    import copy
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    # class weights
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    files = sorted(os.listdir(f"{ROOT}/train/masks"))[::10]
    for f in files:
        m = REMAP[np.array(Image.open(f"{ROOT}/train/masks/{f}"))]
        counts += np.bincount(m[m!=255], minlength=NUM_CLASSES)
    w = 1.0/np.sqrt(counts/counts.sum()+1e-6); w = w/w.mean()
    cw = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    total_steps = max_epochs*len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=0.06)
    scaler = torch.amp.GradScaler()
    best_k, bad, hist = -1.0, 0, []
    for ep in range(1, max_epochs+1):
        model.train(); tr.dataset.epoch_seed = ep
        t0 = time.time(); tot = 0
        for img, lab in tr:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = nn.functional.cross_entropy(out, lab, weight=cw, ignore_index=IGNORE)
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                for es, ms in zip(ema.state_dict().values(), model.state_dict().values()):
                    if es.dtype.is_floating_point: es.mul_(0.999).add_(ms.detach(), alpha=0.001)
                    else: es.copy_(ms)
            tot += loss.item()
        # eval
        ema.eval(); conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        with torch.no_grad():
            for img, lab in va:
                img, lab = img.to(DEVICE), lab.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    pred = ema(img).argmax(1)
                v = lab != IGNORE
                conf += torch.bincount((lab[v]*NUM_CLASSES+pred[v]).cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
        c = conf.cpu().numpy().astype(np.float64)
        oa = np.trace(c)/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
        k = (oa-pe)/(1-pe)
        flag = ""
        if k > best_k:
            best_k, bad = k, 0
            torch.save(ema.state_dict(), f"{CKPT}/{tag}_best.pt")
            flag = " *"
        else: bad += 1
        hist.append(dict(epoch=ep, loss=tot/len(tr), oa=float(oa), kappa=float(k), bad=bad))
        json.dump(hist, open(f"{CKPT}/{tag}_history.json","w"), indent=1)
        print(f"[ep {ep:3d}] {tag} loss={tot/len(tr):.4f} OA={oa:.4f} K={k:.4f} best={best_k:.4f} bad={bad} ({time.time()-t0:.0f}s){flag}", flush=True)
        if bad >= patience: print(f"early stop @{ep}", flush=True); break
    print(f"=== {tag} DONE best Kappa {best_k:.4f} ===", flush=True)
    return model, ema, best_k

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "full_all"
    if which == "full_all":
        run_train("full_all", max_epochs=60, patience=20, lr=2e-4)
    elif which == "full_all_v2":
        # [BUGFIX] 正确忽略 no-data(值0) 后重训, 并与 v1 对比
        run_train("full_all_v2", max_epochs=60, patience=20, lr=2e-4)
    elif which == "post_all_v2":
        run_train("post_all_v2", init_ckpt=f"{CKPT}/full_all_v2_best.pt", max_epochs=30, patience=12, lr=5e-5)
    elif which == "post_all":
        # 真·后训练: 从 full_all 权重继续低LR训练 (continued training)
        run_train("post_all", init_ckpt=f"{CKPT}/full_all_best.pt", max_epochs=30, patience=12, lr=5e-5)
