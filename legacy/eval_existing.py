"""公平性交叉验证（评估only，不训练）:
1. 旧协议产物 (mssact_v2_best) 在 新/旧 两个验证集上的差异 -> 分离"验证集难度"因素
2. 矩阵产物 (unet_matrix_best / mssact_full60_best) 在 旧 patch 验证集上的表现 -> 分离"协议"因素
3. 重跑 v1 U-Net (baseline_unet 协议, 12轮) 恢复被覆盖的权重并交叉评估
结论: 旧U-Net 0.2704 与 矩阵U-Net 0.0939 的差异归因
"""
import os, sys, json
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, paths.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, EMA, DEVICE, NUM_CLASSES, IGNORE
from train_v3 import LoveDADataset, compute_stats
from train import PatchDS

CKPT = paths.CKPT
RESULTS = {}

@torch.no_grad()
def eval_conf(model, loader):
    model.eval(); conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, lab in loader:
        img, lab = img.to(DEVICE), lab.to(DEVICE)
        with torch.amp.autocast("cuda"):
            out = model(img)
        if out.shape[-2:] != lab.shape[-2:]:
            out = nn.functional.interpolate(out, size=lab.shape[-2:], mode='bilinear', align_corners=False)
        pred = out.argmax(1); v = lab != IGNORE
        conf += torch.bincount((lab[v]*NUM_CLASSES+pred[v]).cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    c = conf.cpu().numpy().astype(np.float64)
    oa = np.trace(c)/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
    kappa = (oa-pe)/(1-pe)
    f1s = [2*c[k,k]/max(2*c[k,k]+c[:,k].sum()+c[k,:].sum()-2*c[k,k],1) for k in range(NUM_CLASSES)]
    return dict(oa=float(oa), kappa=float(kappa), mf1=float(np.mean(f1s)))

def new_val_loader():
    mean, std = compute_stats()
    return DataLoader(LoveDADataset("val", mean, std, train=False, n_val_patches=4), batch_size=8, num_workers=0)

def old_val_loader():
    mean = torch.tensor([0.4145,0.4223,0.4101]).view(3,1,1)  # 不重要: PatchDS返回原始DN? 不, v2协议已归一化255
    std = torch.tensor([0.2,0.2,0.2]).view(3,1,1)
    # PatchDS 输出 0-1 归一化? 查 train.PatchDS: img float /255 后再 mean/std? 实际: (x-mean)/std, mean/std为参数
    # v2训练时用的 mean/std 由 compute_stats() 得到 (0-1域)
    m, s = __import__("train").compute_stats()
    return DataLoader(PatchDS("val", m, s), batch_size=8, num_workers=0)

def load_model(kind, path):
    if kind == "unet":
        m = UNet()
    else:
        m = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES, embed_dims=[32,64,128,256],
                      transformer_layers=2, transformer_heads=4)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()

if __name__ == "__main__":
    nv = new_val_loader(); ov = old_val_loader()
    checks = [
        ("mssact_v2_best", "mssact", f"{CKPT}/mssact_v2_best.pt"),
        ("unet_matrix_best", "unet", f"{CKPT}/unet_matrix_best.pt"),
        ("mssact_full60_best", "mssact", f"{CKPT}/mssact_full60_best.pt"),
    ]
    for name, kind, path in checks:
        if not os.path.exists(path):
            print(f"SKIP {name} (no ckpt)"); continue
        m = load_model(kind, path)
        r_new = eval_conf(m, nv)
        r_old = eval_conf(m, ov)
        RESULTS[name] = dict(new_val=r_new, old_patchval=r_old)
        print(f"{name}: NEW-val Kappa={r_new['kappa']:.4f} | OLD-patch-val Kappa={r_old['kappa']:.4f} "
              f"(OA {r_new['oa']:.4f}/{r_old['oa']:.4f}, mF1 {r_new['mf1']:.4f}/{r_old['mf1']:.4f})", flush=True)
    json.dump(RESULTS, open(f"{CKPT}/fairness_crosseval.json","w"), indent=1)
    print("saved fairness_crosseval.json")
