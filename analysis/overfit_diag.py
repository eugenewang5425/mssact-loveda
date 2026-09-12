"""过拟合诊断 #3:
对每个模型, 在训练集(随机100张) vs 验证集(全199张)上做滑窗推理,
统计: (a) logits熵均值 (b) 预测一致性 (argmax翻转率)
若 train entropy << val entropy, 模型过拟合; 一致性 train>>val 同样佐证
"""

# --- 路径引导（本文件位于子目录 analysis/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, PSPNet, FCN, DeepLabV3Plus, SegFormerLite
from experiment_matrix_v2 import FPNSeg, SwinUnetLite
from train_v3 import REMAP

ROOT = paths.DATA_ROOT
CKPT = paths.CKPT
OUT = paths.OVERFIT
os.makedirs(OUT, exist_ok=True)
NUM_CLASSES, S, STRIDE = 7, 256, 128
MEAN = np.array([0.298,0.3224,0.2949], np.float32); STD = np.array([0.1713,0.1415,0.1328], np.float32)

MODELS = {
    "mssact_full60":("mssact", f"{CKPT}/mssact_full60_best.pt"),
    "v3_s256":      ("mssact", f"{CKPT}/v3_s256_best.pt"),
    "deeplab":      ("deeplab",f"{CKPT}/deeplab_best.pt"),
    "fcn":          ("fcn",    f"{CKPT}/fcn_best.pt"),
    "pspnet":       ("pspnet", f"{CKPT}/pspnet_best.pt"),
    "segformer":    ("segformer",f"{CKPT}/segformer_best.pt"),
    "unet_matrix":  ("unet",   f"{CKPT}/unet_matrix_best.pt"),
    "fpn_seg":      ("fpn_seg",f"{CKPT}/fpn_seg_best.pt"),
    "swin_unet":    ("swin_unet",f"{CKPT}/swin_unet_best.pt"),
}
def build(kind):
    if kind=="mssact": return MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4)
    if kind=="unet": return UNet()
    if kind=="pspnet": return PSPNet()
    if kind=="fcn": return FCN()
    if kind=="deeplab": return DeepLabV3Plus()
    if kind=="segformer": return SegFormerLite()
    if kind=="fpn_seg": return FPNSeg()
    if kind=="swin_unet": return SwinUnetLite()

@torch.no_grad()
def sample_logits(model, names, subdir):
    """采100张, 随机裁剪4个256x256 patch, 输出mean entropy & patch预测"""
    import random
    random.seed(42)
    H_list, preds = [], []
    for n in names[:100]:
        img = np.array(Image.open(f"{ROOT}/{subdir}/images/{n}").convert("RGB"))
        msk = REMAP[np.array(Image.open(f"{ROOT}/{subdir}/masks/{n}"))]
        H, W = msk.shape
        for _ in range(4):
            y = random.randint(0, max(0, H-S))
            x = random.randint(0, max(0, W-S))
            patch = (img[y:y+S, x:x+S]/255.0 - MEAN) / STD
            x_t = torch.from_numpy(patch.transpose(2,0,1)[None]).float().to("cuda")
            with torch.amp.autocast("cuda"):
                logits = model(x_t).float()
            probs = torch.softmax(logits, dim=1)[0]
            H_list.append(float(-(probs * (probs + 1e-8).log()).sum(dim=0).mean().item()))
            preds.append(int(probs.argmax(1).flatten().mode(0).values.item()))
    return float(np.mean(H_list)), preds

train_names = sorted(os.listdir(f"{ROOT}/train/images"))[:100]
val_names = sorted(os.listdir(f"{ROOT}/val/images"))

print("Training-set entropy analysis (model 'confidence' on seen data)\n", flush=True)
results = {}
for name,(kind,ckpt) in MODELS.items():
    if not os.path.exists(ckpt): continue
    model = build(kind).to("cuda").eval()
    try:
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    except Exception as e:
        print(f"FAIL load {name}: {e}"); del model; torch.cuda.empty_cache(); continue
    t0=time.time()
    H_tr, p_tr = sample_logits(model, train_names, "train")
    H_va, p_va = sample_logits(model, val_names, "val")
    H_gap = H_tr - H_va   # 负 = train比val自信
    diag = dict(H_train=H_tr, H_val=H_va, H_gap=H_gap, train_preds_dominant=p_tr[:5], val_preds_dominant=p_va[:5], seconds=round(time.time()-t0))
    results[name] = diag
    json.dump(diag, open(f"{OUT}/{name}_diag.json","w"), indent=1)
    print(f"{name:22s} H_train={H_tr:.4f}  H_val={H_va:.4f}  ΔH={H_gap:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache()
json.dump(results, open(f"{OUT}/overfit_summary.json","w"), indent=1)
print("\noverfit_summary saved", flush=True)
print("\n判据: H_gap << 0 -> 过拟合 (训练集比验证集自信得多)", flush=True)
print("      H_gap ≈ 0  -> 校准良好", flush=True)
EOF