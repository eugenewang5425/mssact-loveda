"""
预解码数据集: 1024² PNG -> npy memmap
动机: 原管线每样本要解码两个 1024² PNG, 只裁出 256² (94% 解码结果被丢弃);
     单样本 31-37ms, 是 GPU 空闲(49.9% 利用率)的主因。
方案: 一次性解码为 uint8 memmap, 训练时用 np.memmap 随机裁剪 —— 保留原有的
     "每轮每图随机位置"增强语义, 但消除 PNG 解码成本。

产出: fast_dataset/{images.npy, masks.npy, index.json}
     images.npy  (N,1024,1024,3) uint8
     masks.npy   (N,1024,1024)   uint8  (已应用官方 REMAP: 0/255->255, 1..7->0..6)
"""

# --- 路径引导（本文件位于子目录 prep/，仓库根为上一级）---
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
import numpy as np
from PIL import Image

sys.path.insert(0, _HERE)
import paths

SRC = paths.DATA_NEWSPLIT2          # 含 train/ val/ test/ test_clean/
OUT = os.path.join(paths.REPO, "fast_dataset")
SIZE = 1024

# 官方标签映射: 1..7 -> 0..6; 0(no-data) 及其他 -> 255(ignore)
REMAP = np.full(256, 255, dtype=np.uint8)
for s in range(1, 8):
    REMAP[s] = s - 1

def build(split):
    img_dir = os.path.join(SRC, split, "images")
    msk_dir = os.path.join(SRC, split, "masks")
    if not os.path.isdir(img_dir):
        print(f"  [{split}] 无数据, 跳过")
        return None
    names = sorted(set(os.listdir(img_dir)) & set(os.listdir(msk_dir)))
    n = len(names)
    print(f"  [{split}] {n} 张 -> 解码中...", flush=True)

    imgs = np.lib.format.open_memmap(os.path.join(OUT, f"{split}_images.npy"),
                                     mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    msks = np.lib.format.open_memmap(os.path.join(OUT, f"{split}_masks.npy"),
                                     mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE))
    for i, nm in enumerate(names):
        im = Image.open(os.path.join(img_dir, nm)).convert("RGB")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BILINEAR)
        imgs[i] = np.asarray(im, dtype=np.uint8)
        mk = np.array(Image.open(os.path.join(msk_dir, nm)))
        if mk.shape != (SIZE, SIZE):
            mk = np.array(Image.fromarray(mk).resize((SIZE, SIZE), Image.NEAREST))
        msks[i] = REMAP[mk]
        if (i + 1) % 400 == 0:
            print(f"    {i+1}/{n}", flush=True)
    imgs.flush(); msks.flush()
    del imgs, msks
    print(f"  [{split}] 完成: {split}_images.npy / {split}_masks.npy", flush=True)
    return n

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for split in ["train", "val", "test", "test_clean"]:
        c = build(split)
        if c: meta[split] = c
    json.dump(meta, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    total = sum(meta.values())
    print(f"\n全部完成: {meta}  共 {total} 张")
    # 磁盘占用
    sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".npy"))
    print(f"memmap 体积: {sz/1e9:.2f} GB")
