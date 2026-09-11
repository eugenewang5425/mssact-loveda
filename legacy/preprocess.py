"""LoveDA 预处理：PNG -> 256x256 patch npz（img: uint8 RGB, label: 0-6, 255=ignore）

LoveDA 官方标签（经视觉核对）: 0=no_data(ignore), 1=背景, 2=建筑, 3=道路, 4=水域,
5=裸地, 6=森林, 7=农田
映射后: 0=背景, 1=建筑, 2=道路, 3=水域, 4=裸地, 5=森林, 6=农田; 0 -> 255(ignore)
"""
import os, sys, json, random
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
from PIL import Image

ROOT = paths.DATA_ROOT
OUT = os.path.join(paths.REPO, "patches")
PATCH, TRAIN_STRIDE, VAL_STRIDE = 256, 256, 256
CLASS_NAMES = ["背景", "建筑", "道路", "水域", "裸地", "森林", "农田"]

def remap(mask):
    out = mask.astype(np.uint8) - 1          # 1..7 -> 0..6
    out[mask == 0] = 255                      # no_data
    return out

def process(split, limit_patches, seed=42):
    img_dir, msk_dir = f"{ROOT}/{split}/images", f"{ROOT}/{split}/masks"
    names = sorted(n for n in os.listdir(img_dir) if n.endswith(".png"))
    rng = random.Random(seed)
    patches = []
    stats = np.zeros(7, dtype=np.int64)
    for n in names:
        img = np.array(Image.open(f"{img_dir}/{n}").convert("RGB"))
        msk = remap(np.array(Image.open(f"{msk_dir}/{n}")))
        H, W = msk.shape
        stride = TRAIN_STRIDE if split=="train" else VAL_STRIDE
        for y in range(0, H-PATCH+1, stride):
            for x in range(0, W-PATCH+1, stride):
                patches.append((n, x, y, img[y:y+PATCH, x:x+PATCH], msk[y:y+PATCH, x:x+PATCH]))
    rng.shuffle(patches)
    out_dir = f"{OUT}/{split}"
    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    for i, (n, x, y, img, msk) in enumerate(patches):
        if kept >= limit_patches: break
        np.savez_compressed(f"{out_dir}/{kept:05d}.npz", img=img, label=msk)
        for c in range(7): stats[c] += int((msk==c).sum())
        kept += 1
    print(f"{split}: {len(names)} images -> {kept} patches")
    print("  pixel stats:", {CLASS_NAMES[c]: int(stats[c]) for c in range(6)})
    return stats

if __name__ == "__main__":
    tr = process("train", 3000)
    va = process("val", 600)
    json.dump({"classes": CLASS_NAMES}, open(f"{OUT}/meta.json","w"), ensure_ascii=False, indent=2)
