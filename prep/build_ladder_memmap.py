# -*- coding: utf-8 -*-
"""
为数据阶梯子集建 memmap (prep/build_ladder_memmap.py)

动机
----
数据阶梯此前传 `fast_data=False`（因为 fast_dataset/ 只预解码了完整数据集），
于是退回 **PNG 管线**：每个样本解码两张 1024² PNG 却只裁出 256²，94% 的解码结果
被丢弃，且解码在 CPU 上——GPU 只能等。

实测代价（本轮真实数字）：
    n250 仅 250 张，却要 42-49 s/轮  ≈ 170 ms/张
    对照: 主数据集 1768 张 = 43 s/轮 ≈  24 ms/张
    GPU 利用率均值 16%、中位 6%，功耗 32 W / 180 W，进程吃 13/20 逻辑核
即阶梯每张图代价是主数据集的 **7 倍**，纯属解码开销，与学习无关。

产出
----
    fast_dataset/ladder_n{250,500,1000}/train_images.npy   (N,1024,1024,3) uint8
                                         /train_masks.npy    (N,1024,1024)   uint8 (已 REMAP)
                                         /index.json         {"train": N}
窗口不预解码（阶梯的验证集固定为 newsplit2 的 val，已由主 memmap 覆盖）。

注意: 本脚本是**重量级磁盘 I/O**（解码 1750 张 PNG + 写约 7.3 GB），
     会与训练的 memmap 随机读争抢磁盘（实测可让轮次耗时恶化约 10 倍）。
     **务必在训练队列停止时运行。**

用法: python prep/build_ladder_memmap.py
"""
import os, sys, json, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

SIZES = (250, 500, 1000)
SIZE_PX = 1024

# 官方标签映射: 1..7 -> 0..6; 0(no-data) 及其他 -> 255(ignore)
REMAP = np.full(256, 255, dtype=np.uint8)
for s in range(1, 8):
    REMAP[s] = s - 1


def build(n, src_root, out_root):
    img_dir = os.path.join(src_root, "train", "images")
    msk_dir = os.path.join(src_root, "train", "masks")
    names = sorted(set(os.listdir(img_dir)) & set(os.listdir(msk_dir)))
    n_actual = len(names)
    if n_actual == 0:
        print(f"  [n{n}] 无数据，跳过"); return None
    if n_actual != n:
        print(f"  [n{n}] ⚠ 实际 {n_actual} 张，与预期不符")

    out = os.path.join(out_root, f"ladder_n{n}")
    os.makedirs(out, exist_ok=True)
    imgs = np.lib.format.open_memmap(os.path.join(out, "train_images.npy"), mode="w+",
                                     dtype=np.uint8, shape=(n_actual, SIZE_PX, SIZE_PX, 3))
    msks = np.lib.format.open_memmap(os.path.join(out, "train_masks.npy"), mode="w+",
                                     dtype=np.uint8, shape=(n_actual, SIZE_PX, SIZE_PX))
    t0 = time.time()
    for i, nm in enumerate(names):
        im = Image.open(os.path.join(img_dir, nm)).convert("RGB")
        if im.size != (SIZE_PX, SIZE_PX):
            im = im.resize((SIZE_PX, SIZE_PX), Image.BILINEAR)
        imgs[i] = np.asarray(im, dtype=np.uint8)
        mk = np.array(Image.open(os.path.join(msk_dir, nm)))
        if mk.shape != (SIZE_PX, SIZE_PX):
            mk = np.array(Image.fromarray(mk).resize((SIZE_PX, SIZE_PX), Image.NEAREST))
        msks[i] = REMAP[mk]
        if (i + 1) % 250 == 0:
            print(f"    {i+1}/{n_actual}  ({time.time()-t0:.0f}s)", flush=True)
    imgs.flush(); msks.flush()
    del imgs, msks
    json.dump({"train": n_actual, "source": src_root,
               "note": "阶梯子集预解码 memmap；mask 已应用官方 REMAP"},
              open(os.path.join(out, "index.json"), "w"), indent=1)
    mb = (n_actual * SIZE_PX * SIZE_PX * 4) / 1e6
    print(f"  [n{n}] 完成 {n_actual} 张, 约 {mb:.0f} MB, 用时 {time.time()-t0:.0f}s", flush=True)
    return n_actual


if __name__ == "__main__":
    src_root = paths.DATA_LADDER
    out_root = os.path.join(paths.REPO, "fast_dataset")
    print(f"源目录: {src_root}")
    print(f"输出  : {out_root}/ladder_n*/")
    if not os.path.isdir(src_root):
        print(f"!! 源目录不存在: {src_root}"); raise SystemExit(1)
    total = 0
    for n in SIZES:
        d = os.path.join(src_root, f"n{n}")
        if not os.path.isdir(d):
            print(f"  [n{n}] 目录不存在，跳过"); continue
        total += build(n, d, out_root) or 0
    print(f"\n全部完成: {total} 张")
