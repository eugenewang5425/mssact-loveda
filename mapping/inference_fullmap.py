"""GF 高分影像全图滑窗推理 + 高斯加权融合 + 行政区 shp 裁剪成图

用法:
  python inference_fullmap.py --ckpt checkpoints/v3_s256_best.pt --crop 256 --center 121.51 50.78 --size 8192
  python inference_fullmap.py --ckpt checkpoints/v3_s128gf_best.pt --crop 128 --gf_matched
输入: 多波段 GeoTIFF（前3波段 RGB）+ 行政区矢量 shp（路径由 .env 配置）
输出: GeoTIFF分类图(带坐标) + 彩色PNG + 裁剪掩膜叠加
"""

# --- 路径引导（本文件位于子目录 mapping/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os, sys, argparse
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from PIL import Image

sys.path.insert(0, _HERE)  # repo-local models/
sys.path.insert(0, paths.REPO)
sys.path.insert(0, _HERE)
from models.msscactnet import MSSACTNet

ROOT = paths.DATA_ROOT
SHP = paths.GF_SHP
TIF = paths.GF_TIF
NUM_CLASSES = 7
CLASS_NAMES = ["背景", "建筑", "道路", "水域", "裸地", "森林", "农田"]
# LoveDA训练集统计 (RGB 0-1)
MEAN = np.array([0.298, 0.3224, 0.2949], np.float32)
STD  = np.array([0.1713, 0.1415, 0.1328], np.float32)
PALETTE = np.array([[128,128,128],[255,0,0],[255,255,0],[0,0,255],[160,82,45],[0,128,0],[255,165,0]], np.uint8)

def gauss2d(s, sigma):
    ax = np.arange(s) - (s-1)/2.0
    g = np.exp(-(ax**2)/(2*sigma**2))
    return np.outer(g, g).astype(np.float32)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--center", nargs=2, type=float, default=[121.51, 50.78], help="经度 纬度")
    ap.add_argument("--size", type=int, default=8192, help="推理窗口像素数(边长)")
    ap.add_argument("--out", default=os.path.join(paths.REPO, "genhe_map"))
    ap.add_argument("--stride-ratio", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S, stride = args.crop, int(args.crop*args.stride_ratio)

    import geopandas as gpd
    ds = rasterio.open(TIF)
    cx, cy = args.center
    # 经纬度中心 -> 像素
    inv = ~ds.transform
    pcx, pcy = inv * (cx, cy)
    half = args.size//2
    col0 = int(max(0, pcx-half)); row0 = int(max(0, pcy-half))
    col1 = int(min(ds.width, col0+args.size)); row1 = int(min(ds.height, row0+args.size))
    win = Window(col0, row0, col1-col0, row1-row0)
    print(f"window cols {col0}-{col1} rows {row0}-{row1} ({col1-col0}x{row1-row0})", flush=True)

    img = ds.read([1,2,3], window=win).transpose(1,2,0).astype(np.float32)  # RGB
    transform_win = ds.window_transform(win)

    x = (img/255.0 - MEAN)/STD
    x = torch.from_numpy(x.transpose(2,0,1)[None])  # 1,3,H,W

    device = "cuda"
    model = MSSACTNet(in_channels=3, num_classes=NUM_CLASSES,
                      embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd); model.eval()

    H, W = img.shape[:2]
    prob = np.zeros((NUM_CLASSES, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    gw = gauss2d(S, S/4.0)
    pad_h = (stride - (H - S) % stride) % stride
    pad_w = (stride - (W - S) % stride) % stride
    xp = torch.nn.functional.pad(x, (0,pad_w,0,pad_h), mode="reflect")
    probp = np.zeros((NUM_CLASSES, H+pad_h, W+pad_w), np.float32)
    wsump = np.zeros((H+pad_h, W+pad_w), np.float32)
    Hp, Wp = H+pad_h, W+pad_w
    n_steps = ((Hp-S)//stride+1) * ((Wp-S)//stride+1)
    print(f"sliding {S}x{S} stride {stride}: {n_steps} windows", flush=True)
    done = 0
    for yy in range(0, Hp-S+1, stride):
        for xx in range(0, Wp-S+1, stride):
            patch = xp[:, :, yy:yy+S, xx:xx+S].to(device)
            with torch.amp.autocast("cuda"):
                p = torch.softmax(model(patch), 1)
            probp[:, yy:yy+S, xx:xx+S] += p[0].float().cpu().numpy() * gw
            wsump[yy:yy+S, xx:xx+S] += gw
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{n_steps}", flush=True)
    predp = probp.argmax(0)
    pred = predp[:H, :W].astype(np.uint8)

    # ---- 行政区shp裁剪 ----
    gdf = gpd.read_file(SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4490:
        gdf = gdf.to_crs(4490)
    # 窗口范围
    left, top = transform_win * (0, 0)
    right, bottom = transform_win * (W, H)
    from shapely.geometry import box
    win_box = box(left, bottom, right, top)
    clipped = gdf[gdf.intersects(win_box)]
    print(f"boundary features intersecting window: {len(clipped)}", flush=True)
    if len(clipped):
        mask = geometry_mask(clipped.geometry, out_shape=(H, W), transform=transform_win, invert=True)
        pred[~mask] = 255
        nodata_out = 255
    else:
        nodata_out = None

    # ---- 输出 GeoTIFF (EPSG:4490) ----
    prof = ds.profile.copy()
    prof.update(count=1, dtype="uint8", transform=transform_win, width=W, height=H,
                nodata=255 if nodata_out else None, compress="lzw")
    with rasterio.open(f"{args.out}/genhe_pred_clipped.tif", "w", **prof) as dst:
        dst.write(pred, 1)
    rgb = PALETTE[np.where(pred==255, 0, pred)]
    Image.fromarray(rgb).save(f"{args.out}/genhe_pred_color.png")
    # 叠加原图
    import rasterio as rio
    img8 = np.clip(img/np.percentile(img, 99) * 255, 0, 255).astype(np.uint8)
    overlay = (0.45*img8 + 0.55*rgb).astype(np.uint8)
    Image.fromarray(overlay).save(f"{args.out}/genhe_pred_overlay.png")
    # 指标
    valid = pred != 255
    counts = np.bincount(pred[valid], minlength=NUM_CLASSES)
    print("class px:", dict(zip(CLASS_NAMES, counts.tolist())))
    print("saved to", args.out, flush=True)

if __name__ == "__main__":
    main()
