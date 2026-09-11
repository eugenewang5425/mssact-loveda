"""全量滑窗推理评估（验证集全量 299 tiles, 滑窗 256/stride128 + 高斯融合, 去除边缘效应）
对每个模型: 全tile混淆矩阵 -> OA/Kappa/mF1/mIoU/每类F1; 另存3个展示tile的预测图
"""
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch, torch.nn as nn
from PIL import Image

sys.path.insert(0, paths.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, PSPNet, FCN, DeepLabV3Plus, SegFormerLite
from experiment_matrix_v2 import FPNSeg, SwinUnetLite, mssact_light
from train_v3 import REMAP

ROOT = paths.DATA_ROOT
CKPT = paths.CKPT
OUT = os.path.join(paths.REPO, "fulltile_eval")
NUM_CLASSES, IGNORE = 7, 255
S, STRIDE = 256, 128
CLASS_NAMES = ["背景","建筑","道路","水域","裸地","森林","农田"]
MEAN = np.array([0.298,0.3224,0.2949], np.float32)
STD  = np.array([0.1713,0.1415,0.1328], np.float32)

MODELS = {
    "mssact_full60": ("mssact", f"{CKPT}/mssact_full60_best.pt"),
    "v3_s256":       ("mssact", f"{CKPT}/v3_s256_best.pt"),
    "deeplab":       ("deeplab", f"{CKPT}/deeplab_best.pt"),
    "fcn":           ("fcn",     f"{CKPT}/fcn_best.pt"),
    "pspnet":        ("pspnet",  f"{CKPT}/pspnet_best.pt"),
    "segformer":     ("segformer", f"{CKPT}/segformer_best.pt"),
    "unet_matrix":   ("unet",    f"{CKPT}/unet_matrix_best.pt"),
    "fpn_seg":       ("fpn_seg", f"{CKPT}/fpn_seg_best.pt"),
    "swin_unet":     ("swin_unet", f"{CKPT}/swin_unet_best.pt"),
    "ablate_no_ecsam":("mssact", f"{CKPT}/ablate_no_ecsam_best.pt"),
    "ablate_no_fpn": ("mssact",  f"{CKPT}/ablate_no_fpn_best.pt"),
    "ablate_no_trans":("mssact", f"{CKPT}/ablate_no_trans_best.pt"),
    "ablate_no_adapter":("mssact", f"{CKPT}/ablate_no_adapter_best.pt"),
}
SHOWCASE = []  # 训练中填充: 3个tile名

def build(kind):
    if kind == "mssact":
        return MSSACTNet(in_channels=3, num_classes=NUM_CLASSES, embed_dims=[32,64,128,256],
                         transformer_layers=2, transformer_heads=4)
    if kind == "unet": return UNet()
    if kind == "pspnet": return PSPNet()
    if kind == "fcn": return FCN()
    if kind == "deeplab": return DeepLabV3Plus()
    if kind == "segformer": return SegFormerLite()
    if kind == "fpn_seg": return FPNSeg()
    if kind == "swin_unet": return SwinUnetLite()
    raise ValueError(kind)

def gauss2d(s, sigma):
    ax = np.arange(s)-(s-1)/2.0
    g = np.exp(-(ax**2)/(2*sigma**2))
    return np.outer(g,g).astype(np.float32)

@torch.no_grad()
def infer_tile(model, img_u8, device="cuda"):
    """滑窗+高斯融合, 返回 (H,W) uint8 预测"""
    H, W = img_u8.shape[:2]
    ph = (STRIDE - (H-S) % STRIDE) % STRIDE; pw = (STRIDE - (W-S) % STRIDE) % STRIDE
    x = np.pad(img_u8, ((0,ph),(0,pw),(0,0)), mode="reflect").astype(np.float32)/255.0
    x = (x - MEAN) / STD
    xp = torch.from_numpy(x.transpose(2,0,1)[None])
    Hp, Wp = H+ph, W+pw
    prob = np.zeros((NUM_CLASSES, Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    gw = gauss2d(S, S/4.0)
    wins = [(yy,xx) for yy in range(0, Hp-S+1, STRIDE) for xx in range(0, Wp-S+1, STRIDE)]
    for i in range(0, len(wins), 16):
        batch = [xp[:,:,y:y+S,x:x+S] for y,x in wins[i:i+16]]
        bx = torch.cat(batch).to(device)
        with torch.amp.autocast("cuda"):
            p = torch.softmax(model(bx), 1).float()
        p = p.cpu().numpy()
        for j,(y,x) in enumerate(wins[i:i+16]):
            prob[:, y:y+S, x:x+S] += p[j]*gw
            wsum[y:y+S, x:x+S] += gw
    wsum[wsum==0]=1
    prob /= wsum
    return prob[:, :H, :W].argmax(0).astype(np.uint8)

def main(model_filter=None):
    os.makedirs(OUT, exist_ok=True)
    names = sorted(set(os.listdir(f"{ROOT}/val/images")) & set(os.listdir(f"{ROOT}/val/masks")))
    print(f"val tiles: {len(names)}", flush=True)
    results = {}
    for name,(kind, ckpt) in MODELS.items():
        if model_filter and name not in model_filter: continue
        if not os.path.exists(ckpt):
            print(f"SKIP {name} (no ckpt)"); continue
        pred_dir = f"{OUT}/preds_{name}"
        os.makedirs(pred_dir, exist_ok=True)
        # 检查是否已完成
        done_flag = f"{OUT}/{name}_conf.json"
        if os.path.exists(done_flag):
            print(f"SKIP {name} (done)"); continue
        model = build(kind)
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        model.to("cuda").eval()
        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        t0 = time.time()
        for t_i, n in enumerate(names):
            img = np.array(Image.open(f"{ROOT}/val/images/{n}").convert("RGB"))
            msk = REMAP[np.array(Image.open(f"{ROOT}/val/masks/{n}"))]
            pred = infer_tile(model, img)
            v = msk != 255
            conf += np.bincount((msk[v].astype(np.int64)*NUM_CLASSES + pred[v]).ravel(),
                                minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
            if name in ("mssact_full60","deeplab","fcn","pspnet","segformer","unet_matrix","v3_s256","fpn_seg","swin_unet"):
                Image.fromarray(pred).save(f"{pred_dir}/{n}")
            if (t_i+1) % 50 == 0:
                print(f"  {name} {t_i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
        c = conf.astype(np.float64)
        oa = np.trace(c)/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
        kappa = (oa-pe)/(1-pe)
        f1s, ious = [], []
        for k in range(NUM_CLASSES):
            tp=c[k,k]; fp=c[:,k].sum()-tp; fn=c[k,:].sum()-tp
            f1s.append(float(2*tp/max(2*tp+fp+fn,1))); ious.append(float(tp/max(tp+fp+fn,1)))
        results[name] = dict(oa=float(oa), kappa=float(kappa), mf1=float(np.mean(f1s)),
                             miou=float(np.mean(ious)), f1=f1s, iou=ious,
                             confusion=c.tolist(), seconds=round(time.time()-t0))
        json.dump(results[name], open(done_flag,"w"), indent=1)
        print(f"== {name}: OA={oa:.4f} Kappa={kappa:.4f} mF1={np.mean(f1s):.4f} mIoU={np.mean(ious):.4f} ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(results, open(f"{OUT}/fulltile_results.json","w"), indent=1)
    print("ALL DONE -> fulltile_results.json", flush=True)

if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv)>1 else None)
