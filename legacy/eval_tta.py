"""TTA (Test-Time Augmentation) 评估: 8x 几何变换平均
对 test 集 282 张全量滑窗推理, 对比 无TTA vs 有TTA
变换: 4旋转 × 2翻转 = 8 种, 概率图逆变换后平均
"""
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from train_v3 import REMAP

ROOT = paths.DATA_NEWSPLIT
CKPT = paths.CKPT
OUT = os.path.join(paths.REPO, "tta_eval")
os.makedirs(OUT, exist_ok=True)
NUM_CLASSES, IGNORE, S, STRIDE = 7, 255, 256, 128
MEAN = np.array([0.298,0.3224,0.2949], np.float32); STD = np.array([0.1713,0.1415,0.1328], np.float32)

def gauss2d(s, sigma):
    ax = np.arange(s)-(s-1)/2.0
    g = np.exp(-(ax**2)/(2*sigma**2))
    return np.outer(g,g).astype(np.float32)

def build(ckpt):
    m = MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256],
                  transformer_layers=2, transformer_heads=4)
    m.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
    return m.to("cuda").eval()

@torch.no_grad()
def infer_tile(model, img_u8, tta=False):
    """滑窗+高斯融合; tta=True 时做 8 变换平均"""
    H, W = img_u8.shape[:2]
    ph=(STRIDE-(H-S)%STRIDE)%STRIDE; pw=(STRIDE-(W-S)%STRIDE)%STRIDE
    x = np.pad(img_u8,((0,ph),(0,pw),(0,0)),mode="reflect").astype(np.float32)/255.0
    x = (x-MEAN)/STD
    xp = torch.from_numpy(x.transpose(2,0,1)[None])
    Hp, Wp = H+ph, W+pw
    gw = gauss2d(S, S/4.0)
    wins = [(y,x) for y in range(0,Hp-S+1,STRIDE) for x in range(0,Wp-S+1,STRIDE)]

    def run(flip_h=False, flip_w=False, rot=0):
        prob = np.zeros((NUM_CLASSES,Hp,Wp), np.float32); wsum = np.zeros((Hp,Wp), np.float32)
        src = xp
        if flip_h: src = torch.flip(src, dims=[3])
        if flip_w: src = torch.flip(src, dims=[2])
        if rot: src = torch.rot90(src, rot, dims=[2,3])
        for i in range(0,len(wins),16):
            batch = [src[:,:,y:y+S,x:x+S] for y,x in wins[i:i+16]]
            bx = torch.cat(batch).to("cuda")
            with torch.amp.autocast("cuda"):
                p = torch.softmax(model(bx),1).float()
            p = p.cpu().numpy()
            for j,(y,x) in enumerate(wins[i:i+16]):
                prob[:,y:y+S,x:x+S] += p[j]*gw
                wsum[y:y+S,x:x+S] += gw
        wsum[wsum==0]=1
        prob /= wsum
        # 逆变换
        if rot: prob = np.rot90(prob, -rot, axes=(1,2))
        if flip_w: prob = prob[:, ::-1, :]
        if flip_h: prob = prob[:, :, ::-1]
        return prob

    if not tta:
        return run().argmax(0).astype(np.uint8)
    acc = np.zeros((NUM_CLASSES,Hp,Wp), np.float32)
    for rot in range(4):
        for fh in (False, True):
            for fw in (False, True):
                acc += run(fw, fh, rot)
    return acc.argmax(0).astype(np.uint8)

def evaluate(model, names, tta=False, tag=""):
    conf = np.zeros((NUM_CLASSES,NUM_CLASSES), np.int64)
    t0 = time.time()
    for i, n in enumerate(names):
        img = np.array(Image.open(f"{ROOT}/test/images/{n}").convert("RGB"))
        msk = REMAP[np.array(Image.open(f"{ROOT}/test/masks/{n}"))]
        pred = infer_tile(model, img, tta=tta)
        v = msk != IGNORE
        conf += np.bincount((msk[v].astype(np.int64)*NUM_CLASSES+pred[v]).ravel(),
                            minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
        if (i+1) % 50 == 0:
            print(f"  {tag} {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    c = conf.astype(np.float64)
    oa = np.trace(c)/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
    kappa = (oa-pe)/(1-pe)
    f1s, ious = [], []
    for k in range(NUM_CLASSES):
        tp=c[k,k]; fp=c[:,k].sum()-tp; fn=c[k,:].sum()-tp
        f1s.append(float(2*tp/max(2*tp+fp+fn,1))); ious.append(float(tp/max(tp+fp+fn,1)))
    return dict(oa=float(oa),kappa=float(kappa),mf1=float(np.mean(f1s)),
                miou=float(np.mean(ious)),f1=f1s,iou=ious,confusion=c.tolist(),
                seconds=round(time.time()-t0),tta=tta)

if __name__ == "__main__":
    names = sorted(set(os.listdir(f"{ROOT}/test/images")) & set(os.listdir(f"{ROOT}/test/masks")))
    print(f"test tiles: {len(names)}", flush=True)
    results = {}
    for model_tag in ["full_all", "mssact_full60"]:
        ckpt = f"{CKPT}/{model_tag}_best.pt"
        if not os.path.exists(ckpt):
            print(f"SKIP {model_tag} (no ckpt)"); continue
        model = build(ckpt)
        for tta in [False, True]:
            key = f"{model_tag}{'_TTA' if tta else ''}"
            r = evaluate(model, names, tta=tta, tag=key)
            results[key] = r
            json.dump(r, open(f"{OUT}/{key}_conf.json","w"), indent=1)
            print(f"== {key}: Kappa={r['kappa']:.4f} OA={r['oa']:.4f} mF1={r['mf1']:.4f} mIoU={r['miou']:.4f} ({r['seconds']}s)", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(results, open(f"{OUT}/tta_results.json","w"), indent=1)
    print("\n=== TTA 对比 ===")
    for k,v in results.items():
        print(f"  {k:24s} Kappa={v['kappa']:.4f} OA={v['oa']:.4f}")
    print("DONE", flush=True)
