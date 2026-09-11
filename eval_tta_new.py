"""新数据(newsplit2) TTA 评估: val(221) + test_clean(141) × 有/无TTA
用法: python eval_tta_new.py [tag1,tag2,...]   # 默认评估全部已有模型
"""
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, PSPNet, FCN, DeepLabV3Plus, SegFormerLite
from experiment_matrix_v2 import FPNSeg, SwinUnetLite, mssact_light

ROOT = paths.DATA_NEWSPLIT2
CKPT = paths.CKPT
OUT = paths.TTA2
os.makedirs(OUT, exist_ok=True)
NUM_CLASSES, IGNORE, S, STRIDE = 7, 255, 256, 128
MEAN = np.array([0.298,0.3224,0.2949], np.float32); STD = np.array([0.1713,0.1415,0.1328], np.float32)

def gauss2d(s, sigma):
    ax = np.arange(s)-(s-1)/2.0
    g = np.exp(-(ax**2)/(2*sigma**2))
    return np.outer(g,g).astype(np.float32)

def build_for(tag):
    """按 tag 前缀推断模型类型"""
    if tag.startswith("nd_abl_") or tag in ("full_all_v2","post_all_v2","v3_s256","mssact_full60"):
        d = dict(embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4)
        if "trans4l" in tag: d["transformer_layers"] = 4
        if "trans6l" in tag: d["transformer_layers"] = 6
        if "no_emr" in tag: d["use_emr"] = False
        if "no_ecsam" in tag: d["use_ecsam"] = False
        if "no_fpn" in tag: d["use_fpn"] = False
        if "no_trans" in tag: d["use_transformer"] = False
        if "no_adapter" in tag: d["use_adapter"] = False
        if "bilinear" in tag: d["upsample_mode"] = "bilinear"
        return MSSACTNet(in_channels=3, num_classes=7, **d)
    if tag in ("nd_mssact30m","nd_post30m","strat_mssact_full30m_oc60"):
        return MSSACTNet(in_channels=3, num_classes=7, embed_dims=[64,128,256,512],
                         transformer_layers=4, transformer_heads=8)
    if tag == "nd_unet": return UNet()
    if tag == "nd_pspnet": return PSPNet()
    if tag == "nd_fcn": return FCN()
    if tag == "nd_deeplab": return DeepLabV3Plus()
    if tag == "nd_segformer": return SegFormerLite()
    if tag == "nd_fpn_seg": return FPNSeg()
    if tag == "nd_swin_unet": return SwinUnetLite()
    # 回退: light MSSACT
    return MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256],
                     transformer_layers=2, transformer_heads=4)

@torch.no_grad()
def infer_tile(model, img_u8, tta=False, batch=8):
    H, W = img_u8.shape[:2]
    ph=(STRIDE-(H-S)%STRIDE)%STRIDE; pw=(STRIDE-(W-S)%STRIDE)%STRIDE
    x = np.pad(img_u8,((0,ph),(0,pw),(0,0)),mode="reflect").astype(np.float32)/255.0
    x = (x-MEAN)/STD
    xp = torch.from_numpy(x.transpose(2,0,1)[None])
    Hp, Wp = H+ph, W+pw
    gw = gauss2d(S, S/4.0)
    wins = [(y,x) for y in range(0,Hp-S+1,STRIDE) for x in range(0,Wp-S+1,STRIDE)]

    def run(fh=False, fw=False, rot=0):
        prob = np.zeros((NUM_CLASSES,Hp,Wp), np.float32); wsum = np.zeros((Hp,Wp), np.float32)
        src = xp
        if fh: src = torch.flip(src, dims=[3])
        if fw: src = torch.flip(src, dims=[2])
        if rot: src = torch.rot90(src, rot, dims=[2,3])
        for i in range(0,len(wins),batch):
            bx = torch.cat([src[:,:,y:y+S,x:x+S] for y,x in wins[i:i+batch]]).to("cuda")
            with torch.amp.autocast("cuda"):
                p = torch.softmax(model(bx),1).float().cpu().numpy()
            for j,(y,x) in enumerate(wins[i:i+batch]):
                prob[:,y:y+S,x:x+S] += p[j]*gw; wsum[y:y+S,x:x+S] += gw
        wsum[wsum==0]=1; prob/=wsum
        if rot: prob = np.rot90(prob, -rot, axes=(1,2))
        if fw: prob = prob[:, ::-1, :]
        if fh: prob = prob[:, :, ::-1]
        return prob
    if not tta:
        return run().argmax(0).astype(np.uint8)
    acc = np.zeros((NUM_CLASSES,Hp,Wp), np.float32)
    for rot in range(4):
        for fh in (False,True):
            for fw in (False,True): acc += run(fh,fw,rot)
    return acc.argmax(0).astype(np.uint8)

def evaluate(model, split, tta, tag, batch=8):
    d = f"{ROOT}/{split}"
    names = sorted(set(os.listdir(f"{d}/images")) & set(os.listdir(f"{d}/masks")))
    conf = np.zeros((NUM_CLASSES,NUM_CLASSES), np.int64); t0=time.time()
    for i,n in enumerate(names):
        img = np.array(Image.open(f"{d}/images/{n}").convert("RGB"))
        msk = np.array(Image.open(f"{d}/masks/{n}"))
        # 正确 REMAP: 0(no-data) 及非1-7 -> 255(ignore); 1..7 -> 0..6
        remap = np.full(256, 255, dtype=np.uint8)
        for s in range(1,8): remap[s] = s-1
        msk = remap[msk]
        pred = infer_tile(model, img, tta=tta, batch=batch)
        v = msk != IGNORE
        conf += np.bincount((msk[v].astype(np.int64)*NUM_CLASSES+pred[v]).ravel(),
                            minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
        if (i+1) % 60 == 0: print(f"  {tag} {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    c = conf.astype(np.float64)
    oa = np.trace(c)/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
    kappa = (oa-pe)/(1-pe)
    f1s = [float(2*c[k,k]/max(2*c[k,k]+c[:,k].sum()+c[k,:].sum()-2*c[k,k],1)) for k in range(NUM_CLASSES)]
    ious = [float(c[k,k]/max(c[k,k]+c[:,k].sum()+c[k,:].sum()-2*c[k,k],1)) for k in range(NUM_CLASSES)]
    return dict(oa=float(oa),kappa=float(kappa),mf1=float(np.mean(f1s)),miou=float(np.mean(ious)),
                f1=f1s,iou=ious,confusion=c.tolist(),seconds=round(time.time()-t0),n_tiles=len(names))

if __name__ == "__main__":
    tags = sys.argv[1].split(",") if len(sys.argv)>1 else None
    if tags is None:
        tags = [t.replace("_best.pt","") for t in sorted(os.listdir(CKPT))
                if t.endswith("_best.pt") and (t.startswith("nd_") or t.startswith("full_all") or t.startswith("post_all"))]
    print(f"评估 {len(tags)} 个模型: {tags}", flush=True)
    results = {}
    for tag in tags:
        ckpt = f"{CKPT}/{tag}_best.pt"
        if not os.path.exists(ckpt): print(f"SKIP {tag} (no ckpt)"); continue
        model = build_for(tag).to("cuda").eval()
        r = model.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
        if len(r.missing_keys) + len(r.unexpected_keys) > 0:
            print(f"  [warn] {tag} miss={len(r.missing_keys)} unexp={len(r.unexpected_keys)}", flush=True)
        for split in ["val","test_clean"]:
            for tta in [False, True]:
                key = f"{tag}_{split}{'_TTA' if tta else ''}"
                if os.path.exists(f"{OUT}/{key}.json"):
                    print(f"  SKIP {key} (done)", flush=True); continue
                res = evaluate(model, split, tta, key)
                json.dump(res, open(f"{OUT}/{key}.json","w"), indent=1)
                results[key] = res
                print(f"== {key:36s} K={res['kappa']:.4f} OA={res['oa']:.4f} mF1={res['mf1']:.4f} ({res['seconds']}s)", flush=True)
        del model; torch.cuda.empty_cache()
    print("DONE", flush=True)
