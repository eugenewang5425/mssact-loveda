"""数据阶梯/交互实验评价: 泛化能力 + 过拟合度 + 学习速率
产出: ladder_eval/ 下的 json + 汇总表 + 图
用法: python eval_ladder.py
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
import os, sys, json, time, glob
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, _HERE)
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, FCN, DeepLabV3Plus

BASE = _BASE
CKPT = f"{BASE}/checkpoints"
OUT = f"{BASE}/ladder_eval"
os.makedirs(OUT, exist_ok=True)
MAIN = paths.DATA_NEWSPLIT2
NUM_CLASSES, IGNORE, S, STRIDE = 7, 255, 256, 128
MEAN = np.array([0.298,0.3224,0.2949], np.float32); STD = np.array([0.1713,0.1415,0.1328], np.float32)
REMAP = np.full(256, 255, dtype=np.uint8)
for s in range(1, 8): REMAP[s] = s - 1        # 官方: 1..7->0..6; 0(no-data)->255(ignore)

def gauss2d(s, sigma):
    ax = np.arange(s)-(s-1)/2.0
    g = np.exp(-(ax**2)/(2*sigma**2))
    return np.outer(g,g).astype(np.float32)

def build_for(tag):
    if "deeplab" in tag:
        return DeepLabV3Plus()
    if "unet" in tag:
        return UNet()
    if "fcn" in tag:
        return FCN()
    # mssact variants
    d = dict(embed_dims=[32,64,128,256], transformer_layers=2, transformer_heads=4)
    if "noecsam" in tag: d["use_ecsam"] = False
    if "noemr" in tag: d["use_emr"] = False
    return MSSACTNet(in_channels=3, num_classes=7, **d)

@torch.no_grad()
def infer_tile(model, img_u8, batch=8):
    H, W = img_u8.shape[:2]
    ph=(STRIDE-(H-S)%STRIDE)%STRIDE; pw=(STRIDE-(W-S)%STRIDE)%STRIDE
    x = np.pad(img_u8,((0,ph),(0,pw),(0,0)),mode="reflect").astype(np.float32)/255.0
    x = (x-MEAN)/STD
    xp = torch.from_numpy(x.transpose(2,0,1)[None])
    Hp, Wp = H+ph, W+pw
    prob = np.zeros((NUM_CLASSES,Hp,Wp), np.float32); wsum = np.zeros((Hp,Wp), np.float32)
    gw = gauss2d(S, S/4.0)
    wins = [(y,x) for y in range(0,Hp-S+1,STRIDE) for x in range(0,Wp-S+1,STRIDE)]
    for i in range(0,len(wins),batch):
        bx = torch.cat([xp[:,:,y:y+S,x:x+S] for y,x in wins[i:i+batch]]).to("cuda")
        with torch.amp.autocast("cuda"):
            p = torch.softmax(model(bx),1).float().cpu().numpy()
        for j,(y,x) in enumerate(wins[i:i+batch]):
            prob[:,y:y+S,x:x+S] += p[j]*gw; wsum[y:y+S,x:x+S] += gw
    wsum[wsum==0]=1
    return (prob/wsum).argmax(0).astype(np.uint8)

def evaluate_split(model, split, max_tiles=None):
    d = f"{MAIN}/{split}"
    names = sorted(set(os.listdir(f"{d}/images")) & set(os.listdir(f"{d}/masks")))
    if max_tiles: names = names[:max_tiles]
    conf = np.zeros((NUM_CLASSES,NUM_CLASSES), np.int64)
    for n in names:
        img = np.array(Image.open(f"{d}/images/{n}").convert("RGB"))
        msk = REMAP[np.array(Image.open(f"{d}/masks/{n}"))]
        pred = infer_tile(model, img)
        v = msk != IGNORE
        conf += np.bincount((msk[v].astype(np.int64)*NUM_CLASSES+pred[v]).ravel(),
                            minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    c = conf.astype(np.float64)
    oa = np.trace(c)/c.sum(); pe = (c.sum(1)*c.sum(0)).sum()/c.sum()**2
    kappa = (oa-pe)/(1-pe)
    f1s = [float(2*c[k,k]/max(2*c[k,k]+c[:,k].sum()+c[k,:].sum()-2*c[k,k],1)) for k in range(NUM_CLASSES)]
    return dict(oa=float(oa), kappa=float(kappa), mf1=float(np.mean(f1s)), n_tiles=len(names))

@torch.no_grad()
def entropy_gap(model, n_patches=40):
    """过拟合度: 训练集 vs 验证集 的平均预测熵差 (ΔH<0 => 过拟合)"""
    import random
    random.seed(0)
    Hs = {}
    for split in ["train","val"]:
        d = f"{MAIN}/{split}"
        names = sorted(os.listdir(f"{d}/images"))
        random.shuffle(names); names = names[:n_patches]
        H = []
        for n in names:
            img = np.array(Image.open(f"{d}/images/{n}").convert("RGB"))
            Hh, Ww = img.shape[:2]
            y = random.randint(0, max(0,Hh-S)); x = random.randint(0, max(0,Ww-S))
            patch = (img[y:y+S, x:x+S]/255.0 - MEAN)/STD
            xt = torch.from_numpy(patch.transpose(2,0,1)[None]).float().to("cuda")
            with torch.amp.autocast("cuda"):
                logits = model(xt).float()
            probs = torch.softmax(logits, dim=1)[0]
            H.append(float(-(probs*(probs+1e-8).log()).sum(dim=0).mean().item()))
        Hs[split] = float(np.mean(H))
    return Hs["train"], Hs["val"], Hs["train"]-Hs["val"]

def curve_metrics(h):
    """学习速率指标"""
    ks = np.array([r["kappa"] for r in h]); eps = np.array([r["epoch"] for r in h])
    def first(th):
        i = np.where(ks >= th)[0]
        return int(eps[i[0]]) if len(i) else None
    return dict(n_epochs=len(h), k_final=float(ks[-1]), k_max=float(ks.max()),
                auc=float(ks.mean()), e40=first(0.4), e50=first(0.5), e60=first(0.6))

if __name__ == "__main__":
    tags = sorted({os.path.basename(p).replace("_history.json","")
                   for p in glob.glob(f"{CKPT}/lg_*_history.json")})
    print(f"待评估: {len(tags)} 个实验", flush=True)
    rows = []
    for tag in tags:
        ckpt = f"{CKPT}/{tag}_best.pt"
        if not os.path.exists(ckpt): continue
        rec = {"tag": tag}
        # 1) 学习速率 (从 history)
        try:
            h = json.load(open(f"{CKPT}/{tag}_history.json"))
            rec.update(curve_metrics(h))
        except Exception as e:
            print(f"  skip {tag} history: {e}"); continue
        # 2) 泛化 + 过拟合 (需推理)
        cache = f"{OUT}/{tag}.json"
        if os.path.exists(cache):
            rec.update(json.load(open(cache)))
        else:
            model = build_for(tag).to("cuda").eval()
            r = model.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
            val = evaluate_split(model, "val")
            tst = evaluate_split(model, "test_clean")
            Htr, Hva, dH = entropy_gap(model)
            rec.update(dict(val_kappa=val["kappa"], val_oa=val["oa"],
                            test_kappa=tst["kappa"], test_oa=tst["oa"],
                            gen_gap=val["kappa"]-tst["kappa"],
                            H_train=Htr, H_val=Hva, dH=dH))
            json.dump(rec, open(cache,"w"), indent=1)
            del model; torch.cuda.empty_cache()
        rows.append(rec)
        print(f"== {tag:22s} K_final={rec.get('k_final',0):.4f} val={rec.get('val_kappa',0):.4f} "
              f"test={rec.get('test_kappa',0):.4f} gap={rec.get('gen_gap',0):+.4f} dH={rec.get('dH',0):+.3f}", flush=True)

    json.dump(rows, open(f"{OUT}/ladder_summary.json","w"), indent=1)
    # 打印汇总表
    print("\n" + "="*118)
    print(f"{'实验':22s} {'轮':>3s} {'K_final':>8s} {'K_max':>8s} {'AUC':>7s} {'e40':>4s} {'e50':>4s} {'e60':>4s} "
          f"{'val_K':>7s} {'test_K':>7s} {'泛化gap':>8s} {'ΔH':>7s}")
    print("="*118)
    for r in sorted(rows, key=lambda x: (len(x['tag']), x['tag'])):
        print(f"{r['tag']:22s} {r.get('n_epochs',0):3d} {r.get('k_final',0):8.4f} {r.get('k_max',0):8.4f} "
              f"{r.get('auc',0):7.4f} {str(r.get('e40') or '--'):>4s} {str(r.get('e50') or '--'):>4s} {str(r.get('e60') or '--'):>4s} "
              f"{r.get('val_kappa',0):7.4f} {r.get('test_kappa',0):7.4f} {r.get('gen_gap',0):+8.4f} {r.get('dH',0):+7.3f}")
    print("="*118)
    print("列说明: K_final=末轮验证Kappa, K_max=最优, AUC=曲线均值(学习速率面积), e40/50/60=首次达阈值轮次")
    print("        val_K/test_K=滑窗推理, 泛化gap=val-test(越接近0越好), ΔH=训练/验证熵差(负=过拟合)")
    print("DONE", flush=True)
