"""模型间一致性分析: 检验"标签质量是否已成为模型能力上限"
思路:
  1) 用多个架构/容量差异极大的模型在 test_clean 上推理, 保存逐像素预测
  2) 计算 模型-模型 一致性 vs 模型-标签 一致性
     若 model-model 一致性 >> model-label 一致性, 则模型们"互相认同但不认同标签",
     说明分歧主要来自标签质量而非模型能力
  3) 用"多模型多数投票"作为代理真值(consensus label), 重新评估各模型
     若在共识标签下模型间差异被放大/缩小, 可判断能力差异是否被标签噪声掩盖
  4) 估计标签噪声率下界
产出: consensus_analysis/{preds.npz, consistency.json, report.md}
"""
import os, sys, json, time
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, FCN, DeepLabV3Plus
from experiment_matrix_v2 import FPNSeg

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = f"{BASE}/checkpoints"
OUT = f"{BASE}/consensus_analysis"
os.makedirs(OUT, exist_ok=True)
MAIN = paths.DATA_NEWSPLIT2
NUM_CLASSES, IGNORE, S, STRIDE = 7, 255, 256, 128
MEAN = np.array([0.298,0.3224,0.2949], np.float32); STD = np.array([0.1713,0.1415,0.1328], np.float32)
REMAP = np.full(256, 255, dtype=np.uint8)
for s in range(1, 8): REMAP[s] = s - 1

# 参与一致性分析的模型 (架构/容量差异大)
MODELS = [
    ("full_all_v2",  "MSSACT-light 6.1M", "light"),
    ("nd_deeplab",   "DeepLabV3+ 39.6M",  "deeplab"),
    ("nd_unet",      "U-Net 2.4M",        "unet"),
    ("nd_fcn",       "FCN 23.7M",         "fcn"),
    ("nd_fpn_seg",   "FPN 27.2M",         "fpn"),
    ("nd_swin_unet", "Swin-Unet 32.6M",   "swin"),
]

def gauss2d(s, sigma):
    ax = np.arange(s)-(s-1)/2.0
    g = np.exp(-(ax**2)/(2*sigma**2))
    return np.outer(g,g).astype(np.float32)

def build(kind):
    if kind == "light":
        return MSSACTNet(in_channels=3, num_classes=7, embed_dims=[32,64,128,256],
                         transformer_layers=2, transformer_heads=4)
    if kind == "deeplab": return DeepLabV3Plus()
    if kind == "unet": return UNet()
    if kind == "fcn": return FCN()
    if kind == "fpn": return FPNSeg()
    if kind == "swin": return SwinUnetLite() if False else __import__("experiment_matrix_v2").SwinUnetLite()
    raise ValueError(kind)

@torch.no_grad()
def infer_tile(model, img_u8, batch=4):
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

def main():
    d = f"{MAIN}/test_clean"
    names = sorted(set(os.listdir(f"{d}/images")) & set(os.listdir(f"{d}/masks")))
    print(f"test_clean: {len(names)} 张", flush=True)

    # 标签
    labels = np.stack([REMAP[np.array(Image.open(f"{d}/masks/{n}"))] for n in names])
    np.savez_compressed(f"{OUT}/labels.npz", labels=labels, names=names)

    preds = {}
    for tag, label, kind in MODELS:
        ckpt = f"{CKPT}/{tag}_best.pt"
        if not os.path.exists(ckpt): print(f"SKIP {tag} (no ckpt)", flush=True); continue
        cache = f"{OUT}/pred_{tag}.npz"
        if os.path.exists(cache):
            preds[label] = np.load(cache)["pred"]; print(f"  {tag} (cached)", flush=True); continue
        model = build(kind).to("cuda").eval()
        model.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
        t0 = time.time()
        P = np.stack([infer_tile(model, np.array(Image.open(f"{d}/images/{n}").convert("RGB"))) for n in names])
        np.savez_compressed(cache, pred=P)
        preds[label] = P
        print(f"  {tag} done ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    labels_list = list(preds.keys())
    print(f"\n参与分析: {labels_list}", flush=True)

    # ===== 1) 有效性掩码 (标签非 ignore) =====
    valid = labels != IGNORE

    # ===== 2) 模型-标签 与 模型-模型 一致性 =====
    n = len(labels_list)
    agree_label = {}; agree_model = {}
    for i, li in enumerate(labels_list):
        Pi = preds[li]
        agree_label[li] = float((Pi[valid] == labels[valid]).mean())
        for j, lj in enumerate(labels_list):
            if i >= j: continue
            Pj = preds[lj]
            agree_model[(li,lj)] = float((Pi[valid] == Pj[valid]).mean())

    # ===== 3) 多数投票共识标签 =====
    stack = np.stack([preds[l] for l in labels_list])            # (M,H,W)
    # 只在 valid 区域统计
    from scipy import stats as _st
    consensus = np.zeros_like(labels, dtype=np.uint8)
    for k in range(labels.shape[0]):
        v = valid[k]
        if v.sum() == 0: continue
        arr = stack[:, k, :, :]                                  # (M,H,W)
        # 逐像素众数
        cnt = np.zeros((NUM_CLASSES,) + arr.shape[1:], np.int32)
        for m in range(n):
            msk = arr[m]
            for c in range(NUM_CLASSES):
                cnt[c] += (msk == c)
        consensus[k] = cnt.argmax(0).astype(np.uint8)

    # ===== 4) 在共识标签下重新评估 =====
    results = {}
    for li in labels_list:
        P = preds[li]
        m = valid
        agree_cons = float((P[m] == consensus[m]).mean())
        agree_lbl  = float((P[m] == labels[m]).mean())
        results[li] = dict(agree_label=agree_lbl, agree_consensus=agree_cons,
                           delta=agree_cons-agree_lbl)
    # 共识标签 vs 人工标签
    cons_vs_label = float((consensus[valid] == labels[valid]).mean())

    out = dict(models=labels_list, n_tiles=len(names),
               model_label_agreement=agree_label,
               model_model_agreement={f"{a}|{b}": v for (a,b),v in agree_model.items()},
               consensus_vs_label=cons_vs_label,
               per_model_on_consensus=results)
    json.dump(out, open(f"{OUT}/consistency.json","w"), indent=1)

    # ===== 报告 =====
    lines = ["# 模型间一致性分析 (标签质量检验)\n",
             f"- 测试集: test_clean {len(names)} 张 (含 no-data 忽略)",
             f"- 参与模型: {len(labels_list)} 个 (架构/容量差异大)\n",
             "## 1. 模型 vs 人工标签 的一致性 (= OA)\n",
             "| 模型 | 与人工标签一致率 | 与共识(多数投票)一致率 | 差 |","|---|---|---|---|"]
    for li, r in sorted(results.items(), key=lambda t:-t[1]["agree_label"]):
        lines.append(f"| {li} | {r['agree_label']:.4f} | {r['agree_consensus']:.4f} | {r['delta']:+.4f} |")
    lines += ["\n## 2. 模型之间的一致性 (像素级)\n", "| 模型对 | 一致率 |","|---|---|"]
    for (a,b), v in sorted(agree_model.items(), key=lambda t:-t[1]):
        lines.append(f"| {a} vs {b} | {v:.4f} |")
    mm = np.mean(list(agree_model.values()))
    ml = np.mean(list(agree_label.values()))
    lines += [f"\n**平均 模型-模型 一致率: {mm:.4f}**",
              f"**平均 模型-标签 一致率: {ml:.4f}**",
              f"**共识标签 vs 人工标签: {cons_vs_label:.4f}**",
              "",
              "## 3. 判据",
              f"- 若 模型间一致率 ({mm:.3f}) 显著高于 模型-标签一致率 ({ml:.3f}) → 模型互相认同但不认同标签, 分歧主要来自标签质量",
              f"- 共识标签与人工标签的一致率 {cons_vs_label:.3f} → 约 {(1-cons_vs_label)*100:.1f}% 的像素上多数模型与人工标注不符"]
    open(f"{OUT}/report.md","w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n产出: {OUT}/", flush=True)

if __name__ == "__main__":
    main()
