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
import torch
from PIL import Image

sys.path.insert(0, _HERE)
sys.path.insert(0, paths.REPO)
from models.msscactnet import MSSACTNet
from experiment_matrix import UNet, FCN, DeepLabV3Plus
from experiment_matrix_v2 import FPNSeg

BASE = _BASE
CKPT = paths.CKPT
OUT = paths.CONSENSUS
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
    """按 tag 构造模型。

    2026-09-14: 本地 build(kind) 的 kind 是**架构成分**的字符串, 与 tag 不是一对一,
    故新增 tag 必须改这里 —— 这正是 G1 消融/lg_D1..D4/lgR_D1..D4 长期无法推理的
    一部分原因。现改为直接复用 analysis/model_registry.py 的 tag -> 模型登记表
    （实验/评估/推理三处共用一份映射）。保留 kind 形式的调用以兼容旧调用点。
    """
    from model_registry import build_for_tag, TAG_CFG
    if kind in TAG_CFG:                 # 直接传 tag
        return build_for_tag(kind)
    # 兼容旧的 kind 字符串
    _alias = {"light": "full_all_v2", "deeplab": "nd_deeplab", "unet": "nd_unet",
              "fcn": "nd_fcn", "fpn": "nd_fpn_seg", "swin": "nd_swin_unet"}
    if kind in _alias:
        return build_for_tag(_alias[kind])
    raise ValueError("未知 kind/tag: %r" % kind)

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

def test_clean_names():
    """test_clean 的 (images ∩ masks) 文件名, 已排序"""
    d = f"{MAIN}/test_clean"
    return sorted(set(os.listdir(f"{d}/images")) & set(os.listdir(f"{d}/masks")))


def expected_shape():
    """labels.npz 的形状; 缺失返回 None"""
    lp = f"{OUT}/labels.npz"
    if not os.path.exists(lp):
        return None
    return np.load(lp, allow_pickle=True)["labels"].shape


def cache_is_valid(tag):
    """缓存存在且形状与 labels 一致才算有效。

    历史遗留: 早期 eval_rigor.infer_tag 走 LoveDADataset(train=False)
    (n_val_patches=4, crop=256) 会产出 (564,256,256) 的**分块**布局, 与
    labels.npz 的 (141,1024,1024) 永远不一致。那种缓存必须判为无效并重推 ——
    否则它既不重推、又被形状检查跳过, 该 tag 就永远进不了测试集协议。
    """
    p = f"{OUT}/pred_{tag}.npz"
    if not os.path.exists(p):
        return False, "无缓存"
    want = expected_shape()
    if want is None:
        return True, "无 labels.npz 可校验"
    got = np.load(p)["pred"].shape
    if got != want:
        return False, "缓存形状 %s != 标签 %s" % (got, want)
    return True, "有效"


def infer_and_cache(tag, names=None, force=False, batch=4):
    """对任意**已登记** tag 做整图滑窗推理, 写 pred_{tag}.npz, 返回预测数组。

    这是全项目整图推理的**唯一实现**（eval_rigor 与 run_post_fix 都调用它）:
      * 滑窗 S=256 / STRIDE=128 + 高斯加权, 与训练裁剪尺度一致
      * 输出 (141, 1024, 1024), 与 labels.npz 同布局
    """
    ok, why = cache_is_valid(tag)
    if ok and not force:
        return np.load(f"{OUT}/pred_{tag}.npz")["pred"]
    if (not ok) and os.path.exists(f"{OUT}/pred_{tag}.npz"):
        print("    %s: %s -> 重新推理" % (tag, why), flush=True)
    ck = f"{CKPT}/{tag}_best.pt"
    if not os.path.exists(ck):
        print("    %s: 无权重, 跳过" % tag, flush=True)
        return None
    from model_registry import build_for_tag
    names = names or test_clean_names()
    d = f"{MAIN}/test_clean"
    model = build_for_tag(tag).to("cuda").eval()
    sd = torch.load(ck, map_location="cuda", weights_only=True)
    model.load_state_dict(sd, strict=False)
    t0 = time.time()
    P = np.stack([infer_tile(model, np.array(Image.open(f"{d}/images/{n}").convert("RGB")),
                             batch=batch) for n in names])
    np.savez_compressed(f"{OUT}/pred_{tag}.npz", pred=P)
    print("    %s: 推理完成 %s (%.0fs)" % (tag, P.shape, time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()
    return P


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
        ok, why = cache_is_valid(tag)
        if ok:
            preds[label] = np.load(f"{OUT}/pred_{tag}.npz")["pred"]
            print(f"  {tag} (cached)", flush=True); continue
        print(f"  {tag}: {why} -> 重新推理", flush=True)
        model = build(tag).to("cuda").eval()
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
