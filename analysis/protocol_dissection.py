# -*- coding: utf-8 -*-
"""协议解剖实验 —— 把 val 与整图两种评估协议的差异分解为可单独测量的因子

背景（为什么需要这个实验）
------------------------
同一组消融在 val 协议与整图协议下 **6/8 符号相反**（报告 §5.6.16）。但在此之前
必须先纠正一个事实错误：**两种协议的推理窗口都是 256²** ——

* val 协议: 每图取 **4 个固定角 patch**（256²），无重叠、无混合，混淆矩阵在
  全部 crop 上汇总（train_v3.LoveDADataset.__getitem__ 的 else 分支）
* 整图协议: 256² 滑窗（stride 128）+ **高斯混合**，覆盖整张 1024²，
  逐图混淆后汇总（consensus_analysis.infer_tile）

故此前报告里"整图协议测跨尺度泛化、模型从未见过 1024²"的表述**是错的**——
模型在两种协议下看到的都是 256² 窗口。真实的协议差异有四个因子：

    F1 图像集   val(221 张) vs test_clean(141 张)
    F2 覆盖率   仅 4 角 patch(每图 262k/1049k 像素 ≈ 25%) vs 全图
    F3 窗口混合 无 vs 重叠滑窗 + 高斯加权
    F4 窗口尺度 256²（两者相同）vs 更大窗口（未测过）

本实验用**同一批 checkpoint** 在 test_clean 上跑四臂，逐因子隔离：

    臂 A  val-corner      [已有] = 训练 history 里的 best Kappa（val 协议原值）
    臂 B  test-corner     [新]   = test_clean 上复刻 4 角 patch、同 A 的汇聚方式
                                   → 与 D 之差 = F1+F2+F3 的联合
    臂 C  test-dense      [新]   = test_clean 上 256² 网格 stride 256、无混合
                                   → 与 B 之差 = F2 覆盖率；与 D 之差 = F3 混合
    臂 D  test-slide-blend[已有] = rigor.json 的 kappa_pooled（现整图协议）
    臂 E  test-win512     [新]   = 512² 窗口 stride 256 + 高斯
                                   → 与 D 之差 = F4 窗口尺度

预登记的判读规则（写在代码里，不是看结果再定）：
    若 Δ(变体) 在 B ≈ A            → 符号翻转主要由**图像集**驱动
    若 Δ(变体) 在 B ≈ D            → 主要由**评估流程**（覆盖/混合/汇聚）驱动
    C 与 D 之差显著                → 高斯混合是必要成分
    E 与 D 之差显著                → 窗口尺度有独立作用

Checkpoint 集 = 未修复架构上符号翻转过的 6 个：full_all_v2 与 5 个消融。

用法: python analysis/protocol_dissection.py
输出: results/artifacts/protocol_dissection.json
"""

import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from PIL import Image

import paths
import eval_rigor as ER                      # pooled_kappa / conf_mat 口径一致
import consensus_analysis as CA              # MEAN/STD/REMAP 与整图协议完全一致

S = 256                                      # 窗口边长（与训练/整图协议一致）
TAGS = ["full_all_v2", "nd_abl_no_emr", "nd_abl_no_ecsam", "nd_abl_no_fpn",
        "nd_abl_bilinear", "nd_abl_no_trans"]
REF = "full_all_v2"
OUT = os.path.join(paths.RESULTS, "artifacts", "protocol_dissection.json")


def log(msg):
    print(msg, flush=True)


def load_model(tag):
    from model_registry import build_for_tag
    ck = os.path.join(paths.CKPT, "%s_best.pt" % tag)
    if not os.path.exists(ck):
        return None
    m = build_for_tag(tag).to("cuda").eval()
    sd = torch.load(ck, map_location="cuda", weights_only=True)
    m.load_state_dict(sd, strict=False)
    return m


def norm(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    x = (x - CA.MEAN) / CA.STD
    return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def infer_corners(model, img_u8, msk):
    """臂 B: 复刻 val 协议的 4 固定角 patch; 返回每 crop 的混淆矩阵列表"""
    H, W = msk.shape[:2]
    ys, xs = [0, H - S], [0, W - S]
    crops = []
    for p in range(4):
        y, x = ys[p // 2], xs[p % 2]
        crops.append((y, x))
    batch = torch.stack([norm(img_u8[y:y + S, x:x + S]) for y, x in crops]).to("cuda")
    with torch.amp.autocast("cuda"):
        out = model(batch)
    pred = out.argmax(1).cpu().numpy()
    cms = []
    for j, (y, x) in enumerate(crops):
        cms.append(ER.conf_mat(pred[j], CA.REMAP[msk[y:y + S, x:x + S]]))
    return cms


@torch.no_grad()
def infer_windows(model, img_u8, win, stride, blend):
    """臂 C/E 通用: win² 窗口按 stride 滑动; blend=True 时高斯加权(同 infer_tile)"""
    H, W = img_u8.shape[:2]
    ph = (stride - (H - win) % stride) % stride
    pw = (stride - (W - win) % stride) % stride
    x = np.pad(img_u8, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    xp = norm(x)[None]
    Hp, Wp = H + ph, W + pw
    if blend:
        gw = CA.gauss2d(win, win / 4.0)
    prob = np.zeros((CA.NUM_CLASSES, Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32) if blend else None
    pos = [(y, xx) for y in range(0, Hp - win + 1, stride)
           for xx in range(0, Wp - win + 1, stride)]
    for i in range(0, len(pos), 4):
        bx = torch.cat([xp[:, :, y:y + win, xx:xx + win] for y, xx in pos[i:i + 4]]).to("cuda")
        with torch.amp.autocast("cuda"):
            p = torch.softmax(model(bx), 1).float().cpu().numpy()
        for j, (y, xx) in enumerate(pos[i:i + 4]):
            if blend:
                prob[:, y:y + win, xx:xx + win] += p[j] * gw
                wsum[y:y + win, xx:xx + win] += gw
            else:
                prob[:, y:y + win, xx:xx + win] = p[j]          # 无重叠 → 直接写
    if blend:
        wsum[wsum == 0] = 1
        pred = (prob / wsum).argmax(0).astype(np.uint8)
    else:
        pred = prob.argmax(0).astype(np.uint8)
    return pred[:H, :W]


def val_kappa_from_history(tag):
    """臂 A: val 协议原值 = history 里的 best Kappa（val 即选模依据）"""
    p = os.path.join(paths.CKPT, "%s_history.json" % tag)
    if not os.path.exists(p):
        return None
    h = json.load(open(p, encoding="utf-8"))
    return max(r["kappa"] for r in h)


def main():
    d = os.path.join(CA.MAIN, "test_clean")     # MAIN 定义在 consensus_analysis 侧
    names = CA.test_clean_names()
    log("test_clean: %d 张" % len(names))

    rigor = json.load(open(os.path.join(paths.CONSENSUS, "rigor.json"),
                           encoding="utf-8"))["results"]

    rows = {}
    for tag in TAGS:
        t0 = time.time()
        model = load_model(tag)
        if model is None:
            log("  %s: 无权重, 跳过" % tag)
            continue
        cms_B, cms_C, cms_E = [], [], []
        for n in names:
            img = np.array(Image.open(os.path.join(d, "images", n)).convert("RGB"))
            msk = np.array(Image.open(os.path.join(d, "masks", n)))
            # B: 4 角 patch（val 协议复刻）
            cms_B += infer_corners(model, img, msk)
            # C: 256² 网格 stride 256, 无混合（全覆盖、无混合）
            prC = infer_windows(model, img, S, S, blend=False)
            cms_C.append(ER.conf_mat(prC, CA.REMAP[msk]))
            # E: 512² 窗口 stride 256 + 高斯（窗口尺度因子）
            prE = infer_windows(model, img, 512, 256, blend=True)
            cms_E.append(ER.conf_mat(prE, CA.REMAP[msk]))
        kA = val_kappa_from_history(tag)
        kB = float(ER.pooled_kappa(np.array(cms_B)))
        kC = float(ER.pooled_kappa(np.array(cms_C)))
        kD = (rigor.get(tag) or {}).get("kappa_pooled")
        kE = float(ER.pooled_kappa(np.array(cms_E)))
        rows[tag] = dict(
            A_val_corner=kA, B_test_corner=round(kB, 6),
            C_test_dense_noblend=round(kC, 6),
            D_test_slide_blend=kD, E_test_win512=round(kE, 6))
        del model
        torch.cuda.empty_cache()
        log("  %-16s A=%.4f B=%.4f C=%.4f D=%s E=%.4f  (%.0fs)" % (
            tag, kA, kB, kC, "%.4f" % kD if kD else "-", kE, time.time() - t0))

    # 预登记的判读: 对每个变体, 给出各臂相对 full 的 Δ, 并自动归类
    ref = rows.get(REF, {})
    variants = {}
    for tag, r in rows.items():
        if tag == REF:
            continue
        d_arms = {}
        for arm in ("A_val_corner", "B_test_corner", "C_test_dense_noblend",
                    "D_test_slide_blend", "E_test_win512"):
            if r.get(arm) is not None and ref.get(arm) is not None:
                d_arms[arm] = round(r[arm] - ref[arm], 6)
        # 归类: B 更接近 A 还是 D?
        def close(x, y):
            return abs(x - y)
        verdict = None
        if "A_val_corner" in d_arms and "B_test_corner" in d_arms and "D_test_slide_blend" in d_arms:
            dA, dB, dD = d_arms["A_val_corner"], d_arms["B_test_corner"], d_arms["D_test_slide_blend"]
            if close(dB, dA) <= close(dB, dD):
                verdict = "图像集驱动(ΔB≈ΔA)"
            else:
                verdict = "评估流程驱动(ΔB≈ΔD)"
        variants[tag] = dict(deltas=d_arms, verdict=verdict)

    out = dict(
        note=("协议解剖: 两协议窗口均为 256²; 差异因子 = 图像集(F1)/覆盖率(F2)/"
              "混合(F3)/窗口尺度(F4)。臂定义见模块 docstring。"),
        arms=dict(A="val 4角patch(原值=history best)",
                  B="test_clean 4角patch(复刻val)",
                  C="test_clean 256²网格stride256无混合",
                  D="test_clean 滑窗+高斯(=现整图协议,取自rigor.json)",
                  E="test_clean 512²窗口stride256+高斯"),
        metric="pooled Kappa (eval_rigor.pooled_kappa, 与报告同口径)",
        ref=REF, rows=rows, variants=variants,
        reading_rule=("ΔB≈ΔA → 图像集驱动; ΔB≈ΔD → 评估流程驱动; "
                      "C vs D → 混合作用; E vs D → 窗口尺度作用"),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("\n写入 %s" % OUT)

    log("\n=== 预登记判读 ===")
    log("%-22s %10s %10s %10s %10s %10s  %s" % (
        "变体", "ΔA(val)", "ΔB(corner)", "ΔC(dense)", "ΔD(blend)", "ΔE(win512)", "归类"))
    for tag, v in variants.items():
        dd = v["deltas"]
        log("%-22s %10s %10s %10s %10s %10s  %s" % (
            tag.replace("nd_abl_", "w/o "),
            "%+.4f" % dd["A_val_corner"] if "A_val_corner" in dd else "-",
            "%+.4f" % dd.get("B_test_corner", 0) if "B_test_corner" in dd else "-",
            "%+.4f" % dd.get("C_test_dense_noblend", 0) if "C_test_dense_noblend" in dd else "-",
            "%+.4f" % dd.get("D_test_slide_blend", 0) if "D_test_slide_blend" in dd else "-",
            "%+.4f" % dd.get("E_test_win512", 0) if "E_test_win512" in dd else "-",
            v["verdict"] or "-"))


if __name__ == "__main__":
    main()
