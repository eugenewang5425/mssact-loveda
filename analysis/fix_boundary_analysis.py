"""
修正 boundary_analysis.json —— 原版存在定义错误, 导致报告误报"93% 像素在类边界上"

原错误
------
原实现把整批标签 (N,H,W) 三维数组直接做 `np.diff(..., axis=0/1)`, axis=0 实际是
"图像索引轴", 于是**跨图**的像素差 (图与图之间本来就不同) 被全部误判为类边界,
得出 0.9296 的假边界比例。同时把 valid<->255(ignore) 的过渡也算作类边界。

正确做法
--------
逐图 (2D) 计算, 且只把"两个不同**有效**类别"之间的相邻像素记为边界。

真实结论 (test_clean 141 张)
  边界像素(距离0) 仅 2.4%, 距边界 >8px 的"内部"像素占 78.4%
  -> 标签并不碎片化; 原报告"标签碎片化"与"内部一致性更差"两条结论均不成立
  -> 正确规律: 准确率随距边界距离**单调上升** (边界 0.45 -> 内部 0.70)

用法: python fix_boundary_analysis.py
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
import os, sys, json, time, shutil
import numpy as np
from scipy import ndimage

sys.path.insert(0, _HERE)
import paths

CONS = paths.CONSENSUS
IGNORE = 255
N_CLS = 7
BANDS = [(0, 0.5, "0 (边界像素)"), (0.5, 2.5, "1-2"), (2.5, 4.5, "3-4"),
         (4.5, 8.5, "5-8"), (8.5, 1e9, ">8 (内部)")]

MODELS = ["full_all_v2", "nd_deeplab", "nd_unet", "nd_fcn", "nd_fpn_seg", "nd_swin_unet"]
LABELS = {
    "full_all_v2": "MSSACT-light 6.1M", "nd_deeplab": "DeepLabV3+ 39.6M",
    "nd_unet": "U-Net 2.4M", "nd_fcn": "FCN 23.7M", "nd_fpn_seg": "FPN 27.2M",
    "nd_swin_unet": "Swin-Unet 32.6M",
}


def edge_mask(lab2d):
    valid = lab2d != IGNORE
    e = np.zeros_like(valid, dtype=bool)
    for ax in (0, 1):
        a = np.take(lab2d, range(0, lab2d.shape[ax] - 1), axis=ax)
        b = np.take(lab2d, range(1, lab2d.shape[ax]), axis=ax)
        d = (a != b) & (a != IGNORE) & (b != IGNORE)
        s = [slice(None)] * 2; s[ax] = slice(0, -1); e[tuple(s)] |= d
        s2 = [slice(None)] * 2; s2[ax] = slice(1, None); e[tuple(s2)] |= d
    return e & valid, valid


def main():
    z = np.load(os.path.join(CONS, "labels.npz"), allow_pickle=True)
    labels = z["labels"]
    n = labels.shape[0]
    preds = {}
    for m in MODELS:
        p = os.path.join(CONS, f"pred_{m}.npz")
        if os.path.exists(p):
            preds[m] = np.load(p)["pred"]

    # 逐图距离场 + 逐带统计
    band_n = {nm: 0 for _, _, nm in BANDS}
    n_valid = 0
    band_agree = {m: {nm: [0, 0] for _, _, nm in BANDS} for m in preds}   # 模型 vs 标签
    band_cons = {nm: [0, 0] for _, _, nm in BANDS}                        # 共识 vs 标签
    band_pair = {nm: [0, 0] for _, _, nm in BANDS}                        # 模型间两两一致性

    for i in range(n):
        lab = labels[i]
        e, valid = edge_mask(lab)
        dist = ndimage.distance_transform_edt(~e)
        n_valid += int(valid.sum())
        # 共识 = 多数投票 (含 ignore: 任一模型给出有效类才计入)
        stack = np.stack([preds[m][i] for m in preds if m in preds], 0)   # (M,H,W)
        cnt = np.stack([(stack == c).sum(0) for c in range(N_CLS)], 0)    # (C,H,W)
        cons = cnt.argmax(0).astype(np.uint8)
        cmax = cnt.max(0)
        tie = (cnt == cmax).sum(0) > 1
        for lo, hi, nm in BANDS:
            m = valid & (dist >= lo) & (dist < hi)
            k = int(m.sum())
            if k == 0:
                continue
            band_n[nm] += k
            for mm in preds:
                band_agree[mm][nm][0] += int((preds[mm][i][m] == lab[m]).sum())
                band_agree[mm][nm][1] += k
            ok = m & ~tie
            band_cons[nm][0] += int((cons[ok] == lab[ok]).sum())
            band_cons[nm][1] += int(ok.sum())
            # 模型间: 平均两两一致率 (逐像素)
            ms = list(preds.keys())
            tot = corr = 0
            for x in range(len(ms)):
                for y in range(x + 1, len(ms)):
                    a, b = preds[ms[x]][i][m], preds[ms[y]][i][m]
                    corr += int((a == b).sum()); tot += k
            if tot:
                band_pair[nm][0] += corr; band_pair[nm][1] += tot

    def frac(nm): return round(band_n[nm] / n_valid, 4) if n_valid else None
    out_bands = {nm: dict(frac=frac(nm), n=band_n[nm]) for _, _, nm in BANDS}

    res = {
        "_correction": {
            "date": time.strftime("%Y-%m-%d"),
            "reason": "原实现把 (N,H,W) 三维标签直接沿 axis=0 求差, 跨图像素差被误判为类边界, "
                      "得出 0.9296 的假边界比例; 且未排除 valid<->255(ignore) 过渡。",
            "fix": "逐图 (2D) 计算, 仅统计两个不同有效类别之间的相邻像素。",
            "old_boundary_frac": 0.9296,
            "new_boundary_frac": frac("0 (边界像素)"),
            "impact": "原报告'标签碎片化 93%'与'越纯净区域模型越不同意标签'两条结论不成立; "
                      "正确规律为准确率随距边界距离单调上升。",
        },
        "boundary_frac": frac("0 (边界像素)"),
        "boundary_frac_within8px": round(sum(band_n[nm] for _, _, nm in BANDS
                                             if nm != ">8 (内部)") / n_valid, 4),
        "band_frac": out_bands,
        "consensus_vs_label_by_band": {
            nm: (round(band_cons[nm][0] / band_cons[nm][1], 4) if band_cons[nm][1] else None)
            for _, _, nm in BANDS},
        "per_model_vs_label_by_band": {
            LABELS.get(m, m): {nm: (round(band_agree[m][nm][0] / band_agree[m][nm][1], 4)
                                    if band_agree[m][nm][1] else None) for _, _, nm in BANDS}
            for m in preds},
        "pair_disagreement_by_band": {
            nm: (round(1 - band_pair[nm][0] / band_pair[nm][1], 4) if band_pair[nm][1] else None)
            for _, _, nm in BANDS},
        "n_images": n,
        "n_valid_pixels": n_valid,
    }

    dst = os.path.join(CONS, "boundary_analysis.json")
    if os.path.exists(dst):
        bak = os.path.join(CONS, f"boundary_analysis.bak_{time.strftime('%Y%m%d_%H%M')}.json")
        shutil.copy2(dst, bak)
        print(f"已备份原文件 -> {os.path.basename(bak)}", flush=True)
    json.dump(res, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n修正后 boundary_analysis.json")
    print(f"  边界像素(距离0)占比 = {res['boundary_frac']:.4f}  (原误报 0.9296)")
    print(f"  ≤8px 占比          = {res['boundary_frac_within8px']:.4f}")
    print(f"\n  {'band':>12s} {'占比':>7s} {'共识vs标签':>11s} {'模型间分歧':>11s}")
    for _, _, nm in BANDS:
        c = res["consensus_vs_label_by_band"][nm]
        d = res["pair_disagreement_by_band"][nm]
        print(f"  {nm:>12s} {res['band_frac'][nm]['frac']:>7.4f} {c:>11} {d:>11}")
    print(f"\n  逐模型 vs 标签 一致性:")
    for m, r in res["per_model_vs_label_by_band"].items():
        print(f"    {m:22s} " + " ".join(f"{r[nm]:.3f}" for _, _, nm in BANDS))


if __name__ == "__main__":
    main()
