"""
严格评价脚本 (eval_rigor.py) —— 为"无差异"结论提供统计学依据

动机
----
原始报告用单次运行的 val Kappa 比较架构, 存在三个被文献明确指出的缺陷:
  1. 从未测量随机种子方差 -> ΔKappa 0.006 无法与噪声比较
  2. 聚合指标 (pooled Kappa/OA) 没有置信区间 -> "无差异"只是未做检验
  3. 93% 像素被判为"边界"(膨胀半径过大) -> 边界分层不具区分力

本脚本提供:
  A. 逐图指标 (Kappa/OA/mIoU/per-class IoU) —— 以"图"为重采样单位
  B. 配对 bootstrap 95% CI (图像级重采样, 1000 次)
  C. 等效性检验 TOST (预指定 SESOI = ΔKappa 0.02), 区分
     "无证据有差异" 与 "有证据无差异"
  D. 逐类 IoU 表 (含/不含 barren), 检验聚合指标是否结构性地看不见模块差异
  E. 紧边界分层 (距离类边界 0 / 1-2 / 3-4 / 5-8 / >8 px), 取代过松的膨胀带

用法
----
    python eval_rigor.py                # 仅用已缓存预测 (不占 GPU)
    python eval_rigor.py --infer        # 对缺失 tag 跑 GPU 推理并缓存
    python eval_rigor.py --infer --only nd_abl_no_fpn lgR_D1_join_nofpn_notrans
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
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, _HERE)
import paths

CONS = paths.CONSENSUS
CKPT = paths.CKPT
N_CLS = 7
IGNORE = 255
CLASS_NAMES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
SESOI = 0.02          # smallest effect size of interest for Kappa (预指定)
N_BOOT = 1000
SEED = 12345


# ----------------------------------------------------------- 指标
def per_image_kappa(pred, lab):
    m = lab != IGNORE
    p, l = pred[m].astype(np.int64), lab[m].astype(np.int64)
    if p.size == 0:
        return np.nan
    cm = np.bincount(l * N_CLS + p, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def per_image_oa(pred, lab):
    m = lab != IGNORE
    return float((pred[m] == lab[m]).mean()) if m.any() else np.nan


def per_image_iou(pred, lab):
    """返回 (miou, per_class_iou[nan=类不存在])"""
    m = lab != IGNORE
    p, l = pred[m].astype(np.int64), lab[m].astype(np.int64)
    ious = []
    for c in range(N_CLS):
        inter = int(((p == c) & (l == c)).sum())
        union = int(((p == c) | (l == c)).sum())
        ious.append(inter / union if union > 0 else np.nan)
    ious = np.array(ious, dtype=np.float64)
    return float(np.nanmean(ious)), ious


def pooled_kappa(cms):
    cm = cms.sum(0)
    n = cm.sum()
    if n == 0:
        return np.nan
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def conf_mat(pred, lab):
    m = lab != IGNORE
    p, l = pred[m].astype(np.int64), lab[m].astype(np.int64)
    return np.bincount(l * N_CLS + p, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)


# ----------------------------------------------------------- 预测读取 / 推理
def load_cached(tag):
    p = os.path.join(CONS, f"pred_{tag}.npz")
    if os.path.exists(p):
        return np.load(p)["pred"]
    return None


def infer_tag(tag, labels_meta):
    """GPU 推理: 用该 tag 的 best.pt 在 test_clean 上滑窗/整图预测"""
    import torch
    from train_v3 import LoveDADataset, compute_stats
    from torch.utils.data import DataLoader
    import experiment_matrix as EM
    from experiment_matrix_v2 import FPNSeg, SwinUnetLite, mssact_light
    from models.msscactnet import MSSACTNet

    def light(**kw):
        return mssact_light(**kw)

    REG = {
        "full_all_v2": light, "post_all_v2": light,
        "nd_unet": EM.UNet, "nd_pspnet": EM.PSPNet, "nd_fcn": EM.FCN,
        "nd_deeplab": EM.DeepLabV3Plus, "nd_segformer": EM.SegFormerLite,
        "nd_fpn_seg": FPNSeg, "nd_swin_unet": SwinUnetLite,
        "nd_abl_no_emr": lambda: light(use_emr=False),
        "nd_abl_no_ecsam": lambda: light(use_ecsam=False),
        "nd_abl_no_fpn": lambda: light(use_fpn=False),
        "nd_abl_no_trans": lambda: light(use_transformer=False),
        "nd_abl_no_adapter": lambda: light(use_adapter=False),
        "nd_abl_bilinear": lambda: light(upsample_mode="bilinear"),
        "nd_abl_trans4l": lambda: light(transformer_layers=4),
        "nd_abl_trans6l": lambda: light(transformer_layers=6),
        "lgR_bs8_full_lr2e4": light,
        "lgR_D1_join_nofpn_notrans": lambda: light(use_fpn=False, use_transformer=False),
        "lgR_D2_decoder_ca": lambda: light(decoder_ca_stages=(0, 1)),
        "lgR_D3_ch_tiny": lambda: MSSACTNet(in_channels=3, num_classes=7,
                                            embed_dims=[16, 32, 64, 128],
                                            transformer_layers=2, transformer_heads=4),
        "lgR_D4_ch_large": lambda: MSSACTNet(in_channels=3, num_classes=7,
                                             embed_dims=[64, 128, 256, 512],
                                             transformer_layers=2, transformer_heads=4),
    }
    if tag not in REG:
        print(f"  [skip] {tag}: 未注册构造函数", flush=True)
        return None
    ck = os.path.join(CKPT, f"{tag}_best.pt")
    if not os.path.exists(ck):
        print(f"  [skip] {tag}: 无权重", flush=True)
        return None

    DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    mean, std = compute_stats(root=paths.DATA_NEWSPLIT2)
    ds = LoveDADataset("test_clean", mean, std, train=False, root=paths.DATA_NEWSPLIT2)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = REG[tag]().to(DEV)
    sd = torch.load(ck, map_location=DEV, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for img, _ in dl:
            with torch.amp.autocast("cuda"):
                out = model(img.to(DEV))
            preds.append(out.argmax(1).to(torch.uint8).cpu().numpy())
    pred = np.concatenate(preds, 0)
    np.savez_compressed(os.path.join(CONS, f"pred_{tag}.npz"), pred=pred)
    del model
    import torch as _t
    _t.cuda.empty_cache()
    return pred


# ----------------------------------------------------------- 边界分层
def distance_bands(lab2d):
    """单张 2D 标签图到最近类边界的距离 (px).

    注意: 必须逐图 (2D) 调用。若传入 (N,H,W) 三维数组, axis=0 会把"跨图"的
    像素差误判为类边界, 从而得出 ~92% 的假边界比例 (曾导致报告误报 "93% 像素
    在类边界上")。正确的边界定义还须排除 valid<->255(ignore) 的过渡。
    """
    from scipy import ndimage
    valid = lab2d != IGNORE
    edge = np.zeros_like(valid, dtype=bool)
    for ax in (0, 1):
        a = np.take(lab2d, range(0, lab2d.shape[ax] - 1), axis=ax)
        b = np.take(lab2d, range(1, lab2d.shape[ax]), axis=ax)
        diff = (a != b) & (a != IGNORE) & (b != IGNORE)   # 仅有效类别之间的边界
        s = [slice(None)] * 2
        s[ax] = slice(0, -1)
        edge[tuple(s)] |= diff
        s2 = [slice(None)] * 2
        s2[ax] = slice(1, None)
        edge[tuple(s2)] |= diff
    edge &= valid
    return ndimage.distance_transform_edt(~edge), valid


BANDS = [(0, 0.5, "0 (边界像素)"), (0.5, 2.5, "1-2"), (2.5, 4.5, "3-4"),
         (4.5, 8.5, "5-8"), (8.5, 1e9, ">8 (内部)")]


def accumulate_bands(pred, lab, dist, valid, acc):
    """把单图的逐带 (正确数, 总数) 累加进 acc"""
    for lo, hi, name in BANDS:
        m = valid & (dist >= lo) & (dist < hi)
        n = int(m.sum())
        if n == 0:
            continue
        c = int((pred[m] == lab[m]).sum())
        a = acc.setdefault(name, [0, 0])
        a[0] += c
        a[1] += n


def finalize_bands(acc, n_valid):
    out = {}
    for _, _, name in BANDS:
        if name in acc and acc[name][1] > 0:
            c, n = acc[name]
            out[name] = dict(frac=round(n / n_valid, 5), oa=round(c / n, 4), n=n)
        else:
            out[name] = None
    return out


# ----------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true", help="对缺失 tag 跑 GPU 推理")
    ap.add_argument("--only", nargs="*", default=None, help="只处理这些 tag")
    args = ap.parse_args()

    lp = os.path.join(CONS, "labels.npz")
    if not os.path.exists(lp):
        print("缺少 labels.npz, 请先运行 consensus_analysis.py"); raise SystemExit(1)
    z = np.load(lp, allow_pickle=True)
    labels, names = z["labels"], z["names"]
    print(f"测试集: {labels.shape[0]} 张 {labels.shape[1]}² (test_clean)", flush=True)

    CANON = ["full_all_v2", "post_all_v2", "nd_unet", "nd_pspnet", "nd_fcn", "nd_deeplab",
             "nd_segformer", "nd_fpn_seg", "nd_swin_unet",
             "nd_abl_no_emr", "nd_abl_no_ecsam", "nd_abl_no_fpn", "nd_abl_no_trans",
             "nd_abl_no_adapter", "nd_abl_bilinear", "nd_abl_trans4l", "nd_abl_trans6l",
             "lgR_bs8_full_lr2e4", "lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca",
             "lgR_D3_ch_tiny", "lgR_D4_ch_large"]
    tags = args.only or CANON

    # 逐图计算边界距离场 (必须逐图, 见 distance_bands 注释)
    print("逐图计算类边界距离场...", flush=True)
    dists, valids = [], []
    for i in range(labels.shape[0]):
        d, v = distance_bands(labels[i])
        dists.append(d)
        valids.append(v)
    n_valid_total = int(sum(int(v.sum()) for v in valids))
    # 全局边界比例 (仅由标签决定, 与模型无关)
    bfrac = 0.0
    for d, v in zip(dists, valids):
        bfrac += float(((d < 0.5) & v).sum())
    bfrac /= n_valid_total
    print(f"  类边界(距离0)像素占比 = {bfrac:.4f}  "
          f"(注意: 不是 92%, 后者源于把跨图像素差误判为边界)", flush=True)

    res, cms, per_img = {}, {}, {}
    for tag in tags:
        pred = load_cached(tag)
        src = "cache"
        if pred is None:
            if not args.infer:
                print(f"  - {tag}: 无缓存 (用 --infer 生成)", flush=True); continue
            t0 = time.time()
            pred = infer_tag(tag, names)
            src = f"infer {time.time()-t0:.0f}s"
            if pred is None:
                continue
        if pred.shape != labels.shape:
            print(f"  ! {tag}: 形状 {pred.shape} != {labels.shape}, 跳过", flush=True); continue

        ks, oas = [], []
        cms[tag] = []
        ious = []
        bandacc = {}
        for i in range(labels.shape[0]):
            ks.append(per_image_kappa(pred[i], labels[i]))
            oas.append(per_image_oa(pred[i], labels[i]))
            mi, pc = per_image_iou(pred[i], labels[i])
            ious.append(pc)
            cms[tag].append(conf_mat(pred[i], labels[i]))
            accumulate_bands(pred[i], labels[i], dists[i], valids[i], bandacc)
        ks = np.array(ks); oas = np.array(oas)
        ious = np.array(ious)                      # (n_img, 7)
        mean_iou = np.nanmean(ious, 0)             # 逐类 mIoU
        cm = np.array(cms[tag])
        res[tag] = dict(
            source=src,
            kappa_pooled=round(float(pooled_kappa(cm)), 6),
            kappa_img_mean=round(float(np.nanmean(ks)), 6),
            oa_img_mean=round(float(np.nanmean(oas)), 6),
            miou_pooled=round(float(np.nanmean(mean_iou)), 6),
            per_class_iou=[None if np.isnan(v) else round(float(v), 4) for v in mean_iou],
            miou_no_barren=round(float(np.nanmean(np.delete(mean_iou, 4))), 6),
        )
        per_img[tag] = dict(kappa=ks.tolist(), oa=oas.tolist(),
                            iou=ious.tolist())
        res[tag]["boundary"] = finalize_bands(bandacc, n_valid_total)
        print(f"  ✓ {tag:30s} K={res[tag]['kappa_pooled']:.4f} "
              f"mIoU={res[tag]['miou_pooled']:.4f} [{src}]", flush=True)

    if not res:
        print("无可用预测。"); return

    # ---- 逐类 IoU 表 ----
    print(f"\n{'tag':30s} " + " ".join(f"{c[:4]:>7s}" for c in CLASS_NAMES) + f" {'mIoU':>7s} {'noBarren':>8s}")
    for t, r in sorted(res.items(), key=lambda kv: -kv[1]["miou_pooled"]):
        cells = " ".join(f"{(v if v is not None else float('nan')):7.4f}" for v in r["per_class_iou"])
        print(f"{t:30s} {cells} {r['miou_pooled']:7.4f} {r['miou_no_barren']:8.4f}")

    # ---- 配对 bootstrap + TOST (对照 = full_all_v2 / lgR_bs8_full) ----
    rng = np.random.default_rng(SEED)
    n = labels.shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def boot_kappa(tag):
        cm = np.array(cms[tag])
        return np.array([pooled_kappa(cm[i]) for i in idx])

    for ref in [t for t in ("full_all_v2", "lgR_bs8_full_lr2e4") if t in res]:
        base = boot_kappa(ref)
        print(f"\n=== 配对 bootstrap 对照 {ref} (K={res[ref]['kappa_pooled']:.4f}, "
              f"{N_BOOT} 次图像级重采样) ===")
        print(f"{'tag':30s} {'ΔKappa':>9s} {'95% CI':>20s} {'判定':>28s}")
        for t in res:
            if t == ref:
                continue
            d = boot_kappa(t) - base
            lo, hi = np.percentile(d, [2.5, 97.5])
            lo90, hi90 = np.percentile(d, [5, 95])
            if lo90 > -SESOI and hi90 < SESOI:
                verdict = f"等效 (TOST 90%CI ⊂ ±{SESOI})"
            elif lo > 0 or hi < 0:
                verdict = "显著差异"
            else:
                verdict = "未定 (CI 跨 0 且超界)"
            print(f"{t:30s} {res[t]['kappa_pooled']-res[ref]['kappa_pooled']:+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {verdict:>28s}")

    # ---- 边界分层 ----
    if "full_all_v2" in res and "nd_deeplab" in res:
        print(f"\n=== 紧边界分层 (一致性, 非 93% 膨胀带) ===")
        keys = list(res["full_all_v2"]["boundary"].keys())
        print(f"{'band(px)':>12s} " + " ".join(f"{t[:14]:>16s}" for t in
              ("full_all_v2", "nd_deeplab")))
        for k in keys:
            row = []
            for t in ("full_all_v2", "nd_deeplab"):
                b = res[t]["boundary"].get(k)
                row.append(f"{b['oa']:.4f}({b['frac']*100:.1f}%)" if b else "  -  ")
            print(f"{k:>12s} " + " ".join(f"{c:>16s}" for c in row))

    out = dict(sesoi=SESOI, n_boot=N_BOOT, n_images=int(n),
               boundary_frac_correct=bfrac,
               boundary_def="逐图 2D; 仅有效类别之间的边界 (排除 valid<->255 过渡)",
               class_names=CLASS_NAMES, results=res)
    dst = os.path.join(CONS, "rigor.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(per_img, open(os.path.join(CONS, "per_image_metrics.json"), "w"), indent=1)
    print(f"\n写入 {dst} 与 per_image_metrics.json", flush=True)


if __name__ == "__main__":
    main()
