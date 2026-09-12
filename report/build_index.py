"""
构建 experiments_index.json —— 实验索引（单一事实来源）

原则
----
1. 参数量直接读 checkpoint 张量（不靠文件名猜测）
2. Kappa 直接读 *_history.json
3. 每个 tag 的配置/数据根/协议来自"启动它的队列脚本"（权威出处）
4. 不同管线世代的产物严格分区，禁止跨管线比较

管线世代
--------
  A-legacy : 957 张子集, REMAP 含 no-data bug (0->背景)      -> 已被 C 取代
  B-2257   : 官方全量去重 2821 池, REMAP 仍含 no-data bug    -> 已被 C 取代
  C-final  : no-data<=10% 筛选 + MD5 去重 + REMAP 修复       -> 当前对比/消融正式结果
  lg-old   : 架构实验, 旧增强管线 (全局 RNG)
  lgF-det  : 架构实验, 确定性增强 -> 已否决 (Kappa -0.0246)
  lgR-roll : 架构实验, 回退管线 (memmap + 全局 RNG + workers=0) -> 架构实验正式结果

用法: python build_index.py
"""

# --- 路径引导（本文件位于子目录 report/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os, sys, json, glob, time

import paths

CKPT = paths.CKPT
REPO = paths.REPO

# ---------------------------------------------------------------- 配置注册表
# 出处: 启动各 tag 的队列脚本（已核对源码）
#
# 训练管线世代（决定"能否比较"的关键，已用 checkpoint/日志时间戳核实）
#   P-PNG    : 每样本解码两张 1024² PNG, 裁剪 256², num_workers=0, 全局RNG增强
#   P-MEM-DET: memmap 预解码, 确定性增强     -> 已否决 (D1 Kappa 0.6277 -> 0.6031)
#   P-MEM-ROLL: memmap 预解码, 全局RNG增强, num_workers=0 -> 回退后正式管线
#
# 时间戳证据:
#   fast_dataset/*.npy 建立于 2026-09-12 14:51-14:53
#   nd_* / full_all_v2 写完于 09-10 15:45 ~ 09-12 06:14  -> 早于 memmap, 必为 P-PNG
#   lg_D1/D2/D3 写完于 09-12 08:37/11:16/13:39           -> 早于 memmap, 必为 P-PNG
#   lgF_* 为确定性增强世代 (已否决); lgR_* 为回退后管线
REGISTRY = {}


def reg(tags, series, script, root_key, epochs, patience, batch, lr, note="",
        pipeline="P-PNG", **kw):
    for t in (tags if isinstance(tags, (list, tuple)) else [tags]):
        REGISTRY[t] = dict(series=series, script=script, root=root_key,
                           target_epochs=epochs, patience=patience, batch=batch,
                           lr=lr, note=note, pipeline=pipeline, **kw)


# ---- C-final: 对比实验 (7) + 消融实验 (8) —— 正式结果 ----
_C_NOTE = "REMAP修复后正式结果"
for t in ["nd_unet", "nd_pspnet", "nd_fcn", "nd_deeplab", "nd_segformer",
          "nd_fpn_seg", "nd_swin_unet"]:
    reg(t, "C-final/对比实验", "queues/run_queue_new.py", "DATA_NEWSPLIT2", 60, 20, 8, 2e-4, _C_NOTE)
for t in ["nd_abl_no_emr", "nd_abl_no_ecsam", "nd_abl_no_fpn", "nd_abl_no_trans",
          "nd_abl_no_adapter", "nd_abl_trans4l", "nd_abl_trans6l", "nd_abl_bilinear"]:
    reg(t, "C-final/消融实验", "queues/run_queue_new.py", "DATA_NEWSPLIT2", 60, 20, 8, 2e-4, _C_NOTE)
reg("full_all_v2", "C-final/主模型", "queues/run_queue_new.py", "DATA_NEWSPLIT2", 60, 20, 8, 2e-4,
    "完整模型基线 = 消融参考系")
reg("post_all_v2", "C-final/主模型", "queues/run_queue_new.py", "DATA_NEWSPLIT2", 30, 12, 8, 5e-5,
    "微调自 full_all_v2")

# ---- B-2257: 官方全量 (REMAP bug 未修) ----
reg("full_all", "B-2257", "legacy/run_queue.py", "DATA_NEWSPLIT", 60, 20, 8, 2e-4,
    "REMAP 仍含 no-data bug")
reg("post_all", "B-2257", "legacy/run_queue.py", "DATA_NEWSPLIT", 30, 12, 8, 5e-5,
    "REMAP 仍含 no-data bug")

# ---- A-legacy: 957 张 ----
reg("mssact_full60", "A-957", "legacy/run_queue.py", "DATA_LEGACY", 60, 15, 8, 2e-4,
    "REMAP bug + 957 张子集")
reg(["deeplab", "fcn", "pspnet", "segformer", "fpn_seg", "swin_unet", "unet_lr1e4",
     "ablate_no_emr", "ablate_no_ecsam", "ablate_no_fpn", "ablate_no_trans",
     "ablate_no_adapter"],
    "A-957/对比+消融(60轮)", "experiment_matrix.py", "DATA_LEGACY", 60, 15, 8, 2e-4,
    "REMAP bug + 957 张子集")
reg(["ablate_trans4l", "ablate_trans6l", "ablate_bilinear"],
    "A-957/对比+消融(60轮)", "experiment_matrix_v2.py", "DATA_LEGACY", 60, 15, 8, 2e-4,
    "REMAP bug + 957 张子集")
reg(["abl120_no_emr", "abl120_no_ecsam", "abl120_no_fpn", "abl120_no_trans",
     "abl120_no_adapter"],
    "A-957/消融(120轮)", "legacy/run_abl120.py", "DATA_LEGACY", 120, 20, 8, 2e-4,
    "REMAP bug; 用户要求训练到收敛")
reg(["strat_mssact_full30m_oc60", "strat_mssact_sgdr120", "strat_mssact_oc60_lr5e4",
     "strat_bilinear_oc60_lr1e4", "strat_no_emr_oc60_lr1e4", "strat_deeplab_oc30",
     "strat_bilinear_oc30", "strat_unet_oc12_lr1e3", "strat_mssact_oc30",
     "strat_unet_oc12_lr2e4", "strat_no_emr_oc30"],
    "A-957/训练策略", "queues/run_queue_30m.py", "DATA_LEGACY", 60, 20, 8, None,
    "LR/调度/预算 策略对比")
reg(["v3_s256", "v3_s512", "v3_s1024", "v3_s128gf"],
    "A-957/尺度策略", "legacy/train_v3.py", "DATA_LEGACY", 120, 20, 8, 2e-4,
    "裁剪尺度对比")

# ---- 架构实验: 三代管线 ----
# v1 (P-PNG): 与 nd_* 同管线同协议 -> 可与 full_all_v2 直接比较
_ARCH_OLD = ["lg_D1_join_nofpn_notrans", "lg_D2_decoder_ca", "lg_D3_ch_tiny",
             "lg_D4_ch_large"]
reg(_ARCH_OLD, "lg-old/架构(旧管线)", "queues/run_queue_arch.py (v1)", "DATA_NEWSPLIT2",
    60, 20, 8, 2e-4, "P-PNG: 与 nd_* 同管线同协议, 可比较")
reg(["lg_bs8_full_lr2e4"], "lg-old/架构基线", "queues/run_queue_arch.py (v1)", "DATA_NEWSPLIT2",
    60, 20, 8, 2e-4, "P-PNG")
reg(["lg_bs16_full_lr4e4"], "lg-old/架构基线", "queues/run_queue_batch.py", "DATA_NEWSPLIT2",
    60, 20, 16, 4e-4, "P-PNG; bs16 lr 线性缩放")
reg(["lg_b15_full", "lg_b15_noecsam", "lg_b15_noemr", "lg_b15_unet", "lg_b15_deeplab"],
    "lg-old/预算攻击(15轮)", "queues/run_queue_interaction.py", "DATA_NEWSPLIT2", 15, 15, 8, 2e-4,
    "P-PNG")
reg(["lg_const_full", "lg_const_unet", "lg_const_deeplab"],
    "lg-old/恒定LR", "queues/run_queue_interaction.py", "DATA_NEWSPLIT2", 60, 60, 8, 1e-4, "P-PNG")

_lgF = ["lgF_D1_join_nofpn_notrans", "lgF_D2_decoder_ca", "lgF_D3_ch_tiny",
        "lgF_D4_ch_large", "lgF_bs8_full_lr2e4", "lgF_bs16_full_lr4e4"]
reg(_lgF, "lgF-det/架构(确定性增强)", "queues/run_queue_arch.py (v2)", "DATA_NEWSPLIT2",
    60, 20, 8, 2e-4, "确定性增强 -> Kappa 相对 P-PNG -0.0246, 已否决",
    pipeline="P-MEM-DET", rejected=True)

_lgR = ["lgR_bs8_full_lr2e4", "lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca",
        "lgR_D3_ch_tiny", "lgR_D4_ch_large"]
reg(_lgR[:1], "lgR-roll/架构基线", "queues/run_queue_arch.py (v3)", "DATA_NEWSPLIT2", 60, 20, 8, 2e-4,
    "★ 完整模型 6.102M 同管线对照", pipeline="P-MEM-ROLL")
reg(_lgR[1:], "lgR-roll/架构消融与容量", "queues/run_queue_arch.py (v3)", "DATA_NEWSPLIT2",
    60, 20, 8, 2e-4, "P-MEM-ROLL", pipeline="P-MEM-ROLL")
# DeepLabV3+ 从零训练对照（分离 ImageNet 预训练贡献）
reg(["lgR_deeplab_scr"], "lgR-roll/DeepLab从零对照", "queues/run_queue_deeplab_scr.py",
    "DATA_NEWSPLIT2", 60, 20, 8, 2e-4,
    "★ pretrained_backbone=False；对照 lgR_bs8_full_lr2e4(0.6320, 从零)", pipeline="P-MEM-ROLL")
for _n in (250, 500, 1000):
    reg([f"lgR_n{_n}_deeplab_scr"], f"lgR-roll/数据阶梯 n{_n}", "queues/run_queue_deeplab_scr.py",
        f"data_ladder/n{_n}", 30, 30, 8, 2e-4,
        "从零训练；与同档预训练版 lgR_n%d_deeplab 配对可分离预训练贡献" % _n,
        pipeline="P-MEM-ROLL")
reg(["lgR_b15_full", "lgR_b15_noecsam", "lgR_b15_noemr", "lgR_b15_unet", "lgR_b15_deeplab"],
    "lgR-roll/预算攻击(15轮)", "queues/run_queue_arch.py (v3)", "DATA_NEWSPLIT2", 15, 15, 8, 2e-4,
    "P-MEM-ROLL", pipeline="P-MEM-ROLL")
for _n in (250, 500, 1000):
    reg([f"lgR_n{_n}_{m}" for m in ("full", "noecsam", "unet", "deeplab")],
        f"lgR-roll/数据阶梯 n{_n}", "queues/run_queue_arch.py (v3)", f"data_ladder/n{_n}",
        30, 30, 8, 2e-4,
        "验证集固定为 newsplit2/val; fast_data=False(PNG 子集)", pipeline="P-MEM-ROLL")
reg(["lgV_D1_rollback"], "lgR-roll/回退验证", "verify/verify_rollback.py", "DATA_NEWSPLIT2",
    60, 20, 8, 2e-4, "回退管线性能验证, 已并入 lgR_D1", pipeline="P-MEM-ROLL")

# ---- 废弃/失败运行 (保留记录, 明确不可引用) ----
reg(["mssact_v2", "unet_matrix"], "DEPRECATED", None, "DATA_LEGACY", None, None, 8, 2e-4,
    "空 history, 无有效训练记录, 不可引用")
reg("unet", "DEPRECATED", "experiment_matrix.py", "DATA_LEGACY", 120, 20, 8, 2e-4,
    "早期失败运行, 被 nd_unet 取代")

# ------------------------------------------------- 可比性分组（核心科学约束）
# 只有 (数据划分, 训练管线, 协议) 三者全同的 tag 才可比较 Kappa
COMPARABLE_GROUPS = {
    "G1-PNG-60ep": {
        "pipeline": "P-PNG", "root": "DATA_NEWSPLIT2",
        "protocol": "60ep/pt20/bs8/lr2e-4",
        "members": ["full_all_v2"] +
                   [f"nd_abl_{s}" for s in ("no_emr", "no_ecsam", "no_fpn", "no_trans",
                                            "no_adapter", "bilinear", "trans4l", "trans6l")] +
                   ["lg_D1_join_nofpn_notrans", "lg_D2_decoder_ca", "lg_D3_ch_tiny",
                    "lg_D4_ch_large"],
        "note": "对比实验基线组 (nd_unet/pspnet/fcn/deeplab/segformer/fpn_seg/swin_unet) "
                "同为 P-PNG/60ep/pt20/bs8/lr2e-4, 与本组可比较",
    },
    "G2-MEM-ROLL-60ep": {
        "pipeline": "P-MEM-ROLL", "root": "DATA_NEWSPLIT2",
        "protocol": "60ep/pt20/bs8/lr2e-4",
        "members": ["lgR_bs8_full_lr2e4", "lgR_D1_join_nofpn_notrans",
                    "lgR_D2_decoder_ca", "lgR_D3_ch_tiny", "lgR_D4_ch_large"],
        "note": "架构结论的复现组; 与 G1 的差异仅为数据投递后端 "
                "(memmap vs PNG), 该差异已实测为 +0.0016 (lgR_D1 0.6293 vs lg_D1 0.6277)",
    },
    "G3-b15": {
        "pipeline": "P-PNG", "root": "DATA_NEWSPLIT2", "protocol": "15ep/pt15/bs8/lr2e-4",
        "members": ["lg_b15_full", "lg_b15_noecsam", "lg_b15_noemr", "lg_b15_unet",
                    "lg_b15_deeplab"],
        "note": "预算攻击: 检验'充足预算是否掩盖模块差异'",
    },
    "REJECTED": {
        "pipeline": "P-MEM-DET", "members": _lgF,
        "note": "确定性增强世代, 相对 P-PNG 损失 0.0246 Kappa, 结论一律不引用",
    },
}

# ------------------------------------------------------- 参数量 (直读张量)
sys.path.insert(0, REPO)


def param_counts(tags):
    """直读 checkpoint 张量求参数量; 失败则回退到构造函数实例化"""
    import torch
    out = {}
    for t in sorted(tags):
        p = os.path.join(CKPT, f"{t}_best.pt")
        if not os.path.exists(p):
            continue
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            if hasattr(sd, "state_dict"):
                sd = sd.state_dict()
            if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]
            out[t] = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
            del sd
        except Exception as e:
            out[t] = None
            print(f"  [warn] {t}: {type(e).__name__} {e}", flush=True)
    return out


def arch_probe(tags):
    """从张量形状反推架构: stem 输出通道 / 各 stage 通道 / transformer 层数 / 关键模块有无"""
    import torch
    info = {}
    for t in sorted(tags):
        p = os.path.join(CKPT, f"{t}_best.pt")
        if not os.path.exists(p):
            continue
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        ks = list(sd.keys())
        d = {}
        # stem 输出通道
        for k in ks:
            if k.startswith("stem.0.weight"):
                d["stem_ch"] = int(sd[k].shape[0])
        # encoder 各阶段通道
        for i in range(4):
            for k in ks:
                if k.startswith(f"encoder.{i}.conv1.weight") or \
                   k.startswith(f"encoder.{i}.block.0.weight") or \
                   k.startswith(f"encoder.{i}.") and k.endswith("conv1.weight"):
                    d.setdefault("stage_ch", []).append(int(sd[k].shape[0]))
                    break
        # transformer 层数
        tl = set()
        for k in ks:
            if k.startswith("transformer") and ".layers." in k:
                try:
                    tl.add(int(k.split(".layers.")[1].split(".")[0]))
                except Exception:
                    pass
        d["transformer_layers"] = (max(tl) + 1) if tl else 0
        d["has_transformer"] = d["transformer_layers"] > 0
        d["has_fpn"] = any(k.startswith("fpn") for k in ks)
        d["has_ecsam"] = any("ecsam" in k for k in ks)
        d["has_adapter"] = any("adapter" in k for k in ks)
        d["has_emr"] = any(k.startswith("encoder") and "emr" in k.lower() for k in ks)
        # 上采样模式: 转置卷积核存在 = deconv
        d["deconv_kernels"] = sum(1 for k in ks if "deconv" in k.lower() or
                                  (k.endswith(".weight") and sd[k].dim() == 4
                                   and sd[k].shape[-1] == 2))
        info[t] = d
        del sd
    return info


def main():
    os.makedirs(CKPT, exist_ok=True)
    hist_files = sorted(glob.glob(os.path.join(CKPT, "*_history.json")))
    tags = [os.path.basename(f).replace("_history.json", "") for f in hist_files]
    pt_tags = [os.path.basename(f).replace("_best.pt", "")
               for f in sorted(glob.glob(os.path.join(CKPT, "*_best.pt")))]
    all_tags = sorted(set(tags) | set(pt_tags))

    print(f"扫描: {len(all_tags)} 个 tag ({len(tags)} 有 history, {len(pt_tags)} 有权重)", flush=True)
    print("读取参数量...", flush=True)
    pc = param_counts(all_tags)
    print("反推架构...", flush=True)
    ap = arch_probe(all_tags)

    exps, unregistered = {}, []
    for t in all_tags:
        hp = os.path.join(CKPT, f"{t}_history.json")
        h = None
        if os.path.exists(hp):
            try:
                h = json.load(open(hp, encoding="utf-8"))
            except Exception:
                h = None
        epochs_run = len(h) if isinstance(h, list) else 0
        best_k = best_ep = None
        if epochs_run:
            b = max(h, key=lambda r: r.get("kappa", -9))
            best_k, best_ep = b.get("kappa"), b.get("epoch")
        cfg = REGISTRY.get(t)
        if cfg is None:
            unregistered.append(t)
        target = (cfg or {}).get("target_epochs")
        if (cfg or {}).get("rejected"):
            status = "REJECTED"
        elif epochs_run == 0:
            status = "no-data"
        elif target and epochs_run >= target:
            status = "complete"
        elif target:
            status = f"partial({epochs_run}/{target})"
        else:
            status = "unregistered"
        rec = {
            "series": (cfg or {}).get("series", "UNREGISTERED"),
            "status": status,
            "pipeline": (cfg or {}).get("pipeline"),
            "best_kappa": round(best_k, 6) if isinstance(best_k, (int, float)) else None,
            "best_epoch": best_ep,
            "epochs_run": epochs_run,
            "target_epochs": target,
            "params_M": round(pc[t] / 1e6, 3) if pc.get(t) else None,
            "has_weight": t in pt_tags,
            "script": (cfg or {}).get("script"),
            "root": (cfg or {}).get("root"),
            "protocol": (f"{target}ep/pt{(cfg or {}).get('patience')}/"
                         f"bs{(cfg or {}).get('batch')}/lr{(cfg or {}).get('lr')}"
                         if cfg else None),
            "note": (cfg or {}).get("note", ""),
        }
        if t in ap:
            rec["arch"] = ap[t]
        exps[t] = rec

    # ---------------- 汇总统计 ----------------
    by_series = {}
    for t, r in exps.items():
        by_series.setdefault(r["series"], []).append(t)
    summary = {}
    for s, ts in sorted(by_series.items()):
        ks = [exps[t]["best_kappa"] for t in ts if exps[t]["best_kappa"] is not None]
        summary[s] = {"n_experiments": len(ts), "n_complete": sum(
            1 for t in ts if exps[t]["status"] == "complete"),
            "best_kappa_in_series": round(max(ks), 4) if ks else None}

    out = {
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_by": "build_index.py",
            "protocol": "AdamW(wd=0.02) + OneCycle(6% warmup, cosine) + EMA(0.999) + "
                        "类别加权CE + 验证Kappa早停",
            "criterion": "val Kappa (OA 被多数类支配, 不作选择依据)",
            "splits": {
                "A-957": "train957/val199/test100 (早期子集, REMAP 含 no-data bug)",
                "B-2257": "官方全量去重 2821 池 8:1:1 (REMAP 含 no-data bug)",
                "C-final": "train1768/val221/test221/test_clean141 "
                           "(no-data<=10% 筛选 + MD5 去重 + REMAP 修复)",
            },
            "pipeline_generations": {
                "C-final": "正式对比/消融结果 (train_v3 原管线增强)",
                "lg-old": "架构实验旧管线, 已被 lgR 取代",
                "lgF-det": "确定性增强管线, 已否决 (Kappa -0.0246)",
                "lgR-roll": "memmap 预解码 + 原增强(全局RNG) + workers=0, 架构实验正式结果",
            },
            "hard_rule": "禁止跨管线世代比较 Kappa; 参数量为直读张量求和, 非估算",
            "n_tags": len(exps),
            "unregistered_tags": unregistered,
        },
        "comparable_groups": COMPARABLE_GROUPS,
        "series_summary": summary,
        "experiments": exps,
    }
    dst = os.path.join(REPO, "experiments_index.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {dst}", flush=True)
    print(f"  已登记 {len(exps) - len(unregistered)} / {len(exps)} 个 tag", flush=True)
    if unregistered:
        print(f"  未登记(需补配置): {unregistered}", flush=True)
    for s, v in sorted(summary.items()):
        print(f"  {s:34s} n={v['n_experiments']:2d} complete={v['n_complete']:2d} "
              f"bestK={v['best_kappa_in_series']}", flush=True)


if __name__ == "__main__":
    main()
