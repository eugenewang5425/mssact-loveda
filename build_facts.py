# -*- coding: utf-8 -*-
"""事实汇总: 从 checkpoints/*_history.json + 评估json 提取权威数据, 生成 FACTS.json
用于报告与 README 的唯一数据源, 避免手写数字造成矛盾
"""
import os, json, glob
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = f"{BASE}/checkpoints"
TT = f"{BASE}/tta_eval"
TT2 = f"{BASE}/tta_eval2"
CONS = f"{BASE}/consensus_analysis"

def best_of(tag):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return None
    try:
        h = json.load(open(p))
        if not isinstance(h, list) or not h: return None
        b = max(h, key=lambda r: r.get("kappa", -9))
        return dict(tag=tag, n_epochs=len(h), best_epoch=b["epoch"],
                    kappa=round(b["kappa"],4), oa=round(b.get("oa",0),4),
                    mf1=round(b.get("mf1",0),4), finalized=b["epoch"]==len(h))
    except Exception:
        return None

def eval_of(path):
    if not os.path.exists(path): return None
    try:
        d = json.load(open(path))
        return {k: (round(v,4) if isinstance(v,float) else v) for k,v in d.items()
                if k in ("kappa","oa","mf1","miou","n_tiles")}
    except Exception:
        return None

F = {}

# ========== 阶段A: 早期实验 (957张, 含 REMAP bug) [历史] ==========
F["stage_A_957_legacy"] = {
    "note": "早期实验: 957张(镜像子集), REMAP含no-data bug(0->背景), 结果偏高约0.007-0.009",
    "baselines_60ep": {t: best_of(t) for t in
        ["mssact_full60","deeplab","fcn","pspnet","segformer","unet_matrix","fpn_seg","swin_unet"]},
    "ablation_60ep": {t: best_of(t) for t in
        ["ablate_no_emr","ablate_no_ecsam","ablate_no_fpn","ablate_no_trans","ablate_no_adapter",
         "ablate_trans4l","ablate_trans6l","ablate_bilinear"]},
    "ablation_120ep": {t: best_of(t) for t in
        ["abl120_no_emr","abl120_no_ecsam","abl120_no_fpn","abl120_no_trans","abl120_no_adapter"]},
    "scale_study": {t: best_of(t) for t in ["v3_s128gf","v3_s256","v3_s512","v3_s1024"]},
    "strategies": {t: best_of(t) for t in
        ["strat_mssact_sgdr120","strat_mssact_full30m_oc60","strat_mssact_oc60_lr5e4",
         "strat_bilinear_oc60_lr1e4","strat_no_emr_oc60_lr1e4","strat_deeplab_oc30"]},
}

# ========== 阶段B: 扩大数据 (2257张, 含 bug) ==========
F["stage_B_2257"] = {
    "note": "官方全量训练区(去重后2821池, 8:1:1), 但 REMAP 仍含 no-data bug",
    "full_all": best_of("full_all"),
    "post_all": best_of("post_all"),
    "ttA": {k: eval_of(f"{TT}/{k}.json") for k in
            ["full_all_val","full_all_val_TTA","full_all_test","full_all_test_TTA"]},
}

# ========== 阶段C: 最终实验 (1768张筛选 + REMAP修复) ==========
F["stage_C_1768_final"] = {
    "note": "最终: no-data<=10%筛选(剔除312张) + MD5去重 + 8:1:1 + REMAP修复(0->ignore)",
    "split": dict(train=1768, val=221, test=221, test_clean=141),
    "main": {t: best_of(t) for t in ["full_all_v2","post_all_v2"]},
    "baselines": {t: best_of(t) for t in
        ["nd_unet","nd_pspnet","nd_fcn","nd_deeplab","nd_segformer","nd_fpn_seg","nd_swin_unet"]},
    "ablation": {t: best_of(t) for t in
        ["nd_abl_no_emr","nd_abl_no_ecsam","nd_abl_no_fpn","nd_abl_no_trans",
         "nd_abl_no_adapter","nd_abl_trans4l","nd_abl_trans6l","nd_abl_bilinear"]},
}

# ========== 一致性分析 (标签质量) ==========
F["consensus"] = None
if os.path.exists(f"{CONS}/consistency.json"):
    c = json.load(open(f"{CONS}/consistency.json"))
    b = json.load(open(f"{CONS}/boundary_analysis.json")) if os.path.exists(f"{CONS}/boundary_analysis.json") else {}
    F["consensus"] = dict(
        n_models=len(c["models"]), n_tiles=c["n_tiles"],
        mean_model_model=round(float(np.mean(list(c["model_model_agreement"].values()))),4),
        mean_model_label=round(float(np.mean(list(c["model_label_agreement"].values()))),4),
        consensus_vs_label=round(c["consensus_vs_label"],4),
        per_model={k: {kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items()}
                   for k,v in c["per_model_on_consensus"].items()},
        boundary_frac=round(b.get("boundary_frac",0),4) if b else None,
        consensus_vs_label_by_band=b.get("consensus_vs_label") if b else None)

# ========== 收敛速度分析 ==========
F["convergence"] = None
if os.path.exists(f"{BASE}/convergence_learning_amount.json"):
    F["convergence"] = json.load(open(f"{BASE}/convergence_learning_amount.json"))

# ========== 过拟合诊断 ==========
F["overfit"] = None
if os.path.exists(f"{BASE}/overfit_diag/overfit_summary.json"):
    F["overfit"] = json.load(open(f"{BASE}/overfit_diag/overfit_summary.json"))

json.dump(F, open(f"{BASE}/FACTS.json","w"), indent=1, ensure_ascii=False)

# 打印摘要
print("="*90)
print("阶段C (最终) — 主模型与基线")
print("="*90)
for k in ["full_all_v2","post_all_v2"]:
    r = F["stage_C_1768_final"]["main"].get(k)
    if r: print(f"  {k:18s} K={r['kappa']:.4f} OA={r['oa']:.4f} F1={r['mf1']:.4f} ({r['n_epochs']}轮)")
for k,v in F["stage_C_1768_final"]["baselines"].items():
    if v: print(f"  {k:18s} K={v['kappa']:.4f} OA={v['oa']:.4f} ({v['n_epochs']}轮)")
print("\n阶段C — 消融")
for k,v in F["stage_C_1768_final"]["ablation"].items():
    if v: print(f"  {k:20s} K={v['kappa']:.4f} ({v['n_epochs']}轮)")
if F["consensus"]:
    c = F["consensus"]
    print(f"\n一致性: 模型-模型 {c['mean_model_model']:.4f} | 模型-标签 {c['mean_model_label']:.4f} | 共识-标签 {c['consensus_vs_label']:.4f}")
print("\n-> FACTS.json 已生成")
