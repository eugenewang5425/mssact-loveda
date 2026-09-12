# -*- coding: utf-8 -*-
"""事实汇总: 从 checkpoints/*_history.json + 评估json 提取权威数据, 生成 FACTS.json
用于报告与 README 的唯一数据源, 避免手写数字造成矛盾
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
import os, json, glob
import numpy as np

BASE = _BASE
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
    hr = json.load(open(f"{CONS}/headroom.json")) if os.path.exists(f"{CONS}/headroom.json") else {}
    F["consensus"] = dict(
        n_models=len(c["models"]), n_tiles=c["n_tiles"],
        mean_model_model=round(float(np.mean(list(c["model_model_agreement"].values()))),4),
        mean_model_label=round(float(np.mean(list(c["model_label_agreement"].values()))),4),
        consensus_vs_label=round(c["consensus_vs_label"],4),
        per_model={k: {kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items()}
                   for k,v in c["per_model_on_consensus"].items()},
        # 边界统计: 见 fix_boundary_analysis.py 的 _correction 说明
        # (原版把 (N,H,W) 沿 axis=0 求差, 跨图像素差被误判为边界, 误报 0.9296)
        boundary_frac=round(b.get("boundary_frac",0),4) if b else None,
        boundary_frac_within8px=b.get("boundary_frac_within8px") if b else None,
        band_frac=b.get("band_frac") if b else None,
        consensus_vs_label_by_band=b.get("consensus_vs_label_by_band") if b else None,
        per_model_vs_label_by_band=b.get("per_model_vs_label_by_band") if b else None,
        pair_disagreement_by_band=b.get("pair_disagreement_by_band") if b else None,
        # Kappa 归属上限: 完美解决某一部分像素所能获得的 Kappa 增量
        headroom=hr.get("oracle_kappa") if hr else None,
        headroom_band_frac=hr.get("band_frac") if hr else None)

# ========== 收敛速度分析 ==========
F["convergence"] = None
# 收敛速度分析: 原始产物已归入 artifacts/（根目录只留 FACTS/index/seed_variance 三项权威事实）
if os.path.exists(f"{BASE}/artifacts/convergence_learning_amount.json"):
    F["convergence"] = json.load(open(f"{BASE}/artifacts/convergence_learning_amount.json"))

# ========== 过拟合诊断 ==========
F["overfit"] = None
if os.path.exists(f"{BASE}/overfit_diag/overfit_summary.json"):
    F["overfit"] = json.load(open(f"{BASE}/overfit_diag/overfit_summary.json"))

# ========== 随机种子噪声底线 ==========
F["seed_variance"] = None
if os.path.exists(f"{BASE}/seed_variance.json"):
    F["seed_variance"] = json.load(open(f"{BASE}/seed_variance.json"))

# ========== 解码器伪影 / 边缘密度 ==========
F["artifact"] = None
if os.path.exists(f"{CONS}/artifact.json"):
    F["artifact"] = json.load(open(f"{CONS}/artifact.json"))

# ========== 严格性评估 (逐类 IoU / 配对 bootstrap / TOST) ==========
F["rigor"] = None
if os.path.exists(f"{CONS}/rigor.json"):
    F["rigor"] = json.load(open(f"{CONS}/rigor.json"))

# ========== 架构缺陷 (数值验证结论, 静态登记) ==========
# 依据 docs/架构有效性分析.md; 均由可复现数值实验证实
F["arch_defects"] = {
    "D1_fpn_dead_branches": {
        "claim": "FPN 自顶向下融合只写入 laterals[0..2], 而 forward 只用 fpn_features[-1]",
        "verification": "归零 lateral_convs[0..2] / 用'仅第4支路'等价算子替换整个 FPN",
        "output_diff": 0.0,
        "dead_params": 1828352, "model_params": 6101784,
        "dead_frac": 0.2996, "effective_frac": 0.7004},
    "D2_no_skip_connections": {
        "claim": "解码器只接收 32x32 特征, 无 encoder->decoder 跳连, 上采样 8 倍",
        "resolutions": [256, 128, 64, 32], "decoder_input": 32,
        "note": "256/128/64 特征从未到达输出; 输出细节为合成而非测量"},
    "D3_no_positional_encoding": {
        "claim": "TransformerEncoder 摊平为 token 后未加位置编码",
        "verification": "随机置换 token 后按逆置换还原输出",
        "output_diff": 7.153e-07,
        "note": "置换等变 -> 无空间位置感知; 但卷积特征提供隐式位置锚点, 属机制降级而非失效"},
    "D4_coord_attention_share": {"ecsam_params": 55044, "frac": 0.009},
}

# ========== 架构实验 (lgR_* 回退管线) 汇总 ==========
# 来源: experiments_index.json (参数量直读张量, 由 build_index.py 生成)
F["arch_experiments"] = None
_idx = f"{BASE}/experiments_index.json"
if os.path.exists(_idx):
    _d = json.load(open(_idx, encoding="utf-8"))
    _e = _d.get("experiments", {})
    _want = ["lgR_bs8_full_lr2e4", "lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca",
             "lgR_D3_ch_tiny", "lgR_D4_ch_large",
             "lg_D1_join_nofpn_notrans", "lg_D2_decoder_ca", "lg_D3_ch_tiny",
             "lg_D4_ch_large"]
    _rows = {}
    for _t in _want:
        _r = _e.get(_t)
        if _r:
            _rows[_t] = dict(series=_r.get("series"), pipeline=_r.get("pipeline"),
                             status=_r.get("status"), kappa=_r.get("best_kappa"),
                             best_epoch=_r.get("best_epoch"),
                             epochs_run=_r.get("epochs_run"),
                             params_M=_r.get("params_M"),
                             epochs_total=_r.get("target_epochs"))
    F["arch_experiments"] = dict(
        rows=_rows,
        comparable_groups=_d.get("comparable_groups"),
        # 同管线内的关键对照 (G2 = P-MEM-ROLL, 回退管线)
        g2_full=_rows.get("lgR_bs8_full_lr2e4", {}).get("kappa"),
        g2_d1=_rows.get("lgR_D1_join_nofpn_notrans", {}).get("kappa"),
        # G1 = P-PNG (与正式对比/消融同管线)
        g1_full=0.6312, g1_d1=_rows.get("lg_D1_join_nofpn_notrans", {}).get("kappa"))

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
