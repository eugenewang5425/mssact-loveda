"""多方核验 & 实验索引生成器
核验维度:
  1. checkpoint 架构自动推断 vs 期望配置注册表
  2. history.json 完整性/可解析性 (损坏的标注 deprecated)
  3. 4角验证 Kappa vs 全量滑窗 Kappa vs held-out Kappa 一致性 (排名/差值)
输出: verify_report.txt + experiments_index.json (健壮版)
"""

# --- 路径引导（本文件位于子目录 verify/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os, sys, json, glob
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
import torch

BASE = paths.REPO
CKPT = f"{BASE}/checkpoints"
FT = f"{BASE}/fulltile_eval"
HO = f"{BASE}/heldout_test"
OUT = f"{BASE}/experiments_index.json"
REPORT = f"{BASE}/verify_report.txt"
NUM_CLASSES = 7

# ===== 期望配置注册表 (训练时的真实构造) =====
EXPECTED = {
    "v3_s256":        dict(kind="mssact_light", layers=2, embed=256, desc="主实验 OneCycle120"),
    "mssact_full60":  dict(kind="mssact_light", layers=2, embed=256, desc="公平对照 OneCycle60"),
    "ablate_trans4l": dict(kind="mssact_light", layers=4, embed=256),
    "ablate_trans6l": dict(kind="mssact_light", layers=6, embed=256),
    "ablate_no_emr":  dict(kind="mssact_light", layers=2, embed=256),
    "ablate_no_ecsam":dict(kind="mssact_light", layers=2, embed=256),
    "ablate_no_fpn":  dict(kind="mssact_light", layers=2, embed=256),
    "ablate_no_trans":dict(kind="mssact_light", layers=0, embed=0),
    "ablate_no_adapter":dict(kind="mssact_light", layers=2, embed=256),
    "mssact_v2":      dict(kind="mssact_light", layers=2, embed=256, deprecated=True),
}
NON_MSSACT = {"unet","pspnet","fcn","deeplab","segformer","unet_matrix","fpn_seg","swin_unet","unet_lr1e4"}

def infer_arch(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    has_trans = any(k.startswith("transformer.transformer.layers.") for k in sd)
    layers = len({int(k.split(".")[3]) for k in sd if k.startswith("transformer.transformer.layers.")})
    if "stem.0.weight" in sd and has_trans and layers > 0:
        inproj = sd["transformer.transformer.layers.0.self_attn.in_proj_weight"].shape[0]//3
        return dict(layers=layers, embed=inproj)
    if "stem.0.weight" in sd and not has_trans:
        return dict(layers=0, embed=0)
    return dict(layers=None, embed=None, custom=True)

def safe_json(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"__corrupt__": str(e)[:80]}

def best_of(obj):
    if isinstance(obj, dict) and "__corrupt__" in obj: return None
    if not isinstance(obj, list) or not obj: return None
    return max(obj, key=lambda r: r.get("kappa", -9))

def kappa_rank(res):
    return sorted(res, key=lambda t: -t[1])

def main():
    report = []
    idx = {"_meta": dict(generated_at="2026-09-07", protocol="AdamW+OneCycle+EMA+earlystop(patience=15/20)",
                          val_split="train957/val199/test100(held-out)", note="mssact构造核验: 统一关键字参数")}
    groups = {"scale": [], "baseline": [], "ablation60": [], "ablation120": [],
              "strat": [], "fulltile": [], "heldout": [], "deprecated_corrupt": []}

    # ---- 1. checkpoint 架构核验 ----
    report.append("="*60 + "\n[1] CHECKPOINT ARCHITECTURE CHECK\n" + "="*60)
    for tag, exp in EXPECTED.items():
        p = f"{CKPT}/{tag}_best.pt"
        if not os.path.exists(p):
            report.append(f"  {tag:22s} NO-CKPT"); continue
        a = infer_arch(p)
        ok = (a["layers"] == exp.get("layers")) and (a.get("embed") == exp.get("embed"))
        status = "OK " if ok else "MISMATCH"
        report.append(f"  {status} {tag:22s} inferred L={a['layers']} d={a.get('embed')} expected L={exp.get('layers')} d={exp.get('embed')}")
        if not ok:
            report.append(f"         !!! 该 checkpoint 与注册配置不符, 需重新训练或改注册表")
    for tag in NON_MSSACT:
        p = f"{CKPT}/{tag}_best.pt"
        if not os.path.exists(p):
            report.append(f"  {tag:22s} NO-CKPT"); continue
        report.append(f"  OK  {tag:22s} (non-MSSACT, 构造无参, 无此风险)")

    # ---- 2. history 完整性 ----
    report.append("\n" + "="*60 + "\n[2] HISTORY FILE INTEGRITY\n" + "="*60)
    for tag in list(EXPECTED) + list(NON_MSSACT):
        p = f"{CKPT}/{tag}_history.json"
        if not os.path.exists(p):
            if tag in ("unet_matrix",): continue  # cp 备份无 history
            report.append(f"  {tag:22s} NO-HISTORY"); continue
        obj = safe_json(p)
        if isinstance(obj, dict) and obj.get("__corrupt__"):
            report.append(f"  CORRUPT {tag:22s} {obj['__corrupt__']}  -> 标记 deprecated, 不参与结论")
            groups["deprecated_corrupt"].append(tag)
            continue
        b = best_of(obj)
        report.append(f"  OK  {tag:22s} n={len(obj)} best_k={b['kappa']:.4f}@ep{b['epoch']}")
        if tag.startswith("v3_"): groups["scale"].append(tag)
        elif tag.startswith("ablate_"): groups["ablation60"].append(tag)
        elif tag in ("mssact_full60",): groups["baseline"].append(tag)
        elif tag in NON_MSSACT: groups["baseline"].append(tag)
        elif tag.startswith("strat_"): groups["strat"].append(tag)

    # ---- 3. 三方指标一致性 (4角 / fulltile / heldout) ----
    report.append("\n" + "="*60 + "\n[3] CROSS-VALIDATION CONSISTENCY (4corner vs fulltile vs heldout)\n" + "="*60)
    vals = {}
    for tag in EXPECTED: vals[tag] = []; 
    for tag in list(NON_MSSACT): vals[tag] = []
    # 4角
    for tag in vals:
        obj = safe_json(f"{CKPT}/{tag}_history.json")
        b = best_of(obj)
        if b: vals[tag].append(("4corner", b["kappa"]))
    # fulltile
    for p in glob.glob(f"{FT}/*_conf.json"):
        tag = os.path.basename(p).replace("_conf.json","")
        r = safe_json(p)
        if isinstance(r, dict) and r.get("kappa") is not None:
            vals.setdefault(tag, []).append(("fulltile", r["kappa"]))
    # heldout
    merged = safe_json(f"{HO}/heldout_results.json")
    if isinstance(merged, dict) and "__corrupt__" not in merged:
        for tag, r in merged.items():
            vals.setdefault(tag, []).append(("heldout", r["kappa"]))
    for tag, v in sorted(vals.items()):
        if len(v) >= 1:
            s = "  ".join(f"{k}={x:.4f}" for k,x in v)
            spreads = [abs(x1[1]-x2[1]) for x1 in v for x2 in v if x1[0]!=x2[0]]
            note = f" (max spread {max(spreads):.4f})" if spreads else ""
            report.append(f"  {tag:22s} {s}{note}")

    # ---- 4. 索引 ----
    idx["checkpoints_verified"] = {t: dict(arch=infer_arch(f"{CKPT}/{t}_best.pt")) for t in list(EXPECTED) if os.path.exists(f"{CKPT}/{t}_best.pt")}
    idx["cross_scores"] = {t: {k: x for k,x in v} for t,v in vals.items() if v}
    json.dump(idx, open(OUT, "w"), indent=2, ensure_ascii=False)
    open(REPORT, "w", encoding="utf-8").write("\n".join(report))
    print("\n".join(report))
    print(f"\n-> verify_report.txt & experiments_index.json 已生成")

if __name__ == "__main__":
    main()
