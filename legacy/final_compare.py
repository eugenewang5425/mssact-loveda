"""最终对比: 数据量/后训练/TTA 的全维度结果汇总
数据源: tta_eval/complete_eval.json + checkpoints/*_history.json
"""
import os, sys, json, glob
import paths  # 集中路径配置 (环境变量/.env)

BASE = paths.REPO
CKPT = f"{BASE}/checkpoints"
TT = f"{BASE}/tta_eval"

def load_tt():
    p = f"{TT}/complete_eval.json"
    return json.load(open(p)) if os.path.exists(p) else {}

def hist_best(tag):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return None
    try:
        h = json.load(open(p))
        return max(h, key=lambda r: r.get("kappa",-9))
    except Exception:
        return None

def main():
    tt = load_tt()
    # 也扫单独的 conf 文件
    for f in glob.glob(f"{TT}/*.json"):
        tag = os.path.basename(f).replace(".json","")
        if tag in ("complete_eval","tta_results"): continue
        if tag not in tt:
            try: tt[tag] = json.load(open(f))
            except Exception: pass

    md = []
    md.append("# 数据量 / 后训练 / TTA 完整对比\n")
    md.append("## 1. 数据量效应（同一模型架构 light 6.1M, 同一 OneCycle-60 协议）\n")
    md.append("| 模型 | 训练数据 | val Kappa | val OA | test Kappa | test OA |")
    md.append("|---|---|---|---|---|---|")
    for tag, n, label in [("mssact_full60", 957, "MSSACT-light"), ("full_all", 2257, "MSSACT-light")]:
        b = hist_best(tag)
        v = tt.get(f"{tag}_val", {}); t = tt.get(f"{tag}_test", {})
        md.append(f"| {label} | {n} | "
                  f"{b['kappa']:.4f} (4角) | {b['oa']:.4f} | "
                  f"{t.get('kappa', float('nan')):.4f} | {t.get('oa', float('nan')):.4f} |")
    md.append("\n## 2. TTA 效应（8× 几何变换平均）\n")
    md.append("| 配置 | Kappa | OA | mF1 | mIoU |")
    md.append("|---|---|---|---|---|")
    for tag in ["full_all_val","full_all_val_TTA","full_all_test","full_all_test_TTA"]:
        if tag in tt:
            r = tt[tag]
            md.append(f"| {tag} | {r['kappa']:.4f} | {r['oa']:.4f} | {r['mf1']:.4f} | {r['miou']:.4f} |")
    md.append("\n## 3. 每类 F1 对比（test 集）\n")
    names = ["背景","建筑","道路","水域","裸地","森林","农田"]
    rows = {}
    for tag in ["mssact_full60_test","full_all_test","full_all_test_TTA"]:
        if tag in tt: rows[tag] = tt[tag]["f1"]
    if rows:
        md.append("| 类别 | " + " | ".join(rows.keys()) + " |")
        md.append("|---" * (len(rows)+1) + "|")
        for i,n in enumerate(names):
            md.append(f"| {n} | " + " | ".join(f"{v[i]:.3f}" for v in rows.values()) + " |")
    out = "\n".join(md)
    open(f"{BASE}/最终对比_数据量_TTA.md","w",encoding="utf-8").write(out)
    print(out)

if __name__ == "__main__":
    main()
