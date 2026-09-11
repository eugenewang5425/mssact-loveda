"""汇总全部实验结果 -> 对比/消融表(markdown+json)

数据源: checkpoints/{name}_history.json
  多尺度: v3_s128gf / v3_s256 / v3_s512 / v3_s1024
  基线:   unet / pspnet / fcn / deeplab / segformer
  消融:   ablate_no_emr / ablate_no_ecsam / ablate_no_fpn / ablate_no_trans / ablate_no_adapter
"""
import json, os
import os
import paths  # 集中路径配置 (环境变量/.env)
CKPT = paths.CKPT

def best_of(tag):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return None
    h = json.load(open(p))
    b = max(h, key=lambda r: r["kappa"])
    return dict(tag=tag, epochs=len(h), best_epoch=b["epoch"],
                oa=b.get("oa"), kappa=b["kappa"], mf1=b.get("mf1"),
                miou=b.get("miou"))

def load(tag): return best_of(tag)

def main():
    # ---- 1. 多尺度 (v3) ----
    scale_rows = [r for r in [load(t) for t in ["v3_s256","v3_s512","v3_s1024","v3_s128gf"]] if r]
    # ---- 2. 基线对比 ----
    base_rows = [r for r in [load(t) for t in ["unet","pspnet","fcn","deeplab","segformer","mssact_full60"]] if r]
    mssact = load("v3_s256")
    if mssact: mssact["tag"] = "mssact(s256,120ep)"
    # ---- 3. 消融 ----
    abl_rows = [r for r in [load(t) for t in ["mssact_full60","ablate_no_emr","ablate_no_ecsam","ablate_no_fpn","ablate_no_trans","ablate_no_adapter"]] if r]
    for r in abl_rows:
        if r["tag"]=="mssact_full60": r["tag"]="full(60ep)"
    full = load("v3_s256")
    if full: full = dict(full, tag="full(s256)")

    def table(rows, ref=None):
        md = ["| 配置 | 轮数 | 最优轮 | OA | Kappa | mF1 | mIoU |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            miou = f"{r['miou']:.4f}" if r.get("miou") is not None else "—"
            md.append(f"| {r['tag']} | {r['epochs']} | {r['best_epoch']} | {r['oa']:.4f} | {r['kappa']:.4f} | {r['mf1']:.4f} | {miou} |")
        return "\n".join(md)

    md = ["# 实验汇总 (LoveDA 验证集, 256px @0.3m, 统一收敛协议)\n",
          "统一协议: AdamW(2e-4, wd=0.02) + OneCycle(6% warmup) + EMA(0.999) + 类别加权CE + 验证Kappa早停\n"]
    md += ["## 1. 多尺度对照\n", table(scale_rows), ""]
    if mssact:
        md += ["## 2. 对比实验（基线 vs MSSACT-Net）\n", table(base_rows + [mssact]), ""]
    if full:
        md += ["## 3. 消融实验（MSSACT-Net 组件）\n", table([full] + abl_rows), ""]
    md += ["注: mIoU 仅 v3 系列记录; 基线/消融的 mIoU 为 — (协议相同, Kappa/mF1/OA 完全可比)。"]
    out_md = "\n".join(md)
    open(f"{CKPT}/experiments_summary.md","w",encoding="utf-8").write(out_md)
    json.dump(dict(scales=scale_rows, baselines=base_rows+[mssact] if mssact else base_rows,
                   ablations=[full]+abl_rows if full else abl_rows),
              open(f"{CKPT}/experiments_summary.json","w"), indent=1)
    print(out_md)

if __name__ == "__main__":
    main()
