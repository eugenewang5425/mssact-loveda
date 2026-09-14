# -*- coding: utf-8 -*-
"""缺陷修复归因分析 —— 一处命令产出全部 lgR120 结论

为什么这个脚本必须存在
--------------------
修复了 D1/D2/D3 之后, "修复是否有效"不能靠肉眼看 Kappa。必须同时回答四件事,
且每一件都有明确的口径:

1. **修复的效应量**: ΔKappa 相对**同为 120 轮**的基线 `lgR120_full`,
   按 sigma_seed = 0.0050 分级（|Δ| >= 2σ 记为可分辨）。
2. **调度效应的对照量**: 同时给出相对 60 轮基线 `lgR_bs8_full_lr2e4` 的差值,
   并**显式标注它不可用于归因** —— 它混合了"延长退火"与"修复缺陷"两种效应。
   实测 120 轮基线已达 0.6655 vs 60 轮 0.6320（+0.0335）, 若不重跑基线,
   这 +0.033 会被误读成修复的功劳。**这一列是本脚本存在的主要理由。**
3. **收敛轮次**（用户明确要求"收敛轮次也是重要数据"）:
   G4 组内**学习率调度完全相同**, 故 best_epoch / 收敛速度的差异**可直接归因于
   架构** —— 无需再借助累计学习量 S=Ση 去分离调度效应（那是对跨调度比较才需要的）。
4. **边界带归属**: 跳连的预期收益主要在紧边界带。逐带 OA 取自
   `analysis/eval_rigor.py` 的 boundary 分段（紧边界仅占有效像素 2.36%）,
   故即使边界带 OA 大幅提升, 总 Kappa 的变动也可能很小 —— 这两件事必须分开陈述。

用法: python analysis/fix_attribution.py
输出: results/artifacts/fix_attribution.json
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths

BASE120 = "lgR120_full"          # 同调度基线
BASE60 = "lgR_bs8_full_lr2e4"    # 仅作调度效应对照, 不可用于归因
BASE_POS = "lgR120_fx_pos"       # 阶段 C 消融的底 (pos_enc=True, 0.6716)

FIX_ROWS = [
    ("lgR120_full", "基线（无修复）", "—"),
    ("lgR120_fx_skip", "修 D1+D2：跳连 + FPN 四级接入", "use_skip=True"),
    ("lgR120_fx_pos", "修 D3：2D 正弦位置编码", "pos_enc=True"),
    ("lgR120_fx_all", "三处全修", "use_skip=True, pos_enc=True"),
]
DIAG_ROWS = [
    ("lgR120_fx_d2", "H1 判别：只融合 128²/64²（不做 256² 全分辨率融合）"),
    ("lgR120_fxskip_lr1e4", "H2 判别：修 skip + lr 降到 1e-4"),
    ("lgR120_fxall_lr1e4", "H2 判别：全修 + lr 降到 1e-4"),
]
POS_ABL_ROWS = [
    ("lgR120_pos_abl_no_emr", "去掉 EMR（换标准残差块）"),
    ("lgR120_pos_abl_no_ecsam", "去掉 ECSAM 坐标注意力"),
    ("lgR120_pos_abl_no_fpn", "去掉 FPN"),
    ("lgR120_pos_abl_no_trans", "去掉 Transformer"),
    ("lgR120_pos_abl_no_adapter", "去掉 Adapter-Scale"),
    ("lgR120_pos_abl_bilinear", "转置卷积 → 双线性上采样"),
    ("lgR120_pos_abl_trans4l", "Transformer 层数 2 → 4"),
    ("lgR120_pos_abl_trans6l", "Transformer 层数 2 → 6"),
]
ABL_ROWS = [
    ("lgR120_abl_no_emr", "去掉 EMR（换标准残差块）"),
    ("lgR120_abl_no_ecsam", "去掉 ECSAM 坐标注意力"),
    ("lgR120_abl_no_fpn", "去掉 FPN（跳连改接编码器浅三层）"),
    ("lgR120_abl_no_trans", "去掉 Transformer"),
    ("lgR120_abl_no_adapter", "去掉 Adapter-Scale"),
    ("lgR120_abl_bilinear", "转置卷积 → 双线性上采样"),
    ("lgR120_abl_trans4l", "Transformer 层数 2 → 4"),
    ("lgR120_abl_trans6l", "Transformer 层数 2 → 6"),
]
TARGET_EPOCHS = 120
PATIENCE = 20
OUT = os.path.join(paths.RESULTS, "artifacts", "fix_attribution.json")


def read_hist(tag):
    p = os.path.join(paths.CKPT, "%s_history.json" % tag)
    if not os.path.exists(p):
        return None
    h = json.load(open(p, encoding="utf-8"))
    if not isinstance(h, list) or not h:
        return None
    return h


def summarise(tag):
    """返回 tag 的训练摘要; 缺失返回 None"""
    h = read_hist(tag)
    if h is None:
        return None
    best = max(h, key=lambda r: r.get("kappa", -9))
    ks = [r.get("kappa", float("nan")) for r in h]
    secs = [r["sec"] for r in h if r.get("sec")]
    n = len(h)
    # 收敛速度: 首次达到"自身最优的 95%"的轮次 (取 95% 是为了避开平台期的噪声抖动)
    def thresh_epoch(frac):
        if best["kappa"] is None or best["kappa"] <= 0:
            return None
        thr = best["kappa"] * frac
        for r in h:
            if r.get("kappa", -9) >= thr:
                return r["epoch"]
        return None
    # 完成判定: 跑满预算, 或已在最优之后再跑满一整个 patience 窗口(即早停已触发)。
    # 只看 n >= TARGET 会把"早停收敛"误判为未完成(lgR120_full 就是 best@91、跑了 111 轮),
    # 而把未完成的中间轮次拿去和基线比会得出"可分辨 -106σ"这类荒谬结论。
    complete = (n >= TARGET_EPOCHS) or (n >= best["epoch"] + PATIENCE - 1)
    # 数值失效标记: 发散后退化为常量输出, 指标会**精确冻结**, 看起来像"停在较差的
    # 平台"。若不标记, 会把发散当成模块效应的读数 —— lgR120_abl_no_ecsam 正是如此
    # (loss 恒为 0.0, OA 精确恒定, best.pt 权重最大绝对值 1.26e4)。
    n_zero_loss = sum(1 for r in h if r.get("loss") == 0.0)
    return dict(
        tag=tag, epochs_run=n, complete=bool(complete),
        kappa_best=round(float(best["kappa"]), 6), best_epoch=best["epoch"],
        kappa_final=round(float(ks[-1]), 6),
        diverged=bool(n_zero_loss >= 3), n_zero_loss=n_zero_loss,
        # 收敛轮次是数据: 达到自身最优 90%/95%/99% 的轮次
        epoch_at_90pct=thresh_epoch(0.90),
        epoch_at_95pct=thresh_epoch(0.95),
        epoch_at_99pct=thresh_epoch(0.99),
        sec_per_epoch=round(sum(secs) / len(secs), 1) if secs else None,
        n_sigma=None, delta_120=None, delta_60_unusable=None,
    )


def load_sigma():
    try:
        d = json.load(open(paths.SEEDVAR_JSON, encoding="utf-8"))
        return float(d.get("sigma_seed_used") or 0.0050), int(
            d.get("groups", {}).get("完整模型 MSSACT (P-MEM-ROLL)", {}).get("n", 0))
    except Exception:
        return 0.0050, 0


def load_boundary(tag):
    """从 rigor.json 取逐带 OA（若该 tag 已被评测）"""
    p = os.path.join(paths.CONSENSUS, "rigor.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
        r = d.get("results", {}).get(tag)
        return (r or {}).get("boundary")
    except Exception:
        return None


def grade(delta, sigma):
    if delta is None:
        return "待完成"
    ns = delta / sigma
    if abs(ns) >= 2:
        return "可分辨(%+.1fσ)" % ns
    if abs(ns) >= 1:
        return "弱(%+.1fσ)" % ns
    return "不可分辨(%+.1fσ)" % ns


def main():
    sigma, n_seed = load_sigma()
    rows = {}
    for tag, desc, cfg in FIX_ROWS:
        s = summarise(tag)
        if s:
            s["desc"], s["config"] = desc, cfg
        rows[tag] = s
    for tag, desc in DIAG_ROWS:
        s = summarise(tag)
        if s:
            s["desc"], s["config"] = desc, "判别实验"
        rows[tag] = s
    for tag, desc in POS_ABL_ROWS:
        s = summarise(tag)
        if s:
            s["desc"], s["config"] = desc, "底 = pos_enc=True (lgR120_fx_pos)"
        rows[tag] = s
    for tag, desc in ABL_ROWS:
        s = summarise(tag)
        if s:
            s["desc"], s["config"] = desc, "use_skip+pos_enc + " + tag.split("abl_")[-1]
        rows[tag] = s

    base = rows.get(BASE120)
    base60 = summarise(BASE60)
    if base:
        for tag, s in rows.items():
            if not s or tag == BASE120:
                continue
            if not s["complete"]:
                # 未跑完的实验**不做任何判定**, 也不填 delta —— 中间轮次的 Kappa
                # 与收敛值的差可能远大于任何效应量, 拿去比较只会产生假结论。
                s["delta_120"] = None
                s["n_sigma"] = None
                continue
            s["delta_120"] = round(s["kappa_best"] - base["kappa_best"], 6)
            s["n_sigma"] = round(s["delta_120"] / sigma, 2)
            if base60:
                s["delta_60_unusable"] = round(s["kappa_best"] - base60["kappa_best"], 6)
            b = load_boundary(tag)
            if b:
                s["boundary"] = b
    # 阶段 C 的消融相对**自己的底**(fx_pos), 而不是未修复基线 —— 底换了, 参照也要换
    pbase = rows.get(BASE_POS)
    if pbase:
        for tag, s in rows.items():
            if (not s or not s.get("complete") or tag == BASE_POS
                    or "pos_abl" not in tag):
                continue
            s["delta_pos"] = round(s["kappa_best"] - pbase["kappa_best"], 6)
            s["n_sigma_pos"] = round(s["delta_pos"] / sigma, 2)

    print("=" * 96)
    print("缺陷修复归因   σ_seed = %.4f (n=%d)   判别: |Δ| ≥ 2σ = %.4f 记为可分辨"
          % (sigma, n_seed, 2 * sigma))
    print("=" * 96)
    if base is None:
        print("基线 %s 尚未产出; 待训练完成后再运行。" % BASE120)
    else:
        print("基线 %s: K=%.4f  best@ep%s  %d 轮  %s s/轮"
              % (BASE120, base["kappa_best"], base["best_epoch"], base["epochs_run"],
                 base["sec_per_epoch"]))
        if base60:
            print("（对照）60 轮基线 %s: K=%.4f  best@ep%s"
                  % (BASE60, base60["kappa_best"], base60["best_epoch"]))
            print("  注意: 120 轮基线 - 60 轮基线 = %+.4f 是**调度效应**(延长退火), "
                  "不是修复效应。" % (base["kappa_best"] - base60["kappa_best"]))

    for title, specs in (("阶段一：修复归因（对照 = 同为 120 轮的基线）", FIX_ROWS),
                         ("阶段 A：判别实验（H1 全分辨率融合 / H2 学习率）", DIAG_ROWS),
                         ("阶段 B：修复后消融（底 = use_skip + pos_enc，该底不稳定）", ABL_ROWS),
                         ("阶段 C：修复后消融（底 = pos_enc，稳定且最优）", POS_ABL_ROWS)):
        print()
        print("--- %s ---" % title)
        is_c = title.startswith("阶段 C")
        print("%-24s %8s %8s %12s %10s %-18s" % (
            "tag", "K_best", "bestEp", "Δ(vs pos底)" if is_c else "Δ(120轮)",
            "Δ/σ", "判定"))
        for tag, *_ in specs:
            s = rows.get(tag)
            if not s:
                print("%-24s %8s %8s %9s %10s %s" % (tag, "-", "-", "-", "-", "未开始"))
                continue
            if tag == BASE120:
                print("%-24s %8.4f %8s %9s %10s %s" % (
                    tag, s["kappa_best"], s["best_epoch"], "-", "-", "（基线）"))
                continue
            if tag == BASE_POS and is_c:
                # 只在阶段 C 里显示为"底"; 在阶段一里它是有实测 Δ 的修复变体
                print("%-24s %8.4f %8s %9s %10s %s" % (
                    tag, s["kappa_best"], s["best_epoch"], "-", "-", "（pos 底）"))
                continue
            if not s["complete"]:
                print("%-24s %8.4f %8s %9s %10s %s" % (
                    tag, s["kappa_best"], s["best_epoch"], "-", "-",
                    "进行中(%d/%d 轮)" % (s["epochs_run"], TARGET_EPOCHS)))
                continue
            if s.get("diverged"):
                # 发散后退化为常量输出, 指标精确冻结; 这种数字**不可**当作模块效应
                print("%-24s %8.4f %8s %9s %10s %s" % (
                    tag, s["kappa_best"], s["best_epoch"], "-", "-",
                    "**发散(loss=0 共%d轮), 不可引用**" % s["n_zero_loss"]))
                continue
            d = s.get("delta_pos") if is_c else s.get("delta_120")
            ns = s.get("n_sigma_pos") if is_c else s.get("n_sigma")
            if d is None:
                print("%-24s %8.4f %8s %9s %10s %s" % (
                    tag, s["kappa_best"], s["best_epoch"], "-", "-", "无参照"))
                continue
            print("%-24s %8.4f %8s %9s %10s %-18s" % (
                tag, s["kappa_best"], s["best_epoch"],
                "%+.4f" % d, "%+.2f" % ns, grade(d, sigma)))

    # 收敛轮次表（用户要求: 收敛轮次本身是数据）
    print()
    print("--- 收敛轮次与收敛速度（G4 组内调度相同 ⇒ 差异可归因于架构）---")
    print("%-24s %6s %9s %10s %11s %11s" % (
        "tag", "轮数", "best@", "达90%@", "达95%@", "达99%@"))
    for tag, s in rows.items():
        if not s:
            continue
        mark = "" if s["complete"] else "  (进行中)"
        print("%-24s %6d %9s %10s %11s %11s%s" % (
            tag, s["epochs_run"], s["best_epoch"], s["epoch_at_90pct"],
            s["epoch_at_95pct"], s["epoch_at_99pct"], mark))

    # 边界带
    have_b = [t for t, s in rows.items() if s and s.get("boundary")]
    if have_b:
        print()
        print("--- 紧边界带 OA（跳连的预期收益主要在此带; 该带仅占有效像素 2.36%）---")
        keys = list(rows[have_b[0]]["boundary"].keys())
        print("%-14s " % "band(px)" + " ".join("%-18s" % t for t in have_b))
        for k in keys:
            cells = []
            for t in have_b:
                b = rows[t]["boundary"].get(k)
                cells.append("%-18s" % ("%.4f (%.1f%%)" % (b["oa"], b["frac"] * 100) if b else "-"))
            print("%-14s " % k + " ".join(cells))
    else:
        print()
        print("紧边界带 OA 暂无数据: 需先跑 analysis/eval_rigor.py --infer --only <tags>")

    out = dict(
        sigma_seed=sigma, sigma_seed_n_runs=n_seed, target_epochs=TARGET_EPOCHS,
        base_120=BASE120, base_60_reference_only=BASE60,
        schedule_effect_note=("120 轮与 60 轮是不同 OneCycle 调度; "
                             "两者差值含调度效应, 不可用于修复归因"),
        convergence_note=("G4 组内学习率调度完全相同, 故 best_epoch / 收敛速度差异"
                          "可直接归因于架构, 无需 S=Ση 分离"),
        rows=rows,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print("写入 %s" % OUT)
    n_done = sum(1 for s in rows.values() if s and s["complete"])
    print("  已完成 %d / %d 个实验（跑满 %d 轮, 或已在最优后再跑满 patience=%d 窗口）"
          % (n_done, len(rows), TARGET_EPOCHS, PATIENCE))


if __name__ == "__main__":
    main()
