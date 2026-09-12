# -*- coding: utf-8 -*-
"""
剩余时间估算 (eta_queue.py)

方法：按队列定义逐项累加，每项用**实测**每轮耗时（无实测的用同管线已知值按
计算量比例外推），并明确标注假设来源。可随时重跑，会自动扣除已完成轮次。

关键实测事实（来自主队列日志与 verify_rollback.py）
--------------------------------------------------
memmap 管线（计算受限，耗时随模型规模变化）：
    完整模型 6.102M = 43 s/轮      D1 2.005M = 14.7 s/轮
    D2 6.115M      = 44 s/轮      D3 1.547M = 29 s/轮
    D4 24.242M     = 750 s/轮（ep1 866, ep2 732）
PNG 管线（**数据受限**，耗时与模型规模几乎无关）：
    完整模型 144 s/轮   nd_deeplab 151 s/轮
    nd_abl_no_ecsam 158 s/轮   nd_abl_no_emr 162 s/轮
-> 故 PNG 管线不能按模型规模外推；memmap 管线可以。

用法: python eta_queue.py
"""
import os, sys, json, re, time, glob

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, "checkpoints")
LOGDIR = r"C:/Users/Administrator/.zcode/cli/exec/sess_79382b4e-b5fc-48a3-9f7f-73768fa1f2f2"
MAIN_LOG = os.path.join(LOGDIR, "call_00_p2ZpXJD7LuBTVfXsAVqp0139-stdout.log")

# ---- 实测每轮秒数（memmap 管线） ----
T = {"full": 43, "d1": 14.7, "d2": 44, "d3": 29, "d4": 750,
     "unet": 38, "deeplab": 50, "noecsam": 43, "noemr": 43}
# PNG 管线每图成本（数据瓶颈），用于数据阶梯（fast_data=False）
PNG_S_PER_IMAGE = {"full": 81, "noecsam": 81, "unet": 78, "deeplab": 85}  # 81ms/图 = 144s/1768

# ---- 队列定义（与 run_queue_arch.py / final / crop 一致） ----
B15 = [("lgR_b15_full", "full"), ("lgR_b15_noecsam", "noecsam"),
       ("lgR_b15_noemr", "noemr"), ("lgR_b15_unet", "unet"),
       ("lgR_b15_deeplab", "deeplab")]
LADDER_N = [250, 500, 1000]
LADDER_M = ["full", "noecsam", "unet", "deeplab"]
SEEDS = [7, 2024, 31337]
CROP = [("lgR_c256_center", 256, 8), ("lgR_c384_rand", 384, 4), ("lgR_c512_rand", 512, 2)]


def done_epochs(tag):
    p = os.path.join(CKPT, f"{tag}_history.json")
    if not os.path.exists(p):
        return 0
    try:
        h = json.load(open(p))
        return len(h) if isinstance(h, list) else 0
    except Exception:
        return 0


def main():
    rows = []          # (阶段, 任务, 剩余轮, 秒/轮, 小时)
    P1 = P2 = P3 = 0.0

    # ===== 阶段1: run_queue_arch.py 剩余 =====
    for tag, key in [("lgR_D4_ch_large", "d4")] + B15:
        target = 15 if tag.startswith("lgR_b15_") else 60
        dn = done_epochs(tag)
        if dn >= target:
            continue
        rem = target - dn if dn > 0 else target
        h = rem * T[key] / 3600.0
        rows.append(("1 主队列", tag, rem, T[key], h)); P1 += h
    # 数据阶梯（PNG 管线, 每轮 = n × 每图秒数）
    for n in LADDER_N:
        for m in LADDER_M:
            tag = f"lgR_n{n}_{m}"
            dn = done_epochs(tag); rem = max(0, 30 - dn)
            if dn >= 30:
                continue
            s = n * PNG_S_PER_IMAGE[m] / 1000.0
            h = rem * s / 3600.0
            rows.append((f"1 主队列", tag, rem, round(s), h)); P1 += h
    rows.append(("1 主队列", "eval_ladder.py", 1, 600, 600 / 3600)); P1 += 600 / 3600

    # ===== 阶段2: run_queue_final.py =====
    for s in SEEDS:
        for key, tag in (("full", f"sd{s}_full"), ("deeplab", f"sd{s}_deeplab")):
            dn = done_epochs(tag); rem = max(0, 60 - dn)
            if dn >= 60:
                continue
            h = rem * T[key] / 3600.0
            rows.append(("2 后置队列", tag, rem, T[key], h)); P2 += h
    for tag, sec, note in [("eval_rigor --infer (22 tag)", 90, "每 tag 约 90s"),
                           ("headroom_analysis.py", 600, ""),
                           ("seed_variance+build_index+build_facts", 180, ""),
                           ("make_report_v2 + make_pdf", 120, "")]:
        h = sec / 3600.0
        rows.append(("2 后置队列", tag, 1, sec, h)); P2 += h
    P2 += 22 * 90 / 3600   # eval_rigor 是 22 个 tag, 上面记了 1 次, 补齐
    rows[-4] = ("2 后置队列", "eval_rigor --infer (22 tag)", 22, 90, 22 * 90 / 3600)

    # ===== 阶段3: run_queue_crop.py =====
    for tag, crop, _bs in CROP:
        dn = done_epochs(tag); rem = max(0, 60 - dn)
        if dn >= 60:
            continue
        s = T["full"] * (crop / 256.0) ** 2      # 像素量平方增长
        h = rem * s / 3600.0
        rows.append(("3 裁剪队列", tag, rem, round(s), h)); P3 += h
    for tag, sec in [("build_index+build_facts", 180), ("make_report_v2 + make_pdf", 120)]:
        h = sec / 3600.0
        rows.append(("3 裁剪队列", tag, 1, sec, h)); P3 += h

    total = P1 + P2 + P3
    print("=" * 92)
    print("剩余时间估算（基于实测每轮耗时；无实测项按同管线计算量外推）")
    print("=" * 92)
    print(f"{'阶段':12s} {'任务':32s} {'剩余轮':>7s} {'秒/轮':>8s} {'小时':>7s}")
    for st, tag, rem, s, h in rows:
        if h < 0.001:
            continue
        print(f"{st:12s} {tag:32s} {rem:7d} {s:8.0f} {h:7.2f}")
    print("-" * 92)
    print(f"{'阶段1 主队列合计':44s} {P1:7.2f} h")
    print(f"{'阶段2 后置队列合计':44s} {P2:7.2f} h")
    print(f"{'阶段3 裁剪队列合计':44s} {P3:7.2f} h")
    print(f"{'总计':44s} {total:7.2f} h  ({total/24:.1f} 天)")
    print()
    # 关键占比
    d4 = [r for r in rows if r[1] == "lgR_D4_ch_large"]
    if d4:
        print(f"其中 lgR_D4_ch_large 占 {d4[0][4]:.2f} h "
              f"（总时长的 {100*d4[0][4]/total:.0f}%）")
    print()
    print("假设与来源：")
    print("  memmap 管线: full 43s / D1 14.7s / D2 44s / D3 29s / D4 750s  （日志实测）")
    print("  unet 38s, deeplab 50s: 由 PNG 管线实测值扣掉数据成分后得到")
    print("  PNG 管线（数据阶梯 fast_data=False）: 81 ms/图（= 144s/1768 图, 实测）")
    print("  裁剪 384/512 按像素量平方外推（2.25× / 4×）")


if __name__ == "__main__":
    main()
