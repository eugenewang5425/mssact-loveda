"""
随机种子噪声底线 (seed_variance.py) —— 回答"ΔKappa 0.006 是否可分辨"

背景（文献要求，见 docs/架构有效性分析.md §9）
--------------------------------------------
本项目此前所有比较都是**单次运行**（seed=42），从未测量种子方差。文献指出:
  - GEO-Bench (NeurIPS 2023 D&B): 建议 >=10 seed, 3-5 seed 不足以给出可靠 CI
  - mmseg 实测: 同一 DeepLabV3+ 两次训练在 Cityscapes 上可差 0.7 mIoU
    (双线性插值反向非确定)
  - 严格区分"无证据有差异"与"有证据无差异"需要 TOST 等效性检验
因此必须实测 σ_seed，才能判断消融变体散布 0.006 是否落在噪声内。

重要: 只在**同一管线世代**内比较（跨世代 Kappa 不可比）
------------------------------------------------------
  MEM-ROLL 世代: lgR_bs8_full_lr2e4(seed42), sd7_full, sd2024_full, sd31337_full
                 nd_deeplab 属 P-PNG 世代 -> 仅作参照, 不混入均值
                 sd7/sd2024/sd31337_deeplab 同属 MEM-ROLL -> 可成组

输出
----
  1. 各组 mean / std / min / max / range
  2. σ_seed（样本标准差, 注意 n 很小 -> 同时报告 range）
  3. 与消融散布 0.006 的比较（含 n 小导致的 σ 不确定性警告）
  4. 判读表: 每个 ΔKappa 折算为 "多少个 σ"

用法: python seed_variance.py
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

CKPT = paths.CKPT

GROUPS = {
    "完整模型 MSSACT 6.10M (P-MEM-ROLL)": [
        ("lgR_bs8_full_lr2e4", 42), ("sd7_full", 7), ("sd2024_full", 2024),
        ("sd31337_full", 31337)],
    "DeepLabV3+ 39.69M (P-MEM-ROLL)": [
        ("sd7_deeplab", 7), ("sd2024_deeplab", 2024), ("sd31337_deeplab", 31337)],
    "参照: 完整模型/DeepLab 的 P-PNG 单次结果": [
        ("full_all_v2", 42), ("nd_deeplab", 42)],
}
# 消融变体（用于比较散布）—— 同属 G1/G2 组
ABLATION_MEMROLL = ["lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca",
                    "lgR_D3_ch_tiny", "lgR_D4_ch_large"]
ABLATION_PNG = ["nd_abl_no_emr", "nd_abl_no_ecsam", "nd_abl_no_fpn", "nd_abl_no_trans",
                "nd_abl_no_adapter", "nd_abl_bilinear", "nd_abl_trans4l", "nd_abl_trans6l"]


def best_k(tag):
    p = os.path.join(CKPT, f"{tag}_history.json")
    if not os.path.exists(p):
        return None
    try:
        h = json.load(open(p))
        if not h:
            return None
        return float(max(h, key=lambda r: r.get("kappa", -9)).get("kappa"))
    except Exception:
        return None


def main():
    out = {"groups": {}, "ablations": {}}
    print("=" * 78)
    print("随机种子噪声底线")
    print("=" * 78)
    for gname, members in GROUPS.items():
        rows = []
        for tag, seed in members:
            k = best_k(tag)
            rows.append((tag, seed, k))
        ks = np.array([k for _, _, k in rows if k is not None])
        print(f"\n{gname}")
        for tag, seed, k in rows:
            print(f"   seed={seed:<6d} {tag:<26s} Kappa=" +
                  (f"{k:.4f}" if k is not None else "  缺失"))
        if len(ks) >= 2:
            rec = dict(n=len(ks), mean=round(float(ks.mean()), 4),
                       std=round(float(ks.std(ddof=1)), 4),
                       min=round(float(ks.min()), 4), max=round(float(ks.max()), 4),
                       range=round(float(ks.max() - ks.min()), 4),
                       members={t: (round(k, 4) if k is not None else None) for t, _, k in rows})
            out["groups"][gname] = rec
            print(f"   -> n={rec['n']}  mean={rec['mean']:.4f}  sigma={rec['std']:.4f}  "
                  f"range={rec['range']:.4f}  [{rec['min']:.4f}, {rec['max']:.4f}]")
        else:
            out["groups"][gname] = {"n": len(ks), "note": "样本不足, 仅作参照"}

    # 消融散布
    print("\n" + "=" * 78)
    print("消融变体散布（同管线内）")
    print("=" * 78)
    abl = {}
    for label, tags in (("P-MEM-ROLL (lgR_*)", ABLATION_MEMROLL),
                        ("P-PNG (nd_abl_* + lg_D*)", ABLATION_PNG)):
        vals = {t: best_k(t) for t in tags}
        vals = {t: v for t, v in vals.items() if v is not None}
        if vals:
            arr = np.array(list(vals.values()))
            abl[label] = dict(
                n=len(arr), spread=round(float(arr.max() - arr.min()), 4),
                min=round(float(arr.min()), 4), max=round(float(arr.max()), 4),
                values={t: round(v, 4) for t, v in vals.items()})
            print(f"\n{label}: n={len(arr)} 散布={abl[label]['spread']:.4f} "
                  f"[{abl[label]['min']:.4f}, {abl[label]['max']:.4f}]")
            for t, v in sorted(vals.items(), key=lambda kv: -kv[1]):
                print(f"   {t:<28s} {v:.4f}")
    out["ablations"] = abl

    # 判读
    print("\n" + "=" * 78)
    print("判读: ΔKappa 折算为多少个 σ_seed")
    print("=" * 78)
    sigma = None
    for g, r in out["groups"].items():
        if r.get("n", 0) >= 3 and "sigma" in r:
            sigma = r["sigma"] if sigma is None else max(sigma, r["sigma"])
    if sigma:
        out["sigma_seed_used"] = sigma
        print(f"\n采用各管线内最大 σ_seed = {sigma:.4f} (保守)")
        rows = []
        for label, tags in (("P-MEM-ROLL", ABLATION_MEMROLL), ("P-PNG", ABLATION_PNG)):
            ref = "lgR_bs8_full_lr2e4" if label == "P-MEM-ROLL" else "full_all_v2"
            kr = best_k(ref)
            if kr is None:
                continue
            for t in tags:
                v = best_k(t)
                if v is None:
                    continue
                d = v - kr
                rows.append((label, t, v, d, d / sigma))
        print(f"\n{'管线':11s} {'变体':28s} {'Kappa':>7s} {'ΔKappa':>8s} {'σ倍数':>7s}  判读")
        for label, t, v, d, ns in sorted(rows, key=lambda r: -abs(r[4])):
            verdict = ("不可分辨 (|Δ|<1σ)" if abs(ns) < 1 else
                       ("弱" if abs(ns) < 2 else "可分辨"))
            print(f"{label:11s} {t:28s} {v:7.4f} {d:+8.4f} {ns:+7.2f}  {verdict}")
        out["deltas_in_sigma"] = [
            dict(pipeline=l, tag=t, kappa=round(v, 4), delta=round(d, 4),
                 n_sigma=round(ns, 3)) for l, t, v, d, ns in rows]
        print(f"\n基准: P-MEM-ROLL -> lgR_bs8_full_lr2e4;  P-PNG -> full_all_v2")
        print("说明: n 很小 (3-4 seed) 时 σ 本身有较大不确定性; GEO-Bench 建议 >=10 seed。")
        print("      故本表用于**量级**判断(0.006 是 1σ 还是 5σ), 不宜作精确检验。")
    else:
        print("\n多种子结果尚不完整, 无法计算 σ_seed。")

    # 警告
    print("\n" + "=" * 78)
    print("注意事项")
    print("=" * 78)
    print("1. nd_deeplab 属 P-PNG 世代, 不得与 sd*_deeplab (P-MEM-ROLL) 混合求均值")
    print("   (跨管线 Kappa 不可比; 实测后端差异 +0.0008)")
    print("2. 训练路径只用 torch RNG, 故 seed 参数不影响 seed=42 的既有结果")
    print("3. 裁剪位置由 epoch_seed 决定, 与 seed 无关; seed 控制的是")
    print("   模型初始化 / DataLoader 打乱顺序 / _geom 增强抽样")

    dst = os.path.join(paths.REPO, "seed_variance.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {dst}")


if __name__ == "__main__":
    main()
