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
import os, sys, json, glob
import numpy as np

sys.path.insert(0, _HERE)
import paths

CKPT = paths.CKPT

GROUPS = {
    "完整模型 MSSACT (P-MEM-ROLL)": [
        ("lgR_bs8_full_lr2e4", 42), ("sd7_full", 7), ("sd2024_full", 2024),
        ("sd31337_full", 31337)] + [
        # 2026-09-15 种子扩展: n=4 -> 10 (GEO-Bench 建议 >=10)
        ("sd%d_full" % s, s) for s in (1234, 5555, 8888, 31415, 27182, 9999)],
    # /!\ DeepLab 必须按**主干预训练状态**拆开, 不可混为一组 (2026-09-15 事故):
    # 旧 3 个 seed 训练于 pretrained_backbone 默认仍为 True 的时期
    # (主干 bn1.weight 均值 0.248), 新 6 个训练于默认改为 False 之后 (0.973)。
    # 混成一组会把"配置差 0.065"当成"种子噪声", 得到 sigma=0.0334 (真实种子噪声
    # 仅约 0.003); 而该 sigma 又会被下游拿去给 MSSACT 消融定门限 -> 一切差异都
    # 判成"不可分辨"。拆分依据 = 实测 BN 统计, 见 check_group_homogeneity()。
    "DeepLabV3+ 预训练主干 (P-MEM-ROLL)": [
        ("sd7_deeplab", 7), ("sd2024_deeplab", 2024), ("sd31337_deeplab", 31337)],
    "DeepLabV3+ 从零 (P-MEM-ROLL)": [
        ("sd%d_deeplab" % s, s) for s in (1234, 5555, 8888, 31415, 27182, 9999)],
    "REF-参照(非种子组): 完整模型/DeepLab 的 P-PNG 单次结果": [
        ("full_all_v2", 42), ("nd_deeplab", 42)],
}
# 必须区分两类，否则散布会被容量效应主导而误读为"模块效应":
#   模块消融  = 移除/移动某个模块，容量基本不变
#   容量变体  = 改变通道宽度（D3 tiny / D4 large），是"容量"实验
MODULE_ABL_MEMROLL = ["lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca"]
CAPACITY_MEMROLL   = ["lgR_D3_ch_tiny", "lgR_D4_ch_large"]
MODULE_ABL_PNG = ["nd_abl_no_emr", "nd_abl_no_ecsam", "nd_abl_no_fpn", "nd_abl_no_trans",
                  "nd_abl_no_adapter", "nd_abl_bilinear",
                  "nd_abl_trans4l", "nd_abl_trans6l"]


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


def bn_pretrain_stat(tag):
    """用主干首个 BN 层的 weight 均值判断该 checkpoint 是否加载过 ImageNet 预训练。

    判定依据(此前对 DeepLab 取证时确立): 从零训练 ≈1.0, ImageNet 预训练 ≈0.25。
    用途: 组内同质性检查 —— 种子组必须是**同一配置**, 否则其 std 混入配置差而非噪声。
    """
    import torch as _t
    p = os.path.join(CKPT, f"{tag}_best.pt")
    if not os.path.exists(p):
        return None
    try:
        sd = _t.load(p, map_location="cpu", weights_only=True)
        keys = [k for k in sd if k.endswith("bn1.weight") and "backbone" in k]
        if not keys:
            keys = [k for k in sd if k.endswith("bn1.weight")][:1]
        if not keys:
            return None
        return float(sd[keys[0]].float().mean())
    except Exception:
        return None


def param_count(tag):
    """checkpoint 的 nn.Parameter 计数(架构指纹); 失败返回 None"""
    import torch as _t
    p = os.path.join(CKPT, f"{tag}_best.pt")
    if not os.path.exists(p):
        return None
    try:
        sd = _t.load(p, map_location="cpu", weights_only=True)
        return sum(int(v.numel()) for k, v in sd.items()
                   if hasattr(v, "numel") and not k.endswith("num_batches_tracked")
                   and ".running_" not in k)
    except Exception:
        return None


def check_group_homogeneity(groups):
    """组内同质性: 种子组必须是同一**配置**。检出两类混合, 二者都会污染 σ:

    (a) 主干预训练状态不一致 —— 2026-09-15 实际发生: 3 个预训练版 + 6 个从零版
        DeepLab 混成一组, 得到 σ=0.0334(真实种子噪声仅 ~0.003, 差 11 倍),
        而该 σ 会被下游拿去给 MSSACT 消融定门限, 使一切差异都判成"不可分辨"。
    (b) 架构不一致 —— 用参数量做指纹检出(如"参照"行曾把 6.1M 与 39.6M 并成"一组")。

    返回 [(组名, {原因: [tag...]})]; 空列表 = 全部通过。
    """
    bad = []
    for name, members in groups.items():
        if name.startswith("REF-"):
            continue          # 参照行不是种子组(本就混架构), 不参与 sigma
        kinds, params = {}, {}
        for tag, _seed in members:
            v = bn_pretrain_stat(tag)
            if v is not None:
                kinds[tag] = "预训练" if v < 0.6 else "从零"
            params[tag] = param_count(tag)
        mixed = {}
        if len(set(kinds.values())) > 1:
            for t, k in kinds.items():
                mixed.setdefault("主干预训练=" + k, []).append(t)
        pc = {t: c for t, c in params.items() if c}
        if len(set(pc.values())) > 1:
            for t, c in pc.items():
                mixed.setdefault("参数量=%d" % c, []).append(t)
        if mixed:
            bad.append((name, mixed))
    return bad


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
    for label, tags in (("P-MEM-ROLL 模块消融 (D1,D2)", MODULE_ABL_MEMROLL),
                        ("P-MEM-ROLL 容量变体 (D3,D4)", CAPACITY_MEMROLL),
                        ("P-PNG 模块消融 (nd_abl_*)", MODULE_ABL_PNG)):
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

    # ---- 组内同质性检查(自动拦截"混合配置"造成的假 sigma) ----
    _bad = check_group_homogeneity(GROUPS)
    out["group_homogeneity"] = {"ok": not _bad, "mixed": [
        {"group": g, "by": k} for g, k in _bad]}
    print("")
    if _bad:
        print("!" * 78)
        print("!! 组内同质性检查未通过 —— 下列组混入了不同配置的成员,")
        print("!! 其 sigma 混有'配置差'而非纯'种子噪声', **不可用于任何门限判定**:")
        for g, k in _bad:
            print("!!   %s" % g)
            for kind, tags in k.items():
                print("!!      %s: %s" % (kind, tags))
        print("!" * 78)
    else:
        print("组内同质性检查: 通过 (各组的主干预训练状态与架构均一致)")

    # 判读
    print("\n" + "=" * 78)
    print("判读: ΔKappa 折算为多少个 σ_seed")
    print("=" * 78)
    # 注意: 记录里的键是 "std"（打印时才标为 sigma）。
    # 此前这里写成 `"sigma" in r`，恒为假 -> sigma 永远 None、折算表从未生成。
    # sigma_seed_used = **与消融同配置**的种子噪声, 即 MSSACT 完整模型组。
    # 不能用"各管线取最大(保守)" —— σ 是**配置相关**的: 不同架构的种子噪声不可互换。
    # 事故记录(2026-09-15): 曾用 max(各管线), 因 DeepLab 组被污染(σ 0.0334, 真实 0.003)
    # 而把 MSSACT 消融的门限放大 8 倍, 所有真实差异都会被判成"不可分辨"。
    # 跨配置的 σ 仍各自记录在 groups 里, 供对应架构的判定使用。
    _KEY = "完整模型 MSSACT (P-MEM-ROLL)"
    sigma = None
    if _KEY in out["groups"] and "std" in out["groups"][_KEY]:
        sigma = out["groups"][_KEY]["std"]
    if sigma:
        out["sigma_seed_used"] = sigma
        out["sigma_seed_used_source"] = ("与消融同配置(" + _KEY + ")的种子噪声; "
                                         "不用跨架构取最大, 因为 sigma 是配置相关的")
        print("")
        print("采用与消融同配置的 σ_seed = %.4f (%s)" % (sigma, _KEY))
        for g, r in out["groups"].items():
            if g != _KEY and "std" in r and not g.startswith("REF-"):
                print("    参照(不用于 MSSACT 判定): %s σ=%.4f n=%d" % (g, r["std"], r["n"]))
        rows = []
        for label, tags in (("P-MEM-ROLL", MODULE_ABL_MEMROLL + CAPACITY_MEMROLL),
                            ("P-PNG", MODULE_ABL_PNG)):
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

    dst = paths.SEEDVAR_JSON
    # ---- 协议匹配的 sigma: 测试集协议下的种子噪声 ----
    # 为什么必须分开测: sigma_seed 是**协议相关**的。用 val 协议测出的 0.0050 去判定
    # 整图测试协议(1024^2 滑窗)上的差值属于口径错配 —— 两个协议的评估集、分辨率、
    # 推理方式都不同, 噪声水平没有理由相同。实测 val 0.0050 / test 0.0041。
    try:
        import os as _os2
        rigor_p = _os2.path.join(paths.CONSENSUS, "rigor.json")
        if _os2.path.exists(rigor_p):
            _r = json.load(open(rigor_p, encoding="utf-8"))["results"]
            _grp = ([("lgR_bs8_full_lr2e4", 42), ("sd7_full", 7),
                     ("sd2024_full", 2024), ("sd31337_full", 31337)] +
                    [("sd%d_full" % s, s) for s in (1234, 5555, 8888, 31415, 27182, 9999)])
            _v = [(t, s, _r[t]["kappa_pooled"]) for t, s in _grp if t in _r]
            if len(_v) >= 2:
                import statistics as _st
                _ks = [k for _, _, k in _v]
                out["sigma_seed_test_protocol"] = dict(
                    n=len(_ks), mean=round(_st.mean(_ks), 6),
                    std=round(_st.stdev(_ks), 6),
                    range=round(max(_ks) - min(_ks), 6),
                    members={t: round(k, 6) for t, _, k in _v},
                    protocol="test_clean 141 张 1024^2 整图, 256^2 滑窗/stride 128 高斯加权",
                    note="与 sigma_seed_used(val 协议)不同口径; 判定测试集协议的差值须用本值")
                print("  测试集协议 sigma_seed = %.4f (n=%d)" % (_st.stdev(_ks), len(_ks)))
    except Exception as _e:
        out["sigma_seed_test_protocol"] = None
        print("  测试集协议 sigma 未测得: %s" % _e)

    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {dst}")


if __name__ == "__main__":
    main()
