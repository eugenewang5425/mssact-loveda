# -*- coding: utf-8 -*-
"""修复实验 第二轮: 判别实验 + 修复后消融 (lgR120_* 续)

第一轮的实测结果（全部 120 轮 / pt20 / bs8 / σ_seed = 0.0050）
-----------------------------------------------------------
| tag | Kappa | best@ | 相对 120 轮基线 |
|---|---:|---:|---:|
| lgR120_full（基线，未修复） | 0.6655 | 91 | — |
| lgR120_fx_skip（修 D1+D2） | 0.6505 | 114 | **−0.0150（3σ）** |
| lgR120_fx_pos（修 D3） | 0.6716 | 96 | +0.0061（1.2σ） |
| lgR120_fx_all（全修） | 0.6330 | 62 | **−0.0325** |

即：**修复在当前协议下并未带来收益，全修反而明显更差。** 且两项修复合起来
（−0.0325）比任何单项（−0.0150 / +0.0061）都差得多 —— 非加性，说明另有原因。

**过拟合假设已被推翻**：fx_all 的训练 loss 是四者中**最高**的（0.7237 vs 基线
0.6269），fx_skip 次之（0.6734）。它训练集拟合更差、验证也更差 ⇒ 是**欠拟合 /
优化困难**，不是过拟合。训练 loss 偏高又叠加了容量，指向**优化/条件数问题**。

两个待判别的假设
--------------
H1「全分辨率融合有害」: `_fuse` 在 k=2 处把主干**最浅层**特征（仅一个 EMR 块）
    直接接到 256² 输出, 同时让 FPN 三条全分辨率 3×3 支路（每条约 59 万参数）
    首次参与训练。检验: `skip_stages=(1,2)` 只融合 128²/64², 去掉 256² 那一级。
H2「学习率不再匹配」: 固定 max_lr=2e-4 是给未修复模型调的; 新增容量与新增梯度
    通路后可能需要更小的 LR。检验: 同配置下 lr 1e-4 vs 2e-4 直接对照。

阶段设计
--------
A 判别实验 (3 个, 约 5-6 小时): fx_d2 / fxskip_lr1e4 / fxall_lr1e4
B 修复后消融 (8 个, 约 13 小时): 底 = use_skip+pos_enc, **LR 由预先写定的规则选定**

**预登记的基线选择规则**（写在代码里, 不是看到结果再挑）:
    比较 lgR120_fx_all(lr 2e-4) 与 lgR120_fxall_lr1e4(lr 1e-4) 的 Kappa,
    取较优者所用的 LR 作为阶段 B 的 LR; 底配置固定为 use_skip=True + pos_enc=True。
这样阶段 B 一定跑在"全修复"的底上, 而 LR 取实测更优的那个, 避免用退化配置跑 13 小时。

用法: python queues/run_queue_fix2.py
"""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import os
import sys
import json
import time
import subprocess

import paths
from queue_guard import queues_running, POLL_INTERVAL, _no_window_kwargs   # noqa: E402
from experiment_matrix import train_one                                    # noqa: E402
from experiment_matrix_v2 import mssact_light                              # noqa: E402

BASE = _BASE
CKPT = paths.CKPT
ROOT_MAIN = paths.DATA_NEWSPLIT2
LOG = os.path.join(paths.LOGS, "queue_fix2.log")
EPOCHS, PATIENCE = 120, 20


def log(msg):
    line = "[fix2 %s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def hist(tag):
    p = os.path.join(CKPT, "%s_history.json" % tag)
    if not os.path.exists(p):
        return None
    try:
        h = json.load(open(p, encoding="utf-8"))
        return h if isinstance(h, list) and h else None
    except Exception:
        return None


def complete(tag):
    """跑满预算, 或已在最优之后再跑满一个 patience 窗口(早停已触发)"""
    h = hist(tag)
    if not h:
        return False
    b = max(h, key=lambda r: r.get("kappa", -9))
    return len(h) >= EPOCHS or len(h) >= b["epoch"] + PATIENCE - 1


def kappa(tag):
    h = hist(tag)
    if not h:
        return None
    return max(r.get("kappa", -9) for r in h)


def verify(tag):
    h = hist(tag)
    if not h:
        log("  !!! %s 无 history" % tag)
        return None
    b = max(h, key=lambda r: r.get("kappa", -9))
    secs = [r["sec"] for r in h if r.get("sec")]
    log("  OK %s: n=%d best_k=%.4f@ep%s%s" % (
        tag, len(h), b["kappa"], b["epoch"],
        "  %.1fs/轮" % (sum(secs) / len(secs)) if secs else ""))
    return b["kappa"]


# ==================== 阶段 A: 判别实验 ====================
PHASE_A = [
    ("lgR120_fx_d2", lambda: mssact_light(use_skip=True, skip_stages=(1, 2)), 2e-4,
     "H1: 只融合 128²/64², 不做 256² 全分辨率融合 (对照 fx_skip=0.6505)"),
    ("lgR120_fxskip_lr1e4", lambda: mssact_light(use_skip=True), 1e-4,
     "H2: 修 skip 但 lr 降到 1e-4 (对照 fx_skip=0.6505@2e-4)"),
    ("lgR120_fxall_lr1e4", lambda: mssact_light(use_skip=True, pos_enc=True), 1e-4,
     "H2: 全修 + lr 1e-4 (对照 fx_all=0.6330@2e-4)"),
]

# ==================== 阶段 B: 修复后消融 ====================
ABLATIONS = [
    ("lgR120_abl_no_emr", dict(use_emr=False), "去掉 EMR (换标准残差块)"),
    ("lgR120_abl_no_ecsam", dict(use_ecsam=False), "去掉 ECSAM 坐标注意力"),
    ("lgR120_abl_no_fpn", dict(use_fpn=False), "去掉 FPN (跳连改接编码器浅三层)"),
    ("lgR120_abl_no_trans", dict(use_transformer=False), "去掉 Transformer"),
    ("lgR120_abl_no_adapter", dict(use_adapter=False), "去掉 Adapter-Scale"),
    ("lgR120_abl_bilinear", dict(upsample_mode="bilinear"), "转置卷积 -> 双线性上采样"),
    ("lgR120_abl_trans4l", dict(transformer_layers=4), "Transformer 层数 2 -> 4"),
    ("lgR120_abl_trans6l", dict(transformer_layers=6), "Transformer 层数 2 -> 6"),
]


def pick_base_lr():
    """预登记的基线选择规则: 取 fx_all 与 fxall_lr1e4 中 Kappa 较高者所用的 LR"""
    cands = [("lgR120_fx_all", 2e-4), ("lgR120_fxall_lr1e4", 1e-4)]
    scored = [(kappa(t), lr, t) for t, lr in cands if complete(t)]
    if not scored:
        log("  两个候选都未完成, 回退到 lr=2e-4")
        return 2e-4, "(回退: 无已完成候选)"
    scored.sort(key=lambda x: -x[0])
    k, lr, t = scored[0]
    for kk, ll, tt in scored:
        log("  候选 %s: K=%.4f @lr=%.0e" % (tt, kk, ll))
    return lr, "%s (K=%.4f)" % (t, k)


def run_rows(rows, phase):
    log("")
    log("=" * 72)
    log("阶段%s: %d 个实验 x %d 轮" % (phase, len(rows), EPOCHS))
    log("=" * 72)
    t_phase = time.time()
    for i, (tag, fn, lr, desc) in enumerate(rows, 1):
        if complete(tag):
            log("[%d/%d] SKIP %s (已完成)" % (i, len(rows), tag))
            verify(tag)
            continue
        log("[%d/%d] === %s === lr=%.0e  %s" % (i, len(rows), tag, lr, desc))
        t0 = time.time()
        try:
            train_one(fn, tag, max_epochs=EPOCHS, patience=PATIENCE,
                      batch=8, lr=lr, root=ROOT_MAIN)
        except Exception as e:
            log("  FAILED %s: %s %s" % (tag, type(e).__name__, str(e)[:300]))
            continue
        verify(tag)
        el = (time.time() - t0) / 60.0
        eta = (time.time() - t_phase) / 60.0 / i * (len(rows) - i)
        log("  用时 %.1f 分钟; 本阶段 ETA 剩余约 %.1f 小时" % (el, eta / 60.0))
    log("阶段%s 完成, 用时 %.1f 小时" % (phase, (time.time() - t_phase) / 3600.0))


def post_steps():
    log("后置: run_post_fix.py (全量整图推理 + TOST + 修复归因 + 索引/事实)")
    py = sys.executable
    if os.name == "nt" and py.lower().endswith("pythonw.exe"):
        c = os.path.join(os.path.dirname(py), "python.exe")
        if os.path.exists(c):
            py = c
    out = os.path.join(paths.LOGS, "queue_fix2_post.log")
    with open(out, "a", encoding="utf-8") as fh:
        rc = subprocess.run([py, "-u", os.path.join(_HERE, "run_post_fix.py")],
                            cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
                            **_no_window_kwargs()).returncode
    log("run_post_fix exit=%d -> %s" % (rc, os.path.basename(out)))


if __name__ == "__main__":
    log("启动: 等待其他 run_queue* 结束 (轮询 %ds, 无窗口)" % POLL_INTERVAL)
    waited = 0
    while queues_running():
        log("其他队列运行中, 已等待 %d 分钟" % waited)
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL // 60

    log("数据根 %s" % ROOT_MAIN)
    run_rows(PHASE_A, "A(判别实验: H1 全分辨率融合 / H2 学习率)")

    lr, why = pick_base_lr()
    log("")
    log("阶段 B 的底: use_skip=True + pos_enc=True, lr=%.0e  [%s]" % (lr, why))
    rows_b = [(tag, (lambda kw=kw: mssact_light(use_skip=True, pos_enc=True, **kw)),
               lr, desc) for tag, kw, desc in ABLATIONS]
    run_rows(rows_b, "B(修复后消融)")

    post_steps()
    log("全部完成")
