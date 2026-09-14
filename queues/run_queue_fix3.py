# -*- coding: utf-8 -*-
"""修复后消融 (第三轮): 底 = pos_enc=True (fx_pos, 0.6716)

为什么换底
--------
第二轮在 `use_skip+pos_enc`（fx_all, 0.6330）上跑消融, 三个完成的结果显示**这个底
不稳定**, 不适合做消融:

| 已完成 | 结果 | 性质 |
|---|---|---|
| `lgR120_abl_no_emr` | 0.6600（+0.0271 vs 底, 5.4σ） | 移除 EMR 大幅**改善** |
| `lgR120_abl_no_ecsam` | 0.4657@ep19 后**发散** | 见下 |
| `lgR120_abl_no_fpn` | 未跑完即停 | — |

`lgR120_abl_no_ecsam` 的发散已确证为**数值失效**而非塌陷期: epoch 20 起验证预测冻结
（OA 精确恒定在 0.36989283961174824）, 训练 loss 变成精确的 `0.0`, best.pt 权重最大
绝对值达 **1.26e4**（爆炸; 无 NaN, 但是退化到常量输出）。

底自身不稳定 ⇒ 其上的消融摆动（+0.027 / 发散）主要反映**底的病态**, 而不是模块作用。

第三轮的底选 `fx_pos`（`pos_enc=True`, Kappa 0.6716）, 理由**在跑之前就写定**:
1. 它是**稳定**的（116 轮无异常, 曲线正常收敛于 best@96）;
2. 它是修复族里**唯一高于未修复基线**（0.6655）的配置;
3. 它**不含跳连** —— 而跳连的有害性已在两轮、四种配置下稳健复现
   （见下方"已证伪的假设"）。

**已证伪的两个假设**（第二轮判定实验, 各为单变量）
------------------------------------------------
H1「256² 全分辨率融合有害」→ 证伪: 去掉 256² 融合的 `fx_d2` = 0.6466, 与全融合的
   `fx_skip`(0.6505) 仅差 −0.0039（<1σ）, 距基线仍差 −0.0189（3.8σ）。
H2「学习率不再匹配」→ 证伪: 降 LR 到 1e-4 使 skip 从 0.6505 → 0.6395（−0.0110）,
   使全修从 0.6330 → 0.6170（−0.0160）——**更低的学习率更差**。

⇒ **稳健的事实: 含跳连的配置全部低于基线**（0.6170–0.6505 vs 0.6655）, 不含跳连的
`fx_pos` 反而高于基线。有害的成分是**跳连本身**, 与融合层级、学习率均无关。
故 D1/D2 的"修复"在本项目中被**实测否决**, 不再进入消融的底。

发散检测
-------
本队列每个实验结束后做**数值失效检测**（见 `check_divergence`）: 若某阶段 loss 变成
精确 0 或验证指标长时间精确冻结, 就显式标记为"发散", 而不是把那个 Kappa 当作
模块效应的读数。第二轮就是靠这一步才发现 `abl_no_ecsam` 是发散而非塌陷。

用法: python queues/run_queue_fix3.py
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
LOG = os.path.join(paths.LOGS, "queue_fix3.log")
EPOCHS, PATIENCE = 120, 20
BASE_TAG = "lgR120_fx_pos"          # 已验证稳定: Kappa 0.6716, best@96, 116 轮


def log(msg):
    line = "[fix3 %s] %s" % (time.strftime("%F %T"), msg)
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
    h = hist(tag)
    if not h:
        return False
    b = max(h, key=lambda r: r.get("kappa", -9))
    return len(h) >= EPOCHS or len(h) >= b["epoch"] + PATIENCE - 1


def check_divergence(tag):
    """数值失效检测: 训练 loss 精确为 0, 或验证指标长时间精确冻结。

    为什么必须显式检查: 发散后模型退化为常量输出, 其 Kappa/OA 会**精确恒定**。
    这种数字看起来像"收敛到一个较差的平台", 会被误读成模块效应 —— 第二轮的
    `lgR120_abl_no_ecsam` 正是如此（weight max |.| = 1.26e4, loss 恒为 0.0）。
    """
    h = hist(tag)
    if not h:
        return None
    zeros = [r for r in h if r.get("loss") == 0.0]
    # 连续 5 轮 OA 精确相同 -> 预测已冻结
    frozen = 0
    for i in range(1, len(h)):
        if h[i].get("oa") is not None and h[i].get("oa") == h[i - 1].get("oa"):
            frozen += 1
        else:
            frozen = 0
    if len(zeros) >= 3 or frozen >= 5:
        return "发散(loss=0 共 %d 轮, OA 最长冻结 %d 轮)" % (len(zeros), frozen)
    return "正常"


def verify(tag):
    h = hist(tag)
    if not h:
        log("  !!! %s 无 history" % tag)
        return None
    b = max(h, key=lambda r: r.get("kappa", -9))
    secs = [r["sec"] for r in h if r.get("sec")]
    div = check_divergence(tag)
    log("  OK %s: n=%d best_k=%.4f@ep%s%s  数值状态=%s" % (
        tag, len(h), b["kappa"], b["epoch"],
        "  %.1fs/轮" % (sum(secs) / len(secs)) if secs else "", div))
    if div != "正常":
        log("     !!! 该结果**不可**用作模块效应读数 —— 见 check_divergence 注释")
    return b["kappa"]


# ==================== 消融表 (底 = pos_enc=True) ====================
def _abl(**flags):
    return lambda: mssact_light(pos_enc=True, **flags)


ABLATIONS = [
    ("lgR120_pos_abl_no_emr", _abl(use_emr=False), "去掉 EMR (换标准残差块)"),
    ("lgR120_pos_abl_no_ecsam", _abl(use_ecsam=False), "去掉 ECSAM 坐标注意力"),
    ("lgR120_pos_abl_no_fpn", _abl(use_fpn=False), "去掉 FPN"),
    ("lgR120_pos_abl_no_trans", _abl(use_transformer=False), "去掉 Transformer"),
    ("lgR120_pos_abl_no_adapter", _abl(use_adapter=False), "去掉 Adapter-Scale"),
    ("lgR120_pos_abl_bilinear", _abl(upsample_mode="bilinear"), "转置卷积 -> 双线性"),
    ("lgR120_pos_abl_trans4l", _abl(transformer_layers=4), "Transformer 2 -> 4 层"),
    ("lgR120_pos_abl_trans6l", _abl(transformer_layers=6), "Transformer 2 -> 6 层"),
]


def run_rows(rows, phase):
    log("")
    log("=" * 72)
    log("阶段%s: %d 个实验 x %d 轮" % (phase, len(rows), EPOCHS))
    log("=" * 72)
    t_phase = time.time()
    for i, (tag, fn, desc) in enumerate(rows, 1):
        if complete(tag):
            log("[%d/%d] SKIP %s (已完成)" % (i, len(rows), tag))
            verify(tag)
            continue
        log("[%d/%d] === %s ===  %s" % (i, len(rows), tag, desc))
        t0 = time.time()
        try:
            train_one(fn, tag, max_epochs=EPOCHS, patience=PATIENCE,
                      batch=8, lr=2e-4, root=ROOT_MAIN)
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
    out = os.path.join(paths.LOGS, "queue_fix3_post.log")
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

    b = hist(BASE_TAG)
    if not b:
        log("!! 底 %s 无 history, 中止" % BASE_TAG)
        raise SystemExit(1)
    bk = max(r["kappa"] for r in b)
    log("底 = %s (pos_enc=True): K=%.4f, %d 轮; 数值状态=%s"
        % (BASE_TAG, bk, len(b), check_divergence(BASE_TAG)))
    log("对照: 未修复基线 lgR120_full = 0.6655")
    run_rows(ABLATIONS, "(修复后消融, 底=pos_enc)")
    post_steps()
    log("全部完成")
