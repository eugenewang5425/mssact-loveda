# -*- coding: utf-8 -*-
"""修复配置的种子复现队列（lgR120_fx_* 各补 2 个种子）

为什么只做这一件、不做审核建议的另两件
------------------------------------
逐句审核（docs/报告审核_20260915.md）提了 4 项需要新实验的待办，我按"是否改变
承重结论"逐条判过：

**做** —— Q-28：`lgR120_fx_skip` 的"跳连有害"是本报告最重的**否定性**结论
（它决定了 D1/D2 修复被否决、也决定 §8.1 的路线图）。目前它建立在
**单次运行**上：−0.0150，按更正后的 √2σ 口径 = 2.53σ（勉强可分辨）。
虽然方向由 4 个配置一致给出（fx_skip −0.0150、fx_all −0.0325、fx_d2 −0.0189、
fxskip_lr1e4 −0.0260，4/4 同号），但**没有一个配置有复现**。
补 2 个种子后：若效应稳定，2.53σ 的点估计会获得独立的量级确认；
若不稳定，则必须降为"工作假设"。**无论哪个结果都改变表述强度，故值得做。**

同时补 `lgR120_fx_pos`（位置编码）——它是 §5.6.15 那张"首次可解释消融表"的**底**，
其 +0.0061 目前也是单次。

**不做** —— Q-34（梯度累积对照分离裁剪尺度与 batch）：审核自己给了替代方案
（"或改写为工程取舍"）。该结论的用途是**工程决策**（选 384²），不是机制归因；
而对照要新增 3–4 个长训练（约 10 h）才能把"裁剪尺度 × batch × 调度"拆开，
且拆开后也不改变"384²/bs4 最优"这个可执行结论。**收益不抵代价**，改为在正文
明确写出混淆项。

**不做** —— Q-35（数据阶梯各档 σ）：各档 30 轮、样本量小，测 σ 需 3 档 × 3 种子
≈ 9 次短训练。但阶梯结论（"自研需 ≥1000 张才体现优势"）本身就是**弱证据链**，
正确处理是**停止用满数据档的 σ 去给它定级**（改为描述性呈现），而不是为一条弱
结论再买 9 次训练。

种子取值：7 / 2024（与既有 sd* 系列同值，便于横向对照）。
协议：120 轮上限 / pt20 / bs8 / lr2e-4（与 lgR120_* 组完全一致）。

用法: python queues/run_queue_fx_seeds.py
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
LOG = os.path.join(paths.LOGS, "queue_fx_seeds.log")
EPOCHS, PATIENCE = 120, 20
SEEDS = [7, 2024]

ROWS = [
    ("lgR120_fx_skip_s7",    lambda: mssact_light(use_skip=True), 7),
    ("lgR120_fx_skip_s2024", lambda: mssact_light(use_skip=True), 2024),
    ("lgR120_fx_pos_s7",     lambda: mssact_light(pos_enc=True), 7),
    ("lgR120_fx_pos_s2024",  lambda: mssact_light(pos_enc=True), 2024),
]


def log(msg):
    line = "[fxseeds %s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def verify(tag):
    h = hist(tag)
    if not h:
        log("  !!! %s 无 history" % tag); return None
    b = max(h, key=lambda r: r.get("kappa", -9))
    # 发散检测（与 run_queue_fix3 同规则）
    z = sum(1 for r in h if r.get("loss") == 0.0)
    div = "**发散(loss=0 共%d轮)**" % z if z >= 3 else "正常"
    log("  OK %s: n=%d best_k=%.4f@ep%s  数值状态=%s" % (
        tag, len(h), b["kappa"], b["epoch"], div))
    return b["kappa"]


def run_rows():
    log("=" * 72)
    log("修复配置种子复现: %d 个实验 x %d 轮" % (len(ROWS), EPOCHS))
    log("=" * 72)
    t0 = time.time()
    for i, (tag, fn, seed) in enumerate(ROWS, 1):
        if complete(tag):
            log("[%d/%d] SKIP %s (已完成)" % (i, len(ROWS), tag)); verify(tag); continue
        log("[%d/%d] === %s === seed=%d" % (i, len(ROWS), tag, seed))
        ts = time.time()
        try:
            train_one(fn, tag, max_epochs=EPOCHS, patience=PATIENCE,
                      batch=8, lr=2e-4, root=ROOT_MAIN, seed=seed)
        except Exception as e:
            log("  FAILED %s: %s %s" % (tag, type(e).__name__, str(e)[:300])); continue
        verify(tag)
        eta = (time.time() - t0) / 60.0 / i * (len(ROWS) - i)
        log("  用时 %.1f 分钟; ETA 剩余约 %.1f 小时" % ((time.time() - ts) / 60.0, eta / 60.0))


def post_steps():
    log("后置: run_post_fix.py (新 tag 整图推理 + seed_variance + 索引/事实)")
    py = sys.executable
    if os.name == "nt" and py.lower().endswith("pythonw.exe"):
        c = os.path.join(os.path.dirname(py), "python.exe")
        if os.path.exists(c):
            py = c
    out = os.path.join(paths.LOGS, "queue_fx_seeds_post.log")
    with open(out, "a", encoding="utf-8") as fh:
        rc = subprocess.run([py, "-u", os.path.join(_HERE, "run_post_fix.py"),
                             "--parent-pid", str(os.getpid())],
                            cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
                            **_no_window_kwargs()).returncode
    log("run_post_fix exit=%d -> %s" % (rc, os.path.basename(out)))


if __name__ == "__main__":
    log("启动: 等待其他 run_queue* 结束 (轮询 %ds)" % POLL_INTERVAL)
    waited = 0
    while queues_running():
        log("其他队列运行中, 已等待 %d 分钟" % waited)
        time.sleep(POLL_INTERVAL); waited += POLL_INTERVAL // 60
    run_rows()
    post_steps()
    log("全部完成")
