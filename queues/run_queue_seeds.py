# -*- coding: utf-8 -*-
"""种子扩展队列 —— 把 σ_seed 从 n=4 扩到 n≥10（两个评估协议同时收紧）

依据
----
GEO-Bench (NeurIPS 2023 D&B) 建议 ≥10 seed; 3–5 seed 不足以给出可靠 CI。
当前 n=4（seed 42/7/2024/31337），且 2026-09-15 已实测 **σ_seed 依评估协议而变**
（val 0.0050 / 整图 0.0041，见 analysis/seed_variance.py 的 sigma_seed_test_protocol）。
两个口径都由**同一批 checkpoint** 导出（训练一次 → val 与整图各评一次），
故扩种子同时收紧两者。

设计
----
新种子 6 个: 1234 / 5555 / 8888 / 31415 / 27182 / 9999（确定值, 便于复现）
  × 完整模型 (light)   → full 组 n=4+6=10
  × DeepLabV3+         → deeplab 组 n=3+6=9
协议 = G2 (60ep/pt20/bs8/lr2e-4, P-MEM-ROLL), 与既有 sd* 完全一致。

先跑 analysis/protocol_dissection.py（协议解剖, 已单独跑过一次则幂等重跑），
再训练, 后置: 新 12 个 tag 整图推理 → seed_variance(双口径 σ) → TOST(σ 更新)
→ 索引/事实。

用法: python queues/run_queue_seeds.py
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
LOG = os.path.join(paths.LOGS, "queue_seeds.log")
EPOCHS, PATIENCE = 60, 20
NEW_SEEDS = [1234, 5555, 8888, 31415, 27182, 9999]


def log(msg):
    line = "[seeds %s] %s" % (time.strftime("%F %T"), msg)
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


def verify(tag):
    h = hist(tag)
    if not h:
        log("  !!! %s 无 history" % tag)
        return None
    b = max(h, key=lambda r: r.get("kappa", -9))
    log("  OK %s: n=%d best_k=%.4f@ep%s" % (tag, len(h), b["kappa"], b["epoch"]))
    return b["kappa"]


def run_rows(rows, phase):
    log("")
    log("=" * 72)
    log("阶段%s: %d 个实验 x %d 轮" % (phase, len(rows), EPOCHS))
    log("=" * 72)
    t_phase = time.time()
    for i, (tag, fn, seed) in enumerate(rows, 1):
        if complete(tag):
            log("[%d/%d] SKIP %s (已完成)" % (i, len(rows), tag))
            verify(tag)
            continue
        log("[%d/%d] === %s === seed=%d" % (i, len(rows), tag, seed))
        t0 = time.time()
        try:
            train_one(fn, tag, max_epochs=EPOCHS, patience=PATIENCE,
                      batch=8, lr=2e-4, root=ROOT_MAIN, seed=seed)
        except Exception as e:
            log("  FAILED %s: %s %s" % (tag, type(e).__name__, str(e)[:300]))
            continue
        verify(tag)
        el = (time.time() - t0) / 60.0
        eta = (time.time() - t_phase) / 60.0 / i * (len(rows) - i)
        log("  用时 %.1f 分钟; 本阶段 ETA 剩余约 %.1f 小时" % (el, eta / 60.0))
    log("阶段%s 完成, 用时 %.1f 小时" % (phase, (time.time() - t_phase) / 3600.0))


def post_steps():
    log("后置: run_post_fix.py (新 tag 整图推理 + TOST + seed_variance + 索引/事实)")
    py = sys.executable
    if os.name == "nt" and py.lower().endswith("pythonw.exe"):
        c = os.path.join(os.path.dirname(py), "python.exe")
        if os.path.exists(c):
            py = c
    out = os.path.join(paths.LOGS, "queue_seeds_post.log")
    with open(out, "a", encoding="utf-8") as fh:
        rc = subprocess.run([py, "-u", os.path.join(_HERE, "run_post_fix.py"),
                             "--parent-pid", str(os.getpid())],
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

    # 0) 协议解剖（推理级, ~10 分钟; 幂等——每次重跑都全量重算, 结果可直接覆盖）
    log("阶段0: 协议解剖 protocol_dissection")
    out0 = os.path.join(paths.LOGS, "queue_seeds_dissection.log")
    with open(out0, "a", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "-u",
                             os.path.join(_BASE, "analysis", "protocol_dissection.py")],
                            cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
                            **_no_window_kwargs()).returncode
    log("protocol_dissection exit=%d -> %s" % (rc, os.path.basename(out0)))

    # 1) 完整模型 × 6 新种子
    rows_full = [("sd%d_full" % s, (lambda: mssact_light()), s) for s in NEW_SEEDS]
    run_rows(rows_full, "一(完整模型种子扩展: n=4→10)")

    # 2) DeepLabV3+ × 6 新种子
    from experiment_matrix import DeepLabV3Plus
    rows_dl = [("sd%d_deeplab" % s, (lambda: DeepLabV3Plus()), s) for s in NEW_SEEDS]
    run_rows(rows_dl, "二(DeepLab 种子扩展: n=3→9)")

    post_steps()
    log("全部完成")
