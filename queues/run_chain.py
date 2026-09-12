# -*- coding: utf-8 -*-
"""
统一等待链 (run_chain.py) —— 串行执行后置队列与裁剪队列

为什么需要它
-----------
此前是两个彼此独立的"守卫"各自轮询进程列表：
  * run_queue_final.py 的等待循环
  * _watch_and_run_crop.sh（bash）
两者都调用 `powershell` 做进程检测。当它们由 `nohup ... &` 从**无控制台**的会话
启动时，powershell.exe（控制台子系统）会向 Windows 申请**新的控制台**，于是
每轮询一次就弹出一个终端窗口（实测每 300 秒一次，严重干扰使用者）。

本脚本改为**单一等待点**：
  1. 用 `queue_guard.queues_running()`（带 CREATE_NO_WINDOW，无窗口）等待
  2. 依次以 `--no-wait` 运行 run_queue_final.py、run_queue_crop.py
     （它们内部的等待循环因此不会再次执行）
  3. 所有子进程调用均带 CREATE_NO_WINDOW

这样全流程只有一处轮询，且完全不可见。

用法: python run_chain.py         # 直接运行（会先等待其他队列结束）
"""

# --- 路径引导（本文件位于子目录 queues/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os
import sys
import time
import subprocess

BASE = _BASE
sys.path.insert(0, BASE)

from queue_guard import queues_running, POLL_INTERVAL, _no_window_kwargs   # noqa: E402

LOG = os.path.join(BASE, "chain.log")
STEPS = ["queues/run_queue_final.py", "queues/run_queue_crop.py"]


def _child_python():
    """子进程一律用 python.exe，而非 pythonw.exe

    原因: 本脚本通常以 pythonw.exe 启动（GUI 子系统，无控制台，避免弹窗）。
    但 pythonw **没有 stdout/stderr**——若子进程继承 sys.executable，则其全部
    输出（含错误堆栈）会被静默丢弃。实测后果: eval_rigor 在链里 exit=1 且
    日志中一行输出都没有，无法定位。
    改用同目录的 python.exe，并配合 CREATE_NO_WINDOW（见 queue_guard）与
    显式 stdout 重定向，即可既无窗口、又有完整日志。
    """
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def log(msg):
    line = f"[chain {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def main():
    log(f"启动：等待其他 run_queue* 结束（轮询间隔 {POLL_INTERVAL}s，CREATE_NO_WINDOW）")
    waited = 0
    while queues_running():
        log(f"其他队列运行中，已等待 {waited} 分钟…")
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL // 60

    for step in STEPS:
        path = os.path.join(BASE, step)
        if not os.path.exists(path):
            log(f"跳过（不存在）: {step}")
            continue
        log(f"开始 {step} --no-wait")
        t0 = time.time()
        # 输出重定向到各自日志，避免污染 chain.log
        out_log = os.path.join(_HERE, f"{os.path.basename(step).replace('.py','')}.log")
        with open(out_log, "a", encoding="utf-8") as fh:
            rc = subprocess.run([_child_python(), "-u", path, "--no-wait"],
                                cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
                                **_no_window_kwargs()).returncode
        log(f"{step} 结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} 分钟 -> {os.path.basename(out_log)}")

    log("全部完成")


if __name__ == "__main__":
    main()
