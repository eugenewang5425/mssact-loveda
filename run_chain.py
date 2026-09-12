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
import os
import sys
import time
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from queue_guard import queues_running, POLL_INTERVAL, _no_window_kwargs   # noqa: E402

LOG = os.path.join(BASE, "chain.log")
STEPS = ["run_queue_final.py", "run_queue_crop.py"]


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
        out_log = os.path.join(BASE, f"{step.replace('.py','')}.log")
        with open(out_log, "a", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-u", path, "--no-wait"],
                                cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
                                **_no_window_kwargs()).returncode
        log(f"{step} 结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} 分钟 -> {os.path.basename(out_log)}")

    log("全部完成")


if __name__ == "__main__":
    main()
