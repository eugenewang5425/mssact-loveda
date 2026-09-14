# -*- coding: utf-8 -*-
"""队列互斥守卫 —— 检测其他 run_queue* 进程时**不弹出控制台窗口**

背景（已踩过的坑）
----------------
本模块此前的实现直接 `subprocess.run(["powershell", ...])`。当调用方由
`nohup ... &` 从无控制台的会话启动时，powershell.exe 作为**控制台子系统**程序
会向 Windows 申请一个**新的控制台**，于是每轮询一次就弹出一个终端窗口
（实测每 300 秒一次，严重干扰使用者）。

修复：传 `creationflags=CREATE_NO_WINDOW`（0x08000000）——这会阻止系统为子进程
分配控制台，从而完全消除窗口。同时提高轮询间隔到 600 秒，降低无谓开销。

注意：不要改回 `shell=True` 或去掉 creationflags，否则窗口会再次弹出。
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
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
POLL_INTERVAL = 600          # 秒


def _no_window_kwargs():
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0       # SW_HIDE，双保险
        kw["startupinfo"] = si
    return kw


def queues_running(marker="run_queue", exclude_pids=()):
    """是否有其他含 marker 的 python 进程在跑（排除自身 PID 与 exclude_pids）

    exclude_pids 的用途: 队列 A 在结束时调用 run_post_fix, 而 run_post_fix 的互斥
    检查会把 A 自己（命令行含 "run_queue"）当成"还在训练"从而拒绝启动 —— 实测
    `run_post_fix exit=1`, 日志"仍有 run_queue* 在运行"。故调用方把自己的 PID 传进来
    排除掉, 保护逻辑本身不变（其他任何 run_queue 进程仍会拦住它）。
    """
    my_pid = os.getpid()
    excl = set(int(p) for p in exclude_pids) | {my_pid}
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=40, **_no_window_kwargs())
        for line in r.stdout.splitlines():
            if marker not in line:
                continue
            pid = line.split(",")[0].strip('"')
            if pid.isdigit() and int(pid) not in excl:
                return True
        return False
    except Exception:
        return True          # 检测失败时保守等待
