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


def queues_running(marker="run_queue"):
    """是否有其他含 marker 的 python 进程在跑（排除自身 PID）"""
    my_pid = os.getpid()
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
            if pid.isdigit() and int(pid) != my_pid:
                return True
        return False
    except Exception:
        return True          # 检测失败时保守等待
