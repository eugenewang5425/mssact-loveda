# -*- coding: utf-8 -*-
"""无窗口分离启动器 —— 把长任务从当前会话彻底脱离

为什么需要
----------
1. 用 `nohup python ... &` 从无控制台会话启动时, 子进程若调用 powershell 等
   控制台子系统程序会申请新控制台, 于是**每隔一段时间弹出一个终端窗口**
   (实测每 300 秒一次, 严重干扰使用者)。用 CREATE_NO_WINDOW | DETACHED_PROCESS
   可同时做到: 不分配控制台 + 不随父进程退出而结束。
2. 必须用 python.exe 而非 pythonw.exe —— pythonw 没有 stdout/stderr, 子进程
   全部输出(含错误堆栈)会被静默丢弃 (实测: eval_rigor 在链里 exit=1 且日志中
   一行输出都没有, 无法定位)。

用法:
    python queues/launch_detached.py queues/run_queue_fix.py [日志文件名]
"""

import os
import sys
import subprocess

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, _BASE)
import paths                                                        # noqa: E402


def child_python():
    """子进程一律用 python.exe, 见模块 docstring 第 2 条"""
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def launch(script, log_name=None):
    script = os.path.abspath(script)
    if not os.path.exists(script):
        raise SystemExit("脚本不存在: %s" % script)
    log_name = log_name or (os.path.basename(script).replace(".py", "") + ".out.log")
    log_path = os.path.join(paths.LOGS, log_name)
    flags = 0
    if os.name == "nt":
        flags = CREATE_NO_WINDOW | DETACHED_PROCESS
    fh = open(log_path, "ab")
    p = subprocess.Popen(
        [child_python(), "-u", script],
        cwd=_BASE, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=flags, close_fds=True,
    )
    print("已启动 PID=%d" % p.pid)
    print("  脚本: %s" % script)
    print("  日志: %s" % log_path)
    return p.pid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    launch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
