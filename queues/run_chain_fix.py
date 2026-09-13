# -*- coding: utf-8 -*-
"""修复实验的等待链 —— 等 run_queue_fix.py 结束后自动跑评测后置链

为什么需要它（而不是让队列自己跑后置）
------------------------------------
`run_queue_fix.py` 在 21:16 启动, 那时它的 `post_steps()` 还是**旧版本**
（只对 12 个新 tag 推理 + build_index + build_facts）。Python 在启动时就把模块读进
内存, 之后再改文件对**正在运行的进程无效** —— 所以本次运行结束时**不会**自动跑
`run_post_fix.py`（全量 44 tag 推理 + TOST + 修复归因 + 兼容性回归）。

为什么不直接重启队列: `lgR120_full` 已完成（111 轮, 早停于 best@91）。重启会让
`tag_state` 判定为 `partial(111/120)` 而**从头重训**, 白丢 1.5 小时。
故沿用本仓库既有的"单一等待点"模式（见 run_chain.py 的 docstring）:
一个独立的 watcher 等队列结束后再跑后置链。

注意: 本文件名不含 "run_queue", 故不会被队列自身的互斥检测看到（否则会互相等待死锁）。

用法: python queues/run_chain_fix.py
"""

import os
import sys
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths
from queue_guard import queues_running, POLL_INTERVAL, _no_window_kwargs   # noqa: E402

LOG = os.path.join(paths.LOGS, "chain_fix.log")


def log(msg):
    line = "[chain-fix %s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def child_python():
    """子进程一律用 python.exe（pythonw 会丢 stdout/stderr）"""
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def main():
    log("启动: 等待 run_queue* 结束（轮询 %ds, 无窗口）" % POLL_INTERVAL)
    waited = 0
    while queues_running():
        log("队列仍在运行, 已等待 %d 分钟" % waited)
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL // 60

    log("队列已结束; 开始评测后置链")
    script = os.path.join(_HERE, "run_post_fix.py")
    out = os.path.join(paths.LOGS, "chain_fix_post.log")
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        rc = subprocess.run([child_python(), "-u", script], cwd=_BASE,
                            stdout=fh, stderr=subprocess.STDOUT,
                            **_no_window_kwargs()).returncode
    log("run_post_fix exit=%d 用时 %.1f 分钟 -> %s"
        % (rc, (time.time() - t0) / 60, os.path.basename(out)))
    log("全部完成" if rc == 0 else "!! 后置链失败, 请查看上述日志")


if __name__ == "__main__":
    main()
