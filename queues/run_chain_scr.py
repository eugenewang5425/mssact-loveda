# -*- coding: utf-8 -*-
"""
收尾链 (queues/run_chain_scr.py)

等待所有 run_queue* 结束后依次做两件事:
  1. 运行 DeepLabV3+ 从零训练对照（queues/run_queue_deeplab_scr.py）
  2. 清理仓库内被污染的 ./~ 目录（torch 在 HOME 缺失时把它当相对路径建在这里）

为什么用 Python 而不是 bash 轮询
--------------------------------
bash 里用 `$(powershell ...)` 轮询会为 powershell 申请**新的控制台**，
于是每轮询一次弹出一个终端窗口（已踩过，见 queue_guard.py）。
本脚本复用 queue_guard（CREATE_NO_WINDOW + SW_HIDE），全程无窗口。

以 pythonw 启动本脚本（GUI 子系统，无控制台）；子进程一律用 python.exe，
因为 pythonw **没有 stdout/stderr**，会让子进程日志被静默丢弃（也踩过）。

用法: pythonw queues/run_chain_scr.py
"""
import os
import sys
import time
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

from queue_guard import queues_running, POLL_INTERVAL, _no_window_kwargs   # noqa: E402

LOG = os.path.join(BASE, "chain_scr.log")
SCR_QUEUE = os.path.join(HERE, "run_queue_deeplab_scr.py")
POLLUTED_TILDE = os.path.join(BASE, "~")          # torch 在 HOME 缺失时建的字面 ~


def log(msg):
    line = f"[chain_scr {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _child_python():
    """子进程用 python.exe —— pythonw 会丢弃 stdout/stderr，日志无从保留"""
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def cleanup_tilde():
    """删除仓库内的 ./~ 目录（97MB 预训练权重曾被它带进 git 历史）

    仅在确认没有训练进程时才删——正在跑的 deeplab 训练若把 HOME 解析成字面 "~"，
    会从这里读权重；删早了会触发重新下载。
    """
    if not os.path.isdir(POLLUTED_TILDE):
        log("仓库内无 ./~ 目录，无需清理")
        return
    if queues_running():
        log("仍有队列在跑，暂不清理 ./~（避免触发重复下载）")
        return
    try:
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(POLLUTED_TILDE) for f in fs)
        shutil.rmtree(POLLUTED_TILDE, ignore_errors=True)
        if os.path.isdir(POLLUTED_TILDE):
            log(f"清理 ./~ 未完全成功（可能有句柄占用），请稍后手动删除")
        else:
            log(f"已清理仓库内 ./~ 目录（释放约 {size/1048576:.1f} MB）")
    except Exception as e:
        log(f"清理 ./~ 失败: {type(e).__name__} {e}")


def main():
    log(f"启动：等待其他 run_queue* 结束（轮询间隔 {POLL_INTERVAL}s，CREATE_NO_WINDOW）")
    waited = 0
    while queues_running():
        log(f"其他队列运行中，已等待 {waited} 分钟…")
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL // 60

    if os.path.exists(SCR_QUEUE):
        log("开始 DeepLab 从零训练对照 (run_queue_deeplab_scr.py --no-wait)")
        t0 = time.time()
        out_log = os.path.join(HERE, "run_queue_deeplab_scr.log")
        with open(out_log, "a", encoding="utf-8") as fh:
            rc = subprocess.run([_child_python(), "-u", SCR_QUEUE, "--no-wait"],
                                cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
                                **_no_window_kwargs()).returncode
        log(f"从零对照结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} 分钟 -> "
            f"{os.path.basename(out_log)}")
    else:
        log(f"未找到 {SCR_QUEUE}，跳过")

    cleanup_tilde()
    log("全部完成")


if __name__ == "__main__":
    main()
