# -*- coding: utf-8 -*-
"""
DeepLabV3+ 协议内从零基线 (queues/run_queue_deeplab_protocol.py)

为什么需要
----------
本项目**从未计划使用预训练**。所有模型的设定是"同协议、同数据、从零训练"。
但 `deeplabv3_resnet50(weights=None)` 仍会加载 ImageNet 预训练主干
（该工厂另有独立的 `weights_backbone`，默认即 IMAGENET1K_V1）——这是**代码 bug**，
不是实验设计。

更严重的是：该 bug 的产物 `nd_deeplab`（0.6876 val / 0.6495 test_clean）
一直作为**对比实验表里的头号基线**，并被写成"基线模型（同协议）"——
它根本不同协议。用 bug 产物立论是不可接受的。

本队列补齐**协议内**的从零 DeepLab 基线，使对比表真正同协议：
    nd_deeplab_scr      PNG 管线（与 nd_unet/fcn/pspnet/... 等基线一致），60ep
    lgR_b15_deeplab_scr memmap 管线（与 lgR_b15_* 预算攻击组一致），15ep

既有 `deeplab*`（预训练）结果**保留但降级为旁证**：仅用于量化"预训练贡献"，
不再作为基线，且在任何表中都必须显式标注"预训练主干"。

用法: python queues/run_queue_deeplab_protocol.py [--no-wait]
"""
import os, sys, json, time, subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import queue_guard
from queue_guard import POLL_INTERVAL

BASE = paths.REPO
CKPT = paths.CKPT
ROOT_MAIN = paths.DATA_NEWSPLIT2


def queues_running():
    from queue_guard import queues_running as _qr
    return _qr()


def state(tag, target):
    p = os.path.join(CKPT, f"{tag}_history.json")
    if not os.path.exists(p):
        return "absent"
    try:
        n = len(json.load(open(p)))
        return "complete" if n >= target else f"partial({n}/{target})"
    except Exception:
        return "corrupt"


def verify(tag):
    p = os.path.join(CKPT, f"{tag}_history.json")
    if not os.path.exists(p):
        print(f"  !!! {tag} 无 history", flush=True); return
    try:
        h = json.load(open(p))
        b = max(h, key=lambda r: r.get("kappa", -9))
        print(f"  OK {tag}: n={len(h)} best_k={b['kappa']:.4f}@ep{b['epoch']}", flush=True)
    except Exception as e:
        print(f"  !!! {tag} 损坏: {e}", flush=True)


if __name__ == "__main__":
    print("=== 队列: DeepLabV3+ 协议内从零基线（替换 bug 产物作为基线）===", flush=True)
    if "--no-wait" not in sys.argv:
        while queues_running():
            print(f"  其他队列运行中, 等待 {POLL_INTERVAL}s...", flush=True)
            time.sleep(POLL_INTERVAL)
    print("开始执行", flush=True)
    time.sleep(20)

    from experiment_matrix import train_one, DeepLabV3Plus
    scr = lambda: DeepLabV3Plus()          # 默认即 pretrained_backbone=False

    # (tag, 轮数, 是否 PNG 管线, 说明)
    JOBS = [
        ("nd_deeplab_scr", 60, True,
         "与 nd_* 基线同管线(PNG)/同协议，替代 nd_deeplab 作为对比表基线"),
        ("lgR_b15_deeplab_scr", 15, False,
         "与 lgR_b15_* 预算攻击组同管线(memmap)/同协议"),
    ]
    for tag, ep, use_png, desc in JOBS:
        st = state(tag, ep)
        if st == "complete":
            print(f"SKIP {tag} (complete)", flush=True); continue
        print(f"=== {tag} | {ep}轮 | {'PNG' if use_png else 'memmap'} 管线 ===\n"
              f"    {desc}", flush=True)
        t0 = time.time()
        try:
            train_one(scr, tag, max_epochs=ep, patience=20, batch=8, lr=2e-4,
                      root=ROOT_MAIN, fast_data=(not use_png))
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify(tag)
        print(f"  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    print("\n===== 汇总产物 =====", flush=True)
    for _sub, step in (("report", "build_index.py"), ("report", "build_facts.py"),
                       ("viz", "make_report_figures.py"),
                       ("report", "make_report_v3.py"), ("report", "make_pdf.py")):
        print(f"\n=== {_sub}/{step} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(BASE, _sub, step)],
                           cwd=BASE, **queue_guard._no_window_kwargs())
        print(f"{_sub}/{step} exit={r.returncode}", flush=True)

    print("\n===== DeepLab 协议内从零基线 全部完成 =====", flush=True)
