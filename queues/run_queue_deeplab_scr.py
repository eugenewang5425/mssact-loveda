# -*- coding: utf-8 -*-
"""
DeepLabV3+ 从零训练对照 (queues/run_queue_deeplab_scr.py)

为什么要跑这个
--------------
发现并已确认一处**公平性缺陷**（2026-09-13）:
    torchvision 的 `deeplabv3_resnet50` 有**两个独立**权重参数 --
    `weights`（整体/分割头）与 `weights_backbone`（主干），
    后者的默认值是 `ResNet50_Weights.IMAGENET1K_V1`。
    因此只传 `weights=None` **并不会关闭主干预训练**。

实测（首个 BatchNorm 的 weight 均值，新初始化必为 1.0）:
    U-Net / PSPNet / FCN / SegFormerLite / FPN-Seg = 1.0000  -> 确为从零
    DeepLabV3Plus                                  = 0.2574  -> 加载了预训练

后果: 既有 `*deeplab*` 结果（nd_deeplab 0.6876/0.6495、lgR_n*_deeplab …）都是
**带 ImageNet 预训练主干**的，与自研模型的 +0.072 差距中，
**架构贡献与预训练贡献无法分离**。初稿归因于"低层跳连"属过度归因。

本队列的做法
------------
**全部走 memmap 管线**（与 lgR_bs8_full_lr2e4 / 阶梯各档同世代，直接可比）:
  lgR_deeplab_scr          主数据集 60ep  <- 对照 lgR_bs8_full_lr2e4 (0.6320, 从零)
  lgR_n{250,500,1000}_deeplab_scr  阶梯 30ep <- 对照 lgR_n*_full / _unet (从零)

同时，与**同管线同协议**的预训练版本配对，可直接分离预训练贡献:
    lgR_n1000_deeplab (预训练) vs lgR_n1000_deeplab_scr (从零)

用法: python queues/run_queue_deeplab_scr.py [--no-wait]
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
LADDER = paths.DATA_LADDER


def queues_running():
    """见 queue_guard.py：CREATE_NO_WINDOW，避免弹出控制台窗口"""
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
    print("=== 队列: DeepLabV3+ 从零训练对照（分离预训练贡献）===", flush=True)
    if "--no-wait" not in sys.argv:
        while queues_running():
            print(f"  其他队列运行中, 等待 {POLL_INTERVAL}s...", flush=True)
            time.sleep(POLL_INTERVAL)
    print("开始执行", flush=True)
    time.sleep(20)

    from experiment_matrix import train_one, DeepLabV3Plus

    # 从零训练（关闭主干预训练）
    scr = lambda: DeepLabV3Plus(pretrained_backbone=False)

    JOBS = [("lgR_deeplab_scr", 60, None)]
    for n in (250, 500, 1000):
        JOBS.append((f"lgR_n{n}_deeplab_scr", 30, n))

    for tag, ep, n in JOBS:
        st = state(tag, ep)
        if st == "complete":
            print(f"SKIP {tag} (complete)", flush=True); continue
        kw = {}
        if n is not None:
            fd = os.path.join(BASE, "fast_dataset", f"ladder_n{n}")
            if not os.path.isdir(fd):
                print(f"SKIP {tag}: 缺子集 memmap {fd}", flush=True); continue
            kw = dict(root=os.path.join(LADDER, f"n{n}"), val_root=ROOT_MAIN,
                      fast_dir=fd)
        else:
            kw = dict(root=ROOT_MAIN)
        print(f"=== {tag} | {ep}轮 | {'阶梯 n%d' % n if n else '主数据集'} ===", flush=True)
        t0 = time.time()
        try:
            train_one(scr, tag, max_epochs=ep, patience=ep if n is not None else 20,
                      batch=8, lr=2e-4, **kw)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify(tag)
        print(f"  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    print("\n===== 汇总产物 =====", flush=True)
    for _sub, step in (("report", "build_index.py"), ("report", "build_facts.py"),
                       ("report", "make_report_v2.py"), ("report", "make_pdf.py")):
        print(f"\n=== {_sub}/{step} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(BASE, _sub, step)],
                           cwd=BASE, **queue_guard._no_window_kwargs())
        print(f"{_sub}/{step} exit={r.returncode}", flush=True)

    print("\n===== DeepLab 从零对照 全部完成 =====", flush=True)
