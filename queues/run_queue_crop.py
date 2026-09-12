"""
队列: 训练策略对比（裁剪尺度 + 随机 vs 固定）—— 在**最终数据（阶段 C）**上

动机
----
用户早期明确要求："我们不同的训练策略要进行对比，确保我们训练手法的科学性，
比如随机裁剪、不同固定尺寸，你也别太随机，过小的尺度完全没有任何训练意义"。

该对比此前**只在阶段 A（957 张）**上做过（`v3_s256/s512/s1024/s128gf`），
而阶段 A 已被阶段 C（1,768 张 + REMAP 修复）取代。本队列在**结论依据数据**上补齐。

协议（与主实验一致，仅裁剪参数不同）
--------------------------------
60 ep / patience=20 / batch=8 / lr=2e-4 / OneCycle / EMA(0.999) / 类别加权CE
数据 newsplit2（train 1768 / val 221），回退管线（memmap + 原增强 + workers=0）

对比设计
--------
| tag            | crop | 位置       | 每样本像素（相对 256²） |
|----------------|------|-----------|------------------------|
| lgR_sd_c256_rand | 256 | 随机(已有多条同配置对照) | 1×  |
| lgR_c256_center  | 256 | 固定中心   | 1×  |
| lgR_c384_rand    | 384 | 随机       | 2.25× |
| lgR_c512_rand    | 512 | 随机       | 4×  |

判读
----
- 256 随机 vs 256 中心：隔离"随机裁剪"这一增强本身的贡献（同尺度、同像素量）
- 256 / 384 / 512 随机：尺度效应；注意 crop 越大每样本像素越多、每轮越慢，
  **故不能只看 Kappa，须同时报告每轮耗时与总时长**（尺度的计算代价）
- 用户明确排除过小尺度（如 128），故不设该档

用法: python run_queue_crop.py
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
import os, sys, json, time, subprocess

sys.path.insert(0, _HERE)
BASE = _BASE
import paths
import queue_guard

CKPT = paths.CKPT
ROOT_MAIN = paths.DATA_NEWSPLIT2
JOBS = [
    ("lgR_c256_center", dict(crop=256, center_crop=True),
     "256 固定中心（隔离随机裁剪的贡献）"),
    ("lgR_c384_rand",   dict(crop=384), "384 随机（2.25× 像素）"),
    ("lgR_c512_rand",   dict(crop=512), "512 随机（4× 像素）"),
]


def queues_running():
    """是否有其他 run_queue* 进程在跑（见 queue_guard.py）

    必须走 queue_guard：它用 CREATE_NO_WINDOW 抑制控制台分配。
    此前直接 subprocess.run(["powershell", ...])，而本脚本由 nohup 从无控制台
    会话启动，powershell 会申请**新的控制台**，于是每轮询一次弹出一个终端窗口。
    """
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
    print("=== 队列: 训练策略对比（裁剪尺度 + 随机vs固定, 阶段 C 数据）===", flush=True)
    if "--no-wait" not in sys.argv:
        while queues_running():
            print("  其他队列运行中, 等待 300s...", flush=True)
            time.sleep(600)
    print("开始执行", flush=True)
    time.sleep(30)

    from experiment_matrix import train_one
    from experiment_matrix_v2 import mssact_light
    from models.msscactnet import MSSACTNet
    import torch

    for tag, kw, desc in JOBS:
        st = state(tag, 60)
        if st == "complete":
            print(f"SKIP {tag} (complete)", flush=True); continue
        print(f"=== {tag} ===\n    {desc}", flush=True)
        t0 = time.time()
        # batch 需按裁剪面积下调, 否则显存溢出 (256²: 8; 384²: 4; 512²: 2)
        c = kw.get("crop", 256)
        bs = 8 if c <= 288 else (4 if c <= 420 else 2)
        try:
            train_one(lambda: mssact_light(), tag, max_epochs=60, patience=20,
                      batch=bs, lr=2e-4, root=ROOT_MAIN, **kw)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify(tag)
        print(f"  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    print("\n===== 汇总产物 =====", flush=True)
    for _sub, step in (("analysis", "seed_variance.py"), ("report", "build_index.py"),
                       ("report", "build_facts.py"), ("report", "make_report_v2.py"),
                       ("report", "make_pdf.py")):
        print(f"\n=== {step} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(BASE, _sub, step)], cwd=BASE,
                           **queue_guard._no_window_kwargs())
        print(f"{_sub}/{step} exit={r.returncode}", flush=True)

    print("\n===== 队列 crop 全部完成 =====", flush=True)
