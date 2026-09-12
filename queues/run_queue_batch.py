"""batch 缩放对照队列（等主队列结束后执行）
目的: 验证 batch 8->16 + 线性缩放 LR 的精度/速度权衡
依据: Goyal et al. 2017 线性缩放规则 lr ∝ batch; 显存实测 batch=16 峰值 4.61GB(安全)
对照: lgF_bs8_full_lr2e4 (batch=8, lr=2e-4) vs lgF_bs16_full_lr4e4 (batch=16, lr=4e-4)
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
import paths
BASE = _BASE
CKPT = os.path.join(BASE, "checkpoints")
ROOT_MAIN = paths.DATA_NEWSPLIT2

def queues_running():
    my_pid = os.getpid()
    try:
        r = subprocess.run(["powershell","-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=25)
        for line in r.stdout.splitlines():
            if "run_queue" not in line: continue
            pid = line.split(",")[0].strip('"')
            if pid.isdigit() and int(pid) != my_pid:
                return True
        return False
    except Exception:
        return True

def state(tag, target):
    p = os.path.join(CKPT, f"{tag}_history.json")
    if not os.path.exists(p): return "absent"
    try:
        n = len(json.load(open(p)))
        return "complete" if n >= target else f"partial({n}/{target})"
    except Exception: return "corrupt"

if __name__ == "__main__":
    print("=== batch 缩放对照（等待主队列）===", flush=True)
    while queues_running():
        print("  其他队列运行中, 等待...", flush=True)
        time.sleep(300)
    print("开始", flush=True); time.sleep(30)

    from experiment_matrix import train_one
    from experiment_matrix_v2 import mssact_light

    JOBS = [
        ("lgF_bs8_full_lr2e4",  8, 2e-4, "对照: batch=8,  lr=2e-4"),
        ("lgF_bs16_full_lr4e4", 16, 4e-4, "实验: batch=16, lr=4e-4 (线性缩放)"),
    ]
    for tag, bs, lr, desc in JOBS:
        st = state(tag, 60)
        if st == "complete": print(f"SKIP {tag}", flush=True); continue
        print(f"=== {tag} ===\n    {desc}", flush=True)
        try:
            train_one(lambda: mssact_light(), tag, max_epochs=60, patience=20,
                      batch=bs, lr=lr, root=ROOT_MAIN)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
    print("=== batch 对照完成 ===", flush=True)
