"""等待GPU空闲后自动运行 abl120_* 系列 (120轮上限, patience=20, 与v3_s256主协议一致)
另存命名: abl120_*, 不覆盖 60 轮结果
"""
import subprocess, time, os, sys
import os
import paths  # 集中路径配置 (环境变量/.env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CKPT = paths.CKPT

def gpu_free():
    r = subprocess.run(["powershell","-Command","(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count"],
                       capture_output=True, text=True)
    try: return int(r.stdout.strip()) == 0
    except: return False

print("waiting for current pipeline to finish...", flush=True)
while True:
    if gpu_free():
        time.sleep(300)
        if gpu_free():
            print("GPU free, starting abl120 series", flush=True)
            break
    time.sleep(300)

from experiment_matrix import train_one
from experiment_matrix_v2 import mssact_light

ablations = {
    "abl120_no_emr":     dict(use_emr=False),
    "abl120_no_ecsam":   dict(use_ecsam=False),
    "abl120_no_fpn":     dict(use_fpn=False),
    "abl120_no_trans":   dict(use_transformer=False),
    "abl120_no_adapter": dict(use_adapter=False),
}
for tag, flags in ablations.items():
    if os.path.exists(f"{CKPT}/{tag}_history.json"):
        print(f"SKIP {tag}", flush=True); continue
    print(f"=== RUN {tag} ===", flush=True)
    try:
        train_one(lambda: mssact_light(**flags), tag, max_epochs=120, patience=20, batch=8, lr=2e-4)
    except Exception as e:
        print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
print("abl120 series DONE", flush=True)
