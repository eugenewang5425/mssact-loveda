# -*- coding: utf-8 -*-
"""GPU 利用率采样统计（复现用户此前的分析方法）"""

# --- 路径引导（本文件位于子目录 verify/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import subprocess, time, json, statistics as st
N, IV = 100, 1.0   # 100 样本 × 1s
utils, powers = [], []
t0 = time.time()
for i in range(N):
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        utils.append(float(parts[0])); powers.append(float(parts[1]))
    except Exception:
        pass
    time.sleep(IV)
el = time.time() - t0
def pct(v, th): return sum(1 for x in v if x < th)/len(v)*100
res = dict(samples=len(utils), seconds=round(el,1),
           util_mean=round(st.mean(utils),1), util_median=round(st.median(utils),1),
           util_min=min(utils), util_max=max(utils),
           idle_lt10=round(pct(utils,10),1), idle_lt30=round(pct(utils,30),1),
           busy_gt90=round(pct(utils,90),1),
           power_mean=round(st.mean(powers),1), power_min=min(powers), power_max=max(powers))
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(os.path.join(_BASE, "artifacts", "gpu_sample_after.json"), "w"), indent=1)
# 时序字符图
chars = "".join("_" if u<10 else ("-" if u<30 else ("=" if u<60 else ("+" if u<90 else "#"))) for u in utils)
print("\n时序图 (每字符=1秒; _ <10%, - <30%, = <60%, + <90%, # >=90%):")
for i in range(0, len(chars), 100): print("  " + chars[i:i+100])
