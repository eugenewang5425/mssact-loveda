"""验证实验队列: 数据量 × 模型能力 × 训练策略 的交互
按用户要求, 试图【攻击】以下两个解释:
  (1) "预算充足说": 所有模型都跑满轮次才收敛, 所以模块差异被掩盖
      -> 实验A: 给不足预算(15轮), 看模块差异是否显现
  (2) "退火调度说": 收敛快慢由调好的 OneCycle 退火决定, 而非架构
      -> 实验B: 固定LR(不做退火)训练, 看"收敛快慢"差异是否消失

实验C: 数据量阶梯 (250/500/1000, 嵌套子集) 评价泛化能力/过拟合度/学习速率

统一: 与轻量主队列同协议 (OneCycle/pt20/lr2e-4/batch8, 除特殊说明)
"""
import os, sys, json, time, subprocess
import paths
import queue_guard  # 集中路径配置 (环境变量/.env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = paths.REPO
CKPT = f"{BASE}/checkpoints"
ROOT_MAIN = paths.DATA_NEWSPLIT2      # 1768 训练
LADDER = paths.DATA_LADDER        # 阶梯子集

def jobs_running():
    try:
        r = subprocess.run(["powershell","-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object {$_.CommandLine -match 'run_queue'} | "
            "Measure-Object | Select-Object -ExpandProperty Count"], capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip()) > 0
    except Exception:
        return False

def tag_state(tag, target):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return "absent"
    try:
        h = json.load(open(p)); n = len(h) if isinstance(h,list) else 0
        return "complete" if n >= target else f"partial({n}/{target})"
    except Exception:
        return "corrupt"

def verify(tag):
    try:
        h = json.load(open(f"{CKPT}/{tag}_history.json"))
        b = max(h, key=lambda r: r.get("kappa",-9))
        print(f"  OK {tag}: n={len(h)} best_k={b['kappa']:.4f}@ep{b['epoch']}", flush=True)
    except Exception as e:
        print(f"  !!! {tag} verify fail: {e}", flush=True)

if __name__ == "__main__":
    # 等待所有现有队列结束
    print("=== 验证实验队列 (等待现有队列) ===", flush=True)
    while jobs_running():
        print("  其他队列运行中, 等待...", flush=True)
        time.sleep(600)
    print("开始验证实验", flush=True); time.sleep(60)

    from experiment_matrix import train_one, UNet, FCN, DeepLabV3Plus
    from experiment_matrix_v2 import mssact_light

    MODELS = [("full",   lambda: mssact_light()),
              ("noecsam",lambda: mssact_light(use_ecsam=False)),
              ("noemr",  lambda: mssact_light(use_emr=False)),
              ("unet",   lambda: UNet()),
              ("deeplab",lambda: DeepLabV3Plus()),
              ("fcn",    lambda: FCN())]

    # ================= 实验A: 预算攻击 (15轮) =================
    print("\n===== 实验A: 预算攻击 (1768张, 仅15轮) =====", flush=True)
    for mtag, fn in MODELS:
        tag = f"lg_b15_{mtag}"
        st = tag_state(tag, 15)
        if st == "complete": print(f"SKIP {tag}", flush=True); continue
        print(f"=== {tag} ===", flush=True)
        try:
            train_one(fn, tag, max_epochs=15, patience=15, batch=8, lr=2e-4, root=ROOT_MAIN)
        except Exception as e:
            print(f"FAILED {tag}: {e}", flush=True)
        verify(tag)

    # ================= 实验B: 固定LR攻击 (不退火) =================
    print("\n===== 实验B: 固定LR (1768张, lr=1e-4恒定, 60轮) =====", flush=True)
    for mtag, fn in [("full", lambda: mssact_light()), ("unet", lambda: UNet()),
                     ("deeplab", lambda: DeepLabV3Plus())]:
        tag = f"lg_const_{mtag}"
        st = tag_state(tag, 60)
        if st == "complete": print(f"SKIP {tag}", flush=True); continue
        print(f"=== {tag} ===", flush=True)
        try:
            train_one(fn, tag, max_epochs=60, patience=60, batch=8, lr=1e-4,
                      sched_kind="constant", root=ROOT_MAIN)
        except Exception as e:
            print(f"FAILED {tag}: {e}", flush=True)
        verify(tag)

    # ================= 实验C: 数据量阶梯 (250/500/1000, 30轮) =================
    print("\n===== 实验C: 数据量阶梯 (30轮) =====", flush=True)
    for n in [250, 500, 1000]:
        root = f"{LADDER}/n{n}"
        if not os.path.isdir(f"{root}/train/images"):
            print(f"SKIP ladder n{n} (no data)", flush=True); continue
        for mtag, fn in [("full", lambda: mssact_light()),
                         ("noecsam", lambda: mssact_light(use_ecsam=False)),
                         ("unet", lambda: UNet()),
                         ("deeplab", lambda: DeepLabV3Plus())]:
            tag = f"lg_n{n}_{mtag}"
            st = tag_state(tag, 30)
            if st == "complete": print(f"SKIP {tag}", flush=True); continue
            print(f"=== {tag} (root={root}) ===", flush=True)
            try:
                # val 保持主划分(221张)以便跨档位可比
                train_one(fn, tag, max_epochs=30, patience=30, batch=8, lr=2e-4,
                          root=root, val_root=ROOT_MAIN)
            except Exception as e:
                print(f"FAILED {tag}: {e}", flush=True)
            verify(tag)

    print("\n===== 验证队列全部完成 =====", flush=True)
    # 完成后自动做评估
    r = subprocess.run([sys.executable, "-u", "eval_ladder.py"], cwd=BASE,
                       **queue_guard._no_window_kwargs())
    print(f"eval_ladder exit={r.returncode}", flush=True)
