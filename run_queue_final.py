"""
队列: 多种子噪声底线 + 全量推理严格性评估

背景 (文献要求)
--------------
此前所有比较都基于**单次运行** (seed=42)。文献明确指出:
  - GEO-Bench (NeurIPS 2023 D&B) 建议 >=10 seed, 并指出 3-5 seed 不足以给出可靠 CI
  - mmseg 实测同一 DeepLabV3+ 两次训练在 Cityscapes 上差 0.7 mIoU (双线性插值反向非确定)
  - "未测量种子方差" -> ΔKappa 0.006 的'无差异'结论只是断言, 不是测量结果
  - "无证据差异" vs "有证据无差异" 必须用 TOST 等效性检验区分

因此本队列补齐:
  A. 多种子: 完整模型 + DeepLabV3+ 各 3 个额外种子 (seed 7/2024/31337)
     -> 得到两次运行的 Kappa 标准差 = 噪声底线, 用于判定 ΔKappa 0.006 是否可分辨
  B. 全量推理: 对全部 22 个 tag 在 test_clean 上推理并缓存
     -> eval_rigor.py 可算逐类 IoU / 配对 bootstrap / TOST / 逐带 Kappa

用法: python run_queue_final.py
"""
import os, sys, json, time, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
import paths

CKPT = paths.CKPT
ROOT_MAIN = paths.DATA_NEWSPLIT2
SEEDS = [7, 2024, 31337]


def queues_running():
    """排除自身 PID, 避免自我死锁"""
    my_pid = os.getpid()
    try:
        r = subprocess.run(["powershell", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=30)
        for line in r.stdout.splitlines():
            if "run_queue" not in line:
                continue
            pid = line.split(",")[0].strip('"')
            if pid.isdigit() and int(pid) != my_pid:
                return True
        return False
    except Exception:
        return True


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
    print("=== 队列 final: 多种子 + 全量推理（等待其他队列）===", flush=True)
    while queues_running():
        print("  其他队列运行中, 等待 300s...", flush=True)
        time.sleep(300)
    print("开始执行", flush=True)
    time.sleep(30)

    from experiment_matrix import train_one, DeepLabV3Plus
    from experiment_matrix_v2 import mssact_light

    # ============ A. 多种子噪声底线 ============
    print("\n===== A. 多种子噪声底线 (3 seed × {完整模型, DeepLabV3+}) =====", flush=True)
    print("     协议与主实验完全一致 (60ep/pt20/bs8/lr2e-4, 回退管线)", flush=True)
    JOBS = []
    for s in SEEDS:
        JOBS.append((f"sd{s}_full", lambda: mssact_light(), s))
        JOBS.append((f"sd{s}_deeplab", lambda: DeepLabV3Plus(), s))
    for tag, fn, s in JOBS:
        st = state(tag, 60)
        if st == "complete":
            print(f"SKIP {tag} (complete)", flush=True); continue
        print(f"=== {tag} | seed={s} ===", flush=True)
        t0 = time.time()
        try:
            train_one(fn, tag, max_epochs=60, patience=20, batch=8, lr=2e-4,
                      root=ROOT_MAIN, seed=s)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify(tag)
        print(f"  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    # ============ B. 全量推理 + 严格性评估 ============
    print("\n===== B. 全量推理 (22 tag, test_clean 141 张) =====", flush=True)
    script = os.path.join(BASE, "eval_rigor.py")
    r = subprocess.run([sys.executable, "-u", script, "--infer"], cwd=BASE)
    print(f"eval_rigor exit={r.returncode}", flush=True)

    print("\n===== C. Kappa 归属分解 (逐带) =====", flush=True)
    r = subprocess.run([sys.executable, "-u", os.path.join(BASE, "headroom_analysis.py")],
                       cwd=BASE)
    print(f"headroom exit={r.returncode}", flush=True)

    # ============ D. 汇总产物 ============
    for step in ("build_index.py", "build_facts.py", "make_report_v2.py", "make_pdf.py"):
        print(f"\n===== D. {step} =====", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(BASE, step)], cwd=BASE)
        print(f"{step} exit={r.returncode}", flush=True)

    print("\n===== 队列 final 全部完成 =====", flush=True)
