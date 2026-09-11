"""30.5M 原版 MSSACT 队列 (等待主队列结束后自动接力)
1. nd_mssact30m    : 30.5M 原版架构重训 (筛选数据 1768张, OneCycle-60, pt20)
2. nd_post30m      : 其后训练 (lr=5e-5, 30轮)
3. TTA 评估        : nd_mssact30m / nd_post30m (val + test_clean, 含/不含TTA)
"""
import os, sys, json, time, subprocess
import paths  # 集中路径配置 (环境变量/.env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = paths.REPO
CKPT = f"{BASE}/checkpoints"
ROOT_NEW = paths.DATA_NEWSPLIT2

def main_queue_running():
    """检查主队列(run_queue_new)是否还在跑"""
    try:
        r = subprocess.run(["powershell","-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object {$_.CommandLine -match 'run_queue_new'} | "
            "Measure-Object | Select-Object -ExpandProperty Count"], capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip()) > 0
    except Exception:
        return False

def tag_state(tag, target):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return "absent"
    try:
        h = json.load(open(p)); n = len(h) if isinstance(h, list) else 0
        return "complete" if n >= target else f"partial({n}/{target})"
    except Exception:
        return "corrupt"

def verify(tag):
    p = f"{CKPT}/{tag}_history.json"
    try:
        h = json.load(open(p)); b = max(h, key=lambda r: r.get("kappa",-9))
        print(f"OK {tag}: n={len(h)} best_k={b['kappa']:.4f}@ep{b['epoch']}", flush=True)
    except Exception as e:
        print(f"!!! {tag} verify failed: {e}", flush=True)

if __name__ == "__main__":
    print("=== 30.5M 原版队列 (等待主队列) ===", flush=True)
    while main_queue_running():
        print("  主队列运行中, 等待...", flush=True)
        time.sleep(600)
    print("主队列已结束, 开始 30M 实验", flush=True)
    time.sleep(60)

    from experiment_matrix import train_one
    from models.msscactnet import MSSACTNet

    def mssact_30m():
        return MSSACTNet(in_channels=3, num_classes=7, embed_dims=[64,128,256,512],
                         transformer_layers=4, transformer_heads=8)

    # 1) 30.5M 原版重训
    st = tag_state("nd_mssact30m", 60)
    if st == "complete":
        print("SKIP nd_mssact30m (complete)", flush=True)
    else:
        print(f"=== nd_mssact30m (30.54M, {st}) ===", flush=True)
        try:
            train_one(mssact_30m, "nd_mssact30m", max_epochs=60, patience=20, batch=8,
                      lr=2e-4, root=ROOT_NEW)
        except Exception as e:
            print(f"FAILED nd_mssact30m: {e}", flush=True)
        verify("nd_mssact30m")

    # 2) 后训练
    st = tag_state("nd_post30m", 30)
    if st == "complete":
        print("SKIP nd_post30m (complete)", flush=True)
    else:
        print(f"=== nd_post30m (后训练, {st}) ===", flush=True)
        try:
            train_one(mssact_30m, "nd_post30m", max_epochs=30, patience=12, batch=8,
                      lr=5e-5, root=ROOT_NEW, init_ckpt=f"{CKPT}/nd_mssact30m_best.pt")
        except Exception as e:
            print(f"FAILED nd_post30m: {e}", flush=True)
        verify("nd_post30m")

    # 3) TTA 评估
    print("=== TTA 评估 (30M) ===", flush=True)
    r = subprocess.run([sys.executable, "-u", "eval_tta_new.py", "nd_mssact30m,nd_post30m"],
                       cwd=BASE)
    print(f"eval_tta_new exit={r.returncode}", flush=True)
    print("=== 30.5M 队列全部完成 ===", flush=True)
