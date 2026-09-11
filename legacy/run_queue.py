"""最终 run_queue（经多方核验后编写）
顺序:
  PHASE-1 过拟合诊断 (overfit_diag.py, 已修复构造bug)   ~15min
  PHASE-2 abl120_* 5个消融 ×120轮/pt20 (收敛轮次研究)  ~15h
  PHASE-3 策略探索 strat_* 11个 (坍缩救援/调度分离/30M full验证)  ~20h
每任务: 开始前 tag 唯一性断言 + 结束后 history JSON 完整性断言; 全部新文件名
"""
import os, sys, json, time, subprocess
import paths  # 集中路径配置 (环境变量/.env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CKPT = paths.CKPT
BASE = paths.REPO

def tag_state(tag, target_epochs):
    """complete 判定: 队列日志已出现该任务 DONE 行 (早停/跑满都会打印); 否则按 history 轮数"""
    QLOG_FILES = [
        "C:/Users/Administrator/.zcode/cli/exec/sess_79382b4e-b5fc-48a3-9f7f-73768fa1f2f2/call_00_ET_Gfj9wsuTCGbGVS5Ecka75306-stdout.log",
        "C:/Users/Administrator/.zcode/cli/exec/sess_79382b4e-b5fc-48a3-9f7f-73768fa1f2f2/call_00_ET_9VgeDSos6jbcb3bzHDUE5056-stdout.log",
    ]
    for lp in QLOG_FILES:
        if os.path.exists(lp):
            try:
                logtxt = open(lp, encoding="utf-8", errors="ignore").read()
                if ("=== " + tag + " DONE") in logtxt:
                    return "complete"
            except Exception:
                pass
    hp = CKPT + "/" + tag + "_history.json"
    if not os.path.exists(hp): return "absent"
    try:
        h = json.load(open(hp))
        n = len(h) if isinstance(h, list) else 0
        if n >= target_epochs: return "complete"
        return "partial(%d/%d)" % (n, target_epochs)
    except Exception:
        return "corrupt-retry"

def gpu_free():
    # 排除自身 + 监听器(monitor_queue), 检查是否有其他 python 工作进程
    my_pid = os.getpid()
    try:
        r = subprocess.run(["powershell","-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object {$_.ProcessId -ne %d -and $_.CommandLine -notmatch 'monitor_queue'} | "
            "Measure-Object | Select-Object -ExpandProperty Count" % my_pid],
            capture_output=True, text=True)
        return int(r.stdout.strip()) == 0
    except Exception:
        return True

def wait_gpu():
    print("waiting for GPU/pipeline idle...", flush=True)
    while not gpu_free():
        time.sleep(300)
    time.sleep(60)
    print("GPU idle, proceeding", flush=True)

def verify_history(tag):
    """训练后完整性核验: history JSON 可解析且非空, 打印最优"""
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p):
        print(f"!!! {tag} NO history after run", flush=True); return False
    try:
        h = json.load(open(p))
        if not isinstance(h, list) or not h:
            print(f"!!! {tag} EMPTY history", flush=True); return False
        b = max(h, key=lambda r: r.get("kappa", -9))
        print(f"OK {tag}: n={len(h)} best_k={b['kappa']:.4f}@ep{b['epoch']}", flush=True)
        return True
    except Exception as e:
        print(f"!!! {tag} CORRUPT history: {e}", flush=True); return False

if __name__ == "__main__":
    from experiment_matrix import train_one
    from experiment_matrix import UNet, DeepLabV3Plus
    from experiment_matrix_v2 import mssact_light
    from models.msscactnet import MSSACTNet

    # ========== PHASE-1 过拟合诊断 ==========
    wait_gpu()
    print("\n===== PHASE-1 overfit diagnostic =====\n", flush=True)
    if os.path.exists(f"{BASE}/overfit_diag/overfit_summary.json"):
        print("SKIP overfit (done)", flush=True)
    else:
        r = subprocess.run([sys.executable, "-u", "overfit_diag.py"], cwd=BASE)
        print(f"overfit_diag exit={r.returncode}", flush=True)

    # ========== PHASE-2 消融120轮收敛 ==========
    wait_gpu()
    print("\n===== PHASE-2 ablations to convergence (120ep/pt20) =====\n", flush=True)
    abl120 = {
        "abl120_no_emr":     dict(use_emr=False),
        "abl120_no_ecsam":   dict(use_ecsam=False),
        "abl120_no_fpn":     dict(use_fpn=False),
        "abl120_no_trans":   dict(use_transformer=False),
        "abl120_no_adapter": dict(use_adapter=False),
    }
    for tag, flags in abl120.items():
        st = tag_state(tag, 120)
        if st == "complete":
            print(f"SKIP {tag} (complete)", flush=True); continue
        if st.startswith("partial") or st == "corrupt-retry":
            print(f"RERUN {tag} ({st})", flush=True)
        print(f"=== {tag} ===", flush=True)
        try:
            train_one(lambda f=flags: mssact_light(**f), tag, max_epochs=120, patience=20, batch=8, lr=2e-4)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify_history(tag)

    # ========== PHASE-3 策略探索 ==========
    wait_gpu()
    print("\n===== PHASE-3 strategy exploration =====\n", flush=True)
    strat = [
        ("strat_no_emr_oc30",        lambda: mssact_light(use_emr=False),           30, 30, 8, 2e-4, "onecycle"),
        ("strat_bilinear_oc30",      lambda: mssact_light(upsample_mode="bilinear"), 30, 30, 8, 2e-4, "onecycle"),
        ("strat_no_emr_oc60_lr1e4",  lambda: mssact_light(use_emr=False),           60, 60, 8, 1e-4, "onecycle"),
        ("strat_bilinear_oc60_lr1e4",lambda: mssact_light(upsample_mode="bilinear"),60, 60, 8, 1e-4, "onecycle"),
        ("strat_unet_oc12_lr2e4",    lambda: UNet(),                                12, 12, 8, 2e-4, "onecycle"),
        ("strat_unet_oc12_lr1e3",    lambda: UNet(),                                12, 12, 8, 1e-3, "onecycle"),
        ("strat_mssact_oc30",        lambda: mssact_light(),                        30, 30, 8, 2e-4, "onecycle"),
        ("strat_mssact_oc60_lr5e4",  lambda: mssact_light(),                        60, 60, 8, 5e-4, "onecycle"),
        ("strat_mssact_sgdr120",     lambda: mssact_light(),                        120, 60, 8, 2e-4, "sgdr30"),
        ("strat_deeplab_oc30",       lambda: DeepLabV3Plus(),                       30, 30, 8, 2e-4, "onecycle"),
        ("strat_mssact_full30m_oc60", lambda: MSSACTNet(in_channels=3, num_classes=7,
               embed_dims=[64,128,256,512], transformer_layers=4, transformer_heads=8),
                                                                                     60, 60, 8, 2e-4, "onecycle"),
    ]
    for tag, fn, ep, pt, b, lr, sk in strat:
        st = tag_state(tag, ep)
        if st == "complete":
            print(f"SKIP {tag} (complete)", flush=True); continue
        if st.startswith("partial") or st == "corrupt-retry":
            print(f"RERUN {tag} ({st})", flush=True)
        print(f"=== {tag} (ep{ep} pt{pt} lr{lr} {sk}) ===", flush=True)
        try:
            train_one(fn, tag, max_epochs=ep, patience=pt, batch=b, lr=lr, sched_kind=sk)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify_history(tag)

    print("\n===== QUEUE COMPLETE =====", flush=True)
