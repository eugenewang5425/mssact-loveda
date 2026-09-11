"""进度监听器 v3: 动态刷新 + 每任务完整指标(轮次/OA/Kappa/mF1) + 当前活动高亮
用法: python monitor_queue.py [--interval 15]
纯监控只读, 不启动 GPU 工作
"""
import os, sys, json, time, re, subprocess, datetime, glob
import paths  # 集中路径配置 (环境变量/.env)

BASE = paths.REPO
CKPT = f"{BASE}/checkpoints"
SESSION_DIR = r"C:\Users\Administrator\.zcode\cli\exec\sess_79382b4e-b5fc-48a3-9f7f-73768fa1f2f2"

ABL120 = [("abl120_no_emr",120,100),("abl120_no_ecsam",120,95),("abl120_no_fpn",120,88),
          ("abl120_no_trans",120,90),("abl120_no_adapter",120,95)]
STRATS = [("strat_no_emr_oc30",30,100),("strat_bilinear_oc30",30,95),
          ("strat_no_emr_oc60_lr1e4",60,100),("strat_bilinear_oc60_lr1e4",60,95),
          ("strat_unet_oc12_lr2e4",12,80),("strat_unet_oc12_lr1e3",12,80),
          ("strat_mssact_oc30",30,95),("strat_mssact_oc60_lr5e4",60,95),
          ("strat_mssact_sgdr120",120,95),("strat_deeplab_oc30",30,85),
          ("strat_mssact_full30m_oc60",60,180)]
TAIL_OVERHEAD = 20*60
N_MODELS_OVERFIT = 9
ALL_TASKS = ABL120 + STRATS

def discover_logs():
    cands = glob.glob(os.path.join(SESSION_DIR, "call_00_ET_*-stdout.log"))
    def size(p):
        try: return os.path.getsize(p)
        except: return 0
    # 按 mtime 排序 (最新优先), 再过滤空文件
    cands.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return [p for p in cands if size(p) >= 200][:5]

def enable_ansi():
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass

def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu","--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "N/A"

def fmt_dur(sec):
    if sec < 0: sec = 0
    h = int(sec//3600); m = int((sec%3600)//60); s = int(sec%60)
    if h > 0: return f"{h}h{m:02d}m"
    if m > 0: return f"{m}m{s:02d}s"
    return f"{s}s"

def hist_stats(tag):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p): return (None,)*9
    try:
        h = json.load(open(p))
        if not h: return (None,)*9
        b = max(h, key=lambda r: r.get("kappa", -9))
        l = h[-1]
        return (len(h), b["epoch"], b.get("kappa"), b.get("oa"), b.get("mf1"),
                l["epoch"], l.get("kappa"), l.get("oa"), l.get("mf1"))
    except Exception:
        return (None,)*9

def fmt_metrics(k, oa, mf1):
    parts = []
    if k is not None: parts.append(f"K={k:.4f}")
    if oa is not None: parts.append(f"OA={oa:.4f}")
    if mf1 is not None: parts.append(f"F1={mf1:.4f}")
    return "  ".join(parts)

def read_all_logs():
    """返回 (当前日志文本, 全部日志的DONE集合)"""
    logs = discover_logs()
    done_all = set()
    cur_txt = ""
    if logs:
        try: cur_txt = open(logs[0], encoding="utf-8", errors="ignore").read()
        except Exception: cur_txt = ""
        for p in logs:
            try:
                t = open(p, encoding="utf-8", errors="ignore").read()
                for m in re.finditer(r"=== (\S+) DONE", t):
                    done_all.add(m.group(1))
            except Exception:
                pass
    return cur_txt, done_all

def main(interval=15, default_sps=95):
    enable_ansi()
    while True:
        txt, dones = read_all_logs()
        lines = txt.splitlines()

        # ---- 当前活动 ----
        cur = None; ep_now = None; target_now = None; sps = default_sps
        live = {}
        if "PHASE-1" in txt and any(re.search(r"H_train=", l) for l in lines):
            n_done = sum(1 for l in lines if re.search(r"H_train=", l))
            cur = "PHASE-1 过拟合诊断"
            ep_now, target_now = n_done, N_MODELS_OVERFIT
        else:
            for l in reversed(lines):
                m = re.search(r"=== ([a-z0-9_]+)(?:\(ep|\||\s|$)", l)
                if m and " DONE " not in l:
                    cand = m.group(1)
                    if cand not in dones:
                        cur = cand; break
            if cur:
                target_now = next((ep for t,ep,_ in ALL_TASKS if t==cur), None)
                for l in reversed(lines):
                    m = re.search(r"\[ep\s+(\d+)(?:/(\d+))?\]", l)
                    if m and (" " + cur + " " in l):
                        ep_now = int(m.group(1))
                        km = re.search(r"Kappa=([-\d.]+)", l); om = re.search(r"OA=([-\d.]+)", l)
                        fm = re.search(r"mF1=([-\d.]+)", l); sm = re.search(r"\((\d+)s\)", l)
                        if km: live["kappa"] = float(km.group(1))
                        if om: live["oa"] = float(om.group(1))
                        if fm: live["mf1"] = float(fm.group(1))
                        if sm: sps = int(sm.group(1))
                        break

        # ---- 阶段 (由当前任务归属推断, 最稳) ----
        if any("QUEUE COMPLETE" in l for l in lines):
            phase = "✅完成"
        elif cur:
            if any(t == cur for t,_,_ in STRATS): phase = "PHASE-3"
            elif any(t == cur for t,_,_ in ABL120): phase = "PHASE-2"
            elif cur == "PHASE-1 过拟合诊断": phase = "PHASE-1"
        else:
            phase = "待命"

        def row(tag, target):
            n, be, bk, boa, bf1, le, lk, loa, lf1 = hist_stats(tag)
            if tag in dones:
                st = "✅完成"
                epstr = f"ep{be}/{target}" if be else ""
                met = fmt_metrics(bk, boa, bf1)
            elif n is None:
                st, epstr, met = "未开始", "", ""
            elif tag == cur:
                st = "▶进行中"
                ep = ep_now if ep_now else (n or 0)
                epstr = f"ep{ep}/{target}"
                met = fmt_metrics(live.get("kappa", lk), live.get("oa", loa), live.get("mf1", lf1))
            elif n is not None and n >= target and tag not in dones:
                st = "✅完成(旧实例)"
                epstr = f"ep{be or le}/{target}"
                met = fmt_metrics(bk, boa, bf1)
            else:
                st = "⏳待覆盖"
                epstr = f"ep{le}/{target}" if le else ""
                met = fmt_metrics(lk if lk is not None else bk,
                                  loa if loa is not None else boa,
                                  lf1 if lf1 is not None else bf1)
            return f"  {tag:26s} {st:8s} {epstr:12s} {met}"

        # ---- 剩余估算 ----
        remain = TAIL_OVERHEAD
        if cur and cur != "PHASE-1 过拟合诊断" and target_now:
            e = ep_now if ep_now is not None else 0
            remain += max(0, target_now - e) * sps
        elif cur == "PHASE-1 过拟合诊断":
            remain += max(0, N_MODELS_OVERFIT - (ep_now or 0)) * 25
        for t, ep, bsps in ALL_TASKS:
            if t == cur: continue
            if t in dones: continue
            n, *_ = hist_stats(t)
            if n is not None and n >= ep: continue
            remain += ep * bsps
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remain)

        print("\033[2J\033[H", end="")
        print("="*100)
        print("  MSSACT-LoveDA 队列进度监听器 | Ctrl+C 退出 (纯监控)")
        print("="*100)
        print(f"[{datetime.datetime.now():%H:%M:%S}] 阶段: {phase}")
        if cur:
            if cur == "PHASE-1 过拟合诊断":
                print(f"  ▶ 当前: {cur}  ({ep_now if ep_now else 0}/{N_MODELS_OVERFIT} 模型完成)")
            else:
                epstr = f"ep{ep_now}" + (f"/{target_now}" if target_now else "")
                met = fmt_metrics(live.get("kappa"), live.get("oa"), live.get("mf1"))
                print(f"  ▶ 当前: {cur}  {epstr}  ({sps}s/轮)  {met}")
        print(f"  预计剩余: {fmt_dur(remain)}  (ETA {eta:%m-%d %H:%M} | 跑满上限假设, 早停会更快)")
        print(f"  GPU: {gpu_info()}")
        print(" ── 阶段2 消融收敛 (abl120×5) ──")
        for t,ep,_ in ABL120: print(row(t,ep))
        print(" ── 阶段3 策略探索 (strat×11) ──")
        for t,ep,_ in STRATS: print(row(t,ep))
        sys.stdout.flush()
        time.sleep(interval)

if __name__ == "__main__":
    iv = 15; dsp = 95
    args = sys.argv[1:]
    for i,a in enumerate(args):
        if a == "--interval" and i+1 < len(args): iv = int(args[i+1])
        elif a == "--sps" and i+1 < len(args): dsp = int(args[i+1])
    try:
        main(iv, dsp)
    except KeyboardInterrupt:
        print("\n监听退出 (训练不受影响)")
