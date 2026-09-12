"""实验队列 v4（回退版: memmap 加速 + 原增强语义，性能已恢复）:
  - tag 前缀 lgR_ = "rollback pipeline"（memmap 加速 + 全局RNG增强 + workers=0）
  - 已验证: 回退版 D1 = 0.6293 vs 旧管线 0.6277（+0.0016，性能恢复）
  - 加速来源: memmap 消除 PNG 解码（14.7s/轮 vs 旧 144s/轮，10x）

原 v3 说明（已被回退）:
  - tag 前缀 lgF_ 表示"fast pipeline"，与旧管线实验区分
  - 阶梯子集用 PNG + workers=8（未预解码）

实验队列 v2（按用户决策调整）:
  - 取消: 30.5M 全量模型（性价比低、自身问题大）
  - 保留: 数据量阶梯（用户明确要求）
  - 新增: 三个架构改进实验（针对分析发现的问题）
  - 砍掉: 固定LR实验（累计学习量分析已在数学上回答调度效应）
  - 保留: 预算攻击（便宜，验证"预算是否掩盖模块差异"）

架构实验设计依据（见分析）:
  D1 联合消融    : use_fpn=False + use_transformer=False
                   依据: FPN(40.7%参数)+Transformer(26.4%)各自消融ΔK≈0，
                   若联合消融亦无显著下降 -> 直接证明功能冗余
  D2 解码器CA    : ECSAM 从编码器移到解码器上采样路径（文献通行做法）
                   依据: CA 的价值在位置敏感/定位任务; 我们的编码器用法在
                   深层降采样特征上（位置已稀释）且任务位置无关
  D3 通道-空间比 : embed_dims 缩至 [16,32,64,128]（超轻量）
                   依据: 通道/像素比达标准设计4倍, 过拟合诊断ΔH=-0.241
  D4 大容量对照  : embed_dims=[64,128,256,512]（同数据下容量对照）

协议: 与阶段C一致（newsplit2/train 1768张, OneCycle-60, pt20, lr2e-4, batch8）
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
CKPT = os.path.join(BASE, "checkpoints")
import paths
import queue_guard                     # 集中路径配置
LADDER = paths.DATA_LADDER
ROOT_MAIN = paths.DATA_NEWSPLIT2

def queues_running():
    """等待【其他】队列结束; 必须排除自身 PID, 否则会自检死锁"""
    my_pid = os.getpid()
    try:
        r = subprocess.run(["powershell","-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=25)
        for line in r.stdout.splitlines():
            if "run_queue" not in line: continue
            parts = line.split(",")
            if not parts: continue
            pid = parts[0].strip('"')
            if pid.isdigit() and int(pid) != my_pid:
                return True
        return False
    except Exception:
        return True

def tag_state(tag, target):
    p = os.path.join(CKPT, f"{tag}_history.json")
    if not os.path.exists(p): return "absent"
    try:
        h = json.load(open(p)); n = len(h) if isinstance(h,list) else 0
        return "complete" if n >= target else f"partial({n}/{target})"
    except Exception:
        return "corrupt"

def verify(tag):
    try:
        h = json.load(open(os.path.join(CKPT, f"{tag}_history.json")))
        b = max(h, key=lambda r: r.get("kappa",-9))
        print(f"  OK {tag}: n={len(h)} best_k={b['kappa']:.4f}@ep{b['epoch']}", flush=True)
    except Exception as e:
        print(f"  !!! {tag} verify fail: {e}", flush=True)

if __name__ == "__main__":
    print("=== 实验队列 v2（等待主队列结束）===", flush=True)
    while queues_running():
        print("  其他队列运行中, 等待...", flush=True)
        time.sleep(600)
    print("开始执行", flush=True); time.sleep(60)

    from experiment_matrix import train_one, UNet, DeepLabV3Plus
    from experiment_matrix_v2 import mssact_light
    from models.msscactnet import MSSACTNet

    def light(**kw):
        return mssact_light(**kw)

    # ============ D. 架构改进实验 ============
    print("\n===== D. 架构改进实验（针对分析发现）=====", flush=True)
    ARCH = [
        # (tag, 构造函数, 说明, 预算)
        ("lgR_D1_join_nofpn_notrans",
         lambda: light(use_fpn=False, use_transformer=False),
         "联合消融: FPN+Transformer 同时移除 (验证功能冗余)", 60),
        ("lgR_D2_decoder_ca",
         lambda: light(decoder_ca_stages=(0, 1)),
         "ECSAM 移到解码器上采样路径 (文献通行做法)", 60),
        ("lgR_D3_ch_tiny",
         lambda: MSSACTNet(in_channels=3, num_classes=7, embed_dims=[16,32,64,128],
                           transformer_layers=2, transformer_heads=4),
         "超轻量通道 [16,32,64,128] (验证通道-空间比失衡)", 60),
        ("lgR_D4_ch_large",
         lambda: MSSACTNet(in_channels=3, num_classes=7, embed_dims=[64,128,256,512],
                           transformer_layers=2, transformer_heads=4),
         "大容量通道 [64,128,256,512] (容量对照)", 60),
    ]
    # batch 缩放对照（用户建议）: batch=16 (显存实测 4.61GB, 安全上限) vs 已有 batch=8
    # 线性缩放规则 (Goyal et al. 2017): lr = 2e-4 * (16/8) = 4e-4
    # 完整模型基线（新管线）: 架构冗余结论的必需同管线对照 -> 插到最前
    ARCH = [
        ("lgR_bs8_full_lr2e4", lambda: light(),
         "★ 新管线完整模型基线（D1-D4 的同管线对照）", 60),
    ] + ARCH

    for tag, fn, desc, ep in ARCH:
        st = tag_state(tag, ep)
        if st == "complete": print(f"SKIP {tag}", flush=True); continue
        print(f"=== {tag} ===\n    {desc}", flush=True)
        # 按 tag 决定 batch/lr（bs16 实验用线性缩放后的 lr）
        _bs = 16 if "bs16" in tag else 8
        _lr = 4e-4 if "bs16" in tag else 2e-4
        try:
            train_one(fn, tag, max_epochs=ep, patience=20, batch=_bs, lr=_lr, root=ROOT_MAIN)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify(tag)

    # ============ A. 预算攻击 ============
    print("\n===== A. 预算攻击（15轮, 验证'预算是否掩盖模块差异'）=====", flush=True)
    B15 = [("full", lambda: light()), ("noecsam", lambda: light(use_ecsam=False)),
           ("noemr", lambda: light(use_emr=False)), ("unet", lambda: UNet()),
           ("deeplab", lambda: DeepLabV3Plus())]
    for mtag, fn in B15:
        tag = f"lgR_b15_{mtag}"
        st = tag_state(tag, 15)
        if st == "complete": print(f"SKIP {tag}", flush=True); continue
        print(f"=== {tag} ===", flush=True)
        try:
            train_one(fn, tag, max_epochs=15, patience=15, batch=8, lr=2e-4, root=ROOT_MAIN)
        except Exception as e:
            print(f"FAILED {tag}: {e}", flush=True)
        verify(tag)

    # ============ C. 数据量阶梯（用户明确保留）============
    print("\n===== C. 数据量阶梯（250/500/1000, 30轮）=====", flush=True)
    for n in [250, 500, 1000]:
        root = os.path.join(LADDER, f"n{n}")
        if not os.path.isdir(os.path.join(root, "train", "images")):
            print(f"SKIP n{n} (无数据)", flush=True); continue
        for mtag, fn in [("full", lambda: light()), ("noecsam", lambda: light(use_ecsam=False)),
                         ("unet", lambda: UNet()), ("deeplab", lambda: DeepLabV3Plus())]:
            tag = f"lgR_n{n}_{mtag}"
            st = tag_state(tag, 30)
            if st == "complete": print(f"SKIP {tag}", flush=True); continue
            print(f"=== {tag} ===", flush=True)
            try:
                train_one(fn, tag, max_epochs=30, patience=30, batch=8, lr=2e-4,
                          root=root, val_root=ROOT_MAIN, fast_data=False)  # 阶梯为 PNG 子集
            except Exception as e:
                print(f"FAILED {tag}: {e}", flush=True)
            verify(tag)

    print("\n===== 队列 v2 全部完成 =====", flush=True)
    r = subprocess.run([sys.executable, "-u", os.path.join("analysis", "eval_ladder.py")], cwd=BASE,
                       **queue_guard._no_window_kwargs())
    print(f"eval_ladder exit={r.returncode}", flush=True)
