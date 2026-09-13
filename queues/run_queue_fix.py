# -*- coding: utf-8 -*-
"""缺陷修复验证队列 (lgR120_*)

背景
----
`docs/项目阶段评估_20260913.md` 的收尾清单要求: 先修复三处**已被数值证实**的架构
缺陷, 再重评 —— 这才是让消融实验变得可解释的前提。

已证实的缺陷 (证据见 results/facts/FACTS.json 的 arch_defects 段)
--------------------------------------------------------------
D1  FPN 死支路        self.fpn() 产出 4 级特征, 但 forward 只取 [-1]; 归零
                      lateral_convs[0..2] 后输出**逐位不变** (diff = 0.0)。
                      实测 1,828,352 / 6,099,432 = 29.976% 参数收不到梯度。
D2  无编码器->解码器跳连  解码器 forward 只接收 32x32 特征, 256/128/64 三级
                      细节从未到达输出; 上采样 8 倍全靠转置卷积"合成"。
D3  Transformer 无位置编码  摊平为 token 后未加 PE, 模块**置换等变**
                      (置换 token 后按逆置换还原, 残差 7.153e-07)。
D5  ecsam[0] 死模块   forward 里是 `i > 0` 才应用 ECSAM, 故第 0 个实例
                      (1040 参数, 0.017%) 恒不被使用。(本轮不修, 仅记录)

修复方式 (models/msscactnet.py, 全部**向后兼容**)
-----------------------------------------------
新增三个开关, 默认值 == 修复前行为, 故既有 checkpoint 仍可 strict 加载、
既有结论仍可复现 (已实测: 默认路径输出与旧版逐位相同, 最大差 0.000e+00):
  use_skip=True        解码器逐级接入主干同分辨率特征; 有 FPN 时接入 FPN 的
                       4 级输出 (顺带修好 D1 —— 死支路变活), 无 FPN 时接入
                       编码器浅三层
  pos_enc=True         Transformer 注入 2D 正弦位置编码 (行/列各占一半通道)
  ecsam_stage0=True    第 0 阶段也应用 ECSAM (使死参数归零; 本轮不启用)

实验设计
--------
**为什么必须重跑基线**: OneCycle 的收敛轮次由退火终点决定 (best@49 对应
60 轮 OneCycle、best@109 对应 120 轮), 故 60 轮与 120 轮是**不同学习率调度**,
Kappa 不可直接比较。要做修复归因, 必须有一条**同为 120 轮**的基线。

阶段一 (修复归因, 4 个): 回答"修复是否有效"
  lgR120_full      120 轮同调度基线 (对照)
  lgR120_fx_skip   仅修 D1+D2 (跳连 + 多尺度接入)
  lgR120_fx_pos    仅修 D3   (位置编码)
  lgR120_fx_all    三处全修
  判别门限: sigma_seed = 0.0050 (n=3 种子); |dK| >= 2*sigma 记为可分辨

阶段二 (重跑消融, 8 个): 修复后基线之上的**首次可解释的消融**
  lgR120_abl_{no_emr,no_ecsam,no_fpn,no_trans,no_adapter,bilinear,trans4l,trans6l}
  全部以 use_skip=True + pos_enc=True 为底, 故每个模块的作用不再被缺陷掩盖。

协议: 与 G2 组一致 (newsplit2/train 1768, bs8, lr2e-4, OneCycle, EMA0.999,
      pt20), 但预算为 120 轮 —— 并入新的可比组 G4-MEM-ROLL-120ep。
      **禁止**把 lgR120_* 与 lgR_* (60 轮) 的 Kappa 直接比较。

耗时: 实测 ~48-59 s/轮 (跳连使解码器计算量增加约 10%)。
      12 个实验 x 120 轮 约 19-21 小时。

幂等: 每个 tag 先查 history 长度, 已 complete 则跳过 —— 中断后可直接重启续跑。
"""

# --- 路径引导（本文件位于子目录 queues/，仓库根为上一级）---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os
import sys
import json
import time
import subprocess

import paths
from queue_guard import queues_running, POLL_INTERVAL, _no_window_kwargs   # noqa: E402

# 在模块级导入: PHASE1/PHASE2 的 lambda 会在模块加载时就引用这两个名字
from experiment_matrix import train_one                       # noqa: E402
from experiment_matrix_v2 import mssact_light                 # noqa: E402

BASE = _BASE
CKPT = paths.CKPT
ROOT_MAIN = paths.DATA_NEWSPLIT2
LOG = os.path.join(paths.LOGS, "queue_fix.log")


def log(msg):
    line = "[fix %s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def tag_state(tag, target):
    p = os.path.join(CKPT, "%s_history.json" % tag)
    if not os.path.exists(p):
        return "absent"
    try:
        h = json.load(open(p, encoding="utf-8"))
        n = len(h) if isinstance(h, list) else 0
        return "complete" if n >= target else "partial(%d/%d)" % (n, target)
    except Exception:
        return "corrupt"


def verify(tag):
    try:
        h = json.load(open(os.path.join(CKPT, "%s_history.json" % tag), encoding="utf-8"))
        b = max(h, key=lambda r: r.get("kappa", -9))
        secs = [r.get("sec") for r in h if r.get("sec")]
        log("  OK %s: n=%d best_k=%.4f@ep%s%s" % (
            tag, len(h), b["kappa"], b["epoch"],
            "  %.1fs/轮" % (sum(secs) / len(secs)) if secs else ""))
        return b["kappa"]
    except Exception as e:
        log("  !!! %s verify fail: %s" % (tag, e))
        return None


# ============================ 实验表 ============================
_FIX = dict(use_skip=True, pos_enc=True)

PHASE1 = [
    ("lgR120_full",    lambda: mssact_light(),
     "120轮同调度基线 (修复归因的对照)"),
    ("lgR120_fx_skip", lambda: mssact_light(use_skip=True),
     "仅修 D1+D2: 解码器跳连 + FPN 四级全部接入"),
    ("lgR120_fx_pos",  lambda: mssact_light(pos_enc=True),
     "仅修 D3: Transformer 2D 正弦位置编码"),
    ("lgR120_fx_all",  lambda: mssact_light(**_FIX),
     "三处全修 (D1+D2+D3)"),
]

# 注意: 第二个元素必须是**可调用对象** —— run_table 会直接把它传给 train_one 当
# model_fn。首版误写成 dict(use_emr=False), 于是 8 个实验全部
# `TypeError: 'dict' object is not callable` 秒失败。预检脚本当时用
# `mssact_light(**kw)` 自己解包, 只验证了配置、没验证调用约定, 故未能发现。
def _abl(**flags):
    """消融变体的构造函数: 以 use_skip+pos_enc 为底"""
    return lambda: mssact_light(use_skip=True, pos_enc=True, **flags)


PHASE2 = [
    ("lgR120_abl_no_emr",     _abl(use_emr=False),           "消融: EMR -> 标准残差块"),
    ("lgR120_abl_no_ecsam",   _abl(use_ecsam=False),         "消融: 去掉 ECSAM 空间注意力"),
    ("lgR120_abl_no_fpn",     _abl(use_fpn=False),           "消融: 去掉 FPN (跳连改接编码器浅三层)"),
    ("lgR120_abl_no_trans",   _abl(use_transformer=False),   "消融: 去掉 Transformer 全局上下文"),
    ("lgR120_abl_no_adapter", _abl(use_adapter=False),       "消融: 去掉 Adapter-Scale"),
    ("lgR120_abl_bilinear",   _abl(upsample_mode="bilinear"), "消融: 转置卷积 -> 双线性上采样"),
    ("lgR120_abl_trans4l",    _abl(transformer_layers=4),    "深度: Transformer 2 -> 4 层"),
    ("lgR120_abl_trans6l",    _abl(transformer_layers=6),    "深度: Transformer 2 -> 6 层"),
]


def run_table(rows, phase, epochs=120):
    log("")
    log("=" * 72)
    log("阶段%s: %d 个实验 x %d 轮" % (phase, len(rows), epochs))
    log("=" * 72)
    t_phase = time.time()
    for i, row in enumerate(rows, 1):
        tag, fn, desc = row
        st = tag_state(tag, epochs)
        if st == "complete":
            log("[%d/%d] SKIP %s (已完成)" % (i, len(rows), tag))
            verify(tag)
            continue
        log("[%d/%d] === %s ===  %s" % (i, len(rows), tag, desc))
        if st.startswith("partial"):
            log("        续跑: 现有 %s, 将从头重训 (history 会被覆盖, 旧文件已归档)" % st)
        t0 = time.time()
        try:
            train_one(fn, tag, max_epochs=epochs, patience=20,
                      batch=8, lr=2e-4, root=ROOT_MAIN)
        except Exception as e:
            log("  FAILED %s: %s %s" % (tag, type(e).__name__, str(e)[:300]))
            continue
        k = verify(tag)
        el = (time.time() - t0) / 60.0
        # ETA: 按本阶段已用时间线性外推
        done = i
        eta = (time.time() - t_phase) / 60.0 / done * (len(rows) - done)
        log("  用时 %.1f 分钟; 本阶段 ETA 剩余约 %.1f 小时" % (el, eta / 60.0))
    log("阶段%s 完成, 用时 %.1f 小时" % (phase, (time.time() - t_phase) / 3600.0))


def post_steps():
    """训练后置: 交给 run_post_fix.py (全量推理 + 等效性检验 + 索引/事实汇总)

    注意: 不要把评测列表收窄成只有本轮新 tag。`pred_{tag}.npz` 存在两种不兼容
    布局 (旧 564x256x256 / 新 141x1024x1024), 只对 G1 消融与 lg_D1..D4、
    lgR_D1..D4 做"只推理新 tag"会**永久跳过**它们, 使测试集协议只覆盖一半结论。
    详见 run_post_fix.py 的 docstring。
    """
    steps = [
        ("后置链 run_post_fix", [os.path.join(_HERE, "run_post_fix.py")]),
    ]
    for name, cmd in steps:
        log("后置: %s" % name)
        try:
            out = os.path.join(paths.LOGS, "queue_fix_post.log")
            with open(out, "a", encoding="utf-8") as fh:
                rc = subprocess.run([sys.executable, "-u"] + cmd, cwd=BASE, stdout=fh,
                                    stderr=subprocess.STDOUT, **_no_window_kwargs()).returncode
            log("   %s exit=%d -> %s" % (name, rc, os.path.basename(out)))
        except Exception as e:
            log("   %s 失败: %s" % (name, e))


if __name__ == "__main__":
    log("启动: 等待其他 run_queue* 结束 (轮询 %ds, 无窗口)" % POLL_INTERVAL)
    waited = 0
    while queues_running():
        log("其他队列运行中, 已等待 %d 分钟" % waited)
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL // 60

    log("开始执行; 数据根 %s" % ROOT_MAIN)
    run_table(PHASE1, "一(修复归因)")
    run_table(PHASE2, "二(重跑消融)")
    post_steps()
    log("全部完成")
