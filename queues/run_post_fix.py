# -*- coding: utf-8 -*-
"""修复实验的评测后置链 (训练结束后运行)

为什么需要一个"全量"评测列表
--------------------------
`analysis/eval_rigor.py` 的预测缓存 `results/consensus_analysis/pred_{tag}.npz`
存在**两种不兼容的布局**:
  - 旧布局 (564, 256, 256)  —— consensus_analysis.py 时代产物
  - 新布局 (141, 1024, 1024) —— 与 labels.npz 一致, eval_rigor 的口径
后果: 凡缓存仍是旧布局的 tag, eval_rigor 一律 `形状 != 跳过`, 于是
**G1 的 8 个消融、lg_D1..D4、lgR_D1..D4 从未进入测试集协议**(逐波段/伪影/
等效性)分析 —— 报告里那些结论此前只有 val Kappa 单一来源。

本脚本显式列出全部需要(重新)推理的 tag, 让测试集协议覆盖**每一条**要引用的
结论; 随后跑等效性检验、种子方差、索引与事实汇总。

注意
----
* `--infer` 会**覆盖** `pred_{tag}.npz`。运行前会先把整个 consensus 目录备份到
  `results/_local_archive/consensus_backup_<时间戳>/`(命名纪律: 覆盖前必备份)。
* 必须在训练队列结束后运行 —— 1024² 整图推理会与训练争抢显存。
"""

import os
import sys
import json
import time
import shutil
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths
from queue_guard import queues_running, _no_window_kwargs      # noqa: E402

LOG = os.path.join(paths.LOGS, "post_fix.log")

# ---- 需要重新推理的 tag (旧布局或尚无缓存) ----
# G1 对比: nd_pspnet/nd_segformer 缓存为旧布局
# G1 消融: 8 个全部旧布局
# G1/G2 架构: lg_D1..D4, lgR_D1..D4, lgR 基线, lgR_deeplab_scr
# G4: 12 个新实验 (队列产物)
TAGS = (
    ["nd_pspnet", "nd_segformer",
     "nd_abl_no_emr", "nd_abl_no_ecsam", "nd_abl_no_fpn", "nd_abl_no_trans",
     "nd_abl_no_adapter", "nd_abl_bilinear", "nd_abl_trans4l", "nd_abl_trans6l",
     "lg_D1_join_nofpn_notrans", "lg_D2_decoder_ca", "lg_D3_ch_tiny", "lg_D4_ch_large",
     "lgR_bs8_full_lr2e4", "lgR_D1_join_nofpn_notrans", "lgR_D2_decoder_ca",
     "lgR_D3_ch_tiny", "lgR_D4_ch_large", "lgR_deeplab_scr", "lg_deeplab_scr",
     "lgR120_full", "lgR120_fx_skip", "lgR120_fx_pos", "lgR120_fx_all",
     "lgR120_fx_d2", "lgR120_fxskip_lr1e4", "lgR120_fxall_lr1e4",
     "lgR120_abl_no_emr", "lgR120_abl_no_ecsam", "lgR120_abl_no_fpn",
     "lgR120_abl_no_trans", "lgR120_abl_no_adapter", "lgR120_abl_bilinear",
     "lgR120_abl_trans4l", "lgR120_abl_trans6l",
     "lgR120_pos_abl_no_emr", "lgR120_pos_abl_no_ecsam", "lgR120_pos_abl_no_fpn",
     "lgR120_pos_abl_no_trans", "lgR120_pos_abl_no_adapter", "lgR120_pos_abl_bilinear",
     "lgR120_pos_abl_trans4l", "lgR120_pos_abl_trans6l",
     # 已有有效缓存, 一并列入以保证 rigor.json 覆盖完整
     "full_all_v2", "nd_unet", "nd_fcn", "nd_deeplab", "nd_fpn_seg", "nd_swin_unet",
     "sd7_full", "sd2024_full", "sd31337_full",
     "sd7_deeplab", "sd2024_deeplab", "sd31337_deeplab"])


def log(msg):
    line = "[post %s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def py():
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        c = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(c):
            return c
    return exe


def run(name, cmd, check=True):
    log("开始 %s" % name)
    t0 = time.time()
    out = os.path.join(paths.LOGS, "post_fix_%s.log" % name.replace(" ", "_"))
    with open(out, "a", encoding="utf-8") as fh:
        rc = subprocess.run([py(), "-u"] + cmd, cwd=_BASE, stdout=fh,
                            stderr=subprocess.STDOUT, **_no_window_kwargs()).returncode
    log("%s exit=%d 用时 %.1f 分钟 -> %s" % (name, rc, (time.time() - t0) / 60,
                                             os.path.basename(out)))
    if check and rc != 0:
        log("  !!! %s 失败, 中止后续步骤" % name)
        raise SystemExit(1)
    return rc


def backup_consensus():
    src = paths.CONSENSUS
    dst = os.path.join(paths.RESULTS, "_local_archive",
                       "consensus_backup_%s" % time.strftime("%Y%m%d_%H%M"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst)
    n = sum(len(f) for _, _, f in os.walk(dst))
    log("已备份 consensus 目录 -> %s (%d 个文件)" % (dst, n))
    return dst


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # 由训练队列调用时传自己的 PID: 否则互斥检查会把父队列当成"还在训练"而拒绝启动
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="调用本脚本的队列 PID, 从互斥检查中排除")
    a = ap.parse_args()
    excl = (a.parent_pid,) if a.parent_pid else ()
    if queues_running(exclude_pids=excl):
        log("仍有其他 run_queue* 在运行, 拒绝开始 (推理会与训练争显存)")
        raise SystemExit(1)
    backup_consensus()

    # 0) 修复兼容性回归 (默认行为未被改变 / 缺陷确已消除)
    run("verify_fix_compat", [os.path.join(_BASE, "verify", "verify_fix_compat.py")])
    # 1) 全量推理 (重生成新布局缓存)
    run("eval_rigor_infer",
        [os.path.join(_BASE, "analysis", "eval_rigor.py"), "--infer", "--only"] + TAGS)
    # 2) 等效性检验 (一等产物)
    run("tost_equivalence", [os.path.join(_BASE, "analysis", "tost_equivalence.py")])
    # 3) 修复归因 (效应量 / 收敛轮次 / 边界带)
    run("fix_attribution", [os.path.join(_BASE, "analysis", "fix_attribution.py")])
    # 4) 种子方差
    run("seed_variance", [os.path.join(_BASE, "analysis", "seed_variance.py")])
    # 5) 索引与事实汇总
    run("build_index", [os.path.join(_BASE, "report", "build_index.py")])
    run("build_facts", [os.path.join(_BASE, "report", "build_facts.py")])
    log("全部完成")


if __name__ == "__main__":
    main()
