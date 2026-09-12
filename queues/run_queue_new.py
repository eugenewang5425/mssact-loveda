"""新数据集(官方全量 2257张) 对比实验 + 消融实验 总队列

协议: 与 full_all 完全一致 —— OneCycle-60轮 / patience=20 / lr=2e-4 / batch=8 /
      EMA(0.999) / 类别加权CE / 验证Kappa早停

== 模块移除的架构适配与训练策略响应 (重要标注) ==
MSSACT-Net 各消融变体不是简单"删模块"，而是需要架构层面适配：

1. w/o EMR  (use_emr=False): EMR残差块 -> 标准ResBlock替换
   * 适配: 结构相似但无coord-attention辅助; 参数略减
   * 训练响应: 历史在60轮/pt15下坍缩(Kappa 0.0004), 120轮/pt20恢复(0.4055)
     -> 本队列统一60轮/pt20, 观察在更长patience下是否稳定

2. w/o ECSAM (use_ecsam=False): 移除3个阶段的坐标注意力
   * 适配: 前向直接跳过ecsam[i], 无形状问题; 参数量-0.06M
   * 训练响应: 历史稳定(60轮0.3134 / 120轮0.3587), 无特殊处理

3. w/o FPN (use_fpn=False): 解码器只接收最深特征
   * 适配: 跳过fpn, 直接把encoder_features[-1]给transformer/decoder
   * 说明: 多尺度融合信息丢失, 但通道数一致(256), 无形状冲突
   * 训练响应: 历史最优(60轮0.3760 / 120轮0.4416), 收敛更快

4. w/o Transformer (use_transformer=False): 移除全局自注意力+Adapter
   * 适配: x 直接进decoder; 参数量-1.6M
   * 训练响应: 稳定但上限略低(60轮0.3636 / 120轮0.4580)

5. w/o Adapter (use_adapter=False): 仅移除Adapter-Scale(scale_factor+升/降维)
   * 适配: transformer输出不做残差缩放适配
   * 训练响应: 120轮最佳(0.4621), 说明Adapter在小数据上是负担

6. Transformer-4L/6L: 层数2->4/6, 参数量+1.6M/+3.2M
   * 适配: 显存需求增加, batch可能需下调(依赖OOM自动降级)
   * 训练响应: 60轮下4L=0.3907 / 6L=0.4026, 接近默认2L(0.4065)

7. Bilinear-Up (upsample_mode='bilinear'): 转置卷积 -> 双线性+1x1卷积
   * 适配: 解码器参数-0.7M, 无学习上采样核
   * 训练响应: 60轮/pt15下坍缩(0.0414); lr=1e-4救援到0.3903
     -> 本队列用同协议观察, 若坍缩则记录为"策略敏感"证据

== 与 full_all 的公平性 ==
- 参考系: full_all (完整模型, 2257张, 60轮/pt20) val Kappa 0.6155
- 所有对比/消融变体使用相同数据划分(train 2257/val 282)、相同协议、相同评估
- 最终在 192张干净测试集 上做统一评估
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
import paths  # 集中路径配置 (环境变量/.env)

sys.path.insert(0, _HERE)
CKPT = paths.CKPT
ROOT_NEW = paths.DATA_NEWSPLIT2  # 筛选后(no-data<=10%) 1768 train

def tag_state(tag, target_epochs):
    hp = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(hp): return "absent"
    try:
        h = json.load(open(hp))
        n = len(h) if isinstance(h, list) else 0
        if n >= target_epochs: return "complete"
        return f"partial({n}/{target_epochs})"
    except Exception:
        return "corrupt-retry"

def verify_history(tag):
    p = f"{CKPT}/{tag}_history.json"
    if not os.path.exists(p):
        print(f"!!! {tag} NO history", flush=True); return False
    try:
        h = json.load(open(p))
        b = max(h, key=lambda r: r.get("kappa",-9))
        print(f"OK {tag}: n={len(h)} best_k={b['kappa']:.4f}@ep{b['epoch']}", flush=True)
        return True
    except Exception as e:
        print(f"!!! {tag} CORRUPT: {e}", flush=True); return False

if __name__ == "__main__":
    from experiment_matrix import train_one, UNet, PSPNet, FCN, DeepLabV3Plus, SegFormerLite
    from experiment_matrix_v2 import FPNSeg, SwinUnetLite, mssact_light
    from train_full_data import run_train as run_train_full

    MAX_EP, PATIENCE, LR, BATCH = 60, 20, 2e-4, 8

    # ==== PHASE 0: 修复 no-data bug 后的主模型重训 (与新队列同协议) ====
    print("\n===== PHASE-0: 主模型重训 (REMAP fix: no-data->ignore) =====", flush=True)
    if not os.path.exists(f"{CKPT}/full_all_v2_history.json"):
        print("=== full_all_v2 (60ep) ===", flush=True)
        try:
            run_train_full("full_all_v2", max_epochs=60, patience=20, lr=2e-4)
        except Exception as e:
            print(f"FAILED full_all_v2: {e}", flush=True)
        verify_history("full_all_v2")
    else:
        print("SKIP full_all_v2 (exists)", flush=True)
    if not os.path.exists(f"{CKPT}/post_all_v2_history.json"):
        print("=== post_all_v2 (30ep, 从 full_all_v2) ===", flush=True)
        try:
            run_train_full("post_all_v2", init_ckpt=f"{CKPT}/full_all_v2_best.pt", max_epochs=30, patience=12, lr=5e-5)
        except Exception as e:
            print(f"FAILED post_all_v2: {e}", flush=True)
        verify_history("post_all_v2")
    else:
        print("SKIP post_all_v2 (exists)", flush=True)
    print("===== PHASE-0 完成 =====\n", flush=True)

    JOBS = [
        # === 对比实验 (7) ===
        ("nd_unet",       lambda: UNet()),
        ("nd_pspnet",     lambda: PSPNet()),
        ("nd_fcn",        lambda: FCN()),
        ("nd_deeplab",    lambda: DeepLabV3Plus()),
        ("nd_segformer",  lambda: SegFormerLite()),
        ("nd_fpn_seg",    lambda: FPNSeg()),
        ("nd_swin_unet",  lambda: SwinUnetLite()),
        # === 消融实验 (8) ===
        ("nd_abl_no_emr",     lambda: mssact_light(use_emr=False)),
        ("nd_abl_no_ecsam",   lambda: mssact_light(use_ecsam=False)),
        ("nd_abl_no_fpn",     lambda: mssact_light(use_fpn=False)),
        ("nd_abl_no_trans",   lambda: mssact_light(use_transformer=False)),
        ("nd_abl_no_adapter", lambda: mssact_light(use_adapter=False)),
        ("nd_abl_trans4l",    lambda: mssact_light(transformer_layers=4)),
        ("nd_abl_trans6l",    lambda: mssact_light(transformer_layers=6)),
        ("nd_abl_bilinear",   lambda: mssact_light(upsample_mode="bilinear")),
    ]

    print(f"=== 新数据集队列: {len(JOBS)} 个实验 ===", flush=True)
    print(f"数据: {ROOT_NEW} (train 1768 / val 221 / test 221 / test_clean 141)", flush=True)
    print(f"协议: OneCycle-{MAX_EP}轮 / pt{PATIENCE} / lr{LR} / batch{BATCH}", flush=True)
    print(f"参考系: full_all (完整模型) val Kappa 0.6155", flush=True)
    for i, (tag, fn) in enumerate(JOBS, 1):
        st = tag_state(tag, MAX_EP)
        if st == "complete":
            print(f"[{i}/{len(JOBS)}] SKIP {tag} (complete)", flush=True); continue
        if st.startswith("partial"):
            print(f"[{i}/{len(JOBS)}] RERUN {tag} ({st})", flush=True)
        print(f"[{i}/{len(JOBS)}] === {tag} ===", flush=True)
        t0 = time.time()
        try:
            train_one(fn, tag, max_epochs=MAX_EP, patience=PATIENCE, batch=BATCH, lr=LR, root=ROOT_NEW)
        except Exception as e:
            print(f"FAILED {tag}: {type(e).__name__} {str(e)[:200]}", flush=True)
        verify_history(tag)
        print(f"[{i}/{len(JOBS)}] {tag} 用时 {(time.time()-t0)/60:.0f} 分钟", flush=True)
    print("=== 新数据集队列全部完成 ===", flush=True)
