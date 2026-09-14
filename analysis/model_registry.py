# -*- coding: utf-8 -*-
"""模型构造登记表 —— 推理侧的唯一来源

存在的理由（踩过的坑）
--------------------
本项目的**整图推理**此前有两套互不兼容的实现:

* `analysis/consensus_analysis.py` 的 `infer_tile()` —— 256² 滑窗 + 高斯加权在
  1024² 整图上推理, 与训练裁剪尺度一致, 输出 `(141, 1024, 1024)`。**这是对的**。
* `analysis/eval_rigor.py` 的 `infer_tag()` —— 走 `LoveDADataset(train=False)`,
  而该数据集在非训练模式下 `n_val_patches=4`、crop=256, 于是每个 1024² 图被拆成
  4 个 256² 块, 输出 `(564, 256, 256)`。与 `labels.npz` 的 `(141,1024,1024)`
  **永远不一致**, 故 `eval_rigor` 的 `--infer` 产出的缓存**必定被自己的形状检查跳过**。

后果: 任何"只靠 eval_rigor 补推理"的努力都不可能让新 tag 进入测试集协议。
G1 的 8 个消融、`lg_D1..D4`、`lgR_D1..D4`、`np_pspnet`、`nd_segformer` 因此长期
缺席, 使消融结论只有 val Kappa 单一来源。

本模块把"tag → 模型"的映射收敛到一处, 两个脚本共用, 从此只有一套实现。

用法
----
    from model_registry import build_for_tag, is_registered
    model = build_for_tag("lgR120_fx_skip")     # 未登记则抛 KeyError
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
for _p in (_BASE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_LIGHT = "mssact_light"      # 标记: 用 experiment_matrix_v2.mssact_light(**kw)
_BIG = "mssact_big"          # 标记: 直接实例化 MSSACTNet 并给 embed_dims


def _big(dims):
    return (_BIG, dict(embed_dims=dims))


# tag -> (构造方式, 关键字参数)
# 顺序与 build_index.py 的实验登记保持一致, 便于交叉核对
TAG_CFG = {
    # ---- C-final: 主模型 / 对比 / 消融 (P-PNG) ----
    "full_all_v2": (_LIGHT, {}),
    "post_all_v2": (_LIGHT, {}),
    "nd_unet": ("unet", {}),
    "nd_pspnet": ("pspnet", {}),
    "nd_fcn": ("fcn", {}),
    "nd_deeplab": ("deeplab", {}),
    "nd_deeplab_scr": ("deeplab_scr", {}),
    "nd_segformer": ("segformer", {}),
    "nd_fpn_seg": ("fpn", {}),
    "nd_swin_unet": ("swin", {}),
    "nd_abl_no_emr": (_LIGHT, dict(use_emr=False)),
    "nd_abl_no_ecsam": (_LIGHT, dict(use_ecsam=False)),
    "nd_abl_no_fpn": (_LIGHT, dict(use_fpn=False)),
    "nd_abl_no_trans": (_LIGHT, dict(use_transformer=False)),
    "nd_abl_no_adapter": (_LIGHT, dict(use_adapter=False)),
    "nd_abl_bilinear": (_LIGHT, dict(upsample_mode="bilinear")),
    "nd_abl_trans4l": (_LIGHT, dict(transformer_layers=4)),
    "nd_abl_trans6l": (_LIGHT, dict(transformer_layers=6)),
    # ---- lg-old: 架构 (P-PNG) ----
    "lg_D1_join_nofpn_notrans": (_LIGHT, dict(use_fpn=False, use_transformer=False)),
    "lg_D2_decoder_ca": (_LIGHT, dict(decoder_ca_stages=(0, 1))),
    "lg_D3_ch_tiny": _big([16, 32, 64, 128]),
    "lg_D4_ch_large": _big([64, 128, 256, 512]),
    "lg_bs8_full_lr2e4": (_LIGHT, {}),
    "lg_bs16_full_lr4e4": (_LIGHT, {}),
    "lg_deeplab_scr": ("deeplab_scr", {}),
    # ---- lgF: 已否决世代 (P-MEM-DET) ----
    "lgF_bs8_full_lr2e4": (_LIGHT, {}),
    "lgF_D1_join_nofpn_notrans": (_LIGHT, dict(use_fpn=False, use_transformer=False)),
    "lgF_D2_decoder_ca": (_LIGHT, dict(decoder_ca_stages=(0, 1))),
    "lgF_D3_ch_tiny": _big([16, 32, 64, 128]),
    "lgF_D4_ch_large": _big([64, 128, 256, 512]),
    # ---- lgR: 回退管线 (P-MEM-ROLL) ----
    "lgR_bs8_full_lr2e4": (_LIGHT, {}),
    "lgR_D1_join_nofpn_notrans": (_LIGHT, dict(use_fpn=False, use_transformer=False)),
    "lgR_D2_decoder_ca": (_LIGHT, dict(decoder_ca_stages=(0, 1))),
    "lgR_D3_ch_tiny": _big([16, 32, 64, 128]),
    "lgR_D4_ch_large": _big([64, 128, 256, 512]),
    "lgR_deeplab_scr": ("deeplab_scr", {}),
    "lgR_c256_center": (_LIGHT, {}),
    "lgR_c384_rand": (_LIGHT, {}),
    "lgR_c512_rand": (_LIGHT, {}),
    # ---- 种子方差 ----
    "sd7_full": (_LIGHT, {}), "sd2024_full": (_LIGHT, {}), "sd31337_full": (_LIGHT, {}),
    "sd7_deeplab": ("deeplab", {}), "sd2024_deeplab": ("deeplab", {}),
    "sd31337_deeplab": ("deeplab", {}),
    # ---- lgR120: 缺陷修复组 (120 轮) ----
    "lgR120_full": (_LIGHT, {}),
    "lgR120_fx_skip": (_LIGHT, dict(use_skip=True)),
    "lgR120_fx_pos": (_LIGHT, dict(pos_enc=True)),
    "lgR120_fx_all": (_LIGHT, dict(use_skip=True, pos_enc=True)),
    # 只融合深层两级(不做 256² 融合): 用于分离"全分辨率融合"的影响
    "lgR120_fx_d2": (_LIGHT, dict(use_skip=True, skip_stages=(1, 2))),
    # 判别实验: 分离"全分辨率融合有害"(H1) 与"学习率不再匹配"(H2)
    # 注: 同一份模型配置在 1e-4 与 2e-4 下各训一次, 故 TAG_CFG 只描述模型,
    #     LR 属于训练超参, 由队列脚本决定(见 queues/run_queue_fix2.py)
    "lgR120_fxskip_lr1e4": (_LIGHT, dict(use_skip=True)),
    "lgR120_fxall_lr1e4": (_LIGHT, dict(use_skip=True, pos_enc=True)),
    # 修复后消融第二轮: 底 = pos_enc=True（第二轮证实 use_skip 有害且 fx_all 底发散）
    "lgR120_pos_abl_no_emr": (_LIGHT, dict(pos_enc=True, use_emr=False)),
    "lgR120_pos_abl_no_ecsam": (_LIGHT, dict(pos_enc=True, use_ecsam=False)),
    "lgR120_pos_abl_no_fpn": (_LIGHT, dict(pos_enc=True, use_fpn=False)),
    "lgR120_pos_abl_no_trans": (_LIGHT, dict(pos_enc=True, use_transformer=False)),
    "lgR120_pos_abl_no_adapter": (_LIGHT, dict(pos_enc=True, use_adapter=False)),
    "lgR120_pos_abl_bilinear": (_LIGHT, dict(pos_enc=True, upsample_mode="bilinear")),
    "lgR120_pos_abl_trans4l": (_LIGHT, dict(pos_enc=True, transformer_layers=4)),
    "lgR120_pos_abl_trans6l": (_LIGHT, dict(pos_enc=True, transformer_layers=6)),
    # 修复后的消融 (更早一轮, 底 = use_skip + pos_enc; 该底不稳定, 见 run_queue_fix3)
    "lgR120_abl_no_emr": (_LIGHT, dict(use_skip=True, pos_enc=True, use_emr=False)),
    "lgR120_abl_no_ecsam": (_LIGHT, dict(use_skip=True, pos_enc=True, use_ecsam=False)),
    "lgR120_abl_no_fpn": (_LIGHT, dict(use_skip=True, pos_enc=True, use_fpn=False)),
    "lgR120_abl_no_trans": (_LIGHT, dict(use_skip=True, pos_enc=True, use_transformer=False)),
    "lgR120_abl_no_adapter": (_LIGHT, dict(use_skip=True, pos_enc=True, use_adapter=False)),
    "lgR120_abl_bilinear": (_LIGHT, dict(use_skip=True, pos_enc=True,
                                         upsample_mode="bilinear")),
    "lgR120_abl_trans4l": (_LIGHT, dict(use_skip=True, pos_enc=True, transformer_layers=4)),
    "lgR120_abl_trans6l": (_LIGHT, dict(use_skip=True, pos_enc=True, transformer_layers=6)),
}

# ---- 种子扩展 (2026-09-15): n=4→10 / n=3→9 ----
for _s in (1234, 5555, 8888, 31415, 27182, 9999):
    TAG_CFG["sd%d_full" % _s] = (_LIGHT, {})
    TAG_CFG["sd%d_deeplab" % _s] = ("deeplab", {})

# 允许按前缀批量登记同一配置(如 lgR_fx_lr1e4 这类探索性 tag), 避免每次实验都要改本文件
PREFIX_FALLBACK = ()


def is_registered(tag):
    return tag in TAG_CFG


def build_for_tag(tag):
    """按 tag 构造模型(未加载权重)。未登记则 KeyError 并提示如何补登记。"""
    if tag not in TAG_CFG:
        raise KeyError(
            "tag %r 未在 model_registry.TAG_CFG 登记; 请补登记后再推理" % tag)
    how, kw = TAG_CFG[tag]

    if how == _LIGHT:
        from experiment_matrix_v2 import mssact_light
        return mssact_light(**kw)
    if how == _BIG:
        from models.msscactnet import MSSACTNet
        return MSSACTNet(in_channels=3, num_classes=7, transformer_layers=2,
                         transformer_heads=4, **kw)

    from experiment_matrix import UNet, PSPNet, FCN, DeepLabV3Plus, SegFormerLite
    from experiment_matrix_v2 import FPNSeg, SwinUnetLite
    table = {
        "unet": UNet, "pspnet": PSPNet, "fcn": FCN, "segformer": SegFormerLite,
        "fpn": FPNSeg, "swin": SwinUnetLite,
        "deeplab": DeepLabV3Plus,
        # 从零对照: 必须显式关掉主干预训练, 见 README 的 pretrained_backbone 取证
        "deeplab_scr": lambda: DeepLabV3Plus(pretrained_backbone=False),
    }
    if how not in table:
        raise KeyError("未知构造方式 %r (tag=%s)" % (how, tag))
    return table[how](**kw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="登记表自检: 构造每个已登记的模型并报参数量")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    tags = a.only or sorted(TAG_CFG)
    bad = []
    for t in tags:
        try:
            m = build_for_tag(t)
            n = sum(p.numel() for p in m.parameters())
            print("  %-30s %.4fM" % (t, n / 1e6))
        except Exception as e:
            bad.append((t, "%s: %s" % (type(e).__name__, e)))
    print("\n  登记 %d 个; 构造失败 %d 个" % (len(tags), len(bad)))
    for t, why in bad:
        print("    FAIL %s -> %s" % (t, why))
    raise SystemExit(1 if bad else 0)
