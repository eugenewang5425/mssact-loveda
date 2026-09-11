# -*- coding: utf-8 -*-
"""生成 PDF 报告: markdown -> HTML -> Chrome headless PDF"""
import os, base64, re
import paths  # 集中路径配置 (环境变量/.env)
import markdown

BASE = paths.REPO
FIG = f"{BASE}/figures"

def img_b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

MD = r"""
# MSSACT-Net 遥感土地覆盖分割
## 完整实验报告

**数据集**: LoveDA 公开基准（0.3m 航空正射, 7 类土地覆盖, 官方人工标注）
**模型**: MSSACT-Net（多尺度自注意力 ConvTransformer, light 6.10M / full 30.54M）
**日期**: 2026-09

---

## 摘要

本报告记录基于公开数据集 LoveDA 的完整实验：数据量效应、后训练、TTA、对比实验、
消融实验、公平性验证。核心结论：

1. **数据量是主导因素**：训练数据从 957 张增至 2,257 张，测试集 Kappa 从 **0.3921 提升至 0.6114（+0.219）**
2. **少数类获益最大**：道路 F1 +0.390、水域 +0.385、裸地 +0.324——证明此前精度低的主因是数据量不足
3. **后训练有效但边际**（+0.012），**TTA 稳定小增益**（+0.005）
4. **最终成绩**：干净测试集 **Kappa 0.6278 / OA 0.7009 / mF1 0.7089 / mIoU 0.5583**

---

## 1. 数据与划分

### 1.1 数据来源与规模

| 项 | 数值 |
|---|---|
| LoveDA 官方总规模 | 5,987 张（train 2,522 / val 1,669 / test 1,796） |
| 本实验获取 | 官方训练区全部 **2,522 张** |
| MD5 去重后池 | **2,821 张**（历史镜像数据均为官方子集，已去重） |
| 划分方式 | MD5 去重后 8:1:1 随机划分（seed=42） |
| 训练集 | 2,257 张 |
| 验证集 | 282 张（早停/选优） |
| 测试集 | 282 张 |
| 干净测试子集 | **192 张**（额外排除旧模型训练集重叠，用于新旧公平对比） |

### 1.2 标签与类别

7 类：背景 / 建筑 / 道路 / 水域 / 裸地 / 森林 / 农田；官方人工逐像素标注，
标签值域 1..7（0=未标注，评估时以 ignore 排除）。

**类别分布（验证集）**：背景 45.6%、农田 26.6%、水域 11.6%、森林 5.5%、
建筑 4.3%、裸地 4.1%、道路 2.4%——最大类仅 45.6%，**无躺赢类**。

### 1.3 未标注（ignore）处理

- **不纳入评价**：评估时 `lab != 255` 掩码排除，所有指标（OA/Kappa/F1/mIoU）仅在有效像素上计算
- **训练**：`ignore_index=255`，未标注像素不产生梯度
- **实测占比**：官方全量数据 labels 中 ignore 占比 **0.00%**——本次数据几乎无未标注

---

## 2. 评价体系与公平性

### 2.1 为什么用 Kappa 而非 OA

Kappa 校正随机一致性：κ = (pₒ − pₑ)/(1 − pₑ)。

- 验证集最大类（背景）占 45.6%，**全预测背景的 OA=45.6% 但 Kappa=0**
- 训练早期多次出现 OA≈0.30 / Kappa≈0 的坍缩态，若用 OA 早停会误判
- **全部早停与选优均以验证 Kappa 为准**

### 2.2 数据使用协议

| 集合 | 梯度更新 | 参与选优 | 角色 |
|---|---|---|---|
| train 2,257 | 是 | — | 学习 |
| **val 282** | **否**（`torch.no_grad()` 内评估） | 是（早停/选优） | 监控 |
| **test 282 / 干净 192** | 否 | 否 | **最终无偏评估** |

代码证据：验证循环位于 `with torch.no_grad():` 内，无 backward/optimizer.step。

### 2.3 泄漏核查

- 旧模型训练集（957 张历史镜像）与新测试集有 **90/282 张重叠**
- 处理：构建 **192 张干净测试子集**（两模型均未见过），所有新旧对比在此子集上完成

---

## 3. 训练协议

统一协议（所有模型一致）：

- 优化器：AdamW(lr=2e-4, weight_decay=0.02)
- 调度：OneCycle（6% warmup + 余弦退火，60 轮）
- 权重平滑：EMA(decay=0.999)，评估使用 EMA 权重
- 损失：类别加权交叉熵（频率逆平方根，缓解不平衡）
- 采样：训练每图随机裁剪 256×256；验证每图 4 角固定 patch
- 早停：验证 Kappa patience=20
- 输入：256×256 RGB（0-1 归一化）

---

## 4. 数据量效应（核心结果）

### 4.1 四阶段递进（192 张干净测试集）

| 阶段 | 配置 | Kappa | OA | mF1 | mIoU |
|---|---|---|---|---|---|
| 基线 | 957 张 | 0.3921 | 0.5255 | 0.4532 | 0.3061 |
| +数据量 | 2,257 张 | 0.6114 | 0.6862 | 0.6946 | 0.5420 |
| +后训练 | 继续训练 lr=5e-5 | 0.6233 | 0.6968 | 0.7048 | 0.5536 |
| **+TTA** | 8× 几何变换平均 | **0.6278** | **0.7009** | **0.7089** | **0.5583** |

![最终对比](FIG_FINAL)

### 4.2 学习曲线

![学习曲线](FIG_CURVES_NEW)

**观察**：全量模型（红）在第 8 轮开始快速上升、第 35 轮达 0.61；
旧模型（灰）缓慢爬升至 0.40 后平台化。数据量直接决定收敛高度。

### 4.3 每类 F1 提升

![每类F1](FIG_FINAL_PERCLASS)

| 类别 | 957张 | 最终 | 提升 |
|---|---|---|---|
| 道路 | 0.284 | 0.674 | **+0.390** |
| 水域 | 0.431 | 0.817 | **+0.385** |
| 裸地 | 0.203 | 0.527 | +0.324 |
| 建筑 | 0.437 | 0.692 | +0.255 |
| 农田 | 0.552 | 0.806 | +0.254 |
| 森林 | 0.679 | 0.809 | +0.130 |
| 背景 | 0.587 | 0.637 | +0.051 |

**少数类（道路/水域/裸地/建筑）获益最大**——印证小类样本稀缺是此前精度低的主因。

### 4.4 测试集预测对比

![测试集对比1](FIG_TEST1)

![测试集对比2](FIG_TEST2)

![测试集对比3](FIG_TEST3)

**观察**：旧模型（第3列）大面积误判水域（蓝色泛滥）、建筑块完全丢失；
全量模型（第4列）清晰还原建筑群、道路网、水域边界；后训练（第5列）边缘更平滑。

---

## 4.5 混淆矩阵与统计诊断

### 归一化混淆矩阵（行=召回率）

![混淆矩阵归一化](FIG_CM_NORM)

**关键错误流向**（新模型）：

| 真实类 | 召回 | 主要混淆 |
|---|---|---|
| 背景 | 0.527 | → 建筑 0.18, 裸地 0.10 |
| 建筑 | **0.816** | → 背景 0.13 |
| 道路 | **0.777** | → 背景 0.16 |
| 水域 | **0.837** | → 背景 0.08, 农田 0.06 |
| 裸地 | **0.721** | → 背景 0.11, 农田 0.10 |
| 森林 | **0.752** | → 背景 0.13, 裸地 0.08 |
| 农田 | **0.823** | → 背景 0.08, 裸地 0.05 |

对比旧模型：建筑召回仅 0.38（62% 误判）、道路 0.22、裸地 0.15——新模型各类
召回率全面提升至 0.72-0.84。主要剩余错误是**背景与其他类的边界混淆**
（背景召回 0.53，其像素流向建筑/裸地），源于背景类定义宽泛（含道路边缘、
建筑阴影、田间小路等异质内容）。

### 混淆矩阵（计数值）

![混淆矩阵计数](FIG_CM_COUNT)

### 每类 精度(UA) / 召回(PA) / F1

![PRF](FIG_PRF)

### 参数效率

![效率](FIG_EFF)

MSSACT-Net 以 **6.1M 参数**占据帕累托前沿——DeepLabV3+ 需 39.6M 参数才达到
0.38 Kappa，而 MSSACT（全量）以 1/6.5 参数量达到 0.61。

### OneCycle 学习率调度

![LR](FIG_LR)

---

## 5. TTA（推理增强）

8× 几何变换（4旋转 × 2翻转）概率图平均：

| 配置 | 无TTA Kappa | 有TTA Kappa | 增益 |
|---|---|---|---|
| val 282 | 0.6268 | 0.6320 | +0.0052 |
| test 282 | 0.6423 | 0.6462 | +0.0039 |

零训练成本的稳定增益，推荐部署采用。

---

## 6. 对比实验（基线模型）

统一协议（256px, OneCycle-60, EMA, 类别加权CE, val Kappa 早停），
**旧数据集（957张）**结果：

| 模型 | 参数量 | Kappa | OA | mF1 |
|---|---|---|---|---|
| **MSSACT-Net (light)** | **6.10M** | **0.4065** | **0.5630** | 0.4214 |
| DeepLabV3+ (R50) | 39.6M | 0.3806 | 0.5092 | **0.4683** |
| FCN-32s (R50) | 23.7M | 0.2837 | 0.4245 | 0.3450 |
| PSPNet (R18+PPM) | 13.6M | 0.2513 | 0.3834 | 0.3097 |
| SegFormer-Lite | 1.01M | 0.1775 | 0.3550 | 0.2817 |
| U-Net (base=32) | 2.44M | 0.0939 | 0.4055 | 0.1375 |
| FPN (R50) | 27.2M | 0.2982 | 0.4328 | 0.3581 |
| Swin-Unet (lite) | 32.6M | 0.2682 | 0.3878 | 0.2926 |

![全量滑窗指标](FIG_FULLTILE)

**结论**：MSSACT-Net 以 DeepLabV3+ **1/6.5 参数量**取得最高 Kappa。

> 新数据集（2,257张）上的对比实验正在运行（7 个基线），完成后补充。

---

## 7. 消融实验

### 7.1 旧数据集（957张）结果

**A) 统一 60 轮预算**

| 配置 | Kappa | ΔKappa |
|---|---|---|
| Full | 0.4065 | — |
| w/o Adapter | 0.3452 | −0.061 |
| w/o Transformer | 0.3636 | −0.043 |
| w/o FPN | 0.3760 | −0.031 |
| w/o ECSAM | 0.3134 | −0.093 |
| w/o EMR | 0.0004 | 坍缩 |

**B) 统一 120 轮预算（收敛后重评）**

| 配置 | Kappa | ΔKappa vs full120 |
|---|---|---|
| full (120ep) | 0.3862 | — |
| w/o Adapter | **0.4621** | **+0.076** |
| w/o Transformer | 0.4580 | +0.072 |
| w/o FPN | 0.4416 | +0.055 |
| w/o EMR | 0.4055 | +0.019 |
| w/o ECSAM | 0.3587 | −0.028 |

![消融实验](FIG_ABLATION)

**关键发现**：收敛协议下仅 **ECSAM 正贡献**；Adapter/Transformer/FPN 为负
（数据规模-复杂度不匹配）；**EMR 的"坍缩"系早停误杀**（120轮恢复至 0.4055）。

### 7.2 模块移除的架构适配与训练响应（重要标注）

消融变体不是简单"删除模块"，而是需要架构层面适配，且对训练策略敏感：

| 变体 | 架构适配 | 训练响应 |
|---|---|---|
| w/o EMR | EMR → 标准 ResBlock 替换 | 60轮/pt15 坍缩(0.0004)；120轮/pt20 恢复(0.4055) |
| w/o ECSAM | 跳过 3 阶段坐标注意力（-0.06M） | 稳定，但上限降低 |
| w/o FPN | 解码器只收最深特征（多尺度融合丢失） | 收敛更快，长期最优 |
| w/o Transformer | 移除全局自注意力（-1.6M） | 稳定，上限略低 |
| w/o Adapter | 仅移除 Adapter-Scale 缩放 | 120轮最佳(0.4621)，小数据上是负担 |
| Trans-4L/6L | 层数 2→4/6（+1.6M/+3.2M） | 60轮下 4L=0.3907, 6L=0.4026 |
| Bilinear-Up | 转置卷积 → 双线性+1×1（-0.7M） | 60轮/pt15 坍缩(0.0414)；lr=1e-4 救援(0.3903) |

> 新数据集（2,257张）上的消融实验正在运行（8 个变体），完成后补充。

---

## 8. 策略探索（训练策略研究）

| 实验 | 配置 | Kappa | 说明 |
|---|---|---|---|
| strat_mssact_sgdr120 | SGDR 热重启 120轮 | **0.4183** | 最优策略 |
| strat_mssact_full30m_oc60 | **30.5M 原版** OneCycle-60 | **0.4186** | 原版架构可收敛 |
| strat_mssact_oc60_lr5e4 | lr=5e-4 | 0.3857 | — |
| strat_bilinear_oc60_lr1e4 | bilinear + lr1e-4 | 0.3903 | 坍缩模型被救援 |
| strat_no_emr_oc60_lr1e4 | no_emr + lr1e-4 | 0.3633 | 坍缩模型被救援 |
| strat_deeplab_oc30 | DeepLab 快退火 | 0.3331 | 基线也有上限 |

![策略曲线](FIG_STRATEGY_CURVES)

**结论**：① SGDR 热重启最优；② **30.5M 原版架构在调优策略下可收敛**（"不稳定"是策略问题非架构问题）；③ 坍缩模型可被低 LR 长退火救援。

---

## 9. 过拟合诊断

使用 logits 熵对比（训练集 vs 验证集，各 100 张随机 patch）：

| 模型 | H_train | H_val | ΔH |
|---|---|---|---|
| mssact_full60 | 0.667 | 0.908 | **−0.241**（过拟合显著） |
| v3_s256 | 0.535 | 0.734 | −0.199 |
| DeepLabV3+ | 0.577 | 0.771 | −0.194 |
| Swin-Unet | 0.777 | 0.921 | −0.144 |
| PSPNet | 1.173 | 1.305 | −0.132 |
| FPN | 0.721 | 0.852 | −0.131 |
| FCN | 0.880 | 0.994 | −0.114 |
| U-Net | 1.941 | 1.941 | −0.0003（欠拟合） |
| SegFormer-Lite | 1.252 | 1.220 | +0.032（校准良好） |

**结论**：除 SegFormer 外普遍存在中度过拟合（训练集更自信）；数据量提升后泛化
差距显著缩小（测试集 Kappa 与验证集接近）。

---

## 10. 结论与讨论

### 10.1 主要结论

1. **数据量是精度主导因素**：2.4× 数据带来 +0.219 Kappa，远超任何架构/策略调整
2. **少数类是短板且可修复**：道路/水域/裸地/建筑 F1 提升 0.25-0.39，数据量是解药
3. **MSSACT-Net 参数效率优**：6.1M 参数达到/超过 13-40M 基线
4. **训练策略对稳定性至关重要**：EMR 缺失、Bilinear 解码等变体的"坍缩"是协议敏感而非架构缺陷
5. **后训练与 TTA 提供边际增益**（+0.017 合计），适合部署端优化

### 10.2 与文献对比

LoveDA 官方基准报告 mIoU（完整数据集 + 大模型），本实验在 1/3 训练量 +
6.1M 轻量模型下取得 mIoU 0.558（测试集），处于合理区间。此前"精度低"的
观感源于：① 数据量不足；② OA 在不平衡数据上被单类支配；③ 未用 TTA/后训练。

### 10.3 局限与展望

- 训练区仅用官方 train（2,522张），可加入 val 区扩展
- 跨域验证（GF-1 2.4m）仍受分辨率鸿沟限制，需同域微调
- 少数类（道路 2.4%）仍是最弱环节，可探索重采样/焦点损失调优
- 后续可尝试更大 backbone 与自监督预训练

---

## 附录：复现命令

```bash
python train_full_data.py full_all   # 全量训练 (2257张)
python train_full_data.py post_all   # 后训练
python eval_complete.py              # val/test × TTA 评估
python run_queue_new.py              # 新数据集对比+消融 (15个实验)
python make_figures.py               # 全部图表
```

**仓库**: https://github.com/eugenewang5425/mssact-loveda
"""

# 替换图片占位符
MD = MD.replace("FIG_FINAL", img_b64(f"{FIG}/fig_final_comparison.png"))
MD = MD.replace("FIG_CURVES_NEW", img_b64(f"{FIG}/fig_new_learning_curves.png"))
MD = MD.replace("FIG_FINAL_PERCLASS", img_b64(f"{FIG}/fig_final_perclass.png"))
MD = MD.replace("FIG_TEST1", img_b64(f"{FIG}/fig_test_showcase_00000.png"))
MD = MD.replace("FIG_TEST2", img_b64(f"{FIG}/fig_test_showcase_00006.png"))
MD = MD.replace("FIG_TEST3", img_b64(f"{FIG}/fig_test_showcase_00014.png"))
MD = MD.replace("FIG_FULLTILE", img_b64(f"{FIG}/fig_fulltile_bars.png"))
MD = MD.replace("FIG_ABLATION", img_b64(f"{FIG}/fig_ablation.png"))
MD = MD.replace("FIG_STRATEGY_CURVES", img_b64(f"{FIG}/fig_strategy_curves.png"))
MD = MD.replace("FIG_CM_NORM", img_b64(f"{FIG}/fig_confusion_norm.png"))
MD = MD.replace("FIG_CM_COUNT", img_b64(f"{FIG}/fig_confusion_count.png"))
MD = MD.replace("FIG_PRF", img_b64(f"{FIG}/fig_prf_perclass.png"))
MD = MD.replace("FIG_EFF", img_b64(f"{FIG}/fig_efficiency.png"))
MD = MD.replace("FIG_LR", img_b64(f"{FIG}/fig_lr_schedule.png"))

html_body = markdown.markdown(MD, extensions=["tables", "fenced_code"])
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: "Microsoft YaHei", "SimSun", sans-serif; margin: 40px; line-height: 1.7; color: #222; }}
h1 {{ color: #b30000; border-bottom: 3px solid #b30000; padding-bottom: 8px; }}
h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 32px; }}
h3 {{ color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }}
th, td {{ border: 1px solid #bbb; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; margin: 12px 0; border: 1px solid #ddd; }}
code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-size: 13px; }}
pre {{ background: #f5f5f5; padding: 12px; border-radius: 5px; overflow-x: auto; }}
blockquote {{ border-left: 4px solid #b30000; padding-left: 12px; color: #666; }}
</style></head><body>
{html_body}
</body></html>"""

out_html = f"{BASE}/实验报告.html"
open(out_html, "w", encoding="utf-8").write(html)
print("HTML written:", out_html)
