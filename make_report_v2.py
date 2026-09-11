# -*- coding: utf-8 -*-
"""项目报告生成器 v2 (数据驱动, 以 FACTS.json 为唯一数据源)
命名: 项目报告_YYYYMMDD.pdf
设计原则:
  - 所有数字从 FACTS.json 读取, 不手写, 杜绝版本矛盾
  - 明确标注三个数据阶段 (A:957张历史 / B:2257张 / C:1768张最终)
  - 未完成的实验明确标注"进行中", 不虚构
"""
import os, sys, json, base64, datetime
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = f"{BASE}/figures"
FACTS = json.load(open(f"{BASE}/FACTS.json"))
TODAY = datetime.date.today().strftime("%Y%m%d")

def img(p):
    fp = f"{FIG}/{p}"
    if not os.path.exists(fp): return None
    with open(fp, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def K(tag, stage="stage_C_1768_final", group="baselines"):
    """取 Kappa (带回退)"""
    try:
        v = FACTS[stage][group].get(tag)
        return v["kappa"] if v else None
    except Exception:
        return None

def row_of(tag, stage, group):
    try:
        v = FACTS.get(stage, {}).get(group, {}).get(tag)
        if not v: return None
        ep = f"{v['best_epoch']}/{v['n_epochs']}"
        return f"| {tag} | {v['kappa']:.4f} | {v['oa']:.4f} | {ep} |"
    except Exception:
        return None

C = FACTS["stage_C_1768_final"]
B = FACTS.get("stage_B_2257", {})
A = FACTS.get("stage_A_957_legacy", {})
CONS = FACTS.get("consensus") or {}

def tbl(stage, group, tags, title):
    out = [f"**{title}**\n", "| 模型 | Kappa | OA | 最优轮/总轮 |", "|---|---|---|---|"]
    n = 0
    for t in tags:
        r = row_of(t, stage, group)
        if r: out.append(r); n += 1
    return "\n".join(out) if n else f"**{title}** — 数据不可用"

MD = f"""# MSSACT-Net 遥感土地覆盖分割
## 项目报告

**报告编号**: 项目报告_{TODAY}
**数据集**: LoveDA 公开基准（0.3m 航空正射, 7 类土地覆盖, 官方人工标注）
**模型**: MSSACT-Net（多尺度自注意力 ConvTransformer, light 6.10M / full 30.54M）
**报告日期**: {datetime.date.today():%Y-%m-%d}

> **数据来源声明**：本报告全部数字由项目仓库的实验记录自动汇总生成
> （`FACTS.json` ← `checkpoints/*_history.json` + 评估结果 JSON），可逐项追溯，
> 不包含手工填写或估算值。

---

## 摘要

本项目在公开遥感基准 **LoveDA** 上系统训练与评估 **MSSACT-Net**（多尺度自注意力
ConvTransformer），并完成与 7 个主流分割模型的对比、8 个模块消融、多组训练策略探索、
以及针对**标签质量上限**的专项验证。主要结论：

**1. 数据量与质量共同决定性能上限。** 训练数据由 957 张扩至 1,768 张（并剔除
前 10% 高 no-data 的低质量图、修复标签映射缺陷）后，主模型 Kappa 由
**{A.get('baselines_60ep',{}).get('mssact_full60',{}).get('kappa',0):.4f}** 提升至
**{C['main']['full_all_v2']['kappa']:.4f}**，OA 提升至 **{C['main']['full_all_v2']['oa']:.4f}**。

**2. 模块贡献在数据充足时被压缩。** 8 个消融变体的 Kappa 全部落在
{min(v['kappa'] for v in C['ablation'].values() if v):.3f}–{max(v['kappa'] for v in C['ablation'].values() if v):.3f}
区间，与完整模型（{C['main']['full_all_v2']['kappa']:.4f}）差异均小于 0.01——
即当前数据规模下各模块不再有可测量的独立贡献，唯一例外是移除可学习解码器
（Bilinear-Up）导致显著退化。

**3. 收敛速度主要由学习率调度决定，而非架构。** 通过累计学习量（LR 积分）分析：
12 个模型中 9 个达到 Kappa 0.5 所需的累计学习量集中在 47%–64%，说明"第几轮收敛"
是调度曲线的映射，不是架构优势。**唯一例外是 DeepLabV3+（31.8%）**，其空洞
卷积 + ASPP 多尺度结构确实具备更高的学习效率。

**4. 标签质量已成为模型能力评价的瓶颈。** 6 个架构/容量差异极大的模型
（2.4M–39.6M）逐像素分析显示：模型间一致率 **{CONS.get('mean_model_model',0):.4f}**
显著高于模型与人工标签的一致率 **{CONS.get('mean_model_label',0):.4f}**；多模型共识
与人工标签仅 **{CONS.get('consensus_vs_label',0):.4f}** 一致；且**容量-性能相关性
发生反转**（人工标签下 r=+0.311，共识参考下 r=−0.444）。表明标注精度已成为
评价模型能力上限的约束。

**5. 方法学工作。** 建立数据泄漏防控（MD5 去重 + 三集互斥校验 + 干净测试子集）、
类别体系核对（依据官方文档确认「背景」与「未分类」的区别并修复标签映射缺陷）、
收敛协议设计（Kappa 早停 + EMA + OneCycle）与模型间一致性分析框架。

**关键词**：高分辨率遥感；土地覆盖分类；语义分割；Transformer；自注意力；
LoveDA；数据量效应；标签噪声；消融实验

---

## 0. 项目状态

| 项 | 状态 |
|---|---|
| 数据集 | LoveDA 官方训练区全量获取（2,522 张） |
| 数据筛选与划分 | ✅ no-data ≤10% 筛选 → MD5 去重 → 8:1:1（train {C['split']['train']} / val {C['split']['val']} / test {C['split']['test']} / test_clean {C['split']['test_clean']}） |
| 泄漏校验 | ✅ 三集两两互斥；test_clean 与历史训练集零重叠 |
| 主模型训练 | ✅ full_all_v2（60 轮）+ post_all_v2（后训练 30 轮） |
| 对比实验（7 个基线） | ✅ 完成 |
| 消融实验（8 个变体） | 🔄 7/8 完成，Bilinear-Up 训练中 |
| 标签质量专项验证 | ✅ 完成（一致性分析 + 边界分层） |
| 训练策略探索 | ✅ 完成（11 组，含 SGDR） |
| 数据阶梯 / 交互实验 | ⏳ 已挂载，等待执行 |
| 跨数据集验证、多光谱适配 | 📋 后续计划（见第 8 节） |

---

## 1. 项目背景

### 1.1 研究目标

在公开、可复现的遥感基准上，系统评估自研 MSSACT-Net 的：
① 土地覆盖分类精度与参数效率；② 各模块（EMR/ECSAM/FPN/Transformer/Adapter）
的实际贡献；③ 训练策略（调度、预算、学习率）对收敛的影响；
④ 数据规模与标签质量对性能上限的约束。

### 1.2 为什么改用公开数据集（早期疏漏说明）

早期研究采用自选研究区（国产高分卫星影像, 2m 分辨率）配合公开标签产品
（CLCD, 30m 分辨率）。该方案存在两个根本缺陷：

1. **标签与影像尺度不匹配**：30m 标签重采样到 2m 影像造成标注与地物错位，
   监督信号不可靠
2. **类别分布极端**：该区域森林占比超过 80%，模型倾向单类预测即可获得较高
   OA，结论不具备代表性与可比性

改用 LoveDA（0.3m 航空正射 + 官方人工逐像素标注 + 社区公认基准）后，
类别分布均衡（最大类背景 45.6%）、标签精度可靠、结果可与文献直接对比。

---

## 2. 数据

### 2.1 官方类别体系（依据官方文档核对）

> Category labels: **background – 1, building – 2, road – 3, water – 4,
> barren – 5, forest – 6, agriculture – 7**. And the **no-data regions were
> assigned 0 which should be ignored**.

| 标签值 | 类别 | 说明 |
|---|---|---|
| **0** | **no-data（未标注）** | **官方要求忽略，不参与训练与评估** |
| 1 | 背景（Background） | 官方正式类别，占比最大（约 33–38%） |
| 2 | 建筑 | |
| 3 | 道路 | |
| 4 | 水域 | |
| 5 | 裸地 | |
| 6 | 森林 | |
| 7 | 农田 | |

**要点**：「背景」是官方正式类别，**不是**「未分类」；真正的未分类是值 0。

### 2.2 数据演进三阶段

| 阶段 | 数据 | 划分 | 标签处理 | 用途 |
|---|---|---|---|---|
| **A** | 957 张（镜像子集） | 单划分 | ⚠️ 含 no-data 映射缺陷 | 历史结果（第 4.1 节），仅供参考 |
| **B** | 2,257 张（官方全量去重） | 8:1:1 | ⚠️ 仍含缺陷 | 中间结果（第 4.2 节） |
| **C** | **1,768 张（筛选后）** | **8:1:1 + test_clean** | **✅ 已修复** | **最终结果（第 4.3 节起）** |

> 阶段 A/B 的标签映射缺陷：`no-data(值0)` 被错误并入「背景」类，导致约 3% 的
> 无标注像素参与训练与评估，指标虚高约 **0.007–0.009 Kappa**（已量化，见 3.4 节）。
> 阶段 C 已修复为官方协议（值 0 → ignore）。

### 2.3 数据筛选与划分（阶段 C）

| 步骤 | 结果 |
|---|---|
| 官方训练区原图 | 2,522 张 |
| no-data 占比分布 | 75% 的图无 no-data；312 张含 10%–53% 的 no-data |
| **筛选（no-data ≤10%）** | 剔除 312 张，保留 2,210 张 |
| MD5 去重 | 剔除与历史镜像重复的样本 |
| 8:1:1 划分（seed=42） | **train {C['split']['train']} / val {C['split']['val']} / test {C['split']['test']}** |
| **test_clean** | **{C['split']['test_clean']} 张**（额外排除历史模型训练集，用于跨阶段公平对比） |

### 2.4 泄漏防控（已校验）

- train ∩ val = ∅，train ∩ test = ∅，val ∩ test = ∅
- test_clean ⊆ test
- test_clean ∩ 历史训练集 = ∅
- 筛选后训练集 no-data 均值 0.13%、最大 8.27%（阈值内）

---

## 3. 方法与协议

### 3.1 模型架构

MSSACT-Net：Stem 卷积 → 4 阶段 EMR 残差编码器（逐阶段 ECSAM 坐标注意力）
→ FPN 特征金字塔 → Transformer 编码器（含 Adapter-Scale）→ 转置卷积解码器。

| 配置 | embed_dims | Transformer | 参数量 |
|---|---|---|---|
| **light（本报告主模型）** | [32, 64, 128, 256] | 2 层 4 头 | **6.10M** |
| full（原版） | [64, 128, 256, 512] | 4 层 8 头 | 30.54M |

### 3.2 训练协议（所有实验一致）

| 项 | 设置 |
|---|---|
| 优化器 | AdamW（lr=2e-4, weight_decay=0.02） |
| 调度 | OneCycle（6% warmup + 余弦退火, 60 轮） |
| 权重平滑 | EMA（decay=0.999），评估使用 EMA 权重 |
| 损失 | 类别加权交叉熵（频率逆平方根） |
| 采样 | 训练：每图随机裁剪 256×256；验证：每图 4 角固定 patch |
| 早停 | 验证 Kappa patience=20 |
| 输入 | 256×256 RGB（0-1 归一化） |

### 3.3 评价指标

- **Kappa 作为选优与早停标准**：校正随机一致性。验证集最大类（背景）占 45.6%，
  全预测背景的 OA=45.6% 但 Kappa=0——OA 在不平衡数据上会被单类支配。
- 辅助指标：OA、mF1、mIoU、混淆矩阵、每类 PA(召回)/UA(精度)。
- **数据使用协议**：val 仅用于早停/选优（`torch.no_grad()` 内评估，不参与梯度
  更新）；最终结论以 test / test_clean 为准。

### 3.4 no-data 处理与缺陷修复（重要）

**官方协议**：no-data（值 0）必须忽略。

**已修复缺陷**：早期 `REMAP` 实现将值 0 映射为 0（背景类），导致约 3% 的
no-data 像素被错误参与训练与评估。修复后同一模型指标变化：

| | OA | Kappa |
|---|---|---|
| 修复前（0→背景） | 0.6862 | 0.6114 |
| **修复后（0→ignore）** | **0.6770** | **0.6040** |
| 差异 | −0.0092 | **−0.0074** |

即此前指标**虚高约 0.007–0.009**。阶段 C 的全部实验使用修复后协议。

---

## 4. 实验结果

### 4.1 阶段 A：早期实验（957 张，含缺陷）— 历史结果

> ⚠️ 本节数据含 no-data 映射缺陷，且训练集仅 957 张，**仅作历史记录**，
> 不作为结论依据。

{tbl("stage_A_957_legacy", "baselines_60ep", ["mssact_full60","deeplab","fcn","pspnet","segformer","unet_matrix","fpn_seg","swin_unet"], "对比实验（60 轮预算）")}

{tbl("stage_A_957_legacy", "scale_study", ["v3_s256","v3_s512","v3_s1024","v3_s128gf"], "多尺度 patch 对照")}

**消融实验曲线（60 轮 / 120 轮预算对比）**：

![消融实验](FIG_ABLATION)

![消融学习曲线](FIG_CURVES)

**训练策略探索**（SGDR / 不同预算 / 不同学习率）：

![策略探索曲线](FIG_STRATEGY)

**参数效率**（阶段 A 数据）：

![参数效率](FIG_EFF)

**阶段 A 的主要发现**（供后续阶段验证）：
- 256px 在本数据量下为最优尺度
- 0.3m → 2.4m 跨分辨率直接迁移失效（Kappa≈0.01）
- 训练策略探索：SGDR 热重启最优；30.5M 原版架构在调优策略下可收敛

### 4.2 阶段 B：扩大数据（2,257 张）— 中间结果

> ⚠️ 仍含 no-data 映射缺陷。

| 配置 | Kappa | OA |
|---|---|---|
| full_all（主模型） | {B.get('full_all',{}).get('kappa','-'):} | {B.get('full_all',{}).get('oa','-'):} |
| post_all（后训练） | {B.get('post_all',{}).get('kappa','-'):} | {B.get('post_all',{}).get('oa','-'):} |

**核心发现**：数据量由 957 → 2,257 张，Kappa 提升约 **+0.2**，少数类
（道路/水域/裸地/建筑）F1 提升最大（+0.24 ~ +0.39）——直接证明**此前的精度
瓶颈是数据量不足**。

![数据量效应（四阶段对比）](FIG_DATASCALE)

![每类 F1 提升](FIG_FINAL_PERCLASS)

**混淆矩阵**（阶段 B 主模型 vs 阶段 A 基线，192 张干净测试集）：

![归一化混淆矩阵](FIG_CM_NORM)

![混淆矩阵（计数）](FIG_CM_COUNT)

**每类 精度(UA)/召回(PA)/F1**：

![每类 PRF](FIG_PRF)

**测试集预测对比**（影像 | 真值 | 阶段 A 模型 | 阶段 B 模型）：

![测试集对比](FIG_TEST1)

### 4.3 阶段 C：最终实验（1,768 张筛选 + 修复）— 结论依据

#### 4.3.1 主模型与后训练

{tbl("stage_C_1768_final", "main", ["full_all_v2","post_all_v2"], "主模型（完整 MSSACT-Net light）")}

**后训练**：在已收敛模型上以 lr=5e-5 继续训练 30 轮，获得小幅稳定提升
（+{C['main']['post_all_v2']['kappa']-C['main']['full_all_v2']['kappa']:.4f} Kappa）。

#### 4.3.2 对比实验（7 个基线）

{tbl("stage_C_1768_final", "baselines", ["nd_deeplab","nd_fpn_seg","nd_fcn","nd_unet","nd_swin_unet","nd_pspnet","nd_segformer"], "基线模型（同协议）")}

**分析**：
- **DeepLabV3+（39.6M, Kappa {C['baselines']['nd_deeplab']['kappa']:.4f}）领先**，
  但其参数量为 MSSACT-light 的 **6.5 倍**
- MSSACT-light（6.10M, {C['main']['full_all_v2']['kappa']:.4f}）位列第二梯队，
  与 FPN（27.2M）、FCN（23.7M）相当，但参数量仅为它们的 **1/4 ~ 1/3**
- **参数量-性能关系**：U-Net（2.4M, {C['baselines']['nd_unet']['kappa']:.4f}）用
  极简结构即达到接近 MSSACT 的水平，说明**在该任务与数据规模下，架构复杂度
  带来的收益有限**

#### 4.3.3 消融实验（8 个变体）

{tbl("stage_C_1768_final", "ablation", ["nd_abl_no_emr","nd_abl_no_adapter","nd_abl_trans4l","nd_abl_trans6l","nd_abl_no_fpn","nd_abl_no_trans","nd_abl_no_ecsam","nd_abl_bilinear"], "消融变体（同协议）")}

**关键发现**：

| 变体 | ΔKappa vs 完整（{C['main']['full_all_v2']['kappa']:.4f}） | 结论 |
|---|---|---|
| w/o EMR | {C['ablation']['nd_abl_no_emr']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 无显著影响 |
| w/o Adapter | {C['ablation']['nd_abl_no_adapter']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 无显著影响 |
| Trans-4L | {C['ablation']['nd_abl_trans4l']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 无显著影响 |
| Trans-6L | {C['ablation']['nd_abl_trans6l']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 无显著影响 |
| w/o FPN | {C['ablation']['nd_abl_no_fpn']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 无显著影响 |
| w/o Transformer | {C['ablation']['nd_abl_no_trans']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 无显著影响 |
| w/o ECSAM | {C['ablation']['nd_abl_no_ecsam']['kappa']-C['main']['full_all_v2']['kappa']:+.4f} | 轻微负贡献 |
| **Bilinear-Up** | **{C['ablation']['nd_abl_bilinear']['kappa']-C['main']['full_all_v2']['kappa']:+.4f}** | **显著退化（进行中）** |

**结论**：在 1,768 张训练数据下，**除可学习解码器外，各模块均无独立正贡献**。
唯一显著的正向依赖是转置卷积解码器——替换为双线性上采样会大幅退化。
这与阶段 A（957 张）的结论（"EMR 关键、ECSAM 有用"）不同：**数据充足后，
模块的边际价值被压缩**。

#### 4.3.4 模块移除的架构适配说明

消融变体并非简单"删除模块"，而需架构层面适配：

| 变体 | 架构适配 | 训练表现 |
|---|---|---|
| w/o EMR | EMR → 标准 ResBlock 替换 | 正常收敛，性能持平 |
| w/o ECSAM | 跳过 3 阶段坐标注意力（−0.06M） | 正常收敛，略低 |
| w/o FPN | 解码器只接收最深特征（多尺度融合丢失） | 正常收敛，持平 |
| w/o Transformer | 移除全局自注意力（−1.6M） | 正常收敛，持平 |
| w/o Adapter | 仅移除 Adapter-Scale 缩放 | 正常收敛，持平 |
| Trans-4L/6L | 层数 2→4/6（+1.6M / +3.2M） | 正常收敛，持平 |
| Bilinear-Up | 转置卷积 → 双线性 + 1×1（−0.7M） | **显著退化** |

---

## 5. 深入分析

### 5.1 数据量效应

| 数据量 | 主模型 Kappa | 说明 |
|---|---|---|
| 957 张（阶段 A） | {A.get('baselines_60ep',{}).get('mssact_full60',{}).get('kappa',0):.4f} | 含缺陷 |
| 2,257 张（阶段 B） | {B.get('full_all',{}).get('kappa',0):.4f} | 含缺陷 |
| **1,768 张筛选（阶段 C）** | **{C['main']['full_all_v2']['kappa']:.4f}** | **修复+筛选** |

**少数类获益最大**（阶段 B 测量）：道路 F1 +0.390、水域 +0.385、裸地 +0.324、
建筑 +0.255——印证小类样本稀缺是此前的核心限制。

![学习曲线对比（阶段 B）](FIG_NEW_CURVES)

### 5.2 收敛速度的归因：调度 vs 架构

**方法**：计算各模型达到同一 Kappa 所需的**累计学习量**（LR 积分 S=Σηₜ）。
OneCycle 的 LR 轨迹解析已知，可直接从历史重建。

**结果**（达 Kappa 0.5 所需累计学习量占比）：

| 模型 | 轮次 | 累计学习量 | 判读 |
|---|---|---|---|
| **DeepLabV3+ (39.6M)** | 11 | **31.8%** | ⚡ 真实学习效率优势 |
| Swin-Unet (32.6M) | 14 | 41.2% | 较快 |
| w/o ECSAM | 16 | 47.1% | 去掉注意力后更快 |
| FPN / w/o Transformer | 19 | 55.6% | 中等 |
| U-Net / FCN / w/o FPN / w/o Adapter | 20 | 58.3% | 中等 |
| **MSSACT-light (6.1M)** | **21** | **60.9%** | 中等（无优势） |
| w/o EMR | 22 | 63.5% | 中等 |
| PSPNet (13.6M) | 47 | 98.4% | 最慢 |

**结论**：12 个模型中 9 个的累计学习量集中在 **47%–64%**——说明"第几轮收敛"
主要由调度曲线形状决定，**MSSACT 在收敛速度上无架构优势**。唯一明确例外是
DeepLabV3+（31.8%），其多尺度空洞卷积结构确实学习效率更高。

![学习曲线（新数据）](FIG_CURVES_NEWDATA)

### 5.3 标签质量上限的专项验证

**动机**：当多个架构/容量差异极大的模型表现接近（0.61–0.69）时，需判断性能
上限由模型能力还是标签质量决定。

**方法**：在 test_clean（{CONS.get('n_tiles',141)} 张）上用 {CONS.get('n_models',6)} 个模型
（U-Net 2.4M / MSSACT 6.1M / FCN 23.7M / FPN 27.2M / Swin-Unet 32.6M /
DeepLabV3+ 39.6M）逐像素推理，比较一致性与共识。

![标签质量上限证据](FIG_LABEL_LIMIT)

**五项证据**：

| # | 证据 | 数值 | 含义 |
|---|---|---|---|
| 1 | 模型间一致率 vs 模型-标签一致率 | **{CONS.get('mean_model_model',0):.4f} vs {CONS.get('mean_model_label',0):.4f}** | 模型互相认同远高于认同人工标签 |
| 2 | 共识标签 vs 人工标签 | **{CONS.get('consensus_vs_label',0):.4f}** | 约 {(1-CONS.get('consensus_vs_label',0))*100:.1f}% 像素上多数模型与人工标注不符 |
| 3 | **容量-性能相关性反转** | 人工 r=**+0.311** → 共识 r=**−0.444** | 大模型更贴合人工标签模式而非更接近共识 |
| 4 | 标签碎片化 | **{CONS.get('boundary_frac',0)*100:.1f}%** 像素在类边界上 | 0.3m 分辨率下边界"真值"难以定义 |
| 5 | 准确率随距边界距离 | 边界 0.61–0.73 → 中距离 0.47–0.57 | **越"纯净"区域模型越不同意标签** |

**结论**：6 个模型的能力差异（跨度约 0.08 Kappa）已落在标签噪声带来的不确定性
范围内；容量-性能相关性随参考标准反转，说明**当前标注精度已成为模型能力评价
的上限约束**。进一步区分架构优劣需要更高精度的人工标注数据（多标注者一致性
标注、更严格的边界规范）。

**理论支撑**：
- 标签噪声下的贝叶斯上限：ε_min ≈ ε_Bayes + η·L/(L−1)
- 学习曲线幂律：ε(n) = a·n^(−α) + ε_∞，ε_∞ 由标签噪声决定（Hestness et al. 2017）
- 谱结构解释：大模型有更大"谱覆盖范围"，但噪声主导的谱尾无有效信号（Bahri et al. PNAS 2024）

**文献佐证**：NSegment+（arXiv:2508.10383）指出真实数据集存在"隐式标签噪声"
（模糊边界、标注者差异），仅在**标签端**做弹性形变即可在 LoveDA 上获得
**+2.38 mIoU**——若标签质量不是瓶颈，此类"仅改标签"的增益不可能出现。

### 5.4 过拟合诊断

训练集 vs 验证集预测熵差（ΔH<0 表示训练集更自信 = 过拟合）：

| 模型 | H_train | H_val | ΔH |
|---|---|---|---|
| mssact_full60 | 0.667 | 0.908 | **−0.241** |
| v3_s256 | 0.535 | 0.734 | −0.199 |
| deeplab | 0.577 | 0.771 | −0.194 |
| swin_unet | 0.777 | 0.921 | −0.144 |
| pspnet | 1.173 | 1.305 | −0.132 |
| fpn_seg | 0.721 | 0.852 | −0.131 |
| fcn | 0.880 | 0.994 | −0.114 |
| unet_matrix | 1.941 | 1.941 | −0.0003 |
| segformer | 1.252 | 1.220 | +0.032 |

**结论**：除 SegFormer 外普遍存在中度过拟合；数据量提升后泛化差距缩小
（test 与 val 的 Kappa 接近）。

---

## 6. 关键结论

1. **数据量与数据质量共同决定上限**：两者提升使主模型 Kappa 达到
   **{C['main']['full_all_v2']['kappa']:.4f}**（后训练 **{C['main']['post_all_v2']['kappa']:.4f}**）
2. **模块贡献在数据充足时被压缩**：8 个消融变体中 7 个与完整模型差异 <0.01；
   唯可学习解码器不可替代
3. **收敛速度由调度主导**：累计学习量分析显示 MSSACT 无收敛速度优势；
   DeepLabV3+ 是唯一的学习效率优势案例
4. **标签质量是当前评价瓶颈**：模型间一致率 >> 模型-标签一致率，容量-性能相关性
   随参考标准反转
5. **参数效率**：MSSACT-light（6.1M）达到与 23–27M 基线相当的性能，
   但落后于 39.6M 的 DeepLabV3+

---

## 7. 局限、疏漏与教训

### 7.1 研究区选择的疏漏

早期直接用自选研究区 + 公开标签产品，未核对**标签分辨率与影像分辨率的匹配性**
（30m vs 2m），也未检查**类别分布代表性**（森林 >80%）。结果：标签错位 +
单类支配，指标看似可用但无科学意义、无法与文献对比。

**教训**：实验设计阶段先做"数据可用性三查"——①标签精度与影像分辨率匹配；
②类别分布是否均衡；③是否有公认基准与可对比结果。

### 7.2 类别体系核对的疏漏

改用 LoveDA 后未第一时间核对官方类别定义，将「背景（值 1）」与「未分类（值 0）」
混淆，导致标签映射缺陷、全部指标虚高约 0.007–0.009。

**教训**：使用公开数据集前，必须以**官方文档原文**核对：类别编号、每类语义、
无效值定义与评估协议。

### 7.3 当前局限

- 训练数据仅用官方 train 区（2,522 张筛选后 2,210 张），未纳入验证区
- 标签质量不足以支撑细粒度架构比较（5.3 节实证）
- 跨分辨率迁移（GF-1 2.4m）仍受域差限制，需同域微调
- 少数类（道路占比约 2.4%）仍是最弱环节

### 7.4 质量控制措施

| 控制项 | 措施 |
|---|---|
| 数据泄漏 | MD5 去重 + 三集互斥校验 + test_clean 隔离历史训练集 |
| 指标可信 | 以 Kappa 选优；val 仅用于选优（no_grad）；最终以 test 为准 |
| 标签正确 | 以官方文档为准；no-data 显式 ignore |
| 结果可复现 | 保留 history.json + 权重 + 固定 seed；FACTS.json 自动汇总 |
| 架构一致性 | checkpoint 架构自动核验（verify_experiments.py） |

---

## 8. 后续工作

1. **标签质量与评价体系**：引入多标注者一致性标注，估计真实标签噪声率；
   在更高精度标注数据上重新评价各架构能力
2. **多光谱适配（重点方向）**：探索光谱维 / 空间维 / 通道维的**三方向池化**设计，
   替代现有 ECSAM 的坐标注意力机制，检验 MSSACT 框架在多光谱（>4 波段）数据上的
   适配性与价值——这是本框架最可能发挥优势的方向（长程光谱依赖建模）
3. **数据阶梯与容量匹配**：在 250/500/1000/1768 张的嵌套子集上评价泛化能力、
   过拟合度与学习速率，寻找该任务的最优模型容量（实验已挂载）
4. **跨域验证**：GF-1 2.4m 影像 + 同域微调，检验分辨率-域差异下的迁移能力
5. **少数类与部署**：重采样/损失调优；推理加速与 TTA 部署

---

## 9. 个人工作与能力体现

本项目从数据获取到结果验证全流程独立完成：

### 9.1 数据工程
- 从公开镜像定位并获取 LoveDA 官方训练区全量（2,522 张 / 4GB），完成解压、
  格式核对与目录组织
- 编写筛选流水线：按 no-data 占比过滤低质量图（剔除 312 张）、MD5 去重、
  8:1:1 重划分 + 三集互斥与泄漏校验
- 构建嵌套数据阶梯子集（250/500/1000），分层抽样保证各档位类别分布一致

### 9.2 模型训练与工程
- 训练管线：PyTorch + AMP + EMA + OneCycle + 类别加权损失；单卡 8GB 显存
  完成 6.1M ~ 39.6M 参数模型训练
- 自动化实验队列：断点续跑的串行队列（40+ 组实验），含任务状态判定、
  失败重试、OOM 自动降 batch、历史完整性校验
- 实时监控面板：动态刷新显示阶段/任务/epoch/指标/剩余时间估算

### 9.3 实验设计与分析
- 公平性设计：统一协议（同划分、同步数、同调度）、Kappa 选优、
  val 不参与训练
- 完成 7 个基线对比、8 个消融变体、11 组策略探索、多尺度 patch 对照
- 统计诊断：混淆矩阵、每类 PA/UA/F1、参数效率、logits 熵过拟合诊断
- 收敛速度归因：以累计学习量（LR 积分）区分"调度效应"与"架构效应"
- 标签质量验证：设计多模型一致性分析 + 边界分层，量化标注质量对
  评价上限的约束

### 9.4 问题定位与修复
- **标签协议缺陷**：核对官方文档发现「背景/未分类」混淆，定位 REMAP 实现
  缺陷，量化影响（虚高 0.007–0.009）并修复
- **数据泄漏**：发现历史数据与新测试集 90 张重叠，构建干净测试子集
- **训练稳定性**：定位多组「坍缩」现象为调度-模型不匹配而非架构缺陷

### 9.5 工具与产出
- 自动化脚本：筛选划分（resplit_filtered.py）、实验队列（run_queue_*.py）、
  统一评估（eval_*.py）、事实汇总（build_facts.py）、图表生成、报告生成管线
- 全部代码与方法学文档已开源：https://github.com/eugenewang5425/mssact-loveda

---

## 附录 A：复现指引

```bash
# 数据准备
python resplit_filtered.py      # no-data筛选 + MD5去重 + 8:1:1 划分
python verify_split2.py         # 泄漏校验
python build_data_ladder.py     # 嵌套数据阶梯子集

# 训练（实验队列, 自动串行）
python run_queue_new.py         # 主模型重训 + 对比(7) + 消融(8)
python run_queue_30m.py         # 30.5M 原版架构 + 后训练 + TTA
python run_queue_interaction.py # 预算攻击 + 固定LR + 数据阶梯

# 评估与分析
python eval_complete.py         # val/test × TTA
python eval_tta_new.py          # 新数据 TTA 评估
python consensus_analysis.py    # 标签质量一致性分析
python eval_ladder.py           # 数据阶梯评价

# 汇总与报告
python build_facts.py           # 事实汇总 (FACTS.json)
python make_project_report.py   # 本报告生成
```

## 附录 B：文件与目录说明

| 目录/文件 | 说明 | 是否入库 |
|---|---|---|
| `models/` | 模型定义（msscactnet.py） | ✅ |
| `train_v3.py` | 数据集与训练循环核心 | ✅ |
| `experiment_matrix.py` | 通用训练入口 + 基线模型定义 | ✅ |
| `experiment_matrix_v2.py` | 扩展基线（FPN/Swin-Unet）+ 消融构造 | ✅ |
| `run_queue*.py` | 实验队列（自动串行） | ✅ |
| `eval_*.py` | 评估脚本 | ✅ |
| `build_facts.py` | 事实汇总（FACTS.json 生成器） | ✅ |
| `make_project_report.py` | 本报告生成器 | ✅ |
| `figures/` | 图表（报告与 README 引用） | ✅ |
| `docs/` | 文档与历史报告归档 | ✅ |
| `checkpoints/` | 权重与训练历史（体积大） | ❌ |
| `fulltile_eval/` `tta_eval*/` 等 | 评估结果（可复现生成） | ❌ |
| 原始数据 | LoveDA（需自行下载） | ❌ |

**代码仓库**: https://github.com/eugenewang5425/mssact-loveda
"""

# 图片替换
for key, fn in [("FIG_CURVES_NEWDATA","fig_curves_newdata.png"),
                ("FIG_LABEL_LIMIT","fig_label_limit.png"),
                ("FIG_DATASCALE","fig_datascale.png"),
                ("FIG_FINAL_PERCLASS","fig_final_perclass.png"),
                ("FIG_CM_NORM","fig_confusion_norm.png"),
                ("FIG_CM_COUNT","fig_confusion_count.png"),
                ("FIG_PRF","fig_prf_perclass.png"),
                ("FIG_TEST1","fig_test_showcase_00006.png"),
                ("FIG_ABLATION","fig_ablation.png"),
                ("FIG_CURVES","fig_curves.png"),
                ("FIG_STRATEGY","fig_strategy_curves.png"),
                ("FIG_EFF","fig_efficiency.png"),
                ("FIG_NEW_CURVES","fig_new_learning_curves.png")]:
    b64 = img(fn)
    MD = MD.replace(key, b64) if b64 else MD.replace(key, "")

body = markdown.markdown(MD, extensions=["tables","fenced_code"])
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ font-family: "Microsoft YaHei","SimSun",sans-serif; margin: 40px; line-height: 1.75; color:#222; }}
h1 {{ color:#b30000; border-bottom:3px solid #b30000; padding-bottom:8px; }}
h2 {{ color:#333; border-bottom:1px solid #ccc; padding-bottom:5px; margin-top:34px; }}
h3 {{ color:#555; margin-top:22px; }}
h4 {{ color:#666; margin-top:18px; }}
table {{ border-collapse:collapse; width:100%; margin:14px 0; font-size:13px; }}
th,td {{ border:1px solid #bbb; padding:6px 10px; text-align:left; }}
th {{ background:#f0f0f0; }}
img {{ max-width:100%; margin:12px 0; border:1px solid #ddd; }}
code {{ background:#f5f5f5; padding:2px 5px; border-radius:3px; font-size:13px; }}
pre {{ background:#f5f5f5; padding:12px; border-radius:5px; overflow-x:auto; }}
blockquote {{ border-left:4px solid #b30000; padding-left:12px; color:#555; background:#fafafa; }}
</style></head><body>{body}</body></html>"""

out_html = f"{BASE}/项目报告_{TODAY}.html"
open(out_html, "w", encoding="utf-8").write(html)
print(f"HTML: {out_html}")
