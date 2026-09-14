# MSSACT-Net — 多尺度自注意力遥感土地覆盖分割（LoveDA 基准）

Multi-Scale Self-Attention ConvTransformer (MSSACT-Net) for high-resolution
land-cover semantic segmentation, with systematic evaluation on the public
**LoveDA** benchmark: baseline comparison, module ablation, training-strategy
search, and an empirical study on the **label-quality ceiling** of model evaluation.

> 📄 **项目报告**：[项目报告_20260914.pdf](reports/项目报告_20260914.pdf)（43 页）——
> 含全部实验、图表与公式；[HTML 版](reports/项目报告_20260913.html)（可再生）
>
> 🧭 **导航**：[核心结果](#核心结果) · [重要发现](#重要发现) ·
> [项目结构](#项目结构) · [快速开始](#快速开始) · [后续工作](#后续工作) ·
> [目录整理记录](docs/整理记录_20260913.md) · [报告问题研究](docs/报告问题研究_20260913.md) ·
> [项目阶段评估](docs/项目阶段评估_20260913.md) · [架构有效性分析](docs/架构有效性分析.md) · [缺陷修复记录](docs/缺陷修复_20260913.md)
>
> **数据来源声明**：本 README 全部数字由仓库实验记录自动汇总生成
> （`build_facts.py` → `FACTS.json` ← `checkpoints/*_history.json` + 评估结果 JSON），
> 可逐项追溯，不含手工填写或估算值。

---

## 核心结果

### 主模型（MSSACT-Net light, 6.10M 参数）

| 配置 | Kappa | OA | 训练 |
|---|---|---|---|
| **full_all_v2** | **0.6312** | **0.7030** | 1,768 张 / 60 轮 |
| **post_all_v2**（后训练） | **0.6413** | **0.7112** | 续训 30 轮（lr=5e-5） |

### 对比实验（7 个基线，统一协议）

| 模型 | 参数量 | Kappa | OA |
|---|---|---|---|
| **MSSACT-Net light + 后训练** | **6.10M** | **0.6413** | 0.7112 |
| **MSSACT-Net light** | **6.10M** | **0.6312** | 0.7030 |
| FPN (R50) | 27.2M | 0.6151 | 0.6912 |
| FCN-32s (R50) | 23.7M | 0.6144 | 0.6903 |
| U-Net | 2.44M | 0.6054 | 0.6833 |
| Swin-Unet (lite) | 32.6M | 0.5859 | 0.6693 |
| PSPNet (R18+PPM) | 13.6M | 0.5175 | 0.6160 |
| SegFormer-Lite | 1.01M | 0.3043 | 0.4441 |
| DeepLabV3+（从零，协议内） | 39.63M | ⏳ 训练中 | — |
| ~~DeepLabV3+（预训练主干）~~ | 39.6M | ~~0.6876~~ | 旁证，非协议内 |

**解读**：
- MSSACT-light 以 **1/4 ~ 1/3 的参数量**达到与 23–27M 基线相当的性能。
- **DeepLabV3+ 的 0.6876 已从基线表中移除**：那一行是**代码 bug 的产物**——本项目
  **从未计划使用预训练**，但 `deeplabv3_resnet50(weights=None)` 仍会加载 ImageNet
  预训练主干（该工厂另有独立的 `weights_backbone`，默认即 `IMAGENET1K_V1`）。
  它与其余从零训练的模型**不同协议**，不能作为基线。
- **协议内（从零）对比下，DeepLabV3+ 39.63M 与自研 6.10M 不可分辨**
  （Δ=+0.0039，0.78σ）。协议内从零版本正在训练（`nd_deeplab_scr`）。
- 详见 5.6.6：原 +0.0720 中约 **95% 来自预训练主干**，不是架构。

### 消融实验（8 个变体）

| 变体 | Kappa | ΔKappa vs 完整 |
|---|---|---|
| **完整模型** | **0.6312** | — |
| Trans-4L | 0.6362 | +0.0050 |
| w/o EMR | 0.6355 | +0.0043 |
| w/o Adapter | 0.6342 | +0.0030 |
| Trans-6L | 0.6335 | +0.0023 |
| w/o FPN | 0.6314 | +0.0002 |
| w/o Transformer | 0.6302 | −0.0010 |
| w/o ECSAM | 0.6272 | −0.0040 |
| **Bilinear-Up（解码器替换）** | **0.6369** | **+0.0057** |

**发现**：8 个消融变体的 |ΔKappa| **全部 ≤ 0.006**，其中 **5 个变体反而略优于
完整模型**（Bilinear-Up、Trans-4L、w/o EMR、w/o Adapter、Trans-6L）。

即在 1,768 张数据规模下，**EMR / ECSAM / FPN / Transformer / Adapter 以及
可学习解码器均无可测量的独立贡献**。FPN（40.7% 参数）+ Transformer（26.4%）
合计 67% 的模型容量对性能无显著影响。

> 该结论与阶段 A（957 张，"EMR 关键、ECSAM 有用"）**方向相反**——模块的边际
> 价值随数据量增加被压缩至噪声水平。联合移除 FPN+Transformer（参数量降至
> 2.00M）的验证实验正在运行。

---

## 重要发现

### 1. 数据量与数据质量共同决定性能上限

| 阶段 | 数据 | 标签处理 | Kappa |
|---|---|---|---|
| A | 957 张 | 含映射缺陷 | 0.4065 |
| B | 2,257 张（官方全量去重） | 含缺陷 | 0.6114 |
| **C** | **1,768 张（筛选 + 修复）** | **✅ 官方协议** | **0.6312** |

少数类获益最大（阶段 B 测量）：道路 F1 **+0.390**、水域 **+0.385**、
裸地 **+0.324**、建筑 **+0.255**。

### 2. 收敛速度由学习率调度决定，而非架构

以**累计学习量**（LR 积分 S=Σηₜ）衡量"学到同一水平所需的有效更新量"：

| 模型 | 达 Kappa 0.5 轮次 | 累计学习量 |
|---|---|---|
| **DeepLabV3+ (39.6M)** | 11 | **31.8%** ⚡ |
| Swin-Unet (32.6M) | 14 | 41.2% |
| U-Net / FCN / w/o FPN / w/o Adapter | 20 | 58.3% |
| **MSSACT-light (6.1M)** | 21 | **60.9%** |
| PSPNet (13.6M) | 47 | 98.4% |

12 个模型中 9 个集中在 47%–64%——**"第几轮收敛"是调度曲线的映射，不是架构优势**。
唯一例外是 DeepLabV3+（31.8%），其空洞卷积 + ASPP 结构确实学习效率更高。

### 3. 标签质量已成为模型能力评价的瓶颈

6 个架构/容量差异极大的模型（2.4M – 39.6M）在干净测试集（141 张）上逐像素分析：

| 证据 | 数值 | 含义 |
|---|---|---|
| 模型间一致率 vs 模型-标签一致率 | **0.759 vs 0.650** | 模型互相认同远高于认同人工标签 |
| 共识标签 vs 人工标签 | **0.682** | 31.8% 像素上多数模型与标注不符 |
| **容量-性能相关性反转** | 人工 r=**+0.311** → 共识 r=**−0.444** | 大模型更贴合标注模式而非更接近共识 |
| 边界像素占比 | 边界(距离0) **2.4%**；内部(>8px) **78.4%** | 标签**并不碎片化**；聚合指标 78.4% 权重来自内部像素 |
| 准确率随距边界距离 | 边界 0.45 → 内部 0.70（单调上升） | 边界是标签最不确定处（模型间分歧 0.288 vs 内部 0.229） |
| **Kappa 归属上限** | 完美解决边界仅 **+0.016**；完美解决内部 **+0.293** | **指标在结构上几乎看不见边界质量** |

> **勘误**：早期版本此处报告"93.0% 像素在类边界上"，并据此得出"标签碎片化"与
> "越纯净区域模型越不同意标签"两条结论。原实现把标签批次按 `(N,H,W)` 沿 `axis=0`
> 求差分，**跨图像的像素差被误判为类边界**，故得出虚高比例。改为逐图 (2D)、仅统计
> 两个**不同有效类别**之间的相邻像素后，真实边界比例为 **2.4%**，结论方向亦反转。
> 修正见 `fix_boundary_analysis.py`，原文件备份为 `boundary_analysis.bak_*.json`。

**结论**：模型间差异（跨度约 0.08 Kappa）已落在标签噪声引入的不确定性范围内——
进一步区分架构优劣需要**更高精度的人工标注数据**（多标注者一致性标注、更严格
的边界规范）。文献佐证：NSegment+ ([arXiv:2508.10383](https://arxiv.org/abs/2508.10383))
仅在标签端做形变即可在 LoveDA 上获得 +2.38 mIoU。

**机制性解释（Kappa 归属上限）**：完美预测全部边界像素（2.4%）只能把 Kappa 从
0.5774 提到 0.5936（+0.016），而完美预测内部像素（78.4%）可提到 0.8700（+0.293）。
六模型的该上限高度一致。因此任何**面向边界锐化**的模块（多尺度融合 FPN、坐标
注意力、转置卷积解码器）在聚合 Kappa 上的可见空间被限制在 ≈0.016 内，与观测到的
消融总散布 0.006 **同一量级**——指标主要在度量内部区域的语义分类能力。

### 4. 消融实验"全部无差异"的根因：四处**已数值证实**的结构性缺陷

发现 3 的解释（标签上限）**不完整**。对 checkpoint 张量与前向传播做直接数值检验后，
证实本模型有多个模块**在实现层面并未按设计意图工作**——这是"移除后无影响"的
直接原因。四项检验均可复现，详见 [`docs/架构有效性分析.md`](docs/架构有效性分析.md)
与 [`docs/缺陷修复_20260913.md`](docs/缺陷修复_20260913.md)。

| # | 缺陷 | 检验方式 | 结果 |
|---|---|---|---|
| **D1** | FPN 多尺度融合是**死计算** | 归零 `lateral_convs[0..2]`；把整个 FPN 换成"仅第 4 支路"的等价算子 | 两次输出差**恰好 0.0**（逐位相同） |
| **D2** | 解码器**无跳连**，仅凭 32² 特征上采样 8 倍 | 前向路径 + 分辨率链追踪 + 边缘密度/FFT | 256²/128²/64² 特征**从未到达输出**。注意**不要**说"输出被限制在 8 px 粒度"——转置卷积**能够**产生高频结构（实测预测边缘密度达标签的 4.56×，且周期 k=8 功率超额 1.66×）。正确表述是：**没有任何通路让高频的输入证据到达输出**，细节是解码器*合成*的，而非在高分辨率上*测量*到的 |
| **D3** | Transformer **无位置编码** | 置换 token 后按逆置换还原 | 差 **7.15e-07** → **置换等变**，无空间感知 |
| **D5** | `ecsam[0]` 是**死模块** | 构造时 4 个阶段全建，前向条件是 `i > 0`；查各张量 `.grad` | 第 0 个恒不执行，其 7 个张量的 `.grad is None`；1,040 个参数（**0.017%**） |

**参数账目（直读张量，`nn.Parameter` 口径）**：FPN **40.73%** / Transformer
**26.44%** / EMR 编码器 20.08% / 解码器 11.84% / **坐标注意力仅 0.90%**。
其中 FPN 的 **1,828,352 参数（真实参数的 29.98%）永不参与输出**——实际参与前向的
有效参数仅 **70.02%**（另有 `ecsam[0]` 的 1,040 个参数即 0.017% 同样不参与前向）。

> **口径更正（2026-09-13）**：此前引用的 6,101,784 是 `state_dict` 总元素数，
> 含 16 层 BatchNorm 的 2,640 个 buffer，**高估参数 0.0433%**。
> 真实参数 = **6,099,144（6.0991M）**。相应地把含 buffer 的旧值一并更正：
> DeepLabV3+ 39.692M → **39.635M**、D4 24.242M → **24.237M**、
> D1 2.005M → **2.002M**、D3 1.547M → **1.545M**、D2 6.115M → **6.112M**。

**推论**：**没有一个消融变体削弱了它声称削弱的功能**，因为对应功能在实现中本就
不存在（FPN）或不可用（Transformer 的位置感知）。故这组消融**不能**作为"模块
有效性"的证据。同时，DeepLabV3+ 的 **+0.0720**（配对 bootstrap 95% CI
[+0.0525, +0.0929]，显著）也有了机制解释：它**拥有**本模型缺失的低层跳连。
（**已更正**：初稿称"所有基线 `weights=None` 从零训练"——该核验是错的。
torchvision 的 `deeplabv3_resnet50` 另有独立的 `weights_backbone` 参数，默认即
`IMAGENET1K_V1`，故只传 `weights=None` **仍会加载预训练主干**。实测首个 BN 的
`weight` 均值：U-Net/PSPNet/FCN/SegFormerLite/FPN-Seg 均为 **1.0000**（确为从零），
而 **DeepLabV3+ 为 0.2574**（预训练）。故 +0.0720 实为
"6.10M 从零模型 vs 39.63M **+预训练主干**"，**架构贡献与预训练贡献无法分离**，
初稿归因于"低层跳连"属过度归因。已加 `pretrained_backbone` 参数并用 `_scr`
后缀的从零对照分离二者。）

**从零对照结果（已完成，同管线同协议）**：

| 模型 | 参数量 | Kappa | Δ | 判读 |
|---|---:|---:|---:|---|
| **MSSACT-Net（自研，从零）** | 6.10M | **0.6320** | — | 参照 |
| DeepLabV3+（**从零**） | 39.63M | **0.6359** | **+0.0039** | **0.78σ，不可分辨** |

**即：从零训练的公平对比下，自研 6.10M 与 DeepLabV3+ 39.63M 不可分辨**，
而后者参数是前者的 6.5 倍。原来那 **+0.0720 中约 95% 来自预训练主干**，不是架构。

**预训练主干的独立贡献**（同一模型、同一管线，仅差权重初始化）：
阶梯三档实测 **+0.0535 ~ +0.1540 Kappa**。作为对照，同管线**全部模块消融都
≤1.13σ_seed（≈0.0057）**，裁剪策略为 0.0199–0.0223。
**杠杆排序：换主干权重 ≫ 改训练策略 > 增删架构模块。**

**修复已完成并验证**（2026-09-13，全部**向后兼容**：三个新开关默认值等于修复前
行为，既有 checkpoint 仍可 `strict=True` 加载，既有结论仍可复现）：

| 开关 | 作用 | 修复 | 状态 |
|---|---|---|---|
| `use_skip=True` | 解码器逐级接入主干同分辨率特征；有 FPN 时接入其**四级输出**（死支路变活） | D1+D2 | ✅ 死参数 **29.99% → 0.016%** |
| `pos_enc=True` | Transformer 注入 2D 正弦位置编码 | D3 | ✅ 置换等变性残差 9.5e-07 → **1.94** |
| `ecsam_stage0=True` | 第 0 阶段也应用 ECSAM | D5 | ✅ 死参数归零 |

五项验证（默认路径**逐位一致**、键集合一致、置换等变性被打破、梯度可达性、
14 种组合路径可前向反向）由 [`verify/verify_fix_compat.py`](verify/verify_fix_compat.py)
对修复前快照自动回归。**仍未做**：P2 容量膨胀（2.67×）。
**修复的性能效果：跳连有害（被否决），位置编码弱正向**（120 轮同协议，σ_seed = 0.0050）：

| 实验 | 修复内容 | Kappa | Δ vs 基线 | 判定 |
|---|---|---:|---:|---|
| 未修复（基线） | — | **0.6655** | — | 参照 |
| `fx_pos` | D3 位置编码 | **0.6716** | +0.0061 | 弱（1.2σ） |
| `fx_skip` | D1+D2 跳连 | 0.6505 | **−0.0150** | 可分辨（3σ） |
| `fx_all` | 三处全修 | 0.6330 | **−0.0325** | 可分辨 |

即**修复没有带来收益，全修反而明显更差**，且两项修复合起来比任何单项都差得多（非加性）。
**过拟合假设已被推翻**：`fx_all` 的训练 loss 是四者中最高的（0.7237 vs 基线 0.6269）——
训练集拟合更差、验证也更差，属**欠拟合/优化困难**而非过拟合。
两个假设都已检验并**被证伪**：**H1**（256² 全分辨率融合有害）——只融合 128²/64² 得
0.6466，与全融合 0.6505 仅差 <1σ，仍远低于基线；**H2**（学习率不匹配）——降到 lr=1e-4
反而更差（skip −0.0110、全修 −0.0160）。⇒ **有害的是跳连本身，与融合层级、学习率无关**，
D1/D2 的修复被**实测否决**。

消融因此换了底：`use_skip+pos_enc` 这个底不稳定（其上一个消融使训练**数值发散**——
loss 恒为 0、OA 精确冻结、权重最大绝对值 1.26e4），改用稳定且唯一高于基线的
`fx_pos`（`pos_enc=True`, 0.6716）为底，新变体以 `lgR120_pos_abl_*` 另存命名。
详见 `queues/run_queue_fix3.py`。

### 但"修好缺陷"的真正价值在**可解释性**，不在表头 Kappa

在 `pos_enc` 底上重跑同一组消融后，结论比"修复有害"完整得多。**未修复架构上 6 项里
有 4 项是"移除反而更好"；修复后 7 项全部偏负、无一改善**：

| 变体 | 未修复架构（底 0.6312） | `pos_enc` 底（底 0.6716） |
|---|---:|---:|
| 去掉 Adapter | **+0.0030**（移除反而更好） | −0.0012 |
| 去掉 FPN | **+0.0002**（移除反而更好） | −0.0037 |
| 去掉 EMR | **+0.0043**（移除反而更好） | −0.0067 |
| 双线性 | **+0.0057**（移除反而更好） | −0.0073 |
| 去掉 Transformer | −0.0010（几乎为零） | **−0.0116（2.33σ，可分辨）** |
| 去掉 ECSAM | −0.0040（正常收敛） | **训练发散** |
| Transformer 2→4 / 2→6 层 | — | +0.0012 / −0.0012（不可分辨） |

**位置编码修复自己只值 +0.0061（1.2σ，弱），但它把"移除 Transformer"的代价从
−0.0010（等于没有）变成 −0.0116（可分辨）。** 机制上说得通：无位置编码时注意力是
**置换等变集合算子**（做不了空间建模），移除自然无所谓；注入 2D 正弦编码后它才能做
邻近建模，于是开始承载性能。**这就是"缺陷掩盖了模块作用"的直接正向证据。**

两条限定必须同时说：**ECSAM 移除的发散在两个不同的底上各复现一次**，而在未修复架构上
它能正常收敛 ⇒ ECSAM 是**条件性的训练稳定器**，此前"移除 ECSAM 一致更好"只在未修复
架构上成立；**双线性 vs 转置卷积不可分辨**（两次测量符号相反、均落在噪声内），
伪影**存在**是实测事实，但"换双线性会提精度"不是。

> 这意味着两件事要分开说："缺陷导致消融无差异"的诊断成立；
> "修好缺陷就能提升精度"**被实测否证**（跳连甚至有害）；
> 但"修好缺陷让消融变得可解释"**成立且有正向证据**。

**诚实声明**：上表 ΔKappa 均来自**单次运行**（seed=42），但**种子方差已实测**：
σ_seed = **0.0050**（n=4 次独立训练：seed 42/7/2024/31337），故 |Δ| ≥ 2σ = 0.0100
才记为可分辨。多种子只覆盖完整模型与 DeepLabV3+ 两个配置，n=4 仍低于文献建议的
≥10，故适用于**量级判断**而非精确检验。

等效性检验（TOST，SESOI = 0.02 预指定）已完成并落盘为
`results/artifacts/tost_equivalence.json`：配对 bootstrap 的 95% CI 半宽实测
**δ ≈ 0.010 < SESOI**，故"无差异"已从"未达显著"升级为**统计等效**。

---

## 数据

### 官方类别体系（依据官方文档）

> Category labels: **background – 1, building – 2, road – 3, water – 4,
> barren – 5, forest – 6, agriculture – 7**; and the **no-data regions were
> assigned 0 which should be ignored**.

| 标签值 | 类别 | 占比（验证集） |
|---|---|---|
| **0** | **no-data（未标注，必须忽略）** | ~3% |
| 1 | 背景 | 45.6% |
| 2 | 建筑 | 4.3% |
| 3 | 道路 | 2.4% |
| 4 | 水域 | 11.6% |
| 5 | 裸地 | 4.1% |
| 6 | 森林 | 5.5% |
| 7 | 农田 | 26.6% |

**注意**：「背景」是官方正式类别，**不是**「未分类」；真正的未分类是值 0。

### 筛选与划分

```
官方训练区 2,522 张
  ├─ no-data ≤10% 筛选     →  保留 2,210 张（剔除 312 张）
  ├─ MD5 去重              →  去除与历史镜像重复样本
  └─ 8:1:1 划分 (seed=42)  →  train 1,768 / val 221 / test 221
     └─ test_clean 141 张  →  额外排除历史训练集（跨阶段公平对比）
```

**泄漏校验**：train ∩ val = train ∩ test = val ∩ test = ∅；test_clean ⊆ test；
test_clean ∩ 历史训练集 = ∅。

---

## 方法

### 模型架构

MSSACT-Net：Stem 卷积 → 4 阶段 EMR 残差编码器（逐阶段 ECSAM 坐标注意力）
→ FPN 特征金字塔 → Transformer 编码器（含 Adapter-Scale）→ 转置卷积解码器。

| 配置 | embed_dims | Transformer | 参数量 |
|---|---|---|---|
| **light（主模型）** | [32, 64, 128, 256] | 2 层 4 头 | **6.10M** |
| full（原版） | [64, 128, 256, 512] | 4 层 8 头 | 30.54M |

### 训练协议

AdamW(lr=2e-4, wd=0.02) + OneCycle(6% warmup + 余弦退火, 60 轮) + EMA(0.999)
+ 类别加权交叉熵（频率逆平方根）+ 验证 Kappa 早停(patience=20)；输入 256×256 RGB。

### 评价协议

- **以 Kappa 选优**（OA 在类别不平衡下会被单类支配：全预测背景即得 OA=45.6%，Kappa=0）
- **val 仅用于早停/选优**（`torch.no_grad()`，不参与梯度更新）
- **最终结论以 test / test_clean 为准**

---

## 项目结构

根目录**只保留** 5 个核心模块与文档；脚本按用途分 7 个子目录；
全部产物集中在 `results/`；报告交付件在 `reports/`。

```
loveda/
├── paths.py                    # ★ 路径枢纽：全部路径的唯一来源（含 FACTS/REPORTS/LOGS）
├── train_v3.py                 # 数据集 + 训练循环 + REMAP（核心）
├── experiment_matrix.py        # 通用训练入口 train_one() + 基线模型
├── experiment_matrix_v2.py     # 扩展基线(FPN/Swin-Unet) + 消融构造
├── train_full_data.py          # 全量数据训练（主模型 + 后训练）
├── models/msscactnet.py        # MSSACT-Net 定义（模块可开关）
│
├── queues/                     # 实验队列（自动串行、断点续跑、互斥、无窗口）
│   ├── run_chain.py            #   统一等待链：静默等待 → final → crop
│   ├── run_chain_scr.py        #   收尾链：DeepLab 从零对照
│   ├── run_queue_new.py        #   主模型重训 + 对比(7) + 消融(8)
│   ├── run_queue_arch.py       #   架构实验 D1-D4 + 预算攻击 + 数据阶梯
│   ├── run_queue_final.py      #   多种子噪声底线 + 全量推理
│   ├── run_queue_crop.py       #   裁剪策略对比（256/384/512、随机/固定）
│   ├── run_queue_deeplab_scr.py / _protocol.py   # 从零 DeepLab 对照与协议内基线
│   ├── run_queue_30m.py / _interaction.py / _batch.py   # 早期队列（历史）
│   ├── monitor_queue.py        #   实时进度监控（动态刷新 + ETA）
│   ├── eta_queue.py            #   剩余时间估算（按实测每轮耗时累加）
│   └── queue_guard.py          #   队列互斥守卫（CREATE_NO_WINDOW，避免弹窗）
│
├── analysis/                   # 评估与分析
│   ├── eval_rigor.py           #   严格性评估：逐类 IoU / 配对 bootstrap / TOST
│   ├── eval_ladder.py          #   数据阶梯评价（泛化 / 过拟合 / 学习速率）
│   ├── consensus_analysis.py   #   标签质量一致性分析
│   ├── headroom_analysis.py    #   Kappa 归属分解（oracle 作弊实验）
│   ├── artifact_analysis.py    #   解码器伪影 / 边缘密度 / FFT 周期
│   ├── convergence_amount.py   #   累计学习量（分离调度效应与架构效应）
│   ├── seed_variance.py        #   随机种子噪声底线（σ_seed）
│   ├── fix_boundary_analysis.py#   修正边界分层定义（原实现有跨图求差缺陷）
│   ├── overfit_diag.py         #   过拟合诊断（logits 熵）
│   └── eval_complete.py / eval_tta_new.py    # 早期评估
│
├── verify/                     # 核验与审核
│   ├── pre_push_audit.py       #   ★ 推送前审核（本机路径/敏感语境/大文件/误提交）
│   ├── verify_experiments.py   #   实验产物核验
│   ├── verify_rollback.py      #   回退管线性能验证
│   ├── verify_split2.py        #   划分泄漏校验
│   └── bench_parity.py / bench_pipeline.py / sample_gpu_now.py   # 基准与采样
│
├── prep/                       # 数据准备
│   ├── resplit_filtered.py     #   no-data 筛选 + MD5 去重 + 8:1:1 划分
│   ├── build_data_ladder.py    #   嵌套数据阶梯子集（250/500/1000）
│   └── build_fast_dataset.py / build_ladder_memmap.py   # 预解码 memmap
│
├── viz/                        # 图表
│   ├── make_report_figures.py  #   ★ 报告全部图（从权威数据源生成，22 张）
│   ├── make_figures.py         #   早期图（阶段 A/B，历史）
│   └── make_stats_figures.py   #   早期统计图（历史）
│
├── report/                     # 汇总与报告生成
│   ├── build_index.py          #   实验索引 → results/facts/experiments_index.json
│   ├── build_facts.py          #   事实汇总 → results/facts/FACTS.json（唯一数据源）
│   ├── make_report_v3.py       #   ★ 当前报告生成器（Markdown → HTML）
│   └── make_pdf.py             #   HTML → PDF（含 25 项内容自检 + 页脚泄露检查）
│
├── mapping/inference_fullmap.py# 全图滑窗推理 + shp 裁剪成图（制图管线）
│
├── docs/                       # 文档
│   ├── 架构有效性分析.md         #   三项已数值证实的架构缺陷 + 文献 + 修复路径
│   ├── 报告问题研究_20260913.md  #   报告 7 类 27 条问题的审查与处置
│   ├── 整理记录_20260913.md      #   目录整理记录 + 路径适配 + 验证
│   ├── 项目阶段评估_20260913.md  #   三阶段划分与本项目定位
│   └── archive/                #   历史文档归档
├── legacy/                     # 历史脚本归档（含被 v3 取代的 make_report_v2.py）
│
├── reports/                    # ★ 报告交付件
│   ├── 项目报告_20260913.pdf    #   当前报告（44 页）
│   └── 项目报告_20260913.html   #   HTML 版（可再生，不入库）
├── figures/                    # 报告图（入库）
│
└── results/                    # ★ 全部产物集中于此
    ├── facts/                  #   权威事实：FACTS.json / experiments_index.json / seed_variance.json
    ├── logs/                   #   运行日志（不入库）
    ├── checkpoints/            #   权重 + 训练历史（约 5.2 GB）
    ├── fast_dataset/           #   预解码 memmap（约 17 GB）
    ├── consensus_analysis/     #   一致性分析（汇总 JSON 入库）
    ├── overfit_diag/  ladder_eval/  fulltile_eval/  tta_eval/  heldout_test/
    └── artifacts/              #   中间分析产物（类别分布、收敛量、每轮耗时等）
```

> **为什么核心库留在根目录**：`paths.py` 以 `REPO = dirname(abspath(__file__))`
> 作为**所有相对路径的基准**。它一旦移动，全部输出路径都会错位。因此 `paths.py`
> 与 4 个核心模块固定留在根目录；子目录脚本通过注入的**路径引导块**
> （把仓库根与自身目录加入 `sys.path`）导入它们。
>
> **为什么其余一切都进子目录**：脚本一律通过 `paths.py` 取路径，
> 因此重组只需改 `paths.py` 一处。本次把产物收进 `results/`、事实收进
> `results/facts/`、报告收进 `reports/`、日志收进 `results/logs/`，
> 修正了 12 个脚本的路径引用，并**端到端实跑验证**（FACTS 14 节无 null、
> 报告 44 页自检全过）。详见 [docs/整理记录_20260913.md](docs/整理记录_20260913.md)。

> **路径配置**：代码中不含本地绝对路径。本地使用时创建 `.env` 提供数据根目录
> （详见 [STRUCTURE.md](STRUCTURE.md) 第四节）。

**不入库**（体积大 / 可复现生成）：`results/checkpoints/`、
`results/fast_dataset/`、`results/logs/`、
各评估目录下的预测数组与 PNG（`results/*/pred_*.npz`、`*.png`）、
`reports/*.html`、`_local_archive/`、`backups/`、原始 LoveDA 数据。

**入库**：`results/facts/` 三项权威事实、`reports/项目报告_*.pdf`、`figures/`、
`results/*/` 下的汇总性统计 JSON（如 `ladder_summary.json`、`*_conf.json`）。

---

## 快速开始

> 全部命令**在仓库根目录执行**；脚本内不含绝对路径，路径由 `paths.py` 统一提供。

```bash
conda activate pytorch
pip install -r requirements.txt

# 0) 配置本地路径（创建 .env，见 STRUCTURE.md 第四节）
#    LDA_DATA_ROOT=<数据根目录>
#    LDA_CKPT=<检查点目录>

# 1) 数据准备（需先下载 LoveDA 官方训练区）
python prep/resplit_filtered.py       # no-data 筛选 + 去重 + 划分
python verify/verify_split2.py        # 泄漏校验
python prep/build_data_ladder.py      # 数据阶梯子集（可选）
python prep/build_fast_dataset.py     # 预解码主数据集 memmap（强烈建议，提速约 10 倍）
python prep/build_ladder_memmap.py    # 预解码阶梯子集 memmap（数据阶梯必需；跑前先停队列）

# 2) 训练（队列自动串行、断点续跑、OOM 自动降 batch）
python queues/run_queue_new.py        # 主模型 + 对比(7) + 消融(8)
python queues/run_queue_arch.py       # 架构实验 D1-D4 + 预算攻击 + 数据阶梯

# 2b) 或一次性挂起全链（静默等待其他队列，无窗口）
pythonw queues/run_chain.py           # 等主队列结束 → final(多种子) → crop(裁剪策略)

# 3) 监控 / 估算剩余时间（另开终端）
python queues/monitor_queue.py --interval 15
python queues/eta_queue.py            # 按实测每轮耗时累加出剩余小时数

# 4) 评估与分析（--infer 会补齐缺失 tag 的推理并缓存）
python analysis/eval_rigor.py --infer
python analysis/consensus_analysis.py
python analysis/headroom_analysis.py
python analysis/artifact_analysis.py

# 6) 推送前审核（必须通过才 push）
python verify/pre_push_audit.py   # 本机路径 / 内部语境词 / 大文件 / 误提交

# 5) 汇总与报告
python report/build_index.py          # 实验索引 → results/facts/experiments_index.json
python report/build_facts.py          # 事实汇总 → results/facts/FACTS.json
python viz/make_report_figures.py     # 报告全部图 → figures/（22 张，从权威数据源生成）
python report/make_report_v3.py       # 报告 HTML → reports/
python report/make_pdf.py             # HTML → PDF（含 25 项内容自检 + 页脚泄露检查）
```

---

## 局限与后续工作

**当前局限**：
- 标签精度不足以支撑细粒度架构比较（见"重要发现 3"）
- 仅使用官方训练区（未纳入验证区 1,669 张）
- 跨分辨率迁移（GF-1 2.4m）受域差限制，需同域微调

**后续方向**：
1. **多光谱适配（重点）**：探索光谱维 / 空间维 / 通道维的**三方向池化**设计，
   替代现有 ECSAM 坐标注意力，检验框架在多光谱（>4 波段）数据上的适配价值
2. **标签质量升级**：多标注者一致性标注，估计真实噪声率，在更高精度标注上重评架构
3. **数据阶梯与容量匹配**：在 250/500/1000/1768 张上评价泛化/过拟合/学习速率，
   寻找最优模型容量
4. **跨域验证**：GF-1 2.4m + 同域微调
5. **少数类与部署**：重采样/损失调优；推理加速

---

## 引用

```bibtex
@inproceedings{wang2021loveda,
  title={LoveDA: A remote sensing land-cover dataset for domain adaptive semantic segmentation},
  author={Wang, Junjue and Zheng, Zhuo and Ma, Ailong and Lu, Xiaolei and Zhong, Yanfei},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```

## License

MIT
