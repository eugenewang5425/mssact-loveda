# MSSACT-Net — 多尺度自注意力遥感土地覆盖分割（LoveDA 基准）

Multi-Scale Self-Attention ConvTransformer (MSSACT-Net) for high-resolution
land-cover semantic segmentation, with systematic evaluation on the public
**LoveDA** benchmark: baseline comparison, module ablation, training-strategy
search, and an empirical study on the **label-quality ceiling** of model evaluation.

> **项目报告**：[项目报告_20260912.pdf](项目报告_20260912.pdf)（22 页，含全部实验与图表）
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
| **DeepLabV3+** | 39.6M | **0.6876** | 0.7497 |
| **MSSACT-Net light + 后训练** | **6.10M** | **0.6413** | 0.7112 |
| **MSSACT-Net light** | **6.10M** | **0.6312** | 0.7030 |
| FPN (R50) | 27.2M | 0.6151 | 0.6912 |
| FCN-32s (R50) | 23.7M | 0.6144 | 0.6903 |
| U-Net | 2.44M | 0.6054 | 0.6833 |
| Swin-Unet (lite) | 32.6M | 0.5859 | 0.6693 |
| PSPNet (R18+PPM) | 13.6M | 0.5175 | 0.6160 |
| SegFormer-Lite | 1.01M | 0.3043 | 0.4441 |

**解读**：MSSACT-light 以 **1/4 ~ 1/3 的参数量**达到与 23–27M 基线相当的性能；
DeepLabV3+ 仍领先，但其参数量为 MSSACT 的 **6.5 倍**。

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
| Bilinear-Up（解码器替换） | 0.4885¹ | −0.1427 |

¹ Bilinear-Up 仍在训练中

**发现**：7 个模块消融变体与完整模型的差异**均小于 0.01 Kappa**——在 1,768 张
数据规模下，EMR / ECSAM / FPN / Transformer / Adapter 均无独立可测贡献；
**唯一显著依赖是可学习转置卷积解码器**（替换为双线性上采样大幅退化）。

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
| 标签碎片化 | **93.0%** 像素在类边界上 | 0.3m 分辨率下边界"真值"难定义 |
| 准确率随距边界距离 | 边界 0.61–0.73 → 中距离 0.47–0.57 | 越"纯净"区域模型越不同意标签 |

**结论**：模型间差异（跨度约 0.08 Kappa）已落在标签噪声引入的不确定性范围内——
进一步区分架构优劣需要**更高精度的人工标注数据**（多标注者一致性标注、更严格
的边界规范）。文献佐证：NSegment+ ([arXiv:2508.10383](https://arxiv.org/abs/2508.10383))
仅在标签端做形变即可在 LoveDA 上获得 +2.38 mIoU。

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

```
loveda/
├── paths.py                    # 路径配置中心（环境变量 / .env / 默认值）
├── models/msscactnet.py        # MSSACT-Net 模型定义（支持模块开关）
├── train_v3.py                 # 数据集 + 训练循环 + REMAP（核心）
├── experiment_matrix.py        # 通用训练入口(train_one) + 基线模型
├── experiment_matrix_v2.py     # 扩展基线(FPN/Swin-Unet) + 消融构造
├── train_full_data.py          # 全量数据训练（主模型 + 后训练）
│
├── run_queue_new.py            # 实验队列: 主模型重训 + 对比(7) + 消融(8)
├── run_queue_30m.py            # 30.5M 原版架构 + 后训练 + TTA
├── run_queue_interaction.py    # 交互实验: 预算攻击 + 固定LR + 数据阶梯
├── monitor_queue.py            # 实时进度监控（动态刷新 + ETA）
│
├── resplit_filtered.py         # 数据筛选 + MD5去重 + 8:1:1 划分
├── verify_split2.py            # 划分泄漏校验
├── build_data_ladder.py        # 嵌套数据阶梯子集（250/500/1000）
│
├── eval_complete.py            # val/test × TTA 评估
├── eval_tta_new.py             # 新数据 TTA 评估
├── eval_fulltile.py            # 全图滑窗推理评估
├── eval_ladder.py              # 数据阶梯评价（泛化/过拟合/学习速率）
├── consensus_analysis.py       # 标签质量一致性分析
├── overfit_diag.py             # 过拟合诊断（logits 熵）
├── verify_experiments.py       # 实验产物核验（架构/完整性/交叉一致）
│
├── build_facts.py              # 事实汇总 → FACTS.json（报告唯一数据源）
├── make_figures.py             # 图表生成
├── make_stats_figures.py       # 统计图表（混淆矩阵/PRF/效率）
├── make_report_v2.py           # 报告生成（Markdown→HTML→PDF）
│
├── figures/                    # 26 张图表（报告/README 引用）
├── legacy/                     # 历史脚本归档（16 个早期脚本 + 说明）
├── docs/archive/               # 历史文档归档
├── 项目报告_20260912.pdf        # 项目报告（22 页）
├── FACTS.json                  # 全部实验结果的结构化汇总
└── .env                        # 本地路径配置（不入库，见 STRUCTURE.md）
```

> **路径配置**：代码中不含本地绝对路径。本地使用时创建 `.env` 提供数据根目录
> （详见 [STRUCTURE.md](STRUCTURE.md) 第四节）。

**不入库**（体积大 / 可复现生成）：`checkpoints/`（权重与训练历史）、
`fulltile_eval/` `tta_eval*/` `heldout_test/` `consensus_analysis/`（评估结果）、
原始 LoveDA 数据。

---

## 快速开始

```bash
conda activate pytorch
pip install -r requirements.txt

# 0) 配置本地路径（创建 .env，见 STRUCTURE.md 第四节）
#    LDA_DATA_ROOT=<数据根目录>
#    LDA_CKPT=<检查点目录>

# 1) 数据准备（需先下载 LoveDA 官方训练区）
python resplit_filtered.py        # 筛选 + 去重 + 划分
python verify_split2.py           # 泄漏校验
python build_data_ladder.py       # 数据阶梯子集（可选）

# 2) 训练（实验队列，自动串行、断点续跑）
python run_queue_new.py           # 主模型 + 对比(7) + 消融(8)
python run_queue_30m.py           # 30.5M 原版 + 后训练 + TTA
python run_queue_interaction.py   # 预算攻击 + 固定LR + 数据阶梯

# 3) 监控（另开终端）
python monitor_queue.py --interval 15

# 4) 评估与分析
python eval_tta_new.py            # val/test_clean × TTA
python consensus_analysis.py      # 标签质量一致性分析
python eval_ladder.py             # 数据阶梯评价

# 5) 汇总与报告
python build_facts.py             # 事实汇总 → FACTS.json
python make_report_v2.py          # 生成报告 HTML（再用 Chrome 转 PDF）
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
