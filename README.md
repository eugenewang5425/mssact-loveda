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

### 4. 消融实验"全部无差异"的根因：三处**已数值证实**的结构性缺陷

发现 3 的解释（标签上限）**不完整**。对 checkpoint 张量与前向传播做直接数值检验后，
证实本模型有多个模块**在实现层面并未按设计意图工作**——这是"移除后无影响"的
直接原因。三项检验均可复现，详见 [`docs/架构有效性分析.md`](docs/架构有效性分析.md)。

| # | 缺陷 | 检验方式 | 结果 |
|---|---|---|---|
| **D1** | FPN 多尺度融合是**死计算** | 归零 `lateral_convs[0..2]`；把整个 FPN 换成"仅第 4 支路"的等价算子 | 两次输出差**恰好 0.0**（逐位相同） |
| **D2** | 解码器**无跳连**，仅凭 32² 特征上采样 8 倍 | 前向路径 + 分辨率链追踪 | 256²/128²/64² 特征**从未到达输出**；粒度 8 px（≈2.4 m） |
| **D3** | Transformer **无位置编码** | 置换 token 后按逆置换还原 | 差 **7.15e-07** → **置换等变**，无空间感知 |

**参数账目（直读张量）**：FPN **40.71%** / Transformer **26.43%** / EMR 编码器
20.10% / 解码器 11.84% / **坐标注意力仅 0.90%**。其中 FPN 的 **1,828,352 参数
（全模型 29.96%）永不参与输出**——实际参与前向的有效参数仅 **70.04%**。

**推论**：**没有一个消融变体削弱了它声称削弱的功能**，因为对应功能在实现中本就
不存在（FPN）或不可用（Transformer 的位置感知）。故这组消融**不能**作为"模块
有效性"的证据。同时，DeepLabV3+ 的 **+0.0720**（配对 bootstrap 95% CI
[+0.0525, +0.0929]，显著）也有了机制解释：它**拥有**本模型缺失的低层跳连。
（公平性已核验：所有基线 `weights=None` 从零训练，非预训练优势。）

**已给出可验证的修复路径**（P0 解码器加跳连、FPN 真正使用多尺度输出；P1
Transformer 加位置编码；P2 修正 2.67× 容量膨胀）。

**诚实声明**：ΔKappa 均来自**单次运行**（seed=42），**种子方差尚未测量**，
故"差异落在噪声内"目前是推断而非测量；多种子队列已在 `run_queue_final.py` 排队。

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

脚本按用途分目录；**仓库根目录只保留路径枢纽、核心库、权威事实与交付件**。

```
loveda/
├── paths.py                    # 路径配置中心（环境变量 / .env / 默认值）
├── train_v3.py                 # 数据集 + 训练循环 + REMAP（核心）
├── experiment_matrix.py        # 通用训练入口 train_one() + 基线模型
├── experiment_matrix_v2.py     # 扩展基线(FPN/Swin-Unet) + 消融构造
├── train_full_data.py          # 全量数据训练（主模型 + 后训练）
├── models/msscactnet.py        # MSSACT-Net 定义（模块可开关）
│
├── queues/                     # 实验队列（自动串行、断点续跑）
│   ├── run_chain.py            #   统一等待链：静默等待 → final → crop
│   ├── run_queue_new.py        #   主模型重训 + 对比(7) + 消融(8)
│   ├── run_queue_arch.py       #   架构实验 D1-D4 + 预算攻击 + 数据阶梯
│   ├── run_queue_final.py      #   多种子噪声底线 + 22 tag 全量推理
│   ├── run_queue_crop.py       #   裁剪策略对比（256/384/512、随机/固定）
│   ├── run_queue_30m.py        #   30.5M 原版架构 + 后训练 + TTA
│   ├── run_queue_interaction.py#   预算攻击 + 固定 LR
│   ├── run_queue_batch.py      #   batch 缩放对照（bs8 vs bs16）
│   ├── monitor_queue.py        #   实时进度监控（动态刷新 + ETA）
│   ├── eta_queue.py            #   剩余时间估算（按实测每轮耗时累加）
│   └── queue_guard.py          #   队列互斥守卫（CREATE_NO_WINDOW，避免弹窗）
│
├── analysis/                   # 评估与分析
│   ├── eval_rigor.py           #   严格性评估：逐类 IoU / 配对 bootstrap / TOST
│   ├── eval_ladder.py          #   数据阶梯评价（泛化 / 过拟合 / 学习速率）
│   ├── eval_complete.py        #   val/test × TTA
│   ├── eval_tta_new.py         #   新数据（newsplit2）TTA
│   ├── consensus_analysis.py   #   标签质量一致性分析
│   ├── headroom_analysis.py    #   Kappa 归属分解（oracle 作弊实验）
│   ├── artifact_analysis.py    #   解码器伪影 / 边缘密度
│   ├── fix_boundary_analysis.py#   修正边界分层定义（原实现有跨图求差缺陷）
│   ├── overfit_diag.py         #   过拟合诊断（logits 熵）
│   └── seed_variance.py        #   随机种子噪声底线（σ_seed）
│
├── verify/                     # 校验与基准
│   ├── verify_experiments.py   #   实验产物核验（架构/完整性/交叉一致）
│   ├── verify_rollback.py      #   回退管线性能验证
│   ├── verify_split2.py        #   划分泄漏校验
│   ├── bench_parity.py         #   PNG vs memmap 数值一致性
│   ├── bench_pipeline.py       #   管线提速基准
│   └── sample_gpu_now.py       #   GPU 利用率采样
│
├── prep/                       # 数据准备
│   ├── resplit_filtered.py     #   no-data 筛选 + MD5 去重 + 8:1:1 划分
│   ├── build_data_ladder.py    #   嵌套数据阶梯子集（250/500/1000）
│   └── build_fast_dataset.py   #   预解码 memmap（消除 PNG 解码瓶颈）
│
├── viz/                        # 图表
│   ├── make_figures.py         #   学习曲线 / 消融 / 每类 F1 / 同 tile 对比
│   └── make_stats_figures.py   #   混淆矩阵 / PRF / 参数效率 / LR 调度
│
├── report/                     # 汇总与报告
│   ├── build_index.py          #   实验索引 → experiments_index.json
│   ├── build_facts.py          #   事实汇总 → FACTS.json（报告唯一数据源）
│   ├── make_report_v2.py       #   报告 Markdown → HTML
│   └── make_pdf.py             #   HTML → PDF（Chrome headless）+ 内容自检
│
├── mapping/inference_fullmap.py# 全图滑窗推理 + shp 裁剪成图（制图管线）
│
├── docs/                       # 文档
│   ├── 架构有效性分析.md         #   三项已数值证实的架构缺陷 + 文献 + 修复路径
│   ├── verify_report.txt       #   实验产物核验输出
│   └── archive/                #   历史文档归档
├── legacy/                     # 历史脚本归档（18 个早期脚本 + 说明）
│
├── artifacts/                  # 中间分析产物（非权威事实）
├── figures/                    # 图表输出
├── checkpoints/                # 权重 + 训练历史（不入库）
├── FACTS.json                  # 全部结果的结构化汇总（报告唯一数据源）
├── experiments_index.json      # 全部实验登记 + 管线世代 + 可比性分组
├── seed_variance.json          # 多种子噪声底线结果
└── 项目报告_20260912.pdf        # 项目报告（交付件）
```

> **为什么核心库留在根目录**：`paths.py` 以 `REPO = dirname(abspath(__file__))`
> 作为**所有相对路径的基准**（`checkpoints/`、`figures/`、`FACTS.json` …）。
> 它一旦移动，全部输出路径都会错位。因此 `paths.py` 与 4 个核心模块
> （`train_v3.py` / `experiment_matrix*.py` / `train_full_data.py`）固定留在根目录；
> 子目录脚本通过注入的**路径引导块**（把仓库根与自身目录加入 `sys.path`）导入它们。

> **路径配置**：代码中不含本地绝对路径。本地使用时创建 `.env` 提供数据根目录
> （详见 [STRUCTURE.md](STRUCTURE.md) 第四节）。

**不入库**（体积大 / 可复现生成）：`checkpoints/`（权重与训练历史）、
`fast_dataset/`（预解码 memmap，约 9.9 GB）、
`fulltile_eval/` `tta_eval*/` `heldout_test/` `consensus_analysis/`（评估结果与预测数组）、
`项目报告_*.html`（可再生）、原始 LoveDA 数据。

---

## 快速开始

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
python prep/build_fast_dataset.py     # 预解码 memmap（强烈建议，提速约 10 倍）

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

# 5) 汇总与报告
python report/build_index.py          # 实验索引（参数量直读张量）
python report/build_facts.py          # 事实汇总 → FACTS.json
python report/make_report_v2.py       # 报告 HTML
python report/make_pdf.py             # HTML → PDF（含内容自检）
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
