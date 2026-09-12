# 项目结构说明（STRUCTURE）

> 本文档说明项目文件组织、**执行链依赖**（不可移动）、数据与结果位置。
> 修改任何文件前请先阅读「执行链」一节。

---

## 一、执行链依赖（⚠️ 不可移动 / 不可重命名）

所有训练与评估脚本均通过 `os.path.dirname(os.path.abspath(__file__))` 定位自身，
并用**同目录导入**与**硬编码目录名**引用资源。因此以下文件必须留在
`loveda/` 根目录：

```
run_queue_new.py / run_queue_30m.py / run_queue_interaction.py   (实验队列)
run_queue_arch.py / run_queue_batch.py / run_queue_final.py       (架构/多种子队列)
        │
        ├─→ experiment_matrix.py ──┐
        ├─→ experiment_matrix_v2.py┤
        └─→ train_full_data.py ────┤
                                   ├─→ train_v3.py ──→ models/msscactnet.py
                                   │      （LoveDADataset / REMAP / compute_stats）
                                   │
        eval_complete.py ──────────┤
        eval_tta_new.py ───────────┤
        eval_ladder.py ────────────┤
        eval_rigor.py ─────────────┤
        eval_fulltile.py ──────────┤
        consensus_analysis.py ─────┤
        overfit_diag.py ───────────┘
```

**硬编码的关键常量**（改动会导致实验不可复现）：

| 文件 | 常量 | 作用 |
|---|---|---|
| `train_v3.py` | `ROOT` | 默认数据根目录 |
| `train_v3.py` | `REMAP` | **官方标签映射：0(no-data)→255(ignore)；1..7→0..6** |
| `train_v3.py` | `CLASS_NAMES` | 类别名称顺序（背景/建筑/道路/水域/裸地/森林/农田） |
| `experiment_matrix.py` | `train_one(...)` | 统一训练入口（所有实验共用）；`seed=42` 默认值与原行为一致 |
| `(*).py` | `CKPT` | checkpoints 目录 |

**目录名被脚本引用**（同样不可移动）：`figures/`、`checkpoints/`、
`fast_dataset/`、`fulltile_eval/`、`tta_eval/`、`tta_eval2/`、`heldout_test/`、
`consensus_analysis/`、`overfit_diag/`、`ladder_eval/`。

---

## 二、文件组织

### 2.1 模型与训练核心

| 文件 | 说明 |
|---|---|
| `models/msscactnet.py` | MSSACT-Net 定义（EMR/ECSAM/FPN/Transformer/Adapter 可开关） |
| `train_v3.py` | **核心**：数据集类、标签映射 REMAP、训练循环、评估函数 |
| `experiment_matrix.py` | 通用训练入口 `train_one()` + 基线模型（U-Net/PSPNet/FCN/DeepLabV3+/SegFormer-Lite） |
| `experiment_matrix_v2.py` | 扩展基线（FPN/Swin-Unet-Lite）+ 消融构造辅助 |
| `train_full_data.py` | 全量数据训练与后训练（主模型专用） |
| `baseline_unet.py` `train.py` `train_v2.py` | 早期版本（历史保留，不参与当前实验） |

> ⚠️ **架构缺陷登记**：`models/msscactnet.py` 存在三处**已数值证实**的结构性缺陷
> （FPN 多尺度融合未被使用 / 解码器无跳连 / Transformer 无位置编码），
> 详见 [`docs/架构有效性分析.md`](docs/架构有效性分析.md)。修复前，
> `use_fpn` / `use_transformer` / `use_ecsam` 的消融结果**不能**解释为
> "模块有效性"的证据。

### 2.2 实验队列（自动串行）

| 文件 | 内容 |
|---|---|
| `run_queue_new.py` | PHASE-0 主模型重训 → PHASE-1 对比(7) → PHASE-2 消融(8) |
| `run_queue_30m.py` | 30.5M 原版架构 → 后训练 → TTA 评估 |
| `run_queue_interaction.py` | 实验A 预算攻击(15轮) / 实验B 固定LR / 实验C 数据阶梯 |
| `run_queue_arch.py` | **架构实验**：D1 联合消融 / D2 解码器CA / D3 通道缩小 / D4 通道放大 + 预算攻击 + 数据阶梯（`lgR_*` 前缀） |
| `run_queue_batch.py` | batch 缩放对照（`bs8` vs `bs16` + 线性缩放 LR） |
| `run_queue_final.py` | **多种子噪声底线**（3 seed × {完整模型, DeepLabV3+}）→ 22 tag 全量推理 → 严格性评估 → 汇总产物 |
| `monitor_queue.py` | 进度监控（动态刷新、指标、ETA） |
| `run_abl120.py` `run_queue.py` | 早期队列（已被上述替代，历史保留） |

**队列特性**：断点续跑（按 `*_history.json` 轮数判定完成）、失败重试、
OOM 自动降 batch、训练后校验、**互斥等待**（检测到其他 `run_queue*` 进程时挂起）。

### 2.3 数据准备

| 文件 | 说明 |
|---|---|
| `resplit_filtered.py` | no-data ≤10% 筛选 + MD5 去重 + 8:1:1 划分 |
| `verify_split2.py` | 划分泄漏校验（三集互斥 + test_clean 隔离） |
| `build_data_ladder.py` | 嵌套数据阶梯子集（250/500/1000），分层抽样 |
| `preprocess.py` | 早期 patch 预处理（历史保留） |

### 2.4 评估与分析

| 文件 | 说明 |
|---|---|
| `eval_complete.py` | val/test × {有/无 TTA} 全量评估 |
| `eval_tta_new.py` | 新数据（newsplit2）TTA 评估，含正确 REMAP |
| `eval_rigor.py` | **严格性评估**：逐图指标 / 逐类 IoU / 配对 bootstrap CI / TOST 等效性检验 / 紧边界分层（`--infer` 可补齐缺失 tag 的推理并缓存） |
| `fix_boundary_analysis.py` | **修正**边界分层定义（原实现把 `(N,H,W)` 沿 `axis=0` 求差，跨图像素差被误判为边界；改为逐图 2D + 仅有效类别之间） |
| `headroom_analysis.py` | **Kappa 归属分解**：oracle 作弊实验，量化"完美解决边界/内部"各自能换取的 Kappa 上限 |
| `build_index.py` | **实验索引生成器** → `experiments_index.json`（参数量直读张量 / 架构形状反推 / 管线世代与可比性分组） |
| `eval_fulltile.py` | 全图滑窗推理 + 高斯融合评估 |
| `eval_ladder.py` | 数据阶梯评价：泛化能力 / 过拟合度 / 学习速率 |
| `consensus_analysis.py` | 标签质量一致性分析（模型间一致率 + 共识标签） |
| `overfit_diag.py` | 过拟合诊断（训练/验证 logits 熵差） |
| `verify_experiments.py` | 实验产物核验（架构推断 / history 完整性 / 交叉一致） |
| `verify_rollback.py` | 回退管线性能验证 |
| `evaluate.py` `visualize.py` `compare.py` `final_compare.py` `eval_existing.py` | 早期评估脚本（历史保留） |

### 2.5 可视化与报告

| 文件 | 说明 |
|---|---|
| `build_facts.py` | **事实汇总** → `FACTS.json`（报告与 README 的唯一数据源） |
| `make_figures.py` | 学习曲线 / 消融图 / 每类F1 / 同 tile 对比 |
| `make_stats_figures.py` | 混淆矩阵 / PRF / 参数效率 / LR 调度图 |
| `fig_test_showcase.py` | 测试集 patch 预测对比图 |
| `make_report_v2.py` | **当前报告生成器**（数据驱动，Markdown→HTML） |
| `make_pdf.py` | **HTML→PDF**（Chrome headless）+ 页数与**关键内容自检**（防止回退为错误结论） |
| `make_project_report.py` `make_report_pdf.py` | 早期报告生成器（历史保留） |
| `inference_fullmap.py` | GF-1 全图滑窗推理 + shp 裁剪成图（制图管线） |

> ⚠️ `make_pdf.py` 必须显式传 `--user-data-dir`：用户自身的 Chrome 占用默认
> profile，否则 headless 实例会因等不到 profile 锁而挂起（实测 600 s 超时）。

### 2.6 结果与产物

| 路径 | 内容 | 入库 |
|---|---|---|
| `checkpoints/` | 权重（`*_best.pt`）+ 训练历史（`*_history.json`） | ❌ |
| `fast_dataset/` | 预解码 memmap（`*_images.npy` / `*_masks.npy` / `index.json`，约 9.9 GB） | ❌ |
| `fulltile_eval/` | 全图滑窗评估结果 + 预测 PNG | ❌ |
| `tta_eval/` `tta_eval2/` | TTA 评估结果 | ❌ |
| `heldout_test/` | 早期持有测试集评估 | ❌ |
| `consensus_analysis/` | 一致性分析 + 预测数组 + `boundary_analysis.json`（含 `.bak_*` 备份）+ `rigor.json` + `headroom.json` | ❌ |
| `overfit_diag/` | 过拟合诊断结果 | ❌ |
| `ladder_eval/` | 数据阶梯评价结果 | ❌ |
| `figures/` | 26 张图表 | ✅ |
| `FACTS.json` | 全部结果结构化汇总 | ✅ |
| `experiments_index.json` | 全部实验登记（69 tag）+ 管线世代 + 可比性分组 | ✅ |
| `docs/架构有效性分析.md` | **架构缺陷专项分析**（三项数值验证的缺陷 + 文献 + 修复路径） | ✅ |
| `docs/archive/` | 历史版本报告归档 | ✅ |
| `项目报告_20260912.pdf` | 当前项目报告（33 页） | ✅ |

---

## 三、实验阶段与数据版本

| 阶段 | 数据 | 标签处理 | 对应实验前缀 | 结论地位 |
|---|---|---|---|---|
| **A** | 957 张（镜像子集） | ⚠️ 含 no-data 映射缺陷 | `v3_*` `ablate_*` `abl120_*` `strat_*` | 历史参考 |
| **B** | 2,257 张（官方全量去重） | ⚠️ 仍含缺陷 | `full_all` `post_all` | 中间结果 |
| **C** | **1,768 张（筛选后）** | **✅ 已修复** | **`nd_*` `full_all_v2` `post_all_v2`** | **结论依据** |

**标签映射修复要点**：官方标签 `1..7` 为类别（背景/建筑/道路/水域/裸地/森林/农田），
`0` 为 no-data **必须忽略**。早期实现误将 `0` 映射为「背景」，导致约 3% 像素
错误参与训练与评估，指标虚高约 0.007–0.009 Kappa。

### 三之二、训练管线世代（⚠️ 决定"哪些实验可以比较"）

Kappa **禁止跨管线世代比较**。各世代由 `checkpoint`/日志时间戳核实
（`fast_dataset/*.npy` 建立于 2026-09-12 14:51–14:53）：

| 世代 | 数据投递 | 增强随机性 | `num_workers` | 实验前缀 | 地位 |
|---|---|---|---|---|---|
| **P-PNG** | 每样本解码两张 1024² PNG | 全局 RNG | 0 | `nd_*` `lg_*` `abl120_*` 等 | 正式对比/消融结果 |
| **P-MEM-DET** | memmap 预解码 | **确定性 generator** | 8 | `lgF_*` | ❌ **已否决**（Kappa −0.0246） |
| **P-MEM-ROLL** | memmap 预解码 | 全局 RNG | 0 | `lgR_*` | 架构实验正式结果 |

**关键事实**：加速的真实来源是 **memmap 预解码**（消除 PNG 解码），而非增强或并发
改动。回退验证：`lgR_D1` = **0.6293** vs `lg_D1` = **0.6277**（+0.0016，噪声内），
每轮耗时由 144 s 降至 14.7 s。

**可比性分组**（详见 `experiments_index.json` 的 `comparable_groups`）：

| 组 | 管线 | 协议 | 成员 |
|---|---|---|---|
| **G1-PNG-60ep** | P-PNG | 60ep/pt20/bs8/lr2e-4 | `full_all_v2` + `nd_abl_*`(8) + `lg_D1..D4` + `nd_*` 基线(7) |
| **G2-MEM-ROLL-60ep** | P-MEM-ROLL | 同上 | `lgR_bs8_full` + `lgR_D1..D4` |
| **G3-b15** | P-PNG | 15ep/pt15/bs8/lr2e-4 | `lg_b15_*`(5) |

**跨组差异已实测**：仅数据投递后端不同（memmap vs PNG），效应为 **+0.0008**
（回退管线完整模型 0.6320 vs PNG 管线 0.6312），故两组的架构结论可互相印证。

---

## 四、路径配置（仓库不含本地绝对路径）

路径通过 **`paths.py`** 集中管理，优先级：**环境变量 > 本地 `.env` > 仓库相对默认值**。

本地使用时在仓库根目录创建 `.env`（已被 `.gitignore` 排除，不会入库）：

```bash
# .env 示例（路径按本地实际情况填写）
LDA_DATA_ROOT=<数据根目录>      # 内含 newsplit2/ loveda_official/ data_ladder/ 等
LDA_CKPT=<检查点目录>
LDA_GF_DATA=<GF 影像目录>       # 可选，制图管线用
LDA_GF_TIF=<多波段标签影像>
LDA_GF_SHP=<行政区矢量>
```

| 变量 | 默认值 | 用途 |
|---|---|---|
| `LDA_DATA_ROOT` | `./_data` | 数据根目录 |
| `LDA_DATA_NEWSPLIT2` | `$DATA_ROOT/newsplit2` | 当前使用（筛选 + 划分） |
| `LDA_DATA_NEWSPLIT` | `$DATA_ROOT/newsplit` | 中间版本（阶段 B） |
| `LDA_DATA_OFFICIAL` | `$DATA_ROOT/loveda_official` | 官方全量训练区 |
| `LDA_DATA_LADDER` | `$DATA_ROOT/data_ladder` | 数据阶梯子集 |
| `LDA_CKPT` | `./checkpoints` | 权重与训练历史 |
| `LDA_GF_DATA` | `./_gf_data` | GF 影像（制图管线） |

> 仓库代码中**不包含**任何本地绝对路径；所有脚本通过 `import paths` 获取路径，
> 因此可直接在其他机器上复现（只需提供 `.env` 或设置环境变量）。


## 五、维护约定

1. **实验产物命名**：`{前缀}_{变体}_history.json` + `{前缀}_{变体}_best.pt`
   - 阶段 A：`v3_*` / `ablate_*` / `abl120_*` / `strat_*`
   - 阶段 B：`full_all` / `post_all`
   - 阶段 C：`nd_*` / `full_all_v2` / `post_all_v2`
2. **新增实验**：在 `run_queue_*.py` 的 JOBS 列表中添加，**不改动已有条目**；
   队列会自动跳过已完成任务（按 history 轮数判定）
3. **结果更新后**：运行 `build_facts.py` 刷新 `FACTS.json`，
   再运行 `make_report_v2.py` 重生成报告，保证数字一致
4. **不移动**：执行链文件、结果目录（见第一节）
5. **历史归档**：旧版报告、旧 README 放入 `docs/archive/`
