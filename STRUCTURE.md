# 项目结构说明（STRUCTURE）

> 本文档说明项目文件组织、**执行链依赖**（不可移动）、数据与结果位置。
> 修改任何文件前请先阅读「执行链」一节。

---

## 一、执行链依赖（⚠️ 不可移动 / 不可重命名）

### 1.1 仓库根目录固定项

`paths.py` 以

```python
REPO = os.path.dirname(os.path.abspath(__file__))
```

作为**所有相对路径的基准**（`checkpoints/`、`figures/`、`consensus_analysis/`、
`FACTS.json` …）。它一旦移动，全部输出路径都会错位。因此以下 5 个文件
**固定留在仓库根目录**：

| 文件 | 作用 |
|---|---|
| `paths.py` | **路径枢纽**：全部数据/产物路径的唯一来源 |
| `train_v3.py` | 数据集类 `LoveDADataset`、标签映射 `REMAP`、评估函数 |
| `experiment_matrix.py` | 统一训练入口 `train_one()` + 基线模型 |
| `experiment_matrix_v2.py` | 扩展基线（FPN / Swin-Unet） |
| `train_full_data.py` | 全量数据训练（主模型 + 后训练） |

### 1.2 子目录脚本如何找到它们

每个子目录脚本在模块 docstring 之后有一段**路径引导块**（由迁移时自动注入）：

```python
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
```

- 导入 `paths` / `train_v3` / `experiment_matrix*` → 走 `_BASE`（仓库根）
- 导入同目录模块（如 `queues/` 内的 `queue_guard`）→ 走 `_HERE`

因此脚本**可从任意工作目录调用**（不必先 `cd` 到仓库根）：

```bash
python analysis/eval_rigor.py      # 等效于 cd loveda && python analysis/eval_rigor.py
```

### 1.3 调用关系

```
queues/run_chain.py                     （静默等待 → 串行驱动下面两步）
        │
        ├─→ queues/run_queue_final.py ──→ analysis/eval_rigor.py
        │                                analysis/eval_ladder.py
        │                                analysis/headroom_analysis.py
        │                                analysis/seed_variance.py
        │                                report/build_index.py
        │                                report/build_facts.py
        │                                report/make_report_v2.py
        │                                report/make_pdf.py
        └─→ queues/run_queue_crop.py  ──→ （同上后置步骤）

queues/run_queue_*.py ──→ experiment_matrix.py ──┐
                                                 ├─→ train_v3.py ──→ models/msscactnet.py
analysis/*.py ───────────────────────────────────┘
verify/*.py   ───────────────────────────────────┘
```

> **子目录路径是硬编码的**：队列脚本用 `os.path.join(BASE, "<子目录>", "<脚本>")`
> 调用后置步骤。移动子目录或改名必须同步修改 `queues/run_queue_final.py`、
> `queues/run_queue_crop.py`、`queues/run_chain.py` 中的字符串。

### 1.4 硬编码的关键常量（改动会导致实验不可复现）

| 文件 | 常量 | 作用 |
|---|---|---|
| `train_v3.py` | `ROOT` | 默认数据根目录 |
| `train_v3.py` | `REMAP` | **官方标签映射：0(no-data)→255(ignore)；1..7→0..6** |
| `train_v3.py` | `CLASS_NAMES` | 类别顺序（背景/建筑/道路/水域/裸地/森林/农田） |
| `experiment_matrix.py` | `train_one(..., seed=42, crop=256, center_crop=False)` | 统一训练入口；三个默认值均与原行为一致 |
| `queues/queue_guard.py` | `CREATE_NO_WINDOW` | **不可去掉**：否则轮询会弹出终端窗口（见 2.2） |
| `analysis/eval_rigor.py` | `SESOI = 0.02` | TOST 等效性检验的最小可关注效应 |

### 1.5 目录名被脚本引用（同样不可移动）

`figures/`、`checkpoints/`、`fast_dataset/`、`fulltile_eval/`、`tta_eval/`、
`tta_eval2/`、`heldout_test/`、`consensus_analysis/`、`overfit_diag/`、`ladder_eval/`。

---

## 二、文件组织

根目录只保留路径枢纽、核心库、权威事实与交付件；其余脚本按用途分 7 个子目录。

### 2.1 模型与训练核心（根目录）

| 文件 | 说明 |
|---|---|
| `models/msscactnet.py` | MSSACT-Net 定义（EMR/ECSAM/FPN/Transformer/Adapter 可开关） |
| `train_v3.py` | **核心**：数据集类、标签映射 REMAP、训练循环、评估函数 |
| `experiment_matrix.py` | 通用训练入口 `train_one()` + 基线（U-Net/PSPNet/FCN/DeepLabV3+/SegFormer-Lite） |
| `experiment_matrix_v2.py` | 扩展基线（FPN/Swin-Unet-Lite）+ 消融构造辅助 |
| `train_full_data.py` | 全量数据训练与后训练（主模型专用） |

> ⚠️ **架构缺陷登记**：`models/msscactnet.py` 存在三处**已数值证实**的结构性缺陷
> （FPN 的多尺度融合未被使用 / 解码器无跳连 / Transformer 无位置编码），
> 详见 [`docs/架构有效性分析.md`](docs/架构有效性分析.md)。修复前，
> `use_fpn` / `use_transformer` / `use_ecsam` 的消融结果**不能**解释为
> "模块有效性"的证据。

### 2.2 `queues/` — 实验队列（自动串行）

| 文件 | 内容 |
|---|---|
| `run_chain.py` | **统一等待链**：等所有 `run_queue*` 结束后，依次跑 final → crop |
| `run_queue_new.py` | PHASE-0 主模型重训 → 对比(7) → 消融(8) |
| `run_queue_arch.py` | 架构实验 D1-D4 + 预算攻击(15轮) + 数据阶梯（`lgR_*` 前缀） |
| `run_queue_final.py` | **多种子噪声底线**（3 seed × {完整模型, DeepLabV3+}）→ 22 tag 推理 → 后置步骤 |
| `run_queue_crop.py` | 裁剪策略对比（256 随机/固定中心、384、512） |
| `run_queue_30m.py` | 30.5M 原版架构 → 后训练 → TTA |
| `run_queue_interaction.py` | 预算攻击(15轮) / 固定 LR |
| `run_queue_batch.py` | batch 缩放对照（bs8 vs bs16 + 线性缩放 LR） |
| `monitor_queue.py` | 进度监控（动态刷新、逐模型指标、ETA） |
| `eta_queue.py` | **剩余时间估算**：按实测每轮耗时逐项累加，自动扣除已完成轮次 |
| `queue_guard.py` | 队列互斥守卫 |

**队列特性**：断点续跑（按 `*_history.json` 轮数判定完成）、失败重试、
OOM 自动降 batch、训练后校验、**互斥等待**（检测到其他 `run_queue*` 进程时挂起）。

> ⚠️ **`queue_guard.py` 的 `CREATE_NO_WINDOW` 不可去掉**。此前用
> `subprocess.run(["powershell", ...])` 轮询进程列表，而队列由 `nohup ... &`
> 从无控制台会话启动，powershell 会申请**新的控制台**——每轮询一次弹出一个
> 终端窗口（实测每 300 秒一次）。`CREATE_NO_WINDOW`(0x08000000) + `SW_HIDE`
> 抑制该行为。**单一等待点**（只由 `run_chain.py` 轮询）也是为此。

### 2.3 `analysis/` — 评估与分析

| 文件 | 说明 |
|---|---|
| `eval_rigor.py` | **严格性评估**：逐图指标 / 逐类 IoU / 配对 bootstrap CI / TOST / 紧边界分层（`--infer` 补齐缺失 tag 的推理并缓存） |
| `eval_ladder.py` | 数据阶梯评价：泛化能力 / 过拟合度 / 学习速率 |
| `eval_complete.py` | val/test × {有/无 TTA} 全量评估 |
| `eval_tta_new.py` | 新数据（newsplit2）TTA 评估，含正确 REMAP |
| `consensus_analysis.py` | 标签质量一致性分析（模型间一致率 + 共识标签） |
| `headroom_analysis.py` | **Kappa 归属分解**：oracle 作弊实验，量化"完美解决边界/内部"各自能换取的 Kappa 上限 |
| `artifact_analysis.py` | **解码器伪影分析**：边缘密度 + FFT 功率占比（棋盘伪影） |
| `fix_boundary_analysis.py` | **修正**边界分层定义（原实现把 `(N,H,W)` 沿 `axis=0` 求差，跨图像素差被误判为边界） |
| `overfit_diag.py` | 过拟合诊断（训练/验证 logits 熵差） |
| `seed_variance.py` | **随机种子噪声底线**：σ_seed + ΔKappa 折算为 σ 倍数 |

### 2.4 `verify/` — 校验与基准

| 文件 | 说明 |
|---|---|
| `verify_experiments.py` | 实验产物核验（架构推断 / history 完整性 / 交叉一致） |
| `verify_rollback.py` | 回退管线性能验证 |
| `verify_split2.py` | 划分泄漏校验（三集互斥 + test_clean 隔离） |
| `bench_parity.py` | PNG 与 memmap 两条管线的数值一致性 |
| `bench_pipeline.py` | 管线提速基准 |
| `sample_gpu_now.py` | GPU 利用率采样 |

### 2.5 `prep/` — 数据准备

| 文件 | 说明 |
|---|---|
| `resplit_filtered.py` | no-data ≤10% 筛选 + MD5 去重 + 8:1:1 划分 |
| `build_data_ladder.py` | 嵌套数据阶梯子集（250/500/1000），分层抽样 |
| `build_fast_dataset.py` | 预解码主数据集 memmap（`fast_dataset/`），消除 PNG 解码瓶颈 |
| `build_ladder_memmap.py` | 预解码**数据阶梯子集** memmap（`fast_dataset/ladder_n{250,500,1000}/`） |

> ⚠️ 两个预解码脚本都是**重量级磁盘 I/O**（解码 PNG + 写数 GB），会与训练的
> memmap 随机读争抢磁盘——实测可使轮次耗时恶化约 10 倍（D4 从 87 s/轮 到 866 s/轮）。
> **务必在训练队列停止时运行。**

### 2.6 `viz/` 与 `report/`

| 文件 | 说明 |
|---|---|
| `viz/make_figures.py` | 学习曲线 / 消融图 / 每类 F1 / 同 tile 对比 |
| `viz/make_stats_figures.py` | 混淆矩阵 / PRF / 参数效率 / LR 调度图 |
| `report/build_index.py` | **实验索引** → `experiments_index.json`（参数量直读张量 / 架构形状反推 / 管线世代与可比性分组） |
| `report/build_facts.py` | **事实汇总** → `FACTS.json`（报告与 README 的唯一数据源） |
| `report/make_report_v2.py` | 报告生成（Markdown → HTML） |
| `report/make_pdf.py` | HTML → PDF（Chrome headless）+ **内容自检** |

> ⚠️ `make_pdf.py` 必须显式传 `--user-data-dir`：用户自身的 Chrome 占用默认
> profile，否则 headless 实例会因等不到 profile 锁而挂起（实测 600 s 超时）。

### 2.7 `mapping/` 与归档

| 文件 | 说明 |
|---|---|
| `mapping/inference_fullmap.py` | GF-1 全图滑窗推理 + shp 裁剪成图（制图管线） |
| `legacy/` | 历史脚本归档（18 个早期脚本，不参与当前实验） |
| `docs/` | `架构有效性分析.md`、`verify_report.txt`、`archive/`（历史文档） |

### 2.8 结果与产物

| 路径 | 内容 | 入库 |
|---|---|---|
| `checkpoints/` | 权重（`*_best.pt`）+ 训练历史（`*_history.json`） | ❌ |
| `fast_dataset/` | 预解码 memmap：主数据集（约 9.9 GB）+ 阶梯子集 `ladder_n*`（约 7.3 GB） | ❌ |
| `fulltile_eval/` `tta_eval*/` `heldout_test/` | 全图/TTA/持有测试评估 + 预测 PNG | ❌ |
| `consensus_analysis/` | 一致性分析 + 预测数组 + `boundary_analysis.json`（含 `.bak_*`）+ `rigor.json` + `headroom.json` + `artifact.json` | ❌（仅汇总 JSON ✅） |
| `overfit_diag/` `ladder_eval/` | 诊断与阶梯评价结果 | ❌ |
| `figures/` | 图表（报告/README 引用） | ✅ |
| `artifacts/` | 中间分析产物（`convergence_learning_amount.json` 等） | ✅ |
| `FACTS.json` | **权威事实**：全部结果结构化汇总 | ✅ |
| `experiments_index.json` | **权威事实**：实验登记 + 管线世代 + 可比性分组 | ✅ |
| `seed_variance.json` | **权威事实**：多种子噪声底线 | ✅ |
| `项目报告_20260912.pdf` | 项目报告（交付件） | ✅ |
| `项目报告_*.html` | 报告 HTML（可再生，故不入库） | ❌ |

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

Kappa **禁止跨管线世代比较**。各世代由 checkpoint/日志时间戳核实
（`fast_dataset/*.npy` 建立于 2026-09-12 14:51–14:53）：

| 世代 | 数据投递 | 增强随机性 | `num_workers` | 实验前缀 | 地位 |
|---|---|---|---|---|---|
| **P-PNG** | 每样本解码两张 1024² PNG | 全局 RNG | 0 | `nd_*` `lg_*` `abl120_*` 等 | 正式对比/消融结果 |
| **P-MEM-DET** | memmap 预解码 | **确定性 generator** | 8 | `lgF_*` | ❌ **已否决**（Kappa −0.0246） |
| **P-MEM-ROLL** | memmap 预解码 | 全局 RNG | 0 | `lgR_*` `sd*_*` | 架构实验正式结果 |

**关键事实**：加速的真实来源是 **memmap 预解码**（消除 PNG 解码），而非增强或并发
改动。回退验证：`lgR_D1` = **0.6293** vs `lg_D1` = **0.6277**（+0.0016，噪声内），
每轮耗时由 144 s 降至 14.7 s。

**可比性分组**（详见 `experiments_index.json` 的 `comparable_groups`）：

| 组 | 管线 | 协议 | 成员 |
|---|---|---|---|
| **G1-PNG-60ep** | P-PNG | 60ep/pt20/bs8/lr2e-4 | `full_all_v2` + `nd_abl_*`(8) + `lg_D1..D4` + `nd_*` 基线(7) |
| **G2-MEM-ROLL-60ep** | P-MEM-ROLL | 同上 | `lgR_bs8_full` + `lgR_D1..D4` + `sd*_*` |
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

---

## 五、维护约定

1. **实验产物命名**：`{tag}_history.json` + `{tag}_best.pt`
   - 阶段 A：`v3_*` / `ablate_*` / `abl120_*` / `strat_*`
   - 阶段 B：`full_all` / `post_all`
   - 阶段 C：`nd_*` / `full_all_v2` / `post_all_v2`
   - 架构实验：`lg_*`（旧管线，仅参照）/ `lgF_*`（**已否决，勿引用**）/ `lgR_*`
   - 多种子：`sd{seed}_full` / `sd{seed}_deeplab`
   - **一律使用新文件名**，不覆盖既有产物；覆盖前先备份
2. **新增实验**：在 `queues/run_queue_*.py` 的 JOBS 列表中添加，**不改动已有条目**；
   队列会自动跳过已完成任务（按 history 轮数判定）
3. **新增脚本**：放入对应子目录即可（路径引导块会处理导入）。
   **但若被队列按文件名调用，必须同步更新 `queues/` 中的路径字符串**
4. **结果更新后**：依次运行
   `report/build_index.py` → `report/build_facts.py` →
   `report/make_report_v2.py` → `report/make_pdf.py`
   （`build_index` 必须在前：`build_facts` 会读取它；顺序颠倒会读到过期索引）
5. **不移动**：`paths.py`、4 个核心模块、结果目录（见第一节）
6. **历史归档**：旧版报告、旧 README 放入 `docs/archive/`；
   被取代的脚本放入 `legacy/`
