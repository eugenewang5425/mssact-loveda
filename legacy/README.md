# legacy/ — 历史脚本归档

本目录保存早期版本的脚本，**不参与当前实验流程**，仅供追溯参考。

| 脚本 | 早期用途 | 被谁替代 |
|---|---|---|
| `train.py` `train_v2.py` | 早期 patch 管线训练（npz） | `train_v3.py` + `train_full_data.py` |
| `baseline_unet.py` | U-Net 基线（早期协议） | `experiment_matrix.py`（统一协议） |
| `preprocess.py` | PNG → npz patch 预处理 | 直接读原图（`train_v3.py`） |
| `evaluate.py` `eval_tta.py` `eval_existing.py` `eval_fulltile.py` | 早期评估脚本 | `eval_complete.py` / `eval_tta_new.py` / `eval_ladder.py` |
| `visualize.py` `fig_test_showcase.py` | 可视化 | `make_figures.py` / `make_stats_figures.py` |
| `compare.py` `final_compare.py` `run_queue.py` `run_abl120.py` | 早期汇总与队列 | `build_facts.py` / `run_queue_*.py` |
| `make_project_report.py` `make_report_pdf.py` | 早期报告生成 | `make_report_v2.py` |

**注意**：
- 这些脚本仍可通过 `legacy/paths.py`（转发模块）导入仓库根目录的路径配置
- 运行前请先确认对应数据目录存在（路径由根目录 `.env` 提供）
- 当前实验请使用仓库根目录的脚本（见 `../STRUCTURE.md`）
