# tools/dev —— 开发期脚本（不是使用必需的）

这些文件是开发/调参/压测时用的，普通使用不需要，保留是为了可复现我当时的实测结论。

| 文件 | 作用 |
| --- | --- |
| `gpu_probe.py` | 探测 CUDA / ctranslate2 可用性、显存占用、加载模型耗时 |
| `asr_budget.py` | 微小基准：解码耗时 vs 推理耗时，估算全量转写预算 |
| `_lead_build_test.py` | 200 句媒体链路计时（拼接 / MP3 / MP4 编码速度） |
| `_lead_scale_test.py` | 全量母带拼接计时（验证 193x 实时） |
| `_lead_scene_audit.py` | 读 `filter_report.csv` 做场景可疑度统计、模拟整场剔除规则 |
| `_lead_finalize.py` | 批量收尾：逐文件夹 ASR → refine → build（我跑 4 个文件夹用的） |
| `_lead_after_mrxyy.sh` | 当初的串行编排脚本（已被 `_lead_finalize.py` 取代，留作参考） |

> 这些脚本里的路径是开发机上的示例路径（`E:\mrxyy` 等），按需改。

用户侧常用脚本在上一级 `tools/`：
`qa_check.py`（产物质检）、`calibrate_filter.py`（重新校准阈值）、`remote_selftest.py`（远程算力自测）、
`run_pipeline.py` / `run_all_folders.py`（多文件夹批量）。
