# 7B 16 卡固定图数回放

[六页 PDF 报告](7b-16card-report.pdf)

GitHub 审阅分支收录 PDF、图和汇总数据。下方重绘命令在原实验工作区运行，使用该工作区的 `common/` 绘图脚本。

[正式图](fixed.png) · [封存数据](fixed-matrix-verified.json)

Qwen2.5-VL-7B-Instruct，双机共 16 张 H20。A 为 16 个完整 TP1 副本。B、C70、C50 均为 8E＋4P＋4D，C 在发送前应用 VisionZip。每条请求带 1、4 或 8 张 1080p 图，生成 64 token。三轮使用配对种子 `23011`、`23012`、`23013`，每轮每组每档 128 条。各组各档均完成 **384/384**，108 个单元共完成 **13,824/13,824**。本图对应阶段补测前的固定矩阵。

图保留 P95 TTFT 和成功吞吐两行，各组每档全部完成 384/384（100%）。大点取三轮汇总，小点在各组大点右侧按种子顺序排列。横轴按名义 QPS 档位等距。P95 TTFT 取完成请求首个非空 SSE 内容块的客户端延迟，三轮合并后取分位数，单位秒，使用对数轴。成功吞吐为成功数除以三轮观察窗时长之和，观察窗从首条发出到末条终止，包含排空。各图数的吞吐纵轴分别定标。

八图重点如下。每个单元格为 **P95 TTFT（秒）／成功吞吐（req/s）**。

| 名义 QPS | A | B | C70 | C50 |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 4.805 ／ 0.516 | 3.859 ／ 0.518 | 2.725 ／ 0.520 | 2.086 ／ 0.521 |
| 1.5 | 5.053 ／ 1.487 | 22.184 ／ 1.319 | 5.235 ／ 1.516 | 2.822 ／ 1.535 |
| 4 | 11.279 ／ 2.861 | 58.809 ／ 1.406 | 31.168 ／ 2.110 | 14.986 ／ 2.995 |

八图 4 QPS 下，C50 的成功吞吐在三个种子上均高于 A，汇总高 4.7%。P95 首字仍为 14.99 秒对 11.28 秒。八图 0.5 QPS 时，B 把八图分发给多个 E，A 由单个副本处理。分发记录支持多 E 并行这一解释，独立时间贡献待单独度量。

| 八图 4 QPS | 完整副本 | E | P | D |
|---|---:|---:|---:|---:|
| A | 74.5% | — | — | — |
| B | — | 20.7% | 96.5% | 13.3% |
| C70 | — | 39.3% | 94.8% | 19.7% |
| C50 | — | 54.4% | 92.1% | 25.8% |

利用率以 `report-16card.json` 的三轮 `gpu_devices.samples` 加权，A 有 2,152 个完整副本样本，B/C70/C50 的 E/P/D 样本分别为 2,192/1,096/1,096、1,464/728/728、1,024/512/512。

Campaign 为 `20261123T093855Z-7b-16-campaign`。部署运行 ID、原始响应、回执路径和响应 SHA-256 保存在封存 JSON。输入文件 SHA-256 为 `098645c94cfd473e4cd89a3422d3f05bbedb126e6c5aae3d508fd8c1036ce42b`，同时写入 PNG 元数据。

在仓库根目录重绘。使用已有中文字体，或传 `--font` 指定字体文件。输出路径须为新文件。

```bash
python3 common/plot_16card_fixed.py \
  --input figures/7b-16-formal/fixed-matrix-verified.json \
  --output fixed-redraw.png
```

## 单图 TPOT 和重图干扰

[重图干扰图](interference.png) · [干扰汇总及实际路由](interference-summary.json) · [请求级 TPOT 汇总](tail-p99-summary.json) · [32 个种子回执审计](tail-p99-receipt-audit.json)

重图干扰每条件 10 次。共 90/90 条长流和 50/50 条八图请求完成并核对实际路由。A 的两个重图条件使用 qjy001 上的同一或另一完整副本。B/C 在 E/P/D 分离拓扑下插入重图。使用客户端重图提交时间对齐，无重图取长流首块后 50 ms。图中横轴的 long 统计有有效间隔的长流，heavy 统计完成的八图请求。相邻非空 SSE 内容块间隔按条件汇总为 P95/P99，95% 区间按长流重采样。

插入重图时，A 同副本、A 同节点另一副本、B 三段分离的 SSE 内容块间隔 P99 分别为 304.92、15.61、15.20 ms。TPOT 使用单图 4 QPS、生成 64 token 的独立配对回放，每组八个种子 1,024/1,024 有效。分子为调度器记录的首末 token 时间差，分母为 63。拆分组的时间差包含 P→D 交接。交接的独立贡献需分别度量首→次 token 和次→末 token 间隔。

在仓库根目录从汇总 JSON 重绘干扰图。

```bash
PYTHONPATH=common python3 -c 'import json,sys; from pathlib import Path; from compare_interference import plot; plot(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), Path(sys.argv[2]))' figures/7b-16-formal/interference-summary.json interference-redraw.png
```

## 通信、Prefill 与质量

[阶段和质量图](stage-quality.png) · [正式与诊断汇总](report-16card.json) · [阶段计时](phase-stage-report.json) · [TCP 窗口](tcp-summary.json) · [逐题质量配对](quality-paired-summary.json) · [MuirBench 同批通信审计](muirbench-request-traffic-audit.json) · [运行索引](evidence.json)

阶段图的 E→P 张量和元数据、Prefill 耗时取自每组 6 条相同探针。P→D 使用独立的 16-output-token Mooncake 资格探针，逐请求有 KV 页与接收校验。质量取四组普通部署的三个全量数据集，图中显示两条 B→C 配对差及 95% 区间。MuirBench 通信审计另在 1 QPS 下核对每组 2,600 条响应。B 的阶段探针使用 `20261124T000206Z-7b-16-b-phase-probes`，B 的同题通信数据使用 `20261123T230931Z-7b-16-b-phase`。

C50 八图探针的 8 个并行编码批次，取分中位数 31.315 ms，合并中位数 1.477 ms。B 缺同口径编码计时，端到端拆解留待同一请求的关键路径分析。

| MuirBench 2,600 题 | A | B | C70 | C50 |
|---|---:|---:|---:|---:|
| 跨节点应用流量总字节 | 3,788,778,943 | 91,613,328,411 | 66,361,514,312 | 49,552,719,311 |
| 全样本均值（MB/题） | 1.457 | 35.236 | 25.524 | 19.059 |
| 总量（GB） | 3.789 | 91.613 | 66.362 | 49.553 |
| 题对双正确（共 1,300 对） | 420 | 420 | 424 | 417 |

A 有 1,304/2,600 条请求跨机，仅跨机均值 2.906 MB/题。统计以十进制字节单位为准。总量和题对统计来自封存的逐请求记录 `muirbench-request-traffic.json`，其 SHA-256 为 `b8f68df6b2443d968c376bad00e453932762dbab7899ced2ed16779b5f8e0e54`。原有通信审计 JSON 保留中位数。

| 数据集 / 视觉 token 保留 | VisionZip 作者公开表 | 本地单卡 HF | 16 卡 SGLang |
|---|---:|---:|---:|
| OCRBench 100% | 81.5% | 76.7% | 82.2% |
| OCRBench 70% | 80.9% | 76.5% | 78.1% |
| OCRBench 50% | 70.5% | 73.3% | 68.3% |
| RealWorldQA 100% | 68.6% | 65.10% | 62.88% |
| RealWorldQA 70% | 68.2% | 64.58% | 63.14% |
| RealWorldQA 50% | 68.6% | 64.31% | 62.61% |

两条本地路线使用 7B 和冻结 PNG，图文顺序不同。JPEG 的影响来自默认 Qwen 适配器的抽样预处理诊断，具体分项贡献留待同输入、同精度核对。 [VisionZip 作者公开表](https://github.com/JIA-Lab-research/VisionZip/blob/8f86b55c6f000eb033e6912538af2dd7dcb30502/Qwen2_5_VL/README.md#results)

原始日志、分片与逐题响应保留在 qjy000 和 qjy001 的 `/home/boyjiang/mllm-epd/runs`。阶段、质量和计时输入的来源可按运行索引和各汇总中的 SHA-256 核查。官方五档性能单列于[随机多图图](../7b-16-official/official.png)。

```bash
python3 common/plot_16card_stage_quality.py \
  --transport figures/7b-16-formal/report-16card.json \
  --stage figures/7b-16-formal/phase-stage-report.json \
  --quality figures/7b-16-formal/quality-paired-summary.json \
  --output stage-quality-redraw.png
```
