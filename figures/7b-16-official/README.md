# 7B 16 卡官方随机多图回放

[对照图](official.png) · [可重绘数据](official-report.json)

实测目录为 `20261123T093855Z-7b-16-campaign`。四组各用 16 张 H20。A 为 16 个完整 TP1 副本，B 与 C 为 8 个 E、4 个 P、4 个 D。C70/C50 在发送前将视觉 token 保留至约 70%/50%。五档名义请求率为 0.2、0.4、0.6、0.8、1.0 QPS。同一批 seed 3467 的 64 条请求跨档位、跨组回放，每条随机带 1–8 张 1080p 图，生成 256 token。四组共完成 1,280/1,280。

| 1.0 QPS | A | B | C70 | C50 |
|---|---:|---:|---:|---:|
| P95 TTFT（秒） | 4.664 | 3.469 | 2.646 | 1.966 |
| 成功吞吐（req/s） | 0.967 | 0.973 | 0.976 | 0.978 |

TTFT 以客户端收到第一个非空 SSE 内容块为终点，按完成请求计算 P95。吞吐以首条发出到末条请求终止的时长计算，包含排空。图中的五档用于按相同到达计划比较四组。每点的部署运行 ID、原始响应及回执 SHA-256 在 JSON 的 `cells[].official_provenance`，配对差存于 `pairs`。

在原实验工作区的仓库根目录，从封存汇总重绘到空目录。

```bash
PYTHONPATH=common python -c 'from pathlib import Path; from report_16card import plot, read; plot(read(Path("figures/7b-16-official/official-report.json")), Path("redraw"), None)'
```
