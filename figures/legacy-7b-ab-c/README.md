# Qwen2.5-VL-7B 双卡 A/B/C 对照

运行 ID 为 `20261122T173622Z-7b-ab`。`qjy000`、`qjy001` 各用一张 H20。A 是两个完整副本。B 是 `qjy000` 编码、`qjy001` 共置 Prefill 和 Decode，采用固定编码器的 `B.no-probe` 回放。C70 和 C50 沿用 B 的卡位和拓扑，在发送前压缩视觉特征。C100 用于载荷和恒等检查。原 [A/B 三张历史图](../ab/) 可独立查阅。

| 图 | 样本与指标 |
|---|---|
| [01 固定多图矩阵](01-fixed-matrix.png) | 1、4、8 图，每个负载三个配对种子 `23011`、`23012`、`23013`，每种子每组 128 条，固定生成 64 token。汇总完成率保留全部提交分母，成功吞吐为完成数除以三个重复的观测时长之和。 |
| [02 官方随机多图](02-official-multi-image.png) | 0.2、0.4、0.6、0.8、1.0 QPS，每档每组 64 条，每请求随机 1–8 图，固定生成 256 token。 |
| [03 重图干扰](03-heavy-image-interference.png) | 每条件 10 条长流。A 保留同副本、跨副本路由。干扰条件按实际重图发送时刻对齐，无干扰条件按首块后 50 ms 对齐。展示相邻非空 SSE 内容块间隔，95% 区间按整条长请求重采样 2,000 次。 |
| [04 通信载荷](04-communication-payload.png) | C100/C70/C50 各六条同轨探针。应用层计量分别记录 E→PD 特征和序列化元数据、PD→E 的图像 HTTP 请求体。请求体内的控制字段随请求体一起计量。TCP 独立展示 B 的 E→PD 持久连接前后差分及各 C 组 PD→E 的发送记录。 |
| [05 质量与配对差](05-quality-paired.png) | OCRBench 1,000 题、RealWorldQA 765 题。相对 B 的 95% 配对区间用 5,000 次重采样，OCRBench 按图像 SHA-256 聚类，RealWorldQA 按题。 |
| [06 OCRBench 题型](06-ocrbench-types.png) | 原始十个题型的提交数、各组得分和相对 B 的变化。 |

TTFT 从客户端提交到首个非空 SSE 内容块。P95 使用成功请求，完成率使用全部提交。应用层与 TCP 部分各自说明方向、单位和样本。旧双卡的 MuirBench 记录限于 72 题预跑，四组全量和请求级 TPOT P99 补测随新 16 卡协议执行。

逐点分母和数据出处见 [性能汇总](performance-data.json) 与 [通信和质量汇总](payload-quality-data.json)。原始记录位于共享运行目录 `/home/boyjiang/mllm-epd/runs/20261122T173622Z-7b-ab`。用 Python、Matplotlib 和支持简体中文的字体可重新绘图。

```bash
python common/plot_legacy_7b_performance.py --data figures/legacy-7b-ab-c/performance-data.json --output-dir figures/legacy-7b-ab-c
python common/plot_legacy_7b_payload_quality.py --data figures/legacy-7b-ab-c/payload-quality-data.json --output-dir figures/legacy-7b-ab-c
```
