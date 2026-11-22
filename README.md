# SGLang E/PD 与 VisionZip 复现

用两台机器、每台一张 H20 复现 Qwen2.5-VL-7B-Instruct 的跨节点推理，并在 7B 结果冻结后，用 Qwen3-VL-235B-A22B-Instruct-FP8 复测阶段分离的规模趋势。

| 目录 | 实验 | 比较 |
|---|---|---|
| [A](A/) | 两台各跑一个完整 7B 副本，客户端轮询 | 相同两卡预算的起点 |
| [B](B/) | 第一台运行原生视觉编码 E，第二台运行共置的 Prefill 和 Decode | A→B 测部署变化对时延、生成卡顿、吞吐和完成率的影响 |
| [C](C/) | 保持 B 的拓扑，在 E 发送前执行 VisionZip | B→C100 校验恒等传输，B→C70/C50 测时延、流量和质量变化 |
| [235B](235B/) | 同机 4 卡共置、5 卡 1E+TP4、6 卡 2E+TP4 | 复测 2–3 个预先登记的负载点，报告总吞吐和每卡吞吐 |
| [common](common/) | 生成和回放请求，运行官方评分和指标汇总 | 各组共用数据、负载、统计口径和逻辑请求 ID |

## 固定版本

| 组件 | 修订 |
|---|---|
| [SGLang](https://github.com/sgl-project/sglang) | `8396d60a14ef03b23d2cdaa5c64adb18eba7c703` |
| [VisionZip](https://github.com/JIA-Lab-research/VisionZip) | `8f86b55c6f000eb033e6912538af2dd7dcb30502` |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | `d38f640d302911fcd3e83bf2f99f7dbac4ab9b89` |
| [MuirBench](https://github.com/muirbench/MuirBench) | `840b85fe960ab6160cd566fe8360bfa80f9164db` |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| [Qwen3-VL-235B-A22B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8) | `7fbcd8c9e2ad011808ed8a57d64c462605c3e381` |

两端使用同一 SGLang Python 环境。已通过 7B GPU 预跑的组合为 CUDA 13.0、PyTorch 2.14.1、Transformers 5.19.0、`sglang-kernel` 0.4.11 和 FlashInfer 0.7.4。驱动兼容按 NVIDIA 支持表检查。C 的单机作者参照单独使用 Transformers 4.49.0。[SGLang 依赖定义](https://github.com/sgl-project/sglang/blob/8396d60a14ef03b23d2cdaa5c64adb18eba7c703/python/pyproject.toml) · [NVIDIA CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)

## 复现顺序

1. 为每台服务器准备同一 SGLang 修订和权重，校验环境、GPU UUID、双向带宽和 RTT。用两台机器固定的 GPU 顺序完成 A、B、C。
2. 用 [正式矩阵](plans/formal-design.json) 冻结每个条件的输入、到达时刻、种子和输出长度。先运行 A，再回放 B 和 C70/C50。C100 用于传输等价验证。
3. 在 7B 上另跑随机多图工作负载、长生成受 8 图请求干扰，以及三套全量质量数据。逐请求保留分流、完成情况、流式输出和时间，逐题保留原始答案与评分。
4. 冻结 7B 结果后执行 [235B](235B/) 的规模对照。原文配置的占卡数不同，报告总吞吐、每卡吞吐、TTFT 和完成率。[SGLang H20 实验](https://www.lmsys.org/blog/2026-01-12-epd/#benchmark)

性能所用的合成输入固定输出长度，评分题使用自然 EOS。实测先区分编码器特征载荷、序列化元数据和两个方向的网络传输，再从记录判断收益来自传输、排队还是 prefill。客户端只测到 SSE 内容块时，将块间隔与逐 token 间隔分开报告。

代码包含原生 A/B 入口、VisionZip 适配与 CPU 检查、正式回放计划、评分及作图入口。实测结果以对应运行目录的逐请求记录、服务日志和模型版本为准。
