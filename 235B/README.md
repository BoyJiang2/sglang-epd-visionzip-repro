# Qwen3-VL-235B FP8 规模验证

使用一台 H20 工作机与官方 Qwen3-VL-235B-A22B-Instruct-FP8 检查点。三个条件共用同一组四卡 TP4、EP1 和相同的模型与运行环境。同一节点分别运行 0、1 或 2 个独立 TP1 视觉编码进程，总资源占用为 4、5 或 6 卡。5/6 卡组使用原生 `zmq_to_scheduler`，PD 进程内依次完成 prefill 和 decode。[官方评测部署](https://www.lmsys.org/blog/2026-01-12-epd/#experimental-setup)

```bash
export SGLANG_ROOT=/path/to/sglang
export MODEL_PATH=/path/to/Qwen3-VL-235B-A22B-Instruct-FP8

# 4 卡。TP4_UUIDS 是四个已分配且互异的 UUID，以逗号连接。
GPU_UUIDS="$TP4_UUIDS" bash 235B/start.sh colocated

# 5 卡。先在独立终端启动一个编码器，再启动 PD。
GPU_UUIDS="$ENCODER1_UUID" PORT=31231 bash 235B/start.sh encoder
GPU_UUIDS="$TP4_UUIDS" ENCODER_URLS="http://127.0.0.1:31231" bash 235B/start.sh pd

# 6 卡。再增加一张独立编码卡，并让 PD 连接两个编码端点。
GPU_UUIDS="$ENCODER2_UUID" PORT=31232 bash 235B/start.sh encoder
GPU_UUIDS="$TP4_UUIDS" ENCODER_URLS="http://127.0.0.1:31231 http://127.0.0.1:31232" bash 235B/start.sh pd
```

逐组运行，保留同一 TP4 UUID 顺序。两个 E 的 GPU 与 TP4 不重叠。每组等待角色健康后执行单图和多图预检，记录四个 NCCL rank、角色分配、各卡显存、自动选择的 FP8 与 MoE 后端和实际处理的请求。启动脚本以 32,768 token 上下文、0.85 静态显存比例和自动后端为统一预检配置。模型配置自动识别预量化的 128×128 块级 FP8，`--dtype bfloat16` 对应未量化运算。

使用固定请求轨迹，报告有效请求率、TTFT、TPOT、完成/失败数、总完成吞吐及每卡完成吞吐。三组的卡数不同，卡数与每卡指标须和延迟一同呈现。`bash 235B/start.sh --help`、各角色的 `--dry-run` 可检查配置。
