# A · 双机完整模型副本

在两台 H20 工作机各运行一次 `start.sh`。两台都使用相同的 Qwen2.5-VL-7B-Instruct 权重 revision、相同 `MODEL_ID` 和 SGLang 环境。各进程占一张卡，同时完成视觉编码、prefill 和 decode。启动脚本固定 BF16、TP1、静态显存比例 0.65、上下文 128,000、chunked prefill 8,192、FA3 注意力与 FlashInfer 采样。RadixCache 关闭。

```bash
export SGLANG_ROOT=/path/to/sglang
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export GPU_UUID="$ALLOCATED_GPU_UUID"
HOST="$WORKER_BIND_ADDRESS" PORT=30002 bash A/start.sh
```

`SGLANG_ROOT` 是包含 `python/` 的 SGLang 裸源码目录。执行前在同一 Python 环境中安装固定提交的依赖，必要时设置 `PYTHON_BIN`。容器内执行时先向容器提供该 UUID 和模型路径。

等两台的 `/health` 就绪，再向每个副本分别发一条带图像的请求。正式回放使用 [公共回放入口](../common/README.md)，A 组传两个 endpoint，记录每个副本实际收到的请求和图片数。保留客户端逐请求记录、两个服务日志与对照条件。

预热完成后，在客户端设置 `RUN_DIR`、`MODEL_PATH`、`A_ENDPOINT_1` 和 `A_ENDPOINT_2`，再运行 `bash A/run_matrix.sh`。脚本按 [冻结计划](../plans/formal-design.json) 生成轨迹和 A 的原始记录，B/C 在相同目录回放。

`bash A/start.sh --help` 显示参数。`--dry-run` 显示启动命令。
