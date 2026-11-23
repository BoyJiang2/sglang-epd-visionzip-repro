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

规模回放使用运行目录中的 `scale-design.json`。填入模型 `model_revision`、按启动顺序排列的四个 `tp4` UUID、两个 `encoders` UUID，以及 2–3 个 `cases`。每个 case 填 `name`、`requests` 和 `trace_sha256`，并保存对应的 `<name>.trace.jsonl`。等待 7B 结果冻结后，同一负载点在三个配置中重放相同轨迹。

本次双机流程在核对 7B 全量分析和服务释放后，使用 `python3 235B/register.py --run-dir "$PARENT_RUN_DIR"` 登记单图 0.5 QPS、八图 1 QPS 的原始 128 请求轨迹。命令核对分析哈希和轨迹与 A 组的配对身份，先原子写入 `235b/scale-design.json`，再发布 `7b-frozen.json`。等待中的 qjy001 控制器通过模型、GPU 和端口预检后开始规模回放。

```bash
# GROUP 依次为 235B-4、235B-5、235B-6。CASE 为登记的负载名。
python3 common/epd_replay.py replay --group "$GROUP" --model "$MODEL_ID" \
  --endpoints "$MODEL_ENDPOINT" --trace "$RUN_DIR/$CASE.trace.jsonl" \
  --output "$RUN_DIR/$CASE.$GROUP.jsonl" --timeout 180

python3 235B/summarize.py --run-dir "$RUN_DIR" --output "$RUN_DIR/scale-summary.json"
```

汇总检查轨迹 SHA-256、请求 ID、图像哈希、到达时刻和生成预算。平均 TTFT 与 TPOT 使用有效样本，完成率和总吞吐保留已提交请求与失败记录。每卡吞吐按 4、5、6 张分配卡分别归一。保留启动回执和 GPU 日志核实实际卡数、NCCL 初始化与编码器分配。
