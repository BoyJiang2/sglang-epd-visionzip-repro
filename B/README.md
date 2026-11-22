# B · 双机原生 E/PD

编码主机仅运行视觉编码。语言主机在同一进程中完成 prefill 和 decode。两边以 SGLang 原生 `zmq_to_scheduler` 传输完整视觉特征。语言服务沿用 A 的模型、FA3、TP1、显存比例和 prefill 参数。

两台主机都设置各自的 `SGLANG_ROOT`、`MODEL_PATH` 和 `GPU_UUID`，使用同一份 7B 权重 revision 与同一 `MODEL_ID`。在编码主机执行第一行，确认服务健康后在语言主机执行第二行。

```bash
HOST="$ENCODER_BIND_ADDRESS" PORT=30000 bash B/start.sh encoder
SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=0 \
HOST="$PD_BIND_ADDRESS" PORT=30002 ENCODER_URL="$REACHABLE_ENCODER_URL" bash B/start.sh pd
```

`REACHABLE_ENCODER_URL` 包含可从 PD 主机访问的编码地址和端口。若绑定地址与通告地址不同，两台都显式设置各自的 `SGLANG_HOST_IP`，使对端能够连接。每端只选定一张已分配 H20。用单图和四图各验证一次，按请求日志核对 E、PD 都处理了同一请求。

专用双机复跑使用固定编码器地址，B/C 都关闭 PD 自动探活。原生默认探活连续失败后曾摘除忙碌的编码器，触发 PD 本地视觉处理，影响四图正式轮次。计时前后检查编码服务健康，计时期间通过分派记录核对编码器仍参与每条请求。复跑时给回放结果加 `--label no-probe`，保留默认探活的原始记录。

复用 [A 的固定回放轨迹](../A/README.md) 与 [公共回放入口](../common/README.md)。B 只向 PD 发送请求。独立回放轮次使用互异的请求 ID，保存原始客户端记录、E/PD 日志与实际特征传输。可用 `bash B/start.sh --help` 或在角色后加 `--dry-run` 查看参数。
