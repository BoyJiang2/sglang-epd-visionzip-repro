# B · 双机原生 E/PD

编码主机仅运行视觉编码。语言主机在同一进程中完成 prefill 和 decode。两边以 SGLang 原生 `zmq_to_scheduler` 传输完整视觉特征。语言服务沿用 A 的模型、FA3、TP1、显存比例和 prefill 参数。

两台主机都设置各自的 `SGLANG_ROOT`、`MODEL_PATH` 和 `GPU_UUID`，使用同一份 7B 权重 revision 与同一 `MODEL_ID`。在编码主机执行第一行，确认服务健康后在语言主机执行第二行。

```bash
HOST="$ENCODER_BIND_ADDRESS" PORT=30000 bash B/start.sh encoder
HOST="$PD_BIND_ADDRESS" PORT=30002 ENCODER_URL="$REACHABLE_ENCODER_URL" bash B/start.sh pd
```

`REACHABLE_ENCODER_URL` 包含可从 PD 主机访问的编码地址和端口。若绑定地址与通告地址不同，两台都显式设置各自的 `SGLANG_HOST_IP`，使对端能够连接。每端只选定一张已分配 H20。用单图和四图各验证一次，按请求日志核对 E、PD 都处理了同一请求。

复用 [A 的固定回放轨迹](../A/README.md) 与 [公共回放入口](../common/README.md)。B 只向 PD 发送请求。独立回放轮次使用互异的请求 ID，保存原始客户端记录、E/PD 日志与实际特征传输。可用 `bash B/start.sh --help` 或在角色后加 `--dry-run` 查看参数。
