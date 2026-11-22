# Optional SGLang diagnostics

The patches target SGLang `8396d60a14ef03b23d2cdaa5c64adb18eba7c703`. The default baseline uses the upstream checkout.

- `epd-http-payload-trace.patch` logs the serialized PD→E HTTP body and optional Linux `TCP_INFO` counters when `SGLANG_EPD_HTTP_TRACE=1`. It applies to both the pinned baseline and the VisionZip checkout. The TCP measurement covers transport payload bytes. Link, IP, and TCP headers are excluded. A closed transport or unsupported counter is reported as unavailable. Use an independent diagnostic replay for this instrumentation.
- `epd-native-feature-payload-trace.patch` logs the native encoder's feature and metadata frame sizes when `SGLANG_EPD_NATIVE_PAYLOAD_TRACE=1`. The encoder keeps its native model path and transfer backend. The patch was verified with a local ZMQ sender and receiver.
- `epd-unadmitted-input-ids.patch` lets an E/PD request canceled before its token layout arrives retire without crashing. Patch application and the CPU regression passed on the pinned source. After applying the patch, run `python3 common/diagnostics/epd_unadmitted_cpu_checks.py "$SGLANG_DIR"`.

## Persistent encoder connection

Call `snapshot_embedding_tcp.py` on the encoder host. Use the owner PID reported by `ss` and the PD scheduler's ZMQ receive port. The scheduler log prints that port as `embedding_port`. Warm up the connection before the first snapshot.

```bash
python3 common/diagnostics/snapshot_embedding_tcp.py snapshot \
  --pid "$ENCODER_PID" --peer "$PD_IP" --port "$PD_ZMQ_PORT" \
  --output "$RUN_DIR/embedding-tcp.before.json"
# Run one isolated untimed probe and wait for the response.
python3 common/diagnostics/snapshot_embedding_tcp.py snapshot \
  --pid "$ENCODER_PID" --peer "$PD_IP" --port "$PD_ZMQ_PORT" \
  --output "$RUN_DIR/embedding-tcp.after.json"
python3 common/diagnostics/snapshot_embedding_tcp.py delta \
  "$RUN_DIR/embedding-tcp.before.json" "$RUN_DIR/embedding-tcp.after.json"
```

The tool reads the sender's Linux TCP counters without restarting the service. It rejects comparisons across different socket identities or unordered samples. Inspect the send queues and acknowledged bytes to check that the probe drained. Report TCP payload separately from the application frame counts and PD→E HTTP bytes. [Linux TCP counter definitions](https://github.com/torvalds/linux/blob/v5.15/include/uapi/linux/tcp.h)

## Paired probe payload summary

During the six untimed probes for each C ratio, enable both the encoder payload trace and the PD HTTP trace. Collect the probe response JSONL and the two service logs for each ratio. Use `report_probe_transport.py --c100 RESPONSES E_LOG PD_LOG --c70 RESPONSES E_LOG PD_LOG --c50 RESPONSES E_LOG PD_LOG --output REPORT.json`.

The report matches logical requests and ordered image fingerprints across ratios. It sums measured feature frames, embedding metadata and PD→E HTTP bodies at the application boundary. PD→E TCP payload is reported separately, with the number of available socket samples. Savings are relative to C100, which has its own index metadata.

## MuirBench request-level table

For the MuirBench quality replay, retain the response and score files alongside the role logs. Use the native encoder and PD instrumentation for B and the VisionZip encoder and PD instrumentation for C70/C50. The formal synthetic timing runs keep diagnostic logging off.

```bash
python3 common/diagnostics/report_muir_traffic.py --client-host "$PD_CLIENT_IP" \
  --a "$A_RESPONSES" "$A_SCORES" \
  --b "$B_RESPONSES" "$B_SCORES" "$B_ENCODER_LOG" "$B_PD_LOG" \
  --c70 "$C70_RESPONSES" "$C70_SCORES" "$C70_ENCODER_LOG" "$C70_PD_LOG" \
  --c50 "$C50_RESPONSES" "$C50_SCORES" "$C50_ENCODER_LOG" "$C50_PD_LOG" \
  --output "$RUN_DIR/muir-traffic.json"
```

For remote A requests, the client counts the submitted JSON body and the SSE response through the `[DONE]` line. A local A request has zero inter-node traffic. B/C use the HTTP body and returned feature frames. Match rows by the benchmark question, ordered images and request identity. Unknown application counts stay null, and TCP counts remain separate from application bytes.

## OCRBench and RealWorldQA traffic probes

Run `quality_probe_trace.py --run-dir "$RUN_DIR" --selection "$RUN_DIR/quality-transport-probe-selection.json"` with the frozen full traces and the registered selection of 32 questions per dataset. Replay each probe trace at its recorded arrival times for B, C70 and C50 with the encoder and PD traces enabled.

```bash
python3 common/diagnostics/report_quality_probes.py --run-dir "$RUN_DIR" \
  --client-host "$PD_CLIENT_IP" \
  --b-logs "$B_ENCODER_LOG" "$B_PD_LOG" \
  --c70-logs "$C70_ENCODER_LOG" "$C70_PD_LOG" \
  --c50-logs "$C50_ENCODER_LOG" "$C50_PD_LOG" \
  --output "$RUN_DIR/quality-traffic.json" --figure "$RUN_DIR/quality-traffic.png"
```

The plot uses mean application MiB per request over the probes measured in all three groups. It takes accuracy from the full 1,000-question OCRBench and 765-question RealWorldQA score files and shows the paired traffic sample size on each panel.
