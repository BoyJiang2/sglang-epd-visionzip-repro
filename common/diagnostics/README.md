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

The joined report includes each task category's submitted and completed question counts, accuracy, and the fraction of reciprocal pairs answered correctly on both questions. The reports preserve attempted and completed denominators. When a request fails, the category can come from the paired question or an aligned group. Pairs with no observed category appear under `unlabeled`.

The HTTP diagnostic also records `rpc_headers_ms` per encoder part. For a successful RPC, this is the PD monotonic interval from entering the HTTP call to receiving response headers. It includes request encoding and upload, E queueing and work, and the response path through headers. Use the recorded HTTP status to separate successful RPCs from elapsed times to failure.

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

The plot uses mean application MiB per request over the probes measured in all three groups. Its vertical axis shows correct answers divided by submissions for the 1,000-question OCRBench and 765-question RealWorldQA sets. The legend identifies any failed or skipped requests. The JSON keeps completed, correct and submitted counts, and reports official accuracy for groups with complete coverage. Each panel shows the paired traffic sample size.

## Paired stage timing

In a separate B/C70/C50 run, start E with `--enable-metrics` and PD with `--enable-metrics --export-metrics-to-file --export-metrics-to-file-dir "$PD_METRICS_DIR"`. Warm up each service, save E's `/metrics` immediately before and after the replay, and retain the client JSONL, both role logs and PD export files. Use the same ordered requests and planned offsets across the three groups.

For a managed 7B diagnostic, start `python3 common/diagnostics/sample_7b_roles.py --run-dir "$RUN_DIR" --node "$NODE" --gpu-uuid "$GPU_UUID" --deadline-epoch-s "$CUTOFF_EPOCH"` on each node. The sampler reads the assigned GPU about once a second while its recorded service is live. The timing and TPOT reports restrict samples to the window from first request send to last request outcome. They retain sample counts, query failures, mean GPU utilization, and peak device memory for each role or replica.

For phase logs, use the HTTP patch on PD with `SGLANG_EPD_HTTP_TRACE=1`. On B's encoder use the native feature patch with `SGLANG_EPD_NATIVE_PAYLOAD_TRACE=1`. On C's encoder apply `epd-visionzip-phase-trace.patch` to the pinned C sources and set `SGLANG_EPD_VISIONZIP_PHASE_TRACE=1`.

Pass a manifest to `python3 common/diagnostics/report_stage_timing.py MANIFEST.json --output REPORT.json`. Give each entry in `configurations` a `name`, an `expected_requests` count, and a `groups` object with `B`, `C70`, `C50`. Each group entry points to `responses`, `e_metrics_before`, `e_metrics_after`, and the `pd_export` array of native log files. Paths are relative to the manifest directory.

PD prefill time includes any scheduling gaps between chunks. Encoder preprocess and forward observations are per batch. The C forward measurement includes VisionZip and the copy to host. The ZMQ send measurement ends when E releases its local send buffer. The report keeps these observation units and PD matching coverage alongside client TTFT.

Pass `--pd-log GROUP PATH` and `--encoder-log GROUP PATH` for each diagnostic group to add the phase logs. The report keeps successful HTTP time to headers and failed elapsed time in separate distributions. ViT, VisionZip scoring and merging use one CUDA-event observation per eligible C batch. The enclosing wall span includes the copy to host. Metadata serialization and buffer-view creation are timed per request part in both B and C. The report includes measurement coverage and excludes C batches containing work outside the paired comparison.
