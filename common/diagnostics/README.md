# Optional SGLang diagnostics

Both patches target SGLang `8396d60a14ef03b23d2cdaa5c64adb18eba7c703`. The default baseline uses the upstream checkout.

- `epd-http-payload-trace.patch` logs the serialized PD→E HTTP body and optional Linux `TCP_INFO` counters when `SGLANG_EPD_HTTP_TRACE=1`. The TCP measurement covers transport payload bytes. Link, IP, and TCP headers are excluded. A closed transport or unsupported counter is reported as unavailable. Patch application to the pinned source was checked.
- `epd-unadmitted-input-ids.patch` lets an E/PD request canceled before its token layout arrives retire without crashing. Patch application and the CPU regression passed on the pinned source. After applying the patch, run `python3 common/diagnostics/epd_unadmitted_cpu_checks.py "$SGLANG_DIR"`.
