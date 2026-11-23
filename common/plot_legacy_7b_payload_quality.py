"""Audit and render communication and quality for the archived dual-H20 run."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from audit_7b import GROUPS, matched_transfer, score
from compare_quality import compare, summarize, unique
from diagnostics.report_probe_transport import collect, report
from diagnostics.snapshot_embedding_tcp import difference
from plot_ab_results import GRID, INK, MUTED, STYLES, axis_style, configure_style, panel, plt
from matplotlib.lines import Line2D


RUN_ID = "20261122T173622Z-7b-ab"
TASKS = (("ocrbench", "OCRBench"), ("realworldqa", "RealWorldQA"))
PAYLOAD_GROUPS = ("C100", "C70", "C50")
COLORS = {**{group: style[0] for group, style in STYLES.items()},
          "C100": "#64778B", "C70": "#258276", "C50": "#6F62A0"}
MARKERS = {"A": "o", "C70": "D", "C50": "^"}
SEED, REPEATS = 23011, 5000
MIB = 2 ** 20
OCR_TYPES = {
    "Regular Text Recognition": "规则文本识别",
    "Irregular Text Recognition": "不规则文本识别",
    "Artistic Text Recognition": "艺术字体识别",
    "Handwriting Recognition": "手写文本识别",
    "Digit String Recognition": "数字串识别",
    "Non-Semantic Text Recognition": "无语义文本识别",
    "Scene Text-centric VQA": "场景文字问答",
    "Doc-oriented VQA": "文档问答",
    "Key Information Extraction": "关键信息提取",
    "Handwritten Mathematical Expression Recognition": "手写数学表达式识别",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def probe_paths(run, group):
    phase = "identity" if group == "C100" else "payload"
    return [run / f"identity.{group}.jsonl",
            *(run / f"{group}-{node}-{phase}.service.log" for node in ("qjy000", "qjy001"))]


def collect_communication(run):
    layouts = matched_transfer(run)
    application = report({int(group[1:]): collect(int(group[1:]), *probe_paths(run, group))
                          for group in PAYLOAD_GROUPS})
    for group, item in application.items():
        if (len(item["probes"]) != layouts[group]["probes"]
                or item["feature_bytes"] != layouts[group]["feature_bytes"]
                or item["embedding_metadata_bytes"] != layouts[group]["metadata_bytes"]):
            raise ValueError(f"{group}: transport counters disagree with the checked E/PD layout")

    stem = run / "identity.B.native-tcp"
    before, after, receipt, request, done = (
        read_json(stem.with_name(stem.name + f".{kind}.json"))
        for kind in ("before", "after", "receipt", "request", "done")
    )
    tcp = difference(before, after)
    if (receipt.get("valid") is not True or done["status"] != receipt["status"]
            or done["status"] != "identity_complete"
            or Path(request["output"]).name != "identity.B.no-probe.jsonl"
            or any(receipt[name] != value for name, value in tcp.items())
            or not tcp["both_send_queues_empty"]
            or tcp["tcp_payload_bytes_sent"] != tcp["tcp_payload_bytes_acked"]):
        raise ValueError("Native B TCP receipt does not match its bounded socket snapshots")

    pd_rows = {}
    for group, item in application.items():
        probes = item["probes"]
        retrans = [row["pd_to_e_tcp_retrans_bytes"] for row in probes]
        pd_rows[group] = dict(samples=item["pd_to_e_tcp_payload_samples"],
                              tcp_payload_bytes=item["pd_to_e_tcp_payload_bytes"],
                              retrans_bytes=sum(retrans) if None not in retrans else None)
    return dict(application_layer=application, native_e_to_pd_tcp=dict(samples=6, **tcp),
                pd_to_e_tcp=pd_rows)


def ocr_subtypes(reports, images):
    scores = {group: unique(report["items"], "request_id") for group, report in reports.items()}
    if any(scores["A"][rid]["correct"] != item["correct"] for rid, item in scores["B"].items()):
        raise ValueError("The archived A/B scores do not support a shared subtype column")
    types = defaultdict(list)
    for rid, item in sorted(scores["B"].items()):
        category = item["metrics"]["ocrbench_accuracy"]["question_type"]
        for rows in scores.values():
            metric = rows[rid]["metrics"]["ocrbench_accuracy"]
            if metric["question_type"] != category or metric["score"] != rows[rid]["correct"]:
                raise ValueError(f"{rid}: OCR subtype or per-question score differs")
        types[category].append(rid)
    if set(types) != set(OCR_TYPES):
        raise ValueError("Unexpected OCRBench question taxonomy")

    result = []
    for category in OCR_TYPES:
        ids = types[category]
        correct = {group: sum(rows[rid]["correct"] for rid in ids) for group, rows in scores.items()}
        paired = {}
        for group in ("C70", "C50"):
            clusters = defaultdict(lambda: [0, 0])
            for rid in ids:
                values = clusters[images[rid]]
                values[0] += int(scores[group][rid]["correct"]) - int(scores["B"][rid]["correct"])
                values[1] += 1
            paired[group] = summarize(clusters, SEED, REPEATS)
        result.append(dict(type=category, submitted=len(ids), correct=correct, paired_vs_B=paired))
    return result


def collect_quality(run):
    tasks = {}
    types = None
    for task, _ in TASKS:
        reports, summaries = {}, {}
        images, schedule = None, None
        for group in GROUPS:
            stem = f"quality-{task}.{group}"
            audited = score(run, task, group, stem, True, images, schedule)
            reports[group] = read_json(run / f"{stem}.scores.json")
            images = audited.pop("fingerprints")
            schedule = audited.pop("arrivals")
            summaries[group] = audited
        clusters = {}
        if task == "ocrbench":
            clusters = {rid: digests for rid, (digests, _) in images.items()}
            if any(len(digests) != 1 for digests in clusters.values()):
                raise ValueError("OCRBench clustering requires one image SHA-256 for every question")
            types = ocr_subtypes(reports, clusters)
        paired = {group: compare(reports["B"], reports[group], clusters, SEED, REPEATS)
                  for group in GROUPS if group != "B"}
        tasks[task] = dict(groups=summaries, paired_vs_B=paired)
    return dict(tasks=tasks, ocr_subtypes=types)


def collect_data(run):
    if run.name != RUN_ID:
        raise ValueError(f"Use the archived {RUN_ID} evidence directory")
    communication = collect_communication(run)
    quality = collect_quality(run)
    sources = [path for group in PAYLOAD_GROUPS for path in probe_paths(run, group)]
    sources.append(run / "identity.B.no-probe.jsonl")
    sources.extend(run / f"identity.B.native-tcp.{kind}.json"
                   for kind in ("before", "after", "receipt", "request", "done"))
    sources.extend(run / f"quality-{task}.{group}.{suffix}"
                   for task, _ in TASKS for group in GROUPS for suffix in ("jsonl", "scores.json"))
    return dict(run_id=RUN_ID, model="Qwen2.5-VL-7B-Instruct",
                topology="two nodes, one H20 per node, legacy E/PD",
                communication=communication, quality=quality,
                source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(sources)})


def frame(fig, title, subtitle, footer):
    fig.text(.07, .953, title, va="top", fontsize=24, weight="bold")
    fig.text(.07, .881, subtitle, va="top", fontsize=11, color=MUTED)
    for y in (.829, .084):
        fig.add_artist(Line2D([.07, .962], [y, y], transform=fig.transFigure, color=GRID, lw=1))
    fig.text(.07, .053, footer + f"\nQwen2.5-VL-7B-Instruct · {RUN_ID}",
             fontsize=9, color=MUTED, va="center", linespacing=1.8)


def save(fig, destination):
    fig.savefig(destination)
    plt.close(fig)


def plot_communication(data, destination):
    fig = plt.figure(figsize=(15.3, 7.8))
    grid = fig.add_gridspec(2, 2, left=.12, right=.96, bottom=.145, top=.748,
                           width_ratios=(1.37, 1), height_ratios=(1, .48), hspace=.25, wspace=.28)
    bars, table_ax, tcp_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    frame(fig, "VisionZip 压缩的是返回的视觉特征", "旧双卡 E/PD · 1、4、8 图各 2 条 · 每配置 6 条相同探针",
          "应用层按图像 HTTP 请求体、视觉特征和序列化元数据计量。右侧 TCP 各行独立计量所标方向。")
    panel(bars, "双向应用层载荷", "串行 3 条与混合并发 3 条合计")
    groups = data["application_layer"]
    positions = range(3)
    offsets = [0.] * 3
    for key, color, title in (("feature_bytes", "#258276", "E→PD 视觉特征"),
                              ("pd_to_e_http_body_bytes", "#CFD9E4", "PD→E 图像请求体"),
                              ("embedding_metadata_bytes", "#AA7344", "E→PD 元数据")):
        values = [groups[group][key] / MIB for group in PAYLOAD_GROUPS]
        bars.barh(positions, values, left=offsets, color=color, height=.52, label=title)
        offsets = [current + value for current, value in zip(offsets, values)]
    for y, group, total in zip(positions, PAYLOAD_GROUPS, offsets, strict=True):
        saved = groups[group]["two_way_application_bytes_saved_fraction"]
        indicator = f"（−{saved:.2%}）" if saved else ""
        bars.text(total + 9, y, f"{total:.1f}{indicator}", va="center", color=INK, weight="semibold")
    bars.set(xlim=(0, max(offsets) * 1.22), ylim=(2.62, -.62), xlabel="MiB / 6 条探针")
    bars.set_yticks(positions, ["C100  全量", "C70", "C50"])
    bars.legend(loc="lower right", ncol=1, fontsize=9, frameon=False)
    axis_style(bars, grid="x")

    table_ax.set_axis_off()
    metrics = ("feature_bytes", "embedding_metadata_bytes", "pd_to_e_http_body_bytes")
    table = table_ax.table(
        cellText=[[group] + [f"{groups[group][key] / MIB:.3f}" for key in metrics]
                  for group in PAYLOAD_GROUPS],
        colLabels=["配置", "特征（MiB）", "元数据（MiB）", "请求体（MiB）"],
        cellLoc="center", colWidths=[.2, .26, .26, .28], bbox=[0, 0, 1, .9],
    )
    style_table(table)

    panel(tcp_ax, "独立 TCP 计量", "各方向按该 6 条探针的实际发送记录汇总")
    tcp_ax.set_axis_off()
    native = data["native_e_to_pd_tcp"]
    rows = [["B · E→PD", str(native["samples"]), f'{native["tcp_payload_bytes_sent"] / MIB:.3f}', "—"]]
    for group in PAYLOAD_GROUPS:
        values = data["pd_to_e_tcp"][group]
        rows.append([f"{group} · PD→E", str(values["samples"]),
                     f'{values["tcp_payload_bytes"] / MIB:.3f}' if values["tcp_payload_bytes"] is not None else "—",
                     f'{values["retrans_bytes"]:,}' if values["retrans_bytes"] is not None else "—"])
    table = tcp_ax.table(cellText=rows, colLabels=["方向", "样本", "载荷（MiB）", "重传（字节）"],
                         cellLoc="center", colWidths=[.29, .15, .27, .29], bbox=[0, .36, 1, .59])
    style_table(table)
    tcp_ax.text(0, .26, "B 取同一 E→PD 长连接的前后差分。", fontsize=10, color=MUTED)
    tcp_ax.text(0, .194, "B 的发送与确认载荷字节数一致。", fontsize=10, color=MUTED)
    tcp_ax.text(0, .128, "C 取 PD→E 每条 HTTP 发送日志的 TCP 计数。", fontsize=10, color=MUTED)
    tcp_ax.text(0, .062, "— 表示该字段未记录。", fontsize=9, color=MUTED)
    save(fig, destination)


def style_table(table):
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(.65)
        cell.set_facecolor("#EDF3F7" if row == 0 else "white")
        cell.set_text_props(weight="semibold" if row == 0 or col == 0 else "normal", color=INK)


def interval_point(ax, y, summary, group):
    val = summary["delta_pp"]
    lo, hi = summary["ci95_pp"]
    c = COLORS[group]
    ax.hlines(y, lo, hi, color=c, linewidth=1.7, zorder=2)
    ax.vlines((lo, hi), y - .07, y + .07, color=c, linewidth=1.15, zorder=2)
    ax.scatter(val, y, marker=MARKERS[group], s=52, color=c,
               edgecolors="white", linewidth=.6, zorder=3)
    return val, lo, hi


def plot_quality(data, destination):
    fig, axes = plt.subplots(2, 2, figsize=(15.3, 8.4))
    fig.subplots_adjust(left=.105, right=.956, bottom=.17, top=.74, hspace=.83, wspace=.4)
    frame(fig, "压缩后，OCRBench 与 RealWorldQA 的变化不同", "A、B、C70、C50 使用同一批原图、题目、回放计划与评分器",
          "95% 配对区间使用 5,000 次 bootstrap。OCRBench 按图像 SHA-256 聚类，RealWorldQA 按题。")
    paired_groups = ("A", "C70", "C50")
    for (task, label), (absolute, paired) in zip(TASKS, axes, strict=True):
        task_data = data["tasks"][task]
        groups = task_data["groups"]
        count = groups["B"]["submitted"]
        panel(absolute, f"{label} · {count:,} 题", "答对数 / 全部提交数")
        values = [100 * groups[group]["official_score"] for group in GROUPS]
        absolute.barh(range(4), values, color=[COLORS[group] for group in GROUPS], height=.57)
        for y, group in enumerate(GROUPS):
            text = f"{values[y]:.2f}% · {groups[group]['correct']:,}/{count:,}"
            absolute.text(values[y] - 2, y, text, ha="right", va="center", color="white",
                          weight="semibold", fontsize=9.5)
        absolute.set(xlim=(0, 100), ylim=(3.57, -.57), xlabel="准确率（%）")
        absolute.set_yticks(range(4), GROUPS)
        axis_style(absolute, grid="x")

        panel(paired, f"{label} · 相对原生 B", "点为配对准确率差 · 横线为 95% 区间")
        records = [task_data["paired_vs_B"][group]["submitted_accuracy"] for group in paired_groups]
        lower = min(r["ci95_pp"][0] for r in records)
        upper = max(r["ci95_pp"][1] for r in records)
        span = max(upper - lower, 1)
        paired.axvline(0, color=MUTED, lw=1, linestyle=":", zorder=1)
        for y, (group, record) in enumerate(zip(paired_groups, records, strict=True)):
            val, lo, hi = interval_point(paired, y, record, group)
            paired.text(hi + span * .055, y, f"{val:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
                        va="center", color=COLORS[group], fontsize=9.5)
        paired.set(xlim=(lower - span * .08, upper + span * .76), ylim=(2.58, -.55),
                   xlabel="准确率差（百分点）")
        paired.set_yticks(range(3), paired_groups)
        axis_style(paired, grid="x")
    save(fig, destination)


def plot_ocr_types(data, destination):
    fig = plt.figure(figsize=(15.3, 9.2))
    grid = fig.add_gridspec(1, 2, left=.195, right=.961, bottom=.165, top=.75,
                           width_ratios=(2.12, 1), wspace=.09)
    plot, counts = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    frame(fig, "OCRBench 题型准确率对压缩的响应", "按原始 10 类题型配对 · 题量列在左侧 · 颜色对应 C70 与 C50",
          "每类单独按图像 SHA-256 聚类并重采样 5,000 次。A 与 B 的逐题评分相同。")
    records = data["ocr_subtypes"]
    all_values = [edge for row in records for group in ("C70", "C50")
                  for edge in row["paired_vs_B"][group]["ci95_pp"]]
    lo, hi = min(min(all_values), -1), max(max(all_values), 1)
    panel(plot, "相对 B 的配对准确率差", "点为差值 · 横线为 95% 区间")
    panel(counts, "各组答对数")
    plot.axvline(0, color=MUTED, lw=1.1, linestyle=":", zorder=1)
    labels = [f'{OCR_TYPES[row["type"]]}  ({row["submitted"]})' for row in records]
    for y, row in enumerate(records):
        for group, vertical in (("C70", -.16), ("C50", .16)):
            interval_point(plot, y + vertical, row["paired_vs_B"][group], group)
        for group, x in (("B", .2), ("C70", .5), ("C50", .8)):
            counts.text(x, y, str(row["correct"][group]), ha="center", va="center", color=COLORS[group])
        if y % 2 == 0:
            plot.axhspan(y - .48, y + .48, color="#F2F6F9", zorder=0)
            counts.axhspan(y - .48, y + .48, color="#F2F6F9", zorder=0)
    plot.set(xlim=(lo - 1.5, hi + 1.5), ylim=(len(records) - .46, -.54), xlabel="准确率差（百分点）")
    plot.set_yticks(range(len(records)), labels)
    axis_style(plot, grid="x")
    counts.set(xlim=(0, 1), ylim=plot.get_ylim())
    counts.set_axis_off()
    for group, x in (("A / B", .2), ("C70", .5), ("C50", .8)):
        counts.text(x, -.68, group, ha="center", va="bottom", fontsize=10, weight="semibold",
                    color=COLORS.get(group, COLORS["B"]))
    handles = [Line2D([], [], color=COLORS[group], marker=MARKERS[group], lw=1.5, label=group)
               for group in ("C70", "C50")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.968, .898), ncol=2, frameon=False)
    save(fig, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help=f"Archived {RUN_ID} directory with raw evidence")
    source.add_argument("--data", type=Path, help="Redraw from the audited figure JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--font-file", help="Optional Simplified Chinese font file")
    args = parser.parse_args()
    data = collect_data(args.run_dir) if args.run_dir else read_json(args.data)
    if data["run_id"] != RUN_ID:
        raise ValueError("The data must identify the exact archived legacy run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style(args.font_file)
    plot_communication(data["communication"], args.output_dir / "04-communication-payload.png")
    plot_quality(data["quality"], args.output_dir / "05-quality-paired.png")
    plot_ocr_types(data["quality"], args.output_dir / "06-ocrbench-types.png")
    with (args.output_dir / "payload-quality-data.json").open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
