"""Build the fixed, random and interference figures for the legacy two-H20 run."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re


RUN_ID = "20261122T173622Z-7b-ab"
MODEL = "Qwen2.5-VL-7B-Instruct"
GROUPS = ("A", "B", "C70", "C50")
SEEDS = (23011, 23012, 23013)
RATES = {1: (0.5, 2, 4), 4: (0.5, 1, 2), 8: (0.25, 0.5, 1)}
OFFICIAL_RATES = (0.2, 0.4, 0.6, 0.8, 1.0)
STEMS = {"A": "A", "B": "B.no-probe", "C70": "C70", "C50": "C50"}
CONDITIONS = (
    ("A", "control", "A  无重图"), ("A", "same", "A  同副本重图"),
    ("A", "separate", "A  跨副本重图"), ("B", "control", "B  无重图"),
    ("B", "same", "B  独立编码重图"), ("C70", "control", "C70  无重图"),
    ("C70", "same", "C70  独立编码重图"), ("C50", "control", "C50  无重图"),
    ("C50", "same", "C50  独立编码重图"),
)


def collect(run_dir):
    from compare_interference import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, analyze_mode
    from epd_replay import percentiles, summarize_run
    from report_performance import fingerprint, load_cell

    if run_dir.name != RUN_ID:
        raise ValueError(f"Expected the legacy run {RUN_ID}")
    sources = {}

    def source(name):
        path = run_dir / name
        sources[name] = sha256(path.read_bytes()).hexdigest()
        return path

    plan = json.loads(source("formal-design.json").read_bytes())
    paper = plan["paper"]
    if (
        tuple(plan["groups"]) != GROUPS
        or tuple(plan["seeds"]) != SEEDS
        or {int(k): tuple(sorted(v)) for k, v in plan["rates_qps"].items()} != RATES
        or (plan["requests_per_cell"], plan["output_tokens"]) != (128, 64)
        or tuple(sorted(paper["target_rates"])) != OFFICIAL_RATES
        or (paper["random_images_upper_bound"], paper["seed"], paper["requests"], paper["output_tokens"])
        != (8, 3467, 64, 256)
    ):
        raise ValueError("Frozen formal or official design differs from this run")

    nodes = ("qjy000", "qjy001")
    topology = {}
    for group in GROUPS:
        topology[group] = {}
        label = "no-probe" if group == "B" else "formal"
        for node in nodes:
            name = f"{group}-{node}-{label}.launch.json"
            launch = json.loads(source(name).read_bytes())
            if not launch["gpu_uuid"].startswith("GPU-") or "," in launch["gpu_uuid"]:
                raise ValueError(f"{name}: expected one H20 worker")
            role = "colocated" if group == "A" else "encode" if node == "qjy000" else "prefill_decode"
            flags = {"encode": "--encoder-only", "prefill_decode": "--language-only"}
            actual_roles = set(flags.values()).intersection(launch["command"])
            if actual_roles != ({flags[role]} if role in flags else set()):
                raise ValueError(f"{name}: deployment receipt disagrees with role {role}")
            topology[group][node] = dict(role=role, gpu_uuid=launch["gpu_uuid"],
                                         source_commit=launch["source_commit"], receipt=name)
    devices_by_node = [{topology[group][node]["gpu_uuid"] for group in GROUPS} for node in nodes]
    if any(len(devices) != 1 for devices in devices_by_node) or devices_by_node[0] == devices_by_node[1]:
        raise ValueError("Expected one shared pair of H20 devices across all groups")

    cases = [
        ("formal", images, qps, seed, 128, 64)
        for images, rates in RATES.items() for qps in rates for seed in SEEDS
    ] + [("official", 8, qps, 3467, 64, 256) for qps in OFFICIAL_RATES]
    pooled = defaultdict(list)
    runs = []
    for kind, images, qps, seed, expected, tokens in cases:
        prefix = f"{kind}-{images}img-{qps:g}qps-seed{seed}"
        matched = None
        for group in GROUPS:
            path = source(f"{prefix}.{STEMS[group]}.jsonl")
            rows = load_cell(path, group, expected)
            signatures = {rid: fingerprint(row) for rid, row in rows.items()}
            if matched is not None and signatures != matched:
                raise ValueError(f"{path.name}: paired requests differ")
            matched = signatures
            complete_ttft = []
            for row in rows.values():
                if (
                    row["expected_output_tokens"] != tokens
                    or not 1 <= row["image_count"] <= images
                    or row["image_count"] != len(row["image_digests"])
                    or any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in row["image_digests"])
                    or (kind == "formal" and row["image_count"] != images)
                ):
                    raise ValueError(f"{path.name}: frozen input or output budget differs")
                if row["status"] == "complete":
                    if row["completion_tokens"] != tokens or row["finish_reason"] != "length":
                        raise ValueError(f"{path.name}: completed response has wrong output budget")
                    complete_ttft.append(row["content_chunks"][0][0] * 1000)
            stats = summarize_run(path, expected)
            if (stats["submitted"], stats["completed"]) != (expected, len(complete_ttft)):
                raise ValueError(f"{path.name}: plotted denominators disagree with replay")
            record = {key: stats[key] for key in (
                "submitted", "completed", "failed", "failures_by_type", "observation_s",
                "completed_requests_per_s", "ttft_ms", "request_tpot_ms",
            )}
            after = sum(row["status"] == "failed" and bool(row["content_chunks"]) for row in rows.values())
            runs.append(dict(kind=kind, images=images, qps=qps, seed=seed, group=group,
                             file=path.name, failed_after_first_content=after,
                             failed_before_first_content=stats["failed"] - after, **record))
            if kind == "formal":
                pooled[group, images, qps].append((record, complete_ttft))

    summaries = []
    for (group, images, qps), runs_at_rate in pooled.items():
        submitted = sum(row["submitted"] for row, _ in runs_at_rate)
        completed = sum(row["completed"] for row, _ in runs_at_rate)
        observation = sum(row["observation_s"] for row, _ in runs_at_rate)
        summaries.append(dict(
            group=group, images=images, qps=qps, submitted=submitted, completed=completed,
            observation_s=observation, completed_requests_per_s=completed / observation,
            ttft_ms=percentiles(t for _, ttft in runs_at_rate for t in ttft),
        ))

    interference = {}
    paired_itl = {}
    for group, mode, _ in CONDITIONS:
        starts = []
        for repeat in range(1, 11):
            path = source(f"itl-{group}-{mode}-{repeat}.jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            by_role = {row["interference_role"]: row for row in rows}
            for role, row in by_role.items():
                signature = (row["request_id"], *fingerprint(row)[1:])
                if paired_itl.setdefault(role, signature) != signature:
                    raise ValueError(f"{path.name}: interference inputs differ")
                if row["status"] == "complete" and row["completion_tokens"] != row["expected_output_tokens"]:
                    raise ValueError(f"{path.name}: completed stream has wrong output length")
            long_request = by_role["long"]
            onset = long_request["send_offset_s"] + long_request["content_chunks"][0][0]
            heavy = by_role.get("heavy")
            timing = dict(source=path.name, long_request_id=long_request["request_id"], long_first_content_s=onset)
            if heavy:
                if (
                    not isclose(heavy["trigger_offset_s"], onset, abs_tol=1e-8)
                    or not isclose(heavy["planned_offset_s"], onset + 0.05, abs_tol=1e-8)
                    or heavy["send_offset_s"] < onset
                ):
                    raise ValueError(f"{path.name}: invalid heavy-request trigger or actual send")
                timing.update(heavy_request_id=heavy["request_id"], heavy_actual_send_s=heavy["send_offset_s"])
            starts.append(timing)
        summary = analyze_mode(run_dir, group, mode, 10)
        for repeat, timing in zip(summary["per_repeat"], starts, strict=True):
            target = timing.get("heavy_actual_send_s", timing["long_first_content_s"] + 0.05)
            if not isclose(repeat["aligned_t_s"], target, abs_tol=1e-8):
                raise ValueError(f"{timing['source']}: incorrect alignment")
            repeat.update(timing)
        interference.setdefault(group, {})[mode] = summary

    return dict(
        run_id=RUN_ID, model=MODEL, topology=topology, plan=plan, runs=runs, pooled=summaries,
        interference=interference, source_sha256=sources,
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, unit="long_request", confidence=0.95),
        definitions=dict(
            ttft="Client send to first nonempty SSE content block; completed requests only.",
            throughput="Completions divided by time from first send to last terminal outcome.",
            pooled="Pool successful TTFTs; sum submissions, completions and observation seconds.",
            interference="Consecutive nonempty SSE content blocks within each completed long request. "
                         "Trigger is recorded heavy-request send, or first long content plus 50 ms in control. "
                         "Include the interval straddling the trigger and all later intervals.",
        ),
    )


def styles():
    from plot_ab_results import STYLES
    return {**STYLES, "C70": ("#3B7954", "^", "C70 · VisionZip 70%"), "C50": ("#75539B", "D", "C50 · VisionZip 50%")}


def header(fig, title, descriptor, curves=True):
    from plot_ab_results import GRID, MUTED, Line2D

    height = fig.get_figheight()
    top = 1 - 0.32 / height
    fig.text(0.075, top, title, fontsize=24, weight="bold", va="top")
    fig.text(0.075, top - 0.51 / height, descriptor, fontsize=10.7, color=MUTED, va="top")
    fig.text(0.075, top - 0.79 / height, f"{RUN_ID}  ·  {MODEL}  ·  qjy000/qjy001 各 1 张 H20",
             fontsize=9.4, color=MUTED, va="top")
    line_y = top - 1.09 / height
    fig.add_artist(Line2D([0.075, 0.965], [line_y, line_y], transform=fig.transFigure, color=GRID, lw=1))
    if curves:
        handles = [Line2D([], [], color=color, marker=marker, lw=1.9, ms=6, label=label)
                   for color, marker, label in styles().values()]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.067, line_y - 0.10 / height),
                   frameon=False, ncol=4, handlelength=1.35, columnspacing=2, fontsize=10)


def plot_matrix(data, destination):
    from plot_ab_results import INK, MUTED, PercentFormatter, axis_style, log_axis, plt

    fig, axes = plt.subplots(4, 3, figsize=(17.2, 12.5), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.73]})
    fig.subplots_adjust(left=0.094, right=0.965, bottom=0.145, top=0.805, hspace=0.24, wspace=0.23)
    header(fig, "固定多图负载矩阵", "四组各 27 个正式单元 · 1080p · 输出 64 token · 每单元提交 128 条")
    repeats = {(row["group"], row["images"], row["qps"], row["seed"]): row
               for row in data["runs"] if row["kind"] == "formal"}
    pooled = {(row["group"], row["images"], row["qps"]): row for row in data["pooled"]}
    metrics = (
        ("P95 TTFT（秒）", lambda row: row["ttft_ms"]["p95"] / 1000),
        ("完成率", lambda row: row["completed"] / row["submitted"]),
        ("成功吞吐（请求/s）", lambda row: row["completed_requests_per_s"]),
    )
    seed_markers = ("o", "^", "D")
    for col, (images, rates) in enumerate(RATES.items()):
        count_ax = axes[3, col]
        count_ax.axis("off")
        count_ax.set_ylim(-0.15, 4.2)
        for boundary in (0.5, 1.5):
            count_ax.axvline(boundary, color="#DCE4EB", linewidth=0.8, zorder=1)
        for i in (1, 2) if images == 8 else ():
            for ax in axes[:, col]:
                ax.axvspan(i - 0.42, i + 0.42, color="#F8EFDB", zorder=0)
        axes[0, col].set_title(f"每请求 {images} 图", weight="bold", fontsize=15, pad=9)
        for row_index, (label, metric) in enumerate(metrics):
            ax = axes[row_index, col]
            for gidx, (group, (color, marker, _)) in enumerate(styles().items()):
                offset = (gidx - 1.5) * 0.163
                xpos = [i + offset for i in range(3)]
                values = [metric(pooled[group, images, qps]) for qps in rates]
                ax.plot([x + 0.05 for x in xpos], values, color=color, lw=1.7, marker=marker, ms=6.7, mew=0.65,
                        mec="white", zorder=3)
                for i, qps in enumerate(rates):
                    for idx, seed in enumerate(SEEDS):
                        ax.scatter(xpos[i] - 0.051 + idx * 0.034,
                                   metric(repeats[group, images, qps, seed]), s=16,
                                   marker=seed_markers[idx], facecolors="white",
                                   edgecolors=color, linewidths=0.8, zorder=4)
            ax.set(xlim=(-0.5, 2.5), xticks=range(3))
            if col == 0:
                ax.set_ylabel(label)
            if row_index == 0:
                log_axis(ax, "y", (0.35, 260), [0.5, 1, 3, 10, 30, 100])
            elif row_index == 1:
                ax.set(ylim=(0, 1.125), yticks=[0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            else:
                throughput_ticks = {1: (0, 1, 2, 3, 4, 5), 4: (0, 0.4, 0.8, 1.2, 1.6),
                                    8: (0, 0.2, 0.4, 0.6, 0.8)}[images]
                ax.set(ylim=(0, throughput_ticks[-1]), yticks=throughput_ticks)
                ax.set_xticks(range(3), [f"{rate:g}" for rate in rates])
                ax.tick_params(labelbottom=True)
            axis_style(ax)
        for i, qps in enumerate(rates):
            for gidx, (group, (color, _, _)) in enumerate(styles().items()):
                cells = [repeats[group, images, qps, seed] for seed in SEEDS]
                total = pooled[group, images, qps]
                label = "  ".join(f"{cell['completed']}/{cell['submitted']}" for cell in cells)
                label += f"\n合计 {total['completed']}/{total['submitted']}"
                exceptional = any(cell["failed"] for cell in cells)
                count_ax.text(i, 3.5 - gidx, label, ha="center", va="center", fontsize=7.2,
                              color=color, weight="bold" if exceptional else "normal")
                if i == 0:
                    count_ax.text(-0.533, 3.5 - gidx, group, ha="right", va="center",
                                  fontsize=8.2, weight="bold", color=color, clip_on=False)
    fig.text(0.075, 0.105, "各档请求到达率（QPS）等距排列。大实心点合并三个种子，空心 ○ / △ / ◇ 依次对应 23011 / 23012 / 23013。",
             fontsize=10.1, color=INK)
    fig.text(0.075, 0.078, "底表标注各个种子及其合计的完成 / 提交数。浅底标出八图 0.5 和 1 QPS。",
             fontsize=10.1, color=MUTED)
    fig.text(0.075, 0.051, "P95 合并成功请求的 TTFT。完成率保留超时分母。成功吞吐为完成数除以各轮观测时长之和，各分面纵轴独立。",
             fontsize=10.1, color=MUTED)
    fig.savefig(destination)
    plt.close(fig)


def plot_official(data, destination):
    from plot_ab_results import MUTED, axis_style, panel, plt

    fig, axes = plt.subplots(1, 2, figsize=(14.1, 6.7))
    fig.subplots_adjust(left=0.085, right=0.951, top=0.635, bottom=0.19, wspace=0.27)
    header(fig, "官方随机多图负载", "随机 1–8 图 · 1080p · 每档每组提交 64 条 · 输出 256 token")
    official = [row for row in data["runs"] if row["kind"] == "official"]
    for ax, title, ylabel, decimals, measure in (
        (axes[0], "首 token 延迟", "P95 TTFT（秒）", 2, lambda row: row["ttft_ms"]["p95"] / 1000),
        (axes[1], "成功吞吐", "成功吞吐（请求/s）", 3, lambda row: row["completed_requests_per_s"]),
    ):
        panel(ax, title, title_pad=10)
        for group, (color, marker, _) in styles().items():
            rows = sorted((row for row in official if row["group"] == group), key=lambda row: row["qps"])
            values = [measure(row) for row in rows]
            ax.plot([row["qps"] for row in rows], values,
                    color=color, marker=marker, lw=2, ms=6.8, mew=0.7, mec="white")
            ax.annotate(f"{values[-1]:.{decimals}f}", (1.0, values[-1]), xytext=(7, 0),
                        textcoords="offset points", va="center", color=color, fontsize=9.6)
        ax.set(xlim=(0.15, 1.115), ylim=(0, None), xlabel="请求到达率（QPS）", ylabel=ylabel)
        ax.set_xticks(OFFICIAL_RATES, [f"{rate:.1f}" for rate in OFFICIAL_RATES])
        axis_style(ax)
    sizes = [f"{group} {sum(row['completed'] for row in official if row['group'] == group)}/"
             f"{sum(row['submitted'] for row in official if row['group'] == group)}" for group in GROUPS]
    fig.text(0.075, 0.077, "完成 / 提交合计　" + "　".join(sizes) + "。每档为一次配对回放。",
             fontsize=10.6, color=MUTED)
    fig.text(0.075, 0.045, "TTFT 从客户端发送计至首个非空 SSE 内容块，P95 统计成功请求。成功吞吐使用该档各组的实际观测时长。",
             fontsize=10.4, color=MUTED)
    fig.savefig(destination)
    plt.close(fig)


def plot_interference(data, destination):
    from plot_ab_results import GRID, MUTED, Line2D, axis_style, log_axis, panel, plt

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.4), sharey=True)
    fig.subplots_adjust(left=0.336, right=0.964, top=0.665, bottom=0.196, wspace=0.27)
    header(fig, "重图对长流生成的干扰", "长请求固定生成 256 token · 每个条件 10 次 · 扰动请求含八张图", curves=False)
    handles = [
        Line2D([], [], marker="o", mfc="white", mec=MUTED, color=MUTED,
               label="无重图 · SSE 内容块间隔", linestyle="none"),
        Line2D([], [], marker="o", color=MUTED, label="加入重图 · SSE 内容块间隔", linestyle="none"),
        Line2D([], [], color=MUTED, marker="|", lw=1.4, label="按长请求重采样的 95% 区间"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.067, 0.786), frameon=False, ncol=3, fontsize=10)
    labels = []
    for group, mode, label in CONDITIONS:
        summary = data["interference"][group][mode]
        long_done = summary["long_outcomes"].get("complete", 0)
        long_title = f"长流 {long_done}/{summary['submitted_repeats']}"
        if mode != "control":
            long_title += f"  ·  重图 {summary['heavy_outcomes'].get('complete', 0)}/{summary['submitted_repeats']}"
        labels.append(f"{label}　{long_title}")
    for axis, key, title in zip(axes, ("p95", "p99"), ("内容块间隔 P95", "内容块间隔 P99"), strict=True):
        panel(axis, title, title_pad=10)
        for y, (group, mode, _) in enumerate(CONDITIONS):
            summary = data["interference"][group][mode]
            color, marker, _ = styles()[group]
            value = summary["after_gap_ms"][key]
            lower, upper = summary["after_gap_ci95_ms"][key]
            axis.hlines(y, lower, upper, color=color, linewidth=1.7)
            axis.vlines([lower, upper], y - 0.11, y + 0.11, color=color, linewidth=1.4)
            axis.scatter(value, y, marker=marker, s=48, edgecolor=color,
                         facecolor="white" if mode == "control" else color, zorder=3)
            side = -1 if value > 120 else 1
            axis.annotate(f"{value:.2f}  [{lower:.2f}, {upper:.2f}]", (upper if side == 1 else lower, y),
                          xytext=(side * 8, 0), textcoords="offset points",
                          ha="left" if side == 1 else "right", va="center", color=color, fontsize=8.4)
        axis.set_ylim(len(CONDITIONS) - 0.5, -0.5)
        axis.set_xlabel("相邻 SSE 内容块间隔（ms）")
        axis.set_yticks(range(len(CONDITIONS)), labels)
        axis.tick_params(labelleft=axis is axes[0])
        if key == "p99":
            log_axis(axis, "x", (8.5, 475), [10, 15, 30, 70, 150, 350])
        else:
            values = [data["interference"][g][m]["after_gap_ci95_ms"][key][1] for g, m, _ in CONDITIONS]
            axis.set_xlim(8.5, max(values) + 2.7)
        for y in (2.5, 4.5, 6.5):
            axis.axhline(y, color=GRID, linewidth=0.65)
        axis_style(axis, grid="x")
    fig.text(0.075, 0.096, "以重图在客户端的实际发送时刻对齐。无重图以长流首块后 50 ms 对齐。纳入跨过对齐点及之后的间隔。",
             fontsize=10.5, color=MUTED)
    fig.text(0.075, 0.061, f"以整个长请求为单位，在各条件内重采样 {data['bootstrap']['draws']:,} 次。P99 横轴采用对数刻度。",
             fontsize=10.5, color=MUTED)
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--run-dir", type=Path, help="Verify and collect the original logs to stdout.")
    sources.add_argument("--data", type=Path, help="Render the checked performance aggregate.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.run_dir:
        print(json.dumps(collect(args.run_dir), sort_keys=True, indent=2, allow_nan=False))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required with --data")
    from plot_ab_results import configure_style

    data = json.loads(args.data.read_text(encoding="utf-8"))
    if data["run_id"] != RUN_ID:
        raise ValueError(f"Expected {RUN_ID}")
    configure_style(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for plot, name in (
        (plot_matrix, "01-fixed-matrix.png"),
        (plot_official, "02-official-multi-image.png"),
        (plot_interference, "03-heavy-image-interference.png"),
    ):
        plot(data, args.output_dir / name)


if __name__ == "__main__":
    main()
