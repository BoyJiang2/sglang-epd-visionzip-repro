"""Render the three A/B visuals from the audited 7B aggregate."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter


STYLES = {
    "A": ("#216597", "o", "A · 完整双副本"),
    "B": ("#AB4A27", "s", "B · 原生 E/PD"),
}
INK = "#18293A"
MUTED = "#4C6173"
GRID = "#DCE4EB"


def configure_style(font_file):
    if font_file:
        font_manager.fontManager.addfont(font_file)
        font = font_manager.FontProperties(fname=font_file).get_name()
    else:
        available = {entry.name for entry in font_manager.fontManager.ttflist}
        candidates = ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei", "Noto Sans SC")
        font = next((name for name in candidates if name in available), None)
    if not font:
        raise RuntimeError("Install Noto Sans CJK SC or another supported Simplified Chinese font.")
    plt.rcParams.update({
        "font.family": [font, "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.unicode_minus": False,
        "savefig.dpi": 180,
    })


def header(fig, title, subtitle):
    fig.text(0.075, 0.947, title, fontsize=25, weight="bold", va="top")
    subtitle_y = 0.947 - 0.56 / fig.get_figheight()
    rule_y = subtitle_y - 0.43 / fig.get_figheight()
    fig.text(0.075, subtitle_y, subtitle, fontsize=11.5, color=MUTED, va="top")
    fig.add_artist(Line2D([0.075, 0.965], [rule_y, rule_y], transform=fig.transFigure, color=GRID, lw=1))
    handles = [
        Line2D([], [], color=color, marker=marker, lw=2, ms=7, label=label)
        for color, marker, label in STYLES.values()
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.967, subtitle_y + 0.008),
               frameon=False, ncol=2, handlelength=1.5, columnspacing=1.3, fontsize=10)


def panel(ax, title, subtitle=None, *, title_pad=24):
    ax.set_title(title, loc="left", pad=title_pad, fontsize=15, weight="bold", color=INK)
    if subtitle:
        ax.text(0, 1.013, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")


def axis_style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.tick_params(length=0, labelsize=10, pad=5)


def log_axis(ax, direction, limits, ticks):
    axis = getattr(ax, f"{direction}axis")
    getattr(ax, f"set_{direction}scale")("log")
    getattr(ax, f"set_{direction}lim")(*limits)
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_locator(NullLocator())


def topology(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel(ax, "a  部署拓扑")

    def box(x, y, width, height, lines, group):
        color = STYLES[group][0]
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#F3F8FC" if group == "A" else "#FCF4EF", edgecolor=color,
            linewidth=1.0,
        ))
        for i, (label, fs) in enumerate(lines):
            ypos = y + height * (0.5 if len(lines) == 1 else (0.68 if i == 0 else 0.28))
            ax.text(x + width / 2, ypos, label, ha="center", va="center", fontsize=fs,
                    weight="semibold" if i == 0 else "normal", color=INK)

    def arrow(start, end, color=MUTED):
        ax.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": "-|>", "color": color, "lw": 1.15, "shrinkA": 1, "shrinkB": 1,
        })

    ax.text(0, 0.952, "A  轮询完整副本", color=STYLES["A"][0], fontsize=11.5, weight="bold")
    box(0.01, 0.673, 0.19, 0.14, [("客户端轮询", 9.5)], "A")
    layers = ("视觉编码 · Prefill · Decode", 8.8)
    box(0.48, 0.76, 0.46, 0.152, [("qjy000  ·  1 张 H20", 9.6), layers], "A")
    box(0.48, 0.554, 0.46, 0.152, [("qjy001  ·  1 张 H20", 9.6), layers], "A")
    ax.plot([0.21, 0.32, 0.32], [0.743, 0.743, 0.836], color=STYLES["A"][0], lw=1.15)
    ax.plot([0.32, 0.32], [0.743, 0.63], color=STYLES["A"][0], lw=1.15)
    for y in (0.836, 0.63):
        arrow((0.32, y), (0.473, y), STYLES["A"][0])

    ax.plot([0, 0.96], [0.492, 0.492], color=GRID, lw=0.8)
    ax.text(0, 0.422, "B  拆分视觉编码", color=STYLES["B"][0], fontsize=11.5, weight="bold")
    box(0.01, 0.175, 0.29, 0.16, [("qjy000  ·  1 张 H20", 9.6), ("视觉编码", 9)], "B")
    box(0.66, 0.175, 0.28, 0.16, [("qjy001  ·  1 张 H20", 9.6), ("Prefill · Decode", 9)], "B")
    ax.text(0.477, 0.331, "完整视觉特征", ha="center", fontsize=9, color=STYLES["B"][0])
    arrow((0.308, 0.276), (0.652, 0.276), STYLES["B"][0])
    arrow((0.652, 0.233), (0.308, 0.233))
    ax.text(0.477, 0.176, "图像请求", ha="center", fontsize=8.6, color=MUTED)
    ax.annotate("客户端 → qjy001 请求入口", xy=(0.80, 0.168), xytext=(0.80, 0.034),
                ha="center", va="bottom", fontsize=9, color=MUTED, arrowprops={
                    "arrowstyle": "-|>", "color": MUTED, "lw": 1.1,
                })


def plot_main(data, destination):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.4))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.145, top=0.798, hspace=0.56, wspace=0.35)
    header(fig, "同样两张 H20，拆分视觉编码后，哪些请求受益？",
           "同一 Qwen2.5-VL-7B-Instruct · 每台一张 H20 · SGLang 双机实测")
    topology(axes[0, 0])

    latency = axes[0, 1]
    panel(latency, "b  首 token 延迟", "0.5 QPS · 合并三次重复 · 仅统计完成请求")
    main_rows = sorted((r for r in data["pooled"] if r["qps"] == data["common_qps"]),
                       key=lambda row: row["images"])
    for group, (color, marker, _) in STYLES.items():
        y = [r["ttft_ms"]["p95"] / 1000 for r in main_rows if r["group"] == group]
        latency.plot(range(3), y, color=color, marker=marker, lw=2, ms=7)
        for x, value in enumerate(y):
            above = group == "B"
            first_a = group == "A" and x == 0
            dx, dy = (-9, 0) if first_a else (0, 9 if above else -9)
            latency.annotate(f"{value:.2f}", (x, value), xytext=(dx, dy),
                             textcoords="offset points", ha="right" if first_a else "center",
                             va="center" if first_a else ("bottom" if above else "top"),
                             color=color, fontsize=10.5, weight="semibold")
    latency.set(xlim=(-0.38, 2.25), ylabel="P95 TTFT（秒，对数刻度）")
    latency.set_xticks(range(3), ["1 图", "4 图", "8 图"])
    log_axis(latency, "y", (0.44, 215), [0.5, 1, 3, 10, 30, 100])
    axis_style(latency)

    complete = axes[1, 0]
    panel(complete, "c  八图请求完成率", "每档每组共 384 条请求 · 标注完成数 / 提交数")
    eight_rows = sorted((r for r in data["pooled"] if r["images"] == 8), key=lambda row: row["qps"])
    rates = sorted({row["qps"] for row in eight_rows})
    for index, (group, (color, _, _)) in enumerate(STYLES.items()):
        rows = [r for r in eight_rows if r["group"] == group]
        positions = [i + (index - 0.5) * 0.36 for i in range(len(rows))]
        values = [row["completed"] / row["submitted"] for row in rows]
        bars = complete.bar(positions, values, width=0.32, color=color, alpha=0.94)
        labels = [f"{row['completed']}/{row['submitted']}" for row in rows]
        complete.bar_label(bars, labels=labels, padding=4, fontsize=9.5, color=color, weight="semibold")
    complete.set(ylim=(0, 1.13), ylabel="完成率", xlabel="请求到达率（QPS）")
    complete.set_xticks(range(3), [f"{qps:g}" for qps in rates])
    complete.set_yticks([0, 0.25, 0.5, 0.75, 1])
    complete.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis_style(complete)

    gap = axes[1, 1]
    panel(gap, "d  重图对长生成的干扰", "误差线为 95% 区间 · 按整条长请求重采样 · 每条件 10 次")
    conditions = [
        ("A", "control", "A 无重图"),
        ("A", "same", "A 重图到同一副本"),
        ("A", "separate", "A 重图到另一副本"),
        ("B", "control", "B 无重图"),
        ("B", "same", "B 重图经独立编码"),
    ]
    for y, (group, case, _) in enumerate(conditions):
        color, marker, _ = STYLES[group]
        row = data["interference"][group][case]
        value = row["after_gap_ms"]["p99"]
        low, high = row["after_gap_ci95_ms"]["p99"]
        gap.hlines(y, low, high, color=color, lw=1.6, zorder=2)
        gap.vlines([low, high], y - 0.10, y + 0.10, color=color, lw=1.25, zorder=2)
        gap.scatter(value, y, marker=marker, s=57, color=color, linewidth=1.45,
                    facecolors="white" if case == "control" else color, zorder=3)
        right = value < 100
        gap.annotate(f"{value:.2f}  [{low:.2f}, {high:.2f}]", (value, y), xytext=(9 if right else -12, 0),
                     textcoords="offset points", ha="left" if right else "right", va="center",
                     fontsize=9.4, color=color, weight="normal" if case == "control" else "semibold")
    gap.set(ylim=(4.55, -0.55), xlabel="相邻 SSE 内容块间隔的 P99（ms，对数刻度）")
    gap.set_yticks(range(5), [label for _, _, label in conditions])
    log_axis(gap, "x", (8, 490), [10, 20, 50, 100, 200, 400])
    axis_style(gap, grid="x")

    fig.add_artist(Line2D([0.075, 0.965], [0.074, 0.074], transform=fig.transFigure, color=GRID, lw=1))
    fig.text(0.075, 0.04, "独立编码缓解了同副本卡顿。八图高负载下，B 的首字较慢，完成请求较少。",
             fontsize=11.5, weight="semibold", va="center")
    fig.savefig(destination)
    plt.close(fig)


def plot_matrix(data, destination):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.8), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.092, right=0.963, bottom=0.117, top=0.797, hspace=0.15, wspace=0.17)
    header(fig, "附图 1 · 完整负载矩阵", "固定图数 · 输出 64 token · 每组 27 个单元 · 每单元 128 条请求")
    metrics = [
        ("P95 TTFT（秒，对数刻度）", lambda row: row["ttft_ms"]["p95"] / 1000),
        ("完成率", lambda row: row["completed"] / row["submitted"]),
        ("成功吞吐（请求/s）", lambda row: row["completed_requests_per_s"]),
    ]
    formal = [row for row in data["runs"] if row["kind"] == "formal"]
    for col, images in enumerate((1, 4, 8)):
        pools = [row for row in data["pooled"] if row["images"] == images]
        relevant = [row for row in formal if row["images"] == images]
        rates = sorted({r["qps"] for r in pools})
        spacing = min(b - a for a, b in zip(rates, rates[1:]))
        axes[0, col].set_title(f"每请求 {images} 图", fontsize=14, weight="bold", pad=11)
        for row, (label, metric) in enumerate(metrics):
            ax = axes[row, col]
            for index, (group, (color, marker, _)) in enumerate(STYLES.items()):
                offset = (index - 0.5) * 0.52 * spacing
                for rate in rates:
                    seeds = sorted((r for r in relevant if r["group"] == group and r["qps"] == rate),
                                   key=lambda cell: cell["seed"])
                    points = [rate + offset + delta * spacing for delta in (-0.20, -0.11, 0.11)]
                    ax.scatter(points, [metric(cell) for cell in seeds], marker=marker, s=19,
                               facecolors="none", edgecolors=color, alpha=0.6, lw=0.85, zorder=2)
                pooled = sorted((r for r in pools if r["group"] == group), key=lambda r: r["qps"])
                ax.scatter([r["qps"] + offset for r in pooled], [metric(r) for r in pooled],
                           marker=marker, s=51, color=color, lw=0.75, edgecolor="white", zorder=3)
            ax.set_xlim(rates[0] - 0.58 * spacing, rates[-1] + 0.58 * spacing)
            ax.set_xticks(rates, [f"{r:g}" for r in rates])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                log_axis(ax, "y", (0.42, 260), [0.5, 1, 3, 10, 30, 100])
            elif row == 1:
                ax.set(ylim=(0, 1.075), yticks=[0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            else:
                ax.set(ylim=(0, 4.55), yticks=[0, 1, 2, 3, 4], xlabel="请求到达率（QPS）")
            axis_style(ax)
    fig.text(0.075, 0.064, "小空心点＝单次重复　大实心点＝三次汇总。横向微移用于区分重叠样本。",
             fontsize=10.5, color=MUTED)
    fig.text(0.075, 0.036, "P95 汇集成功请求。完成率按总完成数 / 总提交数计算。成功吞吐按总完成数 / 总观测秒数计算。",
             fontsize=10.5, color=MUTED)
    fig.savefig(destination)
    plt.close(fig)


def plot_official(data, destination):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.9))
    fig.subplots_adjust(left=0.09, right=0.945, bottom=0.19, top=0.7, wspace=0.3)
    header(fig, "附图 2 · 官方随机多图负载复现", "随机 1–8 图 · 1080p · 每档每组 64 条 · 输出 256 token")
    official = [r for r in data["runs"] if r["kind"] == "official"]
    for ax, (title, ylabel, metric, decimals) in zip(axes, [
        ("首 token 延迟", "P95 TTFT（秒）", lambda r: r["ttft_ms"]["p95"] / 1000, 2),
        ("成功吞吐", "成功吞吐（请求/s）", lambda r: r["completed_requests_per_s"], 3),
    ], strict=True):
        panel(ax, title, title_pad=11)
        for group, (color, marker, _) in STYLES.items():
            rows = sorted((r for r in official if r["group"] == group), key=lambda row: row["qps"])
            x = [r["qps"] for r in rows]
            y = [metric(r) for r in rows]
            ax.plot(x, y, color=color, marker=marker, lw=2, ms=7 if group == "A" else 8,
                    mfc=color if group == "A" else "none", mew=1.35)
            ax.annotate(f"{y[-1]:.{decimals}f}", (x[-1], y[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=color, fontsize=10.5, weight="semibold")
        ax.set(xlim=(0.16, 1.14), ylim=(0, None), xlabel="请求到达率（QPS）", ylabel=ylabel)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
        axis_style(ax)
    fig.text(0.075, 0.061, "A、B 各完成 320 / 320 条。首字延迟只统计完成请求，成功吞吐按实际观测秒数计算。",
             fontsize=10.5, color=MUTED)
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, help="Path to a local font that supports Simplified Chinese.")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    configure_style(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for plot, name in (
        (plot_main, "01-main.png"),
        (plot_matrix, "02-formal-matrix.png"),
        (plot_official, "03-official-multi-image.png"),
    ):
        plot(data, args.output_dir / name)


if __name__ == "__main__":
    main()
