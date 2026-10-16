#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.plotting.paper_plot_style import apply_paper_preset  # noqa: E402

DEFAULT_INPUT_CSV = (
    ROOT
    / "experiments"
    / "figure-12-planner-overhead"
    / "merged_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "experiments" / "figure-12-planner-overhead"
TIMEOUT_MS = 120_000.0


def repo_relative(path: Path) -> str:
    """Return portable repository-relative paths for generated metadata."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())

DATASET_ORDER = ["SIFT1M", "GLOVE"]
DATASET_DISPLAY = {"SIFT1M": "SIFT1M", "GLOVE": "GloVe"}
SETTING_ORDER = [
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 64),
    (256, 128),
    (256, 256),
    (512, 128),
    (512, 256),
    (512, 512),
]
METHOD_SPECS = [
    {
        "key": "plain",
        "label": "Exact",
        "time_field": "plain_p95_ms",
        "timeout_field": "plain_timed_out",
        "legend_label": "Baseline",
        "hatch": None,
    },
    {
        "key": "dc",
        "label": "+D&C",
        "time_field": "dc_p95_ms",
        "timeout_field": "dc_timed_out",
        "legend_label": "+D&C",
        "hatch": "///",
    },
    {
        "key": "adaptive_dc",
        "label": "+Adaptive",
        "time_field": "adaptive_dc_p95_ms",
        "timeout_field": "adaptive_dc_timed_out",
        "legend_label": "+Ada. Search",
        "hatch": None,
    },
    {
        "key": "wqs",
        "label": "+WQS",
        "time_field": "wqs_p95_ms",
        "timeout_field": "wqs_timed_out",
        "legend_label": "+Lagrangian",
        "hatch": None,
    },
]

BLUE_4 = ["#f1eef6", "#bdc9e1", "#74a9cf", "#0570b0"]
RED_4 = ["#fef0d9", "#fdcc8a", "#fc8d59", "#d7301f"]
BAR_EDGE_COLORS = ["#4a4a4a", "#4a4a4a", "#2f3f53", "#23384f"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render four-way planner overhead table and figure.")
    ap.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--palette-family", choices=("blue", "red"), default="blue")
    ap.add_argument("--figure-nprobe", type=int, default=256)
    ap.add_argument("--figure-k", type=int, default=128)
    return ap.parse_args()


def _safe_float(value: str) -> float:
    v = float(value)
    if math.isnan(v):
        return float("nan")
    return v


def _safe_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = [dict(row) for row in csv.DictReader(handle)]

    rows: list[dict] = []
    for row in raw_rows:
        rows.append(
            {
                "dataset": str(row["dataset"]).upper(),
                "nprobe": int(row["nprobe"]),
                "K": int(row["K"]),
                "plain_timed_out": _safe_bool(row["plain_timed_out"]),
                "plain_p95_ms": _safe_float(row["plain_p95_ms"]),
                "dc_timed_out": _safe_bool(row["dc_timed_out"]),
                "dc_p95_ms": _safe_float(row["dc_p95_ms"]),
                "adaptive_dc_timed_out": _safe_bool(row.get("adaptive_dc_timed_out", "false")),
                "adaptive_dc_p95_ms": _safe_float(row["adaptive_dc_p95_ms"]),
                "wqs_timed_out": _safe_bool(row.get("wqs_timed_out", "false")),
                "wqs_p95_ms": _safe_float(row["wqs_p95_ms"]),
                "adaptive_dc_err_avg_pct": _safe_float(row["adaptive_dc_err_avg_pct"]),
                "adaptive_dc_err_max_pct": _safe_float(row["adaptive_dc_err_max_pct"]),
                "wqs_err_avg_pct": _safe_float(row["wqs_err_avg_pct"]),
                "wqs_err_max_pct": _safe_float(row["wqs_err_max_pct"]),
            }
        )
    rows.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), r["nprobe"], r["K"]))
    return rows


def index_rows(rows: list[dict]) -> dict[tuple[str, int, int], dict]:
    return {(row["dataset"], row["nprobe"], row["K"]): row for row in rows}


def format_ms(value: float, timed_out: bool) -> str:
    if timed_out:
        return r"$>$120s"
    if value >= 1000.0:
        return f"{value:.0f}"
    if value >= 100.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def format_err(avg: float, max_v: float) -> str:
    if math.isnan(avg) or math.isnan(max_v):
        return "--"
    return f"{avg:.2f} / {max_v:.2f}"


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "nprobe",
        "K",
        "plain_timed_out",
        "plain_p95_ms",
        "full_opt_p95_ms",
        "plain_over_full_opt",
        "full_opt_err_avg_pct",
        "full_opt_err_max_pct",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ratio = TIMEOUT_MS / row["wqs_p95_ms"] if row["plain_timed_out"] else row["plain_p95_ms"] / row["wqs_p95_ms"]
            out_row = {
                "dataset": DATASET_DISPLAY[row["dataset"]],
                "nprobe": row["nprobe"],
                "K": row["K"],
                "plain_timed_out": row["plain_timed_out"],
                "plain_p95_ms": row["plain_p95_ms"],
                "full_opt_p95_ms": row["wqs_p95_ms"],
                "plain_over_full_opt": ratio,
                "full_opt_err_avg_pct": row["wqs_err_avg_pct"],
                "full_opt_err_max_pct": row["wqs_err_max_pct"],
            }
            writer.writerow(out_row)


def format_speedup(plain_value: float, plain_timed_out: bool, final_value: float) -> str:
    ratio = TIMEOUT_MS / final_value if plain_timed_out else plain_value / final_value
    prefix = ">" if plain_timed_out else ""
    if ratio >= 100.0:
        return f"{prefix}{ratio:.0f}$\\times$"
    return f"{prefix}{ratio:.1f}$\\times$"


def render_table(rows: list[dict]) -> str:
    by_key = index_rows(rows)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\setlength{\abovecaptionskip}{2pt}")
    lines.append(r"\setlength{\belowcaptionskip}{0pt}")
    lines.append(
        r"\caption{\textbf{Planner Overhead Endpoints.} All entries report per-query p95 time in milliseconds on real probe lists. ``Full optimization'' denotes the final planner with adaptive $B$ search and the large-$K$ WQS engineering fast path (enabled for $K \ge 32$). Approximation errors are measured against the exact D\&C exhaustive-$B$ baseline when that baseline completes. For $nprobe{=}512$, Plain DP exceeds the 120\,s timeout in every setting, so the reported speedups are lower bounds.}"
    )
    lines.append(r"\label{tab:eval-planner-overhead}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.0pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.00}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}llrcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & $nprobe$ & $K$ & \shortstack{Plain DP\\p95 ms} & \shortstack{Full Opt.\\p95 ms} & \shortstack{Speedup\\vs. Plain} & \shortstack{Error (\%)\\avg./max} \\"
    )
    lines.append(r"\midrule")

    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        for probe_idx, nprobe in enumerate((128, 256, 512)):
            probe_rows = [k for (np_, k) in SETTING_ORDER if np_ == nprobe]
            for k_idx, k in enumerate(probe_rows):
                row = by_key[(dataset, nprobe, k)]
                dataset_cell = (
                    rf"\multirow{{9}}{{*}}{{{DATASET_DISPLAY[dataset]}}}"
                    if probe_idx == 0 and k_idx == 0
                    else ""
                )
                nprobe_cell = rf"\multirow{{3}}{{*}}{{{nprobe}}}" if k_idx == 0 else ""
                lines.append(
                    "  "
                    + f"{dataset_cell} & {nprobe_cell} & {k} "
                    + f"& {format_ms(row['plain_p95_ms'], row['plain_timed_out'])} "
                    + f"& {format_ms(row['wqs_p95_ms'], row['wqs_timed_out'])} "
                    + f"& {format_speedup(row['plain_p95_ms'], row['plain_timed_out'], row['wqs_p95_ms'])} "
                    + f"& {format_err(row['wqs_err_avg_pct'], row['wqs_err_max_pct'])} \\\\"
                )
            if nprobe != 512:
                lines.append(r"\addlinespace[1.1pt]")
        if dataset_idx != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _tick_formatter(value: float, _pos: int | None = None) -> str:
    if value >= 1000.0:
        return f"{int(value / 1000)}k"
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.1f}"


def render_figure(
    rows: list[dict],
    out_dir: Path,
    palette_family: str,
    figure_nprobe: int,
    figure_k: int,
) -> dict:
    apply_paper_preset(
        plt,
        "double_col",
        font_size=8.8,
        axes_label_size=9.2,
        axes_title_size=8.8,
        legend_size=7.2,
        xtick_size=8.0,
        ytick_size=8.0,
        axes_linewidth=0.95,
        extra_rc={
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.major.size": 3.4,
            "ytick.major.size": 3.4,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "hatch.linewidth": 0.80,
        },
    )

    if palette_family == "red":
        colors = RED_4
    else:
        colors = BLUE_4
    by_key = index_rows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(4.20, 1.60), dpi=320, sharey=True)
    x = np.arange(len(METHOD_SPECS), dtype=float)
    bar_width = 0.78

    rendered_summary: dict[str, list[dict]] = {}
    all_bars: list[list] = []

    for ax, dataset, dataset_label in zip(
        axes,
        ("SIFT1M", "GLOVE"),
        ("SIFT1M", "GloVe"),
    ):
        panel_row = by_key[(dataset, figure_nprobe, figure_k)]
        rendered_summary[DATASET_DISPLAY[dataset]] = []
        heights = []
        timeout_flags = []
        for spec in METHOD_SPECS:
            timed_out = panel_row[spec["timeout_field"]]
            value = TIMEOUT_MS if timed_out else panel_row[spec["time_field"]]
            heights.append(value)
            timeout_flags.append(timed_out)
            rendered_summary[DATASET_DISPLAY[dataset]].append(
                {
                    "setting": f"{panel_row['nprobe']}/{panel_row['K']}",
                    "method": spec["key"],
                    "p95_ms": value,
                    "timed_out": timed_out,
                }
            )

        bars = []
        for idx, (height, color) in enumerate(zip(heights, colors)):
            bar = ax.bar(
                x[idx],
                height,
                width=bar_width,
                color=color,
                edgecolor=BAR_EDGE_COLORS[idx],
                linewidth=0.72,
                zorder=3,
                hatch=METHOD_SPECS[idx]["hatch"],
            )[0]
            bars.append(bar)
        all_bars.append(bars)
        for bar, timed_out in zip(bars, timeout_flags):
            if timed_out:
                bar.set_hatch("////")
                bar.set_linewidth(0.82)

        ax.set_yscale("log")
        ax.set_ylim(1.0, 2600.0)
        ax.set_xlabel(dataset_label, labelpad=4.0)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", dashes=(2.2, 2.2), zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(_tick_formatter))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-0.65, len(METHOD_SPECS) - 0.35)

        for bar, value, timed_out in zip(bars, heights, timeout_flags):
            label = r"$>$120s" if timed_out else format_ms(value, False)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.04,
                label,
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#2f3f53",
            )

    axes[0].set_ylabel("p95 time (ms)")

    legend_handles = [all_bars[1][i] for i in range(len(METHOD_SPECS))]
    legend_labels = [spec["legend_label"] for spec in METHOD_SPECS]
    axes[1].legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#aaaaaa",
        facecolor="white",
        framealpha=1.0,
        fontsize=7.0,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.4,
        labelspacing=0.3,
    )

    fig.subplots_adjust(top=0.94, bottom=0.20, left=0.11, right=0.995, wspace=0.26)

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_pdf = out_dir / "figure_03_real_data_planner_overhead_piecewise.pdf"
    figure_png = out_dir / "figure_03_real_data_planner_overhead_piecewise.png"
    fig.savefig(figure_pdf)
    fig.savefig(figure_png)
    plt.close(fig)
    return {
        "figure_pdf": repo_relative(figure_pdf),
        "figure_png": repo_relative(figure_png),
        "panels": rendered_summary,
    }


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.out_dir).resolve()

    rows = load_rows(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_tex = render_table(rows)
    table_tex_path = out_dir / "table_03_real_data_planner_overhead_piecewise.tex"
    table_tex_path.write_text(table_tex, encoding="utf-8")

    summary_csv_path = out_dir / "table_03_real_data_planner_overhead_piecewise.csv"
    write_summary_csv(rows, summary_csv_path)

    figure_summary = render_figure(
        rows,
        out_dir,
        args.palette_family,
        args.figure_nprobe,
        args.figure_k,
    )
    summary_json_path = out_dir / "planner_overhead_fourway_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "input_csv": repo_relative(input_csv),
                "output_table_tex": repo_relative(table_tex_path),
                "output_summary_csv": repo_relative(summary_csv_path),
                "figure_nprobe": args.figure_nprobe,
                "figure_k": args.figure_k,
                **figure_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] table={table_tex_path}")
    print(f"[ok] csv={summary_csv_path}")
    print(f"[ok] figure={figure_summary['figure_pdf']}")
    print(f"[ok] summary={summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
