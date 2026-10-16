#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.plotting.paper_plot_style import apply_paper_preset  # noqa: E402
from artifact.data import paper_data_path  # noqa: E402

# Canonical closed-loop env3 data lives under experiments/<id>/data/ in this repo.
LOCAL_DATA_JSON = paper_data_path(
    ROOT
    / "experiments"
    / "figure-10-p-sweep-closedloop"
    / "data"
    / "paper"
    / "data_gap_env3_v3"
    / "figure_closed_loop_env3_reorder_only_data.json"
)
OUT_DIR = ROOT / "experiments" / "figure-10-p-sweep-closedloop"

DATASETS = ["sift1m", "glove", "deep1m", "openimage", "msong", "random1m128"]
DATASET_TITLES = {
    "sift1m": "SIFT1M",
    "glove": "GloVe",
    "deep1m": "DEEP1M",
    "openimage": "OpenImage",
    "msong": "MSong",
    "random1m128": "Random-1M",
}
ENV_ORDER = [(1, 64), (4, 64), (4, 8)]
ENV_TITLES = {
    (1, 64): "Relaxed",
    (4, 64): "Moderate",
    (4, 8): "Stressed",
}
ENV_SUBTITLES = {
    (1, 64): "Q=1, R=64",
    (4, 64): "Q=4, R=64",
    (4, 8): "Q=4, R=8",
}
P_ORDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
P_TO_X = {float(p): i for i, p in enumerate(P_ORDER)}
P_TICKS = [0, 2, 4, 6, 8]
P_TICKLABELS = [r"$2^0$", r"$2^2$", r"$2^4$", r"$2^6$", r"$2^8$"]

# Vivid colors for baseline dashed lines (easier to distinguish)
SEPARATE_COLOR = "#374151"
LANCE_COLOR = "#2563EB"
GREEDY_COLOR = "#16A34A"
WARP_R_COLOR = "#EF6548"
STAR_FACE = "#FDBA74"
STAR_EDGE = "#C2410C"
BUDGET_K = 64


def load_data() -> dict:
    override = os.environ.get("VECIO_CLOSED_LOOP_JSON", "").strip()
    if override:
        src = Path(override).expanduser().resolve()
    else:
        src = LOCAL_DATA_JSON
    return json.loads(src.read_text(encoding="utf-8"))


def setup_style() -> None:
    apply_paper_preset(
        plt,
        "dense_grid",
        axes_title_size=8.0,
        axes_linewidth=0.78,
        grid_color="#d7d7d7",
        grid_alpha=0.28,
        grid_linewidth=0.45,
        extra_rc={
            "axes.titlepad": 4.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
        },
    )


def rows_for_env(data: dict, dataset: str, q: int, io_threads: int) -> list[dict]:
    return [
        r
        for r in data.get("rows", [])
        if str(r.get("dataset")) == dataset
        and int(r.get("query_concurrency", -1)) == q
        and int(r.get("process_io_threads_limit", -1)) == io_threads
    ]


def selected_for_env(data: dict, dataset: str, q: int, io_threads: int) -> list[dict]:
    return [
        r
        for r in data.get("selected_points", [])
        if str(r.get("dataset")) == dataset
        and int(r.get("query_concurrency", -1)) == q
        and int(r.get("process_io_threads_limit", -1)) == io_threads
        and str(r.get("layout")) == "reordered"
    ]


def warp_curve_rows(rows: list[dict], metric_key: str) -> list[dict]:
    out = []
    for row in rows:
        if str(row.get("layout")) != "reordered":
            continue
        if not str(row.get("method", "")).startswith("warp_reorder_p"):
            continue
        if int(row.get("budget_k") or -1) != BUDGET_K:
            continue
        if row.get(metric_key) is None or row.get("p") is None:
            continue
        out.append(row)
    out.sort(key=lambda r: float(r["p"]))
    return out


def baseline_value(rows: list[dict], method: str, metric_key: str) -> float | None:
    for row in rows:
        if str(row.get("method")) == method and row.get(metric_key) is not None:
            return float(row[metric_key])
    return None


def selected_value(selected: list[dict], method: str, metric_key: str) -> float | None:
    for row in selected:
        if str(row.get("method")) == method and row.get(metric_key) is not None:
            return float(row[metric_key])
    return None


def limits_linear(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None]
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, hi * 0.12, 1e-6)
    return max(0.0, lo - 0.12 * span), hi + 0.12 * span


def limits_log(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    return min(vals) / 1.15, max(vals) * 1.15


def render_metric(data: dict, *, metric_key: str, out_name: str, y_label: str,
                  log_scale: bool, datasets_override: list[str] | None = None) -> None:
    ds_list = datasets_override or DATASETS
    n_envs = len(ENV_ORDER)
    n_ds = len(ds_list)
    fig, axes = plt.subplots(n_envs, n_ds, figsize=(7.00, 1.25 * n_envs))

    per_panel = []
    row_values: dict[int, list[float]] = {i: [] for i in range(n_envs)}
    for row_idx, (q, io_threads) in enumerate(ENV_ORDER):
        for col_idx, dataset in enumerate(ds_list):
            rows = rows_for_env(data, dataset, q, io_threads)
            selected = selected_for_env(data, dataset, q, io_threads)
            curve = warp_curve_rows(rows, metric_key)
            lance = baseline_value(rows, "lance_native_reorder", metric_key)
            separate = baseline_value(rows, "separate_unbounded", metric_key)
            greedy_key = "greedy_best_qps" if metric_key == "qps" else "greedy_best_latency"
            greedy = selected_value(selected, greedy_key, metric_key)
            highlight_p = float(io_threads) / float(q)
            vals = [float(r[metric_key]) for r in curve if r.get(metric_key) is not None]
            vals.extend(v for v in [lance, separate, greedy] if v is not None)
            row_values[row_idx].extend(vals)
            per_panel.append((dataset, q, io_threads, curve, lance, separate, greedy, highlight_p))

    # Per-row y-limits so each env regime uses its own scale
    row_ylims = {}
    for row_idx, vals in row_values.items():
        row_ylims[row_idx] = limits_log(vals) if log_scale else limits_linear(vals)

    for idx, (dataset, q, io_threads, curve, lance, separate, greedy, highlight_p) in enumerate(per_panel):
        row_idx = idx // n_ds
        col_idx = idx % n_ds
        ax = axes[row_idx][col_idx]
        if log_scale:
            ax.set_yscale("log", base=10)
        ax.set_ylim(*row_ylims[row_idx])
        ax.set_xlim(-0.12, len(P_ORDER) - 0.88)

        # Baselines as horizontal reference lines. Each method gets its
        # own linestyle (and colour) so they're distinguishable even in
        # small/greyscale prints.
        if separate is not None:
            ax.axhline(separate, color=SEPARATE_COLOR, linestyle=(0, (4, 2)),
                       linewidth=0.9, alpha=0.85, zorder=1)
        if lance is not None:
            ax.axhline(lance, color=LANCE_COLOR, linestyle=(0, (5, 1.5, 1, 1.5)),
                       linewidth=0.9, alpha=0.85, zorder=2)
        if greedy is not None:
            ax.axhline(greedy, color=GREEDY_COLOR, linestyle=(0, (1, 1.5)),
                       linewidth=0.9, alpha=0.85, zorder=3)

        # Main curve: solid line + white-filled circles (matching Fig 5 draw_budget_curve)
        plotted = [r for r in curve if float(r["p"]) in P_TO_X]
        xs = [P_TO_X[float(r["p"])] for r in plotted]
        ys = [float(r[metric_key]) for r in plotted]
        if xs:
            ax.plot(
                xs,
                ys,
                color=WARP_R_COLOR,
                linewidth=1.05,
                linestyle="-",
                zorder=4,
            )
            ax.scatter(
                xs,
                ys,
                s=10,
                marker="o",
                facecolors="white",
                edgecolors=WARP_R_COLOR,
                linewidths=0.6,
                zorder=5,
            )
            hi = next((r for r in plotted if abs(float(r["p"]) - highlight_p) < 1e-9), None)
            if hi is not None:
                ax.scatter(
                    [P_TO_X[float(hi["p"])]],
                    [float(hi[metric_key])],
                    marker="*",
                    s=48,
                    facecolor=STAR_FACE,
                    edgecolor=STAR_EDGE,
                    linewidth=0.6,
                    zorder=6,
                )

        # Y-axis label: "QPS" on the leftmost column, kept short and close to the panels.
        # Env-row descriptors ("Moderate / Q=4, R=64", etc.) are drawn separately on
        # the outer figure margin below.
        if col_idx == 0:
            ax.set_ylabel(y_label, fontsize=7.0, labelpad=2.5)
        # Column headers: dataset name on first row
        if row_idx == 0:
            ax.set_title(DATASET_TITLES[dataset], fontsize=7.8, pad=3.0)
        # X label on last row
        if row_idx == n_envs - 1:
            ax.set_xlabel(r"$P$")
            ax.set_xticks(P_TICKS)
            ax.set_xticklabels(P_TICKLABELS)
        else:
            ax.set_xticks(P_TICKS)
            ax.tick_params(axis="x", labelbottom=False)

        ax.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.26)
        ax.tick_params(axis="x", which="both", length=2.6, width=0.72, pad=1.8)
        ax.tick_params(axis="y", which="both", length=2.6, width=0.72, pad=1.2, labelsize=5.8)

    # --- Legend matching Figure 5 style ---
    handles = [
        Line2D([0], [0], color=WARP_R_COLOR, linewidth=1.05, linestyle="-",
               label=f"WaRP + reorder, K\u2009\u2264\u2009{BUDGET_K}"),
        Line2D([0], [0], color=GREEDY_COLOR, linewidth=0.9, linestyle=(0, (1, 1.5)),
               label="Greedy-oracle + reorder"),
        Line2D([0], [0], color=LANCE_COLOR, linewidth=0.9, linestyle=(0, (5, 1.5, 1, 1.5)),
               label="Lance + reorder"),
        Line2D([0], [0], color=SEPARATE_COLOR, linewidth=0.9, linestyle=(0, (4, 2)),
               label="Separate"),
        Line2D([0], [0], marker="*", markersize=6.0, markerfacecolor=STAR_FACE, markeredgecolor=STAR_EDGE, linewidth=0, label=r"Default $P{=}R/Q$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#c0c0c0",
        facecolor="white",
        framealpha=0.95,
        borderpad=0.3,
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=0.7,
        labelspacing=0.2,
        fontsize=6.0,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.92, bottom=0.13, wspace=0.25, hspace=0.40)

    # Env-row descriptors as figure-level annotations, placed just outside the
    # per-panel "QPS" y-label so they stay visually close to the panels.
    for row_idx, (q, io_threads) in enumerate(ENV_ORDER):
        ax = axes[row_idx][0]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(
            0.030, y_center,
            f"{ENV_TITLES[(q, io_threads)]}\n{ENV_SUBTITLES[(q, io_threads)]}",
            ha="left", va="center",
            fontsize=6.5, color="#1F2937",
            rotation=90,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    data = load_data()

    # Single figure: 3 rows (envs) x 6 cols (datasets)
    render_metric(data, metric_key="qps",
                  out_name="figure_closed_loop_env3_reorder_only_combined",
                  y_label="QPS", log_scale=False)


if __name__ == "__main__":
    main()
