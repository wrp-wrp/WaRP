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

LOCAL_AGGREGATED_JSON = (
    ROOT
    / "experiments"
    / "figure-09-qps-cost"
    / "aggregated_results_override.json"
)
REMOTE_AGGREGATED_JSON = paper_data_path(
    ROOT
    / "experiments"
    / "figure-09-qps-cost"
    / "data"
    / "paper"
    / "aggregated_results.json"
)
OUT_DIR = ROOT / "experiments" / "figure-09-qps-cost"
OUT_COMBINED_PDF = OUT_DIR / "figure_qps_vs_savings_q4_io64_all6_grid_qps_p95_3x4.pdf"
OUT_COMBINED_PNG = OUT_DIR / "figure_qps_vs_savings_q4_io64_all6_grid_qps_p95_3x4.png"

DATASETS = ["sift1m", "glove", "deep1m", "openimage", "msong", "random1m128"]
DATASET_TITLES = {
    "sift1m": "SIFT1M",
    "glove": "GloVe",
    "deep1m": "DEEP1M",
    "openimage": "OpenImage",
    "msong": "MSong",
    "random1m128": "Random-1M",
}

QUERY_CONCURRENCY = 4
IO_THREADS = 64
BALANCED_P = 16.0

ALLOWED_K = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128}
ALLOWED_GAP_KB = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
ALLOWED_CAPS_MB = {1, 2, 4, 8, 16, 32}

SEPARATE_COLOR = "#111827"
LANCE_COLOR = "#a6bddb"
GREEDY_COLOR = "#74a9cf"
WARP_COLOR = "#2b8cbe"
LANCE_R_COLOR = "#fdbb84"
GREEDY_R_COLOR = "#fc8d59"
WARP_R_COLOR = "#ef6548"


def load_data() -> dict:
    override = os.environ.get("VECIO_FIG7_AGGREGATED_JSON", "").strip()
    if override:
        src = Path(override).expanduser().resolve()
    else:
        src = LOCAL_AGGREGATED_JSON if LOCAL_AGGREGATED_JSON.exists() else REMOTE_AGGREGATED_JSON
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


def rows_for_env(data: dict, dataset: str) -> list[dict]:
    return [
        r
        for r in data.get("rows", [])
        if str(r.get("dataset")) == dataset
        and int(r.get("query_concurrency", -1)) == QUERY_CONCURRENCY
        and int(r.get("process_io_threads_limit", -1)) == IO_THREADS
    ]


def greedy_for_env(data: dict, dataset: str) -> list[dict]:
    return [
        r
        for r in data.get("greedy_unbounded_candidates", [])
        if str(r.get("dataset")) == dataset
        and int(r.get("query_concurrency", -1)) == QUERY_CONCURRENCY
        and int(r.get("process_io_threads_limit", -1)) == IO_THREADS
    ]


def find_row(rows: list[dict], method: str) -> dict | None:
    for r in rows:
        if str(r.get("method")) == method:
            return r
    return None


def savings(avg_req: float, separate_req: float) -> float:
    return 100.0 * (1.0 - float(avg_req) / float(separate_req))


def balanced_method(layout: str, p: float) -> str:
    prefix = "warp_reorder_p" if layout == "reordered" else "warp_original_p"
    return f"{prefix}{int(round(p))}" if abs(p - round(p)) < 1e-9 else f"{prefix}{p:g}"


def warp_curve_metric(rows: list[dict], *, layout: str, p: float, sep_req: float, metric_key: str) -> list[dict]:
    specific = balanced_method(layout, p)
    canonical = "warp_reorder" if layout == "reordered" else "warp_original"
    pts = []
    for r in rows:
        method = str(r.get("method"))
        if method not in {specific, canonical}:
            continue
        k = int(r.get("budget_k") or -1)
        if k not in ALLOWED_K:
            continue
        if r.get(metric_key) is None or r.get("avg_num_requests") is None:
            continue
        pts.append(
            {
                "k": k,
                "x": savings(float(r["avg_num_requests"]), sep_req),
                "y": float(r[metric_key]),
            }
        )
    pts.sort(key=lambda z: z["k"])
    return pts


def greedy_cloud_metric(candidates: list[dict], *, layout: str, sep_req: float, metric_key: str) -> list[dict]:
    pts = []
    for r in candidates:
        if str(r.get("layout")) != layout:
            continue
        if r.get(metric_key) is None or r.get("avg_num_requests") is None:
            continue
        gap_kb = int(r.get("gap_kb") or 0)
        cap_mb = int(int(r.get("span_cap_bytes") or 0) // (1024 * 1024))
        if gap_kb not in ALLOWED_GAP_KB or cap_mb not in ALLOWED_CAPS_MB:
            continue
        pts.append(
            {
                "gap_kb": gap_kb,
                "cap_mb": cap_mb,
                "x": savings(float(r["avg_num_requests"]), sep_req),
                "y": float(r[metric_key]),
            }
        )
    pts.sort(key=lambda z: (z["gap_kb"], z["cap_mb"]))
    return pts


def scatter_point(ax, x: float, y: float, *, marker: str, color: str, size: float, filled: bool = True, zorder: int = 4) -> None:
    ax.scatter(
        [x],
        [y],
        s=size,
        marker=marker,
        facecolors=color if filled else "white",
        edgecolors=color,
        linewidths=1.15,
        zorder=zorder,
    )


def draw_budget_curve(ax, pts: list[dict], *, color: str, linestyle: str) -> None:
    if not pts:
        return
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=1.2, linestyle=linestyle, zorder=4)
    ax.scatter(xs, ys, s=13, marker="o", facecolors="white", edgecolors=color, linewidths=0.75, zorder=5)


def savings_ticks(xlo: float, xhi: float) -> list[int]:
    preferred = [0, 20, 40, 60, 80, 90]
    return [t for t in preferred if xlo <= t <= xhi]


def draw_greedy_cloud(ax, pts: list[dict], *, color: str, marker: str) -> None:
    if not pts:
        return
    ax.scatter(
        [p["x"] for p in pts],
        [p["y"] for p in pts],
        s=7,
        marker=marker,
        facecolors=color,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.68,
        zorder=3,
    )


def render_combined_qps_p95() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    setup_style()

    fig = plt.figure(figsize=(7.75, 2.95))
    gs = fig.add_gridspec(
        3, 4,
        left=0.075, right=0.995, top=0.97, bottom=0.32,
        wspace=0.24, hspace=0.55,
        width_ratios=[1, 1, 1, 1],
    )
    # Insert extra horizontal gap between QPS pair (cols 0-1) and p95 pair (cols 2-3)
    # by using a nested approach: two column groups with a gap
    gs_left = fig.add_gridspec(
        3, 2,
        left=0.075, right=0.475, top=0.92, bottom=0.14,
        wspace=0.28, hspace=0.48,
    )
    gs_right = fig.add_gridspec(
        3, 2,
        left=0.555, right=0.995, top=0.92, bottom=0.14,
        wspace=0.28, hspace=0.48,
    )
    axes = [[None] * 4 for _ in range(3)]
    for r in range(3):
        for c in range(2):
            axes[r][c] = fig.add_subplot(gs_left[r, c])
            axes[r][c + 2] = fig.add_subplot(gs_right[r, c])

    qps_xmins: list[float] = []
    qps_xmaxs: list[float] = []
    qps_ymins: list[float] = []
    qps_ymaxs: list[float] = []
    p95_xmins: list[float] = []
    p95_xmaxs: list[float] = []
    p95_ymins: list[float] = []
    p95_ymaxs: list[float] = []
    per_dataset = []

    for dataset in DATASETS:
        rows = rows_for_env(data, dataset)
        greedy_rows = greedy_for_env(data, dataset)
        separate = find_row(rows, "separate_unbounded")
        if separate is None or separate.get("avg_num_requests") is None:
            raise SystemExit(f"missing Separate for {dataset}")
        sep_req = float(separate["avg_num_requests"])
        lance = find_row(rows, "lance_native")
        lance_r = find_row(rows, "lance_native_reorder")
        warp_qps = warp_curve_metric(rows, layout="original", p=BALANCED_P, sep_req=sep_req, metric_key="qps")
        warp_r_qps = warp_curve_metric(rows, layout="reordered", p=BALANCED_P, sep_req=sep_req, metric_key="qps")
        greedy_qps = greedy_cloud_metric(greedy_rows, layout="original", sep_req=sep_req, metric_key="qps")
        greedy_r_qps = greedy_cloud_metric(greedy_rows, layout="reordered", sep_req=sep_req, metric_key="qps")
        warp_p95 = warp_curve_metric(rows, layout="original", p=BALANCED_P, sep_req=sep_req, metric_key="p95_latency_ms")
        warp_r_p95 = warp_curve_metric(rows, layout="reordered", p=BALANCED_P, sep_req=sep_req, metric_key="p95_latency_ms")
        greedy_p95 = greedy_cloud_metric(greedy_rows, layout="original", sep_req=sep_req, metric_key="p95_latency_ms")
        greedy_r_p95 = greedy_cloud_metric(greedy_rows, layout="reordered", sep_req=sep_req, metric_key="p95_latency_ms")

        xs_qps = [0.0]
        ys_qps = [float(separate["qps"])]
        xs_p95 = [0.0]
        ys_p95 = [float(separate["p95_latency_ms"])]
        for row in (lance, lance_r):
            if row and row.get("avg_num_requests") is not None:
                x = savings(float(row["avg_num_requests"]), sep_req)
                if row.get("qps") is not None:
                    xs_qps.append(x)
                    ys_qps.append(float(row["qps"]))
                if row.get("p95_latency_ms") is not None:
                    xs_p95.append(x)
                    ys_p95.append(float(row["p95_latency_ms"]))
        for pts in (warp_qps, warp_r_qps, greedy_qps, greedy_r_qps):
            xs_qps.extend(p["x"] for p in pts)
            ys_qps.extend(p["y"] for p in pts)
        for pts in (warp_p95, warp_r_p95, greedy_p95, greedy_r_p95):
            xs_p95.extend(p["x"] for p in pts)
            ys_p95.extend(p["y"] for p in pts)

        qps_xmins.append(min(xs_qps))
        qps_xmaxs.append(max(xs_qps))
        qps_ymins.append(min(ys_qps))
        qps_ymaxs.append(max(ys_qps))
        p95_xmins.append(min(xs_p95))
        p95_xmaxs.append(max(xs_p95))
        p95_ymins.append(min(ys_p95))
        p95_ymaxs.append(max(ys_p95))
        per_dataset.append(
            {
                "rows": rows,
                "separate": separate,
                "lance": lance,
                "lance_r": lance_r,
                "warp_qps": warp_qps,
                "warp_r_qps": warp_r_qps,
                "greedy_qps": greedy_qps,
                "greedy_r_qps": greedy_r_qps,
                "warp_p95": warp_p95,
                "warp_r_p95": warp_r_p95,
                "greedy_p95": greedy_p95,
                "greedy_r_p95": greedy_r_p95,
                "sep_req": sep_req,
            }
        )

    qps_xlo = min(qps_xmins) - 2.0
    qps_xhi = max(qps_xmaxs) + 2.0
    qps_ylo = max(0.0, min(qps_ymins) * 0.88)
    qps_yhi = max(qps_ymaxs) * 1.08

    p95_xlo = min(p95_xmins) - 2.0
    p95_xhi = max(p95_xmaxs) + 2.0
    p95_ylo = max(1e-1, min(p95_ymins) * 0.88)
    p95_yhi = max(p95_ymaxs) * 1.10

    for i, dataset in enumerate(DATASETS):
        row = i // 2
        dataset_pair = i % 2
        qps_ax = axes[row][dataset_pair]
        p95_ax = axes[row][dataset_pair + 2]
        d = per_dataset[i]
        sep_req = d["sep_req"]

        for ax, metric_key, warp, warp_r, greedy, greedy_r in (
            (qps_ax, "qps", d["warp_qps"], d["warp_r_qps"], d["greedy_qps"], d["greedy_r_qps"]),
            (p95_ax, "p95_latency_ms", d["warp_p95"], d["warp_r_p95"], d["greedy_p95"], d["greedy_r_p95"]),
        ):
            sep_y = float(d["separate"][metric_key])
            scatter_point(ax, 0.0, sep_y, marker="X", color=SEPARATE_COLOR, size=38, filled=True, zorder=6)
            for lr, color, filled in ((d["lance"], LANCE_COLOR, True), (d["lance_r"], LANCE_R_COLOR, False)):
                if lr and lr.get("avg_num_requests") is not None and lr.get(metric_key) is not None:
                    scatter_point(
                        ax,
                        savings(float(lr["avg_num_requests"]), sep_req),
                        float(lr[metric_key]),
                        marker="^",
                        color=color,
                        size=26,
                        filled=filled,
                    )
            draw_greedy_cloud(ax, greedy, color=GREEDY_COLOR, marker="o")
            draw_greedy_cloud(ax, greedy_r, color=GREEDY_R_COLOR, marker="s")
            draw_budget_curve(ax, warp, color=WARP_COLOR, linestyle="-")
            draw_budget_curve(ax, warp_r, color=WARP_R_COLOR, linestyle="--")
            ax.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.26)
            ax.tick_params(axis="both", which="both", length=2.4, width=0.72, pad=1.7)

        for ax in (qps_ax, p95_ax):
            ax.text(
                0.03, 0.94, DATASET_TITLES[dataset],
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=7.0,
                bbox=dict(facecolor="white", edgecolor="#c0c0c0",
                          linewidth=0.5, boxstyle="round,pad=0.18"),
                zorder=5,
            )

        qps_ax.set_xlim(qps_xlo, qps_xhi)
        qps_ax.set_ylim(qps_ylo, qps_yhi)
        p95_ax.set_xlim(p95_xlo, p95_xhi)
        p95_ax.set_ylim(p95_ylo, p95_yhi)
        p95_ax.set_yscale("log")
        qps_ax.set_xticks(savings_ticks(qps_xlo, qps_xhi))
        p95_ax.set_xticks(savings_ticks(p95_xlo, p95_xhi))

        if row == 2:
            qps_ax.set_xlabel("Savings vs Separate (%)")
            p95_ax.set_xlabel("Savings vs Separate (%)")
        if dataset_pair == 0:
            qps_ax.set_ylabel("QPS")
            p95_ax.set_ylabel("p95 latency (ms)")

    # ── Legend: single horizontal row of all seven method handles ──
    handles_grid = [
        Line2D([0], [0], marker="X", color=SEPARATE_COLOR, linestyle="None", markersize=6.0, label="Separate"),
        Line2D([0], [0], marker="^", color=LANCE_COLOR, linestyle="None", markersize=5.5, label="Lance"),
        Line2D([0], [0], marker="^", color=LANCE_R_COLOR, markerfacecolor="white", linestyle="None", markersize=5.5, label="Lance + reorder"),
        Line2D([0], [0], color=WARP_COLOR, linewidth=1.55, linestyle="-", label="WaRP"),
        Line2D([0], [0], color=WARP_R_COLOR, linewidth=1.55, linestyle="--", label="WaRP + reorder"),
        Line2D([0], [0], marker="o", color=GREEDY_COLOR, markerfacecolor=GREEDY_COLOR, linestyle="None", markersize=4.5, label="Greedy-oracle"),
        Line2D([0], [0], marker="s", color=GREEDY_R_COLOR, markerfacecolor=GREEDY_R_COLOR, linestyle="None", markersize=4.5, label="Greedy-oracle + reorder"),
    ]
    fig.legend(
        handles=handles_grid,
        loc="lower center",
        ncol=len(handles_grid),
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#c0c0c0",
        facecolor="white",
        framealpha=0.95,
        borderpad=0.35,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=0.9,
        handletextpad=0.35,
        labelspacing=0.30,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.20, wspace=0.24, hspace=0.55)
    fig.savefig(OUT_COMBINED_PDF, bbox_inches="tight")
    fig.savefig(OUT_COMBINED_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render_combined_qps_p95()
