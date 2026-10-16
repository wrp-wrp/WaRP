#!/usr/bin/env python3
"""Render Figure 1: S3 Range-GET latency decomposition (stacked bar)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.plotting.paper_plot_style import apply_paper_preset  # noqa: E402
from artifact.data import paper_data_path  # noqa: E402

DATA_CSV = paper_data_path(
    ROOT / "experiments/figure-01-s3-range-get-latency/data/paper/data.csv"
)
OUT_STEM = ROOT / "experiments/figure-01-s3-range-get-latency/latency_composition"
BAR_COLORS = {
    "bottom": "#3690c0",
    "top": "#d0d1e6",
}
BAR_EDGE_COLORS = {
    "bottom": "#2f3f53",
    "top": "#4a4a4a",
}


def setup_style():
    apply_paper_preset(
        plt, "single_col",
        font_size=8.4, axes_label_size=8.8, axes_title_size=8.8,
        legend_size=7.2, xtick_size=7.2, ytick_size=7.6,
        axes_linewidth=0.85,
        extra_rc={
            "axes.spines.top": True, "axes.spines.right": True,
            "xtick.major.width": 0.75, "ytick.major.width": 0.75,
            "xtick.major.size": 3.0, "ytick.major.size": 3.0,
            "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        },
    )


def fmt_label(kb: int) -> str:
    if kb >= 1000:
        return f"{kb // 1000} MB"
    return f"{kb} KB"


def main():
    setup_style()

    with DATA_CSV.open() as f:
        rows = list(csv.DictReader(f))

    labels = [fmt_label(int(r["payload_kb"])) for r in rows]
    overhead = np.array([float(r["request_overhead_ms"]) for r in rows])
    transfer = np.array([float(r["transfer_time_ms"]) for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(3.35, 2.0), dpi=320)

    ax.bar(x, overhead, width=0.6, color=BAR_COLORS["bottom"], edgecolor=BAR_EDGE_COLORS["bottom"],
           linewidth=0.7, label="Request Overhead", zorder=3)
    ax.bar(x, transfer, width=0.6, bottom=overhead, color=BAR_COLORS["top"],
           edgecolor=BAR_EDGE_COLORS["top"], linewidth=0.7, hatch="///", label="Transfer Time", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(overhead + transfer) * 1.12)
    ax.grid(axis="y", linestyle="--", dashes=(2.0, 2.0), linewidth=0.5,
            alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=2)

    ax.legend(
        frameon=True, fancybox=True, edgecolor="#c0c0c0", facecolor="white",
        framealpha=0.95, borderpad=0.35, handlelength=1.4, handletextpad=0.35,
        loc="upper left",
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print("[ok] figure 1 rendered")


if __name__ == "__main__":
    main()
