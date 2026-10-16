#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.plotting.bar_palette import BAR_PALETTE_CHOICES, get_bar_palette  # noqa: E402
from src.plotting.paper_plot_style import apply_paper_preset  # noqa: E402
from artifact.data import paper_data_path  # noqa: E402

CSV_PATH = paper_data_path(
    ROOT / "experiments/figure-14-sift1m-sensitivity/data/paper/table_sift1m_real_s3_sensitivity.csv"
)
OUT_DIR = ROOT / "experiments/figure-14-sift1m-sensitivity"


apply_paper_preset(
    plt,
    "double_col",
    font_size=9.0,
    axes_label_size=9.4,
    axes_title_size=9.2,
    legend_size=8.0,
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
    },
)


def load_rows():
    rows = []
    with CSV_PATH.open() as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        for raw in reader:
            if not raw:
                continue
            if len(raw) == len(header):
                row = {header[i]: raw[i].strip() for i in range(len(header))}
            elif raw[0] == "cluster" and len(raw) == len(header) + 1:
                row = {
                    "panel": raw[0].strip(),
                    "setting_id": raw[1].strip(),
                    "setting_label": f"{raw[2].strip()}, {raw[3].strip()}",
                    "nlist": raw[4].strip(),
                    "nprobe": raw[5].strip(),
                    "recall_at_10": raw[6].strip(),
                    "budget_k1": raw[7].strip(),
                    "budget_k2": raw[8].strip(),
                    "separate_qps": raw[9].strip(),
                    "separate_p95": raw[10].strip(),
                    "lance_r_qps": raw[11].strip(),
                    "lance_r_p95": raw[12].strip(),
                    "warp_k1_qps": raw[13].strip(),
                    "warp_k1_p95": raw[14].strip(),
                    "warp_k2_qps": raw[15].strip(),
                    "warp_k2_p95": raw[16].strip(),
                }
            else:
                raise ValueError(f"Unexpected CSV row shape: {raw}")
            rows.append(row)
    return rows


def draw_panel(ax, rows, xlabels, palette_family="blue"):
    x = np.arange(len(rows))
    width = 0.18
    separate = np.array([float(r["separate_qps"]) for r in rows])
    lance_r = np.array([float(r["lance_r_qps"]) for r in rows])
    warp_k1 = np.array([float(r["warp_k1_qps"]) for r in rows])
    warp_k2 = np.array([float(r["warp_k2_qps"]) for r in rows])
    recall = np.array([100.0 * float(r["recall_at_10"]) for r in rows])

    k1 = rows[0]["budget_k1"]
    k2 = rows[0]["budget_k2"]

    if palette_family == "paper":
        colors = {
            "separate": "#c7ccd4",
            "lance_r": "#95a9c6",
            "warp1": "#5f87b3",
            "warp2": "#274f7d",
            "recall": "#374151",
        }
    else:
        bar_colors = get_bar_palette(palette_family, 4)
        colors = {
            "separate": bar_colors[0],
            "lance_r": bar_colors[1],
            "warp1": bar_colors[2],
            "warp2": bar_colors[3],
            "recall": "#374151",
        }

    ax.bar(
        x - 1.5 * width,
        separate,
        width,
        color=colors["separate"],
        edgecolor="#4a4a4a",
        linewidth=0.7,
        label="Separate",
        zorder=3,
    )
    ax.bar(
        x - 0.5 * width,
        lance_r,
        width,
        color=colors["lance_r"],
        edgecolor="#4a4a4a",
        linewidth=0.7,
        hatch="///",
        label="Lance + reorder",
        zorder=3,
    )
    ax.bar(
        x + 0.5 * width,
        warp_k1,
        width,
        color=colors["warp1"],
        edgecolor="#2f3f53",
        linewidth=0.7,
        label=f"WaRP + reorder, K <= {k1}",
        zorder=3,
    )
    ax.bar(
        x + 1.5 * width,
        warp_k2,
        width,
        color=colors["warp2"],
        edgecolor="#23384f",
        linewidth=0.7,
        label=f"WaRP + reorder, K <= {k2}",
        zorder=3,
    )

    ax.set_ylabel("QPS")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.grid(axis="y", linestyle="--", dashes=(2.2, 2.2), zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=2)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        recall,
        color=colors["recall"],
        marker="o",
        markerfacecolor=colors["recall"],
        markeredgecolor=colors["recall"],
        markersize=3.6,
        linewidth=1.4,
        linestyle=(0, (3, 2)),
        label="Recall (%)",
        zorder=4,
    )
    ax2.set_ylabel("Recall (%)", color=colors["recall"])
    ax2.set_ylim(90, 100.5)
    ax2.set_yticks([90, 95, 100])
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", colors=colors["recall"], pad=2)

    return ax, ax2


def plot_combined(nprobe_rows, nprobe_labels, cluster_rows, cluster_labels, palette_family="blue"):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.90), dpi=320)
    ax_l, ax2_l = draw_panel(axes[0], nprobe_rows, nprobe_labels, palette_family=palette_family)
    ax_r, ax2_r = draw_panel(axes[1], cluster_rows, cluster_labels, palette_family=palette_family)

    ax_l.set_title("(a) Varying Probe Count", pad=4.0)
    ax_r.set_title("(b) Varying Cluster Granularity", pad=4.0)

    handles1, labels1 = ax_l.get_legend_handles_labels()
    handles2, labels2 = ax2_l.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#c0c0c0",
        facecolor="white",
        framealpha=0.95,
        borderpad=0.4,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    ax2_l.set_ylabel("")
    ax2_l.tick_params(axis="y", right=False, labelright=False)
    ax_r.set_ylabel("")
    ax2_r.set_ylabel("Recall (%)", color="#374151", labelpad=6)
    ax2_r.tick_params(axis="y", colors="#374151", pad=2)

    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.08, right=0.965, wspace=0.24)

    fig.savefig(OUT_DIR / "figure_sift1m_real_s3_sensitivity_combined.pdf", dpi=320)
    fig.savefig(OUT_DIR / "figure_sift1m_real_s3_sensitivity_combined.png", dpi=320)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette-family", choices=BAR_PALETTE_CHOICES, default="blue")
    args = parser.parse_args()

    rows = load_rows()
    nprobe_rows = [r for r in rows if r["panel"] == "nprobe"]
    cluster_rows = [r for r in rows if r["panel"] == "cluster"]

    nprobe_labels = [r["setting_label"].replace("nprobe=", "") for r in nprobe_rows]
    cluster_labels = [r["setting_label"].replace("nlist=", "").replace(", nprobe=", "\nnprobe=") for r in cluster_rows]

    plot_combined(
        nprobe_rows,
        nprobe_labels,
        cluster_rows,
        cluster_labels,
        palette_family=args.palette_family,
    )


if __name__ == "__main__":
    main()
