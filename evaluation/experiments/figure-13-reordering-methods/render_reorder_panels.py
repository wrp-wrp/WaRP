from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.plotting.bar_palette import get_bar_palette  # noqa: E402
from src.plotting.paper_plot_style import apply_paper_preset, style_legend_frame  # noqa: E402
from artifact.data import paper_data_path  # noqa: E402


OUT_DIR = HERE
LEFT_SOURCE_CSV = paper_data_path(
    HERE / "data" / "paper" / "stages" / "reorder_cost.csv"
)
RIGHT_SOURCE_CSV = paper_data_path(
    HERE / "data" / "paper" / "stages" / "locality_gap.csv"
)
PREFERRED_METHOD_ORDER = [
    "Identity",
    "Random proj.",
    "PCA-1D",
    "PCA-2D + Z-order",
    "PCA-2D + Hilbert",
    "Spectral (cold)",
    "Spectral (log)",
]
def _load_left_df() -> pd.DataFrame:
    df = pd.read_csv(LEFT_SOURCE_CSV)
    required = {"num_clusters", "method_label", "reorder_time_ms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {LEFT_SOURCE_CSV}: {missing}")
    return df


def _load_right_df() -> pd.DataFrame:
    df = pd.read_csv(RIGHT_SOURCE_CSV)
    required = {"dataset_label", "method_label", "weighted_gap_ratio"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {RIGHT_SOURCE_CSV}: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette-family", choices=("red", "blue"), default="red")
    args = parser.parse_args()

    apply_paper_preset(
        plt,
        "single_col_microbench",
        extra_rc={
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
        },
    )

    left_df = _load_left_df()
    right_df = _load_right_df()
    left_df = left_df[left_df["num_clusters"].astype(int).isin({1024, 8192})]
    right_df = right_df[right_df["dataset_label"].isin({"SIFT1M", "GloVe"})]
    methods = [m for m in PREFERRED_METHOD_ORDER if m in set(right_df["method_label"].tolist())]
    if not methods:
        raise SystemExit("no supported method_label values found in reorder panels source csv")
    colors = {
        method: color
        for method, color in zip(methods, get_bar_palette(args.palette_family, len(methods)))
    }

    fig, (ax_cost, ax_gap) = plt.subplots(1, 2, figsize=(3.45, 1.43), dpi=320, gridspec_kw={"width_ratios": [1.0, 1.0]})

    counts = sorted(int(v) for v in left_df["num_clusters"].dropna().astype(int).unique().tolist())
    cost_grouped = (
        left_df.groupby(["num_clusters", "method_label"], as_index=False)["reorder_time_ms"]
        .mean()
        .sort_values(["num_clusters", "method_label"])
    )
    x_cost = np.arange(len(counts))
    width = min(0.11, 0.80 / max(1, len(methods)))
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
    for idx, method in enumerate(methods):
        sub = cost_grouped[cost_grouped["method_label"] == method].set_index("num_clusters")
        vals = np.array([float(sub.loc[count, "reorder_time_ms"]) for count in counts], dtype=float)
        ax_cost.bar(
            x_cost + offsets[idx],
            vals,
            width=width,
            color=colors[method],
            edgecolor="#4a4a4a",
            linewidth=0.7,
            label=method,
            zorder=3,
        )
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("Time (ms)")
    ax_cost.set_xticks(x_cost)
    ax_cost.set_xticklabels([f"C={value}" for value in counts])
    ax_cost.set_ylim(0.03, float(cost_grouped["reorder_time_ms"].max()) * 1.25)
    ax_cost.grid(axis="y", linestyle="--", dashes=(2.2, 2.2), zorder=0)
    ax_cost.set_axisbelow(True)
    ax_cost.set_title("(a) Permutation Computation", pad=2.0)

    sort_anchor = "PCA-2D + Hilbert" if "PCA-2D + Hilbert" in methods else methods[-1]
    dataset_order = (
        right_df[right_df["method_label"] == sort_anchor][["dataset_label", "weighted_gap_ratio"]]
        .sort_values("weighted_gap_ratio")["dataset_label"]
        .tolist()
    )
    x_gap = np.arange(len(dataset_order), dtype=float)
    gap_width = min(0.11, 0.78 / max(1, len(methods)))
    gap_offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * gap_width
    for idx, method in enumerate(methods):
        sub = right_df[right_df["method_label"] == method].set_index("dataset_label").reindex(dataset_order)
        vals = sub["weighted_gap_ratio"].to_numpy(dtype=float)
        ax_gap.bar(
            x_gap + gap_offsets[idx],
            vals,
            width=gap_width,
            color=colors[method],
            edgecolor="#2e2e2e",
            linewidth=0.7,
            label=method,
            zorder=3,
        )
    ax_gap.set_ylabel(r"$G(\sigma)/G(\sigma_{\mathrm{id}})$")
    ax_gap.set_xticks(x_gap)
    ax_gap.set_xticklabels(dataset_order)
    ax_gap.set_ylim(0.0, float(right_df["weighted_gap_ratio"].max()) * 1.18)
    ax_gap.grid(axis="y", linestyle="--", dashes=(2.2, 2.2), zorder=0)
    ax_gap.tick_params(axis="x", length=0)
    ax_gap.set_axisbelow(True)
    ax_gap.set_title("(b) Reordering Quality", pad=2.0)

    display_label = {
        "Identity": "Identity",
        "Random proj.": "Random proj.",
        "PCA-1D": "PCA-1D",
        "PCA-2D + Z-order": r"PCA $n$-D + Z-order",
        "PCA-2D + Hilbert": r"PCA $n$-D + Hilbert",
        "Spectral (cold)": "Spectral (cold)",
        "Spectral (log)": "Spectral (log)",
    }
    handles = [plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=colors[method], edgecolor="#4a4a4a", linewidth=0.7) for method in methods]
    labels = [display_label.get(m, m) for m in methods]
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=True,
        fancybox=True,
        shadow=False,
        facecolor="white",
        borderpad=0.18,
        handlelength=0.9,
        handletextpad=0.30,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    style_legend_frame(legend)
    fig.subplots_adjust(top=0.72, bottom=0.195, left=0.14, right=0.99, wspace=0.32)
    fig.savefig(OUT_DIR / "figure_reorder_panels.pdf")
    fig.savefig(OUT_DIR / "figure_reorder_panels.png")


if __name__ == "__main__":
    main()
