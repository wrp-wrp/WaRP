#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT / "experiments" / "figure-14-sift1m-sensitivity" / "data" / "latest"
PANEL_ORDER = ["nprobe", "cluster"]
PANEL_TITLES = {
    "nprobe": "nprobe sweep",
    "cluster": "cluster-size sweep",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate SIFT1M real-S3 sensitivity figures.")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out-dir", default=str(DATA_ROOT))
    ap.add_argument("--stem", default="figure_sift1m_real_s3_sensitivity")
    return ap.parse_args()


def _load_settings(data_root: Path) -> list[dict]:
    manifest = json.loads((data_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    out: list[dict] = []
    for spec in manifest.get("settings", []):
        setting_dir = data_root / "settings" / str(spec["setting_id"])
        summary_path = setting_dir / "setting_summary.json"
        agg_path = setting_dir / "aggregated_results.json"
        if not summary_path.exists() or not agg_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = json.loads(agg_path.read_text(encoding="utf-8"))["rows"]

        by_method_and_k: dict[tuple[str, int | None], dict] = {}
        for row in rows:
            by_method_and_k[(str(row.get("method")), row.get("budget_k"))] = row

        budget_ks = list(spec["budget_ks"])

        def row(method: str, budget_k: int | None = None) -> dict | None:
            return by_method_and_k.get((method, budget_k))

        warp_candidates = [row("warp_reorder", budget_ks[0]), row("warp_reorder", budget_ks[1])]
        warp_candidates = [r for r in warp_candidates if r is not None]
        warp_k1 = row("warp_reorder", budget_ks[0])
        warp_k2 = row("warp_reorder", budget_ks[1])

        out.append(
            {
                "setting_id": str(spec["setting_id"]),
                "panel": str(spec["panel"]),
                "setting_label": str(spec["setting_label"]),
                "nlist": int(spec["nlist"]),
                "nprobe": int(spec["nprobe"]),
                "budget_ks": budget_ks,
                "recall_at_10": float(summary["recall_at_10"]),
                "separate": row("separate_unbounded"),
                "lance_reorder": row("lance_native_reorder"),
                "warp_k1": warp_k1,
                "warp_k2": warp_k2,
            }
        )
    out.sort(
        key=lambda row: (
            PANEL_ORDER.index(row["panel"]),
            row["nprobe"] if row["panel"] == "nprobe" else row["nlist"],
        )
    )
    return out


def _panel_series(rows: list[dict], panel: str) -> dict:
    chosen = [r for r in rows if r["panel"] == panel]
    x = [r["nprobe"] if panel == "nprobe" else r["nlist"] for r in chosen]
    labels = [str(v) for v in x]
    return {
        "x": list(range(len(x))),
        "labels": labels,
        "recall": [100.0 * r["recall_at_10"] for r in chosen],
        "separate_qps": [float(r["separate"]["qps"]) for r in chosen],
        "lance_qps": [float(r["lance_reorder"]["qps"]) for r in chosen],
        "warp_k1_qps": [float(r["warp_k1"]["qps"]) for r in chosen],
        "warp_k2_qps": [float(r["warp_k2"]["qps"]) for r in chosen],
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.stem).strip() or "figure_sift1m_real_s3_sensitivity"
    rows = _load_settings(data_root)

    summary = {"figure": stem, "rows": rows}
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.0,
            "axes.labelsize": 12.2,
            "axes.titlesize": 12.2,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.4,
            "legend.fontsize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.titlepad": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.20, top=0.80, wspace=0.28)
    method_styles = {
        "Separate": dict(color="#4C78A8", hatch=None),
        "Lance + reorder": dict(color="#9EC1E6", hatch=None),
        "WaRP + reorder ($K\\leq nprobe/4$)": dict(color="#F7C66B", hatch="//"),
        "WaRP + reorder ($K\\leq nprobe/2$)": dict(color="#F28E2B", hatch="xx"),
    }
    width = 0.18

    for col, panel in enumerate(PANEL_ORDER):
        s = _panel_series(rows, panel)
        ax_qps = axes[col]
        x = np.arange(len(s["x"]), dtype=float)
        x_sep = x - 1.5 * width
        x_lance = x - 0.5 * width
        x_warp_k1 = x + 0.5 * width
        x_warp_k2 = x + 1.5 * width

        ax_qps.bar(
            x_sep,
            s["separate_qps"],
            width=width,
            label="Separate",
            color=method_styles["Separate"]["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        ax_qps.bar(
            x_lance,
            s["lance_qps"],
            width=width,
            label="Lance + reorder",
            color=method_styles["Lance + reorder"]["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        ax_qps.bar(
            x_warp_k1,
            s["warp_k1_qps"],
            width=width,
            label="WaRP + reorder, $K\\leq nprobe/4$",
            color=method_styles["WaRP + reorder ($K\\leq nprobe/4$)"]["color"],
            edgecolor="#C46A00",
            linewidth=0.9,
            hatch=method_styles["WaRP + reorder ($K\\leq nprobe/4$)"]["hatch"],
        )
        ax_qps.bar(
            x_warp_k2,
            s["warp_k2_qps"],
            width=width,
            label="WaRP + reorder, $K\\leq nprobe/2$",
            color=method_styles["WaRP + reorder ($K\\leq nprobe/2$)"]["color"],
            edgecolor="#A65100",
            linewidth=0.9,
            hatch=method_styles["WaRP + reorder ($K\\leq nprobe/2$)"]["hatch"],
        )
        ax_qps.set_title(PANEL_TITLES[panel], pad=4)
        if col == 0:
            ax_qps.set_ylabel("QPS")
        else:
            ax_qps.set_ylabel("")
        ax_qps.set_xticks(x)
        ax_qps.set_xticklabels(s["labels"])
        ax_qps.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.35)
        ax_qps.set_axisbelow(True)
        ax_qps.margins(x=0.08)

        # show recall on a light twin axis in the top row
        ax_r = ax_qps.twinx()
        ax_r.plot(x, s["recall"], color="#8e44ad", marker="^", markersize=5.5, linewidth=1.6, linestyle="--", label="Recall@10")
        ax_r.set_ylim(89.5, 100.5)
        ax_r.yaxis.set_major_locator(FixedLocator([90.0, 95.0, 100.0]))
        ax_r.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        if col == len(PANEL_ORDER) - 1:
            ax_r.set_ylabel("Recall (%)", color="#8e44ad")
            ax_r.tick_params(axis="y", colors="#8e44ad", labelsize=10.0)
        else:
            ax_r.set_ylabel("")
            ax_r.set_yticks([])
            ax_r.tick_params(axis="y", right=False, labelright=False)
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.spines["right"].set_linewidth(1.0)

        ax_qps.set_xlabel("nprobe" if panel == "nprobe" else "nlist")

    handles1, labels1 = axes[0].get_legend_handles_labels()
    twin = axes[0].figure.axes[2]
    handles2, labels2 = twin.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.5,
        columnspacing=1.1,
        handletextpad=0.6,
    )

    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] figure={out_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
