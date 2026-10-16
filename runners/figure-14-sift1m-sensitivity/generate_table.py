#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT / "experiments" / "figure-14-sift1m-sensitivity" / "data" / "latest"
PANEL_ORDER = ["nprobe", "cluster"]
PANEL_TITLES = {
    "nprobe": "Panel A: nprobe sensitivity (nlist=4096)",
    "cluster": "Panel B: cluster-size sensitivity",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate the SIFT1M real-S3 sensitivity table.")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out-dir", default=str(DATA_ROOT))
    ap.add_argument("--stem", default="table_sift1m_real_s3_sensitivity")
    return ap.parse_args()


def _fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_pair(qps: float | None, p95: float | None) -> str:
    return f"{_fmt_num(qps, 1)} / {_fmt_num(p95, 1)}"


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

        row_by_method_and_k: dict[tuple[str, int | None], dict] = {}
        for row in rows:
            row_by_method_and_k[(str(row.get("method")), row.get("budget_k"))] = row

        budget_ks = list(spec["budget_ks"])

        def get_row(method: str, budget_k: int | None = None) -> dict | None:
            return row_by_method_and_k.get((method, budget_k))

        def pair(row: dict | None) -> tuple[float | None, float | None]:
            if row is None:
                return None, None
            return row.get("qps"), row.get("p95_latency_ms")

        out.append(
            {
                "setting_id": str(spec["setting_id"]),
                "panel": str(spec["panel"]),
                "setting_label": str(spec["setting_label"]),
                "nlist": int(spec["nlist"]),
                "nprobe": int(spec["nprobe"]),
                "budget_ks": budget_ks,
                "recall_at_10": float(summary["recall_at_10"]),
                "separate": pair(get_row("separate_unbounded")),
                "lance_reorder": pair(get_row("lance_native_reorder")),
                "warp_k1": pair(get_row("warp_reorder", budget_ks[0])),
                "warp_k2": pair(get_row("warp_reorder", budget_ks[1])),
            }
        )
    out.sort(
        key=lambda row: (
            PANEL_ORDER.index(row["panel"]),
            row["nprobe"] if row["panel"] == "nprobe" else row["nlist"],
        )
    )
    return out


def _render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"  \toprule",
        r"  Setting & Recall@10 & Separate & Lance+r & WaRP+r ($K_1$) & WaRP+r ($K_2$) & Budget pair \\",
        r"  \midrule",
    ]
    current_panel = None
    for row in rows:
        if row["panel"] != current_panel:
            if current_panel is not None:
                lines.append(r"  \midrule")
            lines.append(rf"  \multicolumn{{7}}{{l}}{{\textbf{{{PANEL_TITLES[row['panel']]}}}}} \\")
            current_panel = row["panel"]
        budget_pair = rf"$K\leq {row['budget_ks'][0]}, {row['budget_ks'][1]}$"
        lines.append(
            "  {setting} & {recall} & {sep} & {lance} & {warp1} & {warp2} & {budget_pair} \\\\".format(
                setting=row["setting_label"],
                recall=f"{100.0 * row['recall_at_10']:.2f}\\%",
                sep=_fmt_pair(*row["separate"]),
                lance=_fmt_pair(*row["lance_reorder"]),
                warp1=_fmt_pair(*row["warp_k1"]),
                warp2=_fmt_pair(*row["warp_k2"]),
                budget_pair=budget_pair,
            )
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.stem).strip() or "table_sift1m_real_s3_sensitivity"
    rows = _load_settings(data_root)

    csv_lines = [
        "panel,setting_id,setting_label,nlist,nprobe,recall_at_10,budget_k1,budget_k2,separate_qps,separate_p95,lance_r_qps,lance_r_p95,warp_k1_qps,warp_k1_p95,warp_k2_qps,warp_k2_p95"
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    row["panel"],
                    row["setting_id"],
                    row["setting_label"],
                    str(row["nlist"]),
                    str(row["nprobe"]),
                    f"{row['recall_at_10']:.8f}",
                    str(row["budget_ks"][0]),
                    str(row["budget_ks"][1]),
                    "" if row["separate"][0] is None else f"{float(row['separate'][0]):.8f}",
                    "" if row["separate"][1] is None else f"{float(row['separate'][1]):.8f}",
                    "" if row["lance_reorder"][0] is None else f"{float(row['lance_reorder'][0]):.8f}",
                    "" if row["lance_reorder"][1] is None else f"{float(row['lance_reorder'][1]):.8f}",
                    "" if row["warp_k1"][0] is None else f"{float(row['warp_k1'][0]):.8f}",
                    "" if row["warp_k1"][1] is None else f"{float(row['warp_k1'][1]):.8f}",
                    "" if row["warp_k2"][0] is None else f"{float(row['warp_k2'][0]):.8f}",
                    "" if row["warp_k2"][1] is None else f"{float(row['warp_k2'][1]):.8f}",
                ]
            )
        )

    (out_dir / f"{stem}.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.tex").write_text(_render_latex(rows), encoding="utf-8")
    print(f"[ok] table={out_dir / (stem + '.tex')}")


if __name__ == "__main__":
    main()
