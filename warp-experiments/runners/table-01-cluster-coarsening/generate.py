#!/usr/bin/env python3
"""Generate the introduction's recall-matched clustering table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT_DIR = ROOT / "experiments" / "table-01-cluster-coarsening"
DEFAULT_INPUT = ARTIFACT_DIR / "data" / "latest" / "summary.json"
DEFAULT_CSV = ARTIFACT_DIR / "intro_cluster_coarsening_sanity_check.csv"
DEFAULT_TEX = ARTIFACT_DIR / "intro_cluster_coarsening_sanity_check.tex"

ROW_ORDER = ("fine_grained", "coarse_grained")
DISPLAY_NAMES = {
    "fine_grained": "Fine clustering",
    "coarse_grained": "Coarse clustering",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out-tex", default=str(DEFAULT_TEX))
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("recall_k", -1)) != 10:
        raise ValueError("the paper table requires recall_k=10")
    rows = {str(row.get("label")): row for row in payload.get("rows", [])}
    if set(rows) != set(ROW_ORDER):
        raise ValueError(
            f"expected exactly {ROW_ORDER}, got {tuple(sorted(rows))}"
        )
    return [rows[label] for label in ROW_ORDER]


def _candidate_display(value: Any) -> str:
    return f"{float(value) / 1000.0:.1f}K"


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "setting": DISPLAY_NAMES[str(row["label"])],
            "nlist": int(row["nlist"]),
            "nprobe": int(row["nprobe"]),
            "recall_at_10": f'{float(row["recall_at_k"]):.4f}',
            "requests_per_query": f'{float(row["requests_per_query"]):.0f}',
            "candidates_per_query": f'{float(row["candidates_per_query"]):.1f}',
        }
        for row in rows
    ]


def _render_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!tbp]",
        r"\caption{\textbf{Recall-Matched Comparison of Fine vs.\ Coarse Clustering on SIFT1M.} Coarser clustering reduces requests but enlarges the candidate set, raising per-query compute.}",
        r"\label{tab:intro-coarse-cluster}",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Setting & Lists & Probes & Recall@10 & Req./q & Cand./q \\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f'{DISPLAY_NAMES[str(row["label"])]} & {int(row["nlist"])} & '
            f'{int(row["nprobe"])} & {float(row["recall_at_k"]):.4f} & '
            f'{float(row["requests_per_query"]):.0f} & '
            f'{_candidate_display(row["candidates_per_query"])} \\\\'
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_tex = Path(args.out_tex).resolve()
    rows = _load_rows(input_json)
    csv_rows = _csv_rows(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(csv_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    out_tex.write_text(_render_latex(rows), encoding="utf-8")

    print(f"[ok] csv={out_csv}")
    print(f"[ok] tex={out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
