#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "experiments" / "table-01-cluster-coarsening" / "data" / "latest"
PAPER_DIR = ROOT / "experiments" / "table-01-cluster-coarsening" / "data" / "paper"
DEFAULT_TUNING_JSON = DATA_DIR / "tuning.json"
DEFAULT_TUNING_CSV = DATA_DIR / "tuning_grid.csv"
DEFAULT_PLAN_MANIFEST = DATA_DIR / "manifest.json"
DEFAULT_SUMMARIZE_MANIFEST = PAPER_DIR / "manifest.json"
DEFAULT_SUMMARY_JSON = DATA_DIR / "summary.json"
DEFAULT_SUMMARY_CSV = DATA_DIR / "intro_cluster_coarsening_sanity_check.csv"
DEFAULT_BASE_CONFIG = ROOT / "experiments" / "table-01-cluster-coarsening" / "config" / "base.yaml"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _parse_int_list(raw: str) -> list[int]:
    values = sorted({int(tok.strip()) for tok in str(raw).split(",") if tok.strip()})
    if not values:
        raise SystemExit("expected a non-empty integer list")
    return values


def _subset_queries(queries: np.ndarray, gt: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, min(int(count), len(queries)))
    if count == len(queries):
        return np.asarray(queries, dtype=np.float32), np.asarray(gt)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(queries), size=count, replace=False))
    return np.asarray(queries[idx], dtype=np.float32), np.asarray(gt[idx])


def _load_dataset(dataset_name: str, query_count: int, query_seed: int, recall_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.datasets import get_dataset

    ds = get_dataset(dataset_name)
    ds.prepare()
    base = np.asarray(ds.get_base_vectors(), dtype=np.float32)
    queries = np.asarray(ds.get_queries(), dtype=np.float32)
    gt, _ = ds.get_groundtruth(k=max(100, recall_k))
    if gt is None:
        raise SystemExit(f"dataset={dataset_name} has no ground truth; cannot tune recall")
    queries, gt = _subset_queries(queries, np.asarray(gt), query_count, query_seed)
    return base, queries, gt


def _scan_config(base: np.ndarray, queries: np.ndarray, gt: np.ndarray, *, nlist: int, nprobes: list[int], train_size: int | None, recall_k: int) -> list[dict[str, Any]]:
    from src.index_builder import IVFIndexBuilder
    from src.metrics import compute_recall

    builder = IVFIndexBuilder(dimension=int(base.shape[1]), num_clusters=int(nlist))
    builder.build(base, train_size=train_size)
    inv = builder.extract_inverted_lists()
    sizes = {int(cid): int(len(vec_ids)) for cid, vec_ids in inv.items()}
    vector_bytes = int(base.shape[1]) * 4
    rows: list[dict[str, Any]] = []
    for nprobe in nprobes:
        _dist, indices = builder.search(queries, k=recall_k, nprobe=int(nprobe))
        recall = float(compute_recall(indices, gt, k=recall_k))
        cluster_ids = [builder.find_nearest_clusters(q, int(nprobe)) for q in queries]
        candidates = [sum(sizes.get(int(cid), 0) for cid in probe_ids) for probe_ids in cluster_ids]
        bytes_per_query = [int(c) * int(vector_bytes) for c in candidates]
        rows.append(
            {
                "nlist": int(nlist),
                "nprobe": int(nprobe),
                "recall_at_k": recall,
                "avg_requests_per_query": float(nprobe),
                "avg_candidates_per_query": float(np.mean(candidates)) if candidates else float("nan"),
                "avg_bytes_per_query": float(np.mean(bytes_per_query)) if bytes_per_query else float("nan"),
                "avg_cluster_size": float(np.mean(list(sizes.values()))) if sizes else float("nan"),
            }
        )
    return rows


def _pick_match(rows: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    scored = sorted(rows, key=lambda row: (abs(float(row["recall_at_k"]) - float(target_recall)), int(row["nprobe"])))
    best = deepcopy(scored[0])
    best["target_recall"] = float(target_recall)
    best["delta_to_target"] = abs(float(best["recall_at_k"]) - float(target_recall))
    return best


def cmd_tune(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base, queries, gt = _load_dataset(args.dataset, args.queries, args.query_seed, args.recall_k)
    fine_rows = _scan_config(
        base,
        queries,
        gt,
        nlist=args.fine_nlist,
        nprobes=_parse_int_list(args.fine_nprobes),
        train_size=args.train_size,
        recall_k=args.recall_k,
    )
    coarse_rows = _scan_config(
        base,
        queries,
        gt,
        nlist=args.coarse_nlist,
        nprobes=_parse_int_list(args.coarse_nprobes),
        train_size=args.train_size,
        recall_k=args.recall_k,
    )
    target_recall = float(args.target_recall) if args.target_recall is not None else max(float(row["recall_at_k"]) for row in fine_rows)
    doc = {
        "experiment": "intro_cluster_coarsening_sanity_check",
        "dataset": args.dataset,
        "recall_k": int(args.recall_k),
        "query_count": int(args.queries),
        "query_seed": int(args.query_seed),
        "target_recall": float(target_recall),
        "fine": _pick_match(fine_rows, target_recall),
        "coarse": _pick_match(coarse_rows, target_recall),
        "fine_grid": fine_rows,
        "coarse_grid": coarse_rows,
    }
    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "regime",
                "nlist",
                "nprobe",
                "recall_at_k",
                "avg_requests_per_query",
                "avg_bytes_per_query",
                "avg_candidates_per_query",
                "avg_cluster_size",
            ],
        )
        writer.writeheader()
        for regime, rows in (("fine", fine_rows), ("coarse", coarse_rows)):
            for row in rows:
                writer.writerow({"regime": regime, **row})
    print(f"[ok] tuning_json={out_json}")
    print(f"[ok] tuning_csv={out_csv}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required for config generation")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    if yaml is None:
        raise SystemExit("PyYAML is required for config generation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def cmd_plan(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tuning_path = Path(args.tuning_json).resolve()
    if not tuning_path.exists():
        raise SystemExit("plan expects a prior tuning result")
    tuned = json.loads(tuning_path.read_text(encoding="utf-8"))
    manifest = {
        "experiment": "intro_cluster_coarsening_sanity_check",
        "dataset": args.dataset,
        "recall_k": int(tuned["recall_k"]),
        "target_recall": float(tuned["target_recall"]),
        "rows": [
            {
                "label": "fine_grained",
                "config": "Fine-grained",
                "dataset": args.dataset,
                "nlist": int(args.fine_nlist),
                "nprobe": int(tuned["fine"]["nprobe"]),
                "avg_cluster_size": _safe_float(tuned["fine"].get("avg_cluster_size")),
                "recall_at_k": float(tuned["fine"]["recall_at_k"]),
                "requests_per_query": _safe_float(tuned["fine"].get("avg_requests_per_query")) or float(tuned["fine"]["nprobe"]),
                "bytes_per_query": _safe_float(tuned["fine"].get("avg_bytes_per_query")),
                "candidates_per_query": float(tuned["fine"]["avg_candidates_per_query"]),
                "run_dir": None,
            },
            {
                "label": "coarse_grained",
                "config": "Coarse-grained",
                "dataset": args.dataset,
                "nlist": int(args.coarse_nlist),
                "nprobe": int(tuned["coarse"]["nprobe"]),
                "avg_cluster_size": _safe_float(tuned["coarse"].get("avg_cluster_size")),
                "recall_at_k": float(tuned["coarse"]["recall_at_k"]),
                "requests_per_query": _safe_float(tuned["coarse"].get("avg_requests_per_query")) or float(tuned["coarse"]["nprobe"]),
                "bytes_per_query": _safe_float(tuned["coarse"].get("avg_bytes_per_query")),
                "candidates_per_query": float(tuned["coarse"]["avg_candidates_per_query"]),
                "run_dir": None,
            },
        ],
    }
    if args.emit_configs:
        base_cfg = _load_yaml(Path(args.base_config).resolve())
        out_dir = DATA_DIR / "generated_configs"
        for row in manifest["rows"]:
            cfg = deepcopy(base_cfg)
            cfg.setdefault("dataset", {})
            cfg.setdefault("index", {})
            cfg["dataset"]["name"] = row["dataset"]
            cfg["index"]["num_clusters"] = int(row["nlist"])
            cfg["index"]["nprobe"] = int(row["nprobe"])
            cfg_name = f"{row['label']}_{row['dataset']}_nlist{row['nlist']}_nprobe{row['nprobe']}.yaml"
            cfg_path = out_dir / cfg_name
            _write_yaml(cfg_path, cfg)
            row["generated_config"] = str(cfg_path.relative_to(ROOT))
    out_manifest = Path(args.out_manifest).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] manifest={out_manifest}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_row(stage5_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in stage5_rows if str(row.get("status", "")) == "ok"]
    if not ok_rows:
        raise FileNotFoundError("no ok rows found in stage5/s3_e2e_benchmark.json")
    ok_rows.sort(
        key=lambda row: (
            -float(row.get("coverage", 0.0) or 0.0),
            float(row.get("avg_num_requests", float("inf")) or float("inf")),
        )
    )
    return ok_rows[0]


def cmd_summarize(args: argparse.Namespace) -> int:
    manifest = _read_json(Path(args.manifest).resolve())
    rows_out: list[dict[str, Any]] = []
    for row in manifest.get("rows", []):
        run_dir_raw = row.get("run_dir")
        stage2 = None
        selected = None
        run_dir = None
        if run_dir_raw:
            run_dir = Path(str(run_dir_raw))
            if not run_dir.is_absolute():
                run_dir = (ROOT / run_dir).resolve()
            stage2 = _read_json(run_dir / "stage2" / "index_stats.json")
            stage5 = _read_json(run_dir / "stage5" / "s3_e2e_benchmark.json")
            selected = _pick_row(list(stage5.get("rows", [])))
        bytes_per_query = (
            _safe_float(selected.get("avg_total_bytes"))
            if selected is not None
            else _safe_float(row.get("bytes_per_query"))
        )
        requests_per_query = (
            _safe_float(selected.get("avg_num_requests"))
            if selected is not None
            else _safe_float(row.get("requests_per_query"))
        )
        coverage = _safe_float(selected.get("coverage")) if selected is not None else 1.0
        p95_latency_ms = _safe_float(selected.get("p95_latency_ms")) if selected is not None else None
        rows_out.append(
            {
                "label": str(row["label"]),
                "config": str(row["config"]),
                "nlist": int(row["nlist"]),
                "avg_cluster_size": (
                    _safe_float((stage2.get("cluster_sizes") or {}).get("mean"))
                    if isinstance(stage2, dict)
                    else _safe_float(row.get("avg_cluster_size"))
                ),
                "nprobe": int(row["nprobe"]),
                "recall_at_k": _safe_float(row.get("recall_at_k")),
                "requests_per_query": requests_per_query,
                "bytes_per_query": bytes_per_query,
                "bytes_per_query_mb": (float(bytes_per_query) / (1024.0 * 1024.0)) if bytes_per_query is not None else None,
                "candidates_per_query": _safe_float(row.get("candidates_per_query")),
                "p95_latency_ms": p95_latency_ms,
                "coverage": coverage,
                "run_dir": str(run_dir) if run_dir is not None else None,
            }
        )
    doc = {
        "experiment": manifest.get("experiment"),
        "dataset": manifest.get("dataset"),
        "recall_k": manifest.get("recall_k"),
        "target_recall": manifest.get("target_recall"),
        "rows": rows_out,
    }
    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    pd_fields = [
        "label",
        "config",
        "nlist",
        "avg_cluster_size",
        "nprobe",
        "recall_at_k",
        "requests_per_query",
        "bytes_per_query",
        "bytes_per_query_mb",
        "candidates_per_query",
        "p95_latency_ms",
        "coverage",
        "run_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pd_fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[ok] summary_json={out_json}")
    print(f"[ok] summary_csv={out_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tiny intro-side coarse-cluster sanity check.")
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Print the planned summary output paths without running any stage.",
    )
    sub = ap.add_subparsers(dest="cmd", required=False)

    tune = sub.add_parser("tune")
    tune.add_argument("--dataset", default="sift1m")
    tune.add_argument("--fine-nlist", type=int, required=True)
    tune.add_argument("--coarse-nlist", type=int, required=True)
    tune.add_argument("--fine-nprobes", required=True)
    tune.add_argument("--coarse-nprobes", required=True)
    tune.add_argument("--queries", type=int, default=300)
    tune.add_argument("--query-seed", type=int, default=12345)
    tune.add_argument("--train-size", type=int, default=100000)
    tune.add_argument("--recall-k", type=int, default=10)
    tune.add_argument("--target-recall", type=float, default=None)
    tune.add_argument("--out-json", default=str(DEFAULT_TUNING_JSON))
    tune.add_argument("--out-csv", default=str(DEFAULT_TUNING_CSV))
    tune.set_defaults(func=cmd_tune)

    plan = sub.add_parser("plan")
    plan.add_argument("--dataset", default="sift1m")
    plan.add_argument("--fine-nlist", type=int, required=True)
    plan.add_argument("--coarse-nlist", type=int, required=True)
    plan.add_argument("--tuning-json", default=str(DEFAULT_TUNING_JSON))
    plan.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    plan.add_argument("--emit-configs", action="store_true")
    plan.add_argument("--out-manifest", default=str(DEFAULT_PLAN_MANIFEST))
    plan.set_defaults(func=cmd_plan)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--manifest", default=str(DEFAULT_SUMMARIZE_MANIFEST))
    summarize.add_argument("--out-json", default=str(DEFAULT_SUMMARY_JSON))
    summarize.add_argument("--out-csv", default=str(DEFAULT_SUMMARY_CSV))
    summarize.set_defaults(func=cmd_summarize)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "check_only", False):
        print(f"[plan] summary_json -> {DEFAULT_SUMMARY_JSON}")
        print(f"[plan] summary_csv -> {DEFAULT_SUMMARY_CSV}")
        print(f"[plan] manifest (input) -> {DEFAULT_SUMMARIZE_MANIFEST}")
        return 0
    if not getattr(args, "cmd", None):
        build_parser().error("the following arguments are required: cmd")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
