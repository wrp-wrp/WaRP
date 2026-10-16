#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterRange:
    offset: int
    size: int


def _generate_synth_clusters(file_size: int, nlist: int, cluster_bytes: int) -> list[ClusterRange]:
    if nlist <= 0:
        return []
    clamped_cluster = min(cluster_bytes, max(file_size, 1))
    if nlist == 1 or file_size <= clamped_cluster:
        return [ClusterRange(offset=0, size=clamped_cluster)]

    max_start = max(0, file_size - clamped_cluster)
    step = max(1, max_start // (nlist - 1))
    out: list[ClusterRange] = []
    for i in range(nlist):
        offset = i * step
        if offset > max_start:
            offset = max_start
        out.append(ClusterRange(offset=offset, size=clamped_cluster))
    return out


def _generate_queries(
    clusters: list[ClusterRange],
    nprobe: int,
    num_queries: int,
    seed: int,
) -> list[list[list[int]]]:
    rng = random.Random(seed)
    nprobe = min(nprobe, len(clusters))
    out: list[list[list[int]]] = []
    ids_base = list(range(len(clusters)))
    for _ in range(num_queries):
        ids = ids_base[:]
        rng.shuffle(ids)
        ids = ids[:nprobe]
        ids.sort(key=lambda idx: clusters[idx].offset)
        out.append([[clusters[idx].offset, clusters[idx].offset + clusters[idx].size] for idx in ids])
    return out


def _truncate_queries(queries: list[list[list[int]]], fanout: int) -> list[list[list[int]]]:
    return [q[:fanout] for q in queries]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate a fixed prefix-query pool for clean 1KB fanout sweeps."
    )
    ap.add_argument("--uri", required=True, help="Object URI for metadata in the query JSON.")
    ap.add_argument("--file-size-bytes", type=int, required=True, help="Exact object size in bytes.")
    ap.add_argument("--out-dir", required=True, help="Output directory for generated query JSONs.")
    ap.add_argument("--nlist", type=int, default=4096, help="Synthetic cluster count. Default: 4096.")
    ap.add_argument("--cluster-bytes", type=int, default=1024, help="Cluster size in bytes. Default: 1024.")
    ap.add_argument("--max-fanout", type=int, default=128, help="Maximum fanout / prefix length. Default: 128.")
    ap.add_argument("--num-queries", type=int, default=128, help="Measured query count. Default: 128.")
    ap.add_argument("--warmup-queries", type=int, default=16, help="Warmup query count. Default: 16.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed. Default: 42.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_fanout <= 0:
        raise SystemExit("--max-fanout must be > 0")
    if args.nlist < args.max_fanout:
        raise SystemExit("--nlist must be >= --max-fanout")

    clusters = _generate_synth_clusters(
        file_size=int(args.file_size_bytes),
        nlist=int(args.nlist),
        cluster_bytes=int(args.cluster_bytes),
    )
    warmup_full = _generate_queries(
        clusters,
        nprobe=int(args.max_fanout),
        num_queries=int(args.warmup_queries),
        seed=int(args.seed) + 7,
    )
    measured_full = _generate_queries(
        clusters,
        nprobe=int(args.max_fanout),
        num_queries=int(args.num_queries),
        seed=int(args.seed) + 17,
    )

    manifest = {
        "uri": args.uri,
        "file_size_bytes": int(args.file_size_bytes),
        "nlist": int(args.nlist),
        "cluster_bytes": int(args.cluster_bytes),
        "max_fanout": int(args.max_fanout),
        "num_queries": int(args.num_queries),
        "warmup_queries": int(args.warmup_queries),
        "seed": int(args.seed),
        "generated_files": [],
    }

    base_path = out_dir / "base_maxfanout_queries.json"
    base_payload = {
        "uri": args.uri,
        "nlist": int(args.nlist),
        "nprobe": int(args.max_fanout),
        "queries": measured_full,
        "warmup_queries": warmup_full,
    }
    base_path.write_text(json.dumps(base_payload, indent=2) + "\n", encoding="utf-8")

    for fanout in range(1, int(args.max_fanout) + 1):
        payload = {
            "uri": args.uri,
            "nlist": int(args.nlist),
            "nprobe": int(fanout),
            "queries": _truncate_queries(measured_full, fanout),
            "warmup_queries": _truncate_queries(warmup_full, fanout),
        }
        path = out_dir / f"fanout_{fanout:03d}_queries.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest["generated_files"].append(path.name)

    readme = [
        "Fixed prefix-query pool for clean 1KB fanout sweep",
        f"uri={args.uri}",
        f"file_size_bytes={int(args.file_size_bytes)}",
        f"nlist={int(args.nlist)}",
        f"cluster_bytes={int(args.cluster_bytes)}",
        f"max_fanout={int(args.max_fanout)}",
        f"num_queries={int(args.num_queries)}",
        f"warmup_queries={int(args.warmup_queries)}",
        f"seed={int(args.seed)}",
        "semantics=same measured/warmup query pool for all fanouts; fanout N uses the first N sorted ranges of each base query",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote fixed query pool to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
