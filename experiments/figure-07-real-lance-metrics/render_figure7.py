#!/usr/bin/env python3
"""Render paper Figure 7 with SIFT100M next to the cache groups."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "render_figure7_base.py"
SPEC = importlib.util.spec_from_file_location("figure7_normalized_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load shared Figure 7 renderer: {BASE_SCRIPT}")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

# All numerical and normalization logic stays in the hash-validating base
# renderer. This review direction changes only category order and visual theme.
renderer.OUT = HERE / "figure_real_lance_metrics"
renderer.WORKLOADS = (
    ("sift1m", "SIFT1M", "dataset"),
    ("glove", "GloVe", "dataset"),
    ("openimage", "Open\nImage", "dataset"),
    ("msong", "MSong", "dataset"),
    ("gist1m", "GIST1M", "dataset"),
    ("deep100m", "DEEP\n100M", "dataset"),
    ("sift100m", "SIFT\n100M", "dataset"),
    ("sift100m_cache10", "10%", "cache"),
    ("sift100m_cache50", "50%", "cache"),
)
renderer.POSITIONS = np.asarray(
    (0, 0.88, 1.76, 2.64, 3.52, 4.40, 5.28, 6.16, 7.04),
    dtype=float,
)
renderer.BAR_WIDTH = 0.22
renderer.FIGSIZE = (7.2, 2.44)
renderer.BOTTOM_MARGIN = 0.095
renderer.SUBPLOT_LEFT = 0.050
renderer.SUBPLOT_RIGHT = 0.999
renderer.SUBPLOT_WSPACE = 0.18

# Match Figure 9's paper-scale typography while keeping the dense value and
# configuration annotations legible. Explicit point sizes are slightly larger
# than the previous Figure 7 export; line weights remain unchanged.
renderer.FONT_SIZE = 7.1
renderer.AXES_LABEL_SIZE = 7.1
renderer.AXES_TITLE_SIZE = 7.1
renderer.LEGEND_SIZE = 6.2
renderer.XTICK_SIZE = 5.7
renderer.YTICK_SIZE = 5.7
renderer.REGION_LABEL_SIZE = 4.8
renderer.CACHE_LABEL_SIZE = 4.9
renderer.VALUE_LABEL_SIZE = 4.6
renderer.CACHE_BAND = (5.72, 7.49)
renderer.CACHE_SEPARATOR = 5.72
renderer.CACHE_CENTER = 6.60
renderer.X_LIMITS = (-0.45, 7.49)
renderer.METHODS = (
    ("stock_lance", "Lance", "#a6bddb"),
    ("warp", "WaRP", "#74a9cf"),
    ("warp_reorder", "WaRP + Reorder", "#2b8cbe"),
)
renderer.CACHE_FILL = "#edf4f8"
renderer.CACHE_EDGE = "#5a8eaa"
renderer.CACHE_LABEL = "#356f90"
renderer.LEGEND_EDGE = "#b8b8b8"
renderer.SVG_HASHSALT = "warp-figure7-normalized-blue"
renderer.PALETTE_FAMILY = "paper blue"
renderer.PAPER_INTEGRATION = True
renderer.STATUS = "paper-frozen"
renderer.CONFIG_BANDS = (
    (-0.45, 3.96, "#f3f7fb", "#5e7788", 1.755, "IVF-PQ 4×  ·  nlist=4,096"),
    (3.96, 5.72, "#e8f1f7", "#356f90", 4.84, "IVF-PQ 32×  ·  nlist=65,536"),
)
renderer.PANELS = (
    ("(a) Throughput speedup", "qps_mean", "qps_sample_sd", False, (0.0, 5.50), (0, 1, 2, 3, 4, 5)),
    ("(b) p95 latency reduction", "p95_ms_mean", "p95_ms_sample_sd", True, (0.0, 6.25), (0, 1, 2, 3, 4, 5, 6)),
    ("(c) p99 latency reduction", "p99_ms_mean", "p99_ms_sample_sd", True, (0.0, 7.25), (0, 1, 2, 3, 4, 5, 6, 7)),
    ("(d) Number of GETs reduction", "get_per_q_mean", "get_per_q_sample_sd", True, (0.0, 7.25), (0, 1, 2, 3, 4, 5, 6, 7)),
)


if __name__ == "__main__":
    renderer.main()
