"""Pinned production contracts for the paper experiments.

This module is deliberately data-only.  Runners and preparation tools import
the same records so that the public workflow cannot silently drift from the
campaigns that produced the paper evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


LANCE_BASE_COMMIT: Final[str] = "c768d004506231f4f5d81f61b9cbc55ccfc904f3"
LANCE_EXPORTED_SOURCE_SHA256: Final[str] = (
    "4076f9aa657084a8c871ebef0c6c0337e9f9d3751b3f2d2c4d98192347edadd5"
)
LAYOUT_SEED: Final[int] = 20260811
FIGURE11_PAPER_MSONG_INDEX_UUID: Final[str] = "0a564119-434e-4ce4-b723-228578916085"
FIGURE11_PAPER_MSONG_OBJECT_ETAG: Final[str] = '"b22bd56cb48146c2427802adf29ddcdb-12"'
FIGURE11_PAPER_MSONG_OBJECT_SIZE: Final[int] = 62_565_067


@dataclass(frozen=True)
class Campaign:
    id: str
    measured_queries: int
    warmup_queries: int
    repeats: int
    query_concurrency: int
    io_limit: int
    description: str


@dataclass(frozen=True)
class MatchedDataset:
    key: str
    rows: int
    dimension: int
    nlist: int
    num_sub_vectors: int
    ordinal: int
    random_object: str
    reordered_object: str
    nprobe: int
    refine_factor: int
    source_key: str | None = None
    expected_query_sha256: str = ""
    centroid_preparation: str = "legacy_random_source"
    random_seed: int = LAYOUT_SEED

    @property
    def compression(self) -> int:
        return 4 * self.dimension // self.num_sub_vectors

    @property
    def local_key(self) -> str:
        """Public-array directory used to build this index contract."""
        return self.source_key or self.key


PAPER_CAMPAIGNS: Final[dict[str, Campaign]] = {
    "figure7-main": Campaign(
        "figure7-main", 10_000, 200, 3, 32, 2048,
        "Figure 7 cache-off seven-dataset real-Lance comparison",
    ),
    "figure7-cache": Campaign(
        "figure7-cache", 10_000, 200, 3, 32, 2048,
        "Figure 7 bounded-cache SIFT100M comparison",
    ),
    "figure8-query-concurrency": Campaign(
        "figure8-query-concurrency", 10_000, 200, 3, 32, 2048,
        "Figure 8 SIFT100M query-concurrency sweep",
    ),
    "figure8-io-limit": Campaign(
        "figure8-io-limit", 1_000, 200, 3, 32, 2048,
        "Figure 8 SIFT100M I/O-limit sweeps",
    ),
    "figure11-sift-transfer": Campaign(
        "figure11-sift-transfer", 200, 20, 3, 4, 64,
        "Figure 11 SIFT1M workload-shift transfer",
    ),
    "figure11-msong-transfer": Campaign(
        "figure11-msong-transfer", 200, 20, 3, 4, 64,
        "Figure 11 held-out MSong cross-dataset transfer",
    ),
}


MATCHED_DATASETS: Final[dict[str, MatchedDataset]] = {
    "sift1m": MatchedDataset(
        "sift1m", 1_000_000, 128, 4096, 128, 0,
        "s23_sift1m_pq128_random.lance", "s23_sift1m_pq128_reorder.lance", 128, 1,
        expected_query_sha256="88e61604b7ad2efee410551fbf6fd800594f4008ac599e2f75fe5a58f569f7b9",
    ),
    "glove": MatchedDataset(
        "glove", 1_183_514, 100, 4096, 100, 1,
        "s23_glove_pq100_random.lance", "s23_glove_pq100_reorder.lance", 128, 1,
        expected_query_sha256="f5ff41135dd49293f958dc0df7c52a70fb8c725f850d856ccaa8af2e27c5c788",
    ),
    "openimage": MatchedDataset(
        "openimage", 1_000_000, 512, 4096, 512, 2,
        "s23_openimage_pq512_random.lance", "s23_openimage_pq512_reorder.lance", 128, 1,
        expected_query_sha256="79e09ab40c217727a34473ce3a638b96ee3cb24e8a904d4ed147d10cd7d6fdba",
    ),
    "msong": MatchedDataset(
        "msong", 992_272, 420, 4096, 420, 3,
        "s23_msong_pq420_random.lance", "s23_msong_pq420_reorder.lance", 128, 1,
        expected_query_sha256="ea146b3dc0ef1dafc2fccdad363e7e584ffff2919430c8ba4f625c47c4938409",
    ),
    "gist1m": MatchedDataset(
        "gist1m", 1_000_000, 960, 4096, 960, 4,
        "s22_gist1m_pq960_random.lance", "s22_gist1m_pq960_reorder.lance", 128, 1,
        expected_query_sha256="9c7b5a612125255850d750113795afb43c96944156d9c890f7a4dd9ca794a060",
    ),
    "sift100m": MatchedDataset(
        "sift100m", 100_000_000, 128, 65_536, 16, 5,
        "s21_sift100m_pq16_random.lance", "s21_sift100m_pq16_reorder.lance", 256, 8,
        expected_query_sha256="d1eebedbfde1665eea8d97a8e972c2218009e74b57c06b390a318e4516137e92",
        centroid_preparation="legacy_source_random_target",
        random_seed=LAYOUT_SEED,
    ),
    "deep100m": MatchedDataset(
        "deep100m", 100_000_000, 96, 65_536, 12, 6,
        "s21_deep100m_pq12_random.lance", "s21_deep100m_pq12_reorder.lance", 256, 8,
        expected_query_sha256="6f189a29ee40e11c406432167ca66509f9eb044b8eb91c3d6eeb1ad4e1960512",
        centroid_preparation="legacy_source_random_target",
        random_seed=LAYOUT_SEED,
    ),
}


# Figure 11 deliberately uses its historical M=60 MSong index geometry instead
# of the later Figure 7 four-byte-vector MSong geometry.  Keep it out of
# MATCHED_DATASETS so a Figure 7/8 preparation cannot silently substitute one
# object for the other.
FIGURE11_MATCHED_DATASETS: Final[dict[str, MatchedDataset]] = {
    "msong-figure11": MatchedDataset(
        "msong-figure11", 992_272, 420, 4096, 60, 7,
        "msong_pq_random.lance", "msong_pq_reorder.lance", 64, 5,
        source_key="msong",
        expected_query_sha256="ea146b3dc0ef1dafc2fccdad363e7e584ffff2919430c8ba4f625c47c4938409",
        centroid_preparation="reconstructed_source",
    ),
}


FIGURE11_REQUIRED_CAPABILITIES: Final[dict[str, tuple[str, ...]]] = {
    "strict_budget_real_s3": ("--only-method",),
    "full_pipeline_real_s3": ("--out-query-ranges-json",),
}
