import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "rust" / "tests" / "data"
SIFT_FIXTURES = (
    ("sift_paper_4096x128", "none"),
    ("sift_paper_pca_4096x128", "pca"),
)
SIFT1M_BASE_SHA256 = "4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d"
SIFT1M_QUERIES_SHA256 = "88e61604b7ad2efee410551fbf6fd800594f4008ac599e2f75fe5a58f569f7b9"


class SiftFixtureManifestTests(unittest.TestCase):
    def _load(self, directory: str) -> dict:
        path = DATA / directory / "fixtures_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_sift_manifests_record_the_complete_input_contract(self) -> None:
        manifests = []
        for directory, reorder in SIFT_FIXTURES:
            manifest = self._load(directory)
            manifests.append(manifest)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["dataset"], "sift1m")
            self.assertEqual(manifest["n"], 1_000_000)
            self.assertEqual(manifest["d"], 128)
            self.assertEqual(manifest["index_type"], "faiss.IndexIVFFlat")
            self.assertEqual(manifest["metric"], "L2")
            self.assertEqual(manifest["train_size"], 200_000)
            self.assertEqual(
                manifest["clustering"],
                {
                    "seed": 42,
                    "iterations": 10,
                    "restarts": 1,
                    "max_points_per_centroid": 256,
                    "min_points_per_centroid": 39,
                    "faiss_threads": 16,
                },
            )
            self.assertEqual(manifest["nlist"], 4096)
            self.assertEqual(manifest["nprobe"], 128)
            self.assertEqual(manifest["num_queries"], 32)
            self.assertEqual(manifest["cluster_bytes"], 512)
            self.assertEqual(manifest["total_blob_bytes"], 512_000_000)
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["reorder"], reorder)
            self.assertTrue(manifest["toolchain"]["faiss"])
            self.assertTrue(manifest["toolchain"]["numpy"])
            self.assertEqual(manifest["source_arrays"]["base_file"], "base.npy")
            self.assertEqual(manifest["source_arrays"]["queries_file"], "queries.npy")
            self.assertEqual(
                manifest["source_arrays"]["base_sha256"], SIFT1M_BASE_SHA256
            )
            self.assertEqual(
                manifest["source_arrays"]["queries_sha256"], SIFT1M_QUERIES_SHA256
            )

        self.assertEqual(manifests[0]["source_arrays"], manifests[1]["source_arrays"])

    def test_sift_query_files_match_their_manifests(self) -> None:
        for directory, _ in SIFT_FIXTURES:
            fixture_dir = DATA / directory
            manifest = self._load(directory)
            self.assertEqual(len(manifest["queries"]), 32)
            for entry in manifest["queries"]:
                self.assertRegex(entry["file"], r"^query_[0-9]{4}\.json$")
                requests = json.loads(
                    (fixture_dir / entry["file"]).read_text(encoding="utf-8")
                )
                self.assertEqual(len(requests), 128)
                self.assertEqual(entry["cluster_count"], len(requests))
                offsets = [request["offset"] for request in requests]
                self.assertEqual(offsets, sorted(offsets))
                self.assertTrue(all(request["length"] > 0 for request in requests))
                self.assertEqual(entry["first_offset"], requests[0]["offset"])
                self.assertEqual(
                    entry["last_end"],
                    requests[-1]["offset"] + requests[-1]["length"],
                )

if __name__ == "__main__":
    unittest.main()
