from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock
import sys

import numpy as np

import runners.shared.full_pipeline_campaign as full_pipeline_campaign
import runners.shared.lance_build_receipt as lance_build_receipt
from runners.shared.full_pipeline_campaign import aggregate, index_identity
from runners.shared.matched_lance_datasets import (
    Preparation,
    _index_record,
    ensure_ground_truth,
    preparation_plan,
    production_centroid_layouts,
)
from artifact.campaigns import (
    FIGURE11_PAPER_MSONG_INDEX_UUID,
    FIGURE11_PAPER_MSONG_OBJECT_ETAG,
    FIGURE11_PAPER_MSONG_OBJECT_SIZE,
    LANCE_BASE_COMMIT,
    LANCE_EXPORTED_SOURCE_SHA256,
    MATCHED_DATASETS,
    PAPER_CAMPAIGNS,
)
from artifact.lance_toolchain import (
    _binary_has_flag as toolchain_binary_has_flag,
    companion_root_for_binary,
    figure11_build_commands,
    source_capability_errors,
    source_tree_sha256,
    validate_lance_source,
    verify_lance_history,
)
from artifact.prepare_dataset import (
    PREPARERS,
    _assemble_floatbin_prefix,
    prepare_glove,
    repeated_query_pool,
)


ROOT = Path(__file__).resolve().parents[1]


def _minimal_lance_source(root: Path) -> None:
    for directory in (".cargo", "protos", "rust/example/src", "python/python/lance"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative, contents in {
        "Cargo.toml": "[workspace]\n",
        "Cargo.lock": "lock-v1\n",
        "rust-toolchain.toml": '[toolchain]\nchannel = "stable"\n',
        ".cargo/config.toml": "[build]\n",
        "rust/example/src/lib.rs": "pub fn value() -> u8 { 1 }\n",
        "python/Cargo.toml": '[package]\nname = "lance"\n',
        "python/python/lance/__init__.py": "VALUE = 1\n",
    }.items():
        (root / relative).write_text(contents, encoding="utf-8")


class ProductionCampaignLockTests(unittest.TestCase):
    def test_paper_campaigns_and_lance_base_are_pinned(self) -> None:
        self.assertEqual(
            LANCE_BASE_COMMIT,
            "c768d004506231f4f5d81f61b9cbc55ccfc904f3",
        )
        self.assertEqual(
            set(PAPER_CAMPAIGNS),
            {
                "figure7-main",
                "figure7-cache",
                "figure8-query-concurrency",
                "figure8-io-limit",
                "figure11-sift-transfer",
                "figure11-msong-transfer",
            },
        )
        self.assertEqual(PAPER_CAMPAIGNS["figure7-main"].measured_queries, 10_000)
        self.assertEqual(PAPER_CAMPAIGNS["figure7-main"].repeats, 3)
        self.assertEqual(PAPER_CAMPAIGNS["figure8-io-limit"].measured_queries, 1_000)
        self.assertEqual(PAPER_CAMPAIGNS["figure8-io-limit"].warmup_queries, 200)
        self.assertEqual(PAPER_CAMPAIGNS["figure11-msong-transfer"].measured_queries, 200)

    def test_matched_dataset_plan_owns_exact_paper_object_names(self) -> None:
        plan = preparation_plan(
            ["sift1m", "sift100m"],
            data_root=Path("/datasets"),
            s3_root="s3://example/warp",
            manifest_root=Path("/manifests"),
        )
        self.assertEqual(
            [item.random_uri for item in plan],
            [
                "s3://example/warp/s23_sift1m_pq128_random.lance",
                "s3://example/warp/s21_sift100m_pq16_random.lance",
            ],
        )
        self.assertEqual(plan[0].base_path, Path("/datasets/sift1m/base.npy"))
        self.assertEqual(
            plan[0].spec.expected_query_sha256,
            "88e61604b7ad2efee410551fbf6fd800594f4008ac599e2f75fe5a58f569f7b9",
        )
        self.assertEqual(plan[1].manifest_path, Path("/manifests/sift100m/manifest.json"))
        self.assertEqual(plan[0].spec.random_seed, 20260811)
        self.assertEqual(plan[1].spec.random_seed, 20260811)

    def test_figure11_msong_uses_its_own_m60_index_contract(self) -> None:
        plan = preparation_plan(
            ["msong-figure11"],
            data_root=Path("/datasets"),
            s3_root="s3://example/warp",
            manifest_root=Path("/manifests"),
        )
        self.assertEqual(plan[0].base_path, Path("/datasets/msong/base.npy"))
        self.assertEqual(plan[0].spec.num_sub_vectors, 60)
        self.assertEqual(plan[0].spec.nprobe, 64)
        self.assertEqual(
            plan[0].reordered_uri,
            "s3://example/warp/msong_pq_reorder.lance",
        )

    def test_figure11_toolchain_is_a_pure_pinned_build_plan(self) -> None:
        root = Path("/lance")
        commands = figure11_build_commands(root, root / "target")
        rendered = [" ".join(command) for command in commands]
        self.assertEqual(len(commands), 2)
        self.assertIn("-p lance-io --example strict_budget_real_s3", rendered[0])
        self.assertIn("-p lance --example full_pipeline_real_s3", rendered[1])

    def test_figure11_source_capabilities_are_checked_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strict = root / "rust/lance-io/examples/strict_budget_real_s3.rs"
            trace = root / "rust/lance/examples/full_pipeline_real_s3.rs"
            strict.parent.mkdir(parents=True)
            trace.parent.mkdir(parents=True)
            strict.write_text("struct Args { only_method: Option<String> }", encoding="utf-8")
            trace.write_text('match flag { "--out-query-ranges-json" => {} }', encoding="utf-8")
            self.assertEqual(source_capability_errors(root), [])
            strict.write_text("struct Args {}", encoding="utf-8")
            self.assertIn("only_method", "\n".join(source_capability_errors(root)))

    def test_figure11_binary_capabilities_are_read_from_help(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "binary"
            binary.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Usage: binary --only-method'\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            self.assertTrue(toolchain_binary_has_flag(binary, "--only-method"))
            self.assertFalse(toolchain_binary_has_flag(binary, "--out-query-ranges-json"))

    def test_companion_checkout_must_be_at_the_exact_public_release(self) -> None:
        root = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="/lance\n",
            stderr="",
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=LANCE_BASE_COMMIT + "\n",
            stderr="",
        )
        clean = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with mock.patch(
            "artifact.lance_toolchain.subprocess.run",
            side_effect=(root, completed, clean, clean),
        ), mock.patch(
            "artifact.lance_toolchain.source_tree_sha256",
            return_value=LANCE_EXPORTED_SOURCE_SHA256,
        ):
            verify_lance_history(Path("/lance"))
        completed.stdout = "0" * 40 + "\n"
        with mock.patch(
            "artifact.lance_toolchain.subprocess.run",
            side_effect=(root, completed),
        ):
            with self.assertRaisesRegex(SystemExit, "not at the public paper release"):
                verify_lance_history(Path("/lance"))

    def test_export_nested_in_an_unrelated_git_repository_is_not_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            export = parent / "downloaded-lance"
            export.mkdir()
            subprocess.run(["git", "init", "-q", str(parent)], check=True)
            with mock.patch(
                "artifact.lance_toolchain.source_tree_sha256",
                return_value=LANCE_EXPORTED_SOURCE_SHA256,
            ):
                identity = validate_lance_source(export)
        self.assertEqual(identity["source_mode"], "exported-source")

    def test_an_exported_companion_is_verified_by_its_source_tree(self) -> None:
        unavailable = subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="not a git repository",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "artifact.lance_toolchain.subprocess.run",
            return_value=unavailable,
        ), mock.patch(
            "artifact.lance_toolchain.source_tree_sha256",
            return_value=LANCE_EXPORTED_SOURCE_SHA256,
        ):
            identity = validate_lance_source(Path(directory))
        self.assertEqual(identity["lance_commit"], LANCE_BASE_COMMIT)
        self.assertEqual(identity["source_mode"], "exported-source")
        self.assertEqual(
            identity["source_tree_sha256"],
            LANCE_EXPORTED_SOURCE_SHA256,
        )

    def test_an_exported_companion_with_changed_sources_is_rejected(self) -> None:
        unavailable = subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="not a git repository",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "artifact.lance_toolchain.subprocess.run",
            return_value=unavailable,
        ), mock.patch(
            "artifact.lance_toolchain.source_tree_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(SystemExit, "exported source tree"):
                validate_lance_source(Path(directory))

    def test_source_tree_fingerprint_changes_with_build_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rust/example/src").mkdir(parents=True)
            (root / "protos").mkdir()
            (root / "python").mkdir()
            (root / ".cargo").mkdir()
            (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("lock-v1\n", encoding="utf-8")
            (root / "rust-toolchain.toml").write_text(
                "[toolchain]\nchannel = \"stable\"\n",
                encoding="utf-8",
            )
            source = root / "rust/example/src/lib.rs"
            source.write_text("pub fn value() -> u8 { 1 }\n", encoding="utf-8")
            first = source_tree_sha256(root)
            source.write_text("pub fn value() -> u8 { 2 }\n", encoding="utf-8")
            self.assertNotEqual(source_tree_sha256(root), first)

            python_source = root / "python/lib.py"
            python_source.write_text("VALUE = 1\n", encoding="utf-8")
            with_python = source_tree_sha256(root)
            python_source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(source_tree_sha256(root), with_python)

    def test_editable_binding_preserves_exported_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _minimal_lance_source(root)
            expected = source_tree_sha256(root)
            with mock.patch("artifact.lance_toolchain.LANCE_EXPORTED_SOURCE_SHA256", expected):
                before = validate_lance_source(root)
                extension = root / "python/python/lance/lance.abi3.so"
                for contents in (b"compiled-binding-v1", b"compiled-binding-v2"):
                    extension.write_bytes(contents)
                    self.assertEqual(validate_lance_source(root), before)

    def test_editable_binding_does_not_hide_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _minimal_lance_source(root)
            (root / "python/python/lance/lance.abi3.so").write_bytes(b"compiled-binding")
            expected = source_tree_sha256(root)
            with mock.patch("artifact.lance_toolchain.LANCE_EXPORTED_SOURCE_SHA256", expected):
                for relative in (
                    "python/python/lance/__init__.py",
                    "rust/example/src/lib.rs",
                    "python/Cargo.toml",
                    ".cargo/config.toml",
                ):
                    with self.subTest(path=relative):
                        source = root / relative
                        original = source.read_bytes()
                        source.write_bytes(original + b"\nchanged\n")
                        with self.assertRaisesRegex(SystemExit, "exported source tree"):
                            validate_lance_source(root)
                        source.write_bytes(original)

    def test_editable_binding_exclusion_keeps_other_native_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _minimal_lance_source(root)
            expected = source_tree_sha256(root)
            with mock.patch("artifact.lance_toolchain.LANCE_EXPORTED_SOURCE_SHA256", expected):
                for relative in (
                    "python/python/lance/helper.abi3.so",
                    "python/python/lance/lance.abi3.so.config",
                    "rust/example/src/lance.abi3.so",
                ):
                    with self.subTest(path=relative):
                        native_input = root / relative
                        native_input.write_bytes(b"native-build-input")
                        with self.assertRaisesRegex(SystemExit, "exported source tree"):
                            validate_lance_source(root)
                        native_input.unlink()

    def test_source_tree_fingerprint_rejects_an_untracked_cargo_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rust").mkdir()
            (root / "protos").mkdir()
            (root / "python").mkdir()
            (root / ".cargo").mkdir()
            (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("lock\n", encoding="utf-8")
            (root / "rust-toolchain.toml").write_text("[toolchain]\n", encoding="utf-8")
            before = source_tree_sha256(root)
            (root / ".cargo/config.toml").write_text(
                '[build]\nrustflags = ["--cfg", "changed"]\n',
                encoding="utf-8",
            )
            self.assertNotEqual(source_tree_sha256(root), before)

    def test_source_fingerprint_ignores_only_the_anonymized_repository_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rust").mkdir()
            (root / "protos").mkdir()
            (root / "python").mkdir()
            (root / ".cargo").mkdir()
            (root / "Cargo.lock").write_text("lock\n", encoding="utf-8")
            (root / "rust-toolchain.toml").write_text("[toolchain]\n", encoding="utf-8")
            manifest = root / "Cargo.toml"
            manifest.write_text(
                '[workspace.package]\nrepository = "https://github.com/private/repo"\n',
                encoding="utf-8",
            )
            before = source_tree_sha256(root)
            manifest.write_text(
                '[workspace.package]\nrepository = "https://github.com/ANON-USER/repo"\n',
                encoding="utf-8",
            )
            self.assertEqual(source_tree_sha256(root), before)

    def test_binary_resolves_to_an_exported_companion_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("lock\n", encoding="utf-8")
            (root / "rust").mkdir()
            binary = root / "target/release/examples/full_pipeline_real_s3"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"binary")
            with mock.patch.dict(os.environ, {"VECIO_LANCE_ROOT": ""}):
                self.assertEqual(companion_root_for_binary(binary), root)

    def test_lance_python_setup_verifies_history_before_installing(self) -> None:
        source = (ROOT / "scripts/setup_lance_python.sh").read_text(encoding="utf-8")
        self.assertLess(
            source.index("verify_lance_base.py"),
            source.index("pip install"),
        )

    def test_missing_ground_truth_is_computed_from_public_arrays(self) -> None:
        class IndexFlatL2:
            def __init__(self, dimension: int):
                self.dimension = dimension
                self.base = np.empty((0, dimension), dtype=np.float32)

            def add(self, values: np.ndarray) -> None:
                self.base = np.concatenate([self.base, values], axis=0)

            def search(self, queries: np.ndarray, k: int):
                distances = np.sum(
                    (queries[:, np.newaxis, :] - self.base[np.newaxis, :, :]) ** 2,
                    axis=2,
                )
                ids = np.argsort(distances, axis=1, kind="stable")[:, :k]
                return np.take_along_axis(distances, ids, axis=1), ids.astype(np.int64)

        fake_faiss = types.SimpleNamespace(
            IndexFlatL2=IndexFlatL2,
            omp_set_num_threads=lambda _threads: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = np.asarray([[0.0], [2.0], [5.0], [9.0]], dtype=np.float32)
            queries = np.asarray([[1.0], [8.0]], dtype=np.float32)
            np.save(root / "base.npy", base)
            np.save(root / "queries.npy", queries)
            output = root / "gt_I.npy"
            with mock.patch.dict(sys.modules, {"faiss": fake_faiss}):
                ensure_ground_truth(
                    root / "base.npy",
                    root / "queries.npy",
                    output,
                    k=2,
                    threads=1,
                )
            self.assertEqual(np.load(output).tolist(), [[0, 1], [3, 2]])
            self.assertEqual(list(root.glob(".*gt_I.npy*.tmp")), [])

    def test_truncated_existing_ground_truth_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "base.npy", np.zeros((20, 2), dtype=np.float32))
            np.save(root / "queries.npy", np.zeros((4, 2), dtype=np.float32))
            np.save(root / "gt_I.npy", np.zeros((4, 1), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "ground-truth shape mismatch"):
                ensure_ground_truth(
                    root / "base.npy",
                    root / "queries.npy",
                    root / "gt_I.npy",
                    k=10,
                )

    def test_index_contract_reads_observed_pq_m(self) -> None:
        spec = MATCHED_DATASETS["sift1m"]
        item = Preparation(
            spec=spec,
            base_path=Path("/base.npy"),
            queries_path=Path("/queries.npy"),
            ground_truth_path=Path("/gt.npy"),
            random_uri="s3://bucket/random.lance",
            reordered_uri="s3://bucket/reordered.lance",
            manifest_path=Path("/manifest.json"),
        )

        class FakeDataset:
            version = 7

            def list_indices(self):
                return [{"name": "vector_idx", "type": "IVF_PQ"}]

            stats = types.SimpleNamespace(
                index_stats=lambda _name: {
                    "index_type": "IVF_PQ",
                    "num_indices": 1,
                    "num_indexed_rows": spec.rows,
                    "num_unindexed_rows": 0,
                    "num_unindexed_fragments": 0,
                    "indices": [
                        {
                            "uuid": "actual-index-uuid",
                            "index_type": "IVF_PQ",
                            "num_partitions": spec.nlist,
                            "sub_index": {
                                "index_type": "PQ",
                                "num_sub_vectors": 999,
                                "nbits": 8,
                            },
                        }
                    ],
                }
            )

        with self.assertRaisesRegex(ValueError, "num_sub_vectors"):
            _index_record(FakeDataset(), item, "random")

    def test_pca_is_applied_to_the_legacy_random_source_order(self) -> None:
        spec = MATCHED_DATASETS["sift1m"]
        trained = np.arange(spec.nlist * 2, dtype=np.float32).reshape(spec.nlist, 2)
        random_order = np.random.default_rng(
            [20260811, spec.ordinal]
        ).permutation(spec.nlist)
        observed: list[np.ndarray] = []

        def fake_pca(values: np.ndarray):
            observed.append(values.copy())
            return np.arange(len(values) - 1, -1, -1), {"method": "test"}

        with mock.patch(
            "runners.shared.matched_lance_datasets.pca_nd_zorder",
            side_effect=fake_pca,
        ):
            source, random_centroids, reordered, _, _ = production_centroid_layouts(
                trained, spec
            )
        expected_source = trained[random_order]
        self.assertTrue(np.array_equal(source, expected_source))
        self.assertTrue(np.array_equal(random_centroids, expected_source))
        self.assertTrue(np.array_equal(observed[0], expected_source))
        self.assertTrue(np.array_equal(reordered, expected_source[::-1]))

    def test_floatbin_prefix_is_atomically_rebuilt_from_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            output = root / "base.fbin"
            values = np.arange(12, dtype=np.float32)
            payload.write_bytes(values.tobytes())
            output.write_bytes(b"truncated")
            _assemble_floatbin_prefix(payload, output, rows=3, dimension=4)
            with output.open("rb") as stream:
                header = np.fromfile(stream, dtype=np.int32, count=2)
                observed = np.fromfile(stream, dtype=np.float32)
            self.assertEqual(header.tolist(), [3, 4])
            self.assertEqual(observed.tolist(), values.tolist())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_public_dataset_preparation_covers_the_seven_dataset_campaign(self) -> None:
        self.assertTrue(set(("sift1m", "glove", "openimage", "msong", "gist1m", "sift100m", "deep100m")).issubset(PREPARERS))
        self.assertEqual(
            inspect.signature(prepare_glove).parameters["target_rows"].default,
            1_183_514,
        )

    def test_gist_query_pool_repeats_every_public_query_uniformly(self) -> None:
        source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        queries, origin = repeated_query_pool(source, repeats=3)
        expected = np.concatenate(
            [np.random.default_rng([20260811, epoch]).permutation(2) for epoch in range(3)]
        )
        self.assertEqual(origin.tolist(), expected.tolist())
        self.assertEqual(queries.tolist(), source[expected].tolist())
        self.assertEqual(np.bincount(origin, minlength=2).tolist(), [3, 3])

    def test_gist_production_query_order_identity_is_locked(self) -> None:
        source = np.zeros((1_000, 1), dtype=np.float32)
        _, origin = repeated_query_pool(source, repeats=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_origin_10k.npy"
            np.save(path, origin)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "279ee47b547e4ef9986a2879c5322429bdd78274f73a5f6ed422afd48d0ffc4e",
        )


def load_runner(name: str):
    path = ROOT / "runners" / name / "run_data.py"
    spec = importlib.util.spec_from_file_location(f"test_runner_{name.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyFanoutRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner("real-s3-tiny-request-fanout-sanity")

    def test_companion_root_supplies_built_lance_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lance_root = Path(directory) / "lance"
            with mock.patch.dict(
                os.environ,
                {"VECIO_LANCE_ROOT": str(lance_root)},
                clear=False,
            ), mock.patch.object(sys, "argv", ["run_data.py", "--check-only"]):
                args = self.runner.parse_args()
        self.assertEqual(
            args.lance_bin,
            str(lance_root / "target/release/examples/lance_scheduler_qps_s3"),
        )
        self.assertEqual(
            args.lance_per_iop_bin,
            str(lance_root / "target/release/examples/lance_scheduler_qps_s3_per_iop"),
        )

    def test_provenance_identity_is_derived_from_resolved_s3_uri(self) -> None:
        args = argparse.Namespace(
            uri="s3://bucket/figure04_synth/run/blob.bin",
            s3_bucket="bucket",
            s3_key="",
        )
        self.assertEqual(
            self.runner._resolved_s3_identity(args),
            ("bucket", "figure04_synth/run/blob.bin"),
        )

    def test_generated_object_metadata_uses_the_current_figure_number(self) -> None:
        class FakeS3:
            metadata: dict[str, str] | None = None

            def create_multipart_upload(self, **kwargs):
                self.metadata = kwargs["Metadata"]
                return {"UploadId": "upload-1"}

            def upload_part(self, **_kwargs):
                return {"ETag": "etag-1"}

            def complete_multipart_upload(self, **_kwargs):
                return None

            def abort_multipart_upload(self, **_kwargs):
                return None

            def head_object(self, **_kwargs):
                return {"ContentLength": 1}

        fake_s3 = FakeS3()
        fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: fake_s3)
        with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
            self.runner._upload_synthetic_blob(
                bucket="bucket",
                key="figure04_synth/blob.bin",
                size_bytes=1,
                region="us-east-1",
            )
        self.assertEqual(
            fake_s3.metadata,
            {"vecio-figure": "4", "vecio-generated": "true"},
        )


class Figure7ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner("figure-07-real-lance-metrics")

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            s3_root="s3://bucket/prefix",
            data_root=str(ROOT / "data/processed"),
            index_manifest_root=str(ROOT / "manifests"),
            query_concurrency=32,
            io_limit=2048,
            num_queries=10_000,
            warmup_queries=200,
            repeats=3,
            network_interface="eth-test",
        )

    def test_performance_schedule_is_exactly_seven_by_three_by_three(self) -> None:
        schedule = self.runner._performance_schedule(self.args())
        counts = Counter((cell.dataset, cell.method.name) for cell in schedule)
        self.assertEqual(len(schedule), 63)
        self.assertEqual(set(counts.values()), {3})
        self.assertTrue(all(cell.network_interface == "eth-test" for cell in schedule))

    def test_source_csv_has_complete_paper_schema(self) -> None:
        metric_names = (
            "qps", "p50_ms", "p95_ms", "p99_ms", "iops_per_q",
            "bytes_per_q", "measured_payload_gbps", "rx_gbps", "tx_gbps",
        )
        metrics = {
            name: {"mean": 1.0, "sample_sd": 0.0, "values": [1.0, 1.0, 1.0]}
            for name in metric_names
        }
        performance = {
            "aggregate": [
                {"dataset": dataset, "method": method, "metrics": metrics}
                for dataset in self.runner.DATASETS
                for method in self.runner.METHODS
            ]
        }
        cache_metrics = {
            name: {"mean": 1.0, "sample_sd": 0.0, "values": [1.0, 1.0, 1.0]}
            for name in ("qps", "p50_ms", "p95_ms", "p99_ms", "iops_per_q")
        }
        cache = {
            "groups": [
                {
                    "cache_fraction_percent": fraction,
                    "config": method,
                    **{
                        f"{metric}_{stat}": cache_metrics[metric][stat]
                        for metric in cache_metrics
                        for stat in ("mean", "sample_sd")
                    },
                }
                for fraction in (10, 50)
                for method in self.runner.METHODS
            ]
        }
        recall = {
            "rows": [
                {"dataset": dataset, "method": method, "recall_at_10": 0.9}
                for dataset in self.runner.DATASETS
                for method in self.runner.METHODS
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure7_source.csv"
            self.runner._write_source_csv(path, performance, cache, recall)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fields = stream.seek(0) or next(csv.reader(stream))
        self.assertEqual(len(rows), 27)
        self.assertIn("recall_at_10", fields)
        self.assertIn("payload_gbps_mean", fields)
        self.assertIn("rx_gbps_mean", fields)
        self.assertIn("tx_gbps_mean", fields)


class Figure7RenderingTests(unittest.TestCase):
    def test_bundled_dejavu_fallback_keeps_value_labels_inside_axes(self) -> None:
        path = (
            ROOT
            / "experiments"
            / "figure-07-real-lance-metrics"
            / "render_figure7.py"
        )
        spec = importlib.util.spec_from_file_location("test_figure7_renderer", path)
        assert spec is not None and spec.loader is not None
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        renderer = wrapper.renderer

        from src.plotting import paper_plot_style

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            paper_plot_style,
            "PAPER_SANS_FONTS",
            ["DejaVu Sans"],
        ), mock.patch.object(
            renderer,
            "OUT",
            Path(directory) / "figure7",
        ), mock.patch.object(
            renderer.subprocess,
            "run",
        ):
            rows = renderer.load_rows()
            renderer.render(renderer.normalized_matrix(rows))


class SharedAggregationTests(unittest.TestCase):
    def test_lance_build_receipt_binds_binary_and_cargo_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "target" / "release" / "examples" / "paper-example"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"paper-binary")
            binary.chmod(0o755)
            (root / "Cargo.lock").write_bytes(b"paper-lock")
            with mock.patch.object(
                lance_build_receipt,
                "validate_source_identity",
                return_value={
                    "lance_commit": LANCE_BASE_COMMIT,
                    "source_mode": "git-checkout",
                    "source_tree_sha256": LANCE_EXPORTED_SOURCE_SHA256,
                },
            ):
                path = lance_build_receipt.write_receipt(
                    root,
                    binary,
                    package="lance",
                    example="paper-example",
                )
                identity = lance_build_receipt.validate_receipt(
                    root,
                    binary,
                    package="lance",
                    example="paper-example",
                )
                self.assertEqual(identity["receipt_sha256"], lance_build_receipt.sha256(path))
                self.assertEqual(identity["schema_version"], 2)
                self.assertEqual(
                    identity["source_tree_sha256"],
                    LANCE_EXPORTED_SOURCE_SHA256,
                )
                binary.write_bytes(b"different-binary")
                binary.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "binary_sha256"):
                    lance_build_receipt.validate_receipt(
                        root,
                        binary,
                        package="lance",
                        example="paper-example",
                    )

    def test_lance_build_receipt_accepts_a_verified_source_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "target/release/examples/paper-example"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"paper-binary")
            binary.chmod(0o755)
            (root / "Cargo.lock").write_bytes(b"paper-lock")
            exported = {
                "lance_commit": LANCE_BASE_COMMIT,
                "source_mode": "exported-source",
                "source_tree_sha256": LANCE_EXPORTED_SOURCE_SHA256,
            }
            with mock.patch.object(
                lance_build_receipt,
                "validate_source_identity",
                return_value=exported,
            ):
                path = lance_build_receipt.write_receipt(
                    root,
                    binary,
                    package="lance",
                    example="paper-example",
                )
                identity = lance_build_receipt.validate_receipt(
                    root,
                    binary,
                    package="lance",
                    example="paper-example",
                )
                receipt_digest = lance_build_receipt.sha256(path)
        self.assertEqual(identity["source_mode"], "exported-source")
        self.assertEqual(identity["receipt_sha256"], receipt_digest)

    def test_campaign_environment_removes_behavioral_overrides(self) -> None:
        cell = full_pipeline_campaign.Cell(
            experiment="test",
            dataset="sift100m",
            method=full_pipeline_campaign.Method("stock", "random", "native"),
            repeat=1,
            uri="s3://bucket/sift100m.lance",
            queries_npy=ROOT / "queries.npy",
            index_manifest=ROOT / "manifest.json",
            query_concurrency=32,
            io_limit=64,
            process_io_limit=2048,
            nprobe=256,
            refine_factor=8,
            k=10,
            num_queries=1_000,
            warmup_queries=200,
        )
        env = full_pipeline_campaign._method_env(
            cell,
            {
                "LANCE_MAX_IOP_SIZE": "1",
                "LANCE_PAYLOAD_PLAN_ONLY": "1",
                "LANCE_IO_MERGE_EXACT_REQUESTS": "1",
                "LANCE_IO_MERGE_LATENCY_PER_REQUEST_MS": "999",
                "LANCE_IO_MERGE_LATENCY_PER_BYTE_MS": "999",
                "LANCE_IO_MERGE_LATENCY_STD_MS": "999",
                "UNRELATED_VALUE": "retained",
            },
        )
        for name in (
            "LANCE_MAX_IOP_SIZE",
            "LANCE_PAYLOAD_PLAN_ONLY",
            "LANCE_IO_MERGE_EXACT_REQUESTS",
            "LANCE_IO_MERGE_LATENCY_PER_REQUEST_MS",
            "LANCE_IO_MERGE_LATENCY_PER_BYTE_MS",
            "LANCE_IO_MERGE_LATENCY_STD_MS",
        ):
            self.assertNotIn(name, env)
        self.assertEqual(env["UNRELATED_VALUE"], "retained")

    def test_requested_metrics_are_not_shadowed(self) -> None:
        rows = [
            {"meta": {"dataset": "sift"}, "result": {"qps": 10.0}},
            {"meta": {"dataset": "sift"}, "result": {"qps": 14.0}},
        ]
        result = aggregate(
            rows,
            group_fields=("dataset",),
            expected_repeats=2,
            metrics=("qps",),
        )
        self.assertEqual(result[0]["metrics"]["qps"]["mean"], 12.0)
        self.assertEqual(result[0]["metrics"]["qps"]["values"], [10.0, 14.0])

    def test_index_manifest_geometry_is_checked_against_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            record = {
                "uri": "s3://bucket/sift100m.lance",
                "dataset_version": 7,
                "logical_index_name": "vector_idx",
                "physical_index_uuid": "physical-uuid",
                "index_type": "IVF_PQ",
                "nlist": 1,
                "num_sub_vectors": 1,
                "bits_per_subcode": 8,
            }
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "dataset": "sift100m",
                        "nlist": 1,
                        "num_sub_vectors": 1,
                        "bits_per_subcode": 8,
                        "indices": {"random": record},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "nlist"):
                index_identity(
                    path,
                    dataset="sift100m",
                    layout="random",
                    uri=record["uri"],
                )

    def test_remote_physical_index_must_match_manifest(self) -> None:
        spec = MATCHED_DATASETS["sift100m"]
        uri = "s3://bucket/sift100m.lance"
        record = {
            "uri": uri,
            "dataset_version": 7,
            "logical_index_name": "vector_idx",
            "physical_index_uuid": "physical-uuid",
            "index_type": "IVF_PQ",
            "nlist": spec.nlist,
            "num_sub_vectors": spec.num_sub_vectors,
            "bits_per_subcode": 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "dataset": "sift100m",
                        "nlist": spec.nlist,
                        "num_sub_vectors": spec.num_sub_vectors,
                        "bits_per_subcode": 8,
                        "indices": {"random": record},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                full_pipeline_campaign,
                "_observe_remote_index",
                return_value={
                    key: record[key]
                    for key in (
                        "dataset_version",
                        "logical_index_name",
                        "physical_index_uuid",
                        "index_type",
                        "nlist",
                        "num_sub_vectors",
                        "bits_per_subcode",
                    )
                },
            ):
                identity = full_pipeline_campaign.observed_index_identity(
                    path,
                    dataset="sift100m",
                    layout="random",
                    uri=uri,
                )
        self.assertEqual(
            identity["observed_remote_index"]["physical_index_uuid"],
            "physical-uuid",
        )

    def test_completed_cell_cannot_be_reused_with_a_different_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.npy"
            np.save(queries, np.zeros((1, 1), dtype=np.float32))
            old_binary = root / "old-binary"
            new_binary = root / "new-binary"
            old_binary.write_bytes(b"old")
            new_binary.write_bytes(b"new")
            cell = full_pipeline_campaign.Cell(
                experiment="test",
                dataset="sift100m",
                method=full_pipeline_campaign.Method("stock", "random", "native"),
                repeat=1,
                uri="s3://bucket/sift100m.lance",
                queries_npy=queries,
                index_manifest=root / "manifest.json",
                query_concurrency=1,
                io_limit=64,
                nprobe=1,
                refine_factor=1,
                k=1,
                num_queries=1,
                warmup_queries=0,
                process_io_limit=2048,
            )
            cell_dir = root / "cell"
            cell_dir.mkdir()
            identity = {"manifest_sha256": "a" * 64}
            with mock.patch.object(
                full_pipeline_campaign,
                "observed_index_identity",
                return_value=identity,
            ):
                metadata = full_pipeline_campaign._cell_metadata(cell, old_binary)
                (cell_dir / "meta.json").write_text(
                    json.dumps({"schema_version": 1, "cell": metadata}),
                    encoding="utf-8",
                )
                result = {
                    "num_q": 1,
                    "Q": 1,
                    "R": 2048,
                    "qps": 1,
                    "p50_ms": 1,
                    "p95_ms": 1,
                    "p99_ms": 1,
                    "iops_per_q": 1,
                    "bytes_per_q": 1,
                    "elapsed_s": 1,
                }
                (cell_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                (cell_dir / "latencies.jsonl").write_text("{}\n", encoding="utf-8")
                (cell_dir / "stdout").write_text("", encoding="utf-8")
                (cell_dir / "stderr").write_text("", encoding="utf-8")
                files = ("meta.json", "result.json", "latencies.jsonl", "stdout", "stderr")
                (cell_dir / "sha256.json").write_text(
                    json.dumps(
                        {
                            name: full_pipeline_campaign.sha256(cell_dir / name)
                            for name in files
                        }
                    ),
                    encoding="utf-8",
                )
                row = full_pipeline_campaign.validate_cell(cell, cell_dir, old_binary)
                self.assertEqual(row["result"]["R"], 2048)
                with self.assertRaisesRegex(ValueError, "metadata does not match"):
                    full_pipeline_campaign.validate_cell(cell, cell_dir, new_binary)


class Figure8ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner("figure-08-sift100m-concurrency-scaling")

    def test_paper_cell_counts_and_io_contract(self) -> None:
        self.assertEqual(len(self.runner.Q_LEVELS) * 3 * 3, 54)
        self.assertEqual(sum(map(len, self.runner.IO_CAMPAIGNS.values())) * 3, 30)
        args = argparse.Namespace(
            s3_root="s3://bucket/prefix",
            queries_npy=str(ROOT / "queries.npy"),
            index_manifest_root=str(ROOT / "manifests"),
            num_queries=10_000,
            warmup_queries=200,
        )
        cell = self.runner._cell(
            args,
            experiment="figure8-io-test",
            method=self.runner.METHODS["stock_lance"],
            repeat=1,
            q=32,
            io_limit=64,
            process_io_limit=2048,
            num_queries=1_000,
            refine_factor=8,
        )
        self.assertEqual(cell.num_queries, 1_000)
        self.assertEqual(cell.refine_factor, 8)
        self.assertEqual(cell.process_io_limit, 2048)
        self.assertEqual(
            self.runner.IO_SCHEDULES["ladder"][3],
            (
                (256, 2048),
                (64, 2048),
                (1024, 2048),
                (128, 2048),
                (2048, 2048),
                (512, 2048),
            ),
        )

    def test_io_schedule_is_the_exact_three_campaign_protocol(self) -> None:
        self.assertEqual(
            self.runner.IO_SCHEDULES,
            {
                "ladder": {
                    1: (
                        (64, 2048),
                        (128, 2048),
                        (256, 2048),
                        (512, 2048),
                        (1024, 2048),
                        (2048, 2048),
                    ),
                    2: (
                        (2048, 2048),
                        (1024, 2048),
                        (512, 2048),
                        (256, 2048),
                        (128, 2048),
                        (64, 2048),
                    ),
                    3: (
                        (256, 2048),
                        (64, 2048),
                        (1024, 2048),
                        (128, 2048),
                        (2048, 2048),
                        (512, 2048),
                    ),
                },
                "2048-4096": {
                    1: ((2048, 2048), (4096, 4096)),
                    2: ((4096, 4096), (2048, 2048)),
                    3: ((2048, 2048), (4096, 4096)),
                },
                "4096-8192": {
                    1: ((4096, 4096), (8192, 8192)),
                    2: ((8192, 8192), (4096, 4096)),
                    3: ((4096, 4096), (8192, 8192)),
                },
            },
        )

    def test_paired_chained_series_uses_repeatwise_ratios_and_propagates_sd(self) -> None:
        renderer_path = (
            ROOT
            / "experiments"
            / "figure-08-sift100m-concurrency-scaling"
            / "render_io_series.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_figure8_io_renderer",
            renderer_path,
        )
        assert spec is not None and spec.loader is not None
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)

        def metric(values: list[float]) -> dict:
            return {
                "values": values,
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
            }

        ladder_groups = {
            str(limit): {"p50_ms": metric([90.0, 100.0, 110.0])}
            for limit in renderer.ALL_LIMITS[:6]
        }
        summaries = {
            "ladder": {"groups": ladder_groups},
            "2048-4096": {
                "groups": {
                    "2048": {"p50_ms": metric([10.0, 20.0, 40.0])},
                    "4096": {"p50_ms": metric([15.0, 50.0, 60.0])},
                }
            },
            "4096-8192": {
                "groups": {
                    "4096": {"p50_ms": metric([20.0, 25.0, 50.0])},
                    "8192": {"p50_ms": metric([18.0, 30.0, 55.0])},
                }
            },
        }
        means, errors, evidence = renderer.paired_chained_series(
            summaries,
            "p50_ms",
        )

        first_ratio_mean = 125.0 / 70.0
        first_ratio_sd = float(np.std([1.5, 2.5, 1.5], ddof=1))
        expected_4096 = 100.0 * first_ratio_mean
        expected_4096_sd = expected_4096 * np.sqrt(
            (10.0 / 100.0) ** 2 + (first_ratio_sd / first_ratio_mean) ** 2
        )
        second_ratio_mean = 103.0 / 95.0
        second_ratio_sd = float(np.std([0.9, 1.2, 1.1], ddof=1))
        expected_8192 = expected_4096 * second_ratio_mean
        expected_8192_sd = expected_8192 * np.sqrt(
            (expected_4096_sd / expected_4096) ** 2
            + (second_ratio_sd / second_ratio_mean) ** 2
        )

        np.testing.assert_allclose(means[-2:], [expected_4096, expected_8192])
        np.testing.assert_allclose(
            errors[-2:],
            [expected_4096_sd, expected_8192_sd],
        )
        self.assertEqual(
            [(row["from_limit"], row["to_limit"]) for row in evidence],
            [(2048.0, 4096.0), (4096.0, 8192.0)],
        )

    def test_frozen_io_summaries_retain_measurement_provenance(self) -> None:
        root = (
            ROOT
            / "experiments"
            / "figure-08-sift100m-concurrency-scaling"
            / "data"
            / "paper"
            / "io_summaries"
        )
        expected_pairs = {
            "ladder": [(64, 2048), (128, 2048), (256, 2048), (512, 2048), (1024, 2048), (2048, 2048)],
            "2048-4096": [(2048, 2048), (4096, 4096)],
            "4096-8192": [(4096, 4096), (8192, 8192)],
        }
        for name, pairs in expected_pairs.items():
            payload = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            contract = payload["measurement_contract"]
            self.assertEqual(contract["query_concurrency"], 32)
            self.assertEqual(contract["num_queries"], 1_000)
            self.assertEqual(contract["warmup_queries"], 200)
            self.assertEqual(contract["nprobe"], 256)
            self.assertEqual(contract["refine_factor"], 8)
            self.assertEqual(
                [(row["file"], row["process"]) for row in contract["limit_pairs"]],
                pairs,
            )
            self.assertRegex(contract["query_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(contract["binary_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(contract["lance_commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(
                contract["lance_tracked_diff_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(contract["source_archive_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(contract["index_identity"])
            self.assertTrue(contract["source_hashes"])

    def test_frozen_query_summary_retains_measurement_provenance(self) -> None:
        path = (
            ROOT
            / "experiments"
            / "figure-08-sift100m-concurrency-scaling"
            / "data"
            / "paper"
            / "query_summary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["campaign"], "figure8-query-concurrency")
        contract = payload["measurement_contract"]
        self.assertEqual(contract["query_concurrency_levels"], [16, 32, 48, 64, 96, 128])
        self.assertEqual(contract["num_queries"], 10_000)
        self.assertEqual(contract["warmup_queries"], 200)
        self.assertEqual(contract["refine_factor"], 8)
        self.assertEqual(contract["io_limit"], 2048)
        self.assertRegex(contract["query_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(contract["binary_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(contract["source_archive_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(contract["source_hashes"])
        self.assertTrue(
            all(
                cell["meta"]["query_sha256"] == contract["query_sha256"]
                and cell["meta"]["binary_sha256"] == contract["binary_sha256"]
                for cell in payload["cells"]
            )
        )

    def test_fresh_io_summary_matches_the_frozen_metric_schema(self) -> None:
        metrics = ("elapsed_s", "iops_per_q", "p50_ms", "p95_ms", "p99_ms", "qps")
        rows = []
        for repeat in (3, 1, 2):
            for limit in (2048, 4096):
                value = float(repeat * 10 + limit // 2048)
                rows.append(
                    {
                        "meta": {
                            "io_limit": limit,
                            "repeat": repeat,
                            "queries_sha256": "a" * 64,
                            "binary_sha256": "b" * 64,
                        },
                        "result": {metric: value for metric in metrics},
                    }
                )
        args = argparse.Namespace(io_num_queries=1_000, warmup_queries=200)
        with mock.patch.object(
            self.runner,
            "_companion_identity",
            return_value={
                "lance_commit": "c" * 40,
                "receipt_sha256": "e" * 64,
                "cargo_lock_sha256": "f" * 64,
            },
        ):
            payload = self.runner._io_summary(
                args,
                ROOT / "binary",
                "2048-4096",
                rows,
                (2048, 4096),
                {"manifest_sha256": "d" * 64},
            )
        self.assertEqual(set(payload["groups"]), {"2048", "4096"})
        self.assertEqual(set(payload["groups"]["2048"]), set(metrics))
        self.assertEqual(
            payload["groups"]["2048"]["qps"]["values"],
            [11.0, 21.0, 31.0],
        )
        self.assertEqual(payload["measurement_contract"]["num_queries"], 1_000)
        self.assertEqual(payload["measurement_contract"]["refine_factor"], 8)
        self.assertEqual(
            payload["measurement_contract"]["binary_build_receipt_sha256"],
            "e" * 64,
        )

    def test_fresh_query_summary_binds_cells_to_build_and_sources(self) -> None:
        identities = {
            "random": {"manifest_sha256": "1" * 64},
            "reorder": {"manifest_sha256": "2" * 64},
        }
        source_hashes = {
            "runner_sha256": self.runner.sha256(Path(self.runner.__file__).resolve()),
            "shared_harness_sha256": self.runner.sha256(
                ROOT / "runners" / "shared" / "full_pipeline_campaign.py"
            ),
            "campaign_contract_sha256": self.runner.sha256(
                ROOT / "artifact" / "campaigns.py"
            ),
        }
        metrics = {
            metric: 1.0
            for metric in full_pipeline_campaign.AGGREGATE_METRICS
        }
        rows = []
        for repeat in (1, 2, 3):
            ordinal = 0
            for q in self.runner.Q_LEVELS:
                for method in self.runner.METHODS:
                    ordinal += 1
                    layout = "reorder" if method == "warp_reorder" else "random"
                    rows.append(
                        {
                            "cell": f"r{repeat}-{q}-{method}",
                            "meta": {
                                "dataset": "sift100m",
                                "experiment": "figure8-query-concurrency",
                                "method": {"name": method},
                                "query_concurrency": q,
                                "io_limit": 2048,
                                "nprobe": 256,
                                "refine_factor": 8,
                                "k": 10,
                                "num_queries": 10_000,
                                "warmup_queries": 200,
                                "max_merge_span_bytes": 524_288,
                                "repeat": repeat,
                                "order_ordinal": ordinal,
                                "queries_sha256": "a" * 64,
                                "binary_sha256": "b" * 64,
                                "index_identity": identities[layout],
                                **source_hashes,
                            },
                            "result": dict(metrics),
                        }
                    )
        args = argparse.Namespace(num_queries=10_000, warmup_queries=200)
        with mock.patch.object(
            self.runner,
            "_companion_identity",
            return_value={
                "lance_commit": "c" * 40,
                "receipt_sha256": "d" * 64,
                "cargo_lock_sha256": "e" * 64,
            },
        ):
            payload = self.runner._query_summary(
                args,
                ROOT / "binary",
                rows,
                identities,
            )
        contract = payload["measurement_contract"]
        self.assertEqual(contract["binary_build_receipt_sha256"], "d" * 64)
        self.assertEqual(contract["cargo_lock_sha256"], "e" * 64)
        self.assertEqual(contract["index_identities"], identities)
        self.assertEqual(contract["source_hashes"]["shared_harness"], source_hashes["shared_harness_sha256"])
        self.assertTrue(
            all(cell["meta"]["binary_sha256"] == "b" * 64 for cell in payload["cells"])
        )

    def test_runner_requires_the_exact_clean_companion_checkout(self) -> None:
        with mock.patch.object(
            self.runner,
            "_companion_root",
            return_value=ROOT,
        ), mock.patch.object(
            self.runner,
            "validate_lance_source",
            return_value={
                "lance_commit": LANCE_BASE_COMMIT,
                "source_mode": "git-checkout",
                "source_tree_sha256": LANCE_EXPORTED_SOURCE_SHA256,
            },
        ):
            self.assertEqual(
                self.runner._validate_companion_checkout(ROOT / "binary"),
                LANCE_BASE_COMMIT,
            )

        with mock.patch.object(
            self.runner,
            "_companion_root",
            return_value=ROOT,
        ), mock.patch.object(
            self.runner,
            "validate_lance_source",
            side_effect=SystemExit("the Lance checkout must be at the paper release"),
        ):
            with self.assertRaisesRegex(RuntimeError, "must be at"):
                self.runner._validate_companion_checkout(ROOT / "binary")

    def test_runner_probes_required_binary_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "full_pipeline_real_s3"
            binary.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Usage: benchmark "
                "--reuse-warmup-queries --out-latencies-jsonl'\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            self.runner._validate_binary_capabilities(binary)
            binary.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Usage: benchmark --out-json'\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lacks required Figure 8 flags"):
                self.runner._validate_binary_capabilities(binary)


class Figure11ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner("figure-11-greedy-locality-shift")

    def test_compiled_flag_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "binary"
            binary.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Usage: binary --only-method'\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            self.assertTrue(self.runner._binary_has_flag(binary, "--only-method"))
            self.assertFalse(self.runner._binary_has_flag(binary, "--out-query-ranges-json"))

    def test_msong_uri_follows_prepared_dataset_root(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "VECIO_REAL_LANCE_S3_ROOT": "s3://bucket/rebuilt/",
                "VECIO_S3_BUCKET": "ignored",
            },
            clear=True,
        ):
            self.assertEqual(
                self.runner._default_msong_uri(),
                "s3://bucket/rebuilt/msong_pq_reorder.lance",
            )

    def test_managed_msong_identity_pins_uuid_and_labels_replication(self) -> None:
        uri = "s3://bucket/rebuilt/msong_pq_reorder.lance"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "dataset": "msong-figure11",
                        "nlist": 4096,
                        "num_sub_vectors": 60,
                        "bits_per_subcode": 8,
                        "indices": {
                            "reordered": {
                                "uri": uri,
                                "physical_index_uuid": "rebuilt-uuid",
                                "dataset_version": 7,
                                "logical_index_name": "vector_idx",
                                "index_type": "IVF_PQ",
                                "nlist": 4096,
                                "num_sub_vectors": 60,
                                "bits_per_subcode": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            contract = self.runner._msong_index_contract(path, uri)
        self.assertEqual(contract["expected_index_uuid"], "rebuilt-uuid")
        self.assertEqual(contract["mode"], "reconstructed-replication")

    def test_frozen_paper_identity_is_distinguished(self) -> None:
        uri = "s3://paper-bucket/paper/msong.lance"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(
                json.dumps(
                    {
                        "expected_index_uuid": FIGURE11_PAPER_MSONG_INDEX_UUID,
                        "payload_object_etag": FIGURE11_PAPER_MSONG_OBJECT_ETAG,
                        "payload_object_size": FIGURE11_PAPER_MSONG_OBJECT_SIZE,
                        "dataset_uri": uri,
                    }
                ),
                encoding="utf-8",
            )
            contract = self.runner._msong_index_contract(path, uri)
        self.assertEqual(contract["mode"], "paper-object")

    def test_external_inventory_requires_uri_etag_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(
                json.dumps({"expected_index_uuid": "only-a-uuid"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inventory URI differs"):
                self.runner._msong_index_contract(
                    path,
                    "s3://bucket/msong.lance",
                )

    def test_managed_msong_remote_index_must_match_manifest_contract(self) -> None:
        contract = {
            "mode": "reconstructed-replication",
            "expected_dataset_version": 7,
            "expected_index_uuid": "physical-uuid",
            "logical_index_name": "vector_idx",
        }

        class FakeDataset:
            version = 7
            stats = types.SimpleNamespace(
                index_stats=lambda _name: {
                    "indices": [
                        {
                            "uuid": "physical-uuid",
                            "index_type": "IVF_PQ",
                            "num_partitions": 4096,
                            "sub_index": {
                                "num_sub_vectors": 60,
                                "nbits": 8,
                            },
                        }
                    ]
                }
            )

        fake_lance = types.SimpleNamespace(dataset=lambda _uri: FakeDataset())
        with mock.patch.dict(sys.modules, {"lance": fake_lance}):
            observed = self.runner._validate_observed_managed_msong_index(
                "s3://bucket/msong.lance",
                contract,
            )
        self.assertEqual(observed["physical_index_uuid"], "physical-uuid")
        with mock.patch.dict(sys.modules, {"lance": fake_lance}):
            with self.assertRaisesRegex(ValueError, "dataset_version"):
                self.runner._validate_observed_managed_msong_index(
                    "s3://bucket/msong.lance",
                    {**contract, "expected_dataset_version": 8},
                )

    def test_paper_label_requires_uri_and_observed_etag(self) -> None:
        uri = "s3://paper-bucket/paper/msong.lance"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(
                json.dumps(
                    {
                        "expected_index_uuid": FIGURE11_PAPER_MSONG_INDEX_UUID,
                        "payload_object_etag": FIGURE11_PAPER_MSONG_OBJECT_ETAG,
                        "payload_object_size": FIGURE11_PAPER_MSONG_OBJECT_SIZE,
                        "bucket": "paper-bucket",
                        "prefix": "paper/msong.lance",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inventory URI differs"):
                self.runner._msong_index_contract(path, "s3://other/object")
            paper = self.runner._msong_index_contract(path, uri)
        with self.assertRaisesRegex(ValueError, "payload ETag differs"):
            self.runner._bind_observed_payload_object(
                paper,
                etag='"different"',
                size=1,
            )
        with self.assertRaisesRegex(ValueError, "payload size differs"):
            self.runner._bind_observed_payload_object(
                paper,
                etag=FIGURE11_PAPER_MSONG_OBJECT_ETAG,
                size=1,
            )

    def test_observed_msong_identity_is_stable_across_cache_reuse(self) -> None:
        uri = "s3://bucket/msong.lance"
        contract = {
            "mode": "external",
            "expected_index_uuid": "index-uuid",
            "identity_path": "/identity.json",
            "identity_sha256": "a" * 64,
        }

        class FakeClient:
            def head_object(self, **_kwargs):
                return {"ETag": '"etag"', "ContentLength": 123}

        fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeClient())
        with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
            _, first = self.runner._observe_msong_payload(uri, contract, region="test")
            _, second = self.runner._observe_msong_payload(uri, contract, region="test")
        self.assertEqual(first, second)

    def test_precomputed_manifests_are_bound_to_identity_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_uri = "s3://bucket/msong.lance"
            selected_uuid = "selected-uuid"
            payload_uri = f"{dataset_uri}/_indices/{selected_uuid}/auxiliary.idx"
            manifest = {
                "uri": payload_uri,
                "nlist": 4096,
                "nprobe": 64,
                "warmup_queries": [[[0, 1]] for _ in range(20)],
                "queries": [[[1, 2]] for _ in range(200)],
            }
            calibration = root / "calibration.json"
            evaluation = root / "evaluation.json"
            calibration.write_text(json.dumps(manifest), encoding="utf-8")
            evaluation.write_text(json.dumps(manifest), encoding="utf-8")
            identity = {
                "protocol_id": self.runner.MSONG_PROTOCOL_ID,
                "dataset_uri": dataset_uri,
                "selected_index_uuid": selected_uuid,
                "payload_object_uri": payload_uri,
                "payload_object_size": 123,
                "payload_object_etag": '"etag"',
                "calibration_manifest_sha256": self.runner._sha256(calibration),
                "evaluation_manifest_sha256": self.runner._sha256(evaluation),
                "warmup_queries": 20,
                "calibration_queries": 200,
                "evaluation_queries": 200,
                "nlist": 4096,
                "nprobe": 64,
            }
            (root / "input_identity.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            self.runner._validate_msong_manifest_root(root)
            calibration.write_text(json.dumps({**manifest, "tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration_manifest_sha256"):
                self.runner._validate_msong_manifest_root(root)
            calibration.write_text(
                json.dumps({**manifest, "warmup_queries": [[] for _ in range(20)]}),
                encoding="utf-8",
            )
            identity["calibration_manifest_sha256"] = self.runner._sha256(calibration)
            (root / "input_identity.json").write_text(
                json.dumps(identity),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "calibration_warmup_queries_ranges"):
                self.runner._validate_msong_manifest_root(root)

    def test_precomputed_payload_identity_is_checked_against_s3(self) -> None:
        identity = {
            "payload_object_uri": "s3://bucket/prefix/auxiliary.idx",
            "payload_object_etag": '"etag"',
            "payload_object_size": 123,
        }

        class FakeClient:
            def head_object(self, **kwargs):
                self.kwargs = kwargs
                return {"ETag": '"etag"', "ContentLength": 123}

        client = FakeClient()
        fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: client)
        with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
            observed = self.runner._validate_observed_msong_payload(
                identity,
                region="test-region",
            )
        self.assertEqual(observed["payload_object_size"], 123)
        self.assertEqual(client.kwargs["Bucket"], "bucket")
        self.assertEqual(client.kwargs["Key"], "prefix/auxiliary.idx")

        identity["payload_object_size"] = 999
        with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
            with self.assertRaisesRegex(ValueError, "payload_object_size"):
                self.runner._validate_observed_msong_payload(
                    identity,
                    region="test-region",
                )

    def test_fresh_summaries_can_record_binary_hashes_and_companion_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strict = root / "strict"
            trace = root / "trace"
            strict.write_bytes(b"strict-binary")
            trace.write_bytes(b"trace-binary")
            git = {"root": "/lance", "head": "abc123", "dirty": False}
            with mock.patch.object(
                self.runner, "_companion_git_metadata", return_value=git
            ):
                provenance = self.runner._toolchain_provenance(strict, trace)
        binaries = provenance["binaries"]
        self.assertEqual(binaries["strict_budget_real_s3"]["source_checkout"], git)
        self.assertEqual(
            binaries["strict_budget_real_s3"]["sha256"],
            hashlib.sha256(b"strict-binary").hexdigest(),
        )

    def test_work_directory_cannot_mix_toolchains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            self.runner._bind_work_directory(
                work,
                {"schema_version": 1, "toolchain": {"sha256": "first"}},
                force=False,
            )
            with self.assertRaisesRegex(SystemExit, "different input/toolchain"):
                self.runner._bind_work_directory(
                    work,
                    {"schema_version": 1, "toolchain": {"sha256": "second"}},
                    force=False,
                )


class OrchestratorPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner("table-01-cluster-coarsening")

    def test_cluster_check_only_never_invokes_tuning_subcommand(self) -> None:
        argv = [
            "run_data.py",
            "--check-only",
            "tune",
            "--fine-nlist",
            "4096",
            "--coarse-nlist",
            "1024",
            "--fine-nprobes",
            "128",
            "--coarse-nprobes",
            "32",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            self.runner, "cmd_tune", return_value=0
        ) as tune:
            self.assertEqual(self.runner.main(), 0)
        tune.assert_not_called()


class PlannerSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner("figure-12-planner-overhead")

    def test_raw_four_way_output_is_normalized_to_paper_summary(self) -> None:
        fields = sorted(self.runner.FOUR_WAY_FIELDS)
        rows = []
        for dataset in ("GLOVE", "SIFT1M"):
            for nprobe in (128, 256, 512):
                for k in (16, 32, 64, 128, 256, 512):
                    row = {field: "1.0" for field in fields}
                    row.update(
                        {
                            "dataset": dataset,
                            "nprobe": nprobe,
                            "K": k,
                            "plain_timed_out": "false",
                            "dc_timed_out": "false",
                            "adaptive_dc_timed_out": "false",
                            "wqs_timed_out": "false",
                        }
                    )
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            summary = Path(directory) / "merged_summary.csv"
            with raw.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.runner._publish_four_way_summary(raw, summary)
            with summary.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                normalized = list(reader)
                columns = reader.fieldnames
        self.assertEqual(len(normalized), 18)
        self.assertEqual(
            columns,
            [
                "dataset",
                "nprobe",
                "K",
                "plain_timed_out",
                "plain_p95_ms",
                "dc_timed_out",
                "dc_p95_ms",
                "adaptive_dc_p95_ms",
                "wqs_p95_ms",
                "wqs_over_adaptive_dc_p95",
                "adaptive_dc_err_avg_pct",
                "adaptive_dc_err_max_pct",
                "wqs_err_avg_pct",
                "wqs_err_max_pct",
            ],
        )


if __name__ == "__main__":
    unittest.main()
