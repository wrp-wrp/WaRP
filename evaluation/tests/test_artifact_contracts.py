from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from artifact.prepare_dataset import DATASET_OUTPUT_CONTRACTS, _validate_prepared_output
from artifact.data import data_slot, paper_data_path
from artifact.cli import main as artifact_main
from artifact.checks import public_hygiene
from artifact.checks.public_hygiene import (
    OPAQUE_REVISION_ID,
    OPAQUE_STAGE_ID,
    _is_physical_object_identifier,
    _privacy_violations,
    _read_text_if_probable,
    _resolved_static_strings,
    _unallowlisted_matches,
    history_violations,
)
from artifact.registry import get, ordered_for_run, paper_artifact_specs


ROOT = Path(__file__).resolve().parents[1]


class DataSlotTests(unittest.TestCase):
    def test_paper_slot_is_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WARP_DATA_SLOT", None)
            self.assertEqual(data_slot(), "paper")

    def test_latest_slot_rewrites_only_data_paper_segment(self) -> None:
        source = ROOT / "experiments/example/data/paper/result.json"
        with mock.patch.dict(os.environ, {"WARP_DATA_SLOT": "latest"}):
            self.assertEqual(
                paper_data_path(source),
                ROOT / "experiments/example/data/latest/result.json",
            )

    def test_invalid_slot_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"WARP_DATA_SLOT": "invalid"}):
            with self.assertRaises(ValueError):
                data_slot()


class RegistryCoverageTests(unittest.TestCase):
    def test_dependency_closure_precedes_target(self) -> None:
        ids = [spec.id for spec in ordered_for_run(["figure-13-reordering-methods"])]
        self.assertEqual(
            ids,
            ["reorder-cost", "locality-aware-coaccess-gap", "figure-13-reordering-methods"],
        )

    def test_every_measurement_backed_paper_artifact_has_a_run_path(self) -> None:
        deterministic_render_only = {"figure-05-planning-frontier"}
        missing = []
        for spec in paper_artifact_specs():
            if spec.id in deterministic_render_only:
                continue
            closure = ordered_for_run([spec.id])
            if not any(item.run.commands for item in closure):
                missing.append(spec.id)
        self.assertEqual(missing, [])

    def test_heavyweight_campaigns_have_real_run_commands(self) -> None:
        for spec_id in (
            "figure-07-real-lance-metrics",
            "figure-08-sift100m-concurrency-scaling",
            "figure-11-greedy-locality-shift",
        ):
            spec = get(spec_id)
            self.assertTrue(spec.run.commands, spec_id)
            runner = next(
                part
                for command in spec.run.commands
                for part in command.argv
                if part.endswith(".py")
            )
            self.assertTrue((ROOT / runner).is_file(), runner)

    def test_paper_campaigns_declare_public_preparation(self) -> None:
        for spec_id in (
            "figure-07-real-lance-metrics",
            "figure-08-sift100m-concurrency-scaling",
            "figure-11-greedy-locality-shift",
        ):
            spec = get(spec_id)
            self.assertTrue(spec.prepare.commands, spec_id)
            for command in spec.prepare.commands:
                scripts = [part for part in command.argv if part.endswith(".py")]
                self.assertEqual(len(scripts), 1, command.argv)
                self.assertTrue((ROOT / scripts[0]).is_file(), scripts[0])

    def test_public_array_consumers_declare_fresh_clone_preparation(self) -> None:
        expected = {
            "table-01-cluster-coarsening": {"sift1m"},
            "figure-09-qps-cost": {
                "sift1m", "glove", "deep1m", "openimage", "msong", "random1m128",
            },
            "figure-10-p-sweep-closedloop": {
                "sift1m", "glove", "deep1m", "openimage", "msong", "random1m128",
            },
            "figure-12-planner-overhead": {"sift1m", "glove"},
            "locality-aware-coaccess-gap": {
                "sift1m", "glove", "deep1m", "openimage", "msong", "random1m128",
            },
            "reorder-cost": {"sift1m"},
            "figure-14-sift1m-sensitivity": {"sift1m"},
        }
        for spec_id, datasets in expected.items():
            commands = [command.argv for command in get(spec_id).prepare.commands]
            declared = {
                command[command.index("--dataset") + 1]
                for command in commands
                if "prepare-dataset" in command and "--dataset" in command
            }
            self.assertEqual(declared, datasets, spec_id)
            self.assertTrue(
                all("--download" in command for command in commands),
                spec_id,
            )

    def test_requirements_identify_figure4_topology_gate(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "artifact",
                "requirements",
                "--only",
                "figure-04-object-store-io",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.stdout.strip().split()[-1], "1")


class PublicRepositoryHygieneTests(unittest.TestCase):
    def test_privacy_scanner_reads_sensitive_and_extensionless_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in (".env", "deploy.pem", "credentials"):
                path = root / name
                path.write_text("tracked text fixture\n", encoding="utf-8")
                self.assertIsNotNone(_read_text_if_probable(path), name)
            binary = root / "result.bin"
            binary.write_bytes(b"header\0payload")
            self.assertIsNone(_read_text_if_probable(binary))

    def test_privacy_scanner_resolves_python_constant_concatenation(self) -> None:
        bucket = f"private-{uuid.uuid4().hex}"
        token_tail = "A" * 16
        source = f'''\
BUCKET = {bucket!r}
URI = "s3://" + BUCKET
PARTS = ("person", "example.invalid")
EMAIL = chr(64).join(PARTS)
TOKEN = "AKIA" + {token_tail!r}
MESSAGE = f"endpoint={{URI}}"
'''
        resolved = _resolved_static_strings(source)
        values = [value for _line, value in resolved]
        self.assertIn(f"s3://{bucket}", values)
        self.assertTrue(any(value.endswith("@example.invalid") for value in values))
        self.assertTrue(any(value.startswith("AKIA") for value in values))
        self.assertIn(f"endpoint=s3://{bucket}", values)
        violations = [label for value in values for label in _privacy_violations(value)]
        self.assertIn("non-placeholder S3 bucket", violations)
        self.assertIn("email address", violations)
        self.assertIn("credential or private-key material", violations)

    def test_privacy_scanner_rejects_real_endpoints_and_personal_paths(self) -> None:
        identity = uuid.uuid4().hex
        private_uri = "s3://" + f"private-{identity}/paper/run.json"
        private_path = "/home/" + f"private-{identity}/project/result.json"
        example_email = f"person-{identity}" + chr(64) + "example.invalid"
        self.assertIn(
            "non-placeholder S3 bucket",
            _privacy_violations(private_uri),
        )
        self.assertIn(
            "personal absolute path",
            _privacy_violations(private_path),
        )
        self.assertIn("email address", _privacy_violations(example_email))
        example_token = f"sk-{uuid.uuid4().hex}"
        self.assertIn(
            "credential or private-key material",
            _privacy_violations(example_token),
        )
        self.assertEqual(_privacy_violations("s3://your-bucket/reproduction"), [])

    def test_privacy_scanner_rejects_private_github_repository_urls(self) -> None:
        owner = public_hygiene.PUBLIC_GIT_EMAIL.split("@", 1)[0]
        url = f"https://github.com/{owner}/private-artifact"
        self.assertIn("private GitHub repository URL", _privacy_violations(url))

    def test_privacy_scanner_allows_the_artifact_repository(self) -> None:
        owner = public_hygiene.PUBLIC_GIT_EMAIL.split("@", 1)[0]
        for path in ("WaRP", "WaRP.git", "WaRP/blob/main/planner/README.md"):
            with self.subTest(path=path):
                self.assertEqual(_privacy_violations(f"https://github.com/{owner}/{path}"), [])

    def test_privacy_scanner_rejects_other_repositories_with_the_same_prefix(self) -> None:
        owner = public_hygiene.PUBLIC_GIT_EMAIL.split("@", 1)[0]
        for name in ("WaRP-private", "WaRP.private", "WaRP_extra", "WaRP.git-private"):
            with self.subTest(name=name):
                self.assertIn(
                    "private GitHub repository URL",
                    _privacy_violations(f"https://github.com/{owner}/{name}"),
                )

    def test_opaque_identifiers_are_case_insensitive_and_physical_objects_are_explicit(self) -> None:
        number = 100 + int(uuid.uuid4().hex[:6], 16) % 900
        revision_id = f"F{number:03d}"
        stage_id = f"D{number:03d}"
        self.assertIsNotNone(OPAQUE_REVISION_ID.search(f'experiment="{revision_id}"'))
        self.assertIsNotNone(OPAQUE_STAGE_ID.search(f'flag="--{stage_id}-root"'))
        physical = f'"s{number:03d}_example_index.lance"'
        match = OPAQUE_STAGE_ID.search(physical)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertTrue(_is_physical_object_identifier(physical, match))

        mixed = physical + f" --{stage_id}-root"
        self.assertEqual(
            [match.group(0).lower() for match in _unallowlisted_matches(OPAQUE_STAGE_ID, mixed)],
            [stage_id.lower()],
        )
        self.assertEqual(_unallowlisted_matches(OPAQUE_REVISION_ID, "# noqa: F401"), [])

    def test_public_metadata_has_no_private_workflow_markers(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "artifact.checks.public_hygiene"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_source_export_uses_the_committed_public_file_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "PUBLIC_RELEASE_FILES.txt"
            payload = root / "README.md"
            payload.write_text("public artifact\n", encoding="utf-8")
            manifest.write_text(
                "PUBLIC_RELEASE_FILES.txt\nREADME.md\n",
                encoding="utf-8",
            )
            with mock.patch.object(public_hygiene, "ROOT", root), mock.patch.object(
                public_hygiene,
                "PUBLIC_FILE_MANIFEST",
                manifest,
            ), mock.patch.object(public_hygiene, "_exact_git_root", return_value=None):
                self.assertEqual(
                    public_hygiene.tracked_files(),
                    [manifest, payload],
                )

    def test_experiment_wrappers_do_not_require_git_metadata(self) -> None:
        wrappers = sorted(
            path
            for path in public_hygiene.tracked_files()
            if path.parent != ROOT
            and "experiments" in path.relative_to(ROOT).parts
            and path.name in {"run.sh", "render.sh"}
        )
        self.assertTrue(wrappers)
        for wrapper in wrappers:
            source = wrapper.read_text(encoding="utf-8")
            self.assertNotIn("rev-parse --show-toplevel", source, wrapper)
            self.assertIn('REPO_ROOT="$(cd "$(dirname "$0")/', source, wrapper)

    def test_history_gate_matches_the_current_branch_shape(self) -> None:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            root_result.returncode != 0
            or Path(root_result.stdout.strip()).resolve() != ROOT
        ):
            self.assertEqual(
                history_violations(),
                ["Git history is unavailable in this source export"],
            )
            return
        commit_count = int(
            subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        errors = history_violations()
        if commit_count == 1:
            self.assertEqual(errors, [])
        else:
            self.assertTrue(
                any("one clean root commit" in error for error in errors)
            )


class CliDependencyTests(unittest.TestCase):
    def test_prepare_dataset_rejects_arrays_that_impersonate_a_public_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "sift1m"
            output.mkdir()
            np.save(output / "base.npy", np.zeros((4, 2), dtype=np.float32))
            np.save(output / "queries.npy", np.zeros((3, 2), dtype=np.float32))
            (output / "meta.json").write_text(
                json.dumps(
                    {
                        "dataset": "sift1m",
                        "format": "clivf-npy-v1",
                        "metric": "euclidean",
                        "base": {"rows": 4, "dim": 2, "dtype": "float32", "path": "base.npy"},
                        "queries": {"rows": 3, "dim": 2, "dtype": "float32", "path": "queries.npy"},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "artifact",
                    "prepare-dataset",
                    "--dataset",
                    "sift1m",
                    "--out-dir",
                    str(output),
                    "--download",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("metadata base.rows differs", completed.stderr)

    def test_prepare_dataset_reuse_requires_canonical_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fixture"
            output.mkdir()
            np.save(output / "base.npy", np.zeros((4, 2), dtype=np.float32))
            np.save(output / "queries.npy", np.zeros((3, 2), dtype=np.float32))
            np.save(output / "gt_I.npy", np.zeros((3, 2), dtype=np.int32))
            (output / "meta.json").write_text(
                json.dumps(
                    {
                        "dataset": "sift1m",
                        "format": "clivf-npy-v1",
                        "metric": "euclidean",
                        "base": {"rows": 4, "dim": 2, "dtype": "float32", "path": "base.npy"},
                        "queries": {"rows": 3, "dim": 2, "dtype": "float32", "path": "queries.npy"},
                        "groundtruth": {"rows": 3, "k": 2, "dtype": "int32", "I_path": "gt_I.npy"},
                    }
                ),
                encoding="utf-8",
            )
            fixture_contract = {
                "metric": "euclidean",
                "base": {"rows": [4], "dim": 2, "dtype": "float32", "path": "base.npy"},
                "queries": {"rows": [3], "dim": 2, "dtype": "float32", "path": "queries.npy"},
                "groundtruth": {"rows": [3], "dim": 2, "dtype": "int32", "path": "gt_I.npy"},
            }
            with mock.patch.dict(DATASET_OUTPUT_CONTRACTS, {"sift1m": fixture_contract}):
                _validate_prepared_output(output, "sift1m")
                (output / "gt_I.npy").unlink()
                with self.assertRaises((OSError, ValueError)):
                    _validate_prepared_output(output, "sift1m")

    def test_show_parameters_exposes_canonical_paper_contract(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "artifact",
                "show",
                "figure-08-sift100m-concurrency-scaling",
                "--parameters",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["experiment"], "figure-08-sift100m-concurrency-scaling")
        self.assertEqual(payload["paper_parameters"]["query_concurrency_levels"], [16, 32, 48, 64, 96, 128])
        self.assertEqual(payload["paper_parameters"]["measured_queries"], 10_000)
        self.assertEqual(payload["reader_inputs"]["VECIO_S3_BUCKET"]["kind"], "infrastructure")
        self.assertEqual(payload["custom_parameter_policy"], "not-paper-reproduction")

        rejected = subprocess.run(
            [sys.executable, "-m", "artifact", "list", "--parameters"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("supported only with show", rejected.stderr)

    def test_every_registered_spec_exposes_a_public_parameter_contract(self) -> None:
        listed = subprocess.run(
            [sys.executable, "-m", "artifact", "list"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        for line in listed.stdout.splitlines():
            experiment_id = line.split("\t", 1)[0]
            completed = subprocess.run(
                [sys.executable, "-m", "artifact", "show", experiment_id, "--parameters"],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, f"{experiment_id}: {completed.stderr}")
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["experiment"], experiment_id)
            self.assertIn("paper_parameters", payload)
            self.assertIn("reader_inputs", payload)
            self.assertIn("custom_parameter_policy", payload)

    def test_resolved_contracts_include_measurement_semantics(self) -> None:
        closed_loop = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "artifact",
                    "show",
                    "figure-10-p-sweep-closedloop",
                    "--parameters",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )["paper_parameters"]
        self.assertEqual(closed_loop["request_budgets"], [16, 64])
        self.assertEqual(closed_loop["greedy_gap_kib"], [64, 256, 1024, 4096])
        self.assertEqual(closed_loop["max_iop_size_bytes"], 512 * 1024**3)

        planner = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "artifact",
                    "show",
                    "figure-12-planner-overhead",
                    "--parameters",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )["paper_parameters"]
        self.assertEqual(planner["jobs"], 1)
        self.assertEqual(planner["cost_model"]["breaks_bytes"], [1_048_576])

        transfer = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "artifact",
                    "show",
                    "figure-11-greedy-locality-shift",
                    "--parameters",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )["paper_parameters"]
        self.assertEqual(transfer["msong"]["query_concurrency"], 4)
        self.assertEqual(transfer["msong"]["io_limit"], 64)

        sensitivity = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "artifact",
                    "show",
                    "figure-14-sift1m-sensitivity",
                    "--parameters",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )["paper_parameters"]
        self.assertEqual(sensitivity["balanced_p"], 16)
        self.assertEqual(sensitivity["settings"][0]["request_budgets"], [8, 16])

    def test_prepare_dry_run_exposes_real_asset_builder(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                artifact_main(["prepare", "figure-07-real-lance-metrics", "--dry-run"]),
                0,
            )
        rendered = output.getvalue()
        self.assertIn("prepare_matched_datasets.py", rendered)

    def test_run_dry_run_includes_dependencies_before_owner(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                artifact_main(["run", "figure-13-reordering-methods", "--dry-run"]),
                0,
            )
        rendered = output.getvalue()
        positions = [
            rendered.index("runners/reorder-cost/run_data.py"),
            rendered.index("runners/locality-aware-coaccess-gap/run_data.py"),
            rendered.index("runners/figure-13-reordering-methods/run_data.py"),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_full_render_uses_safe_latest_slot(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                artifact_main(["render", "figure-07-real-lance-metrics", "--dry-run"]),
                0,
            )
        rendered = output.getvalue()
        self.assertIn("artifact rebuild", rendered)
        self.assertIn("--data-slot latest", rendered)

    def test_light_render_uses_safe_paper_slot(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                artifact_main(
                    ["render", "figure-07-real-lance-metrics", "--profile", "light", "--dry-run"]
                ),
                0,
            )
        self.assertIn("--data-slot paper", output.getvalue())

    def test_run_all_plan_never_runs_repository_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / "python3"
            log = root / "commands.log"
            fake_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$WARP_TEST_COMMAND_LOG\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{root}:{env['PATH']}"
            env["VECIO_PYTHON"] = "python3"
            env["WARP_TEST_COMMAND_LOG"] = str(log)
            subprocess.run(
                ["/bin/bash", "scripts/run_all.sh", "--plan", "--only", "reorder-cost"],
                cwd=ROOT,
                check=True,
                env=env,
                text=True,
                capture_output=True,
            )
            commands = log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("artifact orchestrate" in command for command in commands))
        self.assertFalse(any("artifact check" in command for command in commands))

    def test_run_all_loads_rustup_cargo_from_a_fresh_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_bin = root / "initial-bin"
            cargo_bin = root / "cargo-bin"
            cargo_home = root / ".cargo"
            lance_root = root / "lance"
            initial_bin.mkdir()
            cargo_bin.mkdir()
            cargo_home.mkdir()
            lance_root.mkdir()

            fake_python = initial_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *'artifact requirements'*) printf '1 0 1 0 0\\n' ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            cargo_log = root / "cargo.log"
            fake_cargo = cargo_bin / "cargo"
            fake_cargo.write_text(
                "#!/bin/sh\n"
                "printf '%s\\t%s\\n' \"$PWD\" \"$*\" >> \"$WARP_TEST_CARGO_LOG\"\n",
                encoding="utf-8",
            )
            fake_cargo.chmod(0o755)
            (cargo_home / "env").write_text(
                'export PATH="$WARP_TEST_CARGO_BIN:$PATH"\n',
                encoding="utf-8",
            )

            env = dict(os.environ)
            env["HOME"] = str(root)
            env["PATH"] = f"{initial_bin}:/usr/bin:/bin"
            env["VECIO_PYTHON"] = "python3"
            env["WARP_TEST_CARGO_BIN"] = str(cargo_bin)
            env["WARP_TEST_CARGO_LOG"] = str(cargo_log)
            subprocess.run(
                [
                    "/bin/bash",
                    "scripts/run_all.sh",
                    "--only",
                    "real-s3-tiny-request-fanout-sanity",
                    "--lance-root",
                    str(lance_root),
                    "--no-check",
                    "--no-preflight",
                    "--no-prepare",
                    "--no-render",
                ],
                cwd=ROOT,
                check=True,
                env=env,
                text=True,
                capture_output=True,
            )
            cargo_commands = cargo_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(cargo_commands), 2)
        for command in cargo_commands:
            cargo_cwd, cargo_args = command.split("\t", maxsplit=1)
            self.assertEqual(cargo_cwd, str(lance_root))
            self.assertTrue(cargo_args.startswith("build "))


if __name__ == "__main__":
    unittest.main()
