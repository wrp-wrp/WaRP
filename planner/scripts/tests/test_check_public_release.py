from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_public_release.py"
SPEC = importlib.util.spec_from_file_location("check_public_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class TreeViolationsTest(unittest.TestCase):
    def violations(self, filename: str, content: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / filename
            path.write_text(content, encoding="utf-8")
            with (
                mock.patch.object(CHECK, "ROOT", root),
                mock.patch.object(CHECK, "_tracked_paths", return_value=[path]),
            ):
                return CHECK.tree_violations()

    def test_env_variant_is_rejected_by_filename(self) -> None:
        self.assertTrue(self.violations(".env.production", "MODE=production\n"))

    def test_generic_secret_assignment_is_rejected(self) -> None:
        value = "-".join(("private", "token", "value"))
        for key_parts in (("API", "TOKEN"), ("TOKEN",), ("PASSWORD",)):
            with self.subTest(key_parts=key_parts):
                key = "_".join(key_parts)
                self.assertTrue(self.violations("settings.txt", f"{key}={value}\n"))

    def test_private_resource_assignment_is_rejected(self) -> None:
        cases = (
            (("S3", "BUCKET"), "-".join(("private", "bucket", "name"))),
            (("VECIO", "S3", "BUCKET"), "-".join(("private", "bucket", "name"))),
            (("URI",), "s3:" + "//" + "private-bucket/path"),
        )
        for key_parts, value in cases:
            with self.subTest(key_parts=key_parts):
                key = "_".join(key_parts)
                self.assertTrue(self.violations("settings.txt", f"{key}={value}\n"))

    def test_private_github_repository_url_is_rejected(self) -> None:
        url = "https://github.com/" + "wrp-wrp/private-artifact"
        self.assertTrue(self.violations("README.md", url + "\n"))

    def test_artifact_repository_urls_are_allowed(self) -> None:
        owner = CHECK.RELEASE_IDENTITY.split("<", 1)[1].split("@", 1)[0]
        for path in ("WaRP", "WaRP.git", "WaRP/blob/main/planner/README.md"):
            with self.subTest(path=path):
                self.assertEqual(
                    self.violations("README.md", f"https://github.com/{owner}/{path}\n"),
                    [],
                )

    def test_other_repository_names_with_the_artifact_prefix_are_rejected(self) -> None:
        owner = CHECK.RELEASE_IDENTITY.split("<", 1)[1].split("@", 1)[0]
        for name in ("WaRP-private", "WaRP.private", "WaRP_extra", "WaRP.git-private"):
            with self.subTest(name=name):
                self.assertTrue(
                    self.violations("README.md", f"https://github.com/{owner}/{name}\n")
                )

    def test_placeholders_are_allowed(self) -> None:
        self.assertEqual(
            self.violations(
                "example.env",
                "API_TOKEN=<your-token>\nS3_BUCKET=${PUBLIC_BUCKET}\n",
            ),
            [],
        )

    def test_source_export_uses_the_committed_public_file_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "PUBLIC_RELEASE_FILES.txt"
            payload = root / "README.md"
            payload.write_text("public planner\n", encoding="utf-8")
            manifest.write_text(
                "PUBLIC_RELEASE_FILES.txt\nREADME.md\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(CHECK, "ROOT", root),
                mock.patch.object(CHECK, "PUBLIC_FILE_MANIFEST", manifest),
                mock.patch.object(CHECK, "_exact_git_root", return_value=None),
            ):
                self.assertEqual(CHECK._tracked_paths(), [manifest, payload])


class HistoryViolationsTest(unittest.TestCase):
    def git_result(self, *args: str) -> str:
        command = tuple(args)
        if command == ("status", "--porcelain"):
            return ""
        if command == ("rev-list", "--count", "HEAD"):
            return "1\n"
        if command == ("show", "-s", "--format=%P%n%an <%ae>%n%cn <%ce>", "HEAD"):
            return (
                "\nWaRP Authors <wrp-wrp@users.noreply.github.com>\n"
                "WaRP Authors <wrp-wrp@users.noreply.github.com>\n"
            )
        if command == ("log", "-1", "--format=%B", "HEAD"):
            return "warp-planner public release\n"
        self.fail(f"unexpected git command: {command}")

    def test_clean_root_release_passes(self) -> None:
        with mock.patch.object(
            CHECK, "_exact_git_root", return_value=CHECK.ROOT
        ), mock.patch.object(CHECK, "_git", side_effect=self.git_result):
            self.assertEqual(CHECK.history_violations(), [])

    def test_multiple_commits_are_rejected(self) -> None:
        def result(*args: str) -> str:
            if tuple(args) == ("rev-list", "--count", "HEAD"):
                return "2\n"
            return self.git_result(*args)

        with mock.patch.object(
            CHECK, "_exact_git_root", return_value=CHECK.ROOT
        ), mock.patch.object(CHECK, "_git", side_effect=result):
            self.assertIn(
                "release history must contain exactly one root commit",
                CHECK.history_violations(),
            )

    def test_commit_message_leak_is_rejected(self) -> None:
        private_path = "/".join(("", "home", "person", "workspace")) + "/"

        def result(*args: str) -> str:
            if tuple(args) == ("log", "-1", "--format=%B", "HEAD"):
                return f"release built from {private_path}\n"
            return self.git_result(*args)

        with mock.patch.object(
            CHECK, "_exact_git_root", return_value=CHECK.ROOT
        ), mock.patch.object(CHECK, "_git", side_effect=result):
            self.assertTrue(
                any(
                    "release commit message" in violation
                    for violation in CHECK.history_violations()
                )
            )


if __name__ == "__main__":
    unittest.main()
