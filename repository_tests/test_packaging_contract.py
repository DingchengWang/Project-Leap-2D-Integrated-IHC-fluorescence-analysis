from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "Project Leap 2D (8-23-26)"
PAYLOAD = REPO_ROOT / "packaging" / "payload_baseline.json"

REQUIRED_ROOT_FILES = (
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "LICENSES",
)
FORBIDDEN_ROOT_RELATIVE_PATHS = frozenset(
    (
        "validation/release_validation.json",
        "validation/release_validation",
        "validation/release_validation.txt",
    )
)
FORBIDDEN_PATH_MARKERS = (
    "pre_change_reference",
    "post_change_real_sample_validation",
    "audit records",
    "audit_records",
    "historical_results",
    "work_log",
    "worklog",
)
FORBIDDEN_README_PHRASE = "historical mature egfp"
FORBIDDEN_ACCOUNT_TOKEN = str(Path.home())

_TEXT_EXTENSIONS = {
    ".cfg",
    ".command",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
_BINARY_EXTENSIONS = {
    ".ipynb",
    ".parquet",
    ".pt",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tar",
    ".whl",
    ".zip",
}


def _load_payload() -> dict:
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_exempt_relative_path(relative_posix: str, exempt_prefixes: set[str]) -> bool:
    if not relative_posix:
        return False
    return any(
        relative_posix == prefix or relative_posix.startswith(prefix + "/")
        for prefix in exempt_prefixes
    )


def _git_mode_from_path(path: Path) -> int:
    mode = path.lstat().st_mode
    return int(stat.S_IFREG | stat.S_IMODE(mode))


def _normalize_git_mode(raw_mode: int | str) -> int:
    if isinstance(raw_mode, int):
        return raw_mode
    return int(str(raw_mode), 8)


class PackagingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_payload()
        cls.payload_files = cls.payload["files"]
        cls.required_dirs = set(cls.payload["required_directories"])
        cls.executable_files = set(cls.payload["executable_files"])

    def test_01_payload_schema_and_counts(self) -> None:
        self.assertEqual(self.payload["file_count"], len(self.payload_files))
        self.assertEqual(len(self.payload["executable_files"]), 5)

    def test_02_no_nested_git_directory_within_packaging_payload(self) -> None:
        self.assertFalse((PACKAGE_ROOT / ".git").exists())
        for dirpath, dirnames, filenames in os.walk(
            PACKAGE_ROOT, topdown=True, followlinks=False
        ):
            current = Path(dirpath)
            for name in list(dirnames) + filenames:
                path = current / name
                self.assertNotEqual(
                    path.name,
                    ".git",
                    f"nested .git found at {path.relative_to(PACKAGE_ROOT)}",
                )

    def test_03_no_symlink_allowed(self) -> None:
        links = []
        for dirpath, dirnames, filenames in os.walk(
            REPO_ROOT, topdown=True, followlinks=False
        ):
            current = Path(dirpath)
            for name in list(dirnames) + filenames:
                path = current / name
                if path.is_symlink():
                    links.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(links, [])

    def test_04_prepare_workspace_command_exists_and_executable(self) -> None:
        prepare_script = REPO_ROOT / "prepare_workspace.command"
        self.assertTrue(prepare_script.exists())
        self.assertTrue(prepare_script.is_file())
        self.assertTrue(os.access(prepare_script, os.X_OK))

    def test_05_legal_artifacts_exist(self) -> None:
        for item in REQUIRED_ROOT_FILES:
            self.assertTrue((REPO_ROOT / item).exists(), f"missing {item}")

        license_dir = REPO_ROOT / "LICENSES"
        self.assertTrue(license_dir.is_dir())
        license_files = sorted(
            p.name for p in license_dir.iterdir() if p.is_file()
        )
        self.assertGreaterEqual(len(license_files), 2, license_files)
        lower_names = [name.lower() for name in license_files]
        self.assertTrue(any("instanseg" in name for name in lower_names))
        self.assertTrue(any("cellpose" in name for name in lower_names))

    def test_06_inner_package_manifest_files_match_payload(self) -> None:
        expected_files = set(self.payload_files.keys())
        actual_files = set()
        for dirpath, dirnames, filenames in os.walk(
            PACKAGE_ROOT, topdown=True, followlinks=False
        ):
            current = Path(dirpath)
            relative_dir = (
                current.relative_to(PACKAGE_ROOT).as_posix()
                if current != PACKAGE_ROOT
                else ""
            )
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not _is_exempt_relative_path(
                    f"{relative_dir}/{dirname}" if relative_dir else dirname,
                    self.required_dirs,
                )
            ]
            for name in filenames:
                path = current / name
                rel = (
                    f"{relative_dir}/{name}"
                    if relative_dir
                    else name
                )
                if not _is_exempt_relative_path(rel, self.required_dirs):
                    self.assertTrue(path.is_file(), path)
                    actual_files.add(rel)

        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_07_inner_package_manifest_metadata(self) -> None:
        for relative_path, expected in sorted(self.payload_files.items()):
            path = PACKAGE_ROOT / relative_path
            self.assertTrue(path.exists(), relative_path)
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(path.stat().st_size, expected["bytes"], relative_path)
            self.assertEqual(_sha256(path), expected["sha256"], relative_path)
            self.assertEqual(
                _git_mode_from_path(path),
                _normalize_git_mode(expected["git_mode"]),
                relative_path,
            )

        for relative_path in sorted(self.payload["executable_files"]):
            path = PACKAGE_ROOT / relative_path
            self.assertTrue(path.exists(), relative_path)
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(
                _git_mode_from_path(path),
                _normalize_git_mode("100755"),
                relative_path,
            )

    def test_08_no_private_history_artifacts_in_candidate(self) -> None:
        forbidden_paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(
            REPO_ROOT, topdown=True, followlinks=False
        ):
            current = Path(dirpath)
            for name in list(dirnames) + list(filenames):
                path = current / name
                rel = path.relative_to(REPO_ROOT).as_posix().lower()

                if rel in FORBIDDEN_ROOT_RELATIVE_PATHS:
                    forbidden_paths.append(rel)
                    continue

                if any(
                    marker in rel
                    for marker in FORBIDDEN_PATH_MARKERS
                    if marker
                ):
                    forbidden_paths.append(rel)

        self.assertEqual(
            forbidden_paths,
            [],
            f"found private/artifact files that should not be in candidate: {forbidden_paths}",
        )

    def test_09_no_readme_internal_historical_sample_evidence_claim(self) -> None:
        violating_readmes: list[str] = []
        for readme in REPO_ROOT.rglob("README*"):
            if not readme.is_file():
                continue
            try:
                text = readme.read_text(encoding="utf-8")
            except OSError:
                continue
            if FORBIDDEN_README_PHRASE in text.lower():
                violating_readmes.append(readme.relative_to(REPO_ROOT).as_posix())

        self.assertEqual(
            violating_readmes,
            [],
            f"found README text containing historical real-sample evidence: {violating_readmes}",
        )

    def test_10_no_local_account_path_string(self) -> None:
        bad_files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(
            REPO_ROOT, topdown=True, followlinks=False
        ):
            current = Path(dirpath)
            if current == REPO_ROOT:
                dirnames[:] = [
                    dirname for dirname in dirnames if dirname != ".git"
                ]
                filenames = [name for name in filenames if name != ".git"]
            for name in filenames:
                path = current / name
                if path.name == ".DS_Store":
                    continue
                if not path.is_file():
                    continue

                rel = path.relative_to(REPO_ROOT)
                if rel.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                if rel.suffix.lower() not in _TEXT_EXTENSIONS and len(rel.suffix) > 0:
                    continue

                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if FORBIDDEN_ACCOUNT_TOKEN in text:
                    bad_files.append(rel.as_posix())

        self.assertEqual(
            bad_files,
            [],
            f"found private account string in candidate files: {bad_files}",
        )


if __name__ == "__main__":
    unittest.main()
