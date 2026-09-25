from __future__ import annotations

import hashlib
import importlib.util
import io
import shutil
import tempfile
import subprocess
import zipfile
from unittest.mock import patch
import json
import os
import stat
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "Project Leap 2D V1.0.1"
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
        self.assertEqual(self.payload["file_count"], 95)
        self.assertEqual(self.payload["project_version"], "1.0.1")
        self.assertEqual(self.payload["release_folder"], PACKAGE_ROOT.name)
        self.assertEqual(self.required_dirs, {"Sample Image", "Result", "Analysis Package/Run State"})
        self.assertEqual(self.payload["file_count"], len(self.payload_files))
        self.assertEqual(len(self.payload["executable_files"]), 6)

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
                if path.name != ".DS_Store" and not _is_exempt_relative_path(rel, self.required_dirs):
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


_BUILDER_SPEC = importlib.util.spec_from_file_location("build_release_zip", REPO_ROOT / "tools/build_release_zip.py")
builder = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(builder)


class ReleaseBuilderTests(unittest.TestCase):
    def test_baseline_and_actual_package_are_valid(self):
        baseline, entries = builder._load_baseline()
        self.assertEqual(len(builder._validate_inner_payload(baseline, entries)), 95)
        self.assertEqual((PACKAGE_ROOT / "Analysis Package/VERSION").read_text().strip(), "1.0.1")

    def test_archives_are_deterministic_and_include_licenses_and_empty_directories(self):
        baseline, entries = builder._load_baseline()
        directories, files = builder._archive_plan(baseline, entries)
        with tempfile.TemporaryDirectory(prefix="leap-release-build-test-") as temp:
            first = Path(temp) / "first.zip"
            second = Path(temp) / "second.zip"
            self.assertEqual(builder._build_zip(first, directories, files), builder._build_zip(second, directories, files))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                root = "Project Leap 2D V1.0.1 Distribution/"
                names = set(archive.namelist())
                for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "install_macos.command", "INSTALL_中文.md", "INSTALL_English.md", "payload_sha256.txt"):
                    self.assertIn(root + name, names)
                for name in builder.RELEASE_LICENSE_FILES:
                    self.assertIn(root + "LICENSES/" + name, names)
                for name in ("Sample Image", "Result", "Analysis Package/Run State"):
                    self.assertIn(root + PACKAGE_ROOT.name + "/" + name + "/", names)
                self.assertFalse(any(".DS_Store" in name or "/.git/" in name for name in names))
                self.assertEqual(archive.read(root + "payload_sha256.txt"), builder._payload_checksums(entries))
                self.assertEqual(archive.getinfo(root + "install_macos.command").external_attr >> 16, stat.S_IFREG | 0o755)
            with self.assertRaises(builder.ReleaseBuildError):
                builder._build_zip(first, directories, files)

    def test_build_outputs_archive_digest_and_program_manifests(self):
        with tempfile.TemporaryDirectory(prefix="leap-release-artifacts-test-") as temp:
            output = Path(temp) / "Project-Leap-2D-V1.0.1.zip"
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                self.assertEqual(builder.main(["--output", str(output)]), 0)
            manifest_path = output.parent / "Project-Leap-2D-V1.0.1.manifest.json"
            payload_path = output.parent / "payload_sha256.txt"
            self.assertEqual(set(output.parent.iterdir()), {output, manifest_path, payload_path})
            self.assertIn(f"SHA-256: {_sha256(output)}\n", stdout.getvalue())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["files"]), 95)
            checksums = {}
            for line in payload_path.read_text().splitlines():
                digest, relative_path = line.split("  ", 1)
                self.assertNotIn(relative_path, checksums)
                checksums[relative_path] = digest
            self.assertEqual(set(checksums), set(manifest["files"]))
            with zipfile.ZipFile(output) as archive:
                root = "Project Leap 2D V1.0.1 Distribution/"
                self.assertEqual(archive.read(root + "payload_sha256.txt"), payload_path.read_bytes())
                for relative_path, metadata in manifest["files"].items():
                    archived = archive.read(root + PACKAGE_ROOT.name + "/" + relative_path)
                    digest = hashlib.sha256(archived).hexdigest()
                    self.assertEqual(digest, checksums[relative_path], relative_path)
                    self.assertEqual(digest, metadata["sha256"], relative_path)
                    self.assertEqual(len(archived), metadata["size"], relative_path)

    def test_workspace_directories_reject_files_subdirectories_and_links(self):
        with tempfile.TemporaryDirectory(prefix="leap-release-empty-test-") as temp:
            root = Path(temp)
            builder._validate_empty_workspace_directories(root)
            for relative in ("Sample Image", "Result", "Analysis Package/Run State"):
                directory = root / relative
                directory.mkdir(parents=True)
                metadata = directory / ".DS_Store"
                metadata.write_bytes(b"Finder metadata")
                builder._validate_empty_workspace_directories(root)
                for kind in ("file", "directory", "symlink", "finder_symlink"):
                    with self.subTest(relative=relative, kind=kind):
                        unexpected = directory / (".DS_Store" if kind == "finder_symlink" else "unexpected")
                        if kind == "file":
                            unexpected.write_bytes(b"private data")
                        elif kind == "directory":
                            unexpected.mkdir()
                        elif kind == "finder_symlink":
                            metadata.unlink()
                            metadata.symlink_to(root / "outside")
                        else:
                            unexpected.symlink_to(root / "outside")
                        with self.assertRaises(builder.ReleaseBuildError):
                            builder._validate_empty_workspace_directories(root)
                        if unexpected.is_dir():
                            unexpected.rmdir()
                        else:
                            unexpected.unlink()

    def test_public_manifest_matches_program_repair_contract(self):
        _, entries = builder._load_baseline()
        manifest = json.loads(builder._public_manifest(entries))
        self.assertEqual(manifest["version"], "1.0.1")
        self.assertEqual(manifest["package_name"], PACKAGE_ROOT.name)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["files"]), 95)
        for entry in entries:
            self.assertEqual(manifest["files"][entry.relative_path], {
                "sha256": entry.sha256, "size": entry.size, "mode": stat.S_IMODE(entry.mode)
            })
        self.assertEqual(hashlib.sha256(builder._payload_checksums(entries)).hexdigest(), _load_payload()["immutable_baseline_sha256"])

    def test_changed_program_bytes_permissions_missing_and_extra_files_are_rejected(self):
        baseline, entries = builder._load_baseline()
        with tempfile.TemporaryDirectory(prefix="leap-release-tamper-test-") as temp:
            fake_repo = Path(temp)
            package = fake_repo / PACKAGE_ROOT.name
            shutil.copytree(PACKAGE_ROOT, package)
            target = package / "Manual Command.txt"
            original = target.read_bytes()
            with patch.object(builder, "REPOSITORY_ROOT", fake_repo):
                for kind in ("bytes", "mode", "missing", "extra", "symlink"):
                    with self.subTest(kind=kind):
                        if kind == "bytes":
                            target.write_bytes(b"X" + original[1:])
                        elif kind == "mode":
                            target.chmod(0o755)
                        elif kind == "missing":
                            target.unlink()
                        elif kind == "extra":
                            (package / "unreviewed.py").write_text("unexpected")
                        else:
                            target.unlink()
                            target.symlink_to(PACKAGE_ROOT / "Manual Command.txt")
                        with self.assertRaises(builder.ReleaseBuildError):
                            builder._validate_inner_payload(baseline, entries)
                        if target.is_symlink():
                            target.unlink()
                        target.write_bytes(original)
                        target.chmod(0o644)
                        (package / "unreviewed.py").unlink(missing_ok=True)

    def test_wrong_version_is_rejected(self):
        baseline, entries = builder._load_baseline()
        with tempfile.TemporaryDirectory(prefix="leap-release-version-test-") as temp:
            fake_repo = Path(temp)
            version = fake_repo / PACKAGE_ROOT.name / "Analysis Package/VERSION"
            version.parent.mkdir(parents=True)
            version.write_text("9.9.9\n")
            with patch.object(builder, "REPOSITORY_ROOT", fake_repo):
                with self.assertRaisesRegex(builder.ReleaseBuildError, "VERSION"):
                    builder._validate_inner_payload(baseline, entries)

    def test_unsafe_manifest_paths_and_runtime_entries_are_rejected(self):
        for value in ("../escape", "/absolute", "foo//bar", "foo/../bar", "foo\\bar", "bad\npath"):
            with self.subTest(value=value):
                with self.assertRaises(builder.ReleaseBuildError):
                    builder._safe_relative_path(value, field="test")
        for value in ("Sample Image/private.tif", "Result/private.xlsx", "Analysis Package/Run State/recovery.json"):
            self.assertFalse(builder._is_controlled_path(value))

    def test_generated_outputs_are_never_silently_replaced(self):
        with tempfile.TemporaryDirectory(prefix="leap-release-generated-test-") as temp:
            output = Path(temp) / "manifest.json"
            builder._write_generated_output(output, b"expected")
            builder._write_generated_output(output, b"expected")
            with self.assertRaises(builder.ReleaseBuildError):
                builder._write_generated_output(output, b"different")
            self.assertEqual(output.read_bytes(), b"expected")
            link = Path(temp) / "link"
            link.symlink_to(output)
            with self.assertRaises(builder.ReleaseBuildError):
                builder._write_generated_output(link, b"expected")

WORK = Path(__file__).resolve().parents[1]
INSTALLER = WORK / "distribution" / "install_macos.command"
PACKAGE_NAME = "Project Leap 2D V1.0.1"
MAINTENANCE = Path("Analysis Package/project_leap_2d/maintenance/manage_environment.command")
REQUIRED_DIRECTORIES = ("Sample Image", "Result", "Analysis Package", "Analysis Package/Run State", "README")


@unittest.skipUnless(all(Path(path).exists() for path in ("/usr/bin/ditto", "/bin/zsh", "/usr/bin/sandbox-exec")), "macOS tools and network sandbox required")
class DistributionInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="project-leap-distribution-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "download bundle"
        self.bundle.mkdir()
        self.installer = self.bundle / "install_macos.command"
        shutil.copy2(INSTALLER, self.installer)
        self.source = self.bundle / PACKAGE_NAME
        self.entry = self.source / MAINTENANCE
        self.entry.parent.mkdir(parents=True)
        for required_directory in REQUIRED_DIRECTORIES:
            (self.source / required_directory).mkdir(parents=True, exist_ok=True)
        self.entry.write_text(
            "#!/bin/zsh\nset -euo pipefail\n"
            "[[ $# == 0 ]] || exit 89\n"
            "[[ $PROJECT_LEAP_SUPPORT_DIR == $PROJECT_LEAP_TEST_ROOT/* ]] || exit 90\n"
            "/bin/mkdir -p \"$PROJECT_LEAP_SUPPORT_DIR\"\n"
            "print -r -- \"${0:A}\" > \"$PROJECT_LEAP_SUPPORT_DIR/maintenance.invoked\"\n"
            "exit \"${PROJECT_LEAP_TEST_EXIT:-0}\"\n"
        )
        self.entry.chmod(0o755)
        self.launcher = self.source / "Run Analysis.command"
        self.launcher.write_text(
            '#!/bin/zsh\nprint -r -- unexpected > "$PROJECT_LEAP_TEST_ROOT/analysis.unexpected"\nexit 87\n'
        )
        self.launcher.chmod(0o755)
        self.repair = self.source / "Repair.command"
        self.repair.write_text(
            '#!/bin/zsh\nprint -r -- unexpected > "$PROJECT_LEAP_TEST_ROOT/repair.unexpected"\nexit 88\n'
        )
        self.repair.chmod(0o755)
        for relative in ("README/README EN.md", "README/README CN.md", "Manual Command.txt"):
            (self.source / relative).write_text("local fixture documentation\n")
        (self.source / "Analysis Package/VERSION").write_text("1.0.1\n")
        self.data = self.source / "Analysis Package/project_leap_2d/example data.txt"
        self.data.write_text("small fixture, no real software or downloads\n")
        (self.source / "Analysis Package/.fixture_metadata").write_text("dotfiles must be verified\n")
        self.manifest = self.bundle / "payload_sha256.txt"
        self.write_manifest()
        self.destination_parent = self.root / "isolated target"
        self.destination_parent.mkdir()
        self.destination = self.destination_parent / PACKAGE_NAME
        self.support = self.root / "isolated support"
        self.fake_home = self.root / "isolated home"
        self.fake_home.mkdir()
        self.env = dict(os.environ)
        self.env.update(
            HOME=str(self.fake_home),
            ZDOTDIR=str(self.fake_home),
            PROJECT_LEAP_SUPPORT_DIR=str(self.support),
            PROJECT_LEAP_TEST_ROOT=str(self.root),
        )
        self.env.pop("PROJECT_LEAP_TEST_EXIT", None)

    def write_manifest(self):
        self.manifest.write_text(
            "".join(
                f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(self.source).as_posix()}\n"
                for file in sorted(self.source.rglob("*"))
                if file.is_file()
            )
        )

    def run_installer(self, *extra, use_default=False):
        args = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
                "/bin/zsh", str(self.installer)]
        if not use_default:
            args += ["--destination", str(self.destination)]
        args += extra
        result = subprocess.run(args, env=self.env, text=True, capture_output=True, timeout=30)
        self.assertFalse(list(self.destination_parent.glob(".project-leap-v1.0.1.*")), result.stderr)
        self.assertFalse((self.root / "analysis.unexpected").exists(), result.stdout)
        self.assertFalse((self.root / "repair.unexpected").exists(), result.stdout)
        return result

    def assert_stopped_before_deployment(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("INSTALLATION COMPLETE", result.stdout)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.support.exists())

    def test_success_copies_complete_package_and_runs_deployed_maintenance(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("INSTALLATION COMPLETE", result.stdout)
        self.assertEqual((self.destination / self.data.relative_to(self.source)).read_bytes(), self.data.read_bytes())
        self.assertEqual((self.destination / "Analysis Package/.fixture_metadata").read_bytes(), (self.source / "Analysis Package/.fixture_metadata").read_bytes())
        self.assertTrue(os.access(self.destination / self.launcher.name, os.X_OK))
        self.assertTrue(os.access(self.destination / self.repair.name, os.X_OK))
        self.assertIn("Open Run Analysis.command", result.stdout)
        self.assertNotIn("run_project_leap_2d.command", result.stdout)
        invoked = (self.support / "maintenance.invoked").read_text().strip()
        self.assertEqual(Path(invoked), (self.destination / MAINTENANCE).resolve())
        self.assertTrue(self.source.exists())
        self.assertTrue(self.installer.exists())
        self.assertFalse((self.destination / "payload_sha256.txt").exists())
        self.assertEqual({path.name for path in self.destination.iterdir()}, {
            "Sample Image", "Result", "Analysis Package", "README",
            "Run Analysis.command", "Repair.command", "Manual Command.txt",
        })
        for required_directory in REQUIRED_DIRECTORIES:
            self.assertTrue((self.destination / required_directory).is_dir())

    def test_default_destination_uses_only_isolated_home(self):
        desktop = self.fake_home / "Desktop"
        desktop.mkdir()
        result = self.run_installer(use_default=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((desktop / PACKAGE_NAME / MAINTENANCE).exists())
        self.assertFalse(self.destination.exists())

    def test_dry_run_does_not_copy_or_invoke_maintenance(self):
        before = sorted(str(file.relative_to(self.root)) for file in self.root.rglob("*"))
        result = self.run_installer("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN PASSED", result.stdout)
        after = sorted(str(file.relative_to(self.root)) for file in self.root.rglob("*"))
        self.assertEqual(before, after)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.support.exists())

    def test_failed_maintenance_retains_package_and_reports_repair(self):
        self.env["PROJECT_LEAP_TEST_EXIT"] = "37"
        result = self.run_installer()
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertTrue((self.destination / MAINTENANCE).exists())
        self.assertIn("retained", result.stderr)
        self.assertIn("./Repair.command", result.stderr)
        self.assertNotIn("--repair-environment", result.stderr)
        self.assertNotIn("run_project_leap_2d.command", result.stderr)
        self.assertIn("Installation is not complete", result.stderr)
        self.assertNotIn("INSTALLATION COMPLETE", result.stdout)
        self.assertTrue(self.source.exists())

    def test_existing_directory_preserves_user_data(self):
        self.destination.mkdir()
        marker = self.destination / "existing user data.txt"
        marker.write_text("keep exactly\n")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "keep exactly\n")
        self.assertEqual(list(self.destination.iterdir()), [marker])
        self.assertFalse(self.support.exists())

    def test_existing_file_is_not_overwritten(self):
        self.destination.write_text("existing file\n")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.destination.read_text(), "existing file\n")
        self.assertFalse(self.support.exists())

    def test_existing_target_symlink_is_not_followed(self):
        self.destination.symlink_to(self.support, target_is_directory=True)
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.destination.is_symlink())
        self.assertFalse(self.support.exists())

    def test_corrupt_payload_stops_before_copy(self):
        self.data.write_text("tampered\n")
        result = self.run_installer()
        self.assert_stopped_before_deployment(result)
        self.assertIn("checksum mismatch", result.stderr)

    def test_dry_run_also_checks_checksums(self):
        self.data.write_text("tampered\n")
        self.assert_stopped_before_deployment(self.run_installer("--dry-run"))

    def test_missing_payload_file_stops_before_copy(self):
        self.data.unlink()
        result = self.run_installer()
        self.assert_stopped_before_deployment(result)
        self.assertIn("missing", result.stderr)

    def test_unlisted_dotfile_stops_before_copy(self):
        (self.source / ".unlisted").write_text("unexpected\n")
        result = self.run_installer()
        self.assert_stopped_before_deployment(result)
        self.assertIn("Unlisted file", result.stderr)

    def test_unlisted_finder_metadata_is_preserved_in_source_and_omitted_from_target(self):
        metadata_files = [self.source / ".DS_Store", self.source / "Sample Image" / ".DS_Store"]
        for metadata_file in metadata_files:
            metadata_file.write_bytes(b"Finder fixture metadata")
        dry_run = self.run_installer("--dry-run")
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.support.exists())
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        for metadata_file in metadata_files:
            self.assertEqual(metadata_file.read_bytes(), b"Finder fixture metadata")
            self.assertFalse((self.destination / metadata_file.relative_to(self.source)).exists())

    def test_unlisted_finder_metadata_symlink_is_rejected(self):
        (self.source / ".DS_Store").symlink_to(self.data)
        self.assert_stopped_before_deployment(self.run_installer())

    def test_missing_required_directories_are_rejected(self):
        for required_directory in REQUIRED_DIRECTORIES:
            with self.subTest(required_directory=required_directory):
                folder = self.source / required_directory
                moved = self.root / "temporarily removed directory"
                folder.rename(moved)
                try:
                    self.assert_stopped_before_deployment(self.run_installer())
                finally:
                    moved.rename(folder)

    def test_program_container_symlink_is_rejected_before_copy(self):
        container = self.source / "Analysis Package"
        real_container = self.root / "real program container"
        container.rename(real_container)
        container.symlink_to(real_container, target_is_directory=True)
        result = self.run_installer()
        self.assert_stopped_before_deployment(result)
        self.assertTrue(container.is_symlink())
        self.assertTrue((real_container / "project_leap_2d/maintenance/manage_environment.command").is_file())

    def test_nested_run_state_symlink_is_rejected_before_copy(self):
        state = self.source / "Analysis Package/Run State"
        state.rmdir()
        target = self.root / "protected state target"
        target.mkdir()
        marker = target / "keep user state.txt"
        marker.write_text("keep exactly\n")
        state.symlink_to(target, target_is_directory=True)
        result = self.run_installer()
        self.assert_stopped_before_deployment(result)
        self.assertEqual(marker.read_text(), "keep exactly\n")

    def test_payload_symlink_is_rejected(self):
        self.data.unlink()
        self.data.symlink_to(self.launcher)
        self.assert_stopped_before_deployment(self.run_installer())

    def test_payload_directory_symlink_is_rejected(self):
        (self.source / "linked folder").symlink_to(self.destination_parent, target_is_directory=True)
        self.assert_stopped_before_deployment(self.run_installer())

    def test_source_package_symlink_is_rejected(self):
        real_source = self.root / "real source"
        self.source.rename(real_source)
        self.source.symlink_to(real_source, target_is_directory=True)
        self.assert_stopped_before_deployment(self.run_installer())

    def test_manifest_symlink_is_rejected(self):
        real_manifest = self.root / "real manifest"
        self.manifest.rename(real_manifest)
        self.manifest.symlink_to(real_manifest)
        self.assert_stopped_before_deployment(self.run_installer())

    def test_missing_maintenance_entry_is_rejected_even_with_matching_manifest(self):
        self.entry.unlink()
        self.write_manifest()
        self.assert_stopped_before_deployment(self.run_installer())

    def test_invalid_manifests_are_rejected(self):
        valid_manifest = self.manifest.read_text()
        invalid_manifests = {
            "empty": "",
            "duplicate": valid_manifest + valid_manifest.splitlines()[0] + "\n",
            "malformed": "not a sha256 manifest\n",
            "absolute": "0" * 64 + "  /tmp/escaped\n",
            "traversal": "0" * 64 + "  ../escaped\n",
            "embedded_traversal": "0" * 64 + "  project_leap_2d/../escaped\n",
            "dot_component": "0" * 64 + "  ./example data.txt\n",
        }
        for label, content in invalid_manifests.items():
            with self.subTest(label=label):
                self.manifest.write_text(content)
                self.assert_stopped_before_deployment(self.run_installer())

    def test_missing_destination_parent_is_not_created(self):
        self.destination = self.root / "not created" / PACKAGE_NAME
        self.assert_stopped_before_deployment(self.run_installer())
        self.assertFalse(self.destination.parent.exists())

    def test_destination_inside_source_is_rejected_before_staging(self):
        self.destination = self.source / "nested deployment"
        self.assert_stopped_before_deployment(self.run_installer())
        self.assertFalse(list(self.source.glob(".project-leap-v1.0.1.*")))
        self.assertEqual(self.data.read_text(), "small fixture, no real software or downloads\n")

    def test_invalid_arguments_do_not_deploy(self):
        for arguments in [("--unknown",), ("--destination",), ("--destination", "relative/path")]:
            with self.subTest(arguments=arguments):
                self.assert_stopped_before_deployment(self.run_installer(*arguments))


if __name__ == "__main__":
    unittest.main()
