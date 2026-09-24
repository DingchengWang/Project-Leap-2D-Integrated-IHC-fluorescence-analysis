"""Isolated tests: only tiny payloads and a local maintenance stub are run."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


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
            PATH="/usr/bin:/bin:/usr/sbin:/sbin",
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
