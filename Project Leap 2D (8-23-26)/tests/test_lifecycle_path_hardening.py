from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import project_leap_2d.workspace.input_cleanup_recovery as input_recovery
import project_leap_2d.workspace.publication_recovery as publication_recovery
from project_leap_2d.workspace.input_cleanup import InputSnapshot
from project_leap_2d.workspace.input_cleanup_recovery import (
    begin_input_trash_transaction,
    input_trash_journal_path,
    move_inputs_to_macos_trash_recoverable,
    recover_input_trash_transaction,
)
from project_leap_2d.workspace.publication_recovery import (
    _begin_publication,
    _write_journal,
    publication_journal_path,
    publish_with_publication_journal,
    recover_publication_transaction,
)
from project_leap_2d.workspace.workspace_preflight import (
    validate_workspace_layout,
)


FORMAL_KEYS = ("whole", "processes", "soma", "report", "workbook")


class LifecyclePathHardeningTests(unittest.TestCase):
    @staticmethod
    def make_workspace(root: Path) -> tuple[Path, Path, Path]:
        original = root / "Original Image"
        result = root / "Result"
        runtime = root / "Runtime"
        original.mkdir(parents=True)
        result.mkdir()
        runtime.mkdir()
        return original, result, runtime

    @staticmethod
    def make_publication(
        root: Path,
        *,
        with_old: bool,
    ) -> tuple[
        Path,
        Path,
        Path,
        dict[str, Path],
        dict[str, Path],
        dict[str, dict[str, object]],
    ]:
        result = root / "Result"
        runtime = root / "Runtime"
        run_dir = root / "run"
        result.mkdir(parents=True)
        runtime.mkdir()
        run_dir.mkdir()
        staged: dict[str, Path] = {}
        final: dict[str, Path] = {}
        for key in FORMAL_KEYS:
            staged[key] = run_dir / f"{key}.staged"
            staged[key].write_text(f"new-{key}", encoding="utf-8")
            final[key] = result / f"{key}.final"
            if with_old:
                final[key].write_text(f"old-{key}", encoding="utf-8")
        fingerprints = {
            key: publication_recovery._regular_file_fingerprint(
                path,
                "Test staged output",
            )
            for key, path in staged.items()
        }
        return result, runtime, run_dir, staged, final, fingerprints

    @staticmethod
    def publisher(
        *,
        staged_files: dict[str, Path],
        final_files: dict[str, Path],
        run_dir: Path,
    ) -> None:
        del run_dir
        for key in FORMAL_KEYS:
            temporary = final_files[key].with_suffix(".tmp")
            shutil.copy2(staged_files[key], temporary)
            os.replace(temporary, final_files[key])

    def test_workspace_rejects_real_and_broken_symlink_children(self) -> None:
        for folder_name in ("Original Image", "Result", "Runtime"):
            for broken in (False, True):
                with self.subTest(folder=folder_name, broken=broken):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory) / "Project"
                        self.make_workspace(root)
                        selected = root / folder_name
                        selected.rmdir()
                        target = Path(directory) / f"outside-{folder_name}"
                        if not broken:
                            target.mkdir()
                        selected.symlink_to(target, target_is_directory=True)
                        with self.assertRaises(RuntimeError):
                            validate_workspace_layout(root)
                        if not broken:
                            self.assertTrue(target.is_dir())

    def test_publication_rejects_staged_and_final_symlinks(self) -> None:
        for kind in ("staged", "final"):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (
                        result,
                        runtime,
                        run_dir,
                        staged,
                        final,
                        _,
                    ) = self.make_publication(root, with_old=False)
                    outside = root / "outside.bin"
                    outside.write_bytes(b"outside")
                    selected = staged["whole"] if kind == "staged" else final["whole"]
                    if selected.exists():
                        selected.unlink()
                    selected.symlink_to(outside)
                    with self.assertRaises((RuntimeError, ValueError)):
                        publish_with_publication_journal(
                            publish_callback=self.publisher,
                            call_args=(),
                            call_kwargs={
                                "staged_files": staged,
                                "final_files": final,
                                "run_dir": run_dir,
                            },
                            result_root=result,
                            runtime_root=runtime,
                        )
                    self.assertEqual(outside.read_bytes(), b"outside")
                    self.assertFalse(publication_journal_path(runtime).exists())

    def test_publication_rejects_staged_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, run_dir, staged, final, _ = self.make_publication(
                root,
                with_old=False,
            )
            outside = root / "outside-staging"
            outside.mkdir()
            nested_link = run_dir / "linked"
            nested_link.symlink_to(outside, target_is_directory=True)
            staged["whole"] = nested_link / "whole.staged"
            staged["whole"].write_text("new-whole", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "inside the Fiji run directory|parents must not be symlinks",
            ):
                publish_with_publication_journal(
                    publish_callback=self.publisher,
                    call_args=(),
                    call_kwargs={
                        "staged_files": staged,
                        "final_files": final,
                        "run_dir": run_dir,
                    },
                    result_root=result,
                    runtime_root=runtime,
                )
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_recovery_directory_symlink_is_rejected_by_both_transactions(
        self,
    ) -> None:
        for transaction in ("publication", "input"):
            with self.subTest(transaction=transaction):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    original = root / "Original Image"
                    result = root / "Result"
                    runtime = root / "Runtime"
                    home = root / "home"
                    outside = root / "outside-recovery"
                    original.mkdir()
                    result.mkdir()
                    runtime.mkdir()
                    outside.mkdir()
                    (home / ".Trash").mkdir(parents=True)
                    (runtime / "recovery").symlink_to(
                        outside,
                        target_is_directory=True,
                    )
                    with (
                        patch.object(Path, "home", return_value=home),
                        self.assertRaisesRegex(RuntimeError, "Runtime/recovery"),
                    ):
                        if transaction == "publication":
                            recover_publication_transaction(
                                result_root=result,
                                runtime_root=runtime,
                            )
                        else:
                            recover_input_trash_transaction(
                                original_image=original,
                                runtime_root=runtime,
                            )
                    self.assertEqual(list(outside.iterdir()), [])

    def test_publication_rejects_backup_directory_symlink_before_journal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, _, _, final, fingerprints = self.make_publication(
                root,
                with_old=True,
            )
            recovery = runtime / "recovery"
            recovery.mkdir()
            outside = root / "outside-backups"
            outside.mkdir()
            (recovery / "publication_backups").symlink_to(
                outside,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(RuntimeError, "backup directory"):
                _begin_publication(
                    final_files=final,
                    new_fingerprints=fingerprints,
                    result_root=result,
                    runtime_root=runtime,
                )
            self.assertFalse(publication_journal_path(runtime).exists())
            self.assertEqual(list(outside.iterdir()), [])

    def test_publication_rejects_journal_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "Result"
            runtime = root / "Runtime"
            recovery = runtime / "recovery"
            result.mkdir()
            recovery.mkdir(parents=True)
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            publication_journal_path(runtime).symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            self.assertEqual(outside.read_text(encoding="utf-8"), "{}")

    def test_publication_rejects_backup_symlink_without_touching_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, _, staged, final, fingerprints = (
                self.make_publication(root, with_old=True)
            )
            _begin_publication(
                final_files=final,
                new_fingerprints=fingerprints,
                result_root=result,
                runtime_root=runtime,
            )
            backup = runtime / "recovery" / "publication_backups" / "whole.backup"
            backup.unlink()
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            backup.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "backup path is unsafe"):
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertEqual(outside.read_bytes(), b"outside")
            self.assertTrue(publication_journal_path(runtime).is_file())

    def test_committed_content_change_on_same_inode_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, _, staged, final, fingerprints = (
                self.make_publication(root, with_old=True)
            )
            journal, payload = _begin_publication(
                final_files=final,
                new_fingerprints=fingerprints,
                result_root=result,
                runtime_root=runtime,
            )
            entries = {entry["key"]: entry for entry in payload["entries"]}
            for key in FORMAL_KEYS:
                shutil.copy2(staged[key], final[key])
                metadata = final[key].lstat()
                entries[key]["published_device"] = int(metadata.st_dev)
                entries[key]["published_inode"] = int(metadata.st_ino)
            payload["phase"] = "committed"
            _write_journal(journal, payload)
            original_inode = final["whole"].lstat().st_ino
            final["whole"].write_text("damaged", encoding="utf-8")
            self.assertEqual(final["whole"].lstat().st_ino, original_inode)

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_committed_replacement_is_retained_as_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, _, staged, final, fingerprints = (
                self.make_publication(root, with_old=True)
            )
            journal, payload = _begin_publication(
                final_files=final,
                new_fingerprints=fingerprints,
                result_root=result,
                runtime_root=runtime,
            )
            entries = {entry["key"]: entry for entry in payload["entries"]}
            for key in FORMAL_KEYS:
                shutil.copy2(staged[key], final[key])
                metadata = final[key].lstat()
                entries[key]["published_device"] = int(metadata.st_dev)
                entries[key]["published_inode"] = int(metadata.st_ino)
            payload["phase"] = "committed"
            _write_journal(journal, payload)
            unknown = result / "unknown.tmp"
            unknown.write_text("user replacement", encoding="utf-8")
            os.replace(unknown, final["whole"])

            with self.assertRaisesRegex(RuntimeError, "unrecognized file"):
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            self.assertEqual(
                final["whole"].read_text(encoding="utf-8"),
                "user replacement",
            )
            self.assertTrue(publication_journal_path(runtime).is_file())

    def test_cleanup_removes_only_current_publication_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, _, staged, final, fingerprints = (
                self.make_publication(root, with_old=True)
            )
            _begin_publication(
                final_files=final,
                new_fingerprints=fingerprints,
                result_root=result,
                runtime_root=runtime,
            )
            shutil.copy2(staged["whole"], final["whole"])
            token = "a" * 32
            owned = result / f"temporary_IHC_{token}_whole.tmp"
            unrelated = result / "temporary_IHC_user_whole.tmp"
            owned.write_text("transaction", encoding="utf-8")
            unrelated.write_text("keep", encoding="utf-8")

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            self.assertFalse(owned.exists())
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_input_rejects_journal_and_broken_staging_symlinks(self) -> None:
        for kind in ("journal", "staging"):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    original = root / "Original Image"
                    runtime = root / "Runtime"
                    home = root / "home"
                    original.mkdir()
                    (runtime / "recovery").mkdir(parents=True)
                    (home / ".Trash").mkdir(parents=True)
                    outside = root / "outside"
                    path = (
                        input_trash_journal_path(runtime)
                        if kind == "journal"
                        else runtime / "recovery" / "input_trash_staging"
                    )
                    path.symlink_to(outside)
                    with (
                        patch.object(Path, "home", return_value=home),
                        self.assertRaises(RuntimeError),
                    ):
                        recover_input_trash_transaction(
                            original_image=original,
                            runtime_root=runtime,
                        )
                    self.assertTrue(path.is_symlink())

    def test_foreign_trash_target_collision_restores_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "Original Image"
            runtime = root / "Runtime"
            home = root / "home"
            original.mkdir()
            runtime.mkdir()
            (home / ".Trash").mkdir(parents=True)
            paths = []
            for index in range(3):
                path = original / f"C{index + 1}.tif"
                path.write_bytes(f"image-{index}".encode())
                paths.append(path)
            snapshots = tuple(InputSnapshot.capture(path) for path in paths)
            target = home / ".Trash" / "Project Leap collision"
            with patch.object(Path, "home", return_value=home):
                staging = begin_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                    target=target,
                    snapshots=snapshots,
                )
                for snapshot in snapshots:
                    os.replace(snapshot.path, staging / snapshot.path.name)
                target.mkdir()
                (target / "foreign.txt").write_text("keep", encoding="utf-8")
                self.assertTrue(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
            for snapshot in snapshots:
                snapshot.verify_unchanged()
            self.assertEqual(
                (target / "foreign.txt").read_text(encoding="utf-8"),
                "keep",
            )
            self.assertFalse(input_trash_journal_path(runtime).exists())

    def test_each_source_is_reverified_immediately_before_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "Original Image"
            runtime = root / "Runtime"
            home = root / "home"
            original.mkdir()
            runtime.mkdir()
            (home / ".Trash").mkdir(parents=True)
            first = original / "C1.tif"
            second = original / "C2.tif"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            snapshots = (
                InputSnapshot.capture(first),
                InputSnapshot.capture(second),
            )
            real_replace = os.replace
            source_move_count = 0

            def replace_and_change_next(source, destination):
                nonlocal source_move_count
                result = real_replace(source, destination)
                if Path(source).parent == original:
                    source_move_count += 1
                    if source_move_count == 1:
                        second.write_bytes(b"changed-second")
                return result

            class OSProxy:
                replace = staticmethod(replace_and_change_next)

                def __getattr__(self, name):
                    return getattr(os, name)

            with (
                patch.object(Path, "home", return_value=home),
                patch.object(input_recovery, "os", OSProxy()),
                self.assertRaisesRegex(RuntimeError, "recovery journal was retained"),
            ):
                move_inputs_to_macos_trash_recoverable(
                    original_image=original,
                    runtime_root=runtime,
                    snapshots=snapshots,
                )
            self.assertTrue(second.is_file())
            self.assertEqual(second.read_bytes(), b"changed-second")
            self.assertTrue(input_trash_journal_path(runtime).is_file())

    def test_sigkill_ready_publication_recovers_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, runtime, _, staged, final, _ = self.make_publication(
                root,
                with_old=True,
            )
            child = textwrap.dedent(
                """
                import os, shutil, signal, sys
                from pathlib import Path
                import project_leap_2d.workspace.publication_recovery as pr
                root = Path(sys.argv[1])
                result, runtime, run = root / "Result", root / "Runtime", root / "run"
                keys = ("whole", "processes", "soma", "report", "workbook")
                staged = {key: run / f"{key}.staged" for key in keys}
                final = {key: result / f"{key}.final" for key in keys}
                fingerprints = {
                    key: pr._regular_file_fingerprint(path, "child staged")
                    for key, path in staged.items()
                }
                pr._begin_publication(
                    final_files=final,
                    new_fingerprints=fingerprints,
                    result_root=result,
                    runtime_root=runtime,
                )
                shutil.copy2(staged["whole"], final["whole"])
                token = "b" * 32
                (result / f"temporary_IHC_{token}_whole.tmp").write_bytes(b"temp")
                os.kill(os.getpid(), signal.SIGKILL)
                """
            )
            completed = subprocess.run(
                [sys.executable, "-c", child, str(root)],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            self.assertEqual(completed.returncode, -signal.SIGKILL)
            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            self.assertFalse(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertEqual(
                list(result.glob("temporary_IHC_b*.tmp")),
                [],
            )

    def test_sigkill_partial_input_move_recovers_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "Original Image"
            runtime = root / "Runtime"
            home = root / "home"
            original.mkdir()
            runtime.mkdir()
            (home / ".Trash").mkdir(parents=True)
            for index in range(4):
                (original / f"C{index + 1}.tif").write_bytes(
                    f"image-{index}".encode()
                )
            target = home / ".Trash" / "Project Leap killed"
            child = textwrap.dedent(
                """
                import os, signal, sys
                from pathlib import Path
                from project_leap_2d.workspace.input_cleanup import InputSnapshot
                from project_leap_2d.workspace.input_cleanup_recovery import begin_input_trash_transaction
                root = Path(sys.argv[1])
                original, runtime = root / "Original Image", root / "Runtime"
                target = Path(sys.argv[2])
                snapshots = tuple(
                    InputSnapshot.capture(path)
                    for path in sorted(original.glob("*.tif"))
                )
                staging = begin_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                    target=target,
                    snapshots=snapshots,
                )
                for snapshot in snapshots[:2]:
                    os.replace(snapshot.path, staging / snapshot.path.name)
                os.kill(os.getpid(), signal.SIGKILL)
                """
            )
            environment = dict(os.environ)
            environment["HOME"] = str(home)
            completed = subprocess.run(
                [sys.executable, "-c", child, str(root), str(target)],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, -signal.SIGKILL)
            with patch.object(Path, "home", return_value=home):
                self.assertTrue(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
                self.assertFalse(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
            self.assertEqual(len(list(original.glob("*.tif"))), 4)
            self.assertFalse(target.exists())

    def test_sigkill_after_trash_rename_finalizes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "Original Image"
            runtime = root / "Runtime"
            home = root / "home"
            original.mkdir()
            runtime.mkdir()
            (home / ".Trash").mkdir(parents=True)
            for index in range(3):
                (original / f"C{index + 1}.tif").write_bytes(
                    f"image-{index}".encode()
                )
            target = home / ".Trash" / "Project Leap committed"
            child = textwrap.dedent(
                """
                import os, signal, sys
                from pathlib import Path
                from project_leap_2d.workspace.input_cleanup import InputSnapshot
                from project_leap_2d.workspace.input_cleanup_recovery import begin_input_trash_transaction
                root = Path(sys.argv[1])
                original, runtime = root / "Original Image", root / "Runtime"
                target = Path(sys.argv[2])
                snapshots = tuple(
                    InputSnapshot.capture(path)
                    for path in sorted(original.glob("*.tif"))
                )
                staging = begin_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                    target=target,
                    snapshots=snapshots,
                )
                for snapshot in snapshots:
                    os.replace(snapshot.path, staging / snapshot.path.name)
                os.replace(staging, target)
                os.kill(os.getpid(), signal.SIGKILL)
                """
            )
            environment = dict(os.environ)
            environment["HOME"] = str(home)
            completed = subprocess.run(
                [sys.executable, "-c", child, str(root), str(target)],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, -signal.SIGKILL)
            with patch.object(Path, "home", return_value=home):
                self.assertTrue(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
                self.assertFalse(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
            self.assertEqual(len(list(original.glob("*.tif"))), 0)
            self.assertEqual(len(list(target.glob("*.tif"))), 3)


if __name__ == "__main__":
    unittest.main()
