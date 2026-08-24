from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project_leap_2d.workspace.input_cleanup_recovery as input_cleanup_recovery
from project_leap_2d.workspace.input_cleanup import InputSnapshot
from project_leap_2d.workspace.input_cleanup_recovery import (
    begin_input_trash_transaction,
    input_trash_journal_path,
    move_inputs_to_macos_trash_recoverable,
    recover_input_trash_transaction,
)


class InputCleanupRecoveryTests(unittest.TestCase):
    def make_inputs(
        self,
        root: Path,
        *,
        count: int = 4,
    ) -> tuple[Path, Path, Path, tuple[InputSnapshot, ...]]:
        original = root / "Original Image"
        runtime = root / "Runtime"
        home = root / "home"
        original.mkdir(parents=True)
        runtime.mkdir()
        (home / ".Trash").mkdir(parents=True)
        snapshots = []
        for index in range(count):
            path = original / f"C{index + 1}-channel.tif"
            path.write_bytes(f"image-{index}".encode("utf-8"))
            snapshots.append(InputSnapshot.capture(path.resolve()))
        return original, runtime, home, tuple(snapshots)

    def test_every_partial_move_count_restores_all_inputs(self) -> None:
        for moved_count in range(5):
            with self.subTest(moved_count=moved_count):
                with tempfile.TemporaryDirectory() as directory:
                    original, runtime, home, snapshots = self.make_inputs(
                        Path(directory)
                    )
                    target = home / ".Trash" / "Project Leap interrupted"
                    with patch.object(Path, "home", return_value=home):
                        staging = begin_input_trash_transaction(
                            original_image=original,
                            runtime_root=runtime,
                            target=target,
                            snapshots=snapshots,
                        )
                        for snapshot in snapshots[:moved_count]:
                            os.replace(
                                snapshot.path,
                                staging / snapshot.path.name,
                            )
                        self.assertTrue(
                            recover_input_trash_transaction(
                                original_image=original,
                                runtime_root=runtime,
                            )
                        )

                    for snapshot in snapshots:
                        self.assertTrue(snapshot.path.is_file())
                        snapshot.verify_unchanged()
                    self.assertFalse(input_trash_journal_path(runtime).exists())
                    self.assertFalse(staging.exists())

    def test_recovery_is_reentrant_after_partial_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, snapshots = self.make_inputs(Path(directory))
            target = home / ".Trash" / "Project Leap interrupted"
            with patch.object(Path, "home", return_value=home):
                staging = begin_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                    target=target,
                    snapshots=snapshots,
                )
                for snapshot in snapshots:
                    os.replace(snapshot.path, staging / snapshot.path.name)
                first = snapshots[0]
                os.replace(staging / first.path.name, first.path)

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
            for snapshot in snapshots:
                snapshot.verify_unchanged()

    def test_complete_trash_rename_is_finalized_without_restoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, snapshots = self.make_inputs(Path(directory))
            target = home / ".Trash" / "Project Leap completed"
            with patch.object(Path, "home", return_value=home):
                staging = begin_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                    target=target,
                    snapshots=snapshots,
                )
                for snapshot in snapshots:
                    os.replace(snapshot.path, staging / snapshot.path.name)
                os.replace(staging, target)

                self.assertTrue(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
            self.assertTrue(target.is_dir())
            for snapshot in snapshots:
                self.assertFalse(snapshot.path.exists())
                self.assertTrue((target / snapshot.path.name).is_file())
            self.assertFalse(input_trash_journal_path(runtime).exists())

    def test_original_image_conflict_stops_and_retains_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, snapshots = self.make_inputs(Path(directory))
            target = home / ".Trash" / "Project Leap interrupted"
            with patch.object(Path, "home", return_value=home):
                staging = begin_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                    target=target,
                    snapshots=snapshots,
                )
                first = snapshots[0]
                os.replace(first.path, staging / first.path.name)
                first.path.write_bytes(b"conflicting replacement")

                with self.assertRaisesRegex(
                    RuntimeError,
                    "conflict in Original Image",
                ):
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
            self.assertTrue(input_trash_journal_path(runtime).is_file())
            self.assertTrue((staging / first.path.name).is_file())

    def test_python_move_failure_restores_every_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, snapshots = self.make_inputs(Path(directory))
            calls = 0

            def fail_second_move(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected input move failure")
                return os.replace(source, destination)

            class OSProxy:
                replace = staticmethod(fail_second_move)

                def __getattr__(self, name):
                    return getattr(os, name)

            fake_os = OSProxy()
            with (
                patch.object(Path, "home", return_value=home),
                patch.object(input_cleanup_recovery, "os", fake_os),
                self.assertRaisesRegex(OSError, "injected input move failure"),
            ):
                move_inputs_to_macos_trash_recoverable(
                    original_image=original,
                    runtime_root=runtime,
                    snapshots=snapshots,
                )
            for snapshot in snapshots:
                snapshot.verify_unchanged()
            self.assertFalse(input_trash_journal_path(runtime).exists())

    def test_recoverable_wrapper_commits_once_and_cleans_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, snapshots = self.make_inputs(Path(directory))
            with patch.object(Path, "home", return_value=home):
                target = move_inputs_to_macos_trash_recoverable(
                    original_image=original,
                    runtime_root=runtime,
                    snapshots=snapshots,
                )
            self.assertTrue(target.is_dir())
            for snapshot in snapshots:
                self.assertFalse(snapshot.path.exists())
                self.assertTrue((target / snapshot.path.name).is_file())
            self.assertFalse(input_trash_journal_path(runtime).exists())
            self.assertFalse(
                (runtime / "recovery" / "input_trash_staging").exists()
            )

    def test_no_journal_cleans_only_fixed_orphan_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, _ = self.make_inputs(
                Path(directory),
                count=1,
            )
            recovery = runtime / "recovery"
            recovery.mkdir()
            orphan = recovery / (
                "temporary_input_trash_transaction.json.orphan.tmp"
            )
            unrelated = recovery / "unrelated.tmp"
            orphan.write_text("partial", encoding="utf-8")
            unrelated.write_text("keep", encoding="utf-8")
            with patch.object(Path, "home", return_value=home):
                self.assertFalse(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
            self.assertFalse(orphan.exists())
            self.assertTrue(unrelated.is_file())

    def test_no_journal_removes_empty_staging_but_rejects_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, runtime, home, _ = self.make_inputs(
                Path(directory),
                count=1,
            )
            staging = runtime / "recovery" / "input_trash_staging"
            staging.mkdir(parents=True)
            with patch.object(Path, "home", return_value=home):
                self.assertFalse(
                    recover_input_trash_transaction(
                        original_image=original,
                        runtime_root=runtime,
                    )
                )
            self.assertFalse(staging.exists())

            staging.mkdir()
            (staging / "unknown.tif").write_bytes(b"unknown")
            with (
                patch.object(Path, "home", return_value=home),
                self.assertRaisesRegex(
                    RuntimeError,
                    "without a recovery journal",
                ),
            ):
                recover_input_trash_transaction(
                    original_image=original,
                    runtime_root=runtime,
                )
            self.assertTrue((staging / "unknown.tif").is_file())


if __name__ == "__main__":
    unittest.main()
