from __future__ import annotations

import tempfile
import unittest
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import project_leap_2d.workspace.publication_recovery as publication_recovery
from project_leap_2d.workspace_launcher import _run_workspace
from project_leap_2d.workspace.publication_recovery import (
    _begin_publication,
    publication_journal_path,
)
from project_leap_2d.workspace.pending_results import archive_result_root_files
from project_leap_2d.workspace.workspace_preflight import (
    project_workspace_lock,
    require_nonempty_original_image,
)


class WorkspaceLifecycleTests(unittest.TestCase):
    def make_workspace(self, root: Path) -> tuple[Path, Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        original = root / "Original Image"
        result = root / "Result"
        runtime = root / "Runtime"
        original.mkdir()
        result.mkdir()
        (runtime / "staging").mkdir(parents=True)
        (runtime / "recovery").mkdir(parents=True)
        return original, result, runtime

    def test_empty_input_stops_before_result_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original, result, runtime = self.make_workspace(Path(directory))
            (result / "old.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Original Image is empty"):
                require_nonempty_original_image(original)
            self.assertTrue((result / "old.txt").is_file())
            self.assertIsNone(
                next((runtime / "staging").iterdir(), None)
            )

    def test_pending_sequence_and_no_gap_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, result, runtime = self.make_workspace(Path(directory))
            (result / "one.txt").write_text("one", encoding="utf-8")
            first = archive_result_root_files(result, runtime)
            self.assertEqual(first, result / "Pending")
            self.assertTrue((first / "one.txt").is_file())

            (result / "two.txt").write_text("two", encoding="utf-8")
            second = archive_result_root_files(result, runtime)
            self.assertEqual(second, result / "Pending 1")
            self.assertTrue((second / "two.txt").is_file())

            (result / "Pending 2").mkdir()
            (result / "three.txt").write_text("three", encoding="utf-8")
            third = archive_result_root_files(result, runtime)
            self.assertEqual(third, result / "Pending 3")

    def test_partial_output_files_move_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, result, runtime = self.make_workspace(Path(directory))
            for name in ("overlay.png", "report.txt", "unexpected.log"):
                (result / name).write_text(name, encoding="utf-8")
            target = archive_result_root_files(result, runtime)
            self.assertIsNotNone(target)
            self.assertEqual(
                {path.name for path in target.iterdir()},
                {"overlay.png", "report.txt", "unexpected.log"},
            )

    def test_pending_failure_rolls_every_file_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, result, runtime = self.make_workspace(Path(directory))
            (result / "one.txt").write_text("one", encoding="utf-8")
            (result / "two.txt").write_text("two", encoding="utf-8")
            real_replace = os.replace
            call_count = 0

            def fail_second_data_move(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise OSError("injected archive failure")
                return real_replace(source, destination)

            with patch(
                "project_leap_2d.workspace.pending_results.os.replace",
                side_effect=fail_second_data_move,
            ):
                with self.assertRaisesRegex(OSError, "injected archive failure"):
                    archive_result_root_files(result, runtime)

            self.assertEqual(
                {
                    path.name
                    for path in result.iterdir()
                    if path.is_file()
                },
                {"one.txt", "two.txt"},
            )
            self.assertFalse(
                (runtime / "recovery" / "pending_transaction.json").exists()
            )
            self.assertIsNone(next((runtime / "staging").iterdir(), None))

    def test_unexpected_result_folder_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, result, runtime = self.make_workspace(Path(directory))
            (result / "Unrelated").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Unexpected folder"):
                archive_result_root_files(result, runtime)

    def test_workspace_lock_rejects_a_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "Runtime"
            with project_workspace_lock(runtime):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Another Project Leap 2D run is already using this workspace",
                ):
                    with project_workspace_lock(runtime):
                        self.fail("Second workspace lock was unexpectedly acquired")

    def test_successful_packaged_run_moves_inputs_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "Project Leap 2D"
            original, result, runtime_root = self.make_workspace(project)
            source = original / "C1-DAPI.tif"
            source.write_bytes(b"dapi")
            runtime = SimpleNamespace()

            def discover(input_dir: Path):
                self.assertEqual(Path(input_dir), original)
                return {"DAPI": source}, []

            def publish_output_bundle(
                *,
                staged_files: dict[str, Path],
                final_files: dict[str, Path],
                run_dir: Path,
            ) -> None:
                del run_dir
                for key, staged_path in staged_files.items():
                    shutil.copy2(staged_path, final_files[key])

            runtime.discover_channel_paths = discover
            runtime.publish_output_bundle = publish_output_bundle

            def workflow(active_runtime, argv):
                self.assertIs(active_runtime, runtime)
                active_runtime.discover_channel_paths(original)
                run_dir = project / "fiji-run"
                run_dir.mkdir()
                staged_files: dict[str, Path] = {}
                final_files: dict[str, Path] = {}
                for key in ("whole", "processes", "soma", "report", "workbook"):
                    staged = run_dir / f"{key}.staged"
                    staged.write_text(f"new-{key}", encoding="utf-8")
                    staged_files[key] = staged
                    final_files[key] = result / f"{key}.final"
                active_runtime.publish_output_bundle(
                    staged_files=staged_files,
                    final_files=final_files,
                    run_dir=run_dir,
                )
                return 0

            expected_trash = Path(directory) / "Trash" / "accepted"
            with (
                patch(
                    "project_leap_2d.workspace_launcher.load_runtime",
                    return_value=runtime,
                ),
                patch(
                    "project_leap_2d.analysis_workflow.run_analysis_workflow",
                    side_effect=workflow,
                ),
                patch(
                    "project_leap_2d.workspace_launcher.move_inputs_to_macos_trash_recoverable",
                    return_value=expected_trash,
                ) as move_to_trash,
            ):
                return_code = _run_workspace(project, [])

            self.assertEqual(return_code, 0)
            move_to_trash.assert_called_once()
            self.assertEqual(
                {
                    path.name
                    for path in result.iterdir()
                    if path.is_file()
                },
                {
                    "whole.final",
                    "processes.final",
                    "soma.final",
                    "report.final",
                    "workbook.final",
                },
            )
            self.assertFalse(publication_journal_path(runtime_root).exists())

    def test_failed_or_unpublished_run_does_not_move_inputs(self) -> None:
        for return_code in (0, 2):
            with self.subTest(return_code=return_code):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory) / "Project Leap 2D"
                    original, _, _ = self.make_workspace(project)
                    source = original / "C1-DAPI.tif"
                    source.write_bytes(b"dapi")
                    runtime = SimpleNamespace(
                        discover_channel_paths=lambda input_dir: (
                            {"DAPI": source},
                            [],
                        ),
                        publish_output_bundle=lambda *args, **kwargs: None,
                    )

                    def workflow(active_runtime, argv):
                        del argv
                        active_runtime.discover_channel_paths(original)
                        return return_code

                    with (
                        patch(
                            "project_leap_2d.workspace_launcher.load_runtime",
                            return_value=runtime,
                        ),
                        patch(
                            "project_leap_2d.analysis_workflow.run_analysis_workflow",
                            side_effect=workflow,
                        ),
                        patch(
                            "project_leap_2d.workspace_launcher.move_inputs_to_macos_trash_recoverable",
                        ) as move_to_trash,
                    ):
                        observed = _run_workspace(project, [])

                    self.assertEqual(observed, return_code)
                    move_to_trash.assert_not_called()
                    self.assertTrue(source.is_file())

    def test_committed_journal_write_failure_never_moves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "Project Leap 2D"
            original, result, runtime_root = self.make_workspace(project)
            source = original / "C1-DAPI.tif"
            source.write_bytes(b"dapi")

            def discover(input_dir: Path):
                return {"DAPI": source}, []

            def publish_output_bundle(
                *,
                staged_files: dict[str, Path],
                final_files: dict[str, Path],
                run_dir: Path,
            ) -> None:
                del run_dir
                for key, staged in staged_files.items():
                    shutil.copy2(staged, final_files[key])

            runtime = SimpleNamespace(
                discover_channel_paths=discover,
                publish_output_bundle=publish_output_bundle,
            )

            def workflow(active_runtime, argv):
                del argv
                active_runtime.discover_channel_paths(original)
                run_dir = project / "fiji-run"
                run_dir.mkdir()
                staged_files = {}
                final_files = {}
                for key in ("whole", "processes", "soma", "report", "workbook"):
                    staged = run_dir / f"{key}.staged"
                    staged.write_text(f"new-{key}", encoding="utf-8")
                    staged_files[key] = staged
                    final_files[key] = result / f"{key}.final"
                active_runtime.publish_output_bundle(
                    staged_files=staged_files,
                    final_files=final_files,
                    run_dir=run_dir,
                )
                return 0

            real_write_journal = publication_recovery._write_journal

            def fail_committed_phase(path, payload):
                if payload.get("phase") == "committed":
                    raise OSError("injected committed journal failure")
                return real_write_journal(path, payload)

            with (
                patch(
                    "project_leap_2d.workspace_launcher.load_runtime",
                    return_value=runtime,
                ),
                patch(
                    "project_leap_2d.analysis_workflow.run_analysis_workflow",
                    side_effect=workflow,
                ),
                patch.object(
                    publication_recovery,
                    "_write_journal",
                    side_effect=fail_committed_phase,
                ),
                patch(
                    "project_leap_2d.workspace_launcher.move_inputs_to_macos_trash_recoverable",
                ) as move_to_trash,
                self.assertRaisesRegex(
                    OSError,
                    "injected committed journal failure",
                ),
            ):
                _run_workspace(project, [])

            move_to_trash.assert_not_called()
            self.assertTrue(source.is_file())
            self.assertFalse(publication_journal_path(runtime_root).exists())
            self.assertEqual(
                [path for path in result.iterdir() if path.is_file()],
                [],
            )

    def test_startup_publication_recovery_precedes_pending_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "Project Leap 2D"
            original, result, runtime_root = self.make_workspace(project)
            (original / "C1-DAPI.tif").write_bytes(b"dapi")
            final_files = {}
            staged_files = {}
            staged_dir = project / "fiji-run"
            staged_dir.mkdir()
            for key in ("whole", "processes", "soma", "report", "workbook"):
                path = result / f"{key}.final"
                path.write_text(f"old-{key}", encoding="utf-8")
                final_files[key] = path
                staged = staged_dir / f"{key}.staged"
                staged.write_text(f"new-{key}", encoding="utf-8")
                staged_files[key] = staged
            _begin_publication(
                final_files=final_files,
                new_fingerprints={
                    key: publication_recovery._regular_file_fingerprint(
                        path,
                        "Test staged output",
                    )
                    for key, path in staged_files.items()
                },
                result_root=result.resolve(),
                runtime_root=runtime_root.resolve(),
            )
            final_files["whole"].write_text("new-whole", encoding="utf-8")

            runtime = SimpleNamespace(
                discover_channel_paths=lambda input_dir: ({}, []),
                publish_output_bundle=lambda *args, **kwargs: None,
            )
            with (
                patch(
                    "project_leap_2d.workspace_launcher.load_runtime",
                    return_value=runtime,
                ),
                patch(
                    "project_leap_2d.analysis_workflow.run_analysis_workflow",
                    return_value=2,
                ),
            ):
                self.assertEqual(_run_workspace(project, []), 2)

            pending = result / "Pending"
            self.assertTrue(pending.is_dir())
            for key in final_files:
                self.assertEqual(
                    (pending / f"{key}.final").read_text(encoding="utf-8"),
                    f"old-{key}",
                )
            self.assertFalse(publication_journal_path(runtime_root).exists())


if __name__ == "__main__":
    unittest.main()
