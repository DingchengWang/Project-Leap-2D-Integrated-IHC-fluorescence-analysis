from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project_leap_2d.workspace.publication_recovery as publication_recovery
from project_leap_2d.workspace.publication_recovery import (
    _begin_publication,
    _write_journal,
    publication_journal_path,
    publish_with_publication_journal,
    recover_publication_transaction,
)


class PublicationRecoveryTests(unittest.TestCase):
    def make_paths(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, dict[str, Path], dict[str, Path]]:
        result = root / "Result"
        runtime = root / "Runtime"
        run_dir = root / "fiji-run"
        result.mkdir(parents=True)
        runtime.mkdir()
        run_dir.mkdir()
        staged: dict[str, Path] = {}
        final: dict[str, Path] = {}
        for key in ("whole", "processes", "soma", "report", "workbook"):
            staged_path = run_dir / f"{key}.staged"
            staged_path.write_text(f"new-{key}", encoding="utf-8")
            staged[key] = staged_path
            final[key] = result / f"{key}.final"
        return result, runtime, run_dir, staged, final

    @staticmethod
    def fingerprints(staged: dict[str, Path]) -> dict[str, dict[str, object]]:
        return {
            key: publication_recovery._regular_file_fingerprint(
                path,
                "Test staged output",
            )
            for key, path in staged.items()
        }

    @staticmethod
    def publisher(
        *,
        staged_files: dict[str, Path],
        final_files: dict[str, Path],
        run_dir: Path,
    ) -> None:
        del run_dir
        for key, staged in staged_files.items():
            temporary = final_files[key].with_suffix(".tmp")
            shutil.copy2(staged, temporary)
            os.replace(temporary, final_files[key])

    def test_successful_publication_removes_journal_and_backups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, run_dir, staged, final = self.make_paths(
                Path(directory)
            )
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
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"new-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())
            self.assertFalse(
                (runtime / "recovery" / "publication_backups").exists()
            )
            self.assertEqual(
                list(result.glob("temporary_IHC_*.tmp")),
                [],
            )
            self.assertEqual(
                list(
                    (runtime / "recovery").glob(
                        "temporary_publication_transaction*.tmp"
                    )
                ),
                [],
            )

    def test_python_failure_rolls_back_every_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, run_dir, staged, final = self.make_paths(
                Path(directory)
            )
            for key, path in final.items():
                path.write_text(f"old-{key}", encoding="utf-8")

            def partial_publisher(
                *,
                staged_files: dict[str, Path],
                final_files: dict[str, Path],
                run_dir: Path,
            ) -> None:
                del run_dir
                first = next(iter(staged_files))
                shutil.copy2(staged_files[first], final_files[first])
                raise OSError("injected publication failure")

            with self.assertRaisesRegex(OSError, "injected publication failure"):
                publish_with_publication_journal(
                    publish_callback=partial_publisher,
                    call_args=(),
                    call_kwargs={
                        "staged_files": staged,
                        "final_files": final,
                        "run_dir": run_dir,
                    },
                    result_root=result,
                    runtime_root=runtime,
                )

            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_startup_recovery_rolls_back_interrupted_ready_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, _, staged, final = self.make_paths(Path(directory))
            for key, path in final.items():
                path.write_text(f"old-{key}", encoding="utf-8")
            _, _ = _begin_publication(
                final_files=final,
                new_fingerprints=self.fingerprints(staged),
                result_root=result.resolve(),
                runtime_root=runtime.resolve(),
            )
            first_key = next(iter(final))
            final[first_key].write_text(f"new-{first_key}", encoding="utf-8")

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_ready_recovery_handles_every_partial_replace_and_mixed_existence(
        self,
    ) -> None:
        keys = ("whole", "processes", "soma", "report", "workbook")
        for replace_count in range(6):
            with self.subTest(replace_count=replace_count):
                with tempfile.TemporaryDirectory() as directory:
                    result, runtime, _, staged, final = self.make_paths(
                        Path(directory)
                    )
                    for key in keys[:2]:
                        final[key].write_text(f"old-{key}", encoding="utf-8")
                    _begin_publication(
                        final_files=final,
                        new_fingerprints=self.fingerprints(staged),
                        result_root=result.resolve(),
                        runtime_root=runtime.resolve(),
                    )
                    for key in keys[:replace_count]:
                        shutil.copy2(staged[key], final[key])

                    self.assertTrue(
                        recover_publication_transaction(
                            result_root=result,
                            runtime_root=runtime,
                        )
                    )
                    for key in keys[:2]:
                        self.assertEqual(
                            final[key].read_text(encoding="utf-8"),
                            f"old-{key}",
                        )
                    for key in keys[2:]:
                        self.assertFalse(final[key].exists())
                    self.assertFalse(publication_journal_path(runtime).exists())

    def test_preparing_partial_backups_are_cleaned_without_output_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, _, staged, final = self.make_paths(Path(directory))
            for key, path in final.items():
                path.write_text(f"old-{key}", encoding="utf-8")
            journal, payload = _begin_publication(
                final_files=final,
                new_fingerprints=self.fingerprints(staged),
                result_root=result.resolve(),
                runtime_root=runtime.resolve(),
            )
            payload["phase"] = "preparing"
            _write_journal(journal, payload)
            backups = runtime / "recovery" / "publication_backups"
            for path in list(backups.iterdir())[2:]:
                path.unlink()

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertFalse(backups.exists())
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_committed_recovery_keeps_complete_new_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, _, staged, final = self.make_paths(Path(directory))
            journal, payload = _begin_publication(
                final_files=final,
                new_fingerprints=self.fingerprints(staged),
                result_root=result.resolve(),
                runtime_root=runtime.resolve(),
            )
            for key, path in final.items():
                shutil.copy2(staged[key], path)
            payload["phase"] = "committed"
            _write_journal(journal, payload)

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"new-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_committed_missing_file_rolls_back_incomplete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, _, staged, final = self.make_paths(Path(directory))
            for key in ("whole", "processes"):
                final[key].write_text(f"old-{key}", encoding="utf-8")
            journal, payload = _begin_publication(
                final_files=final,
                new_fingerprints=self.fingerprints(staged),
                result_root=result.resolve(),
                runtime_root=runtime.resolve(),
            )
            for key, path in final.items():
                shutil.copy2(staged[key], path)
            payload["phase"] = "committed"
            _write_journal(journal, payload)
            final["workbook"].unlink()

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key in ("whole", "processes"):
                self.assertEqual(
                    final[key].read_text(encoding="utf-8"),
                    f"old-{key}",
                )
            for key in ("soma", "report", "workbook"):
                self.assertFalse(final[key].exists())

    def test_rolled_back_phase_finishes_cleanup_without_backups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, _, staged, final = self.make_paths(Path(directory))
            for key, path in final.items():
                path.write_text(f"old-{key}", encoding="utf-8")
            journal, payload = _begin_publication(
                final_files=final,
                new_fingerprints=self.fingerprints(staged),
                result_root=result.resolve(),
                runtime_root=runtime.resolve(),
            )
            payload["phase"] = "rolled_back"
            _write_journal(journal, payload)
            shutil.rmtree(runtime / "recovery" / "publication_backups")

            self.assertTrue(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())
            self.assertFalse(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )

    def test_fsync_failure_rolls_back_and_leaves_no_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, run_dir, staged, final = self.make_paths(
                Path(directory)
            )
            for key, path in final.items():
                path.write_text(f"old-{key}", encoding="utf-8")
            calls = 0
            real_sync_file = publication_recovery._sync_file

            def fail_first_final(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise OSError("injected final fsync failure")
                real_sync_file(path)

            with (
                patch.object(
                    publication_recovery,
                    "_sync_file",
                    side_effect=fail_first_final,
                ),
                self.assertRaisesRegex(
                    OSError,
                    "injected final fsync failure",
                ),
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

            for key, path in final.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{key}")
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_formal_five_file_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, runtime, run_dir, staged, final = self.make_paths(
                Path(directory)
            )
            incomplete_staged = dict(staged)
            incomplete_staged.pop("workbook")
            with self.assertRaisesRegex(ValueError, "matching staged/final"):
                publish_with_publication_journal(
                    publish_callback=self.publisher,
                    call_args=(),
                    call_kwargs={
                        "staged_files": incomplete_staged,
                        "final_files": final,
                        "run_dir": run_dir,
                    },
                    result_root=result,
                    runtime_root=runtime,
                )
            self.assertFalse(publication_journal_path(runtime).exists())

    def test_actual_runtime_publisher_signature_is_supported(self) -> None:
        from project_leap_2d.runtime_loader import load_runtime

        signature = inspect.signature(load_runtime().publish_output_bundle)
        self.assertEqual(
            tuple(signature.parameters),
            ("staged_files", "final_files", "run_dir"),
        )

    def test_no_journal_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "Result"
            runtime = root / "Runtime"
            result.mkdir()
            runtime.mkdir()
            recovery = runtime / "recovery"
            recovery.mkdir()
            orphan = (
                recovery
                / "temporary_publication_transaction.json.orphan.tmp"
            )
            orphan.write_text("partial", encoding="utf-8")
            self.assertFalse(
                recover_publication_transaction(
                    result_root=result,
                    runtime_root=runtime,
                )
            )
            self.assertFalse(orphan.exists())


if __name__ == "__main__":
    unittest.main()
