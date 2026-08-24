from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from project_leap_2d.fiji_review.failed_run_retention import (
    retain_only_latest_failed_fiji_run,
)
from project_leap_2d.runtime_attributes import temporary_runtime_attributes


class TemporaryRuntimeAttributesTests(unittest.TestCase):
    def test_restores_exact_objects_after_exception(self) -> None:
        first = object()
        second = object()
        runtime = SimpleNamespace(first=first, second=second)
        replacement_first = object()
        replacement_second = object()

        with self.assertRaisesRegex(RuntimeError, "injected"):
            with temporary_runtime_attributes(
                runtime,
                first=replacement_first,
                second=replacement_second,
            ):
                self.assertIs(runtime.first, replacement_first)
                self.assertIs(runtime.second, replacement_second)
                raise RuntimeError("injected")

        self.assertIs(runtime.first, first)
        self.assertIs(runtime.second, second)

    def test_nested_scopes_restore_in_lifo_order(self) -> None:
        original = object()
        outer = object()
        inner = object()
        runtime = SimpleNamespace(value=original)

        with temporary_runtime_attributes(runtime, value=outer):
            self.assertIs(runtime.value, outer)
            with temporary_runtime_attributes(runtime, value=inner):
                self.assertIs(runtime.value, inner)
            self.assertIs(runtime.value, outer)
        self.assertIs(runtime.value, original)

    def test_removes_attribute_that_was_initially_absent(self) -> None:
        runtime = SimpleNamespace()
        with temporary_runtime_attributes(runtime, added=object()):
            self.assertTrue(hasattr(runtime, "added"))
        self.assertFalse(hasattr(runtime, "added"))


class FailedFijiRunRetentionTests(unittest.TestCase):
    @staticmethod
    def make_failed_run(cache_root: Path, suffix: str) -> Path:
        run_dir = cache_root / f"run-{suffix * 32}"
        run_dir.mkdir()
        (run_dir / "analysis_report_failed.txt").write_text(
            "failed",
            encoding="utf-8",
        )
        return run_dir

    def test_keeps_current_failed_run_and_removes_only_prior_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache_root = home / "Library" / "Caches" / "IHC2DAnalysis"
            cache_root.mkdir(parents=True)
            old_failed = self.make_failed_run(cache_root, "a")
            current = self.make_failed_run(cache_root, "b")
            active = cache_root / f"run-{'c' * 32}"
            active.mkdir()
            malformed = cache_root / "run-manual-notes"
            malformed.mkdir()

            with patch.object(Path, "home", return_value=home):
                retain_only_latest_failed_fiji_run(current)

            self.assertFalse(old_failed.exists())
            self.assertTrue(current.is_dir())
            self.assertTrue(active.is_dir())
            self.assertTrue(malformed.is_dir())

    def test_rejects_directory_outside_exact_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            outside = root / f"run-{'d' * 32}"
            outside.mkdir()
            (outside / "analysis_report_failed.txt").write_text(
                "failed",
                encoding="utf-8",
            )

            with patch.object(Path, "home", return_value=home):
                with self.assertRaisesRegex(ValueError, "unrecognized"):
                    retain_only_latest_failed_fiji_run(outside)
            self.assertTrue(outside.is_dir())


if __name__ == "__main__":
    unittest.main()
