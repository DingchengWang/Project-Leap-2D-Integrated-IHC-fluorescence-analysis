"""Bounded retention for failed Fiji runtime directories."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


_RUN_DIRECTORY_PATTERN = re.compile(r"run-[0-9a-f]{32}")
_FAILURE_REPORT = "analysis_report_failed.txt"


def retain_only_latest_failed_fiji_run(run_dir: Path | str) -> None:
    """Keep the current failed run and remove older verified failed runs."""

    supplied = Path(run_dir).expanduser().absolute()
    cache_root = (
        Path.home() / "Library" / "Caches" / "IHC2DAnalysis"
    ).resolve()
    current = supplied.resolve()
    if (
        supplied.is_symlink()
        or current.parent != cache_root
        or _RUN_DIRECTORY_PATTERN.fullmatch(current.name) is None
        or not current.is_dir()
    ):
        raise ValueError("Refusing to clean an unrecognized Fiji runtime directory")
    if not (current / _FAILURE_REPORT).is_file():
        return

    for candidate in cache_root.iterdir():
        if (
            candidate == current
            or _RUN_DIRECTORY_PATTERN.fullmatch(candidate.name) is None
            or candidate.is_symlink()
            or not candidate.is_dir()
            or not (candidate / _FAILURE_REPORT).is_file()
        ):
            continue
        try:
            shutil.rmtree(candidate)
        except OSError:
            pass
