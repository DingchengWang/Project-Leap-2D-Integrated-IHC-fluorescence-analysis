from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path


IGNORED_ENTRY_NAMES = {".DS_Store"}


def validate_workspace_layout(project_root: Path) -> None:
    project_root = Path(project_root)
    if project_root.is_symlink() or not project_root.is_dir():
        raise RuntimeError(
            "Project Leap 2D workspace root must be a real directory, not a symlink."
        )
    resolved_project_root = project_root.resolve()
    analysis_package = project_root / "Analysis Package"
    required = {
        project_root / "Sample Image": resolved_project_root,
        project_root / "Result": resolved_project_root,
        analysis_package: resolved_project_root,
        analysis_package / "Run State": analysis_package.resolve(),
    }
    missing = [
        str(path)
        for path in required
        if not os.path.lexists(path) or not path.is_dir()
    ]
    if missing:
        raise RuntimeError(
            "Project Leap 2D workspace is incomplete; missing folders: "
            + ", ".join(missing)
        )
    unsafe = [
        str(path)
        for path, expected_parent in required.items()
        if path.is_symlink()
        or path.resolve().parent != expected_parent
    ]
    if unsafe:
        raise RuntimeError(
            "Sample Image, Result, and Analysis Package must be real direct "
            "children of the package folder; Run State must be a real direct "
            "child of Analysis Package. Symlinks are not allowed: "
            + ", ".join(unsafe)
        )


def require_nonempty_original_image(original_image: Path) -> None:
    entries = [
        path
        for path in original_image.iterdir()
        if path.name not in IGNORED_ENTRY_NAMES and not path.name.startswith("._")
    ]
    if not entries:
        raise RuntimeError(
            "Sample Image is empty. Add one batch of split-channel TIFF "
            "Z-stacks before starting the analysis."
        )


@contextmanager
def project_workspace_lock(runtime_root: Path):
    lock_dir = runtime_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "workspace.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another Project Leap 2D run is already using this workspace."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        )
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
