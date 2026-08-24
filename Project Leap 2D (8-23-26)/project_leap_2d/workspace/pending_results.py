from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path


PENDING_PATTERN = re.compile(r"^Pending(?: ([1-9][0-9]*))?$")
IGNORED_ROOT_NAMES = {".DS_Store"}


def _next_pending_path(result: Path) -> Path:
    observed: list[int] = []
    for path in result.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        match = PENDING_PATTERN.fullmatch(path.name)
        if match is None:
            if not path.name.startswith("."):
                raise RuntimeError(
                    f"Unexpected folder in Result: {path.name}. Move it out before "
                    "starting a new analysis."
                )
            continue
        observed.append(0 if match.group(1) is None else int(match.group(1)))
    if not observed:
        return result / "Pending"
    return result / f"Pending {max(observed) + 1}"


def _transaction_record(runtime_root: Path) -> Path:
    return runtime_root / "recovery" / "pending_transaction.json"


def _write_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _recover_interrupted_archive(result: Path, runtime_root: Path) -> None:
    record_path = _transaction_record(runtime_root)
    if not record_path.is_file():
        return
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    staging = Path(str(payload["staging"]))
    target = Path(str(payload["target"]))
    names = [str(name) for name in payload["names"]]
    if target.is_dir():
        record_path.unlink(missing_ok=True)
        return
    if staging.is_dir():
        for name in names:
            staged = staging / name
            destination = result / name
            if staged.exists():
                if destination.exists():
                    raise RuntimeError(
                        "Cannot recover interrupted Result archive because "
                        f"{destination.name} already exists."
                    )
                os.replace(staged, destination)
        staging.rmdir()
    record_path.unlink(missing_ok=True)


def archive_result_root_files(result: Path, runtime_root: Path) -> Path | None:
    _recover_interrupted_archive(result, runtime_root)
    root_files: list[Path] = []
    for path in sorted(result.iterdir(), key=lambda item: item.name):
        if path.name in IGNORED_ROOT_NAMES or path.name.startswith("._"):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Symlinks are not allowed in Result: {path.name}")
        if path.is_file():
            root_files.append(path)
        elif path.is_dir():
            if PENDING_PATTERN.fullmatch(path.name) is None and not path.name.startswith(
                "."
            ):
                raise RuntimeError(
                    f"Unexpected folder in Result: {path.name}. Move it out before "
                    "starting a new analysis."
                )
        else:
            raise RuntimeError(f"Unsupported entry in Result: {path.name}")
    if not root_files:
        return None

    target = _next_pending_path(result)
    staging = runtime_root / "staging" / f"pending-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    record_path = _transaction_record(runtime_root)
    payload = {
        "result": str(result),
        "staging": str(staging),
        "target": str(target),
        "names": [path.name for path in root_files],
    }
    _write_record(record_path, payload)
    moved: list[Path] = []
    try:
        for source in root_files:
            destination = staging / source.name
            os.replace(source, destination)
            moved.append(destination)
        os.replace(staging, target)
        record_path.unlink(missing_ok=True)
        return target
    except BaseException:
        if staging.is_dir():
            for staged in reversed(moved):
                if staged.exists():
                    os.replace(staged, result / staged.name)
            shutil.rmtree(staging, ignore_errors=True)
        record_path.unlink(missing_ok=True)
        raise
