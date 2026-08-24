from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


INPUT_TRASH_JOURNAL_NAME = "input_trash_transaction.json"
INPUT_TRASH_STAGING_NAME = "input_trash_staging"
_INPUT_TRASH_SCHEMA_VERSION = 1


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_real_directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{label} must be a real directory, not a symlink")
    return raw.resolve()


def _require_safe_recovery_root(runtime_root: Path, *, create: bool) -> Path:
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    recovery_root = runtime_root / "recovery"
    if _lexists(recovery_root):
        if recovery_root.is_symlink() or not recovery_root.is_dir():
            raise RuntimeError(
                "Runtime/recovery must be a real directory, not a symlink"
            )
    elif create:
        recovery_root.mkdir(parents=False)
        _sync_directory(runtime_root)
    return recovery_root


def input_trash_journal_path(runtime_root: Path) -> Path:
    return (
        Path(runtime_root).resolve()
        / "recovery"
        / INPUT_TRASH_JOURNAL_NAME
    )


def _staging_path(runtime_root: Path) -> Path:
    return (
        Path(runtime_root).resolve()
        / "recovery"
        / INPUT_TRASH_STAGING_NAME
    )


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_journal(path: Path, payload: Mapping[str, Any]) -> None:
    if _lexists(path.parent):
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise RuntimeError(
                "Input cleanup recovery directory must be a real directory"
            )
    else:
        path.parent.mkdir(parents=True)
    if _lexists(path) and path.is_symlink():
        raise RuntimeError("Input cleanup recovery journal must not be a symlink")
    temporary = path.with_name(
        f"temporary_{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_journal_temporaries(recovery_root: Path) -> None:
    if not recovery_root.is_dir():
        return
    for path in recovery_root.glob("temporary_input_trash_transaction*.tmp"):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Input cleanup journal temporary path is unsafe: " + path.name
            )
        path.unlink(missing_ok=True)
    _sync_directory(recovery_root)


def _identity(path: Path) -> dict[str, int]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Input cleanup path is not a regular file: {path}")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _matches_identity(path: Path, record: Mapping[str, Any]) -> bool:
    try:
        observed = _identity(path)
    except (FileNotFoundError, RuntimeError):
        return False
    return all(
        observed[key] == int(record[key])
        for key in ("device", "inode", "size", "mtime_ns")
    )


def _validated_payload(
    payload: Mapping[str, Any],
    *,
    original_image: Path,
    runtime_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    if int(payload.get("schema_version", -1)) != _INPUT_TRASH_SCHEMA_VERSION:
        raise RuntimeError("Unsupported input cleanup recovery journal")
    if Path(str(payload.get("original_image", ""))).resolve() != original_image:
        raise RuntimeError(
            "Input cleanup recovery journal belongs to another Original Image"
        )
    if Path(str(payload.get("staging", ""))).resolve() != _staging_path(
        runtime_root
    ):
        raise RuntimeError("Input cleanup recovery staging path is invalid")
    raw_target = Path(str(payload.get("target", "")))
    if raw_target.is_symlink():
        raise RuntimeError("Input cleanup recovery Trash target is a symlink")
    target = raw_target.resolve()
    expected_trash = _require_real_directory(
        Path.home() / ".Trash",
        "macOS Trash",
    )
    if target.parent != expected_trash or target.name in {"", ".", ".."}:
        raise RuntimeError("Input cleanup recovery Trash target is invalid")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RuntimeError("Input cleanup recovery journal has no inputs")
    entries: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise RuntimeError("Input cleanup recovery entry is invalid")
        name = str(raw.get("name", ""))
        if (
            not name
            or Path(name).name != name
            or name in observed_names
        ):
            raise RuntimeError(
                "Input cleanup recovery entry is unsafe or repeated"
            )
        observed_names.add(name)
        try:
            entry = {
                "name": name,
                "device": int(raw["device"]),
                "inode": int(raw["inode"]),
                "size": int(raw["size"]),
                "mtime_ns": int(raw["mtime_ns"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Input cleanup recovery identity is invalid"
            ) from exc
        if any(entry[key] < 0 for key in ("device", "inode", "size", "mtime_ns")):
            raise RuntimeError("Input cleanup recovery identity is invalid")
        entries.append(entry)
    return target, entries


def _target_is_complete(
    target: Path,
    entries: Iterable[Mapping[str, Any]],
) -> bool:
    if not target.is_dir() or target.is_symlink():
        return False
    expected_names = {str(entry["name"]) for entry in entries}
    observed_names = {
        path.name
        for path in target.iterdir()
        if path.name not in {".DS_Store"}
    }
    if observed_names != expected_names:
        return False
    return all(
        _matches_identity(target / str(entry["name"]), entry)
        for entry in entries
    )


def _clear_transaction(runtime_root: Path) -> None:
    journal = input_trash_journal_path(runtime_root)
    staging = _staging_path(runtime_root)
    if _lexists(staging):
        if staging.is_symlink() or not staging.is_dir():
            raise RuntimeError(
                "Input cleanup staging must be a real directory"
            )
        staging.rmdir()
    if _lexists(journal) and journal.is_symlink():
        raise RuntimeError("Input cleanup recovery journal must not be a symlink")
    journal.unlink(missing_ok=True)
    _remove_journal_temporaries(journal.parent)
    _sync_directory(journal.parent)


def begin_input_trash_transaction(
    *,
    original_image: Path,
    runtime_root: Path,
    target: Path,
    snapshots: Iterable[Any],
) -> Path:
    original_image = _require_real_directory(original_image, "Original Image")
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    _require_safe_recovery_root(runtime_root, create=True)
    raw_target = Path(target)
    if raw_target.is_symlink():
        raise RuntimeError("Input cleanup Trash target must not be a symlink")
    target = raw_target.resolve()
    journal = input_trash_journal_path(runtime_root)
    staging = _staging_path(runtime_root)
    if (
        target.parent
        != _require_real_directory(Path.home() / ".Trash", "macOS Trash")
        or _lexists(target)
    ):
        raise RuntimeError("Input cleanup Trash target is unavailable or unsafe")
    if _lexists(journal):
        if journal.is_symlink():
            raise RuntimeError(
                "Input cleanup recovery journal must not be a symlink"
            )
        raise RuntimeError(
            "An unresolved input cleanup transaction must be recovered first"
        )
    if _lexists(staging):
        if (
            not staging.is_dir()
            or staging.is_symlink()
            or any(staging.iterdir())
        ):
            raise RuntimeError(
                "Input cleanup staging exists without a recoverable journal"
            )
        staging.rmdir()

    entries: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    for snapshot in snapshots:
        raw_source = Path(snapshot.path)
        if raw_source.is_symlink():
            raise RuntimeError(
                "Input cleanup snapshot path must not be a symlink"
            )
        source = raw_source.resolve()
        if (
            source.parent != original_image
            or source.name in observed_names
            or Path(source.name).name != source.name
        ):
            raise RuntimeError("Input cleanup snapshot path is unsafe or repeated")
        observed_names.add(source.name)
        entries.append(
            {
                "name": source.name,
                "device": int(snapshot.device),
                "inode": int(snapshot.inode),
                "size": int(snapshot.size),
                "mtime_ns": int(snapshot.mtime_ns),
            }
        )
    if not entries:
        raise RuntimeError("No accepted channel TIFFs were recorded for cleanup.")

    payload = {
        "schema_version": _INPUT_TRASH_SCHEMA_VERSION,
        "original_image": str(original_image),
        "staging": str(staging),
        "target": str(target),
        "entries": entries,
    }
    _write_journal(journal, payload)
    try:
        staging.mkdir(parents=True)
        _sync_directory(staging.parent)
    except BaseException as preparation_error:
        try:
            recover_input_trash_transaction(
                original_image=original_image,
                runtime_root=runtime_root,
            )
        except BaseException as recovery_error:
            raise RuntimeError(
                "Input cleanup staging preparation failed and automatic "
                f"recovery could not finish: {recovery_error}"
            ) from preparation_error
        raise
    return staging


def recover_input_trash_transaction(
    *,
    original_image: Path,
    runtime_root: Path,
) -> bool:
    """Restore an interrupted pre-Trash move or finalize a completed Trash move."""

    original_image = _require_real_directory(original_image, "Original Image")
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    recovery_root = _require_safe_recovery_root(runtime_root, create=False)
    journal = input_trash_journal_path(runtime_root)
    if not _lexists(journal):
        if recovery_root.is_dir():
            _remove_journal_temporaries(recovery_root)
        staging = _staging_path(runtime_root)
        if _lexists(staging):
            if (
                not staging.is_dir()
                or staging.is_symlink()
                or any(staging.iterdir())
            ):
                raise RuntimeError(
                    "Input cleanup staging exists without a recovery journal"
                )
            staging.rmdir()
            _sync_directory(staging.parent)
        return False
    if journal.is_symlink() or not journal.is_file():
        raise RuntimeError(
            "Input cleanup recovery journal must be a regular non-symlink file"
        )

    payload = json.loads(journal.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Input cleanup recovery journal must be a JSON object")
    target, entries = _validated_payload(
        payload,
        original_image=original_image,
        runtime_root=runtime_root,
    )
    staging = _staging_path(runtime_root)

    if _lexists(target):
        if target.is_symlink():
            raise RuntimeError(
                "Input cleanup Trash target must not be a symlink"
            )
        if _target_is_complete(target, entries):
            if _lexists(staging):
                if (
                    staging.is_symlink()
                    or not staging.is_dir()
                    or any(staging.iterdir())
                ):
                    raise RuntimeError(
                        "Completed input cleanup has conflicting staging files"
                    )
                staging.rmdir()
            _clear_transaction(runtime_root)
            return True
        if not _lexists(staging):
            raise RuntimeError(
                "Input cleanup Trash target conflicts with the recovery journal"
            )
        # A different process created the chosen Trash name after the journal
        # was written. Keep that unrelated target untouched and restore this
        # transaction from its still-present staging directory below.

    if _lexists(staging) and (
        not staging.is_dir() or staging.is_symlink()
    ):
        raise RuntimeError("Input cleanup staging path is invalid")
    staged_names = (
        {path.name for path in staging.iterdir()}
        if staging.is_dir()
        else set()
    )
    expected_names = {str(entry["name"]) for entry in entries}
    if not staged_names.issubset(expected_names):
        raise RuntimeError("Input cleanup staging contains unexpected files")

    moves: list[tuple[Path, Path, Mapping[str, Any]]] = []
    for entry in entries:
        name = str(entry["name"])
        staged = staging / name
        destination = original_image / name
        if _lexists(staged):
            if staged.is_symlink():
                raise RuntimeError(
                    f"Input cleanup staged path is a symlink: {name}"
                )
            if _lexists(destination):
                raise RuntimeError(
                    f"Input cleanup recovery conflict in Original Image: {name}"
                )
            if not _matches_identity(staged, entry):
                raise RuntimeError(
                    f"Input cleanup staged file identity changed: {name}"
                )
            moves.append((staged, destination, entry))
        elif not _matches_identity(destination, entry):
            raise RuntimeError(
                f"Input cleanup recovery cannot locate unchanged input: {name}"
            )

    for staged, destination, entry in moves:
        if not _matches_identity(staged, entry):
            raise RuntimeError(
                f"Input cleanup staged file identity changed: {staged.name}"
            )
        if _lexists(destination):
            raise RuntimeError(
                "Input cleanup recovery conflict in Original Image: "
                f"{destination.name}"
            )
        os.replace(staged, destination)
    _sync_directory(original_image)
    if staging.is_dir():
        staging.rmdir()
    _clear_transaction(runtime_root)
    return True


def move_inputs_to_macos_trash_recoverable(
    *,
    original_image: Path,
    runtime_root: Path,
    snapshots: tuple[Any, ...],
) -> Path:
    """Move accepted unchanged inputs through the recoverable workspace adapter."""

    from .input_cleanup import _available_trash_path

    original_image = _require_real_directory(original_image, "Original Image")
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    _require_safe_recovery_root(runtime_root, create=True)
    if not snapshots:
        raise RuntimeError("No accepted channel TIFFs were recorded for cleanup.")
    for snapshot in snapshots:
        raw_source = Path(snapshot.path)
        if raw_source.is_symlink() or raw_source.resolve().parent != original_image:
            raise RuntimeError(
                "Refusing to move an input outside Original Image: "
                f"{Path(snapshot.path).name}"
            )
        snapshot.verify_unchanged()

    trash_root = Path.home() / ".Trash"
    if trash_root.is_symlink() or not trash_root.is_dir():
        raise RuntimeError("macOS Trash folder is unavailable.")
    target = _available_trash_path(trash_root)
    staging = begin_input_trash_transaction(
        original_image=original_image,
        runtime_root=runtime_root,
        target=target,
        snapshots=snapshots,
    )
    try:
        for snapshot in snapshots:
            source = Path(snapshot.path)
            snapshot.verify_unchanged()
            os.replace(source, staging / source.name)
        _sync_directory(staging)
        _sync_directory(original_image)
        if _lexists(target):
            raise RuntimeError(
                "Input cleanup Trash target became unavailable before commit"
            )
        os.replace(staging, target)
        _sync_directory(target.parent)
        try:
            recover_input_trash_transaction(
                original_image=original_image,
                runtime_root=runtime_root,
            )
        except Exception:
            # The directory rename into Trash already committed. The fixed
            # journal is finalized on the next locked workspace launch.
            pass
        return target
    except BaseException as cleanup_error:
        try:
            recover_input_trash_transaction(
                original_image=original_image,
                runtime_root=runtime_root,
            )
        except BaseException as recovery_error:
            raise RuntimeError(
                "Input cleanup failed and automatic recovery could not safely "
                f"finish; recovery journal was retained: {recovery_error}"
            ) from cleanup_error
        raise
