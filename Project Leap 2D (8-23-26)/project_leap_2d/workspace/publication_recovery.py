from __future__ import annotations

import inspect
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PUBLICATION_JOURNAL_NAME = "publication_transaction.json"
PUBLICATION_BACKUP_DIRECTORY_NAME = "publication_backups"
_PUBLICATION_SCHEMA_VERSION = 2
_PUBLICATION_PHASES = {"preparing", "ready", "rolled_back", "committed"}
_FORMAL_OUTPUT_KEYS = frozenset(
    {"whole", "processes", "soma", "report", "workbook"}
)
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLICATION_TEMPORARY = re.compile(
    r"^temporary_IHC_([0-9a-f]{32})_"
    r"(whole|processes|soma|report|workbook)\.tmp$"
)


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


def _regular_file_fingerprint(path: Path, label: str) -> dict[str, Any]:
    raw = Path(path)
    try:
        descriptor = os.open(
            os.fspath(raw),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing: {raw}") from exc
    except OSError as exc:
        raise RuntimeError(
            f"{label} must be a regular non-symlink file: {raw}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"{label} must be a regular non-symlink file: {raw}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if (
            int(after.st_size) != int(metadata.st_size)
            or int(after.st_mtime_ns) != int(metadata.st_mtime_ns)
            or int(after.st_ino) != int(metadata.st_ino)
        ):
            raise RuntimeError(f"{label} changed while its identity was recorded")
    finally:
        os.close(descriptor)
    return {
        "size": int(metadata.st_size),
        "sha256": digest.hexdigest(),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
    }


def _fingerprint_matches(
    path: Path,
    *,
    size: int,
    sha256: str,
) -> bool:
    if not _lexists(path):
        return False
    observed = _regular_file_fingerprint(path, "Publication file")
    return (
        int(observed["size"]) == int(size)
        and str(observed["sha256"]) == str(sha256)
    )


def publication_journal_path(runtime_root: Path) -> Path:
    return Path(runtime_root).resolve() / "recovery" / PUBLICATION_JOURNAL_NAME


def _backup_directory(runtime_root: Path) -> Path:
    return (
        Path(runtime_root).resolve()
        / "recovery"
        / PUBLICATION_BACKUP_DIRECTORY_NAME
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


def _sync_file(path: Path) -> None:
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"Publication sync path is not a regular file: {path}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_journal(path: Path, payload: Mapping[str, Any]) -> None:
    if _lexists(path.parent):
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise RuntimeError(
                "Publication recovery directory must be a real directory"
            )
    else:
        path.parent.mkdir(parents=True)
    if _lexists(path) and path.is_symlink():
        raise RuntimeError("Publication recovery journal must not be a symlink")
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


def _validated_entries(
    payload: Mapping[str, Any],
    *,
    result_root: Path,
    runtime_root: Path,
) -> list[dict[str, Any]]:
    if int(payload.get("schema_version", -1)) != _PUBLICATION_SCHEMA_VERSION:
        raise RuntimeError("Unsupported publication recovery journal")
    phase = str(payload.get("phase", ""))
    if phase not in _PUBLICATION_PHASES:
        raise RuntimeError("Publication recovery journal has an invalid phase")
    if Path(str(payload.get("result_root", ""))).resolve() != result_root:
        raise RuntimeError(
            "Publication recovery journal belongs to a different Result folder"
        )
    expected_backup_root = _backup_directory(runtime_root)
    if Path(str(payload.get("backup_root", ""))).resolve() != expected_backup_root:
        raise RuntimeError(
            "Publication recovery journal has an invalid backup location"
        )

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RuntimeError("Publication recovery journal has no output entries")
    entries: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    observed_names: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise RuntimeError("Publication recovery entry is invalid")
        key = str(raw.get("key", ""))
        name = str(raw.get("name", ""))
        if (
            _SAFE_KEY.fullmatch(key) is None
            or not name
            or Path(name).name != name
            or key in observed_keys
            or name in observed_names
        ):
            raise RuntimeError("Publication recovery entry is unsafe or repeated")
        backup_name = f"{key}.backup"
        if str(raw.get("backup_name", "")) != backup_name:
            raise RuntimeError("Publication recovery backup name is invalid")
        existed = raw.get("existed")
        if not isinstance(existed, bool):
            raise RuntimeError("Publication recovery existence flag is invalid")
        try:
            new_size = int(raw["new_size"])
            new_sha256 = str(raw["new_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Publication recovery new-file identity is invalid"
            ) from exc
        if new_size < 0 or _SHA256.fullmatch(new_sha256) is None:
            raise RuntimeError(
                "Publication recovery new-file identity is invalid"
            )
        old_size: int | None = None
        old_sha256: str | None = None
        if existed:
            try:
                old_size = int(raw["old_size"])
                old_sha256 = str(raw["old_sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Publication recovery backup identity is invalid"
                ) from exc
            if old_size < 0 or _SHA256.fullmatch(old_sha256) is None:
                raise RuntimeError(
                    "Publication recovery backup identity is invalid"
                )
        published_device = raw.get("published_device")
        published_inode = raw.get("published_inode")
        if (published_device is None) != (published_inode is None):
            raise RuntimeError(
                "Publication recovery committed-file identity is incomplete"
            )
        if published_device is not None:
            try:
                published_device = int(published_device)
                published_inode = int(published_inode)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Publication recovery committed-file identity is invalid"
                ) from exc
            if published_device < 0 or published_inode < 0:
                raise RuntimeError(
                    "Publication recovery committed-file identity is invalid"
                )
        observed_keys.add(key)
        observed_names.add(name)
        entries.append(
            {
                "key": key,
                "name": name,
                "existed": existed,
                "new_size": new_size,
                "new_sha256": new_sha256,
                "old_size": old_size,
                "old_sha256": old_sha256,
                "published_device": published_device,
                "published_inode": published_inode,
                "final_path": result_root / name,
                "backup_path": expected_backup_root / backup_name,
            }
        )
    if observed_keys != _FORMAL_OUTPUT_KEYS:
        raise RuntimeError(
            "Publication recovery journal does not describe the formal "
            "five-file bundle"
        )
    return entries


def _strict_publication_temporaries(result_root: Path) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {}
    for path in result_root.iterdir():
        match = _PUBLICATION_TEMPORARY.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Publication temporary path must be a regular non-symlink file: "
                f"{path.name}"
            )
        observed.setdefault(match.group(1), set()).add(path.name)
    return observed


def _transaction_temporary_token(
    *,
    result_root: Path,
    payload: Mapping[str, Any],
) -> str | None:
    recorded = payload.get("temporary_token")
    if recorded is not None:
        token = str(recorded)
        if re.fullmatch(r"[0-9a-f]{32}", token) is None:
            raise RuntimeError("Publication temporary token is invalid")
        return token
    before_raw = payload.get("temporary_names_before", [])
    if not isinstance(before_raw, list) or any(
        not isinstance(name, str) for name in before_raw
    ):
        raise RuntimeError("Publication temporary baseline is invalid")
    before = set(before_raw)
    observed = _strict_publication_temporaries(result_root)
    candidate_tokens = {
        token
        for token, names in observed.items()
        if any(name not in before for name in names)
    }
    if len(candidate_tokens) > 1:
        raise RuntimeError(
            "More than one publication temporary token was found in Result"
        )
    return next(iter(candidate_tokens), None)


def _remove_transaction_temporaries(
    *,
    result_root: Path,
    token: str | None,
) -> None:
    if token is None:
        return
    for key in sorted(_FORMAL_OUTPUT_KEYS):
        path = result_root / f"temporary_IHC_{token}_{key}.tmp"
        if not _lexists(path):
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Publication temporary path is unsafe: " + path.name
            )
        path.unlink()
    _sync_directory(result_root)


def _remove_journal_temporaries(recovery_root: Path) -> None:
    if not recovery_root.is_dir():
        return
    for path in recovery_root.glob("temporary_publication_transaction*.tmp"):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Publication journal temporary path is unsafe: " + path.name
            )
        path.unlink(missing_ok=True)
    _sync_directory(recovery_root)


def _clean_transaction_storage(
    *,
    result_root: Path,
    runtime_root: Path,
    journal: Path,
    payload: dict[str, Any],
) -> None:
    token = _transaction_temporary_token(
        result_root=result_root,
        payload=payload,
    )
    if token is not None and payload.get("temporary_token") != token:
        payload["temporary_token"] = token
        _write_journal(journal, payload)
    _remove_transaction_temporaries(
        result_root=result_root,
        token=token,
    )
    backup_root = _backup_directory(runtime_root)
    if _lexists(backup_root):
        if backup_root.is_symlink() or not backup_root.is_dir():
            raise RuntimeError(
                "Publication backup directory must be a real directory"
            )
        shutil.rmtree(backup_root)
    journal.unlink(missing_ok=True)
    _remove_journal_temporaries(journal.parent)
    _sync_directory(journal.parent)


def recover_publication_transaction(
    *,
    result_root: Path,
    runtime_root: Path,
) -> bool:
    """Recover one interrupted formal-output publication, if present."""

    result_root = _require_real_directory(result_root, "Result")
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    recovery_root = _require_safe_recovery_root(runtime_root, create=False)
    journal = publication_journal_path(runtime_root)
    if not _lexists(journal):
        if recovery_root.is_dir():
            _remove_journal_temporaries(recovery_root)
        return False
    if journal.is_symlink() or not journal.is_file():
        raise RuntimeError(
            "Publication recovery journal must be a regular non-symlink file"
        )

    payload = json.loads(journal.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Publication recovery journal must be a JSON object")
    entries = _validated_entries(
        payload,
        result_root=result_root,
        runtime_root=runtime_root,
    )
    phase = str(payload["phase"])

    if phase == "committed":
        committed_is_complete = True
        for entry in entries:
            final_path = entry["final_path"]
            if not _lexists(final_path):
                committed_is_complete = False
                continue
            if final_path.is_symlink() or not final_path.is_file():
                raise RuntimeError(
                    "Committed publication output path is unsafe: "
                    f"{final_path.name}"
                )
            if not _fingerprint_matches(
                final_path,
                size=entry["new_size"],
                sha256=entry["new_sha256"],
            ):
                metadata = final_path.lstat()
                same_committed_inode = (
                    entry["published_device"] is not None
                    and int(metadata.st_dev) == entry["published_device"]
                    and int(metadata.st_ino) == entry["published_inode"]
                )
                if not same_committed_inode:
                    raise RuntimeError(
                        "Committed publication output was replaced by an "
                        f"unrecognized file: {final_path.name}"
                    )
                committed_is_complete = False
        if not committed_is_complete:
            phase = "ready"

    if phase == "ready":
        missing_backups = [
            entry["key"]
            for entry in entries
            if entry["existed"] and not _lexists(entry["backup_path"])
        ]
        if missing_backups:
            raise RuntimeError(
                "Publication recovery cannot restore the previous outputs; "
                "missing backups: "
                + ", ".join(missing_backups)
            )
        for entry in entries:
            backup_path = entry["backup_path"]
            if entry["existed"]:
                if backup_path.is_symlink() or not backup_path.is_file():
                    raise RuntimeError(
                        "Publication recovery backup path is unsafe: "
                        f"{backup_path.name}"
                    )
                if not _fingerprint_matches(
                    backup_path,
                    size=entry["old_size"],
                    sha256=entry["old_sha256"],
                ):
                    raise RuntimeError(
                        "Publication recovery backup identity changed: "
                        f"{backup_path.name}"
                    )
            final_path = entry["final_path"]
            if _lexists(final_path) and (
                final_path.is_symlink() or not final_path.is_file()
            ):
                raise RuntimeError(
                    "Publication recovery output path is unsafe: "
                    f"{final_path.name}"
                )
            if _lexists(final_path):
                matches_new = _fingerprint_matches(
                    final_path,
                    size=entry["new_size"],
                    sha256=entry["new_sha256"],
                )
                matches_old = bool(entry["existed"]) and _fingerprint_matches(
                    final_path,
                    size=entry["old_size"],
                    sha256=entry["old_sha256"],
                )
                metadata = final_path.lstat()
                matches_committed_inode = (
                    payload.get("phase") == "committed"
                    and entry["published_device"] is not None
                    and int(metadata.st_dev) == entry["published_device"]
                    and int(metadata.st_ino) == entry["published_inode"]
                )
                if not (matches_new or matches_old or matches_committed_inode):
                    raise RuntimeError(
                        "Publication recovery output was replaced by an "
                        f"unrecognized file: {final_path.name}"
                    )
        for entry in entries:
            final_path = entry["final_path"]
            if entry["existed"]:
                restore_path = result_root / (
                    f"temporary_publication_restore_{entry['key']}.tmp"
                )
                try:
                    shutil.copy2(entry["backup_path"], restore_path)
                    _sync_file(restore_path)
                    if not _fingerprint_matches(
                        restore_path,
                        size=entry["old_size"],
                        sha256=entry["old_sha256"],
                    ):
                        raise RuntimeError(
                            "Publication recovery restore copy failed identity "
                            f"validation: {entry['key']}"
                        )
                    os.replace(restore_path, final_path)
                finally:
                    restore_path.unlink(missing_ok=True)
            else:
                final_path.unlink(missing_ok=True)
        _sync_directory(result_root)
        payload["phase"] = "rolled_back"
        _write_journal(journal, payload)

    _clean_transaction_storage(
        result_root=result_root,
        runtime_root=runtime_root,
        journal=journal,
        payload=payload,
    )
    return True


def _begin_publication(
    *,
    final_files: Mapping[str, Path],
    new_fingerprints: Mapping[str, Mapping[str, Any]],
    result_root: Path,
    runtime_root: Path,
) -> tuple[Path, dict[str, Any]]:
    result_root = _require_real_directory(result_root, "Result")
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    _require_safe_recovery_root(runtime_root, create=True)
    journal = publication_journal_path(runtime_root)
    if _lexists(journal):
        if journal.is_symlink():
            raise RuntimeError(
                "Publication recovery journal must not be a symlink"
            )
        raise RuntimeError(
            "An unresolved publication transaction must be recovered before "
            "starting a new publication"
        )
    backup_root = _backup_directory(runtime_root)
    if _lexists(backup_root) and (
        backup_root.is_symlink() or not backup_root.is_dir()
    ):
        raise RuntimeError(
            "Publication backup directory must be a real directory"
        )
    entries: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    for key, raw_path in final_files.items():
        normalized_key = str(key)
        final_path = Path(raw_path)
        if final_path.is_symlink():
            raise ValueError(
                "Formal publication output paths must not be symlinks"
            )
        if (
            _lexists(final_path)
            and not final_path.is_file()
        ):
            raise ValueError(
                "Formal publication outputs must be regular files"
            )
        final_parent = _require_real_directory(
            final_path.parent,
            "Formal publication output parent",
        )
        if (
            _SAFE_KEY.fullmatch(normalized_key) is None
            or final_parent != result_root
            or final_path.name in observed_names
            or normalized_key not in new_fingerprints
        ):
            raise ValueError(
                "Formal publication outputs must be direct children of Result "
                "with simple keys and distinct names"
            )
        observed_names.add(final_path.name)
        new_identity = new_fingerprints[normalized_key]
        old_identity: dict[str, Any] | None = None
        if final_path.is_file():
            old_identity = _regular_file_fingerprint(
                final_path,
                "Previous formal output",
            )
        entries.append(
            {
                "key": normalized_key,
                "name": final_path.name,
                "backup_name": f"{normalized_key}.backup",
                "existed": final_path.is_file(),
                "new_size": int(new_identity["size"]),
                "new_sha256": str(new_identity["sha256"]),
                "old_size": (
                    None if old_identity is None else int(old_identity["size"])
                ),
                "old_sha256": (
                    None
                    if old_identity is None
                    else str(old_identity["sha256"])
                ),
            }
        )
    if not entries:
        raise ValueError("Formal publication has no output files")

    payload: dict[str, Any] = {
        "schema_version": _PUBLICATION_SCHEMA_VERSION,
        "phase": "preparing",
        "result_root": str(result_root),
        "backup_root": str(backup_root),
        "entries": entries,
        "temporary_names_before": sorted(
            name
            for names in _strict_publication_temporaries(result_root).values()
            for name in names
        ),
    }
    _write_journal(journal, payload)
    try:
        if _lexists(backup_root):
            if backup_root.is_symlink() or not backup_root.is_dir():
                raise RuntimeError(
                    "Publication backup directory must be a real directory"
                )
            shutil.rmtree(backup_root)
        backup_root.mkdir(parents=True)
        for entry in entries:
            if entry["existed"]:
                shutil.copy2(
                    result_root / entry["name"],
                    backup_root / entry["backup_name"],
                )
                _sync_file(backup_root / entry["backup_name"])
                copied_identity = _regular_file_fingerprint(
                    backup_root / entry["backup_name"],
                    "Publication backup",
                )
                if (
                    int(copied_identity["size"]) != int(entry["old_size"])
                    or str(copied_identity["sha256"])
                    != str(entry["old_sha256"])
                ):
                    raise RuntimeError(
                        "Publication backup failed identity validation: "
                        f"{entry['key']}"
                    )
        _sync_directory(backup_root)
        payload["phase"] = "ready"
        _write_journal(journal, payload)
    except BaseException:
        recover_publication_transaction(
            result_root=result_root,
            runtime_root=runtime_root,
        )
        raise
    return journal, payload


def publish_with_publication_journal(
    *,
    publish_callback: Callable[..., Any],
    call_args: Sequence[Any],
    call_kwargs: Mapping[str, Any],
    result_root: Path,
    runtime_root: Path,
) -> Any:
    """Wrap the base publisher with one recoverable workspace transaction."""

    result_root = _require_real_directory(result_root, "Result")
    runtime_root = _require_real_directory(runtime_root, "Runtime")
    bound = inspect.signature(publish_callback).bind(*call_args, **call_kwargs)
    staged_files = bound.arguments.get("staged_files")
    final_files = bound.arguments.get("final_files")
    run_dir = bound.arguments.get("run_dir")
    if (
        not isinstance(staged_files, dict)
        or not isinstance(final_files, dict)
        or run_dir is None
    ):
        raise TypeError(
            "Formal publisher call is missing staged_files, final_files, or run_dir"
        )
    run_dir = _require_real_directory(Path(run_dir), "Publication staging run")
    staged_keys = set(staged_files)
    final_keys = set(final_files)
    if (
        staged_keys != final_keys
        or staged_keys != _FORMAL_OUTPUT_KEYS
    ):
        raise ValueError(
            "Formal publication requires matching staged/final entries for "
            "whole, processes, soma, report, and workbook"
        )
    new_fingerprints: dict[str, dict[str, Any]] = {}
    for key, path in staged_files.items():
        staged_path = Path(path)
        staged_identity = _regular_file_fingerprint(
            staged_path,
            "Formal staged output",
        )
        try:
            staged_path.resolve().relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(
                "Formal staged outputs must be inside the Fiji run directory"
            ) from exc
        current_parent = staged_path.parent
        while True:
            if current_parent.is_symlink():
                raise ValueError(
                    "Formal staged output parents must not be symlinks"
                )
            try:
                if current_parent.samefile(run_dir):
                    break
            except FileNotFoundError as exc:
                raise ValueError(
                    "Formal staged output parent is unavailable"
                ) from exc
            next_parent = current_parent.parent
            if next_parent == current_parent:
                raise ValueError(
                    "Formal staged output is not safely contained in its run"
                )
            current_parent = next_parent
        new_fingerprints[str(key)] = staged_identity

    journal, payload = _begin_publication(
        final_files=final_files,
        new_fingerprints=new_fingerprints,
        result_root=result_root,
        runtime_root=runtime_root,
    )
    try:
        result = publish_callback(*call_args, **call_kwargs)
        entry_by_key = {
            str(entry["key"]): entry for entry in payload["entries"]
        }
        for key, path in final_files.items():
            final_path = Path(path)
            if not _lexists(final_path):
                raise RuntimeError(
                    f"Formal publisher did not create {final_path.name}"
                )
            if final_path.is_symlink() or not final_path.is_file():
                raise RuntimeError(
                    f"Formal publisher created an unsafe output: {final_path.name}"
                )
            _sync_file(final_path)
            expected = entry_by_key[str(key)]
            observed = _regular_file_fingerprint(
                final_path,
                "Published formal output",
            )
            if (
                int(observed["size"]) != int(expected["new_size"])
                or str(observed["sha256"]) != str(expected["new_sha256"])
            ):
                raise RuntimeError(
                    "Formal publisher output failed identity validation: "
                    f"{final_path.name}"
                )
            expected["published_device"] = int(observed["device"])
            expected["published_inode"] = int(observed["inode"])
        _sync_directory(result_root)
        payload["phase"] = "committed"
        _write_journal(journal, payload)
    except BaseException as publication_error:
        try:
            recover_publication_transaction(
                result_root=result_root,
                runtime_root=runtime_root,
            )
        except BaseException as recovery_error:
            raise RuntimeError(
                "Formal output publication failed and automatic recovery "
                f"also failed; journal retained at {journal}: {recovery_error}"
            ) from publication_error
        raise

    recover_publication_transaction(
        result_root=result_root,
        runtime_root=runtime_root,
    )
    return result
