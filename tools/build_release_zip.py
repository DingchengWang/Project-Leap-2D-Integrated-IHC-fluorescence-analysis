#!/usr/bin/env python3
"""Validate the frozen payload and build the deterministic release ZIP.

The inner Project Leap 2D package is immutable.  Its file list, sizes,
SHA-256 digests, and Unix modes come exclusively from
``packaging/payload_baseline.json``.  The release archive also carries the
repository-level documentation and third-party license records, plus the
empty workspace directories that Git cannot represent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Iterable, Mapping, NamedTuple, Optional
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPOSITORY_ROOT / "packaging" / "payload_baseline.json"
EXPECTED_INNER_FOLDER = "Project Leap 2D (8-23-26)"
EXPECTED_VERSION = "1.0.0"
EXPECTED_FILE_COUNT = 127
EXPECTED_REQUIRED_DIRECTORIES = {
    "Original Image",
    "Result",
    "Runtime",
    "Runtime/locks",
    "Runtime/matplotlib",
    "Runtime/recovery",
    "Runtime/staging",
}
RELEASE_TOP_FOLDER = (
    "Project-Leap-2D-Integrated-IHC-fluorescence-analysis-1.0.0"
)

# ZIP timestamps have a lower bound of 1980.  The project release date is
# fixed in the baseline, and midnight is used so repeated builds do not leak
# local wall-clock time.
FIXED_ZIP_TIMESTAMP = (2026, 8, 23, 0, 0, 0)

RELEASE_ROOT_FILES = (
    "README.md",
    "README_中文.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
)
RELEASE_LICENSE_FILES = (
    "Cellpose-BSD-3-Clause.txt",
    "InstanSeg-Apache-2.0.txt",
    "InstanSeg-single_channel_nuclei-README.md",
)

# These names may be produced locally but never belong to a release payload.
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__MACOSX",
    "__pycache__",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
}


class ReleaseBuildError(RuntimeError):
    """A validation or release construction failure safe to show to users."""


class BaselineFile(NamedTuple):
    relative_path: str
    size: int
    sha256: str
    mode: int


class ArchiveFile(NamedTuple):
    source: Path
    archive_path: str
    mode: int
    expected_size: Optional[int] = None
    expected_sha256: Optional[str] = None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseBuildError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise ReleaseBuildError(f"{field} must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReleaseBuildError(f"unsafe or non-canonical {field}: {value!r}")
    return value


def _parse_git_mode(value: object, *, relative_path: str) -> int:
    if isinstance(value, bool):
        raise ReleaseBuildError(f"invalid git_mode for {relative_path}: {value!r}")
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        raise ReleaseBuildError(f"invalid git_mode for {relative_path}: {value!r}")
    if text not in {"100644", "100755"}:
        raise ReleaseBuildError(
            f"unsupported git_mode for {relative_path}: {value!r}; "
            "only 100644 and 100755 are allowed"
        )
    return int(text, 8)


def _load_baseline() -> tuple[dict[str, object], list[BaselineFile]]:
    try:
        raw = BASELINE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseBuildError(f"cannot read baseline {BASELINE_PATH}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReleaseBuildError(f"invalid JSON in {BASELINE_PATH}: {exc}") from exc

    if not isinstance(data, dict):
        raise ReleaseBuildError("payload baseline root must be a JSON object")
    if data.get("schema_version") != 1:
        raise ReleaseBuildError(
            f"unsupported payload schema_version: {data.get('schema_version')!r}"
        )
    if data.get("release_folder") != EXPECTED_INNER_FOLDER:
        raise ReleaseBuildError(
            "payload release_folder differs from the approved inner folder: "
            f"{data.get('release_folder')!r}"
        )
    if data.get("project_version") != EXPECTED_VERSION:
        raise ReleaseBuildError(
            f"payload project_version must be {EXPECTED_VERSION}, got "
            f"{data.get('project_version')!r}"
        )
    if data.get("file_count") != EXPECTED_FILE_COUNT:
        raise ReleaseBuildError(
            f"payload file_count must be {EXPECTED_FILE_COUNT}, got "
            f"{data.get('file_count')!r}"
        )

    files_value = data.get("files")
    if not isinstance(files_value, dict):
        raise ReleaseBuildError("payload files must be a JSON object")
    if len(files_value) != EXPECTED_FILE_COUNT:
        raise ReleaseBuildError(
            f"payload contains {len(files_value)} file records; "
            f"expected {EXPECTED_FILE_COUNT}"
        )

    baseline_files: list[BaselineFile] = []
    for untrusted_path, metadata in files_value.items():
        relative_path = _safe_relative_path(
            untrusted_path, field="payload file path"
        )
        if not isinstance(metadata, dict):
            raise ReleaseBuildError(
                f"metadata for {relative_path} must be a JSON object"
            )
        size = metadata.get("bytes")
        digest = metadata.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseBuildError(f"invalid byte count for {relative_path}: {size!r}")
        if not _is_sha256(digest):
            raise ReleaseBuildError(f"invalid SHA-256 for {relative_path}: {digest!r}")
        mode = _parse_git_mode(
            metadata.get("git_mode"), relative_path=relative_path
        )
        baseline_files.append(
            BaselineFile(relative_path, size, digest, mode)
        )

    required_directories = data.get("required_directories")
    if not isinstance(required_directories, list) or not required_directories:
        raise ReleaseBuildError("payload required_directories must be a non-empty list")
    normalized_required = [
        _safe_relative_path(value, field="required directory")
        for value in required_directories
    ]
    if len(set(normalized_required)) != len(normalized_required):
        raise ReleaseBuildError("payload required_directories contains duplicates")
    if set(normalized_required) != EXPECTED_REQUIRED_DIRECTORIES:
        missing = sorted(EXPECTED_REQUIRED_DIRECTORIES - set(normalized_required))
        unexpected = sorted(set(normalized_required) - EXPECTED_REQUIRED_DIRECTORIES)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ReleaseBuildError(
            "payload required_directories differs from the approved seven paths; "
            + "; ".join(details)
        )

    executable_files = data.get("executable_files")
    if not isinstance(executable_files, list):
        raise ReleaseBuildError("payload executable_files must be a list")
    normalized_executables = {
        _safe_relative_path(value, field="executable file")
        for value in executable_files
    }
    file_map = {entry.relative_path: entry for entry in baseline_files}
    missing_executables = sorted(normalized_executables - file_map.keys())
    if missing_executables:
        raise ReleaseBuildError(
            "executable_files are absent from payload files: "
            + ", ".join(missing_executables)
        )
    modes_marked_executable = {
        entry.relative_path
        for entry in baseline_files
        if stat.S_IMODE(entry.mode) & 0o111
    }
    if normalized_executables != modes_marked_executable:
        raise ReleaseBuildError(
            "payload executable_files does not exactly match executable git modes"
        )

    for digest_field in (
        "immutable_baseline_sha256",
        "release_package_manifest_sha256",
    ):
        if not _is_sha256(data.get(digest_field)):
            raise ReleaseBuildError(f"payload {digest_field} is not a valid SHA-256")

    checksum_text = "".join(
        f"{entry.sha256}  {entry.relative_path}\n"
        for entry in sorted(
            baseline_files, key=lambda entry: entry.relative_path.encode("utf-8")
        )
    )
    rebuilt_baseline_digest = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
    if data["immutable_baseline_sha256"] != rebuilt_baseline_digest:
        raise ReleaseBuildError(
            "payload immutable_baseline_sha256 does not match the file records: "
            f"{data['immutable_baseline_sha256']}, rebuilt {rebuilt_baseline_digest}"
        )

    release_manifest_path = "validation/release_package_files.json"
    try:
        release_manifest_digest = file_map[release_manifest_path].sha256
    except KeyError as exc:
        raise ReleaseBuildError(
            f"payload files lacks required {release_manifest_path} record"
        ) from exc
    if data["release_package_manifest_sha256"] != release_manifest_digest:
        raise ReleaseBuildError(
            "payload release_package_manifest_sha256 does not match the "
            f"{release_manifest_path} file record"
        )

    return data, sorted(baseline_files, key=lambda entry: entry.relative_path.encode("utf-8"))


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise ReleaseBuildError(f"cannot read {path}: {exc}") from exc
    return size, digest.hexdigest()


def _validate_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ReleaseBuildError(f"missing {description}: {path}") from exc
    except OSError as exc:
        raise ReleaseBuildError(f"cannot inspect {description} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ReleaseBuildError(f"symlink is not allowed for {description}: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseBuildError(f"{description} is not a regular file: {path}")
    return metadata


def _is_runtime_or_cache_path(relative_path: str, required_dirs: set[str]) -> bool:
    parts = PurePosixPath(relative_path).parts
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts[:-1]):
        return True
    if parts and (
        parts[-1] in EXCLUDED_FILE_NAMES
        or parts[-1].startswith("._")
        or parts[-1].endswith(".pyc")
    ):
        return True
    return any(
        relative_path == directory or relative_path.startswith(directory + "/")
        for directory in required_dirs
    )


def _actual_nonruntime_files(inner_root: Path, required_dirs: set[str]) -> set[str]:
    actual: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        inner_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        relative_directory = (
            current.relative_to(inner_root).as_posix()
            if current != inner_root
            else ""
        )

        retained_directories: list[str] = []
        for name in directory_names:
            candidate = current / name
            relative = f"{relative_directory}/{name}" if relative_directory else name
            try:
                candidate_metadata = candidate.lstat()
            except OSError as exc:
                raise ReleaseBuildError(f"cannot inspect {candidate}: {exc}") from exc
            if stat.S_ISLNK(candidate_metadata.st_mode):
                raise ReleaseBuildError(f"symlink is not allowed in inner package: {candidate}")
            if not _is_runtime_or_cache_path(relative, required_dirs):
                retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in file_names:
            relative = f"{relative_directory}/{name}" if relative_directory else name
            path = current / name
            if _is_runtime_or_cache_path(relative, required_dirs):
                continue
            _validate_regular_file(path, description="inner package file")
            actual.add(relative)
    return actual


def _validate_inner_payload(
    baseline: Mapping[str, object], baseline_files: Iterable[BaselineFile]
) -> list[BaselineFile]:
    inner_root = REPOSITORY_ROOT / EXPECTED_INNER_FOLDER
    try:
        root_metadata = inner_root.lstat()
    except FileNotFoundError as exc:
        raise ReleaseBuildError(f"missing inner package directory: {inner_root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ReleaseBuildError(f"inner package is not a real directory: {inner_root}")

    entries = list(baseline_files)
    expected_paths = {entry.relative_path for entry in entries}
    required_dirs = set(baseline["required_directories"])  # validated by _load_baseline
    actual_paths = _actual_nonruntime_files(inner_root, required_dirs)
    missing = sorted(expected_paths - actual_paths, key=lambda value: value.encode("utf-8"))
    unexpected = sorted(actual_paths - expected_paths, key=lambda value: value.encode("utf-8"))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ReleaseBuildError("inner package file set differs from baseline; " + "; ".join(details))

    failures: list[str] = []
    for entry in entries:
        path = inner_root / entry.relative_path
        metadata = _validate_regular_file(path, description="baseline payload file")
        actual_mode = stat.S_IFREG | stat.S_IMODE(metadata.st_mode)
        if metadata.st_size != entry.size:
            failures.append(
                f"{entry.relative_path}: size {metadata.st_size}, expected {entry.size}"
            )
            continue
        if actual_mode != entry.mode:
            failures.append(
                f"{entry.relative_path}: mode {actual_mode:o}, expected {entry.mode:o}"
            )
            continue
        _, actual_digest = _hash_file(path)
        if actual_digest != entry.sha256:
            failures.append(
                f"{entry.relative_path}: SHA-256 {actual_digest}, expected {entry.sha256}"
            )
    if failures:
        raise ReleaseBuildError(
            "inner payload validation failed:\n  - " + "\n  - ".join(failures)
        )
    return entries


def _validate_legal_and_readme_files() -> list[ArchiveFile]:
    archive_files: list[ArchiveFile] = []
    for name in RELEASE_ROOT_FILES:
        path = REPOSITORY_ROOT / name
        _validate_regular_file(path, description="required repository file")
        archive_files.append(ArchiveFile(path, name, 0o644))

    licenses_root = REPOSITORY_ROOT / "LICENSES"
    try:
        licenses_metadata = licenses_root.lstat()
    except FileNotFoundError as exc:
        raise ReleaseBuildError(f"missing third-party license directory: {licenses_root}") from exc
    if stat.S_ISLNK(licenses_metadata.st_mode) or not stat.S_ISDIR(licenses_metadata.st_mode):
        raise ReleaseBuildError(f"LICENSES is not a real directory: {licenses_root}")

    approved_names = set(RELEASE_LICENSE_FILES)
    actual_names: set[str] = set()
    try:
        license_entries = list(licenses_root.iterdir())
    except OSError as exc:
        raise ReleaseBuildError(f"cannot inspect LICENSES: {exc}") from exc
    for path in license_entries:
        if path.name in EXCLUDED_FILE_NAMES or path.name.startswith("._"):
            continue
        _validate_regular_file(path, description="third-party license file")
        actual_names.add(path.name)
    if actual_names != approved_names:
        missing = sorted(approved_names - actual_names)
        unexpected = sorted(actual_names - approved_names)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ReleaseBuildError(
            "LICENSES differs from the approved file list; " + "; ".join(details)
        )
    for name in RELEASE_LICENSE_FILES:
        path = licenses_root / name
        archive_files.append(ArchiveFile(path, f"LICENSES/{name}", 0o644))
    return archive_files


def _zip_info(archive_path: str, *, mode: int, is_directory: bool) -> zipfile.ZipInfo:
    normalized = archive_path.rstrip("/") + ("/" if is_directory else "")
    info = zipfile.ZipInfo(normalized, date_time=FIXED_ZIP_TIMESTAMP)
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.compress_type = zipfile.ZIP_STORED
    info.comment = b""
    info.extra = b""
    info.internal_attr = 0
    file_type = stat.S_IFDIR if is_directory else stat.S_IFREG
    info.external_attr = ((file_type | mode) & 0xFFFF) << 16
    if is_directory:
        info.external_attr |= 0x10
    return info


def _parent_directories(archive_path: str) -> set[str]:
    path = PurePosixPath(archive_path)
    return {parent.as_posix() for parent in path.parents if parent.as_posix() != "."}


def _write_archive_file(
    archive: zipfile.ZipFile, entry: ArchiveFile
) -> tuple[int, str]:
    metadata = _validate_regular_file(entry.source, description="release source file")
    if entry.expected_size is not None and metadata.st_size != entry.expected_size:
        raise ReleaseBuildError(
            f"{entry.source} changed after validation: size {metadata.st_size}, "
            f"expected {entry.expected_size}"
        )
    if entry.expected_size is not None:
        actual_mode = stat.S_IFREG | stat.S_IMODE(metadata.st_mode)
        expected_mode = stat.S_IFREG | entry.mode
        if actual_mode != expected_mode:
            raise ReleaseBuildError(
                f"{entry.source} changed after validation: mode {actual_mode:o}, "
                f"expected {expected_mode:o}"
            )

    digest = hashlib.sha256()
    size = 0
    info = _zip_info(entry.archive_path, mode=entry.mode, is_directory=False)
    try:
        with entry.source.open("rb") as source, archive.open(info, "w") as destination:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                destination.write(block)
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise ReleaseBuildError(f"cannot archive {entry.source}: {exc}") from exc

    actual_digest = digest.hexdigest()
    if entry.expected_size is not None and size != entry.expected_size:
        raise ReleaseBuildError(
            f"{entry.source} changed while archiving: size {size}, "
            f"expected {entry.expected_size}"
        )
    if entry.expected_sha256 is not None and actual_digest != entry.expected_sha256:
        raise ReleaseBuildError(
            f"{entry.source} changed while archiving: SHA-256 {actual_digest}, "
            f"expected {entry.expected_sha256}"
        )
    return size, actual_digest


def _archive_plan(
    baseline: Mapping[str, object], baseline_files: Iterable[BaselineFile]
) -> tuple[list[str], list[ArchiveFile]]:
    root_files = _validate_legal_and_readme_files()
    inner_root = REPOSITORY_ROOT / EXPECTED_INNER_FOLDER
    archive_files = list(root_files)
    for entry in baseline_files:
        archive_files.append(
            ArchiveFile(
                inner_root / entry.relative_path,
                f"{EXPECTED_INNER_FOLDER}/{entry.relative_path}",
                stat.S_IMODE(entry.mode),
                entry.size,
                entry.sha256,
            )
        )

    seen_paths: set[str] = set()
    for entry in archive_files:
        if entry.archive_path in seen_paths:
            raise ReleaseBuildError(f"duplicate archive path: {entry.archive_path}")
        seen_paths.add(entry.archive_path)

    directories = {RELEASE_TOP_FOLDER}
    for entry in archive_files:
        full_path = f"{RELEASE_TOP_FOLDER}/{entry.archive_path}"
        directories.update(_parent_directories(full_path))
    for required in baseline["required_directories"]:  # validated by _load_baseline
        full_path = (
            f"{RELEASE_TOP_FOLDER}/{EXPECTED_INNER_FOLDER}/{required}"
        )
        directories.add(full_path)
        directories.update(_parent_directories(full_path))

    prefixed_files = [
        ArchiveFile(
            entry.source,
            f"{RELEASE_TOP_FOLDER}/{entry.archive_path}",
            entry.mode,
            entry.expected_size,
            entry.expected_sha256,
        )
        for entry in archive_files
    ]
    return (
        sorted(directories, key=lambda value: value.encode("utf-8")),
        sorted(prefixed_files, key=lambda entry: entry.archive_path.encode("utf-8")),
    )


def _default_output_path() -> Path:
    return REPOSITORY_ROOT / "dist" / f"{RELEASE_TOP_FOLDER}.zip"


def _build_zip(
    output_path: Path, directories: Iterable[str], archive_files: Iterable[ArchiveFile]
) -> tuple[int, str]:
    output_path = output_path.expanduser().resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise ReleaseBuildError(f"refusing to overwrite existing output: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReleaseBuildError(
            f"cannot create output directory {output_path.parent}: {exc}"
        ) from exc

    directory_list = list(directories)
    file_list = list(archive_files)
    ordered_members = [
        (directory.rstrip("/") + "/", True, directory)
        for directory in directory_list
    ] + [
        (entry.archive_path, False, entry)
        for entry in file_list
    ]
    ordered_members.sort(key=lambda item: item[0].encode("utf-8"))

    created = False
    try:
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(descriptor, "w+b") as output_handle:
            with zipfile.ZipFile(output_handle, mode="w", allowZip64=True) as archive:
                archive.comment = b""
                for _, is_directory, member in ordered_members:
                    if is_directory:
                        archive.writestr(
                            _zip_info(member, mode=0o755, is_directory=True), b""
                        )
                    else:
                        _write_archive_file(archive, member)
    except FileExistsError as exc:
        raise ReleaseBuildError(f"refusing to overwrite existing output: {output_path}") from exc
    except (
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ReleaseBuildError,
    ) as exc:
        if created:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if isinstance(exc, ReleaseBuildError):
            raise
        raise ReleaseBuildError(f"failed to create ZIP {output_path}: {exc}") from exc

    size, digest = _hash_file(output_path)
    return size, digest


def _parse_arguments(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the current payload and build a deterministic "
            "Project Leap 2D release ZIP."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "output ZIP path (default: dist/"
            f"{RELEASE_TOP_FOLDER}.zip under the repository root)"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate all release inputs without writing a ZIP",
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.output is not None:
        parser.error("--output cannot be combined with --check")
    return arguments


def main(argv: Optional[list[str]] = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        baseline, baseline_files = _load_baseline()
        validated_files = _validate_inner_payload(baseline, baseline_files)
        directories, archive_files = _archive_plan(baseline, validated_files)
        if arguments.check:
            print(
                f"OK: validated {len(baseline_files)} payload files, repository legal records, "
                f"and {len(baseline['required_directories'])} required empty directories."
            )
            return 0

        output = arguments.output or _default_output_path()
        size, digest = _build_zip(output, directories, archive_files)
        print(f"Created: {output.expanduser().resolve(strict=False)}")
        print(f"Bytes: {size}")
        print(f"SHA-256: {digest}")
        return 0
    except ReleaseBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
