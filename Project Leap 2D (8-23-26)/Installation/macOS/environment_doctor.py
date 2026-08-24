from __future__ import annotations

import argparse
import base64
import csv
import errno
import hashlib
import importlib.metadata
import io
import json
import os
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


TRUSTED_INTEGRITY_MANIFEST_SHA256 = {
    "python_wheel_integrity.json": (
        "5c2987fccb41781cc0bc35d563027e8fd44ec5fdf873db47fb664301fde4d52d"
    ),
    "fiji_tree_integrity.json": (
        "dd4d945b5db14642d0bf734979fe41afc50d64034fa681c0414b806f9cf0a7e5"
    ),
    "managed_python_integrity.json": (
        "f4b612f2e410142f53c93d8099069affb915686cdd01bc8c5a0cdb748973c987"
    ),
}

TRUSTED_VIRTUAL_ENVIRONMENT_FILES = {
    "lib/python3.9/site-packages/_virtualenv.pth": (
        "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
        18,
    ),
    "lib/python3.9/site-packages/_virtualenv.py": (
        "6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d",
        4342,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_wheel_record_row(
    *,
    package_path: str,
    recorded_hash: str,
    recorded_size: int,
    resolved_path: Path,
    environment_root: Path,
) -> tuple[tuple[str, str, str], bool]:
    """Normalize only uv-generated console scripts that embed the venv path."""

    parts = Path(package_path).parts
    environment_relative = resolved_path.relative_to(environment_root)
    is_generated_entrypoint = (
        len(parts) == 5
        and parts[:4] == ("..", "..", "..", "bin")
        and environment_relative == Path("bin") / parts[-1]
    )
    if not is_generated_entrypoint:
        return (
            package_path,
            f"sha256={recorded_hash}",
            str(recorded_size),
        ), False

    value = resolved_path.read_bytes()
    environment_root_bytes = str(environment_root).encode("utf-8")
    if environment_root_bytes not in value:
        return (
            package_path,
            f"sha256={recorded_hash}",
            str(recorded_size),
        ), False
    normalized = value.replace(
        environment_root_bytes,
        b"${VIRTUAL_ENVIRONMENT_ROOT}",
    )
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(normalized).digest()
    ).rstrip(b"=")
    return (
        package_path,
        f"sha256={digest.decode('ascii')}",
        str(len(normalized)),
    ), True


def canonical_wheel_record_sha256(
    rows: list[tuple[str, str, str]],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(sorted(rows, key=lambda row: row[0]))
    return hashlib.sha256(output.getvalue().encode("utf-8")).hexdigest()


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def locked_versions(lock_file: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(f"Unpinned dependency in lock file: {line}")
        name, version = line.split("==", 1)
        result[canonical_name(name)] = version
    if not result:
        raise RuntimeError("Dependency lock file is empty.")
    return result


def require_locked_environment(lock_file: Path) -> None:
    mismatches: list[str] = []
    for name, expected in locked_versions(lock_file).items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}: missing")
            continue
        if observed != expected:
            mismatches.append(f"{name}: expected {expected}, found {observed}")
    if mismatches:
        raise RuntimeError("Dependency lock mismatch: " + "; ".join(mismatches))


def load_integrity_manifests(
    lock_file: Path,
    *,
    expected_fiji_zip_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_paths = {
        name: lock_file.parent / name
        for name in TRUSTED_INTEGRITY_MANIFEST_SHA256
    }
    for name, path in manifest_paths.items():
        if sha256_file(path) != TRUSTED_INTEGRITY_MANIFEST_SHA256[name]:
            raise RuntimeError(f"{name} does not match its trusted SHA-256.")
    wheel_path = manifest_paths["python_wheel_integrity.json"]
    fiji_path = manifest_paths["fiji_tree_integrity.json"]
    managed_python_path = manifest_paths["managed_python_integrity.json"]
    wheel = json.loads(wheel_path.read_text(encoding="utf-8"))
    fiji = json.loads(fiji_path.read_text(encoding="utf-8"))
    managed_python = json.loads(
        managed_python_path.read_text(encoding="utf-8")
    )
    locked = locked_versions(lock_file)
    if (
        wheel.get("schema_version") != 1
        or wheel.get("algorithm") != "canonical-wheel-record-sha256-v2"
        or set(wheel.get("distributions", {})) != set(locked)
    ):
        raise RuntimeError("Python wheel integrity manifest is inconsistent.")
    for name, expected_version in locked.items():
        row = wheel["distributions"][name]
        if (
            row.get("version") != expected_version
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                row.get("canonical_record_sha256", ""),
            )
            or not isinstance(row.get("record_entries"), int)
            or not isinstance(row.get("hashed_entries"), int)
            or not isinstance(row.get("normalized_environment_files"), int)
        ):
            raise RuntimeError(
                f"Python wheel integrity record is invalid for {name}."
            )
    if (
        fiji.get("schema_version") != 1
        or fiji.get("algorithm") != "sorted-tree-sha256-v1"
        or fiji.get("source_zip_sha256") != expected_fiji_zip_sha256
        or not re.fullmatch(r"[0-9a-f]{64}", fiji.get("tree_sha256", ""))
        or sorted(fiji.get("excluded_mutable_relative_paths", []))
        != [".checksums", "db.xml.gz"]
    ):
        raise RuntimeError("Fiji integrity manifest is inconsistent.")
    if (
        managed_python.get("schema_version") != 1
        or managed_python.get("algorithm")
        != "normalized-python-runtime-sha256-v4"
        or managed_python.get("runtime")
        != "cpython-3.9.25-macos-aarch64-none"
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            managed_python.get("tree_sha256", ""),
        )
        or sorted(managed_python.get("excluded_relative_paths", []))
        != [".lock", ".temp"]
        or managed_python.get("excluded_directory_names") != []
    ):
        raise RuntimeError(
            "Managed Python integrity manifest is inconsistent."
        )
    return wheel, fiji, managed_python


def require_wheel_file_integrity(
    lock_file: Path,
    release_dir: Path,
    manifest: dict[str, Any],
) -> set[Path]:
    locked = locked_versions(lock_file)
    installed: dict[str, importlib.metadata.Distribution] = {}
    site_packages = (
        release_dir
        / "Environment"
        / "lib"
        / "python3.9"
        / "site-packages"
    )
    site_packages_metadata = site_packages.lstat()
    if stat.S_ISLNK(site_packages_metadata.st_mode) or not stat.S_ISDIR(
        site_packages_metadata.st_mode
    ):
        raise RuntimeError(
            "Python site-packages is not a managed directory."
        )
    for distribution in importlib.metadata.distributions(
        path=[str(site_packages)]
    ):
        name = canonical_name(distribution.metadata["Name"])
        if name in installed:
            raise RuntimeError(f"Duplicate installed distribution: {name}")
        installed[name] = distribution
    if set(installed) != set(locked):
        missing = sorted(set(locked) - set(installed))
        extra = sorted(set(installed) - set(locked))
        raise RuntimeError(
            f"Installed distribution set differs from the lock; "
            f"missing={missing}, extra={extra}"
        )

    environment_root = (release_dir / "Environment").resolve(strict=True)
    covered_paths: set[Path] = set()
    for name, expected_version in locked.items():
        distribution = installed[name]
        if distribution.version != expected_version:
            raise RuntimeError(
                f"{name}: expected {expected_version}, "
                f"found {distribution.version}"
            )
        expected = manifest["distributions"][name]
        expected_record_path = Path(expected["record_path"])
        if (
            expected_record_path.is_absolute()
            or ".." in expected_record_path.parts
            or expected_record_path.name != "RECORD"
            or ".dist-info" not in expected_record_path.parent.name
        ):
            raise RuntimeError(f"{name}: trusted wheel RECORD path is invalid.")
        record_file = Path(distribution.locate_file(expected_record_path))
        try:
            resolved_record = record_file.resolve(strict=True)
            resolved_record.relative_to(environment_root)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"{name}: wheel RECORD is missing or outside the managed "
                "environment."
            ) from exc
        if record_file.is_symlink() or not resolved_record.is_file():
            raise RuntimeError(f"{name}: wheel RECORD is not a regular file.")
        covered_paths.add(resolved_record)

        files = list(distribution.files or ())
        record_paths = [
            path
            for path in files
            if path.name == "RECORD" and ".dist-info/" in path.as_posix()
        ]
        if len(record_paths) != 1:
            raise RuntimeError(f"{name}: expected exactly one wheel RECORD.")
        record_path = record_paths[0]
        if record_path.as_posix() != expected_record_path.as_posix():
            raise RuntimeError(f"{name}: wheel RECORD path changed.")
        if len(files) != expected["record_entries"]:
            raise RuntimeError(f"{name}: wheel RECORD entry count changed.")
        file_names = [path.as_posix() for path in files]
        if len(file_names) != len(set(file_names)):
            raise RuntimeError(f"{name}: wheel RECORD contains duplicate paths.")

        hashed_count = 0
        normalized_environment_files = 0
        canonical_rows: list[tuple[str, str, str]] = []
        for package_path in files:
            located = Path(distribution.locate_file(package_path))
            try:
                resolved = located.resolve(strict=True)
                resolved.relative_to(environment_root)
            except (FileNotFoundError, ValueError) as exc:
                raise RuntimeError(
                    f"{name}: installed file is missing or outside the "
                    f"managed environment: {package_path}"
                ) from exc
            if located.is_symlink() or not resolved.is_file():
                raise RuntimeError(
                    f"{name}: installed path is not a regular file: "
                    f"{package_path}"
                )
            covered_paths.add(resolved)
            if package_path == record_path:
                if package_path.hash is not None:
                    raise RuntimeError(f"{name}: RECORD unexpectedly hashes itself.")
                canonical_rows.append((package_path.as_posix(), "", ""))
                continue
            if (
                package_path.hash is None
                or package_path.hash.mode != "sha256"
                or package_path.size is None
            ):
                raise RuntimeError(
                    f"{name}: installed file lacks a SHA-256 and size: "
                    f"{package_path}"
                )
            if resolved.stat().st_size != package_path.size:
                raise RuntimeError(
                    f"{name}: installed file size changed: {package_path}"
                )
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            observed = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=")
            if observed.decode("ascii") != package_path.hash.value:
                raise RuntimeError(
                    f"{name}: installed file digest changed: {package_path}"
                )
            canonical_row, normalized = canonical_wheel_record_row(
                package_path=package_path.as_posix(),
                recorded_hash=package_path.hash.value,
                recorded_size=package_path.size,
                resolved_path=resolved,
                environment_root=environment_root,
            )
            canonical_rows.append(canonical_row)
            normalized_environment_files += int(normalized)
            hashed_count += 1
        if hashed_count != expected["hashed_entries"]:
            raise RuntimeError(f"{name}: hashed wheel entry count changed.")
        if (
            normalized_environment_files
            != expected["normalized_environment_files"]
        ):
            raise RuntimeError(
                f"{name}: generated entry-point inventory changed."
            )
        if (
            canonical_wheel_record_sha256(canonical_rows)
            != expected["canonical_record_sha256"]
        ):
            raise RuntimeError(f"{name}: canonical wheel RECORD changed.")
    return covered_paths


def managed_tree_fingerprint(
    root: Path,
    excluded_relative_paths: set[str],
    excluded_directory_names: set[str],
    *,
    label: str,
    normalization_root: Optional[Path] = None,
    normalization_token: str = "${MANAGED_TREE_ROOT}",
    allowed_symlink_root: Optional[Path] = None,
    strip_libpython_signature: bool = False,
) -> dict[str, int | str]:
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise RuntimeError(f"{label} root is not a managed directory.")
    root = root.resolve(strict=True)
    normalization_roots: list[Path] = []
    if normalization_root is not None:
        normalization_roots = sorted(
            {
                normalization_root.absolute(),
                normalization_root.resolve(strict=True),
            },
            key=lambda value: len(str(value)),
            reverse=True,
        )
    if allowed_symlink_root is None:
        allowed_symlink_root = root
    else:
        allowed_symlink_root = allowed_symlink_root.resolve(strict=True)
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    symlink_count = 0
    total_file_bytes = 0

    for base, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        base_path = Path(base)
        directory_names[:] = sorted(directory_names)
        for name in sorted(directory_names + file_names):
            path = base_path / name
            relative = path.relative_to(root).as_posix()
            if relative in excluded_relative_paths or (
                name in directory_names
                and name in excluded_directory_names
            ):
                if name in directory_names:
                    directory_names.remove(name)
                continue
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    path.resolve(strict=True).relative_to(allowed_symlink_root)
                except (FileNotFoundError, ValueError) as exc:
                    raise RuntimeError(
                        f"{label} symbolic link escapes its managed tree: "
                        f"{relative}"
                    ) from exc
                target = os.readlink(path)
                for normalized_root in normalization_roots:
                    target = target.replace(
                        str(normalized_root),
                        normalization_token,
                    )
                payload = f"L\0{relative}\0{target}\n".encode("utf-8")
                symlink_count += 1
                if name in directory_names:
                    directory_names.remove(name)
            elif stat.S_ISDIR(metadata.st_mode):
                payload = f"D\0{relative}\n".encode("utf-8")
                directory_count += 1
            elif stat.S_ISREG(metadata.st_mode):
                executable = 1 if metadata.st_mode & 0o111 else 0
                file_bytes = path.read_bytes()
                if normalization_roots:
                    if (
                        strip_libpython_signature
                        and relative.endswith("/lib/libpython3.9.dylib")
                    ):
                        file_bytes = normalized_macho_payload(
                            file_bytes,
                            tuple(
                                os.fsencode(value)
                                for value in normalization_roots
                            ),
                            normalization_token.encode("utf-8"),
                        )
                    for normalized_root in normalization_roots:
                        file_bytes = file_bytes.replace(
                            os.fsencode(normalized_root),
                            normalization_token.encode("utf-8"),
                        )
                normalized_size = len(file_bytes)
                payload = (
                    f"F\0{relative}\0{normalized_size}\0{executable}\0"
                    f"{hashlib.sha256(file_bytes).hexdigest()}\n"
                ).encode("utf-8")
                file_count += 1
                total_file_bytes += normalized_size
            else:
                raise RuntimeError(
                    f"{label} contains an unsupported special path: {relative}"
                )
            digest.update(payload)
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "total_file_bytes": total_file_bytes,
    }


def normalized_macho_payload(
    value: bytes,
    installation_roots: tuple[bytes, ...],
    root_token: bytes,
) -> bytes:
    if len(value) < 32 or struct.unpack_from("<I", value, 0)[0] != 0xFEEDFACF:
        raise RuntimeError("Managed Python libpython has an invalid Mach-O header.")
    command_count = struct.unpack_from("<I", value, 16)[0]
    offset = 32
    canonical_commands: list[bytes] = []
    signature_found = False
    signature_offset = 0
    signature_size = 0
    first_section_offset = len(value)
    for _ in range(command_count):
        if offset + 8 > len(value):
            raise RuntimeError(
                "Managed Python libpython has invalid Mach-O commands."
            )
        command, command_size = struct.unpack_from("<II", value, offset)
        if command_size < 8 or offset + command_size > len(value):
            raise RuntimeError(
                "Managed Python libpython has invalid Mach-O commands."
            )
        command_payload = value[offset : offset + command_size]
        if command == 0x0D:
            if command_size < 24:
                raise RuntimeError(
                    "Managed Python libpython has an invalid library identity."
                )
            name_offset = struct.unpack_from("<I", value, offset + 8)[0]
            if name_offset < 24 or name_offset >= command_size:
                raise RuntimeError(
                    "Managed Python libpython has an invalid library identity."
                )
            raw_name = command_payload[name_offset:].split(b"\0", 1)[0]
            normalized_name = raw_name
            for installation_root in installation_roots:
                normalized_name = normalized_name.replace(
                    installation_root,
                    root_token,
                )
            if normalized_name == raw_name:
                raise RuntimeError(
                    "Managed Python libpython identity is outside its "
                    "managed runtime."
                )
            canonical_size = (
                name_offset + len(normalized_name) + 1 + 7
            ) // 8 * 8
            canonical = bytearray(command_payload[:name_offset])
            struct.pack_into("<I", canonical, 4, canonical_size)
            canonical.extend(normalized_name)
            canonical.extend(b"\0" * (canonical_size - len(canonical)))
            command_payload = bytes(canonical)
        if command == 0x1D:
            if command_size < 16 or signature_found:
                raise RuntimeError(
                    "Managed Python libpython has an invalid code signature."
                )
            signature_offset, signature_size = struct.unpack_from(
                "<II",
                value,
                offset + 8,
            )
            if (
                signature_size == 0
                or signature_offset + signature_size > len(value)
            ):
                raise RuntimeError(
                    "Managed Python libpython has an invalid code signature."
                )
            signature_found = True
        if command == 0x19:
            if command_size < 72:
                raise RuntimeError(
                    "Managed Python libpython has invalid segment commands."
                )
            section_count = struct.unpack_from("<I", value, offset + 64)[0]
            if command_size < 72 + section_count * 80:
                raise RuntimeError(
                    "Managed Python libpython has invalid segment commands."
                )
            for index in range(section_count):
                section_offset = struct.unpack_from(
                    "<I",
                    value,
                    offset + 72 + index * 80 + 48,
                )[0]
                if section_offset > 0:
                    first_section_offset = min(
                        first_section_offset,
                        section_offset,
                    )
        canonical_commands.append(command_payload)
        offset += command_size
    if not signature_found:
        raise RuntimeError(
            "Managed Python libpython has no embedded code signature."
        )
    if first_section_offset <= 32 or first_section_offset > len(value):
        raise RuntimeError(
            "Managed Python libpython has invalid section offsets."
        )
    if signature_offset < first_section_offset:
        raise RuntimeError(
            "Managed Python libpython has an invalid signature offset."
        )
    canonical_header = bytearray(value[:32])
    struct.pack_into(
        "<I",
        canonical_header,
        20,
        sum(len(command) for command in canonical_commands),
    )
    # uv rewrites only LC_ID_DYLIB for the installation root and macOS then
    # regenerates the ad-hoc signature. All other commands and executable
    # sections remain byte-for-byte protected. codesign verifies the omitted
    # signature blob against the original file before this digest is accepted.
    return (
        bytes(canonical_header)
        + b"".join(canonical_commands)
        + value[first_section_offset:signature_offset]
        + value[signature_offset + signature_size :]
    )


def fiji_tree_fingerprint(
    root: Path,
    excluded_relative_paths: set[str],
) -> dict[str, int | str]:
    return managed_tree_fingerprint(
        root,
        excluded_relative_paths,
        set(),
        label="Fiji",
    )


def require_fiji_tree_integrity(
    release_dir: Path,
    manifest: dict[str, Any],
) -> None:
    observed = fiji_tree_fingerprint(
        release_dir / "Fiji",
        set(manifest["excluded_mutable_relative_paths"]),
    )
    for key in (
        "tree_sha256",
        "file_count",
        "directory_count",
        "symlink_count",
        "total_file_bytes",
    ):
        if observed[key] != manifest.get(key):
            raise RuntimeError(f"Fiji managed tree integrity changed ({key}).")


def require_managed_python_tree_integrity(
    release_dir: Path,
    manifest: dict[str, Any],
) -> None:
    observed = managed_tree_fingerprint(
        release_dir / "Managed Python",
        set(manifest["excluded_relative_paths"]),
        set(manifest["excluded_directory_names"]),
        label="Managed Python",
        normalization_root=release_dir / "Managed Python",
        normalization_token="${MANAGED_PYTHON_ROOT}",
        strip_libpython_signature=True,
    )
    runtime_root = (
        release_dir
        / "Managed Python"
        / manifest["runtime"]
    )
    for relative in ("bin/python3.9", "lib/libpython3.9.dylib"):
        completed = subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", runtime_root / relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Managed Python code signature is invalid: {relative}"
            )
    for key in (
        "tree_sha256",
        "file_count",
        "directory_count",
        "symlink_count",
        "total_file_bytes",
    ):
        if observed[key] != manifest.get(key):
            raise RuntimeError(
                f"Managed Python tree integrity changed ({key})."
            )


def require_virtual_environment_files(
    release_dir: Path,
    runtime_name: str,
) -> None:
    release_root = release_dir.resolve(strict=True)
    environment_root = release_dir / "Environment"
    environment_metadata = environment_root.lstat()
    if stat.S_ISLNK(environment_metadata.st_mode) or not stat.S_ISDIR(
        environment_metadata.st_mode
    ):
        raise RuntimeError(
            "Python virtual-environment root is not a managed directory."
        )
    environment_root = environment_root.resolve(strict=True)
    environment_root.relative_to(release_root)
    runtime_root = (
        release_dir / "Managed Python" / runtime_name
    ).resolve(strict=True)
    runtime_root.relative_to(release_root)
    runtime_python = (runtime_root / "bin" / "python3.9").resolve(strict=True)

    expected_links = {
        "python": str(
            release_dir
            / "Managed Python"
            / "cpython-3.9-macos-aarch64-none"
            / "bin"
            / "python3.9"
        ),
        "python3": "python",
        "python3.9": "python",
    }
    for name, expected_target in expected_links.items():
        link = environment_root / "bin" / name
        if not link.is_symlink() or os.readlink(link) != expected_target:
            raise RuntimeError(
                f"Python virtual-environment link changed: bin/{name}"
            )
        if link.resolve(strict=True) != runtime_python:
            raise RuntimeError(
                f"Python virtual-environment link escapes its runtime: "
                f"bin/{name}"
            )

    config_path = environment_root / "pyvenv.cfg"
    config_metadata = config_path.lstat()
    if stat.S_ISLNK(config_metadata.st_mode) or not stat.S_ISREG(
        config_metadata.st_mode
    ):
        raise RuntimeError("Python pyvenv.cfg is not a managed file.")
    config: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.count("=") != 1:
            raise RuntimeError("Python pyvenv.cfg is malformed.")
        key, value = (part.strip() for part in line.split("=", 1))
        if key in config:
            raise RuntimeError("Python pyvenv.cfg contains duplicate keys.")
        config[key] = value
    expected_config = {
        "home": str(
            release_dir
            / "Managed Python"
            / "cpython-3.9-macos-aarch64-none"
            / "bin"
        ),
        "implementation": "CPython",
        "uv": "0.11.16",
        "version_info": "3.9.25",
        "include-system-site-packages": "false",
    }
    if config != expected_config:
        raise RuntimeError("Python pyvenv.cfg changed.")


def require_environment_inventory(
    release_dir: Path,
    wheel_paths: set[Path],
) -> None:
    environment_root = (release_dir / "Environment").resolve(strict=True)
    allowed_skeleton = {
        ".gitignore",
        ".lock",
        "CACHEDIR.TAG",
        "pyvenv.cfg",
        "bin/activate",
        "bin/activate.bat",
        "bin/activate.csh",
        "bin/activate.fish",
        "bin/activate.nu",
        "bin/activate.ps1",
        "bin/activate_this.py",
        "bin/deactivate.bat",
        "bin/pydoc.bat",
        "bin/python",
        "bin/python3",
        "bin/python3.9",
    }
    for relative, (expected_sha256, expected_size) in (
        TRUSTED_VIRTUAL_ENVIRONMENT_FILES.items()
    ):
        path = environment_root / relative
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != expected_size
            or sha256_file(path) != expected_sha256
        ):
            raise RuntimeError(
                f"Python virtual-environment bootstrap changed: {relative}"
            )
    for base, directory_names, file_names in os.walk(
        environment_root,
        topdown=True,
        followlinks=False,
    ):
        directory_names[:] = sorted(directory_names)
        base_path = Path(base)
        for name in sorted(directory_names + file_names):
            path = base_path / name
            relative = path.relative_to(environment_root).as_posix()
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if relative in allowed_skeleton:
                continue
            if relative in TRUSTED_VIRTUAL_ENVIRONMENT_FILES:
                continue
            if stat.S_ISREG(metadata.st_mode):
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(environment_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Python environment file escapes its root: {relative}"
                    ) from exc
                if resolved in wheel_paths:
                    continue
            raise RuntimeError(
                f"Unregistered Python environment path: {relative}"
            )


def require_running_virtual_environment(
    release_dir: Path,
    runtime_name: str,
) -> None:
    expected_prefix = (release_dir / "Environment").resolve(strict=True)
    expected_base = (
        release_dir / "Managed Python" / runtime_name
    ).resolve(strict=True)
    if Path(sys.prefix).resolve(strict=True) != expected_prefix:
        raise RuntimeError("Python sys.prefix is outside the managed environment.")
    if Path(sys.base_prefix).resolve(strict=True) != expected_base:
        raise RuntimeError("Python sys.base_prefix is outside the managed runtime.")
    if Path(sys.executable).resolve(strict=True) != (
        expected_base / "bin" / "python3.9"
    ).resolve(strict=True):
        raise RuntimeError("Python executable is outside the managed runtime.")


def require_project_resources(project_dir: Path) -> None:
    model_dir = project_dir / "project_leap_2d" / "resources" / "models"
    metadata_path = model_dir / "instanseg_single_channel_nuclei.json"
    model_path = model_dir / "instanseg_single_channel_nuclei.pt"
    groovy_path = (
        project_dir
        / "project_leap_2d"
        / "fiji_review"
        / "resources"
        / "astrocyte_roi_reviewer.groovy"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed = sha256_file(model_path)
    if observed != metadata.get("sha256"):
        raise RuntimeError("Bundled InstanSeg model hash does not match its metadata.")
    if not groovy_path.is_file() or groovy_path.stat().st_size == 0:
        raise RuntimeError("Bundled Fiji reviewer resource is missing.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("content", "smoke"),
        required=True,
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--cellpose-sha256", required=True)
    parser.add_argument("--fiji-launcher", type=Path, required=True)
    parser.add_argument("--fiji-zip-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "content":
        try:
            require_project_resources(args.project_dir)
            wheel_manifest, fiji_manifest, managed_python_manifest = (
                load_integrity_manifests(
                    args.lock_file,
                    expected_fiji_zip_sha256=args.fiji_zip_sha256,
                )
            )
        except Exception as exc:
            print(
                "FULL ENVIRONMENT CHECK FAILED: "
                f"package resource integrity: {exc}",
                file=sys.stderr,
            )
            return 20

    try:
        if args.mode == "content":
            require_virtual_environment_files(
                args.release_dir,
                managed_python_manifest["runtime"],
            )
            require_managed_python_tree_integrity(
                args.release_dir,
                managed_python_manifest,
            )
            wheel_paths = require_wheel_file_integrity(
                args.lock_file,
                args.release_dir,
                wheel_manifest,
            )
            require_environment_inventory(args.release_dir, wheel_paths)

            cellpose_model = args.release_dir / "Models" / "cpsam_v2"
            if sha256_file(cellpose_model) != args.cellpose_sha256:
                raise RuntimeError(
                    "Cellpose model hash does not match the "
                    "installation contract."
                )
            if not args.fiji_launcher.is_file():
                raise RuntimeError("Fiji launcher is missing.")
            java_roots = (
                args.release_dir / "Fiji" / "java",
                args.release_dir / "Fiji" / "Fiji.app" / "java",
            )
            if not any(path.is_dir() for path in java_roots):
                raise RuntimeError(
                    "Fiji does not contain its managed Java runtime."
                )
            require_fiji_tree_integrity(args.release_dir, fiji_manifest)
        else:
            runtime_name = "cpython-3.9.25-macos-aarch64-none"
            require_running_virtual_environment(
                args.release_dir,
                runtime_name,
            )
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import openpyxl  # noqa: F401
            import scipy  # noqa: F401
            import skimage  # noqa: F401
            import tifffile  # noqa: F401
            import torch
            from PIL import Image  # noqa: F401
            from cellpose import models  # noqa: F401

            instanseg_model = (
                args.project_dir
                / "project_leap_2d"
                / "resources"
                / "models"
                / "instanseg_single_channel_nuclei.pt"
            )
            torch.jit.load(str(instanseg_model), map_location="cpu").eval()
    except (MemoryError, PermissionError) as exc:
        print(
            f"FULL ENVIRONMENT CHECK STOPPED: non-repairable system error: {exc}",
            file=sys.stderr,
        )
        return 22
    except OSError as exc:
        non_repairable = {
            errno.EIO,
            errno.ENOSPC,
            errno.EROFS,
            errno.EACCES,
            errno.EPERM,
        }
        if exc.errno in non_repairable:
            print(
                "FULL ENVIRONMENT CHECK STOPPED: "
                f"non-repairable filesystem error: {exc}",
                file=sys.stderr,
            )
            return 22
        print(
            f"FULL ENVIRONMENT CHECK FAILED: installed environment integrity: {exc}",
            file=sys.stderr,
        )
        return 21
    except Exception as exc:
        print(
            f"FULL ENVIRONMENT CHECK FAILED: installed environment integrity: {exc}",
            file=sys.stderr,
        )
        return 21

    if args.mode == "content":
        print("Project Leap 2D environment content verification passed.")
    else:
        print("Full Project Leap 2D environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
