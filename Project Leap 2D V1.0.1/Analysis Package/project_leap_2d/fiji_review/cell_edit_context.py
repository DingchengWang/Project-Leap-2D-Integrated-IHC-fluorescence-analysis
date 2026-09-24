from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CELL_EDIT_CONTEXT_SCHEMA = "project-leap-2d.cell-edit-context"
CELL_EDIT_CONTEXT_VERSION = 1
CELL_EDIT_CONTEXT_BASENAME = "cell_edit_context"
CELL_EDIT_CONTEXT_MAX_DIMENSION = 16_384
CELL_EDIT_CONTEXT_MAX_PIXELS = 100_000_000
CELL_EDIT_CONTEXT_MAX_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
CELL_EDIT_CONTEXT_MAX_ARCHIVE_BYTES = 768 * 1024 * 1024
CELL_EDIT_SOURCE_FULL_HASH_LIMIT_BYTES = 16 * 1024 * 1024
CELL_EDIT_SOURCE_SAMPLE_BYTES = 1024 * 1024

_ARRAY_NAMES = (
    "dapi_projection",
    "structural_map",
    "canonical_nucleus_core_labels",
    "canonical_nucleus_extent_labels",
    "whole_labels",
    "soma_labels",
    "process_labels",
)
_LABEL_NAMES = frozenset(
    (
        "canonical_nucleus_core_labels",
        "canonical_nucleus_extent_labels",
        "whole_labels",
        "soma_labels",
        "process_labels",
    )
)


class CellEditContextError(RuntimeError):
    """Raised when a cell-edit evidence package is invalid or corrupt."""


@dataclass(frozen=True)
class CellEditContextPaths:
    npz_path: Path
    json_path: Path
    content_sha256: str


@dataclass(frozen=True)
class LoadedCellEditContext:
    npz_path: Path
    json_path: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise CellEditContextError(
                f"Cell-edit context does not contain array {name!r}"
            ) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path, *, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sampled_file_sha256(
    path: Path,
    *,
    file_size: int,
    sample_bytes: int,
) -> str:
    sample_size = min(max(1, int(sample_bytes)), int(file_size))
    offsets = sorted(
        {
            0,
            max(0, (int(file_size) - sample_size) // 2),
            max(0, int(file_size) - sample_size),
        }
    )
    digest = hashlib.sha256()
    digest.update(b"project-leap-2d-sampled-file-sha256-v1\0")
    digest.update(str(int(file_size)).encode("ascii"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            block = handle.read(sample_size)
            digest.update(b"\0offset=")
            digest.update(str(offset).encode("ascii"))
            digest.update(b"\0length=")
            digest.update(str(len(block)).encode("ascii"))
            digest.update(b"\0")
            digest.update(block)
    return digest.hexdigest()


def source_file_fingerprint(
    path: Path | str,
    *,
    full_hash_limit_bytes: int = CELL_EDIT_SOURCE_FULL_HASH_LIMIT_BYTES,
    sample_bytes: int = CELL_EDIT_SOURCE_SAMPLE_BYTES,
) -> dict[str, Any]:
    """Fingerprint a source image without copying or routinely hashing it in full."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise CellEditContextError(f"Source image is not a regular file: {source}")
    stat = source.stat()
    size = int(stat.st_size)
    if size < 1:
        raise CellEditContextError(f"Source image is empty: {source}")
    if int(full_hash_limit_bytes) < 0:
        raise ValueError("full_hash_limit_bytes must not be negative")
    if int(sample_bytes) < 1:
        raise ValueError("sample_bytes must be positive")

    if size <= int(full_hash_limit_bytes):
        strategy = "full_sha256_v1"
        value_sha256 = _sha256_file(source)
        sampled_offsets: list[int] | None = None
    else:
        strategy = "sampled_sha256_v1:first-middle-last"
        sample_size = min(int(sample_bytes), size)
        sampled_offsets = sorted(
            {
                0,
                max(0, (size - sample_size) // 2),
                max(0, size - sample_size),
            }
        )
        value_sha256 = _sampled_file_sha256(
            source,
            file_size=size,
            sample_bytes=sample_size,
        )

    record: dict[str, Any] = {
        "path": str(source),
        "size_bytes": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "sha256_strategy": strategy,
        "sha256": value_sha256,
    }
    if sampled_offsets is not None:
        record["sample_bytes"] = min(int(sample_bytes), size)
        record["sample_offsets"] = sampled_offsets
    fingerprint_basis = {
        key: record[key]
        for key in (
            "path",
            "size_bytes",
            "mtime_ns",
            "device",
            "inode",
            "sha256_strategy",
            "sha256",
        )
    }
    record["fingerprint_sha256"] = hashlib.sha256(
        _canonical_json_bytes(fingerprint_basis)
    ).hexdigest()
    return record


def _normalise_selected_z(selected_z: Mapping[str, Any] | Sequence[int]) -> dict[str, Any]:
    if isinstance(selected_z, Mapping):
        try:
            start = int(selected_z["z_start_1based"])
            end = int(selected_z["z_end_1based_inclusive"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CellEditContextError(
                "selected_z must define integer z_start_1based and "
                "z_end_1based_inclusive"
            ) from exc
        projection = str(selected_z.get("projection", "")).strip()
    else:
        if isinstance(selected_z, (str, bytes)) or len(selected_z) != 2:
            raise CellEditContextError(
                "selected_z must be a mapping or a two-item 1-based range"
            )
        try:
            start, end = (int(value) for value in selected_z)
        except (TypeError, ValueError) as exc:
            raise CellEditContextError(
                "selected_z range values must be integers"
            ) from exc
        projection = ""
    if start < 1 or end < start:
        raise CellEditContextError(
            "selected_z must be a valid inclusive 1-based range"
        )
    result: dict[str, Any] = {
        "z_start_1based": start,
        "z_end_1based_inclusive": end,
    }
    if projection:
        result["projection"] = projection
    return result


def _positive_finite(value: Any, field: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CellEditContextError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise CellEditContextError(f"{field} must be a positive finite number")
    return number


def _normalise_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(calibration, Mapping):
        raise CellEditContextError("calibration must be a mapping")
    width = _positive_finite(calibration.get("pixel_width_um"), "pixel_width_um")
    height = _positive_finite(calibration.get("pixel_height_um"), "pixel_height_um")
    depth = _positive_finite(
        calibration.get("pixel_depth_um"),
        "pixel_depth_um",
        allow_none=True,
    )
    result: dict[str, Any] = {
        "pixel_width_um": width,
        "pixel_height_um": height,
        "pixel_depth_um": depth,
    }
    for key in (
        "pixel_width_source",
        "pixel_height_source",
        "pixel_depth_source",
        "unit",
    ):
        if key in calibration and calibration[key] is not None:
            result[key] = str(calibration[key])
    return result


def _normalise_analysis_context(
    analysis_mode: Any,
    structural_channel: Any,
    structural_paths: Mapping[str, Path | str],
) -> tuple[str, str]:
    mode = str(analysis_mode).strip().lower()
    if mode not in {"egfp", "gfap_only"}:
        raise CellEditContextError(
            "analysis_mode must be 'egfp' or 'gfap_only'"
        )
    channel_names = tuple(str(channel).strip() for channel in structural_paths)
    if structural_channel is None:
        preferred = "GFAP" if mode == "gfap_only" else "eGFP"
        channel = preferred if preferred in channel_names else (
            channel_names[0] if len(channel_names) == 1 else ""
        )
    else:
        channel = str(structural_channel).strip()
    if not channel or channel not in channel_names:
        raise CellEditContextError(
            "structural_channel must identify one structural_paths channel"
        )
    if mode == "gfap_only" and channel != "GFAP":
        raise CellEditContextError(
            "GFAP-only Cell Edit context must use the GFAP structural channel"
        )
    if mode == "egfp" and channel != "eGFP":
        raise CellEditContextError(
            "eGFP Cell Edit context must use the eGFP structural channel"
        )
    return mode, channel


def _normalise_nucleus_records(
    nucleus_records: Sequence[Mapping[str, Any]] | None,
    *,
    canonical_extent_labels: np.ndarray,
    selected_z: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return one strict, JSON-safe record for every canonical 2D nucleus.

    The canonical inventory can be used without a 3D record source.  In that
    case Z bounds remain explicitly unknown so Split can invoke its bounded
    local DAPI model instead of inventing depth evidence.
    """

    extent_ids = tuple(
        int(value) for value in np.unique(canonical_extent_labels) if int(value) > 0
    )
    if nucleus_records is None:
        source_records: Sequence[Mapping[str, Any]] = tuple(
            {
                "instance_id": instance_id,
                "accepted": True,
                "dapi_valid": True,
                "identity_status": "resolved",
                "z_min_0based": None,
                "z_max_0based_inclusive": None,
            }
            for instance_id in extent_ids
        )
    else:
        if isinstance(nucleus_records, (str, bytes)) or not isinstance(
            nucleus_records, Sequence
        ):
            raise CellEditContextError(
                "nucleus_records must be a sequence of mappings"
            )
        source_records = nucleus_records

    allowed_statuses = frozenset(
        ("resolved", "ambiguous", "model_proposal", "raw_dapi", "projection_only")
    )
    normalized: list[dict[str, Any]] = []
    observed_ids: list[int] = []
    z_start_0based = int(selected_z["z_start_1based"]) - 1
    z_end_0based = int(selected_z["z_end_1based_inclusive"]) - 1
    for raw in source_records:
        if not isinstance(raw, Mapping):
            raise CellEditContextError("Each nucleus record must be a mapping")
        try:
            instance_id = int(
                raw.get(
                    "instance_id",
                    raw.get("nucleus_id_2d", raw.get("object_id_3d", 0)),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CellEditContextError(
                "Each nucleus record must have a positive integer instance_id"
            ) from exc
        if instance_id < 1:
            raise CellEditContextError(
                "Each nucleus record must have a positive integer instance_id"
            )
        if not isinstance(raw.get("accepted"), (bool, np.bool_)):
            raise CellEditContextError(
                f"Nucleus {instance_id} accepted must be boolean"
            )
        if not isinstance(raw.get("dapi_valid"), (bool, np.bool_)):
            raise CellEditContextError(
                f"Nucleus {instance_id} dapi_valid must be boolean"
            )
        identity_status = str(raw.get("identity_status", "")).strip().lower()
        if identity_status not in allowed_statuses:
            raise CellEditContextError(
                f"Nucleus {instance_id} has an invalid identity_status"
            )
        z_min_raw = raw.get("z_min_0based")
        z_max_raw = raw.get("z_max_0based_inclusive")
        if (z_min_raw is None) != (z_max_raw is None):
            raise CellEditContextError(
                f"Nucleus {instance_id} must provide both Z bounds or neither"
            )
        if z_min_raw is None:
            z_min = None
            z_max = None
        else:
            try:
                z_min = int(z_min_raw)
                z_max = int(z_max_raw)
            except (TypeError, ValueError) as exc:
                raise CellEditContextError(
                    f"Nucleus {instance_id} Z bounds must be integers or null"
                ) from exc
            if (
                z_min < z_start_0based
                or z_max < z_min
                or z_max > z_end_0based
            ):
                raise CellEditContextError(
                    f"Nucleus {instance_id} Z bounds are outside selected_z"
                )
        normalized_record = {
            "instance_id": instance_id,
            "accepted": bool(raw["accepted"]),
            "dapi_valid": bool(raw["dapi_valid"]),
            "identity_status": identity_status,
            "z_min_0based": z_min,
            "z_max_0based_inclusive": z_max,
        }
        for key in ("source", "confidence"):
            if key in raw and raw[key] is not None:
                if key == "confidence":
                    confidence = float(raw[key])
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        raise CellEditContextError(
                            f"Nucleus {instance_id} confidence must be within [0, 1]"
                        )
                    normalized_record[key] = confidence
                else:
                    normalized_record[key] = str(raw[key])
        if raw.get("owner_display_id") is not None:
            try:
                owner_display_id = int(raw["owner_display_id"])
            except (TypeError, ValueError) as exc:
                raise CellEditContextError(
                    f"Nucleus {instance_id} owner_display_id must be a "
                    "positive integer or null"
                ) from exc
            if owner_display_id < 1:
                raise CellEditContextError(
                    f"Nucleus {instance_id} owner_display_id must be a "
                    "positive integer or null"
                )
            if not bool(raw["accepted"]):
                raise CellEditContextError(
                    f"Nucleus {instance_id} cannot map to an owner display ID "
                    "when accepted is false"
                )
            normalized_record["owner_display_id"] = owner_display_id
        normalized.append(normalized_record)
        observed_ids.append(instance_id)

    if len(observed_ids) != len(set(observed_ids)):
        raise CellEditContextError("nucleus_records repeat an instance_id")
    if tuple(sorted(observed_ids)) != extent_ids:
        raise CellEditContextError(
            "nucleus_records must exactly match canonical nucleus extent IDs"
        )
    return sorted(normalized, key=lambda record: int(record["instance_id"]))


def _validate_nucleus_owner_display_ids(
    nucleus_records: Sequence[Mapping[str, Any]],
    whole_labels: np.ndarray,
    analysis_mode: str,
) -> None:
    mapped = [
        int(record["owner_display_id"])
        for record in nucleus_records
        if record.get("owner_display_id") is not None
    ]
    if len(mapped) != len(set(mapped)):
        raise CellEditContextError(
            "Multiple nuclei map to the same owner display ID"
        )
    whole_ids = {
        int(value) for value in np.unique(whole_labels) if int(value) > 0
    }
    if not set(mapped).issubset(whole_ids):
        raise CellEditContextError(
            "A nucleus owner_display_id is absent from Whole labels"
        )
    if analysis_mode == "gfap_only":
        accepted_ids = {
            int(record["owner_display_id"])
            for record in nucleus_records
            if bool(record["accepted"])
            and record.get("owner_display_id") is not None
        }
        accepted_without_owner = [
            int(record["instance_id"])
            for record in nucleus_records
            if bool(record["accepted"])
            and record.get("owner_display_id") is None
        ]
        if accepted_without_owner or accepted_ids != whole_ids:
            raise CellEditContextError(
                "GFAP-only Cell Edit context must map exactly one accepted "
                "DAPI owner nucleus to every Whole display ID"
            )


def _normalise_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise CellEditContextError(f"{name} must be a 2D YX array")
    if array.dtype.hasobject:
        raise CellEditContextError(f"{name} must not use an object dtype")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise CellEditContextError(f"{name} must use a numeric or boolean dtype")
    if name in _LABEL_NAMES:
        if not np.issubdtype(array.dtype, np.integer):
            raise CellEditContextError(f"{name} must use an integer dtype")
        if array.size and int(array.min()) < 0:
            raise CellEditContextError(f"{name} must not contain negative labels")
        if array.size and int(array.max()) > np.iinfo(np.uint32).max:
            raise CellEditContextError(f"{name} exceeds uint32 label capacity")
        array = np.asarray(array, dtype=np.uint32)
    elif np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise CellEditContextError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _normalise_triplet(
    initial_triplet: Mapping[str, np.ndarray] | Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(initial_triplet, Mapping):
        try:
            values = (
                initial_triplet["whole_labels"],
                initial_triplet["soma_labels"],
                initial_triplet["process_labels"],
            )
        except KeyError as exc:
            raise CellEditContextError(
                "initial_triplet must define whole_labels, soma_labels, and process_labels"
            ) from exc
    else:
        if isinstance(initial_triplet, (str, bytes)) or len(initial_triplet) != 3:
            raise CellEditContextError(
                "initial_triplet must contain Whole, Soma, and Processes arrays"
            )
        values = tuple(initial_triplet)
    return tuple(
        _normalise_array(value, name)
        for value, name in zip(
            values,
            ("whole_labels", "soma_labels", "process_labels"),
        )
    )


def _validate_spatial_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    shapes = {name: tuple(int(v) for v in arrays[name].shape) for name in _ARRAY_NAMES}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise CellEditContextError(
            "All cell-edit context arrays must have the same YX shape: "
            + json.dumps(shapes, sort_keys=True)
        )
    height, width = next(iter(unique_shapes))
    if height < 1 or width < 1:
        raise CellEditContextError("Cell-edit context arrays must not be empty")
    if max(height, width) > CELL_EDIT_CONTEXT_MAX_DIMENSION:
        raise CellEditContextError(
            "Cell-edit context image dimensions exceed the safety limit"
        )
    if height * width > CELL_EDIT_CONTEXT_MAX_PIXELS:
        raise CellEditContextError(
            "Cell-edit context image area exceeds the safety limit"
        )

    whole = arrays["whole_labels"]
    soma = arrays["soma_labels"]
    processes = arrays["process_labels"]
    whole_ids = tuple(int(v) for v in np.unique(whole) if int(v) > 0)
    soma_ids = tuple(int(v) for v in np.unique(soma) if int(v) > 0)
    process_ids = tuple(int(v) for v in np.unique(processes) if int(v) > 0)
    if not whole_ids:
        raise CellEditContextError("The initial triplet contains no Astrocyte ROI")
    if whole_ids != tuple(range(1, len(whole_ids) + 1)):
        raise CellEditContextError(
            "Initial Whole Astrocyte IDs must be consecutive starting at 1"
        )
    if soma_ids != whole_ids or process_ids != whole_ids:
        raise CellEditContextError(
            "Initial Whole, Soma, and Processes must contain identical IDs"
        )
    if np.any((soma > 0) & (soma != whole)):
        raise CellEditContextError(
            "An initial Soma pixel is outside or assigned to a different Whole cell"
        )
    if np.any((processes > 0) & (processes != whole)):
        raise CellEditContextError(
            "An initial Processes pixel is outside or assigned to a different Whole cell"
        )
    occupancy = (soma > 0).astype(np.uint8) + (processes > 0).astype(np.uint8)
    if np.any(occupancy[whole > 0] != 1) or np.any(occupancy[whole == 0] != 0):
        raise CellEditContextError(
            "Initial Soma and Processes must exactly partition Whole"
        )

    core = arrays["canonical_nucleus_core_labels"]
    extent = arrays["canonical_nucleus_extent_labels"]
    if np.any((core > 0) & (extent != core)):
        raise CellEditContextError(
            "Each canonical nucleus core pixel must belong to the same nucleus extent"
        )

    total_bytes = sum(int(arrays[name].nbytes) for name in _ARRAY_NAMES)
    if total_bytes > CELL_EDIT_CONTEXT_MAX_UNCOMPRESSED_BYTES:
        raise CellEditContextError(
            "Cell-edit context exceeds the uncompressed memory safety limit"
        )
    return height, width


def _array_record(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "dtype": contiguous.dtype.str,
        "nbytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
    }


def _content_sha256(metadata_without_content_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"project-leap-2d-cell-edit-context-v1\0"
        + _canonical_json_bytes(metadata_without_content_hash)
    ).hexdigest()


def _replace_file_atomic(temp_path: Path, destination: Path) -> None:
    os.replace(temp_path, destination)
    try:
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f"temporary_{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            np.savez(handle, **{name: arrays[name] for name in _ARRAY_NAMES})
            handle.flush()
            os.fsync(handle.fileno())
        if temp_path.stat().st_size > CELL_EDIT_CONTEXT_MAX_ARCHIVE_BYTES:
            raise CellEditContextError(
                "Cell-edit context archive exceeds the disk safety limit"
            )
        _replace_file_atomic(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"temporary_{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_file_atomic(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def build_cell_edit_context(
    *,
    run_dir: Path | str,
    dapi_path: Path | str,
    structural_paths: Mapping[str, Path | str],
    dapi_projection: np.ndarray,
    structural_map: np.ndarray,
    selected_z: Mapping[str, Any] | Sequence[int],
    calibration: Mapping[str, Any],
    age_profile: str,
    canonical_core_labels: np.ndarray,
    canonical_extent_labels: np.ndarray,
    initial_triplet: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    nucleus_records: Sequence[Mapping[str, Any]] | None = None,
    analysis_mode: str = "egfp",
    structural_channel: str | None = None,
    basename: str = CELL_EDIT_CONTEXT_BASENAME,
) -> CellEditContextPaths:
    """Write the small, immutable evidence package used by Fiji cell edits."""

    destination = Path(run_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise CellEditContextError(
            f"Cell-edit context destination is not a directory: {destination}"
        )
    safe_basename = str(basename).strip()
    if (
        not safe_basename
        or Path(safe_basename).name != safe_basename
        or safe_basename in {".", ".."}
    ):
        raise CellEditContextError("basename must be a plain non-empty filename stem")

    profile = str(age_profile).strip().lower()
    if profile not in {"mature", "neonatal"}:
        raise CellEditContextError("age_profile must be 'mature' or 'neonatal'")
    if not isinstance(structural_paths, Mapping) or not structural_paths:
        raise CellEditContextError(
            "structural_paths must contain at least one structural channel"
        )
    normalized_analysis_mode, normalized_structural_channel = (
        _normalise_analysis_context(
            analysis_mode,
            structural_channel,
            structural_paths,
        )
    )
    if normalized_analysis_mode == "gfap_only" and profile != "mature":
        raise CellEditContextError(
            "GFAP-only cell-edit contexts support mature astrocytes only"
        )

    whole, soma, processes = _normalise_triplet(initial_triplet)
    arrays = {
        "dapi_projection": _normalise_array(dapi_projection, "dapi_projection"),
        "structural_map": _normalise_array(structural_map, "structural_map"),
        "canonical_nucleus_core_labels": _normalise_array(
            canonical_core_labels,
            "canonical_nucleus_core_labels",
        ),
        "canonical_nucleus_extent_labels": _normalise_array(
            canonical_extent_labels,
            "canonical_nucleus_extent_labels",
        ),
        "whole_labels": whole,
        "soma_labels": soma,
        "process_labels": processes,
    }
    height, width = _validate_spatial_arrays(arrays)

    source_records: dict[str, Any] = {
        "DAPI": source_file_fingerprint(dapi_path),
    }
    for channel in sorted(structural_paths):
        name = str(channel).strip()
        if not name or name == "DAPI":
            raise CellEditContextError(
                "structural_paths channel names must be non-empty and must not be DAPI"
            )
        source_records[name] = source_file_fingerprint(structural_paths[channel])

    npz_path = destination / f"{safe_basename}.npz"
    json_path = destination / f"{safe_basename}.json"
    normalized_selected_z = _normalise_selected_z(selected_z)
    normalized_nucleus_records = _normalise_nucleus_records(
        nucleus_records,
        canonical_extent_labels=arrays["canonical_nucleus_extent_labels"],
        selected_z=normalized_selected_z,
    )
    _validate_nucleus_owner_display_ids(
        normalized_nucleus_records,
        arrays["whole_labels"],
        normalized_analysis_mode,
    )
    metadata: dict[str, Any] = {
        "schema": CELL_EDIT_CONTEXT_SCHEMA,
        "schema_version": CELL_EDIT_CONTEXT_VERSION,
        "archive_file": npz_path.name,
        "archive_format": "numpy-npz-stored-v1",
        "image_shape_yx": [height, width],
        "age_profile": profile,
        "analysis_mode": normalized_analysis_mode,
        "structural_channel": normalized_structural_channel,
        "selected_z": normalized_selected_z,
        "calibration": _normalise_calibration(calibration),
        "source_images": source_records,
        "nucleus_records": normalized_nucleus_records,
        "arrays": {name: _array_record(arrays[name]) for name in _ARRAY_NAMES},
    }
    metadata["content_sha256"] = _content_sha256(metadata)

    _write_npz_atomic(npz_path, arrays)
    archive_sha256 = _sha256_file(npz_path)
    committed_metadata = dict(metadata)
    committed_metadata["archive_size_bytes"] = int(npz_path.stat().st_size)
    committed_metadata["archive_sha256"] = archive_sha256
    _write_json_atomic(json_path, committed_metadata)
    return CellEditContextPaths(
        npz_path=npz_path,
        json_path=json_path,
        content_sha256=str(metadata["content_sha256"]),
    )


def _resolve_context_paths(path: Path | str) -> tuple[Path, Path, dict[str, Any]]:
    supplied = Path(path).expanduser().resolve(strict=True)
    json_path = supplied if supplied.suffix.lower() == ".json" else supplied.with_suffix(".json")
    if not json_path.is_file():
        raise CellEditContextError(
            f"Cell-edit context metadata file is missing: {json_path}"
        )
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CellEditContextError(
            f"Cell-edit context metadata is unreadable: {json_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise CellEditContextError("Cell-edit context metadata must be a JSON object")
    archive_name = metadata.get("archive_file")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or not archive_name.endswith(".npz")
    ):
        raise CellEditContextError(
            "Cell-edit context archive_file must be a local NPZ filename"
        )
    npz_path = json_path.parent / archive_name
    if supplied.suffix.lower() == ".npz" and supplied != npz_path:
        raise CellEditContextError(
            "The supplied NPZ path does not match the committed context metadata"
        )
    if not npz_path.is_file():
        raise CellEditContextError(
            f"Cell-edit context archive is missing: {npz_path}"
        )
    return npz_path, json_path, metadata


def validate_cell_edit_source_files(
    metadata: Mapping[str, Any],
    *,
    verify_hashes: bool = False,
) -> None:
    sources = metadata.get("source_images")
    if not isinstance(sources, Mapping) or not sources:
        raise CellEditContextError("Cell-edit context contains no source image records")
    for channel, record in sources.items():
        if not isinstance(record, Mapping):
            raise CellEditContextError(
                f"Invalid source image record for channel {channel!r}"
            )
        try:
            path = Path(str(record["path"]))
            stat = path.stat()
            expected_size = int(record["size_bytes"])
            expected_mtime = int(record["mtime_ns"])
            expected_device = int(record["device"])
            expected_inode = int(record["inode"])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise CellEditContextError(
                f"Source image is missing or unreadable for channel {channel!r}"
            ) from exc
        observed = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_dev),
            int(stat.st_ino),
        )
        expected = (
            expected_size,
            expected_mtime,
            expected_device,
            expected_inode,
        )
        if observed != expected:
            raise CellEditContextError(
                f"Source image fingerprint changed for channel {channel!r}"
            )
        if verify_hashes:
            strategy = str(record.get("sha256_strategy", ""))
            if strategy == "full_sha256_v1":
                observed_sha256 = _sha256_file(path)
            elif strategy == "sampled_sha256_v1:first-middle-last":
                observed_sha256 = _sampled_file_sha256(
                    path,
                    file_size=expected_size,
                    sample_bytes=int(record.get("sample_bytes", 0)),
                )
            else:
                raise CellEditContextError(
                    f"Unknown source SHA-256 strategy for channel {channel!r}"
                )
            if observed_sha256 != record.get("sha256"):
                raise CellEditContextError(
                    f"Source image content changed for channel {channel!r}"
                )


def load_cell_edit_context(
    path: Path | str,
    *,
    verify_sources: bool = False,
    verify_source_hashes: bool = False,
) -> LoadedCellEditContext:
    """Load and verify a committed cell-edit context without pickle support."""

    npz_path, json_path, metadata = _resolve_context_paths(path)
    if metadata.get("schema") != CELL_EDIT_CONTEXT_SCHEMA:
        raise CellEditContextError("Unsupported cell-edit context schema")
    if metadata.get("schema_version") != CELL_EDIT_CONTEXT_VERSION:
        raise CellEditContextError("Unsupported cell-edit context schema version")
    if metadata.get("archive_format") != "numpy-npz-stored-v1":
        raise CellEditContextError("Unsupported cell-edit context archive format")
    if npz_path.stat().st_size > CELL_EDIT_CONTEXT_MAX_ARCHIVE_BYTES:
        raise CellEditContextError(
            "Cell-edit context archive exceeds the disk safety limit"
        )
    try:
        expected_archive_size = int(metadata["archive_size_bytes"])
        expected_archive_sha256 = str(metadata["archive_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CellEditContextError(
            "Cell-edit context archive integrity metadata is incomplete"
        ) from exc
    if npz_path.stat().st_size != expected_archive_size:
        raise CellEditContextError("Cell-edit context archive size check failed")
    if _sha256_file(npz_path) != expected_archive_sha256:
        raise CellEditContextError("Cell-edit context archive integrity check failed")

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(_ARRAY_NAMES):
                raise CellEditContextError(
                    "Cell-edit context archive has unexpected array members"
                )
            arrays = {
                name: _normalise_array(archive[name], name)
                for name in _ARRAY_NAMES
            }
    except CellEditContextError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise CellEditContextError(
            "Cell-edit context archive is unreadable or unsafe"
        ) from exc

    _validate_spatial_arrays(arrays)
    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, Mapping):
        raise CellEditContextError("Cell-edit context array metadata is missing")
    observed_array_records = {
        name: _array_record(arrays[name]) for name in _ARRAY_NAMES
    }
    for name in _ARRAY_NAMES:
        if array_metadata.get(name) != observed_array_records[name]:
            raise CellEditContextError(
                f"Cell-edit context integrity check failed for array {name!r}"
            )

    content_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "content_sha256",
            "archive_size_bytes",
            "archive_sha256",
        }
    }
    expected_content_sha256 = str(metadata.get("content_sha256", ""))
    if _content_sha256(content_metadata) != expected_content_sha256:
        raise CellEditContextError(
            "Cell-edit context content integrity check failed"
        )

    expected_shape = metadata.get("image_shape_yx")
    observed_shape = list(arrays["dapi_projection"].shape)
    if expected_shape != observed_shape:
        raise CellEditContextError(
            "Cell-edit context image shape does not match its metadata"
        )
    normalized_selected_z = _normalise_selected_z(metadata.get("selected_z", {}))
    _normalise_calibration(metadata.get("calibration", {}))
    normalized_mode, normalized_channel = _normalise_analysis_context(
        metadata.get("analysis_mode"),
        metadata.get("structural_channel"),
        {
            str(channel): Path(str(record.get("path", "")))
            for channel, record in metadata.get("source_images", {}).items()
            if str(channel) != "DAPI" and isinstance(record, Mapping)
        },
    )
    normalized_nucleus_records = _normalise_nucleus_records(
        metadata.get("nucleus_records"),
        canonical_extent_labels=arrays["canonical_nucleus_extent_labels"],
        selected_z=normalized_selected_z,
    )
    _validate_nucleus_owner_display_ids(
        normalized_nucleus_records,
        arrays["whole_labels"],
        normalized_mode,
    )
    if metadata.get("nucleus_records") != normalized_nucleus_records:
        raise CellEditContextError(
            "Cell-edit context nucleus_records are not in canonical form"
        )
    if metadata.get("age_profile") not in {"mature", "neonatal"}:
        raise CellEditContextError("Invalid age_profile in cell-edit context")
    if normalized_mode == "gfap_only" and metadata.get("age_profile") != "mature":
        raise CellEditContextError(
            "GFAP-only cell-edit contexts support mature astrocytes only"
        )
    if (
        metadata.get("analysis_mode") != normalized_mode
        or metadata.get("structural_channel") != normalized_channel
    ):
        raise CellEditContextError(
            "Cell-edit analysis mode metadata is not in canonical form"
        )
    if verify_sources or verify_source_hashes:
        validate_cell_edit_source_files(
            metadata,
            verify_hashes=bool(verify_source_hashes),
        )

    frozen_arrays: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        array.setflags(write=False)
        frozen_arrays[name] = array
    return LoadedCellEditContext(
        npz_path=npz_path,
        json_path=json_path,
        metadata=MappingProxyType(metadata),
        arrays=MappingProxyType(frozen_arrays),
    )


def relocate_cell_edit_context(
    path: Path | str,
    *,
    destination_dir: Path | str,
    basename: str = "analysis_context",
) -> CellEditContextPaths:
    """Commit an existing context inside one Fiji run, then remove its source.

    This compatibility helper supports the staged-context API.  The
    production analysis path builds directly in the final Fiji run instead.
    """

    loaded = load_cell_edit_context(path)
    destination = Path(destination_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    safe_basename = str(basename).strip()
    if (
        not safe_basename
        or Path(safe_basename).name != safe_basename
        or safe_basename in {".", ".."}
    ):
        raise CellEditContextError("basename must be a plain non-empty filename stem")
    target_npz = destination / f"{safe_basename}.npz"
    target_json = destination / f"{safe_basename}.json"
    source_npz = loaded.npz_path.resolve()
    source_json = loaded.json_path.resolve()
    if target_npz.exists() or target_json.exists():
        raise CellEditContextError(
            "Cell-edit context destination already contains analysis_context"
        )

    metadata = dict(loaded.metadata)
    metadata["archive_file"] = target_npz.name
    content_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "content_sha256",
            "archive_size_bytes",
            "archive_sha256",
        }
    }
    metadata["content_sha256"] = _content_sha256(content_metadata)
    temporary_npz = (
        destination / f"temporary_{target_npz.name}.{os.getpid()}.tmp"
    )
    try:
        with source_npz.open("rb") as source, temporary_npz.open("xb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        _replace_file_atomic(temporary_npz, target_npz)
        metadata["archive_size_bytes"] = int(target_npz.stat().st_size)
        metadata["archive_sha256"] = _sha256_file(target_npz)
        _write_json_atomic(target_json, metadata)
        committed = load_cell_edit_context(target_json)
    except BaseException:
        temporary_npz.unlink(missing_ok=True)
        target_json.unlink(missing_ok=True)
        target_npz.unlink(missing_ok=True)
        raise

    if source_json != target_json:
        source_json.unlink()
    if source_npz != target_npz:
        source_npz.unlink()
    return CellEditContextPaths(
        npz_path=committed.npz_path,
        json_path=committed.json_path,
        content_sha256=str(committed.metadata["content_sha256"]),
    )
