"""Package the final Fiji geometry and measured identities as a formal output.

The exporter runs after Fiji's final numbering and measurements. This module
checks that export against fiji_done.json; it never edits or reconstructs ROIs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid
import zipfile


ARCHIVE_NAME = "IHC_2D_ROI_Archive.zip"
_COMPARTMENTS = ("whole", "soma", "processes")
_SUFFIXES = {"whole": "Whole", "soma": "Soma", "processes": "Processes"}


class FinalRoiArchiveError(RuntimeError):
    """Final ROI evidence is missing, invalid, or inconsistent with measurement."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalRoiArchiveError(message)


def _file(root: Path, relative: str) -> Path:
    path = root / relative
    _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
             f"Unsafe final ROI path: {relative}")
    current = path
    while current != root:
        _require(not current.is_symlink(), f"Final ROI path is a symlink: {relative}")
        current = current.parent
    _require(path.is_file(), f"Final ROI file is missing: {relative}")
    return path


def _json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinalRoiArchiveError(f"Cannot read final ROI JSON: {path.name}") from exc
    _require(isinstance(payload, dict), f"Final ROI JSON must be an object: {path.name}")
    return payload


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _pixel_coordinate(value: object) -> bool:
    # Groovy resolves java.awt.Rectangle properties through double-returning
    # getters. JSON therefore carries exact integer pixel bounds as e.g. 991.0.
    return _integer(value) or (
        isinstance(value, float) and math.isfinite(value) and value.is_integer()
    )


def _validate_metadata(export: dict, source: dict, done: dict) -> list[dict]:
    _require(export.get("schema_version") == 1, "Unsupported final ROI export schema")
    count = export.get("roi_count")
    _require(_integer(count) and 0 < count <= 65535, "Invalid final ROI count")
    ids = list(range(1, count + 1))
    _require(done.get("roi_count") == count, "Final ROI count differs from measurement")
    for field in ("final_roi_ids", "final_soma_ids", "final_process_ids"):
        _require(done.get(field) == ids, f"Final ROI IDs differ from measurement: {field}")
    records = export.get("identity_records")
    _require(isinstance(records, list) and len(records) == count,
             "Final ROI identities are incomplete")
    for label, record in zip(ids, records):
        _require(isinstance(record, dict) and record.get("label_id") == label,
                 "Final ROI identities must be in final label order")
        _require(_integer(record.get("original_id")) and record["original_id"] > 0,
                 "Final ROI original identity is invalid")
        _require(isinstance(record.get("cell_uid"), str) and bool(record["cell_uid"]),
                 "Final ROI Cell UID is missing")
        _require(isinstance(record.get("parent_uid"), str), "Final ROI parent UID is invalid")
        _require(isinstance(record.get("owner_nucleus_id"), str), "Final ROI nucleus identity is invalid")
        lineage = record.get("lineage")
        _require(isinstance(lineage, list) and bool(lineage)
                 and all(_integer(item) and item > 0 for item in lineage)
                 and len(set(lineage)) == len(lineage), "Final ROI lineage is invalid")
    uids = [record["cell_uid"] for record in records]
    _require(len(set(uids)) == count and done.get("final_cell_uids") == uids,
             "Final ROI Cell UIDs differ from measurement")
    original_map = {str(record["original_id"]): record["label_id"] for record in records}
    source_map: dict[str, list[int]] = {}
    for record in records:
        for original in record["lineage"]:
            source_map.setdefault(str(original), []).append(record["label_id"])
    _require(done.get("original_to_final_id") == original_map,
             "Final ROI original-to-final mapping differs from measurement")
    _require(done.get("source_to_final_ids") == source_map,
             "Final ROI lineage mapping differs from measurement")
    result_sets = done.get("result_sets")
    geometry = export.get("geometry_records")
    _require(isinstance(result_sets, dict) and isinstance(geometry, dict),
             "Final ROI measurement or geometry records are missing")
    for key in _COMPARTMENTS:
        result = result_sets.get(key, {})
        for field, expected in {
            "rows": count, "astrocyte_ids": ids, "cell_uids": uids,
            "original_astrocyte_ids": [r["original_id"] for r in records],
            "source_original_astrocyte_ids": [r["lineage"] for r in records],
            "parent_cell_uids": [r["parent_uid"] for r in records],
            "owner_nucleus_ids": [r["owner_nucleus_id"] for r in records],
        }.items():
            _require(result.get(field) == expected,
                     f"Final ROI identity differs from measured {key}: {field}")
        shapes = geometry.get(key)
        _require(isinstance(shapes, list) and len(shapes) == count,
                 f"Final {key} geometry records are incomplete")
        for record, shape in zip(records, shapes):
            name = f"Astrocyte_{record['label_id']:03d}_{_SUFFIXES[key]}"
            _require(isinstance(shape, dict) and shape.get("label_id") == record["label_id"]
                     and shape.get("cell_uid") == record["cell_uid"]
                     and shape.get("roi_name") == name,
                     f"Final {key} geometry identity differs from measurement")
            bounds = shape.get("bounds_xywh")
            _require(isinstance(bounds, list) and len(bounds) == 4
                     and all(_pixel_coordinate(v) for v in bounds) and bounds[2] > 0 and bounds[3] > 0,
                     f"Invalid final {key} ROI bounds")

    shape = export.get("source_shape_zyx")
    _require(isinstance(shape, list) and len(shape) == 3
             and all(_integer(v) and v > 0 for v in shape)
             and shape == source.get("expected_shape"), "Final ROI source shape is inconsistent")
    selected_z = export.get("selected_z", {})
    start = selected_z.get("start_1based")
    end = selected_z.get("end_1based_inclusive")
    _require(_integer(start) and _integer(end) and 1 <= start <= end <= shape[0]
             and start == source.get("z_start_1based")
             and end == source.get("z_end_1based_inclusive"), "Final ROI Z range is inconsistent")
    _require(export.get("projection") == str(source.get("projection")).lower(),
             "Final ROI projection differs from measurement")
    _require(export.get("measurement_channel") == source.get("measurement_channel")
             and export.get("source_channels") == source.get("channels"),
             "Final ROI source channels differ from measurement")
    coordinates = export.get("coordinates", {})
    for key, expected in {"axes": "YX", "index_origin": 0, "origin": "top-left",
                          "x_direction": "right", "y_direction": "down",
                          "source_xy_offset": [0, 0]}.items():
        _require(coordinates.get(key) == expected, f"Final ROI coordinate convention is invalid: {key}")
    calibration = export.get("calibration", {})
    for key in ("pixel_width_um", "pixel_height_um"):
        _require(_positive(calibration.get(key)) and calibration[key] == source.get(key),
                 f"Final ROI calibration is inconsistent: {key}")
    image_calibrations = export.get("source_image_calibrations")
    _require(isinstance(image_calibrations, dict)
             and set(image_calibrations) == set(source.get("channels", {})),
             "Final ROI source image calibrations are incomplete")
    for values in image_calibrations.values():
        _require(isinstance(values, dict)
                 and all(_positive(values.get(key)) for key in ("pixel_width", "pixel_height", "pixel_depth"))
                 and all(isinstance(values.get(key), str) for key in ("x_unit", "y_unit", "z_unit"))
                 and all(isinstance(values.get(key), (float, int)) and math.isfinite(values[key])
                         for key in ("x_origin", "y_origin", "z_origin")),
                 "Final ROI source image calibration fields are invalid")
    return records


def build_final_roi_archive(run_dir: Path, destination: Path) -> Path:
    """Validate and atomically write the archive, or raise without publishing.

    ``run_dir`` is the exact Fiji staging directory containing manifest.json,
    fiji_done.json and final_roi/export.json. ``destination`` is the staged ZIP
    passed to the six-output publication transaction by the desktop adapter.
    """
    run_dir = Path(run_dir)
    _require(run_dir.is_dir() and not run_dir.is_symlink(), "Invalid Fiji staging directory")
    run_dir = run_dir.resolve()
    done_path = _file(run_dir, "fiji_done.json")
    source_path = _file(run_dir, "manifest.json")
    fingerprints = {"fiji_done.json": _fingerprint(done_path),
                    "source_manifest.json": _fingerprint(source_path)}
    done, source = _json(done_path), _json(source_path)
    _require(done.get("cancelled") is False, "Fiji has not completed measurement")
    reference = done.get("final_roi_export")
    _require(isinstance(reference, dict) and reference.get("schema_version") == 1
             and reference.get("manifest") == "final_roi/export.json",
             "Fiji completion has no final ROI export")
    export_path = _file(run_dir, reference["manifest"])
    fingerprints["final_roi/export.json"] = _fingerprint(export_path)
    _require(fingerprints["final_roi/export.json"]["sha256"] == reference.get("sha256"),
             "Final ROI export manifest hash differs from Fiji completion")
    export = _json(export_path)
    records = _validate_metadata(export, source, done)
    files = export.get("files")
    _require(isinstance(files, dict) and set(files) == {*_COMPARTMENTS, "native_rois"},
             "Final ROI export file set is incomplete")
    members = {"fiji_done.json": done_path, "source_manifest.json": source_path,
               "final_roi/export.json": export_path}
    for key in (*_COMPARTMENTS, "native_rois"):
        entry = files[key]
        filename = "native_rois.zip" if key == "native_rois" else f"{key}_labels.tif"
        _require(isinstance(entry, dict) and entry.get("file") == filename,
                 f"Final ROI export filename is invalid: {key}")
        path = _file(run_dir, f"final_roi/{filename}")
        actual = _fingerprint(path)
        _require(actual == {"size": entry.get("size"), "sha256": entry.get("sha256")},
                 f"Final ROI export file identity differs: {key}")
        members[f"final_roi/{filename}"] = path
        fingerprints[f"final_roi/{filename}"] = actual

    # Adapter-owned provenance schema 1 records the actual app, engine, imported
    # input and installed components. The archive never follows referenced paths.
    provenance_path = _file(run_dir, "runtime_provenance.json")
    fingerprints["runtime_provenance.json"] = _fingerprint(provenance_path)
    provenance = _json(provenance_path)
    _require(provenance.get("schema_version") == 1
             and all(isinstance(provenance.get(key), dict)
                     for key in ("request_provenance", "engine", "runtime", "models"))
             and isinstance(provenance["runtime"].get("descriptor"), dict),
             "Unsupported or incomplete runtime provenance schema")
    members["runtime_provenance.json"] = provenance_path

    # Check serialization dimensions and IDs; no biological or morphology gate.
    import numpy as np
    import tifffile
    for key in _COMPARTMENTS:
        try:
            labels = tifffile.imread(members[f"final_roi/{key}_labels.tif"])
        except (OSError, ValueError) as exc:
            raise FinalRoiArchiveError(f"Cannot read final {key} label mask") from exc
        _require(labels.shape == tuple(export["source_shape_zyx"][1:])
                 and labels.dtype == np.dtype("uint16"), f"Final {key} label mask dimensions or dtype differ")
        _require(np.array_equal(np.unique(labels[labels != 0]), np.arange(1, len(records) + 1)),
                 f"Final {key} label mask IDs differ from measurement")
        del labels
    expected_roi_names = {
        f"Astrocyte_{record['label_id']:03d}_{_SUFFIXES[key]}.roi"
        for record in records for key in _COMPARTMENTS
    }
    try:
        with zipfile.ZipFile(members["final_roi/native_rois.zip"]) as native:
            _require(len(native.namelist()) == len(expected_roi_names)
                     and set(native.namelist()) == expected_roi_names,
                     "Native ROI ZIP identities differ from measurement")
            for name in native.namelist():
                data = native.read(name)
                _require(len(data) >= 64 and data[:4] == b"Iout",
                         f"Native ROI record is invalid: {name}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise FinalRoiArchiveError("Cannot read final native ROI ZIP") from exc

    archive_manifest = {"schema_version": 1, "roi_count": len(records),
                        "files": fingerprints, "export_manifest": "final_roi/export.json",
                        "provenance": "runtime_provenance.json"}
    destination = Path(destination)
    _require(destination.parent.is_dir() and not destination.is_symlink(),
             "Invalid ROI archive destination")
    _require(destination.resolve() not in {path.resolve() for path in members.values()},
             "ROI archive destination would replace its source")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, name)
            archive.writestr("archive_manifest.json", json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            for name, expected in fingerprints.items():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                _require(archive.getinfo(name).file_size == expected["size"]
                         and digest.hexdigest() == expected["sha256"],
                         f"Final ROI file changed while archiving: {name}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
