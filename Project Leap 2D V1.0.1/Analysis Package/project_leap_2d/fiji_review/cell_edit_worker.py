"""Isolated Split/Enlarge worker used by the Fiji Cell Edit bridge.

The public dispatcher starts one fresh Python process for each request.  The
worker reads immutable request masks and a small analysis-context NPZ, performs
one local edit, validates a synchronized Whole/Soma/Processes transaction, and
publishes result files only after every check succeeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from project_leap_2d.fiji_review.cell_edit_context import (
    LoadedCellEditContext,
    load_cell_edit_context,
)

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _worker_context(loaded: LoadedCellEditContext) -> dict[str, Any]:
    """Adapt the single committed context schema to the local edit algorithms."""

    calibration = dict(loaded.metadata["calibration"])
    selected_z = dict(loaded.metadata["selected_z"])
    return {
        "schema_version": int(loaded.metadata["schema_version"]),
        "analysis_mode": str(loaded.metadata["analysis_mode"]),
        "structural_channel": str(loaded.metadata["structural_channel"]),
        "structural_image": loaded.array("structural_map"),
        "canonical_nucleus_core_labels_2d": loaded.array(
            "canonical_nucleus_core_labels"
        ),
        "canonical_nucleus_extent_labels_2d": loaded.array(
            "canonical_nucleus_extent_labels"
        ),
        "nucleus_records": [
            dict(record) for record in loaded.metadata["nucleus_records"]
        ],
        "pixel_width_um": float(calibration["pixel_width_um"]),
        "pixel_height_um": float(calibration["pixel_height_um"]),
        "pixel_depth_um": calibration.get("pixel_depth_um"),
        "z_start_1based": int(selected_z["z_start_1based"]),
        "z_end_1based_inclusive": int(
            selected_z["z_end_1based_inclusive"]
        ),
    }


def _response_base(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": request.get("request_id", ""),
        "action": request.get("action", ""),
        "state_revision": request.get("state_revision", -1),
        "state_token": request.get("state_token", ""),
    }


def _failure(
    request: dict[str, Any], status: str, reason: str
) -> dict[str, Any]:
    return {
        **_response_base(request),
        "status": status,
        "reason": str(reason),
    }


def _validate_runtime_layout(
    request: dict[str, Any], manifest: dict[str, Any]
) -> tuple[Path, Path, Path, Path, Path]:
    config = manifest.get("cell_edit")
    if not isinstance(config, dict):
        raise ValueError("Cell Edit runtime configuration is missing.")
    request_dir = Path(str(config.get("request_dir", ""))).resolve()
    response_dir = Path(str(config.get("response_dir", ""))).resolve()
    cancel_dir = Path(str(config.get("cancel_dir", ""))).resolve()
    state_dir = Path(str(config.get("state_dir", ""))).resolve()
    roots = {path.parent for path in (request_dir, response_dir, cancel_dir, state_dir)}
    if len(roots) != 1 or next(iter(roots)).name != "cell_edit":
        raise ValueError("Cell Edit runtime directories are not isolated.")
    root = next(iter(roots))
    expected = {
        request_dir: "requests",
        response_dir: "responses",
        cancel_dir: "cancel",
        state_dir: "state",
    }
    if any(path.name != name or not path.is_dir() for path, name in expected.items()):
        raise ValueError("Cell Edit runtime directory layout is invalid.")
    request_id = str(request.get("request_id", ""))
    if _HEX32.fullmatch(request_id) is None:
        raise ValueError("Invalid Cell Edit request ID.")
    if _HEX32.fullmatch(str(request.get("state_token", ""))) is None:
        raise ValueError("Invalid Cell Edit state token.")
    for key, raw_path in dict(request.get("label_mask_paths", {})).items():
        if key not in {"whole", "soma", "processes"}:
            raise ValueError("Cell Edit request has an unknown label mask.")
        path = Path(str(raw_path)).resolve()
        if not _inside(path, state_dir):
            raise ValueError("Cell Edit request mask is outside this run.")
    context_path = Path(str(config.get("context_path", ""))).resolve()
    if not _inside(context_path, root) or not context_path.is_file():
        raise ValueError("Cell Edit context is outside this run or missing.")
    program_root = Path(str(config.get("program_root", ""))).resolve()
    worker_module = (
        program_root
        / "project_leap_2d"
        / "fiji_review"
        / "cell_edit_worker.py"
    )
    if not program_root.is_dir() or not worker_module.is_file():
        raise ValueError("Cell Edit program_root is invalid.")
    return root, state_dir, cancel_dir, context_path, program_root


def _read_request_triplet(
    request: dict[str, Any], shape: tuple[int, int], state_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import tifffile as tf

    arrays: dict[str, np.ndarray] = {}
    for key in ("whole", "soma", "processes"):
        path = Path(str(request["label_mask_paths"][key])).resolve()
        expected_hash = str(request["label_mask_sha256"][key]).lower()
        if not _inside(path, state_dir):
            raise ValueError("Cell Edit request mask is outside this run.")
        if _HEX64.fullmatch(expected_hash) is None or _sha256_file(path) != expected_hash:
            raise ValueError(f"Cell Edit {key} request mask failed integrity.")
        array = np.asarray(tf.imread(path))
        if array.shape != shape or array.ndim != 2:
            raise ValueError("Cell Edit request dimensions changed.")
        if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError("Cell Edit request labels are invalid.")
        arrays[key] = array.astype(np.uint32, copy=False)
    return arrays["whole"], arrays["soma"], arrays["processes"]


def _validate_identity_records(
    records: Any, labels: np.ndarray
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("Cell Edit identity records are missing.")
    expected = list(range(1, int(labels.max()) + 1))
    normalized: list[dict[str, Any]] = []
    observed_labels: list[int] = []
    uids: list[str] = []
    originals: list[int] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("Cell Edit identity record is invalid.")
        record = dict(raw)
        label_id = int(record.get("label_id", 0))
        original_id = int(record.get("original_id", 0))
        cell_uid = str(record.get("cell_uid", "")).strip()
        if label_id < 1 or original_id < 1 or not cell_uid:
            raise ValueError("Cell Edit identity record is incomplete.")
        lineage = record.get("lineage", [original_id])
        if not isinstance(lineage, list) or any(int(value) < 1 for value in lineage):
            raise ValueError("Cell Edit identity lineage is invalid.")
        record.update(
            {
                "label_id": label_id,
                "original_id": original_id,
                "cell_uid": cell_uid,
                "parent_uid": str(record.get("parent_uid", "")).strip(),
                "lineage": [int(value) for value in lineage],
                "owner_nucleus_id": str(
                    record.get("owner_nucleus_id", "")
                ).strip(),
            }
        )
        observed_labels.append(label_id)
        originals.append(original_id)
        uids.append(cell_uid)
        normalized.append(record)
    if observed_labels != expected:
        raise ValueError("Cell Edit identity labels are not contiguous.")
    if len(uids) != len(set(uids)) or len(originals) != len(set(originals)):
        raise ValueError("Cell Edit identity records are not unique.")
    return normalized


def _build_base_state(
    whole: np.ndarray,
    soma: np.ndarray,
    processes: np.ndarray,
    records: list[dict[str, Any]],
    revision: int,
):
    from project_leap_2d.fiji_review.cell_edit_transactions import (
        CellEditState,
        CellIdentity,
    )

    identities = {}
    for record in records:
        uid = record["cell_uid"]
        parent_uid = record["parent_uid"] or None
        owner = record["owner_nucleus_id"] or None
        transaction_lineage = tuple(
            value for value in (parent_uid, uid) if value
        )
        identities[int(record["label_id"])] = CellIdentity(
            cell_uid=uid,
            parent_uid=parent_uid,
            owner_nucleus_uid=owner,
            lineage=transaction_lineage,
        )
    return CellEditState(
        whole_labels=whole,
        soma_labels=soma,
        process_labels=processes,
        identities=identities,
        version=int(revision),
    )


def _nucleus_owner_map(
    extent_labels: np.ndarray,
    whole: np.ndarray,
    soma: np.ndarray,
    identity_records: list[dict[str, Any]],
    nucleus_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return nucleus->display owner and display->unique inferred owner."""

    nucleus_ids = [
        int(value) for value in np.unique(extent_labels) if int(value) > 0
    ]
    nucleus_to_cell: dict[int, int] = {}
    cell_to_nucleus: dict[int, int] = {}
    record_by_label = {
        int(record["label_id"]): record for record in identity_records
    }
    for cell_id, record in record_by_label.items():
        text = str(record.get("owner_nucleus_id", "")).strip()
        try:
            declared = int(text)
        except ValueError:
            declared = 0
        if declared in nucleus_ids:
            nucleus_to_cell[declared] = cell_id
            cell_to_nucleus[cell_id] = declared
    for record in nucleus_records or ():
        nucleus_id = int(record.get("instance_id", 0))
        owner_display_id = record.get("owner_display_id")
        if owner_display_id is None:
            continue
        owner = int(owner_display_id)
        if (
            nucleus_id in nucleus_ids
            and 1 <= owner <= int(whole.max())
            and nucleus_id not in nucleus_to_cell
            and owner not in cell_to_nucleus
        ):
            nucleus_to_cell[nucleus_id] = owner
            cell_to_nucleus[owner] = nucleus_id
    for nucleus_id in nucleus_ids:
        if nucleus_id in nucleus_to_cell:
            continue
        nucleus = extent_labels == nucleus_id
        soma_counts = np.bincount(
            soma[nucleus].astype(np.int64), minlength=int(whole.max()) + 1
        )
        whole_counts = np.bincount(
            whole[nucleus].astype(np.int64), minlength=int(whole.max()) + 1
        )
        soma_counts[0] = 0
        whole_counts[0] = 0
        if soma_counts.max(initial=0) > 0:
            owner = int(np.argmax(soma_counts))
        elif whole_counts.max(initial=0) > 0:
            owner = int(np.argmax(whole_counts))
        else:
            owner = 0
        if owner > 0:
            nucleus_to_cell[nucleus_id] = owner
    for cell_id in range(1, int(whole.max()) + 1):
        if cell_id in cell_to_nucleus:
            continue
        candidates = [
            nucleus_id
            for nucleus_id, owner in nucleus_to_cell.items()
            if owner == cell_id
        ]
        if not candidates:
            continue
        cell_soma = soma == cell_id
        cell_whole = whole == cell_id
        cell_to_nucleus[cell_id] = max(
            candidates,
            key=lambda nucleus_id: (
                int(((extent_labels == nucleus_id) & cell_soma).sum()),
                int(((extent_labels == nucleus_id) & cell_whole).sum()),
                -nucleus_id,
            ),
        )
    return nucleus_to_cell, cell_to_nucleus


def _canonical_split_candidates(
    context: dict[str, Any],
    whole: np.ndarray,
    soma: np.ndarray,
    identity_records: list[dict[str, Any]],
):
    from project_leap_2d.compartments.selected_cell_split import (
        SplitNucleusCandidate,
    )

    extent = context["canonical_nucleus_extent_labels_2d"]
    records = {
        int(record["instance_id"]): record
        for record in context["nucleus_records"]
    }
    nucleus_to_cell, _ = _nucleus_owner_map(
        extent,
        whole,
        soma,
        identity_records,
        context["nucleus_records"],
    )
    candidates = []
    for nucleus_id in sorted(records):
        mask = extent == nucleus_id
        if not mask.any():
            continue
        record = records[nucleus_id]
        identity_status = str(record.get("identity_status", "resolved"))
        accepted = bool(record.get("accepted", False))
        confidence = 0.95 if accepted else (0.72 if identity_status == "resolved" else 0.55)
        candidates.append(
            SplitNucleusCandidate(
                nucleus_id=nucleus_id,
                projection_mask=mask,
                dapi_valid=bool(record.get("dapi_valid", True)),
                identity_status=identity_status,
                owner_astrocyte_id=nucleus_to_cell.get(nucleus_id),
                accepted=accepted,
                confidence=confidence,
                z_min_0based=(
                    None
                    if record.get("z_min_0based") is None
                    else int(record["z_min_0based"])
                ),
                z_max_0based=(
                    None
                    if record.get("z_max_0based_inclusive") is None
                    else int(record["z_max_0based_inclusive"])
                ),
                source="dapi_3d_inventory",
            )
        )
    return candidates


def _local_crop_bounds(
    mask: np.ndarray,
    pixel_height_um: float,
    pixel_width_um: float,
    margin_um: float = 3.5,
) -> tuple[int, int, int, int]:
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("The selected Astrocyte is empty.")
    pad_y = max(2, int(math.ceil(margin_um / pixel_height_um)))
    pad_x = max(2, int(math.ceil(margin_um / pixel_width_um)))
    return (
        max(0, int(points[:, 0].min()) - pad_y),
        min(mask.shape[0], int(points[:, 0].max()) + pad_y + 1),
        max(0, int(points[:, 1].min()) - pad_x),
        min(mask.shape[1], int(points[:, 1].max()) + pad_x + 1),
    )


def _read_dapi_crop(
    manifest: dict[str, Any],
    context: dict[str, Any],
    crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    import tifffile as tf

    channels = manifest.get("channels")
    if not isinstance(channels, dict) or "DAPI" not in channels:
        raise ValueError("DAPI source is unavailable for local Split recovery.")
    dapi_path = Path(str(channels["DAPI"])).resolve()
    if not dapi_path.is_file():
        raise ValueError("DAPI source is unavailable for local Split recovery.")
    z_indices = tuple(
        range(
            int(context["z_start_1based"]) - 1,
            int(context["z_end_1based_inclusive"]),
        )
    )
    y0, y1, x0, x1 = crop
    planes = []
    with tf.TiffFile(dapi_path) as tif:
        pages = tif.pages
        if not z_indices or z_indices[-1] >= len(pages):
            raise ValueError("Selected DAPI Z range is outside the source stack.")
        for z_index in z_indices:
            plane = np.asarray(pages[z_index].asarray())
            if plane.ndim != 2 or plane.shape != context["structural_image"].shape:
                raise ValueError("DAPI source dimensions changed.")
            planes.append(np.asarray(plane[y0:y1, x0:x1]))
    return np.stack(planes, axis=0), z_indices


def _link_local_model_candidates(
    labels_zyx: np.ndarray,
    *,
    z_indices: tuple[int, ...],
    crop: tuple[int, int, int, int],
    full_shape: tuple[int, int],
    first_id: int,
    canonical_extent: np.ndarray,
):
    """Conservatively link adjacent slice instances into 3D proposals."""

    from project_leap_2d.compartments.selected_cell_split import (
        SplitNucleusCandidate,
    )

    labels = np.asarray(labels_zyx)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Local nucleus-model labels must be integer ZYX data.")
    # Preserve the model's independent instances within each plane.  Adjacent
    # planes are linked only by one-to-one, mutually strongest real overlap;
    # flattening the volume to binary would incorrectly fuse touching nuclei.
    nodes = [
        (z, int(label))
        for z in range(labels.shape[0])
        for label in np.unique(labels[z])
        if int(label) > 0
    ]
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for z in range(max(0, labels.shape[0] - 1)):
        left_ids = [int(value) for value in np.unique(labels[z]) if int(value) > 0]
        right_ids = [
            int(value) for value in np.unique(labels[z + 1]) if int(value) > 0
        ]
        edges = []
        for left_id in left_ids:
            left_mask = labels[z] == left_id
            left_area = int(left_mask.sum())
            overlaps = np.bincount(
                labels[z + 1][left_mask].astype(np.int64),
                minlength=max(right_ids, default=0) + 1,
            )
            for right_id in right_ids:
                overlap = int(overlaps[right_id])
                if overlap <= 0:
                    continue
                right_area = int((labels[z + 1] == right_id).sum())
                overlap_fraction = overlap / max(min(left_area, right_area), 1)
                if overlap_fraction >= 0.08:
                    edges.append(
                        (overlap_fraction, overlap, left_id, right_id)
                    )
        used_left: set[int] = set()
        used_right: set[int] = set()
        for _, _, left_id, right_id in sorted(edges, reverse=True):
            if left_id in used_left or right_id in used_right:
                continue
            union((z, left_id), (z + 1, right_id))
            used_left.add(left_id)
            used_right.add(right_id)

    tracks: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in nodes:
        tracks.setdefault(find(node), []).append(node)
    candidates = []
    refined_candidates: dict[int, tuple[float, Any]] = {}
    y0, y1, x0, x1 = crop
    next_id = int(first_id)
    for track_nodes in tracks.values():
        z_hits = sorted({int(node[0]) for node in track_nodes})
        local_projection = np.logical_or.reduce(
            [labels[z] == label for z, label in track_nodes]
        )
        if int(local_projection.sum()) < 4:
            continue
        full = np.zeros(full_shape, dtype=bool)
        full[y0:y1, x0:x1] = local_projection
        canonical_ids = [
            int(value)
            for value in np.unique(canonical_extent[full])
            if int(value) > 0
        ]
        best_match: tuple[float, int] | None = None
        for nucleus_id in canonical_ids:
            existing = canonical_extent == nucleus_id
            intersection = int((existing & full).sum())
            match_fraction = intersection / max(
                min(int(existing.sum()), int(full.sum())),
                1,
            )
            if match_fraction >= 0.55 and (
                best_match is None or match_fraction > best_match[0]
            ):
                best_match = (match_fraction, nucleus_id)
        if best_match is not None:
            match_fraction, nucleus_id = best_match
            refined = SplitNucleusCandidate(
                nucleus_id=nucleus_id,
                projection_mask=full,
                dapi_valid=True,
                identity_status="model_proposal",
                owner_astrocyte_id=None,
                accepted=False,
                confidence=min(
                    0.90,
                    0.55 + 0.06 * len(z_hits) + 0.15 * match_fraction,
                ),
                z_min_0based=int(z_indices[z_hits[0]]),
                z_max_0based=int(z_indices[z_hits[-1]]),
                source="instanseg_local_z_refinement",
                locally_confirmed=True,
            )
            prior = refined_candidates.get(nucleus_id)
            if prior is None or match_fraction > prior[0]:
                refined_candidates[nucleus_id] = (match_fraction, refined)
            continue
        candidates.append(
            SplitNucleusCandidate(
                nucleus_id=next_id,
                projection_mask=full,
                dapi_valid=True,
                identity_status="model_proposal",
                owner_astrocyte_id=None,
                accepted=False,
                confidence=min(0.85, 0.52 + 0.06 * len(z_hits)),
                z_min_0based=int(z_indices[z_hits[0]]),
                z_max_0based=int(z_indices[z_hits[-1]]),
                source="instanseg_local_3d_link",
                locally_confirmed=True,
            )
        )
        next_id += 1
    return [
        *(candidate for _, candidate in refined_candidates.values()),
        *candidates,
    ]


def _merge_local_split_candidates(canonical_candidates, model_candidates):
    """Add new model nuclei and locally confirm/refine matched provisional seeds."""

    merged = {int(candidate.nucleus_id): candidate for candidate in canonical_candidates}
    for model in model_candidates:
        nucleus_id = int(model.nucleus_id)
        canonical = merged.get(nucleus_id)
        if canonical is None:
            merged[nucleus_id] = model
            continue
        # Preserve canonical identity/ownership while using the independent
        # local DAPI model's projection and Z extent as the confirmation.  This
        # prevents an unaccepted or tiny inventory fragment from being treated
        # as a complete second nucleus.
        merged[nucleus_id] = replace(
            canonical,
            projection_mask=np.asarray(model.projection_mask, dtype=bool),
            z_min_0based=(
                canonical.z_min_0based
                if model.z_min_0based is None
                else int(model.z_min_0based)
            ),
            z_max_0based=(
                canonical.z_max_0based
                if model.z_max_0based is None
                else int(model.z_max_0based)
            ),
            source=f"{canonical.source}+instanseg_local_confirmation",
            confidence=max(float(canonical.confidence), float(model.confidence)),
            locally_confirmed=True,
        )
    return [merged[nucleus_id] for nucleus_id in sorted(merged)]


def _split_local_fallback_allowed(reason: str, candidates) -> bool:
    """Allow only evidence-recovery refusals, never ownership or known-Z conflicts."""

    if reason == "No additional DAPI nucleus was found.":
        return True
    if reason != "The two nuclear candidates cannot be separated.":
        return False
    return any(
        candidate.z_min_0based is None
        or candidate.z_max_0based is None
        for candidate in candidates
    )


def _instanseg_split_candidates(
    manifest: dict[str, Any],
    context: dict[str, Any],
    selected_mask: np.ndarray,
    first_id: int,
):
    """Lazy local InstanSeg challenger; called only after inventory refusal."""

    from project_leap_2d.nuclei.instanseg_nucleus_detection import (
        detect_instanseg_nuclei,
    )

    crop = _local_crop_bounds(
        selected_mask,
        context["pixel_height_um"],
        context["pixel_width_um"],
    )
    if (crop[1] - crop[0]) * (crop[3] - crop[2]) > 4_000_000:
        raise ValueError("The selected ROI is too large for local nucleus recovery.")
    local_dapi, z_indices = _read_dapi_crop(manifest, context, crop)
    result = detect_instanseg_nuclei(
        local_dapi,
        context["pixel_height_um"],
        context["pixel_width_um"],
    )
    return _link_local_model_candidates(
        result.labels_zyx,
        z_indices=z_indices,
        crop=crop,
        full_shape=selected_mask.shape,
        first_id=first_id,
        canonical_extent=context["canonical_nucleus_extent_labels_2d"],
    )


def _write_result_triplet(
    state_dir: Path,
    request_id: str,
    whole: np.ndarray,
    soma: np.ndarray,
    processes: np.ndarray,
) -> tuple[dict[str, str], dict[str, str]]:
    import tifffile as tf

    result_dir = (state_dir / f"{request_id}_result").resolve()
    if not _inside(result_dir, state_dir):
        raise ValueError("Cell Edit result path is outside this run.")
    result_dir.mkdir(parents=False, exist_ok=False)
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    try:
        for key, array in (
            ("whole", whole),
            ("soma", soma),
            ("processes", processes),
        ):
            if int(np.asarray(array).max()) > np.iinfo(np.uint16).max:
                raise ValueError("Cell Edit result exceeds Fiji label capacity.")
            target = result_dir / f"{key}_labels.tif"
            temporary = (
                result_dir
                / f"temporary_{key}.{uuid.uuid4().hex}.tmp.tif"
            )
            try:
                tf.imwrite(
                    temporary,
                    np.asarray(array, dtype=np.uint16),
                    photometric="minisblack",
                    metadata={"axes": "YX"},
                )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            paths[key] = str(target)
            hashes[key] = _sha256_file(target)
    except BaseException:
        for child in result_dir.iterdir():
            child.unlink(missing_ok=True)
        result_dir.rmdir()
        raise
    return paths, hashes


def _execute_edit(
    request: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    root, state_dir, _, context_path, _ = _validate_runtime_layout(
        request, manifest
    )
    del root
    context = _worker_context(
        load_cell_edit_context(context_path, verify_sources=True)
    )
    shape = tuple(int(value) for value in context["structural_image"].shape)
    expected_shape = tuple(int(value) for value in manifest.get("expected_shape", ()))
    if len(expected_shape) < 2 or tuple(expected_shape[-2:]) != shape:
        raise ValueError("Cell Edit context does not match this Fiji run.")
    whole, soma, processes = _read_request_triplet(request, shape, state_dir)
    identity_records = _validate_identity_records(
        request.get("identity_records"), whole
    )
    from project_leap_2d.fiji_review.review_validation import (
        validate_cell_edit_delta,
        validate_cell_edit_label_triplet,
    )

    validate_cell_edit_label_triplet(
        whole_labels=whole,
        soma_labels=soma,
        process_labels=processes,
        identity_records=identity_records,
    )
    selected_uid = str(request["selected_cell_uid"])
    selected_matches = [
        record for record in identity_records if record["cell_uid"] == selected_uid
    ]
    if len(selected_matches) != 1:
        raise ValueError("The selected Cell UID is stale.")
    selected_record = selected_matches[0]
    if int(selected_record["original_id"]) != int(request["selected_original_id"]):
        raise ValueError("The selected Astrocyte identity is stale.")
    selected_id = int(selected_record["label_id"])
    base = _build_base_state(
        whole,
        soma,
        processes,
        identity_records,
        int(request["state_revision"]),
    )
    action = str(request["action"]).lower()
    if action == "enlarge":
        from project_leap_2d.compartments.selected_soma_enlargement import (
            enlarge_selected_soma,
            selected_soma_enlargement_config_for_mode,
        )
        from project_leap_2d.fiji_review.cell_edit_transactions import (
            propose_enlarge,
        )

        extent = context["canonical_nucleus_extent_labels_2d"]
        _, cell_to_nucleus = _nucleus_owner_map(
            extent,
            whole,
            soma,
            identity_records,
            context["nucleus_records"],
        )
        owner_id = cell_to_nucleus.get(selected_id)
        if owner_id is None:
            return _failure(
                request,
                "rejected",
                "No unique owner nucleus was found for the selected Soma.",
            )
        owner = extent == int(owner_id)
        foreign = (extent > 0) & ~owner
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            selected_id,
            owner,
            context["structural_image"],
            foreign,
            context["pixel_width_um"],
            context["pixel_height_um"],
            config=selected_soma_enlargement_config_for_mode(
                context["analysis_mode"]
            ),
        )
        if not result.approved:
            return _failure(request, "rejected", result.message)
        proposal = propose_enlarge(
            base,
            source_cell_uid=selected_uid,
            whole_labels=result.whole_labels,
            soma_labels=result.soma_labels,
            process_labels=result.process_labels,
            identities=base.identities,
            proposal_id=request["request_id"],
            audit=result.metrics,
        )
    elif action == "split":
        from project_leap_2d.compartments.selected_cell_split import (
            split_selected_cell,
        )
        from project_leap_2d.fiji_review.cell_edit_transactions import (
            make_split_child_identity,
            propose_split,
        )

        candidates = _canonical_split_candidates(
            context, whole, soma, identity_records
        )
        result = split_selected_cell(
            whole,
            soma,
            processes,
            selected_id,
            candidates,
            context["structural_image"],
            context["pixel_width_um"],
            context["pixel_height_um"],
            pixel_depth_um=context["pixel_depth_um"],
        )
        if not result.success and _split_local_fallback_allowed(
            result.reason,
            candidates,
        ):
            model_candidates = _instanseg_split_candidates(
                manifest,
                context,
                whole == selected_id,
                max((candidate.nucleus_id for candidate in candidates), default=0)
                + 1,
            )
            if model_candidates:
                combined_candidates = _merge_local_split_candidates(
                    candidates,
                    model_candidates,
                )
                result = split_selected_cell(
                    whole,
                    soma,
                    processes,
                    selected_id,
                    combined_candidates,
                    context["structural_image"],
                    context["pixel_width_um"],
                    context["pixel_height_um"],
                    pixel_depth_um=context["pixel_depth_um"],
                )
        if not result.success:
            return _failure(request, "rejected", result.reason)
        parent_identity = base.identities[selected_id]
        child_one = make_split_child_identity(
            parent_identity,
            owner_nucleus_uid=str(result.owner_nucleus_id),
            child_index=1,
            edit_nonce=request["request_id"],
        )
        child_two = make_split_child_identity(
            parent_identity,
            owner_nucleus_uid=str(result.second_nucleus_id),
            child_index=2,
            edit_nonce=request["request_id"],
        )
        result_identities = {
            display_id: identity
            for display_id, identity in base.identities.items()
            if display_id != selected_id
        }
        result_identities[selected_id] = child_one
        if result.new_id is None:
            raise ValueError("Split did not create a second display ID.")
        result_identities[int(result.new_id)] = child_two
        proposal = propose_split(
            base,
            source_cell_uid=selected_uid,
            whole_labels=result.whole_labels,
            soma_labels=result.soma_labels,
            process_labels=result.process_labels,
            identities=result_identities,
            proposal_id=request["request_id"],
            audit=result.metrics,
        )
    else:
        raise ValueError("Unsupported Cell Edit action.")

    from project_leap_2d.fiji_review.cell_edit_transactions import (
        commit_cell_edit,
    )
    from project_leap_2d.fiji_review.cell_editing import (
        build_fiji_identity_records,
    )

    committed = commit_cell_edit(base, proposal)
    result_records = build_fiji_identity_records(
        result_identities=dict(committed.identities),
        request_identity_records=identity_records,
    )
    validate_cell_edit_label_triplet(
        whole_labels=committed.whole_labels,
        soma_labels=committed.soma_labels,
        process_labels=committed.process_labels,
        identity_records=result_records,
    )
    validate_cell_edit_delta(
        action=action,
        before_whole=whole,
        before_soma=soma,
        before_processes=processes,
        after_whole=committed.whole_labels,
        after_soma=committed.soma_labels,
        after_processes=committed.process_labels,
        selected_id=selected_id,
    )
    paths, hashes = _write_result_triplet(
        state_dir,
        str(request["request_id"]),
        committed.whole_labels,
        committed.soma_labels,
        committed.process_labels,
    )
    return {
        **_response_base(request),
        "status": "success",
        "roi_count": committed.cell_count,
        "label_mask_paths": paths,
        "label_mask_sha256": hashes,
        "identity_records": result_records,
        "worker_state_hash": committed.state_hash,
        "worker_state_revision": committed.version,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(
        f"temporary_{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _subprocess_entry(task_path_text: str, response_path_text: str) -> None:
    """Private child-process entry point; never raises across the process."""

    task_path = Path(task_path_text).resolve()
    response_path = Path(response_path_text).resolve()
    request: dict[str, Any] = {}
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        request = dict(task["request"])
        manifest = dict(task["manifest"])
        _, state_dir, _, _, _ = _validate_runtime_layout(request, manifest)
        if not _inside(task_path, state_dir) or not _inside(response_path, state_dir):
            raise ValueError("Cell Edit worker files are outside this run.")
        response = _execute_edit(request, manifest)
    except BaseException as exc:
        response = _failure(
            request,
            "error",
            f"Cell Edit failed safely: {exc}",
        )
    try:
        _atomic_json(response_path, response)
    except BaseException:
        pass


def _terminate_worker(process: subprocess.Popen, grace_seconds: float = 0.5) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def dispatch_cell_edit_request(
    request: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Execute one request in a killable child process and return its response."""

    _, state_dir, cancel_dir, _, program_root_path = _validate_runtime_layout(
        request, manifest
    )
    request_id = str(request["request_id"])
    task_path = state_dir / f"temporary_{request_id}.worker-task.json"
    response_path = state_dir / f"temporary_{request_id}.worker-response.json"
    if task_path.exists() or response_path.exists():
        return _failure(
            request,
            "error",
            "Cell Edit request files already exist; ROI state was not changed.",
        )
    _atomic_json(
        task_path,
        {
            "request": request,
            "manifest": manifest,
        },
    )
    timeout = float(manifest["cell_edit"].get("timeout_seconds", 45.0))
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 45.0
    timeout = min(timeout, 45.0)
    project_root = str(program_root_path)
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        project_root
        if not prior_pythonpath
        else project_root + os.pathsep + prior_pythonpath
    )
    command = [
        sys.executable,
        "-c",
        (
            "from project_leap_2d.fiji_review.cell_edit_worker "
            "import _subprocess_entry; "
            "_subprocess_entry(__import__('sys').argv[1], "
            "__import__('sys').argv[2])"
        ),
        str(task_path),
        str(response_path),
    ]
    process: subprocess.Popen | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        while process.poll() is None:
            if (cancel_dir / f"{request_id}.cancel").exists():
                _terminate_worker(process)
                return _failure(
                    request,
                    "cancelled",
                    "Cell Edit was cancelled; ROI state was not changed.",
                )
            if time.monotonic() - started >= timeout:
                _terminate_worker(process)
                return _failure(
                    request,
                    "error",
                    "Cell Edit timed out; ROI state was not changed.",
                )
            time.sleep(0.05)
        if process.returncode != 0 or not response_path.is_file():
            return _failure(
                request,
                "error",
                "Cell Edit worker stopped unexpectedly; ROI state was not changed.",
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise ValueError("Cell Edit worker returned invalid data.")
        return response
    except BaseException as exc:
        if process is not None:
            _terminate_worker(process)
        return _failure(
            request,
            "error",
            f"Cell Edit failed safely: {exc}",
        )
    finally:
        task_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
