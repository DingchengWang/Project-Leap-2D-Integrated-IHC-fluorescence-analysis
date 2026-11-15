"""Three-dimensional DAPI identity and GFAP-associated owner selection.

InstanSeg assigns instance numbers independently on every Z plane.  This
module first links those slice-local instances into globally unique 3D
objects, then decides which valid nuclei have enough local GFAP evidence to
seed an astrocyte ROI.  GFAP enclosure is deliberately not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class GFAPNucleusOwnershipConfig:
    """Physical and evidence gates for the GFAP-only nucleus route."""

    link_min_overlap_fraction: float = 0.18
    link_max_centroid_distance_um: float = 0.85
    link_dilation_um: float = 0.22
    link_ambiguity_score_ratio: float = 0.72
    min_nucleus_volume_um3: float = 3.0
    max_nucleus_volume_um3: float = 1500.0
    min_nucleus_z_span_um: float = 0.48
    reject_xy_border_objects: bool = True
    shell_outer_um: float = 1.50
    background_inner_um: float = 2.0
    background_outer_um: float = 4.0
    shell_support_sigma: float = 1.5
    min_shell_enrichment: float = 4.5
    min_supported_z_count: int = 2
    min_supported_z_fraction: float = 0.25
    min_shell_support_um3: float = 0.30
    structure_contact_um: float = 1.50
    local_evidence_radius_um: float = 8.0
    strong_reference_inner_um: float = 2.0
    strong_reference_outer_um: float = 4.0
    strong_reference_percentile: float = 95.0
    minimum_strong_structural_score: float = 0.30
    min_contact_angular_fraction: float = 0.20
    angular_sector_count: int = 12
    minimum_strong_contact_sector_count: int = 3
    minimum_contiguous_strong_contact_sectors: int = 2
    minimum_strong_contact_sector_area_um2: float = 0.08
    min_anchored_structure_area_um2: float = 0.75
    min_anchored_structure_reach_um: float = 1.6
    min_projection_exclusive_fraction: float = 0.20
    allow_unique_weak_enrichment_fallback: bool = True
    fallback_min_shell_enrichment_fraction: float = 0.75


@dataclass(frozen=True)
class LinkedNucleusInventory:
    """Globally labelled 3D nuclei derived from slice-local instances."""

    labels_zyx: np.ndarray
    valid_ids: tuple[int, ...]
    ambiguous_ids: tuple[int, ...]
    invalid_ids: tuple[int, ...]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GFAPAssociationResult:
    """Per-nucleus GFAP evidence and accepted astrocyte-owner identities."""

    accepted_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExclusiveNucleusProjection:
    """A collision-audited exclusive 2D projection of accepted 3D nuclei."""

    labels_yx: np.ndarray
    retained_ids: tuple[int, ...]
    collision_rejected_ids: tuple[int, ...]
    collision_pixels: int
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GFAPNucleusOwnershipResult:
    """Complete nucleus-identity and GFAP-association resolution."""

    inventory: LinkedNucleusInventory
    association: GFAPAssociationResult
    projection: ExclusiveNucleusProjection


def _apply_unique_weak_enrichment_fallback(
    association: GFAPAssociationResult,
    config: GFAPNucleusOwnershipConfig,
) -> GFAPAssociationResult:
    """Retain one otherwise-complete owner with only mildly weak enrichment.

    This fallback is intentionally unavailable when more than one candidate
    qualifies.  The caller has already restricted association testing to
    physically valid, unambiguous 3D nucleus identities; all local structural,
    angular, Z-continuity, and support-volume gates must also have passed.
    """

    if association.accepted_ids or not config.allow_unique_weak_enrichment_fallback:
        return association
    minimum_enrichment = (
        float(config.min_shell_enrichment)
        * float(config.fallback_min_shell_enrichment_fraction)
    )
    eligible: list[int] = []
    for record in association.records:
        reasons = tuple(str(value) for value in record.get("rejection_reasons", ()))
        enrichment = float(record.get("shell_enrichment", float("-inf")))
        if (
            reasons == ("weak_perinuclear_gfap_enrichment",)
            and np.isfinite(enrichment)
            and enrichment >= minimum_enrichment
        ):
            eligible.append(int(record["nucleus_id"]))
    if len(eligible) != 1:
        return association

    fallback_id = eligible[0]
    updated_records: list[dict[str, Any]] = []
    for record in association.records:
        updated = dict(record)
        used_fallback = int(record["nucleus_id"]) == fallback_id
        updated["accepted_by_limited_fallback"] = used_fallback
        if used_fallback:
            updated["gfap_associated"] = True
            updated["rejection_reasons"] = []
        updated_records.append(updated)
    return GFAPAssociationResult(
        accepted_ids=(fallback_id,),
        rejected_ids=tuple(
            nucleus_id
            for nucleus_id in association.rejected_ids
            if nucleus_id != fallback_id
        ),
        records=tuple(updated_records),
    )


@dataclass(frozen=True)
class _SliceObject:
    label_id: int
    bounds: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    centroid_um: tuple[float, float]


def _validate_spacing(
    pixel_size_um: float | tuple[float, float],
    z_spacing_um: float,
) -> tuple[float, float, float]:
    if np.isscalar(pixel_size_um):
        y_um = x_um = float(pixel_size_um)
    else:
        if len(pixel_size_um) != 2:
            raise ValueError("pixel_size_um must be a scalar or (Y, X)")
        y_um, x_um = (float(pixel_size_um[0]), float(pixel_size_um[1]))
    z_um = float(z_spacing_um)
    if not all(np.isfinite(value) and value > 0 for value in (z_um, y_um, x_um)):
        raise ValueError("Physical voxel spacing must be finite and positive")
    return z_um, y_um, x_um


def _validate_slice_labels(labels_zyx: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_zyx)
    if labels.ndim != 3:
        raise ValueError("Slice-local nucleus labels must have shape (Z, Y, X)")
    if np.issubdtype(labels.dtype, np.floating):
        if not np.all(labels == np.floor(labels)):
            raise ValueError("Nucleus labels must be integer-valued")
    labels = labels.astype(np.int32, copy=False)
    if np.any(labels < 0):
        raise ValueError("Nucleus labels cannot be negative")
    return labels


def _slice_objects(
    plane: np.ndarray,
    y_um: float,
    x_um: float,
) -> dict[int, _SliceObject]:
    objects: dict[int, _SliceObject] = {}
    for label_id in (int(value) for value in np.unique(plane) if value > 0):
        coordinates = np.argwhere(plane == label_id)
        y0 = int(coordinates[:, 0].min())
        y1 = int(coordinates[:, 0].max()) + 1
        x0 = int(coordinates[:, 1].min())
        x1 = int(coordinates[:, 1].max()) + 1
        local_mask = plane[y0:y1, x0:x1] == label_id
        objects[label_id] = _SliceObject(
            label_id=label_id,
            bounds=(y0, y1, x0, x1),
            mask=local_mask,
            area=int(local_mask.sum()),
            centroid_um=(
                float(coordinates[:, 0].mean()) * y_um,
                float(coordinates[:, 1].mean()) * x_um,
            ),
        )
    return objects


def _link_candidate(
    previous: _SliceObject,
    current: _SliceObject,
    *,
    y_um: float,
    x_um: float,
    config: GFAPNucleusOwnershipConfig,
) -> tuple[bool, float]:
    previous_y0, previous_y1, previous_x0, previous_x1 = previous.bounds
    current_y0, current_y1, current_x0, current_x1 = current.bounds
    overlap_y0 = max(previous_y0, current_y0)
    overlap_y1 = min(previous_y1, current_y1)
    overlap_x0 = max(previous_x0, current_x0)
    overlap_x1 = min(previous_x1, current_x1)
    intersection = 0
    if overlap_y0 < overlap_y1 and overlap_x0 < overlap_x1:
        previous_view = previous.mask[
            overlap_y0 - previous_y0 : overlap_y1 - previous_y0,
            overlap_x0 - previous_x0 : overlap_x1 - previous_x0,
        ]
        current_view = current.mask[
            overlap_y0 - current_y0 : overlap_y1 - current_y0,
            overlap_x0 - current_x0 : overlap_x1 - current_x0,
        ]
        intersection = int(np.count_nonzero(previous_view & current_view))
    overlap = float(intersection) / max(
        min(previous.area, current.area),
        1,
    )
    centroid_distance = float(
        np.hypot(
            previous.centroid_um[0] - current.centroid_um[0],
            previous.centroid_um[1] - current.centroid_um[1],
        )
    )
    dilation_pixels = max(
        1,
        int(np.ceil(config.link_dilation_um / min(y_um, x_um))),
    )
    expanded_overlap = False
    if centroid_distance <= config.link_max_centroid_distance_um:
        union_y0 = min(previous_y0, current_y0) - dilation_pixels
        union_y1 = max(previous_y1, current_y1) + dilation_pixels
        union_x0 = min(previous_x0, current_x0) - dilation_pixels
        union_x1 = max(previous_x1, current_x1) + dilation_pixels
        local_shape = (union_y1 - union_y0, union_x1 - union_x0)
        previous_local = np.zeros(local_shape, dtype=bool)
        current_local = np.zeros(local_shape, dtype=bool)
        previous_local[
            previous_y0 - union_y0 : previous_y1 - union_y0,
            previous_x0 - union_x0 : previous_x1 - union_x0,
        ] = previous.mask
        current_local[
            current_y0 - union_y0 : current_y1 - union_y0,
            current_x0 - union_x0 : current_x1 - union_x0,
        ] = current.mask
        expanded_overlap = bool(
            np.any(
                ndi.binary_dilation(
                    previous_local,
                    iterations=dilation_pixels,
                )
                & current_local
            )
        )
    accepted = bool(
        overlap >= config.link_min_overlap_fraction
        or (
            centroid_distance <= config.link_max_centroid_distance_um
            and expanded_overlap
        )
    )
    score = overlap + max(
        0.0,
        1.0 - centroid_distance / max(config.link_max_centroid_distance_um, 1e-6),
    )
    return accepted, float(score)


def link_slice_instances_3d(
    slice_labels_zyx: np.ndarray,
    pixel_size_um: float | tuple[float, float],
    z_spacing_um: float,
    *,
    config: GFAPNucleusOwnershipConfig | None = None,
) -> LinkedNucleusInventory:
    """Link independently numbered adjacent-Z instances into global 3D IDs.

    Only unambiguous one-to-one adjacent-slice links are accepted.  Any
    one-to-many or many-to-one candidate family is retained for diagnostics but
    excluded from the valid owner set.
    """

    active = config or GFAPNucleusOwnershipConfig()
    labels = _validate_slice_labels(slice_labels_zyx)
    z_um, y_um, x_um = _validate_spacing(pixel_size_um, z_spacing_um)
    global_labels = np.zeros(labels.shape, dtype=np.uint32)
    next_global_id = 1
    ambiguous_ids: set[int] = set()
    previous_objects: dict[int, _SliceObject] = {}
    previous_global_ids: dict[int, int] = {}
    track_stats: dict[int, dict[str, int | bool]] = {}

    for z_index in range(labels.shape[0]):
        current_objects = _slice_objects(labels[z_index], y_um, x_um)
        current_global_ids: dict[int, int] = {}
        candidates: dict[tuple[int, int], float] = {}
        for previous_id, previous_object in previous_objects.items():
            for current_id, current_object in current_objects.items():
                accepted, score = _link_candidate(
                    previous_object,
                    current_object,
                    y_um=y_um,
                    x_um=x_um,
                    config=active,
                )
                if accepted:
                    candidates[(previous_id, current_id)] = score

        conflict_edges: set[tuple[int, int]] = set()
        for previous_id in previous_objects:
            row = sorted(
                (
                    (score, current_id)
                    for (candidate_previous, current_id), score in candidates.items()
                    if candidate_previous == previous_id
                ),
                reverse=True,
            )
            if len(row) > 1 and row[1][0] >= active.link_ambiguity_score_ratio * row[0][0]:
                cutoff = active.link_ambiguity_score_ratio * row[0][0]
                conflict_edges.update(
                    (previous_id, current_id)
                    for score, current_id in row
                    if score >= cutoff
                )
        for current_id in current_objects:
            column = sorted(
                (
                    (score, previous_id)
                    for (previous_id, candidate_current), score in candidates.items()
                    if candidate_current == current_id
                ),
                reverse=True,
            )
            if (
                len(column) > 1
                and column[1][0]
                >= active.link_ambiguity_score_ratio * column[0][0]
            ):
                cutoff = active.link_ambiguity_score_ratio * column[0][0]
                conflict_edges.update(
                    (previous_id, current_id)
                    for score, previous_id in column
                    if score >= cutoff
                )
        conflict_previous = {edge[0] for edge in conflict_edges}
        conflict_current = {edge[1] for edge in conflict_edges}
        used_previous: set[int] = set()
        used_current: set[int] = set()
        for (previous_id, current_id), _score in sorted(
            candidates.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        ):
            if (
                previous_id in conflict_previous
                or current_id in conflict_current
                or previous_id in used_previous
                or current_id in used_current
            ):
                continue
            current_global_ids[current_id] = previous_global_ids[previous_id]
            used_previous.add(previous_id)
            used_current.add(current_id)
        for previous_id in conflict_previous:
            ambiguous_ids.add(previous_global_ids[previous_id])
        for current_id in sorted(current_objects):
            if current_id not in current_global_ids:
                current_global_ids[current_id] = next_global_id
                next_global_id += 1
            if current_id in conflict_current:
                ambiguous_ids.add(current_global_ids[current_id])
            current_object = current_objects[current_id]
            y0, y1, x0, x1 = current_object.bounds
            output_view = global_labels[z_index, y0:y1, x0:x1]
            output_view[current_object.mask] = current_global_ids[current_id]
            global_id = current_global_ids[current_id]
            stats = track_stats.setdefault(
                global_id,
                {
                    "voxel_count": 0,
                    "z_first": z_index,
                    "z_last": z_index,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "touches_xy_border": False,
                },
            )
            stats["voxel_count"] = int(stats["voxel_count"]) + current_object.area
            stats["z_first"] = min(int(stats["z_first"]), z_index)
            stats["z_last"] = max(int(stats["z_last"]), z_index)
            stats["y0"] = min(int(stats["y0"]), y0)
            stats["y1"] = max(int(stats["y1"]), y1)
            stats["x0"] = min(int(stats["x0"]), x0)
            stats["x1"] = max(int(stats["x1"]), x1)
            stats["touches_xy_border"] = bool(stats["touches_xy_border"]) or bool(
                y0 == 0
                or y1 == labels.shape[1]
                or x0 == 0
                or x1 == labels.shape[2]
            )

        previous_objects = current_objects
        previous_global_ids = current_global_ids

    voxel_volume_um3 = z_um * y_um * x_um
    records: list[dict[str, Any]] = []
    valid_ids: list[int] = []
    invalid_ids: list[int] = []
    for global_id in sorted(track_stats):
        stats = track_stats[global_id]
        voxel_count = int(stats["voxel_count"])
        z_first = int(stats["z_first"])
        z_last = int(stats["z_last"])
        y0 = int(stats["y0"])
        y1 = int(stats["y1"])
        x0 = int(stats["x0"])
        x1 = int(stats["x1"])
        volume_um3 = float(voxel_count) * voxel_volume_um3
        z_span_um = float(z_last - z_first + 1) * z_um
        touches_xy_border = bool(stats["touches_xy_border"])
        local_mask = (
            global_labels[z_first : z_last + 1, y0:y1, x0:x1] == global_id
        )
        component_count = int(
            ndi.label(
                local_mask,
                structure=ndi.generate_binary_structure(3, 3),
            )[1]
        )
        reasons: list[str] = []
        if global_id in ambiguous_ids:
            reasons.append("ambiguous_adjacent_z_link")
        if volume_um3 < active.min_nucleus_volume_um3:
            reasons.append("volume_below_physical_minimum")
        if volume_um3 > active.max_nucleus_volume_um3:
            reasons.append("volume_above_physical_maximum")
        if z_span_um < active.min_nucleus_z_span_um:
            reasons.append("z_span_below_physical_minimum")
        if active.reject_xy_border_objects and touches_xy_border:
            reasons.append("incomplete_xy_border_object")
        if component_count != 1:
            reasons.append("disconnected_3d_track")
        is_valid = not reasons
        (valid_ids if is_valid else invalid_ids).append(global_id)
        records.append(
            {
                "nucleus_id": global_id,
                "voxel_count": voxel_count,
                "volume_um3": volume_um3,
                "z_span_um": z_span_um,
                "z_first": z_first,
                "z_last": z_last,
                "touches_xy_border": touches_xy_border,
                "component_count": component_count,
                "valid_3d_nucleus": is_valid,
                "rejection_reasons": reasons,
            }
        )
    return LinkedNucleusInventory(
        labels_zyx=global_labels,
        valid_ids=tuple(sorted(valid_ids)),
        ambiguous_ids=tuple(sorted(ambiguous_ids)),
        invalid_ids=tuple(sorted(invalid_ids)),
        records=tuple(records),
    )


def _longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _angular_contact_evidence(
    contact: np.ndarray,
    nucleus_projection: np.ndarray,
    sector_count: int,
    pixel_area_um2: float,
    minimum_sector_area_um2: float,
) -> tuple[int, int, float, tuple[float, ...]]:
    coordinates = np.argwhere(contact)
    nucleus_coordinates = np.argwhere(nucleus_projection)
    if coordinates.size == 0 or nucleus_coordinates.size == 0:
        return 0, 0, 0.0, tuple(0.0 for _ in range(max(int(sector_count), 0)))
    center_y, center_x = nucleus_coordinates.mean(axis=0)
    angles = np.mod(
        np.arctan2(
            coordinates[:, 0] - center_y,
            coordinates[:, 1] - center_x,
        ),
        2.0 * np.pi,
    )
    # Center a sector on the horizontal axis instead of placing a boundary
    # there.  Otherwise a thin horizontal crossing line can numerically occupy
    # four sectors merely because its two-pixel thickness straddles the
    # 0/2π and π boundaries.
    sector_width = 2.0 * np.pi / max(int(sector_count), 1)
    centered_angles = np.mod(angles + 0.5 * sector_width, 2.0 * np.pi)
    sectors = np.floor(
        centered_angles * sector_count / (2.0 * np.pi)
    ).astype(int)
    counts = np.bincount(sectors, minlength=sector_count)
    areas = counts.astype(np.float64) * float(pixel_area_um2)
    supported = areas >= float(minimum_sector_area_um2)
    supported_count = int(np.count_nonzero(supported))
    if supported_count == sector_count:
        contiguous_count = int(sector_count)
    elif supported_count == 0:
        contiguous_count = 0
    else:
        doubled = np.concatenate([supported, supported])
        best = current = 0
        for value in doubled:
            current = current + 1 if value else 0
            best = max(best, current)
        contiguous_count = min(best, int(sector_count))
    return (
        supported_count,
        contiguous_count,
        float(supported_count) / max(int(sector_count), 1),
        tuple(float(value) for value in areas),
    )


def _local_structure_rejection_reasons(
    *,
    anchored_area_um2: float,
    anchored_reach_um: float,
    contact_sector_count: int,
    contiguous_contact_sector_count: int,
    angular_fraction: float,
    config: GFAPNucleusOwnershipConfig,
) -> tuple[str, ...]:
    """Apply the calibrated local-structure gates with inclusive boundaries."""

    reasons: list[str] = []
    if anchored_area_um2 < config.min_anchored_structure_area_um2:
        reasons.append("insufficient_connected_gfap_structure")
    if anchored_reach_um < config.min_anchored_structure_reach_um:
        reasons.append("connected_gfap_structure_too_short")
    if (
        contact_sector_count < config.minimum_strong_contact_sector_count
        or contiguous_contact_sector_count
        < config.minimum_contiguous_strong_contact_sectors
        or angular_fraction < config.min_contact_angular_fraction
    ):
        reasons.append("single_direction_gfap_contact")
    return tuple(reasons)


def select_gfap_associated_owners(
    linked_labels_zyx: np.ndarray,
    gfap_zyx: np.ndarray,
    structural_mask_yx: np.ndarray,
    structural_score_yx: np.ndarray,
    pixel_size_um: float | tuple[float, float],
    z_spacing_um: float,
    *,
    candidate_ids: Iterable[int],
    config: GFAPNucleusOwnershipConfig | None = None,
) -> GFAPAssociationResult:
    """Accept valid 3D nuclei with local, connected GFAP support.

    The rule requires only partial perinuclear GFAP contact, never enclosure.
    A one-plane or single-direction line crossing is rejected.
    """

    active = config or GFAPNucleusOwnershipConfig()
    labels = _validate_slice_labels(linked_labels_zyx)
    gfap = np.asarray(gfap_zyx, dtype=np.float32)
    if gfap.shape != labels.shape:
        raise ValueError("GFAP stack and linked nucleus labels must share ZYX shape")
    if not np.isfinite(gfap).all():
        gfap = np.nan_to_num(gfap, copy=False)
    structural = np.asarray(structural_mask_yx, dtype=bool)
    if structural.shape != labels.shape[1:]:
        raise ValueError("GFAP structural mask and nucleus labels must share XY shape")
    structural_score = np.asarray(structural_score_yx, dtype=np.float32)
    if structural_score.shape != labels.shape[1:]:
        raise ValueError("GFAP structural score and nucleus labels must share XY shape")
    if not np.isfinite(structural_score).all():
        raise ValueError("GFAP structural score contains NaN or infinite values")
    if np.any(structural_score < 0):
        raise ValueError("GFAP structural score cannot be negative")
    z_um, y_um, x_um = _validate_spacing(pixel_size_um, z_spacing_um)
    pixel_area_um2 = y_um * x_um
    voxel_volume_um3 = z_um * pixel_area_um2
    accepted: list[int] = []
    rejected: list[int] = []
    records: list[dict[str, Any]] = []

    nucleus_slices = ndi.find_objects(labels)
    for nucleus_id in sorted({int(value) for value in candidate_ids}):
        reasons: list[str] = []
        nucleus_slice = (
            nucleus_slices[nucleus_id - 1]
            if 0 < nucleus_id <= len(nucleus_slices)
            else None
        )
        if nucleus_slice is None:
            reasons.append("candidate_id_absent_from_3d_labels")
            rejected.append(nucleus_id)
            records.append(
                {
                    "nucleus_id": nucleus_id,
                    "gfap_associated": False,
                    "rejection_reasons": reasons,
                }
            )
            continue

        pad_z = max(1, int(np.ceil(active.background_outer_um / z_um)))
        evidence_radius_um = min(max(float(active.local_evidence_radius_um), 0.0), 8.0)
        xy_context_um = max(active.background_outer_um, evidence_radius_um)
        reference_pad_y = int(
            np.ceil(active.strong_reference_outer_um / y_um)
        )
        reference_pad_x = int(
            np.ceil(active.strong_reference_outer_um / x_um)
        )
        reference_annulus_truncated = bool(
            int(nucleus_slice[1].start) - reference_pad_y < 0
            or int(nucleus_slice[1].stop) + reference_pad_y > labels.shape[1]
            or int(nucleus_slice[2].start) - reference_pad_x < 0
            or int(nucleus_slice[2].stop) + reference_pad_x > labels.shape[2]
        )
        pad_y = max(1, int(np.ceil(xy_context_um / y_um)))
        pad_x = max(1, int(np.ceil(xy_context_um / x_um)))
        z0 = max(0, int(nucleus_slice[0].start) - pad_z)
        z1 = min(labels.shape[0], int(nucleus_slice[0].stop) + pad_z)
        y0 = max(0, int(nucleus_slice[1].start) - pad_y)
        y1 = min(labels.shape[1], int(nucleus_slice[1].stop) + pad_y)
        x0 = max(0, int(nucleus_slice[2].start) - pad_x)
        x1 = min(labels.shape[2], int(nucleus_slice[2].stop) + pad_x)
        local_nucleus = labels[z0:z1, y0:y1, x0:x1] == nucleus_id
        local_gfap = gfap[z0:z1, y0:y1, x0:x1]
        distance = ndi.distance_transform_edt(
            ~local_nucleus,
            sampling=(z_um, y_um, x_um),
        )
        shell = (distance > 0) & (distance <= active.shell_outer_um)
        background = (
            (distance >= active.background_inner_um)
            & (distance <= active.background_outer_um)
        )
        background_values = local_gfap[background]
        if background_values.size < 32:
            background_values = local_gfap[~local_nucleus]
        background_median = (
            float(np.median(background_values)) if background_values.size else 0.0
        )
        background_mad = (
            float(
                1.4826
                * np.median(np.abs(background_values - background_median))
            )
            if background_values.size
            else 0.0
        )
        local_high = float(np.percentile(local_gfap, 99.5))
        numeric_floor = max(
            1.0,
            np.finfo(np.float32).eps * max(abs(local_high), 1.0) * 16.0,
        )
        support_threshold = background_median + max(
            numeric_floor,
            active.shell_support_sigma * background_mad,
            0.08 * max(local_high - background_median, 0.0),
        )
        shell_values = local_gfap[shell]
        shell_p75 = (
            float(np.percentile(shell_values, 75.0))
            if shell_values.size
            else 0.0
        )
        shell_enrichment = (shell_p75 - background_median) / max(
            background_mad,
            numeric_floor,
        )
        supported_shell = shell & (local_gfap >= support_threshold)
        central_z = np.any(local_nucleus, axis=(1, 2))
        supported_z = np.zeros(local_nucleus.shape[0], dtype=bool)
        for local_z in np.flatnonzero(central_z):
            shell_plane = shell[local_z]
            supported_z[local_z] = bool(
                shell_plane.any()
                and float(supported_shell[local_z][shell_plane].mean()) >= 0.03
            )
        supported_z_count = int(np.count_nonzero(supported_z & central_z))
        supported_z_fraction = float(_longest_true_run(supported_z)) / max(
            int(np.count_nonzero(central_z)),
            1,
        )
        shell_support_um3 = float(supported_shell.sum()) * voxel_volume_um3

        nucleus_projection = np.any(local_nucleus, axis=0)
        distance_2d = ndi.distance_transform_edt(
            ~nucleus_projection,
            sampling=(y_um, x_um),
        )
        local_structural = structural[y0:y1, x0:x1]
        local_score = structural_score[y0:y1, x0:x1]
        reference_annulus = (
            (distance_2d >= active.strong_reference_inner_um)
            & (distance_2d <= active.strong_reference_outer_um)
        )
        reference_values = local_score[reference_annulus]
        relative_floor = (
            float(
                np.percentile(
                    reference_values,
                    active.strong_reference_percentile,
                )
            )
            if reference_values.size
            else 0.0
        )
        strong_threshold = max(
            float(active.minimum_strong_structural_score),
            relative_floor,
        )
        local_strong = (
            local_structural
            & (local_score >= strong_threshold)
            & (distance_2d > 0)
            & (distance_2d <= evidence_radius_um)
        )
        local_components, _ = ndi.label(
            local_strong,
            structure=ndi.generate_binary_structure(2, 2),
        )
        contact_shell = (
            (distance_2d > 0)
            & (distance_2d <= active.structure_contact_um)
        )
        component_ids = {
            int(value)
            for value in np.unique(local_components[contact_shell])
            if value > 0
        }
        anchored_local = np.isin(
            local_components,
            tuple(component_ids),
        )
        anchored_area_um2 = float(anchored_local.sum()) * pixel_area_um2
        anchored_reach_um = (
            float(distance_2d[anchored_local].max())
            if anchored_local.any()
            else 0.0
        )
        strong_shell_contact = anchored_local & contact_shell
        (
            contact_sector_count,
            contiguous_contact_sector_count,
            angular_fraction,
            sector_areas_um2,
        ) = (
            _angular_contact_evidence(
            strong_shell_contact,
            nucleus_projection,
            active.angular_sector_count,
            pixel_area_um2,
            active.minimum_strong_contact_sector_area_um2,
        )
        )

        if shell_enrichment < active.min_shell_enrichment:
            reasons.append("weak_perinuclear_gfap_enrichment")
        if reference_annulus_truncated:
            reasons.append("local_reference_annulus_image_truncated")
        if supported_z_count < active.min_supported_z_count:
            reasons.append("gfap_support_not_repeated_across_z")
        if supported_z_fraction < active.min_supported_z_fraction:
            reasons.append("gfap_support_lacks_z_continuity")
        if shell_support_um3 < active.min_shell_support_um3:
            reasons.append("insufficient_perinuclear_gfap_volume")
        reasons.extend(
            _local_structure_rejection_reasons(
                anchored_area_um2=anchored_area_um2,
                anchored_reach_um=anchored_reach_um,
                contact_sector_count=contact_sector_count,
                contiguous_contact_sector_count=(
                    contiguous_contact_sector_count
                ),
                angular_fraction=angular_fraction,
                config=active,
            )
        )
        is_associated = not reasons
        (accepted if is_associated else rejected).append(nucleus_id)
        records.append(
            {
                "nucleus_id": nucleus_id,
                "gfap_associated": is_associated,
                "shell_enrichment": float(shell_enrichment),
                "supported_z_count": supported_z_count,
                "supported_z_fraction": supported_z_fraction,
                "shell_support_um3": shell_support_um3,
                "anchored_structure_area_um2": anchored_area_um2,
                "anchored_structure_reach_um": anchored_reach_um,
                "local_strong_threshold": strong_threshold,
                "local_reference_annulus_image_truncated": (
                    reference_annulus_truncated
                ),
                "local_strong_component_count": len(component_ids),
                "strong_contact_sector_count": contact_sector_count,
                "contiguous_strong_contact_sector_count": (
                    contiguous_contact_sector_count
                ),
                "strong_contact_sector_areas_um2": sector_areas_um2,
                "contact_angular_fraction": angular_fraction,
                "rejection_reasons": reasons,
            }
        )
    return GFAPAssociationResult(
        accepted_ids=tuple(accepted),
        rejected_ids=tuple(rejected),
        records=tuple(records),
    )


def project_exclusive_nucleus_owners(
    linked_labels_zyx: np.ndarray,
    owner_ids: Iterable[int],
    *,
    config: GFAPNucleusOwnershipConfig | None = None,
) -> ExclusiveNucleusProjection:
    """Project accepted owners into an exclusive 2D map with collision audit."""

    active = config or GFAPNucleusOwnershipConfig()
    labels = _validate_slice_labels(linked_labels_zyx)
    requested_ids = tuple(sorted({int(value) for value in owner_ids}))
    projections: dict[int, np.ndarray] = {}
    object_slices = ndi.find_objects(labels)
    for nucleus_id in requested_ids:
        projection = np.zeros(labels.shape[1:], dtype=bool)
        nucleus_slice = (
            object_slices[nucleus_id - 1]
            if 0 < nucleus_id <= len(object_slices)
            else None
        )
        if nucleus_slice is not None:
            z_slice, y_slice, x_slice = nucleus_slice
            projection[y_slice, x_slice] = np.any(
                labels[z_slice, y_slice, x_slice] == nucleus_id,
                axis=0,
            )
        projections[nucleus_id] = projection
    occupancy = np.zeros(labels.shape[1:], dtype=np.uint16)
    for projection in projections.values():
        occupancy += projection.astype(np.uint16)
    collision_mask = occupancy > 1
    rejected: set[int] = set()
    records: list[dict[str, Any]] = []
    for nucleus_id, projection in projections.items():
        projected_area = int(projection.sum())
        exclusive_area = int((projection & (occupancy == 1)).sum())
        exclusive_fraction = float(exclusive_area) / max(projected_area, 1)
        if (
            projected_area == 0
            or exclusive_area == 0
            or exclusive_fraction < active.min_projection_exclusive_fraction
        ):
            rejected.add(nucleus_id)
        records.append(
            {
                "nucleus_id": nucleus_id,
                "projected_area_px": projected_area,
                "exclusive_area_px": exclusive_area,
                "exclusive_fraction": exclusive_fraction,
                "collision_rejected": nucleus_id in rejected,
            }
        )

    retained = tuple(
        nucleus_id for nucleus_id in requested_ids if nucleus_id not in rejected
    )
    retained_projections = {nucleus_id: projections[nucleus_id] for nucleus_id in retained}
    retained_occupancy = np.zeros(labels.shape[1:], dtype=np.uint16)
    for projection in retained_projections.values():
        retained_occupancy += projection.astype(np.uint16)
    output = np.zeros(labels.shape[1:], dtype=np.int32)
    for nucleus_id, projection in retained_projections.items():
        output[projection & (retained_occupancy == 1)] = nucleus_id

    retained_collision = retained_occupancy > 1
    if retained_collision.any():
        distance_to_exclusive: dict[int, np.ndarray] = {}
        for nucleus_id, projection in retained_projections.items():
            exclusive = projection & (retained_occupancy == 1)
            distance_to_exclusive[nucleus_id] = ndi.distance_transform_edt(~exclusive)
        for y_index, x_index in np.argwhere(retained_collision):
            contenders = [
                nucleus_id
                for nucleus_id, projection in retained_projections.items()
                if projection[y_index, x_index]
            ]
            chosen = min(
                contenders,
                key=lambda nucleus_id: (
                    float(distance_to_exclusive[nucleus_id][y_index, x_index]),
                    nucleus_id,
                ),
            )
            output[y_index, x_index] = chosen

    observed = {int(value) for value in np.unique(output) if value > 0}
    missing = set(retained) - observed
    if missing:
        raise RuntimeError(
            "Exclusive nucleus projection unexpectedly lost owner IDs: "
            f"{sorted(missing)}"
        )
    return ExclusiveNucleusProjection(
        labels_yx=output,
        retained_ids=retained,
        collision_rejected_ids=tuple(sorted(rejected)),
        collision_pixels=int(collision_mask.sum()),
        records=tuple(records),
    )


def validate_nucleus_projection(
    nucleus_labels_2d: np.ndarray,
    expected_shape: tuple[int, int],
    valid_3d_ids: set[int],
    *,
    require_all_ids: bool = False,
) -> np.ndarray:
    """Validate an upstream projection without allowing silent owner loss."""

    labels = np.asarray(nucleus_labels_2d)
    if labels.shape != expected_shape:
        raise ValueError("2D nucleus labels and GFAP projection have different shapes")
    if np.issubdtype(labels.dtype, np.floating):
        if not np.all(labels == np.floor(labels)):
            raise ValueError("Nucleus labels must be integer-valued")
    labels = labels.astype(np.int32, copy=False)
    if np.any(labels < 0):
        raise ValueError("Nucleus labels cannot be negative")
    observed = {int(value) for value in np.unique(labels) if value > 0}
    if not observed.issubset(valid_3d_ids):
        raise ValueError("2D nucleus labels contain IDs absent from the 3D owner set")
    if require_all_ids and observed != set(valid_3d_ids):
        missing = sorted(set(valid_3d_ids) - observed)
        raise ValueError(
            "2D nucleus projection silently lost accepted 3D owner IDs: "
            f"{missing}"
        )
    return labels


def resolve_gfap_nucleus_owners(
    slice_labels_zyx: np.ndarray,
    gfap_zyx: np.ndarray,
    structural_mask_yx: np.ndarray,
    structural_score_yx: np.ndarray,
    pixel_size_um: float | tuple[float, float],
    z_spacing_um: float,
    *,
    config: GFAPNucleusOwnershipConfig | None = None,
) -> GFAPNucleusOwnershipResult:
    """Run linking, physical QC, GFAP association, and exclusive projection."""

    active = config or GFAPNucleusOwnershipConfig()
    inventory = link_slice_instances_3d(
        slice_labels_zyx,
        pixel_size_um,
        z_spacing_um,
        config=active,
    )
    association = select_gfap_associated_owners(
        inventory.labels_zyx,
        gfap_zyx,
        structural_mask_yx,
        structural_score_yx,
        pixel_size_um,
        z_spacing_um,
        candidate_ids=inventory.valid_ids,
        config=active,
    )
    association = _apply_unique_weak_enrichment_fallback(
        association,
        active,
    )
    projection = project_exclusive_nucleus_owners(
        inventory.labels_zyx,
        association.accepted_ids,
        config=active,
    )
    return GFAPNucleusOwnershipResult(
        inventory=inventory,
        association=association,
        projection=projection,
    )
