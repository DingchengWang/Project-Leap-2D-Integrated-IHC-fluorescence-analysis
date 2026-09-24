"""Public orchestration API for the independent DAPI + GFAP-only mode."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .gfap_compartments import (
    GFAPCompartmentConfig,
    assign_exclusive_gfap_ownership,
    build_soma_labels,
    partition_compartments,
)
from .gfap_nucleus_ownership import (
    GFAPNucleusOwnershipConfig,
    resolve_gfap_nucleus_owners,
    validate_nucleus_projection,
)
from .gfap_post_compartment_quality import (
    GFAPPostCompartmentQualityConfig,
    apply_gfap_post_compartment_quality,
)
from .gfap_structure import (
    GFAPStructureConfig,
    extract_gfap_structure,
)


@dataclass(frozen=True)
class GFAPOnlyConfig:
    """Complete configuration for the mature-only GFAP analysis route."""

    structure: GFAPStructureConfig = field(default_factory=GFAPStructureConfig)
    compartments: GFAPCompartmentConfig = field(default_factory=GFAPCompartmentConfig)
    nucleus_ownership: GFAPNucleusOwnershipConfig = field(
        default_factory=GFAPNucleusOwnershipConfig
    )
    post_compartment_quality: GFAPPostCompartmentQualityConfig = field(
        default_factory=GFAPPostCompartmentQualityConfig
    )

@dataclass(frozen=True)
class GFAPOnlyResult:
    """Synchronized compartments and auditable intermediate GFAP evidence."""

    whole_labels: np.ndarray
    soma_labels: np.ndarray
    process_labels: np.ndarray
    nucleus_labels_2d: np.ndarray
    valid_nucleus_labels_2d: np.ndarray
    corrected_gfap_projection: np.ndarray
    gfap_intensity: np.ndarray
    gfap_ridge_score: np.ndarray
    gfap_structural_score: np.ndarray
    gfap_structural_mask: np.ndarray
    diagnostics: dict[str, Any]


def _normalize_pixel_size(
    pixel_size_um: float | tuple[float, float],
) -> tuple[float, float]:
    if np.isscalar(pixel_size_um):
        value = float(pixel_size_um)
        result = (value, value)
    else:
        if len(pixel_size_um) != 2:
            raise ValueError("pixel_size_um must be a scalar or (Y, X)")
        result = (float(pixel_size_um[0]), float(pixel_size_um[1]))
    if not all(np.isfinite(value) and value > 0 for value in result):
        raise ValueError("pixel_size_um values must be finite and positive")
    return result


def _project_all_valid_nuclei(
    linked_labels_zyx: np.ndarray,
    valid_ids: tuple[int, ...],
) -> tuple[
    np.ndarray,
    dict[int, tuple[slice, slice, np.ndarray]],
]:
    """Project all valid 3D nuclei with bbox-local work and audited collisions."""

    labels = np.asarray(linked_labels_zyx, dtype=np.int32)
    output = np.zeros(labels.shape[1:], dtype=np.int32)
    occupancy = np.zeros(output.shape, dtype=np.uint16)
    projections: dict[int, tuple[slice, slice, np.ndarray]] = {}
    object_slices = ndi.find_objects(labels)
    for nucleus_id in sorted(int(value) for value in valid_ids):
        nucleus_slice = (
            object_slices[nucleus_id - 1]
            if 0 < nucleus_id <= len(object_slices)
            else None
        )
        if nucleus_slice is None:
            continue
        z_slice, y_slice, x_slice = nucleus_slice
        local_projection = np.any(
            labels[z_slice, y_slice, x_slice] == nucleus_id,
            axis=0,
        )
        projections[nucleus_id] = (y_slice, x_slice, local_projection)
        occupancy[y_slice, x_slice] += local_projection.astype(np.uint16)

    for nucleus_id, (y_slice, x_slice, local_projection) in projections.items():
        occupancy_view = occupancy[y_slice, x_slice]
        output_view = output[y_slice, x_slice]
        output_view[local_projection & (occupancy_view == 1)] = nucleus_id

    valid_set = set(projections)
    valid_values = tuple(sorted(valid_set))
    for y_index, x_index in np.argwhere(occupancy > 1):
        column = labels[:, y_index, x_index]
        column = column[np.isin(column, valid_values)]
        if not column.size:
            continue
        counts = np.bincount(column)
        maximum = int(counts.max())
        output[y_index, x_index] = int(
            np.flatnonzero(counts == maximum)[0]
        )

    # Preserve every independently representable valid identity.  This only
    # affects rare complete XY collisions and never changes the union mask.
    output_counts = np.bincount(
        output.ravel(),
        minlength=max(valid_set, default=0) + 1,
    )
    for nucleus_id in valid_values:
        if output_counts[nucleus_id] > 0:
            continue
        y_slice, x_slice, local_projection = projections[nucleus_id]
        local_coordinates = np.argwhere(local_projection)
        if not local_coordinates.size:
            continue
        center = local_coordinates.mean(axis=0)
        order = np.argsort(
            np.sum((local_coordinates - center) ** 2, axis=1),
            kind="stable",
        )
        for coordinate_index in order:
            local_y, local_x = local_coordinates[coordinate_index]
            y_index = int(y_slice.start) + int(local_y)
            x_index = int(x_slice.start) + int(local_x)
            previous_id = int(output[y_index, x_index])
            if previous_id == 0 or output_counts[previous_id] > 1:
                output[y_index, x_index] = nucleus_id
                output_counts[nucleus_id] += 1
                if previous_id > 0:
                    output_counts[previous_id] -= 1
                break
    return output, projections


def _prioritize_accepted_owner_projection(
    valid_nucleus_projection: np.ndarray,
    accepted_owner_projection: np.ndarray,
) -> np.ndarray:
    """Make Cell Edit collision labels agree with the accepted owner geometry."""

    valid = np.asarray(valid_nucleus_projection, dtype=np.int32)
    accepted = np.asarray(accepted_owner_projection, dtype=np.int32)
    if valid.shape != accepted.shape or valid.ndim != 2:
        raise ValueError("GFAP nucleus projections must be matching 2D arrays")
    if np.any(valid < 0) or np.any(accepted < 0):
        raise ValueError("GFAP nucleus projections cannot contain negative IDs")
    accepted_mask = accepted > 0
    if np.any(accepted_mask & (valid == 0)):
        raise RuntimeError(
            "Accepted GFAP owner projection escaped the valid DAPI nucleus union"
        )
    output = valid.copy()
    output[accepted_mask] = accepted[accepted_mask]
    return output


def analyze_dapi_gfap_only(
    nucleus_labels_3d: np.ndarray,
    gfap_image: np.ndarray,
    pixel_size_um: float | tuple[float, float],
    z_spacing_um: float,
    *,
    nucleus_labels_2d: np.ndarray | None = None,
    config: GFAPOnlyConfig | None = None,
) -> GFAPOnlyResult:
    """Construct Whole/Soma/Processes without eGFP or measurement channels.

    Parameters
    ----------
    nucleus_labels_3d:
        Slice-wise DAPI nuclear instances in ``(Z,Y,X)``.  Instance numbers may
        restart independently on every Z plane; this function resolves global
        3D identity before using any nucleus as an astrocyte owner.
    gfap_image:
        Raw GFAP Z stack ``(Z,Y,X)`` or a selected 2D GFAP projection.
    pixel_size_um:
        Physical ``µm/pixel`` as an isotropic scalar or ``(Y,X)`` tuple.
    z_spacing_um:
        Physical slice spacing, retained in diagnostics for downstream audit.
    nucleus_labels_2d:
        Optional upstream canonical 2D nuclear projection.  Its IDs must match
        the linked, GFAP-associated owner set exactly.

    Notes
    -----
    This API intentionally has no eGFP, KCNN1, or KCNN2 argument.  Those
    channels cannot affect GFAP-only ROI definition.
    """

    pixels_um = _normalize_pixel_size(pixel_size_um)
    z_spacing = float(z_spacing_um)
    if not np.isfinite(z_spacing) or z_spacing <= 0:
        raise ValueError("z_spacing_um must be finite and positive")

    labels_3d = np.asarray(nucleus_labels_3d)
    gfap = np.asarray(gfap_image)
    if labels_3d.ndim != 3:
        raise ValueError("nucleus_labels_3d must have shape (Z, Y, X)")
    if not (labels_3d > 0).any():
        raise ValueError("GFAP-only analysis requires at least one DAPI nucleus")
    expected_shape = tuple(int(value) for value in labels_3d.shape[1:])
    if gfap.ndim == 3:
        if tuple(gfap.shape[1:]) != expected_shape:
            raise ValueError("DAPI nucleus labels and GFAP stack have different XY shapes")
    elif gfap.ndim == 2:
        if tuple(gfap.shape) != expected_shape:
            raise ValueError(
                "DAPI nucleus labels and GFAP projection have different XY shapes"
            )
    else:
        raise ValueError("gfap_image must be a 2D projection or a 3D Z stack")

    active_config = config or GFAPOnlyConfig()

    corrected, intensity, ridge, structural_score, structural_mask, gfap_peak_z = (
        extract_gfap_structure(
            gfap,
            pixels_um,
            active_config.structure,
            return_peak_z=True,
        )
    )
    if gfap.ndim != 3:
        raise ValueError(
            "GFAP-only owner validation requires the raw 3D GFAP Z stack"
        )
    ownership = resolve_gfap_nucleus_owners(
        labels_3d,
        gfap,
        structural_mask,
        structural_score,
        pixels_um,
        z_spacing,
        config=active_config.nucleus_ownership,
    )
    retained_owner_ids = ownership.projection.retained_ids
    if not retained_owner_ids:
        raise ValueError(
            "GFAP-only analysis found no unambiguous GFAP-associated DAPI nucleus"
        )
    projected = ownership.projection.labels_yx
    if nucleus_labels_2d is not None:
        projected = validate_nucleus_projection(
            nucleus_labels_2d,
            expected_shape,
            set(retained_owner_ids),
            require_all_ids=True,
        )
        projected = np.where(
            np.isin(projected, retained_owner_ids),
            projected,
            0,
        ).astype(np.int32)
    accepted_source_projection = projected.copy()

    owner_id_map = {
        owner_id: display_id
        for display_id, owner_id in enumerate(retained_owner_ids, start=1)
    }
    display_projection = np.zeros(projected.shape, dtype=np.int32)
    for owner_id, display_id in owner_id_map.items():
        display_projection[projected == owner_id] = display_id
    projected = display_projection

    valid_nucleus_projection, valid_nucleus_projections = (
        _project_all_valid_nuclei(
            ownership.inventory.labels_zyx,
            ownership.inventory.valid_ids,
        )
    )
    valid_nucleus_projection = _prioritize_accepted_owner_projection(
        valid_nucleus_projection,
        accepted_source_projection,
    )
    competition_projection = np.zeros(projected.shape, dtype=np.int32)
    hard_competitor_mask = np.zeros(projected.shape, dtype=bool)
    next_competition_id = len(owner_id_map) + 1
    marker_z_ranges: dict[int, tuple[int, int]] = {}
    inventory_records = {
        int(record["nucleus_id"]): record
        for record in ownership.inventory.records
    }
    for source_owner_id, display_id in owner_id_map.items():
        record = inventory_records[source_owner_id]
        marker_z_ranges[display_id] = (
            int(record["z_first"]),
            int(record["z_last"]),
        )
    for source_id in ownership.inventory.valid_ids:
        if source_id in owner_id_map:
            continue
        marker_id = next_competition_id
        next_competition_id += 1
        projection_record = valid_nucleus_projections.get(source_id)
        if projection_record is None:
            continue
        y_slice, x_slice, source_projection_local = projection_record
        projected_view = projected[y_slice, x_slice]
        competition_view = competition_projection[y_slice, x_slice]
        hard_view = hard_competitor_mask[y_slice, x_slice]
        true_nonowner_projection = source_projection_local & (projected_view == 0)
        hard_view |= true_nonowner_projection
        available = (
            true_nonowner_projection
            & (competition_view == 0)
        )
        if not available.any():
            pad_y = int(
                np.ceil(
                    (
                        float(
                            active_config.compartments.ownership_seed_max_margin_um
                        )
                        + 2.0 * max(pixels_um)
                    )
                    / pixels_um[0]
                )
            )
            pad_x = int(
                np.ceil(
                    (
                        float(
                            active_config.compartments.ownership_seed_max_margin_um
                        )
                        + 2.0 * max(pixels_um)
                    )
                    / pixels_um[1]
                )
            )
            y0 = max(0, int(y_slice.start) - pad_y)
            y1 = min(projected.shape[0], int(y_slice.stop) + pad_y)
            x0 = max(0, int(x_slice.start) - pad_x)
            x1 = min(projected.shape[1], int(x_slice.stop) + pad_x)
            expanded_source = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            expanded_source[
                int(y_slice.start) - y0 : int(y_slice.stop) - y0,
                int(x_slice.start) - x0 : int(x_slice.stop) - x0,
            ] = source_projection_local
            distance_to_source = ndi.distance_transform_edt(
                ~expanded_source,
                sampling=pixels_um,
            )
            fallback_seed_inner_um = float(
                active_config.compartments.ownership_seed_max_margin_um
            ) + max(pixels_um)
            fallback_seed_radius_um = (
                float(
                    active_config.compartments.ownership_seed_max_margin_um
                )
                + 2.0 * max(pixels_um)
            )
            expanded_available = (
                (distance_to_source >= fallback_seed_inner_um)
                & (distance_to_source <= fallback_seed_radius_um)
                & (projected[y0:y1, x0:x1] == 0)
                & (competition_projection[y0:y1, x0:x1] == 0)
            )
            if expanded_available.any():
                competition_projection[y0:y1, x0:x1][
                    expanded_available
                ] = marker_id
                seeded = True
            else:
                seeded = False
        else:
            competition_view[available] = marker_id
            seeded = True
        if not seeded:
            continue
        record = inventory_records[source_id]
        marker_z_ranges[marker_id] = (
            int(record["z_first"]),
            int(record["z_last"]),
        )

    ownership_seed_config = replace(
        active_config.compartments,
        soma_max_margin_um=(
            active_config.compartments.ownership_seed_max_margin_um
        ),
    )
    ownership_soma = build_soma_labels(
        projected,
        structural_score,
        pixels_um,
        ownership_seed_config,
        hard_exclusion_mask=hard_competitor_mask,
    )
    whole = assign_exclusive_gfap_ownership(
        ownership_soma,
        structural_mask,
        structural_score,
        connectivity=active_config.compartments.ownership_connectivity,
        competition_seed_labels=competition_projection,
        gfap_peak_z_yx=gfap_peak_z,
        marker_z_ranges=marker_z_ranges,
        z_spacing_um=z_spacing,
        pixel_size_um=pixels_um,
        weak_structure_extension_um=(
            active_config.compartments.weak_structure_extension_um
        ),
        weak_structure_score_fraction=(
            active_config.compartments.weak_structure_score_fraction
        ),
        z_competition_weight=active_config.compartments.z_competition_weight,
        z_reassignment_margin_um=(
            active_config.compartments.z_reassignment_margin_um
        ),
    )
    soma = build_soma_labels(
        projected,
        structural_score,
        pixels_um,
        active_config.compartments,
        hard_exclusion_mask=hard_competitor_mask,
    )
    soma = np.where((soma > 0) & (whole == soma), soma, 0).astype(np.int32)
    soma[projected > 0] = projected[projected > 0]
    whole, soma, processes = partition_compartments(whole, soma)
    pre_quality_owner_ids = tuple(retained_owner_ids)
    post_quality = apply_gfap_post_compartment_quality(
        whole,
        soma,
        processes,
        projected,
        pixels_um,
        owner_id_map,
        config=active_config.post_compartment_quality,
    )
    whole = post_quality.whole_labels
    soma = post_quality.soma_labels
    processes = post_quality.process_labels
    projected = post_quality.nucleus_labels_2d
    owner_id_map = dict(post_quality.source_owner_to_display_id)
    retained_owner_ids = tuple(
        source_owner_id
        for source_owner_id, _display_id in sorted(
            owner_id_map.items(),
            key=lambda item: item[1],
        )
    )

    observed_ids = sorted(int(value) for value in np.unique(projected) if value > 0)
    whole_ids = sorted(int(value) for value in np.unique(whole) if value > 0)
    if whole_ids != observed_ids:
        raise RuntimeError("GFAP-only output lost or introduced a DAPI owner identity")
    diagnostics: dict[str, Any] = {
        "analysis_mode": "dapi_gfap_only",
        "age_profile": "mature",
        "pixel_size_yx_um": [float(pixels_um[0]), float(pixels_um[1])],
        "z_spacing_um": z_spacing,
        "nucleus_count": len(observed_ids),
        "nucleus_ids": observed_ids,
        "source_owner_ids": list(retained_owner_ids),
        "pre_quality_source_owner_ids": list(pre_quality_owner_ids),
        "source_owner_to_display_id": {
            str(owner_id): int(display_id)
            for owner_id, display_id in owner_id_map.items()
        },
        "linked_3d_nucleus_count": len(ownership.inventory.records),
        "valid_3d_nucleus_count": len(ownership.inventory.valid_ids),
        "ambiguous_3d_nucleus_ids": list(ownership.inventory.ambiguous_ids),
        "invalid_3d_nucleus_ids": list(ownership.inventory.invalid_ids),
        "gfap_rejected_nucleus_ids": list(ownership.association.rejected_ids),
        "projection_collision_rejected_ids": list(
            ownership.projection.collision_rejected_ids
        ),
        "projection_collision_pixels": ownership.projection.collision_pixels,
        "nucleus_inventory_records": list(ownership.inventory.records),
        "gfap_association_records": list(ownership.association.records),
        "nucleus_projection_records": list(ownership.projection.records),
        "post_compartment_quality_records": list(post_quality.records),
        "post_quality_removed_display_ids": list(
            post_quality.removed_display_ids
        ),
        "post_quality_removed_source_owner_ids": [
            int(record["source_owner_id"])
            for record in post_quality.records
            if not bool(record["passed_post_compartment_quality"])
        ],
        "incomplete_morphology_source_owner_ids": [
            int(record["source_owner_id"])
            for record in post_quality.records
            if bool(record["passed_post_compartment_quality"])
            and bool(record["incomplete_morphology"])
        ],
        "used_supplied_nucleus_projection": nucleus_labels_2d is not None,
        "whole_area_px": int((whole > 0).sum()),
        "soma_area_px": int((soma > 0).sum()),
        "process_area_px": int((processes > 0).sum()),
        "unowned_gfap_structure_px": int(
            (structural_mask & (whole == 0)).sum()
        ),
        "roi_definition_channels": ["DAPI", "GFAP"],
        "measurement_channels_used_for_roi": [],
    }
    return GFAPOnlyResult(
        whole_labels=whole,
        soma_labels=soma,
        process_labels=processes,
        nucleus_labels_2d=projected,
        valid_nucleus_labels_2d=valid_nucleus_projection,
        corrected_gfap_projection=corrected,
        gfap_intensity=intensity,
        gfap_ridge_score=ridge,
        gfap_structural_score=structural_score,
        gfap_structural_mask=structural_mask,
        diagnostics=diagnostics,
    )
