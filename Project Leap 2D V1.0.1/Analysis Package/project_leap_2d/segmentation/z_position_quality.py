# This functional source module is assembled into one shared runtime.
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


Z_POSITION_OPTICAL_WEIGHTS = {
    "sharpness": 4.0 / 11.0,
    "unclipped": 3.0 / 11.0,
    "background_quality": 4.0 / 11.0,
}
Z_POSITION_NUCLEUS_OVERLAP_UM2 = 0.60
# Median final retained-cell count (7, 8, 13) in the three approved primary
# development samples; used by the approved N / (N + Nref) yield mapping.
Z_POSITION_YIELD_REFERENCE_COUNT = 8.0
_ALLOWED_STRUCTURAL_CHANNELS = frozenset({"eGFP", "GFAP"})


def _finite_float64(array: np.ndarray, *, name: str) -> np.ndarray:
    converted = np.asarray(array, dtype=np.float64)
    if converted.ndim not in (2, 3) or not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must be a finite 2D or 3D array")
    return converted


def _robust01_float64(array: np.ndarray) -> np.ndarray:
    values = _finite_float64(array, name="intensity array")
    low, high = np.percentile(values, [0.5, 99.8])
    if high <= low:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _active_structural_weights(
    structural_stacks: Mapping[str, np.ndarray],
) -> dict[str, float]:
    observed = set(structural_stacks)
    unexpected = observed - _ALLOWED_STRUCTURAL_CHANNELS
    if unexpected:
        raise ValueError(
            "Z-position scoring accepts only eGFP and GFAP structural channels; "
            f"unexpected channels: {', '.join(sorted(unexpected))}"
        )
    if not observed:
        raise ValueError("At least one structural channel is required")
    requested = {"eGFP": 0.55, "GFAP": 0.45}
    total = sum(requested[channel] for channel in observed)
    return {channel: requested[channel] / total for channel in sorted(observed)}


def _otsu_support_from_plane(plane: np.ndarray) -> np.ndarray:
    normalized = _robust01_float64(plane)
    if not np.any(normalized > 0.0):
        return np.zeros(normalized.shape, dtype=bool)
    try:
        threshold = float(filters.threshold_otsu(normalized))
    except ValueError:
        return np.zeros(normalized.shape, dtype=bool)
    return np.asarray(normalized > threshold, dtype=bool)


def _z_boundary_continuation_score(
    structural_stacks: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    whole_mask: np.ndarray,
    z0: int,
    z1: int,
) -> tuple[float, float, float]:
    mask = np.asarray(whole_mask, dtype=bool)
    depth = int(np.asarray(structural_stacks[next(iter(weights))]).shape[0])

    def side_risk(boundary_z: int, outside_z: int) -> float:
        if outside_z < 0 or outside_z >= depth:
            return 0.0
        weighted_risk = 0.0
        for channel, weight in weights.items():
            boundary_support = (
                _otsu_support_from_plane(
                    np.asarray(structural_stacks[channel])[boundary_z]
                )
                & mask
            )
            boundary_count = int(np.count_nonzero(boundary_support))
            if boundary_count == 0:
                continue
            outside_support = _otsu_support_from_plane(
                np.asarray(structural_stacks[channel])[outside_z]
            )
            outside_nearby = morphology.binary_dilation(
                outside_support,
                footprint=morphology.disk(1),
            )
            channel_risk = float(
                np.count_nonzero(boundary_support & outside_nearby)
            ) / float(boundary_count)
            weighted_risk += float(weight) * channel_risk
        return float(weighted_risk)

    top_risk = side_risk(z0, z0 - 1)
    bottom_risk = side_risk(z1, z1 + 1)
    risk = 0.5 * (top_risk + bottom_risk)
    return float(np.clip(1.0 - risk, 0.0, 1.0)), top_risk, bottom_risk


def _dapi_plane_normalization_bounds(
    stack: np.ndarray,
    *,
    max_workers: int,
) -> tuple[tuple[float, float], ...]:
    volume = np.asarray(stack)
    if volume.ndim != 3 or volume.shape[0] == 0:
        raise ValueError("DAPI normalization requires a non-empty 3D stack")

    def plane_bounds(plane: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(np.asarray(plane), [0.5, 99.8])
        return float(low), float(high)

    worker_count = max(1, min(int(max_workers), int(volume.shape[0])))
    if worker_count == 1:
        return tuple(plane_bounds(volume[z_index]) for z_index in range(volume.shape[0]))

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="ihc-z-dapi-bounds",
    ) as executor:
        bounds: list[tuple[float, float]] = []
        for chunk_start in range(0, int(volume.shape[0]), worker_count):
            chunk_end = min(chunk_start + worker_count, int(volume.shape[0]))
            bounds.extend(
                executor.map(
                    plane_bounds,
                    (volume[z_index] for z_index in range(chunk_start, chunk_end)),
                )
            )
        return tuple(bounds)


def _normalized_plane_values(
    plane: np.ndarray,
    mask: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(plane)[np.asarray(mask, dtype=bool)].astype(
        np.float64,
        copy=False,
    )
    low, high = bounds
    if high <= low:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _fill_internal_axial_gaps(
    active: np.ndarray,
    *,
    maximum_gap_planes: int = 2,
) -> np.ndarray:
    filled = np.asarray(active, dtype=bool).copy()
    index = 0
    while index < filled.size:
        if filled[index]:
            index += 1
            continue
        gap_start = index
        while index < filled.size and not filled[index]:
            index += 1
        gap_end = index
        if (
            gap_start > 0
            and gap_end < filled.size
            and gap_end - gap_start <= maximum_gap_planes
        ):
            filled[gap_start:gap_end] = True
    return filled


def _remove_short_axial_runs(
    active: np.ndarray,
    *,
    minimum_planes: int,
) -> np.ndarray:
    retained = np.asarray(active, dtype=bool).copy()
    start = 0
    while start < retained.size:
        if not retained[start]:
            start += 1
            continue
        end = start + 1
        while end < retained.size and retained[end]:
            end += 1
        if end - start < int(minimum_planes):
            retained[start:end] = False
        start = end
    return retained


def _axial_runs_for_core(
    *,
    core_mask: np.ndarray,
    extent_mask: np.ndarray,
    dapi_stack: np.ndarray,
    plane_bounds: Sequence[tuple[float, float]],
    halo_z0: int,
    halo_z1: int,
    pixel_depth_um: float,
    validation_config: Any,
) -> tuple[dict[str, Any], ...]:
    profile: list[float] = []
    field_values: list[np.ndarray] = []
    for z_index in range(halo_z0, halo_z1 + 1):
        core_values = _normalized_plane_values(
            dapi_stack[z_index],
            core_mask,
            plane_bounds[z_index],
        )
        extent_values = _normalized_plane_values(
            dapi_stack[z_index],
            extent_mask,
            plane_bounds[z_index],
        )
        profile.append(
            float(np.percentile(core_values, 85.0)) if core_values.size else 0.0
        )
        if extent_values.size:
            field_values.append(extent_values)
    profile_array = np.asarray(profile, dtype=np.float64)
    if profile_array.size == 0 or not field_values:
        return ()
    field = np.concatenate(field_values)
    field_dynamic = max(
        float(np.percentile(field, 99.9)) - float(np.percentile(field, 0.1)),
        0.0,
    )
    baseline = float(np.percentile(profile_array, 10.0))
    peak = float(np.max(profile_array))
    contrast = max(peak - baseline, 0.0)
    numeric_floor = (
        np.finfo(np.float32).eps
        * max(float(np.max(np.abs(field))), float(np.finfo(np.float32).tiny))
        * 16.0
    )
    minimum_dynamic = max(numeric_floor, 0.04 * field_dynamic)
    if contrast < minimum_dynamic:
        return ()
    active_threshold = (
        baseline
        + float(validation_config.dapi_active_profile_fraction) * contrast
    )
    high_threshold = (
        baseline + float(validation_config.dapi_high_fraction) * contrast
    )
    minimum_active_planes = max(
        1,
        int(
            math.ceil(
                float(validation_config.dapi_min_z_span_um)
                / float(pixel_depth_um)
            )
        ),
    )
    active = _remove_short_axial_runs(
        profile_array >= active_threshold,
        minimum_planes=minimum_active_planes,
    )
    active = _fill_internal_axial_gaps(active)
    high_seed = profile_array >= high_threshold
    runs: list[dict[str, Any]] = []
    start = 0
    while start < active.size:
        if not active[start]:
            start += 1
            continue
        end = start + 1
        while end < active.size and active[end]:
            end += 1
        active_count = end - start
        if (
            np.any(high_seed[start:end])
            and active_count * float(pixel_depth_um)
            >= float(validation_config.dapi_min_z_span_um)
        ):
            active_indices = tuple(
                range(halo_z0 + start, halo_z0 + end)
            )
            runs.append(
                {
                    "center_z": 0.5 * (active_indices[0] + active_indices[-1]),
                    "z_min_0based": active_indices[0],
                    "z_max_0based_inclusive": active_indices[-1],
                    "active_z_indices_0based": active_indices,
                    "active_plane_count": active_count,
                    "profile_baseline": baseline,
                    "profile_contrast": contrast,
                    "field_dynamic": field_dynamic,
                }
            )
        start = end
    return tuple(runs)


def _fixed_window_structural_map(
    structural_projections: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    projections: dict[str, np.ndarray] = {}
    combined: np.ndarray | None = None
    for channel, weight in weights.items():
        raw_projection = np.asarray(structural_projections[channel])
        projections[channel] = np.asarray(raw_projection)
        normalized = _robust01_float64(raw_projection)
        combined = (
            normalized * float(weight)
            if combined is None
            else combined + normalized * float(weight)
        )
    assert combined is not None
    maximum = float(np.max(combined))
    if maximum > 0.0:
        combined = combined / maximum
    combined = filters.gaussian(
        combined,
        sigma=1.0,
        preserve_range=True,
    )
    return np.clip(combined, 0.0, 1.0).astype(np.float64), projections


def _fixed_whole_metrics(
    mask: np.ndarray,
    structural_map_2d: np.ndarray,
    bidirectional_component_ids: Sequence[int],
) -> dict[str, float]:
    whole = np.asarray(mask, dtype=bool)
    struct = np.asarray(structural_map_2d, dtype=np.float64)
    area = int(np.count_nonzero(whole))
    if area == 0:
        return {
            "structural_coverage": 0.0,
            "structural_precision": 0.0,
            "unsupported_wide_fraction": 0.0,
            "unanchored_area_fraction": 0.0,
            "xy_edge_area_fraction": 0.0,
        }

    if np.all(struct == struct.flat[0]):
        support = np.zeros(struct.shape, dtype=bool)
    else:
        try:
            support_threshold = float(filters.threshold_otsu(struct))
        except ValueError:
            support = np.zeros(struct.shape, dtype=bool)
        else:
            support = struct > support_threshold
    supported_in_whole = int(np.count_nonzero(support & whole))
    support_area = int(np.count_nonzero(support))
    structural_coverage = (
        float(supported_in_whole) / float(support_area) if support_area else 0.0
    )
    structural_precision = float(supported_in_whole) / float(area)
    distance = ndi.distance_transform_edt(whole)
    unsupported_wide = whole & (distance > 10.0) & ~support
    unsupported_wide_fraction = float(np.count_nonzero(unsupported_wide)) / area

    labels = measure.label(whole)
    component_count = int(labels.max())
    component_areas = np.bincount(
        labels.ravel(),
        minlength=component_count + 1,
    )
    bidirectional_ids = np.asarray(
        sorted({int(value) for value in bidirectional_component_ids}),
        dtype=np.int64,
    )
    all_component_ids = np.arange(1, component_count + 1, dtype=np.int64)
    unanchored_labels = np.setdiff1d(all_component_ids, bidirectional_ids)
    unanchored_area_fraction = (
        float(component_areas[unanchored_labels].sum(dtype=np.int64)) / area
    )

    border_ids = np.unique(
        np.concatenate(
            (
                labels[0],
                labels[-1],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )
    border_ids = border_ids[border_ids > 0]
    xy_edge_area_fraction = (
        float(component_areas[border_ids].sum(dtype=np.int64)) / area
        if border_ids.size
        else 0.0
    )
    return {
        "structural_coverage": float(np.clip(structural_coverage, 0.0, 1.0)),
        "structural_precision": float(np.clip(structural_precision, 0.0, 1.0)),
        "unsupported_wide_fraction": float(
            np.clip(unsupported_wide_fraction, 0.0, 1.0)
        ),
        "unanchored_area_fraction": float(
            np.clip(unanchored_area_fraction, 0.0, 1.0)
        ),
        "xy_edge_area_fraction": float(
            np.clip(xy_edge_area_fraction, 0.0, 1.0)
        ),
    }


def _sensor_rail(dtype: np.dtype[Any]) -> tuple[float, float] | None:
    resolved = np.dtype(dtype)
    if np.issubdtype(resolved, np.integer):
        info = np.iinfo(resolved)
        return float(info.min), float(info.max)
    if np.issubdtype(resolved, np.bool_):
        return 0.0, 1.0
    return None


def _optical_channel_metrics(
    projection: np.ndarray,
    foreground: np.ndarray,
    *,
    rail: tuple[float, float] | None = None,
) -> dict[str, float]:
    raw = np.asarray(projection)
    image = _finite_float64(raw, name="optical projection")
    if image.ndim != 2:
        raise ValueError("Optical quality requires a 2D projection")
    selected = np.asarray(foreground, dtype=bool)
    if selected.shape != image.shape:
        raise ValueError("Optical foreground shape does not match projection")

    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    first_energy = (
        (float(np.mean(dx * dx, dtype=np.float64)) if dx.size else 0.0)
        + (float(np.mean(dy * dy, dtype=np.float64)) if dy.size else 0.0)
    )
    laplacian = ndi.laplace(image, mode="nearest")
    second_energy = float(np.mean(laplacian * laplacian, dtype=np.float64))
    sharp_denominator = first_energy + second_energy
    sharpness = (
        second_energy / sharp_denominator if sharp_denominator > 0.0 else 0.0
    )

    foreground_values = image[selected]
    effective_rail = rail if rail is not None else _sensor_rail(raw.dtype)
    if foreground_values.size == 0:
        unclipped = 0.0
    elif effective_rail is None:
        raise ValueError(
            "Floating-point optical projections require an explicit sensor rail"
        )
    else:
        low_rail, high_rail = map(float, effective_rail)
        low_fraction = float(np.mean(foreground_values <= low_rail))
        high_fraction = float(np.mean(foreground_values >= high_rail))
        unclipped = (1.0 - low_fraction) * (1.0 - high_fraction)

    background = ~selected
    background_values = image[background]
    if foreground_values.size == 0 or background_values.size == 0:
        background_quality = 0.0
    else:
        background_median = float(np.median(background_values))
        foreground_separation = max(
            float(np.median(foreground_values)) - background_median,
            0.0,
        )
        background_noise = 1.4826 * float(
            np.median(np.abs(background_values - background_median))
        )
        row_medians = np.asarray(
            [
                np.median(image[row, background[row]])
                for row in range(image.shape[0])
                if np.any(background[row])
            ],
            dtype=np.float64,
        )
        column_medians = np.asarray(
            [
                np.median(image[background[:, column], column])
                for column in range(image.shape[1])
                if np.any(background[:, column])
            ],
            dtype=np.float64,
        )
        row_offset = (
            float(np.median((row_medians - background_median) ** 2))
            if row_medians.size
            else 0.0
        )
        column_offset = (
            float(np.median((column_medians - background_median) ** 2))
            if column_medians.size
            else 0.0
        )
        nonuniformity = float(np.sqrt(row_offset + column_offset))
        denominator = foreground_separation + background_noise + nonuniformity
        background_quality = (
            foreground_separation / denominator if denominator > 0.0 else 0.0
        )

    return {
        "sharpness": float(np.clip(sharpness, 0.0, 1.0)),
        "unclipped": float(np.clip(unclipped, 0.0, 1.0)),
        "background_quality": float(np.clip(background_quality, 0.0, 1.0)),
    }


def _weighted_channel_metrics(
    metrics_by_channel: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    return {
        metric: float(
            sum(float(weights[channel]) * float(metrics_by_channel[channel][metric])
                for channel in weights)
        )
        for metric in Z_POSITION_OPTICAL_WEIGHTS
    }


def _integer_pair_count_matrix(
    nucleus_labels: np.ndarray,
    whole_component_labels: np.ndarray,
    group_count: int,
    component_count: int,
) -> np.ndarray:
    group_labels = np.asarray(nucleus_labels, dtype=np.int64)
    whole_labels = np.asarray(whole_component_labels, dtype=np.int64)
    if group_labels.shape != whole_labels.shape:
        raise ValueError("Nucleus and Whole label geometry must match")
    active = (group_labels > 0) & (whole_labels > 0)
    pair_counts = np.zeros((group_count + 1, component_count + 1), dtype=np.int64)
    if not np.any(active):
        return pair_counts
    codes = (
        group_labels[active] * (component_count + 1)
        + whole_labels[active]
    )
    pair_counts = np.bincount(
        codes,
        minlength=(group_count + 1) * (component_count + 1),
    ).reshape(group_count + 1, component_count + 1)
    return pair_counts


def _axial_run_association(
    runs: Sequence[Mapping[str, Any]],
    extent_core_labels: np.ndarray,
    whole_mask: np.ndarray,
    pixel_area_um2: float,
) -> tuple[
    list[Mapping[str, Any]],
    np.ndarray,
    list[Mapping[str, Any]],
    tuple[int, ...],
    int,
    int,
    int,
]:
    whole_labels = measure.label(np.asarray(whole_mask, dtype=bool))
    component_count = int(whole_labels.max())
    core_count = int(np.max(extent_core_labels))
    pair_counts = _integer_pair_count_matrix(
        extent_core_labels,
        whole_labels,
        core_count,
        component_count,
    )
    significant_by_core = (
        pair_counts.astype(np.float64) * float(pixel_area_um2)
        >= float(Z_POSITION_NUCLEUS_OVERLAP_UM2)
    )
    run_core_labels = np.asarray(
        [int(run["core_label"]) for run in runs],
        dtype=np.int64,
    )
    significant = significant_by_core[run_core_labels, 1:]
    nucleus_degree = significant.sum(axis=1, dtype=np.int64)
    whole_degree = significant.sum(axis=0, dtype=np.int64)
    associated_indices = np.flatnonzero(nucleus_degree > 0)
    associated_runs = [runs[int(index)] for index in associated_indices]
    associated_core_labels = [
        int(run["core_label"]) for run in associated_runs
    ]
    associated_projection = np.isin(
        extent_core_labels,
        associated_core_labels,
    )
    matched_run_indices: list[int] = []
    matched_component_ids: list[int] = []
    for run_index in np.flatnonzero(nucleus_degree == 1):
        component_index_zero_based = int(
            np.flatnonzero(significant[run_index])[0]
        )
        if int(whole_degree[component_index_zero_based]) != 1:
            continue
        matched_run_indices.append(int(run_index))
        matched_component_ids.append(component_index_zero_based + 1)
    matched_runs = [runs[index] for index in matched_run_indices]
    unique_count = len(matched_runs)
    return (
        associated_runs,
        associated_projection,
        matched_runs,
        tuple(matched_component_ids),
        len(associated_runs),
        component_count,
        unique_count,
    )


def _nuclear_position_score(
    records: Sequence[Mapping[str, Any]],
    z0: int,
    z1: int,
    pixel_depth_um: float,
) -> tuple[float, float]:
    midpoint = (float(z0) + float(z1)) / 2.0
    half_span_um = (float(z1) - float(z0)) * float(pixel_depth_um) / 2.0
    if not records or half_span_um <= 0.0:
        return 0.0, float("inf")
    offsets_um = np.asarray(
        [
            (float(record["center_z"]) - midpoint) * float(pixel_depth_um)
            for record in records
        ],
        dtype=np.float64,
    )
    rms_um = float(np.sqrt(np.mean(offsets_um * offsets_um, dtype=np.float64)))
    return float(np.clip(1.0 - rms_um / half_span_um, 0.0, 1.0)), rms_um


def _nucleus_axial_retention_fraction(
    record: Mapping[str, Any],
    z0: int,
    z1: int,
) -> float:
    active_indices = tuple(
        int(value) for value in record["active_z_indices_0based"]
    )
    nucleus_plane_count = len(active_indices)
    if nucleus_plane_count <= 0:
        raise ValueError("DAPI axial run has no active planes")
    overlap_plane_count = sum(
        z0 <= z_index <= z1 for z_index in active_indices
    )
    return float(overlap_plane_count) / float(nucleus_plane_count)


def _nucleus_quality_score(
    mean_axial_retention_fraction: float,
    association_score: float,
) -> float:
    return float(
        (
            16.0 * float(mean_axial_retention_fraction)
            + 8.0 * float(association_score)
        )
        / 24.0
    )


def _position_total_score(
    *,
    yield_score: float,
    nucleus_score: float,
    whole_score: float,
    centrality_score: float,
    optical_score: float,
) -> float:
    return float(
        0.30 * float(yield_score)
        + 0.24 * float(nucleus_score)
        + 0.27 * float(whole_score)
        + 0.08 * float(centrality_score)
        + 0.11 * float(optical_score)
    )


def _yield_score(unique_nucleus_count: int) -> float:
    count = max(0.0, float(unique_nucleus_count))
    return count / (count + Z_POSITION_YIELD_REFERENCE_COUNT)


def _assign_extent_to_nearest_core(
    nuclei_core: np.ndarray,
    nuclei_extent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    core_labels = measure.label(np.asarray(nuclei_core, dtype=bool), connectivity=2)
    if int(core_labels.max()) == 0:
        empty = np.zeros(core_labels.shape, dtype=np.uint32)
        return core_labels.astype(np.uint32), empty
    _, nearest_indices = ndi.distance_transform_edt(
        core_labels == 0,
        return_indices=True,
    )
    nearest_core = core_labels[tuple(nearest_indices)]
    extent_labels = np.where(
        np.asarray(nuclei_extent, dtype=bool),
        nearest_core,
        0,
    ).astype(np.uint32)
    extent_labels[core_labels > 0] = core_labels[core_labels > 0]
    return core_labels.astype(np.uint32), extent_labels


def _build_position_dapi_axial_inventory(
    *,
    position_index: int,
    whole_mask: np.ndarray,
    dapi_projection: np.ndarray,
    dapi_stack: np.ndarray,
    plane_bounds: Sequence[tuple[float, float]],
    main_z0: int,
    main_z1: int,
    guard_slices: int,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float,
    compartment_config: Any,
    validation_config: Any,
) -> dict[str, Any]:
    nuclei_core, nuclei_extent, _, dapi_metrics = dapi_nuclei_core_and_extent(
        np.asarray(dapi_projection),
        math.sqrt(float(pixel_width_um) * float(pixel_height_um)),
        compartment_config,
    )
    core_labels, extent_core_labels = _assign_extent_to_nearest_core(
        nuclei_core,
        nuclei_extent,
    )
    core_count = int(core_labels.max())
    overlap_counts = np.bincount(
        extent_core_labels[np.asarray(whole_mask, dtype=bool)],
        minlength=core_count + 1,
    )
    candidate_core_ids = tuple(
        core_id
        for core_id in range(1, core_count + 1)
        if float(overlap_counts[core_id])
        * float(pixel_width_um)
        * float(pixel_height_um)
        >= Z_POSITION_NUCLEUS_OVERLAP_UM2
    )
    stack_depth = int(np.asarray(dapi_stack).shape[0])
    halo_z0 = max(0, int(main_z0) - int(guard_slices))
    halo_z1 = min(stack_depth - 1, int(main_z1) + int(guard_slices))
    runs: list[dict[str, Any]] = []
    for core_id in candidate_core_ids:
        core_runs = _axial_runs_for_core(
            core_mask=core_labels == core_id,
            extent_mask=extent_core_labels == core_id,
            dapi_stack=dapi_stack,
            plane_bounds=plane_bounds,
            halo_z0=halo_z0,
            halo_z1=halo_z1,
            pixel_depth_um=float(pixel_depth_um),
            validation_config=validation_config,
        )
        for core_run_index, run in enumerate(core_runs, start=1):
            runs.append(
                {
                    **run,
                    "run_id": len(runs) + 1,
                    "core_label": int(core_id),
                    "core_run_index": core_run_index,
                }
            )
    return {
        "position_index": int(position_index),
        "main_z_start_0based": int(main_z0),
        "main_z_end_0based_inclusive": int(main_z1),
        "profile_z_start_0based": halo_z0,
        "profile_z_end_0based_inclusive": halo_z1,
        "core_count": core_count,
        "whole_linked_core_count": len(candidate_core_ids),
        "axial_run_count": len(runs),
        "runs": tuple(runs),
        "extent_core_labels_2d": extent_core_labels,
        "dapi_metrics": dict(dapi_metrics),
    }


def _build_position_dapi_axial_inventories(
    *,
    masks: Sequence[np.ndarray],
    z_ranges: Sequence[tuple[int, int]],
    position_dapi_projections: Sequence[np.ndarray],
    guard_slices: int,
    dapi_stack: np.ndarray,
    plane_bounds: Sequence[tuple[float, float]],
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float,
    compartment_config: Any,
    validation_config: Any,
    max_workers: int,
) -> tuple[dict[str, Any], ...]:
    if (
        len(masks) != 5
        or len(z_ranges) != 5
        or len(position_dapi_projections) != 5
    ):
        raise ValueError("Exactly five DAPI axial-profile inventories are required")
    arguments: list[dict[str, Any]] = []
    for position_index, (mask, (z0, z1), dapi_projection) in enumerate(
        zip(masks, z_ranges, position_dapi_projections)
    ):
        arguments.append(
            {
                "position_index": position_index,
                "whole_mask": mask,
                "dapi_projection": dapi_projection,
                "dapi_stack": dapi_stack,
                "plane_bounds": plane_bounds,
                "main_z0": z0,
                "main_z1": z1,
                "guard_slices": guard_slices,
                "pixel_width_um": pixel_width_um,
                "pixel_height_um": pixel_height_um,
                "pixel_depth_um": pixel_depth_um,
                "compartment_config": compartment_config,
                "validation_config": validation_config,
            }
        )

    def build_one(argument: Mapping[str, Any]) -> dict[str, Any]:
        return _build_position_dapi_axial_inventory(**argument)

    outer_workers = max(1, min(5, int(max_workers)))
    if outer_workers == 1:
        return tuple(build_one(argument) for argument in arguments)
    with ThreadPoolExecutor(
        max_workers=outer_workers,
        thread_name_prefix="ihc-z-position-inventory",
    ) as inventory_executor:
        return tuple(inventory_executor.map(build_one, arguments))


def _select_unique_highest(position_scores: Sequence[Mapping[str, Any]]) -> tuple[str, int | None]:
    valid = [
        (index, float(position["total_score"]))
        for index, position in enumerate(position_scores)
        if bool(position["valid"])
    ]
    if not valid:
        return "invalid", None
    highest = max(score for _, score in valid)
    best_indices = [index for index, score in valid if score == highest]
    if len(best_indices) != 1:
        return "ambiguous", None
    return "selected", best_indices[0]


def _score_single_z_position(
    *,
    position_index: int,
    mask: np.ndarray,
    z0: int,
    z1: int,
    axial_runs: Sequence[Mapping[str, Any]],
    extent_core_labels: np.ndarray,
    pixel_area_um2: float,
    structural_stacks: Mapping[str, np.ndarray],
    structural_projections: Mapping[str, np.ndarray],
    dapi_projection: np.ndarray,
    weights: Mapping[str, float],
    pixel_depth_um: float,
) -> dict[str, Any]:
    centered_runs = tuple(
        run
        for run in axial_runs
        if z0 <= float(run["center_z"]) <= z1
    )
    (
        associated_runs,
        dapi_foreground,
        unique_runs,
        unique_component_ids,
        associated_nucleus_count,
        component_count,
        unique_count,
    ) = _axial_run_association(
        centered_runs,
        extent_core_labels,
        mask,
        pixel_area_um2,
    )
    valid = bool(unique_count > 0 and component_count > 0)
    yield_score = _yield_score(unique_count)
    axial_retention_fractions = tuple(
        _nucleus_axial_retention_fraction(run, z0, z1)
        for run in associated_runs
    )
    complete_count = sum(value == 1.0 for value in axial_retention_fractions)
    mean_axial_retention = (
        float(np.mean(axial_retention_fractions, dtype=np.float64))
        if axial_retention_fractions
        else 0.0
    )
    association_score = (
        2.0 * float(unique_count)
        / float(associated_nucleus_count + component_count)
        if associated_nucleus_count + component_count > 0
        else 0.0
    )
    nucleus_score = _nucleus_quality_score(
        mean_axial_retention,
        association_score,
    )
    centrality_score, rms_um = _nuclear_position_score(
        associated_runs,
        z0,
        z1,
        float(pixel_depth_um),
    )

    structural_map_2d, cached_structural_projections = (
        _fixed_window_structural_map(
            structural_projections,
            weights,
        )
    )
    fixed_metrics = _fixed_whole_metrics(
        mask,
        structural_map_2d,
        unique_component_ids,
    )
    boundary_completeness, top_boundary_risk, bottom_boundary_risk = (
        _z_boundary_continuation_score(
            structural_stacks,
            weights,
            mask,
            z0,
            z1,
        )
    )
    whole_score = (
        8.0 * boundary_completeness
        + 6.0 * fixed_metrics["structural_coverage"]
        + 5.0 * fixed_metrics["structural_precision"]
        + 3.0 * (1.0 - fixed_metrics["unsupported_wide_fraction"])
        + 3.0 * (1.0 - fixed_metrics["unanchored_area_fraction"])
        + 2.0 * (1.0 - fixed_metrics["xy_edge_area_fraction"])
    ) / 27.0

    dapi_optical = _optical_channel_metrics(
        dapi_projection,
        dapi_foreground,
    )
    structural_optical_by_channel = {
        channel: _optical_channel_metrics(projection, mask)
        for channel, projection in cached_structural_projections.items()
    }
    structural_optical = _weighted_channel_metrics(
        structural_optical_by_channel,
        weights,
    )
    optical_components = {
        metric: 0.5 * dapi_optical[metric] + 0.5 * structural_optical[metric]
        for metric in Z_POSITION_OPTICAL_WEIGHTS
    }
    optical_score = (
        4.0 * optical_components["sharpness"]
        + 3.0 * optical_components["unclipped"]
        + 4.0 * optical_components["background_quality"]
    ) / 11.0
    total_score = _position_total_score(
        yield_score=yield_score,
        nucleus_score=nucleus_score,
        whole_score=whole_score,
        centrality_score=centrality_score,
        optical_score=optical_score,
    )
    return {
        "position_index": position_index,
        "z_start_0based": z0,
        "z_end_0based_inclusive": z1,
        "valid": valid,
        "associated_nucleus_count": associated_nucleus_count,
        "unique_nucleus_count": unique_count,
        "whole_component_count": component_count,
        "unique_whole_component_count": unique_count,
        "complete_nucleus_count": complete_count,
        "yield_score": float(yield_score),
        "nucleus_score": float(nucleus_score),
        "whole_score": float(whole_score),
        "centrality_score": float(centrality_score),
        "optical_score": float(optical_score),
        "total_score": float(total_score),
        "mean_nucleus_axial_retention_fraction": float(mean_axial_retention),
        "association_score": float(association_score),
        "nucleus_center_rms_um": float(rms_um),
        "z_boundary_completeness": boundary_completeness,
        "top_z_boundary_continuation_risk": float(top_boundary_risk),
        "bottom_z_boundary_continuation_risk": float(bottom_boundary_risk),
        **fixed_metrics,
        "dapi_optical": dapi_optical,
        "structural_optical": structural_optical,
        "combined_optical": optical_components,
    }


def score_z_position_best_candidates(
    *,
    position_best_candidates: list[tuple[np.ndarray, dict, Any]],
    dapi_stack: np.ndarray,
    structural_stacks: dict[str, np.ndarray],
    position_projections: Sequence[
        tuple[np.ndarray, Mapping[str, np.ndarray]]
    ],
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float,
    compartment_config: Any,
    validation_config: Any | None = None,
    max_workers: int = 1,
    calibration_source: str = "z_position_selection",
) -> dict[str, Any]:
    """Score five already-selected same-position best Whole candidates.

    This selector consumes only DAPI and structural eGFP/GFAP pixels. Candidate
    brightness activity, cross-Z overlap, cross-position ranks, and measurement
    channels are deliberately absent from the interface and calculation.
    """

    if len(position_best_candidates) != 5:
        raise ValueError("Exactly five best Z-position candidates are required")
    if len(position_projections) != 5:
        raise ValueError("Exactly five cached Z-position projections are required")
    dapi = np.asarray(dapi_stack)
    if dapi.ndim != 3 or dapi.shape[0] < 2 or not np.all(np.isfinite(dapi)):
        raise ValueError("DAPI stack must contain at least two Z planes")
    if min(pixel_width_um, pixel_height_um, pixel_depth_um) <= 0.0:
        raise ValueError("Positive XYZ calibration is required")
    weights = _active_structural_weights(structural_stacks)
    for channel, stack in structural_stacks.items():
        if np.asarray(stack).shape != np.asarray(dapi_stack).shape:
            raise ValueError(f"{channel} stack geometry does not match DAPI")

    masks: list[np.ndarray] = []
    z_ranges: list[tuple[int, int]] = []
    for mask, row, _spec in position_best_candidates:
        whole = np.asarray(mask, dtype=bool)
        if whole.shape != dapi.shape[1:]:
            raise ValueError("Best-candidate Whole geometry does not match input stacks")
        z0 = int(row["z_start_0based"])
        z1 = int(row["z_end_0based_inclusive"])
        if z0 < 0 or z1 >= dapi.shape[0] or z1 <= z0:
            raise ValueError(f"Invalid best-candidate Z interval: {z0}-{z1}")
        masks.append(whole)
        z_ranges.append((z0, z1))

    available_z0 = min(z0 for z0, _ in z_ranges)
    available_z1 = max(z1 for _, z1 in z_ranges)
    cfg = validation_config if validation_config is not None else DapiNucleus3DConfig()
    guard_slices = int(
        math.ceil(
            float(AxialTruncationConfig().guard_depth_um)
            / float(pixel_depth_um)
        )
    )
    position_dapi_projections = tuple(
        np.asarray(projections[0]) for projections in position_projections
    )
    if any(
        projection.shape != masks[index].shape
        for index, projection in enumerate(position_dapi_projections)
    ):
        raise ValueError("Cached DAPI projection geometry does not match Whole")
    dapi_plane_bounds = _dapi_plane_normalization_bounds(
        dapi,
        max_workers=max(1, int(max_workers)),
    )
    position_inventories = _build_position_dapi_axial_inventories(
        masks=masks,
        z_ranges=z_ranges,
        position_dapi_projections=position_dapi_projections,
        guard_slices=guard_slices,
        dapi_stack=dapi,
        plane_bounds=dapi_plane_bounds,
        pixel_width_um=float(pixel_width_um),
        pixel_height_um=float(pixel_height_um),
        pixel_depth_um=float(pixel_depth_um),
        compartment_config=compartment_config,
        validation_config=cfg,
        max_workers=max(1, int(max_workers)),
    )
    pixel_area_um2 = float(pixel_width_um) * float(pixel_height_um)

    position_arguments: list[dict[str, Any]] = []
    for position_index, (
        (mask, _row, _spec),
        (z0, z1),
        (cached_dapi_projection, cached_structural_projections),
        inventory,
    ) in enumerate(
        zip(
            position_best_candidates,
            z_ranges,
            position_projections,
            position_inventories,
        )
    ):
        extent_core_labels = np.asarray(inventory["extent_core_labels_2d"])
        if extent_core_labels.shape != mask.shape:
            raise RuntimeError("DAPI axial-profile extent geometry mismatch")
        dapi_projection = np.asarray(cached_dapi_projection)
        if set(cached_structural_projections) != set(weights):
            raise ValueError("Cached structural projection channels do not match stacks")
        structural_projections = {
            channel: np.asarray(cached_structural_projections[channel])
            for channel in weights
        }
        if any(
            projection.shape != mask.shape
            for projection in structural_projections.values()
        ):
            raise ValueError(
                "Cached structural projection geometry does not match Whole"
            )
        position_arguments.append(
            {
                "position_index": position_index,
                "mask": mask,
                "z0": z0,
                "z1": z1,
                "axial_runs": tuple(inventory["runs"]),
                "extent_core_labels": extent_core_labels,
                "pixel_area_um2": pixel_area_um2,
                "structural_stacks": structural_stacks,
                "structural_projections": structural_projections,
                "dapi_projection": dapi_projection,
                "weights": weights,
                "pixel_depth_um": float(pixel_depth_um),
            }
        )

    position_worker_count = max(1, min(int(max_workers), len(position_arguments)))
    if position_worker_count == 1:
        position_scores = [
            _score_single_z_position(**arguments)
            for arguments in position_arguments
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=position_worker_count,
            thread_name_prefix="ihc-z-position",
        ) as position_executor:
            position_scores = list(
                position_executor.map(
                    lambda arguments: _score_single_z_position(**arguments),
                    position_arguments,
                )
            )

    status, best_index = _select_unique_highest(position_scores)
    return {
        "status": status,
        "best_index": best_index,
        "positions": tuple(position_scores),
        "selection_inventory": {
            "inventory_basis": "dapi_axial_profile_per_position",
            "guard_slices": guard_slices,
            "available_z_start_0based": available_z0,
            "available_z_end_0based_inclusive": available_z1,
            "calibration_source": str(calibration_source),
            "dapi_plane_normalization": "per_plane_p0.5_p99.8",
            "positions": tuple(
                {
                    "position_index": int(inventory["position_index"]),
                    "main_z_start_0based": int(
                        inventory["main_z_start_0based"]
                    ),
                    "main_z_end_0based_inclusive": int(
                        inventory["main_z_end_0based_inclusive"]
                    ),
                    "profile_z_start_0based": int(
                        inventory["profile_z_start_0based"]
                    ),
                    "profile_z_end_0based_inclusive": int(
                        inventory["profile_z_end_0based_inclusive"]
                    ),
                    "dapi_core_count": int(inventory["core_count"]),
                    "whole_linked_dapi_core_count": int(
                        inventory["whole_linked_core_count"]
                    ),
                    "dapi_axial_run_count": int(inventory["axial_run_count"]),
                    "dapi_metrics": inventory["dapi_metrics"],
                }
                for inventory in position_inventories
            ),
            "measurement_channel_used": False,
        },
    }
