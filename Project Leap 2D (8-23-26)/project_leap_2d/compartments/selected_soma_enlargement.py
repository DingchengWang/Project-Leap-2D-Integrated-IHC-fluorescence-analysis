"""Local, calibration-aware manual Soma enlargement.

This module is intentionally independent of the automatic Soma pipeline.  It
computes a proposal for one selected cell and returns synchronized Whole, Soma,
and Processes label maps without mutating any input array.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class SelectedSomaEnlargementConfig:
    """Physical and safety limits for one user-requested enlargement."""

    nucleus_shell_um: float = 0.75
    maximum_supported_radius_um: float = 1.25
    structural_smoothing_um: float = 0.22
    structural_closing_um: float = 0.30
    owner_island_bridge_um: float = 0.35
    background_inner_radius_um: float = 2.80
    background_outer_radius_um: float = 4.80
    structural_support_fraction: float = 0.18
    foreign_nucleus_clearance_um: float = 0.45
    other_cell_clearance_um: float = 0.15
    edge_guard_um: float = 0.15
    minimum_owner_overlap_um2: float = 0.30
    maximum_added_fraction_of_existing_soma: float = 4.00
    maximum_added_area_um2: float = 350.0
    maximum_final_soma_area_um2: float = 650.0
    minimum_background_pixels: int = 32
    maximum_local_crop_pixels: int = 4_000_000


def selected_soma_enlargement_config_for_mode(
    analysis_mode: str,
) -> SelectedSomaEnlargementConfig:
    """Return the isolated manual-Enlarge policy for one analysis route.

    The established eGFP route deliberately returns the unchanged default
    configuration.  GFAP-only uses the full owner DAPI nucleus plus a calibrated
    nuclear shell as its primary target; GFAP can support only a narrow
    additional outer rim.
    """

    mode = str(analysis_mode).strip().lower()
    if mode == "egfp":
        return SelectedSomaEnlargementConfig()
    if mode == "gfap_only":
        return SelectedSomaEnlargementConfig(
            nucleus_shell_um=1.00,
            maximum_supported_radius_um=1.25,
            structural_support_fraction=0.55,
        )
    raise ValueError(f"Unsupported Soma Enlarge analysis mode: {analysis_mode!r}")


@dataclass(frozen=True)
class SelectedSomaEnlargementResult:
    """Atomic result returned to the Fiji cell-edit transaction layer."""

    approved: bool
    changed: bool
    status: str
    message: str
    whole_labels: np.ndarray
    soma_labels: np.ndarray
    process_labels: np.ndarray
    added_soma_mask: np.ndarray
    added_whole_mask: np.ndarray
    metrics: dict


def _copy_result(
    *,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    status: str,
    message: str,
    metrics: dict,
) -> SelectedSomaEnlargementResult:
    shape = whole_labels.shape
    return SelectedSomaEnlargementResult(
        approved=False,
        changed=False,
        status=status,
        message=message,
        whole_labels=whole_labels,
        soma_labels=soma_labels,
        process_labels=process_labels,
        added_soma_mask=np.zeros(shape, dtype=bool),
        added_whole_mask=np.zeros(shape, dtype=bool),
        metrics=metrics,
    )


def _validate_compartment_partition(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
) -> None:
    if (
        whole_labels.ndim != 2
        or soma_labels.ndim != 2
        or process_labels.ndim != 2
    ):
        raise ValueError("Whole, Soma, and Processes labels must be 2D")
    if (
        whole_labels.shape != soma_labels.shape
        or whole_labels.shape != process_labels.shape
    ):
        raise ValueError("Whole, Soma, and Processes label shapes do not match")
    for name, labels in (
        ("Whole", whole_labels),
        ("Soma", soma_labels),
        ("Processes", process_labels),
    ):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name} labels must use an integer dtype")
        if np.any(labels < 0):
            raise ValueError(f"{name} labels contain a negative ID")
    if np.any((soma_labels > 0) & (process_labels > 0)):
        raise ValueError("Soma and Processes overlap")
    if not np.array_equal(
        whole_labels > 0,
        (soma_labels > 0) | (process_labels > 0),
    ):
        raise ValueError("Whole is not the exact union of Soma and Processes")
    if not np.array_equal(
        soma_labels[soma_labels > 0],
        whole_labels[soma_labels > 0],
    ):
        raise ValueError("A Soma pixel has a different Whole ID")
    if not np.array_equal(
        process_labels[process_labels > 0],
        whole_labels[process_labels > 0],
    ):
        raise ValueError("A Processes pixel has a different Whole ID")


def _elliptical_structure(
    radius_um: float,
    pixel_height_um: float,
    pixel_width_um: float,
) -> np.ndarray:
    y_radius = max(1, int(math.ceil(radius_um / pixel_height_um)))
    x_radius = max(1, int(math.ceil(radius_um / pixel_width_um)))
    yy, xx = np.ogrid[-y_radius : y_radius + 1, -x_radius : x_radius + 1]
    return (
        (yy * pixel_height_um) ** 2
        + (xx * pixel_width_um) ** 2
        <= radius_um**2
    )


def enlarge_selected_soma(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    selected_id: int,
    owner_nucleus_mask: np.ndarray,
    structural_image: np.ndarray,
    foreign_nucleus_mask: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: SelectedSomaEnlargementConfig | None = None,
) -> SelectedSomaEnlargementResult:
    """Enlarge one Soma around its single owner nucleus.

    Accepted Soma pixels outside the old Whole are added to both Soma and
    Whole.  Processes are then recomputed exactly as Whole minus Soma.  No
    pixel belonging to another cell can change.
    """

    cfg = config or SelectedSomaEnlargementConfig()
    _validate_compartment_partition(whole_labels, soma_labels, process_labels)
    if selected_id <= 0:
        raise ValueError("selected_id must be a positive integer")
    if (
        not math.isfinite(pixel_width_um)
        or not math.isfinite(pixel_height_um)
        or pixel_width_um <= 0
        or pixel_height_um <= 0
    ):
        raise ValueError("Pixel calibration must contain positive finite values")
    for name, value in (
        ("nucleus_shell_um", cfg.nucleus_shell_um),
        ("maximum_supported_radius_um", cfg.maximum_supported_radius_um),
        ("maximum_added_area_um2", cfg.maximum_added_area_um2),
        ("maximum_final_soma_area_um2", cfg.maximum_final_soma_area_um2),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if cfg.maximum_supported_radius_um < cfg.nucleus_shell_um:
        raise ValueError(
            "maximum_supported_radius_um must be at least nucleus_shell_um"
        )

    if cfg.maximum_local_crop_pixels <= 0:
        raise ValueError("maximum_local_crop_pixels must be positive")

    shape = whole_labels.shape
    owner = np.asarray(owner_nucleus_mask, dtype=bool)
    foreign = np.asarray(foreign_nucleus_mask, dtype=bool)
    structural_source = np.asarray(structural_image)
    if (
        owner.shape != shape
        or foreign.shape != shape
        or structural_source.shape != shape
    ):
        raise ValueError(
            "Owner nucleus, foreign nucleus, and structural image shapes "
            "must match the compartment labels"
        )
    if not np.issubdtype(structural_source.dtype, np.number):
        raise ValueError("Structural image must use a numeric dtype")

    selected_whole = whole_labels == selected_id
    old_soma = soma_labels == selected_id
    pixel_area_um2 = float(pixel_width_um * pixel_height_um)
    selected_whole_area_px = int(selected_whole.sum())
    old_soma_area_px = int(old_soma.sum())
    metrics = {
        "selected_id": int(selected_id),
        "config": asdict(cfg),
        "pixel_size_um": [float(pixel_height_um), float(pixel_width_um)],
        "old_whole_area_px": selected_whole_area_px,
        "old_soma_area_px": old_soma_area_px,
        "owner_nucleus_area_px": int(owner.sum()),
        "added_soma_px": 0,
        "added_whole_px": 0,
        "new_whole_area_px": selected_whole_area_px,
        "new_soma_area_px": old_soma_area_px,
    }

    if not selected_whole.any():
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_selected_cell_missing",
            message="The selected Astrocyte does not exist.",
            metrics=metrics,
        )
    if not old_soma.any():
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_selected_soma_missing",
            message="The selected Astrocyte has no Soma ROI.",
            metrics=metrics,
        )

    owner_coordinates = np.argwhere(owner)
    if owner_coordinates.size == 0:
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_owner_nucleus_missing",
            message="The selected Astrocyte has no owner nucleus.",
            metrics=metrics,
        )

    height, width = shape
    edge_y_px = max(1, int(math.ceil(cfg.edge_guard_um / pixel_height_um)))
    edge_x_px = max(1, int(math.ceil(cfg.edge_guard_um / pixel_width_um)))
    owner_y_min = int(owner_coordinates[:, 0].min())
    owner_y_max = int(owner_coordinates[:, 0].max())
    owner_x_min = int(owner_coordinates[:, 1].min())
    owner_x_max = int(owner_coordinates[:, 1].max())
    if (
        owner_y_min < edge_y_px
        or owner_y_max >= height - edge_y_px
        or owner_x_min < edge_x_px
        or owner_x_max >= width - edge_x_px
    ):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_owner_image_edge",
            message="The owner nucleus reaches the image edge.",
            metrics=metrics,
        )

    calculation_margin_um = max(
        cfg.background_outer_radius_um,
        cfg.maximum_supported_radius_um
        + max(
            cfg.foreign_nucleus_clearance_um,
            cfg.other_cell_clearance_um,
            cfg.structural_closing_um,
            cfg.owner_island_bridge_um,
            4.0 * cfg.structural_smoothing_um,
        ),
    )
    margin_y_px = max(
        1,
        int(math.ceil(calculation_margin_um / pixel_height_um)),
    )
    margin_x_px = max(
        1,
        int(math.ceil(calculation_margin_um / pixel_width_um)),
    )
    y0 = max(0, owner_y_min - margin_y_px)
    y1 = min(height, owner_y_max + margin_y_px + 1)
    x0 = max(0, owner_x_min - margin_x_px)
    x1 = min(width, owner_x_max + margin_x_px + 1)
    crop_pixels = int((y1 - y0) * (x1 - x0))
    metrics["local_crop_bounds_yx"] = [y0, y1, x0, x1]
    metrics["local_crop_shape_yx"] = [y1 - y0, x1 - x0]
    metrics["local_crop_pixels"] = crop_pixels
    metrics["full_image_pixels"] = int(height * width)
    if crop_pixels > cfg.maximum_local_crop_pixels:
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_local_crop_limit",
            message="The local Soma calculation exceeds its memory limit.",
            metrics=metrics,
        )

    crop = np.s_[y0:y1, x0:x1]
    owner_local = owner[crop]
    foreign_local = foreign[crop]
    whole_local = whole_labels[crop]
    soma_local = soma_labels[crop]
    selected_whole_local = whole_local == selected_id
    old_soma_local = soma_local == selected_id
    other_whole_local = (whole_local > 0) & ~selected_whole_local
    structural = np.asarray(structural_source[crop], dtype=np.float64)
    if not np.all(np.isfinite(structural)):
        raise ValueError("Structural image contains non-finite values in the local crop")

    connectivity = np.ones((3, 3), dtype=bool)
    _, owner_component_count = ndi.label(
        owner_local,
        structure=connectivity,
    )
    owner_seed = owner_local.copy()
    if cfg.owner_island_bridge_um > 0:
        owner_seed |= ndi.binary_closing(
            owner_local,
            structure=_elliptical_structure(
                cfg.owner_island_bridge_um,
                pixel_height_um,
                pixel_width_um,
            ),
        )
    _, bridged_component_count = ndi.label(
        owner_seed,
        structure=connectivity,
    )
    metrics["owner_projection_component_count"] = int(owner_component_count)
    metrics["owner_bridged_component_count"] = int(bridged_component_count)

    if np.any(owner_local & foreign_local):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_owner_foreign_nucleus_overlap",
            message="The owner nucleus conflicts with another nucleus.",
            metrics=metrics,
        )
    if np.any(owner_local & other_whole_local):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_owner_other_cell_overlap",
            message="The owner nucleus overlaps another Astrocyte.",
            metrics=metrics,
        )
    owner_overlap_px = int((owner_local & old_soma_local).sum())
    minimum_owner_overlap_px = max(
        1,
        int(math.ceil(cfg.minimum_owner_overlap_um2 / pixel_area_um2)),
    )
    metrics["owner_soma_overlap_px"] = owner_overlap_px
    metrics["minimum_owner_soma_overlap_px"] = minimum_owner_overlap_px
    if owner_overlap_px < minimum_owner_overlap_px:
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_owner_soma_anchor_missing",
            message="The selected Soma is not anchored to its owner nucleus.",
            metrics=metrics,
        )
    if np.any(old_soma_local & foreign_local):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_foreign_nucleus_in_soma",
            message="A foreign nucleus intersects the selected Soma.",
            metrics=metrics,
        )

    sampling = (float(pixel_height_um), float(pixel_width_um))
    distance_to_owner = ndi.distance_transform_edt(
        ~owner_seed,
        sampling=sampling,
    )
    distance_to_foreign = (
        ndi.distance_transform_edt(~foreign_local, sampling=sampling)
        if foreign_local.any()
        else np.full(owner_local.shape, np.inf, dtype=np.float64)
    )
    distance_to_other = (
        ndi.distance_transform_edt(~other_whole_local, sampling=sampling)
        if other_whole_local.any()
        else np.full(owner_local.shape, np.inf, dtype=np.float64)
    )
    safe = (
        (distance_to_foreign > cfg.foreign_nucleus_clearance_um)
        & (distance_to_other > cfg.other_cell_clearance_um)
        & ~other_whole_local
        & ~foreign_local
    )
    if not np.all(safe[owner_local]):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_owner_clearance_conflict",
            message="The owner nucleus is too close to another cell or nucleus.",
            metrics=metrics,
        )

    sigma = (
        max(0.0, cfg.structural_smoothing_um / pixel_height_um),
        max(0.0, cfg.structural_smoothing_um / pixel_width_um),
    )
    smoothed = ndi.gaussian_filter(structural, sigma=sigma, mode="nearest")
    supported_domain = (
        safe & (distance_to_owner <= cfg.maximum_supported_radius_um)
    )
    near = supported_domain & (
        distance_to_owner <= max(cfg.nucleus_shell_um * 2.0, 1.20)
    )
    background = (
        safe
        & (distance_to_owner >= cfg.background_inner_radius_um)
        & (distance_to_owner <= cfg.background_outer_radius_um)
    )
    if int(background.sum()) < cfg.minimum_background_pixels:
        background = safe & (
            distance_to_owner > cfg.nucleus_shell_um
        )
    background_values = smoothed[background]
    near_values = smoothed[near]
    if background_values.size:
        background_level = float(np.percentile(background_values, 70.0))
    else:
        background_level = float(np.percentile(smoothed[safe], 50.0))
    if near_values.size:
        near_high = float(np.percentile(near_values, 85.0))
    else:
        near_high = background_level
    dynamic_range = max(near_high - background_level, 0.0)
    structural_low, structural_high = np.percentile(
        smoothed,
        (1.0, 99.0),
    )
    image_dynamic_range = max(float(structural_high - structural_low), 0.0)
    minimum_detectable_contrast = max(
        np.finfo(np.float64).eps
        * max(abs(near_high), abs(background_level), 1.0)
        * 32.0,
        image_dynamic_range * 0.02,
    )
    has_structural_contrast = dynamic_range >= minimum_detectable_contrast
    structural_threshold = (
        background_level + cfg.structural_support_fraction * dynamic_range
    )
    metrics["structural_background_level"] = background_level
    metrics["structural_near_high"] = near_high
    metrics["structural_threshold"] = structural_threshold
    metrics["structural_contrast_detected"] = bool(has_structural_contrast)

    mandatory_shell = safe & (distance_to_owner <= cfg.nucleus_shell_um)
    supported_shell = (
        supported_domain & (smoothed >= structural_threshold)
        if has_structural_contrast
        else np.zeros(owner_local.shape, dtype=bool)
    )
    proposal_mask = owner_seed | mandatory_shell | supported_shell
    if cfg.structural_closing_um > 0:
        proposal_mask = ndi.binary_closing(
            proposal_mask,
            structure=_elliptical_structure(
                cfg.structural_closing_um,
                pixel_height_um,
                pixel_width_um,
            ),
        )
        proposal_mask &= safe & (
            distance_to_owner <= cfg.maximum_supported_radius_um
        )
        proposal_mask |= owner_seed
    proposal_mask = ndi.binary_propagation(
        owner_seed,
        structure=connectivity,
        mask=proposal_mask,
    )
    proposal_mask = ndi.binary_fill_holes(proposal_mask) & safe
    target_soma_local = old_soma_local | proposal_mask
    added_soma_local = target_soma_local & ~old_soma_local

    if not added_soma_local.any():
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="no_change",
            message="No additional validated Soma pixels were found.",
            metrics=metrics,
        )

    added_y, added_x = np.nonzero(added_soma_local)
    global_added_y = added_y + y0
    global_added_x = added_x + x0
    if np.any(
        (global_added_y < edge_y_px)
        | (global_added_y >= height - edge_y_px)
        | (global_added_x < edge_x_px)
        | (global_added_x >= width - edge_x_px)
    ):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_image_edge",
            message="The enlargement reaches the image edge.",
            metrics=metrics,
        )

    added_px = int(added_soma_local.sum())
    old_soma_px = old_soma_area_px
    final_soma_px = old_soma_px + added_px
    added_area_um2 = added_px * pixel_area_um2
    final_area_um2 = final_soma_px * pixel_area_um2
    added_fraction = added_px / max(old_soma_px, 1)
    metrics.update(
        {
            "candidate_added_soma_px": added_px,
            "candidate_added_area_um2": added_area_um2,
            "candidate_added_fraction": added_fraction,
            "candidate_final_soma_area_um2": final_area_um2,
        }
    )
    if (
        added_fraction > cfg.maximum_added_fraction_of_existing_soma
        or added_area_um2 > cfg.maximum_added_area_um2
        or final_area_um2 > cfg.maximum_final_soma_area_um2
    ):
        return _copy_result(
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            status="rejected_size_limit",
            message="The proposed Soma enlargement exceeds its safety limit.",
            metrics=metrics,
        )

    output_whole = whole_labels.copy()
    output_soma = soma_labels.copy()
    output_whole_local = output_whole[crop]
    output_soma_local = output_soma[crop]
    output_whole_local[added_soma_local] = selected_id
    output_soma_local[added_soma_local] = selected_id
    output_process = process_labels.copy()
    output_process[crop] = np.where(
        (output_whole_local > 0) & (output_soma_local == 0),
        output_whole_local,
        0,
    ).astype(process_labels.dtype, copy=False)
    added_whole_local = added_soma_local & ~selected_whole_local
    added_soma = np.zeros(shape, dtype=bool)
    added_soma[crop] = added_soma_local
    added_whole = np.zeros(shape, dtype=bool)
    added_whole[crop] = added_whole_local

    if np.any(
        output_whole_local[other_whole_local]
        != whole_local[other_whole_local]
    ):
        raise RuntimeError("Enlarge changed another Astrocyte Whole")
    for labels, original, name in (
        (output_soma[crop], soma_local, "Soma"),
        (output_process[crop], process_labels[crop], "Processes"),
    ):
        other_ids = original > 0
        other_ids &= original != selected_id
        if np.any(labels[other_ids] != original[other_ids]):
            raise RuntimeError(f"Enlarge changed another Astrocyte {name}")
    if not np.all(output_soma[old_soma] == selected_id):
        raise RuntimeError("Enlarge removed an existing Soma pixel")
    if not np.array_equal(
        added_whole_local,
        added_soma_local & ~selected_whole_local,
    ):
        raise RuntimeError("Whole expansion differs from approved Soma expansion")
    _validate_compartment_partition(output_whole, output_soma, output_process)

    metrics.update(
        {
            "added_soma_px": added_px,
            "added_whole_px": int(added_whole.sum()),
            "new_whole_area_px": (
                selected_whole_area_px + int(added_whole_local.sum())
            ),
            "new_soma_area_px": final_soma_px,
        }
    )
    return SelectedSomaEnlargementResult(
        approved=True,
        changed=True,
        status="enlarged",
        message="Soma enlarged successfully.",
        whole_labels=output_whole,
        soma_labels=output_soma,
        process_labels=output_process,
        added_soma_mask=added_soma,
        added_whole_mask=added_whole,
        metrics=metrics,
    )
