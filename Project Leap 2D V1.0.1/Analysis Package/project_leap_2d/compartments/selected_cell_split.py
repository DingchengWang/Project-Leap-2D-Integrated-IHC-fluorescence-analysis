"""Local, deterministic scientific action for a user-requested cell split.

The function in this module is deliberately independent of Fiji and file I/O.
It receives one synchronized Whole/Soma/Processes label triplet and returns a
complete replacement triplet.  Expected scientific refusals are represented by
``SelectedCellSplitResult.success == False``; malformed caller input raises
``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation


@dataclass(frozen=True)
class SplitNucleusCandidate:
    """One projected 3D-DAPI identity or a local nuclear-model proposal."""

    nucleus_id: int
    projection_mask: np.ndarray = field(repr=False, compare=False)
    dapi_valid: bool = True
    identity_status: str = "resolved"
    owner_astrocyte_id: int | None = None
    accepted: bool = False
    confidence: float = 0.5
    z_min_0based: int | None = None
    z_max_0based: int | None = None
    source: str = "dapi_3d_inventory"
    locally_confirmed: bool = False


@dataclass(frozen=True)
class SelectedCellSplitConfig:
    """Physical and bounded-work rules for a manual, high-recall split."""

    maximum_candidate_distance_um: float = 2.75
    minimum_nucleus_center_separation_um: float = 1.00
    maximum_nucleus_projection_iou: float = 0.32
    maximum_nucleus_overlap_fraction: float = 0.50
    minimum_exclusive_nucleus_fraction: float = 0.20
    minimum_z_supported_exclusive_nucleus_fraction: float = 0.03
    minimum_nucleus_z_center_separation_um: float = 0.70
    maximum_nucleus_z_overlap_fraction: float = 0.35
    minimum_direct_nucleus_projection_area_um2: float = 3.00
    minimum_second_parent_overlap_fraction: float = 0.35
    minimum_second_parent_core_overlap_fraction: float = 0.50
    maximum_whole_growth_distance_um: float = 3.00
    maximum_added_whole_fraction: float = 0.40
    minimum_structural_support: float = 0.45
    structural_closing_um: float = 0.18
    maximum_branch_attachment_um: float = 2.50
    maximum_recovered_branch_half_width_um: float = 1.25
    soma_nuclear_growth_um: float = 0.65
    minimum_child_area_um2: float = 6.0
    minimum_child_fraction: float = 0.06
    minimum_child_process_area_um2: float = 0.25
    minimum_large_external_growth_um2: float = 10.0
    maximum_external_child_fraction_when_process_poor: float = 0.25
    maximum_process_fraction_for_external_growth: float = 0.03
    maximum_crop_pixels: int = 8_000_000


@dataclass(frozen=True)
class SelectedCellSplitResult:
    """Atomic result returned to the Cell Edit transaction layer."""

    success: bool
    reason: str
    whole_labels: np.ndarray = field(repr=False, compare=False)
    soma_labels: np.ndarray = field(repr=False, compare=False)
    process_labels: np.ndarray = field(repr=False, compare=False)
    selected_id: int
    new_id: int | None = None
    owner_nucleus_id: int | None = None
    second_nucleus_id: int | None = None
    added_whole_px: int = 0
    child_areas_px: tuple[int, int] = (0, 0)
    metrics: dict[str, object] = field(default_factory=dict, compare=False)


def _validate_split_inputs(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    structural_evidence: np.ndarray,
    selected_id: int,
    candidates: Sequence[SplitNucleusCandidate],
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
) -> list[int]:
    arrays = (whole_labels, soma_labels, process_labels, structural_evidence)
    if any(np.asarray(array).ndim != 2 for array in arrays):
        raise ValueError("Split inputs must be two-dimensional arrays")
    if len({np.asarray(array).shape for array in arrays}) != 1:
        raise ValueError("Split inputs must have identical shapes")
    if not np.issubdtype(np.asarray(whole_labels).dtype, np.integer):
        raise ValueError("Whole labels must use an integer dtype")
    if not np.issubdtype(np.asarray(soma_labels).dtype, np.integer):
        raise ValueError("Soma labels must use an integer dtype")
    if not np.issubdtype(np.asarray(process_labels).dtype, np.integer):
        raise ValueError("Processes labels must use an integer dtype")
    if not math.isfinite(pixel_width_um) or pixel_width_um <= 0:
        raise ValueError("pixel_width_um must be finite and positive")
    if not math.isfinite(pixel_height_um) or pixel_height_um <= 0:
        raise ValueError("pixel_height_um must be finite and positive")
    if pixel_depth_um is not None and (
        not math.isfinite(pixel_depth_um) or pixel_depth_um <= 0
    ):
        raise ValueError("pixel_depth_um must be finite and positive when provided")

    ids = sorted(int(value) for value in np.unique(whole_labels) if int(value) > 0)
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("Whole Cell IDs must be contiguous before Split")
    if int(selected_id) not in ids:
        raise ValueError("selected_id is not present in Whole labels")
    if np.any((soma_labels > 0) & (soma_labels != whole_labels)):
        raise ValueError("Soma labels are not contained by matching Whole labels")
    if np.any((process_labels > 0) & (process_labels != whole_labels)):
        raise ValueError("Processes labels are not contained by matching Whole labels")
    occupancy = (soma_labels > 0).astype(np.uint8)
    occupancy += (process_labels > 0).astype(np.uint8)
    if np.any(occupancy[whole_labels > 0] != 1) or np.any(
        occupancy[whole_labels == 0] != 0
    ):
        raise ValueError("Whole/Soma/Processes do not form an exact partition")

    seen: set[int] = set()
    for candidate in candidates:
        if int(candidate.nucleus_id) <= 0 or int(candidate.nucleus_id) in seen:
            raise ValueError("Nucleus candidate IDs must be unique and positive")
        seen.add(int(candidate.nucleus_id))
        projection = np.asarray(candidate.projection_mask)
        if projection.shape != whole_labels.shape or projection.ndim != 2:
            raise ValueError("Every nucleus projection must match the label shape")
        if (candidate.z_min_0based is None) != (candidate.z_max_0based is None):
            raise ValueError("Nucleus candidates must provide both Z bounds or neither")
        if (
            candidate.z_min_0based is not None
            and int(candidate.z_min_0based) > int(candidate.z_max_0based)
        ):
            raise ValueError("Nucleus candidate Z bounds are reversed")
    return ids


def _rejected_result(
    reason: str,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    selected_id: int,
    metrics: dict[str, object] | None = None,
) -> SelectedCellSplitResult:
    return SelectedCellSplitResult(
        success=False,
        reason=reason,
        whole_labels=np.array(whole_labels, copy=True),
        soma_labels=np.array(soma_labels, copy=True),
        process_labels=np.array(process_labels, copy=True),
        selected_id=int(selected_id),
        metrics={} if metrics is None else dict(metrics),
    )


def _bbox_with_padding(
    mask: np.ndarray,
    pad_y: int,
    pad_x: int,
) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot build a crop around an empty mask")
    row0 = max(0, int(coordinates[:, 0].min()) - int(pad_y))
    col0 = max(0, int(coordinates[:, 1].min()) - int(pad_x))
    row1 = min(mask.shape[0], int(coordinates[:, 0].max()) + int(pad_y) + 1)
    col1 = min(mask.shape[1], int(coordinates[:, 1].max()) + int(pad_x) + 1)
    return row0, col0, row1, col1


def _candidate_bbox_is_near(
    projection: np.ndarray,
    parent_bbox: tuple[int, int, int, int],
    maximum_distance_um: float,
    pixel_width_um: float,
    pixel_height_um: float,
) -> bool:
    coordinates = np.argwhere(projection)
    if coordinates.size == 0:
        return False
    row0, col0, row1, col1 = parent_bbox
    candidate_row0 = int(coordinates[:, 0].min())
    candidate_col0 = int(coordinates[:, 1].min())
    candidate_row1 = int(coordinates[:, 0].max()) + 1
    candidate_col1 = int(coordinates[:, 1].max()) + 1
    dy_px = max(row0 - candidate_row1, candidate_row0 - row1, 0)
    dx_px = max(col0 - candidate_col1, candidate_col0 - col1, 0)
    distance_um = math.hypot(dy_px * pixel_height_um, dx_px * pixel_width_um)
    return distance_um <= maximum_distance_um


def _physical_center(
    mask: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
) -> tuple[float, float]:
    center_y, center_x = np.argwhere(mask).mean(axis=0)
    return float(center_y * pixel_height_um), float(center_x * pixel_width_um)


def _physical_center_distance(
    left: np.ndarray,
    right: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
) -> float:
    left_y, left_x = _physical_center(left, pixel_width_um, pixel_height_um)
    right_y, right_x = _physical_center(right, pixel_width_um, pixel_height_um)
    return float(math.hypot(left_y - right_y, left_x - right_x))


def _projection_conflict(
    left: np.ndarray,
    right: np.ndarray,
    config: SelectedCellSplitConfig,
) -> tuple[bool, float, float]:
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    smaller = max(min(int(left.sum()), int(right.sum())), 1)
    iou = float(intersection) / max(union, 1)
    overlap_fraction = float(intersection) / smaller
    conflict = (
        iou > config.maximum_nucleus_projection_iou
        or overlap_fraction > config.maximum_nucleus_overlap_fraction
    )
    return conflict, iou, overlap_fraction


def _candidate_parent_support(
    projection: np.ndarray,
    parent: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
) -> tuple[float, bool, float]:
    """Measure whether a projected nucleus genuinely lies in the parent ROI."""

    coordinates = np.argwhere(projection)
    area = max(int(coordinates.shape[0]), 1)
    overlap_fraction = float(
        parent[coordinates[:, 0], coordinates[:, 1]].sum()
    ) / area
    center_y, center_x = coordinates.mean(axis=0)
    center_row = int(np.clip(np.rint(center_y), 0, projection.shape[0] - 1))
    center_col = int(np.clip(np.rint(center_x), 0, projection.shape[1] - 1))
    center_inside = bool(parent[center_row, center_col])

    row0 = max(0, int(coordinates[:, 0].min()) - 1)
    row1 = min(projection.shape[0], int(coordinates[:, 0].max()) + 2)
    col0 = max(0, int(coordinates[:, 1].min()) - 1)
    col1 = min(projection.shape[1], int(coordinates[:, 1].max()) + 2)
    local_projection = projection[row0:row1, col0:col1]
    distance_inside = ndi.distance_transform_edt(
        local_projection,
        sampling=(pixel_height_um, pixel_width_um),
    )
    maximum_distance = float(distance_inside.max(initial=0.0))
    if maximum_distance <= 0:
        core_overlap_fraction = overlap_fraction
    else:
        core = local_projection & (
            distance_inside >= 0.50 * maximum_distance
        )
        local_parent = parent[row0:row1, col0:col1]
        core_overlap_fraction = float((core & local_parent).sum()) / max(
            int(core.sum()),
            1,
        )
    return overlap_fraction, center_inside, core_overlap_fraction


def _exclusive_nucleus_seeds(
    owner_projection: np.ndarray,
    second_projection: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: SelectedCellSplitConfig,
    z_separated: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve limited 2D overlap between two independent 3D identities.

    Each identity must retain a substantial non-overlapping core.  Pixels in
    the shared projection are then assigned to the nearest exclusive core in
    physical XY distance.  Ties deterministically stay with the owner.
    """

    owner = np.asarray(owner_projection, dtype=bool)
    second = np.asarray(second_projection, dtype=bool)
    owner_only = owner & ~second
    second_only = second & ~owner
    owner_area = max(int(owner.sum()), 1)
    second_area = max(int(second.sum()), 1)
    minimum_exclusive_fraction = (
        config.minimum_z_supported_exclusive_nucleus_fraction
        if z_separated
        else config.minimum_exclusive_nucleus_fraction
    )
    if (
        not owner_only.any()
        or not second_only.any()
        or float(owner_only.sum()) / owner_area
        < float(minimum_exclusive_fraction)
        or float(second_only.sum()) / second_area
        < float(minimum_exclusive_fraction)
    ):
        return None

    overlap = owner & second
    if not overlap.any():
        return owner.copy(), second.copy()

    owner_distance = ndi.distance_transform_edt(
        ~owner_only,
        sampling=(pixel_height_um, pixel_width_um),
    )
    second_distance = ndi.distance_transform_edt(
        ~second_only,
        sampling=(pixel_height_um, pixel_width_um),
    )
    owner_seed = owner_only.copy()
    second_seed = second_only.copy()
    owner_seed[overlap & (owner_distance <= second_distance)] = True
    second_seed[overlap & (owner_distance > second_distance)] = True
    if (
        np.any(owner_seed & second_seed)
        or not np.all((owner | second) == (owner_seed | second_seed))
        or not np.all(owner_only <= owner_seed)
        or not np.all(second_only <= second_seed)
    ):
        return None
    return owner_seed, second_seed


def _z_separation_evidence(
    owner: SplitNucleusCandidate,
    second: SplitNucleusCandidate,
    pixel_depth_um: float | None,
    config: SelectedCellSplitConfig,
) -> tuple[bool, float, float]:
    """Return independent-identity support from calibrated 3D Z ranges."""

    if (
        pixel_depth_um is None
        or owner.z_min_0based is None
        or owner.z_max_0based is None
        or second.z_min_0based is None
        or second.z_max_0based is None
    ):
        return False, 0.0, 1.0
    owner_min = int(owner.z_min_0based)
    owner_max = int(owner.z_max_0based)
    second_min = int(second.z_min_0based)
    second_max = int(second.z_max_0based)
    owner_span = owner_max - owner_min + 1
    second_span = second_max - second_min + 1
    overlap_slices = max(
        0,
        min(owner_max, second_max) - max(owner_min, second_min) + 1,
    )
    overlap_fraction = float(overlap_slices) / max(
        min(owner_span, second_span),
        1,
    )
    owner_center = 0.5 * (owner_min + owner_max)
    second_center = 0.5 * (second_min + second_max)
    center_separation_um = (
        abs(owner_center - second_center) * float(pixel_depth_um)
    )
    separated = (
        center_separation_um
        >= float(config.minimum_nucleus_z_center_separation_um)
        and overlap_fraction
        <= float(config.maximum_nucleus_z_overlap_fraction)
    )
    return bool(separated), float(center_separation_um), float(overlap_fraction)


def _minimum_mask_distance_um(
    left: np.ndarray,
    right: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
) -> float:
    """Measure mask separation inside their minimal union crop."""

    union = left | right
    row0, col0, row1, col1 = _bbox_with_padding(union, 1, 1)
    local_left = left[row0:row1, col0:col1]
    local_right = right[row0:row1, col0:col1]
    if np.any(local_left & local_right):
        return 0.0
    distance = ndi.distance_transform_edt(
        ~local_left,
        sampling=(pixel_height_um, pixel_width_um),
    )
    return float(distance[local_right].min(initial=math.inf))


def _normalize_structural_evidence(local_evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(local_evidence, dtype=np.float32)
    finite = np.isfinite(evidence)
    if not finite.any():
        return np.zeros(evidence.shape, dtype=np.float32)
    output = np.zeros(evidence.shape, dtype=np.float32)
    values = evidence[finite]
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.5))
    if high <= low + np.finfo(np.float32).eps:
        output[finite] = (values > low).astype(np.float32)
        if high > 0 and low == high:
            output[finite] = 1.0
        return output
    output[finite] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return output


def _elliptical_footprint(
    radius_um: float,
    pixel_width_um: float,
    pixel_height_um: float,
) -> np.ndarray:
    radius_y = max(1, int(math.ceil(radius_um / pixel_height_um)))
    radius_x = max(1, int(math.ceil(radius_um / pixel_width_um)))
    yy, xx = np.ogrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    return (yy * pixel_height_um) ** 2 + (xx * pixel_width_um) ** 2 <= radius_um**2


def _select_owner_candidate(
    candidates: Sequence[SplitNucleusCandidate],
    selected_id: int,
    parent: np.ndarray,
    original_soma: np.ndarray,
) -> SplitNucleusCandidate | None:
    eligible = [
        candidate
        for candidate in candidates
        if bool(candidate.dapi_valid)
        and np.asarray(candidate.projection_mask, dtype=bool).any()
        and candidate.owner_astrocyte_id in (None, int(selected_id))
        and np.any(
            np.asarray(candidate.projection_mask, dtype=bool) & parent
        )
    ]
    if not eligible:
        return None

    explicitly_owned = [
        candidate
        for candidate in eligible
        if candidate.owner_astrocyte_id == int(selected_id)
    ]
    pool = explicitly_owned if explicitly_owned else eligible

    def owner_key(candidate: SplitNucleusCandidate) -> tuple[float, ...]:
        projection = np.asarray(candidate.projection_mask, dtype=bool)
        area = max(int(projection.sum()), 1)
        soma_overlap = float((projection & original_soma).sum()) / area
        whole_overlap = float((projection & parent).sum()) / area
        return (
            soma_overlap,
            whole_overlap,
            float(bool(candidate.accepted)),
            float(np.clip(candidate.confidence, 0.0, 1.0)),
            -float(int(candidate.nucleus_id)),
        )

    owner = max(pool, key=owner_key)
    projection = np.asarray(owner.projection_mask, dtype=bool)
    if not np.any(projection & parent):
        return None
    return owner


def _select_second_candidate(
    candidates: Sequence[SplitNucleusCandidate],
    owner: SplitNucleusCandidate,
    selected_id: int,
    parent: np.ndarray,
    original_soma: np.ndarray,
    other_whole: np.ndarray,
    parent_bbox: tuple[int, int, int, int],
    pixel_width_um: float,
    pixel_height_um: float,
    config: SelectedCellSplitConfig,
    pixel_depth_um: float | None,
) -> tuple[SplitNucleusCandidate | None, str, dict[str, object]]:
    owner_projection = np.asarray(owner.projection_mask, dtype=bool)
    observed_foreign_owner = False
    observed_separation_conflict = False
    observed_provisional_or_marginal = False
    scored: list[tuple[tuple[float, ...], SplitNucleusCandidate, dict[str, float]]] = []
    for candidate in candidates:
        if int(candidate.nucleus_id) == int(owner.nucleus_id):
            continue
        projection = np.asarray(candidate.projection_mask, dtype=bool)
        if not bool(candidate.dapi_valid) or not projection.any():
            continue
        if not _candidate_bbox_is_near(
            projection,
            parent_bbox,
            config.maximum_candidate_distance_um,
            pixel_width_um,
            pixel_height_um,
        ):
            continue
        parent_distance_um = _minimum_mask_distance_um(
            parent,
            projection,
            pixel_width_um,
            pixel_height_um,
        )
        if parent_distance_um > float(config.maximum_candidate_distance_um):
            continue
        if np.any(projection & other_whole):
            observed_foreign_owner = True
            continue
        if candidate.owner_astrocyte_id not in (None, int(selected_id)):
            observed_foreign_owner = True
            continue

        (
            whole_overlap_fraction,
            center_inside_parent,
            core_overlap_fraction,
        ) = _candidate_parent_support(
            projection,
            parent,
            pixel_width_um,
            pixel_height_um,
        )
        genuinely_inside_parent = (
            whole_overlap_fraction
            >= float(config.minimum_second_parent_overlap_fraction)
            and (
                center_inside_parent
                or core_overlap_fraction
                >= float(config.minimum_second_parent_core_overlap_fraction)
            )
        )
        projected_area_um2 = float(
            int(projection.sum()) * pixel_width_um * pixel_height_um
        )
        needs_local_confirmation = (
            not bool(candidate.accepted)
            or projected_area_um2
            < float(config.minimum_direct_nucleus_projection_area_um2)
        )
        if not genuinely_inside_parent or (
            needs_local_confirmation and not bool(candidate.locally_confirmed)
        ):
            observed_provisional_or_marginal = True
            continue

        center_separation_um = _physical_center_distance(
            owner_projection,
            projection,
            pixel_width_um,
            pixel_height_um,
        )
        z_separated, z_center_separation_um, z_overlap_fraction = (
            _z_separation_evidence(
                owner,
                candidate,
                pixel_depth_um,
                config,
            )
        )
        conflict, iou, overlap_fraction = _projection_conflict(
            owner_projection,
            projection,
            config,
        )
        if (
            (
                center_separation_um
                < config.minimum_nucleus_center_separation_um
                or conflict
            )
            and not z_separated
        ):
            observed_separation_conflict = True
            continue

        area = max(int(projection.sum()), 1)
        soma_overlap_fraction = float((projection & original_soma).sum()) / area
        identity_bonus = {
            "resolved": 1.0,
            "model_proposal": 0.80,
            "raw_dapi": 0.75,
            "ambiguous": 0.55,
        }.get(str(candidate.identity_status).lower(), 0.35)
        score = (
            3.0 * whole_overlap_fraction
            + 1.5 * soma_overlap_fraction
            + 1.0 / (1.0 + parent_distance_um)
            + 0.45 * float(np.clip(candidate.confidence, 0.0, 1.0))
            + 0.20 * identity_bonus
            + 0.08 * float(bool(candidate.accepted))
        )
        diagnostics = {
            "score": float(score),
            "whole_overlap_fraction": whole_overlap_fraction,
            "center_inside_parent": bool(center_inside_parent),
            "core_overlap_fraction": core_overlap_fraction,
            "projected_area_um2": projected_area_um2,
            "locally_confirmed": bool(candidate.locally_confirmed),
            "soma_overlap_fraction": soma_overlap_fraction,
            "parent_distance_um": parent_distance_um,
            "center_separation_um": center_separation_um,
            "projection_iou": iou,
            "projection_overlap_fraction": overlap_fraction,
            "z_separated": bool(z_separated),
            "z_center_separation_um": z_center_separation_um,
            "z_overlap_fraction": z_overlap_fraction,
        }
        deterministic_key = (
            float(score),
            whole_overlap_fraction,
            soma_overlap_fraction,
            float(np.clip(candidate.confidence, 0.0, 1.0)),
            -float(int(candidate.nucleus_id)),
        )
        scored.append((deterministic_key, candidate, diagnostics))

    if scored:
        _, selected, diagnostics = max(scored, key=lambda row: row[0])
        return selected, "", {
            "second_candidate": diagnostics,
            "eligible_second_candidate_count": len(scored),
        }
    if observed_foreign_owner:
        return None, "The second nucleus already belongs to another astrocyte.", {}
    if observed_separation_conflict:
        return None, "The two nuclear candidates cannot be separated.", {}
    if observed_provisional_or_marginal:
        return None, "No additional DAPI nucleus was found.", {}
    return None, "No additional DAPI nucleus was found.", {}


def _assign_unseeded_domain(
    partition: np.ndarray,
    domain: np.ndarray,
    owner_projection: np.ndarray,
    second_projection: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
) -> np.ndarray:
    missing = domain & (partition == 0)
    if not missing.any():
        return partition
    distance_owner = ndi.distance_transform_edt(
        ~owner_projection,
        sampling=(pixel_height_um, pixel_width_um),
    )
    distance_second = ndi.distance_transform_edt(
        ~second_projection,
        sampling=(pixel_height_um, pixel_width_um),
    )
    output = np.array(partition, copy=True)
    output[missing & (distance_owner <= distance_second)] = 1
    output[missing & (distance_owner > distance_second)] = 2
    return output


def _build_child_soma(
    child: np.ndarray,
    original_soma: np.ndarray,
    nucleus_projection: np.ndarray,
    soma_footprint: np.ndarray,
) -> np.ndarray:
    nuclear_zone = morphology.dilation(
        nucleus_projection,
        footprint=soma_footprint,
    )
    soma = child & (original_soma | nuclear_zone | nucleus_projection)
    soma = morphology.closing(soma, footprint=soma_footprint) & child
    soma_labels = measure.label(soma, connectivity=2)
    supported_labels = np.unique(soma_labels[nucleus_projection & child])
    supported_labels = supported_labels[supported_labels > 0]
    soma = np.isin(soma_labels, supported_labels)
    soma = ndi.binary_fill_holes(soma) & child
    soma |= nucleus_projection & child
    return soma.astype(bool)


def _recover_connected_external_branches(
    parent: np.ndarray,
    structural_support: np.ndarray,
    other_whole: np.ndarray,
    distance_from_parent_um: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: SelectedCellSplitConfig,
) -> np.ndarray:
    """Recover only narrow, strongly supported branches rooted at the parent.

    A broad fluorescence halo can be connected to most of the parent boundary
    and therefore must not become new Whole Cell area during a manual Split.
    Each recovered component instead needs a compact attachment to the trusted
    parent and a branch-like physical width.
    """

    eligible = (
        structural_support
        & ~parent
        & ~other_whole
        & (
            distance_from_parent_um
            <= float(config.maximum_whole_growth_distance_um)
        )
    )
    if not eligible.any():
        return np.zeros(parent.shape, dtype=bool)

    parent_neighborhood = morphology.dilation(
        parent,
        footprint=np.ones((3, 3), dtype=bool),
    )
    labels = measure.label(eligible, connectivity=2)
    recovered = np.zeros(parent.shape, dtype=bool)
    component_slices = ndi.find_objects(labels)
    for label_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue

        # Keep per-component work local.  One pixel of real-image context is
        # sufficient for an exact component distance transform because every
        # non-boundary component then has an explicit zero-valued border.  At
        # an image edge the crop remains clipped, matching the former
        # full-image calculation rather than inventing pixels outside the FOV.
        local_slice = tuple(
            slice(
                max(0, int(axis_slice.start) - 1),
                min(int(axis_size), int(axis_slice.stop) + 1),
            )
            for axis_slice, axis_size in zip(component_slice, parent.shape)
        )
        component = labels[local_slice] == label_id
        attachment = component & parent_neighborhood[local_slice]
        if not attachment.any():
            continue

        attachment_coordinates = np.argwhere(attachment)
        attachment_height_um = (
            int(attachment_coordinates[:, 0].max())
            - int(attachment_coordinates[:, 0].min())
            + 1
        ) * float(pixel_height_um)
        attachment_width_um = (
            int(attachment_coordinates[:, 1].max())
            - int(attachment_coordinates[:, 1].min())
            + 1
        ) * float(pixel_width_um)
        attachment_span_um = math.hypot(
            attachment_height_um,
            attachment_width_um,
        )
        if attachment_span_um > float(config.maximum_branch_attachment_um):
            continue

        half_width_um = float(
            ndi.distance_transform_edt(
                component,
                sampling=(pixel_height_um, pixel_width_um),
            ).max(initial=0.0)
        )
        if half_width_um > float(config.maximum_recovered_branch_half_width_um):
            continue
        recovered[local_slice] |= component
    return recovered


def _unsupported_external_child(
    child: np.ndarray,
    child_process: np.ndarray,
    original_parent: np.ndarray,
    pixel_area_um2: float,
    config: SelectedCellSplitConfig,
) -> tuple[bool, dict[str, float | int]]:
    """Detect only the combined large-exterior/near-processless failure mode."""

    child_area_px = max(int(child.sum()), 1)
    external_px = int((child & ~original_parent).sum())
    process_px = int(child_process.sum())
    external_area_um2 = float(external_px * pixel_area_um2)
    external_fraction = float(external_px) / child_area_px
    process_fraction = float(process_px) / child_area_px
    rejected = (
        external_area_um2
        >= float(config.minimum_large_external_growth_um2)
        and external_fraction
        >= float(config.maximum_external_child_fraction_when_process_poor)
        and process_fraction
        <= float(config.maximum_process_fraction_for_external_growth)
    )
    return bool(rejected), {
        "child_area_px": child_area_px,
        "external_area_px": external_px,
        "external_area_um2": external_area_um2,
        "external_fraction": external_fraction,
        "process_area_px": process_px,
        "process_fraction": process_fraction,
    }


def split_selected_cell(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    selected_id: int,
    nucleus_candidates: Sequence[SplitNucleusCandidate],
    structural_evidence: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: SelectedCellSplitConfig | None = None,
    pixel_depth_um: float | None = None,
) -> SelectedCellSplitResult:
    """Split one selected Astrocyte using one owner and one extra DAPI nucleus.

    A manual Split is intentionally high recall: an unaccepted or ambiguous
    DAPI-valid candidate may be used.  A candidate already assigned to another
    current Astrocyte is never used.  The selected parent is repartitioned in a
    bounded local crop, and only nearby, unassigned structural pixels may be
    added.  Pixels owned by other Whole ROIs are immutable barriers.
    """

    active_config = config or SelectedCellSplitConfig()
    ids = _validate_split_inputs(
        whole_labels,
        soma_labels,
        process_labels,
        structural_evidence,
        selected_id,
        nucleus_candidates,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
    )
    parent = np.asarray(whole_labels) == int(selected_id)
    original_soma = np.asarray(soma_labels) == int(selected_id)
    other_whole = (np.asarray(whole_labels) > 0) & ~parent
    parent_bbox = _bbox_with_padding(parent, 0, 0)

    owner = _select_owner_candidate(
        nucleus_candidates,
        int(selected_id),
        parent,
        original_soma,
    )
    if owner is None:
        return _rejected_result(
            "No valid owner nucleus was found.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
        )
    second, refusal, selection_metrics = _select_second_candidate(
        nucleus_candidates,
        owner,
        int(selected_id),
        parent,
        original_soma,
        other_whole,
        parent_bbox,
        pixel_width_um,
        pixel_height_um,
        active_config,
        pixel_depth_um,
    )
    if second is None:
        return _rejected_result(
            refusal,
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            selection_metrics,
        )

    owner_projection_full = np.asarray(owner.projection_mask, dtype=bool)
    second_projection_full = np.asarray(second.projection_mask, dtype=bool)
    seed_union = parent | owner_projection_full | second_projection_full
    pad_y = max(
        2,
        int(
            math.ceil(
                active_config.maximum_whole_growth_distance_um / pixel_height_um
            )
        ),
    )
    pad_x = max(
        2,
        int(
            math.ceil(
                active_config.maximum_whole_growth_distance_um / pixel_width_um
            )
        ),
    )
    row0, col0, row1, col1 = _bbox_with_padding(seed_union, pad_y, pad_x)
    crop_pixels = int((row1 - row0) * (col1 - col0))
    if crop_pixels > int(active_config.maximum_crop_pixels):
        return _rejected_result(
            "The selected ROI is too large for a safe local split.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            {"crop_pixels": crop_pixels},
        )

    crop = np.s_[row0:row1, col0:col1]
    local_parent = parent[crop]
    local_other = other_whole[crop]
    local_owner_projection = owner_projection_full[crop] & ~local_other
    local_second_projection = second_projection_full[crop] & ~local_other
    z_separated, _, _ = _z_separation_evidence(
        owner,
        second,
        pixel_depth_um,
        active_config,
    )
    exclusive_seeds = _exclusive_nucleus_seeds(
        local_owner_projection,
        local_second_projection,
        pixel_width_um,
        pixel_height_um,
        active_config,
        z_separated=z_separated,
    )
    if exclusive_seeds is None:
        return _rejected_result(
            "The two nuclear candidates cannot be separated.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
        )
    local_owner, local_second = exclusive_seeds
    local_structural = _normalize_structural_evidence(
        np.asarray(structural_evidence)[crop]
    )
    distance_from_parent = ndi.distance_transform_edt(
        ~local_parent,
        sampling=(pixel_height_um, pixel_width_um),
    )
    structural_support = (
        local_structural >= float(active_config.minimum_structural_support)
    )
    if active_config.structural_closing_um > 0:
        structural_support = morphology.closing(
            structural_support,
            footprint=_elliptical_footprint(
                active_config.structural_closing_um,
                pixel_width_um,
                pixel_height_um,
            ),
        )
    local_growth = _recover_connected_external_branches(
        local_parent,
        structural_support,
        local_other,
        distance_from_parent,
        pixel_width_um,
        pixel_height_um,
        active_config,
    )
    soma_footprint = _elliptical_footprint(
        active_config.soma_nuclear_growth_um,
        pixel_width_um,
        pixel_height_um,
    )
    mandatory_soma_zone = (
        morphology.dilation(
            local_owner | local_second,
            footprint=soma_footprint,
        )
        & ~local_other
    )
    domain = (
        local_parent
        | local_growth
        | mandatory_soma_zone
        | local_owner
        | local_second
    ) & ~local_other

    # Retain only connected domain components supported by the original parent
    # or by one of the two DAPI seeds.  This removes isolated structural noise.
    domain_labels = measure.label(domain, connectivity=2)
    supported_labels = np.unique(
        domain_labels[local_parent | local_owner | local_second]
    )
    supported_labels = supported_labels[supported_labels > 0]
    domain = np.isin(domain_labels, supported_labels)

    if not local_owner.any() or not local_second.any():
        return _rejected_result(
            "The two nuclear candidates cannot be separated.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
        )
    markers = np.zeros(domain.shape, dtype=np.uint8)
    markers[local_owner] = 1
    markers[local_second] = 2
    distance_inside = ndi.distance_transform_edt(
        domain,
        sampling=(pixel_height_um, pixel_width_um),
    )
    distance_scale = max(
        float(np.percentile(distance_inside[domain], 99.0)),
        min(pixel_width_um, pixel_height_um),
    )
    core_support = np.clip(distance_inside / distance_scale, 0.0, 1.0)
    elevation = (
        0.58 * (1.0 - local_structural)
        + 0.42 * (1.0 - core_support)
    ).astype(np.float32)
    partition = segmentation.watershed(
        elevation,
        markers=markers,
        mask=domain,
        watershed_line=False,
        connectivity=np.ones((3, 3), dtype=bool),
    ).astype(np.uint8)
    partition = _assign_unseeded_domain(
        partition,
        domain,
        local_owner,
        local_second,
        pixel_width_um,
        pixel_height_um,
    )
    child_owner = partition == 1
    child_second = partition == 2
    child_areas = (int(child_owner.sum()), int(child_second.sum()))
    pixel_area_um2 = float(pixel_width_um * pixel_height_um)
    minimum_child_px = max(
        1,
        int(math.ceil(active_config.minimum_child_area_um2 / pixel_area_um2)),
        int(math.ceil(active_config.minimum_child_fraction * int(domain.sum()))),
    )
    if min(child_areas) < minimum_child_px:
        return _rejected_result(
            "The split would create an invalid child cell.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            {
                **selection_metrics,
                "child_areas_px": child_areas,
                "minimum_child_area_px": minimum_child_px,
            },
        )
    added_whole_px = int(
        ((child_owner | child_second) & ~local_parent).sum()
    )
    maximum_added_whole_px = int(
        math.floor(
            float(active_config.maximum_added_whole_fraction)
            * int(parent.sum())
        )
    )
    if added_whole_px > maximum_added_whole_px:
        return _rejected_result(
            "Split recovery exceeded the safe Whole Cell expansion limit.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            {
                **selection_metrics,
                "added_whole_px": added_whole_px,
                "maximum_added_whole_px": maximum_added_whole_px,
                "parent_whole_px": int(parent.sum()),
            },
        )
    if not np.all(child_owner[local_owner]) or not np.all(
        child_second[local_second]
    ):
        return _rejected_result(
            "The split would create an invalid child cell.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            selection_metrics,
        )

    local_original_soma = original_soma[crop]
    owner_soma = _build_child_soma(
        child_owner,
        local_original_soma,
        local_owner,
        soma_footprint,
    )
    second_soma = _build_child_soma(
        child_second,
        local_original_soma,
        local_second,
        soma_footprint,
    )
    if not owner_soma.any() or not second_soma.any():
        return _rejected_result(
            "The split would create an invalid child cell.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            selection_metrics,
        )

    owner_process = child_owner & ~owner_soma
    second_process = child_second & ~second_soma
    owner_external_failure, owner_external_metrics = _unsupported_external_child(
        child_owner,
        owner_process,
        local_parent,
        pixel_area_um2,
        active_config,
    )
    second_external_failure, second_external_metrics = (
        _unsupported_external_child(
            child_second,
            second_process,
            local_parent,
            pixel_area_um2,
            active_config,
        )
    )
    if owner_external_failure or second_external_failure:
        return _rejected_result(
            "Split recovery produced an unsupported external child cell.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            {
                **selection_metrics,
                "owner_external_child": owner_external_metrics,
                "second_external_child": second_external_metrics,
            },
        )
    minimum_child_process_px = max(
        1,
        int(
            math.ceil(
                float(active_config.minimum_child_process_area_um2)
                / pixel_area_um2
            )
        ),
    )
    local_original_process = (
        np.asarray(process_labels)[crop] == int(selected_id)
    )
    process_evidence = local_original_process | structural_support
    owner_supported_process_px = int((owner_process & process_evidence).sum())
    second_supported_process_px = int((second_process & process_evidence).sum())
    if (
        int(owner_process.sum()) < minimum_child_process_px
        or int(second_process.sum()) < minimum_child_process_px
        or owner_supported_process_px < minimum_child_process_px
        or second_supported_process_px < minimum_child_process_px
    ):
        return _rejected_result(
            "Split could not establish real Processes for both child cells.",
            whole_labels,
            soma_labels,
            process_labels,
            selected_id,
            {
                **selection_metrics,
                "child_process_areas_px": (
                    int(owner_process.sum()),
                    int(second_process.sum()),
                ),
                "child_supported_process_areas_px": (
                    owner_supported_process_px,
                    second_supported_process_px,
                ),
                "minimum_child_process_area_px": minimum_child_process_px,
            },
        )

    new_id = len(ids) + 1
    output_dtype = np.promote_types(
        np.asarray(whole_labels).dtype,
        np.min_scalar_type(new_id),
    )
    output_whole = np.asarray(whole_labels, dtype=output_dtype).copy()
    output_soma = np.asarray(soma_labels, dtype=output_dtype).copy()
    output_process = np.asarray(process_labels, dtype=output_dtype).copy()
    output_whole[parent] = 0
    output_soma[original_soma] = 0
    output_process[np.asarray(process_labels) == int(selected_id)] = 0

    whole_view = output_whole[crop]
    soma_view = output_soma[crop]
    process_view = output_process[crop]
    whole_view[child_owner] = int(selected_id)
    whole_view[child_second] = int(new_id)
    soma_view[owner_soma] = int(selected_id)
    soma_view[second_soma] = int(new_id)
    process_view[owner_process] = int(selected_id)
    process_view[second_process] = int(new_id)

    if np.any((output_soma > 0) & (output_soma != output_whole)):
        raise RuntimeError("Split produced Soma outside its matching Whole")
    if np.any((output_process > 0) & (output_process != output_whole)):
        raise RuntimeError("Split produced Processes outside its matching Whole")
    occupancy = (output_soma > 0).astype(np.uint8)
    occupancy += (output_process > 0).astype(np.uint8)
    if np.any(occupancy[output_whole > 0] != 1) or np.any(
        occupancy[output_whole == 0] != 0
    ):
        raise RuntimeError("Split did not preserve the compartment partition")
    if not np.array_equal(output_whole[other_whole], np.asarray(whole_labels)[other_whole]):
        raise RuntimeError("Split changed a non-selected Whole Cell")
    if int(output_whole.max()) != len(ids) + 1:
        raise RuntimeError("Split did not add exactly one Astrocyte ID")

    added_whole = (output_whole > 0) & ~(np.asarray(whole_labels) > 0)
    metrics = {
        **selection_metrics,
        "crop_bounds_yx_0based": (row0, col0, row1, col1),
        "crop_pixels": crop_pixels,
        "pre_roi_count": len(ids),
        "post_roi_count": len(ids) + 1,
        "owner_identity_status": str(owner.identity_status),
        "second_identity_status": str(second.identity_status),
        "second_candidate_source": str(second.source),
        "added_whole_px": int(added_whole.sum()),
        "child_areas_px": child_areas,
        "child_soma_areas_px": (
            int(owner_soma.sum()),
            int(second_soma.sum()),
        ),
        "child_process_areas_px": (
            int(owner_process.sum()),
            int(second_process.sum()),
        ),
    }
    return SelectedCellSplitResult(
        success=True,
        reason="Split completed.",
        whole_labels=output_whole,
        soma_labels=output_soma,
        process_labels=output_process,
        selected_id=int(selected_id),
        new_id=int(new_id),
        owner_nucleus_id=int(owner.nucleus_id),
        second_nucleus_id=int(second.nucleus_id),
        added_whole_px=int(added_whole.sum()),
        child_areas_px=child_areas,
        metrics=metrics,
    )
