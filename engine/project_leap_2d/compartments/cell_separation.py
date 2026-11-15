# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def retain_primary_anchor_extent(
    selected_extent: np.ndarray,
    selected_core: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Keep the single DAPI-supported extent body belonging to one Soma anchor."""

    extent = (selected_extent | selected_core) & component
    extent_labels = measure.label(extent, connectivity=2)
    component_count = int(extent_labels.max())
    if component_count <= 1:
        return extent.astype(bool), 0, 0

    areas = np.bincount(
        extent_labels.ravel(),
        minlength=component_count + 1,
    )
    core_counts = np.bincount(
        extent_labels[selected_core & component],
        minlength=component_count + 1,
    )
    primary_label = max(
        range(1, component_count + 1),
        key=lambda label_id: (int(core_counts[label_id]), int(areas[label_id])),
    )
    retained = extent_labels == primary_label
    removed_px = int(extent.sum() - retained.sum())
    return retained, component_count - 1, removed_px

def score_nuclei_for_component(
    component: np.ndarray,
    nearest_nucleus_labels: np.ndarray,
    nucleus_distance: np.ndarray,
    distance: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    link_radius_px: int,
    ambiguity_delta: float,
) -> tuple[list[dict], bool]:
    association_zone = component & (nucleus_distance <= link_radius_px)
    candidate_ids = np.unique(nearest_nucleus_labels[association_zone])
    candidate_ids = candidate_ids[candidate_ids > 0]
    if candidate_ids.size == 0:
        return [], False

    core_reference = max(float(np.percentile(distance[component], 99.0)), 1.0)
    scored: list[dict] = []
    for nucleus_id in candidate_ids:
        near_nucleus = (
            component
            & (nearest_nucleus_labels == int(nucleus_id))
            & (nucleus_distance <= link_radius_px)
        )
        if not near_nucleus.any():
            continue
        thickness_support = min(
            1.0,
            float(np.percentile(distance[near_nucleus], 90.0)) / core_reference,
        )
        structural_support = float(np.percentile(struct[near_nucleus], 75.0))
        overlap_fraction = float(
            (component & (nearest_nucleus_labels == int(nucleus_id)) & (nucleus_distance == 0)).sum()
        ) / max(int(((nearest_nucleus_labels == int(nucleus_id)) & (nucleus_distance == 0)).sum()), 1)
        model_support = float(cellpose_mask[near_nucleus].mean()) if near_nucleus.any() else 0.0
        score = (
            0.46 * thickness_support
            + 0.31 * structural_support
            + 0.18 * min(1.0, overlap_fraction * 2.0)
            + 0.05 * model_support
        )
        nucleus_pixels = (nearest_nucleus_labels == int(nucleus_id)) & (nucleus_distance == 0)
        nucleus_coords = np.argwhere(nucleus_pixels)
        if nucleus_coords.size == 0:
            continue
        center_y, center_x = nucleus_coords.mean(axis=0)
        scored.append(
            {
                "nucleus_id": int(nucleus_id),
                "score": float(score),
                "thickness_support": float(thickness_support),
                "structural_support": float(structural_support),
                "overlap_fraction": float(overlap_fraction),
                "model_support": float(model_support),
                "center_y": float(center_y),
                "center_x": float(center_x),
            }
        )
    if not scored:
        return [], False
    scored.sort(key=lambda row: row["score"], reverse=True)
    ambiguous = len(scored) > 1 and scored[0]["score"] - scored[1]["score"] < ambiguity_delta
    return scored, ambiguous

def select_soma_anchor_groups(
    scored_nuclei: list[dict],
    mean_pixel_um: float,
    config: CompartmentConfig,
) -> list[dict]:
    """Keep spatially distinct, high-confidence soma anchors and merge nearby DAPI fragments."""

    if not scored_nuclei:
        return []
    min_separation_px = config.soma_anchor_min_separation_um / mean_pixel_um
    top_score = float(scored_nuclei[0]["score"])
    groups: list[dict] = []
    for index, candidate in enumerate(scored_nuclei):
        nearest_group = None
        nearest_distance = math.inf
        for group in groups:
            distance_px = math.hypot(
                candidate["center_y"] - group["center_y"],
                candidate["center_x"] - group["center_x"],
            )
            if distance_px < nearest_distance:
                nearest_group = group
                nearest_distance = distance_px
        if nearest_group is not None and nearest_distance < min_separation_px:
            nearest_group["nucleus_ids"].append(candidate["nucleus_id"])
            nearest_group["member_scores"].append(candidate["score"])
            continue

        is_primary = index == 0
        passes_primary_threshold = (
            candidate["score"] >= config.primary_anchor_min_score
            and candidate["thickness_support"] >= config.primary_anchor_min_thickness_support
            and candidate["structural_support"] >= config.primary_anchor_min_structural_support
            and (
                candidate["overlap_fraction"] >= config.primary_anchor_min_overlap_fraction
                or candidate["model_support"] >= config.primary_anchor_min_model_support
            )
        )
        has_local_support = (
            candidate["thickness_support"] >= config.multi_anchor_min_thickness_support
            and candidate["structural_support"] >= config.multi_anchor_min_structural_support
            and (
                candidate["overlap_fraction"] >= config.multi_anchor_min_overlap_fraction
                or candidate["model_support"] >= config.multi_anchor_min_model_support
            )
        )
        passes_secondary_threshold = (
            candidate["score"] >= config.multi_anchor_min_score
            and candidate["score"] >= top_score - config.multi_anchor_max_score_delta
            and has_local_support
        )
        if is_primary and not passes_primary_threshold:
            continue
        if not is_primary and not passes_secondary_threshold:
            continue
        if len(groups) >= config.max_soma_anchors_per_whole_roi:
            continue
        groups.append(
            {
                "nucleus_ids": [candidate["nucleus_id"]],
                "member_scores": [candidate["score"]],
                "score": candidate["score"],
                "center_y": candidate["center_y"],
                "center_x": candidate["center_x"],
            }
        )
    return groups

def select_validated_soma_anchor_groups(
    scored_nuclei: list[dict],
    component: np.ndarray,
    local_nuclei_labels: np.ndarray,
    local_grouped_extent_labels: np.ndarray,
    validated_group_by_id: dict[int, dict],
    minimum_overlap_px: int,
    assigned_group_ids: set[int] | None = None,
) -> list[dict]:
    """Use each independently accepted 3D nucleus group as a required Soma anchor."""

    scored_by_id = {
        int(row["nucleus_id"]): row for row in scored_nuclei
    }
    present_group_ids = np.unique(local_grouped_extent_labels[component])
    groups: list[dict] = []
    for group_id_value in present_group_ids:
        group_id = int(group_id_value)
        if group_id <= 0 or group_id not in validated_group_by_id:
            continue
        if assigned_group_ids is not None and group_id not in assigned_group_ids:
            continue
        group = validated_group_by_id[group_id]
        if not bool(group["accepted"]):
            continue
        extent_overlap = component & (local_grouped_extent_labels == group_id)
        overlap_px = int(extent_overlap.sum())
        if overlap_px < minimum_overlap_px:
            continue
        object_ids = tuple(int(value) for value in group["object_ids"])
        nucleus_ids = [
            object_id
            for object_id in object_ids
            if np.any(local_nuclei_labels == object_id)
        ]
        if not nucleus_ids:
            continue
        members = [
            scored_by_id[object_id]
            for object_id in nucleus_ids
            if object_id in scored_by_id
        ]
        nucleus_pixels = component & np.isin(local_nuclei_labels, nucleus_ids)
        center_pixels = nucleus_pixels if nucleus_pixels.any() else extent_overlap
        center_y, center_x = np.argwhere(center_pixels).mean(axis=0)
        score = max(
            (float(row["score"]) for row in members),
            default=float(group["enclosure_score"]),
        )
        groups.append(
            {
                "nucleus_ids": nucleus_ids,
                "member_scores": [float(row["score"]) for row in members],
                "score": score,
                "center_y": float(center_y),
                "center_x": float(center_x),
                "validated_group_id": group_id,
                "validated_object_ids": list(object_ids),
                "extent_overlap_px": overlap_px,
                "source": "independently_accepted_3d_nucleus_group",
            }
        )
    groups.sort(
        key=lambda row: (
            -int(row["extent_overlap_px"]),
            -float(row["score"]),
            int(row["validated_group_id"]),
        )
    )
    return groups

def restore_low_support_branch_gaps(
    whole_mask: np.ndarray,
    dapi_projection: np.ndarray,
    struct: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: CompartmentConfig,
    nuclei_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Restore dark internal valleys without shrinking edges or breaking connectivity."""

    input_mask = whole_mask.astype(bool)
    empty_metrics = {
        "enabled": bool(config.branch_gap_restore_enabled),
        "input_area_px": int(input_mask.sum()),
        "output_area_px": int(input_mask.sum()),
        "removed_px": 0,
        "removed_fraction": 0.0,
        "accepted_gap_count": 0,
        "rejected_disconnect_count": 0,
        "structural_cut": 0.0,
        "removed_structural_mean": 0.0,
    }
    if not config.branch_gap_restore_enabled or not input_mask.any():
        return input_mask, empty_metrics

    structural_cut = float(
        np.percentile(struct[input_mask], config.branch_gap_low_percentile)
    )
    distance_um = ndi.distance_transform_edt(
        input_mask,
        sampling=(pixel_height_um, pixel_width_um),
    )
    nuclei = (
        nuclei_mask.astype(bool, copy=False)
        if nuclei_mask is not None
        else dapi_nuclei_mask(
            dapi_projection,
            percentile_floor=config.dapi_percentile_floor,
        )
    )
    if nuclei.any():
        nucleus_distance_um = ndi.distance_transform_edt(
            ~nuclei,
            sampling=(pixel_height_um, pixel_width_um),
        )
        nucleus_protection = nucleus_distance_um <= config.branch_gap_nucleus_protect_um
    else:
        nucleus_protection = np.zeros_like(input_mask, dtype=bool)

    gap_candidates = (
        input_mask
        & (distance_um >= config.branch_gap_min_depth_um)
        & (struct < structural_cut)
        & ~nucleus_protection
    )
    component_labels = measure.label(input_mask, connectivity=2)
    output = input_mask.copy()
    accepted_gap_count = 0
    rejected_disconnect_count = 0
    pixel_area_um2 = pixel_width_um * pixel_height_um
    mean_pixel_um = math.sqrt(pixel_area_um2)

    for component_id in range(1, int(component_labels.max()) + 1):
        component = component_labels == component_id
        candidate_labels = measure.label(gap_candidates & component, connectivity=2)
        properties = sorted(
            measure.regionprops(candidate_labels),
            key=lambda item: item.area,
            reverse=True,
        )
        for prop in properties:
            area_um2 = float(prop.area) * pixel_area_um2
            major_axis_um = float(prop.major_axis_length) * mean_pixel_um
            gap_like = (
                area_um2 >= config.branch_gap_min_area_um2
                and (
                    float(prop.eccentricity) >= config.branch_gap_min_eccentricity
                    or major_axis_um >= config.branch_gap_min_major_axis_um
                )
            )
            if not gap_like:
                continue
            gap = candidate_labels == prop.label
            trial_component = output & component & ~gap
            if int(measure.label(trial_component, connectivity=2).max()) != 1:
                rejected_disconnect_count += 1
                continue
            output[gap] = False
            accepted_gap_count += 1

    if int(measure.label(output, connectivity=2).max()) != int(component_labels.max()):
        raise RuntimeError("Branch-gap restoration changed the Whole component count")
    removed = input_mask & ~output
    removed_px = int(removed.sum())
    return output, {
        "enabled": True,
        "input_area_px": int(input_mask.sum()),
        "output_area_px": int(output.sum()),
        "removed_px": removed_px,
        "removed_fraction": round(removed_px / max(int(input_mask.sum()), 1), 6),
        "accepted_gap_count": accepted_gap_count,
        "rejected_disconnect_count": rejected_disconnect_count,
        "structural_cut": round(structural_cut, 6),
        "removed_structural_mean": round(
            float(struct[removed].mean()) if removed.any() else 0.0,
            6,
        ),
    }

def split_touching_whole_instances(
    whole_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    nearest_nucleus_labels: np.ndarray,
    nucleus_distance: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    mean_pixel_um: float,
    pixel_area_um2: float,
    link_radius_px: int,
    config: CompartmentConfig,
) -> tuple[np.ndarray, dict]:
    """Split only high-confidence multi-soma components without changing Whole pixels."""

    base_labels = measure.label(whole_mask, connectivity=2).astype(np.uint16)
    base_count = int(base_labels.max())
    empty_metrics = {
        "base_connected_component_count": base_count,
        "final_instance_count": base_count,
        "split_component_count": 0,
        "split_added_roi_count": 0,
        "split_components": [],
        "split_rejected_count": 0,
    }
    if not config.instance_split_enabled or base_count < 1:
        return base_labels, empty_metrics

    output = np.zeros_like(base_labels, dtype=np.uint16)
    next_id = 1
    split_details: list[dict] = []
    rejected_count = 0
    padding = link_radius_px + 4

    for prop in measure.regionprops(base_labels):
        min_row, min_col, max_row, max_col = prop.bbox
        row0 = max(0, min_row - padding)
        col0 = max(0, min_col - padding)
        row1 = min(base_labels.shape[0], max_row + padding)
        col1 = min(base_labels.shape[1], max_col + padding)
        crop = np.s_[row0:row1, col0:col1]
        component = base_labels[crop] == prop.label
        local_struct = struct[crop]
        local_cellpose = cellpose_mask[crop]
        local_nuclei_labels = nuclei_labels[crop]
        local_nearest_labels = nearest_nucleus_labels[crop]
        local_nucleus_distance = nucleus_distance[crop]
        distance = ndi.distance_transform_edt(component)
        scored_nuclei, _ = score_nuclei_for_component(
            component,
            local_nearest_labels,
            local_nucleus_distance,
            distance,
            local_struct,
            local_cellpose,
            link_radius_px,
            config.ambiguity_score_delta,
        )
        anchor_groups = select_soma_anchor_groups(scored_nuclei, mean_pixel_um, config)
        scored_by_id = {row["nucleus_id"]: row for row in scored_nuclei}
        strict_groups: list[dict] = []
        for group in anchor_groups:
            representative = scored_by_id.get(group["nucleus_ids"][0])
            if representative is None:
                continue
            has_channel_support = (
                representative["overlap_fraction"] >= config.multi_anchor_min_overlap_fraction
                or representative["model_support"] >= config.multi_anchor_min_model_support
            )
            if (
                group["score"] >= config.instance_split_min_anchor_score
                and representative["thickness_support"]
                >= config.multi_anchor_min_thickness_support
                and representative["structural_support"]
                >= config.multi_anchor_min_structural_support
                and has_channel_support
            ):
                strict_groups.append(group)

        split_reason = "not_two_strict_anchors"
        partition = None
        anchor_separation_um = 0.0
        neck_core_ratio = math.inf
        child_areas: list[int] = []
        if len(strict_groups) == 2 and len(strict_groups) <= config.instance_split_max_markers:
            anchor_separation_um = mean_pixel_um * math.hypot(
                strict_groups[0]["center_y"] - strict_groups[1]["center_y"],
                strict_groups[0]["center_x"] - strict_groups[1]["center_x"],
            )
            if anchor_separation_um >= config.instance_split_min_anchor_separation_um:
                markers = np.zeros_like(component, dtype=np.int32)
                core_peaks: list[float] = []
                marker_masks: list[np.ndarray] = []
                distance_scale = max(float(np.percentile(distance[component], 99.0)), 1.0)
                for marker_id, group in enumerate(strict_groups, start=1):
                    selected_nucleus = np.isin(local_nuclei_labels, group["nucleus_ids"])
                    selected_distance = ndi.distance_transform_edt(~selected_nucleus)
                    search_region = component & (selected_distance <= link_radius_px)
                    if not search_region.any():
                        break
                    seed_score = (
                        0.72 * np.clip(distance / distance_scale, 0, 1)
                        + 0.23 * local_struct
                        + 0.05 * local_cellpose.astype(np.float32)
                    )
                    seed_score = np.where(search_region, seed_score, -np.inf)
                    seed_y, seed_x = np.unravel_index(
                        int(np.argmax(seed_score)),
                        seed_score.shape,
                    )
                    marker_radius_px = max(1, int(round(0.25 / mean_pixel_um)))
                    marker_mask = component & circular_mask(
                        component.shape,
                        seed_y,
                        seed_x,
                        marker_radius_px,
                    )
                    if not marker_mask.any() or np.any(markers[marker_mask] > 0):
                        break
                    markers[marker_mask] = marker_id
                    marker_masks.append(marker_mask)
                    core_neighborhood = component & circular_mask(
                        component.shape,
                        seed_y,
                        seed_x,
                        max(link_radius_px, int(round(0.75 / mean_pixel_um))),
                    )
                    core_peaks.append(
                        max(float(np.percentile(distance[core_neighborhood], 90.0)), 1.0)
                    )

                if len(marker_masks) == 2:
                    distance_cost = 1.0 - np.clip(distance / distance_scale, 0, 1)
                    structural_cost = 1.0 - filters.gaussian(
                        local_struct,
                        sigma=1.0,
                        preserve_range=True,
                    )
                    elevation = 0.72 * distance_cost + 0.28 * structural_cost
                    candidate_partition = segmentation.watershed(
                        elevation,
                        markers=markers,
                        mask=component,
                        watershed_line=False,
                        connectivity=np.ones((3, 3), dtype=bool),
                    ).astype(np.uint16)
                    child_areas = [
                        int((candidate_partition == marker_id).sum())
                        for marker_id in (1, 2)
                    ]
                    minimum_child_area = max(
                        int(round(config.instance_split_min_child_area_um2 / pixel_area_um2)),
                        int(round(config.instance_split_min_child_fraction * int(component.sum()))),
                    )
                    boundary = np.zeros_like(component, dtype=bool)
                    vertical = (
                        (candidate_partition[1:] > 0)
                        & (candidate_partition[:-1] > 0)
                        & (candidate_partition[1:] != candidate_partition[:-1])
                    )
                    horizontal = (
                        (candidate_partition[:, 1:] > 0)
                        & (candidate_partition[:, :-1] > 0)
                        & (candidate_partition[:, 1:] != candidate_partition[:, :-1])
                    )
                    boundary[1:] |= vertical
                    boundary[:-1] |= vertical
                    boundary[:, 1:] |= horizontal
                    boundary[:, :-1] |= horizontal
                    if boundary.any():
                        neck_core_ratio = float(np.percentile(distance[boundary], 75.0)) / max(
                            min(core_peaks),
                            1.0,
                        )
                    children_connected = all(
                        int(measure.label(candidate_partition == marker_id, connectivity=2).max()) == 1
                        for marker_id in (1, 2)
                    )
                    markers_retained = all(
                        bool(np.all(candidate_partition[marker_masks[index]] == index + 1))
                        for index in range(2)
                    )
                    if min(child_areas) < minimum_child_area:
                        split_reason = "child_too_small"
                    elif not children_connected or not markers_retained:
                        split_reason = "invalid_partition"
                    elif neck_core_ratio > config.instance_split_max_neck_core_ratio:
                        split_reason = "neck_too_thick"
                    else:
                        split_reason = "accepted"
                        partition = candidate_partition
                else:
                    split_reason = "marker_build_failed"
            else:
                split_reason = "anchors_too_close"

        output_view = output[crop]
        if partition is None:
            output_view[component] = next_id
            next_id += 1
            if len(strict_groups) >= 2:
                rejected_count += 1
            continue

        assigned_ids: list[int] = []
        for marker_id in range(1, int(partition.max()) + 1):
            output_view[partition == marker_id] = next_id
            assigned_ids.append(next_id)
            next_id += 1
        split_details.append(
            {
                "base_component_id": int(prop.label),
                "new_astrocyte_ids": assigned_ids,
                "anchor_scores": [round(float(group["score"]), 6) for group in strict_groups],
                "anchor_separation_um": round(anchor_separation_um, 6),
                "child_areas_px": child_areas,
                "neck_core_ratio": round(neck_core_ratio, 6),
                "reason": split_reason,
            }
        )

    if not np.array_equal(output > 0, whole_mask.astype(bool)):
        raise RuntimeError("Instance splitting changed the Whole Astrocyte pixel union")
    final_count = int(output.max())
    return output, {
        "base_connected_component_count": base_count,
        "final_instance_count": final_count,
        "split_component_count": len(split_details),
        "split_added_roi_count": final_count - base_count,
        "split_components": split_details,
        "split_rejected_count": rejected_count,
    }

def split_touching_whole_instances_multi(
    whole_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    nearest_nucleus_labels: np.ndarray,
    nucleus_distance: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    mean_pixel_um: float,
    pixel_area_um2: float,
    link_radius_px: int,
    config: CompartmentConfig,
) -> tuple[np.ndarray, dict]:
    """Neonatal multi-center partition; preserve every Whole pixel and create no seam."""

    base_labels = measure.label(whole_mask, connectivity=2).astype(np.uint16)
    base_count = int(base_labels.max())
    empty_metrics = {
        "base_connected_component_count": base_count,
        "final_instance_count": base_count,
        "split_component_count": 0,
        "split_added_roi_count": 0,
        "split_components": [],
        "split_rejected_count": 0,
        "component_decisions": [],
    }
    if not config.instance_split_enabled or base_count < 1:
        return base_labels, empty_metrics

    output = np.zeros_like(base_labels, dtype=np.uint16)
    next_id = 1
    split_details: list[dict] = []
    component_decisions: list[dict] = []
    rejected_count = 0
    padding = link_radius_px + 4
    max_markers = max(2, int(config.instance_split_max_markers))

    for prop in measure.regionprops(base_labels):
        min_row, min_col, max_row, max_col = prop.bbox
        row0 = max(0, min_row - padding)
        col0 = max(0, min_col - padding)
        row1 = min(base_labels.shape[0], max_row + padding)
        col1 = min(base_labels.shape[1], max_col + padding)
        crop = np.s_[row0:row1, col0:col1]
        component = base_labels[crop] == prop.label
        local_struct = struct[crop]
        local_cellpose = cellpose_mask[crop]
        local_nuclei_labels = nuclei_labels[crop]
        local_nearest_labels = nearest_nucleus_labels[crop]
        local_nucleus_distance = nucleus_distance[crop]
        distance = ndi.distance_transform_edt(component)
        scored_nuclei, _ = score_nuclei_for_component(
            component,
            local_nearest_labels,
            local_nucleus_distance,
            distance,
            local_struct,
            local_cellpose,
            link_radius_px,
            config.ambiguity_score_delta,
        )
        anchor_groups = select_soma_anchor_groups(scored_nuclei, mean_pixel_um, config)
        scored_by_id = {row["nucleus_id"]: row for row in scored_nuclei}
        strict_groups: list[dict] = []
        for group in anchor_groups:
            representative = scored_by_id.get(group["nucleus_ids"][0])
            if representative is None:
                continue
            has_channel_support = (
                representative["overlap_fraction"] >= config.multi_anchor_min_overlap_fraction
                or representative["model_support"] >= config.multi_anchor_min_model_support
            )
            if (
                group["score"] >= config.instance_split_min_anchor_score
                and representative["thickness_support"]
                >= config.multi_anchor_min_thickness_support
                and representative["structural_support"]
                >= config.multi_anchor_min_structural_support
                and has_channel_support
            ):
                strict_groups.append(group)

        selected_groups: list[dict] = []
        for group in strict_groups:
            separated = all(
                mean_pixel_um
                * math.hypot(
                    group["center_y"] - accepted["center_y"],
                    group["center_x"] - accepted["center_x"],
                )
                >= config.instance_split_min_anchor_separation_um
                for accepted in selected_groups
            )
            if separated:
                selected_groups.append(group)
            if len(selected_groups) >= max_markers:
                break

        split_reason = "not_enough_strict_anchors"
        partition = None
        minimum_anchor_separation_um = 0.0
        neck_core_ratio = math.inf
        boundary_structural_ratio = math.inf
        child_areas: list[int] = []
        minimum_child_area = 0
        if len(selected_groups) >= 2:
            pairwise_separations = [
                mean_pixel_um
                * math.hypot(
                    selected_groups[left]["center_y"] - selected_groups[right]["center_y"],
                    selected_groups[left]["center_x"] - selected_groups[right]["center_x"],
                )
                for left in range(len(selected_groups))
                for right in range(left + 1, len(selected_groups))
            ]
            minimum_anchor_separation_um = min(pairwise_separations)
            markers = np.zeros_like(component, dtype=np.int32)
            marker_masks: list[np.ndarray] = []
            core_peaks: list[float] = []
            core_structural_supports: list[float] = []
            distance_scale = max(float(np.percentile(distance[component], 99.0)), 1.0)
            for marker_id, group in enumerate(selected_groups, start=1):
                selected_nucleus = np.isin(local_nuclei_labels, group["nucleus_ids"])
                selected_distance = ndi.distance_transform_edt(~selected_nucleus)
                search_region = component & (selected_distance <= link_radius_px)
                if not search_region.any():
                    break
                seed_score = (
                    0.72 * np.clip(distance / distance_scale, 0, 1)
                    + 0.23 * local_struct
                    + 0.05 * local_cellpose.astype(np.float32)
                )
                seed_score = np.where(search_region, seed_score, -np.inf)
                seed_y, seed_x = np.unravel_index(int(np.argmax(seed_score)), seed_score.shape)
                marker_radius_px = max(1, int(round(0.25 / mean_pixel_um)))
                marker_mask = component & circular_mask(
                    component.shape,
                    seed_y,
                    seed_x,
                    marker_radius_px,
                )
                if not marker_mask.any() or np.any(markers[marker_mask] > 0):
                    break
                markers[marker_mask] = marker_id
                marker_masks.append(marker_mask)
                core_neighborhood = component & circular_mask(
                    component.shape,
                    seed_y,
                    seed_x,
                    max(link_radius_px, int(round(0.75 / mean_pixel_um))),
                )
                core_peaks.append(
                    max(float(np.percentile(distance[core_neighborhood], 90.0)), 1.0)
                )
                core_structural_supports.append(
                    max(float(np.percentile(local_struct[core_neighborhood], 75.0)), 1e-6)
                )

            marker_count = len(selected_groups)
            if len(marker_masks) == marker_count:
                distance_cost = 1.0 - np.clip(distance / distance_scale, 0, 1)
                structural_cost = 1.0 - filters.gaussian(
                    local_struct,
                    sigma=1.0,
                    preserve_range=True,
                )
                elevation = 0.72 * distance_cost + 0.28 * structural_cost
                candidate_partition = segmentation.watershed(
                    elevation,
                    markers=markers,
                    mask=component,
                    watershed_line=False,
                    connectivity=np.ones((3, 3), dtype=bool),
                ).astype(np.uint16)
                marker_ids = list(range(1, marker_count + 1))
                child_areas = [
                    int((candidate_partition == marker_id).sum())
                    for marker_id in marker_ids
                ]
                minimum_child_area = max(
                    int(round(config.instance_split_min_child_area_um2 / pixel_area_um2)),
                    int(round(config.instance_split_min_child_fraction * int(component.sum()))),
                )
                boundary = np.zeros_like(component, dtype=bool)
                vertical = (
                    (candidate_partition[1:] > 0)
                    & (candidate_partition[:-1] > 0)
                    & (candidate_partition[1:] != candidate_partition[:-1])
                )
                horizontal = (
                    (candidate_partition[:, 1:] > 0)
                    & (candidate_partition[:, :-1] > 0)
                    & (candidate_partition[:, 1:] != candidate_partition[:, :-1])
                )
                boundary[1:] |= vertical
                boundary[:-1] |= vertical
                boundary[:, 1:] |= horizontal
                boundary[:, :-1] |= horizontal
                if boundary.any():
                    neck_core_ratio = float(np.percentile(distance[boundary], 75.0)) / max(
                        min(core_peaks),
                        1.0,
                    )
                    boundary_structural_ratio = float(np.median(local_struct[boundary])) / max(
                        min(core_structural_supports),
                        1e-6,
                    )
                children_connected = all(
                    int(measure.label(candidate_partition == marker_id, connectivity=2).max()) == 1
                    for marker_id in marker_ids
                )
                markers_retained = all(
                    bool(np.all(candidate_partition[marker_masks[index]] == index + 1))
                    for index in range(marker_count)
                )
                if min(child_areas) < minimum_child_area:
                    split_reason = "child_too_small"
                elif not children_connected or not markers_retained:
                    split_reason = "invalid_partition"
                elif not (
                    neck_core_ratio <= config.instance_split_strict_neck_core_ratio
                    or (
                        neck_core_ratio <= config.instance_split_max_neck_core_ratio
                        and boundary_structural_ratio
                        <= config.instance_split_max_boundary_structural_ratio
                    )
                ):
                    split_reason = "neck_or_boundary_too_supported"
                else:
                    split_reason = "accepted"
                    partition = candidate_partition
            else:
                split_reason = "marker_build_failed"

        output_view = output[crop]
        if partition is None:
            assigned_id = next_id
            output_view[component] = assigned_id
            next_id += 1
            if len(strict_groups) >= 2:
                rejected_count += 1
            component_decisions.append(
                {
                    "base_component_id": int(prop.label),
                    "output_astrocyte_ids": [assigned_id],
                    "strict_anchor_count": len(strict_groups),
                    "selected_anchor_count": len(selected_groups),
                    "split_required": len(strict_groups) >= 2,
                    "split_accepted": False,
                    "reason": split_reason,
                }
            )
            continue

        assigned_ids: list[int] = []
        for marker_id in range(1, int(partition.max()) + 1):
            output_view[partition == marker_id] = next_id
            assigned_ids.append(next_id)
            next_id += 1
        split_details.append(
            {
                "base_component_id": int(prop.label),
                "new_astrocyte_ids": assigned_ids,
                "anchor_scores": [
                    round(float(group["score"]), 6) for group in selected_groups
                ],
                "anchor_separation_um": round(minimum_anchor_separation_um, 6),
                "child_areas_px": child_areas,
                "neck_core_ratio": round(neck_core_ratio, 6),
                "boundary_structural_ratio": round(boundary_structural_ratio, 6),
                "reason": split_reason,
            }
        )
        component_decisions.append(
            {
                "base_component_id": int(prop.label),
                "output_astrocyte_ids": assigned_ids,
                "strict_anchor_count": len(strict_groups),
                "selected_anchor_count": len(selected_groups),
                "split_required": True,
                "split_accepted": True,
                "reason": split_reason,
                "child_areas_px": child_areas,
                "minimum_child_area_px": int(minimum_child_area),
            }
        )

    if not np.array_equal(output > 0, whole_mask.astype(bool)):
        raise RuntimeError("Neonatal instance splitting changed the Whole Astrocyte pixel union")
    final_count = int(output.max())
    return output, {
        "base_connected_component_count": base_count,
        "final_instance_count": final_count,
        "split_component_count": len(split_details),
        "split_added_roi_count": final_count - base_count,
        "split_components": split_details,
        "split_rejected_count": rejected_count,
        "component_decisions": component_decisions,
    }
