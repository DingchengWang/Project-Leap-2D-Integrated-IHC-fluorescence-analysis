# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def group_inventory_nucleus_objects(
    inventory: ValidatedNucleusAnchors,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float,
    config: NucleusOwnershipConfig,
    prefer_canonical: bool = False,
) -> list[dict]:
    """Group only geometrically overlapping threshold fragments of one 3D nucleus."""

    if prefer_canonical and inventory.nucleus_instance_records:
        return [
            {
                "group_id": int(row["instance_id"]),
                "object_ids": (int(row["instance_id"]),),
                "source_object_ids": tuple(row["source_object_ids"]),
                "accepted": bool(row["accepted"]),
                "independently_accepted": bool(row["accepted"]),
                "accepted_volume_gate_passed": bool(
                    float(row["volume_um3"]) >= config.accepted_min_volume_um3
                ),
                "identity_status": str(row["identity_status"]),
                "volume_um3": float(row["volume_um3"]),
                "enclosure_score": float(row["enclosure_score"]),
                "center_z": float(row["center_z"]),
                "center_y": float(row["center_y"]),
                "center_x": float(row["center_x"]),
                "z_min_0based": int(row["z_min_0based"]),
                "z_max_0based_inclusive": int(row["z_max_0based_inclusive"]),
                "extent_component_2d_ids": (
                    int(row["extent_component_2d_id"]),
                ),
            }
            for row in inventory.nucleus_instance_records
            if bool(row["dapi_valid"])
        ]

    records = [dict(row) for row in inventory.object_records if bool(row["dapi_valid"])]
    if not records:
        return []
    parent = {int(row["object_id"]): int(row["object_id"]) for row in records}

    def find(object_id: int) -> int:
        while parent[object_id] != object_id:
            parent[object_id] = parent[parent[object_id]]
            object_id = parent[object_id]
        return object_id

    def union(left_id: int, right_id: int) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left_index, left in enumerate(records):
        left_extent_component = int(left["extent_component_2d_id"])
        if left_extent_component <= 0:
            continue
        for right in records[left_index + 1 :]:
            if int(right["extent_component_2d_id"]) != left_extent_component:
                continue
            z_overlap = min(
                int(left["z_max_0based_inclusive"]),
                int(right["z_max_0based_inclusive"]),
            ) - max(
                int(left["z_min_0based"]),
                int(right["z_min_0based"]),
            ) + 1
            if z_overlap <= 0:
                continue
            delta_z_um = (
                float(left["center_z"]) - float(right["center_z"])
            ) * pixel_depth_um
            delta_y_um = (
                float(left["center_y"]) - float(right["center_y"])
            ) * pixel_height_um
            delta_x_um = (
                float(left["center_x"]) - float(right["center_x"])
            ) * pixel_width_um
            center_distance_um = math.sqrt(
                delta_z_um**2 + delta_y_um**2 + delta_x_um**2
            )
            left_radius_um = (
                3.0 * float(left["volume_um3"]) / (4.0 * math.pi)
            ) ** (1.0 / 3.0)
            right_radius_um = (
                3.0 * float(right["volume_um3"]) / (4.0 * math.pi)
            ) ** (1.0 / 3.0)
            if center_distance_um <= config.fragment_radius_sum_factor * (
                left_radius_um + right_radius_um
            ):
                union(int(left["object_id"]), int(right["object_id"]))

    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(find(int(record["object_id"])), []).append(record)
    output: list[dict] = []
    for group_id, members in sorted(grouped.items()):
        total_volume = float(sum(float(row["volume_um3"]) for row in members))
        independently_accepted = any(bool(row["accepted"]) for row in members)
        ownership_accepted = bool(
            independently_accepted
            and total_volume >= config.accepted_min_volume_um3
        )
        weights = np.asarray(
            [max(float(row["volume_um3"]), 1e-9) for row in members],
            dtype=np.float64,
        )
        output.append(
            {
                "group_id": int(group_id),
                "object_ids": tuple(sorted(int(row["object_id"]) for row in members)),
                "accepted": ownership_accepted,
                "independently_accepted": independently_accepted,
                "accepted_volume_gate_passed": bool(
                    total_volume >= config.accepted_min_volume_um3
                ),
                "volume_um3": total_volume,
                "enclosure_score": max(float(row["enclosure_score"]) for row in members),
                "center_z": float(
                    np.average([float(row["center_z"]) for row in members], weights=weights)
                ),
                "center_y": float(
                    np.average([float(row["center_y"]) for row in members], weights=weights)
                ),
                "center_x": float(
                    np.average([float(row["center_x"]) for row in members], weights=weights)
                ),
                "z_min_0based": min(int(row["z_min_0based"]) for row in members),
                "z_max_0based_inclusive": max(
                    int(row["z_max_0based_inclusive"]) for row in members
                ),
                "extent_component_2d_ids": tuple(
                    sorted(
                        {
                            int(row["extent_component_2d_id"])
                            for row in members
                            if int(row["extent_component_2d_id"]) > 0
                        }
                    )
                ),
            }
        )
    return output

def nucleus_group_marker(
    component: np.ndarray,
    extent_mask: np.ndarray,
    local_struct: np.ndarray,
    distance_inside_um: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: NucleusOwnershipConfig,
) -> np.ndarray:
    distance_to_extent_um = ndi.distance_transform_edt(
        ~extent_mask,
        sampling=(pixel_height_um, pixel_width_um),
    )
    search = component & (distance_to_extent_um <= config.marker_search_um)
    if not search.any():
        return np.zeros_like(component, dtype=bool)
    distance_scale = max(float(np.percentile(distance_inside_um[component], 99.0)), 1e-6)
    proximity = np.exp(
        -np.square(distance_to_extent_um / max(config.marker_search_um, 1e-6))
    )
    score = (
        0.58 * np.clip(distance_inside_um / distance_scale, 0.0, 1.0)
        + 0.27 * local_struct
        + 0.15 * proximity
    )
    score = np.where(search, score, -np.inf)
    seed_y, seed_x = np.unravel_index(int(np.argmax(score)), score.shape)
    radius_y = max(1, int(round(config.marker_radius_um / pixel_height_um)))
    radius_x = max(1, int(round(config.marker_radius_um / pixel_width_um)))
    yy, xx = np.ogrid[: component.shape[0], : component.shape[1]]
    marker = component & (
        ((yy - seed_y) / radius_y) ** 2 + ((xx - seed_x) / radius_x) ** 2 <= 1.0
    )
    if not marker.any():
        marker[seed_y, seed_x] = True
    return marker

def apply_nucleus_ownership_guard(
    instance_labels: np.ndarray,
    struct: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
    profile: str,
    config: NucleusOwnershipConfig | None = None,
    prefer_canonical: bool = False,
) -> tuple[np.ndarray, dict]:
    """Split validated owners and remove territory assigned to a foreign 3D soma."""

    cfg = config or NucleusOwnershipConfig()
    empty_metrics = {
        "enabled": inventory is not None,
        "profile": profile,
        "method": (
            "object-preserving 3D owner partition with Z-supported foreign-soma barrier"
        ),
        "evaluated_component_count": int(instance_labels.max()),
        "conflict_component_count": 0,
        "split_component_count": 0,
        "foreign_soma_pruned_component_count": 0,
        "fail_closed_component_count": 0,
        "removed_area_px": 0,
        "input_to_output_ids": {},
        "decisions": [],
        "config": asdict(cfg),
    }
    if (
        inventory is None
        or pixel_depth_um is None
        or pixel_depth_um <= 0
        or inventory.object_extent_labels_2d is None
        or not inventory.object_records
    ):
        return instance_labels, empty_metrics

    extent_labels = np.asarray(inventory.object_extent_labels_2d, dtype=np.uint32)
    if extent_labels.shape != instance_labels.shape:
        raise ValueError("3D nucleus inventory and Whole instance labels do not match")
    groups = group_inventory_nucleus_objects(
        inventory,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
        cfg,
        prefer_canonical=prefer_canonical,
    )
    if not groups:
        return instance_labels, empty_metrics

    object_to_group = np.zeros(int(extent_labels.max()) + 1, dtype=np.uint32)
    for group in groups:
        for object_id in group["object_ids"]:
            object_to_group[int(object_id)] = int(group["group_id"])
    grouped_extent_labels = object_to_group[extent_labels]
    group_extent_areas = np.bincount(
        grouped_extent_labels.ravel(),
        minlength=len(object_to_group),
    )
    instance_stride = int(instance_labels.max()) + 1
    group_instance_counts = np.bincount(
        (
            grouped_extent_labels.astype(np.int64) * instance_stride
            + instance_labels.astype(np.int64)
        ).ravel(),
        minlength=len(object_to_group) * instance_stride,
    ).reshape(len(object_to_group), instance_stride)

    pixel_area_um2 = pixel_width_um * pixel_height_um
    owner_min_overlap_px = max(1, int(math.ceil(cfg.owner_min_overlap_um2 / pixel_area_um2)))
    foreign_min_overlap_px = max(
        1,
        int(math.ceil(cfg.foreign_min_overlap_um2 / pixel_area_um2)),
    )
    minimum_child_area_px = max(
        1,
        int(math.ceil(cfg.minimum_child_area_um2 / pixel_area_um2)),
    )
    output = np.zeros_like(instance_labels, dtype=np.uint16)
    next_id = 1
    decisions: list[dict] = []
    split_count = 0
    pruned_count = 0
    fail_closed_count = 0
    removed_area_px = 0
    input_to_output_ids: dict[int, list[int]] = {}

    for prop in measure.regionprops(instance_labels):
        min_row, min_col, max_row, max_col = prop.bbox
        crop = np.s_[min_row:max_row, min_col:max_col]
        component = instance_labels[crop] == int(prop.label)
        local_struct = struct[crop]
        local_grouped_extent_labels = grouped_extent_labels[crop]
        component_area = int(component.sum())
        associations: list[dict] = []
        for group in groups:
            group_id = int(group["group_id"])
            extent_area = int(group_extent_areas[group_id])
            overlap = int(group_instance_counts[group_id, int(prop.label)])
            if overlap == 0 or extent_area == 0:
                continue
            extent = local_grouped_extent_labels == group_id
            total_labelled_overlap = int(group_instance_counts[group_id, 1:].sum())
            dominance = overlap / max(total_labelled_overlap, 1)
            associations.append(
                {
                    **group,
                    "extent": extent,
                    "extent_area_px": extent_area,
                    "overlap_px": overlap,
                    "overlap_fraction": overlap / extent_area,
                    "component_dominance": dominance,
                }
            )

        owner_candidates = [
            row
            for row in associations
            if bool(row["accepted"]) and int(row["overlap_px"]) >= owner_min_overlap_px
        ]
        ambiguous_nuclear_envelopes = [
            row
            for row in associations
            if str(row.get("identity_status", "resolved")) == "ambiguous"
            and int(row["overlap_px"]) >= owner_min_overlap_px
            and float(row["overlap_fraction"]) >= 0.30
        ]
        if ambiguous_nuclear_envelopes:
            input_to_output_ids[int(prop.label)] = []
            removed_area_px += component_area
            fail_closed_count += 1
            empty_metrics["conflict_component_count"] += 1
            decisions.append(
                {
                    "input_instance_id": int(prop.label),
                    "owner_group_id": 0,
                    "foreign_groups": [],
                    "ambiguous_group_ids": [
                        int(row["group_id"]) for row in ambiguous_nuclear_envelopes
                    ],
                    "status": "fail_closed_ambiguous_canonical_nuclear_envelope",
                    "output_instance_ids": [],
                    "removed_area_px": component_area,
                }
            )
            continue
        if not owner_candidates:
            output[crop][component] = next_id
            input_to_output_ids[int(prop.label)] = [next_id]
            next_id += 1
            continue
        owner = max(
            owner_candidates,
            key=lambda row: (
                int(row["overlap_px"]),
                float(row["volume_um3"]),
                float(row["enclosure_score"]),
            ),
        )
        foreign: list[dict] = []
        rejected_candidates: list[dict] = []
        for row in associations:
            if int(row["group_id"]) == int(owner["group_id"]):
                continue
            if bool(row["accepted"]):
                accepted_checks = {
                    "independently_accepted_3d_nucleus": True,
                    "volume": (
                        float(row["volume_um3"]) >= cfg.accepted_min_volume_um3
                    ),
                    "owner_overlap": (
                        int(row["overlap_px"]) >= owner_min_overlap_px
                    ),
                    "extent_overlap_fraction": (
                        float(row["overlap_fraction"])
                        >= cfg.accepted_min_extent_overlap_fraction
                    ),
                }
                if all(accepted_checks.values()):
                    foreign.append(row)
                else:
                    rejected_candidates.append(
                        {
                            "group_id": int(row["group_id"]),
                            "object_ids": list(row["object_ids"]),
                            "checks": accepted_checks,
                        }
                    )
                continue
            minimum_volume = (
                cfg.unowned_min_volume_um3
            )
            checks = {
                "volume": float(row["volume_um3"]) >= minimum_volume,
                "absolute_overlap": int(row["overlap_px"]) >= foreign_min_overlap_px,
                "overlap_fraction": (
                    float(row["overlap_fraction"]) >= cfg.foreign_min_overlap_fraction
                ),
                "owner_overlap_ratio": (
                    int(row["overlap_px"])
                    >= cfg.foreign_min_owner_overlap_ratio * int(owner["overlap_px"])
                ),
                "component_dominance": (
                    float(row["component_dominance"])
                    >= cfg.foreign_min_component_dominance
                ),
                "z_supported_enclosure": (
                    float(row["enclosure_score"])
                    >= cfg.unowned_min_enclosure_score
                ),
            }
            if all(checks.values()):
                foreign.append(row)
            else:
                rejected_candidates.append(
                    {
                        "group_id": int(row["group_id"]),
                        "object_ids": list(row["object_ids"]),
                        "checks": checks,
                    }
                )
        if not foreign:
            output[crop][component] = next_id
            input_to_output_ids[int(prop.label)] = [next_id]
            next_id += 1
            continue

        distance_inside_um = ndi.distance_transform_edt(
            component,
            sampling=(pixel_height_um, pixel_width_um),
        )
        owner_marker = nucleus_group_marker(
            component,
            owner["extent"],
            local_struct,
            distance_inside_um,
            pixel_width_um,
            pixel_height_um,
            cfg,
        )
        accepted_foreign = [row for row in foreign if bool(row["accepted"])]
        unowned_foreign = [row for row in foreign if not bool(row["accepted"])]
        decision = {
            "input_instance_id": int(prop.label),
            "owner_group_id": int(owner["group_id"]),
            "owner_object_ids": list(owner["object_ids"]),
            "foreign_groups": [
                {
                    "group_id": int(row["group_id"]),
                    "object_ids": list(row["object_ids"]),
                    "accepted": bool(row["accepted"]),
                    "volume_um3": float(row["volume_um3"]),
                    "overlap_px": int(row["overlap_px"]),
                    "overlap_fraction": float(row["overlap_fraction"]),
                    "component_dominance": float(row["component_dominance"]),
                    "enclosure_score": float(row["enclosure_score"]),
                }
                for row in foreign
            ],
            "rejected_candidates": rejected_candidates,
            "status": "pending",
            "output_instance_ids": [],
        }
        empty_metrics["conflict_component_count"] += 1

        if not owner_marker.any():
            input_to_output_ids[int(prop.label)] = []
            removed_area_px += component_area
            fail_closed_count += 1
            decision["status"] = "fail_closed_owner_marker_unavailable"
            decisions.append(decision)
            continue

        if not accepted_foreign:
            foreign_extent = np.logical_or.reduce(
                [row["extent"] for row in unowned_foreign]
            )
            distance_to_foreign_um = ndi.distance_transform_edt(
                ~foreign_extent,
                sampling=(pixel_height_um, pixel_width_um),
            )
            structural_cut = float(np.percentile(local_struct[component], 60.0))
            foreign_barrier = (
                component
                & (distance_to_foreign_um <= cfg.unowned_barrier_radius_um)
                & (
                    (distance_inside_um >= cfg.unowned_barrier_inner_width_um)
                    | (local_struct >= structural_cut)
                )
            )
            allowed = component & ~foreign_barrier
            owner_seed = owner_marker & allowed
            if owner_seed.any():
                owner_child = ndi.binary_propagation(
                    owner_seed,
                    structure=np.ones((3, 3), dtype=bool),
                    mask=allowed,
                ).astype(bool)
            else:
                owner_child = np.zeros_like(component, dtype=bool)
            minimum_owner_area = max(
                minimum_child_area_px,
                int(math.ceil(cfg.minimum_owner_child_fraction * component_area)),
            )
            if int(owner_child.sum()) < minimum_owner_area:
                input_to_output_ids[int(prop.label)] = []
                removed_area_px += component_area
                fail_closed_count += 1
                decision["status"] = "fail_closed_foreign_barrier_invalid_owner"
                decisions.append(decision)
                continue
            output_view = output[crop]
            output_view[owner_child] = next_id
            decision["output_instance_ids"] = [next_id]
            input_to_output_ids[int(prop.label)] = [next_id]
            next_id += 1
            removed = component_area - int(owner_child.sum())
            removed_area_px += removed
            pruned_count += 1
            decision["status"] = "foreign_soma_pruned"
            decision["removed_area_px"] = removed
            decisions.append(decision)
            continue

        partition_groups = [owner, *accepted_foreign]
        markers = np.zeros_like(component, dtype=np.int32)
        marker_masks: list[np.ndarray] = []
        marker_failed = False
        for marker_id, group in enumerate(partition_groups, start=1):
            marker = nucleus_group_marker(
                component,
                group["extent"],
                local_struct,
                distance_inside_um,
                pixel_width_um,
                pixel_height_um,
                cfg,
            )
            if not marker.any() or np.any(markers[marker] > 0):
                marker_failed = True
                break
            markers[marker] = marker_id
            marker_masks.append(marker)
        if marker_failed:
            input_to_output_ids[int(prop.label)] = []
            removed_area_px += component_area
            fail_closed_count += 1
            decision["status"] = "fail_closed_marker_build_failed"
            decisions.append(decision)
            continue

        partition_mask = component.copy()
        if unowned_foreign:
            unowned_extent = np.logical_or.reduce(
                [row["extent"] for row in unowned_foreign]
            )
            unowned_distance_um = ndi.distance_transform_edt(
                ~unowned_extent,
                sampling=(pixel_height_um, pixel_width_um),
            )
            partition_mask &= (
                unowned_distance_um > cfg.multi_owner_unowned_exclusion_um
            )
            if any(not np.all(partition_mask[marker]) for marker in marker_masks):
                input_to_output_ids[int(prop.label)] = []
                removed_area_px += component_area
                fail_closed_count += 1
                decision["status"] = "fail_closed_unowned_barrier_hit_owner"
                decisions.append(decision)
                continue

        distance_scale = max(float(np.percentile(distance_inside_um[component], 99.0)), 1e-6)
        distance_cost = 1.0 - np.clip(distance_inside_um / distance_scale, 0.0, 1.0)
        structural_cost = 1.0 - filters.gaussian(
            local_struct,
            sigma=1.0,
            preserve_range=True,
        )
        elevation = 0.72 * distance_cost + 0.28 * structural_cost
        partition = segmentation.watershed(
            elevation,
            markers=markers,
            mask=partition_mask,
            watershed_line=False,
            connectivity=np.ones((3, 3), dtype=bool),
        ).astype(np.uint16)
        child_areas = [
            int((partition == marker_id).sum())
            for marker_id in range(1, len(partition_groups) + 1)
        ]
        minimum_areas = [
            max(
                minimum_child_area_px,
                int(
                    math.ceil(
                        (
                            cfg.minimum_owner_child_fraction
                            if marker_id == 1
                            else cfg.minimum_accepted_child_fraction
                        )
                        * component_area
                    )
                ),
            )
            for marker_id in range(1, len(partition_groups) + 1)
        ]
        connected = all(
            int(measure.label(partition == marker_id, connectivity=2).max()) == 1
            for marker_id in range(1, len(partition_groups) + 1)
        )
        retained = all(
            bool(np.all(partition[marker_masks[index]] == index + 1))
            for index in range(len(marker_masks))
        )
        boundary = np.zeros_like(component, dtype=bool)
        vertical = (
            (partition[1:] > 0)
            & (partition[:-1] > 0)
            & (partition[1:] != partition[:-1])
        )
        horizontal = (
            (partition[:, 1:] > 0)
            & (partition[:, :-1] > 0)
            & (partition[:, 1:] != partition[:, :-1])
        )
        boundary[1:] |= vertical
        boundary[:-1] |= vertical
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
        core_peaks = [
            max(float(np.percentile(distance_inside_um[marker], 90.0)), 1e-6)
            for marker in marker_masks
        ]
        core_structural = [
            max(float(np.percentile(local_struct[marker], 75.0)), 1e-6)
            for marker in marker_masks
        ]
        boundary_core_ratio = (
            float(np.percentile(distance_inside_um[boundary], 75.0))
            / max(min(core_peaks), 1e-6)
            if boundary.any()
            else math.inf
        )
        boundary_structural_ratio = (
            float(np.median(local_struct[boundary])) / max(min(core_structural), 1e-6)
            if boundary.any()
            else math.inf
        )
        quality_passed = bool(
            all(area >= floor for area, floor in zip(child_areas, minimum_areas))
            and connected
            and retained
            and (
                boundary_core_ratio <= cfg.maximum_boundary_core_ratio
                or boundary_structural_ratio <= cfg.maximum_boundary_structural_ratio
            )
        )
        decision["child_areas_px"] = child_areas
        decision["minimum_child_areas_px"] = minimum_areas
        decision["boundary_core_ratio"] = float(boundary_core_ratio)
        decision["boundary_structural_ratio"] = float(boundary_structural_ratio)
        if not quality_passed:
            undersized_foreign = any(
                child_areas[index] < minimum_areas[index]
                for index in range(1, len(partition_groups))
            )
            if prefer_canonical and undersized_foreign:
                output[crop][component] = next_id
                input_to_output_ids[int(prop.label)] = [next_id]
                decision["status"] = (
                    "retained_peripheral_nucleus_without_minimum_cell_territory"
                )
                decision["output_instance_ids"] = [next_id]
                next_id += 1
                decisions.append(decision)
                continue
            input_to_output_ids[int(prop.label)] = []
            removed_area_px += component_area
            fail_closed_count += 1
            decision["status"] = "fail_closed_ambiguous_multi_owner_partition"
            decisions.append(decision)
            continue

        output_view = output[crop]
        assigned_ids: list[int] = []
        for marker_id in range(1, len(partition_groups) + 1):
            output_view[partition == marker_id] = next_id
            assigned_ids.append(next_id)
            next_id += 1
        dropped = int(component.sum()) - int((partition > 0).sum())
        removed_area_px += dropped
        split_count += 1
        decision["status"] = "accepted_multi_owner_split"
        decision["output_instance_ids"] = assigned_ids
        input_to_output_ids[int(prop.label)] = assigned_ids
        decision["removed_area_px"] = dropped
        decisions.append(decision)

    final_mask = output > 0
    if np.any(final_mask & ~(instance_labels > 0)):
        raise RuntimeError("Nucleus ownership guard expanded the frozen Whole geometry")
    empty_metrics.update(
        {
            "conflict_component_count": len(decisions),
            "split_component_count": split_count,
            "foreign_soma_pruned_component_count": pruned_count,
            "fail_closed_component_count": fail_closed_count,
            "removed_area_px": removed_area_px,
            "removed_area_fraction": removed_area_px
            / max(int((instance_labels > 0).sum()), 1),
            "final_instance_count": int(output.max()),
            "input_to_output_ids": {
                str(input_id): output_ids
                for input_id, output_ids in input_to_output_ids.items()
            },
            "decisions": decisions,
        }
    )
    return output, empty_metrics

def inventory_group_geometry(
    inventory: ValidatedNucleusAnchors | None,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
    config: NucleusOwnershipConfig | None = None,
) -> tuple[list[dict], np.ndarray | None]:
    """Return stable nucleus groups and their 2D extent IDs."""

    if inventory is None or pixel_depth_um is None or pixel_depth_um <= 0:
        return [], None
    cfg = config or NucleusOwnershipConfig()
    if (
        inventory.nucleus_instance_extent_labels_2d is not None
        and inventory.nucleus_instance_records
    ):
        labels = np.asarray(
            inventory.nucleus_instance_extent_labels_2d,
            dtype=np.uint32,
        )
        groups = []
        for row in inventory.nucleus_instance_records:
            if not bool(row["dapi_valid"]):
                continue
            instance_id = int(row["instance_id"])
            groups.append(
                {
                    "group_id": instance_id,
                    "object_ids": (instance_id,),
                    "source_object_ids": tuple(row["source_object_ids"]),
                    "accepted": bool(row["accepted"]),
                    "identity_status": str(row["identity_status"]),
                    "volume_um3": float(row["volume_um3"]),
                    "enclosure_score": float(row["enclosure_score"]),
                    "center_z": float(row["center_z"]),
                    "center_y": float(row["center_y"]),
                    "center_x": float(row["center_x"]),
                    "z_min_0based": int(row["z_min_0based"]),
                    "z_max_0based_inclusive": int(row["z_max_0based_inclusive"]),
                    "dapi_low_threshold": (
                        float(row["dapi_low_threshold"])
                        if row.get("dapi_low_threshold") is not None
                        else None
                    ),
                    "extent_component_2d_ids": (
                        int(row["extent_component_2d_id"]),
                    ),
                }
            )
        valid_ids = np.asarray(
            [int(group["group_id"]) for group in groups],
            dtype=np.uint32,
        )
        return groups, np.where(np.isin(labels, valid_ids), labels, 0).astype(np.uint32)
    if inventory.object_extent_labels_2d is None or not inventory.object_records:
        return [], None
    groups = group_inventory_nucleus_objects(
        inventory,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
        cfg,
    )
    object_labels = np.asarray(inventory.object_extent_labels_2d, dtype=np.uint32)
    object_to_group = np.zeros(int(object_labels.max()) + 1, dtype=np.uint32)
    for group in groups:
        for object_id in group["object_ids"]:
            object_to_group[int(object_id)] = int(group["group_id"])
    return groups, object_to_group[object_labels]

def owner_group_for_soma(
    soma_mask: np.ndarray,
    groups: list[dict],
    grouped_extent_labels: np.ndarray,
) -> dict | None:
    """Choose one nucleus envelope by Soma overlap without intensity-peak voting."""

    overlaps: list[tuple[int, bool, float, dict]] = []
    for group in groups:
        group_id = int(group["group_id"])
        overlap = int((soma_mask & (grouped_extent_labels == group_id)).sum())
        if overlap <= 0:
            continue
        overlaps.append(
            (
                overlap,
                bool(group["accepted"]),
                float(group["volume_um3"]),
                group,
            )
        )
    if not overlaps:
        return None
    accepted = [row for row in overlaps if row[1]]
    pool = accepted or overlaps
    return max(pool, key=lambda row: (row[0], row[2]))[3]
