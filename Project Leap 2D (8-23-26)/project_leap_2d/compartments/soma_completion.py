# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def apply_canonical_identity_reconciliation(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    struct: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
    profile: str,
    config: CanonicalIdentityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Correct only explicit nucleus/ROI identity conflicts on validated compartment labels."""

    cfg = config or CanonicalIdentityConfig()
    metrics = {
        "enabled": bool(cfg.enabled and inventory is not None),
        "profile": profile,
        "method": (
            "local canonical identity reconciliation: same canonical nucleus merges "
            "adjacent IDs; multiple accepted nuclei trigger quality-gated partition"
        ),
        "pre_roi_count": int(whole_labels.max()),
        "post_roi_count": int(whole_labels.max()),
        "merge_count": 0,
        "split_count": 0,
        "fail_closed_count": 0,
        "changed_input_ids": [],
        "merge_decisions": [],
        "partition_decisions": [],
        "final_lineage": {},
        "config": asdict(cfg),
    }
    if (
        not metrics["enabled"]
        or pixel_depth_um is None
        or pixel_depth_um <= 0
        or inventory is None
        or inventory.nucleus_instance_extent_labels_2d is None
        or not inventory.nucleus_instance_records
    ):
        return whole_labels, soma_labels, process_labels, metrics

    canonical_records = [
        dict(row)
        for row in inventory.nucleus_instance_records
        if bool(row["accepted"])
        and bool(row["dapi_valid"])
        and str(row["identity_status"]) == "resolved"
    ]
    if not canonical_records:
        return whole_labels, soma_labels, process_labels, metrics
    satellite_aliases: dict[int, int] = {}
    for left_index, left in enumerate(canonical_records):
        for right in canonical_records[left_index + 1 :]:
            smaller, larger = sorted(
                (left, right),
                key=lambda row: float(row["volume_um3"]),
            )
            smaller_sources = set(int(value) for value in smaller["source_object_ids"])
            larger_sources = set(int(value) for value in larger["source_object_ids"])
            if not smaller_sources or not smaller_sources < larger_sources:
                continue
            volume_ratio = float(smaller["volume_um3"]) / max(
                float(larger["volume_um3"]),
                1e-9,
            )
            if volume_ratio > cfg.satellite_max_volume_ratio:
                continue
            z_overlap = max(
                0,
                min(
                    int(smaller["z_max_0based_inclusive"]),
                    int(larger["z_max_0based_inclusive"]),
                )
                - max(
                    int(smaller["z_min_0based"]),
                    int(larger["z_min_0based"]),
                )
                + 1,
            )
            smaller_z_span = (
                int(smaller["z_max_0based_inclusive"])
                - int(smaller["z_min_0based"])
                + 1
            )
            if z_overlap / max(smaller_z_span, 1) < cfg.satellite_min_z_overlap_fraction:
                continue
            delta_z = (
                float(smaller["center_z"]) - float(larger["center_z"])
            ) * pixel_depth_um
            delta_y = (
                float(smaller["center_y"]) - float(larger["center_y"])
            ) * pixel_height_um
            delta_x = (
                float(smaller["center_x"]) - float(larger["center_x"])
            ) * pixel_width_um
            center_distance_um = math.sqrt(delta_z**2 + delta_y**2 + delta_x**2)
            smaller_radius_um = (
                3.0 * float(smaller["volume_um3"]) / (4.0 * math.pi)
            ) ** (1.0 / 3.0)
            larger_radius_um = (
                3.0 * float(larger["volume_um3"]) / (4.0 * math.pi)
            ) ** (1.0 / 3.0)
            if center_distance_um > cfg.satellite_max_radius_sum_factor * (
                smaller_radius_um + larger_radius_um
            ):
                continue
            satellite_aliases[int(smaller["instance_id"])] = int(
                larger["instance_id"]
            )
    if satellite_aliases:
        def alias_root(instance_id: int) -> int:
            while instance_id in satellite_aliases:
                instance_id = satellite_aliases[instance_id]
            return instance_id

        satellite_aliases = {
            instance_id: alias_root(target_id)
            for instance_id, target_id in satellite_aliases.items()
        }
        canonical_records = [
            row
            for row in canonical_records
            if int(row["instance_id"]) not in satellite_aliases
        ]
    metrics["canonical_satellite_collapses"] = {
        str(instance_id): target_id
        for instance_id, target_id in sorted(satellite_aliases.items())
    }
    canonical_ids = np.asarray(
        [int(row["instance_id"]) for row in canonical_records],
        dtype=np.uint32,
    )
    canonical_extent_labels = np.asarray(
        inventory.nucleus_instance_extent_labels_2d,
        dtype=np.uint32,
    )
    canonical_core_labels = np.asarray(
        inventory.nucleus_instance_core_labels_2d,
        dtype=np.uint32,
    )
    if satellite_aliases:
        for source_id, target_id in satellite_aliases.items():
            canonical_extent_labels[canonical_extent_labels == source_id] = target_id
            canonical_core_labels[canonical_core_labels == source_id] = target_id
    canonical_extent_labels = np.where(
        np.isin(canonical_extent_labels, canonical_ids),
        canonical_extent_labels,
        0,
    ).astype(np.uint32)
    canonical_core_labels = np.where(
        np.isin(canonical_core_labels, canonical_ids),
        canonical_core_labels,
        0,
    ).astype(np.uint32)

    original_ids = list(range(1, int(whole_labels.max()) + 1))
    parent = {astrocyte_id: astrocyte_id for astrocyte_id in original_ids}

    def find(astrocyte_id: int) -> int:
        while parent[astrocyte_id] != astrocyte_id:
            parent[astrocyte_id] = parent[parent[astrocyte_id]]
            astrocyte_id = parent[astrocyte_id]
        return astrocyte_id

    def union(left_id: int, right_id: int) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    pixel_area_um2 = pixel_width_um * pixel_height_um
    minimum_overlap_px = max(
        1,
        int(math.ceil(cfg.minimum_extent_overlap_um2 / pixel_area_um2)),
    )
    for canonical_id in canonical_ids:
        extent = canonical_extent_labels == int(canonical_id)
        extent_area = int(extent.sum())
        if extent_area == 0:
            continue
        overlaps = np.bincount(
            whole_labels[extent],
            minlength=int(whole_labels.max()) + 1,
        )
        source_ids = [
            astrocyte_id
            for astrocyte_id in original_ids
            if int(overlaps[astrocyte_id]) >= minimum_overlap_px
            and float(overlaps[astrocyte_id]) / extent_area
            >= cfg.minimum_extent_overlap_fraction
        ]
        if len(source_ids) < 2 or len(source_ids) > cfg.maximum_merge_source_count:
            continue
        accepted_pairs = []
        for left_index, left_id in enumerate(source_ids):
            left = whole_labels == left_id
            contact_distance = ndi.distance_transform_edt(
                ~left,
                sampling=(pixel_height_um, pixel_width_um),
            )
            for right_id in source_ids[left_index + 1 :]:
                right = whole_labels == right_id
                if not np.any(right & (contact_distance <= cfg.merge_contact_distance_um)):
                    continue
                union(left_id, right_id)
                accepted_pairs.append([left_id, right_id])
        if accepted_pairs:
            metrics["merge_decisions"].append(
                {
                    "canonical_nucleus_id": int(canonical_id),
                    "source_astrocyte_ids": source_ids,
                    "accepted_pairs": accepted_pairs,
                    "extent_overlap_px": {
                        str(value): int(overlaps[value]) for value in source_ids
                    },
                }
            )

    root_to_sources: dict[int, list[int]] = {}
    for astrocyte_id in original_ids:
        root_to_sources.setdefault(find(astrocyte_id), []).append(astrocyte_id)
    merged_labels = np.zeros_like(whole_labels, dtype=np.uint16)
    merged_soma = np.zeros_like(soma_labels, dtype=np.uint16)
    merged_sources: dict[int, list[int]] = {}
    for merged_id, root in enumerate(sorted(root_to_sources), start=1):
        sources = sorted(root_to_sources[root])
        merged_sources[merged_id] = sources
        source_mask = np.isin(whole_labels, sources)
        merged_labels[source_mask] = merged_id
        merged_soma[np.isin(soma_labels, sources)] = merged_id
    metrics["merge_count"] = sum(len(values) > 1 for values in merged_sources.values())

    accepted_core = canonical_core_labels > 0
    accepted_extent = canonical_extent_labels > 0
    canonical_inventory = ValidatedNucleusAnchors(
        accepted_core_mask_2d=accepted_core,
        accepted_extent_mask_2d=accepted_extent,
        metrics={"status": "canonical_identity_reconciliation"},
        object_core_labels_2d=canonical_core_labels,
        object_extent_labels_2d=canonical_extent_labels,
        dapi_valid_object_ids=tuple(int(value) for value in canonical_ids),
        accepted_object_ids=tuple(int(value) for value in canonical_ids),
        object_records=tuple(canonical_records),
        nucleus_instance_core_labels_2d=canonical_core_labels,
        nucleus_instance_extent_labels_2d=canonical_extent_labels,
        accepted_instance_ids=tuple(int(value) for value in canonical_ids),
        nucleus_instance_records=tuple(canonical_records),
    )
    ownership_cfg = replace(
        NucleusOwnershipConfig(),
        accepted_min_extent_overlap_fraction=cfg.split_minimum_extent_overlap_fraction,
        unowned_min_volume_um3=math.inf,
        unowned_min_enclosure_score=math.inf,
    )
    reconciled_whole, partition_metrics = apply_nucleus_ownership_guard(
        merged_labels,
        struct,
        canonical_inventory,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
        profile=f"{profile}_canonical_reconciliation",
        config=ownership_cfg,
        prefer_canonical=True,
    )
    metrics["partition_decisions"] = partition_metrics.get("decisions", [])
    metrics["split_count"] = int(partition_metrics.get("split_component_count", 0))
    metrics["fail_closed_count"] = int(
        partition_metrics.get("fail_closed_component_count", 0)
    )
    input_to_output_ids = {
        int(input_id): [int(value) for value in output_ids]
        for input_id, output_ids in partition_metrics.get(
            "input_to_output_ids", {}
        ).items()
    }

    output_soma = np.zeros_like(reconciled_whole, dtype=np.uint16)
    output_process = np.zeros_like(reconciled_whole, dtype=np.uint16)
    final_lineage: dict[int, dict] = {}
    changed_merged_ids = {
        merged_id
        for merged_id, sources in merged_sources.items()
        if len(sources) > 1 or len(input_to_output_ids.get(merged_id, [])) != 1
    }
    for final_id in range(1, int(reconciled_whole.max()) + 1):
        component = reconciled_whole == final_id
        source_merged_ids = [
            merged_id
            for merged_id, output_ids in input_to_output_ids.items()
            if final_id in output_ids
        ]
        if not source_merged_ids:
            source_merged_ids = sorted(
                int(value)
                for value in np.unique(merged_labels[component])
                if int(value) > 0
            )
        source_ids = sorted(
            {
                source_id
                for merged_id in source_merged_ids
                for source_id in merged_sources.get(merged_id, [])
            }
        )
        soma = component & np.isin(soma_labels, source_ids)
        canonical_overlaps = np.bincount(
            canonical_extent_labels[component],
            minlength=int(canonical_extent_labels.max()) + 1,
        )
        owner_id = int(np.argmax(canonical_overlaps[1:]) + 1) if canonical_overlaps[1:].any() else 0
        was_split = any(
            len(input_to_output_ids.get(merged_id, [])) > 1
            for merged_id in source_merged_ids
        )
        if was_split and owner_id > 0:
            soma |= component & (canonical_extent_labels == owner_id)
        if not soma.any() and owner_id > 0:
            soma = component & (canonical_extent_labels == owner_id)
        if not soma.any():
            continue
        output_soma[soma] = final_id
        output_process[component & ~soma] = final_id
        final_lineage[final_id] = {
            "source_astrocyte_ids": source_ids,
            "source_merged_ids": source_merged_ids,
            "canonical_owner_id": owner_id,
            "identity_changed": bool(
                any(value in changed_merged_ids for value in source_merged_ids)
            ),
        }

    retained_ids = sorted(final_lineage)
    if not retained_ids:
        raise RuntimeError("Canonical identity reconciliation removed every Astrocyte ROI")
    output_whole, output_soma, output_process, final_mapping = (
        relabel_compartment_triplet(
            reconciled_whole,
            output_soma,
            output_process,
            retained_ids,
        )
    )
    metrics["final_lineage"] = {
        str(final_mapping[old_id]): row for old_id, row in final_lineage.items()
    }
    metrics["changed_input_ids"] = sorted(
        {
            source_id
            for row in metrics["final_lineage"].values()
            if bool(row["identity_changed"])
            for source_id in row["source_astrocyte_ids"]
        }
    )
    metrics["post_roi_count"] = int(output_whole.max())
    if np.any((output_whole > 0) & ~(whole_labels > 0)):
        raise RuntimeError("Canonical identity reconciliation expanded frozen Whole geometry")
    if np.any((output_soma > 0) & (output_process > 0)):
        raise RuntimeError("Canonical identity reconciliation overlapped compartments")
    if not np.array_equal(
        output_whole > 0,
        (output_soma > 0) | (output_process > 0),
    ):
        raise RuntimeError("Canonical identity reconciliation broke the partition")
    return output_whole, output_soma, output_process, metrics

def apply_axial_truncation_guard(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    context: Neonatal3DContext | None,
    pixel_width_um: float,
    pixel_height_um: float,
    config: AxialTruncationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Delete cells whose owner nucleus is demonstrably truncated by selected Z faces."""

    cfg = config or AxialTruncationConfig()
    metrics = {
        "enabled": bool(cfg.enabled and inventory is not None and context is not None),
        "method": (
            "selected-Z cuboid guard using owner-envelope boundary contact and "
            "connected DAPI continuation in raw guard slices"
        ),
        "evaluated_cell_count": int(whole_labels.max()),
        "removed_cell_count": 0,
        "removed_pre_guard_ids": [],
        "id_mapping": {},
        "decisions": [],
        "config": asdict(cfg),
    }
    if not metrics["enabled"]:
        return whole_labels, soma_labels, process_labels, metrics
    assert context is not None
    groups, grouped_extents = inventory_group_geometry(
        inventory,
        pixel_width_um,
        pixel_height_um,
        context.pixel_depth_um,
    )
    if not groups or grouped_extents is None:
        return whole_labels, soma_labels, process_labels, metrics

    z0 = int(context.z_start_0based)
    z1 = int(context.z_end_0based_inclusive)
    stack_depth = int(context.dapi_stack.shape[0])
    band_slices = max(1, int(math.ceil(cfg.boundary_band_um / context.pixel_depth_um)))
    guard_slices = max(1, int(math.ceil(cfg.guard_depth_um / context.pixel_depth_um)))
    pixel_area_um2 = pixel_width_um * pixel_height_um
    voxel_volume_um3 = pixel_area_um2 * context.pixel_depth_um
    per_object_rows = {
        int(row["object_id_3d"]): row
        for row in inventory.metrics.get("per_nucleus", [])
    }

    owners: dict[int, dict] = {}
    owner_volumes: list[float] = []
    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        owner = owner_group_for_soma(
            soma_labels == astrocyte_id,
            groups,
            grouped_extents,
        )
        if owner is not None:
            owners[astrocyte_id] = owner
            owner_volumes.append(float(owner["volume_um3"]))
    reference_volume = (
        float(np.median(owner_volumes))
        if len(owner_volumes) >= cfg.min_reference_nuclei
        else 0.0
    )

    removed_ids: set[int] = set()
    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        owner = owners.get(astrocyte_id)
        decision = {
            "pre_guard_astrocyte_id": astrocyte_id,
            "owner_group_id": int(owner["group_id"]) if owner else 0,
            "status": "retained",
            "faces": [],
        }
        if owner is None:
            decision["status"] = "not_evaluated_owner_unavailable"
            metrics["decisions"].append(decision)
            continue
        group_id = int(owner["group_id"])
        extent = grouped_extents == group_id
        if not extent.any():
            decision["status"] = "not_evaluated_extent_unavailable"
            metrics["decisions"].append(decision)
            continue
        rows, cols = np.nonzero(extent)
        halo_y = max(1, int(math.ceil(0.45 / pixel_height_um)))
        halo_x = max(1, int(math.ceil(0.45 / pixel_width_um)))
        row0 = max(0, int(rows.min()) - halo_y)
        row1 = min(extent.shape[0], int(rows.max()) + halo_y + 1)
        col0 = max(0, int(cols.min()) - halo_x)
        col1 = min(extent.shape[1], int(cols.max()) + halo_x + 1)
        local_extent = extent[row0:row1, col0:col1]
        support_distance = ndi.distance_transform_edt(
            ~local_extent,
            sampling=(pixel_height_um, pixel_width_um),
        )
        support = support_distance <= 0.45
        thresholds = [
            float(per_object_rows[object_id]["dapi_low_threshold"])
            for object_id in owner["object_ids"]
            if object_id in per_object_rows
            and per_object_rows[object_id].get("dapi_low_threshold") is not None
        ]
        if not thresholds and owner.get("dapi_low_threshold") is not None:
            thresholds = [float(owner["dapi_low_threshold"])]
        if not thresholds:
            decision["status"] = "not_evaluated_threshold_unavailable"
            metrics["decisions"].append(decision)
            continue
        threshold = float(np.median(thresholds))
        inside_volume = float(owner["volume_um3"])
        relative_volume = (
            inside_volume / reference_volume if reference_volume > 0 else math.nan
        )
        faces = []
        if z0 > 0 and int(owner["z_min_0based"]) <= z0 + band_slices - 1:
            faces.append(("front", max(0, z0 - guard_slices), z0 + band_slices, z0))
        if z1 < stack_depth - 1 and int(owner["z_max_0based_inclusive"]) >= z1 - band_slices + 1:
            faces.append(("back", z1 - band_slices + 1, min(stack_depth, z1 + guard_slices + 1), z1 + 1))

        reject = False
        for face_name, slab_start, slab_end, outside_boundary in faces:
            slab = context.dapi_stack[
                slab_start:slab_end,
                row0:row1,
                col0:col1,
            ].astype(np.float32, copy=False)
            binary = (slab >= threshold) & support[None, :, :]
            labelled = measure.label(binary, connectivity=3)
            split_index = outside_boundary - slab_start
            if face_name == "front":
                outside_selector = np.arange(binary.shape[0]) < split_index
                boundary_indices = range(split_index, min(binary.shape[0], split_index + band_slices))
            else:
                outside_selector = np.arange(binary.shape[0]) >= split_index
                boundary_indices = range(max(0, split_index - band_slices), split_index)
            boundary_ids: set[int] = set()
            for index in boundary_indices:
                boundary_ids.update(int(value) for value in np.unique(labelled[index]) if value > 0)
            outside_counts = {
                component_id: int(
                    ((labelled == component_id) & outside_selector[:, None, None]).sum()
                )
                for component_id in boundary_ids
            }
            connected_outside_voxels = max(outside_counts.values(), default=0)
            outside_volume = connected_outside_voxels * voxel_volume_um3
            outside_ratio = outside_volume / max(inside_volume, 1e-9)
            inside_fraction = inside_volume / max(inside_volume + outside_volume, 1e-9)
            boundary_area_px = 0
            for index in boundary_indices:
                boundary_area_px = max(
                    boundary_area_px,
                    int(np.isin(labelled[index], list(boundary_ids)).sum()),
                )
            boundary_area_ratio = (
                boundary_area_px * pixel_area_um2
                / max(int(extent.sum()) * pixel_area_um2, 1e-9)
            )
            face_decision = {
                "face": face_name,
                "outside_volume_um3": float(outside_volume),
                "outside_to_inside_ratio": float(outside_ratio),
                "inside_volume_fraction": float(inside_fraction),
                "boundary_area_ratio": float(boundary_area_ratio),
            }
            continuation = bool(
                outside_volume >= cfg.min_outside_continuation_um3
                and outside_ratio >= cfg.min_outside_to_inside_ratio
                and boundary_area_ratio >= cfg.min_boundary_area_ratio
            )
            small_or_incomplete = bool(
                inside_fraction < cfg.min_inside_volume_fraction
                or float(owner["z_max_0based_inclusive"] - owner["z_min_0based"] + 1)
                * context.pixel_depth_um
                < cfg.min_inside_z_span_um
                or (
                    np.isfinite(relative_volume)
                    and relative_volume < cfg.minimum_relative_volume
                )
            )
            face_decision["continuation_confirmed"] = continuation
            face_decision["small_or_incomplete"] = small_or_incomplete
            decision["faces"].append(face_decision)
            reject |= continuation and small_or_incomplete
        decision["inside_volume_um3"] = inside_volume
        decision["relative_owner_volume"] = float(relative_volume)
        if reject:
            removed_ids.add(astrocyte_id)
            decision["status"] = "removed_confirmed_axial_truncation"
        metrics["decisions"].append(decision)

    retained_ids = [
        astrocyte_id
        for astrocyte_id in range(1, int(whole_labels.max()) + 1)
        if astrocyte_id not in removed_ids
    ]
    if not retained_ids:
        raise RuntimeError("Axial truncation guard would remove every Astrocyte ROI")
    outputs = relabel_compartment_triplet(
        whole_labels,
        soma_labels,
        process_labels,
        retained_ids,
    )
    out_whole, out_soma, out_process, mapping = outputs
    metrics["removed_cell_count"] = len(removed_ids)
    metrics["removed_pre_guard_ids"] = sorted(removed_ids)
    metrics["id_mapping"] = {str(key): value for key, value in mapping.items()}
    return out_whole, out_soma, out_process, metrics

def apply_projected_foreign_soma_guard(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
    config: NucleusOwnershipConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove only true distal tips ending inside another cell's DAPI projection."""

    cfg = config or NucleusOwnershipConfig()
    metrics = {
        "enabled": bool(cfg.projection_occlusion_enabled and inventory is not None),
        "method": (
            "topology-gated distal-tip exclusion at the exact foreign DAPI projection "
            "boundary; no fixed terminal length, halo, or ROI expansion"
        ),
        "evaluated_cell_count": int(whole_labels.max()),
        "changed_cell_count": 0,
        "removed_area_px": 0,
        "true_tip_count": 0,
        "terminal_overlap_component_count": 0,
        "preserved_pass_through_component_count": 0,
        "connectivity_rollback_count": 0,
        "decisions": [],
        "config": asdict(cfg),
    }
    groups, grouped_extents = inventory_group_geometry(
        inventory,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
        cfg,
    )
    if not metrics["enabled"] or not groups or grouped_extents is None:
        return whole_labels, soma_labels, process_labels, metrics
    pixel_area_um2 = pixel_width_um * pixel_height_um
    output_whole = np.zeros_like(whole_labels, dtype=np.uint16)
    output_soma = soma_labels.copy().astype(np.uint16)
    output_process = np.zeros_like(process_labels, dtype=np.uint16)

    eligible_groups = [
        group
        for group in groups
        if float(group["volume_um3"]) >= cfg.projection_min_foreign_volume_um3
        and (
            int(group["z_max_0based_inclusive"]) - int(group["z_min_0based"]) + 1
        ) * float(pixel_depth_um)
        >= cfg.projection_min_foreign_z_span_um
        and int((grouped_extents == int(group["group_id"])).sum()) * pixel_area_um2
        >= cfg.projection_min_foreign_extent_um2
    ]
    eligible_extents = {
        int(group["group_id"]): grouped_extents == int(group["group_id"])
        for group in eligible_groups
    }
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    connectivity = np.ones((3, 3), dtype=np.uint8)
    image_edge = np.zeros_like(whole_labels, dtype=bool)
    image_edge[[0, -1], :] = True
    image_edge[:, [0, -1]] = True

    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        component = whole_labels == astrocyte_id
        soma = soma_labels == astrocyte_id
        process = process_labels == astrocyte_id
        owner = owner_group_for_soma(soma, groups, grouped_extents)
        owner_group_id = int(owner["group_id"]) if owner is not None else 0
        foreign_ids: list[int] = []
        removed_foreign_ids: list[int] = []
        remove_mask = np.zeros_like(component, dtype=bool)

        component_labels = measure.label(component, connectivity=2)
        soma_root_ids = np.unique(component_labels[soma])
        soma_root_ids = soma_root_ids[soma_root_ids > 0]
        soma_rooted_whole = np.isin(component_labels, soma_root_ids)
        skeleton = morphology.skeletonize(soma_rooted_whole)
        neighbor_count = ndi.convolve(
            skeleton.astype(np.uint8),
            neighbor_kernel,
            mode="constant",
            cval=0,
        )
        soma_adjacent = morphology.binary_dilation(
            soma,
            footprint=np.ones((3, 3), dtype=bool),
        )
        true_tips = (
            skeleton
            & process
            & (neighbor_count == 1)
            & ~soma_adjacent
            & ~image_edge
        )
        true_tip_count = int(true_tips.sum())
        metrics["true_tip_count"] += true_tip_count
        terminal_components = 0
        preserved_pass_through = 0

        for group_id, extent in eligible_extents.items():
            if group_id == owner_group_id:
                continue
            overlap = process & extent
            if not np.any(overlap):
                continue
            foreign_ids.append(group_id)
            overlap_labels, overlap_count = ndi.label(
                overlap,
                structure=connectivity,
            )
            tip_labels = np.unique(overlap_labels[true_tips & extent])
            tip_labels = tip_labels[tip_labels > 0]
            for overlap_id in tip_labels:
                overlap_component = overlap_labels == int(overlap_id)
                skeleton_inside = skeleton & overlap_component
                if not np.any(skeleton_inside):
                    continue
                skeleton_entry_pixels = (
                    ndi.binary_dilation(skeleton_inside, structure=connectivity)
                    & skeleton
                    & process
                    & ~extent
                )
                _, entry_count = ndi.label(
                    skeleton_entry_pixels,
                    structure=connectivity,
                )
                contains_branchpoint = bool(
                    np.any(skeleton_inside & (neighbor_count >= 3))
                )
                if entry_count != 1 or contains_branchpoint:
                    preserved_pass_through += 1
                    continue
                remove_mask |= overlap_component
                terminal_components += 1
                removed_foreign_ids.append(group_id)

        retained = soma | (process & ~remove_mask)
        before_component_count = int(measure.label(component, connectivity=2).max())
        after_component_count = int(measure.label(retained, connectivity=2).max())
        connectivity_rollback = after_component_count > before_component_count
        if connectivity_rollback:
            retained = component
            remove_mask.fill(False)
            terminal_components = 0
            removed_foreign_ids = []
            metrics["connectivity_rollback_count"] += 1

        removed = int(component.sum() - retained.sum())
        output_whole[retained] = astrocyte_id
        output_soma[soma] = astrocyte_id
        output_process[retained & ~soma] = astrocyte_id
        metrics["decisions"].append(
            {
                "astrocyte_id": astrocyte_id,
                "owner_group_id": owner_group_id,
                "foreign_group_ids": foreign_ids,
                "removed_foreign_group_ids": sorted(set(removed_foreign_ids)),
                "true_tip_count": true_tip_count,
                "terminal_overlap_component_count": terminal_components,
                "preserved_pass_through_component_count": preserved_pass_through,
                "connectivity_rollback": connectivity_rollback,
                "removed_area_px": removed,
                "status": (
                    "terminal_projection_overlap_pruned"
                    if removed
                    else (
                        "connectivity_rollback"
                        if connectivity_rollback
                        else "unchanged"
                    )
                ),
            }
        )
        metrics["changed_cell_count"] += int(removed > 0)
        metrics["removed_area_px"] += removed
        metrics["terminal_overlap_component_count"] += terminal_components
        metrics["preserved_pass_through_component_count"] += preserved_pass_through
    if np.any((output_whole > 0) & ~(whole_labels > 0)):
        raise RuntimeError("Projected foreign-soma guard expanded Whole geometry")
    if np.any((output_soma > 0) & ~(output_whole > 0)):
        raise RuntimeError("Projected foreign-soma guard removed assigned Soma pixels")
    if np.any((output_soma > 0) & (output_process > 0)):
        raise RuntimeError("Projected foreign-soma guard overlapped Soma and Processes")
    if not np.array_equal(output_whole > 0, (output_soma > 0) | (output_process > 0)):
        raise RuntimeError("Projected foreign-soma guard broke the compartment partition")
    return output_whole, output_soma, output_process, metrics

def _complete_soma_within_whole_owner_extent(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
    config: SomaNuclearCompletionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fill only small missing parts of an already assigned 3D nuclear envelope."""

    cfg = config or SomaNuclearCompletionConfig()
    metrics = {
        "enabled": bool(cfg.enabled and inventory is not None),
        "method": (
            "owner-only canonical nuclear-envelope completion inside frozen Whole; "
            "Processes are recomputed as Whole minus Soma"
        ),
        "evaluated_cell_count": int(whole_labels.max()),
        "changed_cell_count": 0,
        "added_soma_px": 0,
        "decisions": [],
        "config": asdict(cfg),
    }
    groups, grouped_extents = inventory_group_geometry(
        inventory,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
    )
    if not metrics["enabled"] or not groups or grouped_extents is None:
        return whole_labels, soma_labels, process_labels, metrics

    pixel_area_um2 = pixel_width_um * pixel_height_um
    minimum_owner_overlap_px = max(
        1,
        int(math.ceil(cfg.minimum_owner_overlap_um2 / pixel_area_um2)),
    )
    output_soma = soma_labels.copy().astype(np.uint16)
    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        whole = whole_labels == astrocyte_id
        soma = output_soma == astrocyte_id
        owner = owner_group_for_soma(soma, groups, grouped_extents)
        decision = {
            "astrocyte_id": astrocyte_id,
            "owner_group_id": int(owner["group_id"]) if owner is not None else 0,
            "status": "unchanged",
            "owner_extent_inside_whole_px": 0,
            "existing_extent_coverage": 0.0,
            "added_soma_px": 0,
        }
        if owner is None or str(owner.get("identity_status", "")) != "resolved":
            decision["status"] = "skipped_no_resolved_owner"
            metrics["decisions"].append(decision)
            continue
        owner_extent = grouped_extents == int(owner["group_id"])
        inside = owner_extent & whole
        inside_area = int(inside.sum())
        covered_area = int((inside & soma).sum())
        missing = inside & ~soma
        missing_area = int(missing.sum())
        coverage = covered_area / max(inside_area, 1)
        decision["owner_extent_inside_whole_px"] = inside_area
        decision["existing_extent_coverage"] = coverage
        decision["added_soma_px"] = missing_area
        if inside_area < minimum_owner_overlap_px or covered_area < minimum_owner_overlap_px:
            decision["status"] = "skipped_insufficient_owner_overlap"
        elif coverage < cfg.minimum_existing_extent_coverage:
            decision["status"] = "skipped_extent_not_already_soma_owned"
        elif missing_area > cfg.maximum_added_fraction_of_existing_soma * max(
            int(soma.sum()),
            1,
        ):
            decision["status"] = "skipped_completion_too_large"
        elif missing_area:
            output_soma[missing] = astrocyte_id
            decision["status"] = "completed_owner_nuclear_extent"
            metrics["changed_cell_count"] += 1
            metrics["added_soma_px"] += missing_area
        metrics["decisions"].append(decision)

    output_process = np.where(
        (whole_labels > 0) & (output_soma == 0),
        whole_labels,
        0,
    ).astype(np.uint16)
    if not np.array_equal(whole_labels > 0, (output_soma > 0) | (output_process > 0)):
        raise RuntimeError("Soma nuclear-envelope completion broke the compartment partition")
    if np.any((output_soma > 0) & (output_process > 0)):
        raise RuntimeError("Soma nuclear-envelope completion overlapped Soma and Processes")
    if np.any((output_soma > 0) & ~(whole_labels > 0)):
        raise RuntimeError("Soma nuclear-envelope completion expanded outside Whole")
    return whole_labels, output_soma, output_process, metrics

def resolve_canonical_owner_assignments(
    identity_metrics: dict,
    axial_metrics: dict,
    inventory: ValidatedNucleusAnchors | None,
    final_roi_count: int,
) -> tuple[dict[int, dict], dict[int, list[str]]]:
    """Resolve one canonical, axially retained 3D owner for each final cell."""

    assignments: dict[int, dict] = {}
    failures: dict[int, list[str]] = {}

    def reject(astrocyte_id: int, reason: str) -> None:
        failures.setdefault(int(astrocyte_id), []).append(str(reason))

    if inventory is None:
        for astrocyte_id in range(1, int(final_roi_count) + 1):
            reject(astrocyte_id, "owner_inventory_missing")
        return assignments, failures

    records = {
        int(row["instance_id"]): row
        for row in inventory.nucleus_instance_records
    }
    accepted_ids = {
        int(value) for value in inventory.accepted_instance_ids
    }
    identity_lineage = {
        int(key): value
        for key, value in identity_metrics.get("final_lineage", {}).items()
    }
    axial_mapping = {
        int(old_id): int(new_id)
        for old_id, new_id in axial_metrics.get("id_mapping", {}).items()
    }
    inverse_axial_mapping: dict[int, int] = {}
    duplicate_final_ids: set[int] = set()
    for pre_axial_id, final_id in axial_mapping.items():
        if final_id in inverse_axial_mapping:
            duplicate_final_ids.add(final_id)
        else:
            inverse_axial_mapping[final_id] = pre_axial_id
    axial_decisions = {
        int(row["pre_guard_astrocyte_id"]): row
        for row in axial_metrics.get("decisions", [])
        if int(row.get("pre_guard_astrocyte_id", 0)) > 0
    }
    lineage_owner_claims: dict[int, list[int]] = {}
    for astrocyte_id in range(1, int(final_roi_count) + 1):
        pre_axial_id = inverse_axial_mapping.get(astrocyte_id)
        lineage = (
            identity_lineage.get(pre_axial_id)
            if pre_axial_id is not None
            else None
        )
        owner_id = (
            int(lineage.get("canonical_owner_id", 0))
            if lineage is not None
            else 0
        )
        if owner_id > 0:
            lineage_owner_claims.setdefault(owner_id, []).append(
                astrocyte_id
            )
    duplicate_owner_claim_cells = {
        astrocyte_id
        for cell_ids in lineage_owner_claims.values()
        if len(cell_ids) > 1
        for astrocyte_id in cell_ids
    }

    for astrocyte_id in range(1, int(final_roi_count) + 1):
        if astrocyte_id in duplicate_final_ids:
            reject(astrocyte_id, "axial_mapping_not_unique")
            continue
        pre_axial_id = inverse_axial_mapping.get(astrocyte_id)
        if pre_axial_id is None:
            reject(astrocyte_id, "axial_mapping_missing")
            continue
        lineage = identity_lineage.get(pre_axial_id)
        if lineage is None:
            reject(astrocyte_id, "owner_lineage_missing")
            continue
        owner_id = int(lineage.get("canonical_owner_id", 0))
        if owner_id <= 0:
            reject(astrocyte_id, "canonical_owner_missing")
            continue
        if astrocyte_id in duplicate_owner_claim_cells:
            reject(astrocyte_id, "owner_assignment_not_unique")
            continue
        record = records.get(owner_id)
        if record is None:
            reject(astrocyte_id, "owner_record_missing")
            continue
        if not bool(record.get("dapi_valid", False)):
            reject(astrocyte_id, "owner_not_dapi_valid")
            continue
        if str(record.get("identity_status", "")) != "resolved":
            reject(astrocyte_id, "owner_not_resolved")
            continue
        if not bool(record.get("accepted", False)):
            reject(astrocyte_id, "owner_not_accepted")
            continue
        if owner_id not in accepted_ids:
            reject(astrocyte_id, "owner_not_in_accepted_ids")
            continue
        axial_decision = axial_decisions.get(pre_axial_id)
        if axial_decision is None:
            reject(astrocyte_id, "axial_decision_missing")
            continue
        axial_owner_id = int(axial_decision.get("owner_group_id", 0))
        if axial_owner_id != owner_id:
            reject(astrocyte_id, "axial_owner_mismatch")
            continue
        if str(axial_decision.get("status", "")) != "retained":
            reject(astrocyte_id, "axial_owner_not_retained")
            continue
        assignments[astrocyte_id] = {
            "astrocyte_id": astrocyte_id,
            "pre_axial_id": pre_axial_id,
            "owner_id": owner_id,
            "lineage_owner_id": owner_id,
            "axial_owner_id": axial_owner_id,
            "owner_record": record,
            "identity_changed": bool(
                lineage.get("identity_changed", False)
            ),
        }

    return assignments, failures

def complete_soma_to_owner_nuclear_extent(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
    owner_assignments: dict[int, dict],
    owner_assignment_failures: dict[int, list[str]],
    pre_finalization_whole_union: np.ndarray,
    config: SomaNuclearCompletionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """Complete only an unambiguous canonical owner extent, atomically."""

    cfg = config or SomaNuclearCompletionConfig()
    if (
        whole_labels.shape != soma_labels.shape
        or whole_labels.shape != process_labels.shape
    ):
        raise ValueError("Canonical Owner Nuclear-Extent Completion compartment label shapes do not match")
    frozen_pre_finalization_whole = np.asarray(
        pre_finalization_whole_union,
        dtype=bool,
    )
    if frozen_pre_finalization_whole.shape != whole_labels.shape:
        raise ValueError(
            "Canonical Owner Nuclear-Extent Completion pre-finalization Whole shape "
            "does not match compartment labels"
        )
    original_ids = set(int(value) for value in np.unique(whole_labels))
    metrics = {
        "enabled": bool(cfg.enabled and inventory is not None),
        "method": (
            "exact canonical owner-nucleus extent completion after identity and "
            "axial consensus; no dilation, closing, hull, or intensity voting"
        ),
        "pre_finalization_whole_area_px": int(
            frozen_pre_finalization_whole.sum()
        ),
        "pre_canonical_owner_extent_completion_whole_area_px": int(
            (whole_labels > 0).sum()
        ),
        "post_canonical_owner_extent_completion_whole_area_px": int(
            (whole_labels > 0).sum()
        ),
        "pre_finalization_whole_to_final_removed_px": 0,
        "pre_finalization_whole_to_final_approved_added_px": 0,
        "evaluated_cell_count": int(whole_labels.max()),
        "eligible_cell_count": 0,
        "changed_cell_count": 0,
        "no_op_cell_count": 0,
        "fail_closed_cell_count": 0,
        "inside_added_soma_px": 0,
        "outside_added_whole_px": 0,
        "outside_added_soma_px": 0,
        "changed_cell_ids": [],
        "outside_changed_cell_ids": [],
        "no_op_cell_ids": [],
        "fail_closed_cell_ids": [],
        "decisions": [],
        "config": asdict(cfg),
    }
    approved_outside = np.zeros(whole_labels.shape, dtype=bool)
    if not metrics["enabled"]:
        for astrocyte_id in range(1, int(whole_labels.max()) + 1):
            metrics["decisions"].append(
                {
                    "astrocyte_id": astrocyte_id,
                    "status": "fail_closed_owner_assignment_missing",
                    "fail_reasons": ["owner_inventory_missing"],
                }
            )
            metrics["fail_closed_cell_ids"].append(astrocyte_id)
        metrics["fail_closed_cell_count"] = len(
            metrics["fail_closed_cell_ids"]
        )
        return (
            whole_labels,
            soma_labels,
            process_labels,
            metrics,
            approved_outside,
        )

    groups, grouped_extents = inventory_group_geometry(
        inventory,
        pixel_width_um,
        pixel_height_um,
        pixel_depth_um,
    )
    if grouped_extents is None:
        owner_assignment_failures = {
            astrocyte_id: ["owner_extent_inventory_missing"]
            for astrocyte_id in range(1, int(whole_labels.max()) + 1)
        }
        owner_assignments = {}
    group_by_id = {
        int(group["group_id"]): group for group in groups
    }
    pixel_area_um2 = float(pixel_width_um * pixel_height_um)
    minimum_owner_overlap_px = max(
        1,
        int(math.ceil(cfg.minimum_owner_overlap_um2 / pixel_area_um2)),
    )
    plans: dict[int, dict] = {}
    decisions: dict[int, dict] = {}

    def failure_status(reasons: list[str]) -> str:
        if "owner_assignment_not_unique" in reasons:
            return "fail_closed_owner_assignment_not_unique"
        if "axial_owner_mismatch" in reasons:
            return "fail_closed_axial_owner_mismatch"
        if any(reason.startswith("axial_") for reason in reasons):
            return "fail_closed_axial_status_unverified"
        if any(
            reason in {
                "owner_not_accepted",
                "owner_not_resolved",
                "owner_not_dapi_valid",
                "owner_not_in_accepted_ids",
                "owner_record_missing",
            }
            for reason in reasons
        ):
            return "fail_closed_owner_not_accepted_resolved"
        return "fail_closed_owner_assignment_missing"

    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        assignment = owner_assignments.get(astrocyte_id)
        assignment_reasons = list(
            owner_assignment_failures.get(astrocyte_id, [])
        )
        decision = {
            "astrocyte_id": astrocyte_id,
            "pre_axial_id": int(
                assignment.get("pre_axial_id", 0)
                if assignment is not None
                else 0
            ),
            "lineage_owner_id": int(
                assignment.get("lineage_owner_id", 0)
                if assignment is not None
                else 0
            ),
            "axial_owner_id": int(
                assignment.get("axial_owner_id", 0)
                if assignment is not None
                else 0
            ),
            "owner_group_id": int(
                assignment.get("owner_id", 0)
                if assignment is not None
                else 0
            ),
            "owner_assignment_unique": bool(
                assignment is not None and not assignment_reasons
            ),
            "accepted_resolved_soma_candidate_ids": [],
            "resolved_nucleus_ids_in_whole": [],
            "foreign_resolved_nucleus_ids": [],
            "incidental_resolved_nucleus_ids_in_whole": [],
            "foreign_resolved_nucleus_evidence": [],
            "owner_extent_total_px": 0,
            "owner_extent_inside_current_whole_px": 0,
            "owner_extent_current_background_px": 0,
            "owner_extent_outside_pre_finalization_whole_px": 0,
            "prior_guard_removed_owner_px": 0,
            "existing_soma_owner_overlap_px": 0,
            "missing_inside_px": 0,
            "foreign_whole_overlap_px": 0,
            "foreign_soma_overlap_px": 0,
            "owner_extent_component_count": 0,
            "owner_anchor_component_count": 0,
            "selected_owner_extent_component_count": 0,
            "selected_owner_extent_component_ids": [],
            "selected_owner_extent_component_px": 0,
            "selected_owner_core_overlap_px": 0,
            "selected_owner_soma_overlap_px": 0,
            "selected_owner_core_component_count": 0,
            "selected_owner_substantial_core_component_count": 0,
            "selected_owner_substantial_core_component_ids": [],
            "selected_owner_core_component_details": [],
            "ignored_owner_extent_component_count": 0,
            "ignored_owner_extent_px": 0,
            "owner_extent_component_details": [],
            "owner_extent_xy_edge_touch": False,
            "approved_inside_soma_px": 0,
            "approved_outside_whole_px": 0,
            "whole_delta_px": 0,
            "soma_delta_px": 0,
            "process_delta_px": 0,
            "approved": False,
            "approved_owner_extent_added_px": 0,
            "status": "unchanged",
            "fail_reasons": assignment_reasons,
        }
        decisions[astrocyte_id] = decision
        if assignment is None or assignment_reasons:
            decision["status"] = failure_status(assignment_reasons)
            continue

        owner_id = int(assignment["owner_id"])
        owner_group = group_by_id.get(owner_id)
        if owner_group is None:
            decision["fail_reasons"].append("owner_extent_record_missing")
            decision["status"] = "fail_closed_owner_assignment_missing"
            continue
        full_owner_extent = np.asarray(
            grouped_extents == owner_id,
            dtype=bool,
        )
        if (
            inventory.nucleus_instance_core_labels_2d is not None
            and np.asarray(
                inventory.nucleus_instance_core_labels_2d
            ).shape
            == full_owner_extent.shape
        ):
            owner_core = np.asarray(
                inventory.nucleus_instance_core_labels_2d == owner_id,
                dtype=bool,
            )
        else:
            owner_core = np.zeros_like(full_owner_extent, dtype=bool)
        own_whole = whole_labels == astrocyte_id
        own_soma = soma_labels == astrocyte_id
        component_labels = measure.label(
            full_owner_extent,
            connectivity=2,
        )
        component_count = int(component_labels.max())
        owner_soma_distance_um = ndi.distance_transform_edt(
            ~own_soma,
            sampling=(pixel_height_um, pixel_width_um),
        )
        component_details = []
        owner_anchor_component_ids = []
        for component_id in range(1, component_count + 1):
            component = component_labels == component_id
            component_area = int(component.sum())
            core_overlap = int((component & owner_core).sum())
            soma_overlap = int((component & own_soma).sum())
            whole_overlap = int((component & own_whole).sum())
            outside_pre_finalization_whole = int(
                (component & ~frozen_pre_finalization_whole).sum()
            )
            minimum_soma_distance_um = (
                float(owner_soma_distance_um[component].min())
                if component_area > 0
                else math.inf
            )
            core_soma_anchor = bool(
                core_overlap > 0
                and soma_overlap >= minimum_owner_overlap_px
            )
            if core_soma_anchor:
                owner_anchor_component_ids.append(component_id)
            rows, cols = np.nonzero(component)
            component_details.append(
                {
                    "component_id": component_id,
                    "area_px": component_area,
                    "core_overlap_px": core_overlap,
                    "soma_overlap_px": soma_overlap,
                    "whole_overlap_px": whole_overlap,
                    "outside_pre_finalization_whole_px": (
                        outside_pre_finalization_whole
                    ),
                    "minimum_soma_distance_um": float(
                        minimum_soma_distance_um
                    ),
                    "core_soma_anchor": core_soma_anchor,
                    "bbox_yx": [
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()) + 1,
                        int(cols.max()) + 1,
                    ],
                }
            )
        if len(owner_anchor_component_ids) == 1:
            selected_component_ids = list(owner_anchor_component_ids)
            owner_extent = (
                component_labels == selected_component_ids[0]
            )
        else:
            selected_component_ids = []
            owner_extent = full_owner_extent
        ignored_component_ids = [
            component_id
            for component_id in range(1, component_count + 1)
            if component_id not in selected_component_ids
        ]
        ignored_owner_extent_px = int(
            np.isin(component_labels, ignored_component_ids).sum()
        )
        selected_core_component_details = []
        selected_substantial_core_component_ids = []
        selected_core_component_count = 0
        if selected_component_ids:
            selected_core = owner_core & owner_extent
            selected_core_labels = measure.label(
                selected_core,
                connectivity=2,
            )
            selected_core_component_count = int(
                selected_core_labels.max()
            )
            for core_component_id in range(
                1,
                selected_core_component_count + 1,
            ):
                core_component = (
                    selected_core_labels == core_component_id
                )
                core_component_area = int(core_component.sum())
                substantial = bool(
                    core_component_area >= minimum_owner_overlap_px
                )
                if substantial:
                    selected_substantial_core_component_ids.append(
                        core_component_id
                    )
                rows, cols = np.nonzero(core_component)
                selected_core_component_details.append(
                    {
                        "component_id": core_component_id,
                        "area_px": core_component_area,
                        "substantial": substantial,
                        "bbox_yx": [
                            int(rows.min()),
                            int(cols.min()),
                            int(rows.max()) + 1,
                            int(cols.max()) + 1,
                        ],
                    }
                )
        accepted_resolved_candidates = []
        resolved_nuclei_in_whole = []
        foreign_resolved_nucleus_ids = []
        incidental_resolved_nucleus_ids = []
        foreign_resolved_nucleus_evidence = []
        owner_distance_um = ndi.distance_transform_edt(
            ~owner_extent,
            sampling=(pixel_height_um, pixel_width_um),
        )
        local_domain = own_soma | owner_extent
        local_domain_distance_um = ndi.distance_transform_edt(
            ~local_domain,
            sampling=(pixel_height_um, pixel_width_um),
        )
        minimum_foreign_overlap_px = max(
            1,
            int(
                math.ceil(
                    cfg.minimum_foreign_overlap_um2 / pixel_area_um2
                )
            ),
        )
        for group in groups:
            group_id = int(group["group_id"])
            group_extent = np.asarray(
                grouped_extents == group_id,
                dtype=bool,
            )
            whole_overlap = int((own_whole & group_extent).sum())
            soma_overlap = int((own_soma & group_extent).sum())
            is_resolved = (
                str(group.get("identity_status", "")) == "resolved"
            )
            if is_resolved and whole_overlap > 0:
                resolved_nuclei_in_whole.append(group_id)
            if (
                bool(group.get("accepted", False))
                and is_resolved
                and soma_overlap >= minimum_owner_overlap_px
            ):
                accepted_resolved_candidates.append(group_id)
            if (
                group_id == owner_id
                or not is_resolved
                or whole_overlap <= 0
            ):
                continue
            extent_area = int(group_extent.sum())
            whole_overlap_fraction = (
                whole_overlap / max(extent_area, 1)
            )
            minimum_owner_distance_um = (
                float(owner_distance_um[group_extent].min())
                if extent_area > 0
                else math.inf
            )
            minimum_local_domain_distance_um = (
                float(local_domain_distance_um[group_extent].min())
                if extent_area > 0
                else math.inf
            )
            meaningful_soma_overlap = (
                soma_overlap >= minimum_foreign_overlap_px
            )
            meaningful_whole_overlap = (
                whole_overlap >= minimum_foreign_overlap_px
                and whole_overlap_fraction
                >= cfg.minimum_foreign_overlap_fraction
            )
            local_owner_contact = (
                whole_overlap >= minimum_foreign_overlap_px
                and minimum_local_domain_distance_um
                <= cfg.maximum_local_foreign_distance_um
            )
            veto = bool(
                meaningful_soma_overlap
                or local_owner_contact
            )
            foreign_resolved_nucleus_evidence.append(
                {
                    "nucleus_id": group_id,
                    "accepted": bool(group.get("accepted", False)),
                    "whole_overlap_px": whole_overlap,
                    "soma_overlap_px": soma_overlap,
                    "whole_overlap_fraction_of_extent": float(
                        whole_overlap_fraction
                    ),
                    "minimum_owner_extent_distance_um": float(
                        minimum_owner_distance_um
                    ),
                    "minimum_local_domain_distance_um": float(
                        minimum_local_domain_distance_um
                    ),
                    "meaningful_soma_overlap": bool(
                        meaningful_soma_overlap
                    ),
                    "meaningful_whole_overlap": bool(
                        meaningful_whole_overlap
                    ),
                    "local_owner_contact": bool(local_owner_contact),
                    "veto": veto,
                }
            )
            if veto:
                foreign_resolved_nucleus_ids.append(group_id)
            else:
                incidental_resolved_nucleus_ids.append(group_id)
        decision["accepted_resolved_soma_candidate_ids"] = sorted(
            accepted_resolved_candidates
        )
        decision["resolved_nucleus_ids_in_whole"] = sorted(
            resolved_nuclei_in_whole
        )
        decision["foreign_resolved_nucleus_ids"] = sorted(
            foreign_resolved_nucleus_ids
        )
        decision["incidental_resolved_nucleus_ids_in_whole"] = sorted(
            incidental_resolved_nucleus_ids
        )
        decision["foreign_resolved_nucleus_evidence"] = sorted(
            foreign_resolved_nucleus_evidence,
            key=lambda row: int(row["nucleus_id"]),
        )

        inside_current_whole = owner_extent & own_whole
        current_background = owner_extent & (whole_labels == 0)
        prior_guard_removed = current_background & frozen_pre_finalization_whole
        novel_outside_pre_finalization_whole = (
            current_background & ~frozen_pre_finalization_whole
        )
        foreign_whole = (
            owner_extent
            & (whole_labels > 0)
            & (whole_labels != astrocyte_id)
        )
        foreign_soma = (
            owner_extent
            & (soma_labels > 0)
            & (soma_labels != astrocyte_id)
        )
        missing_inside = inside_current_whole & ~own_soma
        existing_overlap = int((inside_current_whole & own_soma).sum())
        edge_touch = bool(
            owner_extent[0, :].any()
            or owner_extent[-1, :].any()
            or owner_extent[:, 0].any()
            or owner_extent[:, -1].any()
        )
        decision.update(
            {
                "owner_extent_total_px": int(
                    full_owner_extent.sum()
                ),
                "owner_extent_inside_current_whole_px": int(
                    inside_current_whole.sum()
                ),
                "owner_extent_current_background_px": int(
                    current_background.sum()
                ),
                "owner_extent_outside_pre_finalization_whole_px": int(
                    novel_outside_pre_finalization_whole.sum()
                ),
                "prior_guard_removed_owner_px": int(
                    prior_guard_removed.sum()
                ),
                "existing_soma_owner_overlap_px": existing_overlap,
                "missing_inside_px": int(missing_inside.sum()),
                "foreign_whole_overlap_px": int(foreign_whole.sum()),
                "foreign_soma_overlap_px": int(foreign_soma.sum()),
                "owner_extent_component_count": component_count,
                "owner_anchor_component_count": int(
                    len(owner_anchor_component_ids)
                ),
                "selected_owner_extent_component_count": len(
                    selected_component_ids
                ),
                "selected_owner_extent_component_ids": (
                    selected_component_ids
                ),
                "selected_owner_extent_component_px": int(
                    owner_extent.sum()
                    if selected_component_ids
                    else 0
                ),
                "selected_owner_core_overlap_px": int(
                    (owner_extent & owner_core).sum()
                    if selected_component_ids
                    else 0
                ),
                "selected_owner_soma_overlap_px": int(
                    (owner_extent & own_soma).sum()
                    if selected_component_ids
                    else 0
                ),
                "selected_owner_core_component_count": int(
                    selected_core_component_count
                ),
                "selected_owner_substantial_core_component_count": len(
                    selected_substantial_core_component_ids
                ),
                "selected_owner_substantial_core_component_ids": (
                    selected_substantial_core_component_ids
                ),
                "selected_owner_core_component_details": (
                    selected_core_component_details
                ),
                "ignored_owner_extent_component_count": len(
                    ignored_component_ids
                ),
                "ignored_owner_extent_px": ignored_owner_extent_px,
                "owner_extent_component_details": component_details,
                "owner_extent_xy_edge_touch": edge_touch,
            }
        )
        fail_reasons: list[str] = []
        if owner_extent.sum() == 0:
            fail_reasons.append("owner_extent_missing")
        if len(owner_anchor_component_ids) == 0:
            fail_reasons.append(
                "owner_extent_no_core_soma_anchored_component"
            )
        elif len(owner_anchor_component_ids) > 1:
            fail_reasons.append(
                "owner_extent_multiple_core_soma_anchored_components"
            )
        if (
            len(owner_anchor_component_ids) == 1
            and len(selected_substantial_core_component_ids) == 0
        ):
            fail_reasons.append(
                "owner_extent_no_substantial_core_component"
            )
        elif (
            len(owner_anchor_component_ids) == 1
            and len(selected_substantial_core_component_ids) > 1
        ):
            fail_reasons.append(
                "owner_extent_multiple_substantial_core_components"
            )
        if accepted_resolved_candidates != [owner_id]:
            fail_reasons.append("owner_assignment_not_unique")
        if decision["foreign_resolved_nucleus_ids"]:
            fail_reasons.append("multiple_resolved_nuclei_in_whole")
        if existing_overlap < minimum_owner_overlap_px:
            fail_reasons.append("insufficient_soma_owner_anchor")
        if edge_touch:
            fail_reasons.append("owner_extent_xy_edge_touch")
        if foreign_whole.any() or foreign_soma.any():
            fail_reasons.append("competing_whole_or_soma")
        if prior_guard_removed.any():
            fail_reasons.append("prior_guard_removed_owner_extent")
        if fail_reasons:
            decision["fail_reasons"].extend(fail_reasons)
            if (
                "owner_assignment_not_unique" in fail_reasons
                or "multiple_resolved_nuclei_in_whole" in fail_reasons
            ):
                decision["status"] = (
                    "fail_closed_owner_assignment_not_unique"
                )
            elif (
                "owner_extent_no_core_soma_anchored_component"
                in fail_reasons
            ):
                decision["status"] = (
                    "fail_closed_owner_component_anchor_missing"
                )
            elif (
                "owner_extent_multiple_core_soma_anchored_components"
                in fail_reasons
            ):
                decision["status"] = (
                    "fail_closed_owner_component_ambiguous"
                )
            elif (
                "owner_extent_no_substantial_core_component"
                in fail_reasons
                or "owner_extent_multiple_substantial_core_components"
                in fail_reasons
            ):
                decision["status"] = (
                    "fail_closed_owner_core_ambiguous"
                )
            elif "insufficient_soma_owner_anchor" in fail_reasons:
                decision["status"] = (
                    "fail_closed_insufficient_soma_anchor"
                )
            elif "owner_extent_xy_edge_touch" in fail_reasons:
                decision["status"] = "fail_closed_xy_edge_touch"
            elif "competing_whole_or_soma" in fail_reasons:
                decision["status"] = "fail_closed_competing_whole"
            elif "prior_guard_removed_owner_extent" in fail_reasons:
                decision["status"] = (
                    "fail_closed_prior_guard_removed_extent"
                )
            else:
                decision["status"] = "fail_closed_owner_assignment_missing"
            continue
        plans[astrocyte_id] = {
            "inside": missing_inside,
            "outside": novel_outside_pre_finalization_whole,
        }
        metrics["eligible_cell_count"] += 1

    planned_owners = sorted(plans)
    conflict_ids: set[int] = set()
    for position, first_id in enumerate(planned_owners):
        first = plans[first_id]["outside"]
        for second_id in planned_owners[position + 1 :]:
            if np.any(first & plans[second_id]["outside"]):
                conflict_ids.update((first_id, second_id))
    for astrocyte_id in sorted(conflict_ids):
        plans.pop(astrocyte_id, None)
        decision = decisions[astrocyte_id]
        decision["fail_reasons"].append("planned_outside_overlap")
        decision["status"] = "fail_closed_planned_outside_conflict"

    output_whole = whole_labels.copy()
    output_soma = soma_labels.copy()
    for astrocyte_id, plan in sorted(plans.items()):
        inside = plan["inside"]
        outside = plan["outside"]
        output_soma[inside] = astrocyte_id
        output_whole[outside] = astrocyte_id
        output_soma[outside] = astrocyte_id
        approved_outside |= outside
        inside_count = int(inside.sum())
        outside_count = int(outside.sum())
        decision = decisions[astrocyte_id]
        decision["approved_inside_soma_px"] = inside_count
        decision["approved_outside_whole_px"] = outside_count
        decision["whole_delta_px"] = outside_count
        decision["soma_delta_px"] = inside_count + outside_count
        decision["process_delta_px"] = -inside_count
        decision["approved"] = bool(inside_count or outside_count)
        decision["approved_owner_extent_added_px"] = (
            inside_count + outside_count
        )
        if inside_count or outside_count:
            decision["status"] = (
                "completed_inside_and_exact_outside_pre_finalization_whole_owner_extent"
                if outside_count
                else "completed_inside_owner_extent"
            )
            metrics["changed_cell_ids"].append(astrocyte_id)
            if outside_count:
                metrics["outside_changed_cell_ids"].append(astrocyte_id)
            metrics["inside_added_soma_px"] += inside_count
            metrics["outside_added_whole_px"] += outside_count
            metrics["outside_added_soma_px"] += outside_count
        else:
            decision["status"] = "no_op_owner_already_complete"
            metrics["no_op_cell_ids"].append(astrocyte_id)

    for astrocyte_id, decision in sorted(decisions.items()):
        if str(decision["status"]).startswith("fail_closed_"):
            metrics["fail_closed_cell_ids"].append(astrocyte_id)
        elif astrocyte_id not in plans:
            metrics["no_op_cell_ids"].append(astrocyte_id)
        metrics["decisions"].append(decision)
    metrics["changed_cell_ids"] = sorted(set(metrics["changed_cell_ids"]))
    metrics["outside_changed_cell_ids"] = sorted(
        set(metrics["outside_changed_cell_ids"])
    )
    metrics["no_op_cell_ids"] = sorted(set(metrics["no_op_cell_ids"]))
    metrics["fail_closed_cell_ids"] = sorted(
        set(metrics["fail_closed_cell_ids"])
    )
    metrics["changed_cell_count"] = len(metrics["changed_cell_ids"])
    metrics["no_op_cell_count"] = len(metrics["no_op_cell_ids"])
    metrics["fail_closed_cell_count"] = len(
        metrics["fail_closed_cell_ids"]
    )

    output_process = np.where(
        (output_whole > 0) & (output_soma == 0),
        output_whole,
        0,
    ).astype(process_labels.dtype, copy=False)
    if output_whole.dtype != whole_labels.dtype:
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion changed the Whole label dtype"
        )
    if output_soma.dtype != soma_labels.dtype:
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion changed the Soma label dtype"
        )
    if set(int(value) for value in np.unique(output_whole)) != original_ids:
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion changed the Whole ID set"
        )
    if np.any((output_soma > 0) & (output_process > 0)):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion overlapped Soma and Processes"
        )
    if not np.array_equal(
        output_whole > 0,
        (output_soma > 0) | (output_process > 0),
    ):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion broke Whole = Soma union Processes"
        )
    if not np.array_equal(
        output_soma[output_soma > 0],
        output_whole[output_soma > 0],
    ):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion changed a Soma pixel to a foreign ID"
        )
    if not np.array_equal(
        output_process[output_process > 0],
        output_whole[output_process > 0],
    ):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion changed a Processes pixel to a foreign ID"
        )
    added_whole = (output_whole > 0) & ~(whole_labels > 0)
    if not np.array_equal(added_whole, approved_outside):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion Whole expansion differs from approved mask"
        )
    metrics["post_canonical_owner_extent_completion_whole_area_px"] = int(
        (output_whole > 0).sum()
    )
    metrics["pre_finalization_whole_to_final_removed_px"] = int(
        (frozen_pre_finalization_whole & ~(output_whole > 0)).sum()
    )
    metrics["pre_finalization_whole_to_final_approved_added_px"] = int(
        approved_outside.sum()
    )
    metrics["approved_owner_extent_added_px"] = int(
        metrics["inside_added_soma_px"]
        + metrics["outside_added_soma_px"]
    )
    return (
        output_whole,
        output_soma,
        output_process,
        metrics,
        approved_outside,
    )

def reconcile_same_id_disconnected_soma(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    inventory: ValidatedNucleusAnchors | None,
    identity_metrics: dict,
    axial_metrics: dict,
    profile: str,
    pixel_width_um: float,
    pixel_height_um: float,
    pixel_depth_um: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Join only short, same-owner Soma islands inside one frozen Whole ID."""

    if (
        whole_labels.shape != soma_labels.shape
        or whole_labels.shape != process_labels.shape
    ):
        raise ValueError(
            "Same-ID Soma Island Reconciliation compartment label shapes do not match"
        )
    frozen_whole_labels = whole_labels.copy()
    ownership_cfg = NucleusOwnershipConfig()
    compartment_cfg = compartment_config_for_profile(profile)
    pixel_area_um2 = float(pixel_width_um * pixel_height_um)
    minimum_owner_overlap_px = max(
        1,
        int(
            math.ceil(
                ownership_cfg.owner_min_overlap_um2 / pixel_area_um2
            )
        ),
    )
    maximum_absolute_added_px = max(
        1,
        int(
            math.floor(
                compartment_cfg.min_soma_area_um2
                * SomaNuclearCompletionConfig().maximum_added_fraction_of_existing_soma
                / pixel_area_um2
            )
        ),
    )
    bridge_radius_y = max(
        1,
        int(
            round(
                compartment_cfg.dapi_extent_closing_um
                / max(float(pixel_height_um), 1e-9)
            )
        ),
    )
    bridge_radius_x = max(
        1,
        int(
            round(
                compartment_cfg.dapi_extent_closing_um
                / max(float(pixel_width_um), 1e-9)
            )
        ),
    )
    bridge_structure = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * bridge_radius_x + 1, 2 * bridge_radius_y + 1),
    ).astype(bool)
    metrics = {
        "enabled": bool(
            inventory is not None
            and pixel_depth_um is not None
            and pixel_depth_um > 0
        ),
        "method": (
            "identity-gated, calibration-aware same-owner island bridging; "
            "Whole is frozen and only same-ID Processes may become Soma"
        ),
        "disconnected_before_ids": [],
        "bridged_ids": [],
        "rejected_identity_split_ids": [],
        "rejected_multiple_owner_ids": [],
        "rejected_local_foreign_ids": [],
        "rejected_ambiguous_owner_ids": [],
        "rejected_gap_ids": [],
        "rejected_path_ids": [],
        "added_soma_px": 0,
        "removed_process_px": 0,
        "approved_process_to_soma_px": 0,
        "canonical_satellite_aliases": {},
        "canonical_satellite_alias_details": [],
        "decisions": [],
        "derived_thresholds": {
            "fragment_bridge_max_gap_um": float(
                ownership_cfg.fragment_bridge_max_gap_um
            ),
            "owner_min_overlap_px": minimum_owner_overlap_px,
            "owner_halo_um": float(
                compartment_cfg.soma_trusted_core_nucleus_margin_um
            ),
            "bridge_radius_y_px": bridge_radius_y,
            "bridge_radius_x_px": bridge_radius_x,
            "maximum_absolute_added_px": maximum_absolute_added_px,
            "maximum_relative_added_fraction": float(
                SomaNuclearCompletionConfig().maximum_added_fraction_of_existing_soma
            ),
            "minimum_process_fraction": float(
                compartment_cfg.min_process_fraction
            ),
        },
    }
    if profile == "neonatal":
        metrics["method"] += (
            "; safe merge-only two-island pairs may use the exact owner "
            "extent inside their pair convex hull"
        )
        metrics["owner_convex_hull_ids"] = []
        metrics["owner_convex_hull_added_px"] = 0
        metrics["derived_thresholds"].update(
            {
                "owner_convex_hull_profile": "neonatal",
                "owner_convex_hull_component_count": 2,
                "owner_convex_hull_relative_cap_only": True,
            }
        )
    approved_labels = np.zeros(whole_labels.shape, dtype=np.uint16)
    output_soma = soma_labels.copy()
    if inventory is not None:
        groups, grouped_extents = inventory_group_geometry(
            inventory,
            pixel_width_um,
            pixel_height_um,
            pixel_depth_um,
        )
    else:
        groups, grouped_extents = [], None
    group_by_id = {
        int(group["group_id"]): group for group in groups
    }
    records_by_id = (
        {
            int(row["instance_id"]): row
            for row in inventory.nucleus_instance_records
        }
        if inventory is not None
        else {}
    )
    canonical_cfg = CanonicalIdentityConfig()
    satellite_aliases: dict[int, int] = {}
    satellite_alias_details: list[dict] = []
    for satellite_id, satellite in sorted(records_by_id.items()):
        if pixel_depth_um is None or pixel_depth_um <= 0:
            break
        if (
            bool(satellite.get("accepted", False))
            or not bool(satellite.get("dapi_valid", False))
            or str(satellite.get("identity_status", "")) != "resolved"
            or bool(
                satellite.get("resolution_diagnostics", {}).get(
                    "split_accepted",
                    False,
                )
            )
        ):
            continue
        satellite_sources = {
            int(value) for value in satellite.get("source_object_ids", ())
        }
        satellite_volume = float(satellite.get("volume_um3", 0.0))
        if not satellite_sources or satellite_volume <= 0:
            continue
        candidates: list[tuple[int, dict]] = []
        for owner_candidate_id, owner_candidate in sorted(
            records_by_id.items()
        ):
            if (
                owner_candidate_id == satellite_id
                or not bool(owner_candidate.get("accepted", False))
                or not bool(owner_candidate.get("dapi_valid", False))
                or str(owner_candidate.get("identity_status", ""))
                != "resolved"
            ):
                continue
            owner_sources = {
                int(value)
                for value in owner_candidate.get("source_object_ids", ())
            }
            if not satellite_sources < owner_sources:
                continue
            owner_volume = float(owner_candidate.get("volume_um3", 0.0))
            if owner_volume <= 0:
                continue
            volume_ratio = satellite_volume / owner_volume
            if volume_ratio > canonical_cfg.satellite_max_volume_ratio:
                continue
            z_overlap = max(
                0,
                min(
                    int(satellite["z_max_0based_inclusive"]),
                    int(owner_candidate["z_max_0based_inclusive"]),
                )
                - max(
                    int(satellite["z_min_0based"]),
                    int(owner_candidate["z_min_0based"]),
                )
                + 1,
            )
            satellite_z_span = (
                int(satellite["z_max_0based_inclusive"])
                - int(satellite["z_min_0based"])
                + 1
            )
            z_overlap_fraction = z_overlap / max(satellite_z_span, 1)
            if (
                z_overlap_fraction
                < canonical_cfg.satellite_min_z_overlap_fraction
            ):
                continue
            delta_z_um = (
                float(satellite["center_z"])
                - float(owner_candidate["center_z"])
            ) * float(pixel_depth_um)
            delta_y_um = (
                float(satellite["center_y"])
                - float(owner_candidate["center_y"])
            ) * float(pixel_height_um)
            delta_x_um = (
                float(satellite["center_x"])
                - float(owner_candidate["center_x"])
            ) * float(pixel_width_um)
            center_distance_um = math.sqrt(
                delta_z_um**2 + delta_y_um**2 + delta_x_um**2
            )
            satellite_radius_um = (
                3.0 * satellite_volume / (4.0 * math.pi)
            ) ** (1.0 / 3.0)
            owner_radius_um = (
                3.0 * owner_volume / (4.0 * math.pi)
            ) ** (1.0 / 3.0)
            radius_limit_um = (
                canonical_cfg.satellite_max_radius_sum_factor
                * (satellite_radius_um + owner_radius_um)
            )
            if center_distance_um > radius_limit_um:
                continue
            candidates.append(
                (
                    owner_candidate_id,
                    {
                        "satellite_id": satellite_id,
                        "owner_id": owner_candidate_id,
                        "satellite_source_object_ids": sorted(
                            satellite_sources
                        ),
                        "owner_source_object_ids": sorted(owner_sources),
                        "volume_ratio": float(volume_ratio),
                        "z_overlap_fraction": float(z_overlap_fraction),
                        "center_distance_um": float(center_distance_um),
                        "radius_limit_um": float(radius_limit_um),
                    },
                )
            )
        if len(candidates) == 1:
            owner_candidate_id, detail = candidates[0]
            satellite_aliases[satellite_id] = owner_candidate_id
            satellite_alias_details.append(detail)
    if grouped_extents is not None and satellite_aliases:
        effective_grouped_extents = np.asarray(
            grouped_extents,
            dtype=np.uint32,
        ).copy()
        for satellite_id, owner_id in sorted(satellite_aliases.items()):
            effective_grouped_extents[
                effective_grouped_extents == satellite_id
            ] = owner_id
        grouped_extents = effective_grouped_extents
    metrics["canonical_satellite_aliases"] = {
        str(satellite_id): owner_id
        for satellite_id, owner_id in sorted(satellite_aliases.items())
    }
    metrics["canonical_satellite_alias_details"] = satellite_alias_details
    identity_lineage = {
        int(key): value
        for key, value in identity_metrics.get("final_lineage", {}).items()
    }
    merge_decisions = [
        value
        for value in identity_metrics.get("merge_decisions", [])
        if isinstance(value, dict)
    ]
    merged_id_claim_counts: dict[int, int] = {}
    for lineage in identity_lineage.values():
        for merged_id in {
            int(value)
            for value in lineage.get("source_merged_ids", [])
        }:
            merged_id_claim_counts[merged_id] = (
                merged_id_claim_counts.get(merged_id, 0) + 1
            )
    axial_mapping = {
        int(old_id): int(new_id)
        for old_id, new_id in axial_metrics.get("id_mapping", {}).items()
    }
    inverse_axial_mapping = {
        final_id: old_id for old_id, final_id in axial_mapping.items()
    }
    retained_owner_claim_counts: dict[int, int] = {}
    for pre_axial_id in axial_mapping:
        lineage = identity_lineage.get(pre_axial_id)
        if lineage is None:
            continue
        owner_id = int(lineage.get("canonical_owner_id", 0))
        if owner_id > 0:
            retained_owner_claim_counts[owner_id] = (
                retained_owner_claim_counts.get(owner_id, 0) + 1
            )

    def is_accepted_merge_only(lineage: dict) -> bool:
        source_ids = sorted(
            {
                int(value)
                for value in lineage.get("source_astrocyte_ids", [])
            }
        )
        source_merged_ids = sorted(
            {
                int(value)
                for value in lineage.get("source_merged_ids", [])
            }
        )
        owner_id = int(lineage.get("canonical_owner_id", 0))
        if (
            len(source_ids) < 2
            or len(source_merged_ids) != 1
            or owner_id <= 0
            or merged_id_claim_counts.get(source_merged_ids[0], 0) != 1
        ):
            return False
        source_set = set(source_ids)
        for merge_decision in merge_decisions:
            if int(merge_decision.get("canonical_nucleus_id", 0)) != owner_id:
                continue
            decision_sources = {
                int(value)
                for value in merge_decision.get(
                    "source_astrocyte_ids",
                    [],
                )
            }
            if decision_sources != source_set:
                continue
            parent = {source_id: source_id for source_id in source_ids}

            def find(source_id: int) -> int:
                while parent[source_id] != source_id:
                    parent[source_id] = parent[parent[source_id]]
                    source_id = parent[source_id]
                return source_id

            for pair in merge_decision.get("accepted_pairs", []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                left_id, right_id = (int(pair[0]), int(pair[1]))
                if left_id not in source_set or right_id not in source_set:
                    continue
                left_root = find(left_id)
                right_root = find(right_id)
                if left_root != right_root:
                    parent[right_root] = left_root
            if len({find(source_id) for source_id in source_ids}) == 1:
                return True
        return False

    def component_count(mask: np.ndarray) -> int:
        return int(measure.label(mask, connectivity=2).max())

    def closest_masks(
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, tuple[int, int], tuple[int, int]]:
        distance, nearest = ndi.distance_transform_edt(
            ~source,
            sampling=(float(pixel_height_um), float(pixel_width_um)),
            return_indices=True,
        )
        target_coordinates = np.argwhere(target)
        target_distances = distance[target]
        minimum_distance = float(np.min(target_distances))
        tied_positions = np.flatnonzero(
            np.isclose(
                target_distances,
                minimum_distance,
                rtol=0.0,
                atol=1e-12,
            )
        )
        source_center = np.mean(np.argwhere(source), axis=0)
        target_center = np.mean(target_coordinates, axis=0)
        tied_rows = []
        for position in tied_positions:
            target_y = int(target_coordinates[position, 0])
            target_x = int(target_coordinates[position, 1])
            source_y = int(nearest[0, target_y, target_x])
            source_x = int(nearest[1, target_y, target_x])
            center_cost = float(
                (target_y - target_center[0]) ** 2
                + (target_x - target_center[1]) ** 2
                + (source_y - source_center[0]) ** 2
                + (source_x - source_center[1]) ** 2
            )
            tied_rows.append(
                (
                    center_cost,
                    target_y,
                    target_x,
                    source_y,
                    source_x,
                )
            )
        _, target_y, target_x, source_y, source_x = min(tied_rows)
        center_distance = minimum_distance
        contact_step = math.hypot(
            float(pixel_height_um) if target_y != source_y else 0.0,
            float(pixel_width_um) if target_x != source_x else 0.0,
        )
        effective_gap = max(0.0, center_distance - contact_step)
        return (
            effective_gap,
            (source_y, source_x),
            (target_y, target_x),
        )

    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        own_whole = whole_labels == astrocyte_id
        original_soma = soma_labels == astrocyte_id
        pre_components = component_count(original_soma)
        decision = {
            "astrocyte_id": astrocyte_id,
            "pre_component_count": pre_components,
            "post_component_count": pre_components,
            "owner_group_id": 0,
            "resolved_nucleus_ids_in_whole": [],
            "foreign_resolved_nucleus_ids_in_whole": [],
            "local_foreign_veto_ids": [],
            "gap_um": [],
            "added_soma_px": 0,
            "approved": False,
            "approved_process_to_soma_px": 0,
            "process_fraction_before": float(
                (process_labels == astrocyte_id).sum()
                / max(int(own_whole.sum()), 1)
            ),
            "process_fraction_after": float(
                (process_labels == astrocyte_id).sum()
                / max(int(own_whole.sum()), 1)
            ),
            "status": (
                "unchanged_connected"
                if pre_components <= 1
                else "skipped_no_owner"
            ),
        }
        if profile == "neonatal":
            decision["bridge_method"] = "none"
            decision["owner_overlap_px_by_component"] = []
        if pre_components <= 1:
            metrics["decisions"].append(decision)
            continue
        metrics["disconnected_before_ids"].append(astrocyte_id)

        pre_axial_id = inverse_axial_mapping.get(astrocyte_id)
        lineage = (
            identity_lineage.get(pre_axial_id)
            if pre_axial_id is not None
            else None
        )
        if lineage is None:
            metrics["decisions"].append(decision)
            continue
        identity_changed = bool(lineage.get("identity_changed", False))
        accepted_merge_only = (
            identity_changed and is_accepted_merge_only(lineage)
        )
        if identity_changed and not accepted_merge_only:
            decision["status"] = "skipped_identity_changed"
            metrics["rejected_identity_split_ids"].append(astrocyte_id)
            metrics["decisions"].append(decision)
            continue
        owner_id = int(lineage.get("canonical_owner_id", 0))
        decision["owner_group_id"] = owner_id
        if (
            accepted_merge_only
            and retained_owner_claim_counts.get(owner_id, 0) != 1
        ):
            decision["status"] = "skipped_ambiguous_owner"
            metrics["rejected_ambiguous_owner_ids"].append(astrocyte_id)
            metrics["decisions"].append(decision)
            continue
        owner_record = records_by_id.get(owner_id)
        if owner_record is None or grouped_extents is None:
            metrics["decisions"].append(decision)
            continue
        if str(owner_record.get("identity_status", "")) != "resolved":
            decision["status"] = "skipped_ambiguous_owner"
            metrics["rejected_ambiguous_owner_ids"].append(astrocyte_id)
            metrics["decisions"].append(decision)
            continue
        if (
            not bool(owner_record.get("dapi_valid", False))
            or not bool(owner_record.get("accepted", False))
            or owner_id not in group_by_id
        ):
            metrics["decisions"].append(decision)
            continue

        owner_extent = np.asarray(
            grouped_extents == owner_id,
            dtype=bool,
        )
        labeled_soma = measure.label(original_soma, connectivity=2)
        owner_overlap_px_by_component = [
            int(
                (
                    (labeled_soma == component_id)
                    & owner_extent
                ).sum()
            )
            for component_id in range(1, pre_components + 1)
        ]
        if profile == "neonatal":
            decision["owner_overlap_px_by_component"] = (
                owner_overlap_px_by_component
            )
        every_island_supported = all(
            overlap_px > 0
            for overlap_px in owner_overlap_px_by_component
        )
        if not every_island_supported:
            decision["status"] = "skipped_island_not_owner_supported"
            metrics["decisions"].append(decision)
            continue
        use_owner_convex_hull = bool(
            profile == "neonatal"
            and accepted_merge_only
            and pre_components == 2
        )
        if (
            use_owner_convex_hull
            and any(
                overlap_px < minimum_owner_overlap_px
                for overlap_px in owner_overlap_px_by_component
            )
        ):
            decision["status"] = "skipped_island_not_owner_supported"
            metrics["rejected_path_ids"].append(astrocyte_id)
            metrics["decisions"].append(decision)
            continue

        resolved_ids = []
        for group_id, record in sorted(records_by_id.items()):
            if (
                not bool(record.get("dapi_valid", False))
                or str(record.get("identity_status", "")) != "resolved"
            ):
                continue
            overlap = int(
                (
                    own_whole
                    & np.asarray(grouped_extents == group_id, dtype=bool)
                ).sum()
            )
            if overlap >= minimum_owner_overlap_px:
                resolved_ids.append(group_id)
        decision["resolved_nucleus_ids_in_whole"] = resolved_ids
        decision["foreign_resolved_nucleus_ids_in_whole"] = [
            group_id for group_id in resolved_ids if group_id != owner_id
        ]
        if owner_id not in resolved_ids:
            decision["status"] = "skipped_owner_not_resolved_in_whole"
            metrics["decisions"].append(decision)
            continue

        if (
            decision["process_fraction_before"]
            < compartment_cfg.min_process_fraction
        ):
            decision["status"] = "skipped_process_fraction"
            metrics["decisions"].append(decision)
            continue

        owner_distance = ndi.distance_transform_edt(
            ~owner_extent,
            sampling=(float(pixel_height_um), float(pixel_width_um)),
        )
        allowed = (
            own_whole
            & (
                owner_distance
                <= float(
                    compartment_cfg.soma_trusted_core_nucleus_margin_um
                )
            )
        )
        trial = original_soma.copy()
        bridge = np.zeros_like(trial)
        failure_status: str | None = None
        gap_values: list[float] = []
        while component_count(trial) > 1:
            labels = measure.label(trial, connectivity=2)
            properties = list(measure.regionprops(labels))
            root_property = max(
                properties,
                key=lambda row: (int(row.area), -int(row.label)),
            )
            root = labels == int(root_property.label)
            closest: tuple[
                float,
                int,
                tuple[int, int],
                tuple[int, int],
            ] | None = None
            for prop in properties:
                if int(prop.label) == int(root_property.label):
                    continue
                candidate = labels == int(prop.label)
                gap, source_point, target_point = closest_masks(
                    root,
                    candidate,
                )
                row = (
                    gap,
                    int(prop.label),
                    source_point,
                    target_point,
                )
                if closest is None or row < closest:
                    closest = row
            if closest is None:
                failure_status = "skipped_no_safe_path"
                break
            gap, target_label, source_point, target_point = closest
            selected_target = labels == int(target_label)
            gap_values.append(float(gap))
            if gap > ownership_cfg.fragment_bridge_max_gap_um + 1e-12:
                failure_status = "skipped_gap_too_large"
                break
            target_foreign_ids = []
            for group_id in sorted(
                int(value)
                for value in np.unique(grouped_extents[selected_target])
                if int(value) > 0 and int(value) != owner_id
            ):
                target_overlap_px = int(
                    (
                        selected_target
                        & np.asarray(grouped_extents == group_id, dtype=bool)
                    ).sum()
                )
                record = records_by_id.get(group_id)
                incidental_unaccepted_resolved_overlap = bool(
                    record is not None
                    and not bool(record.get("accepted", False))
                    and bool(record.get("dapi_valid", False))
                    and str(record.get("identity_status", "")) == "resolved"
                    and target_overlap_px < minimum_owner_overlap_px
                )
                if not incidental_unaccepted_resolved_overlap:
                    target_foreign_ids.append(group_id)
            if target_foreign_ids:
                decision["local_foreign_veto_ids"] = sorted(
                    set(decision["local_foreign_veto_ids"])
                    | set(target_foreign_ids)
                )
                failure_status = "skipped_foreign_nucleus_near_connection"
                break
            if use_owner_convex_hull:
                pair_hull = morphology.convex_hull_image(original_soma)
                eligible_hull = (
                    pair_hull
                    & own_whole
                    & owner_extent
                    & (process_labels == astrocyte_id)
                    & ~original_soma
                )
                eligible_labels = measure.label(
                    eligible_hull,
                    connectivity=2,
                )
                joining_components = []
                for component_id in range(
                    1,
                    int(eligible_labels.max()) + 1,
                ):
                    component = eligible_labels == component_id
                    if component_count(original_soma | component) == 1:
                        joining_components.append(component)
                if len(joining_components) != 1:
                    failure_status = "skipped_no_safe_path"
                    break
                footprint = joining_components[0]
                if profile == "neonatal":
                    decision["bridge_method"] = (
                        "accepted_merge_owner_convex_hull"
                    )
            else:
                line = np.zeros_like(trial, dtype=np.uint8)
                cv2.line(
                    line,
                    (int(source_point[1]), int(source_point[0])),
                    (int(target_point[1]), int(target_point[0])),
                    color=1,
                    thickness=1,
                    lineType=cv2.LINE_8,
                )
                footprint = ndi.binary_dilation(
                    line.astype(bool),
                    structure=bridge_structure,
                )
                if profile == "neonatal":
                    decision["bridge_method"] = (
                        "calibration_aware_shortest_line"
                    )
            distance_to_connection_um = ndi.distance_transform_edt(
                ~footprint,
                sampling=(
                    float(pixel_height_um),
                    float(pixel_width_um),
                ),
            )
            local_foreign_ids = sorted(
                int(value)
                for value in np.unique(
                    grouped_extents[
                        distance_to_connection_um
                        <= ownership_cfg.unowned_barrier_radius_um
                    ]
                )
                if int(value) > 0 and int(value) != owner_id
            )
            if local_foreign_ids:
                decision["local_foreign_veto_ids"] = sorted(
                    set(decision["local_foreign_veto_ids"])
                    | set(local_foreign_ids)
                )
                failure_status = "skipped_foreign_nucleus_near_connection"
                break
            if use_owner_convex_hull:
                proposed = footprint
            else:
                if np.any(footprint & ~(allowed | trial)):
                    failure_status = "skipped_no_safe_path"
                    break
                proposed = footprint & allowed & ~trial
            if not proposed.any():
                failure_status = "skipped_no_safe_path"
                break
            before_count = component_count(trial)
            trial |= proposed
            bridge |= proposed
            if component_count(trial) >= before_count:
                failure_status = "skipped_no_safe_path"
                break
        decision["gap_um"] = [round(value, 9) for value in gap_values]
        if failure_status is not None:
            decision["status"] = failure_status
            if failure_status == "skipped_gap_too_large":
                metrics["rejected_gap_ids"].append(astrocyte_id)
            elif failure_status == "skipped_foreign_nucleus_near_connection":
                metrics["rejected_multiple_owner_ids"].append(astrocyte_id)
                metrics["rejected_local_foreign_ids"].append(astrocyte_id)
            else:
                metrics["rejected_path_ids"].append(astrocyte_id)
            metrics["decisions"].append(decision)
            continue
        if component_count(trial) != 1:
            decision["status"] = "skipped_no_safe_path"
            metrics["rejected_path_ids"].append(astrocyte_id)
            metrics["decisions"].append(decision)
            continue

        added_count = int(bridge.sum())
        maximum_relative_added_px = int(
            math.floor(
                SomaNuclearCompletionConfig().maximum_added_fraction_of_existing_soma
                * max(int(original_soma.sum()), 1)
            )
        )
        maximum_added_px = (
            max(1, maximum_relative_added_px)
            if use_owner_convex_hull
            else min(
                maximum_absolute_added_px,
                max(1, maximum_relative_added_px),
            )
        )
        if profile == "neonatal":
            decision["maximum_added_px"] = maximum_added_px
        if added_count > maximum_added_px:
            decision["status"] = "skipped_bridge_too_large"
            metrics["decisions"].append(decision)
            continue
        process_after = int(
            (process_labels == astrocyte_id).sum()
        ) - added_count
        process_fraction_after = process_after / max(
            int(own_whole.sum()),
            1,
        )
        decision["process_fraction_after"] = float(process_fraction_after)
        if process_fraction_after < compartment_cfg.min_process_fraction:
            decision["status"] = "skipped_process_fraction"
            metrics["decisions"].append(decision)
            continue
        if np.any(bridge & (process_labels != astrocyte_id)):
            raise RuntimeError(
                "Same-ID Soma Island Reconciliation attempted to convert pixels "
                "outside same-ID Processes"
            )
        output_soma[bridge] = astrocyte_id
        approved_labels[bridge] = astrocyte_id
        decision["post_component_count"] = 1
        decision["added_soma_px"] = added_count
        decision["approved"] = True
        decision["approved_process_to_soma_px"] = added_count
        decision["status"] = "bridged_same_owner_islands"
        metrics["bridged_ids"].append(astrocyte_id)
        metrics["added_soma_px"] += added_count
        metrics["removed_process_px"] += added_count
        metrics["approved_process_to_soma_px"] += added_count
        if use_owner_convex_hull:
            metrics["owner_convex_hull_ids"].append(astrocyte_id)
            metrics["owner_convex_hull_added_px"] += added_count
        metrics["decisions"].append(decision)

    output_process = np.where(
        (whole_labels > 0) & (output_soma == 0),
        whole_labels,
        0,
    ).astype(process_labels.dtype, copy=False)
    if not np.array_equal(whole_labels, frozen_whole_labels):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation changed Whole labels"
        )
    if np.any((output_soma > 0) & (output_process > 0)):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation overlapped Soma and Processes"
        )
    if not np.array_equal(
        whole_labels > 0,
        (output_soma > 0) | (output_process > 0),
    ):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation broke Whole = Soma union Processes"
        )
    if np.any((output_soma > 0) & (output_soma != whole_labels)):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation created a foreign Soma label"
        )
    if np.any(
        (approved_labels > 0)
        & (process_labels != approved_labels)
    ):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation approval map is not same-ID Processes"
        )
    metrics["disconnected_before_ids"] = sorted(
        metrics["disconnected_before_ids"]
    )
    metrics["bridged_ids"] = sorted(metrics["bridged_ids"])
    if profile == "neonatal":
        metrics["owner_convex_hull_ids"] = sorted(
            metrics["owner_convex_hull_ids"]
        )
    return whole_labels, output_soma, output_process, metrics

def finalize_compartment_geometry_and_metrics(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    metrics: dict,
    inventory: ValidatedNucleusAnchors | None,
    context: Neonatal3DContext | None,
    struct: np.ndarray,
    profile: str,
    pixel_width_um: float,
    pixel_height_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply post-selection guards and refresh synchronized cell metrics."""

    pre_finalization_whole_union = (whole_labels > 0).copy()
    pre_finalization_whole_union.flags.writeable = False
    whole_labels, soma_labels, process_labels, identity_metrics = (
        apply_canonical_identity_reconciliation(
            whole_labels,
            soma_labels,
            process_labels,
            struct,
            inventory,
            pixel_width_um,
            pixel_height_um,
            context.pixel_depth_um if context is not None else None,
            profile,
        )
    )
    whole_labels, soma_labels, process_labels, axial_metrics = (
        apply_axial_truncation_guard(
            whole_labels,
            soma_labels,
            process_labels,
            inventory,
            context,
            pixel_width_um,
            pixel_height_um,
        )
    )
    if profile == "mature":
        whole_labels, soma_labels, process_labels, projection_metrics = (
            apply_projected_foreign_soma_guard(
                whole_labels,
                soma_labels,
                process_labels,
                inventory,
                pixel_width_um,
                pixel_height_um,
                context.pixel_depth_um if context is not None else None,
            )
        )
    else:
        projection_metrics = {
            "enabled": False,
            "status": "skipped_for_dense_neonatal_projection",
            "method": (
                "projection foreign-soma exclusion is restricted to mature samples; "
                "neonatal identity uses explicit canonical nucleus reconciliation"
            ),
            "evaluated_cell_count": int(whole_labels.max()),
            "changed_cell_count": 0,
            "removed_area_px": 0,
            "decisions": [],
        }
    # Preserve the within-Whole owner-extent completion result first. Canonical
    # owner completion then adds only canonical-owner pixels, so cells outside
    # the eligibility gates remain unchanged.
    whole_labels, soma_labels, process_labels, within_whole_completion_metrics = (
        _complete_soma_within_whole_owner_extent(
            whole_labels,
            soma_labels,
            process_labels,
            inventory,
            pixel_width_um,
            pixel_height_um,
            context.pixel_depth_um if context is not None else None,
        )
    )
    owner_assignments, owner_assignment_failures = (
        resolve_canonical_owner_assignments(
            identity_metrics,
            axial_metrics,
            inventory,
            int(whole_labels.max()),
        )
    )
    pre_canonical_owner_extent_completion_union = (whole_labels > 0).copy()
    pre_canonical_owner_extent_completion_soma_labels = soma_labels.copy()
    (
        whole_labels,
        soma_labels,
        process_labels,
        soma_completion_metrics,
        approved_outside_pre_finalization_whole_mask,
    ) = (
        complete_soma_to_owner_nuclear_extent(
            whole_labels,
            soma_labels,
            process_labels,
            inventory,
            pixel_width_um,
            pixel_height_um,
            context.pixel_depth_um if context is not None else None,
            owner_assignments,
            owner_assignment_failures,
            pre_finalization_whole_union,
        )
    )
    canonical_owner_extent_approved_owner_extent_labels = np.where(
        (pre_canonical_owner_extent_completion_soma_labels == 0)
        & (soma_labels > 0),
        soma_labels,
        0,
    ).astype(np.uint16)
    if (
        int((canonical_owner_extent_approved_owner_extent_labels > 0).sum())
        != int(
            soma_completion_metrics[
                "approved_owner_extent_added_px"
            ]
        )
    ):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion approval labels differ "
            "from recorded approved pixels"
        )
    pre_same_id_soma_reconciliation_whole_labels = whole_labels.copy()
    pre_same_id_soma_reconciliation_soma_labels = soma_labels.copy()
    pre_same_id_soma_reconciliation_process_labels = process_labels.copy()
    (
        whole_labels,
        soma_labels,
        process_labels,
        soma_reconciliation_metrics,
    ) = reconcile_same_id_disconnected_soma(
        whole_labels,
        soma_labels,
        process_labels,
        inventory,
        identity_metrics,
        axial_metrics,
        profile,
        pixel_width_um,
        pixel_height_um,
        context.pixel_depth_um if context is not None else None,
    )
    same_id_soma_reconciliation_approved_process_to_soma_labels = np.where(
        (pre_same_id_soma_reconciliation_soma_labels == 0)
        & (soma_labels > 0),
        soma_labels,
        0,
    ).astype(np.uint16)
    if not np.array_equal(
        whole_labels,
        pre_same_id_soma_reconciliation_whole_labels,
    ):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation changed Whole labels"
        )
    if not np.array_equal(
        same_id_soma_reconciliation_approved_process_to_soma_labels,
        np.where(
            (pre_same_id_soma_reconciliation_process_labels > 0)
            & (process_labels == 0),
            pre_same_id_soma_reconciliation_process_labels,
            0,
        ).astype(np.uint16),
    ):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation Soma additions differ from "
            "Processes removals"
        )
    if (
        int(
            (
                same_id_soma_reconciliation_approved_process_to_soma_labels > 0
            ).sum()
        )
        != int(
            soma_reconciliation_metrics[
                "approved_process_to_soma_px"
            ]
        )
    ):
        raise RuntimeError(
            "Same-ID Soma Island Reconciliation approval labels differ from "
            "recorded approved pixels"
        )
    if not np.array_equal(
        (whole_labels > 0) & ~pre_canonical_owner_extent_completion_union,
        approved_outside_pre_finalization_whole_mask,
    ):
        raise RuntimeError(
            "Canonical Owner Nuclear-Extent Completion final Whole expansion "
            "differs from its approved mask"
        )
    if not np.array_equal(
        (whole_labels > 0) & ~pre_finalization_whole_union,
        approved_outside_pre_finalization_whole_mask,
    ):
        raise RuntimeError(
            "Final Whole expansion outside the pre-finalization Whole is not "
            "exclusively from Canonical Owner Nuclear-Extent Completion"
        )
    if np.any((soma_labels > 0) & (process_labels > 0)):
        raise RuntimeError("Final geometry overlapped Soma and Processes")
    if not np.array_equal(
        whole_labels > 0,
        (soma_labels > 0) | (process_labels > 0),
    ):
        raise RuntimeError("Final geometry violates Whole = Soma union Processes")

    original_rows = {
        int(row["astrocyte_id"]): dict(row) for row in metrics.get("per_cell", [])
    }
    axial_mapping = {
        int(old_id): int(new_id)
        for old_id, new_id in axial_metrics.get("id_mapping", {}).items()
    }
    inverse_axial_mapping = {new_id: old_id for old_id, new_id in axial_mapping.items()}
    identity_lineage = {
        int(final_id): row
        for final_id, row in identity_metrics.get("final_lineage", {}).items()
    }
    refreshed_rows = []
    for astrocyte_id in range(1, int(whole_labels.max()) + 1):
        pre_axial_id = inverse_axial_mapping.get(astrocyte_id, astrocyte_id)
        lineage = identity_lineage.get(
            pre_axial_id,
            {"source_astrocyte_ids": [pre_axial_id]},
        )
        source_ids = [int(value) for value in lineage["source_astrocyte_ids"]]
        source_row = next(
            (original_rows[value] for value in source_ids if value in original_rows),
            {"astrocyte_id": astrocyte_id},
        )
        row = dict(source_row)
        row["astrocyte_id"] = astrocyte_id
        row["source_astrocyte_ids_before_identity_reconciliation"] = source_ids
        row["canonical_nucleus_id"] = int(lineage.get("canonical_owner_id", 0))
        row["identity_reconciled"] = bool(lineage.get("identity_changed", False))
        whole_area = int((whole_labels == astrocyte_id).sum())
        soma_area = int((soma_labels == astrocyte_id).sum())
        process_area = int((process_labels == astrocyte_id).sum())
        row["whole_area_px"] = whole_area
        row["soma_area_px"] = soma_area
        row["process_area_px"] = process_area
        row["soma_fraction"] = soma_area / max(whole_area, 1)
        row["process_fraction"] = process_area / max(whole_area, 1)
        refreshed_rows.append(row)
    metrics["per_cell"] = refreshed_rows
    metrics["canonical_identity_reconciliation"] = identity_metrics
    metrics[
        "within_whole_soma_nuclear_extent_completion"
    ] = within_whole_completion_metrics
    metrics["soma_nuclear_extent_completion"] = soma_completion_metrics
    metrics[
        "_canonical_owner_extent_pre_finalization_whole_union_mask"
    ] = pre_finalization_whole_union
    metrics[
        "_canonical_owner_extent_approved_outside_pre_finalization_whole_mask"
    ] = approved_outside_pre_finalization_whole_mask
    metrics[
        "_canonical_owner_extent_approved_owner_extent_labels"
    ] = canonical_owner_extent_approved_owner_extent_labels
    if (
        inventory is not None
        and inventory.nucleus_instance_core_labels_2d is not None
        and inventory.nucleus_instance_extent_labels_2d is not None
    ):
        metrics[
            "_canonical_nucleus_instance_core_labels_2d"
        ] = np.asarray(
            inventory.nucleus_instance_core_labels_2d,
            dtype=np.uint32,
        )
        metrics[
            "_canonical_nucleus_instance_extent_labels_2d"
        ] = np.asarray(
            inventory.nucleus_instance_extent_labels_2d,
            dtype=np.uint32,
        )
    metrics[
        "same_id_disconnected_soma_reconciliation"
    ] = soma_reconciliation_metrics
    metrics[
        "_same_id_soma_reconciliation_approved_process_to_soma_labels"
    ] = same_id_soma_reconciliation_approved_process_to_soma_labels
    metrics["axial_truncation_guard"] = axial_metrics
    metrics["projected_foreign_soma_guard"] = projection_metrics
    metrics["roi_count"] = int(whole_labels.max())
    metrics["whole_area_px"] = int((whole_labels > 0).sum())
    metrics["soma_area_px"] = int((soma_labels > 0).sum())
    metrics["process_area_px"] = int((process_labels > 0).sum())
    metrics["soma_area_fraction"] = (
        metrics["soma_area_px"] / max(metrics["whole_area_px"], 1)
    )
    metrics["process_area_fraction"] = (
        metrics["process_area_px"] / max(metrics["whole_area_px"], 1)
    )
    return whole_labels, soma_labels, process_labels, metrics
