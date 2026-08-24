# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def prune_soma_to_trusted_core_shell(
    soma: np.ndarray,
    trusted_core: np.ndarray,
    anchor_seeds: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    max_shell_um: float,
    min_soma_area_px: int,
) -> tuple[np.ndarray, bool]:
    """Conservatively remove thin Soma extensions without adding any pixels."""

    if not soma.any() or not trusted_core.any() or not anchor_seeds.any():
        return soma, False
    trusted_core = trusted_core & soma
    anchor_seeds = anchor_seeds & soma
    if not trusted_core.any() or not anchor_seeds.any():
        return soma, False
    distance_um = ndi.distance_transform_edt(
        ~trusted_core,
        sampling=(pixel_height_um, pixel_width_um),
    )
    allowed = soma & (distance_um <= max_shell_um)
    allowed |= anchor_seeds
    pruned = ndi.binary_propagation(
        anchor_seeds,
        structure=np.ones((3, 3), dtype=bool),
        mask=allowed,
    ).astype(bool)
    if not np.all(pruned[anchor_seeds]):
        return soma, False
    if int(pruned.sum()) < min_soma_area_px:
        return soma, False
    if np.any(pruned & ~soma):
        raise RuntimeError("Soma core-shell pruning added pixels outside the original Soma")
    return pruned, bool(np.any(soma & ~pruned))

def compartment_config_for_profile(profile: str) -> CompartmentConfig:
    if profile == "mature":
        return CompartmentConfig()
    if profile != "neonatal":
        raise ValueError(f"Unknown astrocyte profile: {profile}")
    return replace(
        CompartmentConfig(),
        soma_zone_max_um=4.60,
        soma_zone_scale_process_rich=1.95,
        soma_zone_scale_compact=2.45,
        thickness_fraction_process_rich=0.40,
        thickness_fraction_compact=0.30,
        structural_percentile_process_rich=62.0,
        structural_percentile_compact=50.0,
        fallback_soma_radius_um=1.45,
        max_soma_fraction=0.64,
        primary_anchor_min_score=0.60,
        primary_anchor_min_thickness_support=0.56,
        primary_anchor_min_structural_support=0.32,
        primary_anchor_min_overlap_fraction=0.32,
        multi_anchor_min_score=0.58,
        multi_anchor_max_score_delta=0.18,
        multi_anchor_min_thickness_support=0.60,
        multi_anchor_min_structural_support=0.33,
        multi_anchor_min_overlap_fraction=0.48,
        soma_anchor_min_separation_um=3.8,
        soma_part_max_axis_ratio=4.50,
        soma_core_shell_max_um=0.75,
        soma_trusted_core_radius_scale=1.45,
        soma_trusted_core_max_um=3.20,
        soma_trusted_core_nucleus_margin_um=0.60,
        soma_nucleus_shape_preserving=True,
        instance_split_min_anchor_score=0.66,
        instance_split_min_anchor_separation_um=4.0,
        instance_split_min_child_area_um2=12.0,
        instance_split_min_child_fraction=0.10,
        instance_split_max_neck_core_ratio=0.82,
        instance_split_max_boundary_structural_ratio=0.72,
        instance_split_max_markers=4,
        instance_split_strategy="neonatal_multi",
    )

def split_astrocyte_compartments(
    whole_mask: np.ndarray,
    dapi_projection: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: CompartmentConfig | None = None,
    validated_anchors: ValidatedNucleusAnchors | None = None,
    ownership_inventory: ValidatedNucleusAnchors | None = None,
    ownership_pixel_depth_um: float | None = None,
    ownership_profile: str = "mature",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Partition each Whole ROI into its soma union and exact process complement."""

    cfg = config or CompartmentConfig()
    if not whole_mask.any():
        raise ValueError("Cannot split compartments because the Whole ROI mask is empty")

    mean_pixel_um = max(1e-4, math.sqrt(pixel_width_um * pixel_height_um))
    pixel_area_um2 = pixel_width_um * pixel_height_um
    nuclei, nuclei_extent, dapi_norm, dapi_extent_metrics = dapi_nuclei_core_and_extent(
        dapi_projection,
        mean_pixel_um,
        cfg,
    )
    preserved_object_nuclei_labels = None
    if validated_anchors is not None:
        accepted_core = np.asarray(
            validated_anchors.accepted_core_mask_2d,
            dtype=bool,
        )
        accepted_extent = np.asarray(
            validated_anchors.accepted_extent_mask_2d,
            dtype=bool,
        )
        if accepted_core.shape != nuclei.shape or accepted_extent.shape != nuclei.shape:
            raise ValueError(
                "Validated neonatal nucleus masks do not match the 2D compartment geometry"
            )
        pre_validation_core_count = int(measure.label(nuclei, connectivity=2).max())
        nuclei &= accepted_core
        nuclei_extent &= accepted_extent
        nuclei_extent |= nuclei
        dapi_extent_metrics["pre_3d_validation_core_count"] = pre_validation_core_count
        dapi_extent_metrics["post_3d_validation_core_count"] = int(
            measure.label(nuclei, connectivity=2).max()
        )
        dapi_extent_metrics["strict_core_px_after_3d_validation"] = int(nuclei.sum())
        dapi_extent_metrics["extent_px_after_3d_validation"] = int(nuclei_extent.sum())
        if validated_anchors.object_core_labels_2d is not None:
            object_core_labels = np.asarray(
                validated_anchors.object_core_labels_2d,
                dtype=np.uint32,
            )
            preserved_object_nuclei_labels = np.where(
                np.isin(object_core_labels, validated_anchors.accepted_object_ids),
                object_core_labels,
                0,
            ).astype(np.uint32)
            dapi_extent_metrics["preserved_3d_object_id_count"] = len(
                np.unique(preserved_object_nuclei_labels)
            ) - int(np.any(preserved_object_nuclei_labels == 0))
    refined_whole_mask, branch_gap_metrics = restore_low_support_branch_gaps(
        whole_mask,
        dapi_projection,
        struct,
        pixel_width_um,
        pixel_height_um,
        cfg,
        nuclei_mask=nuclei,
    )
    link_radius_px = max(3, int(round(cfg.nucleus_link_um / mean_pixel_um)))
    min_zone_px = max(link_radius_px + 2, int(round(cfg.soma_zone_min_um / mean_pixel_um)))
    max_zone_px = max(min_zone_px, int(round(cfg.soma_zone_max_um / mean_pixel_um)))
    fallback_radius_px = max(4, int(round(cfg.fallback_soma_radius_um / mean_pixel_um)))
    min_soma_area_px = max(40, int(round(cfg.min_soma_area_um2 / pixel_area_um2)))

    nuclei_labels = (
        preserved_object_nuclei_labels
        if preserved_object_nuclei_labels is not None
        else measure.label(nuclei, connectivity=2)
    )
    validated_grouped_extent_labels = None
    validated_group_extent_areas = None
    validated_group_by_id: dict[int, dict] = {}
    validated_anchor_minimum_overlap_px = 1
    if (
        validated_anchors is not None
        and ownership_pixel_depth_um is not None
        and ownership_pixel_depth_um > 0
        and validated_anchors.object_extent_labels_2d is not None
    ):
        ownership_config = NucleusOwnershipConfig()
        validated_groups = group_inventory_nucleus_objects(
            validated_anchors,
            pixel_width_um,
            pixel_height_um,
            ownership_pixel_depth_um,
            ownership_config,
        )
        object_extent_labels = np.asarray(
            validated_anchors.object_extent_labels_2d,
            dtype=np.uint32,
        )
        object_to_group = np.zeros(
            int(object_extent_labels.max()) + 1,
            dtype=np.uint32,
        )
        for group in validated_groups:
            group_id = int(group["group_id"])
            validated_group_by_id[group_id] = group
            for object_id in group["object_ids"]:
                object_to_group[int(object_id)] = group_id
        validated_grouped_extent_labels = object_to_group[object_extent_labels]
        validated_group_extent_areas = np.bincount(
            validated_grouped_extent_labels.ravel(),
            minlength=len(object_to_group),
        )
        validated_anchor_minimum_overlap_px = max(
            1,
            int(
                math.ceil(
                    ownership_config.owner_min_overlap_um2 / pixel_area_um2
                )
            ),
        )
    if nuclei.any():
        nucleus_distance, nearest_indices = ndi.distance_transform_edt(
            ~nuclei,
            return_indices=True,
        )
        nearest_nucleus_labels = nuclei_labels[nearest_indices[0], nearest_indices[1]]
    else:
        nucleus_distance = np.full(nuclei.shape, np.inf, dtype=np.float32)
        nearest_nucleus_labels = np.zeros(nuclei.shape, dtype=np.int32)
    if cfg.instance_split_strategy == "pairwise_soma_anchor_split":
        instance_splitter = split_touching_whole_instances
    elif cfg.instance_split_strategy == "neonatal_multi":
        instance_splitter = split_touching_whole_instances_multi
    else:
        raise ValueError(f"Unknown instance split strategy: {cfg.instance_split_strategy}")
    labels, instance_metrics = instance_splitter(
        refined_whole_mask,
        nuclei_labels,
        nearest_nucleus_labels,
        nucleus_distance,
        struct,
        cellpose_mask,
        mean_pixel_um,
        pixel_area_um2,
        link_radius_px,
        cfg,
    )
    labels, nucleus_ownership_metrics = apply_nucleus_ownership_guard(
        labels,
        struct,
        ownership_inventory,
        pixel_width_um,
        pixel_height_um,
        ownership_pixel_depth_um,
        ownership_profile,
    )
    validated_groups_by_instance_id: dict[int, set[int]] = {}
    if validated_grouped_extent_labels is not None:
        explicit_group_owner: dict[int, int] = {}
        for decision in nucleus_ownership_metrics.get("decisions", []):
            output_ids = [
                int(value) for value in decision.get("output_instance_ids", [])
            ]
            if not output_ids:
                continue
            owner_group_id = int(decision.get("owner_group_id", 0))
            if owner_group_id > 0:
                explicit_group_owner[owner_group_id] = output_ids[0]
            accepted_foreign_group_ids = [
                int(row["group_id"])
                for row in decision.get("foreign_groups", [])
                if bool(row.get("accepted"))
            ]
            for group_id, output_id in zip(
                accepted_foreign_group_ids,
                output_ids[1:],
            ):
                explicit_group_owner[group_id] = output_id
        for group_id, group in validated_group_by_id.items():
            if not bool(group["accepted"]):
                continue
            if group_id in explicit_group_owner:
                validated_groups_by_instance_id.setdefault(
                    explicit_group_owner[group_id],
                    set(),
                ).add(int(group_id))
                continue
            overlapping_labels = labels[validated_grouped_extent_labels == group_id]
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            if overlapping_labels.size == 0:
                continue
            label_ids, label_counts = np.unique(
                overlapping_labels,
                return_counts=True,
            )
            winner_index = int(np.argmax(label_counts))
            if int(label_counts[winner_index]) < validated_anchor_minimum_overlap_px:
                continue
            group_extent_area_px = int(validated_group_extent_areas[group_id])
            if (
                int(label_counts[winner_index]) / max(group_extent_area_px, 1)
                < ownership_config.accepted_min_extent_overlap_fraction
            ):
                continue
            owner_label = int(label_ids[winner_index])
            validated_groups_by_instance_id.setdefault(owner_label, set()).add(
                int(group_id)
            )
    ownership_id_mapping = {
        int(old_id): [int(value) for value in new_ids]
        for old_id, new_ids in nucleus_ownership_metrics.get(
            "input_to_output_ids",
            {},
        ).items()
    }
    if ownership_id_mapping:
        for detail in instance_metrics.get("split_components", []):
            pre_guard_ids = [int(value) for value in detail.get("new_astrocyte_ids", [])]
            detail["pre_ownership_guard_new_astrocyte_ids"] = pre_guard_ids
            detail["new_astrocyte_ids"] = [
                mapped_id
                for old_id in pre_guard_ids
                for mapped_id in ownership_id_mapping.get(old_id, [])
            ]
        for decision in instance_metrics.get("component_decisions", []):
            pre_guard_ids = [
                int(value) for value in decision.get("output_astrocyte_ids", [])
            ]
            decision["pre_ownership_guard_output_astrocyte_ids"] = pre_guard_ids
            decision["output_astrocyte_ids"] = [
                mapped_id
                for old_id in pre_guard_ids
                for mapped_id in ownership_id_mapping.get(old_id, [])
            ]
    roi_count = int(labels.max())
    soma_labels = np.zeros_like(labels, dtype=np.uint16)
    process_labels = np.zeros_like(labels, dtype=np.uint16)
    per_cell: list[dict] = []
    fallback_count = 0
    ambiguous_count = 0
    no_dapi_count = 0
    multi_soma_roi_count = 0
    total_soma_anchor_count = 0
    rejected_soma_anchor_count = 0
    dapi_extent_satellite_component_count = 0
    dapi_extent_satellite_px = 0
    component_properties = {prop.label: prop for prop in measure.regionprops(labels)}
    crop_padding = max_zone_px + link_radius_px + 4

    for astrocyte_id in range(1, roi_count + 1):
        prop = component_properties[astrocyte_id]
        min_row, min_col, max_row, max_col = prop.bbox
        row0 = max(0, min_row - crop_padding)
        col0 = max(0, min_col - crop_padding)
        row1 = min(labels.shape[0], max_row + crop_padding)
        col1 = min(labels.shape[1], max_col + crop_padding)
        crop = np.s_[row0:row1, col0:col1]
        component = labels[crop] == astrocyte_id
        local_struct = struct[crop]
        local_cellpose = cellpose_mask[crop]
        local_nuclei_labels = nuclei_labels[crop]
        local_nuclei_extent = nuclei_extent[crop]
        local_dapi_norm = dapi_norm[crop]
        local_nearest_labels = nearest_nucleus_labels[crop]
        local_nucleus_distance = nucleus_distance[crop]
        local_validated_grouped_extent_labels = (
            validated_grouped_extent_labels[crop]
            if validated_grouped_extent_labels is not None
            else None
        )
        component_area = int(component.sum())
        distance = ndi.distance_transform_edt(component)
        scored_nuclei, ambiguous = score_nuclei_for_component(
            component,
            local_nearest_labels,
            local_nucleus_distance,
            distance,
            local_struct,
            local_cellpose,
            link_radius_px,
            cfg.ambiguity_score_delta,
        )
        if local_validated_grouped_extent_labels is not None:
            anchor_groups = select_validated_soma_anchor_groups(
                scored_nuclei,
                component,
                local_nuclei_labels,
                local_validated_grouped_extent_labels,
                validated_group_by_id,
                validated_anchor_minimum_overlap_px,
                validated_groups_by_instance_id.get(astrocyte_id, set()),
            )
        else:
            anchor_groups = select_soma_anchor_groups(
                scored_nuclei,
                mean_pixel_um,
                cfg,
            )
        nucleus_candidates = len(scored_nuclei)
        nucleus_score = float(scored_nuclei[0]["score"]) if scored_nuclei else 0.0
        ambiguous_count += int(ambiguous)

        soma_parts: list[np.ndarray] = []
        anchor_details: list[dict] = []
        fallback_used = False
        rejected_anchor_count = 0
        distance_scale = max(float(np.percentile(distance[component], 99.0)), 1.0)

        for anchor in anchor_groups:
            nucleus_ids = anchor["nucleus_ids"]
            if nucleus_ids:
                selected_nucleus = np.isin(local_nuclei_labels, nucleus_ids)
                selected_voronoi = np.isin(local_nearest_labels, nucleus_ids)
                selected_nucleus_distance_um = ndi.distance_transform_edt(
                    ~selected_nucleus,
                    sampling=(pixel_height_um, pixel_width_um),
                )
                selected_nucleus_extent = (
                    local_nuclei_extent
                    & selected_voronoi
                    & (selected_nucleus_distance_um <= cfg.dapi_extent_max_expand_um)
                )
                selected_nucleus_extent |= selected_nucleus
                selected_nucleus_extent = morphology.binary_closing(
                    selected_nucleus_extent,
                    footprint=morphology.disk(1),
                )
                (
                    selected_nucleus_extent,
                    removed_extent_components,
                    removed_extent_px,
                ) = retain_primary_anchor_extent(
                    selected_nucleus_extent,
                    selected_nucleus,
                    component,
                )
                dapi_extent_satellite_component_count += removed_extent_components
                dapi_extent_satellite_px += removed_extent_px
                selected_nucleus_distance = ndi.distance_transform_edt(
                    ~selected_nucleus_extent
                )
                selected_nucleus_extent_distance_um = ndi.distance_transform_edt(
                    ~selected_nucleus_extent,
                    sampling=(pixel_height_um, pixel_width_um),
                )
                search_region = component & (selected_nucleus_distance <= link_radius_px)
            else:
                selected_nucleus = np.zeros_like(component, dtype=bool)
                selected_nucleus_extent = np.zeros_like(component, dtype=bool)
                selected_nucleus_distance = np.full(component.shape, np.inf, dtype=np.float32)
                selected_nucleus_extent_distance_um = np.full(
                    component.shape,
                    np.inf,
                    dtype=np.float32,
                )
                search_region = component.copy()
            if not search_region.any():
                search_region = component.copy()

            nucleus_proximity = (
                np.exp(-np.square(selected_nucleus_distance / max(link_radius_px, 1)))
                if nucleus_ids
                else np.zeros_like(distance, dtype=np.float32)
            )
            seed_score = (
                0.64 * np.clip(distance / distance_scale, 0, 1)
                + 0.23 * local_struct
                + 0.05 * local_cellpose.astype(np.float32)
                + 0.08 * nucleus_proximity
            )
            seed_score = np.where(search_region, seed_score, -np.inf)
            seed_y, seed_x = np.unravel_index(int(np.argmax(seed_score)), seed_score.shape)
            seed_point = np.zeros_like(component, dtype=bool)
            seed_point[seed_y, seed_x] = True

            core_neighborhood_radius = max(
                link_radius_px,
                int(round(0.75 / mean_pixel_um)),
            )
            core_neighborhood = component & circular_mask(
                component.shape,
                seed_y,
                seed_x,
                core_neighborhood_radius,
            )
            core_peak_px = max(
                float(np.percentile(distance[core_neighborhood], 90.0)),
                0.55 / mean_pixel_um,
            )
            thin_cut = max(1.5, 0.42 * core_peak_px)
            thin_fraction = float((distance[component] <= thin_cut).mean())
            process_richness = float(np.clip((thin_fraction - 0.20) / 0.55, 0, 1))

            zone_scale = (
                cfg.soma_zone_scale_compact
                + process_richness
                * (cfg.soma_zone_scale_process_rich - cfg.soma_zone_scale_compact)
            )
            zone_radius_px = int(np.clip(round(zone_scale * core_peak_px), min_zone_px, max_zone_px))
            thickness_fraction = (
                cfg.thickness_fraction_compact
                + process_richness
                * (cfg.thickness_fraction_process_rich - cfg.thickness_fraction_compact)
            )
            structural_percentile = (
                cfg.structural_percentile_compact
                + process_richness
                * (cfg.structural_percentile_process_rich - cfg.structural_percentile_compact)
            )
            thickness_cut = max(0.32 / mean_pixel_um, core_peak_px * thickness_fraction)
            structural_cut = float(np.percentile(local_struct[component], structural_percentile))

            if nucleus_ids:
                soma_zone = selected_nucleus_distance <= zone_radius_px
            else:
                soma_zone = circular_mask(
                    component.shape,
                    seed_y,
                    seed_x,
                    zone_radius_px,
                )
            secondary_thickness_cut = max(1.5, 0.14 * core_peak_px)
            soma_domain = component & soma_zone & (
                (distance >= thickness_cut)
                | ((local_struct >= structural_cut) & (distance >= secondary_thickness_cut))
            )
            seed_radius_px = max(2, min(fallback_radius_px, int(round(0.35 * core_peak_px))))
            soma_seed = component & circular_mask(
                component.shape,
                seed_y,
                seed_x,
                seed_radius_px,
            )
            required_nucleus = component & selected_nucleus_extent
            soma_domain |= soma_seed | required_nucleus
            soma_seed |= required_nucleus
            soma_part = ndi.binary_propagation(
                soma_seed,
                structure=np.ones((3, 3), dtype=bool),
                mask=soma_domain,
            ).astype(bool)
            soma_part = morphology.binary_closing(soma_part, footprint=morphology.disk(2)) & component
            soma_part = morphology.remove_small_holes(
                soma_part,
                area_threshold=max(16, int(round(0.35 / pixel_area_um2))),
            ) & component

            part_fallback = False
            soma_fraction = float(soma_part.sum()) / max(component_area, 1)
            if int(soma_part.sum()) < min_soma_area_px or soma_fraction > cfg.max_soma_fraction:
                part_fallback = True
                if nucleus_ids and cfg.soma_nucleus_shape_preserving:
                    fallback_zone = (
                        selected_nucleus_extent_distance_um
                        <= cfg.fallback_soma_radius_um
                    )
                    fallback_domain = component & fallback_zone & (
                        (distance >= secondary_thickness_cut)
                        | (local_struct >= structural_cut)
                    )
                else:
                    fallback_zone = circular_mask(
                        component.shape,
                        seed_y,
                        seed_x,
                        fallback_radius_px,
                    )
                    fallback_domain = component & fallback_zone
                fallback_domain |= component & selected_nucleus_extent
                fallback_soma = ndi.binary_propagation(
                    seed_point | required_nucleus,
                    structure=np.ones((3, 3), dtype=bool),
                    mask=fallback_domain,
                ).astype(bool)
                fallback_soma = morphology.binary_closing(
                    fallback_soma,
                    footprint=morphology.disk(2),
                ) & component
                if fallback_soma.any():
                    soma_part = fallback_soma
            part_properties = measure.regionprops(measure.label(soma_part, connectivity=2))
            if part_properties:
                soma_property = max(part_properties, key=lambda item: item.area)
                axis_ratio = float(soma_property.major_axis_length) / max(
                    float(soma_property.minor_axis_length),
                    1e-6,
                )
            else:
                axis_ratio = math.inf
            core_radius_um = core_peak_px * mean_pixel_um
            if (
                not soma_part.any()
                or core_radius_um < cfg.soma_part_min_core_radius_um
                or axis_ratio > cfg.soma_part_max_axis_ratio
            ):
                rejected_anchor_count += 1
                continue
            fallback_used |= part_fallback
            soma_parts.append(soma_part)
            trusted_core_radius_um = float(
                np.clip(
                    cfg.soma_trusted_core_radius_scale * core_peak_px * mean_pixel_um,
                    cfg.soma_trusted_core_min_um,
                    cfg.soma_trusted_core_max_um,
                )
            )
            trusted_core_radius_px = max(
                seed_radius_px,
                int(round(trusted_core_radius_um / mean_pixel_um)),
            )
            if nucleus_ids and cfg.soma_nucleus_shape_preserving:
                trusted_core_zone = (
                    selected_nucleus_extent_distance_um <= trusted_core_radius_um
                )
            else:
                trusted_core_zone = circular_mask(
                    component.shape,
                    seed_y,
                    seed_x,
                    trusted_core_radius_px,
                )
            nucleus_trusted_zone = (
                selected_nucleus_extent_distance_um
                <= cfg.soma_trusted_core_nucleus_margin_um
            )
            trusted_core = soma_part & trusted_core_zone & nucleus_trusted_zone & (
                (distance >= thickness_cut) | soma_seed
            )
            trusted_core |= required_nucleus
            trusted_core |= seed_point
            anchor_details.append(
                {
                    "seed_y": seed_y,
                    "seed_x": seed_x,
                    "selected_nucleus": selected_nucleus,
                    "selected_nucleus_extent": required_nucleus,
                    "required_nucleus_px": int(required_nucleus.sum()),
                    "required_nucleus_mean": round(
                        float(local_dapi_norm[required_nucleus].mean())
                        if required_nucleus.any()
                        else 0.0,
                        6,
                    ),
                    "core_peak_px": core_peak_px,
                    "thin_fraction": thin_fraction,
                    "process_richness": process_richness,
                    "zone_radius_px": zone_radius_px,
                    "score": float(anchor["score"]),
                    "axis_ratio": axis_ratio,
                    "seed_point": seed_point,
                    "trusted_core": trusted_core,
                    "trusted_core_radius_um": trusted_core_radius_um,
                    "anchor_source": anchor.get("source", "2d_scored_nucleus"),
                    "validated_group_id": anchor.get("validated_group_id"),
                }
            )

        soma_anchor_count = len(soma_parts)
        no_dapi_count += int(soma_anchor_count == 0)
        multi_soma_roi_count += int(soma_anchor_count > 1)
        total_soma_anchor_count += soma_anchor_count
        rejected_soma_anchor_count += rejected_anchor_count
        soma = (
            np.logical_or.reduce(soma_parts) & component
            if soma_parts
            else np.zeros_like(component, dtype=bool)
        )
        process = component & ~soma
        if soma.any() and float(process.sum()) / max(component_area, 1) < cfg.min_process_fraction:
            fallback_used = True
            radial_limit = max(4, int(round(fallback_radius_px * 0.85)))
            restricted = np.zeros_like(component, dtype=bool)
            for detail in anchor_details:
                restricted |= component & circular_mask(
                    component.shape,
                    detail["seed_y"],
                    detail["seed_x"],
                    radial_limit,
                )
                restricted |= detail["selected_nucleus_extent"]
            if restricted.any():
                soma = restricted
                process = component & ~soma

        soma_area_before_core_shell_px = int(soma.sum())
        soma_core_shell_applied = False
        if soma.any() and anchor_details:
            trusted_core_union = np.logical_or.reduce(
                [detail["trusted_core"] for detail in anchor_details]
            ) & soma
            anchor_seed_union = np.logical_or.reduce(
                [
                    detail["seed_point"] | detail["selected_nucleus_extent"]
                    for detail in anchor_details
                ]
            ) & soma
            soma, soma_core_shell_applied = prune_soma_to_trusted_core_shell(
                soma,
                trusted_core_union,
                anchor_seed_union,
                pixel_width_um,
                pixel_height_um,
                cfg.soma_core_shell_max_um,
                min_soma_area_px,
            )
            process = component & ~soma
            required_nucleus_union = np.logical_or.reduce(
                [detail["selected_nucleus_extent"] for detail in anchor_details]
            ) & component
            missing_required_nucleus_px = int((required_nucleus_union & ~soma).sum())
            if missing_required_nucleus_px:
                raise RuntimeError(
                    f"Astrocyte_{astrocyte_id:03d} Soma lost "
                    f"{missing_required_nucleus_px} protected DAPI pixels"
                )
        else:
            required_nucleus_union = np.zeros_like(component, dtype=bool)
        soma_core_shell_removed_px = soma_area_before_core_shell_px - int(soma.sum())

        if not process.any():
            raise RuntimeError(
                f"Astrocyte_{astrocyte_id:03d} could not retain a non-empty Processes compartment"
            )
        fallback_count += int(fallback_used)
        soma_view = soma_labels[crop]
        process_view = process_labels[crop]
        soma_view[soma] = astrocyte_id
        process_view[process] = astrocyte_id
        per_cell.append(
            {
                "astrocyte_id": astrocyte_id,
                "whole_area_px": component_area,
                "soma_area_px": int(soma.sum()),
                "process_area_px": int(process.sum()),
                "soma_fraction": round(float(soma.sum()) / component_area, 6),
                "process_fraction": round(float(process.sum()) / component_area, 6),
                "process_richness": round(
                    float(np.mean([detail["process_richness"] for detail in anchor_details]))
                    if anchor_details else 1.0,
                    6,
                ),
                "thin_fraction": round(
                    float(np.mean([detail["thin_fraction"] for detail in anchor_details]))
                    if anchor_details else 1.0,
                    6,
                ),
                "core_peak_px": round(
                    max((detail["core_peak_px"] for detail in anchor_details), default=0.0),
                    3,
                ),
                "soma_zone_radius_px": max(
                    (detail["zone_radius_px"] for detail in anchor_details),
                    default=0,
                ),
                "nucleus_candidates": nucleus_candidates,
                "nucleus_score": round(nucleus_score, 6),
                "nucleus_ambiguous": bool(ambiguous),
                "soma_anchor_count": soma_anchor_count,
                "soma_anchor_scores": [
                    round(float(detail["score"]), 6) for detail in anchor_details
                ],
                "soma_anchor_sources": [
                    str(detail["anchor_source"]) for detail in anchor_details
                ],
                "validated_soma_group_ids": [
                    int(detail["validated_group_id"])
                    for detail in anchor_details
                    if detail["validated_group_id"] is not None
                ],
                "rejected_soma_anchor_count": rejected_anchor_count,
                "fallback_used": bool(fallback_used),
                "soma_area_before_core_shell_px": soma_area_before_core_shell_px,
                "soma_core_shell_removed_px": soma_core_shell_removed_px,
                "soma_core_shell_applied": bool(soma_core_shell_applied),
                "required_nucleus_px": int(required_nucleus_union.sum()),
                "required_nucleus_coverage": round(
                    float((required_nucleus_union & soma).sum())
                    / max(int(required_nucleus_union.sum()), 1),
                    6,
                ),
            }
        )

    pre_filter_roi_count = int(labels.max())
    labels, soma_labels, process_labels, per_cell, morphology_filter_metrics = (
        filter_morphology_outlier_instances(
            labels,
            soma_labels,
            process_labels,
            struct,
            mean_pixel_um,
            pixel_area_um2,
            per_cell,
            instance_metrics,
            cfg,
        )
    )
    roi_count = int(labels.max())
    id_mapping = {
        int(old_id): int(new_id)
        for old_id, new_id in morphology_filter_metrics["id_mapping"].items()
    }
    for detail in instance_metrics.get("split_components", []):
        original_ids = [int(value) for value in detail["new_astrocyte_ids"]]
        detail["pre_filter_new_astrocyte_ids"] = original_ids
        detail["new_astrocyte_ids"] = [
            id_mapping[value] for value in original_ids if value in id_mapping
        ]
    instance_metrics["pre_morphology_filter_instance_count"] = pre_filter_roi_count
    instance_metrics["final_instance_count"] = roi_count

    final_whole_mask = labels > 0
    soma_mask = soma_labels > 0
    process_mask = process_labels > 0
    overlap_px = int((soma_mask & process_mask).sum())
    gap_px = int((final_whole_mask & ~(soma_mask | process_mask)).sum())
    outside_px = int(((soma_mask | process_mask) & ~final_whole_mask).sum())
    if overlap_px or gap_px or outside_px:
        raise RuntimeError(
            "Compartment partition invariant failed: "
            f"overlap={overlap_px}, gap={gap_px}, outside={outside_px}"
        )

    whole_area_px = int(final_whole_mask.sum())
    soma_area_px = int(soma_mask.sum())
    process_area_px = int(process_mask.sum())
    metrics = {
        "method": "connectivity-preserving low-support branch-gap restoration + high-confidence DAPI/structural marker-controlled instance partition + single-body assigned-DAPI extent protection + local thickness/core-shell Soma + conservative multimetric whole-ID morphology filtering; Processes=Whole-Soma; no trusted Soma is forced",
        "adaptation": "continuous per-cell morphology adaptation; no animal-age label inferred",
        "config": asdict(cfg),
        "pixel_width_um": pixel_width_um,
        "pixel_height_um": pixel_height_um,
        "roi_count": roi_count,
        "whole_area_px": whole_area_px,
        "soma_area_px": soma_area_px,
        "process_area_px": process_area_px,
        "soma_area_fraction": round(soma_area_px / whole_area_px, 6),
        "process_area_fraction": round(process_area_px / whole_area_px, 6),
        "fallback_soma_count": int(sum(bool(row["fallback_used"]) for row in per_cell)),
        "ambiguous_nucleus_count": int(
            sum(bool(row["nucleus_ambiguous"]) for row in per_cell)
        ),
        "no_dapi_anchor_count": int(
            sum(int(row["soma_anchor_count"]) == 0 for row in per_cell)
        ),
        "total_soma_anchor_count": int(
            sum(int(row["soma_anchor_count"]) for row in per_cell)
        ),
        "multi_soma_whole_roi_count": int(
            sum(int(row["soma_anchor_count"]) > 1 for row in per_cell)
        ),
        "rejected_soma_anchor_count": int(
            sum(int(row["rejected_soma_anchor_count"]) for row in per_cell)
        ),
        "instance_split": instance_metrics,
        "nucleus_ownership_guard": nucleus_ownership_metrics,
        "branch_gap_restoration": branch_gap_metrics,
        "dapi_extent": dapi_extent_metrics,
        "dapi_extent_satellite_components_removed": int(
            dapi_extent_satellite_component_count
        ),
        "dapi_extent_satellite_px_removed": int(dapi_extent_satellite_px),
        "morphology_filter": morphology_filter_metrics,
        "soma_core_shell_removed_px": int(
            sum(row["soma_core_shell_removed_px"] for row in per_cell)
        ),
        "soma_core_shell_applied_roi_count": int(
            sum(bool(row["soma_core_shell_applied"]) for row in per_cell)
        ),
        "partition_overlap_px": overlap_px,
        "partition_gap_px": gap_px,
        "partition_outside_whole_px": outside_px,
        "per_cell": per_cell,
    }
    return labels, soma_labels, process_labels, metrics
