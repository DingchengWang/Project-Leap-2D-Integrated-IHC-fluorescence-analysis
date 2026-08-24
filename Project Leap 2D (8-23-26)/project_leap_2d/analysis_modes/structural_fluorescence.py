# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def _linear_evidence(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("Age-profile evidence bounds must increase")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))

def classify_age_profile(
    whole_mask: np.ndarray,
    dapi_projection: np.ndarray,
    struct: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
) -> AgeProfileDecision:
    """Return a deterministic binary morphology profile from structural channels only."""

    mask = whole_mask.astype(bool, copy=False)
    labels = measure.label(mask, connectivity=2)
    pixel_area_um2 = pixel_width_um * pixel_height_um
    mean_pixel_um = math.sqrt(pixel_area_um2)
    all_props = sorted(measure.regionprops(labels), key=lambda prop: prop.area, reverse=True)
    minimum_area_px = max(32, int(round(12.0 / pixel_area_um2)))
    area_eligible = [prop for prop in all_props if int(prop.area) >= minimum_area_px]
    interior_eligible = [
        prop
        for prop in area_eligible
        if prop.bbox[0] > 0
        and prop.bbox[1] > 0
        and prop.bbox[2] < labels.shape[0]
        and prop.bbox[3] < labels.shape[1]
    ]
    eligible = interior_eligible if len(interior_eligible) >= 3 else area_eligible
    if not eligible:
        eligible = all_props
    if not eligible:
        raise ValueError("Cannot classify age profile from an empty Whole mask")
    eligible = eligible[:24]
    eligible_ids = {int(prop.label) for prop in eligible}

    nuclei = dapi_nuclei_mask(dapi_projection, percentile_floor=85.0)
    nuclei_labels = measure.label(nuclei, connectivity=2)
    nucleus_counts = {label_id: 0 for label_id in eligible_ids}
    if mask.any():
        distance_to_whole_um, nearest_indices = ndi.distance_transform_edt(
            ~mask,
            sampling=(pixel_height_um, pixel_width_um),
            return_indices=True,
        )
        nearest_whole_labels = labels[nearest_indices[0], nearest_indices[1]]
        for nucleus_prop in measure.regionprops(nuclei_labels):
            coords = nucleus_prop.coords
            overlapping = labels[coords[:, 0], coords[:, 1]]
            overlapping = overlapping[overlapping > 0]
            assigned = 0
            if overlapping.size:
                ids, counts = np.unique(overlapping, return_counts=True)
                assigned = int(ids[int(np.argmax(counts))])
            else:
                cy = int(np.clip(round(nucleus_prop.centroid[0]), 0, labels.shape[0] - 1))
                cx = int(np.clip(round(nucleus_prop.centroid[1]), 0, labels.shape[1] - 1))
                if distance_to_whole_um[cy, cx] <= 1.2:
                    assigned = int(nearest_whole_labels[cy, cx])
            if assigned in nucleus_counts:
                nucleus_counts[assigned] += 1

    widths_um: list[float] = []
    thin_fractions: list[float] = []
    solidities: list[float] = []
    axis_ratios: list[float] = []
    branch_densities: list[float] = []
    endpoint_densities: list[float] = []
    for prop in eligible:
        min_row, min_col, max_row, max_col = prop.bbox
        component = labels[min_row:max_row, min_col:max_col] == prop.label
        distance_um = ndi.distance_transform_edt(
            component,
            sampling=(pixel_height_um, pixel_width_um),
        )
        skeleton_px, endpoint_count, branchpoint_count = skeleton_topology(component)
        skeleton_length_um = max(skeleton_px * mean_pixel_um, mean_pixel_um)
        area_um2 = float(prop.area) * pixel_area_um2
        widths_um.append(area_um2 / skeleton_length_um)
        thin_fractions.append(float((distance_um[component] <= 0.70).mean()))
        solidities.append(float(prop.solidity))
        axis_ratios.append(
            float(prop.major_axis_length) / max(float(prop.minor_axis_length), 1e-6)
        )
        branch_densities.append(10.0 * branchpoint_count / skeleton_length_um)
        endpoint_densities.append(10.0 * endpoint_count / skeleton_length_um)

    median_width_um = float(np.median(widths_um))
    median_thin_fraction = float(np.median(thin_fractions))
    median_solidity = float(np.median(solidities))
    median_axis_ratio = float(np.median(axis_ratios))
    median_branch_density = float(np.median(branch_densities))
    median_endpoint_density = float(np.median(endpoint_densities))
    multi_nucleus_fraction = float(
        np.mean([nucleus_counts[int(prop.label)] >= 2 for prop in eligible])
    )

    evidence = {
        "broad_structure": _linear_evidence(median_width_um, 0.95, 2.20),
        "solid_structure": _linear_evidence(median_solidity, 0.24, 0.54),
        "limited_thin_arbor": 1.0
        - _linear_evidence(median_thin_fraction, 0.62, 0.88),
        "limited_branching": 1.0
        - _linear_evidence(median_branch_density, 0.14, 0.50),
        "polarized_shape": _linear_evidence(median_axis_ratio, 1.70, 3.20),
        "multi_nucleus_overlap": _linear_evidence(multi_nucleus_fraction, 0.05, 0.30),
    }
    neonatal_score = (
        0.26 * evidence["broad_structure"]
        + 0.20 * evidence["solid_structure"]
        + 0.18 * evidence["limited_thin_arbor"]
        + 0.16 * evidence["limited_branching"]
        + 0.10 * evidence["polarized_shape"]
        + 0.10 * evidence["multi_nucleus_overlap"]
    )
    threshold = AGE_PROFILE_THRESHOLD
    profile = "neonatal" if neonatal_score >= threshold else "mature"
    features: dict[str, float | int] = {
        "component_count": len(eligible),
        "edge_components_excluded": max(0, len(area_eligible) - len(eligible)),
        "median_width_um": round(median_width_um, 6),
        "median_thin_fraction": round(median_thin_fraction, 6),
        "median_solidity": round(median_solidity, 6),
        "median_axis_ratio": round(median_axis_ratio, 6),
        "median_branchpoints_per_10um": round(median_branch_density, 6),
        "median_endpoints_per_10um": round(median_endpoint_density, 6),
        "multi_nucleus_component_fraction": round(multi_nucleus_fraction, 6),
        **{f"evidence_{key}": round(value, 6) for key, value in evidence.items()},
    }
    return AgeProfileDecision(
        profile=profile,
        source="morphology_classifier",
        neonatal_score=round(float(neonatal_score), 6),
        threshold=threshold,
        confidence_margin=round(abs(float(neonatal_score) - threshold), 6),
        tagged_files=(),
        features=features,
    )

def split_astrocyte_compartments_for_profile(
    whole_mask: np.ndarray,
    dapi_projection: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    profile: str,
    neonatal_3d_context: Neonatal3DContext | None = None,
    dapi_fragment_workload_diagnostic_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply shared 3D ownership before profile-specific Soma/Processes rules."""

    mature_config = compartment_config_for_profile("mature")
    ownership_inventory = None
    if neonatal_3d_context is not None:
        ownership_started = time.perf_counter()
        ownership_inventory = build_dapi_object_inventory_3d(
            whole_mask,
            dapi_projection,
            neonatal_3d_context,
            pixel_width_um,
            pixel_height_um,
            mature_config,
            max_workers=_EFFECTIVE_DAPI_INVENTORY_CPU_WORKERS,
            workload_diagnostic_path=dapi_fragment_workload_diagnostic_path,
        )
        nucleus_inventory_metrics = ownership_inventory.metrics
        print(
            "3D DAPI nucleus ownership complete | "
            f"elapsed={time.perf_counter() - ownership_started:.3f} s; "
            "partitioning Whole/Soma/Processes...",
            flush=True,
        )
    else:
        nucleus_inventory_metrics = {
            "status": "not_run_structural_stack_or_Z_calibration_unavailable",
            "method": "object-preserving calibrated 3D DAPI inventory",
            "measurement_channel_used": False,
            "candidate_count": 0,
            "dapi_valid_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "per_nucleus": [],
        }
    if profile == "mature":
        base_partition_started = time.perf_counter()
        labels, soma_labels, process_labels, metrics = split_astrocyte_compartments(
            whole_mask,
            dapi_projection,
            struct,
            cellpose_mask,
            pixel_width_um,
            pixel_height_um,
            config=mature_config,
            ownership_inventory=ownership_inventory,
            ownership_pixel_depth_um=(
                neonatal_3d_context.pixel_depth_um
                if neonatal_3d_context is not None
                else None
            ),
            ownership_profile="mature",
        )
        pre_gate_labels = labels.copy()
        pre_gate_roi_count = int(labels.max())
        (
            labels,
            soma_labels,
            process_labels,
            gated_per_cell,
            soma_identity_gate,
        ) = filter_instances_by_valid_soma(
            labels,
            soma_labels,
            process_labels,
            metrics["per_cell"],
            profile="mature",
        )
        retained_pre_gate_ids = {
            int(value) for value in soma_identity_gate["retained_pre_gate_ids"]
        }
        expected_final_mask = np.isin(pre_gate_labels, list(retained_pre_gate_ids))
        final_whole_mask = labels > 0
        if not np.array_equal(final_whole_mask, expected_final_mask):
            raise RuntimeError(
                "Mature valid-Soma gate changed retained geometry instead of deleting "
                "complete cell IDs"
            )
        if int(final_whole_mask.sum()) + int(soma_identity_gate["removed_area_px"]) != int(
            (pre_gate_labels > 0).sum()
        ):
            raise RuntimeError("Mature valid-Soma gate removed partial cell geometry")

        id_mapping = {
            int(old_id): int(new_id)
            for old_id, new_id in soma_identity_gate["id_mapping"].items()
        }
        for detail in metrics["instance_split"].get("split_components", []):
            pre_gate_ids = [int(value) for value in detail.get("new_astrocyte_ids", [])]
            detail["pre_soma_gate_new_astrocyte_ids"] = pre_gate_ids
            detail["removed_pre_soma_gate_ids"] = [
                value for value in pre_gate_ids if value not in id_mapping
            ]
            detail["new_astrocyte_ids"] = [
                id_mapping[value] for value in pre_gate_ids if value in id_mapping
            ]
        metrics["instance_split"]["pre_soma_gate_instance_count"] = pre_gate_roi_count
        metrics["instance_split"]["final_instance_count"] = int(labels.max())
        metrics["soma_identity_gate"] = soma_identity_gate
        metrics["per_cell"] = gated_per_cell
        metrics["roi_count"] = int(labels.max())
        metrics["whole_area_px"] = int(final_whole_mask.sum())
        metrics["soma_area_px"] = int((soma_labels > 0).sum())
        metrics["process_area_px"] = int((process_labels > 0).sum())
        metrics["soma_area_fraction"] = round(
            metrics["soma_area_px"] / metrics["whole_area_px"],
            6,
        )
        metrics["process_area_fraction"] = round(
            metrics["process_area_px"] / metrics["whole_area_px"],
            6,
        )
        metrics["fallback_soma_count"] = int(
            sum(bool(row["fallback_used"]) for row in gated_per_cell)
        )
        metrics["ambiguous_nucleus_count"] = int(
            sum(bool(row["nucleus_ambiguous"]) for row in gated_per_cell)
        )
        metrics["no_dapi_anchor_count"] = 0
        metrics["total_soma_anchor_count"] = len(gated_per_cell)
        metrics["multi_soma_whole_roi_count"] = 0
        metrics["method"] += (
            "; mature whole-ID valid-Soma gate with synchronized relabeling"
        )
        metrics["nucleus_3d_inventory"] = nucleus_inventory_metrics
        print(
            "Base compartment partition complete | profile=mature | "
            f"elapsed={time.perf_counter() - base_partition_started:.3f} s; "
            "finalizing identity and Soma safeguards...",
            flush=True,
        )
        finalization_started = time.perf_counter()
        labels, soma_labels, process_labels, metrics = (
            finalize_compartment_geometry_and_metrics(
                labels,
                soma_labels,
                process_labels,
                metrics,
                ownership_inventory,
                neonatal_3d_context,
                struct,
                "mature",
                pixel_width_um,
                pixel_height_um,
            )
        )
        print(
            "Compartment finalization complete | profile=mature | "
            f"elapsed={time.perf_counter() - finalization_started:.3f} s",
            flush=True,
        )
        return labels, soma_labels, process_labels, metrics
    if profile != "neonatal":
        raise ValueError(f"Unknown astrocyte profile: {profile}")

    base_partition_started = time.perf_counter()
    shared_whole_labels, _, _, shared_metrics = split_astrocyte_compartments(
        whole_mask,
        dapi_projection,
        struct,
        cellpose_mask,
        pixel_width_um,
        pixel_height_um,
        config=mature_config,
        ownership_inventory=ownership_inventory,
        ownership_pixel_depth_um=(
            neonatal_3d_context.pixel_depth_um
            if neonatal_3d_context is not None
            else None
        ),
        ownership_profile="neonatal_shared_whole",
    )
    frozen_whole_mask = shared_whole_labels > 0
    neonatal_config = replace(
        compartment_config_for_profile("neonatal"),
        branch_gap_restore_enabled=False,
        morphology_outlier_filter_enabled=False,
        instance_split_min_anchor_separation_um=4.0,
    )
    validated_anchors = ownership_inventory
    neonatal_3d_metrics = nucleus_inventory_metrics
    labels, soma_labels, process_labels, metrics = split_astrocyte_compartments(
        frozen_whole_mask,
        dapi_projection,
        struct,
        cellpose_mask,
        pixel_width_um,
        pixel_height_um,
        config=neonatal_config,
        validated_anchors=validated_anchors,
        ownership_inventory=ownership_inventory,
        ownership_pixel_depth_um=(
            neonatal_3d_context.pixel_depth_um
            if neonatal_3d_context is not None
            else None
        ),
        ownership_profile="neonatal",
    )
    if np.any((labels > 0) & ~frozen_whole_mask):
        raise RuntimeError(
            "Neonatal repartition expanded the frozen shared Whole pixel union before "
            "the valid-Soma cell gate"
        )
    pre_gate_roi_count = int(labels.max())
    pre_gate_labels = labels.copy()
    unresolved_multi_soma_ids = {
        int(astrocyte_id)
        for decision in metrics["instance_split"].get("component_decisions", [])
        if bool(decision.get("split_required"))
        and not bool(decision.get("split_accepted"))
        for astrocyte_id in decision.get("output_astrocyte_ids", [])
    }
    (
        labels,
        soma_labels,
        process_labels,
        gated_per_cell,
        soma_identity_gate,
    ) = filter_instances_by_valid_soma(
        labels,
        soma_labels,
        process_labels,
        metrics["per_cell"],
        profile="neonatal",
        unresolved_multi_soma_ids=unresolved_multi_soma_ids,
    )
    final_whole_mask = labels > 0
    if np.any(final_whole_mask & ~frozen_whole_mask):
        raise RuntimeError("Neonatal valid-Soma gate expanded the frozen Whole pixel union")
    retained_pre_gate_ids = {
        int(value) for value in soma_identity_gate["retained_pre_gate_ids"]
    }
    expected_final_mask = np.isin(
        pre_gate_labels,
        list(retained_pre_gate_ids),
    )
    if not np.array_equal(final_whole_mask, expected_final_mask):
        raise RuntimeError(
            "Neonatal valid-Soma gate changed retained cell geometry instead of only relabeling"
        )
    ownership_removed_area_px = int(
        metrics.get("nucleus_ownership_guard", {}).get("removed_area_px", 0)
    )
    if (
        int(final_whole_mask.sum())
        + int(soma_identity_gate["removed_area_px"])
        + ownership_removed_area_px
        != int(frozen_whole_mask.sum())
    ):
        raise RuntimeError(
            "Neonatal ownership and valid-Soma gates do not account for the frozen "
            "Whole geometry exactly"
        )
    del pre_gate_labels, expected_final_mask

    id_mapping = {
        int(old_id): int(new_id)
        for old_id, new_id in soma_identity_gate["id_mapping"].items()
    }
    for detail in metrics["instance_split"].get("split_components", []):
        pre_gate_ids = [int(value) for value in detail.get("new_astrocyte_ids", [])]
        pre_gate_areas = [int(value) for value in detail.get("child_areas_px", [])]
        detail["pre_soma_gate_new_astrocyte_ids"] = pre_gate_ids
        detail["pre_soma_gate_child_areas_px"] = pre_gate_areas
        detail["removed_pre_soma_gate_ids"] = [
            value for value in pre_gate_ids if value not in id_mapping
        ]
        detail["new_astrocyte_ids"] = [
            id_mapping[value] for value in pre_gate_ids if value in id_mapping
        ]
        detail["retained_child_areas_px"] = [
            area
            for value, area in zip(pre_gate_ids, pre_gate_areas)
            if value in id_mapping
        ]
    for decision in metrics["instance_split"].get("component_decisions", []):
        pre_gate_ids = [
            int(value) for value in decision.get("output_astrocyte_ids", [])
        ]
        decision["pre_soma_gate_output_astrocyte_ids"] = pre_gate_ids
        decision["output_astrocyte_ids"] = [
            id_mapping[value] for value in pre_gate_ids if value in id_mapping
        ]
    metrics["instance_split"]["pre_soma_gate_instance_count"] = pre_gate_roi_count
    metrics["instance_split"]["post_soma_gate_instance_count"] = int(labels.max())
    metrics["soma_identity_gate"] = soma_identity_gate
    metrics["per_cell"] = gated_per_cell
    metrics["roi_count"] = int(labels.max())
    metrics["whole_area_px"] = int((labels > 0).sum())
    metrics["soma_area_px"] = int((soma_labels > 0).sum())
    metrics["process_area_px"] = int((process_labels > 0).sum())
    metrics["soma_area_fraction"] = round(
        metrics["soma_area_px"] / metrics["whole_area_px"], 6
    )
    metrics["process_area_fraction"] = round(
        metrics["process_area_px"] / metrics["whole_area_px"], 6
    )
    metrics["fallback_soma_count"] = int(
        sum(bool(row["fallback_used"]) for row in gated_per_cell)
    )
    metrics["ambiguous_nucleus_count"] = int(
        sum(bool(row["nucleus_ambiguous"]) for row in gated_per_cell)
    )
    metrics["no_dapi_anchor_count"] = 0
    metrics["total_soma_anchor_count"] = len(gated_per_cell)
    metrics["multi_soma_whole_roi_count"] = 0
    metrics["rejected_soma_anchor_count"] = int(
        sum(int(row["rejected_soma_anchor_count"]) for row in gated_per_cell)
    )
    metrics["soma_core_shell_removed_px"] = int(
        sum(int(row["soma_core_shell_removed_px"]) for row in gated_per_cell)
    )
    metrics["soma_core_shell_applied_roi_count"] = int(
        sum(bool(row["soma_core_shell_applied"]) for row in gated_per_cell)
    )
    overlap_px = int(((soma_labels > 0) & (process_labels > 0)).sum())
    gap_px = int(
        ((labels > 0) & ~((soma_labels > 0) | (process_labels > 0))).sum()
    )
    outside_px = int(
        (((soma_labels > 0) | (process_labels > 0)) & ~(labels > 0)).sum()
    )
    if overlap_px or gap_px or outside_px:
        raise RuntimeError(
            "Post-gate compartment partition invariant failed: "
            f"overlap={overlap_px}, gap={gap_px}, outside={outside_px}"
        )
    metrics["partition_overlap_px"] = overlap_px
    metrics["partition_gap_px"] = gap_px
    metrics["partition_outside_whole_px"] = outside_px
    metrics["shared_whole_baseline"] = {
        "method": (
            "Frozen Whole geometry/filter with shared object-preserving 3D nucleus "
            "ownership refinement"
        ),
        "roi_count_before_neonatal_repartition": int(shared_whole_labels.max()),
        "whole_area_px": int(frozen_whole_mask.sum()),
        "branch_gap_restoration": shared_metrics["branch_gap_restoration"],
        "instance_split": shared_metrics["instance_split"],
        "nucleus_ownership_guard": shared_metrics["nucleus_ownership_guard"],
        "morphology_filter": shared_metrics["morphology_filter"],
    }
    metrics["neonatal_3d_validation"] = neonatal_3d_metrics
    shared_rows = {
        int(row["astrocyte_id"]): row for row in shared_metrics["per_cell"]
    }
    for row in metrics["per_cell"]:
        astrocyte_id = int(row["astrocyte_id"])
        final_mask = labels == astrocyte_id
        source_ids = sorted(
            int(value)
            for value in np.unique(shared_whole_labels[final_mask])
            if int(value) > 0
        )
        row["shared_whole_ids"] = source_ids
        row["process_component_count"] = int(
            measure.label(process_labels == astrocyte_id, connectivity=2).max()
        )
        if len(source_ids) == 1 and source_ids[0] in shared_rows:
            row["shared_morphology_qc"] = shared_rows[source_ids[0]].get(
                "morphology_qc", {}
            )
    validation_method = (
        "object-preserving calibrated 3D DAPI/"
        f"{neonatal_3d_context.structural_channel} nucleus ownership and anchor gate + "
        if neonatal_3d_context is not None
        else "2D DAPI anchors because calibrated 3D ownership was unavailable + "
    )
    metrics["method"] = (
        "shared frozen Whole geometry/filter with 3D nucleus ownership refinement + "
        + validation_method
        + "neonatal multi-center ID partition + neonatal local thickness/core-shell Soma + "
        "whole-ID valid-Soma gate and synchronized relabeling; "
        "Processes=Whole-Soma"
    )
    print(
        "Base compartment partition complete | profile=neonatal | "
        f"elapsed={time.perf_counter() - base_partition_started:.3f} s; "
        "finalizing identity and Soma safeguards...",
        flush=True,
    )
    finalization_started = time.perf_counter()
    labels, soma_labels, process_labels, metrics = (
        finalize_compartment_geometry_and_metrics(
            labels,
            soma_labels,
            process_labels,
            metrics,
            ownership_inventory,
            neonatal_3d_context,
            struct,
            "neonatal",
            pixel_width_um,
            pixel_height_um,
        )
    )
    print(
        "Compartment finalization complete | profile=neonatal | "
        f"elapsed={time.perf_counter() - finalization_started:.3f} s",
        flush=True,
    )
    return labels, soma_labels, process_labels, metrics
