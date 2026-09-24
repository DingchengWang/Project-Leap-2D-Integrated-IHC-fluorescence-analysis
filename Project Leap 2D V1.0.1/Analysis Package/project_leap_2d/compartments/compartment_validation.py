# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def relabel_compartment_triplet(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    retained_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Relabel retained Whole/Soma/Processes IDs atomically and contiguously."""

    maximum_id = max(
        int(whole_labels.max()),
        int(soma_labels.max()),
        int(process_labels.max()),
    )
    lookup = np.zeros(maximum_id + 1, dtype=np.uint16)
    mapping: dict[int, int] = {
        int(old_id): int(new_id)
        for new_id, old_id in enumerate(retained_ids, start=1)
    }
    for old_id, new_id in mapping.items():
        lookup[old_id] = new_id
    out_whole = lookup[whole_labels]
    out_soma = lookup[soma_labels]
    out_process = lookup[process_labels]
    return out_whole, out_soma, out_process, mapping

def circular_mask(
    shape: tuple[int, int],
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2

def skeleton_topology(mask: np.ndarray) -> tuple[int, int, int]:
    skeleton = morphology.skeletonize(mask.astype(bool))
    if not skeleton.any():
        return 0, 0, 0
    neighbors = ndi.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    endpoints = int((skeleton & (neighbors == 1)).sum())
    branch_clusters = int(
        measure.label(skeleton & (neighbors >= 3), connectivity=2).max()
    )
    return int(skeleton.sum()), endpoints, branch_clusters

def robust_reference(values: list[float], index: int) -> tuple[float, float]:
    reference = np.asarray(
        [value for position, value in enumerate(values) if position != index and np.isfinite(value)],
        dtype=np.float64,
    )
    if reference.size == 0:
        return 0.0, 1.0
    median = float(np.median(reference))
    mad_scale = 1.4826 * float(np.median(np.abs(reference - median)))
    q25, q75 = np.percentile(reference, [25.0, 75.0])
    iqr_scale = float(q75 - q25) / 1.349
    scale = max(mad_scale, iqr_scale, 1e-6)
    return median, scale

def filter_morphology_outlier_instances(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    struct: np.ndarray,
    mean_pixel_um: float,
    pixel_area_um2: float,
    per_cell: list[dict],
    instance_metrics: dict,
    config: CompartmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """Remove only whole-ID morphology outliers supported by multiple independent cues."""

    original_count = int(whole_labels.max())
    empty_metrics = {
        "enabled": bool(config.morphology_outlier_filter_enabled),
        "reference_count": original_count,
        "pre_filter_roi_count": original_count,
        "post_filter_roi_count": original_count,
        "removed_count": 0,
        "removed_area_px": 0,
        "removed_area_fraction": 0.0,
        "removed_original_ids": [],
        "flagged_original_ids": [],
        "id_mapping": {str(index): index for index in range(1, original_count + 1)},
        "details": [],
    }
    if not config.morphology_outlier_filter_enabled or original_count == 0:
        return whole_labels, soma_labels, process_labels, per_cell, empty_metrics

    per_cell_by_id = {int(row["astrocyte_id"]): row for row in per_cell}
    split_child_ids = {
        int(astrocyte_id)
        for detail in instance_metrics.get("split_components", [])
        for astrocyte_id in detail.get("new_astrocyte_ids", [])
    }
    properties = {int(prop.label): prop for prop in measure.regionprops(whole_labels)}
    feature_rows: list[dict] = []
    for astrocyte_id in range(1, original_count + 1):
        prop = properties[astrocyte_id]
        min_row, min_col, max_row, max_col = prop.bbox
        crop = np.s_[min_row:max_row, min_col:max_col]
        component = whole_labels[crop] == astrocyte_id
        local_process = process_labels[crop] == astrocyte_id
        distance_um = ndi.distance_transform_edt(
            component,
            sampling=(mean_pixel_um, mean_pixel_um),
        )
        skeleton_px, endpoint_count, branchpoint_count = skeleton_topology(component)
        process_component_count = int(measure.label(local_process, connectivity=2).max())
        axis_ratio = float(prop.major_axis_length) / max(float(prop.minor_axis_length), 1e-6)
        edge_touch = bool(
            min_row == 0
            or min_col == 0
            or max_row == whole_labels.shape[0]
            or max_col == whole_labels.shape[1]
        )
        local_struct = struct[crop]
        ring = morphology.binary_dilation(component, footprint=morphology.disk(5)) & ~component
        structural_contrast = float(np.median(local_struct[component])) - (
            float(np.median(local_struct[ring])) if ring.any() else 0.0
        )
        cell = per_cell_by_id[astrocyte_id]
        feature_rows.append(
            {
                "original_astrocyte_id": astrocyte_id,
                "area_px": int(prop.area),
                "area_um2": float(prop.area) * pixel_area_um2,
                "core_radius_um": float(np.percentile(distance_um[component], 95.0)),
                "axis_ratio": axis_ratio,
                "eccentricity": float(prop.eccentricity),
                "solidity": float(prop.solidity),
                "skeleton_length_um": skeleton_px * mean_pixel_um,
                "endpoint_count": endpoint_count,
                "branchpoint_count": branchpoint_count,
                "process_component_count": process_component_count,
                "structural_contrast": structural_contrast,
                "soma_anchor_count": int(cell["soma_anchor_count"]),
                "nucleus_score": float(cell["nucleus_score"]),
                "process_fraction": float(cell["process_fraction"]),
                "edge_touch": edge_touch,
                "accepted_split_child": astrocyte_id in split_child_ids,
            }
        )

    # Border-incomplete cells were already removed upstream; preserved border cells
    # remain valid peers, but are still protected from morphology-only deletion.
    reference_indices = list(range(len(feature_rows)))
    reference_count = len(reference_indices)
    metric_values = {
        "log_area": [math.log(max(row["area_um2"], 1e-6)) for row in feature_rows],
        "log_core": [math.log(max(row["core_radius_um"], 1e-6)) for row in feature_rows],
        "log_skeleton": [
            math.log(max(row["skeleton_length_um"], 1e-6)) for row in feature_rows
        ],
        "branchpoints": [float(row["branchpoint_count"]) for row in feature_rows],
        "endpoints": [float(row["endpoint_count"]) for row in feature_rows],
        "solidity": [float(row["solidity"]) for row in feature_rows],
        "axis_ratio": [float(row["axis_ratio"]) for row in feature_rows],
        "structural_contrast": [
            float(row["structural_contrast"]) for row in feature_rows
        ],
    }
    reference_medians = {
        key: float(np.median([values[index] for index in reference_indices]))
        if reference_indices
        else 0.0
        for key, values in metric_values.items()
    }
    process_fraction_median = float(
        np.median([feature_rows[index]["process_fraction"] for index in reference_indices])
    ) if reference_indices else 0.0
    branchpoint_median = reference_medians["branchpoints"]
    ramified_field = bool(process_fraction_median >= 0.60 and branchpoint_median >= 3.0)

    proposed: list[tuple[int, float, str]] = []
    flagged_ids: list[int] = []
    for index, row in enumerate(feature_rows):
        z_scores: dict[str, float] = {}
        for key, values in metric_values.items():
            allowed_reference = [values[position] for position in reference_indices]
            if index in reference_indices:
                local_index = reference_indices.index(index)
                median, scale = robust_reference(allowed_reference, local_index)
            else:
                median = float(np.median(allowed_reference)) if allowed_reference else 0.0
                mad = (
                    1.4826
                    * float(np.median(np.abs(np.asarray(allowed_reference) - median)))
                    if allowed_reference
                    else 1.0
                )
                scale = max(mad, 1e-6)
            z_scores[key] = (values[index] - median) / scale

        median_area_um2 = math.exp(reference_medians["log_area"])
        median_core_um = math.exp(reference_medians["log_core"])
        median_skeleton_um = math.exp(reference_medians["log_skeleton"])
        no_anchor = row["soma_anchor_count"] == 0
        votes = {
            "small_area": bool(
                z_scores["log_area"] <= -config.morphology_outlier_robust_z
                and row["area_um2"] <= 0.45 * median_area_um2
            ),
            "thin_core": bool(
                z_scores["log_core"] <= -config.morphology_outlier_robust_z
                and row["core_radius_um"] <= 0.65 * median_core_um
            ),
            "short_skeleton": bool(
                z_scores["log_skeleton"] <= -config.morphology_outlier_robust_z
                and row["skeleton_length_um"] <= 0.45 * median_skeleton_um
            ),
            "few_branches": bool(
                z_scores["branchpoints"] <= -config.morphology_outlier_robust_z
                and row["branchpoint_count"] <= max(1, int(0.25 * branchpoint_median))
            ),
            "few_endpoints": bool(
                z_scores["endpoints"] <= -config.morphology_outlier_robust_z
                and row["endpoint_count"] <= 2
            ),
            "high_solidity": bool(
                z_scores["solidity"] >= config.morphology_outlier_robust_z
                and row["solidity"] >= 0.82
            ),
            "high_axis_ratio": bool(
                z_scores["axis_ratio"] >= config.morphology_outlier_robust_z
                and row["axis_ratio"] >= config.morphology_fragment_min_axis_ratio
            ),
            "low_structural_contrast": bool(
                z_scores["structural_contrast"] <= -config.morphology_outlier_robust_z
                and row["structural_contrast"] <= 0.0
            ),
        }
        consensus = int(sum(votes.values()))
        absolute_fragment = bool(
            no_anchor
            and row["axis_ratio"] >= config.morphology_fragment_min_axis_ratio
            and row["branchpoint_count"] <= config.morphology_fragment_max_branchpoints
            and row["core_radius_um"] <= config.morphology_fragment_max_core_radius_um
        )
        relative_elongated_fragment = bool(
            reference_count >= config.morphology_outlier_min_reference_count
            and row["axis_ratio"] >= config.morphology_fragment_min_axis_ratio
            and row["area_um2"] <= 0.35 * median_area_um2
            and row["skeleton_length_um"] <= 0.35 * median_skeleton_um
            and row["core_radius_um"] <= 1.10 * median_core_um
        )
        population_outlier = bool(
            reference_count >= config.morphology_outlier_min_reference_count
            and no_anchor
            and consensus >= config.morphology_outlier_min_consensus
        )
        compact_cues = {
            "relative_small_area": bool(row["area_um2"] <= 0.45 * median_area_um2),
            "relative_short_skeleton": bool(
                row["skeleton_length_um"] <= 0.45 * median_skeleton_um
            ),
            "relative_few_branches": bool(
                row["branchpoint_count"] <= max(1, int(0.45 * branchpoint_median))
            ),
            "relative_low_process_fraction": bool(
                row["process_fraction"] <= min(0.55, 0.75 * process_fraction_median)
            ),
            "relative_high_solidity": bool(
                row["solidity"] >= max(0.50, 1.25 * reference_medians["solidity"])
            ),
        }
        compact_consensus = int(sum(compact_cues.values()))
        compact_outlier = bool(
            reference_count >= config.morphology_outlier_min_reference_count
            and ramified_field
            and compact_cues["relative_high_solidity"]
            and compact_consensus >= config.morphology_outlier_min_consensus + 1
        )
        protected = bool(row["edge_touch"] or row["accepted_split_child"])
        reason = ""
        if relative_elongated_fragment:
            reason = "small_elongated_morphology_outlier"
        elif absolute_fragment:
            reason = "unanchored_thin_fragment"
        elif population_outlier:
            reason = "unanchored_multimetric_outlier"
        elif compact_outlier:
            reason = "compact_multimetric_outlier"
        if consensus > 0 or compact_consensus > 0 or absolute_fragment:
            flagged_ids.append(int(row["original_astrocyte_id"]))
        severity = float(consensus + compact_consensus) + max(
            abs(min(z_scores.values(), default=0.0)),
            abs(max(z_scores.values(), default=0.0)),
        ) / 10.0
        if reason and not protected:
            proposed.append((int(row["original_astrocyte_id"]), severity, reason))
        row.update(
            {
                "robust_z": {key: round(value, 4) for key, value in z_scores.items()},
                "outlier_votes": [key for key, value in votes.items() if value],
                "outlier_consensus": consensus,
                "compact_outlier_cues": [
                    key for key, value in compact_cues.items() if value
                ],
                "compact_outlier_consensus": compact_consensus,
                "outlier_reason": reason,
                "outlier_protected": protected,
            }
        )

    max_remove_count = max(1, int(math.floor(0.25 * original_count)))
    proposed.sort(key=lambda item: item[1], reverse=True)
    removed_ids = {item[0] for item in proposed[:max_remove_count]}
    removed_area_px = int(np.isin(whole_labels, list(removed_ids)).sum())
    if (
        original_count - len(removed_ids) < 5
        or removed_area_px > 0.25 * int((whole_labels > 0).sum())
    ):
        removed_ids.clear()
        removed_area_px = 0

    retained_ids = [
        astrocyte_id
        for astrocyte_id in range(1, original_count + 1)
        if astrocyte_id not in removed_ids
    ]
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(retained_ids, start=1)}
    filtered_whole = np.zeros_like(whole_labels, dtype=np.uint16)
    filtered_soma = np.zeros_like(soma_labels, dtype=np.uint16)
    filtered_process = np.zeros_like(process_labels, dtype=np.uint16)
    filtered_per_cell: list[dict] = []
    for old_id, new_id in id_mapping.items():
        filtered_whole[whole_labels == old_id] = new_id
        filtered_soma[soma_labels == old_id] = new_id
        filtered_process[process_labels == old_id] = new_id
        updated = dict(per_cell_by_id[old_id])
        updated["original_astrocyte_id"] = old_id
        updated["astrocyte_id"] = new_id
        feature = next(
            row for row in feature_rows if row["original_astrocyte_id"] == old_id
        )
        updated["morphology_qc"] = feature
        filtered_per_cell.append(updated)

    removed_details = [
        {
            **next(row for row in feature_rows if row["original_astrocyte_id"] == old_id),
            "reason": next(item[2] for item in proposed if item[0] == old_id),
        }
        for old_id in sorted(removed_ids)
    ]
    metrics = {
        "enabled": True,
        "reference_count": reference_count,
        "pre_filter_roi_count": original_count,
        "post_filter_roi_count": len(retained_ids),
        "removed_count": len(removed_ids),
        "removed_area_px": removed_area_px,
        "removed_area_fraction": round(
            removed_area_px / max(int((whole_labels > 0).sum()), 1),
            6,
        ),
        "removed_original_ids": sorted(removed_ids),
        "flagged_original_ids": sorted(set(flagged_ids)),
        "id_mapping": {str(old): new for old, new in id_mapping.items()},
        "ramified_field": ramified_field,
        "reference_medians": {
            "area_um2": round(math.exp(reference_medians["log_area"]), 6),
            "core_radius_um": round(math.exp(reference_medians["log_core"]), 6),
            "skeleton_length_um": round(
                math.exp(reference_medians["log_skeleton"]),
                6,
            ),
            "branchpoints": round(branchpoint_median, 6),
            "process_fraction": round(process_fraction_median, 6),
        },
        "details": removed_details,
        "all_features": feature_rows,
    }
    return (
        filtered_whole,
        filtered_soma,
        filtered_process,
        filtered_per_cell,
        metrics,
    )

def filter_instances_by_valid_soma(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    per_cell: list[dict],
    profile: str,
    unresolved_multi_soma_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """Keep only cell IDs with exactly one connected, trusted Soma."""

    if profile not in {"mature", "neonatal"}:
        raise ValueError(f"Unknown Soma-gate profile: {profile}")
    if whole_labels.shape != soma_labels.shape or whole_labels.shape != process_labels.shape:
        raise ValueError("Valid-Soma gate received mismatched label geometry")
    original_ids = sorted(int(value) for value in np.unique(whole_labels) if int(value) > 0)
    expected_ids = list(range(1, int(whole_labels.max()) + 1))
    if original_ids != expected_ids:
        raise ValueError(
            "Valid-Soma gate requires contiguous pre-gate Whole IDs: "
            f"observed={original_ids}"
        )
    per_cell_by_id = {int(row["astrocyte_id"]): row for row in per_cell}
    if set(per_cell_by_id) != set(original_ids):
        raise ValueError(
            "Valid-Soma gate per-cell rows do not match Whole IDs: "
            f"whole={original_ids}, rows={sorted(per_cell_by_id)}"
        )

    retained_ids: list[int] = []
    removed_details: list[dict] = []
    unresolved_ids = set(unresolved_multi_soma_ids or set())
    if not unresolved_ids.issubset(set(original_ids)):
        raise ValueError(
            "Disqualified split IDs are absent from Whole labels: "
            f"{sorted(unresolved_ids - set(original_ids))}"
        )
    for astrocyte_id in original_ids:
        row = per_cell_by_id[astrocyte_id]
        anchor_count = int(row.get("soma_anchor_count", 0))
        whole_area_px = int((whole_labels == astrocyte_id).sum())
        soma_area_px = int((soma_labels == astrocyte_id).sum())
        process_area_px = int((process_labels == astrocyte_id).sum())
        soma_component_count = int(
            measure.label(soma_labels == astrocyte_id, connectivity=2).max()
        )
        reason = ""
        if astrocyte_id in unresolved_ids:
            reason = "unresolved_multi_soma_instance_split"
        elif anchor_count == 0 or soma_area_px == 0:
            reason = "no_valid_soma"
        elif anchor_count > 1:
            reason = "multiple_valid_somata_after_instance_split"
        elif soma_component_count != 1:
            reason = "disconnected_soma_geometry"
        elif process_area_px == 0:
            reason = "empty_processes_compartment"
        if reason:
            removed_details.append(
                {
                    "pre_gate_astrocyte_id": astrocyte_id,
                    "reason": reason,
                    "soma_anchor_count": anchor_count,
                    "soma_component_count": soma_component_count,
                    "whole_area_px": whole_area_px,
                    "soma_area_px": soma_area_px,
                    "process_area_px": process_area_px,
                }
            )
        else:
            retained_ids.append(astrocyte_id)

    if not retained_ids:
        raise RuntimeError(
            f"No {profile} Whole Astrocyte instance retained exactly one valid Soma; "
            "the run was stopped before measurement"
        )

    id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(retained_ids, start=1)
    }
    filtered_whole = np.zeros_like(whole_labels, dtype=np.uint16)
    filtered_soma = np.zeros_like(soma_labels, dtype=np.uint16)
    filtered_process = np.zeros_like(process_labels, dtype=np.uint16)
    filtered_per_cell: list[dict] = []
    for old_id, new_id in id_mapping.items():
        filtered_whole[whole_labels == old_id] = new_id
        filtered_soma[soma_labels == old_id] = new_id
        filtered_process[process_labels == old_id] = new_id
        updated = dict(per_cell_by_id[old_id])
        updated.setdefault("original_astrocyte_id", old_id)
        updated["pre_soma_gate_astrocyte_id"] = old_id
        updated["astrocyte_id"] = new_id
        filtered_per_cell.append(updated)

    final_ids = set(range(1, len(retained_ids) + 1))
    observed = {
        key: set(int(value) for value in np.unique(labels) if int(value) > 0)
        for key, labels in (
            ("whole", filtered_whole),
            ("soma", filtered_soma),
            ("processes", filtered_process),
        )
    }
    if any(ids != final_ids for ids in observed.values()):
        raise RuntimeError(
            "Valid-Soma gate failed to preserve synchronized compartment IDs: "
            f"{observed}"
        )
    if any(int(row["soma_anchor_count"]) != 1 for row in filtered_per_cell):
        raise RuntimeError("A retained cell does not have exactly one Soma anchor")

    removed_ids = [row["pre_gate_astrocyte_id"] for row in removed_details]
    removed_area_px = int(np.isin(whole_labels, removed_ids).sum())
    metrics = {
        "enabled": True,
        "profile": profile,
        "criterion": "exactly_one_valid_soma_per_final_astrocyte",
        "pre_gate_roi_count": len(original_ids),
        "post_gate_roi_count": len(retained_ids),
        "removed_count": len(removed_ids),
        "removed_area_px": removed_area_px,
        "removed_area_fraction": round(
            removed_area_px / max(int((whole_labels > 0).sum()), 1),
            6,
        ),
        "removed_pre_gate_ids": removed_ids,
        "unresolved_multi_soma_pre_gate_ids": sorted(unresolved_ids),
        "retained_pre_gate_ids": retained_ids,
        "id_mapping": {str(old_id): new_id for old_id, new_id in id_mapping.items()},
        "details": removed_details,
    }
    return (
        filtered_whole,
        filtered_soma,
        filtered_process,
        filtered_per_cell,
        metrics,
    )
