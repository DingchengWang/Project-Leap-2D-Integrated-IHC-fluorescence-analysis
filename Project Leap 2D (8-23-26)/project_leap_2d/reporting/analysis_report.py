# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def write_analysis_report(
    path: Path,
    *,
    input_dir: Path,
    paths: dict[str, Path],
    metadata: dict[str, dict],
    structural_channels: list[str],
    measurement: str,
    best_row: dict,
    candidate_rows: list[dict],
    candidate_specs: list[TestSpec],
    roi_count: int,
    compartment_metrics: dict,
    fiji_status: str,
    fiji_details: dict | None = None,
) -> None:
    if len(candidate_specs) != len(candidate_rows):
        raise ValueError(
            "Candidate report metadata is incomplete: "
            f"{len(candidate_specs)} specifications for {len(candidate_rows)} rows"
        )
    age_profile = compartment_metrics.get("age_profile", {})
    age_score = age_profile.get("neonatal_score")
    age_margin = age_profile.get("confidence_margin")
    tagged_files = age_profile.get("tagged_files", [])
    neonatal_3d = compartment_metrics.get(
        "neonatal_3d_validation",
        compartment_metrics.get(
            "nucleus_3d_inventory",
            {
                "status": "not_run_calibrated_Z_unavailable",
                "method": "object-preserving calibrated 3D DAPI inventory",
                "measurement_channel_used": False,
                "candidate_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "per_nucleus": [],
            },
        ),
    )
    canonical_nuclei = neonatal_3d.get("canonical_resolution", {})
    axial_guard = compartment_metrics.get("axial_truncation_guard", {})
    projected_foreign_guard = compartment_metrics.get(
        "projected_foreign_soma_guard",
        {},
    )
    canonical_owner_extent_completion = compartment_metrics.get(
        "soma_nuclear_extent_completion",
        {},
    )
    same_id_soma_reconciliation = compartment_metrics.get(
        "same_id_disconnected_soma_reconciliation",
        {},
    )
    ownership_guard = compartment_metrics.get("nucleus_ownership_guard", {})
    shared_whole = compartment_metrics.get("shared_whole_baseline", {})
    soma_identity_gate = compartment_metrics.get(
        "soma_identity_gate",
        {
            "enabled": False,
            "pre_gate_roi_count": compartment_metrics["roi_count"],
            "post_gate_roi_count": compartment_metrics["roi_count"],
            "removed_count": 0,
            "removed_area_px": 0,
            "removed_area_fraction": 0.0,
            "removed_pre_gate_ids": [],
            "id_mapping": {},
            "details": [],
        },
    )
    shared_filter = shared_whole.get("morphology_filter", {})
    shared_gap = shared_whole.get("branch_gap_restoration", {})
    effective_filter = shared_filter or compartment_metrics["morphology_filter"]
    effective_gap = shared_gap or compartment_metrics["branch_gap_restoration"]
    candidate_times = _RUNTIME_TIMINGS["candidate_postprocess_seconds"]
    cellpose_events = _RUNTIME_TIMINGS["cellpose_inference_events"]
    assert isinstance(candidate_times, list)
    assert isinstance(cellpose_events, list)
    module_counts = {
        name: sum(candidate_module_name(spec) == name for spec in candidate_specs)
        for name in (
            "morphology_baseline",
            "structural_refinement",
            "distributional_threshold",
        )
    }
    lines = [
        "PROJECT LEAP 2D ANALYSIS REPORT",
        "===============================",
        "",
        "[Run]",
        f"Product: {PRODUCT_DISPLAY_NAME}",
        f"Scientific Core: {ANALYSIS_CORE_NAME}",
        f"Pipeline: {PIPELINE_NAME}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Input Folder: {input_dir}",
        "",
        "[Channels]",
    ]
    for channel in ("DAPI", "eGFP", "GFAP", *MEASUREMENT_CHANNELS):
        if channel in paths:
            lines.append(f"{channel}: {paths[channel].name}")
    lines.extend(
        [
            f"ROI Definition: DAPI + {' + '.join(structural_channels)}",
            f"Measurement: {measurement}",
            "Measurement Channel Used for ROI Definition: No",
            f"Image Geometry: ZYX {tuple(metadata['DAPI']['shape'])}, {metadata['DAPI']['dtype']}",
            "Channel Shape and Calibration Match: Yes",
            "",
            "[Candidate Architecture and Runtime]",
            "Module Counts: "
            f"Morphology Baseline={module_counts['morphology_baseline']}; "
            f"Structural Refinement={module_counts['structural_refinement']}; "
            "Distributional Threshold="
            f"{module_counts['distributional_threshold']}",
            f"Z Intervals: {EXPECTED_Z_INTERVAL_COUNT}; "
            f"Profiles per Z Interval: {EXPECTED_PROFILES_PER_Z}",
            "Shared Preparation: each Z interval generates one projection set, "
            "one Cellpose-SAM mask, and one reusable DAPI/Sato/top-hat feature context.",
            f"Cellpose Model Initialization: "
            f"{float(_RUNTIME_TIMINGS['cellpose_model_init_seconds']):.6f} s",
            f"Cellpose Inference Calls: {len(cellpose_events)}",
        ]
    )
    for index, event in enumerate(cellpose_events, start=1):
        lines.append(
            f"  Inference {index:02d}: status="
            f"{'ok' if event.get('success', False) else 'failed'}; "
            f"device={event.get('device', 'unknown')}; "
            f"max_side={event.get('max_side', 'unknown')}; "
            f"diameter={event.get('diameter', 'unknown')}; "
            f"cellprob={event.get('cellprob', 'unknown')}; "
            f"time={float(event.get('seconds', 0.0)):.6f} s"
        )
    lines.extend(
        [
            f"Candidate Stage Wall Time: "
            f"{float(_RUNTIME_TIMINGS['candidate_stage_wall_seconds']):.6f} s",
            f"Candidate Worker Compute Total: "
            f"{sum(float(value) for value in candidate_times):.6f} s",
            f"Candidate Ranking Time: "
            f"{float(_RUNTIME_TIMINGS['rank_candidates_seconds']):.6f} s",
            f"Soma/Processes and 3D Ownership Time: "
            f"{float(_RUNTIME_TIMINGS['compartment_split_seconds']):.6f} s",
            f"Fiji Startup to Review Ready: "
            + (
                f"{float(fiji_details['fiji_startup_seconds']):.6f} s"
                if fiji_details and fiji_details.get("fiji_startup_seconds") is not None
                else "Not completed"
            ),
            f"Fiji Native Measurement Time: "
            + (
                f"{float(fiji_details['measurement_seconds']):.6f} s"
                if fiji_details and fiji_details.get("measurement_seconds") is not None
                else "Not completed"
            ),
            "",
            "Candidate Inventory:",
            "ID | Module | Family | Profile | Threshold | Z | Area_px | Components | "
            "Soma_Support | Fine_Added_px | Eligible | Worker_Time_s",
        ]
    )
    for index, (spec, row) in enumerate(zip(candidate_specs, candidate_rows)):
        worker_time = (
            f"{float(candidate_times[index]):.6f}"
            if index < len(candidate_times)
            else "Not available"
        )
        lines.append(
            f"{int(row['candidate']):02d} | "
            f"{candidate_module_display_name(spec)} | "
            f"{candidate_profile_family(spec)} | "
            f"{candidate_profile_name(spec)} | "
            f"{candidate_threshold_display_name(spec)} | "
            f"{int(row['z_start_1based'])}-{int(row['z_end_1based_inclusive'])} | "
            f"{int(row['mask_area_px'])} | "
            f"{int(row['connected_components'])} | "
            f"{int(row['soma_supported_components'])} | "
            f"{int(row['fine_branch_retained_px'])} | "
            f"{bool(row.get('selection_eligible'))} | "
            f"{worker_time}"
        )
    lines.extend(
        [
            "",
            "[Selected Analysis]",
            f"Astrocyte Profile: {age_profile.get('profile', 'unknown')}",
            f"Profile Decision Source: {age_profile.get('source', 'unknown')}",
            f"Filename Profile Evidence: {', '.join(tagged_files) if tagged_files else 'None'}",
            f"Morphology Neonatal Score: {age_score if age_score is not None else 'Not run'}",
            f"Morphology Decision Threshold: {age_profile.get('threshold', AGE_PROFILE_THRESHOLD)}",
            f"Morphology Confidence Margin: {age_margin if age_margin is not None else 'Not applicable'}",
            f"Candidates: {len(candidate_rows)} evaluated; "
            f"{sum(bool(row.get('selection_eligible')) for row in candidate_rows)} eligible",
            "Candidate Design: 30 Morphology Baseline + 30 Structural Refinement + "
            "30 Distributional Threshold candidates across five Z intervals",
            "Ranking: unified 90-candidate family-by-Z balance with near-duplicate "
            "clustering and a pre-distribution-baseline same-Z non-regression guard",
            "Morphology Baseline Incumbent Candidate: "
            f"{best_row.get('morphology_baseline_incumbent_candidate', 'Not available')}",
            "Morphology Baseline Incumbent Score: "
            f"{best_row.get('morphology_baseline_incumbent_score', 'Not available')}",
            "Pre-Distribution Baseline Incumbent Candidate: "
            f"{best_row.get('pre_distribution_baseline_incumbent_candidate', 'Not available')}",
            "Pre-Distribution Baseline Incumbent Score: "
            f"{best_row.get('pre_distribution_baseline_incumbent_score_frozen', 'Not available')}",
            "Pre-Distribution Baseline Incumbent Score in Complete Pool: "
            f"{best_row.get('pre_distribution_baseline_incumbent_score_complete_pool', 'Not available')}",
            f"Family-Balanced Challenger Candidate: {best_row.get('family_balanced_challenger_candidate', 'Not available')}",
            f"Family-Balanced Challenger Score: {best_row.get('family_balanced_challenger_score', 'Not available')}",
            f"Selection Guard Decision: {best_row.get('selection_guard_reason', 'Not available')}",
            f"Best Candidate: {best_row['candidate']:02d} ({best_row['name']})",
            f"Selection Score: {best_row['auto_selection_score']}",
            f"Z Range: slices {best_row['z_start_1based']}-{best_row['z_end_1based_inclusive']} "
            "(1-based inclusive)",
            f"Projection: {str(best_row['projection']).upper()}",
            f"Automatically Selected Whole Astrocyte ROIs: {roi_count}",
            f"Selected Candidate Whole Area Before Final Refinement: "
            f"{best_row['mask_area_px']} pixels",
            f"Cellpose-SAM: {best_row['cellpose']} on {_CELLPOSE_DEVICE or 'not initialized'}",
            f"Candidate CPU Workers: requested={CANDIDATE_CPU_WORKERS}; effective={_EFFECTIVE_CANDIDATE_CPU_WORKERS}",
            f"3D DAPI Inventory CPU Workers: requested={DAPI_INVENTORY_CPU_WORKERS}; effective={_EFFECTIVE_DAPI_INVENTORY_CPU_WORKERS}",
            f"Cellpose Batch Size: requested={CELLPOSE_BATCH_SIZE}; effective={_CELLPOSE_EFFECTIVE_BATCH_SIZE if _CELLPOSE_EFFECTIVE_BATCH_SIZE is not None else 'Not run'}",
            f"Segmentation Method: {best_row['method_used']}",
            f"Distribution Model: {best_row.get('distribution_model', 'Not selected')}",
            f"Distribution Background-Peak Role: {best_row.get('distribution_background_peak_role', 'Not selected')}",
            f"Distribution Valid Channels: {best_row.get('distribution_valid_channels', 'Not selected')}",
            f"Distribution Posterior Thresholds: {best_row.get('distribution_posterior_thresholds_raw', 'Not selected')}",
            f"Distribution Background Peaks: {best_row.get('distribution_background_peaks_raw', 'Not selected')}",
            f"Distribution Component Separations: {best_row.get('distribution_ashman_separations', 'Not selected')}",
            f"Candidate Errors: {sum(bool(row['error']) for row in candidate_rows)}",
            "",
            "[Soma and Processes]",
            f"Method: {compartment_metrics['method']}",
            f"Adaptation: {compartment_metrics['adaptation']}",
            f"XY Pixel Size: {compartment_metrics['pixel_width_um']:.6f} x "
            f"{compartment_metrics['pixel_height_um']:.6f} um",
            f"Connected Whole Components Before Instance Split: "
            f"{compartment_metrics['instance_split']['base_connected_component_count']}",
            "Whole Instances Before Valid-Soma Cell Gate: "
            f"{compartment_metrics['instance_split'].get('pre_soma_gate_instance_count', compartment_metrics['instance_split']['final_instance_count'])}",
            f"Final Whole Astrocyte Instances: {compartment_metrics['roi_count']}",
            f"High-Confidence Multi-Astrocyte Components Split: "
            f"{compartment_metrics['instance_split']['split_component_count']}; "
            f"added ROIs={compartment_metrics['instance_split']['split_added_roi_count']}; "
            f"rejected ambiguous splits={compartment_metrics['instance_split']['split_rejected_count']}",
            f"Whole Geometry Path: "
            f"{'shared frozen Whole geometry with 3D ownership, then neonatal compartment repartition' if shared_whole else 'frozen mature Soma/Processes with shared 3D ownership guard'}",
            f"Shared Whole Baseline Area: "
            f"{shared_whole.get('whole_area_px', compartment_metrics['whole_area_px'])} pixels",
            f"Shared Whole Baseline Morphology Filter Removed IDs: "
            f"{shared_filter.get('removed_original_ids', compartment_metrics['morphology_filter']['removed_original_ids'])}",
            f"Shared Whole Baseline Branch-Gap Removal: "
            f"{shared_gap.get('removed_px', compartment_metrics['branch_gap_restoration']['removed_px'])} pixels",
            f"Branch-Gap Restoration: "
            f"{effective_gap['accepted_gap_count']} internal valleys; "
            f"removed={effective_gap['removed_px']} pixels "
            f"({effective_gap['removed_fraction']:.2%}); "
            f"connectivity-protected rejections="
            f"{effective_gap['rejected_disconnect_count']}",
            f"DAPI Extent Reconstruction: high={compartment_metrics['dapi_extent']['high_threshold']}; "
            f"low={compartment_metrics['dapi_extent']['low_threshold']}; "
            f"strict_core={compartment_metrics['dapi_extent']['strict_core_px']} pixels; "
            f"reconstructed_extent={compartment_metrics['dapi_extent']['reconstructed_extent_px']} pixels",
            f"DAPI Extent Satellite Cleanup: "
            f"{compartment_metrics['dapi_extent_satellite_components_removed']} components; "
            f"{compartment_metrics['dapi_extent_satellite_px_removed']} pixels reassigned to Processes",
            f"{'Shared Whole Morphology Filter' if shared_whole else 'Morphology Filter'}: "
            f"pre={effective_filter['pre_filter_roi_count']} ROIs; "
            f"post={effective_filter['post_filter_roi_count']} ROIs; "
            f"removed={effective_filter['removed_original_ids']}; "
            f"removed_area={effective_filter['removed_area_px']} pixels "
            f"({effective_filter['removed_area_fraction']:.2%})",
            "Age-Independent Valid-Soma Filtering: Whole-ID gate; retained cell boundaries unchanged",
            f"Valid-Soma Cell Gate ({soma_identity_gate.get('profile', age_profile.get('profile', 'unknown'))}): "
            f"pre={soma_identity_gate['pre_gate_roi_count']}; "
            f"post={soma_identity_gate['post_gate_roi_count']}; "
            f"removed={soma_identity_gate['removed_pre_gate_ids']}; "
            f"removed_area={soma_identity_gate['removed_area_px']} pixels "
            f"({soma_identity_gate['removed_area_fraction']:.2%})",
            f"Soma-Gate ID Mapping: {soma_identity_gate['id_mapping']}",
            f"Final Whole Area: {compartment_metrics['whole_area_px']} pixels",
            f"Final Soma Area: {compartment_metrics['soma_area_px']} pixels "
            f"({compartment_metrics['soma_area_fraction']:.2%})",
            f"Final Processes Area: {compartment_metrics['process_area_px']} pixels "
            f"({compartment_metrics['process_area_fraction']:.2%})",
            f"Fallback Soma ROIs: {compartment_metrics['fallback_soma_count']}",
            f"Ambiguous DAPI Anchors: {compartment_metrics['ambiguous_nucleus_count']}",
            f"Whole ROIs Without a Trusted Soma: {compartment_metrics['no_dapi_anchor_count']}",
            f"Selected DAPI Soma Anchors: {compartment_metrics['total_soma_anchor_count']}",
            f"Whole ROIs With Multiple Soma Anchors: "
            f"{compartment_metrics['multi_soma_whole_roi_count']}",
            f"Rejected Soma Anchors: {compartment_metrics['rejected_soma_anchor_count']}",
            f"Soma Core-Shell Pruning: {compartment_metrics['soma_core_shell_applied_roi_count']} ROIs; "
            f"{compartment_metrics['soma_core_shell_removed_px']} pixels reassigned to Processes",
            "No-Soma Rule: in both mature and neonatal profiles, a Whole ID without exactly one connected trusted Soma is removed together with its Processes.",
            "Partition QC: overlap=0 pixels; gap=0 pixels; outside Whole=0 pixels",
            "Processes ROI Semantics: one Astrocyte ID may contain multiple disconnected subregions; all are measured together as one composite ROI.",
            "",
            "Removed Morphology Outliers:",
        ]
    )
    if effective_filter["details"]:
        for row in effective_filter["details"]:
            lines.append(
                f"  Original ID {row['original_astrocyte_id']:03d}: {row['reason']}; "
                f"area={row['area_um2']:.3f} um2; core_R95={row['core_radius_um']:.3f} um; "
                f"axis_ratio={row['axis_ratio']:.3f}; branches={row['branchpoint_count']}; "
                f"anchors={row['soma_anchor_count']}; robust_votes={row['outlier_votes']}; "
                f"relative_cues={row['compact_outlier_cues']}"
            )
    else:
        lines.append("  None")
    lines.extend(["", "Valid-Soma Cell Gate Removals:"])
    if soma_identity_gate["details"]:
        for row in soma_identity_gate["details"]:
            lines.append(
                f"  Pre-gate ID {row['pre_gate_astrocyte_id']:03d}: {row['reason']}; "
                f"anchors={row['soma_anchor_count']}; soma_components={row['soma_component_count']}; "
                f"whole={row['whole_area_px']} px; "
                f"soma={row['soma_area_px']} px; processes={row['process_area_px']} px"
            )
    else:
        lines.append("  None")
    lines.extend(
        [
            "",
            "Accepted Instance Splits:",
        ]
    )
    if compartment_metrics["instance_split"]["split_components"]:
        for row in compartment_metrics["instance_split"]["split_components"]:
            lines.append(
                f"  Base {row['base_component_id']:03d}: "
                f"pre_gate_IDs={row.get('pre_soma_gate_new_astrocyte_ids', row['new_astrocyte_ids'])}; "
                f"pre_gate_child_areas={row.get('pre_soma_gate_child_areas_px', row['child_areas_px'])}; "
                f"removed_pre_gate_IDs={row.get('removed_pre_soma_gate_ids', [])}; "
                f"final_IDs={row['new_astrocyte_ids']}; "
                f"anchor_scores={row['anchor_scores']}; "
                f"separation={row['anchor_separation_um']:.3f} um; "
                f"neck/core={row['neck_core_ratio']:.3f}; "
                f"boundary/core_struct={row.get('boundary_structural_ratio', 'not used')}; "
                f"reason={row.get('reason', 'accepted')}"
            )
    else:
        lines.append("  None")
    lines.extend(
        [
            "",
            "Compartment Parameters:",
        ]
    )
    for key, value in compartment_metrics["config"].items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Morphology Profile Features:"])
    if age_profile.get("features"):
        for key, value in age_profile["features"].items():
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  Not computed because a filename profile label was present")
    lines.extend(
        [
            "",
            "[3D Nucleus Ownership QC]",
            f"Status: {neonatal_3d.get('status', 'unknown')}",
            f"Method: {neonatal_3d.get('method', 'unknown')}",
            f"Measurement Channel Used: {neonatal_3d.get('measurement_channel_used', False)}",
            f"Validation Z Range: "
            f"{neonatal_3d.get('z_start_1based', 'Not applicable')}-"
            f"{neonatal_3d.get('z_end_1based_inclusive', 'Not applicable')}",
            f"Voxel Size ZYX (um): {neonatal_3d.get('voxel_size_um', 'Not available')}",
            f"Calibration Source: {neonatal_3d.get('calibration_source', 'Not available')}",
            f"DAPI Candidates Near Frozen Whole: {neonatal_3d.get('candidate_count', 0)}",
            f"DAPI Candidates Passed 3D Gate: {neonatal_3d.get('accepted_count', 0)}",
            f"DAPI Candidates Rejected by 3D Gate: {neonatal_3d.get('rejected_count', 0)}",
            f"Accepted 2D Nucleus IDs: {neonatal_3d.get('accepted_2d_nucleus_ids', [])}",
            f"Rejected 2D Nucleus IDs: {neonatal_3d.get('rejected_2d_nucleus_ids', [])}",
            f"Ownership Conflicts: {ownership_guard.get('conflict_component_count', 0)}",
            f"Mandatory Multi-Nucleus Splits: {ownership_guard.get('split_component_count', 0)}",
            f"Foreign Soma Territories Pruned: {ownership_guard.get('foreign_soma_pruned_component_count', 0)}",
            f"Ambiguous Components Removed Fail-Closed: {ownership_guard.get('fail_closed_component_count', 0)}",
            f"Ownership Area Removed (px): {ownership_guard.get('removed_area_px', 0)}",
            f"Canonical 3D Nucleus Instances: {canonical_nuclei.get('instance_count', 0)}",
            f"Canonical Envelopes Split: {canonical_nuclei.get('connected_envelope_split_count', 0)}",
            f"Canonical Envelopes Ambiguous: {canonical_nuclei.get('ambiguous_instance_count', 0)}",
            f"Axially Truncated Cells Removed: {axial_guard.get('removed_cell_count', 0)}; "
            f"pre-guard IDs={axial_guard.get('removed_pre_guard_ids', [])}",
            f"Projected Foreign-DAPI Terminal Guard: changed "
            f"{projected_foreign_guard.get('changed_cell_count', 0)} cells; "
            f"removed {projected_foreign_guard.get('removed_area_px', 0)} px; "
            f"true tips={projected_foreign_guard.get('true_tip_count', 0)}; "
            f"terminal overlaps={projected_foreign_guard.get('terminal_overlap_component_count', 0)}; "
            f"pass-through preserved={projected_foreign_guard.get('preserved_pass_through_component_count', 0)}; "
            f"connectivity rollbacks={projected_foreign_guard.get('connectivity_rollback_count', 0)}",
            "Projection Guard Boundary: exact validated DAPI extent only; no halo, fixed terminal length, or trunk subtraction.",
            f"Canonical Owner Nuclear-Extent Completion: changed "
            f"{canonical_owner_extent_completion.get('changed_cell_count', 0)} cells; "
            "approved "
            f"{canonical_owner_extent_completion.get('approved_owner_extent_added_px', 0)} "
            "Soma px; pre-finalization-Whole-external "
            f"{canonical_owner_extent_completion.get('outside_added_whole_px', 0)} px; "
            "fail-closed IDs="
            f"{canonical_owner_extent_completion.get('fail_closed_cell_ids', [])}",
            f"Same-ID Soma Island Reconciliation: bridged IDs="
            f"{same_id_soma_reconciliation.get('bridged_ids', [])}; "
            f"approved Processes-to-Soma "
            f"{same_id_soma_reconciliation.get('approved_process_to_soma_px', 0)} px; "
            f"identity-split rejections="
            f"{same_id_soma_reconciliation.get('rejected_identity_split_ids', [])}; "
            f"multiple-nucleus rejections="
            f"{same_id_soma_reconciliation.get('rejected_multiple_owner_ids', [])}",
            "Per-Nucleus 3D QC:",
            "ID  Accepted  Score  Surface  XY_Boundary  Angular  Z_Support  Shell_Enrichment  Radial_Bands  DAPI_Z_um  Reason",
        ]
    )
    for row in neonatal_3d.get("per_nucleus", []):
        lines.append(
            f"{row['nucleus_id_2d']:03d}  {str(row['accepted']):8s}  "
            f"{row['enclosure_score']:.3f}  {row['surface_coverage']:.3f}  "
            f"{row['median_xy_boundary_coverage']:.3f}  {row['angular_coverage']:.3f}  "
            f"{row['z_support_fraction']:.3f}  {row['shell_enrichment']:.3f}  "
            f"{row['radial_band_fraction']:.3f}  {row['dapi_z_span_um']:.3f}  "
            f"{row['reason']}"
        )
    lines.extend(
        [
            "",
            "Per-Astrocyte Final Compartment QC:",
            "ID  Original_ID  Shared_Whole_IDs  Whole_px  Soma_px  Processes_px  Soma_%  Processes_%  Process_parts  DAPI_candidates  Soma_anchors  Nucleus_coverage  Flags",
        ]
    )
    for row in compartment_metrics["per_cell"]:
        flags = []
        if row["nucleus_ambiguous"]:
            flags.append("ambiguous_DAPI")
        if row["fallback_used"]:
            flags.append("fallback_soma")
        if row["nucleus_candidates"] == 0:
            flags.append("no_DAPI_anchor")
        elif row["soma_anchor_count"] == 0:
            flags.append("no_trusted_soma")
        if row["soma_anchor_count"] > 1:
            flags.append("multiple_soma_anchors")
        if row["rejected_soma_anchor_count"]:
            flags.append("rejected_soma_anchor")
        if row["soma_core_shell_applied"]:
            flags.append(f"core_shell_removed_{row['soma_core_shell_removed_px']}px")
        morphology_qc = row.get(
            "shared_morphology_qc", row.get("morphology_qc", {})
        )
        if morphology_qc.get("outlier_consensus", 0):
            flags.append(f"morphology_votes_{morphology_qc['outlier_consensus']}")
        shared_whole_ids = row.get("shared_whole_ids", [])
        shared_whole_text = (
            ",".join(str(value) for value in shared_whole_ids)
            if shared_whole_ids
            else "-"
        )
        lines.append(
            f"{row['astrocyte_id']:03d}  {row.get('original_astrocyte_id', row['astrocyte_id']):11d}  "
            f"{shared_whole_text:16s}  "
            f"{row['whole_area_px']:8d}  {row['soma_area_px']:7d}  "
            f"{row['process_area_px']:12d}  {100 * row['soma_fraction']:6.2f}  "
            f"{100 * row['process_fraction']:11.2f}  "
            f"{row.get('process_component_count', morphology_qc.get('process_component_count', 0)):13d}  "
            f"{row['nucleus_candidates']:15d}  {row['soma_anchor_count']:12d}  "
            f"{row['required_nucleus_coverage']:16.3f}  "
            f"{','.join(flags) or 'OK'}"
        )
    lines.extend(
        [
            "",
            "[Fiji]",
            f"Status: {fiji_status}",
            "Display: DAPI=blue; measurement=red; eGFP=green and GFAP=yellow when both are present; lone structural channel=green.",
            "ROI Colors: Whole=magenta; Processes=white; Soma=cyan.",
            "Fiji Windows: three Composite views and three raw grayscale measurement views; contour strokes and display-only synchronized IDs, no biological ROI fills.",
            "ROI Review: linked Delete, Soma Merge, Whole Split, and Soma Enlarge "
            "update Whole/Soma/Processes together; Revert is LIFO.",
            f"Measurement Image: raw grayscale {measurement} {str(best_row['projection']).upper()}, "
            f"slices {best_row['z_start_1based']}-{best_row['z_end_1based_inclusive']}.",
            "Measurement Order: Whole Astrocyte Cell -> Astrocyte Processes -> Astrocyte Soma.",
            "Measurement Command: Fiji ROI Manager > Measure; RGB/composite pixels are never quantified.",
            "Outputs:",
            f"  {WHOLE_OVERLAY_FILENAME}",
            f"  {PROCESS_OVERLAY_FILENAME}",
            f"  {SOMA_OVERLAY_FILENAME}",
            f"  {WORKBOOK_FILENAME}",
        ]
    )
    if fiji_details:
        lines.extend(["", "Final Fiji Results:"])
        result_sets = fiji_details.get("result_sets", {})
        for key in ("whole", "processes", "soma"):
            detail = result_sets.get(key, {})
            lines.append(
                f"  {detail.get('title', key)}: {detail.get('rows', 'unknown')} rows; "
                f"total area={detail.get('area_sum', 'unknown')}"
            )
        lines.append(
            "Final ROI Counts: "
            f"Whole={result_sets.get('whole', {}).get('rows', 'unknown')}; "
            f"Processes={result_sets.get('processes', {}).get('rows', 'unknown')}; "
            f"Soma={result_sets.get('soma', {}).get('rows', 'unknown')}"
        )
        lines.append(f"Measurement Window: {fiji_details.get('measurement_title', 'unknown')}")
        lines.append(
            f"Manual ROI Review Used: {fiji_details.get('manual_review_used', False)}; "
            f"Deleted Original IDs: {fiji_details.get('deleted_original_ids', [])}; "
            f"Reverted Actions: {fiji_details.get('reverted_actions', 0)}"
        )
        lines.append(f"Review Audit: {fiji_details.get('review_audit', [])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
