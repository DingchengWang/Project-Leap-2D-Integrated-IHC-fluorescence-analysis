# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def _main_locked(argv: list[str] | None = None) -> int:
    global _CELLPOSE_EFFECTIVE_BATCH_SIZE
    global _EFFECTIVE_CANDIDATE_CPU_WORKERS
    global _EFFECTIVE_DAPI_INVENTORY_CPU_WORKERS
    global _RUN_STARTED_AT
    _RUN_STARTED_AT = time.perf_counter()
    _CELLPOSE_EFFECTIVE_BATCH_SIZE = None
    _EFFECTIVE_CANDIDATE_CPU_WORKERS = select_candidate_cpu_workers(
        CANDIDATE_CPU_WORKERS
    )
    _EFFECTIVE_DAPI_INVENTORY_CPU_WORKERS = select_candidate_cpu_workers(
        DAPI_INVENTORY_CPU_WORKERS
    )
    _RUNTIME_TIMINGS.update(
        {
            "cellpose_model_init_seconds": 0.0,
            "cellpose_inference_events": [],
            "candidate_postprocess_seconds": [],
            "candidate_total_seconds": [],
            "candidate_stage_wall_seconds": 0.0,
            "rank_candidates_seconds": 0.0,
            "compartment_split_seconds": 0.0,
        }
    )
    args = parse_args(argv)
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The analysis environment requires openpyxl 3.1.5 before analysis starts"
        ) from exc
    if args.fiji_timeout_minutes <= 0:
        raise ValueError("--fiji-timeout-minutes must be positive")
    if (
        args.dapi_fragment_workload_preflight_only
        and args.dapi_fragment_workload_json is None
    ):
        raise ValueError(
            "--dapi-fragment-workload-json is required with "
            "--dapi-fragment-workload-preflight-only"
        )
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dapi_fragment_workload_path = (
        args.dapi_fragment_workload_json.expanduser().resolve()
        if args.dapi_fragment_workload_json is not None
        else output_dir / "IHC_2D_DAPI_Fragment_Workload_Failure.json"
    )
    whole_overlay_path = output_dir / WHOLE_OVERLAY_FILENAME
    soma_overlay_path = output_dir / SOMA_OVERLAY_FILENAME
    process_overlay_path = output_dir / PROCESS_OVERLAY_FILENAME
    report_path = output_dir / REPORT_FILENAME
    workbook_path = output_dir / WORKBOOK_FILENAME

    paths, ignored_files = discover_channel_paths(input_dir)
    if "eGFP" not in paths:
        raise ValueError(
            "The single-file fallback supports the eGFP route only. Use "
            "run_project_leap_2d.command for mature GFAP-only analysis."
        )
    filename_age_decision = detect_filename_age_profile(paths)
    structural_channels = [channel for channel in STRUCTURAL_CHANNELS if channel in paths]
    measurement = measurement_channel(paths)
    mode = channel_mode(paths)
    metadata = {channel: read_meta(path) for channel, path in paths.items()}
    reference_shape = tuple(metadata["DAPI"]["shape"])
    mismatched = {
        channel: tuple(channel_meta["shape"])
        for channel, channel_meta in metadata.items()
        if tuple(channel_meta["shape"]) != reference_shape
    }
    if mismatched:
        raise ValueError(f"Channel shapes do not match DAPI {reference_shape}: {mismatched}")
    if any(channel_meta["axes"] != "ZYX" for channel_meta in metadata.values()):
        raise ValueError("All input channels must have explicit ZYX axes")
    validate_shared_geometry(metadata)
    if any(
        metadata["DAPI"].get(key) is None
        or float(metadata["DAPI"][key]) <= 0
        for key in ("pixel_width_um", "pixel_height_um")
    ):
        raise ValueError(
            "Positive XY pixel calibration is required for physical Soma and morphology thresholds"
        )

    print_terminal_stage(
        "01 | INPUT CONTRACT",
        f"{PRODUCT_DISPLAY_NAME}\nScientific core: {ANALYSIS_CORE_NAME}",
    )
    print(f"Input: {input_dir}", flush=True)
    print(f"ROI mode: {mode}; measurement: {measurement} (excluded from ROI definition)", flush=True)
    for channel, path in paths.items():
        print(f"  {channel}: {path.name}", flush=True)
    for note in ignored_files:
        print(f"  ignored: {note}", flush=True)
    if filename_age_decision is not None:
        print(
            f"Age profile from filename: {filename_age_decision.profile} "
            f"({', '.join(filename_age_decision.tagged_files)})",
            flush=True,
        )

    print_terminal_stage(
        "02 | STACK LOADING AND ADAPTIVE Z PREPARATION",
        "Common method: load each split ZYX stack once, verify shared calibration, "
        "and build five adaptive contiguous Z intervals.",
    )
    _CELLPOSE_MASK_CACHE.clear()
    clear_candidate_computation_caches()
    dapi_stack = load_stack(paths["DAPI"])
    structural_stacks = {
        channel: load_stack(paths[channel])
        for channel in structural_channels
    }
    z_activity_profile = z_profile(structural_stacks)
    chosen_specs = complete_candidate_specs(reference_shape[0], structural_stacks)
    if args.disable_cellpose:
        chosen_specs = [replace(spec, cellpose=False) for spec in chosen_specs]

    rows: list[dict] = []
    candidate_masks: list[np.ndarray] = []
    projection_keys: list[tuple[int, int, str]] = []
    projection_cache: dict[tuple[int, int, str], tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    structural_map_cache: dict[tuple, np.ndarray] = {}

    def evaluate_timed(
        position: int,
        spec: TestSpec,
        context: CandidateWindowContext,
    ):
        candidate_started = time.perf_counter()
        mask, row, projection_key = evaluate_ihc_candidate(
            candidate_number=position,
            candidate_count=len(chosen_specs),
            spec=spec,
            input_dir=input_dir,
            structural_channels=structural_channels,
            dapi_stack=dapi_stack,
            structural_stacks=structural_stacks,
            profile=z_activity_profile,
            projection_cache=projection_cache,
            structural_map_cache=structural_map_cache,
            emit_progress=False,
            window_context=context,
        )
        candidate_total = time.perf_counter() - candidate_started
        return mask, row, projection_key, candidate_total, candidate_total

    def print_candidate_row(
        row: dict,
        spec: TestSpec,
        completed_count: int,
        worker_seconds: float,
    ) -> None:
        print_terminal_event(
            f"Candidate {int(row['candidate']):02d} | "
            f"completed={completed_count:02d}/{len(chosen_specs)} | "
            f"module={candidate_module_display_name(spec)} | "
            f"family={candidate_profile_family(spec)} | "
            f"profile={candidate_profile_name(spec)} | "
            f"threshold={candidate_threshold_display_name(spec)} | "
            f"Z={int(row['z_start_1based'])}-{int(row['z_end_1based_inclusive'])} | "
            f"area={int(row['mask_area_px'])} | "
            f"components={int(row['connected_components'])} | "
            f"soma_support={int(row['soma_supported_components'])} | "
            f"fine_added={int(row['fine_branch_retained_px'])} | "
            f"worker_time={float(worker_seconds):.3f} s"
        )

    if len(chosen_specs) != TOTAL_CANDIDATE_COUNT:
        raise AssertionError(
            f"Complete candidate evaluation requires exactly {TOTAL_CANDIDATE_COUNT} candidates"
        )
    z_group_counts: dict[str, int] = {}
    for spec in chosen_specs:
        z_group_counts[spec.z_mode] = z_group_counts.get(spec.z_mode, 0) + 1
    if (
        len(z_group_counts) != EXPECTED_Z_INTERVAL_COUNT
        or set(z_group_counts.values()) != {EXPECTED_PROFILES_PER_Z}
    ):
        raise AssertionError(
            "Complete candidate evaluation requires five Z groups with exactly eighteen profiles per group"
        )
    print_terminal_stage(
        "03 | CELLPOSE-SAM GPU INFERENCE",
        "Common method: one serial Apple Metal/MPS Cellpose-SAM inference per Z "
        "interval; masks are cached and reused by all matching candidates.\n"
        f"Cellpose batch requested={CELLPOSE_BATCH_SIZE}; "
        f"candidate CPU workers requested={CANDIDATE_CPU_WORKERS}, "
        f"effective={_EFFECTIVE_CANDIDATE_CPU_WORKERS}; "
        f"3D DAPI workers requested={DAPI_INVENTORY_CPU_WORKERS}, "
        f"effective={_EFFECTIVE_DAPI_INVENTORY_CPU_WORKERS}.",
    )
    candidate_stage_started = time.perf_counter()
    candidate_contexts = build_candidate_window_contexts(
        chosen_specs=chosen_specs,
        structural_channels=structural_channels,
        dapi_stack=dapi_stack,
        structural_stacks=structural_stacks,
        profile=z_activity_profile,
        projection_cache=projection_cache,
        structural_map_cache=structural_map_cache,
    )
    context_groups: list[tuple[CandidateWindowContext, list[TestSpec]]] = []
    context_positions: dict[int, int] = {}
    for spec, context in zip(chosen_specs, candidate_contexts):
        group_position = context_positions.get(id(context))
        if group_position is None:
            context_positions[id(context)] = len(context_groups)
            context_groups.append((context, [spec]))
        else:
            context_groups[group_position][1].append(spec)

    print_terminal_stage(
        "04 | CPU CANDIDATE MORPHOLOGY",
        "Common method: reusable DAPI, Sato, top-hat, distributional-threshold, "
        "and structural features are precomputed once per complete cache key; "
        "90 immutable candidate jobs then run in the global worker pool.",
    )
    with ThreadPoolExecutor(
        max_workers=min(_EFFECTIVE_CANDIDATE_CPU_WORKERS, len(chosen_specs)),
        thread_name_prefix="ihc-candidate",
    ) as executor:
        precompute_jobs = candidate_precompute_jobs(
            context_groups=context_groups,
            input_dir=input_dir,
            structural_channels=structural_channels,
        )
        precompute_futures = [
            executor.submit(function, *job_args)
            for _priority, function, job_args in precompute_jobs
        ]
        for future in precompute_futures:
            future.result()

        primary_jobs: list[tuple[int, TestSpec, CandidateWindowContext]] = []
        reuse_jobs: list[tuple[int, TestSpec, CandidateWindowContext]] = []
        seen_base_keys: set[tuple] = set()
        for position, (spec, context) in enumerate(
            zip(chosen_specs, candidate_contexts),
            start=1,
        ):
            base_key = candidate_base_cache_key(context.structural_map, spec)
            job = (position, spec, context)
            if base_key in seen_base_keys:
                reuse_jobs.append(job)
            else:
                seen_base_keys.add(base_key)
                primary_jobs.append(job)
        def job_priority(job: tuple[int, TestSpec, CandidateWindowContext]) -> tuple:
            position, spec, _context = job
            return (
                float(spec.fine_branch_detail_percentile),
                float(spec.fine_branch_intensity_percentile),
                int(spec.fine_branch_min_area),
                position,
            )

        primary_jobs.sort(key=job_priority)
        reuse_jobs.sort(key=job_priority)
        futures_by_position = {
            position: executor.submit(evaluate_timed, position, spec, context)
            for position, spec, context in (*primary_jobs, *reuse_jobs)
        }
        future_positions = {
            future: position for position, future in futures_by_position.items()
        }
        ordered_results: list[tuple | None] = [None] * len(chosen_specs)
        candidate_errors: dict[int, BaseException] = {}
        completed_candidates = 0
        for future in as_completed(future_positions):
            position = future_positions[future]
            try:
                result = future.result()
                ordered_results[position - 1] = result
                completed_candidates += 1
                print_candidate_row(
                    result[1],
                    chosen_specs[position - 1],
                    completed_candidates,
                    float(result[3]),
                )
            except BaseException as exc:
                candidate_errors[position] = exc
                print_terminal_event(
                    f"Candidate {position:02d} failed | "
                    f"module={candidate_module_display_name(chosen_specs[position - 1])} | "
                    f"profile={candidate_profile_name(chosen_specs[position - 1])} | "
                    f"error={exc!r}"
                )
        if candidate_errors:
            first_failed = min(candidate_errors)
            raise RuntimeError(
                f"Candidate {first_failed:02d} failed during parallel morphology"
            ) from candidate_errors[first_failed]

    for result in ordered_results:
        if result is None:
            raise RuntimeError("Candidate scheduler returned an incomplete result set")
        mask, row, projection_key, candidate_total, candidate_postprocess = result
        candidate_totals = _RUNTIME_TIMINGS["candidate_total_seconds"]
        candidate_postprocess_times = _RUNTIME_TIMINGS[
            "candidate_postprocess_seconds"
        ]
        assert isinstance(candidate_totals, list)
        assert isinstance(candidate_postprocess_times, list)
        candidate_totals.append(float(candidate_total))
        candidate_postprocess_times.append(float(candidate_postprocess))
        candidate_masks.append(mask)
        rows.append(row)
        projection_keys.append(projection_key)
    _RUNTIME_TIMINGS["candidate_stage_wall_seconds"] = float(
        time.perf_counter() - candidate_stage_started
    )

    print_terminal_stage(
        "05 | BALANCED RANKING AND NON-REGRESSION GUARD",
        "Common method: unified family-by-Z ranking, near-duplicate clustering, "
        "and a same-Z challenger comparison against the pre-distribution baseline incumbent.",
    )
    ranking_started = time.perf_counter()
    best_position, selection_guard = rank_complete_production_candidates(
        candidate_masks,
        rows,
    )
    _RUNTIME_TIMINGS["rank_candidates_seconds"] = float(
        time.perf_counter() - ranking_started
    )
    best_mask = candidate_masks[best_position]
    best_row = rows[best_position]
    best_spec = chosen_specs[best_position]
    best_projection_key = projection_keys[best_position]
    dapi_projection, structural_projections = projection_cache[best_projection_key]
    candidate_component_count = int(measure.label(best_mask, connectivity=2).max())
    print(
        f"Auto-selected candidate {best_row['candidate']:02d}: "
        f"score={best_row['auto_selection_score']}, "
        f"slices={best_row['z_start_1based']}-{best_row['z_end_1based_inclusive']}, "
        f"connected_components={candidate_component_count}",
        flush=True,
    )
    print(
        "Selection non-regression guard: "
        f"morphology_baseline={selection_guard['morphology_baseline_incumbent_candidate']:02d}, "
        "pre_distribution_baseline="
        f"{selection_guard['pre_distribution_baseline_incumbent_candidate']:02d}, "
        f"challenger={selection_guard['family_balanced_challenger_candidate']:02d}, "
        f"selected={selection_guard['selected_candidate']:02d}, "
        f"reason={selection_guard['guard']['reason']}",
        flush=True,
    )

    best_struct_key = (
        *best_projection_key,
        round(best_spec.egfp_weight, 6),
        round(best_spec.gfap_weight, 6),
        round(best_spec.smooth_sigma, 6),
    )
    best_struct = structural_map_cache[best_struct_key]
    best_cellpose_mask = np.zeros_like(best_mask, dtype=bool)
    if best_spec.cellpose and not args.disable_cellpose:
        best_cellpose_key = (
            tuple(structural_channels),
            int(best_row["z_start_0based"]),
            int(best_row["z_end_0based_inclusive"]),
            best_row["projection"],
            round(best_spec.egfp_weight, 3),
            round(best_spec.gfap_weight, 3),
            round(best_spec.smooth_sigma, 3),
            round(best_spec.cellpose_cellprob, 3),
            round(best_spec.cellpose_diameter, 3),
            best_spec.cellpose_max_side,
        )
        try:
            best_cellpose_mask, _ = run_cellpose_mask(
                best_struct,
                best_spec,
                best_cellpose_key,
            )
        except Exception as exc:
            print(f"Compartment Cellpose prior unavailable; continuing without it: {exc!r}", flush=True)
            best_cellpose_mask = np.zeros_like(best_mask, dtype=bool)

    del futures_by_position, future_positions, ordered_results
    del precompute_futures, precompute_jobs
    del primary_jobs, reuse_jobs, candidate_contexts, context_groups, context_positions
    del seen_base_keys
    candidate_masks.clear()
    projection_keys.clear()
    projection_cache.clear()
    structural_map_cache.clear()
    _CELLPOSE_MASK_CACHE.clear()
    clear_candidate_computation_caches()
    gc.collect()

    print_terminal_stage(
        "06 | AGE PROFILE, 3D DAPI OWNERSHIP, AND COMPARTMENTS",
        "Common method: determine mature/neonatal profile after Whole/Z selection, "
        "validate calibrated 3D DAPI objects, reconcile identities, and enforce "
        "Whole = Soma + Processes with synchronized IDs.",
    )
    pixel_width_um = float(metadata["DAPI"]["pixel_width_um"])
    pixel_height_um = float(metadata["DAPI"]["pixel_height_um"])
    age_profile_decision = filename_age_decision or classify_age_profile(
        best_mask,
        dapi_projection,
        best_struct,
        pixel_width_um,
        pixel_height_um,
    )
    if age_profile_decision.source == "morphology_classifier":
        print(
            f"Age profile: {age_profile_decision.profile} "
            f"(morphology neonatal_score={age_profile_decision.neonatal_score:.6f}, "
            f"threshold={age_profile_decision.threshold:.2f})",
            flush=True,
        )
    else:
        print(f"Age profile: {age_profile_decision.profile} (filename)", flush=True)
    z0 = int(best_row["z_start_0based"])
    z1 = int(best_row["z_end_0based_inclusive"])
    neonatal_3d_context = None
    if structural_stacks:
        ownership_structural_channel = (
            "eGFP" if "eGFP" in structural_stacks else "GFAP"
        )
        dapi_depth_um = metadata["DAPI"].get("pixel_depth_um")
        structural_depth_um = metadata[ownership_structural_channel].get(
            "pixel_depth_um"
        )
        if dapi_depth_um is not None and structural_depth_um is not None:
            dapi_depth_um = float(dapi_depth_um)
            structural_depth_um = float(structural_depth_um)
            if not math.isclose(
                dapi_depth_um,
                structural_depth_um,
                rel_tol=1e-4,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "DAPI/structural Z calibration mismatch prevents 3D nucleus ownership: "
                    f"DAPI={dapi_depth_um}, "
                    f"{ownership_structural_channel}={structural_depth_um} um"
                )
            neonatal_3d_context = Neonatal3DContext(
                dapi_stack=dapi_stack,
                egfp_stack=structural_stacks[ownership_structural_channel],
                z_start_0based=z0,
                z_end_0based_inclusive=z1,
                pixel_depth_um=dapi_depth_um,
                calibration_source=str(metadata["DAPI"].get("pixel_depth_source") or "unknown"),
                structural_channel=ownership_structural_channel,
            )
        else:
            print(
                "3D nucleus ownership skipped: calibrated DAPI/structural Z spacing is unavailable",
                flush=True,
            )
    if args.dapi_fragment_workload_preflight_only:
        if neonatal_3d_context is None:
            raise RuntimeError(
                "DAPI fragment-workload preflight requires calibrated DAPI and structural "
                "ZYX stacks with matching positive Z spacing"
            )
        print_terminal_stage(
            "06 | DAPI 3D WORKLOAD PREFLIGHT",
            "Validation-only mode: use the production parent reconstruction path, "
            "count fragment workload, and stop before fragment evaluation, the "
            "measurement stack, or Fiji.",
        )
        preflight_started = time.perf_counter()
        preflight_inventory = build_dapi_object_inventory_3d(
            best_mask,
            dapi_projection,
            neonatal_3d_context,
            pixel_width_um,
            pixel_height_um,
            compartment_config_for_profile("mature"),
            max_workers=1,
            preflight_only=True,
            workload_diagnostic_path=dapi_fragment_workload_path,
        )
        _RUNTIME_TIMINGS["compartment_split_seconds"] = float(
            time.perf_counter() - preflight_started
        )
        workload_summary = preflight_inventory.metrics.get(
            "dapi_fragment_workload"
        )
        if not isinstance(workload_summary, dict):
            raise RuntimeError(
                "DAPI fragment-workload preflight did not return a workload summary"
            )
        workload_summary.update(
            {
                "input_dir": str(input_dir),
                "input_channels": {
                    channel: path.name for channel, path in paths.items()
                },
                "product_display_name": PRODUCT_DISPLAY_NAME,
                "analysis_core_name": ANALYSIS_CORE_NAME,
                "selected_candidate": int(best_row["candidate"]),
                "selected_age_profile": age_profile_decision.profile,
                "selected_z_range_1based": [int(z0 + 1), int(z1 + 1)],
                "measurement_stack_loaded": False,
                "fragment_evaluator_called": False,
                "fiji_launched": False,
                "production_outputs_replaced": False,
            }
        )
        atomic_write_dapi_fragment_workload_json(
            dapi_fragment_workload_path,
            workload_summary,
        )
        print(
            "DAPI fragment-workload preflight completed safely: "
            f"parents={int(workload_summary['parents_linked_to_whole'])}, "
            f"fragments={int(workload_summary['total_fragments'])}, "
            "max_parent_voxel_checks="
            f"{int(workload_summary['max_parent_voxel_comparisons']):,}",
            flush=True,
        )
        print(f"Workload JSON: {dapi_fragment_workload_path}", flush=True)
        return 0
    compartment_started = time.perf_counter()
    whole_labels, soma_labels, process_labels, compartment_metrics = (
        split_astrocyte_compartments_for_profile(
            best_mask,
            dapi_projection,
            best_struct,
            best_cellpose_mask,
            pixel_width_um,
            pixel_height_um,
            age_profile_decision.profile,
            neonatal_3d_context=neonatal_3d_context,
            dapi_fragment_workload_diagnostic_path=dapi_fragment_workload_path,
        )
    )
    _RUNTIME_TIMINGS["compartment_split_seconds"] = float(
        time.perf_counter() - compartment_started
    )
    compartment_metrics["age_profile"] = asdict(age_profile_decision)
    compartment_metrics["adaptation"] = (
        (
            "object-preserving calibrated 3D nucleus ownership guard followed by "
            "the frozen mature Soma/Processes path"
            if neonatal_3d_context is not None
            else "frozen mature compartment path without calibrated 3D ownership"
        )
        if age_profile_decision.profile == "mature"
        else (
            "shared Whole geometry followed by object-preserving calibrated 3D DAPI/"
            f"{neonatal_3d_context.structural_channel} ownership, mandatory multi-nucleus "
            "partition, neonatal shape-preserving Soma/Processes, and whole-ID valid-Soma gate"
            if neonatal_3d_context is not None
            else (
                "frozen Whole geometry followed by neonatal 2D anchors, multi-center ID, "
                "neonatal Soma/Processes profile, and whole-ID valid-Soma cell gate"
            )
        )
    )
    roi_count = int(whole_labels.max())
    print(
        "Compartments: "
        f"Whole_ROIs={roi_count} "
        f"Whole={compartment_metrics['whole_area_px']} px, "
        f"Soma={compartment_metrics['soma_area_px']} px "
        f"({compartment_metrics['soma_area_fraction']:.1%}), "
        f"Processes={compartment_metrics['process_area_px']} px "
        f"({compartment_metrics['process_area_fraction']:.1%})",
        flush=True,
    )

    print_terminal_stage(
        "07 | RAW MEASUREMENT PROJECTION",
        f"Common method: project the untouched grayscale {measurement} stack over "
        f"the selected inclusive Z range {z0 + 1}-{z1 + 1}; rendered composites "
        "remain display-only.",
    )
    measurement_projection_started = time.perf_counter()
    measurement_substack = tf.imread(str(paths[measurement]), key=range(z0, z1 + 1))
    measurement_projection = project(
        measurement_substack,
        0,
        measurement_substack.shape[0] - 1,
        best_row["projection"],
    )
    print(
        "Raw fluorescence measurement projection complete | "
        f"elapsed={time.perf_counter() - measurement_projection_started:.3f} s",
        flush=True,
    )

    if args.skip_fiji:
        debug_whole_overlay_path = output_dir / DEBUG_WHOLE_OVERLAY_FILENAME
        debug_soma_overlay_path = output_dir / DEBUG_SOMA_OVERLAY_FILENAME
        debug_process_overlay_path = output_dir / DEBUG_PROCESS_OVERLAY_FILENAME
        debug_report_path = output_dir / DEBUG_REPORT_FILENAME
        debug_state_path = output_dir / DEBUG_STATE_FILENAME
        preview_rgb = make_fiji_like_composite(
            dapi_projection,
            structural_projections,
            measurement_projection,
        )
        Image.fromarray(
            draw_smooth_label_contours(preview_rgb, whole_labels, best_spec, (255, 0, 255))
        ).save(debug_whole_overlay_path)
        Image.fromarray(
            draw_smooth_label_contours(preview_rgb, soma_labels, best_spec, (0, 255, 255))
        ).save(debug_soma_overlay_path)
        Image.fromarray(
            draw_smooth_label_contours(preview_rgb, process_labels, best_spec, (255, 255, 255))
        ).save(debug_process_overlay_path)
        np.savez_compressed(
            debug_state_path,
            whole_mask=(whole_labels > 0).astype(np.uint8),
            candidate_whole_mask=best_mask.astype(np.uint8),
            whole_labels=whole_labels.astype(np.uint16),
            soma_labels=soma_labels.astype(np.uint16),
            process_labels=process_labels.astype(np.uint16),
            canonical_owner_extent_approved_owner_extent_labels=np.asarray(
                compartment_metrics.get(
                    "_canonical_owner_extent_approved_owner_extent_labels",
                    np.zeros_like(whole_labels, dtype=np.uint16),
                ),
                dtype=np.uint16,
            ),
            same_id_soma_reconciliation_approved_process_to_soma_labels=np.asarray(
                compartment_metrics.get(
                    "_same_id_soma_reconciliation_approved_process_to_soma_labels",
                    np.zeros_like(whole_labels, dtype=np.uint16),
                ),
                dtype=np.uint16,
            ),
            canonical_nucleus_instance_core_labels_2d=np.asarray(
                compartment_metrics.get(
                    "_canonical_nucleus_instance_core_labels_2d",
                    np.zeros_like(whole_labels, dtype=np.uint32),
                ),
                dtype=np.uint32,
            ),
            canonical_nucleus_instance_extent_labels_2d=np.asarray(
                compartment_metrics.get(
                    "_canonical_nucleus_instance_extent_labels_2d",
                    np.zeros_like(whole_labels, dtype=np.uint32),
                ),
                dtype=np.uint32,
            ),
            dapi_projection=dapi_projection,
            structural_map=best_struct,
            cellpose_mask=best_cellpose_mask.astype(np.uint8),
            measurement_projection=measurement_projection,
            pixel_width_um=np.asarray(pixel_width_um),
            pixel_height_um=np.asarray(pixel_height_um),
            age_profile=np.asarray(age_profile_decision.profile),
            age_profile_source=np.asarray(age_profile_decision.source),
            age_profile_neonatal_score=np.asarray(
                np.nan
                if age_profile_decision.neonatal_score is None
                else age_profile_decision.neonatal_score
            ),
        )
        write_analysis_report(
            debug_report_path,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            structural_channels=structural_channels,
            measurement=measurement,
            best_row=best_row,
            candidate_rows=rows,
            candidate_specs=chosen_specs,
            roi_count=roi_count,
            compartment_metrics=compartment_metrics,
            fiji_status="Skipped by --skip-fiji (debug preview only; no measurement performed)",
        )
        print(f"Debug Whole overlay: {debug_whole_overlay_path}", flush=True)
        print(f"Debug Soma overlay: {debug_soma_overlay_path}", flush=True)
        print(f"Debug Processes overlay: {debug_process_overlay_path}", flush=True)
        print(f"Debug compartment state: {debug_state_path}", flush=True)
        print(f"Debug report: {debug_report_path}", flush=True)
        return 0

    print_terminal_stage(
        "08 | FIJI REVIEW, RAW-GRAYSCALE MEASUREMENT, AND PUBLICATION",
        "Common method: open three Composite and three raw grayscale views; "
        "optionally apply linked Delete, Soma-only proximity-chain Merge, or "
        "Revert; then measure Whole, Processes, and Soma on the raw grayscale projection.",
    )
    run_dir: Path | None = None
    try:
        fiji_runtime_started = time.perf_counter()
        selected_projections = {
            "DAPI": dapi_projection,
            measurement: measurement_projection,
            **structural_projections,
        }
        run_dir, manifest_path = prepare_fiji_runtime(
            output_dir=output_dir,
            paths=paths,
            metadata=metadata,
            structural_channels=structural_channels,
            measurement=measurement,
            best_row=best_row,
            whole_labels=whole_labels,
            soma_labels=soma_labels,
            process_labels=process_labels,
            selected_projections=selected_projections,
            auto_continue=args.fiji_auto_continue,
        )
        runtime_report = run_dir / "analysis_report_pending.txt"
        write_analysis_report(
            runtime_report,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            structural_channels=structural_channels,
            measurement=measurement,
            best_row=best_row,
            candidate_rows=rows,
            candidate_specs=chosen_specs,
            roi_count=roi_count,
            compartment_metrics=compartment_metrics,
            fiji_status="Pending automatic Fiji six-window display and native measurement",
        )
        launcher = find_fiji_launcher(args.fiji_launcher)

        del measurement_substack, measurement_projection, selected_projections
        del neonatal_3d_context
        del dapi_stack, structural_stacks, candidate_masks, projection_cache, structural_map_cache
        _CELLPOSE_MASK_CACHE.clear()
        clear_candidate_computation_caches()
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        print(
            "Fiji runtime package prepared and analysis memory released | "
            f"elapsed={time.perf_counter() - fiji_runtime_started:.3f} s; "
            "launching Fiji...",
            flush=True,
        )
        fiji_details = launch_fiji_workflow(
            launcher=launcher,
            run_dir=run_dir,
            manifest_path=manifest_path,
            timeout_minutes=args.fiji_timeout_minutes,
        )
        print_terminal_event(
            "Fiji workflow returned; validating synchronized ROI IDs, measurement "
            "partitions, overlays, report, and workbook before atomic publication."
        )
        if bool(fiji_details.get("cancelled", False)):
            shutil.rmtree(run_dir, ignore_errors=True)
            print("Analysis cancelled in Fiji before measurement; no production output was replaced.", flush=True)
            return 0
        final_roi_count = int(fiji_details.get("roi_count", -1))
        if final_roi_count < 1:
            raise RuntimeError(f"Fiji returned an invalid final ROI count: {fiji_details}")
        required_headings = {
            "Label",
            "Area",
            "Mean",
            "Median",
            "Min",
            "Max",
            "IntDen",
            "ROI_Index",
            "Astrocyte_ID",
            "Original_Astrocyte_ID",
            "Source_Original_Astrocyte_IDs",
            "Compartment",
            "ROI_Name",
        }
        final_roi_ids = [int(value) for value in fiji_details.get("final_roi_ids", [])]
        if final_roi_ids != list(range(1, final_roi_count + 1)):
            raise RuntimeError(f"Fiji returned invalid final Astrocyte IDs: {final_roi_ids}")
        final_process_ids = [int(value) for value in fiji_details.get("final_process_ids", [])]
        final_soma_ids = [int(value) for value in fiji_details.get("final_soma_ids", [])]
        if final_process_ids != final_roi_ids:
            raise RuntimeError(f"Fiji Processes IDs do not match Whole IDs: {final_process_ids}")
        if final_soma_ids != final_roi_ids:
            raise RuntimeError(
                "Fiji Soma IDs do not match Whole IDs exactly: "
                f"Soma={final_soma_ids}, Whole={final_roi_ids}"
            )
        result_sets = fiji_details.get("result_sets", {})
        expected_original_ids = [
            int(value)
            for value in result_sets.get("whole", {}).get(
                "original_astrocyte_ids",
                [],
            )
        ]
        if len(expected_original_ids) != final_roi_count or len(
            set(expected_original_ids)
        ) != final_roi_count:
            raise RuntimeError(
                f"Fiji returned invalid Original Astrocyte IDs: {expected_original_ids}"
            )
        for key in ("whole", "processes", "soma"):
            detail = result_sets.get(key, {})
            expected_count = final_roi_count
            if int(detail.get("rows", -1)) != expected_count:
                raise RuntimeError(f"Fiji {key} Results row count is inconsistent: {detail}")
            result_ids = [int(value) for value in detail.get("astrocyte_ids", [])]
            expected_result_ids = final_roi_ids
            if result_ids != expected_result_ids:
                raise RuntimeError(f"Fiji {key} Results Astrocyte IDs are inconsistent: {detail}")
            result_original_ids = [
                int(value) for value in detail.get("original_astrocyte_ids", [])
            ]
            if result_original_ids != expected_original_ids:
                raise RuntimeError(
                    f"Fiji {key} Original Astrocyte IDs are inconsistent: {detail}"
                )
            observed_headings = set(detail.get("headings", []))
            if expected_count > 0 and not required_headings.issubset(observed_headings):
                raise RuntimeError(
                    f"Fiji {key} Results columns are incomplete: expected {sorted(required_headings)}, "
                    f"observed {sorted(observed_headings)}"
                )
            row_data = detail.get("row_data", [])
            if len(row_data) != expected_count:
                raise RuntimeError(f"Fiji {key} Results row payload is incomplete: {detail}")
            payload_ids = [int(row["Astrocyte_ID"]) for row in row_data]
            if payload_ids != expected_result_ids:
                raise RuntimeError(
                    f"Fiji {key} row payload Astrocyte IDs are inconsistent: {payload_ids}"
                )
            payload_original_ids = [
                int(row["Original_Astrocyte_ID"]) for row in row_data
            ]
            if payload_original_ids != expected_original_ids:
                raise RuntimeError(
                    f"Fiji {key} row payload Original IDs are inconsistent: "
                    f"{payload_original_ids}"
                )
            if any(not np.isfinite(float(row["Median"])) for row in row_data):
                raise RuntimeError(f"Fiji {key} returned a non-finite Median value")
        whole_area = float(result_sets["whole"]["area_sum"])
        partition_area = float(result_sets["processes"]["area_sum"]) + float(
            result_sets["soma"]["area_sum"]
        )
        area_tolerance = max(
            pixel_width_um * pixel_height_um * final_roi_count * 2.0,
            abs(whole_area) * 1e-6,
        )
        if abs(whole_area - partition_area) > area_tolerance:
            raise RuntimeError(
                "Fiji compartment area identity failed: "
                f"Whole={whole_area}, Processes+Soma={partition_area}"
            )
        whole_intden = float(result_sets["whole"]["integrated_density_sum"])
        partition_intden = float(result_sets["processes"]["integrated_density_sum"]) + float(
            result_sets["soma"]["integrated_density_sum"]
        )
        if abs(whole_intden - partition_intden) > max(1.0, abs(whole_intden) * 1e-6):
            raise RuntimeError(
                "Fiji compartment integrated-density identity failed: "
                f"Whole={whole_intden}, Processes+Soma={partition_intden}"
            )

        overlay_paths = fiji_details.get("overlay_paths", {})
        staged_overlays: dict[str, Path] = {}
        for key in ("whole", "processes", "soma"):
            staged_overlay = Path(str(overlay_paths.get(key, ""))).resolve()
            if staged_overlay.parent != run_dir.resolve() or not staged_overlay.is_file():
                raise RuntimeError(
                    f"Fiji returned an invalid run-specific {key} overlay path: {staged_overlay}"
                )
            with Image.open(staged_overlay) as image:
                if image.size != (reference_shape[2], reference_shape[1]) or image.mode != "RGB":
                    raise RuntimeError(
                        f"Unexpected Fiji {key} overlay format: mode={image.mode}, size={image.size}"
                    )
                image.verify()
            staged_overlays[key] = staged_overlay
        completed_report = run_dir / "analysis_report_completed.txt"
        staged_workbook = run_dir / WORKBOOK_FILENAME
        write_measurement_workbook(
            staged_workbook,
            fiji_details=fiji_details,
            measurement=measurement,
            best_row=best_row,
            age_profile=age_profile_decision.profile,
        )
        validate_measurement_workbook(staged_workbook, fiji_details)
        write_analysis_report(
            completed_report,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            structural_channels=structural_channels,
            measurement=measurement,
            best_row=best_row,
            candidate_rows=rows,
            candidate_specs=chosen_specs,
            roi_count=roi_count,
            compartment_metrics=compartment_metrics,
            fiji_status=(
                "Completed after the Fiji pre-measurement review decision: six ROI image windows, "
                "three native measurement Results windows, and one synchronized XLSX workbook"
            ),
            fiji_details=fiji_details,
        )
        publish_output_bundle(
            staged_files={
                "whole": staged_overlays["whole"],
                "processes": staged_overlays["processes"],
                "soma": staged_overlays["soma"],
                "report": completed_report,
                "workbook": staged_workbook,
            },
            final_files={
                "whole": whole_overlay_path,
                "processes": process_overlay_path,
                "soma": soma_overlay_path,
                "report": report_path,
                "workbook": workbook_path,
            },
            run_dir=run_dir,
        )
        print_terminal_event("Validated production bundle published atomically.")
    except Exception as exc:
        if run_dir is not None:
            try:
                failed_report = run_dir / "analysis_report_failed.txt"
                write_analysis_report(
                    failed_report,
                    input_dir=input_dir,
                    paths=paths,
                    metadata=metadata,
                    structural_channels=structural_channels,
                    measurement=measurement,
                    best_row=best_row,
                    candidate_rows=rows,
                    candidate_specs=chosen_specs,
                    roi_count=roi_count,
                    compartment_metrics=compartment_metrics,
                    fiji_status=f"Failed: {exc!r}; previous production outputs were preserved",
                )
                print(f"Failure diagnostics retained at: {run_dir}", flush=True)
            except Exception:
                pass
        raise

    shutil.rmtree(run_dir, ignore_errors=True)
    print(f"Whole overlay: {whole_overlay_path}", flush=True)
    print(f"Processes overlay: {process_overlay_path}", flush=True)
    print(f"Soma overlay: {soma_overlay_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Workbook: {workbook_path}", flush=True)
    for key in ("whole", "processes", "soma"):
        detail = fiji_details["result_sets"][key]
        print(f"Fiji {detail['title']}: {detail['rows']} rows", flush=True)
    print_runtime_timing_summary(fiji_details)
    return 0

def main(argv: list[str] | None = None) -> int:
    try:
        with analysis_lock():
            return _main_locked(argv)
    except DapiFragmentWorkloadLimitExceeded as exc:
        print(exc.user_message(), flush=True)
        return 2
