# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best

def neonatal_3d_slice_angular_coverage(
    boundary: np.ndarray,
    covered: np.ndarray,
    sector_count: int,
    sector_support_fraction: float,
) -> float:
    coords = np.argwhere(boundary)
    if coords.size == 0:
        return 0.0
    center_y, center_x = coords.mean(axis=0)
    angles = np.mod(
        np.arctan2(coords[:, 0] - center_y, coords[:, 1] - center_x),
        2.0 * math.pi,
    )
    sectors = np.floor(angles * sector_count / (2.0 * math.pi)).astype(int)
    supported = 0
    observed = 0
    covered_values = covered[coords[:, 0], coords[:, 1]]
    for sector in range(sector_count):
        selected = sectors == sector
        if not np.any(selected):
            continue
        observed += 1
        supported += int(float(covered_values[selected].mean()) >= sector_support_fraction)
    return float(supported) / max(observed, 1)

def evaluate_nucleus_object_3d(
    nucleus_volume: np.ndarray,
    local_core: np.ndarray,
    local_dapi: np.ndarray,
    local_structural: np.ndarray,
    voxel_sampling: tuple[float, float, float],
    dapi_dynamic: float,
    dapi_min_dynamic: float,
    low_threshold: float,
    high_threshold: float,
    cfg: Neonatal3DConfig,
    projection_overlap_denominator: int,
) -> dict:
    voxel_volume_um3 = float(np.prod(voxel_sampling))
    volume_px = int(nucleus_volume.sum())
    volume_um3 = volume_px * voxel_volume_um3
    z_present = np.any(nucleus_volume, axis=(1, 2))
    z_indices = np.flatnonzero(z_present)
    z_span_um = (
        (int(z_indices[-1]) - int(z_indices[0]) + 1) * voxel_sampling[0]
        if z_indices.size
        else 0.0
    )
    projection = np.any(nucleus_volume, axis=0)
    projection_overlap = float((projection & local_core).sum()) / max(
        int(projection_overlap_denominator),
        1,
    )
    dapi_valid = bool(
        dapi_dynamic >= dapi_min_dynamic
        and volume_um3 >= cfg.dapi_min_volume_um3
        and z_span_um >= cfg.dapi_min_z_span_um
        and projection_overlap >= cfg.dapi_min_projection_overlap
    )

    surface_coverage = 0.0
    median_boundary_coverage = 0.0
    median_angular_coverage = 0.0
    z_support_fraction = 0.0
    shell_enrichment = -1.0
    radial_band_fraction = 0.0
    enclosure_score = 0.0
    structural_threshold = math.nan
    if dapi_valid:
        smooth_sigma = tuple(
            max(0.0, cfg.egfp_smooth_um / spacing) for spacing in voxel_sampling
        )
        smooth_structural = ndi.gaussian_filter(
            local_structural,
            sigma=smooth_sigma,
            mode="nearest",
        )
        distance_from_nucleus = ndi.distance_transform_edt(
            ~nucleus_volume,
            sampling=voxel_sampling,
        )
        shell = (
            (distance_from_nucleus >= cfg.shell_inner_um)
            & (distance_from_nucleus <= cfg.shell_outer_um)
            & ~nucleus_volume
        )
        background_shell = (
            (distance_from_nucleus >= cfg.background_inner_um)
            & (distance_from_nucleus <= cfg.background_outer_um)
        )
        background_values = smooth_structural[background_shell]
        if background_values.size < 32:
            background_values = smooth_structural[~nucleus_volume]
        background_median = float(np.percentile(background_values, 50.0))
        background_mad = float(
            1.4826 * np.median(np.abs(background_values - background_median))
        )
        local_high = float(np.percentile(smooth_structural, 99.0))
        structural_numeric_floor = (
            np.finfo(np.float32).eps
            * max(
                float(np.max(np.abs(smooth_structural))),
                float(np.finfo(np.float32).tiny),
            )
            * 16.0
        )
        structural_threshold = background_median + max(
            structural_numeric_floor,
            1.5 * background_mad,
            0.12 * max(local_high - background_median, 0.0),
        )
        positive = (smooth_structural > structural_threshold) & ~nucleus_volume
        if positive.any():
            distance_to_positive = ndi.distance_transform_edt(
                ~positive,
                sampling=voxel_sampling,
            )
        else:
            distance_to_positive = np.full(
                nucleus_volume.shape,
                np.inf,
                dtype=np.float32,
            )
        surface = nucleus_volume & ~ndi.binary_erosion(
            nucleus_volume,
            structure=ndi.generate_binary_structure(3, 1),
            border_value=0,
        )
        surface_coverage = (
            float((distance_to_positive[surface] <= cfg.surface_contact_um).mean())
            if surface.any()
            else 0.0
        )

        shell_values = smooth_structural[shell]
        shell_p75 = float(np.percentile(shell_values, 75.0)) if shell_values.size else 0.0
        background_p75 = float(np.percentile(background_values, 75.0))
        local_dynamic = max(
            local_high - background_median,
            structural_numeric_floor,
        )
        shell_enrichment = float(
            np.clip((shell_p75 - background_p75) / local_dynamic, -1.0, 1.0)
        )
        radial_support: list[bool] = []
        radial_edges = np.linspace(cfg.shell_inner_um, cfg.shell_outer_um, 4)
        for inner, outer in zip(radial_edges[:-1], radial_edges[1:]):
            band = (
                (distance_from_nucleus >= inner)
                & (distance_from_nucleus < outer)
                & ~nucleus_volume
            )
            radial_support.append(
                bool(band.any())
                and float(positive[band].mean()) >= 0.08
                and float(np.percentile(smooth_structural[band], 70.0))
                >= structural_threshold
            )
        radial_band_fraction = float(np.mean(radial_support)) if radial_support else 0.0

        areas = nucleus_volume.sum(axis=(1, 2))
        central = z_present & (
            areas >= cfg.central_slice_area_fraction * max(int(areas.max()), 1)
        )
        boundary_coverages: list[float] = []
        angular_coverages: list[float] = []
        supported_slices = np.zeros(nucleus_volume.shape[0], dtype=bool)
        for z_index in np.flatnonzero(central):
            plane = nucleus_volume[z_index]
            boundary = plane & ~morphology.binary_erosion(
                plane,
                footprint=morphology.disk(1),
            )
            if not boundary.any():
                continue
            positive_plane = positive[z_index]
            if positive_plane.any():
                distance_2d = ndi.distance_transform_edt(
                    ~positive_plane,
                    sampling=voxel_sampling[1:],
                )
                covered = boundary & (distance_2d <= cfg.surface_contact_um)
            else:
                covered = np.zeros_like(boundary, dtype=bool)
            boundary_coverage = float(covered.sum()) / max(int(boundary.sum()), 1)
            angular_coverage = neonatal_3d_slice_angular_coverage(
                boundary,
                covered,
                cfg.angular_sector_count,
                cfg.angular_sector_support_fraction,
            )
            boundary_coverages.append(boundary_coverage)
            angular_coverages.append(angular_coverage)
            supported_slices[z_index] = (
                boundary_coverage >= cfg.slice_support_threshold
                and angular_coverage >= cfg.min_angular_coverage
            )
        median_boundary_coverage = (
            float(np.median(boundary_coverages)) if boundary_coverages else 0.0
        )
        median_angular_coverage = (
            float(np.median(angular_coverages)) if angular_coverages else 0.0
        )
        z_support_fraction = float(longest_true_run(supported_slices)) / max(
            int(central.sum()),
            1,
        )

        surface_score = float(np.clip((surface_coverage - 0.20) / 0.55, 0.0, 1.0))
        angular_score = float(
            np.clip((median_angular_coverage - 0.20) / 0.65, 0.0, 1.0)
        )
        z_score = float(np.clip((z_support_fraction - 0.15) / 0.70, 0.0, 1.0))
        enrichment_score = float(
            np.clip((shell_enrichment + 0.02) / 0.35, 0.0, 1.0)
        )
        enclosure_score = (
            0.30 * surface_score
            + 0.25 * angular_score
            + 0.20 * z_score
            + 0.15 * enrichment_score
            + 0.10 * radial_band_fraction
        )

    criteria = {
        "surface": surface_coverage >= cfg.min_surface_coverage,
        "angular": median_angular_coverage >= cfg.min_angular_coverage,
        "z_continuity": z_support_fraction >= cfg.min_z_support_fraction,
        "shell_enrichment": shell_enrichment >= cfg.min_shell_enrichment,
    }
    accepted = bool(
        dapi_valid
        and enclosure_score >= cfg.min_enclosure_score
        and criteria["surface"]
        and criteria["angular"]
        and criteria["z_continuity"]
        and sum(criteria.values()) >= 3
    )
    failed: list[str] = []
    if not dapi_valid:
        failed.append("invalid_3d_dapi_object")
    if enclosure_score < cfg.min_enclosure_score:
        failed.append("low_enclosure_score")
    failed.extend(name for name, passed in criteria.items() if not passed)
    coords = np.argwhere(nucleus_volume)
    center_z, center_y, center_x = (
        coords.mean(axis=0) if coords.size else np.asarray([math.nan, math.nan, math.nan])
    )
    return {
        "accepted": accepted,
        "dapi_valid": dapi_valid,
        "reason": "accepted" if accepted else ",".join(dict.fromkeys(failed)),
        "center_z_local": float(center_z),
        "center_y_local": float(center_y),
        "center_x_local": float(center_x),
        "dapi_volume_um3": float(volume_um3),
        "dapi_z_span_um": float(z_span_um),
        "dapi_projection_overlap": float(projection_overlap),
        "surface_coverage": float(surface_coverage),
        "median_xy_boundary_coverage": float(median_boundary_coverage),
        "angular_coverage": float(median_angular_coverage),
        "z_support_fraction": float(z_support_fraction),
        "shell_enrichment": float(shell_enrichment),
        "radial_band_fraction": float(radial_band_fraction),
        "enclosure_score": float(enclosure_score),
        "dapi_low_threshold": float(low_threshold),
        "dapi_high_threshold": float(high_threshold),
        "structural_threshold": (
            float(structural_threshold) if np.isfinite(structural_threshold) else None
        ),
        "projection": projection,
    }

def evaluate_raw_inventory_object_3d(
    local_volume_id: int,
    volume_labels: np.ndarray,
    volume_count: int,
    local_core: np.ndarray,
    local_dapi: np.ndarray,
    local_structural: np.ndarray,
    extent_component_labels_crop: np.ndarray,
    voxel_sampling: tuple[float, float, float],
    dapi_dynamic: float,
    dapi_min_dynamic: float,
    low_threshold: float,
    high_threshold: float,
    cfg: Neonatal3DConfig,
) -> dict:
    """Evaluate one raw 3D DAPI object without assigning global IDs or labels."""

    object_volume = volume_labels == int(local_volume_id)
    object_projection = np.any(object_volume, axis=0)
    overlap_px = int((object_projection & local_core).sum())
    denominator = (
        int(local_core.sum())
        if int(volume_count) == 1
        else min(int(local_core.sum()), int(object_projection.sum()))
    )
    evaluated = evaluate_nucleus_object_3d(
        object_volume,
        local_core,
        local_dapi,
        local_structural,
        voxel_sampling,
        dapi_dynamic,
        dapi_min_dynamic,
        low_threshold,
        high_threshold,
        cfg,
        denominator,
    )
    z_coordinates = np.flatnonzero(np.any(object_volume, axis=(1, 2)))
    extent_component_values = extent_component_labels_crop[object_projection]
    extent_component_values = extent_component_values[
        extent_component_values > 0
    ]
    extent_component_id = (
        int(np.bincount(extent_component_values).argmax())
        if extent_component_values.size
        else 0
    )
    peak_map = np.max(
        np.where(object_volume, local_dapi, -np.inf),
        axis=0,
    )
    return {
        "local_volume_id": int(local_volume_id),
        "projection": object_projection,
        "overlap_px": overlap_px,
        "evaluated": evaluated,
        "z_coordinates": z_coordinates,
        "extent_component_id": extent_component_id,
        "peak_map": peak_map,
    }

def dapi_parent_fragment_workload(
    *,
    parent_core_id: int,
    parent_index_1based: int,
    bbox_yx_0based: tuple[int, int, int, int],
    volume_labels: np.ndarray,
    fragment_count: int,
) -> DapiParentFragmentWorkload:
    crop_shape = tuple(int(value) for value in volume_labels.shape)
    crop_voxels = int(volume_labels.size)
    crop_xy_pixels = int(crop_shape[1] * crop_shape[2])
    fragments = int(fragment_count)
    return DapiParentFragmentWorkload(
        parent_core_id=int(parent_core_id),
        parent_index_1based=int(parent_index_1based),
        bbox_yx_0based=tuple(int(value) for value in bbox_yx_0based),
        crop_shape_zyx=crop_shape,
        crop_voxels=crop_voxels,
        crop_xy_pixels=crop_xy_pixels,
        fragment_count=fragments,
        estimated_voxel_comparisons=int(fragments * crop_voxels),
        estimated_result_payload_bytes_lower_bound=int(
            fragments
            * crop_xy_pixels
            * (
                np.dtype(np.bool_).itemsize
                + np.dtype(np.float32).itemsize
            )
        ),
    )

def append_dapi_parent_fragment_workload(
    summary: dict[str, object],
    workload: DapiParentFragmentWorkload,
) -> dict[str, object]:
    record = asdict(workload)
    parent_records = summary["parent_records"]
    assert isinstance(parent_records, list)
    parent_records.append(record)
    summary["parents_linked_to_whole"] = int(
        summary["parents_linked_to_whole"]
    ) + 1
    summary["total_fragments"] = int(summary["total_fragments"]) + int(
        workload.fragment_count
    )
    summary["total_voxel_comparisons"] = int(
        summary["total_voxel_comparisons"]
    ) + int(workload.estimated_voxel_comparisons)
    summary["max_parent_fragments"] = max(
        int(summary["max_parent_fragments"]),
        int(workload.fragment_count),
    )
    summary["max_parent_voxel_comparisons"] = max(
        int(summary["max_parent_voxel_comparisons"]),
        int(workload.estimated_voxel_comparisons),
    )
    summary["max_parent_result_payload_bytes_lower_bound"] = max(
        int(summary["max_parent_result_payload_bytes_lower_bound"]),
        int(workload.estimated_result_payload_bytes_lower_bound),
    )
    return record

def dapi_fragment_workload_violation(
    summary: dict[str, object],
    parent: DapiParentFragmentWorkload,
    limits: DapiFragmentWorkloadLimits,
) -> tuple[str, int, int] | None:
    checks = (
        (
            "parent_result_payload_bytes_lower_bound",
            int(parent.estimated_result_payload_bytes_lower_bound),
            int(limits.max_parent_result_payload_bytes_lower_bound),
        ),
        (
            "parent_voxel_comparisons",
            int(parent.estimated_voxel_comparisons),
            int(limits.max_parent_voxel_comparisons),
        ),
        (
            "parent_fragments",
            int(parent.fragment_count),
            int(limits.max_parent_fragments),
        ),
        (
            "total_voxel_comparisons",
            int(summary["total_voxel_comparisons"]),
            int(limits.max_total_voxel_comparisons),
        ),
        (
            "total_fragments",
            int(summary["total_fragments"]),
            int(limits.max_total_fragments),
        ),
    )
    for metric, observed, limit in checks:
        if observed > limit:
            return metric, observed, limit
    return None

def atomic_write_dapi_fragment_workload_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        resolved.parent
        / f"temporary_{resolved.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)

def bounded_ordered_map(
    executor: ThreadPoolExecutor,
    function,
    arguments,
    *,
    max_pending: int,
    cancel_event: threading.Event,
    heartbeat_seconds: float,
    progress_callback=None,
) -> list:
    """Evaluate a bounded number of Futures while yielding input order."""

    if max_pending <= 0:
        raise ValueError("max_pending must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    indexed_arguments = iter(enumerate(arguments))
    pending: dict[object, int] = {}
    completed: dict[int, object] = {}
    results: list[object] = []
    next_yield_index = 0
    submitted_count = 0

    def fill_window() -> None:
        nonlocal submitted_count
        while len(pending) + len(completed) < max_pending:
            if cancel_event.is_set():
                return
            try:
                index, call_arguments = next(indexed_arguments)
            except StopIteration:
                return
            future = executor.submit(function, *call_arguments)
            pending[future] = int(index)
            submitted_count += 1

    fill_window()
    try:
        while pending:
            if cancel_event.is_set():
                raise RuntimeError("DAPI fragment evaluation was cancelled")
            done, _ = wait(
                tuple(pending),
                timeout=float(heartbeat_seconds),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "heartbeat",
                            "submitted": int(submitted_count),
                            "yielded": int(next_yield_index),
                            "pending": int(len(pending)),
                        }
                    )
                continue
            for future in done:
                index = pending.pop(future)
                completed[int(index)] = future.result()
            while next_yield_index in completed:
                results.append(completed.pop(next_yield_index))
                next_yield_index += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "completed",
                            "submitted": int(submitted_count),
                            "yielded": int(next_yield_index),
                            "pending": int(len(pending)),
                        }
                    )
            fill_window()
    except BaseException:
        cancel_event.set()
        for future in pending:
            future.cancel()
        raise
    if completed:
        raise RuntimeError(
            "DAPI fragment scheduler retained out-of-order results"
        )
    return results

def separable_physical_binary_closing(
    mask: np.ndarray,
    radii_um: tuple[float, float, float],
    sampling: tuple[float, float, float],
) -> np.ndarray:
    """Close small intranuclear gaps with three bounded one-dimensional passes."""

    output = np.asarray(mask, dtype=bool)
    for axis, (radius_um, spacing_um) in enumerate(zip(radii_um, sampling)):
        radius_px = max(1, int(math.ceil(radius_um / spacing_um)))
        shape = [1, 1, 1]
        shape[axis] = 2 * radius_px + 1
        output = ndi.binary_closing(output, structure=np.ones(shape, dtype=bool))
    return output

def resolve_connected_nuclear_envelope(
    envelope: np.ndarray,
    voxel_sampling: tuple[float, float, float],
    config: Neonatal3DConfig,
) -> tuple[list[np.ndarray], str, dict]:
    """Resolve one connected DAPI envelope by shape basins, never intensity peaks."""

    distance_um = ndi.distance_transform_edt(envelope, sampling=voxel_sampling)
    maxima = morphology.h_maxima(distance_um, config.canonical_peak_h_um)
    maxima &= envelope & (distance_um >= config.canonical_min_peak_radius_um)
    maxima_labels = measure.label(maxima, connectivity=3)
    peak_rows: list[tuple[float, tuple[int, int, int]]] = []
    for prop in measure.regionprops(maxima_labels, intensity_image=distance_um):
        coordinates = prop.coords
        values = distance_um[tuple(coordinates.T)]
        best = coordinates[int(np.argmax(values))]
        peak_rows.append(
            (
                float(values.max()),
                (int(best[0]), int(best[1]), int(best[2])),
            )
        )
    selected: list[tuple[float, tuple[int, int, int]]] = []
    for row in sorted(peak_rows, reverse=True):
        coordinate = row[1]
        if any(
            math.sqrt(
                sum(
                    ((coordinate[axis] - other[1][axis]) * voxel_sampling[axis]) ** 2
                    for axis in range(3)
                )
            )
            < config.canonical_min_peak_separation_um
            for other in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= config.canonical_max_instances_per_envelope:
            break
    if len(selected) < 2:
        return [envelope], "single", {
            "shape_peak_count": len(selected),
            "split_accepted": False,
            "neck_peak_ratio": None,
        }

    markers = np.zeros(envelope.shape, dtype=np.int32)
    for marker_id, (_peak, coordinate) in enumerate(selected, start=1):
        markers[coordinate] = marker_id
    partition = segmentation.watershed(
        -distance_um,
        markers=markers,
        mask=envelope,
        connectivity=np.ones((3, 3, 3), dtype=bool),
        watershed_line=False,
    )
    voxel_volume_um3 = float(np.prod(voxel_sampling))
    children = [partition == marker_id for marker_id in range(1, len(selected) + 1)]
    child_volumes = [int(child.sum()) * voxel_volume_um3 for child in children]
    child_spans = [
        int(np.any(child, axis=(1, 2)).sum()) * voxel_sampling[0]
        for child in children
    ]
    boundary = np.zeros(envelope.shape, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        different = (
            (partition[tuple(left)] > 0)
            & (partition[tuple(right)] > 0)
            & (partition[tuple(left)] != partition[tuple(right)])
        )
        boundary[tuple(left)] |= different
        boundary[tuple(right)] |= different
    minimum_peak = min(row[0] for row in selected)
    neck_peak_ratio = (
        float(np.percentile(distance_um[boundary], 75.0)) / max(minimum_peak, 1e-9)
        if boundary.any()
        else math.inf
    )
    quality_passed = bool(
        all(
            volume >= config.canonical_min_child_volume_um3
            for volume in child_volumes
        )
        and all(
            span >= config.canonical_min_child_z_span_um for span in child_spans
        )
        and neck_peak_ratio <= config.canonical_max_neck_peak_ratio
    )
    diagnostics = {
        "shape_peak_count": len(selected),
        "split_accepted": quality_passed,
        "child_volumes_um3": child_volumes,
        "child_z_spans_um": child_spans,
        "neck_peak_ratio": float(neck_peak_ratio),
    }
    if quality_passed:
        ordered = sorted(
            children,
            key=lambda child: tuple(np.argwhere(child).mean(axis=0)),
        )
        return ordered, "connected_object_split", diagnostics
    return [envelope], "ambiguous", diagnostics

def resolve_2d_nuclear_extent_families(
    nuclei_extent: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    config: Neonatal3DConfig,
) -> np.ndarray:
    """Partition projection-connected DAPI extent into bounded shape-basin families."""

    connected_labels = measure.label(nuclei_extent, connectivity=2)
    output = np.zeros(nuclei_extent.shape, dtype=np.uint32)
    next_id = 1
    minimum_area_px = max(
        12,
        int(
            math.ceil(
                math.pi * config.canonical_min_peak_radius_um**2
                / (pixel_width_um * pixel_height_um)
            )
        ),
    )
    for prop in measure.regionprops(connected_labels):
        min_row, min_col, max_row, max_col = prop.bbox
        crop = np.s_[min_row:max_row, min_col:max_col]
        component = connected_labels[crop] == int(prop.label)
        distance_um = ndi.distance_transform_edt(
            component,
            sampling=(pixel_height_um, pixel_width_um),
        )
        maxima = morphology.h_maxima(distance_um, config.canonical_peak_h_um)
        maxima &= component & (distance_um >= config.canonical_min_peak_radius_um)
        maxima_labels = measure.label(maxima, connectivity=2)
        peaks: list[tuple[float, tuple[int, int]]] = []
        for maximum in measure.regionprops(maxima_labels, intensity_image=distance_um):
            coordinates = maximum.coords
            values = distance_um[tuple(coordinates.T)]
            best = coordinates[int(np.argmax(values))]
            peaks.append((float(values.max()), (int(best[0]), int(best[1]))))
        selected: list[tuple[float, tuple[int, int]]] = []
        for peak in sorted(peaks, reverse=True):
            coordinate = peak[1]
            if any(
                math.hypot(
                    (coordinate[0] - other[1][0]) * pixel_height_um,
                    (coordinate[1] - other[1][1]) * pixel_width_um,
                )
                < config.canonical_min_peak_separation_um
                for other in selected
            ):
                continue
            selected.append(peak)
        if len(selected) <= 1:
            local_output = output[crop]
            local_output[component] = next_id
            next_id += 1
            continue
        markers = np.zeros(component.shape, dtype=np.int32)
        for marker_id, (_value, coordinate) in enumerate(selected, start=1):
            markers[coordinate] = marker_id
        partition = segmentation.watershed(
            -distance_um,
            markers=markers,
            mask=component,
            watershed_line=False,
            connectivity=np.ones((3, 3), dtype=bool),
        )
        child_ids = [
            child_id
            for child_id in range(1, len(selected) + 1)
            if int((partition == child_id).sum()) >= minimum_area_px
        ]
        if len(child_ids) <= 1:
            local_output = output[crop]
            local_output[component] = next_id
            next_id += 1
            continue
        retained_markers = np.zeros(component.shape, dtype=np.int32)
        for marker_id, child_id in enumerate(child_ids, start=1):
            coordinate = selected[child_id - 1][1]
            retained_markers[coordinate] = marker_id
        partition = segmentation.watershed(
            -distance_um,
            markers=retained_markers,
            mask=component,
            watershed_line=False,
            connectivity=np.ones((3, 3), dtype=bool),
        )
        local_output = output[crop]
        for child_id in range(1, len(child_ids) + 1):
            local_output[partition == child_id] = next_id
            next_id += 1
    return output

def evaluate_canonical_extent_family_3d(
    family_id: int,
    bbox: tuple[int, int, int, int],
    extent_labels: np.ndarray,
    nuclei_core: np.ndarray,
    distance_to_whole: np.ndarray,
    dapi_substack: np.ndarray,
    structural_substack: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    voxel_sampling: tuple[float, float, float],
    pad_y: int,
    pad_x: int,
    minimum_component_voxels: int,
    config: Neonatal3DConfig,
) -> dict | None:
    """Compute one canonical DAPI family without assigning global instance IDs."""

    min_row, min_col, max_row, max_col = bbox
    row0 = max(0, min_row - pad_y)
    col0 = max(0, min_col - pad_x)
    row1 = min(extent_labels.shape[0], max_row + pad_y)
    col1 = min(extent_labels.shape[1], max_col + pad_x)
    crop = np.s_[row0:row1, col0:col1]
    local_family = extent_labels[crop] == int(family_id)
    if (
        float(distance_to_whole[crop][local_family].min(initial=math.inf))
        > config.candidate_link_um
    ):
        return None
    local_core = nuclei_core[crop] & local_family
    if not local_core.any():
        return None
    support_distance = ndi.distance_transform_edt(
        ~local_family,
        sampling=(pixel_height_um, pixel_width_um),
    )
    support = support_distance <= config.dapi_xy_support_margin_um
    local_dapi = dapi_substack[:, row0:row1, col0:col1].astype(
        np.float32,
        copy=False,
    )
    local_structural = structural_substack[:, row0:row1, col0:col1].astype(
        np.float32,
        copy=False,
    )
    ring = support & (
        support_distance >= 0.50 * config.dapi_xy_support_margin_um
    )
    if not ring.any():
        ring = ~support
    background_values = local_dapi[:, ring] if ring.any() else local_dapi.ravel()
    dapi_background = float(np.percentile(background_values, 50.0))
    dapi_peak = float(np.percentile(local_dapi[:, local_core], 99.0))
    dapi_dynamic = max(dapi_peak - dapi_background, 0.0)
    field_dynamic = max(
        float(np.percentile(local_dapi, 99.9))
        - float(np.percentile(local_dapi, 0.1)),
        0.0,
    )
    numeric_floor = (
        np.finfo(np.float32).eps
        * max(
            float(np.max(np.abs(local_dapi))),
            float(np.finfo(np.float32).tiny),
        )
        * 16.0
    )
    minimum_dynamic = max(numeric_floor, 0.04 * field_dynamic)
    low_threshold = dapi_background + config.dapi_low_fraction * dapi_dynamic
    high_threshold = dapi_background + config.dapi_high_fraction * dapi_dynamic
    envelope = (local_dapi >= low_threshold) & support[None, :, :]
    envelope = separable_physical_binary_closing(
        envelope,
        (
            config.canonical_envelope_closing_z_um,
            config.canonical_envelope_closing_xy_um,
            config.canonical_envelope_closing_xy_um,
        ),
        voxel_sampling,
    )
    envelope = ndi.binary_fill_holes(envelope)
    envelope = morphology.remove_small_objects(
        envelope,
        min_size=minimum_component_voxels,
    )
    connected_labels = measure.label(envelope, connectivity=3)
    family_instances: list[tuple[np.ndarray, str, dict]] = []
    split_envelope_count = 0
    ambiguous_count = 0
    for connected_id in range(1, int(connected_labels.max()) + 1):
        connected = connected_labels == connected_id
        if int(connected.sum()) < minimum_component_voxels:
            continue
        children, resolution, diagnostics = resolve_connected_nuclear_envelope(
            connected,
            voxel_sampling,
            config,
        )
        split_envelope_count += int(resolution == "connected_object_split")
        ambiguous_count += int(resolution == "ambiguous")
        family_instances.extend(
            (child, resolution, diagnostics) for child in children
        )

    family_instances.sort(
        key=lambda item: tuple(np.argwhere(item[0]).mean(axis=0))
    )
    evaluated_instances: list[dict] = []
    for instance_volume, resolution, diagnostics in family_instances:
        projection = np.any(instance_volume, axis=0) & local_family
        if not projection.any():
            continue
        distance_um = ndi.distance_transform_edt(
            instance_volume,
            sampling=voxel_sampling,
        )
        peak_radius = float(distance_um.max(initial=0.0))
        interior = instance_volume & (
            distance_um
            >= config.canonical_core_radius_fraction
            * max(peak_radius, 1e-9)
        )
        core_projection = np.any(interior, axis=0) & projection
        if not core_projection.any():
            projection_coordinates = np.argwhere(projection)
            projection_center = projection_coordinates.mean(axis=0)
            nearest = projection_coordinates[
                int(
                    np.argmin(
                        np.square(
                            projection_coordinates - projection_center
                        ).sum(axis=1)
                    )
                )
            ]
            core_projection[int(nearest[0]), int(nearest[1])] = True
        evaluated = evaluate_nucleus_object_3d(
            instance_volume,
            projection,
            local_dapi,
            local_structural,
            voxel_sampling,
            dapi_dynamic,
            minimum_dynamic,
            low_threshold,
            high_threshold,
            config,
            int(projection.sum()),
        )
        coordinates = np.argwhere(instance_volume)
        z_coordinates = np.flatnonzero(
            np.any(instance_volume, axis=(1, 2))
        )
        evaluated_instances.append(
            {
                "resolution": resolution,
                "diagnostics": diagnostics,
                "projection": projection,
                "core_projection": core_projection,
                "evaluated": evaluated,
                "center_z_local": float(coordinates[:, 0].mean()),
                "center_y_local": float(coordinates[:, 1].mean()),
                "center_x_local": float(coordinates[:, 2].mean()),
                "z_min_local": int(z_coordinates.min()),
                "z_max_local": int(z_coordinates.max()),
                "volume_px": int(instance_volume.sum()),
            }
        )
    return {
        "family_id": int(family_id),
        "row0": int(row0),
        "col0": int(col0),
        "row1": int(row1),
        "col1": int(col1),
        "split_envelope_count": int(split_envelope_count),
        "ambiguous_count": int(ambiguous_count),
        "instances": evaluated_instances,
    }

def resolve_canonical_nucleus_instances_3d(
    frozen_whole_mask: np.ndarray,
    nuclei_core: np.ndarray,
    nuclei_extent: np.ndarray,
    context: Neonatal3DContext,
    pixel_width_um: float,
    pixel_height_um: float,
    config: Neonatal3DConfig,
    raw_object_extent_labels: np.ndarray | None = None,
    max_workers: int = 1,
) -> CanonicalNucleusResolution:
    """Resolve heterogeneous DAPI into one canonical identity per 3D nuclear envelope."""

    z0 = int(context.z_start_0based)
    z1 = int(context.z_end_0based_inclusive)
    dapi_substack = context.dapi_stack[z0 : z1 + 1]
    structural_substack = context.egfp_stack[z0 : z1 + 1]
    voxel_sampling = (
        float(context.pixel_depth_um),
        float(pixel_height_um),
        float(pixel_width_um),
    )
    voxel_volume_um3 = float(np.prod(voxel_sampling))
    extent_labels = resolve_2d_nuclear_extent_families(
        nuclei_extent,
        pixel_width_um,
        pixel_height_um,
        config,
    )
    distance_to_whole = ndi.distance_transform_edt(
        ~frozen_whole_mask.astype(bool),
        sampling=(pixel_height_um, pixel_width_um),
    )
    pad_y = max(2, int(math.ceil(config.canonical_crop_margin_um / pixel_height_um)))
    pad_x = max(2, int(math.ceil(config.canonical_crop_margin_um / pixel_width_um)))
    core_output = np.zeros(nuclei_extent.shape, dtype=np.uint32)
    extent_output = np.zeros(nuclei_extent.shape, dtype=np.uint32)
    records: list[dict] = []
    next_instance_id = 1
    ambiguous_count = 0
    split_envelope_count = 0

    minimum_component_voxels = max(
        1,
        int(math.ceil(config.dapi_min_volume_um3 / voxel_volume_um3)),
    )

    family_arguments = [
        (
            int(prop.label),
            tuple(int(value) for value in prop.bbox),
            extent_labels,
            nuclei_core,
            distance_to_whole,
            dapi_substack,
            structural_substack,
            pixel_width_um,
            pixel_height_um,
            voxel_sampling,
            pad_y,
            pad_x,
            minimum_component_voxels,
            config,
        )
        for prop in measure.regionprops(extent_labels)
    ]
    worker_count = max(1, min(int(max_workers), 12))
    if worker_count == 1:
        family_results = [
            evaluate_canonical_extent_family_3d(*arguments)
            for arguments in family_arguments
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=min(worker_count, max(len(family_arguments), 1)),
            thread_name_prefix="ihc-dapi-canonical",
        ) as canonical_executor:
            canonical_futures = [
                canonical_executor.submit(
                    evaluate_canonical_extent_family_3d,
                    *arguments,
                )
                for arguments in family_arguments
            ]
            family_results = [
                future.result() for future in canonical_futures
            ]

    for family_result in family_results:
        if family_result is None:
            continue
        family_id = int(family_result["family_id"])
        row0 = int(family_result["row0"])
        col0 = int(family_result["col0"])
        row1 = int(family_result["row1"])
        col1 = int(family_result["col1"])
        crop = np.s_[row0:row1, col0:col1]
        split_envelope_count += int(
            family_result["split_envelope_count"]
        )
        ambiguous_count += int(family_result["ambiguous_count"])
        for instance_result in family_result["instances"]:
            resolution = str(instance_result["resolution"])
            diagnostics = instance_result["diagnostics"]
            projection = instance_result["projection"]
            core_projection = instance_result["core_projection"]
            evaluated = instance_result["evaluated"]
            instance_id = next_instance_id
            next_instance_id += 1
            core_view = core_output[crop]
            extent_view = extent_output[crop]
            unclaimed_extent = projection & (extent_view == 0)
            unclaimed_core = core_projection & (core_view == 0)
            extent_view[unclaimed_extent] = instance_id
            core_view[unclaimed_core] = instance_id
            extent_view[unclaimed_core] = instance_id
            claimed_projection = unclaimed_extent | unclaimed_core
            source_object_ids: tuple[int, ...] = ()
            if raw_object_extent_labels is not None:
                source_object_ids = tuple(
                    sorted(
                        int(value)
                        for value in np.unique(
                            raw_object_extent_labels[crop][claimed_projection]
                        )
                        if int(value) > 0
                    )
                )
            identity_status = "ambiguous" if resolution == "ambiguous" else "resolved"
            accepted = bool(evaluated["accepted"] and identity_status == "resolved")
            records.append(
                {
                    "instance_id": instance_id,
                    "object_id": instance_id,
                    "nucleus_id_2d": instance_id,
                    "object_id_3d": instance_id,
                    "source_object_ids": source_object_ids,
                    "extent_component_2d_id": family_id,
                    "resolution": resolution,
                    "identity_status": identity_status,
                    "accepted": accepted,
                    "dapi_valid": bool(evaluated["dapi_valid"]),
                    "center_z": float(
                        z0 + instance_result["center_z_local"]
                    ),
                    "center_y": float(
                        row0 + instance_result["center_y_local"]
                    ),
                    "center_x": float(
                        col0 + instance_result["center_x_local"]
                    ),
                    "z_min_0based": int(
                        z0 + instance_result["z_min_local"]
                    ),
                    "z_max_0based_inclusive": int(
                        z0 + instance_result["z_max_local"]
                    ),
                    "volume_um3": float(
                        int(instance_result["volume_px"])
                        * voxel_volume_um3
                    ),
                    "projection_area_px": int(claimed_projection.sum()),
                    "extent_area_px": int(claimed_projection.sum()),
                    "enclosure_score": float(evaluated["enclosure_score"]),
                    "dapi_low_threshold": float(
                        evaluated["dapi_low_threshold"]
                    ),
                    "dapi_high_threshold": float(
                        evaluated["dapi_high_threshold"]
                    ),
                    "dapi_z_span_um": float(evaluated["dapi_z_span_um"]),
                    "dapi_projection_overlap": float(evaluated["dapi_projection_overlap"]),
                    "surface_coverage": float(evaluated["surface_coverage"]),
                    "median_xy_boundary_coverage": float(
                        evaluated["median_xy_boundary_coverage"]
                    ),
                    "angular_coverage": float(evaluated["angular_coverage"]),
                    "z_support_fraction": float(evaluated["z_support_fraction"]),
                    "shell_enrichment": float(evaluated["shell_enrichment"]),
                    "radial_band_fraction": float(evaluated["radial_band_fraction"]),
                    "reason": (
                        "ambiguous_nuclear_envelope"
                        if identity_status == "ambiguous"
                        else str(evaluated["reason"])
                    ),
                    "resolution_diagnostics": diagnostics,
                }
            )

    accepted_ids = tuple(
        int(row["instance_id"]) for row in records if bool(row["accepted"])
    )
    ambiguous_ids = tuple(
        int(row["instance_id"])
        for row in records
        if row["identity_status"] == "ambiguous"
    )
    return CanonicalNucleusResolution(
        core_labels_2d=core_output,
        extent_labels_2d=extent_output,
        records=tuple(records),
        accepted_ids=accepted_ids,
        ambiguous_ids=ambiguous_ids,
        metrics={
            "method": (
                "canonical 3D nuclear envelope resolution by anisotropic distance basins; "
                "DAPI intensity peaks are not identity markers"
            ),
            "instance_count": len(records),
            "accepted_instance_count": len(accepted_ids),
            "ambiguous_instance_count": len(ambiguous_ids),
            "connected_envelope_split_count": split_envelope_count,
            "source_extent_family_count": int(extent_labels.max()),
        },
    )

def build_dapi_object_inventory_3d(
    frozen_whole_mask: np.ndarray,
    dapi_projection: np.ndarray,
    context: Neonatal3DContext,
    pixel_width_um: float,
    pixel_height_um: float,
    compartment_config: CompartmentConfig,
    validation_config: Neonatal3DConfig | None = None,
    max_workers: int = 1,
    preflight_only: bool = False,
    workload_limits: DapiFragmentWorkloadLimits | None = None,
    workload_diagnostic_path: Path | None = None,
) -> ValidatedNucleusAnchors:
    """Preserve and independently validate every reconstructed 3D DAPI object."""

    inventory_started = time.perf_counter()
    cfg = validation_config or Neonatal3DConfig()
    if context.pixel_depth_um <= 0:
        raise ValueError("Positive Z calibration is required for 3D nucleus inventory")
    z0 = int(context.z_start_0based)
    z1 = int(context.z_end_0based_inclusive)
    if z0 < 0 or z1 < z0 or z1 >= context.dapi_stack.shape[0]:
        raise ValueError(f"Invalid 3D nucleus inventory Z range: {z0}-{z1}")
    if context.dapi_stack.shape != context.egfp_stack.shape:
        raise ValueError("DAPI and structural stacks must have identical shapes")
    limits = (
        workload_limits
        if workload_limits is not None
        else DAPI_FRAGMENT_WORKLOAD_LIMITS
    )
    workload_summary: dict[str, object] = {
        "schema_version": 1,
        "mode": "preflight_only" if preflight_only else "enforce",
        "status": "running",
        "policy_version": (
            limits.policy_version if limits is not None else "calibration_only"
        ),
        "z_start_1based": int(z0 + 1),
        "z_end_1based_inclusive": int(z1 + 1),
        "parents_seen": 0,
        "parents_linked_to_whole": 0,
        "total_fragments": 0,
        "total_voxel_comparisons": 0,
        "max_parent_fragments": 0,
        "max_parent_voxel_comparisons": 0,
        "max_parent_result_payload_bytes_lower_bound": 0,
        "guard_triggered": False,
        "guard_reason": None,
        "parent_records": [],
    }
    mean_pixel_um = math.sqrt(pixel_width_um * pixel_height_um)
    nuclei_core, nuclei_extent, _, dapi_2d_metrics = dapi_nuclei_core_and_extent(
        dapi_projection,
        mean_pixel_um,
        compartment_config,
    )
    core_labels = measure.label(nuclei_core, connectivity=2)
    extent_component_labels = measure.label(nuclei_extent, connectivity=2)
    object_core_labels = np.zeros(nuclei_core.shape, dtype=np.uint32)
    object_extent_labels = np.zeros(nuclei_core.shape, dtype=np.uint32)
    if int(core_labels.max()) == 0:
        workload_summary["status"] = (
            "preflight_completed" if preflight_only else "completed"
        )
        return ValidatedNucleusAnchors(
            np.zeros_like(nuclei_core, dtype=bool),
            np.zeros_like(nuclei_extent, dtype=bool),
            {
                "status": "completed_no_2d_dapi_candidates",
                "method": "object-preserving calibrated 3D DAPI inventory",
                "measurement_channel_used": False,
                "z_start_1based": z0 + 1,
                "z_end_1based_inclusive": z1 + 1,
                "voxel_size_um": [
                    round(context.pixel_depth_um, 9),
                    round(pixel_height_um, 9),
                    round(pixel_width_um, 9),
                ],
                "calibration_source": context.calibration_source,
                "structural_channel": context.structural_channel,
                "parent_2d_core_count": 0,
                "candidate_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "dapi_valid_count": 0,
                "multi_object_parent_core_count": 0,
                "per_nucleus": [],
                "config": asdict(cfg),
                "dapi_2d": dapi_2d_metrics,
                "dapi_fragment_workload": workload_summary,
            },
            object_core_labels,
            object_extent_labels,
        )

    core_distance, core_nearest_indices = ndi.distance_transform_edt(
        ~nuclei_core,
        sampling=(pixel_height_um, pixel_width_um),
        return_indices=True,
    )
    nearest_core_labels = core_labels[
        core_nearest_indices[0],
        core_nearest_indices[1],
    ]
    distance_to_whole = ndi.distance_transform_edt(
        ~frozen_whole_mask.astype(bool),
        sampling=(pixel_height_um, pixel_width_um),
    )
    dapi_substack = context.dapi_stack[z0 : z1 + 1]
    structural_substack = context.egfp_stack[z0 : z1 + 1]
    voxel_sampling = (
        float(context.pixel_depth_um),
        float(pixel_height_um),
        float(pixel_width_um),
    )
    pad_y = max(2, int(math.ceil(cfg.crop_margin_um / pixel_height_um)))
    pad_x = max(2, int(math.ceil(cfg.crop_margin_um / pixel_width_um)))
    per_object: list[dict] = []
    object_records: list[dict] = []
    next_object_id = 1
    multi_object_parent_core_count = 0
    worker_count = max(1, min(int(max_workers), 12))

    for parent_index, prop in enumerate(
        measure.regionprops(core_labels),
        start=1,
    ):
        workload_summary["parents_seen"] = int(
            workload_summary["parents_seen"]
        ) + 1
        parent_core_id = int(prop.label)
        min_row, min_col, max_row, max_col = prop.bbox
        row0 = max(0, min_row - pad_y)
        col0 = max(0, min_col - pad_x)
        row1 = min(core_labels.shape[0], max_row + pad_y)
        col1 = min(core_labels.shape[1], max_col + pad_x)
        crop = np.s_[row0:row1, col0:col1]
        local_core = core_labels[crop] == parent_core_id
        if (
            float(distance_to_whole[crop][local_core].min(initial=math.inf))
            > cfg.candidate_link_um
        ):
            continue
        local_extent = (
            nuclei_extent[crop]
            & (nearest_core_labels[crop] == parent_core_id)
            & (core_distance[crop] <= compartment_config.dapi_extent_max_expand_um)
        )
        local_extent |= local_core
        local_support_distance = ndi.distance_transform_edt(
            ~(local_core | local_extent),
            sampling=(pixel_height_um, pixel_width_um),
        )
        xy_support = local_support_distance <= cfg.dapi_xy_support_margin_um
        local_dapi = dapi_substack[:, row0:row1, col0:col1].astype(
            np.float32,
            copy=False,
        )
        local_structural = structural_substack[:, row0:row1, col0:col1].astype(
            np.float32,
            copy=False,
        )

        profile = np.percentile(local_dapi[:, local_core], 85.0, axis=1)
        profile_baseline = float(np.percentile(profile, 10.0))
        profile_peak = float(np.max(profile))
        profile_contrast = profile_peak - profile_baseline
        active = profile >= (
            profile_baseline + cfg.dapi_active_profile_fraction * max(profile_contrast, 0.0)
        )
        active[int(np.argmax(profile))] = True
        active = ndi.binary_closing(active, structure=np.ones(3, dtype=bool))

        support_ring = xy_support & (
            local_support_distance >= 0.50 * cfg.dapi_xy_support_margin_um
        )
        if not support_ring.any():
            support_ring = ~xy_support
        background_values = (
            local_dapi[:, support_ring] if support_ring.any() else local_dapi.ravel()
        )
        dapi_background = float(np.percentile(background_values, 50.0))
        dapi_peak = float(np.percentile(local_dapi[:, local_core], 99.0))
        dapi_dynamic = max(dapi_peak - dapi_background, 0.0)
        dapi_field_dynamic = max(
            float(np.percentile(local_dapi, 99.9))
            - float(np.percentile(local_dapi, 0.1)),
            0.0,
        )
        dapi_numeric_floor = (
            np.finfo(np.float32).eps
            * max(
                float(np.max(np.abs(local_dapi))),
                float(np.finfo(np.float32).tiny),
            )
            * 16.0
        )
        dapi_min_dynamic = max(dapi_numeric_floor, 0.04 * dapi_field_dynamic)
        low_threshold = dapi_background + cfg.dapi_low_fraction * dapi_dynamic
        high_threshold = dapi_background + cfg.dapi_high_fraction * dapi_dynamic
        low_domain = (
            (local_dapi >= low_threshold)
            & xy_support[None, :, :]
            & active[:, None, None]
        )
        seeds = (
            (local_dapi >= high_threshold)
            & local_core[None, :, :]
            & active[:, None, None]
        )
        if not seeds.any() and dapi_dynamic > 0:
            seed_values = np.where(
                local_core[None, :, :] & active[:, None, None],
                local_dapi,
                -np.inf,
            )
            seed = np.unravel_index(int(np.argmax(seed_values)), seed_values.shape)
            seeds[seed] = True
        nucleus_volume = (
            ndi.binary_propagation(
                seeds,
                structure=np.ones((3, 3, 3), dtype=bool),
                mask=low_domain,
            ).astype(bool)
            if seeds.any()
            else np.zeros_like(low_domain, dtype=bool)
        )
        volume_labels = measure.label(nucleus_volume, connectivity=3)
        local_entries: list[dict] = []
        volume_count = int(volume_labels.max())
        multi_object_parent_core_count += int(volume_count > 1)
        parent_workload = dapi_parent_fragment_workload(
            parent_core_id=parent_core_id,
            parent_index_1based=parent_index,
            bbox_yx_0based=(row0, col0, row1, col1),
            volume_labels=volume_labels,
            fragment_count=volume_count,
        )
        parent_workload_record = append_dapi_parent_fragment_workload(
            workload_summary,
            parent_workload,
        )
        if preflight_only:
            continue
        if limits is not None:
            violation = dapi_fragment_workload_violation(
                workload_summary,
                parent_workload,
                limits,
            )
            if violation is not None:
                trigger_metric, observed, limit = violation
                workload_summary["status"] = "blocked"
                workload_summary["guard_triggered"] = True
                workload_summary["guard_reason"] = trigger_metric
                diagnostic: dict[str, object] = {
                    "schema_version": 1,
                    "status": "blocked",
                    "guard_triggered": True,
                    "reason_code": (
                        "DAPI_FRAGMENT_WORKLOAD_LIMIT_EXCEEDED"
                    ),
                    "analysis_stage": "dapi_fragment_pre_submission",
                    "policy_version": limits.policy_version,
                    "trigger_metric": trigger_metric,
                    "observed": int(observed),
                    "limit": int(limit),
                    "selected_z_range_1based": [int(z0 + 1), int(z1 + 1)],
                    "offending_parent": parent_workload_record,
                    "jobs_submitted_for_offending_parent": False,
                    "measurement_stack_loaded": False,
                    "fiji_launched": False,
                    "production_outputs_replaced": False,
                    "workload_summary": workload_summary,
                }
                if workload_diagnostic_path is not None:
                    atomic_write_dapi_fragment_workload_json(
                        workload_diagnostic_path,
                        diagnostic,
                    )
                raise DapiFragmentWorkloadLimitExceeded(
                    diagnostic,
                    workload_diagnostic_path,
                )
        extent_component_labels_crop = extent_component_labels[crop]
        task_arguments = (
            (
                local_volume_id,
                volume_labels,
                volume_count,
                local_core,
                local_dapi,
                local_structural,
                extent_component_labels_crop,
                voxel_sampling,
                dapi_dynamic,
                dapi_min_dynamic,
                low_threshold,
                high_threshold,
                cfg,
            )
            for local_volume_id in range(1, volume_count + 1)
        )
        if worker_count == 1 or volume_count <= 1:
            local_results = [
                evaluate_raw_inventory_object_3d(*arguments)
                for arguments in task_arguments
            ]
        else:
            cancel_event = threading.Event()
            max_pending = min(
                int(volume_count),
                int(
                    limits.max_pending_tasks
                    if limits is not None
                    else max(1, 2 * worker_count)
                ),
            )
            heartbeat_seconds = float(
                limits.heartbeat_seconds if limits is not None else 5.0
            )
            progress_step = max(1, int(math.ceil(volume_count / 10)))

            def report_dapi_fragment_progress(
                progress: dict[str, object],
            ) -> None:
                yielded = int(progress["yielded"])
                if (
                    progress["event"] == "heartbeat"
                    or yielded == volume_count
                    or yielded % progress_step == 0
                ):
                    print_terminal_event(
                        "DAPI fragments | "
                        f"parent={parent_index}/{int(core_labels.max())} | "
                        f"completed={yielded}/{volume_count} | "
                        f"pending={int(progress['pending'])}"
                    )

            with ThreadPoolExecutor(
                max_workers=min(worker_count, volume_count),
                thread_name_prefix="ihc-dapi-object",
            ) as object_executor:
                local_results = bounded_ordered_map(
                    object_executor,
                    evaluate_raw_inventory_object_3d,
                    task_arguments,
                    max_pending=max_pending,
                    cancel_event=cancel_event,
                    heartbeat_seconds=heartbeat_seconds,
                    progress_callback=report_dapi_fragment_progress,
                )
        for object_result in local_results:
            object_projection = object_result["projection"]
            overlap_px = int(object_result["overlap_px"])
            evaluated = object_result["evaluated"]
            z_coordinates = object_result["z_coordinates"]
            extent_component_id = int(object_result["extent_component_id"])
            peak_map = object_result["peak_map"]
            object_id = next_object_id
            next_object_id += 1
            local_entries.append(
                {
                    "object_id": object_id,
                    "projection": object_projection,
                    "peak_map": peak_map,
                    "evaluated": evaluated,
                }
            )
            per_object.append(
                {
                    "nucleus_id_2d": object_id,
                    "object_id_3d": object_id,
                    "parent_core_2d_id": parent_core_id,
                    "parent_core_object_count": volume_count,
                    "accepted": bool(evaluated["accepted"]),
                    "dapi_valid": bool(evaluated["dapi_valid"]),
                    "reason": str(evaluated["reason"]),
                    "center_z": round(z0 + evaluated["center_z_local"], 3),
                    "center_y": round(row0 + evaluated["center_y_local"], 3),
                    "center_x": round(col0 + evaluated["center_x_local"], 3),
                    "dapi_volume_um3": round(evaluated["dapi_volume_um3"], 6),
                    "dapi_z_span_um": round(evaluated["dapi_z_span_um"], 6),
                    "dapi_projection_overlap": round(
                        evaluated["dapi_projection_overlap"],
                        6,
                    ),
                    "surface_coverage": round(evaluated["surface_coverage"], 6),
                    "median_xy_boundary_coverage": round(
                        evaluated["median_xy_boundary_coverage"],
                        6,
                    ),
                    "angular_coverage": round(evaluated["angular_coverage"], 6),
                    "z_support_fraction": round(
                        evaluated["z_support_fraction"],
                        6,
                    ),
                    "shell_enrichment": round(evaluated["shell_enrichment"], 6),
                    "radial_band_fraction": round(
                        evaluated["radial_band_fraction"],
                        6,
                    ),
                    "enclosure_score": round(evaluated["enclosure_score"], 6),
                    "dapi_low_threshold": round(evaluated["dapi_low_threshold"], 6),
                    "dapi_high_threshold": round(evaluated["dapi_high_threshold"], 6),
                    "egfp_threshold": (
                        round(float(evaluated["structural_threshold"]), 6)
                        if evaluated["structural_threshold"] is not None
                        else None
                    ),
                    "structural_channel": context.structural_channel,
                    "raw_projection_overlap_px": overlap_px,
                    "z_min_1based": int(z0 + z_coordinates.min() + 1),
                    "z_max_1based_inclusive": int(z0 + z_coordinates.max() + 1),
                    "projection_area_px": int(object_projection.sum()),
                    "extent_component_2d_id": extent_component_id,
                }
            )
            object_records.append(
                {
                    "object_id": object_id,
                    "parent_core_2d_id": parent_core_id,
                    "extent_component_2d_id": extent_component_id,
                    "accepted": bool(evaluated["accepted"]),
                    "dapi_valid": bool(evaluated["dapi_valid"]),
                    "center_z": float(z0 + evaluated["center_z_local"]),
                    "center_y": float(row0 + evaluated["center_y_local"]),
                    "center_x": float(col0 + evaluated["center_x_local"]),
                    "z_min_0based": int(z0 + z_coordinates.min()),
                    "z_max_0based_inclusive": int(z0 + z_coordinates.max()),
                    "volume_um3": float(evaluated["dapi_volume_um3"]),
                    "projection_area_px": int(object_projection.sum()),
                    "enclosure_score": float(evaluated["enclosure_score"]),
                }
            )

        label_entries = [
            entry for entry in local_entries if bool(entry["evaluated"]["dapi_valid"])
        ]
        if not label_entries:
            continue
        strengths = np.stack(
            [np.asarray(entry["peak_map"], dtype=np.float32) for entry in label_entries],
            axis=0,
        )
        supported = np.isfinite(strengths)
        assigned_index = np.argmax(strengths, axis=0)
        any_supported = np.any(supported, axis=0)
        projection_stack = np.stack(
            [np.asarray(entry["projection"], dtype=bool) for entry in label_entries],
            axis=0,
        )
        distance_stack = np.stack(
            [ndi.distance_transform_edt(~projection) for projection in projection_stack],
            axis=0,
        )
        assigned_index[~any_supported] = np.argmin(distance_stack[:, ~any_supported], axis=0)
        local_object_core_labels = np.zeros(local_core.shape, dtype=np.uint32)
        for index, entry in enumerate(label_entries):
            local_object_core_labels[local_core & (assigned_index == index)] = int(
                entry["object_id"]
            )

        used_seed_pixels: set[tuple[int, int]] = set()
        core_coords = np.argwhere(local_core)
        for entry in label_entries:
            object_id = int(entry["object_id"])
            object_core = local_object_core_labels == object_id
            if object_core.any():
                seed_coord = tuple(np.argwhere(object_core)[0])
                used_seed_pixels.add((int(seed_coord[0]), int(seed_coord[1])))
                continue
            center = np.asarray(
                [
                    float(entry["evaluated"]["center_y_local"]),
                    float(entry["evaluated"]["center_x_local"]),
                ]
            )
            order = np.argsort(np.square(core_coords - center).sum(axis=1))
            for coordinate_index in order:
                seed_y, seed_x = map(int, core_coords[int(coordinate_index)])
                if (seed_y, seed_x) in used_seed_pixels:
                    continue
                local_object_core_labels[seed_y, seed_x] = object_id
                used_seed_pixels.add((seed_y, seed_x))
                break

        object_ids = [int(entry["object_id"]) for entry in label_entries]
        object_seed_masks = [local_object_core_labels == object_id for object_id in object_ids]
        seed_distance_stack = np.stack(
            [ndi.distance_transform_edt(~seed_mask) for seed_mask in object_seed_masks],
            axis=0,
        )
        extent_assignment = np.argmin(seed_distance_stack, axis=0)
        local_object_extent_labels = np.zeros(local_extent.shape, dtype=np.uint32)
        for index, object_id in enumerate(object_ids):
            local_object_extent_labels[local_extent & (extent_assignment == index)] = object_id
        local_object_extent_labels[local_object_core_labels > 0] = local_object_core_labels[
            local_object_core_labels > 0
        ]
        core_view = object_core_labels[crop]
        extent_view = object_extent_labels[crop]
        if np.any((core_view > 0) & (local_object_core_labels > 0)):
            raise RuntimeError("3D nucleus inventory produced overlapping parent-core labels")
        if np.any((extent_view > 0) & (local_object_extent_labels > 0)):
            raise RuntimeError("3D nucleus inventory produced overlapping extent labels")
        core_view[local_object_core_labels > 0] = local_object_core_labels[
            local_object_core_labels > 0
        ]
        extent_view[local_object_extent_labels > 0] = local_object_extent_labels[
            local_object_extent_labels > 0
        ]

    if preflight_only:
        workload_summary["status"] = "preflight_completed"
        return ValidatedNucleusAnchors(
            accepted_core_mask_2d=np.zeros_like(nuclei_core, dtype=bool),
            accepted_extent_mask_2d=np.zeros_like(nuclei_extent, dtype=bool),
            metrics={
                "status": "preflight_completed",
                "method": (
                    "DAPI fragment workload preflight using the production DAPI "
                    "parent reconstruction path without fragment evaluation"
                ),
                "measurement_channel_used": False,
                "z_start_1based": z0 + 1,
                "z_end_1based_inclusive": z1 + 1,
                "parent_2d_core_count": int(core_labels.max()),
                "candidate_count": int(workload_summary["total_fragments"]),
                "accepted_count": 0,
                "rejected_count": 0,
                "dapi_valid_count": 0,
                "per_nucleus": [],
                "config": asdict(cfg),
                "dapi_2d": dapi_2d_metrics,
                "dapi_fragment_workload": workload_summary,
            },
            object_core_labels_2d=object_core_labels,
            object_extent_labels_2d=object_extent_labels,
        )

    workload_summary["status"] = "completed"

    raw_dapi_valid_ids = tuple(
        int(row["object_id_3d"]) for row in per_object if bool(row["dapi_valid"])
    )
    raw_accepted_ids = tuple(
        int(row["object_id_3d"]) for row in per_object if bool(row["accepted"])
    )
    per_object_by_id = {
        int(row["object_id_3d"]): row for row in per_object
    }
    extent_area_counts = np.bincount(
        object_extent_labels.ravel(),
        minlength=next_object_id,
    )
    for record in object_records:
        object_id = int(record["object_id"])
        extent_area_px = int(extent_area_counts[object_id])
        record["extent_area_px"] = extent_area_px
        per_object_by_id[object_id]["extent_area_px"] = extent_area_px
    print(
        "3D DAPI inventory complete | "
        f"objects={len(object_records)} | "
        f"elapsed={time.perf_counter() - inventory_started:.3f} s; "
        "resolving canonical nuclei...",
        flush=True,
    )
    canonical_started = time.perf_counter()
    canonical = resolve_canonical_nucleus_instances_3d(
        frozen_whole_mask,
        nuclei_core,
        nuclei_extent,
        context,
        pixel_width_um,
        pixel_height_um,
        cfg,
        raw_object_extent_labels=object_extent_labels,
        max_workers=worker_count,
    )
    print(
        "Canonical nucleus resolution complete | "
        f"instances={len(canonical.records)} | "
        f"elapsed={time.perf_counter() - canonical_started:.3f} s",
        flush=True,
    )
    canonical_records = [dict(row) for row in canonical.records]
    dapi_valid_ids = raw_dapi_valid_ids
    accepted_ids = raw_accepted_ids
    accepted_core = np.isin(object_core_labels, accepted_ids)
    accepted_extent = np.isin(object_extent_labels, accepted_ids) | accepted_core
    source_object_to_instance_ids: dict[int, list[int]] = {}
    for row in canonical_records:
        for source_object_id in row["source_object_ids"]:
            source_object_to_instance_ids.setdefault(int(source_object_id), []).append(
                int(row["instance_id"])
            )
    return ValidatedNucleusAnchors(
        accepted_core_mask_2d=accepted_core,
        accepted_extent_mask_2d=accepted_extent,
        metrics={
            "status": "completed",
            "method": (
                "object-preserving calibrated 3D DAPI inventory + "
                "independent canonical nuclear-envelope audit layer"
            ),
            "measurement_channel_used": False,
            "z_start_1based": z0 + 1,
            "z_end_1based_inclusive": z1 + 1,
            "voxel_size_um": [
                round(context.pixel_depth_um, 9),
                round(pixel_height_um, 9),
                round(pixel_width_um, 9),
            ],
            "calibration_source": context.calibration_source,
            "structural_channel": context.structural_channel,
            "parent_2d_core_count": int(core_labels.max()),
            "candidate_count": len(per_object),
            "dapi_valid_count": len(dapi_valid_ids),
            "accepted_count": len(accepted_ids),
            "rejected_count": len(per_object) - len(accepted_ids),
            "unowned_dapi_valid_count": len(set(dapi_valid_ids) - set(accepted_ids)),
            "multi_object_parent_core_count": multi_object_parent_core_count,
            "accepted_2d_nucleus_ids": list(accepted_ids),
            "rejected_2d_nucleus_ids": [
                int(row["object_id_3d"])
                for row in per_object
                if not bool(row["accepted"])
            ],
            "config": asdict(cfg),
            "dapi_2d": dapi_2d_metrics,
            "dapi_fragment_workload": workload_summary,
            "canonical_resolution": canonical.metrics,
            "raw_object_qc": {
                "candidate_count": len(per_object),
                "dapi_valid_count": len(raw_dapi_valid_ids),
                "accepted_count": len(raw_accepted_ids),
                "multi_object_parent_core_count": multi_object_parent_core_count,
            },
            "canonical_per_nucleus": canonical_records,
            "per_nucleus": per_object,
        },
        object_core_labels_2d=object_core_labels,
        object_extent_labels_2d=object_extent_labels,
        dapi_valid_object_ids=dapi_valid_ids,
        accepted_object_ids=accepted_ids,
        object_records=tuple(object_records),
        nucleus_instance_core_labels_2d=canonical.core_labels_2d,
        nucleus_instance_extent_labels_2d=canonical.extent_labels_2d,
        accepted_instance_ids=tuple(int(value) for value in canonical.accepted_ids),
        ambiguous_instance_ids=canonical.ambiguous_ids,
        nucleus_instance_records=tuple(canonical_records),
        source_object_to_instance_ids={
            object_id: tuple(sorted(instance_ids))
            for object_id, instance_ids in source_object_to_instance_ids.items()
        },
    )
