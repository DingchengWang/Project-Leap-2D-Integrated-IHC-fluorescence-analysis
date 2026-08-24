# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def dapi_supported_anchor(dapi_proj: np.ndarray, struct: np.ndarray, candidate: np.ndarray, spec: TestSpec) -> np.ndarray:
    nuclei = dapi_nuclei_mask(dapi_proj)
    nuclei = morphology.binary_dilation(nuclei, footprint=morphology.disk(spec.dapi_support_radius))
    structural_high = struct >= full_array_percentile(struct, 74)
    support = candidate & nuclei & structural_high
    support = morphology.remove_small_objects(support, min_size=max(40, spec.min_area // 2))
    return support

def strict_soma_anchor(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    spec: TestSpec,
    distance: np.ndarray | None = None,
) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    nuclei = dapi_nuclei_mask(dapi_proj, percentile_floor=85.0)
    near_nuclei = morphology.binary_dilation(
        nuclei,
        footprint=morphology.disk(spec.soma_anchor_radius),
    )
    if distance is None:
        distance = ndi.distance_transform_edt(mask)
    soma_core = mask & (distance >= spec.soma_core_radius)
    structural_core = struct >= full_array_percentile(
        struct,
        spec.soma_anchor_percentile,
    )
    model_or_structure = structural_core | (cellpose_mask & mask)
    return (soma_core & near_nuclei & model_or_structure).astype(bool)

def retain_soma_connected_components(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    spec: TestSpec,
) -> np.ndarray:
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask.astype(bool)

    soma_anchor = strict_soma_anchor(mask, dapi_proj, struct, cellpose_mask, spec)
    component_count = int(labels.max())
    component_areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    anchor_counts = np.bincount(labels[soma_anchor], minlength=component_count + 1)
    primary_labels = np.flatnonzero(
        (component_areas >= spec.anchor_component_min_area)
        & (anchor_counts >= spec.soma_anchor_min_pixels)
    )
    primary_labels = primary_labels[primary_labels > 0]
    primary = np.isin(labels, primary_labels)
    if not primary.any():
        return np.zeros_like(mask, dtype=bool)

    if spec.connection_radius <= 0:
        return primary.astype(bool)

    support_cut = full_array_percentile(
        struct,
        spec.connection_support_percentile,
    )
    bridge_band = morphology.binary_dilation(
        primary,
        footprint=morphology.disk(spec.connection_radius),
    ) & (struct >= support_cut)
    connection_domain = mask | bridge_band
    domain_labels = measure.label(connection_domain)
    touching_labels = np.unique(domain_labels[primary])
    touching_labels = touching_labels[touching_labels > 0]
    connected = np.isin(domain_labels, touching_labels)
    connected = morphology.remove_small_objects(connected, min_size=spec.min_area)
    return connected.astype(bool)

def empty_branch_recovery_metrics() -> dict:
    return {
        "fine_branch_evidence_px": 0,
        "fine_branch_added_px": 0,
        "fine_branch_added_fraction": 0.0,
        "fine_branch_added_structural_mean": 0.0,
        "fine_branch_consensus_evidence_px": 0,
        "fine_branch_single_channel_retained_px": 0,
        "fine_branch_topology_bridge_px": 0,
        "fine_branch_topology_skeleton_px": 0,
        "fine_branch_topology_rejected_px": 0,
    }

def fine_branch_features(
    structural_projections: dict[str, np.ndarray],
    cache_key: tuple,
    background_sigma: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    projection_identity = tuple(
        (channel, array_identity_key(projection))
        for channel, projection in sorted(structural_projections.items())
    )
    feature_key = (
        *cache_key,
        projection_identity,
        round(float(background_sigma), 3),
    )
    with _CACHE_LOCK:
        cached = _BRANCH_FEATURE_CACHE.get(feature_key)
    if cached is not None:
        return cached
    with cache_key_lock(("branch_features", *feature_key)):
        with _CACHE_LOCK:
            cached = _BRANCH_FEATURE_CACHE.get(feature_key)
        if cached is not None:
            return cached
        features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for channel, projection in structural_projections.items():
            normalized = normalized_projection(projection)
            smoothed = filters.gaussian(normalized, sigma=0.8, preserve_range=True)
            local_background = filters.gaussian(
                smoothed,
                sigma=background_sigma,
                preserve_range=True,
            )
            local_detail = np.clip(smoothed - local_background, 0, None).astype(
                np.float32
            )
            ridge = filters.sato(
                smoothed,
                sigmas=(1, 2, 3),
                black_ridges=False,
            ).astype(np.float32)
            local_detail.setflags(write=False)
            ridge.setflags(write=False)
            features[channel] = (normalized, local_detail, ridge)

        with _CACHE_LOCK:
            _BRANCH_FEATURE_CACHE[feature_key] = features
        return features

def channel_consensus_branch_evidence(
    mask: np.ndarray,
    channel_evidence: list[np.ndarray],
    channel_support: list[np.ndarray],
    radius: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    evidence = np.logical_or.reduce(channel_evidence)
    low_support = np.logical_or.reduce(channel_support)
    if len(channel_evidence) < 2:
        footprint = morphology.disk(max(1, int(radius)))
        evidence_labels = measure.label(evidence)
        directly_connected = morphology.binary_dilation(
            mask,
            footprint=footprint,
        )
        retained_labels = np.unique(evidence_labels[directly_connected])
        retained_labels = retained_labels[retained_labels > 0]
        guarded_evidence = np.isin(evidence_labels, retained_labels)
        guarded_support = low_support & morphology.binary_dilation(
            mask | guarded_evidence,
            footprint=footprint,
        )
        return guarded_evidence, guarded_support, {
            "fine_branch_consensus_evidence_px": 0,
            "fine_branch_single_channel_retained_px": int(guarded_evidence.sum()),
        }

    footprint = morphology.disk(max(1, int(radius)))
    consensus = np.zeros_like(mask, dtype=bool)
    consensus_support = np.zeros_like(mask, dtype=bool)
    for left in range(len(channel_evidence)):
        for right in range(left + 1, len(channel_evidence)):
            left_evidence = channel_evidence[left]
            right_evidence = channel_evidence[right]
            consensus |= left_evidence & morphology.binary_dilation(
                right_evidence,
                footprint=footprint,
            )
            consensus |= right_evidence & morphology.binary_dilation(
                left_evidence,
                footprint=footprint,
            )
            left_support = channel_support[left]
            right_support = channel_support[right]
            consensus_support |= left_support & morphology.binary_dilation(
                right_support,
                footprint=footprint,
            )
            consensus_support |= right_support & morphology.binary_dilation(
                left_support,
                footprint=footprint,
            )

    single_channel = evidence & ~consensus
    single_labels = measure.label(single_channel)
    trusted_seed = mask | consensus
    trusted_neighborhood = morphology.binary_dilation(
        trusted_seed,
        footprint=morphology.disk(max(1, int(radius) + 1)),
    )
    retained_labels = np.unique(single_labels[trusted_neighborhood])
    retained_labels = retained_labels[retained_labels > 0]
    retained_single = np.isin(single_labels, retained_labels)
    guarded_evidence = consensus | retained_single
    guarded_support = consensus_support | (
        low_support
        & morphology.binary_dilation(
            mask | guarded_evidence,
            footprint=footprint,
        )
    )
    return guarded_evidence, guarded_support, {
        "fine_branch_consensus_evidence_px": int(consensus.sum()),
        "fine_branch_single_channel_retained_px": int(retained_single.sum()),
    }

def topology_continuity_branch_evidence(
    mask: np.ndarray,
    evidence: np.ndarray,
    low_support: np.ndarray,
    *,
    max_gap: int,
    min_skeleton: int,
    max_hops: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    evidence_labels = measure.label(evidence)
    if evidence_labels.max() == 0:
        return evidence, np.zeros_like(mask, dtype=bool), {
            "fine_branch_topology_bridge_px": 0,
            "fine_branch_topology_skeleton_px": 0,
            "fine_branch_topology_rejected_px": 0,
        }

    skeleton = morphology.skeletonize(evidence)
    skeleton_counts = np.bincount(
        evidence_labels[skeleton],
        minlength=int(evidence_labels.max()) + 1,
    )
    valid_labels = np.flatnonzero(skeleton_counts >= max(1, int(min_skeleton)))
    valid_labels = valid_labels[valid_labels > 0]
    line_evidence = np.isin(evidence_labels, valid_labels)
    line_labels = measure.label(line_evidence)
    accepted = np.zeros_like(mask, dtype=bool)
    bridges = np.zeros_like(mask, dtype=bool)
    frontier = mask.astype(bool, copy=True)
    one_pixel = morphology.disk(1)
    max_gap = max(1, int(max_gap))

    for _ in range(max(1, int(max_hops))):
        reachable = frontier.copy()
        for _step in range(max_gap):
            reachable |= morphology.binary_dilation(
                reachable,
                footprint=one_pixel,
            ) & low_support
        touching = morphology.binary_dilation(
            reachable,
            footprint=one_pixel,
        )
        touching_labels = np.unique(line_labels[touching])
        touching_labels = touching_labels[touching_labels > 0]
        newly_accepted = np.isin(line_labels, touching_labels) & ~accepted
        if not newly_accepted.any():
            break
        bridge_target = morphology.binary_dilation(
            newly_accepted,
            footprint=morphology.disk(max_gap),
        )
        bridges |= reachable & bridge_target & low_support
        accepted |= newly_accepted
        frontier = mask | accepted | bridges

    accepted_skeleton_px = int(morphology.skeletonize(accepted).sum())
    return accepted, bridges, {
        "fine_branch_topology_bridge_px": int(bridges.sum()),
        "fine_branch_topology_skeleton_px": accepted_skeleton_px,
        "fine_branch_topology_rejected_px": int((evidence & ~accepted).sum()),
    }

def recover_anchor_connected_fine_processes(
    mask: np.ndarray,
    structural_projections: dict[str, np.ndarray],
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    spec: TestSpec,
    cache_key: tuple,
) -> tuple[np.ndarray, dict]:
    if not spec.fine_branch_recovery or not mask.any():
        return mask.astype(bool), empty_branch_recovery_metrics()

    features = fine_branch_features(
        structural_projections,
        cache_key=cache_key,
        background_sigma=spec.fine_branch_background_sigma,
    )
    evidence = np.zeros_like(mask, dtype=bool)
    low_support = np.zeros_like(mask, dtype=bool)
    channel_evidence_masks: list[np.ndarray] = []
    channel_support_masks: list[np.ndarray] = []
    has_gfap = "GFAP" in structural_projections

    for channel, (normalized, local_detail, ridge) in features.items():
        if channel == "GFAP":
            channel_offset = 0.0
        elif has_gfap:
            channel_offset = spec.fine_branch_single_channel_offset * 0.5
        else:
            channel_offset = spec.fine_branch_single_channel_offset

        detail_percentile = min(99.5, spec.fine_branch_detail_percentile + channel_offset)
        intensity_percentile = min(
            99.0,
            spec.fine_branch_intensity_percentile + 1.5 * channel_offset,
        )
        detail_cut = full_array_percentile(local_detail, detail_percentile)
        ridge_cut = full_array_percentile(ridge, detail_percentile)
        intensity_cut = full_array_percentile(normalized, intensity_percentile)
        channel_evidence = (
            (normalized >= intensity_cut)
            & (local_detail >= detail_cut)
            & (ridge >= ridge_cut)
        )
        channel_min_area = max(
            4,
            int(round(spec.fine_branch_min_area + 1.5 * channel_offset)),
        )
        channel_evidence = morphology.remove_small_objects(
            channel_evidence,
            min_size=channel_min_area,
        )

        labels = measure.label(channel_evidence)
        shaped_labels: list[int] = []
        min_major_axis = spec.fine_branch_min_major_axis + 1.5 * channel_offset
        min_eccentricity = min(
            0.95,
            spec.fine_branch_min_eccentricity + 0.025 * channel_offset,
        )
        for prop in measure.regionprops(labels):
            major_axis = float(
                prop.axis_major_length
                if hasattr(prop, "axis_major_length")
                else prop.major_axis_length
            )
            line_like = (
                prop.area >= channel_min_area
                and major_axis >= min_major_axis
                and (
                    prop.eccentricity >= min_eccentricity
                    or major_axis >= 1.8 * min_major_axis
                )
            )
            if line_like:
                shaped_labels.append(int(prop.label))
        shaped = np.isin(labels, shaped_labels)
        channel_evidence = morphology.binary_closing(
            shaped,
            footprint=morphology.disk(1),
        )
        evidence |= channel_evidence
        channel_evidence_masks.append(channel_evidence)

        support_detail_cut = full_array_percentile(
            local_detail,
            max(50.0, detail_percentile - 20.0),
        )
        support_intensity_cut = full_array_percentile(
            normalized,
            max(45.0, intensity_percentile - 10.0),
        )
        channel_support = (
            (normalized >= support_intensity_cut)
            & (local_detail >= support_detail_cut)
        )
        low_support |= channel_support
        channel_support_masks.append(channel_support)

    mode_metrics: dict[str, int] = {}
    topology_bridge = np.zeros_like(mask, dtype=bool)
    if spec.fine_branch_evidence_mode == "channel_consensus":
        evidence, low_support, mode_metrics = channel_consensus_branch_evidence(
            mask,
            channel_evidence_masks,
            channel_support_masks,
            spec.fine_branch_consensus_radius,
        )
    elif spec.fine_branch_evidence_mode == "topology_continuity":
        evidence, topology_bridge, mode_metrics = topology_continuity_branch_evidence(
            mask,
            evidence,
            low_support,
            max_gap=spec.fine_branch_topology_max_gap,
            min_skeleton=spec.fine_branch_topology_min_skeleton,
            max_hops=spec.fine_branch_topology_max_hops,
        )
    elif spec.fine_branch_evidence_mode != "union":
        raise ValueError(
            f"Unknown fine-branch evidence mode: {spec.fine_branch_evidence_mode}"
        )

    if not evidence.any():
        return mask.astype(bool), empty_branch_recovery_metrics()

    if spec.fine_branch_evidence_mode == "topology_continuity":
        bridge = topology_bridge
    else:
        bridge = morphology.binary_dilation(
            mask,
            footprint=morphology.disk(spec.fine_branch_gap_radius),
        ) & low_support
    domain = mask | evidence | bridge
    base_labels = measure.label(mask)
    distance_from_base = ndi.distance_transform_edt(~mask)
    grown_labels = segmentation.watershed(
        distance_from_base,
        markers=base_labels,
        mask=domain,
    )

    # Keep neighboring astrocytes separated if one permissive line component
    # reaches two pre-existing soma-anchored components.
    seam = np.zeros_like(mask, dtype=bool)
    vertical = (
        (grown_labels[1:] > 0)
        & (grown_labels[:-1] > 0)
        & (grown_labels[1:] != grown_labels[:-1])
    )
    horizontal = (
        (grown_labels[:, 1:] > 0)
        & (grown_labels[:, :-1] > 0)
        & (grown_labels[:, 1:] != grown_labels[:, :-1])
    )
    diagonal_down = (
        (grown_labels[1:, 1:] > 0)
        & (grown_labels[:-1, :-1] > 0)
        & (grown_labels[1:, 1:] != grown_labels[:-1, :-1])
    )
    diagonal_up = (
        (grown_labels[1:, :-1] > 0)
        & (grown_labels[:-1, 1:] > 0)
        & (grown_labels[1:, :-1] != grown_labels[:-1, 1:])
    )
    seam[1:] |= vertical
    seam[:-1] |= vertical
    seam[:, 1:] |= horizontal
    seam[:, :-1] |= horizontal
    seam[1:, 1:] |= diagonal_down
    seam[:-1, :-1] |= diagonal_down
    seam[1:, :-1] |= diagonal_up
    seam[:-1, 1:] |= diagonal_up
    grown = grown_labels > 0
    grown[seam & ~mask] = False
    grown |= mask

    if spec.require_soma_anchor:
        grown = retain_soma_connected_components(
            grown,
            dapi_proj,
            struct,
            cellpose_mask,
            spec,
        )
    added = grown & ~mask
    metrics = {
        "fine_branch_evidence_px": int(evidence.sum()),
        "fine_branch_added_px": int(added.sum()),
        "fine_branch_added_fraction": round(
            float(added.sum()) / max(int(grown.sum()), 1),
            6,
        ),
        "fine_branch_added_structural_mean": round(
            float(struct[added].mean()) if added.any() else 0.0,
            6,
        ),
        **{
            key: int(value)
            for key, value in mode_metrics.items()
        },
    }
    for key, value in empty_branch_recovery_metrics().items():
        metrics.setdefault(key, value)
    return grown.astype(bool), metrics

def empty_border_exclusion_metrics() -> dict:
    return {
        "border_candidate_components": 0,
        "border_preserved_complete_components": 0,
        "border_preserved_complete_area_px": 0,
        "border_preserved_complete_area_fraction": 0.0,
        "border_reference_median_area_px": 0.0,
        "border_removed_components": 0,
        "border_removed_area_px": 0,
        "border_removed_area_fraction": 0.0,
    }

def edge_zone_mask(shape: tuple[int, int], margin: int) -> np.ndarray:
    height, width = shape
    bounded_margin = min(max(1, int(margin)), height // 2, width // 2)
    zone = np.zeros(shape, dtype=bool)
    zone[:bounded_margin] = True
    zone[-bounded_margin:] = True
    zone[:, :bounded_margin] = True
    zone[:, -bounded_margin:] = True
    return zone

def complete_soma_component_labels(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
    labels: np.ndarray | None = None,
    distance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if labels is None:
        labels = measure.label(mask)
    component_count = int(labels.max())
    if component_count == 0:
        return labels, np.empty(0, dtype=np.int32), 0.0

    # A process may leave the field of view while the biological cell remains
    # usable. Require its DAPI-supported structural soma core to be well inside
    # the image, then compare the component with complete interior cells.
    soma_anchor = strict_soma_anchor(
        mask,
        dapi_proj,
        struct,
        np.zeros_like(mask, dtype=bool),
        spec,
        distance=distance,
    )
    soma_interior = ~edge_zone_mask(mask.shape, spec.border_complete_soma_margin)
    soma_anchor &= soma_interior

    component_areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    anchor_counts = np.bincount(labels[soma_anchor], minlength=component_count + 1)
    interior_counts = np.bincount(
        labels[soma_interior],
        minlength=component_count + 1,
    )
    supported = (
        (component_areas >= spec.anchor_component_min_area)
        & (anchor_counts >= spec.soma_anchor_min_pixels)
    )
    supported[0] = False

    border_zone = edge_zone_mask(mask.shape, spec.border_margin)
    border_labels = np.unique(labels[border_zone])
    border_labels = border_labels[border_labels > 0]
    interior_supported = np.flatnonzero(supported & ~np.isin(np.arange(component_count + 1), border_labels))
    if interior_supported.size:
        reference_area = float(np.median(component_areas[interior_supported]))
    else:
        all_supported = np.flatnonzero(supported)
        reference_area = float(np.median(component_areas[all_supported])) if all_supported.size else 0.0

    area_floor = reference_area * spec.border_complete_min_area_ratio
    interior_fraction = interior_counts / np.maximum(component_areas, 1)
    complete = supported & (component_areas >= area_floor)
    complete &= interior_fraction >= spec.border_complete_min_interior_fraction
    complete[0] = False
    return labels, np.flatnonzero(complete).astype(np.int32), reference_area

def exclude_incomplete_border_components(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
) -> tuple[np.ndarray, dict]:
    if not spec.exclude_border_components or not mask.any():
        return mask.astype(bool), empty_border_exclusion_metrics()

    labels, complete_labels, reference_area = complete_soma_component_labels(
        mask,
        dapi_proj,
        struct,
        spec,
    )
    border_zone = edge_zone_mask(mask.shape, spec.border_margin)
    border_labels = np.unique(labels[border_zone])
    border_labels = border_labels[border_labels > 0]
    if spec.preserve_complete_border_components:
        preserved_labels = np.intersect1d(border_labels, complete_labels)
    else:
        preserved_labels = np.empty(0, dtype=border_labels.dtype)
    removed_labels = np.setdiff1d(border_labels, preserved_labels)
    removed = np.isin(labels, removed_labels)
    preserved = np.isin(labels, preserved_labels)
    kept = mask & ~removed
    preserved_fraction = float(preserved.sum()) / max(int(kept.sum()), 1)
    removed_fraction = float(removed.sum()) / max(int(mask.sum()), 1)
    metrics = {
        "border_candidate_components": int(len(border_labels)),
        "border_preserved_complete_components": int(len(preserved_labels)),
        "border_preserved_complete_area_px": int(preserved.sum()),
        "border_preserved_complete_area_fraction": round(preserved_fraction, 6),
        "_raw_border_preserved_complete_area_fraction": preserved_fraction,
        "border_reference_median_area_px": round(reference_area, 3),
        "border_removed_components": int(len(removed_labels)),
        "border_removed_area_px": int(removed.sum()),
        "border_removed_area_fraction": round(removed_fraction, 6),
        "_raw_border_removed_area_fraction": removed_fraction,
    }
    return kept.astype(bool), metrics

def remove_isolated_artifact_fragments(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    cellpose_mask: np.ndarray,
    spec: TestSpec,
) -> np.ndarray:
    if spec.require_soma_anchor:
        return retain_soma_connected_components(mask, dapi_proj, struct, cellpose_mask, spec)

    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    props = measure.regionprops(labels, intensity_image=struct)
    large_labels = [int(prop.label) for prop in props if prop.area >= spec.anchor_area]
    large = np.isin(labels, large_labels)
    near_large = morphology.binary_dilation(large, footprint=morphology.disk(spec.artifact_near_radius))
    dapi_support = dapi_supported_anchor(dapi_proj, struct, mask, spec)
    near_dapi_soma = morphology.binary_dilation(
        dapi_support,
        footprint=morphology.disk(spec.artifact_near_radius),
    )
    soma_min_area = max(180, spec.min_area * 2)
    process_min_area = max(120, spec.min_area)

    keep = np.zeros_like(mask, dtype=bool)
    for prop in props:
        component_slice = prop.slice
        comp = prop.image
        area = prop.area
        near_main = bool((comp & near_large[component_slice]).any())
        has_dapi_support = bool((comp & dapi_support[component_slice]).any())
        process_like = (
            prop.eccentricity >= spec.process_eccentricity
            and prop.major_axis_length >= spec.process_major_axis
        )
        large_enough = area >= spec.artifact_min_area
        near_supported_soma = bool((comp & near_dapi_soma[component_slice]).any())
        supported_soma = has_dapi_support and area >= soma_min_area
        supported_process = process_like and near_supported_soma and area >= process_min_area
        if large_enough or near_main or supported_soma or supported_process:
            keep_view = keep[component_slice]
            keep_view[comp] = True
    keep = morphology.remove_small_objects(keep, min_size=spec.min_area)
    return keep.astype(bool)

def refine_fused_process_regions(
    mask: np.ndarray,
    candidate: np.ndarray,
    cellpose_mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
) -> np.ndarray:
    if not spec.branch_refine or not mask.any():
        return mask.astype(bool)

    distance = ndi.distance_transform_edt(mask)
    values = struct[mask]
    evidence_cut = float(np.percentile(values, spec.branch_support_percentile))
    direct_support = morphology.binary_dilation(
        candidate & mask,
        footprint=morphology.disk(spec.branch_support_radius),
    )
    intensity_support = morphology.binary_dilation(
        mask & (struct >= evidence_cut),
        footprint=morphology.disk(1),
    )
    thin_process_corridor = mask & (distance <= spec.max_process_half_width)

    dapi_anchor = dapi_supported_anchor(dapi_proj, struct, candidate, spec)
    structural_core = struct >= full_array_percentile(struct, 74)
    cellpose_anchor = cellpose_mask & mask & structural_core
    soma_seed = dapi_anchor | cellpose_anchor
    soma_support = morphology.binary_dilation(
        soma_seed,
        footprint=morphology.disk(spec.soma_protect_radius),
    )

    refined = mask & (
        direct_support
        | intensity_support
        | thin_process_corridor
        | soma_support
    )
    refined = morphology.binary_closing(refined, footprint=morphology.disk(1))
    refined = morphology.remove_small_holes(refined, area_threshold=spec.hole_area)
    refined = morphology.remove_small_objects(refined, min_size=spec.min_area)
    return refined.astype(bool)

def refine_with_cellpose_and_dapi(
    raw_mask: np.ndarray,
    cellpose_mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
) -> np.ndarray:
    candidate = cleanup_mask(raw_mask, spec)
    cp_context = morphology.binary_dilation(cellpose_mask, footprint=morphology.disk(spec.bridge_radius))
    candidate = cleanup_mask(candidate, spec)
    cp_anchor = cp_context & candidate & (
        struct >= full_array_percentile(struct, 66)
    )
    cp_anchor = morphology.remove_small_objects(cp_anchor, min_size=max(60, spec.min_area))
    dapi_anchor = dapi_supported_anchor(dapi_proj, struct, candidate, spec)
    extra_anchor = cp_anchor | dapi_anchor
    cleaned = anchor_connected_cleanup(candidate, struct, spec, extra_anchor=extra_anchor)
    cleaned = refine_fused_process_regions(
        cleaned,
        candidate,
        cellpose_mask,
        dapi_proj,
        struct,
        spec,
    )
    if spec.artifact_filter:
        cleaned = remove_isolated_artifact_fragments(cleaned, dapi_proj, struct, cellpose_mask, spec)
    return cleaned

def postprocess_without_cellpose(
    mask: np.ndarray,
    raw_mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
) -> np.ndarray:
    candidate = cleanup_mask(raw_mask, spec)
    empty_cellpose = np.zeros_like(mask, dtype=bool)
    cleaned = refine_fused_process_regions(
        mask,
        candidate,
        empty_cellpose,
        dapi_proj,
        struct,
        spec,
    )
    if spec.artifact_filter:
        cleaned = remove_isolated_artifact_fragments(cleaned, dapi_proj, struct, empty_cellpose, spec)
    return cleaned

def fixed_soma_component_metrics(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    labels: np.ndarray | None = None,
    distance: np.ndarray | None = None,
) -> dict:
    if labels is None:
        labels = measure.label(mask)
    component_count = int(labels.max())
    if component_count == 0:
        return {
            "soma_supported_components": 0,
            "unanchored_components": 0,
            "unanchored_area_fraction": 0.0,
            "median_component_area_px": 0.0,
        }

    nuclei = dapi_nuclei_mask(dapi_proj, percentile_floor=85.0)
    near_nuclei = morphology.binary_dilation(nuclei, footprint=morphology.disk(4))
    if distance is None:
        distance = ndi.distance_transform_edt(mask)
    structural_core = struct >= full_array_percentile(struct, 84.0)
    fixed_anchor = mask & near_nuclei & structural_core & (distance >= 8.0)

    component_areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    anchor_counts = np.bincount(labels[fixed_anchor], minlength=component_count + 1)
    supported = (component_areas >= 3000) & (anchor_counts >= 8)
    supported[0] = False
    unanchored_labels = np.flatnonzero((component_areas > 0) & ~supported)
    unanchored_labels = unanchored_labels[unanchored_labels > 0]
    unanchored_area = int(component_areas[unanchored_labels].sum())
    unanchored_fraction = unanchored_area / max(int(mask.sum()), 1)
    return {
        "soma_supported_components": int(supported.sum()),
        "unanchored_components": int(len(unanchored_labels)),
        "unanchored_area_fraction": round(unanchored_fraction, 6),
        "_raw_unanchored_area_fraction": unanchored_fraction,
        "median_component_area_px": round(float(np.median(component_areas[1:])), 3),
    }

def edge_proximity_metrics(
    mask: np.ndarray,
    dapi_proj: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
    labels: np.ndarray | None = None,
    distance: np.ndarray | None = None,
) -> dict:
    labels, complete_labels, _ = complete_soma_component_labels(
        mask,
        dapi_proj,
        struct,
        spec,
        labels=labels,
        distance=distance,
    )
    if labels.max() == 0:
        return {
            "final_border_touching_components": 0,
            "final_complete_border_touching_components": 0,
            "final_incomplete_border_touching_components": 0,
            "edge_proximity_components": 0,
            "edge_proximity_area_fraction": 0.0,
        }

    exact_border = np.zeros_like(mask, dtype=bool)
    exact_border[[0, -1], :] = True
    exact_border[:, [0, -1]] = True
    exact_labels = np.unique(labels[exact_border])
    exact_labels = exact_labels[exact_labels > 0]
    complete_exact_labels = np.intersect1d(exact_labels, complete_labels)
    incomplete_exact_labels = np.setdiff1d(exact_labels, complete_labels)

    edge_zone = edge_zone_mask(mask.shape, spec.edge_qc_margin)
    near_labels = np.unique(labels[edge_zone])
    near_labels = near_labels[near_labels > 0]
    incomplete_near_labels = np.setdiff1d(near_labels, complete_labels)
    near_area = int(np.isin(labels, incomplete_near_labels).sum())
    edge_fraction = near_area / max(int(mask.sum()), 1)
    return {
        "final_border_touching_components": int(len(exact_labels)),
        "final_complete_border_touching_components": int(len(complete_exact_labels)),
        "final_incomplete_border_touching_components": int(len(incomplete_exact_labels)),
        "edge_proximity_components": int(len(incomplete_near_labels)),
        "edge_proximity_area_fraction": round(edge_fraction, 6),
        "_raw_edge_proximity_area_fraction": edge_fraction,
    }

def qc_metrics(mask: np.ndarray, struct: np.ndarray, dapi_proj: np.ndarray, spec: TestSpec) -> dict:
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    total_signal = full_array_sum(struct)
    mask_area = int(mask.sum())
    distance = ndi.distance_transform_edt(mask) if mask_area else None
    if mask_area:
        masked_struct = struct[mask]
        in_signal = float(np.sum(masked_struct, dtype=np.float64))
        evidence_cut = float(np.percentile(masked_struct, 40))
        unsupported_wide = mask & (distance > 10) & (struct < evidence_cut)
        unsupported_wide_fraction = float(unsupported_wide.sum()) / mask_area
        structural_precision = float(np.mean(masked_struct, dtype=np.float64))
    else:
        in_signal = 0.0
        unsupported_wide_fraction = 0.0
        structural_precision = 0.0

    background_labels = measure.label(~mask)
    border_labels = np.unique(
        np.concatenate(
            [
                background_labels[0],
                background_labels[-1],
                background_labels[:, 0],
                background_labels[:, -1],
            ]
        )
    )
    internal_holes = [
        prop
        for prop in measure.regionprops(background_labels)
        if prop.label not in border_labels and prop.area >= spec.outline_hole_min_area
    ]
    structural_coverage = in_signal / total_signal if total_signal > 0 else 0.0
    mask_area_fraction = float(np.mean(mask, dtype=np.float64))
    metrics = {
        "mask_area_px": mask_area,
        "mask_area_fraction": round(mask_area_fraction, 6),
        "_raw_mask_area_fraction": mask_area_fraction,
        "connected_components": int(len(props)),
        "largest_component_px": int(max((p.area for p in props), default=0)),
        "structural_signal_coverage": round(structural_coverage, 6),
        "_raw_structural_signal_coverage": structural_coverage,
        "structural_precision": round(structural_precision, 6),
        "_raw_structural_precision": structural_precision,
        "unsupported_wide_fraction": round(unsupported_wide_fraction, 6),
        "_raw_unsupported_wide_fraction": unsupported_wide_fraction,
        "internal_holes": int(len(internal_holes)),
        "internal_hole_area_px": int(sum(prop.area for prop in internal_holes)),
    }
    metrics.update(
        fixed_soma_component_metrics(
            mask,
            dapi_proj,
            struct,
            labels=labels,
            distance=distance,
        )
    )
    metrics.update(
        edge_proximity_metrics(
            mask,
            dapi_proj,
            struct,
            spec,
            labels=labels,
            distance=distance,
        )
    )
    return metrics

def candidate_cellpose_cache_key(
    structural_channels: list[str],
    projection_key: tuple[int, int, str],
    spec: TestSpec,
) -> tuple:
    z0, z1, projection_mode = projection_key
    return (
        tuple(structural_channels),
        z0,
        z1,
        projection_mode,
        round(spec.egfp_weight, 3),
        round(spec.gfap_weight, 3),
        round(spec.smooth_sigma, 3),
        round(spec.cellpose_cellprob, 3),
        round(spec.cellpose_diameter, 3),
        spec.cellpose_max_side,
    )

def build_candidate_window_contexts(
    *,
    chosen_specs: list[TestSpec],
    structural_channels: list[str],
    dapi_stack: np.ndarray,
    structural_stacks: dict[str, np.ndarray],
    profile: np.ndarray,
    projection_cache: dict[
        tuple[int, int, str],
        tuple[np.ndarray, dict[str, np.ndarray]],
    ],
    structural_map_cache: dict[tuple, np.ndarray],
) -> list[CandidateWindowContext]:
    contexts: list[CandidateWindowContext] = []
    shared_contexts: dict[tuple, CandidateWindowContext] = {}
    for spec in chosen_specs:
        z0, z1 = z_range_from_mode(spec.z_mode, profile)
        projection_key = (z0, z1, spec.projection)
        if projection_key not in projection_cache:
            dapi_projection = project(dapi_stack, z0, z1, spec.projection)
            structural_projections = {
                channel: project(stack, z0, z1, spec.projection)
                for channel, stack in structural_stacks.items()
            }
            dapi_projection.setflags(write=False)
            for projection in structural_projections.values():
                projection.setflags(write=False)
            projection_cache[projection_key] = (
                dapi_projection,
                structural_projections,
            )
        dapi_projection, structural_projections = projection_cache[projection_key]

        structural_key = (
            *projection_key,
            round(spec.egfp_weight, 6),
            round(spec.gfap_weight, 6),
            round(spec.smooth_sigma, 6),
        )
        if structural_key not in structural_map_cache:
            cached_struct = structural_map(structural_projections, spec)
            cached_struct.setflags(write=False)
            structural_map_cache[structural_key] = cached_struct
        struct = structural_map_cache[structural_key]

        context_key = (
            structural_key,
            bool(spec.cellpose),
            round(spec.cellpose_cellprob, 3),
            round(spec.cellpose_diameter, 3),
            spec.cellpose_max_side,
        )
        context = shared_contexts.get(context_key)
        if context is None:
            if spec.cellpose:
                cellpose_mask, cellpose_note = run_cellpose_mask(
                    struct,
                    spec,
                    candidate_cellpose_cache_key(
                        structural_channels,
                        projection_key,
                        spec,
                    ),
                )
                immutable_cellpose = np.asarray(
                    cellpose_mask,
                    dtype=bool,
                ).copy()
            else:
                immutable_cellpose = np.zeros_like(struct, dtype=bool)
                cellpose_note = "cellpose_disabled"
            immutable_cellpose.setflags(write=False)
            context = CandidateWindowContext(
                projection_key=projection_key,
                structural_key=structural_key,
                dapi_projection=dapi_projection,
                structural_projections=structural_projections,
                structural_map=struct,
                cellpose_mask=immutable_cellpose,
                cellpose_note=cellpose_note,
            )
            shared_contexts[context_key] = context
        contexts.append(context)
    return contexts

def precompute_candidate_top_hat(
    context: CandidateWindowContext,
    representative: TestSpec,
) -> None:
    threshold_mask(
        context.structural_map,
        context.structural_projections,
        representative,
    )

def precompute_candidate_dapi(
    context: CandidateWindowContext,
    percentile_floor: float | None,
) -> None:
    dapi_nuclei_mask(
        context.dapi_projection,
        percentile_floor=percentile_floor,
    )

def precompute_candidate_branches(
    context: CandidateWindowContext,
    input_dir: Path,
    structural_channels: list[str],
    background_sigma: float,
) -> None:
    fine_branch_features(
        context.structural_projections,
        cache_key=(
            str(input_dir),
            tuple(structural_channels),
            *context.projection_key,
        ),
        background_sigma=background_sigma,
    )

def precompute_distribution_models(context: CandidateWindowContext) -> None:
    try:
        get_log1p_gmm_threshold(context.structural_map)
    except ValueError:
        # Distributional-threshold candidates consume the cached failure as a
        # fail-closed QC result.
        pass

def candidate_precompute_jobs(
    *,
    context_groups: list[tuple[CandidateWindowContext, list[TestSpec]]],
    input_dir: Path,
    structural_channels: list[str],
) -> list[tuple[int, object, tuple]]:
    jobs: list[tuple[int, object, tuple]] = []
    for context, specs in context_groups:
        if any(spec.method == "log1p_gmm" for spec in specs):
            jobs.append((0, precompute_distribution_models, (context,)))
        branch_sigmas = sorted(
            {
                float(spec.fine_branch_background_sigma)
                for spec in specs
                if spec.fine_branch_recovery
            }
        )
        for background_sigma in branch_sigmas:
            jobs.append(
                (
                    0,
                    precompute_candidate_branches,
                    (context, input_dir, structural_channels, background_sigma),
                )
            )
        top_hat_specs = [spec for spec in specs if spec.method == "top_hat_union"]
        if top_hat_specs:
            jobs.append(
                (
                    1,
                    precompute_candidate_top_hat,
                    (context, top_hat_specs[0]),
                )
            )
        jobs.append((2, precompute_candidate_dapi, (context, None)))
        jobs.append((2, precompute_candidate_dapi, (context, 85.0)))
    return sorted(jobs, key=lambda item: item[0])

def candidate_base_cache_key(struct: np.ndarray, spec: TestSpec) -> tuple:
    base_spec = asdict(spec)
    base_spec.pop("name", None)
    base_spec.pop("z_mode", None)
    for field_name in list(base_spec):
        if field_name.startswith("fine_branch_"):
            base_spec.pop(field_name)
    return (
        "candidate_base",
        array_identity_key(struct),
        tuple(sorted(base_spec.items())),
    )

def evaluate_ihc_candidate(
    *,
    candidate_number: int,
    candidate_count: int,
    spec: TestSpec,
    input_dir: Path,
    structural_channels: list[str],
    dapi_stack: np.ndarray,
    structural_stacks: dict[str, np.ndarray],
    profile: np.ndarray,
    projection_cache: dict[tuple[int, int, str], tuple[np.ndarray, dict[str, np.ndarray]]],
    structural_map_cache: dict[tuple, np.ndarray],
    emit_progress: bool = True,
    window_context: CandidateWindowContext | None = None,
) -> tuple[np.ndarray, dict, tuple[int, int, str]]:
    z0, z1 = z_range_from_mode(spec.z_mode, profile)
    projection_key = (z0, z1, spec.projection)
    if window_context is None:
        if projection_key not in projection_cache:
            projection_cache[projection_key] = (
                project(dapi_stack, z0, z1, spec.projection),
                {
                    channel: project(stack, z0, z1, spec.projection)
                    for channel, stack in structural_stacks.items()
                },
            )
        d_proj, structural_projections = projection_cache[projection_key]
    else:
        if window_context.projection_key != projection_key:
            raise AssertionError("Candidate context projection key mismatch")
        d_proj = window_context.dapi_projection
        structural_projections = window_context.structural_projections
    structural_key = (
        *projection_key,
        round(spec.egfp_weight, 6),
        round(spec.gfap_weight, 6),
        round(spec.smooth_sigma, 6),
    )
    if window_context is None:
        if structural_key not in structural_map_cache:
            cached_struct = structural_map(structural_projections, spec)
            cached_struct.setflags(write=False)
            structural_map_cache[structural_key] = cached_struct
        struct = structural_map_cache[structural_key]
    else:
        if window_context.structural_key != structural_key:
            raise AssertionError("Candidate context structural key mismatch")
        struct = window_context.structural_map
    normalized_weights = active_channel_weights(structural_projections, spec)
    base_key = candidate_base_cache_key(struct, spec)
    with _CACHE_LOCK:
        base_result = _CANDIDATE_BASE_CACHE.get(base_key)
        base_lock = _CANDIDATE_BASE_LOCKS.setdefault(base_key, threading.Lock())
    if base_result is None:
        with base_lock:
            with _CACHE_LOCK:
                base_result = _CANDIDATE_BASE_CACHE.get(base_key)
            if base_result is None:
                cellpose_mask = np.zeros_like(struct, dtype=bool)
                try:
                    if spec.cellpose:
                        raw_mask = threshold_mask(struct, structural_projections, spec)
                        if window_context is None:
                            cellpose_mask, cellpose_note = run_cellpose_mask(
                                struct,
                                spec,
                                candidate_cellpose_cache_key(
                                    structural_channels,
                                    projection_key,
                                    spec,
                                ),
                            )
                        else:
                            cellpose_mask = window_context.cellpose_mask
                            cellpose_note = window_context.cellpose_note
                        cellpose_mask = cleanup_mask(cellpose_mask, spec)
                        if cellpose_mask.sum() == 0:
                            if spec.cleanup_mode == "basic":
                                mask = cleanup_mask(raw_mask, spec)
                            else:
                                mask = anchor_connected_cleanup(raw_mask, struct, spec)
                            mask = postprocess_without_cellpose(
                                mask,
                                raw_mask,
                                d_proj,
                                struct,
                                spec,
                            )
                            method_used = f"{cellpose_note}_fallback_{spec.method}"
                        else:
                            mask = refine_with_cellpose_and_dapi(
                                raw_mask,
                                cellpose_mask,
                                d_proj,
                                struct,
                                spec,
                            )
                            method_used = (
                                f"{cellpose_note}+{spec.method}+"
                                "cellpose_anchor_only+dapi_anchor"
                            )
                    else:
                        raw_mask = threshold_mask(struct, structural_projections, spec)
                        if spec.cleanup_mode == "basic":
                            mask = cleanup_mask(raw_mask, spec)
                        else:
                            mask = anchor_connected_cleanup(raw_mask, struct, spec)
                        mask = postprocess_without_cellpose(
                            mask,
                            raw_mask,
                            d_proj,
                            struct,
                            spec,
                        )
                        method_used = spec.method
                    error = ""
                except Exception as exc:
                    try:
                        raw_mask = threshold_mask(
                            struct,
                            structural_projections,
                            spec,
                        )
                    except Exception as threshold_exc:
                        if spec.method != "log1p_gmm":
                            raise
                        raw_mask = np.zeros_like(struct, dtype=bool)
                        mask = raw_mask.copy()
                        method_used = "log1p_gmm_qc_failed"
                        error = repr(threshold_exc)
                    else:
                        if spec.cleanup_mode == "basic":
                            mask = cleanup_mask(raw_mask, spec)
                        else:
                            mask = anchor_connected_cleanup(raw_mask, struct, spec)
                        mask = postprocess_without_cellpose(
                            mask,
                            raw_mask,
                            d_proj,
                            struct,
                            spec,
                        )
                        method_used = f"fallback_{spec.method}"
                        error = repr(exc)
                cached_mask = mask.astype(bool, copy=True)
                cached_cellpose = cellpose_mask.astype(bool, copy=True)
                cached_mask.setflags(write=False)
                cached_cellpose.setflags(write=False)
                base_result = CandidateBaseResult(
                    mask=cached_mask,
                    cellpose_mask=cached_cellpose,
                    method_used=method_used,
                    error=error,
                )
                with _CACHE_LOCK:
                    _CANDIDATE_BASE_CACHE[base_key] = base_result
    mask = base_result.mask.copy()
    cellpose_mask = base_result.cellpose_mask
    method_used = base_result.method_used
    error = base_result.error

    branch_base_mask = mask
    branch_metrics = empty_branch_recovery_metrics()
    if spec.fine_branch_recovery:
        try:
            mask, branch_metrics = recover_anchor_connected_fine_processes(
                mask,
                structural_projections,
                d_proj,
                struct,
                cellpose_mask,
                spec,
                cache_key=(str(input_dir), tuple(structural_channels), *projection_key),
            )
            method_used = f"{method_used}+anchored_fine_branch_recovery"
            if spec.fine_branch_evidence_mode == "channel_consensus":
                method_used = f"{method_used}+channel_consensus_guarded"
            elif spec.fine_branch_evidence_mode == "topology_continuity":
                method_used = f"{method_used}+topology_continuity_guarded"
        except Exception as exc:
            detail = f"fine_branch_recovery={exc!r}"
            error = f"{error}; {detail}" if error else detail
            method_used = f"{method_used}+fine_branch_recovery_failed"

    border_metrics = empty_border_exclusion_metrics()
    if spec.exclude_border_components:
        mask, border_metrics = exclude_incomplete_border_components(mask, d_proj, struct, spec)
        method_used = f"{method_used}+border_soma_complete"

    retained_added = mask & ~branch_base_mask
    retained_added_px = int(retained_added.sum())
    final_mask_area_px = int(mask.sum())
    branch_metrics["fine_branch_retained_px"] = retained_added_px
    branch_metrics["fine_branch_retained_fraction"] = round(
        float(retained_added_px) / max(final_mask_area_px, 1),
        6,
    )
    if spec.branch_refine:
        method_used = f"{method_used}+branch_gap_refine"
    if spec.require_soma_anchor:
        method_used = f"{method_used}+soma_connected_filter"

    z_activity_mean = float(np.mean(profile[z0 : z1 + 1], dtype=np.float64))
    z_activity_integral = float(np.sum(profile[z0 : z1 + 1], dtype=np.float64))
    row = {
        "candidate": candidate_number,
        "pipeline_name": PIPELINE_NAME,
        "name": spec.name,
        "candidate_profile": candidate_profile_name(spec),
        "candidate_family": candidate_profile_family(spec),
        "candidate_module": candidate_module_name(spec),
        "z_mode": spec.z_mode,
        "z_start_0based": z0,
        "z_end_0based_inclusive": z1,
        "z_start_1based": z0 + 1,
        "z_end_1based_inclusive": z1 + 1,
        "z_slice_count": z1 - z0 + 1,
        "z_activity_mean": round(z_activity_mean, 6),
        "_raw_z_activity_mean": z_activity_mean,
        "z_activity_integral": round(z_activity_integral, 6),
        "_raw_z_activity_integral": z_activity_integral,
        "projection": spec.projection,
        "method_requested": "cellpose_cpsam_v2" if spec.cellpose else spec.method,
        "method_used": method_used,
        "egfp_weight_effective": normalized_weights.get("eGFP", 0.0),
        "gfap_weight_effective": normalized_weights.get("GFAP", 0.0),
        "threshold_scale": spec.threshold_scale,
        "min_area": spec.min_area,
        "close_radius": spec.close_radius,
        "dilate_radius": spec.dilate_radius,
        "cellpose": bool(spec.cellpose),
        "cellpose_cellprob": spec.cellpose_cellprob,
        "cellpose_diameter": spec.cellpose_diameter,
        "cellpose_max_side": spec.cellpose_max_side,
        "fine_branch_detail_percentile": spec.fine_branch_detail_percentile,
        "fine_branch_intensity_percentile": spec.fine_branch_intensity_percentile,
        "fine_branch_min_area": spec.fine_branch_min_area,
        "fine_branch_min_major_axis": spec.fine_branch_min_major_axis,
        "fine_branch_min_eccentricity": spec.fine_branch_min_eccentricity,
        "fine_branch_gap_radius": spec.fine_branch_gap_radius,
        "fine_branch_evidence_mode": spec.fine_branch_evidence_mode,
        "border_margin": spec.border_margin,
        "edge_qc_margin": spec.edge_qc_margin,
        "border_complete_soma_margin": spec.border_complete_soma_margin,
        "border_complete_min_area_ratio": spec.border_complete_min_area_ratio,
        "border_complete_min_interior_fraction": spec.border_complete_min_interior_fraction,
        "error": error,
    }
    if spec.method == "log1p_gmm":
        row.update(
            distributional_threshold_diagnostics(
                struct,
                structural_projections,
                spec,
            )
        )
    row.update(branch_metrics)
    row.update(border_metrics)
    row.update(qc_metrics(mask, struct, d_proj, spec))
    if emit_progress:
        print(
            f"candidate {candidate_number:02d}/{candidate_count}: "
            f"z={z0 + 1}-{z1 + 1}, area={row['mask_area_px']}, "
            f"components={row['connected_components']}, soma={row['soma_supported_components']}, "
            f"fine_added={row['fine_branch_retained_px']}, method={method_used}",
            flush=True,
        )
    return mask, row, projection_key
