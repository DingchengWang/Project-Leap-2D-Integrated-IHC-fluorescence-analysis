# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def structural_snr_sensitivity(structural_stacks: dict[str, np.ndarray]) -> float:
    """Map structural-channel contrast/noise to a bounded branch sensitivity."""
    channel_scores: list[float] = []
    for stack in structural_stacks.values():
        sample = np.asarray(stack[:, ::8, ::8], dtype=np.float64).reshape(-1)
        if sample.size == 0:
            continue
        p50, p75, p995 = np.percentile(sample, [50.0, 75.0, 99.5])
        background = sample[sample <= p75]
        if background.size:
            background_median = float(np.median(background))
            noise = 1.4826 * float(
                np.median(np.abs(background - background_median))
            )
        else:
            noise = 0.0
        signal = max(float(p995 - p50), 0.0)
        scale = max(noise, 0.02 * signal, np.finfo(np.float64).eps)
        channel_scores.append(signal / scale)
    if not channel_scores:
        return 0.5
    robust_score = float(np.median(np.asarray(channel_scores, dtype=np.float64)))
    lower = math.log1p(3.0)
    upper = math.log1p(20.0)
    return float(np.clip((math.log1p(robust_score) - lower) / (upper - lower), 0.0, 1.0))

def morphology_with_adaptive_refinement_specs(
    n_slices: int,
    structural_stacks: dict[str, np.ndarray],
) -> list[TestSpec]:
    """Retain the validated 30 candidates and add 20 complementary profiles."""
    morphology_baseline_candidates = morphology_baseline_specs(n_slices)
    templates = morphology_baseline_template_specs()
    snr_sensitivity = structural_snr_sensitivity(structural_stacks)
    adaptive_detail = 94.5 - 2.0 * snr_sensitivity
    adaptive_intensity = 73.0 - 3.0 * snr_sensitivity
    adaptive_area = int(round(21.0 - 4.0 * snr_sensitivity))
    adaptive_major_axis = 16.0 - 3.0 * snr_sensitivity
    adaptive_eccentricity = 0.70 - 0.05 * snr_sensitivity
    adaptive_gap = 2 if snr_sensitivity >= 0.35 else 1
    added_profiles = [
        (
            templates[0],
            dict(
                variant="fine_process_guarded",
                fine_branch_detail_percentile=91.5,
                fine_branch_intensity_percentile=66.5,
                fine_branch_min_area=14,
                fine_branch_min_major_axis=10.5,
                fine_branch_min_eccentricity=0.58,
                fine_branch_gap_radius=2,
            ),
        ),
        (
            replace(
                templates[3],
                threshold_scale=1.06,
                min_area=135,
                anchor_area=1950,
                bridge_radius=5,
                low_percentile=86,
                seed_percentile=96,
                seed_min_area=230,
                max_area_fraction=0.24,
                artifact_min_area=1400,
                artifact_near_radius=22,
                branch_support_percentile=58,
                max_process_half_width=6,
                soma_anchor_percentile=88,
                soma_anchor_min_pixels=13,
                anchor_component_min_area=6500,
                connection_radius=1,
                connection_support_percentile=90,
            ),
            dict(
                variant="precision_guarded",
                fine_branch_detail_percentile=96.0,
                fine_branch_intensity_percentile=76.0,
                fine_branch_min_area=24,
                fine_branch_min_major_axis=20.0,
                fine_branch_min_eccentricity=0.75,
                fine_branch_gap_radius=1,
            ),
        ),
        (
            replace(
                templates[2],
                threshold_scale=1.02,
                min_area=105,
                anchor_area=1650,
                bridge_radius=5,
                low_percentile=84,
                seed_percentile=95,
                seed_min_area=190,
                max_area_fraction=0.27,
                artifact_min_area=1050,
                artifact_near_radius=24,
                branch_support_percentile=55,
                max_process_half_width=6,
                soma_anchor_percentile=86,
                soma_anchor_min_pixels=11,
                anchor_component_min_area=5000,
                connection_radius=1,
                connection_support_percentile=90,
            ),
            dict(
                variant="merge_resistant",
                fine_branch_detail_percentile=94.5,
                fine_branch_intensity_percentile=73.0,
                fine_branch_min_area=20,
                fine_branch_min_major_axis=16.0,
                fine_branch_min_eccentricity=0.72,
                fine_branch_gap_radius=1,
            ),
        ),
        (
            templates[1],
            dict(
                variant="structural_snr_adaptive",
                fine_branch_detail_percentile=adaptive_detail,
                fine_branch_intensity_percentile=adaptive_intensity,
                fine_branch_min_area=adaptive_area,
                fine_branch_min_major_axis=adaptive_major_axis,
                fine_branch_min_eccentricity=adaptive_eccentricity,
                fine_branch_gap_radius=adaptive_gap,
            ),
        ),
    ]
    added: list[TestSpec] = []
    for width in adaptive_z_window_widths(n_slices):
        for template, profile in added_profiles:
            variant = str(profile["variant"])
            parameters = {
                key: value for key, value in profile.items() if key != "variant"
            }
            added.append(
                replace(
                    template,
                    name=f"structural_refinement_auto{width}_{variant}",
                    z_mode=f"auto_{width}",
                    fine_branch_recovery=True,
                    fine_branch_background_sigma=7.0,
                    fine_branch_single_channel_offset=4.0,
                    exclude_border_components=True,
                    border_margin=12,
                    edge_qc_margin=48,
                    preserve_complete_border_components=True,
                    border_complete_soma_margin=48,
                    border_complete_min_area_ratio=0.75,
                    border_complete_min_interior_fraction=0.75,
                    **parameters,
                )
            )
    candidates = [*morphology_baseline_candidates, *added]
    if len(candidates) != 50:
        raise AssertionError(
            "Morphology with adaptive refinement requires exactly 50 Whole-ROI "
            f"candidates, got {len(candidates)}"
        )
    widths = adaptive_z_window_widths(n_slices)
    for width in widths:
        group = [spec for spec in candidates if spec.z_mode == f"auto_{width}"]
        if len(group) != 10:
            raise AssertionError(
                f"Z width {width} must contain exactly 10 candidate profiles"
            )
        signatures = {
            tuple(sorted({
                key: value
                for key, value in asdict(spec).items()
                if key not in {"name", "z_mode"}
            }.items()))
            for spec in group
        }
        if len(signatures) != 10:
            raise AssertionError(f"Z width {width} contains duplicate candidates")
    return candidates

def morphology_with_structural_refinement_specs(
    n_slices: int,
    structural_stacks: dict[str, np.ndarray],
) -> list[TestSpec]:
    """Retain 50 candidates and append ten orthogonal structural refinements."""
    candidates = morphology_with_adaptive_refinement_specs(
        n_slices,
        structural_stacks,
    )
    new_candidates: list[TestSpec] = []
    for width in adaptive_z_window_widths(n_slices):
        z_mode = f"auto_{width}"
        fine_process = next(
            spec
            for spec in candidates
            if spec.z_mode == z_mode
            and candidate_profile_name(spec) == "fine_process_guarded"
        )
        new_candidates.extend(
            [
                replace(
                    fine_process,
                    name=f"structural_refinement_auto{width}_channel_consensus_guarded",
                    fine_branch_evidence_mode="channel_consensus",
                    fine_branch_consensus_radius=1,
                ),
                replace(
                    fine_process,
                    name=f"structural_refinement_auto{width}_topology_continuity_guarded",
                    fine_branch_evidence_mode="topology_continuity",
                    fine_branch_topology_max_gap=4,
                    fine_branch_topology_min_skeleton=10,
                    fine_branch_topology_max_hops=3,
                ),
            ]
        )
    candidates.extend(new_candidates)
    if len(candidates) != PRE_DISTRIBUTION_BASELINE_CANDIDATE_COUNT:
        raise AssertionError(
            "Pre-distribution baseline requires exactly "
            f"{PRE_DISTRIBUTION_BASELINE_CANDIDATE_COUNT} Whole-ROI candidates, "
            f"got {len(candidates)}"
        )
    widths = adaptive_z_window_widths(n_slices)
    for width in widths:
        group = [spec for spec in candidates if spec.z_mode == f"auto_{width}"]
        if len(group) != PRE_DISTRIBUTION_BASELINE_PROFILES_PER_Z:
            raise AssertionError(
                "Z width "
                f"{width} must contain exactly "
                f"{PRE_DISTRIBUTION_BASELINE_PROFILES_PER_Z} "
                "candidate profiles"
            )
        signatures = {
            tuple(
                sorted(
                    {
                        key: value
                        for key, value in asdict(spec).items()
                        if key not in {"name", "z_mode"}
                    }.items()
                )
            )
            for spec in group
        }
        if len(signatures) != PRE_DISTRIBUTION_BASELINE_PROFILES_PER_Z:
            raise AssertionError(f"Z width {width} contains duplicate candidates")
    return candidates
