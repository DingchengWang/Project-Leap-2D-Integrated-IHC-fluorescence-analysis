# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def complete_candidate_specs(
    n_slices: int,
    structural_stacks: dict[str, np.ndarray],
) -> list[TestSpec]:
    """Retain the 60-candidate baseline and append 30 distributional challengers."""
    candidates = morphology_with_structural_refinement_specs(
        n_slices,
        structural_stacks,
    )
    added: list[TestSpec] = []
    for width in adaptive_z_window_widths(n_slices):
        z_mode = f"auto_{width}"
        profiles = {
            candidate_profile_name(spec): spec
            for spec in candidates
            if spec.z_mode == z_mode
        }
        templates = (
            (profiles["fine_process_guarded"], "gmm_process_guarded", 0.90),
            (profiles["structural_snr_adaptive"], "gmm_balanced", 0.98),
            (profiles["precision_guarded"], "gmm_precision_guarded", 1.08),
            (profiles["merge_resistant"], "gmm_merge_resistant", 1.04),
            (
                profiles["channel_consensus_guarded"],
                "gmm_channel_consensus_guarded",
                1.00,
            ),
            (
                profiles["topology_continuity_guarded"],
                "gmm_topology_continuity_guarded",
                0.94,
            ),
        )
        for template, profile_name, threshold_scale in templates:
            added.append(
                replace(
                    template,
                    name=f"distributional_threshold_auto{width}_{profile_name}",
                    method="log1p_gmm",
                    threshold_scale=threshold_scale,
                )
            )
    candidates.extend(added)
    if len(candidates) != TOTAL_CANDIDATE_COUNT:
        raise AssertionError(
            f"Complete candidate catalog requires exactly {TOTAL_CANDIDATE_COUNT} "
            "Whole-ROI candidates, "
            f"got {len(candidates)}"
        )
    widths = adaptive_z_window_widths(n_slices)
    for width in widths:
        group = [spec for spec in candidates if spec.z_mode == f"auto_{width}"]
        if len(group) != EXPECTED_PROFILES_PER_Z:
            raise AssertionError(
                f"Z width {width} must contain exactly {EXPECTED_PROFILES_PER_Z} "
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
        if len(signatures) != EXPECTED_PROFILES_PER_Z:
            raise AssertionError(f"Z width {width} contains duplicate candidates")
    if (
        sum(
            candidate_module_name(spec) == "distributional_threshold"
            for spec in candidates
        )
        != 30
    ):
        raise AssertionError(
            "Distributional threshold group must contain exactly 30 candidates"
        )
    return candidates
