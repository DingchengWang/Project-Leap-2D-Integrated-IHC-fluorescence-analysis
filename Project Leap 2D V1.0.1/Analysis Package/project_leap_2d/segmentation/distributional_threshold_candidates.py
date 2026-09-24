# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def complete_candidate_specs(
    windows: list[ZWindowPlan],
    structural_stacks: dict[str, np.ndarray],
) -> list[CandidateSpec]:
    """Retain the 60-candidate baseline and append 30 distributional challengers."""
    validate_fixed_z_windows(windows)
    candidates = morphology_with_structural_refinement_specs(
        windows,
        structural_stacks,
    )
    added: list[CandidateSpec] = []
    for window in windows:
        z_mode = window.z_mode
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
                    name=(
                        "distributional_threshold_"
                        f"z{window.position_1based:02d}_{profile_name}"
                    ),
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
    for window in windows:
        group = [spec for spec in candidates if spec.z_mode == window.z_mode]
        if len(group) != EXPECTED_PROFILES_PER_Z:
            raise AssertionError(
                "Z position "
                f"{window.position_1based} must contain exactly {EXPECTED_PROFILES_PER_Z} "
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
            raise AssertionError(
                f"Z position {window.position_1based} contains duplicate candidates"
            )
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
