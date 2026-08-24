# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def candidate_profile_name(spec: TestSpec) -> str:
    match = re.search(r"_auto\d+_(.+)$", spec.name)
    if match is None:
        raise ValueError(f"Candidate name does not encode a profile: {spec.name}")
    return match.group(1)

def candidate_profile_family(spec: TestSpec) -> str:
    profile_name = candidate_profile_name(spec)
    families = {
        "process": "process_sensitivity",
        "process_refined": "process_sensitivity",
        "fine_process_guarded": "process_sensitivity",
        "balanced": "balanced_adaptive",
        "structural_snr_adaptive": "balanced_adaptive",
        "clean": "precision",
        "clean_refined": "precision",
        "precision_guarded": "precision",
        "strict": "strict_merge",
        "merge_resistant": "strict_merge",
        "channel_consensus_guarded": "channel_consensus",
        "topology_continuity_guarded": "topology_continuity",
        "gmm_process_guarded": "distributional_threshold",
        "gmm_balanced": "distributional_threshold",
        "gmm_precision_guarded": "distributional_threshold",
        "gmm_merge_resistant": "distributional_threshold",
        "gmm_channel_consensus_guarded": "distributional_threshold",
        "gmm_topology_continuity_guarded": "distributional_threshold",
    }
    try:
        return families[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown candidate profile family: {profile_name}") from exc

def candidate_module_name(spec: TestSpec) -> str:
    profile_name = candidate_profile_name(spec)
    if profile_name.startswith("gmm_"):
        return "distributional_threshold"
    if profile_name in {
        "process",
        "balanced",
        "clean",
        "strict",
        "process_refined",
        "clean_refined",
    }:
        return "morphology_baseline"
    return "structural_refinement"

def candidate_module_display_name(spec: TestSpec) -> str:
    return {
        "morphology_baseline": "Morphology Baseline",
        "structural_refinement": "Structural Refinement",
        "distributional_threshold": "Distributional Threshold",
    }[candidate_module_name(spec)]

def candidate_threshold_display_name(spec: TestSpec) -> str:
    return "log1p_gmm" if spec.method == "log1p_gmm" else str(spec.method)
