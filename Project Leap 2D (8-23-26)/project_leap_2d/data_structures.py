# This functional source module is assembled into one shared runtime.
from __future__ import annotations

@dataclass(frozen=True)
class TestSpec:
    name: str
    z_mode: str
    projection: str
    method: str
    egfp_weight: float = 0.55
    gfap_weight: float = 0.45
    smooth_sigma: float = 1.0
    threshold_scale: float = 1.0
    min_area: int = 180
    close_radius: int = 1
    dilate_radius: int = 1
    hole_area: int = 64
    cleanup_mode: str = "basic"
    anchor_area: int = 1600
    bridge_radius: int = 8
    low_percentile: float = 83.0
    seed_percentile: float = 93.0
    seed_min_area: int = 260
    max_area_fraction: float = 0.34
    cellpose: bool = False
    cellpose_cellprob: float = -0.5
    cellpose_diameter: float = 85.0
    cellpose_max_side: int = 2048
    dapi_support_radius: int = 24
    outline_smooth_sigma: float = 1.15
    outline_epsilon: float = 1.6
    artifact_filter: bool = False
    artifact_min_area: int = 1100
    artifact_near_radius: int = 36
    process_eccentricity: float = 0.78
    process_major_axis: float = 42.0
    branch_refine: bool = False
    branch_support_percentile: float = 38.0
    branch_support_radius: int = 2
    max_process_half_width: float = 10.0
    soma_protect_radius: int = 22
    outline_hole_min_area: int = 90
    require_soma_anchor: bool = False
    soma_anchor_radius: int = 4
    soma_anchor_percentile: float = 84.0
    soma_core_radius: float = 8.0
    soma_anchor_min_pixels: int = 8
    anchor_component_min_area: int = 3000
    connection_radius: int = 3
    connection_support_percentile: float = 84.0
    fine_branch_recovery: bool = False
    fine_branch_detail_percentile: float = 92.0
    fine_branch_intensity_percentile: float = 68.0
    fine_branch_min_area: int = 16
    fine_branch_min_major_axis: float = 12.0
    fine_branch_min_eccentricity: float = 0.62
    fine_branch_gap_radius: int = 2
    fine_branch_background_sigma: float = 7.0
    fine_branch_single_channel_offset: float = 4.0
    fine_branch_evidence_mode: str = "union"
    fine_branch_consensus_radius: int = 1
    fine_branch_topology_max_gap: int = 4
    fine_branch_topology_min_skeleton: int = 10
    fine_branch_topology_max_hops: int = 3
    exclude_border_components: bool = False
    border_margin: int = 12
    edge_qc_margin: int = 48
    preserve_complete_border_components: bool = False
    border_complete_soma_margin: int = 48
    border_complete_min_area_ratio: float = 0.75
    border_complete_min_interior_fraction: float = 0.75

@dataclass(frozen=True)
class CompartmentConfig:
    """Calibration-aware 2D soma rules applied after Whole ROI selection."""

    dapi_percentile_floor: float = 85.0
    nucleus_link_um: float = 0.55
    soma_zone_min_um: float = 2.1
    soma_zone_max_um: float = 5.2
    soma_zone_scale_process_rich: float = 2.15
    soma_zone_scale_compact: float = 2.85
    thickness_fraction_process_rich: float = 0.34
    thickness_fraction_compact: float = 0.24
    structural_percentile_process_rich: float = 58.0
    structural_percentile_compact: float = 46.0
    min_soma_area_um2: float = 4.5
    fallback_soma_radius_um: float = 1.55
    max_soma_fraction: float = 0.72
    min_process_fraction: float = 0.08
    ambiguity_score_delta: float = 0.08
    primary_anchor_min_score: float = 0.62
    primary_anchor_min_thickness_support: float = 0.60
    primary_anchor_min_structural_support: float = 0.34
    primary_anchor_min_overlap_fraction: float = 0.35
    primary_anchor_min_model_support: float = 0.55
    multi_anchor_min_score: float = 0.60
    multi_anchor_max_score_delta: float = 0.16
    multi_anchor_min_thickness_support: float = 0.65
    multi_anchor_min_structural_support: float = 0.35
    multi_anchor_min_overlap_fraction: float = 0.55
    multi_anchor_min_model_support: float = 0.60
    soma_anchor_min_separation_um: float = 6.0
    max_soma_anchors_per_whole_roi: int = 4
    soma_part_min_core_radius_um: float = 0.90
    soma_part_max_axis_ratio: float = 3.50
    soma_core_shell_max_um: float = 0.75
    soma_trusted_core_radius_scale: float = 1.55
    soma_trusted_core_min_um: float = 1.80
    soma_trusted_core_max_um: float = 3.60
    soma_trusted_core_nucleus_margin_um: float = 0.65
    soma_nucleus_shape_preserving: bool = False
    dapi_extent_percentile_floor: float = 68.0
    dapi_extent_low_high_ratio: float = 0.58
    dapi_extent_max_expand_um: float = 1.45
    dapi_extent_closing_um: float = 0.18
    instance_split_enabled: bool = True
    instance_split_min_anchor_score: float = 0.68
    instance_split_min_anchor_separation_um: float = 6.0
    instance_split_min_child_area_um2: float = 12.0
    instance_split_min_child_fraction: float = 0.12
    instance_split_strict_neck_core_ratio: float = 0.72
    instance_split_max_neck_core_ratio: float = 0.72
    instance_split_max_boundary_structural_ratio: float = 0.72
    instance_split_max_markers: int = 2
    instance_split_strategy: str = "pairwise_soma_anchor_split"
    branch_gap_restore_enabled: bool = True
    branch_gap_low_percentile: float = 25.0
    branch_gap_min_depth_um: float = 0.27
    branch_gap_nucleus_protect_um: float = 2.10
    branch_gap_min_area_um2: float = 0.14
    branch_gap_min_major_axis_um: float = 0.90
    branch_gap_min_eccentricity: float = 0.62
    morphology_outlier_filter_enabled: bool = True
    morphology_outlier_min_reference_count: int = 8
    morphology_outlier_robust_z: float = 3.50
    morphology_outlier_min_consensus: int = 3
    morphology_fragment_min_axis_ratio: float = 4.25
    morphology_fragment_max_branchpoints: int = 1
    morphology_fragment_max_core_radius_um: float = 0.95

@dataclass(frozen=True)
class AgeProfileDecision:
    profile: str
    source: str
    neonatal_score: float | None
    threshold: float
    confidence_margin: float | None
    tagged_files: tuple[str, ...]
    features: dict[str, float | int]

@dataclass(frozen=True)
class Neonatal3DConfig:
    """Calibrated object/surface gate for neonatal DAPI soma anchors."""

    candidate_link_um: float = 0.75
    crop_margin_um: float = 4.60
    dapi_xy_support_margin_um: float = 0.75
    dapi_active_profile_fraction: float = 0.22
    dapi_low_fraction: float = 0.28
    dapi_high_fraction: float = 0.48
    dapi_min_z_span_um: float = 0.35
    dapi_min_volume_um3: float = 0.45
    dapi_min_projection_overlap: float = 0.30
    egfp_smooth_um: float = 0.12
    shell_inner_um: float = 0.08
    shell_outer_um: float = 0.95
    background_inner_um: float = 1.35
    background_outer_um: float = 2.20
    surface_contact_um: float = 0.55
    central_slice_area_fraction: float = 0.45
    slice_support_threshold: float = 0.42
    angular_sector_count: int = 12
    angular_sector_support_fraction: float = 0.20
    min_surface_coverage: float = 0.34
    min_angular_coverage: float = 0.75
    min_z_support_fraction: float = 0.34
    min_shell_enrichment: float = 0.04
    min_enclosure_score: float = 0.48
    canonical_envelope_closing_xy_um: float = 0.22
    canonical_envelope_closing_z_um: float = 0.28
    canonical_crop_margin_um: float = 2.60
    canonical_min_peak_radius_um: float = 0.72
    canonical_peak_h_um: float = 0.38
    canonical_min_peak_separation_um: float = 2.35
    canonical_min_child_volume_um3: float = 3.0
    canonical_min_child_z_span_um: float = 0.45
    canonical_max_neck_peak_ratio: float = 0.58
    canonical_core_radius_fraction: float = 0.46
    canonical_max_instances_per_envelope: int = 4

@dataclass(frozen=True)
class DapiFragmentWorkloadLimits:
    """Versioned resource limits applied before DAPI fragment-job submission."""

    policy_version: str
    max_parent_fragments: int
    max_total_fragments: int
    max_parent_voxel_comparisons: int
    max_total_voxel_comparisons: int
    max_parent_result_payload_bytes_lower_bound: int
    max_pending_tasks: int = 24
    heartbeat_seconds: float = 5.0

@dataclass(frozen=True)
class DapiParentFragmentWorkload:
    parent_core_id: int
    parent_index_1based: int
    bbox_yx_0based: tuple[int, int, int, int]
    crop_shape_zyx: tuple[int, int, int]
    crop_voxels: int
    crop_xy_pixels: int
    fragment_count: int
    estimated_voxel_comparisons: int
    estimated_result_payload_bytes_lower_bound: int

class DapiFragmentWorkloadLimitExceeded(RuntimeError):
    """Expected safety stop for pathological DAPI fragment workloads."""

    def __init__(
        self,
        diagnostic: dict[str, object],
        diagnostic_path: Path | None,
    ) -> None:
        self.diagnostic = diagnostic
        self.diagnostic_path = diagnostic_path
        super().__init__(
            str(
                diagnostic.get(
                    "reason_code",
                    "DAPI_FRAGMENT_WORKLOAD_LIMIT_EXCEEDED",
                )
            )
        )

    def user_message(self) -> str:
        parent = self.diagnostic["offending_parent"]
        assert isinstance(parent, dict)
        shape = parent["crop_shape_zyx"]
        assert isinstance(shape, (list, tuple))
        selected_z = self.diagnostic["selected_z_range_1based"]
        assert isinstance(selected_z, (list, tuple))
        limit = self.diagnostic["limit"]
        observed = self.diagnostic["observed"]
        details = (
            str(self.diagnostic_path)
            if self.diagnostic_path is not None
            else "Diagnostic JSON could not be written."
        )
        return "\n".join(
            [
                "ANALYSIS STOPPED SAFELY — DAPI fragment workload limit exceeded",
                "",
                "The DAPI 3D check found an abnormal nuclear-fragment workload.",
                "Continuing could keep the CPU busy for a very long time, so the run",
                "was stopped before jobs were submitted for the offending region.",
                "",
                f"Selected Z range: {selected_z[0]}–{selected_z[1]}",
                f"Offending parent: {parent['parent_core_id']}",
                f"3D fragments: {int(parent['fragment_count']):,}",
                "Parent crop: "
                f"{int(shape[0]):,} × {int(shape[1]):,} × {int(shape[2]):,} voxels",
                "Estimated voxel checks: "
                f"{int(parent['estimated_voxel_comparisons']):,}",
                f"Triggered metric: {self.diagnostic['trigger_metric']} "
                f"({int(observed):,} > {int(limit):,})",
                f"Safety policy: {self.diagnostic['policy_version']}",
                "",
                "Protected state:",
                "• No fragment jobs were submitted for this parent.",
                "• The KCNN/KCNJ measurement stack was not loaded.",
                "• Fiji was not launched.",
                "• Existing production outputs were not replaced.",
                "",
                "This does not mean that the TIFF is corrupted, and it does not by",
                "itself prove acquisition overexposure. Common causes include DAPI",
                "saturation or clipping during conversion to 8-bit.",
                "",
                "Next step: inspect the original high-bit-depth DAPI data and the",
                "acquisition histogram. Re-export linear grayscale data if clipping",
                "occurred during export; reacquire or exclude the image if the",
                "original acquisition is saturated.",
                "",
                "Diagnostic:",
                str(self.diagnostic["reason_code"]),
                f"Details: {details}",
            ]
        )

@dataclass(frozen=True)
class NucleusOwnershipConfig:
    """Conservative 3D nucleus ownership and foreign-soma conflict rules."""

    fragment_radius_sum_factor: float = 1.25
    owner_min_overlap_um2: float = 0.60
    accepted_min_extent_overlap_fraction: float = 0.45
    foreign_min_overlap_um2: float = 1.60
    foreign_min_overlap_fraction: float = 0.45
    foreign_min_owner_overlap_ratio: float = 0.15
    foreign_min_component_dominance: float = 0.80
    accepted_min_volume_um3: float = 4.0
    unowned_min_volume_um3: float = 12.0
    unowned_min_enclosure_score: float = 0.20
    unowned_barrier_radius_um: float = 1.10
    unowned_barrier_inner_width_um: float = 0.18
    multi_owner_unowned_exclusion_um: float = 0.45
    marker_search_um: float = 0.75
    marker_radius_um: float = 0.25
    minimum_child_area_um2: float = 12.0
    minimum_owner_child_fraction: float = 0.12
    minimum_accepted_child_fraction: float = 0.08
    maximum_boundary_core_ratio: float = 0.86
    maximum_boundary_structural_ratio: float = 0.88
    fragment_bridge_max_gap_um: float = 0.38
    fragment_bridge_min_z_overlap_fraction: float = 0.35
    fragment_bridge_max_axis_ratio: float = 2.25
    projection_occlusion_enabled: bool = True
    projection_foreign_halo_um: float = 1.10
    projection_contact_stop_um: float = 0.12
    projection_min_foreign_volume_um3: float = 3.0
    projection_min_foreign_z_span_um: float = 0.45
    projection_min_foreign_extent_um2: float = 1.2
    projection_min_retained_child_um2: float = 12.0

@dataclass(frozen=True)
class AxialTruncationConfig:
    """Conservative selected-Z cuboid guard for partially observed nuclei."""

    enabled: bool = True
    guard_depth_um: float = 2.0
    boundary_band_um: float = 0.55
    min_inside_volume_fraction: float = 0.62
    min_inside_z_span_um: float = 0.50
    min_outside_continuation_um3: float = 0.35
    min_outside_to_inside_ratio: float = 0.12
    min_boundary_area_ratio: float = 0.24
    min_reference_nuclei: int = 4
    minimum_relative_volume: float = 0.34

@dataclass(frozen=True)
class CanonicalIdentityConfig:
    """Frozen-baseline gates for local 3D nucleus identity reconciliation."""

    enabled: bool = True
    minimum_extent_overlap_um2: float = 0.55
    minimum_extent_overlap_fraction: float = 0.16
    merge_contact_distance_um: float = 0.30
    maximum_merge_source_count: int = 3
    split_minimum_extent_overlap_fraction: float = 0.32
    satellite_max_volume_ratio: float = 0.38
    satellite_max_radius_sum_factor: float = 1.20
    satellite_min_z_overlap_fraction: float = 0.60

@dataclass(frozen=True)
class SomaNuclearCompletionConfig:
    """Conservative completion of a Soma to its already assigned 3D nucleus."""

    enabled: bool = True
    minimum_existing_extent_coverage: float = 0.70
    maximum_added_fraction_of_existing_soma: float = 0.15
    minimum_owner_overlap_um2: float = 0.55
    minimum_foreign_overlap_um2: float = 0.55
    minimum_foreign_overlap_fraction: float = 0.16
    maximum_local_foreign_distance_um: float = 1.10

@dataclass(frozen=True)
class Neonatal3DContext:
    dapi_stack: np.ndarray
    egfp_stack: np.ndarray
    z_start_0based: int
    z_end_0based_inclusive: int
    pixel_depth_um: float
    calibration_source: str
    structural_channel: str = "eGFP"

@dataclass
class ValidatedNucleusAnchors:
    accepted_core_mask_2d: np.ndarray
    accepted_extent_mask_2d: np.ndarray
    metrics: dict
    object_core_labels_2d: np.ndarray | None = None
    object_extent_labels_2d: np.ndarray | None = None
    dapi_valid_object_ids: tuple[int, ...] = ()
    accepted_object_ids: tuple[int, ...] = ()
    object_records: tuple[dict, ...] = ()
    nucleus_instance_core_labels_2d: np.ndarray | None = None
    nucleus_instance_extent_labels_2d: np.ndarray | None = None
    accepted_instance_ids: tuple[int, ...] = ()
    ambiguous_instance_ids: tuple[int, ...] = ()
    nucleus_instance_records: tuple[dict, ...] = ()
    source_object_to_instance_ids: dict[int, tuple[int, ...]] | None = None

@dataclass
class CanonicalNucleusResolution:
    core_labels_2d: np.ndarray
    extent_labels_2d: np.ndarray
    records: tuple[dict, ...]
    accepted_ids: tuple[int, ...]
    ambiguous_ids: tuple[int, ...]
    metrics: dict

@dataclass(frozen=True)
class CandidateBaseResult:
    mask: np.ndarray
    cellpose_mask: np.ndarray
    method_used: str
    error: str

@dataclass(frozen=True)
class CandidateWindowContext:
    projection_key: tuple[int, int, str]
    structural_key: tuple
    dapi_projection: np.ndarray
    structural_projections: dict[str, np.ndarray]
    structural_map: np.ndarray
    cellpose_mask: np.ndarray
    cellpose_note: str

@dataclass(frozen=True)
class Log1pGMMThreshold:
    threshold_raw: float
    threshold_log1p: float
    background_peak_raw: float
    background_peak_log1p: float
    background_sigma_log1p: float
    means_log1p: tuple[float, float]
    variances_log1p: tuple[float, float]
    weights: tuple[float, float]
    ashman_separation: float
    posterior_at_threshold: float
    clip_bounds_raw: tuple[float, float]
    converged: bool
    iterations: int
    qc_pass: bool
    qc_reasons: tuple[str, ...]
