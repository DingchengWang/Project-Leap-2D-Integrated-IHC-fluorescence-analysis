# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def make_morphology_baseline_spec(
    name: str,
    z_mode: str = "peak_narrow",
    *,
    threshold_scale: float = 0.94,
    smooth_sigma: float = 1.0,
    min_area: int = 70,
    anchor_area: int = 1150,
    bridge_radius: int = 9,
    low_percentile: float = 78,
    seed_percentile: float = 93,
    seed_min_area: int = 140,
    max_area_fraction: float = 0.34,
    cellpose_cellprob: float = 0.0,
    cellpose_diameter: float = 70,
    dapi_support_radius: int = 16,
    outline_smooth_sigma: float = 1.7,
    artifact_min_area: int = 650,
    artifact_near_radius: int = 36,
    process_eccentricity: float = 0.70,
    process_major_axis: float = 30,
    branch_support_percentile: float = 38,
    branch_support_radius: int = 1,
    max_process_half_width: float = 9,
    soma_protect_radius: int = 21,
    outline_hole_min_area: int = 70,
    require_soma_anchor: bool = False,
    soma_anchor_radius: int = 4,
    soma_anchor_percentile: float = 84.0,
    soma_core_radius: float = 8.0,
    soma_anchor_min_pixels: int = 8,
    anchor_component_min_area: int = 3000,
    connection_radius: int = 3,
    connection_support_percentile: float = 84.0,
    fine_branch_recovery: bool = False,
    fine_branch_detail_percentile: float = 92.0,
    fine_branch_intensity_percentile: float = 68.0,
    fine_branch_min_area: int = 16,
    fine_branch_min_major_axis: float = 12.0,
    fine_branch_min_eccentricity: float = 0.62,
    fine_branch_gap_radius: int = 2,
    fine_branch_background_sigma: float = 7.0,
    fine_branch_single_channel_offset: float = 4.0,
    exclude_border_components: bool = False,
    border_margin: int = 12,
    edge_qc_margin: int = 48,
) -> TestSpec:
    return TestSpec(
        name=name,
        z_mode=z_mode,
        projection="max",
        method="top_hat_union",
        egfp_weight=0.55,
        gfap_weight=0.45,
        smooth_sigma=smooth_sigma,
        threshold_scale=threshold_scale,
        min_area=min_area,
        close_radius=1,
        dilate_radius=1,
        cleanup_mode="hybrid_reconstruct",
        anchor_area=anchor_area,
        bridge_radius=bridge_radius,
        low_percentile=low_percentile,
        seed_percentile=seed_percentile,
        seed_min_area=seed_min_area,
        max_area_fraction=max_area_fraction,
        cellpose=True,
        cellpose_cellprob=cellpose_cellprob,
        cellpose_diameter=cellpose_diameter,
        cellpose_max_side=2048,
        dapi_support_radius=dapi_support_radius,
        outline_smooth_sigma=outline_smooth_sigma,
        outline_epsilon=2.2,
        artifact_filter=True,
        artifact_min_area=artifact_min_area,
        artifact_near_radius=artifact_near_radius,
        process_eccentricity=process_eccentricity,
        process_major_axis=process_major_axis,
        branch_refine=True,
        branch_support_percentile=branch_support_percentile,
        branch_support_radius=branch_support_radius,
        max_process_half_width=max_process_half_width,
        soma_protect_radius=soma_protect_radius,
        outline_hole_min_area=outline_hole_min_area,
        require_soma_anchor=require_soma_anchor,
        soma_anchor_radius=soma_anchor_radius,
        soma_anchor_percentile=soma_anchor_percentile,
        soma_core_radius=soma_core_radius,
        soma_anchor_min_pixels=soma_anchor_min_pixels,
        anchor_component_min_area=anchor_component_min_area,
        connection_radius=connection_radius,
        connection_support_percentile=connection_support_percentile,
        fine_branch_recovery=fine_branch_recovery,
        fine_branch_detail_percentile=fine_branch_detail_percentile,
        fine_branch_intensity_percentile=fine_branch_intensity_percentile,
        fine_branch_min_area=fine_branch_min_area,
        fine_branch_min_major_axis=fine_branch_min_major_axis,
        fine_branch_min_eccentricity=fine_branch_min_eccentricity,
        fine_branch_gap_radius=fine_branch_gap_radius,
        fine_branch_background_sigma=fine_branch_background_sigma,
        fine_branch_single_channel_offset=fine_branch_single_channel_offset,
        exclude_border_components=exclude_border_components,
        border_margin=border_margin,
        edge_qc_margin=edge_qc_margin,
    )

def morphology_baseline_template_specs() -> list[TestSpec]:
    variants = [
        (
            "process",
            dict(
                threshold_scale=0.93,
                min_area=55,
                anchor_area=1150,
                bridge_radius=9,
                low_percentile=80,
                seed_min_area=130,
                max_area_fraction=0.32,
                artifact_min_area=650,
                artifact_near_radius=32,
                branch_support_percentile=44,
                max_process_half_width=8,
                soma_protect_radius=19,
                soma_anchor_percentile=82,
                soma_core_radius=7,
                soma_anchor_min_pixels=6,
                anchor_component_min_area=2800,
                connection_radius=5,
                connection_support_percentile=82,
            ),
        ),
        (
            "balanced",
            dict(
                threshold_scale=0.96,
                min_area=75,
                anchor_area=1250,
                bridge_radius=8,
                low_percentile=81,
                seed_min_area=140,
                max_area_fraction=0.30,
                artifact_min_area=750,
                artifact_near_radius=30,
                branch_support_percentile=48,
                max_process_half_width=7,
                soma_protect_radius=18,
                soma_anchor_percentile=84,
                soma_core_radius=8,
                soma_anchor_min_pixels=8,
                anchor_component_min_area=3600,
                connection_radius=3,
                connection_support_percentile=84,
            ),
        ),
        (
            "clean",
            dict(
                threshold_scale=1.00,
                min_area=95,
                anchor_area=1500,
                bridge_radius=7,
                low_percentile=83,
                seed_percentile=94,
                seed_min_area=175,
                max_area_fraction=0.28,
                artifact_min_area=950,
                artifact_near_radius=27,
                process_eccentricity=0.76,
                process_major_axis=36,
                branch_support_percentile=52,
                max_process_half_width=7,
                soma_protect_radius=18,
                soma_anchor_percentile=85,
                soma_core_radius=9,
                soma_anchor_min_pixels=10,
                anchor_component_min_area=4500,
                connection_radius=2,
                connection_support_percentile=86,
            ),
        ),
        (
            "strict",
            dict(
                threshold_scale=1.04,
                min_area=120,
                anchor_area=1800,
                bridge_radius=6,
                low_percentile=85,
                seed_percentile=95,
                seed_min_area=210,
                max_area_fraction=0.26,
                artifact_min_area=1200,
                artifact_near_radius=24,
                process_eccentricity=0.80,
                process_major_axis=42,
                branch_support_percentile=56,
                max_process_half_width=6,
                soma_protect_radius=17,
                soma_anchor_percentile=87,
                soma_core_radius=10,
                soma_anchor_min_pixels=12,
                anchor_component_min_area=6000,
                connection_radius=1,
                connection_support_percentile=88,
            ),
        ),
    ]
    candidates: list[TestSpec] = []
    for variant, parameters in variants:
        candidates.append(
            make_morphology_baseline_spec(
                f"morphology_baseline_template_auto17_{variant}",
                "auto_17",
                require_soma_anchor=True,
                **parameters,
            )
        )
    return candidates

def adaptive_z_window_widths(n_slices: int) -> list[int]:
    if n_slices < 10:
        raise ValueError(
            f"Morphology baseline requires at least 10 Z slices, got {n_slices}"
        )
    if n_slices >= 47:
        return [17, 23, 31, 39, 47]

    allowed = list(range(3, n_slices + 1, 2))
    if n_slices not in allowed:
        allowed.append(n_slices)
    targets = [
        n_slices * 0.49,
        n_slices * 0.66,
        n_slices * 0.83,
        n_slices * 0.94,
        float(n_slices),
    ]
    selected: list[int] = []
    for target in targets:
        available = [width for width in allowed if width not in selected]
        width = min(available, key=lambda value: (abs(value - target), value))
        selected.append(width)
    return sorted(selected)

def morphology_baseline_specs(n_slices: int) -> list[TestSpec]:
    """Build six morphology-baseline profiles per adaptive Z window."""
    templates = morphology_baseline_template_specs()
    profile_templates = [
        (
            templates[0],
            dict(
                variant="process",
                fine_branch_detail_percentile=92.0,
                fine_branch_intensity_percentile=68.0,
                fine_branch_min_area=16,
                fine_branch_min_major_axis=12.0,
                fine_branch_min_eccentricity=0.62,
                fine_branch_gap_radius=2,
            ),
        ),
        (
            templates[1],
            dict(
                variant="balanced",
                fine_branch_detail_percentile=93.0,
                fine_branch_intensity_percentile=70.0,
                fine_branch_min_area=18,
                fine_branch_min_major_axis=13.0,
                fine_branch_min_eccentricity=0.65,
                fine_branch_gap_radius=2,
            ),
        ),
        (
            templates[2],
            dict(
                variant="clean",
                fine_branch_detail_percentile=94.0,
                fine_branch_intensity_percentile=72.0,
                fine_branch_min_area=20,
                fine_branch_min_major_axis=15.0,
                fine_branch_min_eccentricity=0.68,
                fine_branch_gap_radius=2,
            ),
        ),
        (
            templates[3],
            dict(
                variant="strict",
                fine_branch_detail_percentile=95.0,
                fine_branch_intensity_percentile=74.0,
                fine_branch_min_area=22,
                fine_branch_min_major_axis=18.0,
                fine_branch_min_eccentricity=0.72,
                fine_branch_gap_radius=1,
            ),
        ),
        (
            templates[0],
            dict(
                variant="process_refined",
                fine_branch_detail_percentile=92.5,
                fine_branch_intensity_percentile=69.0,
                fine_branch_min_area=17,
                fine_branch_min_major_axis=12.5,
                fine_branch_min_eccentricity=0.64,
                fine_branch_gap_radius=2,
            ),
        ),
        (
            templates[2],
            dict(
                variant="clean_refined",
                fine_branch_detail_percentile=93.5,
                fine_branch_intensity_percentile=71.0,
                fine_branch_min_area=19,
                fine_branch_min_major_axis=14.0,
                fine_branch_min_eccentricity=0.67,
                fine_branch_gap_radius=2,
            ),
        ),
    ]
    candidates: list[TestSpec] = []
    for width in adaptive_z_window_widths(n_slices):
        for template, profile in profile_templates:
            variant = str(profile["variant"])
            parameters = {key: value for key, value in profile.items() if key != "variant"}
            candidates.append(
                replace(
                    template,
                    name=f"morphology_baseline_auto{width}_{variant}",
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
    if len(candidates) != 30:
        raise AssertionError(
            "Morphology baseline must contain exactly 30 Whole-ROI candidates, "
            f"got {len(candidates)}"
        )
    return candidates
