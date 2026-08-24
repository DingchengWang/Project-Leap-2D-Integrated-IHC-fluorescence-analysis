"""Lazy, controller-facing pipeline for the independent DAPI + GFAP mode.

The module owns only the GFAP-only scientific branch.  File discovery,
selected-slice measurement loading, Fiji review, debug rendering, and atomic
publication stay in the main runtime and are connected through small callbacks.
This avoids a second copy of the controller and ensures that target-channel
intensity cannot influence ROI definition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Collection, Optional

import numpy as np
from scipy import ndimage as ndi

from .gfap_only_analysis import GFAPOnlyConfig, GFAPOnlyResult


@dataclass(frozen=True)
class GFAPZSelectionConfig:
    """Physical-scale rules for one inclusive, contiguous GFAP Z interval."""

    activity_percentile: float = 55.0
    minimum_relative_activity: float = 0.15
    smoothing_sigma_um: float = 0.75
    padding_um: float = 1.0
    minimum_span_um: float = 4.0
    projection: str = "max"


@dataclass(frozen=True)
class GFAPOnlyQualityConfig:
    """Gross-failure guards applied before measurement or Fiji publication."""

    maximum_whole_image_fraction: float = 0.75
    maximum_soma_whole_fraction: float = 0.90
    minimum_process_whole_fraction: float = 0.02


@dataclass(frozen=True)
class GFAPZSelection:
    """Auditable active-Z decision; indices are relative to the source stack."""

    start_0based: int
    end_0based_inclusive: int
    indices: tuple[int, ...]
    raw_activity: tuple[float, ...]
    smoothed_activity: tuple[float, ...]
    activity_threshold: float
    projection: str

    @property
    def start_1based(self) -> int:
        return self.start_0based + 1

    @property
    def end_1based_inclusive(self) -> int:
        return self.end_0based_inclusive + 1


@dataclass(frozen=True)
class GFAPOnlyPreparedRun:
    """Complete in-memory handoff to existing debug/Fiji runtime functions."""

    analysis: GFAPOnlyResult
    z_selection: GFAPZSelection
    dapi_projection: np.ndarray
    gfap_projection: np.ndarray
    measurement_projection: np.ndarray | None
    nucleus_detection: dict[str, Any]
    stage_timings_seconds: dict[str, float]

    @property
    def best_row(self) -> dict[str, Any]:
        """Minimal row compatible with the established Fiji preparation API."""

        return {
            "candidate": 0,
            "analysis_mode": "dapi_gfap_only",
            "z_start_0based": self.z_selection.start_0based,
            "z_end_0based_inclusive": self.z_selection.end_0based_inclusive,
            "z_start_1based": self.z_selection.start_1based,
            "z_end_1based_inclusive": self.z_selection.end_1based_inclusive,
            "projection": self.z_selection.projection,
        }


@dataclass(frozen=True)
class GFAPOnlyPipelineResult:
    """Single return object for debug and Fiji executions."""

    prepared: GFAPOnlyPreparedRun
    skip_fiji: bool
    debug_result: Any = None
    fiji_result: Any = None


NucleusDetector = Callable[..., Any]
GFAPAnalyzer = Callable[..., GFAPOnlyResult]
MeasurementProjectionLoader = Callable[[int, int, str], np.ndarray]
PreparedRunHandler = Callable[[GFAPOnlyPreparedRun], Any]
GFAPStageReporter = Callable[[str, str, Optional[float]], None]


def _report_stage(
    reporter: GFAPStageReporter | None,
    stage: str,
    status: str,
    elapsed_seconds: float | None = None,
) -> None:
    if reporter is not None:
        reporter(stage, status, elapsed_seconds)


def validate_gfap_only_route(
    available_channels: Collection[str],
    *,
    egfp_is_valid: bool,
) -> None:
    """Refuse accidental use when the normal eGFP route is available."""

    channels = {str(channel).strip() for channel in available_channels}
    missing = {"DAPI", "GFAP"} - channels
    if missing:
        raise ValueError(
            "GFAP-only analysis requires DAPI and GFAP; missing "
            + ", ".join(sorted(missing))
        )
    if bool(egfp_is_valid):
        raise ValueError(
            "GFAP-only analysis is disabled because a valid eGFP channel is present"
        )


def _validate_stack_pair(
    dapi_stack: np.ndarray,
    gfap_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dapi = np.asarray(dapi_stack)
    gfap = np.asarray(gfap_stack)
    if dapi.ndim != 3 or gfap.ndim != 3:
        raise ValueError("DAPI and GFAP inputs must both have shape (Z, Y, X)")
    if dapi.shape != gfap.shape:
        raise ValueError(
            f"DAPI and GFAP stack shapes differ: {dapi.shape} versus {gfap.shape}"
        )
    if dapi.shape[0] < 1 or dapi.shape[1] < 1 or dapi.shape[2] < 1:
        raise ValueError("DAPI/GFAP stacks cannot be empty")
    return dapi, gfap


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _gfap_plane_activity(plane: np.ndarray) -> float:
    """Robust within-plane contrast, insensitive to a uniform bright offset."""

    values = np.asarray(plane, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median, high = np.percentile(finite, (50.0, 99.2))
    return max(0.0, float(high - median))


def select_gfap_active_z(
    gfap_stack: np.ndarray,
    z_spacing_um: float,
    *,
    config: GFAPZSelectionConfig | None = None,
) -> GFAPZSelection:
    """Choose a contiguous active interval with padding/span expressed in µm."""

    gfap = np.asarray(gfap_stack)
    if gfap.ndim != 3 or gfap.shape[0] < 1:
        raise ValueError("GFAP stack must have shape (Z, Y, X)")
    spacing = _positive_finite(z_spacing_um, "z_spacing_um")
    active = config or GFAPZSelectionConfig()
    if not 0.0 <= float(active.activity_percentile) <= 100.0:
        raise ValueError("activity_percentile must be between 0 and 100")
    if not 0.0 <= float(active.minimum_relative_activity) <= 1.0:
        raise ValueError("minimum_relative_activity must be between 0 and 1")
    if active.projection not in {"max", "mean", "sum"}:
        raise ValueError("GFAP-only projection must be max, mean, or sum")
    if active.smoothing_sigma_um < 0 or active.padding_um < 0:
        raise ValueError("Z smoothing and padding cannot be negative")
    if active.minimum_span_um <= 0:
        raise ValueError("minimum_span_um must be positive")

    raw = np.asarray(
        [_gfap_plane_activity(gfap[z]) for z in range(gfap.shape[0])],
        dtype=np.float64,
    )
    sigma_slices = float(active.smoothing_sigma_um) / spacing
    smoothed = (
        ndi.gaussian_filter1d(raw, sigma=max(sigma_slices, 0.0), mode="nearest")
        if sigma_slices > 0
        else raw.copy()
    )
    dynamic = float(smoothed.max() - smoothed.min())
    numerical_floor = max(
        np.finfo(np.float64).eps * max(abs(float(smoothed.max())), 1.0),
        1e-9,
    )
    if dynamic <= numerical_floor:
        start, end = 0, int(gfap.shape[0]) - 1
        threshold = float(smoothed.min())
    else:
        threshold = max(
            float(np.percentile(smoothed, float(active.activity_percentile))),
            float(smoothed.min())
            + float(active.minimum_relative_activity) * dynamic,
        )
        active_indices = np.flatnonzero(smoothed >= threshold)
        if active_indices.size == 0:
            active_indices = np.asarray([int(np.argmax(smoothed))])
        padding_slices = int(np.ceil(float(active.padding_um) / spacing))
        start = max(0, int(active_indices.min()) - padding_slices)
        end = min(
            int(gfap.shape[0]) - 1,
            int(active_indices.max()) + padding_slices,
        )

        minimum_slices = min(
            int(gfap.shape[0]),
            max(1, int(np.ceil(float(active.minimum_span_um) / spacing)) + 1),
        )
        current_slices = end - start + 1
        if current_slices < minimum_slices:
            peak = int(np.argmax(smoothed))
            before = (minimum_slices - 1) // 2
            start = max(0, peak - before)
            end = start + minimum_slices - 1
            if end >= gfap.shape[0]:
                end = int(gfap.shape[0]) - 1
                start = max(0, end - minimum_slices + 1)

    indices = tuple(range(start, end + 1))
    return GFAPZSelection(
        start_0based=start,
        end_0based_inclusive=end,
        indices=indices,
        raw_activity=tuple(float(value) for value in raw),
        smoothed_activity=tuple(float(value) for value in smoothed),
        activity_threshold=threshold,
        projection=active.projection,
    )


def _project_selected(stack: np.ndarray, selection: GFAPZSelection) -> np.ndarray:
    selected = np.asarray(stack)[selection.start_0based : selection.end_0based_inclusive + 1]
    if selection.projection == "max":
        return selected.max(axis=0)
    if selection.projection == "mean":
        return selected.mean(axis=0, dtype=np.float32)
    return selected.sum(axis=0, dtype=np.float32)


def _default_nucleus_detector(
    dapi_stack: np.ndarray,
    pixel_height_um: float,
    pixel_width_um: float,
    *,
    z_indices: tuple[int, ...],
) -> Any:
    # Delayed import keeps PyTorch/model bytes out of the eGFP route.
    from ...nuclei.instanseg_nucleus_detection import detect_instanseg_nuclei

    return detect_instanseg_nuclei(
        dapi_stack,
        pixel_height_um,
        pixel_width_um,
        z_indices=z_indices,
    )


def _default_gfap_analyzer(*args: Any, **kwargs: Any) -> GFAPOnlyResult:
    # Delayed wrapper makes the ownership boundary explicit and easy to audit.
    from .gfap_only_analysis import analyze_dapi_gfap_only

    return analyze_dapi_gfap_only(*args, **kwargs)


def _default_analysis_config() -> GFAPOnlyConfig:
    """Safe GFAP projection defaults for the mature-only route."""

    base = GFAPOnlyConfig()
    structure = replace(
        base.structure,
        projection_percentile=95.0,
        intensity_floor_percentile=64.0,
        structural_percentile=84.0,
        strong_ridge_percentile=94.0,
        connection_gap_um=0.18,
    )
    nucleus_ownership = replace(
        base.nucleus_ownership,
        min_shell_enrichment=4.5,
    )
    return replace(
        base,
        structure=structure,
        nucleus_ownership=nucleus_ownership,
    )


def _validate_analysis_partition(
    result: GFAPOnlyResult,
    quality: GFAPOnlyQualityConfig,
) -> None:
    whole = np.asarray(result.whole_labels)
    soma = np.asarray(result.soma_labels)
    processes = np.asarray(result.process_labels)
    if whole.shape != soma.shape or whole.shape != processes.shape:
        raise RuntimeError("GFAP-only Whole/Soma/Processes shapes are inconsistent")
    if np.any((soma > 0) & (whole != soma)):
        raise RuntimeError("GFAP-only Soma escaped or changed its Whole owner")
    if np.any((processes > 0) & (whole != processes)):
        raise RuntimeError("GFAP-only Processes escaped or changed their Whole owner")
    occupancy = (soma > 0).astype(np.uint8) + (processes > 0).astype(np.uint8)
    if np.any(occupancy[whole > 0] != 1) or np.any(occupancy[whole == 0] != 0):
        raise RuntimeError("GFAP-only Whole is not exactly Soma union Processes")
    whole_ids = {int(value) for value in np.unique(whole) if value > 0}
    if not whole_ids:
        raise RuntimeError("GFAP-only analysis returned no astrocyte ROI")
    for name, labels in (("Soma", soma), ("Processes", processes)):
        observed = {int(value) for value in np.unique(labels) if value > 0}
        if observed != whole_ids:
            raise RuntimeError(
                f"GFAP-only {name} IDs do not match Whole IDs: "
                f"{sorted(observed)} versus {sorted(whole_ids)}"
            )
    image_pixels = int(whole.size)
    whole_pixels = int(np.count_nonzero(whole))
    soma_pixels = int(np.count_nonzero(soma))
    process_pixels = int(np.count_nonzero(processes))
    whole_fraction = whole_pixels / max(image_pixels, 1)
    soma_fraction = soma_pixels / max(whole_pixels, 1)
    process_fraction = process_pixels / max(whole_pixels, 1)
    if whole_fraction > float(quality.maximum_whole_image_fraction):
        raise RuntimeError(
            "GFAP-only structural mask covers an implausibly large image "
            f"fraction ({whole_fraction:.1%}); analysis stopped before Fiji"
        )
    if soma_fraction > float(quality.maximum_soma_whole_fraction):
        raise RuntimeError(
            "GFAP-only result contains insufficient process structure "
            f"(Soma={soma_fraction:.1%} of Whole); analysis stopped before Fiji"
        )
    if process_fraction < float(quality.minimum_process_whole_fraction):
        raise RuntimeError(
            "GFAP-only result contains insufficient process structure "
            f"(Processes={process_fraction:.1%} of Whole); analysis stopped before Fiji"
        )


def run_gfap_only_pipeline(
    *,
    available_channels: Collection[str],
    egfp_is_valid: bool,
    dapi_stack: np.ndarray,
    gfap_stack: np.ndarray,
    pixel_height_um: float,
    pixel_width_um: float,
    z_spacing_um: float,
    skip_fiji: bool = False,
    debug: bool = False,
    z_config: GFAPZSelectionConfig | None = None,
    analysis_config: GFAPOnlyConfig | None = None,
    quality_config: GFAPOnlyQualityConfig | None = None,
    measurement_projection_loader: MeasurementProjectionLoader | None = None,
    nucleus_detector: NucleusDetector | None = None,
    analyzer: GFAPAnalyzer | None = None,
    debug_handler: PreparedRunHandler | None = None,
    fiji_handler: PreparedRunHandler | None = None,
    stage_reporter: GFAPStageReporter | None = None,
) -> GFAPOnlyPipelineResult:
    """Execute the DAPI/GFAP scientific branch and hand off to shared runtime.

    ``measurement_projection_loader`` is deliberately called only after every
    ROI has been defined.  It receives the inclusive Z bounds and projection
    name, allowing the controller to read only selected untouched target
    slices.  Neither its pixels nor its return value are passed to the analyzer.
    """

    validate_gfap_only_route(
        available_channels,
        egfp_is_valid=egfp_is_valid,
    )
    dapi, gfap = _validate_stack_pair(dapi_stack, gfap_stack)
    height_um = _positive_finite(pixel_height_um, "pixel_height_um")
    width_um = _positive_finite(pixel_width_um, "pixel_width_um")
    spacing_um = _positive_finite(z_spacing_um, "z_spacing_um")
    stage_timings: dict[str, float] = {}

    _report_stage(stage_reporter, "z_selection", "started")
    stage_started = perf_counter()
    selection = select_gfap_active_z(gfap, spacing_um, config=z_config)
    stage_timings["z_selection"] = perf_counter() - stage_started
    _report_stage(
        stage_reporter,
        "z_selection",
        "completed",
        stage_timings["z_selection"],
    )

    _report_stage(stage_reporter, "dapi_nucleus_model", "started")
    stage_started = perf_counter()
    detector = nucleus_detector or _default_nucleus_detector
    detected = detector(
        dapi,
        height_um,
        width_um,
        z_indices=selection.indices,
    )
    if not hasattr(detected, "labels_zyx"):
        raise TypeError("GFAP-only nucleus detector must return labels_zyx")
    labels_zyx = np.asarray(detected.labels_zyx)
    selected_gfap = gfap[
        selection.start_0based : selection.end_0based_inclusive + 1
    ]
    if labels_zyx.shape != selected_gfap.shape:
        raise RuntimeError(
            "InstanSeg DAPI candidate labels do not match the selected GFAP stack: "
            f"{labels_zyx.shape} versus {selected_gfap.shape}"
        )
    stage_timings["dapi_nucleus_model"] = perf_counter() - stage_started
    _report_stage(
        stage_reporter,
        "dapi_nucleus_model",
        "completed",
        stage_timings["dapi_nucleus_model"],
    )

    _report_stage(stage_reporter, "gfap_compartments", "started")
    stage_started = perf_counter()
    analyze = analyzer or _default_gfap_analyzer
    active_analysis_config = analysis_config or _default_analysis_config()
    analysis = analyze(
        labels_zyx,
        selected_gfap,
        (height_um, width_um),
        spacing_um,
        config=active_analysis_config,
    )
    _validate_analysis_partition(
        analysis,
        quality_config or GFAPOnlyQualityConfig(),
    )
    stage_timings["gfap_compartments"] = perf_counter() - stage_started
    _report_stage(
        stage_reporter,
        "gfap_compartments",
        "completed",
        stage_timings["gfap_compartments"],
    )

    _report_stage(stage_reporter, "measurement_preparation", "started")
    stage_started = perf_counter()
    measurement_projection = None
    if measurement_projection_loader is not None:
        measurement_projection = np.asarray(
            measurement_projection_loader(
                selection.start_0based,
                selection.end_0based_inclusive,
                selection.projection,
            )
        )
        if measurement_projection.shape != dapi.shape[1:]:
            raise ValueError(
                "Measurement projection shape differs from the ROI geometry: "
                f"{measurement_projection.shape} versus {dapi.shape[1:]}"
            )

    dapi_projection = _project_selected(dapi, selection)
    gfap_projection = _project_selected(gfap, selection)
    detector_details = {
        "model": "InstanSeg single-channel nuclei candidate generator",
        "z_indices": list(selection.indices),
        "instance_counts": [
            int(value) for value in getattr(detected, "instance_counts", ())
        ],
        "model_sha256": str(getattr(detected, "model_sha256", "")),
        "ownership_decision_source": "GFAP association after 3D DAPI linking",
    }
    prepared = GFAPOnlyPreparedRun(
        analysis=analysis,
        z_selection=selection,
        dapi_projection=dapi_projection,
        gfap_projection=gfap_projection,
        measurement_projection=measurement_projection,
        nucleus_detection=detector_details,
        stage_timings_seconds=stage_timings,
    )
    stage_timings["measurement_preparation"] = perf_counter() - stage_started
    _report_stage(
        stage_reporter,
        "measurement_preparation",
        "completed",
        stage_timings["measurement_preparation"],
    )

    # Fiji may remain open for manual review for many minutes.  The prepared
    # handoff contains only 2D projections/labels, so release every 3D
    # reference owned by this frame before invoking a handler.  The caller may
    # independently retain its own stacks; this block guarantees that this
    # pipeline does not extend their lifetime.
    del detected, labels_zyx, selected_gfap
    del dapi, gfap, dapi_stack, gfap_stack
    import gc

    gc.collect()

    debug_result = None
    if (bool(debug) or bool(skip_fiji)) and debug_handler is not None:
        debug_result = debug_handler(prepared)
    if skip_fiji:
        return GFAPOnlyPipelineResult(
            prepared=prepared,
            skip_fiji=True,
            debug_result=debug_result,
        )
    if fiji_handler is None:
        raise ValueError(
            "fiji_handler is required when skip_fiji is false; the GFAP-only "
            "pipeline will not silently bypass review and measurement"
        )
    _report_stage(stage_reporter, "fiji_review_and_publication", "started")
    stage_started = perf_counter()
    fiji_result = fiji_handler(prepared)
    stage_timings["fiji_review_and_publication"] = perf_counter() - stage_started
    _report_stage(
        stage_reporter,
        "fiji_review_and_publication",
        "completed",
        stage_timings["fiji_review_and_publication"],
    )
    return GFAPOnlyPipelineResult(
        prepared=prepared,
        skip_fiji=False,
        debug_result=debug_result,
        fiji_result=fiji_result,
    )


__all__ = [
    "GFAPOnlyPipelineResult",
    "GFAPOnlyPreparedRun",
    "GFAPOnlyQualityConfig",
    "GFAPStageReporter",
    "GFAPZSelection",
    "GFAPZSelectionConfig",
    "run_gfap_only_pipeline",
    "select_gfap_active_z",
    "validate_gfap_only_route",
]
