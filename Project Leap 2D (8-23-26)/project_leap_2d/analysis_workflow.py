"""Analysis-mode routing and Fiji Cell Edit integration around the shared runtime."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Literal, Mapping, Sequence

import numpy as np

from .runtime_attributes import temporary_runtime_attributes


@dataclass
class _CellEditCapture:
    analysis_mode: str = "egfp"
    structural_channel: str = "eGFP"
    dapi_projection: np.ndarray | None = None
    structural_map: np.ndarray | None = None
    whole_labels: np.ndarray | None = None
    soma_labels: np.ndarray | None = None
    process_labels: np.ndarray | None = None
    compartment_metrics: dict[str, Any] | None = None
    pixel_width_um: float | None = None
    pixel_height_um: float | None = None
    pixel_depth_um: float | None = None
    age_profile: str | None = None
    context_paths: Any = None
    run_dir: Path | None = None

    def release_image_evidence(self) -> None:
        self.dapi_projection = None
        self.structural_map = None
        self.whole_labels = None
        self.soma_labels = None
        self.process_labels = None
        self.compartment_metrics = None


def select_analysis_route(
    available_channels: Collection[str],
) -> Literal["egfp", "gfap_only"]:
    """Select from channel names only; model modules remain lazily imported."""

    channels = {str(channel) for channel in available_channels}
    if "DAPI" not in channels:
        raise ValueError("Analysis requires a DAPI channel")
    if "eGFP" in channels:
        return "egfp"
    if "GFAP" in channels:
        return "gfap_only"
    raise ValueError("Analysis requires eGFP or GFAP structural fluorescence")


def _argument_path(argv: Sequence[str], option: str) -> Path:
    values = list(argv)
    for index, value in enumerate(values):
        if value == option:
            if index + 1 >= len(values):
                break
            return Path(values[index + 1]).expanduser().resolve()
        prefix = f"{option}="
        if value.startswith(prefix):
            return Path(value[len(prefix) :]).expanduser().resolve()
    raise ValueError(f"Analysis workflow requires {option}")


def _canonical_nucleus_records(
    compartment_metrics: Mapping[str, Any],
    canonical_core_labels: np.ndarray,
    canonical_extent_labels: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    """Prefer calibrated 3D records; keep the 2D fallback explicitly uncertain."""

    core = np.asarray(canonical_core_labels)
    extent = np.asarray(canonical_extent_labels)
    if core.ndim != 2 or core.shape != extent.shape:
        raise ValueError("Canonical Cell Edit nucleus maps must be matching 2D arrays")
    instance_ids = tuple(
        sorted(int(value) for value in np.unique(extent) if int(value) > 0)
    )
    inventory_metrics = compartment_metrics.get("nucleus_3d_inventory", {})
    inventory_records = (
        inventory_metrics.get("canonical_per_nucleus", ())
        if isinstance(inventory_metrics, Mapping)
        else ()
    )
    if isinstance(inventory_records, (list, tuple)) and inventory_records:
        records_by_id: dict[int, dict[str, Any]] = {}
        for source_record in inventory_records:
            if not isinstance(source_record, Mapping):
                continue
            record = dict(source_record)
            instance_id = int(
                record.get(
                    "instance_id",
                    record.get("nucleus_id_2d", record.get("object_id_3d", 0)),
                )
            )
            if instance_id < 1:
                continue
            record["instance_id"] = instance_id
            records_by_id[instance_id] = record
        if instance_ids and all(
            instance_id in records_by_id for instance_id in instance_ids
        ):
            return tuple(records_by_id[instance_id] for instance_id in instance_ids)

    fallback: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        core_area = int((core == instance_id).sum())
        extent_area = int((extent == instance_id).sum())
        fallback.append(
            {
                "instance_id": instance_id,
                "object_id": instance_id,
                "nucleus_id_2d": instance_id,
                "object_id_3d": instance_id,
                "accepted": False,
                "dapi_valid": bool(core_area > 0 and extent_area >= core_area),
                "identity_status": "projection_only",
                "resolution": "projection_only",
                "z_min_0based": None,
                "z_max_0based_inclusive": None,
                "projection_area_px": extent_area,
                "extent_area_px": extent_area,
            }
        )
    return tuple(fallback)


def _capture_compartment_result(
    capture: _CellEditCapture,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: tuple[Any, Any, Any, Any],
) -> None:
    if len(args) < 7 or len(result) != 4:
        raise RuntimeError(
            "Frozen compartment call no longer matches the Cell Edit adapter"
        )
    whole, soma, processes, metrics = result
    neonatal_context = kwargs.get("neonatal_3d_context")
    capture.dapi_projection = np.asarray(args[1])
    capture.structural_map = np.asarray(args[2])
    capture.whole_labels = np.asarray(whole)
    capture.soma_labels = np.asarray(soma)
    capture.process_labels = np.asarray(processes)
    capture.compartment_metrics = dict(metrics)
    capture.pixel_width_um = float(args[4])
    capture.pixel_height_um = float(args[5])
    capture.age_profile = str(args[6])
    capture.pixel_depth_um = (
        None
        if neonatal_context is None
        else float(neonatal_context.pixel_depth_um)
    )


def _cleanup_context_artifacts(
    capture: _CellEditCapture,
    *,
    remove_runtime_context: bool,
) -> None:
    """Remove final context evidence only after a finished workflow."""

    if not remove_runtime_context:
        return
    if capture.context_paths is not None:
        for attribute in ("npz_path", "json_path"):
            Path(getattr(capture.context_paths, attribute)).unlink(missing_ok=True)
    if capture.run_dir is not None:
        runtime_root = capture.run_dir / "cell_edit"
        for filename in ("analysis_context.npz", "analysis_context.json"):
            (runtime_root / filename).unlink(missing_ok=True)


def _build_context_and_prepare_fiji(
    *,
    capture: _CellEditCapture,
    base_prepare: Callable[..., tuple[Path, Path]],
    prepare_kwargs: dict[str, Any],
) -> tuple[Path, Path]:
    from .fiji_review.cell_edit_context import build_cell_edit_context
    from .fiji_review.cell_edit_fiji_bridge import prepare_cell_edit_fiji_runtime

    metrics = capture.compartment_metrics
    if (
        metrics is None
        or capture.dapi_projection is None
        or capture.structural_map is None
        or capture.whole_labels is None
        or capture.soma_labels is None
        or capture.process_labels is None
    ):
        result = prepare_cell_edit_fiji_runtime(
            base_prepare=base_prepare,
            cell_edit_context_builder=None,
            **prepare_kwargs,
        )
        capture.run_dir = Path(result[0])
        return result

    shape = capture.whole_labels.shape
    core = np.asarray(
        metrics.get(
            "_canonical_nucleus_instance_core_labels_2d",
            np.zeros(shape, dtype=np.uint32),
        ),
        dtype=np.uint32,
    )
    extent = np.asarray(
        metrics.get(
            "_canonical_nucleus_instance_extent_labels_2d",
            np.zeros(shape, dtype=np.uint32),
        ),
        dtype=np.uint32,
    )
    nucleus_records = _canonical_nucleus_records(metrics, core, extent)
    context_enabled = bool(nucleus_records) and bool(np.any(extent > 0))
    if not context_enabled:
        capture.release_image_evidence()
        result = prepare_cell_edit_fiji_runtime(
            base_prepare=base_prepare,
            cell_edit_context_builder=None,
            **prepare_kwargs,
        )
        capture.run_dir = Path(result[0])
        return result

    paths = prepare_kwargs["paths"]
    metadata = prepare_kwargs["metadata"]
    structural_channels = tuple(prepare_kwargs["structural_channels"])
    best_row = prepare_kwargs["best_row"]
    def build_final_context(context_dir: Path):
        capture.context_paths = build_cell_edit_context(
            run_dir=context_dir,
            basename="analysis_context",
            dapi_path=paths["DAPI"],
            structural_paths={
                channel: paths[channel] for channel in structural_channels
            },
            dapi_projection=capture.dapi_projection,
            structural_map=capture.structural_map,
            selected_z={
                "z_start_1based": int(best_row["z_start_1based"]),
                "z_end_1based_inclusive": int(
                    best_row["z_end_1based_inclusive"]
                ),
                "projection": str(best_row["projection"]),
            },
            calibration={
                "pixel_width_um": capture.pixel_width_um,
                "pixel_height_um": capture.pixel_height_um,
                "pixel_depth_um": (
                    capture.pixel_depth_um
                    if capture.pixel_depth_um is not None
                    else metadata["DAPI"].get("pixel_depth_um")
                ),
                "pixel_width_source": metadata["DAPI"].get("pixel_width_source"),
                "pixel_height_source": metadata["DAPI"].get("pixel_height_source"),
                "pixel_depth_source": metadata["DAPI"].get("pixel_depth_source"),
                "unit": "um",
            },
            age_profile=str(capture.age_profile),
            analysis_mode=capture.analysis_mode,
            structural_channel=capture.structural_channel,
            canonical_core_labels=core,
            canonical_extent_labels=extent,
            nucleus_records=nucleus_records,
            initial_triplet={
                "whole_labels": capture.whole_labels,
                "soma_labels": capture.soma_labels,
                "process_labels": capture.process_labels,
            },
        )
        return capture.context_paths

    result = prepare_cell_edit_fiji_runtime(
        base_prepare=base_prepare,
        cell_edit_context_builder=build_final_context,
        cleanup_unlaunched_run_on_failure=True,
        **prepare_kwargs,
    )
    capture.run_dir = Path(result[0])
    capture.release_image_evidence()
    return result


def _run_egfp_structural(runtime, argv: Sequence[str]) -> int:
    """Run the shared analysis with temporary, exception-safe Cell Edit adapters."""

    from .fiji_review.cell_edit_fiji_bridge import launch_cell_edit_fiji_workflow

    capture = _CellEditCapture()
    base_split = runtime.split_astrocyte_compartments_for_profile
    base_prepare = runtime.prepare_fiji_runtime
    base_launch = runtime.launch_fiji_workflow

    def capture_split(*args, **kwargs):
        result = base_split(*args, **kwargs)
        _capture_compartment_result(capture, args, kwargs, result)
        return result

    def cell_edit_prepare(*args, **kwargs):
        if args:
            raise TypeError("Fiji runtime preparation must use named arguments")
        return _build_context_and_prepare_fiji(
            capture=capture,
            base_prepare=base_prepare,
            prepare_kwargs=dict(kwargs),
        )

    def cell_edit_launch(*args, **kwargs):
        if capture.context_paths is None:
            return base_launch(*args, **kwargs)
        return launch_cell_edit_fiji_workflow(*args, **kwargs)

    workflow_finished = False
    try:
        with temporary_runtime_attributes(
            runtime,
            split_astrocyte_compartments_for_profile=capture_split,
            prepare_fiji_runtime=cell_edit_prepare,
            launch_fiji_workflow=cell_edit_launch,
        ):
            try:
                return_code = int(runtime.main(list(argv)))
            except Exception:
                if capture.run_dir is not None:
                    from .fiji_review.failed_run_retention import (
                        retain_only_latest_failed_fiji_run,
                    )

                    try:
                        retain_only_latest_failed_fiji_run(capture.run_dir)
                    except (OSError, ValueError):
                        pass
                raise
            workflow_finished = True
            return return_code
    finally:
        _cleanup_context_artifacts(
            capture,
            remove_runtime_context=workflow_finished,
        )
        capture.release_image_evidence()


def _run_gfap_only(runtime, argv: Sequence[str], channel_paths: Mapping[str, Path]) -> int:
    """Run the independent DAPI+GFAP route with shared Fiji publication."""

    from .analysis_modes.gfap_only.gfap_only_pipeline import run_gfap_only_pipeline

    paths = dict(channel_paths)
    age_decision = runtime.detect_filename_age_profile(paths)
    if age_decision is not None and age_decision.profile != "mature":
        raise ValueError(
            "This release supports GFAP-only analysis for mature astrocytes "
            "only; neonatal GFAP-only analysis is not supported."
        )

    args = runtime.parse_args(list(argv))
    if args.fiji_timeout_minutes <= 0:
        raise ValueError("--fiji-timeout-minutes must be positive")
    if args.dapi_fragment_workload_preflight_only:
        raise ValueError(
            "The DAPI workload preflight is available only for the eGFP route"
        )
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl 3.1.5 is required before analysis starts"
        ) from exc

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {channel: runtime.read_meta(path) for channel, path in paths.items()}
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
    runtime.validate_shared_geometry(metadata)
    calibration = metadata["DAPI"]
    if any(
        calibration.get(key) is None or float(calibration[key]) <= 0
        for key in ("pixel_width_um", "pixel_height_um", "pixel_depth_um")
    ):
        raise ValueError(
            "GFAP-only analysis requires positive calibrated X, Y, and Z spacing"
        )

    measurement = runtime.measurement_channel(paths)
    dapi_stack = runtime.load_stack(paths["DAPI"])
    gfap_stack = runtime.load_stack(paths["GFAP"])

    def measurement_loader(z0: int, z1: int, projection: str) -> np.ndarray:
        selected = runtime.tf.imread(
            str(paths[measurement]),
            key=range(int(z0), int(z1) + 1),
        )
        return runtime.project(
            selected,
            0,
            int(selected.shape[0]) - 1,
            projection,
        )

    def debug_handler(prepared):
        return _write_gfap_debug_outputs(
            runtime=runtime,
            prepared=prepared,
            output_dir=output_dir,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            measurement=measurement,
        )

    def fiji_handler(prepared):
        return _run_gfap_fiji_workflow(
            runtime=runtime,
            prepared=prepared,
            args=args,
            output_dir=output_dir,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            measurement=measurement,
        )

    runtime.print_terminal_stage(
        "GFAP-ONLY | DAPI NUCLEI AND GFAP STRUCTURE",
        "Independent mode: DAPI defines candidate nuclei; GFAP association "
        "defines astrocyte structure; the measurement channel is excluded from ROI definition.",
    )

    stage_titles = {
        "z_selection": "Z SELECTION",
        "dapi_nucleus_model": "DAPI NUCLEUS MODEL",
        "gfap_compartments": "GFAP STRUCTURE AND COMPARTMENTS",
        "measurement_preparation": "MEASUREMENT PREPARATION",
        "fiji_review_and_publication": "FIJI REVIEW AND PUBLICATION",
    }

    def stage_reporter(
        stage: str,
        status: str,
        elapsed_seconds: float | None,
    ) -> None:
        title = stage_titles.get(stage, stage.replace("_", " ").upper())
        if status == "started":
            runtime.print_terminal_stage(f"GFAP-ONLY | {title}", "Started.")
            return
        print(
            f"GFAP-only | {title} | completed "
            f"| elapsed={float(elapsed_seconds or 0.0):.3f} s",
            flush=True,
        )

    result = run_gfap_only_pipeline(
        available_channels=paths.keys(),
        egfp_is_valid=False,
        dapi_stack=dapi_stack,
        gfap_stack=gfap_stack,
        pixel_height_um=float(calibration["pixel_height_um"]),
        pixel_width_um=float(calibration["pixel_width_um"]),
        z_spacing_um=float(calibration["pixel_depth_um"]),
        skip_fiji=bool(args.skip_fiji),
        debug=bool(args.skip_fiji),
        measurement_projection_loader=measurement_loader,
        debug_handler=debug_handler,
        fiji_handler=fiji_handler,
        stage_reporter=stage_reporter,
    )
    del dapi_stack, gfap_stack
    return 0


def _gfap_nucleus_records(prepared) -> tuple[dict[str, Any], ...]:
    analysis = prepared.analysis
    diagnostics = analysis.diagnostics
    source_to_display = {
        int(source): int(display)
        for source, display in diagnostics.get(
            "source_owner_to_display_id",
            {},
        ).items()
    }
    inventory_by_source = {
        int(record["nucleus_id"]): record
        for record in diagnostics.get("nucleus_inventory_records", [])
    }
    projected_source_ids = tuple(
        int(value)
        for value in np.unique(analysis.valid_nucleus_labels_2d)
        if int(value) > 0
    )
    records: list[dict[str, Any]] = []
    for source_id in projected_source_ids:
        inventory = inventory_by_source.get(source_id, {})
        if not bool(inventory.get("valid_3d_nucleus", False)):
            raise RuntimeError(
                "GFAP Cell Edit nucleus projection contains a nucleus "
                "without valid 3D DAPI identity"
            )
        display_id = source_to_display.get(source_id)
        z_first = inventory.get("z_first")
        z_last = inventory.get("z_last")
        record = {
            "instance_id": source_id,
            "accepted": display_id is not None,
            "dapi_valid": True,
            "identity_status": "resolved",
            "z_min_0based": (
                None
                if z_first is None
                else int(prepared.z_selection.start_0based) + int(z_first)
            ),
            "z_max_0based_inclusive": (
                None
                if z_last is None
                else int(prepared.z_selection.start_0based) + int(z_last)
            ),
            "source": (
                "gfap_only_3d_dapi_owner"
                if display_id is not None
                else "gfap_only_3d_dapi_nonowner"
            ),
        }
        if display_id is not None:
            record["owner_display_id"] = display_id
        records.append(record)
    return tuple(records)


def _gfap_capture(prepared, metadata: Mapping[str, Any]) -> _CellEditCapture:
    analysis = prepared.analysis
    nucleus_labels = np.asarray(
        analysis.valid_nucleus_labels_2d,
        dtype=np.uint32,
    )
    return _CellEditCapture(
        analysis_mode="gfap_only",
        structural_channel="GFAP",
        dapi_projection=np.asarray(prepared.dapi_projection),
        structural_map=np.asarray(
            analysis.gfap_structural_score,
            dtype=np.float32,
        ),
        whole_labels=np.asarray(analysis.whole_labels),
        soma_labels=np.asarray(analysis.soma_labels),
        process_labels=np.asarray(analysis.process_labels),
        compartment_metrics={
            "_canonical_nucleus_instance_core_labels_2d": nucleus_labels,
            "_canonical_nucleus_instance_extent_labels_2d": nucleus_labels,
            "nucleus_3d_inventory": {
                "canonical_per_nucleus": list(_gfap_nucleus_records(prepared)),
            },
        },
        pixel_width_um=float(metadata["DAPI"]["pixel_width_um"]),
        pixel_height_um=float(metadata["DAPI"]["pixel_height_um"]),
        pixel_depth_um=float(metadata["DAPI"]["pixel_depth_um"]),
        age_profile="mature",
    )


def _draw_gfap_contours(runtime, rgb, labels, color: tuple[int, int, int]):
    overlay = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    boundary = runtime.segmentation.find_boundaries(
        np.asarray(labels),
        mode="outer",
    )
    boundary = runtime.morphology.binary_dilation(
        boundary,
        footprint=runtime.morphology.disk(1),
    )
    overlay[boundary] = np.asarray(color, dtype=np.uint8)
    return overlay


def _write_gfap_report(
    path: Path,
    *,
    prepared,
    input_dir: Path,
    paths: Mapping[str, Path],
    metadata: Mapping[str, Any],
    measurement: str,
    fiji_status: str,
    fiji_details: Mapping[str, Any] | None = None,
) -> None:
    analysis = prepared.analysis
    timings = dict(getattr(prepared, "stage_timings_seconds", {}) or {})
    inference_elapsed = sum(
        float(timings.get(stage, 0.0))
        for stage in (
            "z_selection",
            "dapi_nucleus_model",
            "gfap_compartments",
        )
    )
    stage_labels = (
        ("z_selection", "Z selection"),
        ("dapi_nucleus_model", "DAPI nucleus model"),
        ("gfap_compartments", "GFAP structure and compartments"),
        ("measurement_preparation", "Measurement preparation"),
    )
    lines = [
        "PROJECT LEAP 2D - DAPI + GFAP ONLY ANALYSIS",
        "============================================",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Input Folder: {input_dir}",
        f"DAPI: {paths['DAPI'].name}",
        f"GFAP: {paths['GFAP'].name}",
        f"Measurement: {measurement} ({paths[measurement].name})",
        "Measurement Channel Used for ROI Definition: No",
        "ROI Definition Channels: DAPI + GFAP",
        "Supported GFAP-only Age Profile: mature",
        f"Image Geometry: ZYX {tuple(metadata['DAPI']['shape'])}",
        "Pixel Calibration (um): "
        f"X={metadata['DAPI']['pixel_width_um']}, "
        f"Y={metadata['DAPI']['pixel_height_um']}, "
        f"Z={metadata['DAPI']['pixel_depth_um']}",
        "Selected Z (1-based inclusive): "
        f"{prepared.z_selection.start_1based}-"
        f"{prepared.z_selection.end_1based_inclusive}",
        f"Projection: {prepared.z_selection.projection}",
        f"Astrocyte ROI Count: {int(analysis.whole_labels.max())}",
        f"Whole Area (px): {int((analysis.whole_labels > 0).sum())}",
        f"Soma Area (px): {int((analysis.soma_labels > 0).sum())}",
        f"Processes Area (px): {int((analysis.process_labels > 0).sum())}",
        f"Nucleus Model: {prepared.nucleus_detection.get('model', 'unknown')}",
        f"Nucleus Model SHA256: {prepared.nucleus_detection.get('model_sha256', '')}",
        f"Automated Inference Elapsed (s): {inference_elapsed:.3f}",
        f"Fiji Status: {fiji_status}",
        "",
        "Major Stage Status and Elapsed Time:",
        *[
            f"- {label}: Completed ({float(timings.get(stage, 0.0)):.3f} s)"
            for stage, label in stage_labels
        ],
        "",
        "Biological Boundary:",
        "GFAP does not need to enclose the DAPI nucleus. The analysis retains only "
        "DAPI nuclei with validated three-dimensional identity and connected GFAP support.",
        "This release supports mature-astrocyte GFAP-only analysis only. "
        "Neonatal GFAP-only inputs are rejected before image or model loading.",
    ]
    fiji_elapsed = timings.get("fiji_review_and_publication")
    if fiji_status.startswith("Pending"):
        lines.insert(
            lines.index("Biological Boundary:") - 1,
            "- Fiji review and publication: Pending",
        )
    elif fiji_status.startswith("Skipped"):
        lines.insert(
            lines.index("Biological Boundary:") - 1,
            "- Fiji review and publication: Skipped",
        )
    elif fiji_elapsed is not None:
        lines.insert(
            lines.index("Biological Boundary:") - 1,
            "- Fiji review and publication: "
            f"Completed ({float(fiji_elapsed):.3f} s)",
        )
    else:
        lines.insert(
            lines.index("Biological Boundary:") - 1,
            f"- Fiji review and publication: {fiji_status}",
        )
    if fiji_details is not None:
        lines.extend(
            [
                "",
                "Fiji Review:",
                f"Manual Review Used: {bool(fiji_details.get('manual_review_used', False))}",
                f"Final ROI Count: {int(fiji_details.get('roi_count', -1))}",
                f"Review Audit Events: {len(fiji_details.get('review_audit', []))}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gfap_debug_outputs(
    *,
    runtime,
    prepared,
    output_dir: Path,
    input_dir: Path,
    paths: Mapping[str, Path],
    metadata: Mapping[str, Any],
    measurement: str,
) -> dict[str, Path]:
    analysis = prepared.analysis
    composite = runtime.make_fiji_like_composite(
        prepared.dapi_projection,
        {"GFAP": prepared.gfap_projection},
        prepared.measurement_projection,
    )
    outputs = {
        "whole": output_dir / runtime.DEBUG_WHOLE_OVERLAY_FILENAME,
        "soma": output_dir / runtime.DEBUG_SOMA_OVERLAY_FILENAME,
        "processes": output_dir / runtime.DEBUG_PROCESS_OVERLAY_FILENAME,
        "state": output_dir / runtime.DEBUG_STATE_FILENAME,
        "report": output_dir / runtime.DEBUG_REPORT_FILENAME,
    }
    runtime.Image.fromarray(
        _draw_gfap_contours(
            runtime,
            composite,
            analysis.whole_labels,
            (255, 0, 255),
        )
    ).save(outputs["whole"])
    runtime.Image.fromarray(
        _draw_gfap_contours(
            runtime,
            composite,
            analysis.soma_labels,
            (0, 255, 255),
        )
    ).save(outputs["soma"])
    runtime.Image.fromarray(
        _draw_gfap_contours(
            runtime,
            composite,
            analysis.process_labels,
            (255, 255, 255),
        )
    ).save(outputs["processes"])
    np.savez_compressed(
        outputs["state"],
        whole_mask=(analysis.whole_labels > 0).astype(np.uint8),
        candidate_whole_mask=(analysis.whole_labels > 0).astype(np.uint8),
        whole_labels=np.asarray(analysis.whole_labels, dtype=np.uint16),
        soma_labels=np.asarray(analysis.soma_labels, dtype=np.uint16),
        process_labels=np.asarray(analysis.process_labels, dtype=np.uint16),
        canonical_nucleus_instance_core_labels_2d=np.asarray(
            analysis.valid_nucleus_labels_2d,
            dtype=np.uint32,
        ),
        canonical_nucleus_instance_extent_labels_2d=np.asarray(
            analysis.valid_nucleus_labels_2d,
            dtype=np.uint32,
        ),
        dapi_projection=np.asarray(prepared.dapi_projection),
        structural_map=np.asarray(
            analysis.gfap_structural_score,
            dtype=np.float32,
        ),
        measurement_projection=np.asarray(prepared.measurement_projection),
        pixel_width_um=np.asarray(metadata["DAPI"]["pixel_width_um"]),
        pixel_height_um=np.asarray(metadata["DAPI"]["pixel_height_um"]),
        pixel_depth_um=np.asarray(metadata["DAPI"]["pixel_depth_um"]),
        age_profile=np.asarray("mature"),
        analysis_mode=np.asarray("dapi_gfap_only"),
    )
    _write_gfap_report(
        outputs["report"],
        prepared=prepared,
        input_dir=input_dir,
        paths=paths,
        metadata=metadata,
        measurement=measurement,
        fiji_status="Skipped by --skip-fiji; debug previews only",
    )
    for path in outputs.values():
        print(f"GFAP-only debug output: {path}", flush=True)
    return outputs


def _run_gfap_fiji_workflow(
    *,
    runtime,
    prepared,
    args,
    output_dir: Path,
    input_dir: Path,
    paths: Mapping[str, Path],
    metadata: Mapping[str, Any],
    measurement: str,
) -> dict[str, Any]:
    from .fiji_review.cell_edit_fiji_bridge import launch_cell_edit_fiji_workflow
    from .fiji_review.measurement_result_validation import (
        validate_fiji_measurement_result,
    )

    analysis = prepared.analysis
    fiji_stage_started = time.perf_counter()
    capture = _gfap_capture(prepared, metadata)
    run_dir: Path | None = None
    workflow_finished = False
    try:
        run_dir, manifest_path = _build_context_and_prepare_fiji(
            capture=capture,
            base_prepare=runtime.prepare_fiji_runtime,
            prepare_kwargs={
                "output_dir": output_dir,
                "paths": dict(paths),
                "metadata": dict(metadata),
                "structural_channels": ["GFAP"],
                "measurement": measurement,
                "best_row": prepared.best_row,
                "whole_labels": analysis.whole_labels,
                "soma_labels": analysis.soma_labels,
                "process_labels": analysis.process_labels,
                "selected_projections": {
                    "DAPI": prepared.dapi_projection,
                    "GFAP": prepared.gfap_projection,
                    measurement: prepared.measurement_projection,
                },
                "auto_continue": bool(args.fiji_auto_continue),
            },
        )
        _write_gfap_report(
            run_dir / "analysis_report_pending.txt",
            prepared=prepared,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            measurement=measurement,
            fiji_status="Pending Fiji review and raw-grayscale measurement",
        )
        launcher = runtime.find_fiji_launcher(args.fiji_launcher)
        fiji_details = launch_cell_edit_fiji_workflow(
            launcher=launcher,
            run_dir=run_dir,
            manifest_path=manifest_path,
            timeout_minutes=float(args.fiji_timeout_minutes),
        )
        if bool(fiji_details.get("cancelled", False)):
            shutil.rmtree(run_dir, ignore_errors=True)
            workflow_finished = True
            return fiji_details
        overlays = validate_fiji_measurement_result(
            fiji_details=fiji_details,
            run_dir=run_dir,
            image_shape_yx=tuple(int(value) for value in reference_shape_yx(metadata)),
            pixel_width_um=float(metadata["DAPI"]["pixel_width_um"]),
            pixel_height_um=float(metadata["DAPI"]["pixel_height_um"]),
        )
        workbook = run_dir / runtime.WORKBOOK_FILENAME
        runtime.write_measurement_workbook(
            workbook,
            fiji_details=fiji_details,
            measurement=measurement,
            best_row=prepared.best_row,
            age_profile="mature",
        )
        runtime.validate_measurement_workbook(workbook, fiji_details)
        prepared.stage_timings_seconds["fiji_review_and_publication"] = (
            time.perf_counter() - fiji_stage_started
        )
        report = run_dir / "analysis_report_completed.txt"
        _write_gfap_report(
            report,
            prepared=prepared,
            input_dir=input_dir,
            paths=paths,
            metadata=metadata,
            measurement=measurement,
            fiji_status="Completed after Fiji review and native measurement",
            fiji_details=fiji_details,
        )
        runtime.publish_output_bundle(
            staged_files={
                "whole": overlays["whole"],
                "processes": overlays["processes"],
                "soma": overlays["soma"],
                "report": report,
                "workbook": workbook,
            },
            final_files={
                "whole": output_dir / runtime.WHOLE_OVERLAY_FILENAME,
                "processes": output_dir / runtime.PROCESS_OVERLAY_FILENAME,
                "soma": output_dir / runtime.SOMA_OVERLAY_FILENAME,
                "report": output_dir / runtime.REPORT_FILENAME,
                "workbook": output_dir / runtime.WORKBOOK_FILENAME,
            },
            run_dir=run_dir,
        )
        shutil.rmtree(run_dir, ignore_errors=True)
        workflow_finished = True
        return fiji_details
    except Exception as exc:
        if run_dir is not None:
            try:
                _write_gfap_report(
                    run_dir / "analysis_report_failed.txt",
                    prepared=prepared,
                    input_dir=input_dir,
                    paths=paths,
                    metadata=metadata,
                    measurement=measurement,
                    fiji_status=f"Failed: {exc!r}; previous outputs were preserved",
                )
                from .fiji_review.failed_run_retention import (
                    retain_only_latest_failed_fiji_run,
                )

                try:
                    retain_only_latest_failed_fiji_run(run_dir)
                except (OSError, ValueError):
                    pass
                print(f"Failure diagnostics retained at: {run_dir}", flush=True)
            except Exception:
                pass
        raise
    finally:
        _cleanup_context_artifacts(
            capture,
            remove_runtime_context=workflow_finished,
        )


def reference_shape_yx(metadata: Mapping[str, Any]) -> tuple[int, int]:
    shape = tuple(int(value) for value in metadata["DAPI"]["shape"])
    if len(shape) != 3:
        raise ValueError("DAPI metadata must describe a ZYX image")
    return shape[1], shape[2]


def run_analysis_workflow(runtime, argv: Sequence[str]) -> int:
    input_dir = _argument_path(argv, "--input-dir")
    channel_paths, _ignored = runtime.discover_channel_paths(input_dir)
    if select_analysis_route(channel_paths) == "gfap_only":
        return _run_gfap_only(runtime, argv, channel_paths)
    return _run_egfp_structural(runtime, argv)
