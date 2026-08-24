# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def prepare_fiji_runtime(
    *,
    output_dir: Path,
    paths: dict[str, Path],
    metadata: dict[str, dict],
    structural_channels: list[str],
    measurement: str,
    best_row: dict,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    selected_projections: dict[str, np.ndarray],
    auto_continue: bool,
) -> tuple[Path, Path]:
    cache_root = Path.home() / "Library" / "Caches" / "IHC2DAnalysis"
    cache_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old_dir in cache_root.glob("run-*"):
        try:
            if old_dir.is_dir() and now - old_dir.stat().st_mtime > 2 * 24 * 3600:
                shutil.rmtree(old_dir)
        except OSError:
            pass
    run_dir = cache_root / f"run-{uuid.uuid4().hex}"
    run_dir.mkdir()

    roi_count = int(whole_labels.max())
    if roi_count < 1:
        raise ValueError("The selected candidate contains no ROI")
    expected_ids = set(range(1, roi_count + 1))
    observed_ids = {
        key: set(int(value) for value in np.unique(labels) if int(value) > 0)
        for key, labels in (
            ("whole", whole_labels),
            ("soma", soma_labels),
            ("processes", process_labels),
        )
    }
    if any(ids != expected_ids for ids in observed_ids.values()):
        raise ValueError(
            "Whole, Soma, and Processes label IDs must match exactly: "
            f"{observed_ids}"
        )
    if np.any((soma_labels > 0) & (whole_labels != soma_labels)):
        raise ValueError("A Soma label is outside or assigned to a different Whole Astrocyte ID")
    if np.any((process_labels > 0) & (whole_labels != process_labels)):
        raise ValueError("A Processes label is outside or assigned to a different Whole Astrocyte ID")
    partition_count = (soma_labels > 0).astype(np.uint8) + (process_labels > 0).astype(np.uint8)
    if np.any(partition_count[whole_labels > 0] != 1) or np.any(partition_count[whole_labels == 0] != 0):
        raise ValueError("Soma and Processes do not form an exact partition of the Whole labels")
    label_paths = {
        "whole": run_dir / "whole_roi_labels.tif",
        "soma": run_dir / "soma_roi_labels.tif",
        "processes": run_dir / "process_roi_labels.tif",
    }
    for key, labels in (
        ("whole", whole_labels),
        ("soma", soma_labels),
        ("processes", process_labels),
    ):
        tf.imwrite(
            label_paths[key],
            labels.astype(np.uint16, copy=False),
            photometric="minisblack",
            metadata={"axes": "YX"},
        )
    script_path = run_dir / "ihc_fiji_bridge.groovy"
    script_path.write_text(FIJI_GROOVY_SCRIPT, encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    channel_order = ["DAPI", measurement, *structural_channels]
    display_ranges = {
        channel: display_range(selected_projections[channel])
        for channel in channel_order
    }
    manifest_data = {
        "pipeline_name": PIPELINE_NAME,
        "channels": {channel: str(paths[channel]) for channel in channel_order},
        "channel_order": channel_order,
        "structural_channels": structural_channels,
        "measurement_channel": measurement,
        "expected_shape": list(metadata["DAPI"]["shape"]),
        "pixel_width_um": float(metadata["DAPI"]["pixel_width_um"]),
        "pixel_height_um": float(metadata["DAPI"]["pixel_height_um"]),
        "review_merge_max_soma_gap_um": REVIEW_MERGE_MAX_SOMA_GAP_UM,
        "z_start_1based": int(best_row["z_start_1based"]),
        "z_end_1based_inclusive": int(best_row["z_end_1based_inclusive"]),
        "projection": best_row["projection"],
        "display_ranges": display_ranges,
        "roi_count": roi_count,
        "require_complete_soma_ids": True,
        "label_mask_paths": {key: str(value) for key, value in label_paths.items()},
        "overlay_output_paths": {
            "whole": str(run_dir / "reviewed_whole_overlay.png"),
            "soma": str(run_dir / "reviewed_soma_overlay.png"),
            "processes": str(run_dir / "reviewed_processes_overlay.png"),
        },
        "ready_marker": str(run_dir / "fiji_ready.json"),
        "done_marker": str(run_dir / "fiji_done.json"),
        "error_marker": str(run_dir / "fiji_error.txt"),
        "auto_continue": bool(auto_continue),
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return run_dir, manifest_path
