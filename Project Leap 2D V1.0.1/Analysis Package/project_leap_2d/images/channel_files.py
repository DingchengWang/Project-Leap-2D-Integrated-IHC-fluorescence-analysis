# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def discover_channel_paths(input_dir: Path) -> tuple[dict[str, Path], list[str]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    candidates: dict[str, list[Path]] = {name: [] for name in CHANNEL_PATTERNS}
    ignored: list[str] = []
    tif_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    for path in tif_paths:
        matches = [name for name, pattern in CHANNEL_PATTERNS.items() if pattern.search(path.stem)]
        if not matches:
            ignored.append(f"{path.name}: no supported channel token")
            continue
        if len(matches) != 1:
            raise ValueError(f"Ambiguous channel tokens in {path.name}: {matches}")
        try:
            with tf.TiffFile(str(path)) as tif:
                series = tif.series[0]
                shape = tuple(int(x) for x in series.shape)
                axes = str(series.axes)
        except Exception as exc:
            raise ValueError(f"Unreadable TIFF {path.name}: {exc!r}") from exc
        if len(shape) != 3 or axes != "ZYX":
            raise ValueError(
                f"{path.name} must be a split single-channel ZYX stack; found axes={axes!r}, shape={shape}"
            )
        candidates[matches[0]].append(path)

    selected: dict[str, Path] = {}
    for channel, paths in candidates.items():
        if not paths:
            continue
        if len(paths) != 1:
            names = ", ".join(path.name for path in paths)
            raise ValueError(f"Exactly one {channel} stack is allowed; found: {names}")
        selected[channel] = paths[0]

    if "DAPI" not in selected:
        raise ValueError(f"No split DAPI Z-stack found in {input_dir}")
    if not any(channel in selected for channel in STRUCTURAL_CHANNELS):
        raise ValueError(f"No split eGFP or GFAP structural Z-stack found in {input_dir}")
    measurement = [channel for channel in MEASUREMENT_CHANNELS if channel in selected]
    if len(measurement) != 1:
        raise ValueError(
            "Exactly one measurement stack is required: KCNN1, KCNN2, KCNN3, or KCNJ10; "
            f"found {measurement or 'none'}"
        )
    return selected, ignored

def detect_filename_age_profile(paths: dict[str, Path]) -> AgeProfileDecision | None:
    """Use an explicit age token from ROI-defining channel filenames only."""

    evidence: dict[str, list[str]] = {name: [] for name in AGE_PROFILE_PATTERNS}
    for channel in ("DAPI", *STRUCTURAL_CHANNELS):
        path = paths.get(channel)
        if path is None:
            continue
        matched = [
            name
            for name, pattern in AGE_PROFILE_PATTERNS.items()
            if pattern.search(path.stem)
        ]
        if len(matched) > 1:
            raise ValueError(
                f"Conflicting age-profile tokens in one input filename: {path.name}"
            )
        if matched:
            evidence[matched[0]].append(path.name)

    detected = [name for name, filenames in evidence.items() if filenames]
    if len(detected) > 1:
        details = "; ".join(
            f"{name}={','.join(evidence[name])}" for name in detected
        )
        raise ValueError(
            "Input filenames contain conflicting neonatal/mature labels: " + details
        )
    if not detected:
        return None

    profile = detected[0]
    return AgeProfileDecision(
        profile=profile,
        source="filename",
        neonatal_score=None,
        threshold=AGE_PROFILE_THRESHOLD,
        confidence_margin=None,
        tagged_files=tuple(sorted(evidence[profile])),
        features={},
    )

def channel_mode(paths: dict[str, Path]) -> str:
    structural = [channel for channel in STRUCTURAL_CHANNELS if channel in paths]
    return "+".join(["DAPI", *structural])

def measurement_channel(paths: dict[str, Path]) -> str:
    channels = [channel for channel in MEASUREMENT_CHANNELS if channel in paths]
    if len(channels) != 1:
        raise ValueError(f"Expected exactly one measurement channel, found {channels}")
    return channels[0]

def validate_shared_geometry(metadata: dict[str, dict]) -> None:
    def signature(meta: dict) -> tuple:
        return (
            meta.get("pixel_width_um"),
            meta.get("pixel_height_um"),
            meta.get("pixel_depth_um"),
        )

    def signatures_match(left: tuple, right: tuple) -> bool:
        for left_value, right_value in zip(left, right):
            if left_value is None or right_value is None:
                if left_value is not None or right_value is not None:
                    return False
                continue
            if not math.isclose(
                float(left_value),
                float(right_value),
                rel_tol=1e-5,
                abs_tol=1e-9,
            ):
                return False
        return True

    reference = signature(metadata["DAPI"])
    mismatched = {
        channel: signature(channel_meta)
        for channel, channel_meta in metadata.items()
        if not signatures_match(signature(channel_meta), reference)
    }
    if mismatched:
        raise ValueError(
            f"Channel calibration metadata do not match DAPI {reference}: {mismatched}"
        )
