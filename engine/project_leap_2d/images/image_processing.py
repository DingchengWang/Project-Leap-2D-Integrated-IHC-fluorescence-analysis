# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def robust01(img: np.ndarray, lo: float = 0.5, hi: float = 99.8) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    p0, p1 = np.percentile(x, [lo, hi])
    if p1 <= p0:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - p0) / (p1 - p0)
    return np.clip(y, 0, 1).astype(np.float32)

def normalized_projection(img: np.ndarray) -> np.ndarray:
    """Return one immutable robust normalization for a shared projection."""
    key = ("normalized_projection", array_identity_key(img))
    with _CACHE_LOCK:
        entry = _NORMALIZED_PROJECTION_CACHE.get(key)
    if entry is not None and entry[0]() is img:
        return entry[1]
    with cache_key_lock(key):
        with _CACHE_LOCK:
            entry = _NORMALIZED_PROJECTION_CACHE.get(key)
        if entry is None or entry[0]() is not img:
            normalized = robust01(img)
            normalized.setflags(write=False)
            with _CACHE_LOCK:
                _NORMALIZED_PROJECTION_CACHE[key] = (
                    weakref.ref(img),
                    normalized,
                )
            return normalized
        return entry[1]

def array_identity_key(array: np.ndarray) -> tuple:
    return (
        int(array.__array_interface__["data"][0]),
        tuple(int(value) for value in array.shape),
        tuple(int(value) for value in array.strides),
        array.dtype.str,
    )

def cache_key_lock(key: tuple) -> threading.Lock:
    with _CACHE_LOCK:
        return _CACHE_KEY_LOCKS.setdefault(key, threading.Lock())

def full_array_percentile(array: np.ndarray, percentile: float) -> float:
    key = ("percentile", id(array), float(percentile))
    with _CACHE_LOCK:
        entry = _FULL_PERCENTILE_CACHE.get(key)
    if entry is not None and entry[0]() is array:
        return entry[1]
    with cache_key_lock(key):
        with _CACHE_LOCK:
            entry = _FULL_PERCENTILE_CACHE.get(key)
        if entry is None or entry[0]() is not array:
            value = float(np.percentile(array, percentile))
            with _CACHE_LOCK:
                _FULL_PERCENTILE_CACHE[key] = (weakref.ref(array), value)
            return value
        return entry[1]

def full_array_sum(array: np.ndarray) -> float:
    key = ("sum_float64", id(array))
    with _CACHE_LOCK:
        entry = _FULL_SUM_CACHE.get(key)
    if entry is not None and entry[0]() is array:
        return entry[1]
    with cache_key_lock(key):
        with _CACHE_LOCK:
            entry = _FULL_SUM_CACHE.get(key)
        if entry is None or entry[0]() is not array:
            value = float(np.sum(array, dtype=np.float64))
            with _CACHE_LOCK:
                _FULL_SUM_CACHE[key] = (weakref.ref(array), value)
            return value
        return entry[1]

def project(stack: np.ndarray, z0: int, z1: int, mode: str) -> np.ndarray:
    sub = stack[z0 : z1 + 1]
    if mode == "max":
        return sub.max(axis=0)
    if mode == "mean":
        return sub.mean(axis=0).astype(np.float32)
    if mode == "sum":
        return sub.sum(axis=0).astype(np.float32)
    raise ValueError(mode)

def _round_half_up_nonnegative(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected a finite non-negative value, got {value}")
    return int(math.floor(value + 0.5))

def plan_fixed_z_windows(
    n_slices: int,
    z_step_um: float,
) -> list[ZWindowPlan]:
    """Plan five equally spaced 9.5-um windows after 5-um end guards.

    The returned coordinates are zero-based and inclusive.  An empty result is
    the explicit signal that five distinct windows do not fit; the controller
    owns the user-facing ``z_step_limit`` stop.
    """
    n_slices = int(n_slices)
    z_step_um = float(z_step_um)
    if n_slices <= 0:
        raise ValueError(f"Z stack must contain at least one slice, got {n_slices}")
    if not math.isfinite(z_step_um) or z_step_um <= 0:
        raise ValueError(f"Z-step must be finite and positive, got {z_step_um}")

    window_intervals = _round_half_up_nonnegative(Z_WINDOW_TARGET_UM / z_step_um)
    window_slices = window_intervals + 1
    guard_slices = int(math.ceil(Z_SURFACE_GUARD_UM / z_step_um))
    first_start = guard_slices
    last_start = n_slices - guard_slices - window_slices
    start_span = last_start - first_start
    if start_span < EXPECTED_Z_INTERVAL_COUNT - 1:
        return []

    denominator = EXPECTED_Z_INTERVAL_COUNT - 1
    starts = [
        first_start + _round_half_up_nonnegative(position * start_span / denominator)
        for position in range(EXPECTED_Z_INTERVAL_COUNT)
    ]
    if len(set(starts)) != EXPECTED_Z_INTERVAL_COUNT:
        return []

    actual_width_um = window_intervals * z_step_um
    actual_guard_um = guard_slices * z_step_um
    plans: list[ZWindowPlan] = []
    for position_0based, z0 in enumerate(starts):
        z1 = z0 + window_slices - 1
        position_1based = position_0based + 1
        plans.append(
            ZWindowPlan(
                position_1based=position_1based,
                z0=z0,
                z1=z1,
                window_slices=window_slices,
                actual_width_um=actual_width_um,
                guard_slices=guard_slices,
                actual_guard_um=actual_guard_um,
                z_mode=f"fixed_z{position_1based:02d}_{z0}_{z1}",
            )
        )
    return plans

def z_range_from_mode(mode: str) -> tuple[int, int]:
    match = re.fullmatch(r"fixed_z(\d{2})_(\d+)_(\d+)", mode)
    if match is None:
        raise ValueError(f"Unknown fixed Z-window mode: {mode}")
    position, z0, z1 = (int(value) for value in match.groups())
    if not 1 <= position <= EXPECTED_Z_INTERVAL_COUNT or z1 < z0:
        raise ValueError(f"Invalid fixed Z-window mode: {mode}")
    return z0, z1

def active_channel_weights(
    structural_projections: dict[str, np.ndarray],
    spec: CandidateSpec,
) -> dict[str, float]:
    requested = {"eGFP": spec.egfp_weight, "GFAP": spec.gfap_weight}
    weights = {
        channel: max(0.0, float(requested[channel]))
        for channel in STRUCTURAL_CHANNELS
        if channel in structural_projections
    }
    total = sum(weights.values())
    if total <= 0:
        equal = 1.0 / len(weights)
        return {channel: equal for channel in weights}
    return {channel: weight / total for channel, weight in weights.items()}

def structural_map(
    structural_projections: dict[str, np.ndarray],
    spec: CandidateSpec,
) -> np.ndarray:
    weights = active_channel_weights(structural_projections, spec)
    combined = sum(
        weights[channel] * normalized_projection(structural_projections[channel])
        for channel in weights
    )
    combined = combined / max(float(combined.max()), 1e-6)
    if spec.smooth_sigma > 0:
        combined = filters.gaussian(combined, sigma=spec.smooth_sigma, preserve_range=True)
    return np.clip(combined, 0, 1).astype(np.float32)

def smooth_outline_source(mask: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    if spec.outline_smooth_sigma <= 0:
        return mask.astype(bool)
    smoothed = filters.gaussian(mask.astype(np.float32), sigma=spec.outline_smooth_sigma, preserve_range=True)
    outline_mask = smoothed >= 0.42
    outline_mask = morphology.binary_closing(outline_mask, footprint=morphology.disk(1))
    outline_mask = morphology.remove_small_objects(outline_mask, min_size=max(30, spec.min_area // 2))
    return outline_mask.astype(bool)

def draw_smooth_label_contours(
    rgb: np.ndarray,
    labels: np.ndarray,
    spec: CandidateSpec,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw each instance separately so touching labels retain their shared boundary."""

    overlay = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    for label_id in (int(value) for value in np.unique(labels) if int(value) > 0):
        source = smooth_outline_source(labels == label_id, spec)
        contours, hierarchy = cv2.findContours(
            source.astype(np.uint8) * 255,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE,
        )
        hierarchy_rows = hierarchy[0] if hierarchy is not None else []
        drawable_contours = []
        for index, contour in enumerate(contours):
            if len(contour) < 4:
                continue
            is_internal = len(hierarchy_rows) > index and hierarchy_rows[index][3] >= 0
            if is_internal and abs(cv2.contourArea(contour)) < spec.outline_hole_min_area:
                continue
            drawable_contours.append(contour)
        if drawable_contours:
            cv2.drawContours(
                overlay,
                drawable_contours,
                -1,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
    return overlay

def make_fiji_like_composite(
    dapi: np.ndarray,
    structural_projections: dict[str, np.ndarray],
    measurement_projection: np.ndarray,
) -> np.ndarray:
    blue = exposure.adjust_gamma(robust01(dapi), gamma=0.85)
    measurement_red = exposure.adjust_gamma(robust01(measurement_projection), gamma=0.85)
    if "eGFP" in structural_projections and "GFAP" in structural_projections:
        egfp = exposure.adjust_gamma(robust01(structural_projections["eGFP"]), gamma=0.85)
        gfap = exposure.adjust_gamma(robust01(structural_projections["GFAP"]), gamma=0.85)
        red = np.clip(measurement_red + gfap, 0, 1)
        green = np.clip(gfap + egfp, 0, 1)
    else:
        only_channel = next(iter(structural_projections))
        red = measurement_red
        green = exposure.adjust_gamma(
            robust01(structural_projections[only_channel]),
            gamma=0.85,
        )
    return np.stack([red, green, blue], axis=-1).astype(np.float32)

def display_range(image: np.ndarray, low: float = 0.3, high: float = 99.8) -> list[float]:
    sample = image[::4, ::4]
    minimum, maximum = np.percentile(sample, [low, high])
    if maximum <= minimum:
        minimum = float(np.min(sample))
        maximum = float(np.max(sample))
    if maximum <= minimum:
        maximum = minimum + 1.0
    return [round(float(minimum), 6), round(float(maximum), 6)]
