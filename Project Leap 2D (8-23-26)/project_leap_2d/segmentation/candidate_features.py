# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def clear_candidate_computation_caches() -> None:
    with _CACHE_LOCK:
        _DAPI_NUCLEI_CACHE.clear()
        _FULL_PERCENTILE_CACHE.clear()
        _FULL_SUM_CACHE.clear()
        _TOP_HAT_CACHE.clear()
        _NORMALIZED_PROJECTION_CACHE.clear()
        _CANDIDATE_BASE_CACHE.clear()
        _CANDIDATE_BASE_LOCKS.clear()
        _CACHE_KEY_LOCKS.clear()
        _BRANCH_FEATURE_CACHE.clear()
        _DISTRIBUTION_MODEL_CACHE.clear()
        _DISTRIBUTION_MODEL_FAILURES.clear()
        _DISTRIBUTION_DIAGNOSTIC_CACHE.clear()

def weighted_value_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    fraction: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or values.size == 0:
        raise ValueError("Invalid weighted quantile inputs")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Weighted quantile requires positive total weight")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    target = float(np.clip(fraction, 0.0, 1.0)) * float(cumulative[-1])
    position = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(position, ordered_values.size - 1)])

def gaussian_posterior_intersection(
    means: np.ndarray | tuple[float, float],
    variances: np.ndarray | tuple[float, float],
    weights: np.ndarray | tuple[float, float],
) -> float:
    """Return the background-to-signal equal-posterior root between the means."""
    means_array = np.asarray(means, dtype=np.float64)
    variances_array = np.asarray(variances, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if means_array.shape != (2,) or variances_array.shape != (2,) or weights_array.shape != (2,):
        raise ValueError("Two Gaussian components are required")
    order = np.argsort(means_array, kind="stable")
    means_array = means_array[order]
    variances_array = variances_array[order]
    weights_array = weights_array[order]
    if (
        not np.all(np.isfinite(means_array))
        or not np.all(np.isfinite(variances_array))
        or not np.all(np.isfinite(weights_array))
        or np.any(variances_array <= 0)
        or np.any(weights_array <= 0)
        or means_array[1] <= means_array[0]
    ):
        raise ValueError("Invalid Gaussian component parameters")

    mean0, mean1 = map(float, means_array)
    var0, var1 = map(float, variances_array)
    weight0, weight1 = map(float, weights_array)
    sigma0 = math.sqrt(var0)
    sigma1 = math.sqrt(var1)
    a = 0.5 / var1 - 0.5 / var0
    b = mean0 / var0 - mean1 / var1
    c = (
        -0.5 * mean0 * mean0 / var0
        + 0.5 * mean1 * mean1 / var1
        + math.log((weight0 / sigma0) / (weight1 / sigma1))
    )
    scale = max(abs(0.5 / var0), abs(0.5 / var1), 1.0)
    if abs(a) <= 1e-12 * scale:
        if abs(b) <= np.finfo(np.float64).eps:
            raise ValueError("Gaussian posteriors do not have a unique intersection")
        roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        tolerance = 1e-12 * max(b * b, abs(4.0 * a * c), 1.0)
        if discriminant < -tolerance:
            raise ValueError("Gaussian posteriors do not intersect")
        discriminant = max(discriminant, 0.0)
        square_root = math.sqrt(discriminant)
        roots = [
            (-b - square_root) / (2.0 * a),
            (-b + square_root) / (2.0 * a),
        ]
    interval_tolerance = 1e-10 * max(abs(mean0), abs(mean1), 1.0)
    valid_roots = sorted(
        root
        for root in roots
        if np.isfinite(root)
        and mean0 + interval_tolerance < root < mean1 - interval_tolerance
    )
    if len(valid_roots) != 1:
        raise ValueError(
            "Gaussian posteriors require exactly one intersection between component means"
        )
    root = float(valid_roots[0])
    epsilon = max((mean1 - mean0) * 1e-6, 1e-12)

    def signal_log_odds(value: float) -> float:
        background = (
            math.log(weight0 / sigma0)
            - 0.5 * (value - mean0) ** 2 / var0
        )
        signal = (
            math.log(weight1 / sigma1)
            - 0.5 * (value - mean1) ** 2 / var1
        )
        return signal - background

    if not (
        signal_log_odds(root - epsilon) < 0.0
        and signal_log_odds(root + epsilon) > 0.0
    ):
        raise ValueError(
            "Posterior intersection does not switch from background to signal"
        )
    return root

def fit_weighted_two_component_gmm(
    values: np.ndarray,
    counts: np.ndarray,
    *,
    max_iterations: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """Fit an exact weighted univariate GMM to unique log-intensity values."""
    x = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(counts, dtype=np.float64)
    if x.ndim != 1 or x.shape != sample_weights.shape or x.size < 4:
        raise ValueError("Too few unique values for a two-component GMM")
    total_weight = float(sample_weights.sum(dtype=np.float64))
    if total_weight < 200 or np.any(sample_weights < 0):
        raise ValueError("Too few pixels for a two-component GMM")
    global_mean = float(np.sum(sample_weights * x, dtype=np.float64) / total_weight)
    global_variance = float(
        np.sum(sample_weights * (x - global_mean) ** 2, dtype=np.float64)
        / total_weight
    )
    if not np.isfinite(global_variance) or global_variance <= 0:
        raise ValueError("GMM requires a non-constant intensity distribution")
    variance_floor = max(global_variance * 1e-6, 1e-10)
    initial_quantiles = ((0.20, 0.75), (0.35, 0.85), (0.10, 0.65))
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, bool, int] | None = None

    for lower_fraction, upper_fraction in initial_quantiles:
        means = np.asarray(
            [
                weighted_value_quantile(x, sample_weights, lower_fraction),
                weighted_value_quantile(x, sample_weights, upper_fraction),
            ],
            dtype=np.float64,
        )
        if means[1] <= means[0]:
            continue
        hard_labels = np.argmin(np.abs(x[:, None] - means[None, :]), axis=1)
        component_weights = np.asarray(
            [float(sample_weights[hard_labels == index].sum(dtype=np.float64)) for index in range(2)],
            dtype=np.float64,
        )
        if np.any(component_weights <= 0):
            component_weights = np.asarray([0.70, 0.30], dtype=np.float64) * total_weight
        mixing = component_weights / total_weight
        variances = np.full(2, global_variance, dtype=np.float64)
        previous_log_likelihood = -np.inf
        converged = False
        completed_iterations = 0

        for iteration in range(1, max_iterations + 1):
            log_probabilities = []
            for component in range(2):
                variance = max(float(variances[component]), variance_floor)
                log_probabilities.append(
                    math.log(max(float(mixing[component]), 1e-12))
                    - 0.5
                    * (
                        math.log(2.0 * math.pi * variance)
                        + (x - float(means[component])) ** 2 / variance
                    )
                )
            log_probability_0, log_probability_1 = log_probabilities
            log_normalizer = np.logaddexp(log_probability_0, log_probability_1)
            responsibility_0 = np.exp(log_probability_0 - log_normalizer)
            responsibilities = np.column_stack(
                [responsibility_0, 1.0 - responsibility_0]
            )
            weighted_responsibilities = responsibilities * sample_weights[:, None]
            effective_counts = weighted_responsibilities.sum(axis=0, dtype=np.float64)
            if np.any(effective_counts <= max(1e-8 * total_weight, 1e-6)):
                break
            mixing = effective_counts / total_weight
            means = (
                (weighted_responsibilities * x[:, None]).sum(axis=0, dtype=np.float64)
                / effective_counts
            )
            variances = (
                (
                    weighted_responsibilities
                    * (x[:, None] - means[None, :]) ** 2
                ).sum(axis=0, dtype=np.float64)
                / effective_counts
            )
            variances = np.maximum(variances, variance_floor)
            log_likelihood = float(
                np.sum(sample_weights * log_normalizer, dtype=np.float64)
            )
            completed_iterations = iteration
            if (
                iteration >= 5
                and abs(log_likelihood - previous_log_likelihood)
                <= 1e-9 * (1.0 + abs(previous_log_likelihood))
            ):
                converged = True
                previous_log_likelihood = log_likelihood
                break
            previous_log_likelihood = log_likelihood

        if not np.isfinite(previous_log_likelihood):
            continue
        candidate = (
            previous_log_likelihood,
            means.copy(),
            variances.copy(),
            mixing.copy(),
            converged,
            completed_iterations,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("All deterministic GMM initializations failed")
    _likelihood, means, variances, mixing, converged, iterations = best
    order = np.argsort(means, kind="stable")
    return (
        means[order],
        variances[order],
        mixing[order],
        bool(converged),
        int(iterations),
    )

def background_peak_log_model(
    log_values: np.ndarray,
    counts: np.ndarray,
    *,
    upper_bound: float,
    bins: int = 512,
) -> tuple[float, float]:
    """Estimate the low-intensity mode and FWHM sigma using true bin centers."""
    x = np.asarray(log_values, dtype=np.float64)
    sample_weights = np.asarray(counts, dtype=np.float64)
    if x.size < 4 or x.shape != sample_weights.shape or x[-1] <= x[0]:
        raise ValueError("Background-peak modeling requires a non-constant distribution")
    hist, edges = np.histogram(
        x,
        bins=max(64, int(bins)),
        range=(float(x[0]), float(x[-1])),
        weights=sample_weights,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = ndi.gaussian_filter1d(hist.astype(np.float64), sigma=2.0)
    allowed = np.flatnonzero(centers < float(upper_bound))
    if allowed.size < 3:
        raise ValueError("No low-intensity histogram range below the posterior threshold")
    peak_index = int(allowed[np.argmax(smoothed[allowed])])
    baseline = float(np.min(smoothed[allowed]))
    peak_height = float(smoothed[peak_index])
    if peak_height <= baseline:
        raise ValueError("Background peak has no measurable prominence")
    half_height = baseline + 0.5 * (peak_height - baseline)
    left_index = peak_index
    while left_index > 0 and smoothed[left_index] > half_height:
        left_index -= 1
    right_index = peak_index
    while right_index < smoothed.size - 1 and smoothed[right_index] > half_height:
        right_index += 1
    left_width = float(centers[peak_index] - centers[left_index])
    right_width = float(centers[right_index] - centers[peak_index])
    if left_width > 0 and right_width > 0:
        fwhm = left_width + right_width
    elif left_width > 0:
        fwhm = 2.0 * left_width
    elif right_width > 0:
        fwhm = 2.0 * right_width
    else:
        raise ValueError("Background peak width could not be estimated")
    bin_width = float(np.mean(np.diff(centers), dtype=np.float64))
    sigma = max(fwhm / 2.354820045, bin_width)
    return float(centers[peak_index]), float(sigma)

def distribution_model_cache_key(projection: np.ndarray) -> tuple:
    return (
        "log1p_gmm_posterior_v1",
        array_identity_key(projection),
        0.01,
        99.99,
        4096,
        512,
    )

def fit_log1p_gmm_threshold(projection: np.ndarray) -> Log1pGMMThreshold:
    values = np.asarray(projection, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 200:
        raise ValueError("Too few finite pixels for log1p GMM thresholding")
    clip_low, clip_high = np.percentile(values, [0.01, 99.99])
    if not np.isfinite(clip_low) or not np.isfinite(clip_high) or clip_high <= clip_low:
        raise ValueError("log1p GMM requires a non-constant clipped intensity range")
    clipped_log_values = np.log1p(np.clip(values, clip_low, clip_high))
    histogram, edges = np.histogram(
        clipped_log_values,
        bins=4096,
        range=(float(np.log1p(clip_low)), float(np.log1p(clip_high))),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    populated = histogram > 0
    unique_log_values = centers[populated]
    counts = histogram[populated].astype(np.float64, copy=False)
    means, variances, weights, converged, iterations = fit_weighted_two_component_gmm(
        unique_log_values,
        counts,
    )
    threshold_log = gaussian_posterior_intersection(means, variances, weights)
    background_peak_log, background_sigma_log = background_peak_log_model(
        unique_log_values,
        counts,
        upper_bound=threshold_log,
    )
    variance_sum = float(variances[0] + variances[1])
    ashman_separation = float(
        math.sqrt(2.0) * (means[1] - means[0]) / math.sqrt(variance_sum)
    )
    log_densities = np.asarray(
        [
            math.log(float(weights[index]))
            - 0.5
            * (
                math.log(2.0 * math.pi * float(variances[index]))
                + (threshold_log - float(means[index])) ** 2
                / float(variances[index])
            )
            for index in range(2)
        ],
        dtype=np.float64,
    )
    posterior_at_threshold = float(
        math.exp(log_densities[1] - float(np.logaddexp(*log_densities)))
    )
    reasons: list[str] = []
    if not converged:
        reasons.append("gmm_not_converged")
    if float(np.min(weights)) < 0.005:
        reasons.append("component_weight_below_0.005")
    if ashman_separation < 1.0:
        reasons.append("component_separation_below_1.0")
    if not (float(means[0]) < threshold_log < float(means[1])):
        reasons.append("posterior_intersection_outside_means")
    if threshold_log <= background_peak_log + 0.25 * background_sigma_log:
        reasons.append("posterior_intersection_not_above_background_noise")
    if abs(posterior_at_threshold - 0.5) > 1e-6:
        reasons.append("posterior_intersection_numerically_inaccurate")
    threshold_raw = float(np.expm1(threshold_log))
    if not (float(clip_low) <= threshold_raw <= float(clip_high)):
        reasons.append("posterior_threshold_outside_clipped_range")
    return Log1pGMMThreshold(
        threshold_raw=threshold_raw,
        threshold_log1p=float(threshold_log),
        background_peak_raw=float(np.expm1(background_peak_log)),
        background_peak_log1p=float(background_peak_log),
        background_sigma_log1p=float(background_sigma_log),
        means_log1p=(float(means[0]), float(means[1])),
        variances_log1p=(float(variances[0]), float(variances[1])),
        weights=(float(weights[0]), float(weights[1])),
        ashman_separation=ashman_separation,
        posterior_at_threshold=posterior_at_threshold,
        clip_bounds_raw=(float(clip_low), float(clip_high)),
        converged=bool(converged),
        iterations=int(iterations),
        qc_pass=not reasons,
        qc_reasons=tuple(reasons),
    )

def get_log1p_gmm_threshold(projection: np.ndarray) -> Log1pGMMThreshold:
    key = distribution_model_cache_key(projection)
    with _CACHE_LOCK:
        cached = _DISTRIBUTION_MODEL_CACHE.get(key)
        cached_failure = _DISTRIBUTION_MODEL_FAILURES.get(key)
    if cached is not None:
        return cached
    if cached_failure is not None:
        raise ValueError(cached_failure)
    with cache_key_lock(key):
        with _CACHE_LOCK:
            cached = _DISTRIBUTION_MODEL_CACHE.get(key)
            cached_failure = _DISTRIBUTION_MODEL_FAILURES.get(key)
        if cached is not None:
            return cached
        if cached_failure is not None:
            raise ValueError(cached_failure)
        try:
            result = fit_log1p_gmm_threshold(projection)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with _CACHE_LOCK:
                _DISTRIBUTION_MODEL_FAILURES[key] = message
            raise ValueError(message) from exc
        with _CACHE_LOCK:
            _DISTRIBUTION_MODEL_CACHE[key] = result
        return result

def distributional_threshold_diagnostics(
    struct: np.ndarray,
    structural_projections: dict[str, np.ndarray],
    spec: TestSpec,
) -> dict[str, object]:
    key = (
        "distributional_threshold_diagnostics_v1",
        array_identity_key(struct),
        tuple(sorted(structural_projections)),
        round(float(spec.threshold_scale), 12),
    )
    with _CACHE_LOCK:
        cached = _DISTRIBUTION_DIAGNOSTIC_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    model_error = ""
    result: Log1pGMMThreshold | None = None
    try:
        result = get_log1p_gmm_threshold(struct)
    except Exception as exc:
        model_error = repr(exc)
    if result is not None and not result.qc_pass:
        model_error = ",".join(result.qc_reasons)
    valid = result is not None and result.qc_pass
    posterior_threshold = result.threshold_raw if result is not None else None
    scaled_threshold = (
        result.threshold_raw * float(spec.threshold_scale)
        if valid and result is not None
        else None
    )
    diagnostics: dict[str, object] = {
        "distribution_model": "log1p_two_component_gmm_true_posterior",
        "distribution_background_peak_role": "qc_only",
        "distribution_qc_pass": bool(valid),
        "distribution_valid_channels": ",".join(sorted(structural_projections)),
        "distribution_failed_channels": model_error,
        "distribution_thresholds_raw": scaled_threshold,
        "distribution_posterior_thresholds_raw": posterior_threshold,
        "distribution_background_peaks_raw": (
            result.background_peak_raw if result is not None else None
        ),
        "distribution_background_sigmas_log1p": (
            result.background_sigma_log1p if result is not None else None
        ),
        "distribution_ashman_separations": (
            result.ashman_separation if result is not None else None
        ),
        "distribution_component_parameters": (
            json.dumps(
                {
                    "means_log1p": result.means_log1p,
                    "variances_log1p": result.variances_log1p,
                    "weights": result.weights,
                    "posterior_at_threshold": result.posterior_at_threshold,
                    "iterations": result.iterations,
                },
                sort_keys=True,
            )
            if result is not None
            else ""
        ),
    }
    with _CACHE_LOCK:
        _DISTRIBUTION_DIAGNOSTIC_CACHE[key] = dict(diagnostics)
    return diagnostics

def log1p_gmm_mask(
    struct: np.ndarray,
    structural_projections: dict[str, np.ndarray],
    spec: TestSpec,
) -> np.ndarray:
    diagnostics = distributional_threshold_diagnostics(
        struct,
        structural_projections,
        spec,
    )
    if not bool(diagnostics["distribution_qc_pass"]):
        raise ValueError(
            "No structural channel passed log1p GMM/background-peak QC: "
            f"{diagnostics['distribution_failed_channels']}"
        )
    threshold = diagnostics["distribution_thresholds_raw"]
    if threshold is None or not np.isfinite(float(threshold)):
        raise ValueError("No valid structural-composite log1p GMM threshold")
    return np.asarray(struct, dtype=np.float64) >= float(threshold)

def threshold_mask(
    struct: np.ndarray,
    structural_projections: dict[str, np.ndarray],
    spec: TestSpec,
) -> np.ndarray:
    if spec.method == "otsu":
        thr = filters.threshold_otsu(struct) * spec.threshold_scale
        mask = struct >= thr
    elif spec.method == "yen":
        thr = filters.threshold_yen(struct) * spec.threshold_scale
        mask = struct >= thr
    elif spec.method == "li":
        thr = filters.threshold_li(struct) * spec.threshold_scale
        mask = struct >= thr
    elif spec.method == "sauvola":
        # Local thresholding is useful for uneven staining and projection background.
        local = filters.threshold_sauvola(struct, window_size=121, k=0.12)
        mask = struct >= (local * spec.threshold_scale)
    elif spec.method == "hysteresis":
        hi = full_array_percentile(struct, 88) * spec.threshold_scale
        lo = hi * 0.45
        mask = filters.apply_hysteresis_threshold(struct, lo, hi)
    elif spec.method == "dual_channel_union":
        masks = []
        for channel, projection in structural_projections.items():
            normalized = normalized_projection(projection)
            threshold_factor = 0.82 if channel == "eGFP" else 0.76
            threshold = filters.threshold_otsu(filters.gaussian(normalized, sigma=1.0))
            masks.append(normalized >= threshold * threshold_factor * spec.threshold_scale)
        mask = np.logical_or.reduce(masks)
    elif spec.method == "top_hat_union":
        weights = active_channel_weights(structural_projections, spec)
        cache_key = (
            "top_hat_union",
            tuple(
                (
                    channel,
                    round(float(weights[channel]), 12),
                    array_identity_key(structural_projections[channel]),
                )
                for channel in sorted(weights)
            ),
        )
        with _CACHE_LOCK:
            cached = _TOP_HAT_CACHE.get(cache_key)
        if cached is None:
            with cache_key_lock(cache_key):
                with _CACHE_LOCK:
                    cached = _TOP_HAT_CACHE.get(cache_key)
                if cached is None:
                    fp = morphology.disk(9)
                    base_mix = np.zeros_like(struct, dtype=np.float32)
                    top_hat_mix = np.zeros_like(struct, dtype=np.float32)
                    for channel, weight in weights.items():
                        normalized = normalized_projection(
                            structural_projections[channel]
                        )
                        base_mix += weight * normalized
                        top_hat_mix += weight * morphology.white_tophat(
                            normalized,
                            footprint=fp,
                        )
                    mix = np.clip(0.45 * base_mix + 0.55 * top_hat_mix, 0, None)
                    mix = (mix / max(float(mix.max()), 1e-6)).astype(
                        np.float32,
                        copy=False,
                    )
                    threshold = float(filters.threshold_otsu(mix))
                    mix.setflags(write=False)
                    cached = (mix, threshold)
                    with _CACHE_LOCK:
                        _TOP_HAT_CACHE[cache_key] = cached
        mix, threshold = cached
        mask = mix >= (threshold * spec.threshold_scale)
    elif spec.method == "log1p_gmm":
        mask = log1p_gmm_mask(struct, structural_projections, spec)
    else:
        raise ValueError(spec.method)
    return mask

def cleanup_mask(mask: np.ndarray, spec: TestSpec) -> np.ndarray:
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=spec.min_area)
    if spec.close_radius > 0:
        mask = morphology.binary_closing(mask, footprint=morphology.disk(spec.close_radius))
    if spec.hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=spec.hole_area)
    if spec.dilate_radius > 0:
        mask = morphology.binary_dilation(mask, footprint=morphology.disk(spec.dilate_radius))
    mask = morphology.remove_small_objects(mask, min_size=spec.min_area)
    # Avoid accepting a biologically implausible whole-field mask.
    frac = float(mask.mean())
    if frac > 0.42:
        labels = measure.label(mask)
        props = sorted(measure.regionprops(labels), key=lambda p: p.area, reverse=True)
        keep_labels = [p.label for p in props[:12] if p.area >= spec.min_area]
        mask = np.isin(labels, keep_labels)
    return mask.astype(bool)

def anchor_mask(candidate: np.ndarray, struct: np.ndarray, spec: TestSpec) -> np.ndarray:
    labels = measure.label(candidate)
    if labels.max() == 0:
        return np.zeros_like(candidate, dtype=bool)
    candidate_values = struct[candidate]
    if candidate_values.size == 0:
        return np.zeros_like(candidate, dtype=bool)
    mean_cut = float(np.percentile(candidate_values, 72))
    anchor_labels: list[int] = []
    for prop in measure.regionprops(labels, intensity_image=struct):
        if prop.area >= spec.anchor_area:
            anchor_labels.append(int(prop.label))
        elif prop.area >= max(120, spec.anchor_area // 5) and prop.mean_intensity >= mean_cut:
            anchor_labels.append(int(prop.label))
    anchor = np.isin(labels, anchor_labels)

    bright_seed = candidate & (
        struct >= full_array_percentile(struct, spec.seed_percentile)
    )
    bright_seed = morphology.remove_small_objects(bright_seed, min_size=spec.seed_min_area)
    return anchor | bright_seed

def anchor_connected_cleanup(
    candidate: np.ndarray,
    struct: np.ndarray,
    spec: TestSpec,
    extra_anchor: np.ndarray | None = None,
) -> np.ndarray:
    candidate = cleanup_mask(candidate, spec)
    anchors = anchor_mask(candidate, struct, spec)
    if extra_anchor is not None and extra_anchor.any():
        anchors |= extra_anchor.astype(bool)
    if not anchors.any():
        return candidate

    if spec.cleanup_mode == "anchor_bridge":
        expanded = morphology.binary_dilation(candidate, footprint=morphology.disk(spec.bridge_radius))
        labels = measure.label(expanded)
        anchor_labels = np.unique(labels[morphology.binary_dilation(anchors, footprint=morphology.disk(spec.bridge_radius))])
        anchor_labels = anchor_labels[anchor_labels != 0]
        keep_region = np.isin(labels, anchor_labels)
        cleaned = candidate & keep_region
    elif spec.cleanup_mode == "seed_reconstruct":
        low_mask = struct >= full_array_percentile(struct, spec.low_percentile)
        low_mask = morphology.remove_small_objects(low_mask, min_size=max(40, spec.min_area // 2))
        cleaned = ndi.binary_propagation(anchors, mask=low_mask)
    elif spec.cleanup_mode == "hybrid_reconstruct":
        low_mask = struct >= full_array_percentile(struct, spec.low_percentile)
        bridge = morphology.binary_dilation(candidate, footprint=morphology.disk(spec.bridge_radius))
        cleaned = ndi.binary_propagation(anchors, mask=(low_mask & bridge))
        cleaned |= candidate & morphology.binary_dilation(cleaned, footprint=morphology.disk(max(2, spec.bridge_radius // 2)))
    else:
        cleaned = candidate

    cleaned = cleanup_mask(cleaned, spec)
    if float(cleaned.mean()) > spec.max_area_fraction:
        # Revert to conservative anchor-bridge if reconstruction leaks into broad background.
        expanded = morphology.binary_dilation(candidate, footprint=morphology.disk(max(3, spec.bridge_radius // 2)))
        labels = measure.label(expanded)
        anchor_labels = np.unique(labels[anchors])
        anchor_labels = anchor_labels[anchor_labels != 0]
        cleaned = cleanup_mask(candidate & np.isin(labels, anchor_labels), spec)
    return cleaned.astype(bool)
