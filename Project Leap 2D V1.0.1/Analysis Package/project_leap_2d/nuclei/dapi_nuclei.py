# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def dapi_nuclei_mask(dapi_proj: np.ndarray, percentile_floor: float | None = None) -> np.ndarray:
    key = (
        "dapi_nuclei",
        array_identity_key(dapi_proj),
        None if percentile_floor is None else float(percentile_floor),
    )
    with _CACHE_LOCK:
        cached = _DAPI_NUCLEI_CACHE.get(key)
    if cached is not None:
        return cached
    with cache_key_lock(key):
        with _CACHE_LOCK:
            cached = _DAPI_NUCLEI_CACHE.get(key)
        if cached is None:
            d = normalized_projection(dapi_proj)
            try:
                threshold = float(filters.threshold_otsu(d))
            except ValueError:
                threshold = full_array_percentile(d, 96)
            if percentile_floor is not None:
                threshold = max(
                    threshold,
                    full_array_percentile(d, percentile_floor),
                )
            cached = d >= threshold
            cached = morphology.binary_opening(
                cached,
                footprint=morphology.disk(1),
            )
            cached = morphology.remove_small_objects(cached, min_size=40).astype(bool)
            cached.setflags(write=False)
            with _CACHE_LOCK:
                _DAPI_NUCLEI_CACHE[key] = cached
        return cached

def dapi_nuclei_core_and_extent(
    dapi_proj: np.ndarray,
    mean_pixel_um: float,
    config: CompartmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return strict DAPI seeds and their lower-threshold, seed-connected extents."""

    dapi_norm = normalized_projection(dapi_proj)
    try:
        otsu_threshold = float(filters.threshold_otsu(dapi_norm))
    except ValueError:
        otsu_threshold = full_array_percentile(dapi_norm, 96)
    high_threshold = max(
        otsu_threshold,
        full_array_percentile(dapi_norm, config.dapi_percentile_floor),
    )
    nuclei_core = dapi_norm >= high_threshold
    nuclei_core = morphology.binary_opening(
        nuclei_core,
        footprint=morphology.disk(1),
    )
    nuclei_core = morphology.remove_small_objects(nuclei_core, min_size=40)

    low_threshold = min(
        high_threshold,
        max(
            full_array_percentile(dapi_norm, config.dapi_extent_percentile_floor),
            high_threshold * config.dapi_extent_low_high_ratio,
        ),
    )
    extent_domain = dapi_norm >= low_threshold
    nuclei_extent = ndi.binary_propagation(
        nuclei_core,
        structure=np.ones((3, 3), dtype=bool),
        mask=extent_domain,
    ).astype(bool)
    closing_radius_px = max(
        1,
        int(round(config.dapi_extent_closing_um / max(mean_pixel_um, 1e-4))),
    )
    nuclei_extent = morphology.binary_closing(
        nuclei_extent,
        footprint=morphology.disk(closing_radius_px),
    )
    nuclei_extent = morphology.remove_small_holes(
        nuclei_extent,
        area_threshold=max(16, int(round(0.20 / max(mean_pixel_um**2, 1e-6)))),
    )
    nuclei_extent = morphology.remove_small_objects(nuclei_extent, min_size=40)
    nuclei_extent |= nuclei_core
    return nuclei_core.astype(bool), nuclei_extent.astype(bool), dapi_norm, {
        "high_threshold": round(high_threshold, 6),
        "low_threshold": round(low_threshold, 6),
        "strict_core_px": int(nuclei_core.sum()),
        "reconstructed_extent_px": int(nuclei_extent.sum()),
    }
