"""Calibration-aware GFAP background correction and fibre enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


@dataclass(frozen=True)
class GFAPStructureConfig:
    """Physical-scale parameters used to produce a GFAP structural score."""

    background_sigma_um: float = 2.8
    ridge_scales_um: tuple[float, ...] = (0.12, 0.24, 0.48)
    projection_percentile: float = 100.0
    intensity_weight: float = 0.58
    ridge_weight: float = 0.42
    intensity_floor_percentile: float = 52.0
    structural_percentile: float = 72.0
    strong_ridge_percentile: float = 88.0
    connection_gap_um: float = 0.28
    min_structure_area_um2: float = 0.10


def _robust_unit_interval(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.7))
    if not np.isfinite(high) or high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    normalized = (np.asarray(image, dtype=np.float32) - float(low)) / float(high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _elliptical_footprint(radius_um: float, pixel_size_um: tuple[float, float]) -> np.ndarray:
    y_um, x_um = pixel_size_um
    ry = max(1, int(np.ceil(radius_um / y_um)))
    rx = max(1, int(np.ceil(radius_um / x_um)))
    yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    return ((yy * y_um) ** 2 + (xx * x_um) ** 2) <= radius_um**2


def background_correct_gfap(
    gfap_image: np.ndarray,
    pixel_size_um: tuple[float, float],
    config: GFAPStructureConfig,
    *,
    return_peak_z: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Subtract smooth local background before projecting a GFAP Z stack.

    A two-dimensional image is accepted as an already selected projection.  A
    three-dimensional input is corrected slice-by-slice before projection so a
    bright background slice cannot define the signal baseline for every Z.
    """

    image = np.asarray(gfap_image, dtype=np.float32)
    if image.ndim not in (2, 3):
        raise ValueError("GFAP input must be a 2D projection or a 3D Z stack")
    if not np.isfinite(image).all():
        image = np.nan_to_num(image, copy=False)

    sigma_y = max(0.5, config.background_sigma_um / pixel_size_um[0])
    sigma_x = max(0.5, config.background_sigma_um / pixel_size_um[1])
    if image.ndim == 2:
        background = ndi.gaussian_filter(image, sigma=(sigma_y, sigma_x), mode="nearest")
        corrected_2d = np.maximum(image - background, 0.0).astype(
            np.float32,
            copy=False,
        )
        if return_peak_z:
            return corrected_2d, np.zeros(image.shape, dtype=np.float32)
        return corrected_2d

    corrected = np.empty_like(image, dtype=np.float32)
    for z_index in range(image.shape[0]):
        background = ndi.gaussian_filter(
            image[z_index],
            sigma=(sigma_y, sigma_x),
            mode="nearest",
        )
        corrected[z_index] = np.maximum(image[z_index] - background, 0.0)
    if config.projection_percentile >= 100.0:
        projection = corrected.max(axis=0)
    else:
        projection = np.percentile(
            corrected,
            config.projection_percentile,
            axis=0,
        ).astype(np.float32)
    if return_peak_z:
        return projection, np.argmax(corrected, axis=0).astype(np.float32)
    return projection


def multiscale_fibre_score(
    corrected_projection: np.ndarray,
    pixel_size_um: tuple[float, float],
    config: GFAPStructureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized intensity, ridge evidence, and their structural score."""

    intensity = _robust_unit_interval(corrected_projection)
    ridge = np.zeros(intensity.shape, dtype=np.float32)
    y_um, x_um = pixel_size_um
    for scale_um in config.ridge_scales_um:
        sigma = (
            max(0.55, float(scale_um) / y_um),
            max(0.55, float(scale_um) / x_um),
        )
        dyy = ndi.gaussian_filter(intensity, sigma=sigma, order=(2, 0), mode="nearest")
        dxx = ndi.gaussian_filter(intensity, sigma=sigma, order=(0, 2), mode="nearest")
        dxy = ndi.gaussian_filter(intensity, sigma=sigma, order=(1, 1), mode="nearest")
        trace_half = 0.5 * (dxx + dyy)
        discriminant = np.sqrt(np.maximum(0.25 * (dxx - dyy) ** 2 + dxy**2, 0.0))
        lambda_min = trace_half - discriminant
        lambda_max = trace_half + discriminant
        curvature = np.maximum(-lambda_min, 0.0)
        coherence = np.clip(
            1.0 - np.abs(lambda_max) / np.maximum(np.abs(lambda_min), 1e-6),
            0.0,
            1.0,
        )
        ridge = np.maximum(ridge, curvature * coherence)
    ridge = _robust_unit_interval(ridge)
    total_weight = max(config.intensity_weight + config.ridge_weight, 1e-6)
    score = (
        config.intensity_weight * intensity + config.ridge_weight * ridge
    ) / total_weight
    return intensity, ridge, score.astype(np.float32, copy=False)


def structural_candidate_mask(
    intensity: np.ndarray,
    ridge: np.ndarray,
    score: np.ndarray,
    pixel_size_um: tuple[float, float],
    config: GFAPStructureConfig,
) -> np.ndarray:
    """Threshold and clean supported GFAP structures without using target channels."""

    positive = intensity > 0
    if not positive.any():
        return np.zeros(intensity.shape, dtype=bool)
    intensity_floor = float(
        np.percentile(intensity[positive], config.intensity_floor_percentile)
    )
    score_floor = float(np.percentile(score[positive], config.structural_percentile))
    ridge_positive = ridge[ridge > 0]
    ridge_floor = (
        float(np.percentile(ridge_positive, config.strong_ridge_percentile))
        if ridge_positive.size
        else 1.0
    )
    candidate = (
        ((score >= score_floor) & (intensity >= intensity_floor))
        | ((ridge >= ridge_floor) & (intensity >= 0.5 * intensity_floor))
    )
    if config.connection_gap_um > 0:
        footprint = _elliptical_footprint(config.connection_gap_um, pixel_size_um)
        candidate = morphology.binary_closing(candidate, footprint=footprint)
    pixel_area = pixel_size_um[0] * pixel_size_um[1]
    min_pixels = max(2, int(np.ceil(config.min_structure_area_um2 / pixel_area)))
    return morphology.remove_small_objects(candidate, min_size=min_pixels)


def extract_gfap_structure(
    gfap_image: np.ndarray,
    pixel_size_um: tuple[float, float],
    config: GFAPStructureConfig,
    *,
    return_peak_z: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """Run the complete GFAP-only structural preprocessing pipeline."""

    corrected_result = background_correct_gfap(
        gfap_image,
        pixel_size_um,
        config,
        return_peak_z=return_peak_z,
    )
    if return_peak_z:
        corrected, peak_z = corrected_result
    else:
        corrected = corrected_result
    intensity, ridge, score = multiscale_fibre_score(
        corrected,
        pixel_size_um,
        config,
    )
    candidate = structural_candidate_mask(
        intensity,
        ridge,
        score,
        pixel_size_um,
        config,
    )
    if return_peak_z:
        return corrected, intensity, ridge, score, candidate, peak_z
    return corrected, intensity, ridge, score, candidate
