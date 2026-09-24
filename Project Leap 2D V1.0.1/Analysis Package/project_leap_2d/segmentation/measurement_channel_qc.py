"""Read-only technical QC for the KCNN/KCNJ measurement-channel stack.

This module deliberately has no access to Whole masks, candidate scores, or
best-candidate selection.  It reports whether each already planned Z window contains
readable measurement-channel pixels and records descriptive acquisition
metrics for Terminal/report output.  Low signal is a valid biological result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from project_leap_2d.data_structures import ZWindowPlan


MEASUREMENT_MAX_ENDPOINT_FRACTION = 0.05
MEASUREMENT_MIN_NORMALIZED_SPATIAL_SHARPNESS = 0.002
MEASUREMENT_MAX_NORMALIZED_BACKGROUND_NONUNIFORMITY = 0.18


@dataclass(frozen=True)
class MeasurementChannelWindowQC:
    """Immutable technical observations for one planned Z window."""

    position_1based: int
    z0: int
    z1: int
    z_mode: str
    status: str
    invalid_reasons: tuple[str, ...]
    expected_slices: int
    observed_slices: int
    pixel_count: int
    finite_fraction: float
    lower_endpoint_fraction: float | None
    upper_endpoint_fraction: float | None
    constant_slice_count: int
    constant_slice_fraction: float
    normalized_spatial_sharpness: float | None
    normalized_background_nonuniformity: float | None


@dataclass(frozen=True)
class _PlaneTechnicalStats:
    """Small scalar summary retained after one source Z plane is released."""

    pixel_count: int
    finite_count: int
    value_min: float | None
    value_max: float | None
    lower_endpoint_count: int | None
    upper_endpoint_count: int | None
    is_constant: bool
    x_difference_sum: float
    x_difference_count: int
    y_difference_sum: float
    y_difference_count: int
    tile_backgrounds: tuple[float, ...]


def _digital_endpoints(dtype: np.dtype) -> tuple[object, object] | None:
    if np.issubdtype(dtype, np.bool_):
        return False, True
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return limits.min, limits.max
    return None


def _inspect_plane(
    plane: np.ndarray,
    digital_endpoints: tuple[object, object] | None,
) -> _PlaneTechnicalStats:
    """Inspect one native plane, limiting floating work arrays to two dimensions."""

    pixel_count = int(plane.size)
    if np.issubdtype(plane.dtype, np.inexact):
        finite_count = int(np.count_nonzero(np.isfinite(plane)))
    else:
        finite_count = pixel_count
    all_finite = finite_count == pixel_count

    lower_endpoint_count: int | None = None
    upper_endpoint_count: int | None = None
    if digital_endpoints is not None:
        lower_endpoint, upper_endpoint = digital_endpoints
        lower_endpoint_count = int(np.count_nonzero(plane == lower_endpoint))
        upper_endpoint_count = int(np.count_nonzero(plane == upper_endpoint))

    if not all_finite:
        return _PlaneTechnicalStats(
            pixel_count=pixel_count,
            finite_count=finite_count,
            value_min=None,
            value_max=None,
            lower_endpoint_count=lower_endpoint_count,
            upper_endpoint_count=upper_endpoint_count,
            is_constant=False,
            x_difference_sum=0.0,
            x_difference_count=0,
            y_difference_sum=0.0,
            y_difference_count=0,
            tile_backgrounds=(),
        )

    value_min = float(np.min(plane))
    value_max = float(np.max(plane))
    values = np.asarray(plane, dtype=np.float32)
    y_difference_sum = 0.0
    y_difference_count = 0
    if values.shape[0] > 1:
        difference = np.diff(values, axis=0)
        np.abs(difference, out=difference)
        y_difference_sum = float(np.sum(difference, dtype=np.float64))
        y_difference_count = int(difference.size)
        del difference
    x_difference_sum = 0.0
    x_difference_count = 0
    if values.shape[1] > 1:
        difference = np.diff(values, axis=1)
        np.abs(difference, out=difference)
        x_difference_sum = float(np.sum(difference, dtype=np.float64))
        x_difference_count = int(difference.size)
        del difference

    height, width = values.shape
    y_edges = np.linspace(0, height, min(4, height) + 1, dtype=int)
    x_edges = np.linspace(0, width, min(4, width) + 1, dtype=int)
    tile_backgrounds: list[float] = []
    for y_index in range(len(y_edges) - 1):
        y0, y1 = int(y_edges[y_index]), int(y_edges[y_index + 1])
        for x_index in range(len(x_edges) - 1):
            x0, x1 = int(x_edges[x_index]), int(x_edges[x_index + 1])
            tile = plane[y0:y1, x0:x1]
            if tile.size:
                tile_backgrounds.append(float(np.percentile(tile, 20.0)))

    return _PlaneTechnicalStats(
        pixel_count=pixel_count,
        finite_count=finite_count,
        value_min=value_min,
        value_max=value_max,
        lower_endpoint_count=lower_endpoint_count,
        upper_endpoint_count=upper_endpoint_count,
        is_constant=value_min == value_max,
        x_difference_sum=x_difference_sum,
        x_difference_count=x_difference_count,
        y_difference_sum=y_difference_sum,
        y_difference_count=y_difference_count,
        tile_backgrounds=tuple(tile_backgrounds),
    )


def inspect_measurement_channel_windows(
    measurement_stack_zyx: np.ndarray,
    windows: Sequence["ZWindowPlan"],
) -> tuple[MeasurementChannelWindowQC, ...]:
    """Inspect raw KCNN/KCNJ pixels for each of exactly five Z windows.

    The function never produces a score or selection recommendation.  A
    window is ``invalid`` only when its requested geometry cannot be read,
    pixels are non-finite, any complete Z plane is constant, the complete
    window is constant, or at least five percent of pixels reach the raw
    integer dtype's digital maximum, has calibrated severe defocus, or has
    calibrated severe background nonuniformity.  The lower endpoint remains
    descriptive so a valid low-signal biological result is not rejected.
    """

    stack = np.asarray(measurement_stack_zyx)
    if stack.ndim != 3:
        raise ValueError("measurement_stack_zyx must have ZYX dimensions")
    if stack.shape[1] <= 0 or stack.shape[2] <= 0:
        raise ValueError("measurement_stack_zyx must have non-empty Y and X axes")
    if len(windows) != 5:
        raise ValueError("measurement-channel QC requires exactly five Z windows")

    plan_geometry: list[tuple[object, int, int, int, bool]] = []
    required_planes: set[int] = set()
    for plan in windows:
        z0 = int(plan.z0)
        z1 = int(plan.z1)
        expected_slices = int(plan.window_slices)
        geometry_matches = z0 >= 0 and z1 >= z0 and (
            z1 - z0 + 1 == expected_slices
        )
        in_bounds = geometry_matches and z1 < stack.shape[0]
        plan_geometry.append((plan, z0, z1, expected_slices, in_bounds))
        if in_bounds:
            required_planes.update(range(z0, z1 + 1))

    digital_endpoints = _digital_endpoints(stack.dtype)
    plane_stats = {
        z_index: _inspect_plane(stack[z_index], digital_endpoints)
        for z_index in sorted(required_planes)
    }

    results: list[MeasurementChannelWindowQC] = []
    for plan, z0, z1, expected_slices, in_bounds in plan_geometry:
        selected_stats = (
            [plane_stats[z_index] for z_index in range(z0, z1 + 1)]
            if in_bounds
            else []
        )
        observed_slices = len(selected_stats)
        invalid_reasons: list[str] = []
        if not in_bounds or observed_slices != expected_slices:
            invalid_reasons.append("window_slice_count_mismatch")

        pixel_count = sum(item.pixel_count for item in selected_stats)
        finite_count = sum(item.finite_count for item in selected_stats)
        finite_fraction = float(finite_count / pixel_count) if pixel_count else 0.0
        all_finite = bool(pixel_count and finite_count == pixel_count)
        if pixel_count and not all_finite:
            invalid_reasons.append("non_finite_pixels")

        if digital_endpoints is None or not pixel_count:
            lower_fraction = None
            upper_fraction = None
        else:
            lower_fraction = float(
                sum(int(item.lower_endpoint_count or 0) for item in selected_stats)
                / pixel_count
            )
            upper_fraction = float(
                sum(int(item.upper_endpoint_count or 0) for item in selected_stats)
                / pixel_count
            )
        all_digital_maximum = bool(
            pixel_count
            and upper_fraction is not None
            and upper_fraction == 1.0
        )
        if all_digital_maximum:
            invalid_reasons.append("entire_window_digital_maximum")
        elif (
            upper_fraction is not None
            and upper_fraction >= MEASUREMENT_MAX_ENDPOINT_FRACTION
        ):
            invalid_reasons.append("upper_endpoint_saturation")

        constant_slice_count = 0
        constant_slice_fraction = 0.0
        sharpness: float | None = None
        background_nonuniformity: float | None = None
        if all_finite:
            constant_slice_count = sum(item.is_constant for item in selected_stats)
            constant_slice_fraction = float(constant_slice_count / observed_slices)
            if constant_slice_count:
                invalid_reasons.append("constant_z_slice")
            value_min = min(
                float(item.value_min) for item in selected_stats
                if item.value_min is not None
            )
            value_max = max(
                float(item.value_max) for item in selected_stats
                if item.value_max is not None
            )
            dynamic_range = value_max - value_min
            if dynamic_range == 0.0:
                if not all_digital_maximum:
                    invalid_reasons.append("entire_window_constant")
            else:
                axis_means: list[float] = []
                x_count = sum(item.x_difference_count for item in selected_stats)
                if x_count:
                    axis_means.append(
                        sum(item.x_difference_sum for item in selected_stats)
                        / x_count
                    )
                y_count = sum(item.y_difference_count for item in selected_stats)
                if y_count:
                    axis_means.append(
                        sum(item.y_difference_sum for item in selected_stats)
                        / y_count
                    )
                sharpness = (
                    float(np.mean(axis_means) / dynamic_range)
                    if axis_means
                    else 0.0
                )
                tile_backgrounds = tuple(
                    value
                    for item in selected_stats
                    for value in item.tile_backgrounds
                )
                if tile_backgrounds:
                    lower, upper = np.percentile(
                        tile_backgrounds,
                        (10.0, 90.0),
                    )
                    background_nonuniformity = float(
                        (upper - lower) / dynamic_range
                    )
                else:
                    background_nonuniformity = 0.0
                sharpness_is_assessable = bool(
                    lower_fraction is None
                    or 1.0 - lower_fraction
                    >= MEASUREMENT_MAX_ENDPOINT_FRACTION
                )
                if (
                    sharpness_is_assessable
                    and sharpness
                    < MEASUREMENT_MIN_NORMALIZED_SPATIAL_SHARPNESS
                ):
                    invalid_reasons.append("severe_defocus")
                if (
                    background_nonuniformity
                    >= MEASUREMENT_MAX_NORMALIZED_BACKGROUND_NONUNIFORMITY
                ):
                    invalid_reasons.append("severe_background_nonuniformity")

        results.append(
            MeasurementChannelWindowQC(
                position_1based=int(plan.position_1based),
                z0=z0,
                z1=z1,
                z_mode=str(plan.z_mode),
                status="invalid" if invalid_reasons else "valid",
                invalid_reasons=tuple(invalid_reasons),
                expected_slices=expected_slices,
                observed_slices=observed_slices,
                pixel_count=pixel_count,
                finite_fraction=finite_fraction,
                lower_endpoint_fraction=lower_fraction,
                upper_endpoint_fraction=upper_fraction,
                constant_slice_count=constant_slice_count,
                constant_slice_fraction=constant_slice_fraction,
                normalized_spatial_sharpness=sharpness,
                normalized_background_nonuniformity=background_nonuniformity,
            )
        )
    return tuple(results)


__all__ = [
    "MeasurementChannelWindowQC",
    "inspect_measurement_channel_windows",
]
