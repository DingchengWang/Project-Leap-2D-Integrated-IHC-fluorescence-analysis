"""Post-compartment quality filtering for GFAP-only astrocytes.

The filter computes the approved owner-centered hub distance directly from
the unsmoothed assigned-Processes and Soma masks using calibrated Euclidean
distance transforms.  It then filters and renumbers every synchronized label
and source/display identity record as one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class GFAPPostCompartmentQualityConfig:
    """Calibrated mature defaults applied after compartment construction."""

    minimum_process_area_um2: float = 15.0
    minimum_process_whole_fraction: float = 0.12
    maximum_owner_centered_hub_distance_um: float = 15.0


@dataclass(frozen=True)
class GFAPPostCompartmentQualityResult:
    """Synchronized filtered labels plus source/display identity audit."""

    whole_labels: np.ndarray
    soma_labels: np.ndarray
    process_labels: np.ndarray
    nucleus_labels_2d: np.ndarray
    source_owner_to_display_id: dict[int, int]
    old_display_to_new_display_id: dict[int, int]
    removed_display_ids: tuple[int, ...]
    retained_display_ids: tuple[int, ...]
    records: tuple[dict[str, Any], ...]


def _pixel_geometry(
    pixel_size_um: float | tuple[float, float],
) -> tuple[float, float, float]:
    if np.isscalar(pixel_size_um):
        y_um = x_um = float(pixel_size_um)
    else:
        if len(pixel_size_um) != 2:
            raise ValueError("pixel_size_um must be a scalar or (Y, X)")
        y_um, x_um = float(pixel_size_um[0]), float(pixel_size_um[1])
    if not all(np.isfinite(value) and value > 0 for value in (y_um, x_um)):
        raise ValueError("pixel_size_um values must be finite and positive")
    return y_um, x_um, y_um * x_um


def _validate_triplet(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    whole = np.asarray(whole_labels, dtype=np.int32)
    soma = np.asarray(soma_labels, dtype=np.int32)
    processes = np.asarray(process_labels, dtype=np.int32)
    if whole.ndim != 2 or whole.shape != soma.shape or whole.shape != processes.shape:
        raise ValueError("Whole, Soma, and Processes must share one 2D shape")
    if np.any(whole < 0) or np.any(soma < 0) or np.any(processes < 0):
        raise ValueError("Compartment labels cannot be negative")
    ids = tuple(sorted(int(value) for value in np.unique(whole) if value > 0))
    if not ids:
        raise ValueError("Post-compartment quality requires at least one Whole ID")
    expected = set(ids)
    observed_soma = {int(value) for value in np.unique(soma) if value > 0}
    if observed_soma != expected:
        raise ValueError(
            "Soma IDs do not exactly match Whole IDs: "
            f"{sorted(observed_soma)} versus {list(ids)}"
        )
    observed_processes = {
        int(value) for value in np.unique(processes) if value > 0
    }
    if not observed_processes.issubset(expected):
        raise ValueError("Processes contain an ID absent from Whole")
    if np.any((soma > 0) & (whole != soma)):
        raise ValueError("Soma escaped or changed its Whole owner")
    if np.any((processes > 0) & (whole != processes)):
        raise ValueError("Processes escaped or changed their Whole owner")
    occupancy = (soma > 0).astype(np.uint8) + (processes > 0).astype(np.uint8)
    if np.any(occupancy[whole > 0] != 1) or np.any(occupancy[whole == 0] != 0):
        raise ValueError("Whole is not exactly partitioned into Soma and Processes")
    return whole, soma, processes, ids


def _source_by_display(
    source_owner_to_display_id: Mapping[int, int],
    display_ids: tuple[int, ...],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for source_owner_id, display_id in source_owner_to_display_id.items():
        source = int(source_owner_id)
        display = int(display_id)
        if display in result:
            raise ValueError(f"Multiple source owners map to display ID {display}")
        result[display] = source
    if set(result) != set(display_ids):
        raise ValueError(
            "source_owner_to_display_id does not exactly cover active labels"
        )
    return result


def _remap_labels(
    labels: np.ndarray,
    old_to_new: Mapping[int, int],
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        output[labels == int(old_id)] = int(new_id)
    return output


def classify_gfap_post_compartment_metrics(
    *,
    assigned_process_area_um2: float,
    processes_whole_fraction: float,
    owner_centered_hub_distance_um: float | None,
    soma_touches_image_border: bool = False,
    config: GFAPPostCompartmentQualityConfig | None = None,
) -> tuple[str, ...]:
    """Classify already measured metrics using the exact release thresholds."""

    active = config or GFAPPostCompartmentQualityConfig()
    reasons: list[str] = []
    if owner_centered_hub_distance_um is None:
        reasons.append("empty_assigned_processes")
    if float(assigned_process_area_um2) < active.minimum_process_area_um2:
        reasons.append("assigned_process_area_below_minimum")
    if float(processes_whole_fraction) < active.minimum_process_whole_fraction:
        reasons.append("processes_whole_fraction_below_minimum")
    if (
        owner_centered_hub_distance_um is not None
        and float(owner_centered_hub_distance_um)
        > active.maximum_owner_centered_hub_distance_um
    ):
        reasons.append("owner_centered_hub_distance_above_maximum")
    if bool(soma_touches_image_border):
        reasons.append("soma_touches_image_border")
    return tuple(reasons)


def apply_gfap_post_compartment_quality(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    nucleus_labels_2d: np.ndarray,
    pixel_size_um: float | tuple[float, float],
    source_owner_to_display_id: Mapping[int, int],
    *,
    config: GFAPPostCompartmentQualityConfig | None = None,
) -> GFAPPostCompartmentQualityResult:
    """Compute EDT hub metrics, filter cells, and synchronize all identities.

    For each display ID, the Processes EDT gives local thickness.  Every
    Processes pixel within numerical tolerance of that cell's maximum
    thickness is a hub; the recorded hub distance is the minimum unsmoothed
    Soma-distance EDT value across those hub pixels.
    """

    whole, soma, processes, display_ids = _validate_triplet(
        whole_labels,
        soma_labels,
        process_labels,
    )
    nuclei = np.asarray(nucleus_labels_2d, dtype=np.int32)
    if nuclei.shape != whole.shape or np.any(nuclei < 0):
        raise ValueError("nucleus_labels_2d must match the compartment geometry")
    nucleus_ids = {int(value) for value in np.unique(nuclei) if value > 0}
    if nucleus_ids != set(display_ids):
        raise ValueError("Nucleus IDs do not exactly match Whole IDs")
    if np.any((nuclei > 0) & (whole != nuclei)):
        raise ValueError("A nucleus escaped or changed its Whole owner")
    pixel_y_um, pixel_x_um, pixel_area_um2 = _pixel_geometry(pixel_size_um)
    active = config or GFAPPostCompartmentQualityConfig()
    if active.minimum_process_area_um2 < 0:
        raise ValueError("minimum_process_area_um2 cannot be negative")
    if not 0 <= active.minimum_process_whole_fraction <= 1:
        raise ValueError("minimum_process_whole_fraction must be between 0 and 1")
    if active.maximum_owner_centered_hub_distance_um < 0:
        raise ValueError("maximum_owner_centered_hub_distance_um cannot be negative")

    source_by_display = _source_by_display(
        source_owner_to_display_id,
        display_ids,
    )
    retained: list[int] = []
    removed: list[int] = []
    records: list[dict[str, Any]] = []
    for display_id in display_ids:
        whole_pixels = int(np.count_nonzero(whole == display_id))
        soma_pixels = int(np.count_nonzero(soma == display_id))
        process_pixels = int(np.count_nonzero(processes == display_id))
        process_area_um2 = process_pixels * pixel_area_um2
        whole_area_um2 = whole_pixels * pixel_area_um2
        process_fraction = process_pixels / max(whole_pixels, 1)
        process_mask = processes == display_id
        soma_mask = soma == display_id
        whole_mask = whole == display_id
        if process_pixels:
            thickness = np.asarray(
                ndi.distance_transform_edt(
                    process_mask,
                    sampling=(pixel_y_um, pixel_x_um),
                ),
                dtype=np.float64,
            )
            max_thickness_um = float(thickness[process_mask].max())
            hub_tolerance = max(1e-9, 1e-6 * max_thickness_um)
            hubs = process_mask & (
                thickness >= max_thickness_um - hub_tolerance
            )
            soma_distance = ndi.distance_transform_edt(
                ~soma_mask,
                sampling=(pixel_y_um, pixel_x_um),
            )
            hub_distance_um: float | None = float(soma_distance[hubs].min())
            hub_pixel_count = int(np.count_nonzero(hubs))
        else:
            max_thickness_um = 0.0
            hub_distance_um = None
            hub_pixel_count = 0
        soma_touches_image_border = bool(
            soma_mask[0].any()
            or soma_mask[-1].any()
            or soma_mask[:, 0].any()
            or soma_mask[:, -1].any()
        )
        whole_touches_image_border = bool(
            whole_mask[0].any()
            or whole_mask[-1].any()
            or whole_mask[:, 0].any()
            or whole_mask[:, -1].any()
        )
        processes_touch_image_border = bool(
            process_mask[0].any()
            or process_mask[-1].any()
            or process_mask[:, 0].any()
            or process_mask[:, -1].any()
        )
        incomplete_morphology = bool(
            whole_touches_image_border or processes_touch_image_border
        )
        reasons = list(
            classify_gfap_post_compartment_metrics(
                assigned_process_area_um2=process_area_um2,
                processes_whole_fraction=process_fraction,
                owner_centered_hub_distance_um=hub_distance_um,
                soma_touches_image_border=soma_touches_image_border,
                config=active,
            )
        )
        passed = not reasons
        (retained if passed else removed).append(display_id)
        records.append(
            {
                "source_owner_id": source_by_display[display_id],
                "old_display_id": display_id,
                "new_display_id": None,
                "whole_area_um2": whole_area_um2,
                "soma_area_um2": soma_pixels * pixel_area_um2,
                "assigned_process_area_um2": process_area_um2,
                "processes_whole_fraction": process_fraction,
                "owner_centered_hub_distance_um": hub_distance_um,
                "maximum_process_thickness_um": max_thickness_um,
                "hub_pixel_count": hub_pixel_count,
                "soma_touches_image_border": soma_touches_image_border,
                "whole_touches_image_border": whole_touches_image_border,
                "processes_touch_image_border": processes_touch_image_border,
                "incomplete_morphology": incomplete_morphology,
                "passed_post_compartment_quality": passed,
                "rejection_reasons": reasons,
            }
        )

    if not retained:
        raise ValueError(
            "Post-compartment quality rejected every GFAP-only astrocyte"
        )
    old_to_new = {
        old_id: new_id for new_id, old_id in enumerate(retained, start=1)
    }
    filtered_whole = _remap_labels(whole, old_to_new)
    filtered_soma = _remap_labels(soma, old_to_new)
    filtered_processes = _remap_labels(processes, old_to_new)
    filtered_nuclei = _remap_labels(nuclei, old_to_new)
    _validate_triplet(filtered_whole, filtered_soma, filtered_processes)

    filtered_source_map = {
        source_by_display[old_id]: new_id
        for old_id, new_id in old_to_new.items()
    }
    final_records: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        old_id = int(record["old_display_id"])
        updated["new_display_id"] = old_to_new.get(old_id)
        final_records.append(updated)
    return GFAPPostCompartmentQualityResult(
        whole_labels=filtered_whole,
        soma_labels=filtered_soma,
        process_labels=filtered_processes,
        nucleus_labels_2d=filtered_nuclei,
        source_owner_to_display_id=filtered_source_map,
        old_display_to_new_display_id=old_to_new,
        removed_display_ids=tuple(removed),
        retained_display_ids=tuple(retained),
        records=tuple(final_records),
    )


__all__ = [
    "GFAPPostCompartmentQualityConfig",
    "GFAPPostCompartmentQualityResult",
    "apply_gfap_post_compartment_quality",
    "classify_gfap_post_compartment_metrics",
]
