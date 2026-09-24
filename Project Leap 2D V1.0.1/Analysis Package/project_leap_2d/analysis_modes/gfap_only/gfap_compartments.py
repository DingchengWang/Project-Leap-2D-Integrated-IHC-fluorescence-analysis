"""Exclusive nucleus-to-GFAP ownership and compartment construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation


@dataclass(frozen=True)
class GFAPCompartmentConfig:
    """Physical constraints for soma and GFAP-process ownership."""

    soma_base_margin_um: float = 0.80
    soma_max_margin_um: float = 1.25
    ownership_seed_max_margin_um: float = 2.10
    soma_support_percentile: float = 38.0
    soma_closing_um: float = 0.22
    ownership_connectivity: int = 2
    weak_structure_extension_um: float = 0.28
    weak_structure_score_fraction: float = 0.88
    z_competition_weight: float = 0.65
    z_reassignment_margin_um: float = 0.35


def _elliptical_footprint(radius_um: float, pixel_size_um: tuple[float, float]) -> np.ndarray:
    y_um, x_um = pixel_size_um
    ry = max(1, int(np.ceil(radius_um / y_um)))
    rx = max(1, int(np.ceil(radius_um / x_um)))
    yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    return ((yy * y_um) ** 2 + (xx * x_um) ** 2) <= radius_um**2


def build_soma_labels(
    nucleus_labels_2d: np.ndarray,
    structural_score: np.ndarray,
    pixel_size_um: tuple[float, float],
    config: GFAPCompartmentConfig,
    *,
    hard_exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Expand every complete nucleus by a limited, competitively owned margin."""

    nuclei = np.asarray(nucleus_labels_2d, dtype=np.int32)
    nucleus_mask = nuclei > 0
    if not nucleus_mask.any():
        return np.zeros(nuclei.shape, dtype=np.int32)
    if hard_exclusion_mask is None:
        excluded = np.zeros(nuclei.shape, dtype=bool)
    else:
        excluded = np.asarray(hard_exclusion_mask, dtype=bool)
        if excluded.shape != nuclei.shape:
            raise ValueError("Soma hard-exclusion mask must match nucleus geometry")
        excluded = excluded & ~nucleus_mask

    distance, nearest_indices = ndi.distance_transform_edt(
        ~nucleus_mask,
        sampling=pixel_size_um,
        return_indices=True,
    )
    nearest_id = nuclei[tuple(nearest_indices)]
    positive_score = structural_score[structural_score > 0]
    support_floor = (
        float(np.percentile(positive_score, config.soma_support_percentile))
        if positive_score.size
        else 1.0
    )
    mandatory = distance <= config.soma_base_margin_um
    supported = (
        (distance <= config.soma_max_margin_um)
        & (structural_score >= support_floor)
    )
    soma = np.where((mandatory | supported) & ~excluded, nearest_id, 0).astype(
        np.int32
    )
    soma[nucleus_mask] = nuclei[nucleus_mask]

    if config.soma_closing_um > 0:
        footprint = _elliptical_footprint(config.soma_closing_um, pixel_size_um)
        closed = np.zeros_like(soma)
        for nucleus_id in (int(value) for value in np.unique(soma) if value > 0):
            region = morphology.binary_closing(soma == nucleus_id, footprint=footprint)
            available = region & (closed == 0) & ~excluded
            closed[available] = nucleus_id
        closed[excluded] = 0
        closed[nucleus_mask] = nuclei[nucleus_mask]
        soma = closed
    return soma


def assign_exclusive_gfap_ownership(
    soma_labels: np.ndarray,
    structural_mask: np.ndarray,
    structural_score: np.ndarray,
    *,
    connectivity: int = 2,
    competition_seed_labels: np.ndarray | None = None,
    gfap_peak_z_yx: np.ndarray | None = None,
    marker_z_ranges: dict[int, tuple[int, int]] | None = None,
    z_spacing_um: float = 1.0,
    pixel_size_um: tuple[float, float] = (1.0, 1.0),
    weak_structure_extension_um: float = 0.0,
    weak_structure_score_fraction: float = 0.88,
    z_competition_weight: float = 0.65,
    z_reassignment_margin_um: float = 0.35,
) -> np.ndarray:
    """Assign connected GFAP structures by competitive signal-path costs.

    Valid 3D nuclei that are not output owners may be supplied as temporary
    competition seeds.  Their regions are left unowned after competition, so
    a GFAP-poor neighbouring cell cannot silently donate its branches to an
    accepted astrocyte.
    """

    soma = np.asarray(soma_labels, dtype=np.int32)
    structural = np.asarray(structural_mask, dtype=bool)
    score = np.asarray(structural_score, dtype=np.float32)
    if structural.shape != soma.shape or score.shape != soma.shape:
        raise ValueError("GFAP structure arrays must match Soma geometry")
    owner_ids = {int(value) for value in np.unique(soma) if value > 0}
    if not owner_ids:
        return soma.copy()

    domain_structure = structural.copy()
    if weak_structure_extension_um > 0 and structural.any():
        y_um, x_um = (float(pixel_size_um[0]), float(pixel_size_um[1]))
        if not all(np.isfinite(value) and value > 0 for value in (y_um, x_um)):
            raise ValueError("pixel_size_um values must be finite and positive")
        strong_values = score[structural]
        weak_floor = (
            float(np.percentile(strong_values, 25.0))
            * float(weak_structure_score_fraction)
        )
        distance_to_strong = ndi.distance_transform_edt(
            ~structural,
            sampling=(y_um, x_um),
        )
        weak = (
            (score >= weak_floor)
            & (distance_to_strong <= float(weak_structure_extension_um))
        )
        weak_domain = structural | weak
        weak_components, _ = ndi.label(
            weak_domain,
            structure=ndi.generate_binary_structure(2, 2),
        )
        seeded_components = {
            int(value)
            for value in np.unique(weak_components[structural])
            if value > 0
        }
        domain_structure = np.isin(
            weak_components,
            tuple(sorted(seeded_components)),
        )

    markers = soma.copy()
    if competition_seed_labels is not None:
        competition = np.asarray(competition_seed_labels, dtype=np.int32)
        if competition.shape != soma.shape or np.any(competition < 0):
            raise ValueError("Competition nucleus labels must match Soma geometry")
        competition_ids = {int(value) for value in np.unique(competition) if value > 0}
        if competition_ids & owner_ids:
            raise ValueError(
                "Competition marker IDs must be distinct from accepted Soma IDs"
            )
        empty = markers == 0
        markers[empty & (competition > 0)] = competition[empty & (competition > 0)]

    domain = domain_structure | (markers > 0)
    if not domain.any() or not (soma > 0).any():
        return soma.copy()
    elevation = 1.0 - np.clip(score, 0.0, 1.0)
    elevation = elevation.astype(np.float32, copy=False)
    whole = segmentation.watershed(
        elevation,
        markers=markers,
        mask=domain,
        connectivity=connectivity,
        watershed_line=False,
    ).astype(np.int32)

    if (
        gfap_peak_z_yx is not None
        and marker_z_ranges
        and len(marker_z_ranges) > 1
    ):
        peak_z = np.asarray(gfap_peak_z_yx, dtype=np.float32)
        if peak_z.shape != soma.shape or not np.isfinite(peak_z).all():
            raise ValueError("GFAP peak-Z map must be finite and match Soma geometry")
        y_um, x_um = (float(pixel_size_um[0]), float(pixel_size_um[1]))
        marker_ids = tuple(
            marker_id
            for marker_id in sorted(marker_z_ranges)
            if np.any(markers == marker_id)
        )
        best_cost = np.full(soma.shape, np.inf, dtype=np.float32)
        best_marker = np.zeros(soma.shape, dtype=np.int32)
        best_z_cost = np.zeros(soma.shape, dtype=np.float32)
        current_total = np.full(soma.shape, np.inf, dtype=np.float32)
        current_z_cost = np.zeros(soma.shape, dtype=np.float32)
        for marker_id in marker_ids:
            seed = markers == marker_id
            spatial_cost = ndi.distance_transform_edt(
                ~seed,
                sampling=(y_um, x_um),
            ).astype(np.float32, copy=False)
            z_first, z_last = marker_z_ranges[marker_id]
            z_distance = np.maximum(
                np.maximum(float(z_first) - peak_z, peak_z - float(z_last)),
                0.0,
            )
            z_cost = z_distance * float(z_spacing_um)
            total_cost = spatial_cost + float(z_competition_weight) * z_cost
            better = total_cost < best_cost
            best_cost[better] = total_cost[better]
            best_marker[better] = marker_id
            best_z_cost[better] = z_cost[better]
            currently_owned = whole == marker_id
            current_total[currently_owned] = total_cost[currently_owned]
            current_z_cost[currently_owned] = z_cost[currently_owned]
        reassign = (
            domain_structure
            & (best_marker > 0)
            & (best_marker != whole)
            & (best_z_cost + 1e-6 < current_z_cost)
            & (
                best_cost + float(z_reassignment_margin_um)
                < current_total
            )
        )
        whole[reassign] = best_marker[reassign]

    # A reassigned fragment is valid only if it remains connected to its seed.
    for marker_id in sorted(int(value) for value in np.unique(whole) if value > 0):
        region = whole == marker_id
        components, _ = ndi.label(
            region,
            structure=ndi.generate_binary_structure(2, 2),
        )
        seed_components = {
            int(value)
            for value in np.unique(components[markers == marker_id])
            if value > 0
        }
        keep = np.isin(components, tuple(sorted(seed_components)))
        whole[region & ~keep] = 0

    whole[~np.isin(whole, tuple(sorted(owner_ids)))] = 0
    whole[soma > 0] = soma[soma > 0]
    return whole


def partition_compartments(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict Whole, Soma, Processes labels after checking all identities."""

    whole = np.asarray(whole_labels, dtype=np.int32)
    soma = np.asarray(soma_labels, dtype=np.int32)
    if whole.shape != soma.shape:
        raise ValueError("Whole and Soma labels have different shapes")
    if np.any((soma > 0) & (whole != soma)):
        raise RuntimeError("GFAP-only Soma is not an identity-preserving Whole subset")
    processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(np.int32)
    if np.any((soma > 0) & (processes > 0)):
        raise RuntimeError("GFAP-only Soma and Processes overlap")
    recombined = np.where(soma > 0, soma, processes)
    if not np.array_equal(recombined, whole):
        raise RuntimeError("GFAP-only Whole != Soma union Processes")
    return whole, soma, processes
