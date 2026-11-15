"""Shared validation for completed Fiji compartment measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_MEASUREMENT_HEADINGS = frozenset(
    {
        "Label",
        "Area",
        "Mean",
        "Median",
        "Min",
        "Max",
        "IntDen",
        "ROI_Index",
        "Astrocyte_ID",
        "Original_Astrocyte_ID",
        "Source_Original_Astrocyte_IDs",
        "Compartment",
        "ROI_Name",
    }
)


def validate_fiji_measurement_result(
    *,
    fiji_details: dict[str, Any],
    run_dir: Path,
    image_shape_yx: tuple[int, int],
    pixel_width_um: float,
    pixel_height_um: float,
) -> dict[str, Path]:
    """Validate IDs, triplet measurements, partitions, and run-local overlays."""

    final_roi_count = int(fiji_details.get("roi_count", -1))
    if final_roi_count < 1:
        raise RuntimeError(
            f"Fiji returned an invalid final ROI count: {fiji_details}"
        )
    final_ids = [int(value) for value in fiji_details.get("final_roi_ids", [])]
    if final_ids != list(range(1, final_roi_count + 1)):
        raise RuntimeError(f"Fiji returned invalid final Astrocyte IDs: {final_ids}")
    process_ids = [
        int(value) for value in fiji_details.get("final_process_ids", [])
    ]
    soma_ids = [int(value) for value in fiji_details.get("final_soma_ids", [])]
    if process_ids != final_ids or soma_ids != final_ids:
        raise RuntimeError(
            "Fiji Whole, Soma, and Processes IDs are not synchronized: "
            f"Whole={final_ids}, Soma={soma_ids}, Processes={process_ids}"
        )

    result_sets = fiji_details.get("result_sets", {})
    expected_original_ids = [
        int(value)
        for value in result_sets.get("whole", {}).get(
            "original_astrocyte_ids",
            [],
        )
    ]
    if len(expected_original_ids) != final_roi_count or len(
        set(expected_original_ids)
    ) != final_roi_count:
        raise RuntimeError(
            f"Fiji returned invalid Original Astrocyte IDs: {expected_original_ids}"
        )
    for key in ("whole", "processes", "soma"):
        detail = result_sets.get(key, {})
        if int(detail.get("rows", -1)) != final_roi_count:
            raise RuntimeError(f"Fiji {key} Results row count is inconsistent: {detail}")
        if [int(value) for value in detail.get("astrocyte_ids", [])] != final_ids:
            raise RuntimeError(f"Fiji {key} Astrocyte IDs are inconsistent: {detail}")
        observed_original_ids = [
            int(value) for value in detail.get("original_astrocyte_ids", [])
        ]
        if observed_original_ids != expected_original_ids:
            raise RuntimeError(
                f"Fiji {key} Original Astrocyte IDs are inconsistent: {detail}"
            )
        headings = set(detail.get("headings", []))
        if not REQUIRED_MEASUREMENT_HEADINGS.issubset(headings):
            raise RuntimeError(
                f"Fiji {key} Results columns are incomplete: {sorted(headings)}"
            )
        rows = detail.get("row_data", [])
        if len(rows) != final_roi_count:
            raise RuntimeError(f"Fiji {key} row payload is incomplete: {detail}")
        if [int(row["Astrocyte_ID"]) for row in rows] != final_ids:
            raise RuntimeError(f"Fiji {key} row Astrocyte IDs are inconsistent")
        if [
            int(row["Original_Astrocyte_ID"]) for row in rows
        ] != expected_original_ids:
            raise RuntimeError(f"Fiji {key} row Original IDs are inconsistent")
        if any(not np.isfinite(float(row["Median"])) for row in rows):
            raise RuntimeError(f"Fiji {key} returned a non-finite Median value")

    whole_area = float(result_sets["whole"]["area_sum"])
    partition_area = float(result_sets["processes"]["area_sum"]) + float(
        result_sets["soma"]["area_sum"]
    )
    area_tolerance = max(
        float(pixel_width_um) * float(pixel_height_um) * final_roi_count * 2.0,
        abs(whole_area) * 1e-6,
    )
    if abs(whole_area - partition_area) > area_tolerance:
        raise RuntimeError(
            "Fiji compartment area identity failed: "
            f"Whole={whole_area}, Processes+Soma={partition_area}"
        )
    whole_intden = float(result_sets["whole"]["integrated_density_sum"])
    partition_intden = float(
        result_sets["processes"]["integrated_density_sum"]
    ) + float(result_sets["soma"]["integrated_density_sum"])
    if abs(whole_intden - partition_intden) > max(
        1.0,
        abs(whole_intden) * 1e-6,
    ):
        raise RuntimeError(
            "Fiji compartment integrated-density identity failed: "
            f"Whole={whole_intden}, Processes+Soma={partition_intden}"
        )

    expected_size = (int(image_shape_yx[1]), int(image_shape_yx[0]))
    run_root = Path(run_dir).resolve()
    overlays: dict[str, Path] = {}
    for key in ("whole", "processes", "soma"):
        path = Path(
            str(fiji_details.get("overlay_paths", {}).get(key, ""))
        ).resolve()
        if path.parent != run_root or not path.is_file():
            raise RuntimeError(
                f"Fiji returned an invalid run-specific {key} overlay path: {path}"
            )
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != expected_size:
                raise RuntimeError(
                    f"Unexpected Fiji {key} overlay format: "
                    f"mode={image.mode}, size={image.size}"
                )
            image.verify()
        overlays[key] = path
    return overlays


__all__ = ["REQUIRED_MEASUREMENT_HEADINGS", "validate_fiji_measurement_result"]
