from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from project_leap_2d.fiji_review.measurement_result_validation import (
    REQUIRED_MEASUREMENT_HEADINGS,
    validate_fiji_measurement_result,
)


def _result_set(
    *,
    area: float,
    integrated_density: float,
) -> dict:
    return {
        "rows": 1,
        "astrocyte_ids": [1],
        "original_astrocyte_ids": [1],
        "headings": sorted(REQUIRED_MEASUREMENT_HEADINGS),
        "row_data": [
            {
                "Astrocyte_ID": 1,
                "Original_Astrocyte_ID": 1,
                "Median": 4.5,
            }
        ],
        "area_sum": area,
        "integrated_density_sum": integrated_density,
    }


class MeasurementResultValidationTests(unittest.TestCase):
    def _valid_details(self, run_dir: Path) -> dict:
        overlays = {}
        for key in ("whole", "processes", "soma"):
            path = run_dir / f"{key}.png"
            Image.new("RGB", (9, 8), (10, 20, 30)).save(path)
            overlays[key] = str(path)
        return {
            "roi_count": 1,
            "final_roi_ids": [1],
            "final_process_ids": [1],
            "final_soma_ids": [1],
            "result_sets": {
                "whole": _result_set(area=10.0, integrated_density=100.0),
                "processes": _result_set(area=4.0, integrated_density=40.0),
                "soma": _result_set(area=6.0, integrated_density=60.0),
            },
            "overlay_paths": overlays,
        }

    def test_valid_triplet_and_run_local_overlays_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            result = validate_fiji_measurement_result(
                fiji_details=self._valid_details(run_dir),
                run_dir=run_dir,
                image_shape_yx=(8, 9),
                pixel_width_um=0.1,
                pixel_height_um=0.1,
            )
            self.assertEqual(set(result), {"whole", "processes", "soma"})

    def test_partition_integrated_density_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            details = self._valid_details(run_dir)
            details["result_sets"]["soma"]["integrated_density_sum"] = 55.0
            with self.assertRaisesRegex(RuntimeError, "integrated-density"):
                validate_fiji_measurement_result(
                    fiji_details=details,
                    run_dir=run_dir,
                    image_shape_yx=(8, 9),
                    pixel_width_um=0.1,
                    pixel_height_um=0.1,
                )

    def test_overlay_outside_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            details = self._valid_details(run_dir)
            outside = root / "outside.png"
            Image.new("RGB", (9, 8)).save(outside)
            details["overlay_paths"]["whole"] = str(outside)
            with self.assertRaisesRegex(RuntimeError, "run-specific"):
                validate_fiji_measurement_result(
                    fiji_details=details,
                    run_dir=run_dir,
                    image_shape_yx=(8, 9),
                    pixel_width_um=0.1,
                    pixel_height_um=0.1,
                )


if __name__ == "__main__":
    unittest.main()
