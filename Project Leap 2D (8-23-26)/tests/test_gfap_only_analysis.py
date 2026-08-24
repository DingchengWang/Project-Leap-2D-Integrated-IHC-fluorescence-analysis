from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import numpy as np
from scipy import ndimage as ndi

from project_leap_2d.analysis_modes.gfap_only import (
    GFAPOnlyConfig,
    analyze_dapi_gfap_only,
)
from project_leap_2d.analysis_modes.gfap_only.gfap_compartments import (
    GFAPCompartmentConfig,
    assign_exclusive_gfap_ownership,
    build_soma_labels,
)
from project_leap_2d.analysis_modes.gfap_only.gfap_only_analysis import (
    _project_all_valid_nuclei,
    _prioritize_accepted_owner_projection,
)
from project_leap_2d.analysis_modes.gfap_only.gfap_post_compartment_quality import (
    GFAPPostCompartmentQualityConfig,
)
from project_leap_2d.compartments.selected_soma_enlargement import (
    enlarge_selected_soma,
    selected_soma_enlargement_config_for_mode,
)


def _draw_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], value: float) -> None:
    y0, x0 = start
    y1, x1 = end
    steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
    yy = np.rint(np.linspace(y0, y1, steps)).astype(int)
    xx = np.rint(np.linspace(x0, x1, steps)).astype(int)
    image[yy, xx] = value


def make_synthetic_field() -> tuple[np.ndarray, np.ndarray]:
    shape = (5, 112, 160)
    nuclei = np.zeros(shape, dtype=np.int32)
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    first = (yy - 55) ** 2 + (xx - 43) ** 2 <= 7**2
    second = (yy - 55) ** 2 + (xx - 117) ** 2 <= 7**2
    for z_index in (1, 2, 3):
        nuclei[z_index, first] = 1
        nuclei[z_index, second] = 2

    signal = np.zeros(shape[1:], dtype=np.float32)
    signal[first | second] = 8.0
    grid_y, grid_x = np.indices(shape[1:])
    for center_y, center_x, outward_angle in (
        (55, 43, np.pi),
        (55, 117, 0.0),
    ):
        radial_distance = np.hypot(grid_y - center_y, grid_x - center_x)
        angle_delta = np.angle(
            np.exp(
                1j
                * (
                    np.arctan2(grid_y - center_y, grid_x - center_x)
                    - outward_angle
                )
            )
        )
        fan = (
            (radial_distance >= 7)
            & (radial_distance <= 38)
            & (np.abs(angle_delta) <= np.pi / 3)
        )
        signal[fan] = np.maximum(signal[fan], 8.0)
        endpoint_x = 8 if outward_angle == np.pi else 152
        for endpoint_y in (35, 55, 75):
            _draw_line(
                signal,
                (center_y, center_x),
                (endpoint_y, endpoint_x),
                15.0,
            )
    _draw_line(signal, (55, 43), (55, 117), 3.8)
    _draw_line(signal, (55, 43), (25, 20), 3.4)
    _draw_line(signal, (55, 43), (88, 18), 3.1)
    _draw_line(signal, (55, 117), (23, 143), 3.4)
    _draw_line(signal, (55, 117), (90, 146), 3.1)
    signal = ndi.gaussian_filter(signal, sigma=1.15)
    background = np.linspace(0.2, 1.1, shape[2], dtype=np.float32)[None, :]
    gfap = np.stack(
        [background + 0.04 * z for z in range(shape[0])],
        axis=0,
    )
    gfap = gfap + signal[None, :, :]
    return nuclei, gfap.astype(np.float32)


def synthetic_config() -> GFAPOnlyConfig:
    base = GFAPOnlyConfig()
    return GFAPOnlyConfig(
        structure=base.structure,
        compartments=base.compartments,
        nucleus_ownership=base.nucleus_ownership,
        post_compartment_quality=GFAPPostCompartmentQualityConfig(
            minimum_process_area_um2=0.0,
            minimum_process_whole_fraction=0.0,
            maximum_owner_centered_hub_distance_um=1000.0,
        ),
    )


class GFAPOnlyAnalysisTests(unittest.TestCase):
    def test_accepted_owner_priority_preserves_foreign_exclusive_projection(
        self,
    ) -> None:
        valid = np.zeros((24, 30), dtype=np.int32)
        accepted = np.zeros_like(valid)
        accepted[7:17, 7:17] = 81
        valid[7:17, 7:17] = 81
        # Simulate the generic all-valid collision rule assigning part of the
        # accepted owner projection to a rejected-but-valid neighboring nucleus.
        valid[10:15, 14:20] = 150

        corrected = _prioritize_accepted_owner_projection(valid, accepted)

        self.assertTrue(np.all(corrected[accepted == 81] == 81))
        self.assertTrue(np.any(corrected[:, 17:20] == 150))
        np.testing.assert_array_equal(corrected > 0, valid > 0)

    def test_collision_bookkeeping_alone_does_not_reject_gfap_enlarge(
        self,
    ) -> None:
        shape = (70, 80)
        valid = np.zeros(shape, dtype=np.int32)
        accepted = np.zeros_like(valid)
        accepted[27:38, 27:38] = 81
        valid[27:38, 27:38] = 81
        valid[31:35, 35:38] = 150
        valid[30:36, 41:45] = 150
        corrected = _prioritize_accepted_owner_projection(valid, accepted)

        owner = accepted == 81
        foreign = corrected == 150
        distance_to_owner = ndi.distance_transform_edt(
            ~owner,
            sampling=(0.20, 0.20),
        )
        old_soma = (owner | (distance_to_owner <= 0.80)) & ~foreign
        old_whole = (owner | (distance_to_owner <= 2.00)) & ~foreign
        whole = np.where(old_whole, 1, 0).astype(np.int32)
        soma = np.where(old_soma, 1, 0).astype(np.int32)
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.int32
        )
        structural = np.zeros(shape, dtype=np.float32)

        stale = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            valid == 81,
            structural,
            valid == 150,
            0.20,
            0.20,
            config=selected_soma_enlargement_config_for_mode("gfap_only"),
        )
        corrected_result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.20,
            0.20,
            config=selected_soma_enlargement_config_for_mode("gfap_only"),
        )

        self.assertEqual(stale.status, "rejected_foreign_nucleus_in_soma")
        self.assertNotEqual(
            corrected_result.status,
            "rejected_foreign_nucleus_in_soma",
        )
        self.assertTrue(corrected_result.approved, corrected_result.message)
        self.assertFalse(np.any(corrected_result.soma_labels[foreign] == 1))

    def test_all_valid_projection_uses_local_objects_and_preserves_overlap_ids(
        self,
    ) -> None:
        labels = np.zeros((6, 40, 40), dtype=np.int32)
        yy, xx = np.indices(labels.shape[1:])
        first = (yy - 20) ** 2 + (xx - 19) ** 2 <= 5**2
        second = (yy - 20) ** 2 + (xx - 21) ** 2 <= 5**2
        labels[1:3, first] = 1
        labels[4:6, second] = 2
        projected, records = _project_all_valid_nuclei(labels, (1, 2))
        self.assertEqual(set(records), {1, 2})
        self.assertEqual(set(np.unique(projected)), {0, 1, 2})
        np.testing.assert_array_equal(projected > 0, np.any(labels > 0, axis=0))

    def test_nonowner_nucleus_is_a_hard_soma_shell_exclusion(self) -> None:
        shape = (60, 70)
        yy, xx = np.indices(shape)
        owner = (yy - 30) ** 2 + (xx - 25) ** 2 <= 3**2
        foreign = (yy - 30) ** 2 + (xx - 34) ** 2 <= 2**2
        nuclei = np.where(owner, 1, 0).astype(np.int32)
        soma = build_soma_labels(
            nuclei,
            np.ones(shape, dtype=np.float32),
            (0.2, 0.2),
            GFAPCompartmentConfig(),
            hard_exclusion_mask=foreign,
        )
        self.assertTrue(np.all(soma[owner] == 1))
        self.assertFalse(np.any(soma[foreign] > 0))

    def test_nonoutput_valid_nucleus_competes_without_becoming_an_owner(self) -> None:
        shape = (70, 120)
        yy, xx = np.indices(shape)
        soma = np.zeros(shape, dtype=np.int32)
        soma[(yy - 35) ** 2 + (xx - 18) ** 2 <= 3**2] = 1
        competitor = np.zeros(shape, dtype=np.int32)
        competitor[(yy - 35) ** 2 + (xx - 96) ** 2 <= 3**2] = 2
        structural = np.zeros(shape, dtype=bool)
        structural[34:37, 18:97] = True
        score = structural.astype(np.float32)
        whole = assign_exclusive_gfap_ownership(
            soma,
            structural,
            score,
            competition_seed_labels=competitor,
        )
        self.assertEqual(int(whole[35, 35]), 1)
        self.assertEqual(int(whole[35, 80]), 0)
        self.assertNotIn(2, np.unique(whole))

    def test_peak_z_resolves_only_a_materially_inconsistent_boundary(self) -> None:
        shape = (50, 64)
        soma = np.zeros(shape, dtype=np.int32)
        soma[24:27, 8:11] = 1
        soma[24:27, 50:53] = 2
        structural = np.zeros(shape, dtype=bool)
        structural[24:27, 10:51] = True
        score = structural.astype(np.float32)
        peak_z = np.zeros(shape, dtype=np.float32)
        peak_z[:, 27:37] = 9.0
        whole = assign_exclusive_gfap_ownership(
            soma,
            structural,
            score,
            gfap_peak_z_yx=peak_z,
            marker_z_ranges={1: (0, 2), 2: (8, 10)},
            z_spacing_um=1.0,
            pixel_size_um=(1.0, 1.0),
        )
        self.assertEqual(int(whole[25, 29]), 2)
        self.assertEqual(int(whole[25, 20]), 1)

    def test_high_confidence_structure_bridges_only_nearby_supported_gap(self) -> None:
        shape = (40, 60)
        soma = np.zeros(shape, dtype=np.int32)
        soma[19:22, 7:10] = 1
        structural = np.zeros(shape, dtype=bool)
        structural[20, 9:25] = True
        structural[20, 27:42] = True
        score = np.zeros(shape, dtype=np.float32)
        score[structural] = 1.0
        score[20, 25:27] = 0.9
        whole = assign_exclusive_gfap_ownership(
            soma,
            structural,
            score,
            pixel_size_um=(0.2, 0.2),
            weak_structure_extension_um=0.4,
            weak_structure_score_fraction=0.82,
        )
        self.assertEqual(int(whole[20, 35]), 1)
        self.assertEqual(int(whole[10, 35]), 0)

    def test_constructs_strict_exclusive_triplet(self) -> None:
        nuclei, gfap = make_synthetic_field()
        result = analyze_dapi_gfap_only(
            nuclei,
            gfap,
            pixel_size_um=(0.20, 0.20),
            z_spacing_um=0.45,
            config=synthetic_config(),
        )
        self.assertEqual(set(np.unique(result.whole_labels)), {0, 1, 2})
        self.assertFalse(
            np.any((result.soma_labels > 0) & (result.process_labels > 0))
        )
        recombined = np.where(
            result.soma_labels > 0,
            result.soma_labels,
            result.process_labels,
        )
        np.testing.assert_array_equal(recombined, result.whole_labels)
        self.assertGreater(int((result.process_labels == 1).sum()), 0)
        self.assertGreater(int((result.process_labels == 2).sum()), 0)
        self.assertFalse(
            np.any(
                (result.whole_labels == 1)
                & (result.nucleus_labels_2d == 2)
            )
        )
        self.assertEqual(
            result.diagnostics["measurement_channels_used_for_roi"],
            [],
        )
        self.assertEqual(
            set(np.unique(result.valid_nucleus_labels_2d)),
            {0, 1, 2},
        )
        self.assertEqual(result.diagnostics["age_profile"], "mature")

    def test_unseeded_gfap_component_remains_unowned(self) -> None:
        nuclei, gfap = make_synthetic_field()
        gfap[:, 5:12, 70:80] += 8.0
        result = analyze_dapi_gfap_only(
            nuclei,
            gfap,
            pixel_size_um=0.20,
            z_spacing_um=0.45,
            config=synthetic_config(),
        )
        self.assertTrue(result.gfap_structural_mask[7:10, 72:78].any())
        self.assertFalse((result.whole_labels[7:10, 72:78] > 0).any())
        self.assertGreater(result.diagnostics["unowned_gfap_structure_px"], 0)

    def test_complete_nuclei_are_always_inside_soma_and_whole(self) -> None:
        nuclei, gfap = make_synthetic_field()
        projection = np.max(nuclei, axis=0)
        result = analyze_dapi_gfap_only(
            nuclei,
            gfap,
            pixel_size_um=0.20,
            z_spacing_um=0.45,
            nucleus_labels_2d=projection,
            config=synthetic_config(),
        )
        np.testing.assert_array_equal(
            result.soma_labels[projection > 0],
            projection[projection > 0],
        )
        np.testing.assert_array_equal(
            result.whole_labels[projection > 0],
            projection[projection > 0],
        )
        self.assertTrue(result.diagnostics["used_supplied_nucleus_projection"])

    def test_config_exposes_only_frozen_mature_parameters(self) -> None:
        parameters = inspect.signature(GFAPOnlyConfig).parameters
        self.assertNotIn("age_profile", parameters)
        self.assertFalse(hasattr(GFAPOnlyConfig, "for_age_profile"))
        mature = GFAPOnlyConfig()
        self.assertEqual(mature.compartments.soma_base_margin_um, 0.80)
        self.assertEqual(mature.compartments.soma_max_margin_um, 1.25)
        self.assertEqual(
            mature.compartments.ownership_seed_max_margin_um,
            2.10,
        )

    def test_final_soma_margin_does_not_change_whole_ownership(self) -> None:
        nuclei, gfap = make_synthetic_field()
        base = synthetic_config()
        wide_config = replace(
            base,
            compartments=replace(
                base.compartments,
                soma_max_margin_um=1.25,
                ownership_seed_max_margin_um=2.10,
            ),
        )
        narrow_config = replace(
            base,
            compartments=replace(
                base.compartments,
                soma_max_margin_um=0.82,
                ownership_seed_max_margin_um=2.10,
            ),
        )
        wide = analyze_dapi_gfap_only(
            nuclei,
            gfap,
            pixel_size_um=0.20,
            z_spacing_um=0.45,
            config=wide_config,
        )
        narrow = analyze_dapi_gfap_only(
            nuclei,
            gfap,
            pixel_size_um=0.20,
            z_spacing_um=0.45,
            config=narrow_config,
        )
        self.assertLess(
            int(np.count_nonzero(narrow.soma_labels)),
            int(np.count_nonzero(wide.soma_labels)),
        )
        np.testing.assert_array_equal(narrow.whole_labels, wide.whole_labels)
        self.assertEqual(
            narrow.diagnostics["source_owner_ids"],
            wide.diagnostics["source_owner_ids"],
        )
        self.assertEqual(
            set(np.unique(narrow.nucleus_labels_2d)),
            set(np.unique(wide.nucleus_labels_2d)),
        )

    def test_api_cannot_accept_egfp_or_measurement_channels(self) -> None:
        parameters = inspect.signature(analyze_dapi_gfap_only).parameters
        for prohibited in ("egfp", "kcnn1", "kcnn2", "measurement_image"):
            self.assertNotIn(prohibited, parameters)

    def test_rejects_mismatched_calibration_and_shapes(self) -> None:
        nuclei, gfap = make_synthetic_field()
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            analyze_dapi_gfap_only(nuclei, gfap, 0.0, 0.45)
        with self.assertRaisesRegex(ValueError, "different XY shapes"):
            analyze_dapi_gfap_only(nuclei, gfap[:, :-1], 0.20, 0.45)
        with self.assertRaisesRegex(ValueError, "at least one DAPI nucleus"):
            analyze_dapi_gfap_only(
                np.zeros_like(nuclei),
                gfap,
                0.20,
                0.45,
            )


if __name__ == "__main__":
    unittest.main()
