from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from project_leap_2d.compartments import selected_soma_enlargement as enlargement
from project_leap_2d.compartments.selected_soma_enlargement import (
    SelectedSomaEnlargementConfig,
    enlarge_selected_soma,
    selected_soma_enlargement_config_for_mode,
)


class SelectedSomaEnlargementTests(unittest.TestCase):
    def make_case(self):
        yy, xx = np.ogrid[:80, :90]
        whole = np.zeros((80, 90), dtype=np.int32)
        soma = np.zeros_like(whole)
        first_whole = (yy - 40) ** 2 + (xx - 32) ** 2 <= 19**2
        first_soma = (yy - 40) ** 2 + (xx - 32) ** 2 <= 7**2
        second_whole = (yy - 40) ** 2 + (xx - 72) ** 2 <= 8**2
        second_soma = (yy - 40) ** 2 + (xx - 72) ** 2 <= 4**2
        whole[first_whole] = 1
        whole[second_whole] = 2
        soma[first_soma] = 1
        soma[second_soma] = 2
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.int32
        )
        owner = (yy - 40) ** 2 + (xx - 32) ** 2 <= 5**2
        foreign = (yy - 40) ** 2 + (xx - 72) ** 2 <= 3**2
        structural = (
            np.exp(-((yy - 40) ** 2 + (xx - 32) ** 2) / 180.0)
            + 0.05
        )
        return whole, soma, processes, owner, foreign, structural

    def test_enlarges_soma_and_recomputes_processes(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertTrue(result.approved)
        self.assertTrue(result.changed)
        self.assertGreater(int(result.added_soma_mask.sum()), 0)
        self.assertTrue(np.all(result.soma_labels[soma == 1] == 1))
        np.testing.assert_array_equal(
            result.whole_labels > 0,
            (result.soma_labels > 0) | (result.process_labels > 0),
        )
        self.assertFalse(
            np.any((result.soma_labels > 0) & (result.process_labels > 0))
        )

    def test_expansion_outside_old_whole_is_added_to_whole(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        whole[40, 35:52] = 0
        soma[40, 35:52] = 0
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.int32
        )
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertTrue(result.approved)
        self.assertGreater(int(result.added_whole_mask.sum()), 0)
        np.testing.assert_array_equal(
            result.added_whole_mask,
            result.added_soma_mask & ~(whole == 1),
        )
        self.assertTrue(
            np.all(result.whole_labels[result.added_whole_mask] == 1)
        )
        self.assertTrue(
            np.all(result.soma_labels[result.added_whole_mask] == 1)
        )

    def test_other_cells_are_pixel_exact(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertTrue(result.approved)
        np.testing.assert_array_equal(
            result.whole_labels == 2,
            whole == 2,
        )
        np.testing.assert_array_equal(
            result.soma_labels == 2,
            soma == 2,
        )
        np.testing.assert_array_equal(
            result.process_labels == 2,
            processes == 2,
        )

    def test_repeated_call_is_idempotent(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        first = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertTrue(first.approved)
        second = enlarge_selected_soma(
            first.whole_labels,
            first.soma_labels,
            first.process_labels,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertFalse(second.changed)
        self.assertEqual(second.status, "no_change")
        self.assertEqual(
            second.message,
            "No additional validated Soma pixels were found.",
        )
        np.testing.assert_array_equal(second.whole_labels, first.whole_labels)
        np.testing.assert_array_equal(second.soma_labels, first.soma_labels)
        np.testing.assert_array_equal(second.process_labels, first.process_labels)

    def test_mode_policy_freezes_egfp_and_limits_gfap_to_a_narrow_outer_rim(
        self,
    ) -> None:
        egfp = selected_soma_enlargement_config_for_mode("egfp")
        gfap = selected_soma_enlargement_config_for_mode("gfap_only")
        self.assertEqual(egfp, SelectedSomaEnlargementConfig())
        self.assertEqual(gfap.nucleus_shell_um, 1.00)
        self.assertEqual(gfap.maximum_supported_radius_um, 1.25)
        self.assertGreater(
            gfap.structural_support_fraction,
            egfp.structural_support_fraction,
        )

    def test_gfap_mode_adds_a_narrow_dapi_shell_and_respects_foreign_nucleus(
        self,
    ) -> None:
        shape = (120, 140)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        pixel_um = 0.10
        owner = (yy - 60) ** 2 + (xx - 60) ** 2 <= 10**2
        distance_to_owner = enlargement.ndi.distance_transform_edt(
            ~owner,
            sampling=(pixel_um, pixel_um),
        )
        old_soma_mask = owner | (distance_to_owner <= 0.80)
        whole_mask = (yy - 60) ** 2 + (xx - 60) ** 2 <= 35**2
        foreign = (yy - 60) ** 2 + (xx - 83) ** 2 <= 2**2
        whole = np.where(whole_mask, 1, 0).astype(np.int32)
        soma = np.where(old_soma_mask, 1, 0).astype(np.int32)
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.int32
        )
        structural = np.zeros(shape, dtype=np.float32)

        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            pixel_um,
            pixel_um,
            config=selected_soma_enlargement_config_for_mode("gfap_only"),
        )

        self.assertTrue(result.approved, result.message)
        self.assertGreater(int(result.added_soma_mask.sum()), 0)
        self.assertFalse(np.any(result.soma_labels[foreign] == 1))
        distance_to_foreign = enlargement.ndi.distance_transform_edt(
            ~foreign,
            sampling=(pixel_um, pixel_um),
        )
        self.assertFalse(
            np.any(
                result.added_soma_mask
                & (distance_to_foreign <= 0.45)
            )
        )
        self.assertFalse(
            np.any(result.added_soma_mask & (distance_to_owner > 1.0 + 1e-9))
        )

    def test_foreign_nucleus_conflict_rejects_without_changes(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        foreign = foreign | np.roll(owner, 3, axis=1)
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertFalse(result.approved)
        self.assertIn("nucleus", result.message.lower())
        np.testing.assert_array_equal(result.whole_labels, whole)
        np.testing.assert_array_equal(result.soma_labels, soma)
        np.testing.assert_array_equal(result.process_labels, processes)

    def test_one_owner_identity_may_have_multiple_projection_islands(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        owner = owner.copy()
        owner[:, 32] = False
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertTrue(result.approved)
        self.assertGreater(
            result.metrics["owner_projection_component_count"],
            1,
        )
        self.assertTrue(np.all(result.soma_labels[owner] == 1))
        self.assertTrue(np.all(result.whole_labels[owner] == 1))

    def test_size_limit_rejects_without_partial_commit(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        config = SelectedSomaEnlargementConfig(
            maximum_added_fraction_of_existing_soma=0.01,
        )
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
            config,
        )
        self.assertEqual(result.status, "rejected_size_limit")
        np.testing.assert_array_equal(result.whole_labels, whole)
        np.testing.assert_array_equal(result.soma_labels, soma)
        np.testing.assert_array_equal(result.process_labels, processes)

    def test_constant_structural_signal_uses_only_bounded_nuclear_shell(self) -> None:
        whole, soma, processes, owner, foreign, _ = self.make_case()
        structural = np.zeros_like(whole, dtype=np.float64)
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertFalse(result.metrics["structural_contrast_detected"])
        distance_px = np.sqrt(
            (np.indices(owner.shape)[0] - 40) ** 2
            + (np.indices(owner.shape)[1] - 32) ** 2
        )
        # Owner radius is 5 px and the unconditional shell is 3 px.
        self.assertFalse(np.any(result.added_soma_mask & (distance_px > 8.1)))

    def test_default_structural_growth_stays_near_the_owner_nucleus(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertTrue(result.approved)
        distance_um = enlargement.ndi.distance_transform_edt(
            ~owner,
            sampling=(0.25, 0.25),
        )
        maximum_radius_um = (
            SelectedSomaEnlargementConfig().maximum_supported_radius_um
        )
        self.assertFalse(
            np.any(
                result.added_soma_mask
                & (distance_um > maximum_radius_um + 1e-9)
            )
        )

    def test_owner_at_image_edge_rejects_without_changes(self) -> None:
        whole, soma, processes, _, foreign, structural = self.make_case()
        owner = np.zeros_like(whole, dtype=bool)
        owner[0:3, 31:34] = True
        soma = soma.copy()
        whole = whole.copy()
        soma[0:3, 31:34] = 1
        whole[0:3, 31:34] = 1
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.int32
        )
        result = enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        self.assertEqual(result.status, "rejected_owner_image_edge")
        np.testing.assert_array_equal(result.whole_labels, whole)
        np.testing.assert_array_equal(result.soma_labels, soma)
        np.testing.assert_array_equal(result.process_labels, processes)

    def test_inputs_are_never_mutated(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        originals = tuple(
            value.copy()
            for value in (whole, soma, processes, owner, foreign, structural)
        )
        enlarge_selected_soma(
            whole,
            soma,
            processes,
            1,
            owner,
            structural,
            foreign,
            0.25,
            0.25,
        )
        for observed, expected in zip(
            (whole, soma, processes, owner, foreign, structural),
            originals,
        ):
            np.testing.assert_array_equal(observed, expected)

    def test_large_image_uses_only_bounded_local_numeric_kernels(self) -> None:
        shape = (1200, 1600)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        whole = np.zeros(shape, dtype=np.int16)
        soma = np.zeros_like(whole)
        whole_mask = (yy - 600) ** 2 + (xx - 800) ** 2 <= 22**2
        soma_mask = (yy - 600) ** 2 + (xx - 800) ** 2 <= 7**2
        owner = (yy - 600) ** 2 + (xx - 800) ** 2 <= 5**2
        whole[whole_mask] = 1
        soma[soma_mask] = 1
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.int16
        )
        structural = np.asarray(
            np.exp(-((yy - 600) ** 2 + (xx - 800) ** 2) / 180.0),
            dtype=np.float32,
        )
        foreign = np.zeros(shape, dtype=bool)
        kernel_shapes: list[tuple[int, int]] = []
        real_edt = enlargement.ndi.distance_transform_edt
        real_gaussian = enlargement.ndi.gaussian_filter

        def tracked_edt(value, *args, **kwargs):
            kernel_shapes.append(tuple(value.shape))
            return real_edt(value, *args, **kwargs)

        def tracked_gaussian(value, *args, **kwargs):
            kernel_shapes.append(tuple(value.shape))
            return real_gaussian(value, *args, **kwargs)

        with patch.object(
            enlargement.ndi,
            "distance_transform_edt",
            side_effect=tracked_edt,
        ), patch.object(
            enlargement.ndi,
            "gaussian_filter",
            side_effect=tracked_gaussian,
        ):
            result = enlarge_selected_soma(
                whole,
                soma,
                processes,
                1,
                owner,
                structural,
                foreign,
                0.25,
                0.25,
            )

        self.assertTrue(result.approved)
        self.assertTrue(kernel_shapes)
        self.assertTrue(
            all(
                local_height < shape[0] and local_width < shape[1]
                for local_height, local_width in kernel_shapes
            )
        )
        self.assertEqual(
            set(kernel_shapes),
            {tuple(result.metrics["local_crop_shape_yx"])},
        )
        self.assertLess(
            result.metrics["local_crop_pixels"],
            result.metrics["full_image_pixels"] // 100,
        )

    def test_crop_memory_limit_rejects_before_numeric_kernels(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        config = SelectedSomaEnlargementConfig(maximum_local_crop_pixels=16)
        with patch.object(
            enlargement.ndi,
            "distance_transform_edt",
        ) as distance_transform:
            result = enlarge_selected_soma(
                whole,
                soma,
                processes,
                1,
                owner,
                structural,
                foreign,
                0.25,
                0.25,
                config,
            )
        self.assertEqual(result.status, "rejected_local_crop_limit")
        self.assertFalse(distance_transform.called)
        self.assertIs(result.whole_labels, whole)
        self.assertIs(result.soma_labels, soma)
        self.assertIs(result.process_labels, processes)

    def test_invalid_partition_is_rejected_before_analysis(self) -> None:
        whole, soma, processes, owner, foreign, structural = self.make_case()
        processes[40, 32] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            enlarge_selected_soma(
                whole,
                soma,
                processes,
                1,
                owner,
                structural,
                foreign,
                0.25,
                0.25,
            )


if __name__ == "__main__":
    unittest.main()
