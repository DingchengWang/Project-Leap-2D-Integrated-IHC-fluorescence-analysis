from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy import ndimage as ndi
from skimage import draw, measure, morphology


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_leap_2d.compartments.selected_cell_split import (  # noqa: E402
    SelectedCellSplitConfig,
    SplitNucleusCandidate,
    _recover_connected_external_branches,
    _unsupported_external_child,
    split_selected_cell,
)


def disk_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    rr, cc = draw.disk(center, radius, shape=shape)
    output[rr, cc] = True
    return output


class SelectedCellSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        shape = (180, 220)
        left = disk_mask(shape, (90, 70), 30)
        right = disk_mask(shape, (90, 132), 28)
        bridge = np.zeros(shape, dtype=bool)
        bridge[82:99, 70:133] = True
        parent = left | right | bridge
        other = disk_mask(shape, (90, 195), 14)

        self.whole = np.zeros(shape, dtype=np.uint16)
        self.whole[parent] = 1
        self.whole[other] = 2
        owner_nucleus = disk_mask(shape, (90, 70), 11)
        second_nucleus = disk_mask(shape, (90, 132), 10)
        other_nucleus = disk_mask(shape, (90, 195), 7)
        self.owner_nucleus = owner_nucleus
        self.second_nucleus = second_nucleus
        self.other_nucleus = other_nucleus

        self.soma = np.zeros(shape, dtype=np.uint16)
        merged_soma = (
            morphology.dilation(owner_nucleus, morphology.disk(17))
            | morphology.dilation(second_nucleus, morphology.disk(16))
        ) & parent
        self.soma[merged_soma] = 1
        self.soma[other_nucleus] = 2
        self.process = np.where(
            (self.whole > 0) & (self.soma == 0),
            self.whole,
            0,
        ).astype(np.uint16)

        self.structural = (self.whole > 0).astype(np.float32)
        # Unassigned structural branch that may be recovered for the second child.
        self.structural[86:95, 160:178] = 1.0
        self.structural[86:95, 132:161] = 1.0
        self.candidates = [
            SplitNucleusCandidate(
                11,
                owner_nucleus,
                owner_astrocyte_id=1,
                accepted=True,
                confidence=0.99,
            ),
            SplitNucleusCandidate(
                12,
                second_nucleus,
                owner_astrocyte_id=None,
                accepted=False,
                identity_status="ambiguous",
                confidence=0.72,
                locally_confirmed=True,
            ),
            SplitNucleusCandidate(
                90,
                other_nucleus,
                owner_astrocyte_id=2,
                accepted=True,
                confidence=0.99,
            ),
        ]
        self.config = SelectedCellSplitConfig(
            maximum_whole_growth_distance_um=4.0,
            minimum_child_area_um2=3.0,
        )

    def run_split(self, candidates=None):
        return split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            self.candidates if candidates is None else candidates,
            self.structural,
            pixel_width_um=0.20,
            pixel_height_um=0.20,
            config=self.config,
        )

    def test_split_adds_exactly_one_cell_and_preserves_triplet(self) -> None:
        result = self.run_split()
        self.assertTrue(result.success, result.reason)
        self.assertEqual(result.new_id, 3)
        self.assertEqual(result.owner_nucleus_id, 11)
        self.assertEqual(result.second_nucleus_id, 12)
        self.assertEqual(sorted(np.unique(result.whole_labels).tolist()), [0, 1, 2, 3])
        self.assertTrue(np.all(result.whole_labels[self.other_nucleus] == 2))
        self.assertTrue(
            np.array_equal(
                result.whole_labels[self.whole == 2],
                self.whole[self.whole == 2],
            )
        )
        self.assertGreater(result.added_whole_px, 0)
        self.assertTrue(np.any((result.whole_labels == 3) & (self.whole == 0)))
        self.assertFalse(np.any((result.soma_labels > 0) & (result.soma_labels != result.whole_labels)))
        self.assertFalse(
            np.any(
                (result.process_labels > 0)
                & (result.process_labels != result.whole_labels)
            )
        )
        occupancy = (result.soma_labels > 0).astype(np.uint8)
        occupancy += (result.process_labels > 0).astype(np.uint8)
        self.assertTrue(np.all(occupancy[result.whole_labels > 0] == 1))
        self.assertTrue(np.all(occupancy[result.whole_labels == 0] == 0))
        self.assertEqual(
            set(np.unique(result.whole_labels)) - {0},
            set(np.unique(result.soma_labels)) - {0},
        )
        self.assertEqual(
            set(np.unique(result.whole_labels)) - {0},
            set(np.unique(result.process_labels)) - {0},
        )
        self.assertEqual(measure.label(result.soma_labels == 1).max(), 1)
        self.assertEqual(measure.label(result.soma_labels == 3).max(), 1)
        self.assertTrue(np.all(result.soma_labels[self.owner_nucleus] == 1))
        self.assertTrue(np.all(result.soma_labels[self.second_nucleus] == 3))

    def test_locally_confirmed_ambiguous_second_nucleus_is_allowed(self) -> None:
        result = self.run_split()
        self.assertTrue(result.success)
        self.assertEqual(result.metrics["second_identity_status"], "ambiguous")

    def test_unaccepted_canonical_second_nucleus_requires_local_confirmation(
        self,
    ) -> None:
        provisional = [
            self.candidates[0],
            replace(self.candidates[1], locally_confirmed=False),
        ]
        result = self.run_split(provisional)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "No additional DAPI nucleus was found.")
        self.assertTrue(np.array_equal(result.whole_labels, self.whole))

    def test_marginal_exterior_canonical_nucleus_cannot_short_circuit_recovery(
        self,
    ) -> None:
        exterior = disk_mask(self.whole.shape, (90, 168), 9)
        # Retain only a thin intersection with the selected parent, similar to
        # a neighboring canonical object touching the ROI margin.
        overlap_fraction = float((exterior & (self.whole == 1)).sum()) / int(
            exterior.sum()
        )
        self.assertLess(overlap_fraction, 0.10)
        result = self.run_split(
            [
                self.candidates[0],
                SplitNucleusCandidate(
                    72,
                    exterior,
                    accepted=True,
                    confidence=0.99,
                ),
            ]
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "No additional DAPI nucleus was found.")
        self.assertTrue(np.array_equal(result.whole_labels, self.whole))

    def test_tiny_canonical_second_nucleus_is_provisional_even_if_accepted(
        self,
    ) -> None:
        tiny = disk_mask(self.whole.shape, (90, 132), 3)
        projected_area_um2 = float(tiny.sum()) * 0.20 * 0.20
        self.assertLess(
            projected_area_um2,
            self.config.minimum_direct_nucleus_projection_area_um2,
        )
        result = self.run_split(
            [
                self.candidates[0],
                SplitNucleusCandidate(
                    66,
                    tiny,
                    accepted=True,
                    confidence=0.99,
                ),
            ]
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "No additional DAPI nucleus was found.")

    def test_strongest_single_extra_nucleus_is_selected(self) -> None:
        distant = disk_mask(self.whole.shape, (50, 110), 8)
        structural = self.structural.copy()
        structural[distant] = 1.0
        candidates = self.candidates + [
            SplitNucleusCandidate(
                13,
                distant,
                identity_status="model_proposal",
                confidence=0.51,
            )
        ]
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            candidates,
            structural,
            0.20,
            0.20,
            self.config,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.second_nucleus_id, 12)
        self.assertEqual(result.new_id, 3)

    def test_foreign_owned_second_nucleus_is_rejected_without_changes(self) -> None:
        foreign_candidates = [
            self.candidates[0],
            self.candidates[2],
        ]
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            foreign_candidates,
            self.structural,
            0.20,
            0.20,
            replace(self.config, maximum_candidate_distance_um=6.0),
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.reason,
            "The second nucleus already belongs to another astrocyte.",
        )
        self.assertTrue(np.array_equal(result.whole_labels, self.whole))
        self.assertTrue(np.array_equal(result.soma_labels, self.soma))
        self.assertTrue(np.array_equal(result.process_labels, self.process))

    def test_distant_foreign_nucleus_does_not_change_refusal_reason(self) -> None:
        result = self.run_split([self.candidates[0], self.candidates[2]])
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "No additional DAPI nucleus was found.")

    def test_candidate_inside_concave_bbox_but_far_from_mask_is_rejected(self) -> None:
        shape = (170, 170)
        parent = np.zeros(shape, dtype=bool)
        parent[20:145, 20:32] = True
        parent[20:145, 138:150] = True
        parent[133:145, 20:150] = True
        whole = np.where(parent, 1, 0).astype(np.uint16)
        owner = disk_mask(shape, (70, 26), 5)
        distant_inside_bbox = disk_mask(shape, (55, 85), 6)
        soma = np.where(owner, 1, 0).astype(np.uint16)
        process = np.where(parent & ~owner, 1, 0).astype(np.uint16)
        result = split_selected_cell(
            whole,
            soma,
            process,
            1,
            [
                SplitNucleusCandidate(41, owner, owner_astrocyte_id=1),
                SplitNucleusCandidate(42, distant_inside_bbox),
            ],
            parent.astype(np.float32),
            0.20,
            0.20,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "No additional DAPI nucleus was found.")

    def test_missing_second_nucleus_is_rejected_without_changes(self) -> None:
        result = self.run_split([self.candidates[0]])
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "No additional DAPI nucleus was found.")
        self.assertTrue(np.array_equal(result.whole_labels, self.whole))

    def test_split_is_deterministic(self) -> None:
        first = self.run_split()
        second = self.run_split()
        self.assertTrue(first.success and second.success)
        self.assertTrue(np.array_equal(first.whole_labels, second.whole_labels))
        self.assertTrue(np.array_equal(first.soma_labels, second.soma_labels))
        self.assertTrue(np.array_equal(first.process_labels, second.process_labels))
        self.assertEqual(first.metrics, second.metrics)

    def test_overlapping_second_nucleus_is_rejected(self) -> None:
        overlapping = np.roll(self.owner_nucleus, 2, axis=1)
        candidates = [
            self.candidates[0],
            SplitNucleusCandidate(
                14,
                overlapping,
                accepted=True,
                confidence=0.9,
            ),
        ]
        result = self.run_split(candidates)
        self.assertFalse(result.success)
        self.assertEqual(
            result.reason,
            "The two nuclear candidates cannot be separated.",
        )

    def test_limited_projection_overlap_is_resolved_exclusively(self) -> None:
        owner = disk_mask(self.whole.shape, (90, 88), 12)
        second = disk_mask(self.whole.shape, (90, 104), 12)
        overlap = owner & second
        self.assertTrue(overlap.any())
        candidates = [
            SplitNucleusCandidate(
                21,
                owner,
                owner_astrocyte_id=1,
                accepted=True,
                confidence=0.99,
            ),
            SplitNucleusCandidate(
                22,
                second,
                identity_status="resolved",
                confidence=0.90,
                locally_confirmed=True,
            ),
        ]
        result = self.run_split(candidates)
        self.assertTrue(result.success, result.reason)
        self.assertEqual(result.owner_nucleus_id, 21)
        self.assertEqual(result.second_nucleus_id, 22)
        self.assertTrue(np.all(result.whole_labels[owner & ~second] == 1))
        self.assertTrue(np.all(result.whole_labels[second & ~owner] == 3))
        overlap_ids = set(int(value) for value in np.unique(result.whole_labels[overlap]))
        self.assertTrue(overlap_ids.issubset({1, 3}))
        self.assertNotIn(0, overlap_ids)
        self.assertFalse(
            np.any(
                (result.soma_labels > 0)
                & (result.soma_labels != result.whole_labels)
            )
        )

    def test_default_growth_recovers_a_connected_unassigned_process(self) -> None:
        structural = self.structural.copy()
        branch = np.zeros(self.whole.shape, dtype=bool)
        branch[48:65, 67:74] = True
        structural[branch] = 1.0
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            self.candidates,
            structural,
            0.20,
            0.20,
            SelectedCellSplitConfig(minimum_child_area_um2=3.0),
        )
        self.assertTrue(result.success, result.reason)
        self.assertTrue(np.all(result.whole_labels[branch] > 0))
        self.assertFalse(np.any(result.whole_labels[branch] == 2))

    def test_broad_external_halo_is_not_recovered(self) -> None:
        parent = self.whole == 1
        halo = morphology.dilation(parent, morphology.disk(10)) & ~parent
        structural = parent.astype(np.float32)
        structural[halo] = 1.0
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            self.candidates,
            structural,
            0.20,
            0.20,
            SelectedCellSplitConfig(minimum_child_area_um2=3.0),
        )
        self.assertTrue(result.success, result.reason)
        self.assertFalse(np.any(result.whole_labels[halo] > 0))
        self.assertEqual(result.added_whole_px, 0)

    def test_local_branch_recovery_matches_full_canvas_reference(self) -> None:
        shape = (90, 130)
        parent = np.zeros(shape, dtype=bool)
        parent[42:48, 8:122] = True
        structural = np.zeros(shape, dtype=bool)
        structural[28:43, 18:22] = True
        structural[34:43, 54:60] = True
        structural[22:43, 88:104] = True  # Too wide; must remain rejected.
        structural[5:10, 5:10] = True  # No parent attachment.
        other_whole = np.zeros(shape, dtype=bool)
        other_whole[30:43, 56:58] = True
        distance = np.zeros(shape, dtype=np.float32)
        config = SelectedCellSplitConfig(
            maximum_whole_growth_distance_um=4.0,
            maximum_branch_attachment_um=2.5,
            maximum_recovered_branch_half_width_um=1.25,
        )

        eligible = (
            structural
            & ~parent
            & ~other_whole
            & (distance <= config.maximum_whole_growth_distance_um)
        )
        parent_neighborhood = morphology.dilation(
            parent,
            footprint=np.ones((3, 3), dtype=bool),
        )
        labels = measure.label(eligible, connectivity=2)
        expected = np.zeros(shape, dtype=bool)
        for label_id in range(1, int(labels.max()) + 1):
            component = labels == label_id
            attachment = component & parent_neighborhood
            if not attachment.any():
                continue
            coordinates = np.argwhere(attachment)
            height_um = (
                int(coordinates[:, 0].max())
                - int(coordinates[:, 0].min())
                + 1
            ) * 0.20
            width_um = (
                int(coordinates[:, 1].max())
                - int(coordinates[:, 1].min())
                + 1
            ) * 0.25
            if np.hypot(height_um, width_um) > config.maximum_branch_attachment_um:
                continue
            half_width_um = float(
                ndi.distance_transform_edt(
                    component,
                    sampling=(0.20, 0.25),
                ).max(initial=0.0)
            )
            if half_width_um <= config.maximum_recovered_branch_half_width_um:
                expected |= component

        actual = _recover_connected_external_branches(
            parent,
            structural,
            other_whole,
            distance,
            pixel_width_um=0.25,
            pixel_height_um=0.20,
            config=config,
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_many_noise_components_use_bounded_distance_crops(self) -> None:
        shape = (80, 2400)
        parent = np.zeros(shape, dtype=bool)
        parent[45:49, 5:2395] = True
        structural = np.zeros(shape, dtype=bool)
        for x in range(12, 2380, 24):
            structural[32:46, x : x + 3] = True
        distance = np.zeros(shape, dtype=np.float32)
        original_distance_transform = ndi.distance_transform_edt
        transform_shapes: list[tuple[int, ...]] = []

        def recording_distance_transform(array, *args, **kwargs):
            transform_shapes.append(tuple(array.shape))
            return original_distance_transform(array, *args, **kwargs)

        with mock.patch(
            "project_leap_2d.compartments.selected_cell_split."
            "ndi.distance_transform_edt",
            side_effect=recording_distance_transform,
        ):
            first = _recover_connected_external_branches(
                parent,
                structural,
                np.zeros(shape, dtype=bool),
                distance,
                pixel_width_um=0.20,
                pixel_height_um=0.20,
                config=SelectedCellSplitConfig(),
            )
        second = _recover_connected_external_branches(
            parent,
            structural,
            np.zeros(shape, dtype=bool),
            distance,
            pixel_width_um=0.20,
            pixel_height_um=0.20,
            config=SelectedCellSplitConfig(),
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(len(transform_shapes), 90)
        self.assertTrue(
            all(np.prod(transform_shape) <= 80 for transform_shape in transform_shapes)
        )
        self.assertLess(
            max(np.prod(transform_shape) for transform_shape in transform_shapes),
            np.prod(shape) // 1000,
        )

    def test_external_child_safeguard_requires_combined_failure_pattern(
        self,
    ) -> None:
        shape = (100, 140)
        child = disk_mask(shape, (50, 70), 28)
        original_parent = np.zeros(shape, dtype=bool)
        original_parent[:, :70] = child[:, :70]
        sparse_process = np.zeros(shape, dtype=bool)
        sparse_process[50, 70:85] = True
        config = SelectedCellSplitConfig()

        rejected, metrics = _unsupported_external_child(
            child,
            sparse_process,
            original_parent,
            pixel_area_um2=0.04,
            config=config,
        )
        self.assertTrue(rejected)
        self.assertGreaterEqual(
            metrics["external_area_um2"],
            config.minimum_large_external_growth_um2,
        )
        self.assertLessEqual(
            metrics["process_fraction"],
            config.maximum_process_fraction_for_external_growth,
        )

        process_rich = child & ~disk_mask(shape, (50, 70), 20)
        process_rich_rejected, _ = _unsupported_external_child(
            child,
            process_rich,
            original_parent,
            pixel_area_um2=0.04,
            config=config,
        )
        self.assertFalse(process_rich_rejected)

        compact_parent = child.copy()
        compact_rejected, _ = _unsupported_external_child(
            child,
            np.zeros(shape, dtype=bool),
            compact_parent,
            pixel_area_um2=0.04,
            config=config,
        )
        self.assertFalse(compact_rejected)

    def test_split_rejects_children_without_real_processes(self) -> None:
        shape = (100, 140)
        owner = disk_mask(shape, (50, 45), 8)
        second = disk_mask(shape, (50, 92), 8)
        parent = owner | second
        bridge = np.zeros(shape, dtype=bool)
        bridge[48:53, 45:93] = True
        parent |= bridge
        whole = np.where(parent, 1, 0).astype(np.uint16)
        soma = whole.copy()
        processes = np.zeros(shape, dtype=np.uint16)
        result = split_selected_cell(
            whole,
            soma,
            processes,
            1,
            [
                SplitNucleusCandidate(
                    71,
                    owner,
                    owner_astrocyte_id=1,
                    accepted=True,
                ),
                SplitNucleusCandidate(
                    72,
                    second,
                    owner_astrocyte_id=1,
                    accepted=True,
                ),
            ],
            np.zeros(shape, dtype=np.float32),
            0.20,
            0.20,
            SelectedCellSplitConfig(
                minimum_child_area_um2=0.5,
                minimum_child_fraction=0.01,
                maximum_added_whole_fraction=3.0,
            ),
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.reason,
            "Split could not establish real Processes for both child cells.",
        )
        self.assertTrue(np.array_equal(result.whole_labels, whole))
        self.assertTrue(np.array_equal(result.soma_labels, soma))
        self.assertTrue(np.array_equal(result.process_labels, processes))

    def test_excessive_external_growth_is_rejected_without_changes(self) -> None:
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            self.candidates,
            self.structural,
            0.20,
            0.20,
            replace(
                self.config,
                maximum_added_whole_fraction=0.001,
            ),
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.reason,
            "Split recovery exceeded the safe Whole Cell expansion limit.",
        )
        self.assertTrue(np.array_equal(result.whole_labels, self.whole))
        self.assertTrue(np.array_equal(result.soma_labels, self.soma))
        self.assertTrue(np.array_equal(result.process_labels, self.process))
        self.assertGreater(
            result.metrics["added_whole_px"],
            result.metrics["maximum_added_whole_px"],
        )

    def test_stable_owner_remapping_supports_repeated_split(self) -> None:
        shape = (160, 280)
        left = disk_mask(shape, (80, 55), 30)
        middle = disk_mask(shape, (80, 135), 30)
        right = disk_mask(shape, (80, 215), 30)
        bridge = np.zeros(shape, dtype=bool)
        bridge[73:88, 55:216] = True
        parent = left | middle | right | bridge
        whole = np.where(parent, 1, 0).astype(np.uint16)
        nuclei = [
            disk_mask(shape, (80, 55), 10),
            disk_mask(shape, (80, 135), 10),
            disk_mask(shape, (80, 215), 10),
        ]
        soma = np.zeros(shape, dtype=np.uint16)
        for nucleus in nuclei:
            soma[morphology.dilation(nucleus, morphology.disk(15)) & parent] = 1
        process = np.where((whole > 0) & (soma == 0), whole, 0).astype(np.uint16)
        candidates = [
            SplitNucleusCandidate(
                31,
                nuclei[0],
                owner_astrocyte_id=1,
                accepted=True,
                confidence=1.0,
            ),
            SplitNucleusCandidate(
                32,
                nuclei[1],
                owner_astrocyte_id=1,
                accepted=True,
                confidence=0.95,
            ),
            SplitNucleusCandidate(
                33,
                nuclei[2],
                owner_astrocyte_id=1,
                accepted=False,
                confidence=0.70,
                locally_confirmed=True,
            ),
        ]
        config = SelectedCellSplitConfig(
            maximum_candidate_distance_um=30.0,
            minimum_child_area_um2=3.0,
        )
        first = split_selected_cell(
            whole,
            soma,
            process,
            1,
            candidates,
            parent.astype(np.float32),
            0.20,
            0.20,
            config,
        )
        self.assertTrue(first.success, first.reason)
        child_with_third = int(first.whole_labels[nuclei[2]][0])
        self.assertIn(child_with_third, (1, 2))
        remapped_candidates = []
        for candidate in candidates:
            current_owner = int(
                np.bincount(
                    first.whole_labels[candidate.projection_mask].ravel(),
                    minlength=3,
                )[1:].argmax()
                + 1
            )
            remapped_candidates.append(
                replace(candidate, owner_astrocyte_id=current_owner)
            )
        second = split_selected_cell(
            first.whole_labels,
            first.soma_labels,
            first.process_labels,
            child_with_third,
            remapped_candidates,
            parent.astype(np.float32),
            0.20,
            0.20,
            config,
        )
        self.assertTrue(second.success, second.reason)
        self.assertEqual(int(second.whole_labels.max()), 3)
        self.assertEqual(
            sorted(np.unique(second.whole_labels).tolist()),
            [0, 1, 2, 3],
        )

    def test_high_xy_overlap_with_separate_z_ranges_succeeds(self) -> None:
        owner = disk_mask(self.whole.shape, (90, 90), 13)
        second = disk_mask(self.whole.shape, (90, 94), 13)
        overlap_fraction = float((owner & second).sum()) / min(
            int(owner.sum()),
            int(second.sum()),
        )
        self.assertGreater(overlap_fraction, 0.50)
        candidates = [
            SplitNucleusCandidate(
                51,
                owner,
                owner_astrocyte_id=1,
                accepted=True,
                confidence=0.99,
                z_min_0based=2,
                z_max_0based=5,
            ),
            SplitNucleusCandidate(
                52,
                second,
                owner_astrocyte_id=1,
                accepted=True,
                confidence=0.95,
                z_min_0based=10,
                z_max_0based=13,
            ),
        ]
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            candidates,
            self.structural,
            0.20,
            0.20,
            replace(
                self.config,
                minimum_child_area_um2=0.5,
                minimum_child_fraction=0.01,
            ),
            pixel_depth_um=0.50,
        )
        self.assertTrue(result.success, result.reason)
        self.assertTrue(result.metrics["second_candidate"]["z_separated"])
        self.assertGreater(
            result.metrics["second_candidate"]["z_center_separation_um"],
            0.70,
        )
        self.assertTrue(np.all(result.whole_labels[owner & ~second] == 1))
        self.assertTrue(np.all(result.whole_labels[second & ~owner] == 3))
        self.assertFalse(
            np.any(
                (result.soma_labels > 0)
                & (result.soma_labels != result.whole_labels)
            )
        )

    def test_high_xy_overlap_without_z_separation_is_rejected(self) -> None:
        owner = disk_mask(self.whole.shape, (90, 90), 13)
        second = disk_mask(self.whole.shape, (90, 94), 13)
        candidates = [
            SplitNucleusCandidate(
                61,
                owner,
                owner_astrocyte_id=1,
                accepted=True,
                z_min_0based=4,
                z_max_0based=8,
            ),
            SplitNucleusCandidate(
                62,
                second,
                owner_astrocyte_id=1,
                accepted=True,
                z_min_0based=5,
                z_max_0based=9,
            ),
        ]
        result = split_selected_cell(
            self.whole,
            self.soma,
            self.process,
            1,
            candidates,
            self.structural,
            0.20,
            0.20,
            self.config,
            pixel_depth_um=0.50,
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.reason,
            "The two nuclear candidates cannot be separated.",
        )

    def test_malformed_triplet_raises(self) -> None:
        bad_process = self.process.copy()
        bad_process[self.soma == 1] = 1
        with self.assertRaisesRegex(ValueError, "exact partition"):
            split_selected_cell(
                self.whole,
                self.soma,
                bad_process,
                1,
                self.candidates,
                self.structural,
                0.20,
                0.20,
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
