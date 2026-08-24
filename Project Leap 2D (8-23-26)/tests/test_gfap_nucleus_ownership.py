from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage as ndi

from project_leap_2d.analysis_modes.gfap_only.gfap_nucleus_ownership import (
    GFAPAssociationResult,
    GFAPNucleusOwnershipConfig,
    _apply_unique_weak_enrichment_fallback,
    _local_structure_rejection_reasons,
    link_slice_instances_3d,
    project_exclusive_nucleus_owners,
    select_gfap_associated_owners,
    validate_nucleus_projection,
)


def disk(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def permissive_link_config() -> GFAPNucleusOwnershipConfig:
    return GFAPNucleusOwnershipConfig(
        min_nucleus_volume_um3=0.001,
        min_nucleus_z_span_um=0.01,
        reject_xy_border_objects=False,
    )


class GFAPNucleusLinkingTests(unittest.TestCase):
    def test_same_slice_label_at_distant_positions_becomes_two_global_ids(self) -> None:
        labels = np.zeros((2, 64, 96), dtype=np.int32)
        labels[0, disk(labels.shape[1:], (25, 20), 5)] = 1
        labels[1, disk(labels.shape[1:], (25, 75), 5)] = 1
        result = link_slice_instances_3d(
            labels,
            0.2,
            0.4,
            config=permissive_link_config(),
        )
        self.assertEqual(
            {int(value) for value in np.unique(result.labels_zyx) if value > 0},
            {1, 2},
        )
        self.assertEqual(result.valid_ids, (1, 2))

    def test_different_slice_labels_with_overlap_become_one_global_id(self) -> None:
        labels = np.zeros((3, 64, 96), dtype=np.int32)
        labels[0, disk(labels.shape[1:], (25, 30), 5)] = 8
        labels[1, disk(labels.shape[1:], (25, 31), 5)] = 2
        labels[2, disk(labels.shape[1:], (26, 31), 5)] = 19
        result = link_slice_instances_3d(
            labels,
            0.2,
            0.4,
            config=permissive_link_config(),
        )
        self.assertEqual(
            {int(value) for value in np.unique(result.labels_zyx) if value > 0},
            {1},
        )
        self.assertEqual(result.valid_ids, (1,))

    def test_adjacent_z_split_conflict_is_ambiguous_and_not_valid(self) -> None:
        labels = np.zeros((2, 64, 96), dtype=np.int32)
        labels[0, 25:36, 30:47] = 1
        labels[1, 25:36, 30:38] = 4
        labels[1, 25:36, 39:47] = 5
        result = link_slice_instances_3d(
            labels,
            0.2,
            0.4,
            config=permissive_link_config(),
        )
        self.assertEqual(set(result.ambiguous_ids), {1, 2, 3})
        self.assertTrue(set(result.ambiguous_ids).isdisjoint(result.valid_ids))
        self.assertTrue(
            any(
                "ambiguous_adjacent_z_link" in row["rejection_reasons"]
                for row in result.records
            )
        )

    def test_adjacent_z_merge_conflict_marks_every_involved_track_ambiguous(self) -> None:
        labels = np.zeros((2, 64, 96), dtype=np.int32)
        labels[0, 25:36, 30:38] = 4
        labels[0, 25:36, 39:47] = 5
        labels[1, 25:36, 30:47] = 1
        result = link_slice_instances_3d(
            labels,
            0.2,
            0.4,
            config=permissive_link_config(),
        )
        self.assertEqual(set(result.ambiguous_ids), {1, 2, 3})
        self.assertTrue(set(result.ambiguous_ids).isdisjoint(result.valid_ids))


class GFAPNucleusAssociationTests(unittest.TestCase):
    def test_unique_mild_enrichment_failure_has_a_limited_fallback(self) -> None:
        association = GFAPAssociationResult(
            accepted_ids=(),
            rejected_ids=(7,),
            records=(
                {
                    "nucleus_id": 7,
                    "shell_enrichment": 4.0,
                    "gfap_associated": False,
                    "rejection_reasons": [
                        "weak_perinuclear_gfap_enrichment"
                    ],
                },
            ),
        )
        result = _apply_unique_weak_enrichment_fallback(
            association,
            GFAPNucleusOwnershipConfig(),
        )
        self.assertEqual(result.accepted_ids, (7,))
        self.assertEqual(result.rejected_ids, ())
        self.assertTrue(result.records[0]["accepted_by_limited_fallback"])

    def test_fallback_rejects_multiple_eligible_or_multi_gate_failures(self) -> None:
        multiple = GFAPAssociationResult(
            accepted_ids=(),
            rejected_ids=(7, 8),
            records=tuple(
                {
                    "nucleus_id": nucleus_id,
                    "shell_enrichment": 4.0,
                    "gfap_associated": False,
                    "rejection_reasons": [
                        "weak_perinuclear_gfap_enrichment"
                    ],
                }
                for nucleus_id in (7, 8)
            ),
        )
        self.assertEqual(
            _apply_unique_weak_enrichment_fallback(
                multiple,
                GFAPNucleusOwnershipConfig(),
            ).accepted_ids,
            (),
        )
        multi_gate = GFAPAssociationResult(
            accepted_ids=(),
            rejected_ids=(7,),
            records=(
                {
                    "nucleus_id": 7,
                    "shell_enrichment": 4.0,
                    "gfap_associated": False,
                    "rejection_reasons": [
                        "weak_perinuclear_gfap_enrichment",
                        "gfap_support_lacks_z_continuity",
                    ],
                },
            ),
        )
        self.assertEqual(
            _apply_unique_weak_enrichment_fallback(
                multi_gate,
                GFAPNucleusOwnershipConfig(),
            ).accepted_ids,
            (),
        )

    def test_local_area_boundary_is_inclusive_and_source_222_passes(self) -> None:
        config = GFAPNucleusOwnershipConfig()
        self.assertEqual(config.min_anchored_structure_area_um2, 0.75)
        common = {
            "anchored_reach_um": 1.6,
            "contact_sector_count": 3,
            "contiguous_contact_sector_count": 2,
            "angular_fraction": 0.25,
            "config": config,
        }
        self.assertNotIn(
            "insufficient_connected_gfap_structure",
            _local_structure_rejection_reasons(
                anchored_area_um2=0.75,
                **common,
            ),
        )
        self.assertIn(
            "insufficient_connected_gfap_structure",
            _local_structure_rejection_reasons(
                anchored_area_um2=0.74,
                **common,
            ),
        )
        self.assertEqual(
            _local_structure_rejection_reasons(
                anchored_area_um2=0.812802,
                anchored_reach_um=1.71296,
                contact_sector_count=4,
                contiguous_contact_sector_count=2,
                angular_fraction=4 / 12,
                config=config,
            ),
            (),
        )

    def make_owner_evidence(
        self,
        *,
        shape: tuple[int, int, int] = (5, 160, 160),
        center: tuple[int, int] = (80, 80),
        radius: int = 5,
        full_shell_gfap: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        labels = np.zeros(shape, dtype=np.int32)
        nucleus = disk(shape[1:], center, radius)
        for z_index in (1, 2, 3):
            labels[z_index, nucleus] = 1
        gfap = np.ones(shape, dtype=np.float32)
        distance = ndi.distance_transform_edt(~nucleus, sampling=(0.2, 0.2))
        shell = (distance > 0) & (distance <= 1.2)
        yy, xx = np.indices(shape[1:])
        angle = np.arctan2(yy - center[0], xx - center[1])
        raw_support = shell if full_shell_gfap else shell & (np.abs(angle) <= np.pi / 2)
        for z_index in (1, 2, 3):
            gfap[z_index, raw_support] = 30.0
        return (
            labels,
            gfap,
            np.zeros(shape[1:], dtype=bool),
            np.zeros(shape[1:], dtype=np.float32),
            distance,
        )

    def select_single(
        self,
        labels: np.ndarray,
        gfap: np.ndarray,
        structural: np.ndarray,
        score: np.ndarray,
    ):
        return select_gfap_associated_owners(
            labels,
            gfap,
            structural,
            score,
            0.2,
            0.4,
            candidate_ids=(1,),
        )

    def make_two_nuclei(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (5, 84, 112)
        labels = np.zeros(shape, dtype=np.int32)
        first = disk(shape[1:], (42, 32), 5)
        second = disk(shape[1:], (42, 82), 5)
        for z_index in (1, 2, 3):
            labels[z_index, first] = 1
            labels[z_index, second] = 2
        gfap = np.ones(shape, dtype=np.float32)
        distance = ndi.distance_transform_edt(
            ~first,
            sampling=(0.2, 0.2),
        )
        supported_shell = (distance > 0) & (distance <= 1.2)
        # Partial support on three Z planes: sufficient perisomatic evidence,
        # but deliberately not a closed GFAP envelope.
        partial = supported_shell & (np.indices(first.shape)[1] <= 36)
        for z_index in (1, 2, 3):
            gfap[z_index, partial] = 20.0
        structural = np.zeros(shape[1:], dtype=bool)
        structural[34:51, 26:31] = True
        structural[39:45, 18:48] = True
        score = structural.astype(np.float32)
        return labels, gfap, structural, score

    def test_only_gfap_associated_nucleus_becomes_owner(self) -> None:
        labels, gfap, structural, score = self.make_two_nuclei()
        result = select_gfap_associated_owners(
            labels,
            gfap,
            structural,
            score,
            0.2,
            0.4,
            candidate_ids=(1, 2),
        )
        self.assertEqual(result.accepted_ids, (1,))
        self.assertEqual(result.rejected_ids, (2,))
        first = next(row for row in result.records if row["nucleus_id"] == 1)
        self.assertGreaterEqual(first["supported_z_count"], 2)
        self.assertGreaterEqual(first["contact_angular_fraction"], 0.20)

    def test_single_plane_line_crossing_is_not_ownership(self) -> None:
        shape = (5, 84, 112)
        labels = np.zeros(shape, dtype=np.int32)
        nucleus = disk(shape[1:], (42, 52), 5)
        for z_index in (1, 2, 3):
            labels[z_index, nucleus] = 1
        gfap = np.ones(shape, dtype=np.float32)
        gfap[2, 42:44, 10:100] = 30.0
        structural = np.zeros(shape[1:], dtype=bool)
        structural[42:44, 10:100] = True
        result = select_gfap_associated_owners(
            labels,
            gfap,
            structural,
            structural.astype(np.float32),
            0.2,
            0.4,
            candidate_ids=(1,),
        )
        self.assertEqual(result.accepted_ids, ())
        reasons = result.records[0]["rejection_reasons"]
        self.assertIn("gfap_support_not_repeated_across_z", reasons)

    def test_global_network_cannot_contribute_area_outside_local_evidence(self) -> None:
        labels, gfap, structural, score, _distance = self.make_owner_evidence()
        # A narrow local bridge touches the nucleus and joins a very large
        # remote network.  Only the <=8 um local bridge may count.
        structural[79:81, 85:160] = True
        structural[110:160, 110:160] = True
        structural[80:111, 145:147] = True
        score[structural] = 0.9
        result = self.select_single(labels, gfap, structural, score)
        self.assertEqual(result.accepted_ids, ())
        row = result.records[0]
        self.assertLessEqual(row["anchored_structure_reach_um"], 8.0)
        self.assertLess(row["strong_contact_sector_count"], 3)
        self.assertIn("single_direction_gfap_contact", row["rejection_reasons"])

    def test_repeated_z_crossing_line_is_still_not_an_owner(self) -> None:
        labels, gfap, structural, score, _distance = self.make_owner_evidence()
        structural[79:82, 20:140] = True
        score[structural] = 0.9
        # Raw line is bright on every nuclear plane, so Z repetition alone
        # cannot make it an astrocyte owner.
        gfap[:, 79:82, 20:140] = 30.0
        result = self.select_single(labels, gfap, structural, score)
        self.assertEqual(result.accepted_ids, ())
        row = result.records[0]
        self.assertGreaterEqual(row["supported_z_count"], 2)
        self.assertLess(row["contiguous_strong_contact_sector_count"], 3)

    def test_ring_speckles_do_not_replace_reaching_structure(self) -> None:
        labels, gfap, structural, score, distance = self.make_owner_evidence()
        yy, xx = np.indices(structural.shape)
        center = np.asarray([80, 80])
        for angle in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            y = int(round(center[0] + np.sin(angle) * 8))
            x = int(round(center[1] + np.cos(angle) * 8))
            structural[y - 1 : y + 2, x - 1 : x + 2] = True
        structural &= (distance > 0) & (distance <= 1.5)
        score[structural] = 0.9
        result = self.select_single(labels, gfap, structural, score)
        self.assertEqual(result.accepted_ids, ())
        self.assertIn(
            "connected_gfap_structure_too_short",
            result.records[0]["rejection_reasons"],
        )

    def test_unilateral_fan_is_valid_without_nuclear_enclosure(self) -> None:
        labels, gfap, structural, score, distance = self.make_owner_evidence(
            full_shell_gfap=False
        )
        yy, xx = np.indices(structural.shape)
        angle = np.arctan2(yy - 80, xx - 80)
        distance_3d = ndi.distance_transform_edt(
            ~(labels == 1),
            sampling=(0.4, 0.2, 0.2),
        )
        unilateral_shell_3d = (
            (distance_3d > 0)
            & (distance_3d <= 1.5)
            & (np.abs(angle)[None, :, :] <= np.pi / 2)
        )
        gfap[unilateral_shell_3d] = 30.0
        fan = (
            (distance > 0)
            & (distance <= 6.0)
            & (np.abs(angle) <= np.pi / 3)
        )
        structural[fan] = True
        score[fan] = 0.9
        result = self.select_single(labels, gfap, structural, score)
        self.assertEqual(result.accepted_ids, (1,))
        row = result.records[0]
        self.assertGreaterEqual(row["strong_contact_sector_count"], 3)
        self.assertGreaterEqual(row["anchored_structure_reach_um"], 1.6)

    def test_source_222_weak_boundary_is_retained(self) -> None:
        labels, gfap, structural, score, distance = self.make_owner_evidence()
        labels[labels == 1] = 222
        yy, xx = np.indices(structural.shape)
        angle = np.mod(np.arctan2(yy - 80, xx - 80), 2.0 * np.pi)
        sector = np.floor(
            np.mod(angle + np.pi / 12, 2.0 * np.pi)
            * 12
            / (2.0 * np.pi)
        ).astype(int)
        boundary_branches = (
            (distance > 0)
            & (distance <= 1.6000001)
            & np.isin(sector, (0, 1, 3))
        )
        structural[boundary_branches] = True
        score[boundary_branches] = 0.9
        result = select_gfap_associated_owners(
            labels,
            gfap,
            structural,
            score,
            0.2,
            0.4,
            candidate_ids=(222,),
        )
        self.assertEqual(result.accepted_ids, (222,))
        row = result.records[0]
        self.assertGreaterEqual(row["anchored_structure_area_um2"], 1.0)
        self.assertAlmostEqual(row["anchored_structure_reach_um"], 1.6, places=6)
        self.assertEqual(row["strong_contact_sector_count"], 3)
        self.assertEqual(row["contiguous_strong_contact_sector_count"], 2)

    def test_locally_relative_but_absolutely_weak_structure_is_rejected(self) -> None:
        labels, gfap, structural, score, distance = self.make_owner_evidence()
        yy, xx = np.indices(structural.shape)
        angle = np.arctan2(yy - 80, xx - 80)
        fan = (
            (distance > 0)
            & (distance <= 6.0)
            & (np.abs(angle) <= np.pi / 3)
        )
        structural[fan] = True
        score[fan] = 0.29
        result = self.select_single(labels, gfap, structural, score)
        self.assertEqual(result.accepted_ids, ())
        self.assertEqual(result.records[0]["local_strong_component_count"], 0)
        self.assertIn(
            "insufficient_connected_gfap_structure",
            result.records[0]["rejection_reasons"],
        )

    def test_crop_edge_does_not_create_missing_context_evidence(self) -> None:
        labels, gfap, structural, score, _distance = self.make_owner_evidence(
            shape=(5, 96, 96),
            center=(6, 6),
            radius=4,
        )
        structural[5:8, 10:90] = True
        score[structural] = 0.9
        result = self.select_single(labels, gfap, structural, score)
        self.assertEqual(result.accepted_ids, ())
        self.assertLess(result.records[0]["strong_contact_sector_count"], 3)

    def test_foreign_neighboring_process_cannot_seed_owner(self) -> None:
        labels, gfap, structural, score, _distance = self.make_owner_evidence()
        second = disk(labels.shape[1:], (80, 125), 5)
        for z_index in (1, 2, 3):
            labels[z_index, second] = 2
        # A process from the neighboring cell passes one side of nucleus 1.
        structural[76:80, 84:126] = True
        structural[70:92, 116:138] = True
        score[structural] = 0.9
        result = select_gfap_associated_owners(
            labels,
            gfap,
            structural,
            score,
            0.2,
            0.4,
            candidate_ids=(1,),
        )
        self.assertEqual(result.accepted_ids, ())
        self.assertLess(result.records[0]["strong_contact_sector_count"], 3)


class GFAPNucleusProjectionTests(unittest.TestCase):
    def test_z_separated_partial_projection_overlap_keeps_both_ids(self) -> None:
        labels = np.zeros((6, 72, 96), dtype=np.int32)
        first = disk(labels.shape[1:], (36, 42), 8)
        second = disk(labels.shape[1:], (36, 50), 8)
        labels[0:2, first] = 1
        labels[4:6, second] = 2
        result = project_exclusive_nucleus_owners(labels, (1, 2))
        self.assertEqual(result.retained_ids, (1, 2))
        self.assertGreater(result.collision_pixels, 0)
        self.assertEqual(
            {int(value) for value in np.unique(result.labels_yx) if value > 0},
            {1, 2},
        )

    def test_fully_overlapping_z_separated_projection_is_explicitly_rejected(self) -> None:
        labels = np.zeros((6, 72, 96), dtype=np.int32)
        nucleus = disk(labels.shape[1:], (36, 46), 8)
        labels[0:2, nucleus] = 1
        labels[4:6, nucleus] = 2
        result = project_exclusive_nucleus_owners(labels, (1, 2))
        self.assertEqual(result.retained_ids, ())
        self.assertEqual(result.collision_rejected_ids, (1, 2))
        self.assertGreater(result.collision_pixels, 0)

    def test_supplied_projection_cannot_silently_drop_an_owner(self) -> None:
        projection = np.zeros((32, 32), dtype=np.int32)
        projection[5:10, 5:10] = 1
        with self.assertRaisesRegex(ValueError, "silently lost"):
            validate_nucleus_projection(
                projection,
                projection.shape,
                {1, 2},
                require_all_ids=True,
            )


if __name__ == "__main__":
    unittest.main()
