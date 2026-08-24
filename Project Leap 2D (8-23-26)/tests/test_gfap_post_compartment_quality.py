from __future__ import annotations

import unittest

import numpy as np

from project_leap_2d.analysis_modes.gfap_only.gfap_post_compartment_quality import (
    GFAPPostCompartmentQualityConfig,
    apply_gfap_post_compartment_quality,
    classify_gfap_post_compartment_metrics,
)


def make_triplet():
    whole = np.zeros((30, 60), dtype=np.int32)
    soma = np.zeros_like(whole)
    processes = np.zeros_like(whole)
    for display_id, x0 in ((1, 2), (2, 22), (3, 42)):
        whole[2:22, x0 : x0 + 16] = display_id
        soma[2:12, x0 : x0 + 16] = display_id
        processes[12:22, x0 : x0 + 16] = display_id
    return whole, soma, processes


class GFAPPostCompartmentQualityTests(unittest.TestCase):
    def test_filters_renumbers_and_preserves_partition(self):
        whole, soma, processes = make_triplet()
        # IDs 1/3 fail process area; ID 2 passes and becomes display ID 1.
        processes[12:22, 2:18] = 0
        whole[12:22, 2:18] = 0
        processes[12:16, 2:10] = 1
        whole[12:16, 2:10] = 1
        processes[12:22, 42:58] = 0
        whole[12:22, 42:58] = 0
        processes[12:16, 42:50] = 3
        whole[12:16, 42:50] = 3
        result = apply_gfap_post_compartment_quality(
            whole,
            soma,
            processes,
            soma.copy(),
            pixel_size_um=0.5,
            source_owner_to_display_id={21: 1, 222: 2, 275: 3},
        )
        self.assertEqual(result.retained_display_ids, (2,))
        self.assertEqual(result.removed_display_ids, (1, 3))
        self.assertEqual(result.old_display_to_new_display_id, {2: 1})
        self.assertEqual(result.source_owner_to_display_id, {222: 1})
        self.assertEqual(set(np.unique(result.whole_labels)), {0, 1})
        self.assertEqual(set(np.unique(result.soma_labels)), {0, 1})
        self.assertEqual(set(np.unique(result.process_labels)), {0, 1})
        recombined = np.where(
            result.soma_labels > 0,
            result.soma_labels,
            result.process_labels,
        )
        np.testing.assert_array_equal(recombined, result.whole_labels)
        by_old_id = {row["old_display_id"]: row for row in result.records}
        self.assertIn("assigned_process_area_below_minimum", by_old_id[1]["rejection_reasons"])
        self.assertEqual(by_old_id[2]["new_display_id"], 1)
        self.assertIn(
            "assigned_process_area_below_minimum",
            by_old_id[3]["rejection_reasons"],
        )

    def test_mature_default_boundaries_are_inclusive(self):
        whole = np.zeros((30, 40), dtype=np.int32)
        soma = np.zeros_like(whole)
        processes = np.zeros_like(whole)
        # 500 Whole pixels; 60 Process pixels at 0.5 um/px = 15 um2 and
        # exactly 12% of Whole.
        whole[5:25, 5:30] = 1
        soma[5:25, 5:27] = 1
        processes[5:25, 27:30] = 1
        result = apply_gfap_post_compartment_quality(
            whole,
            soma,
            processes,
            soma.copy(),
            pixel_size_um=0.5,
            source_owner_to_display_id={222: 1},
        )
        self.assertEqual(result.retained_display_ids, (1,))
        record = result.records[0]
        self.assertEqual(record["assigned_process_area_um2"], 15.0)
        self.assertAlmostEqual(record["processes_whole_fraction"], 0.12)
        self.assertEqual(record["maximum_process_thickness_um"], 1.0)
        self.assertEqual(record["owner_centered_hub_distance_um"], 1.0)
        self.assertEqual(record["rejection_reasons"], [])

    def test_sample4_metric_table_retains_only_26_222_266(self):
        source_metrics = {
            21: (398.75, 0.807, 24.00),
            25: (6.06, 0.052, None),
            26: (48.61, 0.436, 10.52),
            27: (6.76, 0.073, None),
            29: (1.76, 0.044, None),
            30: (1.58, 0.029, None),
            68: (1.14, 0.024, None),
            146: (14.43, 0.104, None),
            222: (18.14, 0.170, 3.63),
            266: (129.35, 0.476, 1.91),
            275: (415.78, 0.905, 42.66),
        }
        retained = {
            source_id
            for source_id, (area, fraction, hub) in source_metrics.items()
            if not classify_gfap_post_compartment_metrics(
                assigned_process_area_um2=area,
                processes_whole_fraction=fraction,
                owner_centered_hub_distance_um=hub,
            )
        }
        self.assertEqual(retained, {26, 222, 266})

    def test_rejects_nonpartitioned_input(self):
        whole, soma, processes = make_triplet()
        processes[3, 3] = 1
        with self.assertRaisesRegex(ValueError, "exactly partitioned"):
            apply_gfap_post_compartment_quality(
                whole,
                soma,
                processes,
                soma.copy(),
                pixel_size_um=0.5,
                source_owner_to_display_id={21: 1, 222: 2, 275: 3},
            )

    def test_soma_border_rejects_but_process_only_border_is_flagged(self):
        whole = np.zeros((40, 60), dtype=np.int32)
        soma = np.zeros_like(whole)
        processes = np.zeros_like(whole)
        whole[0:30, 2:22] = 1
        soma[0:10, 2:22] = 1
        processes[10:30, 2:22] = 1
        whole[5:35, 35:60] = 2
        soma[10:20, 40:50] = 2
        processes[5:35, 35:60] = 2
        processes[10:20, 40:50] = 0
        with self.assertRaisesRegex(ValueError, "rejected every"):
            apply_gfap_post_compartment_quality(
                whole[:, :30],
                soma[:, :30],
                processes[:, :30],
                soma[:, :30].copy(),
                pixel_size_um=0.5,
                source_owner_to_display_id={21: 1},
            )
        result = apply_gfap_post_compartment_quality(
            whole[:, 30:],
            soma[:, 30:],
            processes[:, 30:],
            soma[:, 30:].copy(),
            pixel_size_um=0.5,
            source_owner_to_display_id={222: 2},
        )
        self.assertTrue(result.records[0]["incomplete_morphology"])
        self.assertFalse(result.records[0]["soma_touches_image_border"])

    def test_configuration_defaults_are_locked(self):
        config = GFAPPostCompartmentQualityConfig()
        self.assertEqual(config.minimum_process_area_um2, 15.0)
        self.assertEqual(config.minimum_process_whole_fraction, 0.12)
        self.assertEqual(config.maximum_owner_centered_hub_distance_um, 15.0)


if __name__ == "__main__":
    unittest.main()
