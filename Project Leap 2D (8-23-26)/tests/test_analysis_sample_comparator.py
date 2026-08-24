from __future__ import annotations

import unittest

import numpy as np

from compare_analysis_samples import compare


def triplet() -> dict[str, np.ndarray]:
    whole = np.zeros((10, 14), dtype=np.uint16)
    whole[2:8, 2:7] = 1
    whole[2:8, 9:13] = 2
    soma = np.zeros_like(whole)
    soma[4:6, 3:5] = 1
    soma[4:6, 10:12] = 2
    processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(np.uint16)
    return {
        "whole_labels": whole,
        "soma_labels": soma,
        "process_labels": processes,
    }


class AnalysisSampleComparatorTests(unittest.TestCase):
    def test_mature_zero_delta_checks_every_array(self) -> None:
        reference = triplet()
        reference["candidate_01"] = np.arange(6, dtype=np.uint16)
        candidate = {name: value.copy() for name, value in reference.items()}
        self.assertEqual(
            compare(
                profile="mature_zero_delta",
                reference=reference,
                candidate=candidate,
                plan={},
            ),
            [],
        )
        candidate["candidate_01"][0] = 9
        self.assertEqual(
            compare(
                profile="mature_zero_delta",
                reference=reference,
                candidate=candidate,
                plan={},
            ),
            ["zero-delta array changed: candidate_01"],
        )

    def test_scientific_zero_delta_checks_complete_gfap_payload(self) -> None:
        reference = triplet()
        reference.update(
            {
                "nucleus_labels_2d": np.arange(140, dtype=np.int32).reshape(10, 14),
                "valid_nucleus_labels_2d": np.arange(
                    140, dtype=np.int32
                ).reshape(10, 14),
                "gfap_structural_score": np.linspace(
                    0.0,
                    1.0,
                    140,
                    dtype=np.float32,
                ).reshape(10, 14),
            }
        )
        candidate = {name: value.copy() for name, value in reference.items()}
        self.assertEqual(
            compare(
                profile="scientific_zero_delta",
                reference=reference,
                candidate=candidate,
                plan={},
            ),
            [],
        )
        candidate["gfap_structural_score"][4, 7] = np.float32(0.25)
        self.assertEqual(
            compare(
                profile="scientific_zero_delta",
                reference=reference,
                candidate=candidate,
                plan={},
            ),
            ["zero-delta array changed: gfap_structural_score"],
        )

    def test_scientific_zero_delta_rejects_dtype_and_shape_drift(self) -> None:
        reference = triplet()
        reference["gfap_structural_score"] = np.ones((10, 14), dtype=np.float32)

        dtype_candidate = {
            name: value.copy() for name, value in reference.items()
        }
        dtype_candidate["gfap_structural_score"] = dtype_candidate[
            "gfap_structural_score"
        ].astype(np.float64)
        self.assertEqual(
            compare(
                profile="scientific_zero_delta",
                reference=reference,
                candidate=dtype_candidate,
                plan={},
            ),
            [
                "zero-delta array dtype changed: "
                "gfap_structural_score float32 -> float64"
            ],
        )

        shape_candidate = {
            name: value.copy() for name, value in reference.items()
        }
        shape_candidate["gfap_structural_score"] = shape_candidate[
            "gfap_structural_score"
        ][:9, :]
        self.assertEqual(
            compare(
                profile="scientific_zero_delta",
                reference=reference,
                candidate=shape_candidate,
                plan={},
            ),
            [
                "zero-delta array shape changed: "
                "gfap_structural_score (10, 14) -> (9, 14)"
            ],
        )

    def test_split_requires_one_extra_cell_and_preserves_untargeted_cell(self) -> None:
        reference = triplet()
        candidate = {name: value.copy() for name, value in reference.items()}
        for name in ("whole_labels", "soma_labels", "process_labels"):
            selected = candidate[name] == 1
            candidate[name][selected & (np.indices(selected.shape)[1] >= 4)] = 3
        errors = compare(
            profile="split",
            reference=reference,
            candidate=candidate,
            plan={
                "target_reference_id": 1,
                "child_candidate_ids": [1, 3],
                "unchanged_id_map": {"2": 2},
            },
        )
        self.assertEqual(errors, [])

    def test_split_rejects_incomplete_untargeted_mapping(self) -> None:
        reference = triplet()
        candidate = {name: value.copy() for name, value in reference.items()}
        for name in ("whole_labels", "soma_labels", "process_labels"):
            selected = candidate[name] == 1
            candidate[name][selected & (np.indices(selected.shape)[1] >= 4)] = 3
        errors = compare(
            profile="split",
            reference=reference,
            candidate=candidate,
            plan={
                "target_reference_id": 1,
                "child_candidate_ids": [1, 3],
                "unchanged_id_map": {},
            },
        )
        self.assertIn(
            (
                "Split unchanged_id_map does not cover every untargeted "
                "reference cell: expected=[2], observed=[]"
            ),
            errors,
        )

    def test_edit_profiles_report_missing_plan_without_crashing(self) -> None:
        reference = triplet()
        candidate = {name: value.copy() for name, value in reference.items()}
        self.assertIn(
            (
                "Split comparison plan requires integer target_reference_id "
                "and two child_candidate_ids"
            ),
            compare(
                profile="split",
                reference=reference,
                candidate=candidate,
                plan={},
            ),
        )
        self.assertIn(
            (
                "Enlarge comparison plan requires an integer "
                "target_reference_id"
            ),
            compare(
                profile="enlarge",
                reference=reference,
                candidate=candidate,
                plan={},
            ),
        )

    def test_enlarge_accepts_soma_and_whole_growth_outside_old_whole(self) -> None:
        reference = triplet()
        candidate = {name: value.copy() for name, value in reference.items()}
        candidate["whole_labels"][1, 3:6] = 1
        candidate["soma_labels"][1, 3:6] = 1
        candidate["process_labels"] = np.where(
            (candidate["whole_labels"] > 0) & (candidate["soma_labels"] == 0),
            candidate["whole_labels"],
            0,
        ).astype(np.uint16)
        errors = compare(
            profile="enlarge",
            reference=reference,
            candidate=candidate,
            plan={
                "target_reference_id": 1,
                "unchanged_id_map": {"2": 2},
            },
        )
        self.assertEqual(errors, [])

    def test_enlarge_rejects_soma_outside_updated_whole(self) -> None:
        reference = triplet()
        candidate = {name: value.copy() for name, value in reference.items()}
        candidate["soma_labels"][1, 3] = 1
        errors = compare(
            profile="enlarge",
            reference=reference,
            candidate=candidate,
            plan={
                "target_reference_id": 1,
                "unchanged_id_map": {"2": 2},
            },
        )
        self.assertIn(
            "candidate Whole != Soma union Processes with matching IDs",
            errors,
        )

    def test_gfap_only_checks_partition_but_requires_visual_review_separately(self) -> None:
        self.assertEqual(
            compare(
                profile="gfap_only",
                reference=None,
                candidate=triplet(),
                plan={},
            ),
            [],
        )

    def test_gfap_only_rejects_noninteger_label_arrays(self) -> None:
        candidate = {
            name: value.astype(np.float32)
            for name, value in triplet().items()
        }
        errors = compare(
            profile="gfap_only",
            reference=None,
            candidate=candidate,
            plan={},
        )
        self.assertIn(
            "candidate compartment arrays do not use integer IDs",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
