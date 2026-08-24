from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from project_leap_2d.analysis_modes.gfap_only import analyze_dapi_gfap_only
from project_leap_2d.fiji_review import cell_edit_transactions as tx
from test_cell_edit_transactions import base_state, identity, split_result
from test_gfap_only_analysis import make_synthetic_field, synthetic_config


def _array_snapshot(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.view(np.uint8))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


GFAP_ARRAY_SNAPSHOTS = {
    "corrected_gfap_projection": {
        "dtype": "float32",
        "sha256": "65b2b68ab70f3b3bf295566b609e4fe67344c7a5ca887e47459eb4a00eac90f9",
        "shape": [112, 160],
    },
    "gfap_intensity": {
        "dtype": "float32",
        "sha256": "d8fd4aa2f132c3f506a16deeb2d4f5e269147b7a4689ec3a6c16c9c831fd6e2c",
        "shape": [112, 160],
    },
    "gfap_ridge_score": {
        "dtype": "float32",
        "sha256": "13bf112b5bebe1347ec3767437728a3b02ef1f85909cf27089025889ddb8a97b",
        "shape": [112, 160],
    },
    "gfap_structural_mask": {
        "dtype": "bool",
        "sha256": "cea88376f8e72c7473d9ef3452f87266300345d3083ad9821230407c6e696b29",
        "shape": [112, 160],
    },
    "gfap_structural_score": {
        "dtype": "float32",
        "sha256": "0e881c46f434d3559a230d85c062300a17263e6ae1af260e338c77368f487e52",
        "shape": [112, 160],
    },
    "nucleus_labels_2d": {
        "dtype": "int32",
        "sha256": "60d683fdbdc69d05ae4deb2f75a2ecfdb4f9d9d402fb66e131d1f7e999a7a636",
        "shape": [112, 160],
    },
    "process_labels": {
        "dtype": "int32",
        "sha256": "8ce473bfbf08f4569c37f5cc6d4a282aa8c95efc4ece6117d68ab3a103781fdd",
        "shape": [112, 160],
    },
    "soma_labels": {
        "dtype": "int32",
        "sha256": "418c9978b81c3df6c869f0e8684b0cd96a0a9763f5c191c5110b48414577d68b",
        "shape": [112, 160],
    },
    "valid_nucleus_labels_2d": {
        "dtype": "int32",
        "sha256": "60d683fdbdc69d05ae4deb2f75a2ecfdb4f9d9d402fb66e131d1f7e999a7a636",
        "shape": [112, 160],
    },
    "whole_labels": {
        "dtype": "int32",
        "sha256": "bde86b3ebc4dc3515e43857b1c972446f3f1b57cc49ce4f8da0456ce6309587f",
        "shape": [112, 160],
    },
}

GFAP_DIAGNOSTIC_KEYS = (
    "age_profile",
    "analysis_mode",
    "measurement_channels_used_for_roi",
    "nucleus_count",
    "nucleus_ids",
    "process_area_px",
    "roi_definition_channels",
    "soma_area_px",
    "source_owner_ids",
    "source_owner_to_display_id",
    "unowned_gfap_structure_px",
    "valid_3d_nucleus_count",
    "whole_area_px",
    "z_spacing_um",
)

GFAP_DIAGNOSTIC_SNAPSHOT = {
    "age_profile": "mature",
    "analysis_mode": "dapi_gfap_only",
    "measurement_channels_used_for_roi": [],
    "nucleus_count": 2,
    "nucleus_ids": [1, 2],
    "process_area_px": 1004,
    "roi_definition_channels": ["DAPI", "GFAP"],
    "soma_area_px": 810,
    "source_owner_ids": [1, 2],
    "source_owner_to_display_id": {"1": 1, "2": 2},
    "unowned_gfap_structure_px": 0,
    "valid_3d_nucleus_count": 2,
    "whole_area_px": 1814,
    "z_spacing_um": 0.45,
}

CELL_EDIT_STATE_HASHES = {
    "base": "9991c06121f268eb2eb0a24ad26f00de2971feafff2bdbfa388b082caf052b37",
    "split": "9acfee4ad1a73a943bc1e6176b6529a489f20b40f3a078b71333b234720d9694",
    "enlarge": "c7f26f7f1dc59ef8282ed63e0ec15393e732edae6f7b7e1996f9d6af58dfaf16",
    "merge": "54471aa497bb975f2c7a907c188ecedf0fb5ef8d977c819839f760bd38e4355c",
}


class ScientificZeroDeltaSnapshotTests(unittest.TestCase):
    def test_complete_gfap_arrays_and_key_diagnostics_are_fixed(self) -> None:
        nuclei, gfap = make_synthetic_field()
        observed_runs = []
        for _repeat in range(3):
            result = analyze_dapi_gfap_only(
                nuclei,
                gfap,
                0.20,
                0.45,
                config=synthetic_config(),
            )
            observed = {
                name: _array_snapshot(getattr(result, name))
                for name in GFAP_ARRAY_SNAPSHOTS
            }
            self.assertEqual(observed, GFAP_ARRAY_SNAPSHOTS)
            diagnostic_snapshot = {
                key: result.diagnostics[key] for key in GFAP_DIAGNOSTIC_KEYS
            }
            self.assertEqual(diagnostic_snapshot, GFAP_DIAGNOSTIC_SNAPSHOT)
            observed_runs.append(
                hashlib.sha256(
                    json.dumps(
                        {
                            "arrays": observed,
                            "diagnostics": diagnostic_snapshot,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
        self.assertEqual(len(set(observed_runs)), 1)

    def test_split_enlarge_merge_and_revert_state_snapshots_are_fixed(self) -> None:
        base = base_state()
        self.assertEqual(base.state_hash, CELL_EDIT_STATE_HASHES["base"])

        split_whole, split_soma, split_processes, split_identities = split_result(
            base
        )
        split_proposal = tx.propose_split(
            base,
            source_cell_uid="cell-a",
            whole_labels=split_whole,
            soma_labels=split_soma,
            process_labels=split_processes,
            identities=split_identities,
            proposal_id="split-regression",
            audit={"fixture": "fixed"},
        )

        enlarged_whole = base.whole_labels.copy()
        enlarged_soma = base.soma_labels.copy()
        enlarged_processes = base.process_labels.copy()
        enlarged_whole[2, 2] = 1
        enlarged_soma[2, 2] = 1
        enlarge_proposal = tx.propose_enlarge(
            base,
            source_cell_uid="cell-a",
            whole_labels=enlarged_whole,
            soma_labels=enlarged_soma,
            process_labels=enlarged_processes,
            identities=base.identities,
            proposal_id="enlarge-regression",
            audit={"fixture": "fixed"},
        )

        merged_whole = np.where(base.whole_labels > 0, 1, 0)
        merged_soma = np.where(base.soma_labels > 0, 1, 0)
        merged_processes = np.where(merged_whole > 0, merged_whole, 0)
        merged_processes[merged_soma > 0] = 0
        merged_identity = identity(
            "merged-ab",
            owner="nucleus-a",
            lineage=("cell-a", "cell-b"),
        )
        merge_proposal = tx.propose_merge(
            base,
            source_cell_uids=("cell-a", "cell-b"),
            whole_labels=merged_whole,
            soma_labels=merged_soma,
            process_labels=merged_processes,
            identities={1: merged_identity},
            proposal_id="merge-regression",
            audit={"fixture": "fixed"},
        )

        for name, proposal in (
            ("split", split_proposal),
            ("enlarge", enlarge_proposal),
            ("merge", merge_proposal),
        ):
            ledger = tx.CellEditLedger(base)
            committed = ledger.commit(proposal)
            self.assertEqual(committed.state_hash, CELL_EDIT_STATE_HASHES[name])
            reverted = ledger.revert()
            self.assertEqual(reverted.state_hash, CELL_EDIT_STATE_HASHES["base"])
            self.assertEqual(reverted.version, 2)
            self.assertEqual(ledger.undo_depth, 0)
            np.testing.assert_array_equal(
                reverted.whole_labels,
                base.whole_labels,
            )
            np.testing.assert_array_equal(
                reverted.soma_labels,
                base.soma_labels,
            )
            np.testing.assert_array_equal(
                reverted.process_labels,
                base.process_labels,
            )


if __name__ == "__main__":
    unittest.main()
