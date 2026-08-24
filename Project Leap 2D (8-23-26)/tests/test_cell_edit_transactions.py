from __future__ import annotations

import unittest

import numpy as np

from project_leap_2d.fiji_review import cell_edit_transactions as tx


def identity(name, *, parent=None, owner=None, lineage=()):
    return tx.CellIdentity(
        cell_uid=name,
        parent_uid=parent,
        owner_nucleus_uid=owner,
        lineage=tuple(lineage) or (name,),
    )


def base_state():
    whole = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
        ],
        dtype=np.uint16,
    )
    soma = np.array(
        [
            [1, 0, 0, 0, 2],
            [1, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    processes = np.where(whole > 0, whole, 0)
    processes[soma > 0] = 0
    return tx.CellEditState(
        whole,
        soma,
        processes,
        {
            1: identity("cell-a", owner="nucleus-a"),
            2: identity("cell-b", owner="nucleus-b"),
        },
    )


def split_result(base):
    whole = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [3, 3, 0, 2, 2],
        ],
        dtype=np.uint32,
    )
    soma = np.array(
        [
            [1, 0, 0, 0, 2],
            [1, 0, 0, 0, 2],
            [3, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    processes = np.where(whole > 0, whole, 0)
    processes[soma > 0] = 0
    identities = {
        1: identity(
            "child-a1", parent="cell-a", owner="nucleus-a", lineage=("cell-a",)
        ),
        2: base.identity_by_uid("cell-b"),
        3: identity(
            "child-a2", parent="cell-a", owner="nucleus-c", lineage=("cell-a",)
        ),
    }
    return whole, soma, processes, identities


class CellEditTransactionTests(unittest.TestCase):
    def test_triplet_requires_exact_partition_and_consecutive_ids(self):
        base = base_state()
        self.assertEqual(base.cell_count, 2)
        self.assertEqual(base.state_hash, base.with_version(99).state_hash)
        broken = base.process_labels.copy()
        broken[0, 1] = 0
        with self.assertRaisesRegex(tx.CellEditValidationError, "exact partition"):
            tx.CellEditState(
                base.whole_labels,
                base.soma_labels,
                broken,
                base.identities,
            )
        with self.assertRaisesRegex(tx.CellEditValidationError, "consecutive"):
            tx.CellEditState(
                np.where(base.whole_labels == 2, 3, base.whole_labels),
                np.where(base.soma_labels == 2, 3, base.soma_labels),
                np.where(base.process_labels == 2, 3, base.process_labels),
                {1: base.identities[1], 3: base.identities[2]},
            )

    def test_identity_derivation_is_reproducible(self):
        first = tx.make_initial_cell_identity(
            identity_namespace="run-hash",
            original_display_id=4,
            owner_nucleus_uid="n4",
        )
        again = tx.make_initial_cell_identity(
            identity_namespace="run-hash",
            original_display_id=4,
            owner_nucleus_uid="n4",
        )
        self.assertEqual(first, again)
        child_1 = tx.make_split_child_identity(
            first,
            owner_nucleus_uid="n4",
            child_index=1,
            edit_nonce="edit-1",
        )
        child_2 = tx.make_split_child_identity(
            first,
            owner_nucleus_uid="n5",
            child_index=2,
            edit_nonce="edit-1",
        )
        self.assertEqual(child_1.parent_uid, first.cell_uid)
        self.assertNotEqual(child_1.cell_uid, child_2.cell_uid)
        merged = tx.make_merged_cell_identity(
            (child_1, child_2),
            owner_nucleus_uid="n4",
            edit_nonce="edit-2",
        )
        self.assertTrue(
            {child_1.cell_uid, child_2.cell_uid}.issubset(set(merged.lineage))
        )

    def test_split_commit_revert_and_stale_rejection(self):
        base = base_state()
        whole, soma, processes, identities = split_result(base)
        proposal = tx.propose_split(
            base,
            source_cell_uid="cell-a",
            whole_labels=whole,
            soma_labels=soma,
            process_labels=processes,
            identities=identities,
            proposal_id="split-1",
            audit={"candidate": "manual"},
        )
        ledger = tx.CellEditLedger(base)
        result = ledger.commit(proposal)
        self.assertEqual(result.cell_count, 3)
        self.assertEqual(result.version, 1)
        restored = ledger.revert()
        self.assertEqual(restored.version, 2)
        self.assertEqual(restored.state_hash, base.state_hash)
        self.assertTrue(np.array_equal(restored.whole_labels, base.whole_labels))

        stale_ledger = tx.CellEditLedger(base.with_version(4))
        unchanged = stale_ledger.current_state
        with self.assertRaisesRegex(tx.StaleCellEditError, "revision"):
            stale_ledger.commit(proposal)
        self.assertIs(stale_ledger.current_state, unchanged)
        self.assertEqual(stale_ledger.undo_depth, 0)

    def test_split_rejects_duplicate_owner_and_changed_unaffected_cell(self):
        base = base_state()
        whole, soma, processes, identities = split_result(base)
        identities[3] = identity(
            "child-a2",
            parent="cell-a",
            owner="nucleus-a",
            lineage=("cell-a",),
        )
        with self.assertRaisesRegex(tx.CellEditValidationError, "distinct owner"):
            tx.propose_split(
                base,
                source_cell_uid="cell-a",
                whole_labels=whole,
                soma_labels=soma,
                process_labels=processes,
                identities=identities,
            )
        identities[3] = identity(
            "child-a2",
            parent="cell-a",
            owner="nucleus-c",
            lineage=("cell-a",),
        )
        changed = whole.copy()
        changed[2, 4] = 0
        changed_processes = processes.copy()
        changed_processes[2, 4] = 0
        with self.assertRaisesRegex(tx.CellEditValidationError, "Unedited whole"):
            tx.propose_split(
                base,
                source_cell_uid="cell-a",
                whole_labels=changed,
                soma_labels=soma,
                process_labels=changed_processes,
                identities=identities,
            )

    def test_merge_exact_union_and_renumbering(self):
        base = base_state()
        whole = np.where(base.whole_labels > 0, 1, 0)
        soma = np.where(base.soma_labels > 0, 1, 0)
        processes = np.where(whole > 0, whole, 0)
        processes[soma > 0] = 0
        merged = identity(
            "merged-ab", owner="nucleus-a", lineage=("cell-a", "cell-b")
        )
        proposal = tx.propose_merge(
            base,
            source_cell_uids=("cell-a", "cell-b"),
            whole_labels=whole,
            soma_labels=soma,
            process_labels=processes,
            identities={1: merged},
        )
        self.assertEqual(tx.commit_cell_edit(base, proposal).cell_count, 1)

        result = tx.renumber_label_triplet(
            np.where(base.whole_labels == 2, 9, np.where(base.whole_labels == 1, 4, 0)),
            np.where(base.soma_labels == 2, 9, np.where(base.soma_labels == 1, 4, 0)),
            np.where(
                base.process_labels == 2,
                9,
                np.where(base.process_labels == 1, 4, 0),
            ),
            {4: base.identities[1], 9: base.identities[2]},
            display_order=(9, 4),
        )
        r_whole, r_soma, r_processes, r_identities, mapping = result
        self.assertEqual(mapping, {9: 1, 4: 2})
        renumbered = tx.CellEditState(
            r_whole, r_soma, r_processes, r_identities
        )
        self.assertEqual(renumbered.identities[1].cell_uid, "cell-b")

    def test_enlarge_synchronizes_whole_and_soma(self):
        base = base_state()
        whole = base.whole_labels.copy()
        soma = base.soma_labels.copy()
        processes = base.process_labels.copy()
        whole[2, 2] = 1
        soma[2, 2] = 1
        proposal = tx.propose_enlarge(
            base,
            source_cell_uid="cell-a",
            whole_labels=whole,
            soma_labels=soma,
            process_labels=processes,
            identities=base.identities,
        )
        result = tx.commit_cell_edit(base, proposal)
        self.assertEqual(result.cell_count, base.cell_count)
        self.assertEqual(result.whole_labels[2, 2], 1)
        self.assertEqual(result.soma_labels[2, 2], 1)

        invalid_processes = processes.copy()
        invalid_processes[2, 2] = 1
        invalid_soma = soma.copy()
        invalid_soma[2, 2] = 0
        invalid_soma[2, 0] = 1
        invalid_processes[2, 0] = 0
        with self.assertRaisesRegex(tx.CellEditValidationError, "simultaneously"):
            tx.propose_enlarge(
                base,
                source_cell_uid="cell-a",
                whole_labels=whole,
                soma_labels=invalid_soma,
                process_labels=invalid_processes,
                identities=base.identities,
            )

    def test_protocol_round_trip_and_corruption_detection(self):
        base = base_state()
        restored = tx.CellEditState.from_protocol_dict(base.to_protocol_dict())
        self.assertEqual(restored.state_hash, base.state_hash)
        self.assertTrue(np.array_equal(restored.soma_labels, base.soma_labels))

        whole, soma, processes, identities = split_result(base)
        proposal = tx.propose_split(
            base,
            source_cell_uid="cell-a",
            whole_labels=whole,
            soma_labels=soma,
            process_labels=processes,
            identities=identities,
            proposal_id="split-roundtrip",
        )
        encoded = tx.protocol_json_dumps(proposal.to_protocol_dict())
        decoded = tx.protocol_json_loads(encoded)
        restored_proposal = tx.CellEditProposal.from_protocol_dict(decoded)
        self.assertEqual(
            restored_proposal.result_state.state_hash,
            proposal.result_state.state_hash,
        )
        self.assertEqual(restored_proposal.source_cell_uids, ("cell-a",))

        corrupt = base.to_protocol_dict()
        corrupt["labels"]["whole"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(tx.CellEditProtocolError, "checksum"):
            tx.CellEditState.from_protocol_dict(corrupt)

    def test_label_arrays_are_immutable_copies(self):
        base = base_state()
        self.assertFalse(base.whole_labels.flags.writeable)
        with self.assertRaises(ValueError):
            base.whole_labels[0, 0] = 0


if __name__ == "__main__":
    unittest.main()
