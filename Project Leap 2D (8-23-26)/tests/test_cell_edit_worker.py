from __future__ import annotations

import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile as tf

from project_leap_2d.fiji_review.cell_edit_worker import (
    _execute_edit,
    _merge_local_split_candidates,
    _split_local_fallback_allowed,
    dispatch_cell_edit_request,
)
from project_leap_2d.fiji_review.cell_edit_context import (
    build_cell_edit_context,
    load_cell_edit_context,
)


PROGRAM_ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CellEditWorkerTests(unittest.TestCase):
    def make_runtime(
        self,
        root: Path,
        *,
        two_nuclei: bool = False,
        canonical_missing_second: bool = False,
        second_nucleus_accepted: bool = True,
        analysis_mode: str = "egfp",
        gfap_foreign_in_growth_zone: bool = False,
    ):
        run = root / "run"
        edit = run / "cell_edit"
        request_dir = edit / "requests"
        response_dir = edit / "responses"
        cancel_dir = edit / "cancel"
        state_dir = edit / "state"
        for path in (request_dir, response_dir, cancel_dir, state_dir):
            path.mkdir(parents=True)

        shape = (180, 220) if two_nuclei else (80, 90)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        if two_nuclei:
            left = (yy - 90) ** 2 + (xx - 70) ** 2 <= 30**2
            right = (yy - 90) ** 2 + (xx - 132) ** 2 <= 28**2
            bridge = np.zeros(shape, dtype=bool)
            bridge[82:99, 70:133] = True
            parent = left | right | bridge
            whole = np.where(parent, 1, 0).astype(np.uint16)
            owner = (yy - 90) ** 2 + (xx - 70) ** 2 <= 11**2
            second = (yy - 90) ** 2 + (xx - 132) ** 2 <= 10**2
            soma_mask = (
                ((yy - 90) ** 2 + (xx - 70) ** 2 <= 23**2)
                | ((yy - 90) ** 2 + (xx - 132) ** 2 <= 21**2)
            ) & parent
            soma = np.where(soma_mask, 1, 0).astype(np.uint16)
            extent = np.zeros(shape, dtype=np.uint32)
            extent[owner] = 11
            extent[second] = 12
            core = extent.copy()
            records = [
                {
                    "instance_id": 11,
                    "accepted": True,
                    "dapi_valid": True,
                    "identity_status": "resolved",
                    "z_min_0based": 3,
                    "z_max_0based_inclusive": 9,
                },
                {
                    "instance_id": 12,
                    "accepted": bool(second_nucleus_accepted),
                    "dapi_valid": True,
                    "identity_status": "ambiguous",
                    "z_min_0based": 5,
                    "z_max_0based_inclusive": 11,
                },
            ]
            if canonical_missing_second:
                extent[second] = 0
                core = extent.copy()
                records = records[:1]
            structural = whole.astype(np.float32)
        else:
            first_whole = (yy - 40) ** 2 + (xx - 32) ** 2 <= 19**2
            second_whole = (yy - 40) ** 2 + (xx - 62) ** 2 <= 8**2
            first_soma = (yy - 40) ** 2 + (xx - 32) ** 2 <= 7**2
            second_soma = (yy - 40) ** 2 + (xx - 62) ** 2 <= 4**2
            whole = np.zeros(shape, dtype=np.uint16)
            soma = np.zeros(shape, dtype=np.uint16)
            whole[first_whole] = 1
            whole[second_whole] = 2
            soma[first_soma] = 1
            soma[second_soma] = 2
            owner = (yy - 40) ** 2 + (xx - 32) ** 2 <= 5**2
            foreign = (yy - 40) ** 2 + (xx - 62) ** 2 <= 3**2
            extent = np.zeros(shape, dtype=np.uint32)
            owner_source_id = 81 if analysis_mode == "gfap_only" else 11
            second_owner_source_id = (
                104 if analysis_mode == "gfap_only" else 22
            )
            extent[owner] = owner_source_id
            extent[foreign] = second_owner_source_id
            core = extent.copy()
            records = [
                {
                    "instance_id": owner_source_id,
                    "accepted": True,
                    "dapi_valid": True,
                    "identity_status": "resolved",
                    "z_min_0based": 2,
                    "z_max_0based_inclusive": 8,
                    **(
                        {"owner_display_id": 1}
                        if analysis_mode == "gfap_only"
                        else {}
                    ),
                },
                {
                    "instance_id": second_owner_source_id,
                    "accepted": True,
                    "dapi_valid": True,
                    "identity_status": "resolved",
                    "z_min_0based": 2,
                    "z_max_0based_inclusive": 8,
                    **(
                        {"owner_display_id": 2}
                        if analysis_mode == "gfap_only"
                        else {}
                    ),
                },
            ]
            if gfap_foreign_in_growth_zone:
                if analysis_mode != "gfap_only":
                    raise ValueError(
                        "The GFAP foreign-nucleus fixture requires gfap_only"
                    )
                rejected_valid = (
                    (yy - 40) ** 2 + (xx - 41) ** 2 <= 1**2
                )
                extent[rejected_valid] = 150
                core = extent.copy()
                records.append(
                    {
                        "instance_id": 150,
                        "accepted": False,
                        "dapi_valid": True,
                        "identity_status": "resolved",
                        "z_min_0based": 3,
                        "z_max_0based_inclusive": 7,
                        "source": "gfap_only_3d_dapi_nonowner",
                    }
                )
            structural = (
                np.exp(-((yy - 40) ** 2 + (xx - 32) ** 2) / 180.0) + 0.05
            ).astype(np.float32)
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.uint16
        )
        dapi_path = root / "DAPI.tif"
        structural_channel = "GFAP" if analysis_mode == "gfap_only" else "eGFP"
        structural_path = root / f"{structural_channel}.tif"
        tf.imwrite(
            dapi_path,
            np.repeat(
                np.asarray((extent > 0) * 65535, dtype=np.uint16)[None, ...],
                12,
                axis=0,
            ),
            metadata={"axes": "ZYX"},
        )
        tf.imwrite(structural_path, structural, metadata={"axes": "YX"})
        context_paths = build_cell_edit_context(
            run_dir=edit,
            dapi_path=dapi_path,
            structural_paths={structural_channel: structural_path},
            dapi_projection=np.asarray((extent > 0) * 65535, dtype=np.uint16),
            structural_map=structural,
            selected_z=(1, 12),
            calibration={
                "pixel_width_um": 0.20 if two_nuclei else 0.25,
                "pixel_height_um": 0.20 if two_nuclei else 0.25,
                "pixel_depth_um": 0.5,
            },
            age_profile="mature",
            canonical_core_labels=core,
            canonical_extent_labels=extent,
            initial_triplet=(whole, soma, processes),
            nucleus_records=records,
            analysis_mode=analysis_mode,
            structural_channel=structural_channel,
            basename="analysis_context",
        )
        mask_paths = {}
        mask_hashes = {}
        for name, array in (
            ("whole", whole),
            ("soma", soma),
            ("processes", processes),
        ):
            path = state_dir / f"request_{name}.tif"
            tf.imwrite(path, array)
            mask_paths[name] = str(path)
            mask_hashes[name] = _hash(path)
        manifest = {
            "expected_shape": [12, *shape],
            "channels": {"DAPI": str(dapi_path)},
            "cell_edit": {
                "enabled_actions": ["split", "enlarge"],
                "request_dir": str(request_dir),
                "response_dir": str(response_dir),
                "cancel_dir": str(cancel_dir),
                "state_dir": str(state_dir),
                "context_path": str(context_paths.json_path),
                "program_root": str(PROGRAM_ROOT),
                "timeout_seconds": 20.0,
            },
        }
        count = int(whole.max())
        identities = [
            {
                "label_id": value,
                "original_id": value,
                "cell_uid": f"cell-{value}",
                "parent_uid": "",
                "lineage": [value],
                "owner_nucleus_id": "",
            }
            for value in range(1, count + 1)
        ]
        request = {
            "schema_version": 1,
            "request_id": uuid.uuid4().hex,
            "action": "split" if two_nuclei else "enlarge",
            "state_revision": 0,
            "state_token": uuid.uuid4().hex,
            "selected_original_id": 1,
            "selected_cell_uid": "cell-1",
            "label_mask_paths": mask_paths,
            "label_mask_sha256": mask_hashes,
            "identity_records": identities,
        }
        return manifest, request, whole, soma, processes

    def test_context_roundtrip_and_strict_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _, _, _ = self.make_runtime(Path(directory))
            loaded = load_cell_edit_context(
                Path(manifest["cell_edit"]["context_path"])
            )
            self.assertEqual(loaded.metadata["schema_version"], 1)
            self.assertEqual(loaded.array("structural_map").shape, (80, 90))
            self.assertEqual(
                [
                    record["instance_id"]
                    for record in loaded.metadata["nucleus_records"]
                ],
                [11, 22],
            )

    def test_enlarge_runs_in_subprocess_and_publishes_atomic_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, request, whole, soma, _ = self.make_runtime(Path(directory))
            response = dispatch_cell_edit_request(request, manifest)
            self.assertEqual(response["status"], "success", response)
            self.assertEqual(response["roi_count"], 2)
            output_whole = tf.imread(response["label_mask_paths"]["whole"])
            output_soma = tf.imread(response["label_mask_paths"]["soma"])
            self.assertTrue(np.all(output_soma[soma == 1] == 1))
            self.assertGreater(int((output_soma == 1).sum()), int((soma == 1).sum()))
            self.assertTrue(np.all(output_whole[whole == 2] == 2))
            for key, path in response["label_mask_paths"].items():
                self.assertEqual(_hash(Path(path)), response["label_mask_sha256"][key])

    def test_gfap_enlarge_selects_the_isolated_dapi_led_policy(self) -> None:
        from project_leap_2d.compartments import selected_soma_enlargement

        with tempfile.TemporaryDirectory() as directory:
            manifest, request, _, _, _ = self.make_runtime(
                Path(directory),
                analysis_mode="gfap_only",
            )
            with mock.patch.object(
                selected_soma_enlargement,
                "selected_soma_enlargement_config_for_mode",
                wraps=(
                    selected_soma_enlargement
                    .selected_soma_enlargement_config_for_mode
                ),
            ) as mode_policy:
                response = _execute_edit(request, manifest)
            self.assertEqual(response["status"], "success", response)
            mode_policy.assert_called_once_with("gfap_only")

    def test_gfap_rejected_valid_nucleus_blocks_enlarge_growth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, request, _, _, _ = self.make_runtime(
                Path(directory),
                analysis_mode="gfap_only",
                gfap_foreign_in_growth_zone=True,
            )
            response = _execute_edit(request, manifest)
            self.assertEqual(response["status"], "success", response)
            output_soma = tf.imread(response["label_mask_paths"]["soma"])
            loaded = load_cell_edit_context(
                Path(manifest["cell_edit"]["context_path"])
            )
            extent = loaded.array("canonical_nucleus_extent_labels")
            rejected_valid = extent == 150
            self.assertTrue(rejected_valid.any())
            self.assertFalse(np.any(output_soma[rejected_valid] == 1))

    def test_split_adds_exactly_one_identity_in_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, request, whole, _, _ = self.make_runtime(
                Path(directory), two_nuclei=True
            )
            response = dispatch_cell_edit_request(request, manifest)
            self.assertEqual(response["status"], "success", response)
            self.assertEqual(response["roi_count"], 2)
            output_whole = tf.imread(response["label_mask_paths"]["whole"])
            self.assertEqual(
                sorted(int(value) for value in np.unique(output_whole) if value > 0),
                [1, 2],
            )
            self.assertEqual(len(response["identity_records"]), 2)
            self.assertNotEqual(
                response["identity_records"][0]["cell_uid"],
                response["identity_records"][1]["cell_uid"],
            )
            self.assertFalse(np.array_equal(output_whole, whole))

    def test_request_mask_path_escape_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, request, _, _, _ = self.make_runtime(root)
            escaped = root / "outside.tif"
            tf.imwrite(escaped, np.zeros((80, 90), dtype=np.uint16))
            request["label_mask_paths"]["whole"] = str(escaped)
            request["label_mask_sha256"]["whole"] = _hash(escaped)
            with self.assertRaisesRegex(ValueError, "outside this run"):
                dispatch_cell_edit_request(request, manifest)

    def test_context_path_escape_is_rejected_before_worker_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, request, _, _, _ = self.make_runtime(root)
            manifest["cell_edit"]["context_path"] = str(
                root / "analysis_context.json"
            )
            Path(manifest["cell_edit"]["context_path"]).write_text(
                "{}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside this run"):
                dispatch_cell_edit_request(request, manifest)

    def test_stale_identity_rejects_without_result_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, request, _, _, _ = self.make_runtime(Path(directory))
            request["selected_cell_uid"] = "stale-cell"
            response = dispatch_cell_edit_request(request, manifest)
            self.assertEqual(response["status"], "error")
            self.assertIn("stale", response["reason"].lower())
            state_dir = Path(manifest["cell_edit"]["state_dir"])
            self.assertFalse(
                (state_dir / f"{request['request_id']}_result").exists()
            )

    def test_local_model_recovers_visually_two_nuclei_when_inventory_missed_one(
        self,
    ) -> None:
        from project_leap_2d.compartments.selected_cell_split import (
            SplitNucleusCandidate,
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest, request, whole, _, _ = self.make_runtime(
                Path(directory),
                two_nuclei=True,
                canonical_missing_second=True,
            )
            yy, xx = np.ogrid[: whole.shape[0], : whole.shape[1]]
            second = (yy - 90) ** 2 + (xx - 132) ** 2 <= 10**2
            model_candidate = SplitNucleusCandidate(
                nucleus_id=12,
                projection_mask=second,
                dapi_valid=True,
                identity_status="model_proposal",
                owner_astrocyte_id=None,
                accepted=False,
                confidence=0.85,
                z_min_0based=5,
                z_max_0based=11,
                source="test-local-model",
                locally_confirmed=True,
            )
            with mock.patch(
                "project_leap_2d.fiji_review.cell_edit_worker."
                "_instanseg_split_candidates",
                return_value=[model_candidate],
            ) as recover:
                response = _execute_edit(request, manifest)
            self.assertEqual(response["status"], "success", response)
            recover.assert_called_once()

    def test_unaccepted_canonical_nucleus_is_locally_confirmed_before_split(
        self,
    ) -> None:
        from project_leap_2d.compartments.selected_cell_split import (
            SplitNucleusCandidate,
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest, request, whole, _, _ = self.make_runtime(
                Path(directory),
                two_nuclei=True,
                second_nucleus_accepted=False,
            )
            yy, xx = np.ogrid[: whole.shape[0], : whole.shape[1]]
            refined_second = (yy - 90) ** 2 + (xx - 132) ** 2 <= 10**2
            model_candidate = SplitNucleusCandidate(
                nucleus_id=12,
                projection_mask=refined_second,
                dapi_valid=True,
                identity_status="model_proposal",
                accepted=False,
                confidence=0.85,
                z_min_0based=5,
                z_max_0based=11,
                source="instanseg_local_z_refinement",
                locally_confirmed=True,
            )
            with mock.patch(
                "project_leap_2d.fiji_review.cell_edit_worker."
                "_instanseg_split_candidates",
                return_value=[model_candidate],
            ) as recover:
                response = _execute_edit(request, manifest)

            self.assertEqual(response["status"], "success", response)
            recover.assert_called_once()

    def test_local_model_can_refine_an_existing_canonical_nucleus(self) -> None:
        from project_leap_2d.fiji_review.cell_edit_worker import (
            _link_local_model_candidates,
        )

        labels = np.zeros((2, 20, 20), dtype=np.uint16)
        labels[:, 5:11, 7:13] = 1
        canonical = np.zeros((20, 20), dtype=np.uint32)
        canonical[5:11, 7:13] = 41

        candidates = _link_local_model_candidates(
            labels,
            z_indices=(3, 4),
            crop=(0, 20, 0, 20),
            full_shape=(20, 20),
            first_id=100,
            canonical_extent=canonical,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].nucleus_id, 41)
        self.assertEqual(
            candidates[0].source,
            "instanseg_local_z_refinement",
        )
        self.assertTrue(candidates[0].locally_confirmed)

    def test_local_model_replaces_tiny_provisional_projection_but_keeps_identity(
        self,
    ) -> None:
        from project_leap_2d.compartments.selected_cell_split import (
            SplitNucleusCandidate,
        )

        canonical_mask = np.zeros((40, 40), dtype=bool)
        canonical_mask[19:22, 19:22] = True
        refined_mask = np.zeros((40, 40), dtype=bool)
        refined_mask[14:27, 14:27] = True
        canonical = SplitNucleusCandidate(
            nucleus_id=66,
            projection_mask=canonical_mask,
            owner_astrocyte_id=10,
            accepted=False,
            confidence=0.55,
            source="dapi_3d_inventory",
        )
        local = SplitNucleusCandidate(
            nucleus_id=66,
            projection_mask=refined_mask,
            accepted=False,
            confidence=0.82,
            z_min_0based=4,
            z_max_0based=9,
            source="instanseg_local_z_refinement",
            locally_confirmed=True,
        )

        merged = _merge_local_split_candidates([canonical], [local])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].nucleus_id, 66)
        self.assertEqual(merged[0].owner_astrocyte_id, 10)
        self.assertFalse(merged[0].accepted)
        self.assertTrue(merged[0].locally_confirmed)
        self.assertTrue(np.array_equal(merged[0].projection_mask, refined_mask))
        self.assertIn("instanseg_local_confirmation", merged[0].source)

    def test_foreign_owned_nucleus_never_invokes_local_model_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, request, _, _, _ = self.make_runtime(Path(directory))
            request["action"] = "split"
            with mock.patch(
                "project_leap_2d.fiji_review.cell_edit_worker."
                "_instanseg_split_candidates"
            ) as recover:
                response = _execute_edit(request, manifest)
            self.assertEqual(response["status"], "rejected", response)
            self.assertIn("another astrocyte", response["reason"])
            recover.assert_not_called()

    def test_unknown_z_allows_narrow_fallback_but_known_conflict_does_not(self):
        from project_leap_2d.compartments.selected_cell_split import (
            SplitNucleusCandidate,
        )

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        unknown = SplitNucleusCandidate(
            nucleus_id=1,
            projection_mask=mask,
            dapi_valid=True,
            identity_status="resolved",
            owner_astrocyte_id=1,
            accepted=True,
            confidence=0.9,
        )
        known = SplitNucleusCandidate(
            nucleus_id=2,
            projection_mask=mask,
            dapi_valid=True,
            identity_status="resolved",
            owner_astrocyte_id=1,
            accepted=True,
            confidence=0.9,
            z_min_0based=2,
            z_max_0based=5,
        )
        refusal = "The two nuclear candidates cannot be separated."
        self.assertTrue(_split_local_fallback_allowed(refusal, [unknown, known]))
        self.assertFalse(_split_local_fallback_allowed(refusal, [known]))
        self.assertFalse(
            _split_local_fallback_allowed(
                "The second nucleus already belongs to another astrocyte.",
                [unknown],
            )
        )


if __name__ == "__main__":
    unittest.main()
