from __future__ import annotations

import json
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile as tf

from project_leap_2d.fiji_review.cell_editing import (
    CELL_EDIT_SCHEMA_VERSION,
    CellEditRequestService,
    atomic_write_cell_edit_json,
    build_fiji_identity_records,
    cell_edit_sha256,
    prepare_cell_edit_runtime,
    validate_cell_edit_request,
    validate_cell_edit_response,
)
from project_leap_2d.fiji_review.review_validation import (
    validate_cell_edit_delta,
    validate_cell_edit_label_triplet,
)
from project_leap_2d.fiji_review.fiji_launcher import (
    terminate_fiji_process_group,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROOVY = (
    PROJECT_ROOT
    / "project_leap_2d"
    / "fiji_review"
    / "resources"
    / "astrocyte_roi_reviewer.groovy"
)


def simple_triplet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    whole = np.zeros((12, 14), dtype=np.uint16)
    whole[2:10, 2:12] = 1
    soma = np.zeros_like(whole)
    soma[4:8, 5:9] = 1
    processes = np.where((whole == 1) & (soma == 0), 1, 0).astype(np.uint16)
    return whole, soma, processes


class FijiCellEditProtocolTests(unittest.TestCase):
    def write_masks(
        self,
        directory: Path,
        prefix: str,
        values: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[dict[str, str], dict[str, str]]:
        paths = {}
        hashes = {}
        for key, value in zip(("whole", "soma", "processes"), values):
            path = directory / f"{prefix}_{key}.tif"
            tf.imwrite(path, value)
            paths[key] = str(path)
            hashes[key] = cell_edit_sha256(path)
        return paths, hashes

    def request_payload(
        self,
        directory: Path,
        values: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> dict:
        paths, hashes = self.write_masks(directory, "request", values)
        return {
            "schema_version": CELL_EDIT_SCHEMA_VERSION,
            "request_id": uuid.uuid4().hex,
            "action": "enlarge",
            "state_revision": 3,
            "state_token": uuid.uuid4().hex,
            "selected_original_id": 1,
            "selected_cell_uid": "cell-000001",
            "label_mask_paths": paths,
            "label_mask_sha256": hashes,
            "identity_records": [
                {
                    "label_id": 1,
                    "original_id": 1,
                    "cell_uid": "cell-000001",
                    "parent_uid": "",
                    "lineage": [1],
                    "owner_nucleus_id": "nucleus-1",
                }
            ],
        }

    def test_request_requires_three_hash_verified_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self.request_payload(root, simple_triplet())
            observed = validate_cell_edit_request(
                request,
                enabled_actions=("split", "enlarge"),
            )
            self.assertEqual(observed["action"], "enlarge")
            request["label_mask_sha256"]["whole"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_cell_edit_request(
                    request,
                    enabled_actions=("split", "enlarge"),
                )

    def test_response_is_bound_to_exact_request_and_unique_uids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self.request_payload(root, simple_triplet())
            response_paths, response_hashes = self.write_masks(
                root,
                "response",
                simple_triplet(),
            )
            response = {
                key: request[key]
                for key in ("request_id", "action", "state_revision", "state_token")
            }
            response.update(
                {
                    "schema_version": CELL_EDIT_SCHEMA_VERSION,
                    "status": "success",
                    "roi_count": 1,
                    "label_mask_paths": response_paths,
                    "label_mask_sha256": response_hashes,
                    "identity_records": request["identity_records"],
                }
            )
            self.assertEqual(
                validate_cell_edit_response(response, request=request)["status"],
                "success",
            )
            response["state_revision"] = 4
            with self.assertRaisesRegex(ValueError, "state_revision"):
                validate_cell_edit_response(response, request=request)

    def test_triplet_partition_and_action_delta_gates(self) -> None:
        whole, soma, processes = simple_triplet()
        identities = [
            {
                "label_id": 1,
                "original_id": 1,
                "cell_uid": "cell-000001",
            }
        ]
        validate_cell_edit_label_triplet(
            whole_labels=whole,
            soma_labels=soma,
            process_labels=processes,
            identity_records=identities,
        )
        enlarged_soma = soma.copy()
        enlarged_soma[3:9, 4:10] = 1
        enlarged_whole = whole.copy()
        enlarged_whole[3:9, 4:10] = 1
        enlarged_processes = np.where(
            (enlarged_whole == 1) & (enlarged_soma == 0),
            1,
            0,
        ).astype(np.uint16)
        validate_cell_edit_delta(
            action="enlarge",
            before_whole=whole,
            before_soma=soma,
            before_processes=processes,
            after_whole=enlarged_whole,
            after_soma=enlarged_soma,
            after_processes=enlarged_processes,
            selected_id=1,
        )
        bad_processes = enlarged_processes.copy()
        bad_processes[4, 5] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_cell_edit_label_triplet(
                whole_labels=enlarged_whole,
                soma_labels=enlarged_soma,
                process_labels=bad_processes,
                identity_records=identities,
            )

    def test_single_flight_service_publishes_validated_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            context = run_dir / "context.npz"
            context.write_bytes(b"context")
            config = prepare_cell_edit_runtime(
                run_dir,
                context_path=context,
                enabled_actions=("enlarge",),
                timeout_seconds=5.0,
            )
            manifest = {"cell_edit": config}
            request = self.request_payload(Path(config["state_dir"]), simple_triplet())
            response_paths, response_hashes = self.write_masks(
                Path(config["state_dir"]),
                "result",
                simple_triplet(),
            )

            def dispatcher(payload, unused_manifest):
                response = {
                    key: payload[key]
                    for key in (
                        "request_id",
                        "action",
                        "state_revision",
                        "state_token",
                    )
                }
                response.update(
                    {
                        "schema_version": CELL_EDIT_SCHEMA_VERSION,
                        "status": "success",
                        "roi_count": 1,
                        "label_mask_paths": response_paths,
                        "label_mask_sha256": response_hashes,
                        "identity_records": request["identity_records"],
                    }
                )
                return response

            service = CellEditRequestService(
                manifest=manifest,
                dispatcher=dispatcher,
            )
            request_path = Path(config["request_dir"]) / f"{request['request_id']}.json"
            atomic_write_cell_edit_json(request_path, request)
            response_path = (
                Path(config["response_dir"]) / f"{request['request_id']}.json"
            )
            deadline = time.monotonic() + 3.0
            while not response_path.exists() and time.monotonic() < deadline:
                service.poll()
                time.sleep(0.01)
            service.close()
            self.assertTrue(response_path.is_file())
            observed = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(observed["status"], "success")

    def test_groovy_exposes_only_manifest_enabled_python_actions(self) -> None:
        source = GROOVY.read_text(encoding="utf-8")
        self.assertIn('enabledPythonEdits.contains("split")', source)
        self.assertIn('enabledPythonEdits.contains("enlarge")', source)
        self.assertIn('new Thread({', source)
        self.assertIn("EventQueue.invokeAndWait", source)
        self.assertIn("state_token", source)
        self.assertIn("label_mask_sha256", source)
        self.assertIn("Cancel Cell Edit", source)
        self.assertIn("validateTripletRoiSets(loadedSets)", source)
        self.assertIn("frame.setSize(620, 310)", source)
        self.assertIn(
            "Revert steps backward through committed edits.",
            source,
        )
        self.assertNotIn("July 23, 2026", source)
        renumber = source[
            source.index("def renumberRoiSets = {") :
            source.index("def refreshPersistentViews = {")
        ]
        self.assertIn(
            "originToFinal[originalRoiId(roi) as Integer] = index + 1",
            renumber,
        )
        self.assertNotIn("roiLineage(roi).each", renumber)

    def test_failed_workflow_cleanup_stops_only_launched_process_group(self) -> None:
        process = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
        )
        try:
            terminate_fiji_process_group(process, grace_seconds=0.2)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)

    def test_split_children_share_source_lineage_but_not_original_id(self) -> None:
        identities = {
            1: SimpleNamespace(
                cell_uid="child-a",
                parent_uid="parent",
                owner_nucleus_uid="nucleus-a",
                lineage=("parent",),
            ),
            2: SimpleNamespace(
                cell_uid="child-b",
                parent_uid="parent",
                owner_nucleus_uid="nucleus-b",
                lineage=("parent",),
            ),
        }
        records = build_fiji_identity_records(
            result_identities=identities,
            request_identity_records=[
                {
                    "label_id": 1,
                    "original_id": 7,
                    "cell_uid": "parent",
                    "parent_uid": "",
                    "lineage": [7],
                    "owner_nucleus_id": "nucleus-a",
                }
            ],
        )
        self.assertEqual([value["lineage"] for value in records], [[7], [7]])
        self.assertEqual(records[0]["original_id"], 7)
        self.assertNotEqual(records[0]["original_id"], records[1]["original_id"])


if __name__ == "__main__":
    unittest.main()
