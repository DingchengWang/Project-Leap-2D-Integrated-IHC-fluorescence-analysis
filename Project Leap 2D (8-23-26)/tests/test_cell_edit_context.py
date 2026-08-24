from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_leap_2d.fiji_review import cell_edit_context as context


def valid_arrays(shape: tuple[int, int] = (18, 24)):
    whole = np.zeros(shape, dtype=np.uint16)
    whole[2:9, 2:10] = 1
    whole[10:17, 13:22] = 2
    soma = np.zeros_like(whole)
    soma[4:8, 4:8] = 1
    soma[12:16, 15:19] = 2
    processes = whole.copy()
    processes[soma > 0] = 0
    core = np.zeros(shape, dtype=np.uint32)
    core[5:7, 5:7] = 101
    core[13:15, 16:18] = 202
    extent = np.zeros(shape, dtype=np.uint32)
    extent[4:8, 4:8] = 101
    extent[12:16, 15:19] = 202
    dapi = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    structural = np.linspace(0, 1, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    return dapi, structural, core, extent, whole, soma, processes


class CellEditContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dapi_path = self.root / "DAPI.tif"
        self.egfp_path = self.root / "eGFP.tif"
        self.gfap_path = self.root / "GFAP.tif"
        self.dapi_path.write_bytes(bytes(range(256)) * 8)
        self.egfp_path.write_bytes(bytes(reversed(range(256))) * 8)
        self.gfap_path.write_bytes(bytes(range(128)) * 16)
        (
            self.dapi,
            self.structural,
            self.core,
            self.extent,
            self.whole,
            self.soma,
            self.processes,
        ) = valid_arrays()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, *, age_profile: str = "mature"):
        return context.build_cell_edit_context(
            run_dir=self.root / "run",
            dapi_path=self.dapi_path,
            structural_paths={"eGFP": self.egfp_path},
            dapi_projection=self.dapi,
            structural_map=self.structural,
            selected_z={
                "z_start_1based": 3,
                "z_end_1based_inclusive": 11,
                "projection": "max",
            },
            calibration={
                "pixel_width_um": 0.09,
                "pixel_height_um": 0.091,
                "pixel_depth_um": 0.25,
                "unit": "um",
            },
            age_profile=age_profile,
            canonical_core_labels=self.core,
            canonical_extent_labels=self.extent,
            initial_triplet={
                "whole_labels": self.whole,
                "soma_labels": self.soma,
                "process_labels": self.processes,
            },
        )

    def gfap_build_kwargs(self, *, age_profile: str) -> dict:
        return {
            "run_dir": self.root / f"gfap-{age_profile}",
            "dapi_path": self.dapi_path,
            "structural_paths": {"GFAP": self.gfap_path},
            "dapi_projection": self.dapi,
            "structural_map": self.structural,
            "selected_z": (3, 11),
            "calibration": {
                "pixel_width_um": 0.09,
                "pixel_height_um": 0.091,
                "pixel_depth_um": 0.25,
            },
            "age_profile": age_profile,
            "canonical_core_labels": self.core,
            "canonical_extent_labels": self.extent,
            "initial_triplet": (self.whole, self.soma, self.processes),
            "nucleus_records": (
                {
                    "instance_id": 101,
                    "accepted": True,
                    "dapi_valid": True,
                    "identity_status": "resolved",
                    "z_min_0based": None,
                    "z_max_0based_inclusive": None,
                    "owner_display_id": 1,
                },
                {
                    "instance_id": 202,
                    "accepted": True,
                    "dapi_valid": True,
                    "identity_status": "resolved",
                    "z_min_0based": None,
                    "z_max_0based_inclusive": None,
                    "owner_display_id": 2,
                },
            ),
            "analysis_mode": "gfap_only",
            "structural_channel": "GFAP",
        }

    def test_egfp_neonatal_round_trip_and_gfap_neonatal_rejection(self) -> None:
        egfp_paths = self.build(age_profile="neonatal")
        egfp_loaded = context.load_cell_edit_context(egfp_paths.json_path)
        self.assertEqual(egfp_loaded.metadata["analysis_mode"], "egfp")
        self.assertEqual(egfp_loaded.metadata["age_profile"], "neonatal")

        with self.assertRaisesRegex(
            context.CellEditContextError,
            "GFAP-only.*mature astrocytes only",
        ):
            context.build_cell_edit_context(
                **self.gfap_build_kwargs(age_profile="neonatal")
            )

        gfap_paths = context.build_cell_edit_context(
            **self.gfap_build_kwargs(age_profile="mature")
        )
        metadata = json.loads(gfap_paths.json_path.read_text(encoding="utf-8"))
        metadata["age_profile"] = "neonatal"
        content_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "content_sha256",
                "archive_size_bytes",
                "archive_sha256",
            }
        }
        metadata["content_sha256"] = context._content_sha256(content_metadata)
        gfap_paths.json_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "GFAP-only.*mature astrocytes only",
        ):
            context.load_cell_edit_context(gfap_paths.json_path)

    def test_round_trip_preserves_evidence_and_validates_sources(self) -> None:
        paths = self.build()
        loaded = context.load_cell_edit_context(
            paths.npz_path,
            verify_sources=True,
            verify_source_hashes=True,
        )
        self.assertEqual(
            loaded.metadata["schema"],
            context.CELL_EDIT_CONTEXT_SCHEMA,
        )
        self.assertEqual(loaded.metadata["selected_z"]["z_start_1based"], 3)
        self.assertEqual(loaded.metadata["age_profile"], "mature")
        self.assertEqual(loaded.metadata["analysis_mode"], "egfp")
        self.assertEqual(loaded.metadata["structural_channel"], "eGFP")
        self.assertEqual(
            loaded.metadata["calibration"]["pixel_depth_um"],
            0.25,
        )
        self.assertEqual(
            loaded.metadata["source_images"]["DAPI"]["size_bytes"],
            self.dapi_path.stat().st_size,
        )
        for name, expected in {
            "dapi_projection": self.dapi,
            "structural_map": self.structural,
            "canonical_nucleus_core_labels": self.core,
            "canonical_nucleus_extent_labels": self.extent,
            "whole_labels": self.whole,
            "soma_labels": self.soma,
            "process_labels": self.processes,
        }.items():
            self.assertTrue(np.array_equal(loaded.array(name), expected))
            self.assertFalse(loaded.array(name).flags.writeable)
        with self.assertRaises(ValueError):
            loaded.array("whole_labels")[0, 0] = 9

    def test_large_source_uses_bounded_sampled_sha256(self) -> None:
        record = context.source_file_fingerprint(
            self.dapi_path,
            full_hash_limit_bytes=1,
            sample_bytes=128,
        )
        self.assertEqual(
            record["sha256_strategy"],
            "sampled_sha256_v1:first-middle-last",
        )
        self.assertLessEqual(len(record["sample_offsets"]), 3)
        self.assertEqual(record["sample_bytes"], 128)
        self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(len(record["fingerprint_sha256"]), 64)

    def test_source_mutation_is_rejected_before_worker_use(self) -> None:
        paths = self.build()
        with self.dapi_path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "fingerprint changed",
        ):
            context.load_cell_edit_context(paths.json_path, verify_sources=True)

    def test_archive_tampering_is_rejected(self) -> None:
        paths = self.build()
        with np.load(paths.npz_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        arrays["dapi_projection"] = arrays["dapi_projection"].copy()
        arrays["dapi_projection"][0, 0] += 1
        with paths.npz_path.open("wb") as handle:
            np.savez(handle, **arrays)
        metadata = json.loads(paths.json_path.read_text(encoding="utf-8"))
        metadata["archive_size_bytes"] = paths.npz_path.stat().st_size
        metadata["archive_sha256"] = context._sha256_file(paths.npz_path)
        paths.json_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "dapi_projection",
        ):
            context.load_cell_edit_context(paths.npz_path)

    def test_metadata_tampering_is_rejected(self) -> None:
        paths = self.build()
        metadata = json.loads(paths.json_path.read_text(encoding="utf-8"))
        metadata["age_profile"] = "neonatal"
        paths.json_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "content integrity",
        ):
            context.load_cell_edit_context(paths.json_path)

    def test_nucleus_records_are_canonical_and_content_hashed(self) -> None:
        records = [
            {
                "instance_id": 101,
                "accepted": True,
                "dapi_valid": True,
                "identity_status": "resolved",
                "z_min_0based": 2,
                "z_max_0based_inclusive": 7,
            },
            {
                "instance_id": 202,
                "accepted": False,
                "dapi_valid": True,
                "identity_status": "ambiguous",
                "z_min_0based": None,
                "z_max_0based_inclusive": None,
            },
        ]
        paths = context.build_cell_edit_context(
            run_dir=self.root / "records",
            dapi_path=self.dapi_path,
            structural_paths={"eGFP": self.egfp_path},
            dapi_projection=self.dapi,
            structural_map=self.structural,
            selected_z=(3, 11),
            calibration={
                "pixel_width_um": 0.09,
                "pixel_height_um": 0.091,
                "pixel_depth_um": 0.25,
            },
            age_profile="mature",
            canonical_core_labels=self.core,
            canonical_extent_labels=self.extent,
            initial_triplet=(self.whole, self.soma, self.processes),
            nucleus_records=records,
        )
        loaded = context.load_cell_edit_context(paths.json_path)
        self.assertEqual(loaded.metadata["nucleus_records"], records)
        metadata = json.loads(paths.json_path.read_text(encoding="utf-8"))
        metadata["nucleus_records"][0]["accepted"] = False
        paths.json_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "content integrity",
        ):
            context.load_cell_edit_context(paths.json_path)

    def test_unknown_z_is_allowed_but_partial_or_out_of_range_z_is_rejected(self):
        paths = self.build()
        loaded = context.load_cell_edit_context(paths.json_path)
        self.assertTrue(
            all(
                record["z_min_0based"] is None
                and record["z_max_0based_inclusive"] is None
                for record in loaded.metadata["nucleus_records"]
            )
        )
        base_record = {
            "instance_id": 101,
            "accepted": True,
            "dapi_valid": True,
            "identity_status": "resolved",
            "z_min_0based": None,
            "z_max_0based_inclusive": 4,
        }
        second_record = {
            "instance_id": 202,
            "accepted": True,
            "dapi_valid": True,
            "identity_status": "resolved",
            "z_min_0based": None,
            "z_max_0based_inclusive": None,
        }
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "both Z bounds",
        ):
            context.build_cell_edit_context(
                run_dir=self.root / "partial-z",
                dapi_path=self.dapi_path,
                structural_paths={"eGFP": self.egfp_path},
                dapi_projection=self.dapi,
                structural_map=self.structural,
                selected_z=(3, 11),
                calibration={
                    "pixel_width_um": 0.09,
                    "pixel_height_um": 0.091,
                    "pixel_depth_um": 0.25,
                },
                age_profile="mature",
                canonical_core_labels=self.core,
                canonical_extent_labels=self.extent,
                initial_triplet=(self.whole, self.soma, self.processes),
                nucleus_records=(base_record, second_record),
            )

    def test_final_named_context_passes_full_committed_validation(self) -> None:
        paths = context.build_cell_edit_context(
            run_dir=self.root / "fiji-run" / "cell_edit",
            basename="analysis_context",
            dapi_path=self.dapi_path,
            structural_paths={"eGFP": self.egfp_path},
            dapi_projection=self.dapi,
            structural_map=self.structural,
            selected_z=(3, 11),
            calibration={
                "pixel_width_um": 0.09,
                "pixel_height_um": 0.091,
                "pixel_depth_um": 0.25,
            },
            age_profile="mature",
            canonical_core_labels=self.core,
            canonical_extent_labels=self.extent,
            initial_triplet=(self.whole, self.soma, self.processes),
        )
        self.assertEqual(paths.npz_path.name, "analysis_context.npz")
        self.assertEqual(paths.json_path.name, "analysis_context.json")
        loaded = context.load_cell_edit_context(
            paths.json_path,
            verify_sources=True,
            verify_source_hashes=True,
        )
        self.assertEqual(
            loaded.metadata["archive_file"],
            "analysis_context.npz",
        )

    def test_relocation_compatibility_api_rehashes_and_removes_staged_pair(self) -> None:
        paths = self.build()
        staged_npz = paths.npz_path
        staged_json = paths.json_path
        relocated = context.relocate_cell_edit_context(
            staged_json,
            destination_dir=self.root / "compatibility-run" / "cell_edit",
        )
        self.assertFalse(staged_npz.exists())
        self.assertFalse(staged_json.exists())
        self.assertEqual(relocated.npz_path.name, "analysis_context.npz")
        self.assertEqual(relocated.json_path.name, "analysis_context.json")
        loaded = context.load_cell_edit_context(relocated.json_path)
        self.assertEqual(loaded.metadata["archive_file"], "analysis_context.npz")

    def test_invalid_partition_and_core_extent_are_rejected(self) -> None:
        broken_processes = self.processes.copy()
        broken_processes[self.soma == 1] = 1
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "partition",
        ):
            context.build_cell_edit_context(
                run_dir=self.root / "broken-partition",
                dapi_path=self.dapi_path,
                structural_paths={"eGFP": self.egfp_path},
                dapi_projection=self.dapi,
                structural_map=self.structural,
                selected_z=(1, 2),
                calibration={
                    "pixel_width_um": 0.09,
                    "pixel_height_um": 0.09,
                    "pixel_depth_um": 0.25,
                },
                age_profile="mature",
                canonical_core_labels=self.core,
                canonical_extent_labels=self.extent,
                initial_triplet=(self.whole, self.soma, broken_processes),
            )

        broken_extent = self.extent.copy()
        broken_extent[self.core == 101] = 202
        with self.assertRaisesRegex(
            context.CellEditContextError,
            "same nucleus extent",
        ):
            context.build_cell_edit_context(
                run_dir=self.root / "broken-core",
                dapi_path=self.dapi_path,
                structural_paths={"eGFP": self.egfp_path},
                dapi_projection=self.dapi,
                structural_map=self.structural,
                selected_z=(1, 2),
                calibration={
                    "pixel_width_um": 0.09,
                    "pixel_height_um": 0.09,
                    "pixel_depth_um": 0.25,
                },
                age_profile="mature",
                canonical_core_labels=self.core,
                canonical_extent_labels=broken_extent,
                initial_triplet=(self.whole, self.soma, self.processes),
            )

    def test_rejects_shape_dtype_calibration_and_path_traversal(self) -> None:
        with self.assertRaisesRegex(context.CellEditContextError, "same YX shape"):
            context.build_cell_edit_context(
                run_dir=self.root / "wrong-shape",
                dapi_path=self.dapi_path,
                structural_paths={"eGFP": self.egfp_path},
                dapi_projection=self.dapi[:-1],
                structural_map=self.structural,
                selected_z=(1, 2),
                calibration={
                    "pixel_width_um": 0.09,
                    "pixel_height_um": 0.09,
                    "pixel_depth_um": 0.25,
                },
                age_profile="mature",
                canonical_core_labels=self.core,
                canonical_extent_labels=self.extent,
                initial_triplet=(self.whole, self.soma, self.processes),
            )
        with self.assertRaisesRegex(context.CellEditContextError, "integer dtype"):
            context.build_cell_edit_context(
                run_dir=self.root / "wrong-dtype",
                dapi_path=self.dapi_path,
                structural_paths={"eGFP": self.egfp_path},
                dapi_projection=self.dapi,
                structural_map=self.structural,
                selected_z=(1, 2),
                calibration={
                    "pixel_width_um": 0.09,
                    "pixel_height_um": 0.09,
                    "pixel_depth_um": 0.25,
                },
                age_profile="mature",
                canonical_core_labels=self.core.astype(np.float32),
                canonical_extent_labels=self.extent,
                initial_triplet=(self.whole, self.soma, self.processes),
            )
        with self.assertRaisesRegex(context.CellEditContextError, "pixel_width_um"):
            context.build_cell_edit_context(
                run_dir=self.root / "wrong-calibration",
                dapi_path=self.dapi_path,
                structural_paths={"eGFP": self.egfp_path},
                dapi_projection=self.dapi,
                structural_map=self.structural,
                selected_z=(1, 2),
                calibration={
                    "pixel_width_um": 0,
                    "pixel_height_um": 0.09,
                    "pixel_depth_um": 0.25,
                },
                age_profile="mature",
                canonical_core_labels=self.core,
                canonical_extent_labels=self.extent,
                initial_triplet=(self.whole, self.soma, self.processes),
            )
        paths = self.build()
        metadata = json.loads(paths.json_path.read_text(encoding="utf-8"))
        metadata["archive_file"] = "../outside.npz"
        paths.json_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(context.CellEditContextError, "local NPZ"):
            context.load_cell_edit_context(paths.json_path)

    def test_commit_has_no_temporary_files_and_does_not_copy_tiffs(self) -> None:
        paths = self.build()
        run_dir = paths.npz_path.parent
        self.assertEqual(
            sorted(path.name for path in run_dir.iterdir()),
            ["cell_edit_context.json", "cell_edit_context.npz"],
        )
        with np.load(paths.npz_path, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(context._ARRAY_NAMES))
        metadata_text = paths.json_path.read_text(encoding="utf-8")
        self.assertIn(str(self.dapi_path.resolve()), metadata_text)
        self.assertFalse(any(path.suffix.lower() in {".tif", ".tiff"} for path in run_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
