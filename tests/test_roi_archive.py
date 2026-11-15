from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from project_leap_2d.final_roi_archive import ARCHIVE_NAME, FinalRoiArchiveError, build_final_roi_archive
from project_leap_2d.workspace import publication_recovery as recovery


KEYS = ("whole", "processes", "soma", "report", "workbook", "roi_archive")
COMPARTMENTS = ("whole", "soma", "processes")
SUFFIX = {"whole": "Whole", "soma": "Soma", "processes": "Processes"}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fingerprint(path):
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fixture(root):
    final = root / "final_roi"
    final.mkdir()
    write_json(root / "runtime_provenance.json", {
        "schema_version": 1, "request_provenance": {"source": "/read-only/source.tif"},
        "engine": {"version": "test-fixture"},
        "runtime": {"descriptor": {"component_version": "test-fixture"}},
        "models": {"hash_source": "synthetic fixture; no model loaded"},
    })
    records = [
        {"label_id": 1, "original_id": 1, "cell_uid": "cell-000001", "parent_uid": "",
         "lineage": [1], "owner_nucleus_id": "nucleus-1"},
        {"label_id": 2, "original_id": 3, "cell_uid": "cell-000003", "parent_uid": "",
         "lineage": [2, 3], "owner_nucleus_id": "nucleus-3"},
    ]
    source = {"expected_shape": [5, 6, 8], "z_start_1based": 2,
              "z_end_1based_inclusive": 4, "projection": "max",
              "pixel_width_um": 0.25, "pixel_height_um": 0.5,
              "channels": {"DAPI": "/read-only/source.tif"}, "measurement_channel": "DAPI"}
    write_json(root / "manifest.json", source)
    files, geometry = {}, {}
    with zipfile.ZipFile(final / "native_rois.zip", "w") as native:
        for key in COMPARTMENTS:
            labels = np.zeros((6, 8), dtype=np.uint16)
            shapes = []
            for record, (x, y) in zip(records, ((1, 1), (4, 3))):
                if key == "processes":
                    x += 1
                width = 2 if key == "whole" else 1
                labels[y:y+2, x:x+width] = record["label_id"]
                name = f"Astrocyte_{record['label_id']:03d}_{SUFFIX[key]}"
                shapes.append({"label_id": record["label_id"], "cell_uid": record["cell_uid"],
                               "roi_name": name, "bounds_xywh": [x, y, width, 2]})
                roi = bytearray(64)
                struct.pack_into(">4sHBBhhhh", roi, 0, b"Iout", 228, 1, 0, y, x, y+2, x+width)
                native.writestr(name + ".roi", roi)
            geometry[key] = shapes
            path = final / f"{key}_labels.tif"
            tifffile.imwrite(path, labels, photometric="minisblack", metadata={"axes": "YX"})
            files[key] = {"file": path.name, **fingerprint(path)}
    files["native_rois"] = {"file": "native_rois.zip", **fingerprint(final / "native_rois.zip")}
    export = {"schema_version": 1, "roi_count": 2, "identity_records": records,
              "geometry_records": geometry, "files": files,
              "source_shape_zyx": [5, 6, 8], "projection": "max",
              "selected_z": {"start_1based": 2, "end_1based_inclusive": 4},
              "coordinates": {"axes": "YX", "index_origin": 0, "origin": "top-left",
                              "x_direction": "right", "y_direction": "down", "source_xy_offset": [0, 0]},
              "calibration": {"pixel_width_um": 0.25, "pixel_height_um": 0.5},
              "source_channels": source["channels"], "measurement_channel": "DAPI",
              "source_image_calibrations": {"DAPI": {"pixel_width": .25, "pixel_height": .5,
                  "pixel_depth": 1.5, "x_origin": 0, "y_origin": 0, "z_origin": 0,
                  "x_unit": "um", "y_unit": "um", "z_unit": "um"}}}
    write_json(final / "export.json", export)
    result = {"rows": 2, "astrocyte_ids": [1, 2], "cell_uids": [r["cell_uid"] for r in records],
              "original_astrocyte_ids": [1, 3], "source_original_astrocyte_ids": [[1], [2, 3]],
              "parent_cell_uids": ["", ""], "owner_nucleus_ids": ["nucleus-1", "nucleus-3"]}
    done = {"cancelled": False, "roi_count": 2, "final_roi_ids": [1, 2],
            "final_soma_ids": [1, 2], "final_process_ids": [1, 2],
            "final_cell_uids": result["cell_uids"], "original_to_final_id": {"1": 1, "3": 2},
            "source_to_final_ids": {"1": [1], "2": [2], "3": [2]},
            "result_sets": {k: result for k in COMPARTMENTS},
            "final_roi_export": {"schema_version": 1, "manifest": "final_roi/export.json",
                                 "sha256": fingerprint(final / "export.json")["sha256"]}}
    write_json(root / "fiji_done.json", done)
    return export, done


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.export, self.done = fixture(self.root)
        self.destination = self.root / ARCHIVE_NAME

    def tearDown(self):
        self.temp.cleanup()

    def bind_export(self):
        write_json(self.root / "final_roi/export.json", self.export)
        self.done["final_roi_export"]["sha256"] = fingerprint(self.root / "final_roi/export.json")["sha256"]
        write_json(self.root / "fiji_done.json", self.done)

    def test_round_trip_preserves_final_bytes_and_identity(self):
        before = {p: fingerprint(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(build_final_roi_archive(self.root, self.destination), self.destination)
        with zipfile.ZipFile(self.destination) as archive:
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("archive_manifest.json"))
            self.assertEqual(manifest["roi_count"], 2)
            for name, identity in manifest["files"].items():
                data = archive.read(name)
                self.assertEqual(hashlib.sha256(data).hexdigest(), identity["sha256"])
            self.assertEqual(json.loads(archive.read("final_roi/export.json")), self.export)
        self.assertEqual(before, {p: fingerprint(p) for p in before})

    def test_initial_masks_are_never_used_as_fallback(self):
        shutil.copy2(self.root / "final_roi/whole_labels.tif", self.root / "whole_roi_labels.tif")
        (self.root / "final_roi/whole_labels.tif").unlink()
        with self.assertRaisesRegex(FinalRoiArchiveError, "missing"):
            build_final_roi_archive(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_runtime_provenance_is_preserved_without_following_paths(self):
        provenance = {"schema_version": 1, "request_provenance": {"source": "/never/read/me"},
                      "engine": {"actual_resource_sha256": "test fingerprint"},
                      "runtime": {"descriptor": {"component_version": "prototype"}},
                      "models": {"hash_source": "installed manifest"}}
        write_json(self.root / "runtime_provenance.json", provenance)
        build_final_roi_archive(self.root, self.destination)
        with zipfile.ZipFile(self.destination) as archive:
            self.assertEqual(json.loads(archive.read("runtime_provenance.json")), provenance)
            self.assertEqual(json.loads(archive.read("archive_manifest.json"))["provenance"],
                             "runtime_provenance.json")

    def test_fiji_rectangle_double_bounds_preserve_exact_geometry(self):
        for shapes in self.export["geometry_records"].values():
            for shape in shapes:
                shape["bounds_xywh"] = [float(value) for value in shape["bounds_xywh"]]
        self.bind_export()
        before = {p: fingerprint(p) for p in (self.root / "final_roi").iterdir()}
        build_final_roi_archive(self.root, self.destination)
        self.assertEqual(before, {p: fingerprint(p) for p in before})
        with zipfile.ZipFile(self.destination) as archive:
            self.assertEqual(json.loads(archive.read("final_roi/export.json")), self.export)

    def test_fractional_nonfinite_and_boolean_pixel_bounds_rejected(self):
        for invalid in (1.5, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                self.export["geometry_records"]["whole"][0]["bounds_xywh"][0] = invalid
                self.bind_export()
                with self.assertRaisesRegex(FinalRoiArchiveError, "Invalid final whole ROI bounds"):
                    build_final_roi_archive(self.root, self.destination)

    def test_unsupported_provenance_schema_is_rejected(self):
        write_json(self.root / "runtime_provenance.json", {"schema_version": 2})
        with self.assertRaisesRegex(FinalRoiArchiveError, "provenance schema"):
            build_final_roi_archive(self.root, self.destination)

    def test_missing_provenance_prevents_formal_archive(self):
        (self.root / "runtime_provenance.json").unlink()
        with self.assertRaisesRegex(FinalRoiArchiveError, "missing: runtime_provenance.json"):
            build_final_roi_archive(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_incomplete_provenance_is_rejected(self):
        write_json(self.root / "runtime_provenance.json", {"schema_version": 1})
        with self.assertRaisesRegex(FinalRoiArchiveError, "incomplete runtime provenance"):
            build_final_roi_archive(self.root, self.destination)

    def test_export_manifest_hash_disagreement_rejected(self):
        self.export["roi_count"] = 1
        write_json(self.root / "final_roi/export.json", self.export)
        with self.assertRaisesRegex(FinalRoiArchiveError, "manifest hash differs"):
            build_final_roi_archive(self.root, self.destination)

    def test_source_changed_after_validation_cannot_be_published(self):
        original_write = zipfile.ZipFile.write
        def changed_write(archive, filename, arcname, *args, **kwargs):
            if arcname == "source_manifest.json":
                Path(filename).write_text("{}")
            return original_write(archive, filename, arcname, *args, **kwargs)
        with patch.object(zipfile.ZipFile, "write", changed_write):
            with self.assertRaisesRegex(FinalRoiArchiveError, "changed while archiving"):
                build_final_roi_archive(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_cancelled_run_cannot_archive(self):
        self.done["cancelled"] = True
        write_json(self.root / "fiji_done.json", self.done)
        with self.assertRaisesRegex(FinalRoiArchiveError, "not completed"):
            build_final_roi_archive(self.root, self.destination)

    def test_modified_geometry_keeps_existing_destination(self):
        self.destination.write_bytes(b"previous completed archive")
        (self.root / "final_roi/soma_labels.tif").write_bytes(b"truncated")
        with self.assertRaisesRegex(FinalRoiArchiveError, "identity differs"):
            build_final_roi_archive(self.root, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"previous completed archive")

    def test_measurement_identity_disagreement_rejected(self):
        self.done["result_sets"]["whole"]["cell_uids"] = ["wrong", "cell-000003"]
        write_json(self.root / "fiji_done.json", self.done)
        with self.assertRaisesRegex(FinalRoiArchiveError, "differs from measured whole"):
            build_final_roi_archive(self.root, self.destination)

    def test_mask_missing_final_label_rejected_even_when_hash_matches(self):
        path = self.root / "final_roi/processes_labels.tif"
        tifffile.imwrite(path, np.zeros((6, 8), dtype=np.uint16))
        self.export["files"]["processes"].update(fingerprint(path))
        self.bind_export()
        with self.assertRaisesRegex(FinalRoiArchiveError, "mask IDs differ"):
            build_final_roi_archive(self.root, self.destination)

    def test_native_roi_identity_disagreement_rejected(self):
        path = self.root / "final_roi/native_rois.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("initial_unreviewed.roi", b"Iout" + bytes(60))
        self.export["files"]["native_rois"].update(fingerprint(path))
        self.bind_export()
        with self.assertRaisesRegex(FinalRoiArchiveError, "Native ROI ZIP identities"):
            build_final_roi_archive(self.root, self.destination)

    def test_z_and_calibration_disagreement_rejected(self):
        for field, replacement in (("selected_z", {"start_1based": 1, "end_1based_inclusive": 4}),
                                   ("calibration", {"pixel_width_um": .1, "pixel_height_um": .5})):
            with self.subTest(field=field):
                previous = self.export[field]
                self.export[field] = replacement
                self.bind_export()
                with self.assertRaises(FinalRoiArchiveError):
                    build_final_roi_archive(self.root, self.destination)
                self.export[field] = previous

    def test_symlink_export_rejected(self):
        path = self.root / "final_roi/whole_labels.tif"
        target = self.root / "copy.tif"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(FinalRoiArchiveError, "symlink"):
            build_final_roi_archive(self.root, self.destination)

    def test_write_failure_leaves_no_partial_archive(self):
        with patch("project_leap_2d.final_roi_archive.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                build_final_roi_archive(self.root, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run, self.result, self.runtime = (self.root / name for name in ("run", "Result", "Run State"))
        for path in (self.run, self.result, self.runtime):
            path.mkdir()
        self.staged = {key: self.run / (ARCHIVE_NAME if key == "roi_archive" else key) for key in KEYS}
        self.final = {key: self.result / path.name for key, path in self.staged.items()}
        for key in KEYS:
            self.staged[key].write_bytes(("new:" + key).encode())
            self.final[key].write_bytes(("old:" + key).encode())

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, callback):
        return recovery.publish_with_publication_journal(
            publish_callback=callback, call_args=(),
            call_kwargs={"staged_files": self.staged, "final_files": self.final, "run_dir": self.run},
            result_root=self.result, runtime_root=self.runtime)

    @staticmethod
    def success(*, staged_files, final_files, run_dir):
        for key in staged_files:
            shutil.copy2(staged_files[key], final_files[key])

    def assert_previous(self):
        for key in KEYS:
            self.assertEqual(self.final[key].read_bytes(), ("old:" + key).encode())

    def test_six_outputs_commit_together(self):
        self.publish(self.success)
        for key in KEYS:
            self.assertEqual(self.final[key].read_bytes(), ("new:" + key).encode())
        self.assertFalse(recovery.publication_journal_path(self.runtime).exists())

    def test_failure_after_each_replace_restores_all_six(self):
        for failed_at in range(1, 7):
            with self.subTest(replaced=failed_at):
                def broken(*, staged_files, final_files, run_dir):
                    for index, key in enumerate(staged_files, 1):
                        shutil.copy2(staged_files[key], final_files[key])
                        if index == failed_at:
                            raise OSError("injected publication failure")
                with self.assertRaisesRegex(OSError, "injected"):
                    self.publish(broken)
                self.assert_previous()

    def test_sixth_temporary_removed_after_failure(self):
        token = "e" * 32
        def broken(*, staged_files, final_files, run_dir):
            for key in KEYS:
                (self.result / f"temporary_IHC_{token}_{key}.tmp").write_text("partial")
            raise SystemExit("interrupted")
        with self.assertRaises(SystemExit):
            self.publish(broken)
        self.assert_previous()
        self.assertEqual(list(self.result.glob("temporary_IHC_*")), [])

    def test_missing_archive_is_rejected_before_any_output_changes(self):
        self.staged.pop("roi_archive")
        self.final.pop("roi_archive")
        with self.assertRaisesRegex(ValueError, "roi_archive"):
            self.publish(self.success)
        self.assertFalse(recovery.publication_journal_path(self.runtime).exists())

    def test_interrupted_recovery_replays_six_entries(self):
        original_recover = recovery.recover_publication_transaction
        def broken(*, staged_files, final_files, run_dir):
            for key in KEYS[:3]:
                shutil.copy2(staged_files[key], final_files[key])
            raise OSError("publish interrupted")
        with patch.object(recovery, "recover_publication_transaction", side_effect=OSError("recovery interrupted")):
            with self.assertRaisesRegex(RuntimeError, "journal retained"):
                self.publish(broken)
        journal = json.loads(recovery.publication_journal_path(self.runtime).read_text())
        self.assertEqual(journal["schema_version"], 3)
        self.assertEqual({entry["key"] for entry in journal["entries"]}, set(KEYS))
        self.assertTrue(original_recover(result_root=self.result, runtime_root=self.runtime))
        self.assert_previous()

    def test_first_publication_failure_removes_partial_six_bundle(self):
        for path in self.final.values():
            path.unlink()
        def broken(*, staged_files, final_files, run_dir):
            self.success(staged_files=staged_files, final_files=final_files, run_dir=run_dir)
            raise OSError("failure after sixth output")
        with self.assertRaises(OSError):
            self.publish(broken)
        self.assertFalse(any(path.exists() for path in self.final.values()))


if __name__ == "__main__":
    unittest.main()
