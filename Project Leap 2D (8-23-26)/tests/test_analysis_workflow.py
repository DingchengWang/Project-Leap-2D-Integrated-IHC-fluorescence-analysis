from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from skimage import morphology, segmentation

from project_leap_2d import analysis_workflow as controller


class AnalysisWorkflowTests(unittest.TestCase):
    def test_analysis_route_is_egfp_first_and_gfap_only_is_explicit(self):
        self.assertEqual(
            controller.select_analysis_route(("DAPI", "eGFP", "GFAP")),
            "egfp",
        )
        self.assertEqual(
            controller.select_analysis_route(("DAPI", "eGFP")),
            "egfp",
        )
        self.assertEqual(
            controller.select_analysis_route(("DAPI", "GFAP")),
            "gfap_only",
        )
        with self.assertRaisesRegex(ValueError, "eGFP or GFAP"):
            controller.select_analysis_route(("DAPI", "KCNN2"))

    def test_gfap_only_no_age_token_and_mature_token_run_mature_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            class Runtime:
                tf = SimpleNamespace(imread=lambda *_args, **_kwargs: None)

                def __init__(self, age_decision):
                    self.calls = []
                    self.age_decision = age_decision

                def detect_filename_age_profile(self, _paths):
                    self.calls.append("detect_filename_age_profile")
                    return self.age_decision

                def parse_args(self, _argv):
                    self.calls.append("parse_args")
                    return SimpleNamespace(
                        fiji_timeout_minutes=1.0,
                        dapi_fragment_workload_preflight_only=False,
                        input_dir=root,
                        output_dir=root / "output",
                        skip_fiji=True,
                    )

                def read_meta(self, _path):
                    self.calls.append("read_meta")
                    return {
                        "shape": (3, 8, 8),
                        "axes": "ZYX",
                        "pixel_width_um": 0.2,
                        "pixel_height_um": 0.2,
                        "pixel_depth_um": 0.5,
                    }

                def validate_shared_geometry(self, _metadata):
                    self.calls.append("validate_shared_geometry")

                def measurement_channel(self, _paths):
                    return "GFAP"

                def load_stack(self, _path):
                    self.calls.append("load_stack")
                    return np.zeros((3, 8, 8), dtype=np.uint8)

                def print_terminal_stage(self, *_args):
                    return None

                @staticmethod
                def project(*_args):
                    raise AssertionError("measurement loader was not requested")

            cases = (
                ("no-token", None),
                ("mature-token", SimpleNamespace(profile="mature")),
            )
            paths = {
                "DAPI": root / "DAPI.tif",
                "GFAP": root / "GFAP.tif",
            }
            for label, age_decision in cases:
                with self.subTest(label=label):
                    runtime = Runtime(age_decision)
                    with mock.patch(
                        "project_leap_2d.analysis_modes.gfap_only."
                        "gfap_only_pipeline.run_gfap_only_pipeline",
                        return_value=SimpleNamespace(),
                    ) as run_pipeline:
                        self.assertEqual(
                            controller._run_gfap_only(runtime, [], paths),
                            0,
                        )
                    run_pipeline.assert_called_once()
                    self.assertNotIn(
                        "age_profile",
                        run_pipeline.call_args.kwargs,
                    )
                    self.assertEqual(runtime.calls.count("load_stack"), 2)

    def test_gfap_only_neonatal_token_stops_before_io_or_model_pipeline(self):
        calls = []

        class Runtime:
            @staticmethod
            def detect_filename_age_profile(_paths):
                calls.append("detect_filename_age_profile")
                return SimpleNamespace(profile="neonatal")

            @staticmethod
            def parse_args(_argv):
                calls.append("parse_args")
                raise AssertionError("argument parsing must not run")

            @staticmethod
            def read_meta(_path):
                calls.append("read_meta")
                raise AssertionError("metadata loading must not run")

            @staticmethod
            def load_stack(_path):
                calls.append("load_stack")
                raise AssertionError("image loading must not run")

        paths = {
            "DAPI": Path("DAPI_neonatal.tif"),
            "GFAP": Path("GFAP_neonatal.tif"),
        }
        with mock.patch(
            "project_leap_2d.analysis_modes.gfap_only."
            "gfap_only_pipeline.run_gfap_only_pipeline",
        ) as run_pipeline:
            with self.assertRaisesRegex(
                ValueError,
                "mature astrocytes only",
            ):
                controller._run_gfap_only(Runtime(), [], paths)
        self.assertEqual(calls, ["detect_filename_age_profile"])
        run_pipeline.assert_not_called()

    def test_authoritative_canonical_inventory_records_are_preserved(self):
        record = {
            "instance_id": 41,
            "accepted": True,
            "dapi_valid": True,
            "identity_status": "resolved",
            "z_min_0based": 7,
            "z_max_0based_inclusive": 13,
        }
        metrics = {
            "nucleus_3d_inventory": {
                "canonical_per_nucleus": [record],
            }
        }
        labels = np.zeros((8, 9), dtype=np.uint32)
        labels[2:5, 3:6] = 41
        observed = controller._canonical_nucleus_records(
            metrics,
            labels,
            labels,
        )
        self.assertEqual(observed, (record,))
        self.assertIsNot(observed[0], record)

    def test_projection_fallback_does_not_invent_z_or_acceptance(self):
        core = np.zeros((8, 9), dtype=np.uint32)
        extent = np.zeros_like(core)
        core[3:5, 4:6] = 12
        extent[2:6, 3:7] = 12
        observed = controller._canonical_nucleus_records(
            {},
            core,
            extent,
        )
        self.assertEqual(len(observed), 1)
        record = observed[0]
        self.assertEqual(record["instance_id"], 12)
        self.assertTrue(record["dapi_valid"])
        self.assertFalse(record["accepted"])
        self.assertEqual(record["identity_status"], "projection_only")
        self.assertIsNone(record["z_min_0based"])
        self.assertIsNone(record["z_max_0based_inclusive"])

    def test_cleanup_removes_only_current_context_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi_path = root / "DAPI.tif"
            structural_path = root / "eGFP.tif"
            dapi_path.write_bytes(b"dapi")
            structural_path.write_bytes(b"structural")

            run_dir = root / "run"
            runtime_context = run_dir / "cell_edit"
            runtime_context.mkdir(parents=True)
            (runtime_context / "analysis_context.npz").write_bytes(
                b"runtime-context"
            )
            (runtime_context / "analysis_context.json").write_text(
                "{}",
                encoding="utf-8",
            )
            npz_path = runtime_context / "analysis_context.npz"
            json_path = runtime_context / "analysis_context.json"
            sentinel = runtime_context / "requests"
            sentinel.mkdir()

            controller._cleanup_context_artifacts(
                controller._CellEditCapture(
                    context_paths=SimpleNamespace(
                        npz_path=npz_path,
                        json_path=json_path,
                    ),
                    run_dir=run_dir,
                ),
                remove_runtime_context=True,
            )

            self.assertFalse(npz_path.exists())
            self.assertFalse(json_path.exists())
            self.assertFalse((runtime_context / "analysis_context.npz").exists())
            self.assertFalse((runtime_context / "analysis_context.json").exists())
            self.assertTrue(dapi_path.is_file())
            self.assertTrue(structural_path.is_file())
            self.assertTrue(sentinel.is_dir())

    def test_egfp_adapter_restores_frozen_runtime_functions_on_failure(self):
        def split(*args, **kwargs):
            raise AssertionError("not reached")

        def prepare(*args, **kwargs):
            raise AssertionError("not reached")

        def launch(*args, **kwargs):
            raise AssertionError("not reached")

        class Runtime:
            split_astrocyte_compartments_for_profile = staticmethod(split)
            prepare_fiji_runtime = staticmethod(prepare)
            launch_fiji_workflow = staticmethod(launch)

            @staticmethod
            def main(_argv):
                raise RuntimeError("simulated analysis failure")

        runtime = Runtime()
        with mock.patch.object(
            controller,
            "_cleanup_context_artifacts",
        ) as cleanup:
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                controller._run_egfp_structural(runtime, ["--input-dir", "/tmp"])
        cleanup.assert_called_once()
        self.assertFalse(cleanup.call_args.kwargs["remove_runtime_context"])
        self.assertIs(
            runtime.split_astrocyte_compartments_for_profile,
            split,
        )
        self.assertIs(runtime.prepare_fiji_runtime, prepare)
        self.assertIs(runtime.launch_fiji_workflow, launch)

    def test_context_is_built_directly_in_final_runtime_and_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi_path = root / "DAPI.tif"
            egfp_path = root / "eGFP.tif"
            dapi_path.write_bytes(b"dapi-source")
            egfp_path.write_bytes(b"egfp-source")
            shape = (12, 14)
            whole = np.ones(shape, dtype=np.uint16)
            soma = np.zeros(shape, dtype=np.uint16)
            soma[3:9, 4:10] = 1
            processes = whole.copy()
            processes[soma > 0] = 0
            core = np.zeros(shape, dtype=np.uint32)
            extent = np.zeros(shape, dtype=np.uint32)
            core[5:7, 6:8] = 7
            extent[4:8, 5:9] = 7
            capture = controller._CellEditCapture(
                dapi_projection=np.arange(
                    shape[0] * shape[1], dtype=np.uint16
                ).reshape(shape),
                structural_map=np.linspace(
                    0.0,
                    1.0,
                    shape[0] * shape[1],
                    dtype=np.float32,
                ).reshape(shape),
                whole_labels=whole,
                soma_labels=soma,
                process_labels=processes,
                compartment_metrics={
                    "_canonical_nucleus_instance_core_labels_2d": core,
                    "_canonical_nucleus_instance_extent_labels_2d": extent,
                    "nucleus_3d_inventory": {
                        "canonical_per_nucleus": [
                            {
                                "instance_id": 7,
                                "accepted": True,
                                "dapi_valid": True,
                                "identity_status": "resolved",
                                "z_min_0based": 2,
                                "z_max_0based_inclusive": 5,
                            }
                        ]
                    },
                },
                pixel_width_um=0.11,
                pixel_height_um=0.12,
                pixel_depth_um=0.4,
                age_profile="mature",
            )

            def base_prepare(**_kwargs):
                run_dir = root / "cache" / f"run-{'a' * 32}"
                run_dir.mkdir(parents=True)
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text("{}", encoding="utf-8")
                return run_dir, manifest_path

            with mock.patch(
                "project_leap_2d.fiji_review.cell_edit_fiji_bridge._fiji_cache_root",
                return_value=(root / "cache").resolve(),
            ):
                run_dir, manifest_path = controller._build_context_and_prepare_fiji(
                    capture=capture,
                    base_prepare=base_prepare,
                    prepare_kwargs={
                        "paths": {"DAPI": dapi_path, "eGFP": egfp_path},
                        "metadata": {
                            "DAPI": {
                                "pixel_depth_um": 0.4,
                                "pixel_width_source": "OME",
                                "pixel_height_source": "OME",
                                "pixel_depth_source": "OME",
                            }
                        },
                        "structural_channels": ["eGFP"],
                        "best_row": {
                            "z_start_1based": 3,
                            "z_end_1based_inclusive": 6,
                            "projection": "max",
                        },
                        "output_dir": root,
                        "measurement": "KCNN2",
                        "whole_labels": whole,
                        "soma_labels": soma,
                        "process_labels": processes,
                        "selected_projections": {},
                        "auto_continue": True,
                    },
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime_context = Path(manifest["cell_edit"]["context_path"])
            self.assertEqual(runtime_context.name, "analysis_context.json")
            self.assertTrue(runtime_context.is_file())
            self.assertTrue(runtime_context.with_suffix(".npz").is_file())
            self.assertEqual(Path(run_dir), capture.run_dir)
            self.assertEqual(
                capture.context_paths.npz_path.resolve(),
                runtime_context.with_suffix(".npz").resolve(),
            )
            self.assertEqual(
                capture.context_paths.json_path.resolve(),
                runtime_context.resolve(),
            )
            self.assertIsNone(capture.dapi_projection)
            controller._cleanup_context_artifacts(
                capture,
                remove_runtime_context=True,
            )
            self.assertFalse(runtime_context.exists())
            self.assertFalse(runtime_context.with_suffix(".npz").exists())
            self.assertTrue(dapi_path.is_file())
            self.assertTrue(egfp_path.is_file())

    def test_failed_run_keeps_final_manifest_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "failed-run"
            runtime_root = run_dir / "cell_edit"
            runtime_root.mkdir(parents=True)
            runtime_npz = runtime_root / "analysis_context.npz"
            runtime_json = runtime_root / "analysis_context.json"
            runtime_npz.write_bytes(b"diagnostic-context")
            runtime_json.write_text("{}", encoding="utf-8")
            manifest = run_dir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cell_edit": {
                            "context_path": str(runtime_json),
                        }
                    }
                ),
                encoding="utf-8",
            )
            capture = controller._CellEditCapture(
                context_paths=SimpleNamespace(
                    npz_path=runtime_npz,
                    json_path=runtime_json,
                ),
                run_dir=run_dir,
            )
            controller._cleanup_context_artifacts(
                capture,
                remove_runtime_context=False,
            )
            self.assertTrue(runtime_npz.is_file())
            self.assertTrue(runtime_json.is_file())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                Path(manifest_data["cell_edit"]["context_path"]),
                runtime_json,
            )

    def test_gfap_owner_records_map_source_ids_and_selected_z(self):
        valid_projection = np.zeros((12, 14), dtype=np.uint32)
        valid_projection[2:5, 2:5] = 81
        valid_projection[6:9, 5:8] = 104
        valid_projection[3:6, 10:13] = 150
        prepared = SimpleNamespace(
            z_selection=SimpleNamespace(start_0based=12),
            analysis=SimpleNamespace(
                valid_nucleus_labels_2d=valid_projection,
                diagnostics={
                    "source_owner_to_display_id": {"81": 1, "104": 2},
                    "nucleus_inventory_records": [
                        {
                            "nucleus_id": 81,
                            "z_first": 1,
                            "z_last": 4,
                            "valid_3d_nucleus": True,
                        },
                        {
                            "nucleus_id": 104,
                            "z_first": 3,
                            "z_last": 8,
                            "valid_3d_nucleus": True,
                        },
                        {
                            "nucleus_id": 150,
                            "z_first": 2,
                            "z_last": 6,
                            "valid_3d_nucleus": True,
                        },
                    ],
                }
            ),
        )
        records = controller._gfap_nucleus_records(prepared)
        self.assertEqual(
            [record["instance_id"] for record in records],
            [81, 104, 150],
        )
        self.assertEqual(
            [record.get("owner_display_id") for record in records],
            [1, 2, None],
        )
        self.assertEqual(
            [record["accepted"] for record in records],
            [True, True, False],
        )
        self.assertEqual(
            (records[0]["z_min_0based"], records[0]["z_max_0based_inclusive"]),
            (13, 16),
        )
        self.assertEqual(
            (records[1]["z_min_0based"], records[1]["z_max_0based_inclusive"]),
            (15, 20),
        )

    def test_gfap_capture_keeps_valid_nonowner_nucleus_for_cell_edit(self):
        shape = (12, 14)
        valid_projection = np.zeros(shape, dtype=np.uint32)
        valid_projection[2:5, 2:5] = 81
        valid_projection[6:9, 8:11] = 150
        whole = np.zeros(shape, dtype=np.uint16)
        whole[1:7, 1:7] = 1
        soma = np.zeros_like(whole)
        soma[2:5, 2:5] = 1
        processes = np.where((whole > 0) & (soma == 0), whole, 0).astype(
            np.uint16
        )
        prepared = SimpleNamespace(
            z_selection=SimpleNamespace(start_0based=4),
            dapi_projection=np.ones(shape, dtype=np.uint16),
            analysis=SimpleNamespace(
                valid_nucleus_labels_2d=valid_projection,
                gfap_structural_score=np.ones(shape, dtype=np.float32),
                whole_labels=whole,
                soma_labels=soma,
                process_labels=processes,
                diagnostics={
                    "source_owner_to_display_id": {"81": 1},
                    "nucleus_inventory_records": [
                        {
                            "nucleus_id": 81,
                            "z_first": 1,
                            "z_last": 4,
                            "valid_3d_nucleus": True,
                        },
                        {
                            "nucleus_id": 150,
                            "z_first": 2,
                            "z_last": 5,
                            "valid_3d_nucleus": True,
                        },
                    ],
                },
            ),
        )
        capture = controller._gfap_capture(
            prepared,
            {
                "DAPI": {
                    "pixel_width_um": 0.2,
                    "pixel_height_um": 0.2,
                    "pixel_depth_um": 0.5,
                }
            },
        )
        np.testing.assert_array_equal(
            capture.compartment_metrics[
                "_canonical_nucleus_instance_extent_labels_2d"
            ],
            valid_projection,
        )
        records = capture.compartment_metrics["nucleus_3d_inventory"][
            "canonical_per_nucleus"
        ]
        self.assertEqual(
            [record["instance_id"] for record in records],
            [81, 150],
        )
        self.assertEqual(records[0]["owner_display_id"], 1)
        self.assertNotIn("owner_display_id", records[1])

    def test_gfap_report_keeps_major_timings_without_full_diagnostics_dump(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = np.zeros((8, 9), dtype=np.uint16)
            labels[2:6, 3:7] = 1
            prepared = SimpleNamespace(
                analysis=SimpleNamespace(
                    whole_labels=labels,
                    soma_labels=labels,
                    process_labels=labels,
                    diagnostics={
                        "internal_record_that_must_not_be_reported": [
                            {"threshold": 123}
                        ],
                    },
                ),
                z_selection=SimpleNamespace(
                    start_1based=2,
                    end_1based_inclusive=7,
                    projection="max",
                ),
                nucleus_detection={
                    "model": "test model",
                    "model_sha256": "a" * 64,
                },
                stage_timings_seconds={
                    "z_selection": 1.0,
                    "dapi_nucleus_model": 2.0,
                    "gfap_compartments": 3.0,
                    "measurement_preparation": 4.0,
                    "fiji_review_and_publication": 5.0,
                },
            )
            report = root / "report.txt"
            controller._write_gfap_report(
                report,
                prepared=prepared,
                input_dir=root,
                paths={
                    "DAPI": root / "DAPI.tif",
                    "GFAP": root / "GFAP.tif",
                    "KCNN2": root / "KCNN2.tif",
                },
                metadata={
                    "DAPI": {
                        "shape": (9, 8, 9),
                        "pixel_width_um": 0.2,
                        "pixel_height_um": 0.2,
                        "pixel_depth_um": 0.5,
                    },
                },
                measurement="KCNN2",
                fiji_status="Completed after Fiji review and native measurement",
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("Automated Inference Elapsed (s): 6.000", text)
            self.assertIn("Z selection: Completed (1.000 s)", text)
            self.assertIn(
                "Fiji review and publication: Completed (5.000 s)",
                text,
            )
            self.assertNotIn("Diagnostics:", text)
            self.assertNotIn("internal_record_that_must_not_be_reported", text)

    def test_gfap_debug_handler_writes_standard_preview_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (10, 12)
            whole = np.zeros(shape, dtype=np.uint16)
            whole[2:9, 2:10] = 1
            soma = np.zeros_like(whole)
            soma[4:7, 4:8] = 1
            processes = whole.copy()
            processes[soma > 0] = 0
            analysis = SimpleNamespace(
                whole_labels=whole,
                soma_labels=soma,
                process_labels=processes,
                nucleus_labels_2d=(soma > 0).astype(np.uint16),
                valid_nucleus_labels_2d=(soma > 0).astype(np.uint16),
                gfap_structural_score=np.ones(shape, dtype=np.float32),
                diagnostics={"analysis_mode": "dapi_gfap_only"},
            )
            prepared = SimpleNamespace(
                analysis=analysis,
                dapi_projection=np.ones(shape, dtype=np.uint16),
                gfap_projection=np.ones(shape, dtype=np.uint16),
                measurement_projection=np.ones(shape, dtype=np.uint16),
                z_selection=SimpleNamespace(
                    start_1based=2,
                    end_1based_inclusive=6,
                    projection="max",
                ),
                nucleus_detection={
                    "model": "test",
                    "model_sha256": "0" * 64,
                },
            )
            runtime = SimpleNamespace(
                DEBUG_WHOLE_OVERLAY_FILENAME="whole.png",
                DEBUG_SOMA_OVERLAY_FILENAME="soma.png",
                DEBUG_PROCESS_OVERLAY_FILENAME="processes.png",
                DEBUG_STATE_FILENAME="state.npz",
                DEBUG_REPORT_FILENAME="report.txt",
                Image=Image,
                morphology=morphology,
                segmentation=segmentation,
                make_fiji_like_composite=lambda *_args: np.zeros(
                    (*shape, 3),
                    dtype=np.float32,
                ),
            )
            dapi = root / "DAPI.tif"
            gfap = root / "GFAP.tif"
            kcnn2 = root / "KCNN2.tif"
            for path in (dapi, gfap, kcnn2):
                path.write_bytes(b"source")
            outputs = controller._write_gfap_debug_outputs(
                runtime=runtime,
                prepared=prepared,
                output_dir=root,
                input_dir=root,
                paths={"DAPI": dapi, "GFAP": gfap, "KCNN2": kcnn2},
                metadata={
                    "DAPI": {
                        "shape": (8, *shape),
                        "pixel_width_um": 0.1,
                        "pixel_height_um": 0.1,
                        "pixel_depth_um": 0.4,
                    }
                },
                measurement="KCNN2",
            )
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            with np.load(outputs["state"], allow_pickle=False) as archive:
                self.assertEqual(
                    str(archive["analysis_mode"]),
                    "dapi_gfap_only",
                )
                self.assertTrue(
                    np.array_equal(
                        archive["whole_labels"],
                        whole,
                    )
                )


if __name__ == "__main__":
    unittest.main()
