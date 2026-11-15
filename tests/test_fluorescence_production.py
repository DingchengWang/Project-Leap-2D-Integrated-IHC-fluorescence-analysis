"""Production review, model integrity, and resource-limit regression checks."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass
import inspect
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine/project_leap_2d"
sys.path.insert(0, str(ROOT / "engine"))
from project_leap_2d.analysis_modes.gfap_only import gfap_only_pipeline as gfap
from project_leap_2d.nuclei import instanseg_nucleus_detection as instanseg
from project_leap_2d import desktop_adapter, runtime_manifest


def load_functional_module(path, namespace):
    exec(compile(path.read_text(), str(path), "exec"), namespace)


class ProductionEntryTests(unittest.TestCase):
    def test_command_line_requires_the_production_path(self):
        namespace = {"argparse": argparse, "Path": Path,
                     "DEFAULT_INPUT_DIR": Path("/input"), "DEFAULT_OUT_DIR": Path("/output")}
        load_functional_module(ENGINE / "command_line.py", namespace)
        self.assertEqual(namespace["parse_args"]([]).fiji_timeout_minutes, 120.0)
        for option in ("--disable-cellpose", "--skip-fiji", "--fiji-auto-continue",
                       "--dapi-fragment-workload-preflight-only", "--dapi-fragment-workload-json"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                namespace["parse_args"]([option])
        self.assertEqual(list(inspect.signature(desktop_adapter.run_request).parameters), ["request_path"])

    def test_fiji_resource_requires_user_choice_and_matches_integrity_pin(self):
        import hashlib
        source = (ENGINE / "fiji_review/resources/astrocyte_roi_reviewer.groovy").read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), runtime_manifest.GROOVY_RESOURCE_SHA256)
        self.assertIn('String action = chooseAction()', source.decode())
        self.assertNotIn('auto_continue', source.decode())

    def test_packaged_instanseg_resources_retain_the_pinned_identity(self):
        details = instanseg.validate_instanseg_model_resources()
        self.assertEqual(details["sha256"], instanseg.INSTANSEG_MODEL_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            damaged = Path(directory) / "damaged.pt"
            damaged.write_bytes(b"not-the-pinned-model")
            with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                instanseg.validate_instanseg_model_resources(model_path=damaged)


class GFAPProductionTests(unittest.TestCase):
    def setUp(self):
        self.stack = np.ones((7, 10, 10), dtype=np.float32)
        whole = np.zeros((10, 10), dtype=np.uint16)
        whole[3:7, 3:7] = 1
        soma = np.zeros_like(whole)
        soma[4:6, 4:6] = 1
        processes = whole.copy()
        processes[soma > 0] = 0
        self.analysis = SimpleNamespace(whole_labels=whole, soma_labels=soma, process_labels=processes)
        self.events = []

    def run_pipeline(self, *, analysis=None, fiji=True):
        def detector(dapi, height, width, *, z_indices):
            self.events.append("detect")
            return SimpleNamespace(labels_zyx=np.ones((len(z_indices), 10, 10), dtype=np.uint32),
                                   model_sha256=instanseg.INSTANSEG_MODEL_SHA256)

        def analyzer(*args, **kwargs):
            self.events.append("analyze")
            self.assertEqual(len(args), 4)
            return analysis or self.analysis

        def measurement(*args):
            self.events.append("measurement")
            return np.full((10, 10), 777, dtype=np.uint16)

        def review(prepared):
            self.events.append("review")
            self.assertTrue(np.all(prepared.measurement_projection == 777))
            return {"reviewed": True}

        with patch.object(gfap, "_default_nucleus_detector", detector), \
             patch.object(gfap, "_default_gfap_analyzer", analyzer):
            return gfap.run_gfap_only_pipeline(
                available_channels={"DAPI", "GFAP", "KCNN1"}, egfp_is_valid=False,
                dapi_stack=self.stack, gfap_stack=self.stack, pixel_height_um=0.5,
                pixel_width_um=0.5, z_spacing_um=1.0,
                measurement_projection_loader=measurement, fiji_handler=review if fiji else None)

    def test_roi_definition_precedes_measurement_and_review(self):
        result = self.run_pipeline()
        self.assertEqual(self.events, ["detect", "analyze", "measurement", "review"])
        self.assertEqual(result.fiji_result, {"reviewed": True})
        self.assertEqual(result.prepared.nucleus_detection["model_sha256"], instanseg.INSTANSEG_MODEL_SHA256)

    def test_invalid_roi_partition_stops_before_measurement(self):
        invalid = SimpleNamespace(**vars(self.analysis))
        invalid.soma_labels = invalid.whole_labels.copy()
        with self.assertRaisesRegex(RuntimeError, "exactly Soma union Processes"):
            self.run_pipeline(analysis=invalid)
        self.assertEqual(self.events, ["detect", "analyze"])

    def test_missing_review_cannot_complete(self):
        with self.assertRaisesRegex(ValueError, "review and measurement"):
            self.run_pipeline(fiji=False)
        self.assertNotIn("review", self.events)


class DAPIWorkloadTests(unittest.TestCase):
    def test_real_workload_guard_writes_diagnostic_before_fragment_evaluation(self):
        names = {"DapiNucleus3DConfig", "DapiFragmentWorkloadLimits", "DapiParentFragmentWorkload",
                 "DapiFragmentWorkloadLimitExceeded"}
        tree = ast.parse((ENGINE / "data_structures.py").read_text())
        selected = [tree.body[0], *[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]]
        namespace = {"dataclass": dataclass, "np": np, "Path": Path, "time": time,
                     "math": math, "ndi": ndi, "measure": measure, "asdict": asdict,
                     "json": json, "os": __import__("os"), "uuid": __import__("uuid")}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "workload-types", "exec"), namespace)
        load_functional_module(ENGINE / "nuclei/dapi_3d_inventory.py", namespace)
        limits = namespace["DapiFragmentWorkloadLimits"](
            policy_version="test-resource-limit", max_parent_fragments=0, max_total_fragments=100,
            max_parent_voxel_comparisons=10**9, max_total_voxel_comparisons=10**9,
            max_parent_result_payload_bytes_lower_bound=10**9)
        stack = np.zeros((5, 10, 10), dtype=np.float32)
        stack[1:4, 4:6, 4:6] = 100
        core = np.zeros((10, 10), dtype=bool)
        core[4:6, 4:6] = True
        namespace["DAPI_FRAGMENT_WORKLOAD_LIMITS"] = limits
        namespace["dapi_nuclei_core_and_extent"] = lambda *_: (core, core, None, {})
        namespace["evaluate_raw_inventory_object_3d"] = lambda *_: self.fail("Over-limit fragments must not run")
        context = SimpleNamespace(pixel_depth_um=1, z_start_0based=0, z_end_0based_inclusive=4,
                                  dapi_stack=stack, egfp_stack=stack)
        with tempfile.TemporaryDirectory() as directory:
            diagnostic = Path(directory) / "failure.json"
            with self.assertRaises(namespace["DapiFragmentWorkloadLimitExceeded"]):
                namespace["build_dapi_object_inventory_3d"](
                    core, stack.max(axis=0), context, 1, 1,
                    SimpleNamespace(dapi_extent_max_expand_um=1), workload_diagnostic_path=diagnostic)
            result = json.loads(diagnostic.read_text())
        self.assertEqual(result["trigger_metric"], "parent_fragments")
        self.assertFalse(result["jobs_submitted_for_offending_parent"])
        self.assertEqual(result["workload_summary"]["mode"], "enforce")
        self.assertNotIn("preflight_only", inspect.signature(namespace["build_dapi_object_inventory_3d"]).parameters)


if __name__ == "__main__":
    unittest.main()
