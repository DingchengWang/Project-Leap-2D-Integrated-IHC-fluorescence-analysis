from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "project_leap_2d"
FALLBACK = PROJECT_ROOT / "fallback" / "single_file_fallback.py"
MANIFEST_PATH = PROJECT_ROOT / "validation" / "source_manifest.json"
RELEASE_CONTRACT_PATH = PROJECT_ROOT / "validation" / "release_contract.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_fallback():
    name = "project_leap_2d_fallback_parity"
    spec = importlib.util.spec_from_file_location(name, FALLBACK)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import fallback")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SourceAndRuntimeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from project_leap_2d.runtime_loader import load_runtime
        from project_leap_2d.runtime_manifest import GROOVY_RESOURCE_SHA256
        from release_audit_support import protected_definition_errors

        cls.runtime = load_runtime()
        cls.fallback = load_fallback()
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.release_contract = json.loads(
            RELEASE_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        cls.groovy_resource_sha256 = GROOVY_RESOURCE_SHA256
        cls.protected_definition_errors = staticmethod(
            protected_definition_errors
        )

    def test_runtime_uses_current_release_display_name(self) -> None:
        self.assertEqual(
            self.runtime.PRODUCT_DISPLAY_NAME,
            "Project Leap 2D",
        )

    def test_fallback_is_exact_canonical_source(self) -> None:
        self.assertEqual(
            sha256_bytes(FALLBACK.read_bytes()),
            self.manifest["canonical_source_sha256"],
        )
        self.assertEqual(
            self.runtime._PROJECT_LEAP_CANONICAL_SOURCE_SHA256,
            self.manifest["canonical_source_sha256"],
        )
        self.assertEqual(
            FALLBACK.stat().st_size,
            self.manifest["canonical_source_bytes"],
        )

    def test_runtime_and_fallback_reject_gfap_only_before_metadata(self) -> None:
        for target in (self.runtime, self.fallback):
            with self.subTest(target=target.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    args = SimpleNamespace(
                        fiji_timeout_minutes=10.0,
                        dapi_fragment_workload_preflight_only=False,
                        dapi_fragment_workload_json=None,
                        input_dir=root / "input",
                        output_dir=root / "output",
                    )
                    paths = {
                        "DAPI": root / "DAPI.tif",
                        "GFAP": root / "GFAP.tif",
                        "KCNN2": root / "KCNN2.tif",
                    }
                    with (
                        mock.patch.object(target, "parse_args", return_value=args),
                        mock.patch.object(
                            target,
                            "discover_channel_paths",
                            return_value=(paths, []),
                        ),
                        mock.patch.object(
                            target,
                            "read_meta",
                            side_effect=AssertionError("metadata must not be read"),
                        ),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "single-file fallback supports the eGFP route only",
                        ):
                            target._main_locked([])

    def test_age_filename_contract_and_egfp_morphology_fallback_are_preserved(
        self,
    ) -> None:
        self.assertIsNone(
            self.runtime.detect_filename_age_profile(
                {
                    "DAPI": Path("DAPI.tif"),
                    "eGFP": Path("eGFP.tif"),
                }
            )
        )
        mature = self.runtime.detect_filename_age_profile(
            {
                "DAPI": Path("DAPI_mature.tif"),
                "eGFP": Path("eGFP.tif"),
            }
        )
        neonatal = self.runtime.detect_filename_age_profile(
            {
                "DAPI": Path("DAPI.tif"),
                "eGFP": Path("eGFP_neonatal.tif"),
            }
        )
        self.assertEqual(mature.profile, "mature")
        self.assertEqual(neonatal.profile, "neonatal")
        with self.assertRaisesRegex(ValueError, "conflicting neonatal/mature"):
            self.runtime.detect_filename_age_profile(
                {
                    "DAPI": Path("DAPI_neonatal.tif"),
                    "eGFP": Path("eGFP_mature.tif"),
                }
            )
        self.assertIn(
            "filename_age_decision or classify_age_profile(",
            inspect.getsource(self.runtime._main_locked),
        )

    def test_every_protected_definition_retains_its_exact_body(self) -> None:
        self.assertEqual(
            len(self.manifest["definitions"]),
            self.release_contract["protected_definition_count"],
        )
        self.assertEqual(
            self.protected_definition_errors(
                code_root=CODE_ROOT,
                baseline_manifest=self.manifest,
                registered_additions=self.release_contract[
                    "protected_module_additions"
                ],
                registered_module_sha256=self.release_contract[
                    "protected_module_sha256"
                ],
            ),
            [],
        )

    def test_candidate_catalog_is_exact_and_ordered(self) -> None:
        rng = np.random.default_rng(20260726)
        stacks = {
            "eGFP": rng.integers(0, 4096, size=(11, 24, 20), dtype=np.uint16),
            "GFAP": rng.integers(0, 4096, size=(11, 24, 20), dtype=np.uint16),
        }
        expected = [
            asdict(value)
            for value in self.fallback.complete_candidate_specs(11, stacks)
        ]
        observed = [
            asdict(value)
            for value in self.runtime.complete_candidate_specs(11, stacks)
        ]
        self.assertEqual(len(observed), 90)
        self.assertEqual(observed, expected)
        self.assertEqual(
            [
                asdict(value)
                for value in self.runtime.morphology_baseline_specs(11)
            ],
            expected[:30],
        )
        self.assertEqual(
            [
                asdict(value)
                for value in self.runtime.morphology_with_adaptive_refinement_specs(
                    11,
                    stacks,
                )
            ],
            expected[:50],
        )
        self.assertEqual(
            [
                asdict(value)
                for value in self.runtime.morphology_with_structural_refinement_specs(
                    11,
                    stacks,
                )
            ],
            expected[:60],
        )

    def test_fiji_resource_matches_current_manifest_and_protocol(self) -> None:
        resource = (
            CODE_ROOT
            / "fiji_review"
            / "resources"
            / "astrocyte_roi_reviewer.groovy"
        )
        value = resource.read_bytes()
        self.assertTrue(value.endswith(b"\n"))
        self.assertEqual(
            sha256_bytes(value),
            self.groovy_resource_sha256,
        )
        self.assertEqual(self.runtime.FIJI_GROOVY_SCRIPT.encode("utf-8"), value)
        source = value.decode("utf-8")
        required_tokens = (
            'enabledPythonEdits.contains("split")',
            'enabledPythonEdits.contains("enlarge")',
            'Button("Split Selected Whole Cell")',
            'Button("Enlarge Selected Soma")',
            'Button("Cancel Cell Edit")',
            'Button("Revert")',
            "validateTripletRoiSets(loadedSets)",
            "state_token",
            "label_mask_sha256",
            "selected_cell_uid",
            "new Thread({",
        )
        for token in required_tokens:
            self.assertIn(token, source)
        renumber = source[
            source.index("def renumberRoiSets = {") :
            source.index("def refreshPersistentViews = {")
        ]
        self.assertIn(
            "originToFinal[originalRoiId(roi) as Integer] = index + 1",
            renumber,
        )
        self.assertNotIn("roiLineage(roi).each", renumber)

    def test_thread_environment_precedes_numpy_import(self) -> None:
        startup = (CODE_ROOT / "startup.py").read_text(encoding="utf-8")
        self.assertLess(startup.index("OMP_NUM_THREADS"), startup.index("import numpy"))

    def test_workspace_entrypoint_does_not_preload_numpy(self) -> None:
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "import project_leap_2d.workspace_launcher\n"
            "print(json.dumps({'numpy_loaded': 'numpy' in sys.modules}))\n"
        )
        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment.pop(name, None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as cache:
            environment["MPLCONFIGDIR"] = cache
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
        self.assertFalse(json.loads(completed.stdout)["numpy_loaded"])

    def test_requested_candidate_module_filenames_exist(self) -> None:
        segmentation = CODE_ROOT / "segmentation"
        self.assertTrue(
            (
                segmentation
                / "structural_refinement_candidates.py"
            ).is_file()
        )
        self.assertTrue(
            (
                segmentation
                / "distributional_threshold_candidates.py"
            ).is_file()
        )

    def test_required_documents_and_removed_internal_documents(self) -> None:
        self.assertTrue((PROJECT_ROOT / "MODULE_MAP_\u4e2d\u6587.md").is_file())
        self.assertTrue((PROJECT_ROOT / "MODULE_MAP_ENGLISH.md").is_file())
        self.assertFalse((PROJECT_ROOT / "MODULE_MAP.md").exists())
        self.assertEqual(
            (PROJECT_ROOT / "RUN_COMMAND.txt").read_text(encoding="utf-8"),
            (
                "# Run these commands from the Project Leap 2D (8-23-26) "
                "package directory.\n"
                "\n"
                "# First-time installation\n"
                "./Installation/macOS/install_macos.command\n"
                "\n"
                "# Run an analysis\n"
                "./run_project_leap_2d.command\n"
            ),
        )
        forbidden = {
            "DECISIONS.md",
            "SECOND_STAGE_WORKLOG.md",
        }
        self.assertFalse(
            forbidden.intersection(path.name for path in PROJECT_ROOT.rglob("*"))
        )


if __name__ == "__main__":
    unittest.main()
