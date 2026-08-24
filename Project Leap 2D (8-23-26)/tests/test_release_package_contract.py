from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from release_audit_support import (
    load_json,
    markdown_reference_errors,
    protected_definition_errors,
    registered_definition_errors,
    registered_file_hash_errors,
    registered_module_surface_errors,
    release_file_manifest,
    required_document_token_errors,
    top_level_optional_import_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "project_leap_2d"
BASELINE_MANIFEST = PROJECT_ROOT / "validation" / "source_manifest.json"
RELEASE_CONTRACT = PROJECT_ROOT / "validation" / "release_contract.json"
RELEASE_FILE_MANIFEST = (
    PROJECT_ROOT / "validation" / "release_package_files.json"
)


class ReleasePackageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = load_json(BASELINE_MANIFEST)
        cls.contract = load_json(RELEASE_CONTRACT)

    def test_all_protected_definitions_retain_exact_bodies(self) -> None:
        self.assertEqual(
            len(self.baseline["definitions"]),
            self.contract["protected_definition_count"],
        )
        self.assertEqual(
            protected_definition_errors(
                code_root=CODE_ROOT,
                baseline_manifest=self.baseline,
                registered_additions=self.contract[
                    "protected_module_additions"
                ],
                registered_module_sha256=self.contract[
                    "protected_module_sha256"
                ],
            ),
            [],
        )

    def test_every_registered_definition_is_explicitly_registered(self) -> None:
        self.assertEqual(
            len(self.contract["registered_modules"]),
            len(set(self.contract["registered_modules"])),
        )
        self.assertEqual(
            self.contract["registered_modules"],
            sorted(self.contract["registered_modules"]),
        )
        self.assertEqual(
            registered_module_surface_errors(
                code_root=CODE_ROOT,
                registered_modules=self.contract["registered_modules"],
                registered_module_discovery_globs=self.contract[
                    "registered_module_discovery_globs"
                ],
            ),
            [],
        )
        self.assertTrue(
            set(self.contract["registered_modules"]).isdisjoint(
                self.baseline["module_order"]
            )
        )
        self.assertEqual(
            registered_definition_errors(
                code_root=CODE_ROOT,
                registered_modules=self.contract["registered_modules"],
                registered=self.contract["registered_definitions"],
                registered_module_sha256=self.contract[
                    "registered_module_sha256"
                ],
            ),
            [],
        )

    def test_runtime_and_lifecycle_modules_are_registered(self) -> None:
        required_modules = {
            "fiji_review/failed_run_retention.py",
            "runtime_attributes.py",
            "workspace/input_cleanup_recovery.py",
            "workspace/publication_recovery.py",
            "workspace/workspace_preflight.py",
        }
        self.assertTrue(
            required_modules.issubset(self.contract["registered_modules"])
        )
        expected_map_tokens = {
            "runtime_attributes.py",
            "failed_run_retention.py",
            "workspace_preflight.py",
            "publication_recovery.py",
            "input_cleanup_recovery.py",
            "Installation/macOS",
        }
        required_tokens = self.contract["description_files"][
            "required_tokens"
        ]
        for document in ("MODULE_MAP_ENGLISH.md", "MODULE_MAP_中文.md"):
            self.assertTrue(
                expected_map_tokens.issubset(required_tokens[document])
            )

    def test_scientific_and_fiji_resources_are_hash_registered(self) -> None:
        self.assertEqual(
            registered_file_hash_errors(
                PROJECT_ROOT,
                self.contract["resource_sha256"],
            ),
            [],
        )

    def test_description_surface_is_small_complete_and_link_safe(self) -> None:
        descriptions = self.contract["description_files"]
        errors = markdown_reference_errors(
            PROJECT_ROOT,
            descriptions["allowed"],
        )
        errors.extend(
            required_document_token_errors(
                PROJECT_ROOT,
                descriptions["required_tokens"],
            )
        )
        self.assertEqual(errors, [])

    def test_release_tree_contains_no_hidden_files_or_folders(self) -> None:
        paths = [PROJECT_ROOT, *PROJECT_ROOT.rglob("*")]
        hidden_names = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in paths[1:]
            if path.name.startswith(".")
        )
        hidden_flags = sorted(
            (
                "."
                if path == PROJECT_ROOT
                else path.relative_to(PROJECT_ROOT).as_posix()
            )
            for path in paths
            if int(getattr(path.stat(), "st_flags", 0))
            & int(getattr(stat, "UF_HIDDEN", 0))
        )
        self.assertEqual(hidden_names, [])
        self.assertEqual(hidden_flags, [])
        self.assertTrue((PROJECT_ROOT / "Runtime").is_dir())
        self.assertFalse((PROJECT_ROOT / ".runtime").exists())

    def test_protected_bootstrap_has_no_eager_optional_model_import(self) -> None:
        baseline_modules = [
            CODE_ROOT / relative for relative in self.baseline["module_order"]
        ]
        self.assertEqual(
            top_level_optional_import_errors(
                baseline_modules,
                self.contract["optional_import_prefixes"],
            ),
            [],
        )

    def test_egfp_route_does_not_load_optional_analysis_modules(self) -> None:
        prefixes = self.contract["optional_import_prefixes"]
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "from project_leap_2d.runtime_loader import load_runtime\n"
            "runtime = load_runtime()\n"
            "from project_leap_2d.analysis_workflow import "
            "select_analysis_route\n"
            "routes = {\n"
            "  'egfp': select_analysis_route(('DAPI', 'eGFP')),\n"
            "  'egfp_gfap': select_analysis_route("
            "('DAPI', 'eGFP', 'GFAP')),\n"
            "  'gfap_only': select_analysis_route(('DAPI', 'GFAP')),\n"
            "}\n"
            f"prefixes = {prefixes!r}\n"
            "unexpected = sorted(name for name in sys.modules "
            "if any(name == p or name.startswith(p + '.') for p in prefixes))\n"
            "print(json.dumps({'routes': routes, 'unexpected': unexpected}))\n"
        )
        with tempfile.TemporaryDirectory() as cache:
            environment = {
                **os.environ,
                "MPLCONFIGDIR": cache,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
        observed = json.loads(completed.stdout.strip())
        self.assertEqual(
            observed["routes"],
            {
                "egfp": "egfp",
                "egfp_gfap": "egfp",
                "gfap_only": "gfap_only",
            },
        )
        self.assertEqual(observed["unexpected"], [])

    def test_contract_json_is_stable_and_has_no_absolute_sample_paths(self) -> None:
        raw = RELEASE_CONTRACT.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        self.assertEqual(parsed["schema_version"], 2)
        self.assertNotIn("/Users/", raw)

    def test_every_immutable_release_file_is_hash_registered(self) -> None:
        manifest = load_json(RELEASE_FILE_MANIFEST)
        observed = release_file_manifest(
            PROJECT_ROOT,
            excluded_relative_paths=manifest["excluded_relative_paths"],
            excluded_directory_names=manifest["excluded_directory_names"],
            excluded_suffixes=manifest["excluded_suffixes"],
        )
        self.assertEqual(observed, manifest["files"])


if __name__ == "__main__":
    unittest.main()
