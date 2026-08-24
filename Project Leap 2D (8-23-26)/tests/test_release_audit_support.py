from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from release_audit_support import (
    ForbiddenOptionalImport,
    forbid_optional_imports,
    markdown_reference_errors,
    protected_definition_errors,
    registered_definition_errors,
    release_file_manifest,
    required_document_token_errors,
    sha256_bytes,
    top_level_definitions,
)


class ReleaseAuditSupportTests(unittest.TestCase):
    def test_protected_definitions_allow_registered_modules_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "protected.py"
            protected.write_text("def kept():\n    return 1\n", encoding="utf-8")
            row = top_level_definitions(protected)["kept"]
            manifest = {
                "definitions": {
                    "kept": {
                        **row,
                        "module": "protected.py",
                    }
                }
            }
            (root / "registered.py").write_text(
                "def added():\n    return 2\n",
                encoding="utf-8",
            )
            self.assertEqual(
                protected_definition_errors(
                    code_root=root,
                    baseline_manifest=manifest,
                ),
                [],
            )

    def test_protected_definition_body_change_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "protected.py"
            baseline = "def kept():\n    return 1\n"
            path.write_text(baseline, encoding="utf-8")
            row = top_level_definitions(path)["kept"]
            manifest = {
                "definitions": {
                    "kept": {
                        **row,
                        "module": "protected.py",
                    }
                }
            }
            path.write_text("def kept():\n    return 2\n", encoding="utf-8")
            errors = protected_definition_errors(
                code_root=root,
                baseline_manifest=manifest,
            )
            self.assertEqual(
                errors,
                ["protected definition body changed: protected.py:kept"],
            )

    def test_module_registry_rejects_unregistered_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registered.py"
            path.write_text("def added():\n    return 2\n", encoding="utf-8")
            errors = registered_definition_errors(
                code_root=root,
                registered_modules=("registered.py",),
                registered={"registered.py": {}},
            )
            self.assertIn(
                "unregistered definitions in registered.py: ['added']",
                errors,
            )

    def test_description_audit_checks_surface_links_and_required_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(
                "# Read me\n[map](MODULE_MAP.md)\n",
                encoding="utf-8",
            )
            (root / "MODULE_MAP.md").write_text(
                "# Map\nnew_feature.py\n",
                encoding="utf-8",
            )
            self.assertEqual(
                markdown_reference_errors(
                    root,
                    ("README.md", "MODULE_MAP.md"),
                ),
                [],
            )
            self.assertEqual(
                required_document_token_errors(
                    root,
                    {"MODULE_MAP.md": ("new_feature.py",)},
                ),
                [],
            )
            (root / "OLD.md").write_text("obsolete", encoding="utf-8")
            errors = markdown_reference_errors(
                root,
                ("README.md", "MODULE_MAP.md"),
            )
            self.assertIn("unexpected description files remain: ['OLD.md']", errors)

    def test_optional_import_guard_is_active_only_inside_context(self) -> None:
        module_name = "_optional_route_guard_fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{module_name}.py").write_text("VALUE = 1\n", encoding="utf-8")
            sys.path.insert(0, str(root))
            try:
                with forbid_optional_imports((module_name,)):
                    with self.assertRaisesRegex(
                        ForbiddenOptionalImport,
                        "optional import was attempted",
                    ):
                        importlib.import_module(module_name)
                loaded = importlib.import_module(module_name)
                self.assertEqual(loaded.VALUE, 1)
            finally:
                sys.modules.pop(module_name, None)
                sys.path.remove(str(root))

    def test_release_manifest_excludes_only_declared_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "Result").mkdir()
            (root / "Result" / "output.txt").write_text("run", encoding="utf-8")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "code.pyc").write_bytes(b"cache")
            observed = release_file_manifest(
                root,
                excluded_relative_paths=(),
                excluded_directory_names=("Result", "__pycache__"),
                excluded_suffixes=(".pyc",),
            )
            self.assertEqual(set(observed), {"code.py"})
            self.assertEqual(observed["code.py"]["bytes"], 10)

    def test_sha256_helper_is_byte_exact(self) -> None:
        fixture = {"value": "µm"}
        payload = json.dumps(fixture, ensure_ascii=False).encode("utf-8")
        self.assertEqual(sha256_bytes(payload), sha256_bytes(bytes(payload)))


if __name__ == "__main__":
    unittest.main()
