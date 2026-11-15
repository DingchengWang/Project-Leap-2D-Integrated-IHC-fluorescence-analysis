"""Protocol and ownership checks; these tests never load models or Fiji."""
from __future__ import annotations

import fcntl
import ast
import io
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from project_leap_2d.desktop_adapter import _run_scientific_workflow, run_request, validate_request
from project_leap_2d import desktop_adapter as adapter
from project_leap_2d.desktop_paths import configure_worker_paths, fiji_cache_root, fiji_process_environment, validate_request_paths


class EngineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="leap-adapter-", dir="/private/tmp")
        self.root = Path(self.temporary.name)
        self.run = self.root / "中文 task" / "run-1"
        self.run.mkdir(parents=True)
        self.components = self.root / "components"
        (self.components / "Models").mkdir(parents=True)
        (self.components / "Models" / "cpsam_v2").write_bytes(b"test-only-placeholder")
        self.launcher = self.components / "fiji"
        self.launcher.write_text("#!/bin/sh\nexit 1\n")
        self.launcher.chmod(0o700)
        self.python = self.components / "python"
        self.python.write_bytes(b"test-only-placeholder")
        (self.components / "runtime.json").write_text(json.dumps({
            "schema_version": 2, "component_id": "project-leap-2d", "component_version": "1.0.1"}))
        self.lock = self.root / "components.lock"
        self.lock.touch()
        self.request = {
            "schema_version": 1, "run_id": "run-1",
            "workspace": str(self.run / "workspace"), "cache_root": str(self.run / "cache"),
            "events_path": str(self.run / "events.jsonl"), "outcome_path": str(self.run / "outcome.json"),
            "fiji_launcher": str(self.launcher),
        }
        self.request_path = self.run / "request.json"
        self.request_path.write_text(json.dumps(self.request))
        self.environment = {
            "PROJECT_LEAP_COMPONENT_ROOT": str(self.components),
            "CELLPOSE_LOCAL_MODELS_PATH": str(self.components / "Models"),
            "PROJECT_LEAP_COMPONENT_LOCK": str(self.lock),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def execute(self, workflow):
        with patch.dict(os.environ, self.environment), patch.object(sys, "executable", str(self.python)), redirect_stderr(io.StringIO()):
            with patch.object(adapter, "_run_scientific_workflow", workflow):
                code = run_request(self.request_path)
        return code, json.loads((self.run / "outcome.json").read_text())

    def events(self):
        return [json.loads(line) for line in (self.run / "events.jsonl").read_text().splitlines()]

    def provenance_fixture(self, *, shared):
        code = self.root / "fixture-engine/project_leap_2d"
        (code / "resources/models").mkdir(parents=True)
        (code / "desktop_adapter.py").write_text("# provenance fixture\n")
        (code / "resources/models/instanseg_single_channel_nuclei.pt").write_bytes(b"fixture instanseg")
        (code.parent / "VERSION").write_text("1.0.1\n")
        (self.run / "workspace/Sample Image").mkdir(parents=True)
        weight = self.components / "Models/cpsam_v2"
        dependency = dict(adapter.SHARED_CELLPOSE_DEPENDENCY, sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
        descriptor = {"schema_version": 2, "component_id": "project-leap-2d", "component_version": "1.0.1"}
        root = self.components
        if shared:
            root = self.root / "shared-model"
            (root / "Models").mkdir(parents=True)
            (root / "Models/cpsam_v2").write_bytes(weight.read_bytes())
            self.environment.update(PROJECT_LEAP_MODEL_ROOT=str(root), CELLPOSE_LOCAL_MODELS_PATH=str(root / "Models"))
            descriptor.update(schema_version=3, model_component=dependency)
            model_descriptor = {"schema_version": 2, "component_id": dependency["id"],
                "component_version": dependency["version"], "models": "Models", "model_file": "cpsam_v2",
                "model_sha256": dependency["sha256"]}
            (root / "runtime.json").write_text(json.dumps(model_descriptor))
        (self.components / "runtime.json").write_text(json.dumps(descriptor))
        model_manifest = {"schema_version": 1, "component_id": "cellpose-model" if shared else "project-leap-2d",
            "component_version": dependency["version"] if shared else "1.0.1", "entries": {
                "Models/cpsam_v2": {"type": "file", "bytes": weight.stat().st_size, "sha256": dependency["sha256"]},
                "runtime.json": {"type": "file", "sha256": hashlib.sha256((root / "runtime.json").read_bytes()).hexdigest()}}}
        (root / "content-manifest.json").write_text(json.dumps(model_manifest))
        if shared:
            (self.components / "content-manifest.json").write_text(json.dumps({"entries": {}}))
        return code, root, dependency

    def read_provenance(self, code, dependency):
        paths = {"workspace": self.run / "workspace", "run_root": self.run}
        with patch.dict(os.environ, self.environment), \
             patch.object(sys, "executable", str(self.python)), \
             patch.object(adapter, "__file__", str(code / "desktop_adapter.py")), \
             patch.object(adapter, "SHARED_CELLPOSE_DEPENDENCY", dependency), \
             patch.object(adapter, "fiji_process_environment", return_value={"JAVA_TOOL_OPTIONS": "fixture"}):
            return adapter.runtime_provenance(self.request, paths)

    def shared_science_fixture(self):
        code, model_root, dependency = self.provenance_fixture(shared=True)
        descriptor_path = self.components / "runtime.json"
        descriptor = json.loads(descriptor_path.read_text())
        descriptor.update(schema_version=4, platform="macos-arm64",
                          science_runtime={"id": "science-runtime", "version": "1.0.0"})
        descriptor_path.write_text(json.dumps(descriptor))
        root = self.root / "shared science runtime"
        self.python = root / "python/bin/python3.9"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"test-only shared scientific Python executable")
        self.python.chmod(0o700)
        (root / "site-packages").mkdir()
        runtime = {"schema_version": 4, "component_id": "science-runtime", "component_version": "1.0.0",
                   "platform": "macos-arm64", "python": "python/bin/python3.9", "site_packages": "site-packages",
                   "environment_contract": "test-only-environment-identity"}
        (root / "runtime.json").write_text(json.dumps(runtime))
        entries = {name: {"type": "file", "bytes": (root / name).stat().st_size,
                          "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                   for name in ("runtime.json", "python/bin/python3.9")}
        (root / "content-manifest.json").write_text(json.dumps({
            "schema_version": 1, "component_id": "science-runtime", "component_version": "1.0.0",
            "entries": entries}))
        self.environment["PROJECT_LEAP_SCIENCE_ROOT"] = str(root)
        return code, root, dependency

    def configure_science_fixture(self):
        paths = validate_request_paths(self.request, self.request_path)
        with patch.dict(os.environ, self.environment), patch.object(sys, "executable", str(self.python)):
            return configure_worker_paths(self.request, paths)

    def test_schema4_shared_science_runtime_accepts_real_split_layout_and_records_provenance(self):
        code, root, dependency = self.shared_science_fixture()
        self.assertEqual(self.configure_science_fixture(), self.launcher)
        payload = self.read_provenance(code, dependency)
        actual = payload["runtime"]["science_runtime"]
        self.assertEqual(actual["root"], str(root))
        self.assertEqual(actual["python_executable"], str(self.python))
        self.assertEqual(actual["python_fingerprint"]["sha256"], hashlib.sha256(self.python.read_bytes()).hexdigest())
        self.assertEqual(actual["descriptor"]["environment_contract"], "test-only-environment-identity")
        self.assertEqual(actual["component_manifest"]["sha256"], hashlib.sha256((root / "content-manifest.json").read_bytes()).hexdigest())
        self.assertEqual(payload["runtime"]["descriptor"]["schema_version"], 4)
        self.assertTrue(payload["models"]["cellpose"]["rehashed_this_run"])

    def test_schema4_requires_explicit_separate_real_science_root(self):
        _code, root, _dependency = self.shared_science_fixture()
        link = self.root / "science-link"
        link.symlink_to(root, target_is_directory=True)
        for value in (None, "relative/runtime", str(link), str(self.components),
                      str(Path.home() / "Applications/Project Leap 2D Support")):
            with self.subTest(root=value):
                environment = dict(self.environment)
                if value is None:
                    environment.pop("PROJECT_LEAP_SCIENCE_ROOT")
                else:
                    environment["PROJECT_LEAP_SCIENCE_ROOT"] = value
                with patch.dict(os.environ, environment, clear=True), patch.object(sys, "executable", str(self.python)):
                    with self.assertRaises(ValueError):
                        configure_worker_paths(self.request, validate_request_paths(self.request, self.request_path))

    def test_schema4_rejects_wrong_dependency_identity_and_manifest(self):
        _code, root, _dependency = self.shared_science_fixture()
        cases = (
            (self.components / "runtime.json", lambda d: d.update(science_runtime={"id": "other", "version": "1.0.0"})),
            (self.components / "runtime.json", lambda d: d.update(science_runtime={"id": "science-runtime", "version": "2.0.0"})),
            (root / "runtime.json", lambda d: d.update(component_id="other")),
            (root / "runtime.json", lambda d: d.update(schema_version=3)),
            (root / "runtime.json", lambda d: d.update(platform="wrong-platform")),
            (root / "content-manifest.json", lambda d: d.update(component_id="other")),
            (root / "content-manifest.json", lambda d: d.update(component_version="2.0.0")),
            (root / "content-manifest.json", lambda d: d["entries"]["runtime.json"].update(sha256="0" * 64)),
            (root / "content-manifest.json", lambda d: d["entries"]["python/bin/python3.9"].update(sha256="0" * 64)),
        )
        for target, mutate in cases:
            original = target.read_bytes()
            try:
                data = json.loads(original)
                mutate(data)
                target.write_text(json.dumps(data))
                with self.subTest(target=str(target), data=data), self.assertRaises(ValueError):
                    self.configure_science_fixture()
            finally:
                target.write_bytes(original)

    def test_schema4_rejects_undeclared_or_damaged_python_and_missing_site_packages(self):
        _code, root, _dependency = self.shared_science_fixture()
        declared = self.python
        for candidate in (self.components / "python", root / "other-python"):
            with self.subTest(python=str(candidate)):
                self.python = candidate
                with self.assertRaisesRegex(ValueError, "declared shared science runtime executable"):
                    self.configure_science_fixture()
        self.python = declared
        before = declared.read_bytes()
        declared.write_bytes(b"X" * len(before))
        with self.assertRaisesRegex(ValueError, "differs from its manifest"):
            self.configure_science_fixture()
        declared.write_bytes(before)
        (root / "site-packages").rmdir()
        with self.assertRaisesRegex(ValueError, "site_packages directory is missing"):
            self.configure_science_fixture()

    def test_schema4_rejects_escaping_descriptor_paths_even_with_updated_manifest(self):
        _code, root, _dependency = self.shared_science_fixture()
        runtime_path, manifest_path = root / "runtime.json", root / "content-manifest.json"
        original_runtime, original_manifest = runtime_path.read_bytes(), manifest_path.read_bytes()
        for field in ("python", "site_packages"):
            for value in ("../outside", "/outside", "nested/../outside", "nested//outside"):
                try:
                    runtime = json.loads(original_runtime)
                    runtime[field] = value
                    runtime_path.write_text(json.dumps(runtime))
                    manifest = json.loads(original_manifest)
                    manifest["entries"]["runtime.json"].update(
                        bytes=runtime_path.stat().st_size, sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
                    manifest_path.write_text(json.dumps(manifest))
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "runtime path is invalid"):
                        self.configure_science_fixture()
                finally:
                    runtime_path.write_bytes(original_runtime)
                    manifest_path.write_bytes(original_manifest)

    def test_schema4_preserves_manifest_declared_internal_python_symlink(self):
        code, root, dependency = self.shared_science_fixture()
        actual_python = self.python
        link = actual_python.with_name("python")
        link.symlink_to(actual_python.name)
        runtime_path, manifest_path = root / "runtime.json", root / "content-manifest.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["python"] = "python/bin/python"
        runtime_path.write_text(json.dumps(runtime))
        manifest = json.loads(manifest_path.read_text())
        manifest["entries"]["runtime.json"].update(
            bytes=runtime_path.stat().st_size, sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
        manifest["entries"]["python/bin/python"] = {"type": "symlink", "target": actual_python.name}
        manifest_path.write_text(json.dumps(manifest))
        self.python = link
        self.assertEqual(self.configure_science_fixture(), self.launcher)
        self.assertEqual(self.read_provenance(code, dependency)["runtime"]["science_runtime"]["python_executable"],
                         str(actual_python))
        manifest["entries"]["python/bin/python"]["target"] = "another-python"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "symlink differs from its manifest"):
            self.configure_science_fixture()

    def test_embedded_model_provenance_preserves_original_contract(self):
        code, _root, dependency = self.provenance_fixture(shared=False)
        payload = self.read_provenance(code, dependency)
        self.assertNotIn("cellpose_model", payload["runtime"])
        self.assertFalse(payload["models"]["cellpose"]["rehashed_this_run"])
        self.assertEqual(payload["models"]["cellpose"]["sha256"], dependency["sha256"])

    def test_shared_model_paths_and_provenance_use_separate_component(self):
        code, root, dependency = self.provenance_fixture(shared=True)
        paths = validate_request_paths(self.request, self.request_path)
        with patch.dict(os.environ, self.environment), patch.object(sys, "executable", str(self.python)):
            self.assertEqual(configure_worker_paths(self.request, paths), self.launcher)
        payload = self.read_provenance(code, dependency)
        self.assertEqual(payload["runtime"]["cellpose_model"]["root"], str(root))
        self.assertEqual(payload["runtime"]["cellpose_model"]["descriptor"]["component_id"], "cellpose-model")
        self.assertEqual(payload["runtime"]["descriptor"]["component_id"], "project-leap-2d")
        self.assertTrue(payload["models"]["cellpose"]["rehashed_this_run"])

    def test_shared_model_provenance_rejects_dependency_manifest_and_bytes_mismatch(self):
        code, root, dependency = self.provenance_fixture(shared=True)
        for target, key in ((self.components / "runtime.json", "dependency"),
                            (root / "runtime.json", "identity"),
                            (root / "content-manifest.json", "manifest"),
                            (root / "Models/cpsam_v2", "bytes")):
            with self.subTest(key=key):
                before = target.read_bytes()
                if key == "bytes":
                    target.write_bytes(b"X" * len(before))
                else:
                    value = json.loads(before)
                    if key == "dependency":
                        value["model_component"]["sha256"] = "0" * 64
                    elif key == "identity":
                        value["component_id"] = "wrong-model"
                    else:
                        value["entries"]["Models/cpsam_v2"]["sha256"] = "0" * 64
                    target.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    self.read_provenance(code, dependency)
                target.write_bytes(before)

    def test_shared_model_root_does_not_relax_path_or_python_ownership(self):
        code, root, dependency = self.provenance_fixture(shared=True)
        paths = validate_request_paths(self.request, self.request_path)
        symlink = self.root / "model-link"
        symlink.symlink_to(root, target_is_directory=True)
        for model_root, python, message in ((str(self.components), str(self.python), "Models must belong"),
                (str(symlink), str(self.python), "symlinks"),
                (str(Path.home() / "Applications/Project Leap 2D Support"), str(self.python), "protected production"),
                (str(root), str(root / "python"), "Scientific Python")):
            with self.subTest(message=message), patch.dict(os.environ, dict(self.environment, PROJECT_LEAP_MODEL_ROOT=model_root)), \
                 patch.object(sys, "executable", python):
                with self.assertRaisesRegex(ValueError, message):
                    configure_worker_paths(self.request, paths)
        self.environment["PROJECT_LEAP_MODEL_ROOT"] = str(self.components)
        with self.assertRaises(ValueError):
            self.read_provenance(code, dependency)

    def test_zero_exit_without_commit_is_failure(self):
        code, result = self.execute(lambda *_: (0, {}))
        self.assertEqual(code, 1)
        self.assertEqual(result["error_code"], "engine_not_published")
        self.assertIsNone(result["results_dir"])
        self.assertEqual(self.events()[-1]["type"], "failed")

    def test_engine_check_has_distinct_terminal_code(self):
        _, result = self.execute(lambda *_: (2, {}))
        self.assertEqual(result["error_code"], "engine_check_stopped")

    def test_shared_fiji_runs_from_its_own_managed_root(self):
        fiji_root = self.root / "shared fiji"
        fiji_root.mkdir()
        launcher = fiji_root / "fiji"
        launcher.write_text("#!/bin/sh\nexit 1\n")
        launcher.chmod(0o700)
        self.request["fiji_launcher"] = str(launcher)
        self.request["module_id"] = "project-leap-2d"
        self.request_path.write_text(json.dumps(self.request))
        self.environment["PROJECT_LEAP_FIJI_ROOT"] = str(fiji_root)

        def workflow(request, paths, actual_launcher, events):
            self.assertEqual(actual_launcher, launcher)
            events.committed = True
            return 0, {}

        code, result = self.execute(workflow)
        self.assertEqual(code, 0)
        self.assertEqual(result["module_id"], "project-leap-2d")

    def test_fiji_launcher_cannot_escape_declared_dependency(self):
        fiji_root = self.root / "shared fiji"
        fiji_root.mkdir()
        self.environment["PROJECT_LEAP_FIJI_ROOT"] = str(fiji_root)
        _, result = self.execute(lambda *_: self.fail("Mismatched Fiji must not start"))
        self.assertIn("isolated Fiji component root", result["message"])

    def test_worker_rejects_request_for_different_module(self):
        with self.assertRaisesRegex(ValueError, "only runs"):
            validate_request(dict(self.request, module_id="different-module"))

    def test_completed_requires_commit_and_preserves_inputs(self):
        original = self.run / "workspace" / "Sample Image" / "DAPI.tif"
        original.parent.mkdir(parents=True)
        original.write_bytes(b"immutable-working-copy")

        def workflow(request, paths, launcher, events):
            events.committed = True
            return 0, {"roi_archive": str(paths["workspace"] / "Result" / "IHC_2D_ROI_Archive.zip")}

        code, result = self.execute(workflow)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(original.read_bytes(), b"immutable-working-copy")
        events = self.events()
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual([event["type"] for event in events if event["type"] in {"completed", "cancelled", "failed"}], ["completed"])

    def test_cleanup_error_after_commit_keeps_completed_artifacts(self):
        def workflow(request, paths, launcher, events):
            events.committed = True
            events.artifacts = {"workbook": "committed.xlsx"}
            raise OSError("synthetic post-publication cleanup error")

        code, result = self.execute(workflow)
        self.assertEqual(code, 0)
        self.assertEqual(result["artifacts"], {"workbook": "committed.xlsx"})
        self.assertIn("warning", [event["type"] for event in self.events()])

    def test_marker_cancels_before_workflow(self):
        (self.run / "cancel.request").touch()
        code, result = self.execute(lambda *_: self.fail("Workflow must not run"))
        self.assertEqual(code, 130)
        self.assertEqual(result["status"], "cancelled")

    def test_fiji_review_cancel_is_not_completed(self):
        def workflow(request, paths, launcher, events):
            events.review_cancelled = True
            return 0, {}
        code, result = self.execute(workflow)
        self.assertEqual((code, result["status"]), (130, "cancelled"))

    def test_cancellation_only_terminates_registered_session(self):
        python = sys.executable
        owned = subprocess.Popen([python, "-c", "import time; time.sleep(60)"], start_new_session=True)
        unrelated = subprocess.Popen([python, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            def workflow(request, paths, launcher, events):
                events.register_process(owned, "fiji")
                events.cancelled = True
                events.check_cancelled()
            code, result = self.execute(workflow)
            self.assertEqual(code, 130)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
            self.assertFalse(result["processes"][0]["running"])
            started = [event for event in self.events() if event["type"] == "process"][0]
            self.assertEqual((started["pid"], started["pgid"], started["process_type"]), (owned.pid, owned.pid, "fiji"))
        finally:
            for process in (owned, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)

    def test_completed_also_closes_its_fiji(self):
        owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            def workflow(request, paths, launcher, events):
                events.register_process(owned, "fiji")
                events.committed = True
                return 0, {}
            code, result = self.execute(workflow)
            self.assertEqual(code, 0)
            self.assertIsNotNone(owned.poll())
            self.assertFalse(result["processes"][0]["running"])
        finally:
            if owned.poll() is None:
                owned.terminate()
            owned.wait(timeout=5)

    def test_exclusive_maintenance_lock_blocks_science(self):
        with self.lock.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _, result = self.execute(lambda *_: self.fail("Workflow must not run under maintenance"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("installed or repaired", result["message"])

    def test_application_analysis_lock_blocks_second_worker(self):
        with (self.root / "analysis.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            code, result = self.execute(lambda *_: self.fail("A second analysis must not start"))
        self.assertEqual(code, 1)
        self.assertEqual(result["error_code"], "analysis_already_running")

    def test_worker_identity_and_lock_are_owned_by_worker(self):
        def workflow(request, paths, launcher, events):
            identity = json.loads((self.run / "worker.json").read_text())
            self.assertEqual(identity, {"pid": os.getpid(), "kind": "analysis", "request_path": str(self.request_path)})
            with (self.root / "analysis.lock").open("r+") as handle:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return 0, {}
        self.execute(workflow)
        with (self.root / "analysis.lock").open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_rejects_path_escape_and_symlinks(self):
        request = dict(self.request, cache_root=str(self.root / "elsewhere"))
        with self.assertRaisesRegex(ValueError, "cache_root must be"):
            validate_request_paths(request, self.request_path)
        (self.run / "cache").symlink_to(self.components, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            validate_request_paths(self.request, self.request_path)

    def test_desktop_cache_never_falls_back_to_production(self):
        with patch.dict(os.environ, {"PROJECT_LEAP_DESKTOP_WORKER": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires PROJECT_LEAP_CACHE_ROOT"):
                fiji_cache_root()

    def test_fiji_java_environment_is_task_local_without_changing_home(self):
        supplied = {"PROJECT_LEAP_DESKTOP_WORKER": "1", "PROJECT_LEAP_CACHE_ROOT": str(self.run / "cache"),
                    "HOME": str(self.root / "unchanged-home"), "JAVA_TOOL_OPTIONS": "-Duser.home=/wrong",
                    "JDK_JAVA_OPTIONS": "-Xmx1m", "_JAVA_OPTIONS": "-Djava.io.tmpdir=/wrong"}
        with patch.dict(os.environ, supplied):
            environment = fiji_process_environment()
            self.assertEqual(os.environ["JAVA_TOOL_OPTIONS"], supplied["JAVA_TOOL_OPTIONS"])
        self.assertEqual(environment["HOME"], supplied["HOME"])
        self.assertNotIn("JDK_JAVA_OPTIONS", environment)
        self.assertNotIn("_JAVA_OPTIONS", environment)
        self.assertNotIn("/wrong", environment["JAVA_TOOL_OPTIONS"])
        for relative in ("java-home", "java-prefs/user", "java-prefs/system", "java-tmp"):
            self.assertTrue((self.run / "cache" / relative).is_dir())
            self.assertIn('"' + str(self.run / "cache" / relative) + '"', environment["JAVA_TOOL_OPTIONS"])

    def test_rejects_scientific_overrides(self):
        for arguments in (["--skip-fiji"], ["--disable-cellpose"], ["--input-dir", "/tmp"]):
            with self.assertRaises(ValueError):
                validate_request(dict(self.request, engine_args=arguments))

    def test_rejects_repeat_run(self):
        self.execute(lambda *_: (0, {}))
        with self.assertRaisesRegex(ValueError, "already has an outcome"):
            validate_request_paths(self.request, self.request_path)

    def _real_progress_callback(self, name, namespace):
        # Execute the actual event callback, without loading or running its
        # enclosing scientific pipeline. The algorithm stays outside this test.
        path = Path(__file__).resolve().parents[1] / "engine/project_leap_2d/analysis_controller.py"
        tree = ast.parse(path.read_text())
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), callback], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), "exec"), namespace)
        return namespace[name]

    def test_cellpose_progress_counts_successful_distinct_windows(self):
        emitted = []
        namespace = {"emit_event": lambda *args, **kwargs: emitted.append((args, kwargs)),
                     "print_terminal_event": lambda *_: None, "completed_cellpose_windows": set(),
                     "fixed_z_windows": list(range(5))}
        callback = self._real_progress_callback("print_shared_preparation", namespace)
        event = {"position_1based": 3, "z_start_1based": 1, "z_end_1based_inclusive": 39,
                 "step": "cellpose", "status": "started"}
        callback(event)
        callback(dict(event, step="dapi_max", status="completed"))
        callback(dict(event, status="disabled"))
        self.assertEqual(emitted, [])
        callback(dict(event, status="completed"))
        callback(dict(event, status="completed"))
        callback(dict(event, position_1based=1, status="completed"))
        self.assertEqual([details["current"] for _, details in emitted], [1, 2])
        self.assertEqual([details["total"] for _, details in emitted], [5, 5])
        self.assertTrue(all(args[:2] == ("progress", "cellpose_inference") for args, _ in emitted))

    def test_candidate_progress_uses_completed_count_not_candidate_id(self):
        emitted = []
        namespace = {"emit_event": lambda *args, **kwargs: emitted.append((args, kwargs)),
                     "print_terminal_event": lambda *_: None, "chosen_specs": list(range(90))}
        for name in ("candidate_module_display_name", "candidate_profile_family", "candidate_profile_name", "candidate_threshold_display_name"):
            namespace[name] = lambda _: "fixture"
        callback = self._real_progress_callback("print_candidate_row", namespace)
        row = {"candidate": 89, "z_start_1based": 1, "z_end_1based_inclusive": 39,
               "mask_area_px": 10, "connected_components": 1, "soma_supported_components": 1,
               "fine_branch_retained_px": 0}
        callback(row, None, 1, 1, 5, 0.1)
        callback(dict(row, candidate=1), None, 2, 1, 1, 0.1)
        self.assertEqual([details["current"] for _, details in emitted], [1, 2])
        self.assertEqual([details["total"] for _, details in emitted], [90, 90])
        self.assertTrue(all(args[:2] == ("progress", "candidate_morphology") for args, _ in emitted))

    def test_publication_wrapper_commits_six_outputs_together(self):
        for relative in ("Sample Image", "Result", "Analysis Package/Run State"):
            (self.run / "workspace" / relative).mkdir(parents=True, exist_ok=True)
        (self.run / "workspace/Sample Image/DAPI.tif").write_bytes(b"working-input")
        observed = {}
        filenames = {
            "whole": "IHC_2D_Whole_Astrocyte_Overlay.png",
            "processes": "IHC_2D_Astrocyte_Processes_Overlay.png",
            "soma": "IHC_2D_Astrocyte_Soma_Overlay.png",
            "report": "IHC_2D_Analysis_Report.txt",
            "workbook": "IHC_2D_Fluorescence_Results.xlsx",
        }

        def publish(*, staged_files, final_files, run_dir):
            observed.update({"staged": set(staged_files), "final": set(final_files)})
            for key in staged_files:
                final_files[key].write_bytes(staged_files[key].read_bytes())

        runtime = types.SimpleNamespace(publish_output_bundle=publish)
        fake_workflow = types.ModuleType("project_leap_2d.analysis_workflow")

        def run_workflow(runtime, args):
            staging = self.run / "cache" / ("run-" + "a" * 32)
            staging.mkdir()
            for key, filename in filenames.items():
                (staging / filename).write_bytes(key.encode())
            runtime.publish_output_bundle(
                staged_files={key: staging / filename for key, filename in filenames.items()},
                final_files={key: self.run / "workspace/Result" / filename for key, filename in filenames.items()},
                run_dir=staging,
            )
            return 0

        fake_workflow.run_analysis_workflow = run_workflow

        def archive(run_dir, destination):
            self.assertEqual(json.loads((run_dir / "runtime_provenance.json").read_text())["schema_version"], 1)
            destination.write_bytes(b"validated-roi-archive-fixture")
            return destination

        with patch("project_leap_2d.runtime_loader.load_runtime", return_value=runtime), \
             patch("project_leap_2d.desktop_adapter.runtime_provenance", return_value={"schema_version": 1}), \
             patch("project_leap_2d.final_roi_archive.build_final_roi_archive", side_effect=archive), \
             patch.dict(sys.modules, {"project_leap_2d.analysis_workflow": fake_workflow}):
            code, result = self.execute(_run_scientific_workflow)
        expected = set(filenames) | {"roi_archive"}
        self.assertEqual(code, 0)
        self.assertEqual(observed, {"staged": expected, "final": expected})
        self.assertEqual(set(result["artifacts"]), expected)
        self.assertEqual(len(list((self.run / "workspace/Result").iterdir())), 6)
        self.assertEqual((self.run / "workspace/Sample Image/DAPI.tif").read_bytes(), b"working-input")


if __name__ == "__main__":
    unittest.main()
