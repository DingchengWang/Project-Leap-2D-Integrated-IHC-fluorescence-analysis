from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from project_leap_2d.fiji_review import cell_edit_fiji_bridge as bridge
from project_leap_2d.fiji_review.cell_edit_context import (
    build_cell_edit_context,
    load_cell_edit_context,
)


class _FakeProcess:
    returncode = None

    def poll(self):
        return self.returncode


class _FakeCellEditService:
    instances = []

    def __init__(self, *, manifest, dispatcher):
        self.manifest = manifest
        self.dispatcher = dispatcher
        self.poll_count = 0
        self.closed = False
        self.__class__.instances.append(self)

    def poll(self):
        self.poll_count += 1

    def close(self):
        self.closed = True


class CellEditFijiBridgeTests(unittest.TestCase):
    _PRODUCTION_RUN_NAME = f"run-{'a' * 32}"

    def setUp(self) -> None:
        _FakeCellEditService.instances.clear()

    @staticmethod
    def base_prepare(root: Path):
        def prepare(**kwargs):
            run_dir = root / "run"
            run_dir.mkdir()
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"base_marker": kwargs["base_marker"]}),
                encoding="utf-8",
            )
            return run_dir, manifest_path

        return prepare

    @classmethod
    def production_base_prepare(cls, root: Path):
        def prepare(**kwargs):
            run_dir = root / "cache" / cls._PRODUCTION_RUN_NAME
            run_dir.mkdir(parents=True)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"base_marker": kwargs["base_marker"]}),
                encoding="utf-8",
            )
            return run_dir, manifest_path

        return prepare

    @staticmethod
    def context_builder(root: Path):
        dapi_path = root / "DAPI.tif"
        structural_path = root / "eGFP.tif"
        dapi_path.write_bytes(b"dapi-source")
        structural_path.write_bytes(b"structural-source")

        def build(context_dir: Path):
            shape = (12, 14)
            whole = np.zeros(shape, dtype=np.uint16)
            whole[2:10, 2:12] = 1
            soma = np.zeros_like(whole)
            soma[4:8, 5:9] = 1
            processes = whole.copy()
            processes[soma > 0] = 0
            extent = np.zeros(shape, dtype=np.uint32)
            extent[4:8, 5:9] = 7
            core = np.zeros_like(extent)
            core[5:7, 6:8] = 7
            return build_cell_edit_context(
                run_dir=context_dir,
                basename="analysis_context",
                dapi_path=dapi_path,
                structural_paths={"eGFP": structural_path},
                dapi_projection=(extent > 0).astype(np.uint16),
                structural_map=whole.astype(np.float32),
                selected_z=(1, 3),
                calibration={
                    "pixel_width_um": 0.2,
                    "pixel_height_um": 0.2,
                    "pixel_depth_um": 0.5,
                },
                age_profile="mature",
                canonical_core_labels=core,
                canonical_extent_labels=extent,
                initial_triplet=(whole, soma, processes),
            )

        return build

    def test_missing_context_keeps_base_manifest_and_hides_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, manifest_path = bridge.prepare_cell_edit_fiji_runtime(
                base_prepare=self.base_prepare(root),
                cell_edit_context_builder=None,
                base_marker="unchanged",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest, {"base_marker": "unchanged"})
            self.assertFalse((run_dir / "cell_edit").exists())

    def test_builder_and_compatibility_path_are_mutually_exclusive(self):
        base_prepare = mock.Mock()
        with self.assertRaisesRegex(ValueError, "not both"):
            bridge.prepare_cell_edit_fiji_runtime(
                base_prepare=base_prepare,
                cell_edit_context_builder=lambda _destination: None,
                cell_edit_context_path=Path("/tmp/context.npz"),
            )
        base_prepare.assert_not_called()

    def test_compatibility_context_path_still_relocates_and_enables_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = self.context_builder(root)(root / "staged")
            run_dir, manifest_path = bridge.prepare_cell_edit_fiji_runtime(
                base_prepare=self.base_prepare(root),
                cell_edit_context_path=staged.npz_path,
                enabled_cell_edit_actions=("split",),
                base_marker="retained",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["cell_edit"]["enabled_actions"], ["split"])
            self.assertEqual(
                Path(manifest["cell_edit"]["context_path"]),
                (run_dir / "cell_edit" / "analysis_context.json").resolve(),
            )
            self.assertFalse(staged.npz_path.exists())
            self.assertFalse(staged.json_path.exists())

    def test_existing_context_atomically_adds_cell_edit_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder = mock.Mock(wraps=self.context_builder(root))
            with mock.patch.object(
                bridge,
                "load_cell_edit_context",
                wraps=load_cell_edit_context,
            ) as validate:
                run_dir, manifest_path = bridge.prepare_cell_edit_fiji_runtime(
                    base_prepare=self.base_prepare(root),
                    cell_edit_context_builder=builder,
                    enabled_cell_edit_actions=("enlarge",),
                    cell_edit_timeout_seconds=12.5,
                    base_marker="retained",
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["base_marker"], "retained")
            self.assertEqual(
                manifest["cell_edit"]["enabled_actions"],
                ["enlarge"],
            )
            self.assertEqual(manifest["cell_edit"]["timeout_seconds"], 12.5)
            self.assertEqual(
                Path(manifest["cell_edit"]["context_path"]),
                (run_dir / "cell_edit" / "analysis_context.json").resolve(),
            )
            self.assertEqual(
                Path(manifest["cell_edit"]["program_root"]),
                Path(bridge.__file__).resolve().parents[2],
            )
            builder.assert_called_once_with((run_dir / "cell_edit").resolve())
            validate.assert_called_once_with(
                (run_dir / "cell_edit" / "analysis_context.json").resolve(),
                verify_sources=True,
                verify_source_hashes=True,
            )
            self.assertFalse((root / "staged").exists())
            loaded = load_cell_edit_context(
                manifest["cell_edit"]["context_path"]
            )
            self.assertEqual(loaded.npz_path.name, "analysis_context.npz")
            self.assertTrue((run_dir / "cell_edit" / "requests").is_dir())
            self.assertEqual(list(run_dir.glob(".manifest.json.*.tmp")), [])

    def test_failed_final_context_build_removes_entire_unlaunched_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fail_after_partial_write(context_dir: Path):
                context_dir.mkdir(parents=True)
                (context_dir / "analysis_context.npz").write_bytes(b"partial")
                raise RuntimeError("simulated context build failure")

            with mock.patch.object(
                bridge,
                "_fiji_cache_root",
                return_value=(root / "cache").resolve(),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated context build failure",
                ):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=self.production_base_prepare(root),
                        cell_edit_context_builder=fail_after_partial_write,
                        cleanup_unlaunched_run_on_failure=True,
                        base_marker="retained",
                    )

            self.assertFalse((root / "cache" / self._PRODUCTION_RUN_NAME).exists())

    def test_failed_final_context_validation_removes_entire_unlaunched_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    bridge,
                    "_fiji_cache_root",
                    return_value=(root / "cache").resolve(),
                ),
                mock.patch.object(
                    bridge,
                    "load_cell_edit_context",
                    side_effect=RuntimeError("simulated final validation failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "final validation failure"):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=self.production_base_prepare(root),
                        cell_edit_context_builder=self.context_builder(root),
                        cleanup_unlaunched_run_on_failure=True,
                        base_marker="retained",
                    )
            self.assertFalse((root / "cache" / self._PRODUCTION_RUN_NAME).exists())

    def test_failed_runtime_directory_preparation_removes_entire_unlaunched_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    bridge,
                    "_fiji_cache_root",
                    return_value=(root / "cache").resolve(),
                ),
                mock.patch.object(
                    bridge,
                    "prepare_cell_edit_runtime",
                    side_effect=RuntimeError("simulated runtime preparation failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime preparation failure"):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=self.production_base_prepare(root),
                        cell_edit_context_builder=self.context_builder(root),
                        cleanup_unlaunched_run_on_failure=True,
                        base_marker="retained",
                    )
            self.assertFalse((root / "cache" / self._PRODUCTION_RUN_NAME).exists())

    def test_failed_manifest_commit_removes_entire_unlaunched_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    bridge,
                    "_fiji_cache_root",
                    return_value=(root / "cache").resolve(),
                ),
                mock.patch.object(
                    bridge,
                    "atomic_write_cell_edit_json",
                    side_effect=RuntimeError("simulated manifest commit failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "manifest commit failure"):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=self.production_base_prepare(root),
                        cell_edit_context_builder=self.context_builder(root),
                        cleanup_unlaunched_run_on_failure=True,
                        base_marker="retained",
                    )
            self.assertFalse((root / "cache" / self._PRODUCTION_RUN_NAME).exists())

    def test_default_builder_failure_preserves_callback_run_and_removes_only_partial_cell_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fail_after_partial_write(context_dir: Path):
                context_dir.mkdir(parents=True)
                (context_dir / "analysis_context.npz").write_bytes(b"partial")
                raise RuntimeError("simulated public callback failure")

            with self.assertRaisesRegex(RuntimeError, "public callback failure"):
                bridge.prepare_cell_edit_fiji_runtime(
                    base_prepare=self.base_prepare(root),
                    cell_edit_context_builder=fail_after_partial_write,
                    base_marker="retained",
                )
            self.assertTrue((root / "run" / "manifest.json").is_file())
            self.assertFalse((root / "run" / "cell_edit").exists())

    def test_cleanup_opt_in_rejects_arbitrary_callback_directory_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valuable = root / "valuable"
            valuable.mkdir()
            sentinel = valuable / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            manifest_path = valuable / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            builder = mock.Mock()

            with mock.patch.object(
                bridge,
                "_fiji_cache_root",
                return_value=(root / "cache").resolve(),
            ):
                with self.assertRaisesRegex(ValueError, "unrecognized"):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=lambda **_kwargs: (
                            valuable,
                            manifest_path,
                        ),
                        cell_edit_context_builder=builder,
                        cleanup_unlaunched_run_on_failure=True,
                    )
            builder.assert_not_called()
            self.assertTrue(sentinel.is_file())
            self.assertTrue(manifest_path.is_file())

    def test_cleanup_opt_in_rejects_preexisting_validly_named_cache_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            run_dir = cache_root / self._PRODUCTION_RUN_NAME
            run_dir.mkdir(parents=True)
            sentinel = run_dir / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            builder = mock.Mock()

            with mock.patch.object(
                bridge,
                "_fiji_cache_root",
                return_value=cache_root.resolve(),
            ):
                with self.assertRaisesRegex(ValueError, "unrecognized"):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=lambda **_kwargs: (
                            run_dir,
                            manifest_path,
                        ),
                        cell_edit_context_builder=builder,
                        cleanup_unlaunched_run_on_failure=True,
                    )
            builder.assert_not_called()
            self.assertTrue(sentinel.is_file())
            self.assertTrue(manifest_path.is_file())

    def test_compatibility_manifest_failure_keeps_relocated_context_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = self.context_builder(root)(root / "staged")
            with mock.patch.object(
                bridge,
                "atomic_write_cell_edit_json",
                side_effect=RuntimeError("simulated compatibility manifest failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "compatibility manifest failure",
                ):
                    bridge.prepare_cell_edit_fiji_runtime(
                        base_prepare=self.base_prepare(root),
                        cell_edit_context_path=staged.npz_path,
                        base_marker="retained",
                    )
            self.assertTrue((root / "run" / "manifest.json").is_file())
            self.assertTrue(
                (root / "run" / "cell_edit" / "analysis_context.npz").is_file()
            )
            self.assertTrue(
                (root / "run" / "cell_edit" / "analysis_context.json").is_file()
            )
            self.assertFalse(staged.npz_path.exists())
            self.assertFalse(staged.json_path.exists())

    def test_launcher_polls_service_and_returns_without_termination(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "ihc_fiji_bridge.groovy").write_text(
                "return null",
                encoding="utf-8",
            )
            (run_dir / "fiji_done.json").write_text(
                json.dumps({"roi_count": 4}),
                encoding="utf-8",
            )
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cell_edit": {
                            "enabled_actions": ["split"],
                            "request_dir": str(run_dir / "requests"),
                            "response_dir": str(run_dir / "responses"),
                            "cancel_dir": str(run_dir / "cancel"),
                        }
                    }
                ),
                encoding="utf-8",
            )
            fake_process = _FakeProcess()
            dispatcher = object()
            with (
                mock.patch.object(
                    bridge.subprocess,
                    "Popen",
                    return_value=fake_process,
                ),
                mock.patch.object(
                    bridge,
                    "CellEditRequestService",
                    _FakeCellEditService,
                ),
                mock.patch.object(
                    bridge,
                    "_cell_edit_dispatcher",
                    return_value=dispatcher,
                ),
                mock.patch.object(
                    bridge,
                    "terminate_fiji_process_group",
                ) as terminate,
            ):
                result = bridge.launch_cell_edit_fiji_workflow(
                    launcher=Path("/Applications/Fiji/fiji"),
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    timeout_minutes=1.0,
                )
            self.assertEqual(result["roi_count"], 4)
            self.assertEqual(len(_FakeCellEditService.instances), 1)
            service = _FakeCellEditService.instances[0]
            self.assertIs(service.dispatcher, dispatcher)
            self.assertEqual(service.poll_count, 1)
            self.assertTrue(service.closed)
            terminate.assert_not_called()

    def test_timeout_closes_service_and_terminates_only_launched_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "ihc_fiji_bridge.groovy").write_text(
                "return null",
                encoding="utf-8",
            )
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cell_edit": {
                            "enabled_actions": ["enlarge"],
                            "request_dir": str(run_dir / "requests"),
                            "response_dir": str(run_dir / "responses"),
                            "cancel_dir": str(run_dir / "cancel"),
                        }
                    }
                ),
                encoding="utf-8",
            )
            fake_process = _FakeProcess()
            with (
                mock.patch.object(
                    bridge.subprocess,
                    "Popen",
                    return_value=fake_process,
                ),
                mock.patch.object(
                    bridge,
                    "CellEditRequestService",
                    _FakeCellEditService,
                ),
                mock.patch.object(
                    bridge,
                    "_cell_edit_dispatcher",
                    return_value=object(),
                ),
                mock.patch.object(
                    bridge,
                    "terminate_fiji_process_group",
                ) as terminate,
            ):
                with self.assertRaises(TimeoutError):
                    bridge.launch_cell_edit_fiji_workflow(
                        launcher=Path("/Applications/Fiji/fiji"),
                        run_dir=run_dir,
                        manifest_path=manifest_path,
                        timeout_minutes=0.0,
                    )
            self.assertTrue(_FakeCellEditService.instances[0].closed)
            terminate.assert_called_once_with(fake_process)

    def test_fiji_error_closes_service_and_terminates_launched_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "ihc_fiji_bridge.groovy").write_text(
                "return null",
                encoding="utf-8",
            )
            (run_dir / "fiji_error.txt").write_text(
                "simulated Fiji failure",
                encoding="utf-8",
            )
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cell_edit": {
                            "enabled_actions": ["split"],
                            "request_dir": str(run_dir / "requests"),
                            "response_dir": str(run_dir / "responses"),
                            "cancel_dir": str(run_dir / "cancel"),
                        }
                    }
                ),
                encoding="utf-8",
            )
            fake_process = _FakeProcess()
            with (
                mock.patch.object(
                    bridge.subprocess,
                    "Popen",
                    return_value=fake_process,
                ) as popen,
                mock.patch.object(
                    bridge,
                    "CellEditRequestService",
                    _FakeCellEditService,
                ),
                mock.patch.object(
                    bridge,
                    "_cell_edit_dispatcher",
                    return_value=object(),
                ),
                mock.patch.object(
                    bridge,
                    "terminate_fiji_process_group",
                ) as terminate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated Fiji failure",
                ):
                    bridge.launch_cell_edit_fiji_workflow(
                        launcher=Path("/Applications/Fiji/fiji"),
                        run_dir=run_dir,
                        manifest_path=manifest_path,
                        timeout_minutes=1.0,
                    )
            self.assertTrue(_FakeCellEditService.instances[0].closed)
            terminate.assert_called_once_with(fake_process)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_base_launch_does_not_resolve_cell_edit_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "ihc_fiji_bridge.groovy").write_text(
                "return null",
                encoding="utf-8",
            )
            (run_dir / "fiji_done.json").write_text(
                json.dumps({"roi_count": 2}),
                encoding="utf-8",
            )
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            fake_process = _FakeProcess()
            with (
                mock.patch.object(
                    bridge.subprocess,
                    "Popen",
                    return_value=fake_process,
                ),
                mock.patch.object(
                    bridge,
                    "_cell_edit_dispatcher",
                ) as resolve_dispatcher,
            ):
                result = bridge.launch_cell_edit_fiji_workflow(
                    launcher=Path("/Applications/Fiji/fiji"),
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    timeout_minutes=1.0,
                )
            self.assertEqual(result["roi_count"], 2)
            resolve_dispatcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
