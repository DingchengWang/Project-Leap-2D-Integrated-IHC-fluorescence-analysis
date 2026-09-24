from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[1]
PACKAGE = WORK / "Project Leap 2D V1.0.1"
sys.path.insert(0, str(PACKAGE / "Analysis Package"))

from project_leap_2d import workspace_launcher
from project_leap_2d.workspace import input_cleanup_recovery as inputs
from project_leap_2d.workspace import pending_results as pending
from project_leap_2d.workspace import publication_recovery as publication
from project_leap_2d.workspace import workspace_preflight as preflight
from project_leap_2d.workspace.input_cleanup import InputSnapshot


class WorkspaceLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="leap-layout-")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name).resolve()
        self.root, self.samples, self.result, self.state = self.make_layout("工作包 空格")
        self.home = self.temporary / "isolated-home"
        (self.home / ".Trash").mkdir(parents=True)
        # Every code path that resolves macOS Trash sees this temporary home.
        home_patch = mock.patch.object(Path, "home", return_value=self.home)
        home_patch.start()
        self.addCleanup(home_patch.stop)

    def make_layout(self, name: str) -> tuple[Path, Path, Path, Path]:
        root = self.temporary / name
        samples = root / "Sample Image"
        result = root / "Result"
        state = root / "Analysis Package" / "Run State"
        for path in (samples, result, state):
            path.mkdir(parents=True)
        return root, samples, result, state

    @staticmethod
    def snapshot(root: Path) -> dict[str, bytes | str]:
        return {
            path.relative_to(root).as_posix(): (
                path.read_bytes() if path.is_file() else "directory"
            )
            for path in root.rglob("*")
        }

    @staticmethod
    def make_inputs(samples: Path) -> tuple[InputSnapshot, ...]:
        records = []
        for index in range(2):
            path = samples / f"C{index + 1}-fixture.tif"
            # These are small opaque files; no image decoder is called.
            path.write_bytes(f"synthetic-input-{index}".encode())
            records.append(InputSnapshot.capture(path))
        return tuple(records)

    def make_publication(self, name: str = "publish"):
        root, _, result, state = self.make_layout(name)
        run = root / "synthetic-fiji-run"
        run.mkdir()
        staged, final = {}, {}
        for key in ("whole", "processes", "soma", "report", "workbook"):
            staged[key] = run / f"{key}.staged"
            final[key] = result / f"{key}.final"
            staged[key].write_bytes(f"new-{key}".encode())
            final[key].write_bytes(f"old-{key}".encode())
        return root, result, state, run, staged, final

    @staticmethod
    def publisher(*, staged_files, final_files, run_dir):
        del run_dir
        for key, source in staged_files.items():
            temporary = final_files[key].with_suffix(".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, final_files[key])

    @staticmethod
    def begin_publication(result, state, staged, final):
        return publication._begin_publication(
            final_files=final,
            new_fingerprints={
                key: publication._regular_file_fingerprint(path, "Synthetic staged file")
                for key, path in staged.items()
            },
            result_root=result,
            runtime_root=state,
        )

    def test_new_nested_layout_accepts_without_old_root_directories(self) -> None:
        preflight.validate_workspace_layout(self.root)
        self.assertFalse((self.root / "Runtime").exists())
        self.assertFalse((self.root / "Original Image").exists())

    def test_missing_nested_state_is_not_replaced_by_old_runtime(self) -> None:
        self.state.rmdir()
        (self.root / "Runtime").mkdir()
        with self.assertRaisesRegex(RuntimeError, "Analysis Package/Run State"):
            preflight.validate_workspace_layout(self.root)

    def test_workspace_root_and_each_required_directory_reject_symlinks(self) -> None:
        root_link = self.temporary / "root-link"
        root_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            preflight.validate_workspace_layout(root_link)
        for index, relative in enumerate(("Sample Image", "Result", "Analysis Package", "Analysis Package/Run State")):
            with self.subTest(relative=relative):
                root, _, _, _ = self.make_layout(f"symlink-{index}")
                original = root / relative
                external = self.temporary / f"external-{index}"
                original.rename(external)
                original.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, "Symlinks are not allowed"):
                    preflight.validate_workspace_layout(root)

    def test_empty_sample_message_ignores_finder_metadata(self) -> None:
        (self.samples / ".DS_Store").write_bytes(b"metadata")
        (self.samples / "._fixture.tif").write_bytes(b"metadata")
        with self.assertRaisesRegex(RuntimeError, "Sample Image is empty"):
            preflight.require_nonempty_original_image(self.samples)

    def test_run_wrapper_rejects_program_or_state_symlink_before_writes(self) -> None:
        contract = (PACKAGE / "Analysis Package" / "project_leap_2d" / "maintenance" / "environment_contract.txt").read_text().strip()
        for index, relative in enumerate(("Analysis Package", "Analysis Package/project_leap_2d", "Analysis Package/Run State")):
            with self.subTest(relative=relative):
                root, _, _, _ = self.make_layout(f"wrapper-symlink-{index}")
                local_contract = root / "Analysis Package" / "project_leap_2d" / "maintenance" / "environment_contract.txt"
                local_contract.parent.mkdir(parents=True)
                local_contract.write_text(contract, encoding="utf-8")
                launcher = root / "Run Analysis.command"
                shutil.copy2(PACKAGE / "Run Analysis.command", launcher)
                supplied = root / relative
                external = self.temporary / f"wrapper-external-{index}"
                supplied.rename(external)
                supplied.symlink_to(external, target_is_directory=True)
                support = self.temporary / f"wrapper-support-{index}"
                support.mkdir()
                invocation = support / "python-invoked"
                python = support / "fake-python"
                python.write_text('#!/bin/sh\nprintf invoked > "$PROJECT_LEAP_SUPPORT_DIR/python-invoked"\n', encoding="utf-8")
                python.chmod(0o755)
                fiji = support / "fake-fiji"
                fiji.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                fiji.chmod(0o755)
                models = support / "Models"
                models.mkdir()
                (models / "cpsam_v2").write_bytes(b"synthetic-model-placeholder")
                (support / "installation_state.json").write_text(json.dumps({
                    "schema_version": "2", "status": "ready", "platform": "macos-arm64",
                    "environment_contract_id": contract, "python_executable": str(python),
                    "fiji_launcher": str(fiji), "cellpose_models_path": str(models),
                }, indent=2), encoding="utf-8")
                before = self.snapshot(external)
                completed = subprocess.run(
                    ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", "/bin/zsh", str(launcher)],
                    cwd=root, env={**os.environ, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(self.home), "ZDOTDIR": str(self.home),
                                   "PROJECT_LEAP_SUPPORT_DIR": str(support), "PYTHONDONTWRITEBYTECODE": "1"},
                    capture_output=True, text=True, timeout=5, check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertRegex(completed.stderr, "Analysis Package|Run State|symlink")
                self.assertFalse(invocation.exists())
                self.assertEqual(self.snapshot(external), before)

    def test_main_resolves_root_above_analysis_package(self) -> None:
        module_path = self.root / "Analysis Package" / "project_leap_2d" / "workspace_launcher.py"
        with mock.patch.object(workspace_launcher, "__file__", str(module_path)), mock.patch.object(
            workspace_launcher, "_run_workspace", return_value=23
        ) as run:
            self.assertEqual(workspace_launcher.main(["--example", "保留 参数"]), 23)
        run.assert_called_once_with(self.root, ["--example", "保留 参数"])

    def test_workspace_routes_new_paths_preserving_order_and_locked_lifecycle(self) -> None:
        records = self.make_inputs(self.samples)
        events = []
        fake_runtime = types.SimpleNamespace()

        def discover(directory):
            self.assertEqual(directory, self.samples)
            events.append("discover")
            return {"DAPI": records[0].path}, []

        def original_publish(**kwargs):
            raise AssertionError("The scientific publisher must be replaced by this fixture")

        fake_runtime.discover_channel_paths = discover
        fake_runtime.publish_output_bundle = original_publish
        fake_workflow = types.ModuleType("project_leap_2d.analysis_workflow")

        def analysis(runtime, argv):
            events.append("analysis")
            self.assertEqual(argv, ["--input-dir", str(self.samples), "--output-dir", str(self.result), "--fixture"])
            with (self.state / "locks" / "workspace.lock").open("a+") as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            runtime.discover_channel_paths(self.samples)
            runtime.publish_output_bundle(staged_files={}, final_files={}, run_dir=self.state)
            return 0

        fake_workflow.run_analysis_workflow = analysis
        actual_nonempty = preflight.require_nonempty_original_image

        def nonempty(path):
            events.append("nonempty")
            actual_nonempty(path)

        with (
            mock.patch.dict(sys.modules, {"project_leap_2d.analysis_workflow": fake_workflow}),
            mock.patch.object(workspace_launcher, "recover_input_trash_transaction", side_effect=lambda **_: events.append("input-recovery")) as input_recover,
            mock.patch.object(workspace_launcher, "recover_publication_transaction", side_effect=lambda **_: events.append("publication-recovery")) as publication_recover,
            mock.patch.object(workspace_launcher, "require_nonempty_original_image", side_effect=nonempty),
            mock.patch.object(workspace_launcher, "archive_result_root_files", side_effect=lambda *_: events.append("pending")) as archive,
            mock.patch.object(workspace_launcher, "load_runtime", side_effect=lambda: events.append("load") or fake_runtime),
            mock.patch.object(workspace_launcher, "publish_with_publication_journal", side_effect=lambda **_: events.append("publish")) as publish,
            mock.patch.object(workspace_launcher, "move_inputs_to_macos_trash_recoverable", side_effect=lambda **_: events.append("trash") or self.home / ".Trash" / "fixture") as trash,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(workspace_launcher._run_workspace(self.root, ["--fixture"]), 0)
        self.assertEqual(events, ["input-recovery", "publication-recovery", "nonempty", "pending", "load", "analysis", "discover", "publish", "trash"])
        input_recover.assert_called_once_with(original_image=self.samples, runtime_root=self.state)
        publication_recover.assert_called_once_with(result_root=self.result, runtime_root=self.state)
        archive.assert_called_once_with(self.result, self.state)
        self.assertEqual(publish.call_args.kwargs["runtime_root"], self.state)
        self.assertEqual(publish.call_args.kwargs["result_root"], self.result)
        self.assertEqual(trash.call_args.kwargs["original_image"], self.samples)
        self.assertEqual(trash.call_args.kwargs["runtime_root"], self.state)
        self.assertEqual(trash.call_args.kwargs["snapshots"], (records[0],))
        self.assertFalse((self.root / "Runtime").exists())
        self.assertIs(fake_runtime.discover_channel_paths, discover)
        self.assertIs(fake_runtime.publish_output_bundle, original_publish)
        with preflight.project_workspace_lock(self.state):
            pass

    def test_packaged_path_overrides_remain_rejected(self) -> None:
        for argv in (["--input-dir", "/unused"], ["--output-dir=/unused"]):
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, "Sample Image and Result"):
                workspace_launcher.main(argv)

    def test_pending_archive_uses_nested_state_and_keeps_existing_names(self) -> None:
        for name in ("one.txt", "two.txt"):
            (self.result / name).write_bytes(name.encode())
        target = pending.archive_result_root_files(self.result, self.state)
        self.assertEqual(target, self.result / "Pending")
        self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, {"one.txt": b"one.txt", "two.txt": b"two.txt"})
        self.assertFalse(pending._transaction_record(self.state).exists())
        self.assertFalse((self.root / "Runtime").exists())

    def test_pending_partial_archive_recovers_from_nested_state(self) -> None:
        staging = self.state / "staging" / ("pending-" + "a" * 32)
        staging.mkdir(parents=True)
        (staging / "one.txt").write_bytes(b"one")
        (self.result / "two.txt").write_bytes(b"two")
        journal = pending._transaction_record(self.state)
        pending._write_record(journal, {
            "result": str(self.result), "staging": str(staging),
            "target": str(self.result / "Pending"), "names": ["one.txt", "two.txt"],
        })
        pending._recover_interrupted_archive(self.result, self.state)
        self.assertEqual((self.result / "one.txt").read_bytes(), b"one")
        self.assertEqual((self.result / "two.txt").read_bytes(), b"two")
        self.assertFalse(staging.exists())
        self.assertFalse(journal.exists())

    def test_pending_old_or_misplaced_paths_preserve_journal_and_files(self) -> None:
        for index, condition in enumerate(("result", "old-state", "staging-name", "target-parent", "target-name")):
            with self.subTest(condition=condition):
                root, _, result, state = self.make_layout(f"pending-invalid-{index}")
                staging = state / "staging" / ("pending-" + "b" * 32)
                staging.mkdir(parents=True)
                (staging / "one.txt").write_bytes(b"preserve")
                target = result / "Pending"
                target.mkdir()
                payload = {"result": str(result), "staging": str(staging), "target": str(target), "names": ["one.txt"]}
                if condition == "result":
                    payload["result"] = str(root / "old-Result")
                elif condition == "old-state":
                    payload["staging"] = str(root / "Runtime" / "staging" / staging.name)
                elif condition == "staging-name":
                    payload["staging"] = str(staging.with_name("unrelated"))
                elif condition == "target-parent":
                    payload["target"] = str(root / "old-Result" / "Pending")
                else:
                    payload["target"] = str(result / "unrelated")
                pending._write_record(pending._transaction_record(state), payload)
                before = self.snapshot(root)
                with self.assertRaisesRegex(RuntimeError, "journal does not match"):
                    pending._recover_interrupted_archive(result, state)
                self.assertEqual(self.snapshot(root), before)

    def test_input_cleanup_commits_only_into_temporary_trash(self) -> None:
        records = self.make_inputs(self.samples)
        target = inputs.move_inputs_to_macos_trash_recoverable(
            original_image=self.samples, runtime_root=self.state, snapshots=records,
        )
        self.assertEqual(target.parent, self.home / ".Trash")
        for record in records:
            self.assertFalse(record.path.exists())
            self.assertEqual((target / record.path.name).read_bytes(), f"synthetic-input-{records.index(record)}".encode())
        self.assertFalse(inputs.input_trash_journal_path(self.state).exists())

    def test_input_cleanup_partial_move_recovers_under_nested_state(self) -> None:
        records = self.make_inputs(self.samples)
        staging = inputs.begin_input_trash_transaction(
            original_image=self.samples, runtime_root=self.state,
            target=self.home / ".Trash" / "interrupted", snapshots=records,
        )
        os.replace(records[0].path, staging / records[0].path.name)
        self.assertTrue(inputs.recover_input_trash_transaction(original_image=self.samples, runtime_root=self.state))
        for record in records:
            record.verify_unchanged()
        self.assertFalse(inputs.input_trash_journal_path(self.state).exists())
        self.assertFalse(staging.exists())

    def test_input_cleanup_old_input_or_state_paths_are_rejected_without_writes(self) -> None:
        for index, field in enumerate(("original_image", "staging")):
            with self.subTest(field=field):
                root, samples, _, state = self.make_layout(f"input-invalid-{index}")
                records = self.make_inputs(samples)
                staging = inputs.begin_input_trash_transaction(
                    original_image=samples, runtime_root=state,
                    target=self.home / ".Trash" / f"old-input-{index}", snapshots=records,
                )
                os.replace(records[0].path, staging / records[0].path.name)
                journal = inputs.input_trash_journal_path(state)
                payload = json.loads(journal.read_text())
                payload[field] = str(root / "Original Image") if field == "original_image" else str(root / "Runtime" / "recovery" / "input_trash_staging")
                journal.write_text(json.dumps(payload), encoding="utf-8")
                before = self.snapshot(self.temporary)
                with self.assertRaisesRegex(RuntimeError, "another Sample Image|staging path is invalid"):
                    inputs.recover_input_trash_transaction(original_image=samples, runtime_root=state)
                self.assertEqual(self.snapshot(self.temporary), before)

    def test_publication_commit_and_recovery_use_nested_state(self) -> None:
        _, result, state, run, staged, final = self.make_publication()
        publication.publish_with_publication_journal(
            publish_callback=self.publisher, call_args=(),
            call_kwargs={"staged_files": staged, "final_files": final, "run_dir": run},
            result_root=result, runtime_root=state,
        )
        for key, path in final.items():
            self.assertEqual(path.read_bytes(), f"new-{key}".encode())
        self.assertFalse(publication.publication_journal_path(state).exists())
        self.assertFalse((state / "recovery" / "publication_backups").exists())

    def test_publication_partial_replace_recovers_all_five_files(self) -> None:
        _, result, state, _, staged, final = self.make_publication()
        self.begin_publication(result, state, staged, final)
        final["whole"].write_bytes(staged["whole"].read_bytes())
        self.assertTrue(publication.recover_publication_transaction(result_root=result, runtime_root=state))
        for key, path in final.items():
            self.assertEqual(path.read_bytes(), f"old-{key}".encode())
        self.assertFalse(publication.publication_journal_path(state).exists())

    def test_publication_old_result_or_backup_paths_are_rejected_without_writes(self) -> None:
        for index, field in enumerate(("result_root", "backup_root")):
            with self.subTest(field=field):
                root, result, state, _, staged, final = self.make_publication(f"publish-invalid-{index}")
                journal, payload = self.begin_publication(result, state, staged, final)
                final["whole"].write_bytes(staged["whole"].read_bytes())
                payload[field] = str(root / "old-Result") if field == "result_root" else str(root / "Runtime" / "recovery" / "publication_backups")
                journal.write_text(json.dumps(payload), encoding="utf-8")
                before = self.snapshot(root)
                with self.assertRaisesRegex(RuntimeError, "different Result|invalid backup location"):
                    publication.recover_publication_transaction(result_root=result, runtime_root=state)
                self.assertEqual(self.snapshot(root), before)


if __name__ == "__main__":
    unittest.main()
