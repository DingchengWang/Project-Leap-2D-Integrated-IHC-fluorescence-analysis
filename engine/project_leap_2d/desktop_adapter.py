"""Single-run desktop entry point around the unchanged scientific workflow."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

from .desktop_events import DesktopEvents, active_events, atomic_json
from .desktop_paths import (AnalysisAlreadyRunning, analysis_worker_lock, component_usage_lock,
                            configure_worker_paths, fiji_process_environment,
                            reject_protected_path, require_absolute_real_path, validate_request_paths)


ROI_ARCHIVE_FILENAME = "IHC_2D_ROI_Archive.zip"
SHARED_CELLPOSE_DEPENDENCY = {
    "id": "cellpose-model", "version": "4.2.1.1", "filename": "cpsam_v2",
    "sha256": "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
}


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def runtime_provenance(request: dict, paths: dict) -> dict:
    """Capture the actual code and installed model identity before analysis."""
    code_root = Path(__file__).resolve().parent
    component_root = require_absolute_real_path(os.environ["PROJECT_LEAP_COMPONENT_ROOT"], "Component root")
    descriptor_path = component_root / "runtime.json"
    content_manifest_path = component_root / "content-manifest.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    component_manifest = json.loads(content_manifest_path.read_text(encoding="utf-8"))
    model_root = require_absolute_real_path(os.environ.get("PROJECT_LEAP_MODEL_ROOT", component_root), "Model component root")
    reject_protected_path(model_root, "Model component root")
    cellpose = require_absolute_real_path(Path(os.environ["CELLPOSE_LOCAL_MODELS_PATH"]) / "cpsam_v2", "Cellpose model")
    relative_model = cellpose.relative_to(model_root).as_posix()
    model_manifest = component_manifest
    shared_model = None
    if model_root != component_root:
        dependency = SHARED_CELLPOSE_DEPENDENCY
        if descriptor.get("schema_version") != 3 or descriptor.get("model_component") != dependency:
            raise ValueError("Module does not declare the pinned shared Cellpose dependency")
        model_descriptor_path = require_absolute_real_path(model_root / "runtime.json", "Model descriptor")
        model_manifest_path = require_absolute_real_path(model_root / "content-manifest.json", "Model manifest")
        model_descriptor = json.loads(model_descriptor_path.read_text(encoding="utf-8"))
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if (model_descriptor.get("schema_version") != 2
                or model_descriptor.get("component_id") != dependency["id"]
                or model_descriptor.get("component_version") != dependency["version"]
                or model_descriptor.get("models") != "Models"
                or model_descriptor.get("model_file") != dependency["filename"]
                or model_descriptor.get("model_sha256") != dependency["sha256"]
                or model_manifest.get("component_id") != dependency["id"]
                or model_manifest.get("component_version") != dependency["version"]
                or relative_model != "Models/" + dependency["filename"]):
            raise ValueError("Shared Cellpose component does not match the pinned dependency")
        if model_manifest.get("entries", {}).get("runtime.json", {}).get("sha256") != _fingerprint(model_descriptor_path)["sha256"]:
            raise ValueError("Shared Cellpose descriptor differs from its component manifest")
        shared_model = {"root": str(model_root), "descriptor": model_descriptor,
                        "component_manifest": _fingerprint(model_manifest_path)}
    model_entry = model_manifest["entries"][relative_model]
    if model_entry.get("type") != "file" or cellpose.stat().st_size != model_entry.get("bytes"):
        raise ValueError("Installed Cellpose model differs from its component manifest")
    if shared_model and (model_entry.get("sha256") != SHARED_CELLPOSE_DEPENDENCY["sha256"]
                         or _fingerprint(cellpose)["sha256"] != SHARED_CELLPOSE_DEPENDENCY["sha256"]):
        raise ValueError("Shared Cellpose model SHA-256 differs from the pinned dependency")
    files = sorted(path for path in code_root.rglob("*")
                   if path.is_file() and path.suffix in {".py", ".groovy", ".java", ".jar"})
    code_fingerprints = {path.relative_to(code_root).as_posix(): _fingerprint(path) for path in files}
    inputs = []
    for path in sorted((paths["workspace"] / "Sample Image").iterdir()):
        if path.suffix.lower() in {".tif", ".tiff"}:
            require_absolute_real_path(path, "Working input")
            stat = path.stat()
            inputs.append({"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "schema_version": 1, "run_id": request["run_id"],
        "request_provenance": request.get("provenance", {}),
        "working_inputs": inputs,
        "engine": {"version": (code_root.parent / "VERSION").read_text().strip(),
                   "code_files": code_fingerprints},
        "runtime": {"descriptor": descriptor, "component_manifest": _fingerprint(content_manifest_path),
                    "python_executable": str(Path(sys.executable).resolve()),
                    "fiji_java_tool_options": fiji_process_environment()["JAVA_TOOL_OPTIONS"]},
        "models": {
            "cellpose": {"path": str(cellpose), "size_bytes": model_entry["bytes"],
                         "sha256": model_entry["sha256"],
                         "hash_source": "shared_component_model_bytes" if shared_model else "installed_component_content_manifest",
                         "rehashed_this_run": bool(shared_model)},
            "instanseg": {"path": str(code_root / "resources/models/instanseg_single_channel_nuclei.pt"),
                          **_fingerprint(code_root / "resources/models/instanseg_single_channel_nuclei.pt"),
                          "hash_source": "packaged_model_bytes", "rehashed_this_run": True},
        },
    }
    if shared_model:
        payload["runtime"]["cellpose_model"] = shared_model
    fiji_root = Path(os.environ.get("PROJECT_LEAP_FIJI_ROOT", component_root))
    if fiji_root != component_root:
        payload["runtime"]["fiji"] = {
            "descriptor": json.loads((fiji_root / "runtime.json").read_text(encoding="utf-8")),
            "component_manifest": _fingerprint(fiji_root / "content-manifest.json"),
            "launcher": request["fiji_launcher"],
        }
    provenance = request.get("provenance", {})
    import_path_value = provenance.get("import_result_path")
    if import_path_value:
        import_path = require_absolute_real_path(import_path_value, "Import result")
        if paths["run_root"] not in import_path.parents:
            raise ValueError("Import result must belong to this task")
        payload["import_result"] = json.loads(import_path.read_text(encoding="utf-8"))
    return payload


def validate_request(request: object) -> dict:
    if not isinstance(request, dict) or type(request.get("schema_version")) is not int or request["schema_version"] != 1:
        raise ValueError("Analysis request schema_version must be 1")
    if not isinstance(request.get("run_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", request["run_id"]):
        raise ValueError("run_id must contain only letters, numbers, hyphens and underscores")
    if request.get("module_id", "project-leap-2d") != "project-leap-2d":
        raise ValueError("This analysis worker only runs the project-leap-2d module")
    arguments = request.get("engine_args", [])
    if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
        raise ValueError("engine_args must be a list of strings")
    # The desktop adapter owns all writable paths and scientific settings.
    # The existing review timeout is the only forwarded user-facing option.
    if arguments:
        if len(arguments) != 2 or arguments[0] != "--fiji-timeout-minutes":
            raise ValueError("Only --fiji-timeout-minutes may be passed in engine_args")
        import math
        timeout = float(arguments[1])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Fiji timeout must be positive and finite")
    if not isinstance(request.get("provenance", {}), dict):
        raise ValueError("provenance must be a JSON object")
    return request


def _run_scientific_workflow(request: dict, paths: dict, launcher: Path, events: DesktopEvents) -> tuple[int, dict[str, str]]:
    # Import only after writable caches and managed models have been configured.
    from .runtime_loader import load_runtime
    from .runtime_attributes import temporary_runtime_attributes
    from .workspace.pending_results import archive_result_root_files
    from .workspace.publication_recovery import publish_with_publication_journal, recover_publication_transaction
    from .workspace.workspace_preflight import (
        project_workspace_lock, require_nonempty_original_image, validate_workspace_layout,
    )

    workspace = paths["workspace"]
    result_root = workspace / "Result"
    runtime_root = workspace / "Analysis Package" / "Run State"
    validate_workspace_layout(workspace)
    artifacts = {}
    with project_workspace_lock(runtime_root):
        events.check_cancelled()
        recover_publication_transaction(result_root=result_root, runtime_root=runtime_root)
        require_nonempty_original_image(workspace / "Sample Image")
        archive_result_root_files(result_root, runtime_root)
        events.emit("stage", "loading_engine", "Loading analysis components")
        runtime = load_runtime()
        provenance = runtime_provenance(request, paths)
        from .analysis_workflow import run_analysis_workflow
        original_publish = runtime.publish_output_bundle

        def desktop_publish(*args, **kwargs):
            from .final_roi_archive import build_final_roi_archive
            events.check_cancelled()
            events.emit("stage", "publishing", "Saving validated results and final ROIs")
            bound = inspect.signature(original_publish).bind(*args, **kwargs)
            run_dir = Path(bound.arguments["run_dir"])
            staged = dict(bound.arguments["staged_files"])
            final = dict(bound.arguments["final_files"])
            archive_path = run_dir / ROI_ARCHIVE_FILENAME
            atomic_json(run_dir / "runtime_provenance.json", provenance)
            build_final_roi_archive(run_dir, archive_path)
            staged["roi_archive"] = archive_path
            final["roi_archive"] = result_root / ROI_ARCHIVE_FILENAME
            bound.arguments["staged_files"] = staged
            bound.arguments["final_files"] = final
            events.check_cancelled()
            # Finish or roll back the existing publication transaction before
            # responding to cancellation; do not interrupt between replacements.
            events.critical = True
            try:
                value = publish_with_publication_journal(
                    publish_callback=original_publish,
                    call_args=bound.args, call_kwargs=bound.kwargs,
                    result_root=result_root, runtime_root=runtime_root,
                )
                artifacts.update({key: str(path) for key, path in final.items()})
                events.artifacts.update(artifacts)
                events.committed = True
                return value
            finally:
                events.critical = False

        arguments = ["--input-dir", str(workspace / "Sample Image"),
                     "--output-dir", str(result_root), "--fiji-launcher", str(launcher),
                     *request.get("engine_args", [])]
        with temporary_runtime_attributes(runtime, publish_output_bundle=desktop_publish):
            return_code = int(run_analysis_workflow(runtime, arguments))
        # Working inputs are retained under the task, never sent to macOS Trash.
        return return_code, artifacts


def run_request(request_path: Path) -> int:
    request_path = Path(request_path)
    request = validate_request(json.loads(request_path.read_text(encoding="utf-8")))
    paths = validate_request_paths(request, request_path)
    events = DesktopEvents(request["run_id"], paths["events_path"], paths["run_root"] / "cancel.request")
    status, message, error_code = "failed", "Analysis did not complete", "engine_not_published"
    artifacts = {}
    with active_events(events), ExitStack() as resources:
        try:
            atomic_json(paths["run_root"] / "worker.json", {
                "pid": os.getpid(), "kind": "analysis", "request_path": str(request_path.resolve()),
            })
            events.emit("stage", "preflight", "Checking isolated task paths and components")
            events.check_cancelled()
            resources.enter_context(analysis_worker_lock(paths["run_root"].parent.parent))
            resources.enter_context(component_usage_lock())
            launcher = configure_worker_paths(request, paths)
            return_code, artifacts = _run_scientific_workflow(request, paths, launcher, events)
            if events.committed:
                status, message, error_code = "completed", "Results and final ROIs were saved", None
                artifacts = dict(events.artifacts or artifacts)
            elif events.review_cancelled or events.cancelled:
                status, message, error_code = "cancelled", "Analysis was cancelled before publication", "user_cancelled"
            elif return_code == 2:
                message, error_code = "The analysis engine stopped at its input or quality checks; no results were published", "engine_check_stopped"
            elif return_code != 0:
                message, error_code = f"The analysis engine stopped with code {return_code}", "engine_failed"
        except KeyboardInterrupt:
            status, message, error_code = "cancelled", "Analysis was cancelled before publication", "user_cancelled"
        except AnalysisAlreadyRunning as exc:
            message, error_code = str(exc), "analysis_already_running"
        except Exception as exc:
            message, error_code = str(exc) or type(exc).__name__, "technical_failure"
            traceback.print_exc()
        finally:
            if events.committed:
                # A later cleanup failure cannot invalidate a committed result.
                if status != "completed":
                    events.emit("warning", "cleanup", message, code=error_code)
                status, message, error_code = "completed", "Results and final ROIs were saved", None
                artifacts = dict(events.artifacts or artifacts)
            events.terminate_owned_processes()
            outcome = {
                "schema_version": 1, "run_id": request["run_id"], "status": status,
                "module_id": request.get("module_id", "project-leap-2d"),
                "results_dir": str(paths["workspace"] / "Result") if events.committed else None,
                "message": message, "artifacts": artifacts,
                "processes": events.process_records(),
            }
            if error_code:
                outcome["error_code"] = error_code
            try:
                atomic_json(paths["outcome_path"], outcome)
                events.emit(status, "complete" if events.committed else events.stage,
                            message, status=status, **({"code": error_code} if error_code else {}))
            finally:
                events.close()
    return {"completed": 0, "cancelled": 130, "failed": 1}[status]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated Project Leap desktop task")
    parser.add_argument("--request", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        return run_request(arguments.request)
    except Exception as exc:
        # Invalid protocol paths cannot safely be used to write an outcome.
        print(f"Invalid analysis request: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
