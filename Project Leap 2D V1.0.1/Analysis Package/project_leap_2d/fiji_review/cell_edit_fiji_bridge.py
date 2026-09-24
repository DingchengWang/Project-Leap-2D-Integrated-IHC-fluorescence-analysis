"""Add optional Cell Edit support without changing the base Fiji preparation path."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .cell_editing import (
    CellEditRequestService,
    atomic_write_cell_edit_json,
    prepare_cell_edit_runtime,
)
from .cell_edit_context import (
    CellEditContextPaths,
    load_cell_edit_context,
    relocate_cell_edit_context,
)
from .fiji_launcher import terminate_fiji_process_group


BasePrepareCallback = Callable[..., tuple[Path, Path]]
CellEditContextBuilder = Callable[[Path], CellEditContextPaths]
_FIJI_RUN_DIRECTORY = re.compile(r"run-[0-9a-f]{32}")


def _fiji_cache_root() -> Path:
    return (Path.home() / "Library" / "Caches" / "IHC2DAnalysis").resolve()


def _known_fiji_runs(cache_root: Path) -> set[Path]:
    if not cache_root.is_dir():
        return set()
    return {
        child.resolve()
        for child in cache_root.iterdir()
        if (
            not child.is_symlink()
            and child.is_dir()
            and _FIJI_RUN_DIRECTORY.fullmatch(child.name) is not None
        )
    }


def _is_new_production_fiji_run(
    run_dir: Path,
    manifest_path: Path,
    *,
    cache_root: Path,
    runs_before_prepare: set[Path],
) -> bool:
    supplied = run_dir.expanduser().absolute()
    resolved = supplied.resolve()
    return bool(
        not supplied.is_symlink()
        and resolved.parent == cache_root
        and _FIJI_RUN_DIRECTORY.fullmatch(resolved.name) is not None
        and resolved not in runs_before_prepare
        and resolved.is_dir()
        and manifest_path.resolve() == resolved / "manifest.json"
    )


def _prepare_compatibility_context_path(
    *,
    run_dir: Path,
    manifest_path: Path,
    context_path: Path,
    cell_edit_timeout_seconds: float,
    enabled_cell_edit_actions: tuple[str, ...],
) -> tuple[Path, Path]:
    """Support staged-context compatibility with the same failure semantics."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("The base Fiji manifest must be a JSON object")
    context_destination = run_dir / "cell_edit"
    target_context_path = context_destination / "analysis_context.json"
    cell_edit_manifest = prepare_cell_edit_runtime(
        run_dir,
        context_path=target_context_path,
        enabled_actions=enabled_cell_edit_actions,
        timeout_seconds=cell_edit_timeout_seconds,
    )
    try:
        relocated = relocate_cell_edit_context(
            context_path,
            destination_dir=context_destination,
            basename="analysis_context",
        )
    except BaseException:
        for key in ("state_dir", "cancel_dir", "response_dir", "request_dir"):
            Path(cell_edit_manifest[key]).rmdir()
        context_destination.rmdir()
        raise
    cell_edit_manifest["context_path"] = str(relocated.json_path)
    cell_edit_manifest["program_root"] = str(Path(__file__).resolve().parents[2])
    manifest["cell_edit"] = cell_edit_manifest
    atomic_write_cell_edit_json(manifest_path, manifest)
    return run_dir, manifest_path


def prepare_cell_edit_fiji_runtime(
    *,
    base_prepare: BasePrepareCallback,
    cell_edit_context_builder: CellEditContextBuilder | None = None,
    cell_edit_context_path: Path | None = None,
    cleanup_unlaunched_run_on_failure: bool = False,
    cell_edit_timeout_seconds: float = 45.0,
    enabled_cell_edit_actions: tuple[str, ...] = ("split", "enlarge"),
    **base_prepare_kwargs: Any,
) -> tuple[Path, Path]:
    """Run the base Fiji preparation, then atomically enable Cell Edit.

    A missing context intentionally leaves the base Fiji manifest untouched, so
    the Fiji reviewer cannot expose actions that have no evidence bundle.
    """

    if cell_edit_context_builder is not None and cell_edit_context_path is not None:
        raise ValueError(
            "Provide either cell_edit_context_builder or cell_edit_context_path, "
            "not both"
        )
    context_path = (
        None
        if cell_edit_context_path is None
        else Path(cell_edit_context_path).expanduser().resolve()
    )
    cache_root = _fiji_cache_root()
    runs_before_prepare = (
        _known_fiji_runs(cache_root)
        if cell_edit_context_builder is not None
        and cleanup_unlaunched_run_on_failure
        else set()
    )
    run_dir, manifest_path = base_prepare(**base_prepare_kwargs)
    run_dir = Path(run_dir)
    manifest_path = Path(manifest_path)
    if cell_edit_context_builder is None and (
        context_path is None or not context_path.is_file()
    ):
        return run_dir, manifest_path
    if cell_edit_context_builder is None:
        return _prepare_compatibility_context_path(
            run_dir=run_dir,
            manifest_path=manifest_path,
            context_path=context_path,
            cell_edit_timeout_seconds=cell_edit_timeout_seconds,
            enabled_cell_edit_actions=enabled_cell_edit_actions,
        )

    owned_production_run = _is_new_production_fiji_run(
        run_dir,
        manifest_path,
        cache_root=cache_root,
        runs_before_prepare=runs_before_prepare,
    )
    if cleanup_unlaunched_run_on_failure and not owned_production_run:
        raise ValueError(
            "Refusing destructive cleanup for an unrecognized Fiji runtime directory"
        )

    context_destination = (run_dir / "cell_edit").resolve()
    target_context_path = context_destination / "analysis_context.json"
    context_destination_preexisted = context_destination.exists()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("The base Fiji manifest must be a JSON object")
        if context_destination.exists():
            raise ValueError(
                "The Fiji runtime already contains a cell_edit directory"
            )
        context_paths = cell_edit_context_builder(context_destination)
        if (
            Path(context_paths.npz_path).resolve()
            != target_context_path.with_suffix(".npz").resolve()
            or Path(context_paths.json_path).resolve()
            != target_context_path.resolve()
        ):
            raise ValueError(
                "The Cell Edit context builder did not write the final "
                "analysis_context pair"
            )
        committed = load_cell_edit_context(
            target_context_path,
            verify_sources=True,
            verify_source_hashes=True,
        )
        if (
            committed.npz_path.resolve()
            != target_context_path.with_suffix(".npz").resolve()
            or committed.json_path.resolve() != target_context_path.resolve()
        ):
            raise ValueError(
                "The final Cell Edit context path changed during validation"
            )
        del committed
        cell_edit_manifest = prepare_cell_edit_runtime(
            run_dir,
            context_path=target_context_path,
            enabled_actions=enabled_cell_edit_actions,
            timeout_seconds=cell_edit_timeout_seconds,
        )
        cell_edit_manifest["context_path"] = str(target_context_path)
        cell_edit_manifest["program_root"] = str(Path(__file__).resolve().parents[2])
        manifest["cell_edit"] = cell_edit_manifest
        atomic_write_cell_edit_json(manifest_path, manifest)
    except BaseException:
        if cleanup_unlaunched_run_on_failure and owned_production_run:
            shutil.rmtree(run_dir, ignore_errors=True)
        elif not context_destination_preexisted:
            shutil.rmtree(context_destination, ignore_errors=True)
        raise
    return run_dir, manifest_path


def _cell_edit_dispatcher() -> Callable[[dict, dict], dict]:
    """Resolve the worker lazily so normal analysis never loads its models."""

    from .cell_edit_worker import dispatch_cell_edit_request

    return dispatch_cell_edit_request


def launch_cell_edit_fiji_workflow(
    *,
    launcher: Path,
    run_dir: Path,
    manifest_path: Path,
    timeout_minutes: float,
) -> dict:
    """Launch Fiji with an optional single-flight Cell Edit request service."""

    run_dir = Path(run_dir)
    manifest_path = Path(manifest_path)
    script_path = run_dir / "ihc_fiji_bridge.groovy"
    ready_path = run_dir / "fiji_ready.json"
    done_path = run_dir / "fiji_done.json"
    error_path = run_dir / "fiji_error.txt"
    log_path = run_dir / "fiji_console.log"
    command = [
        str(launcher),
        "--memory",
        "16G",
        "--allow-multiple",
        "--no-splash",
        "--run",
        str(script_path),
        "manifest='" + str(manifest_path).replace("'", "\\'") + "'",
    ]
    print("Launching Fiji with 16 GB heap...", flush=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process: subprocess.Popen | None = None
    cell_edit_service: CellEditRequestService | None = None
    workflow_returned = False
    try:
        process = subprocess.Popen(
            command,
            cwd=str(run_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cell_edit = manifest.get("cell_edit", {})
        if isinstance(cell_edit, dict) and cell_edit.get("enabled_actions"):
            cell_edit_service = CellEditRequestService(
                manifest=manifest,
                dispatcher=_cell_edit_dispatcher(),
            )

        deadline = time.monotonic() + float(timeout_minutes) * 60.0
        ready_announced = False
        ready_seconds: float | None = None
        started_at = time.monotonic()
        while time.monotonic() < deadline:
            if cell_edit_service is not None:
                cell_edit_service.poll()
            if error_path.exists():
                detail = error_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                raise RuntimeError(f"Fiji workflow failed:\n{detail}")
            if ready_path.exists() and not ready_announced:
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                print(
                    f"Fiji is ready at {ready.get('stage', 'measurement')}: "
                    f"{ready['roi_count']} Whole ROIs. "
                    "Six ROI windows are open and waiting for the review decision.",
                    flush=True,
                )
                ready_seconds = float(time.monotonic() - started_at)
                ready_announced = True
            if done_path.exists():
                details = json.loads(done_path.read_text(encoding="utf-8"))
                details["fiji_startup_seconds"] = ready_seconds
                details["fiji_total_seconds"] = float(
                    time.monotonic() - started_at
                )
                workflow_returned = True
                return details
            if not ready_path.exists() and time.monotonic() - started_at > 8.0:
                console = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                if "[ERROR]" in console or "Exception" in console:
                    raise RuntimeError(
                        "Fiji failed before the workflow initialized:\n"
                        f"{console}"
                    )
            if process.poll() is not None and not done_path.exists():
                time.sleep(1.0)
                if not done_path.exists():
                    console = log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    raise RuntimeError(
                        f"Fiji exited with code {process.returncode} "
                        f"before completion.\n{console}"
                    )
            time.sleep(0.75)
        raise TimeoutError(
            "Fiji display and measurement did not finish within "
            f"{timeout_minutes:g} minutes. Runtime diagnostics were kept at "
            f"{run_dir}"
        )
    finally:
        if cell_edit_service is not None:
            cell_edit_service.close()
        if not workflow_returned and process is not None:
            terminate_fiji_process_group(process)
        log_handle.close()
