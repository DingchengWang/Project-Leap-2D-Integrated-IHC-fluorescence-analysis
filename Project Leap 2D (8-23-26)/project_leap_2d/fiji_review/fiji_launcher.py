# This functional source module is assembled into one shared runtime.
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def terminate_fiji_process_group(
    process: subprocess.Popen,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Stop only the Fiji process group launched for the current analysis."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def find_fiji_launcher(explicit: Path | None = None) -> Path:
    candidates = [
        explicit.expanduser() if explicit else None,
        Path("/Applications/Fiji/fiji"),
        Path("/Applications/Fiji/Fiji.app/Contents/MacOS/fiji-macos-arm64"),
        Path("/Applications/Fiji/Fiji.app/Contents/MacOS/fiji-macos"),
        Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Fiji launcher not found. Expected /Applications/Fiji/fiji or an explicit --fiji-launcher."
    )

def launch_fiji_workflow(
    *,
    launcher: Path,
    run_dir: Path,
    manifest_path: Path,
    timeout_minutes: float,
) -> dict:
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
    try:
        process = subprocess.Popen(
            command,
            cwd=str(run_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_minutes * 60.0
        ready_announced = False
        ready_seconds: float | None = None
        started_at = time.monotonic()
        while time.monotonic() < deadline:
            if error_path.exists():
                detail = error_path.read_text(encoding="utf-8", errors="replace")
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
                details["fiji_total_seconds"] = float(time.monotonic() - started_at)
                return details
            if not ready_path.exists() and time.monotonic() - started_at > 8.0:
                console = log_path.read_text(encoding="utf-8", errors="replace")
                if "[ERROR]" in console or "Exception" in console:
                    raise RuntimeError(f"Fiji failed before the workflow initialized:\n{console}")
            if process.poll() is not None and not done_path.exists():
                time.sleep(1.0)
                if not done_path.exists():
                    console = log_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(
                        f"Fiji exited with code {process.returncode} before completion.\n{console}"
                    )
            time.sleep(0.75)
        raise TimeoutError(
            f"Fiji display and measurement did not finish within {timeout_minutes:g} minutes. "
            f"Runtime diagnostics were kept at {run_dir}"
        )
    finally:
        log_handle.close()
