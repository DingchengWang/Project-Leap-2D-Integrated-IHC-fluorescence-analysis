"""Explicit writable paths for the desktop worker; no production fallback."""
from __future__ import annotations

import os
import sys
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping


class AnalysisAlreadyRunning(RuntimeError):
    pass


def require_absolute_real_path(value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"{label} must not contain symlinks: {parent}")
    return path.resolve()


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def reject_protected_path(path: Path, label: str) -> None:
    protected = (
        Path.home() / "Desktop",
        Path.home() / "Applications" / "Project Leap 2D Support",
        Path.home() / "Library" / "Caches" / "IHC2DAnalysis",
    )
    if any(_within(path, root.resolve()) for root in protected):
        raise ValueError(f"{label} points into a protected production location: {path}")


def fiji_cache_root() -> Path:
    """Retain the script default only outside desktop-worker mode."""
    configured = os.environ.get("PROJECT_LEAP_CACHE_ROOT")
    if configured:
        root = require_absolute_real_path(configured, "PROJECT_LEAP_CACHE_ROOT")
        reject_protected_path(root, "PROJECT_LEAP_CACHE_ROOT")
        return root
    if os.environ.get("PROJECT_LEAP_DESKTOP_WORKER") == "1":
        raise RuntimeError("Desktop worker requires PROJECT_LEAP_CACHE_ROOT")
    return (Path.home() / "Library" / "Caches" / "IHC2DAnalysis").resolve()


def _java_quote(value: str) -> str:
    # HotSpot's JAVA_TOOL_OPTIONS tokenizer handles either quote style, including
    # spaces inside it. It does not use shell backslash escaping.
    if '"' not in value:
        return '"' + value + '"'
    if "'" not in value:
        return "'" + value + "'"
    raise ValueError("Java runtime paths cannot contain both single and double quotes")


def fiji_process_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if environment.get("PROJECT_LEAP_DESKTOP_WORKER") != "1":
        return environment
    cache = fiji_cache_root()
    directories = {
        "user.home": cache / "java-home",
        "java.util.prefs.userRoot": cache / "java-prefs" / "user",
        "java.util.prefs.systemRoot": cache / "java-prefs" / "system",
        "java.io.tmpdir": cache / "java-tmp",
    }
    for label, directory in directories.items():
        require_absolute_real_path(directory, label)
        directory.mkdir(parents=True, exist_ok=True)
    imagej_preferences = directories["user.home"] / "Library" / "Preferences"
    require_absolute_real_path(imagej_preferences, "ImageJ preferences")
    imagej_preferences.mkdir(parents=True, exist_ok=True)
    preferences_jar = Path(__file__).resolve().parent / "fiji_review/resources/desktop-preferences.jar"
    if not preferences_jar.is_file():
        raise FileNotFoundError(f"Desktop Java preferences adapter is missing: {preferences_jar}")
    arguments = ["-D" + key + "=" + _java_quote(str(value)) for key, value in directories.items()]
    arguments.extend([
        "-Djava.util.prefs.PreferencesFactory=org.projectleap.desktop.TaskPreferencesFactory",
        "-Xbootclasspath/a:" + _java_quote(str(preferences_jar)),
    ])
    # These variables can otherwise override paths or JVM settings supplied by
    # the app. OS HOME and the existing scientific heap settings are unchanged.
    for variable in ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"):
        environment.pop(variable, None)
    environment["JAVA_TOOL_OPTIONS"] = " ".join(arguments)
    return environment


@contextmanager
def analysis_worker_lock(app_home: Path):
    path = require_absolute_real_path(app_home / "analysis.lock", "Analysis lock")
    reject_protected_path(path, "Analysis lock")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AnalysisAlreadyRunning("Another analysis is already running in this application workspace") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def component_usage_lock():
    path = require_absolute_real_path(os.environ.get("PROJECT_LEAP_COMPONENT_LOCK"), "PROJECT_LEAP_COMPONENT_LOCK")
    reject_protected_path(path, "Component lock")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Analysis components are being installed or repaired") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_request_paths(request: Mapping[str, object], request_path: Path) -> dict[str, Path]:
    request_path = require_absolute_real_path(request_path, "request")
    if request_path.name != "request.json" or not request_path.is_file():
        raise ValueError("Worker request must be an existing request.json file")
    run_root = request_path.parent
    reject_protected_path(run_root, "Task directory")
    paths = {"run_root": run_root}
    for name, relative in (
        ("workspace", "workspace"), ("cache_root", "cache"),
        ("events_path", "events.jsonl"), ("outcome_path", "outcome.json"),
    ):
        path = require_absolute_real_path(request.get(name), name)
        if path != run_root / relative:
            raise ValueError(f"{name} must be {run_root / relative}")
        paths[name] = path
    if paths["outcome_path"].exists():
        raise ValueError("This task already has an outcome; create a new run_id")
    if paths["events_path"].exists() and paths["events_path"].stat().st_size:
        raise ValueError("This task already has events; create a new run_id")
    return paths


def configure_worker_paths(request: Mapping[str, object], paths: Mapping[str, Path]) -> Path:
    component_root = require_absolute_real_path(
        os.environ.get("PROJECT_LEAP_COMPONENT_ROOT"), "PROJECT_LEAP_COMPONENT_ROOT"
    )
    reject_protected_path(component_root, "Component root")
    models = require_absolute_real_path(
        os.environ.get("CELLPOSE_LOCAL_MODELS_PATH"), "CELLPOSE_LOCAL_MODELS_PATH"
    )
    model_root = require_absolute_real_path(
        os.environ.get("PROJECT_LEAP_MODEL_ROOT", component_root), "PROJECT_LEAP_MODEL_ROOT"
    )
    reject_protected_path(model_root, "Model component root")
    launcher = require_absolute_real_path(request.get("fiji_launcher"), "fiji_launcher")
    # Desktop installations share Fiji independently of the scientific module.
    # The existing reference runner still uses its original combined runtime.
    fiji_root = require_absolute_real_path(
        os.environ.get("PROJECT_LEAP_FIJI_ROOT", component_root), "PROJECT_LEAP_FIJI_ROOT"
    )
    reject_protected_path(fiji_root, "Fiji component root")
    if not _within(models, model_root):
        raise ValueError("Models must belong to the isolated model component root")
    if not _within(launcher, fiji_root):
        raise ValueError("Fiji launcher must belong to the isolated Fiji component root")
    if not _within(Path(sys.executable).resolve(), component_root):
        raise ValueError("Scientific Python must belong to the isolated component root")
    cellpose = require_absolute_real_path(models / "cpsam_v2", "Cellpose model")
    if not cellpose.is_file():
        raise FileNotFoundError(f"The installed Cellpose model is missing: {cellpose}")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise FileNotFoundError(f"The installed Fiji launcher is not executable: {launcher}")
    # Set before importing scientific packages, including Matplotlib and Cellpose.
    os.environ["PROJECT_LEAP_DESKTOP_WORKER"] = "1"
    os.environ["PROJECT_LEAP_CACHE_ROOT"] = str(paths["cache_root"])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    paths["cache_root"].mkdir(parents=True, exist_ok=True)
    for variable, name in (
        ("MPLCONFIGDIR", "matplotlib"), ("TORCH_HOME", "torch"),
        ("XDG_CACHE_HOME", "xdg"), ("TMPDIR", "tmp"),
    ):
        destination = paths["cache_root"] / name
        require_absolute_real_path(destination, variable)
        destination.mkdir(exist_ok=True)
        os.environ[variable] = str(destination)
    # Validate and prepare Java paths before any scientific computation starts.
    fiji_process_environment()
    return launcher
