from __future__ import annotations

import hashlib
import sys
import threading
import types
from pathlib import Path

from .runtime_manifest import (
    CANONICAL_SOURCE_SHA256,
    GROOVY_RESOURCE_SHA256,
    MODULE_ORDER,
    REQUIRED_RUNTIME_SYMBOLS,
)
_LOAD_LOCK = threading.RLock()
_RUNTIME_NAME = "project_leap_2d.runtime"
_RELEASE_DISPLAY_NAME = "Project Leap 2D"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_runtime() -> types.ModuleType:
    """Assemble the validated functional modules into one shared state space.

    The shared namespace is deliberate: caches, DAPI fragment diagnostics, worker
    settings, and Cellpose state must not be copied between imported modules.
    """

    with _LOAD_LOCK:
        existing = sys.modules.get(_RUNTIME_NAME)
        if existing is not None and bool(
            getattr(existing, "_PROJECT_LEAP_RUNTIME_READY", False)
        ):
            return existing

        code_root = Path(__file__).resolve().parent
        runtime = types.ModuleType(_RUNTIME_NAME)
        runtime.__dict__.update(
            {
                "__file__": str(code_root / "runtime_loader.py"),
                "__package__": "project_leap_2d",
                "_PROJECT_LEAP_CANONICAL_SOURCE_SHA256": CANONICAL_SOURCE_SHA256,
                "_PROJECT_LEAP_MODULE_ORDER": MODULE_ORDER,
            }
        )
        sys.modules[_RUNTIME_NAME] = runtime
        try:
            loaded_files: list[str] = []
            for relative in MODULE_ORDER:
                source_path = code_root / relative
                source_bytes = source_path.read_bytes()
                source_text = source_bytes.decode("utf-8")
                code = compile(source_text, str(source_path), "exec")
                exec(code, runtime.__dict__, runtime.__dict__)
                loaded_files.append(str(source_path))

            groovy_path = (
                code_root
                / "fiji_review"
                / "resources"
                / "astrocyte_roi_reviewer.groovy"
            )
            groovy_bytes = groovy_path.read_bytes()
            observed_groovy_sha = _sha256(groovy_bytes)
            if observed_groovy_sha != GROOVY_RESOURCE_SHA256:
                raise RuntimeError(
                    "Fiji reviewer resource integrity check failed: "
                    f"{observed_groovy_sha}"
                )
            runtime.FIJI_GROOVY_SCRIPT = groovy_bytes.decode("utf-8")
            runtime._PROJECT_LEAP_LOADED_FILES = tuple(loaded_files)
            missing = [
                name
                for name in REQUIRED_RUNTIME_SYMBOLS
                if name not in runtime.__dict__
            ]
            if missing:
                raise RuntimeError(
                    "Modular runtime is incomplete; missing symbols: "
                    + ", ".join(missing)
                )
            runtime.PRODUCT_DISPLAY_NAME = _RELEASE_DISPLAY_NAME
            runtime._PROJECT_LEAP_RUNTIME_READY = True
            return runtime
        except BaseException:
            sys.modules.pop(_RUNTIME_NAME, None)
            raise
