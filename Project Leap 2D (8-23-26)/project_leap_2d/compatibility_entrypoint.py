from __future__ import annotations

import sys
import types
from pathlib import Path

_package_parent = str(Path(__file__).resolve().parents[1])
if _package_parent not in sys.path:
    sys.path.insert(0, _package_parent)
from project_leap_2d.runtime_loader import load_runtime


_runtime = load_runtime()


class _RuntimeProxy(types.ModuleType):
    def __getattr__(self, name: str):
        return getattr(_runtime, name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_proxy_") or name in {
            "__class__",
            "__dict__",
            "__doc__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
        }:
            super().__setattr__(name, value)
            return
        setattr(_runtime, name, value)

    def __delattr__(self, name: str) -> None:
        if hasattr(_runtime, name):
            delattr(_runtime, name)
            return
        super().__delattr__(name)


sys.modules[__name__].__class__ = _RuntimeProxy
