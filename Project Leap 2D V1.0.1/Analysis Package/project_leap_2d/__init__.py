"""Project Leap 2D package.

Importing this package is intentionally lightweight. Scientific dependencies
are loaded only after startup has fixed the BLAS/OpenMP thread environment.
"""

from __future__ import annotations

__all__ = ["load_runtime"]


def load_runtime():
    from .runtime_loader import load_runtime as _load_runtime

    return _load_runtime()
