"""Exception-safe temporary replacements on the shared runtime object."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


_MISSING = object()


@contextmanager
def temporary_runtime_attributes(
    runtime: object,
    /,
    **replacements: Any,
) -> Iterator[object]:
    """Temporarily replace attributes and restore their exact prior objects."""

    originals: list[tuple[str, object]] = []
    try:
        for name, replacement in replacements.items():
            original = getattr(runtime, name, _MISSING)
            originals.append((name, original))
            setattr(runtime, name, replacement)
        yield runtime
    finally:
        for name, original in reversed(originals):
            if original is _MISSING:
                if hasattr(runtime, name):
                    delattr(runtime, name)
            else:
                setattr(runtime, name, original)
