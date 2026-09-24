from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> "InputSnapshot":
        stat = path.stat()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Input is not a regular file: {path.name}")
        return cls(
            path=path,
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )

    def verify_unchanged(self) -> None:
        current = self.capture(self.path)
        if current != self:
            raise RuntimeError(
                f"Input changed during analysis and was retained: {self.path.name}"
            )


def _available_trash_path(trash_root: Path) -> Path:
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    base = trash_root / f"Project Leap 2D Original Images {timestamp}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = trash_root / (
            f"Project Leap 2D Original Images {timestamp} {suffix}"
        )
        if not candidate.exists():
            return candidate
        suffix += 1
