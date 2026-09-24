# This functional source module is assembled into one shared runtime.
from __future__ import annotations

@contextmanager
def analysis_lock():
    cache_root = Path.home() / "Library" / "Caches" / "IHC2DAnalysis"
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "analysis.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another IHC 2D Analysis process is already running. Finish or stop it before starting a new run."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
