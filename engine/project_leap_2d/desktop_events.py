"""Structured events and owned-process bookkeeping, independent of stdout."""
from __future__ import annotations

import json
import os
import signal
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


_active = None
TERMINAL_TYPES = frozenset({"completed", "cancelled", "failed"})
EVENT_TYPES = TERMINAL_TYPES | {"stage", "progress", "warning", "review_ready", "process"}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class DesktopEvents:
    def __init__(self, run_id: str, events_path: Path, cancel_path: Path):
        self.run_id = run_id
        self.events_path = events_path
        self.cancel_path = cancel_path
        self.sequence = 0
        self.stage = "preflight"
        self.cancelled = False
        self.review_cancelled = False
        self.committed = False
        self.critical = False
        self.terminal = False
        self.processes = {}
        self.artifacts = {}
        self._lock = threading.RLock()
        self._handle = events_path.open("a", encoding="utf-8")

    def emit(self, event_type: str, stage: str, message: str, **details) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown desktop event type: {event_type}")
        with self._lock:
            if self.terminal:
                return
            self.sequence += 1
            self.stage = stage
            payload = {
                "schema_version": 1, "run_id": self.run_id,
                "sequence": self.sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type, "stage": stage, "message": message,
            }
            payload.update(details)
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._handle.flush()
            self.terminal = event_type in TERMINAL_TYPES

    def check_cancelled(self) -> None:
        if self.cancel_path.exists():
            self.cancelled = True
        if self.cancelled and not self.critical and not self.committed:
            raise KeyboardInterrupt("Analysis cancelled")

    def signal_cancel(self, _signum, _frame) -> None:
        self.cancelled = True
        self.check_cancelled()

    def register_process(self, process, role: str) -> None:
        with self._lock:
            self.processes[process.pid] = (process, role)
        self.emit("process", "fiji_starting" if role == "fiji" else "cell_edit",
                  f"Started {role}", pid=process.pid, pgid=process.pid,
                  process_type=role, process_status="started")
        if self.cancelled:
            from .fiji_review.fiji_launcher import terminate_fiji_process_group
            terminate_fiji_process_group(process)

    def process_records(self) -> list[dict]:
        with self._lock:
            return [{"pid": process.pid, "pgid": process.pid, "process_type": role, "running": process.poll() is None}
                    for process, role in self.processes.values()]

    def terminate_owned_processes(self) -> None:
        # Popen objects are registered only immediately after start_new_session=True.
        from .fiji_review.fiji_launcher import terminate_fiji_process_group
        self.critical = True
        try:
            for process, role in tuple(self.processes.values()):
                if process.poll() is None:
                    terminate_fiji_process_group(process)
                self.emit("process", "cleanup", f"Stopped {role}",
                          pid=process.pid, pgid=process.pid, process_type=role,
                          process_status="exited" if process.poll() is not None else "running")
        finally:
            self.critical = False

    def close(self) -> None:
        self._handle.close()


@contextmanager
def active_events(events: DesktopEvents):
    global _active
    previous = _active
    _active = events
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in old_handlers:
            signal.signal(sig, events.signal_cancel)
        yield events
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        _active = previous


def emit_event(event_type: str, stage: str, message: str, **details) -> None:
    if _active is not None:
        _active.emit(event_type, stage, message, **details)


def check_cancelled() -> None:
    if _active is not None:
        _active.check_cancelled()


def register_process(process, role: str) -> None:
    if _active is not None:
        _active.register_process(process, role)


def record_review_result(details: dict) -> None:
    if _active is not None:
        _active.review_cancelled = bool(details.get("cancelled", False))
        _active.emit("stage", "review_cancelled" if _active.review_cancelled else "measurement_complete",
                     "Fiji review cancelled" if _active.review_cancelled else "Fiji measurement completed")
