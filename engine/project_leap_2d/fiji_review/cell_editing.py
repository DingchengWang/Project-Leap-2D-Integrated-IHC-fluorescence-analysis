from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


CELL_EDIT_SCHEMA_VERSION = 1
PYTHON_CELL_EDIT_ACTIONS = ("split", "enlarge")
_CELL_EDIT_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


def cell_edit_runtime_paths(run_dir: Path) -> dict[str, Path]:
    root = Path(run_dir) / "cell_edit"
    return {
        "root": root,
        "request_dir": root / "requests",
        "response_dir": root / "responses",
        "cancel_dir": root / "cancel",
        "state_dir": root / "state",
    }


def prepare_cell_edit_runtime(
    run_dir: Path,
    *,
    context_path: Path | None,
    enabled_actions: tuple[str, ...] = PYTHON_CELL_EDIT_ACTIONS,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    paths = cell_edit_runtime_paths(run_dir)
    for key in ("request_dir", "response_dir", "cancel_dir", "state_dir"):
        paths[key].mkdir(parents=True, exist_ok=False)
    normalized_actions = tuple(
        action
        for action in PYTHON_CELL_EDIT_ACTIONS
        if action in {str(value).strip().lower() for value in enabled_actions}
    )
    return {
        "schema_version": CELL_EDIT_SCHEMA_VERSION,
        "enabled_actions": list(normalized_actions),
        "request_dir": str(paths["request_dir"]),
        "response_dir": str(paths["response_dir"]),
        "cancel_dir": str(paths["cancel_dir"]),
        "state_dir": str(paths["state_dir"]),
        "context_path": None if context_path is None else str(context_path),
        "timeout_seconds": float(timeout_seconds),
    }


def cell_edit_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_cell_edit_json(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f"temporary_{target.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def validate_cell_edit_request(
    payload: dict[str, Any],
    *,
    enabled_actions: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Cell Edit request must be a JSON object")
    if int(payload.get("schema_version", -1)) != CELL_EDIT_SCHEMA_VERSION:
        raise ValueError("Unsupported Cell Edit request schema")
    request_id = str(payload.get("request_id", ""))
    if _CELL_EDIT_REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError("Invalid Cell Edit request ID")
    action = str(payload.get("action", "")).strip().lower()
    if action not in enabled_actions:
        raise ValueError(f"Cell Edit action is not enabled: {action!r}")
    revision = int(payload.get("state_revision", -1))
    if revision < 0:
        raise ValueError("Invalid Cell Edit state revision")
    state_token = str(payload.get("state_token", ""))
    if _CELL_EDIT_REQUEST_ID.fullmatch(state_token) is None:
        raise ValueError("Invalid Cell Edit state token")
    selected_original_id = int(payload.get("selected_original_id", 0))
    if selected_original_id < 1:
        raise ValueError("A positive selected Original Astrocyte ID is required")
    selected_cell_uid = str(payload.get("selected_cell_uid", "")).strip()
    if not selected_cell_uid:
        raise ValueError("A selected Cell UID is required")
    mask_paths = payload.get("label_mask_paths")
    mask_hashes = payload.get("label_mask_sha256")
    if not isinstance(mask_paths, dict) or not isinstance(mask_hashes, dict):
        raise ValueError("Cell Edit request is missing synchronized label masks")
    normalized_paths: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    for key in ("whole", "soma", "processes"):
        path = Path(str(mask_paths.get(key, ""))).resolve()
        expected_hash = str(mask_hashes.get(key, "")).lower()
        if not path.is_file():
            raise ValueError(f"Cell Edit {key} label mask is missing")
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError(f"Cell Edit {key} label mask hash is invalid")
        if cell_edit_sha256(path) != expected_hash:
            raise ValueError(f"Cell Edit {key} label mask hash mismatch")
        normalized_paths[key] = str(path)
        normalized_hashes[key] = expected_hash
    normalized = dict(payload)
    normalized.update(
        {
            "request_id": request_id,
            "action": action,
            "state_revision": revision,
            "state_token": state_token,
            "selected_original_id": selected_original_id,
            "selected_cell_uid": selected_cell_uid,
            "label_mask_paths": normalized_paths,
            "label_mask_sha256": normalized_hashes,
        }
    )
    return normalized


def validate_cell_edit_response(
    payload: dict[str, Any],
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Cell Edit response must be a JSON object")
    for key in ("request_id", "action", "state_revision", "state_token"):
        if payload.get(key) != request.get(key):
            raise ValueError(f"Cell Edit response does not match request field {key}")
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"success", "rejected", "cancelled", "error"}:
        raise ValueError("Cell Edit response has an invalid status")
    normalized = dict(payload)
    normalized["status"] = status
    if status != "success":
        normalized["reason"] = str(
            payload.get("reason", "Cell Edit did not produce a result")
        )
        return normalized
    roi_count = int(payload.get("roi_count", 0))
    if roi_count < 1:
        raise ValueError("Successful Cell Edit response has no ROIs")
    mask_paths = payload.get("label_mask_paths")
    mask_hashes = payload.get("label_mask_sha256")
    identities = payload.get("identity_records")
    if not isinstance(mask_paths, dict) or not isinstance(mask_hashes, dict):
        raise ValueError("Successful Cell Edit response is missing label masks")
    if not isinstance(identities, list) or len(identities) != roi_count:
        raise ValueError("Successful Cell Edit response has invalid identity records")
    labels = []
    cell_uids = []
    original_ids = []
    for record in identities:
        if not isinstance(record, dict):
            raise ValueError("Cell Edit identity record must be an object")
        labels.append(int(record.get("label_id", 0)))
        cell_uids.append(str(record.get("cell_uid", "")))
        original_ids.append(int(record.get("original_id", 0)))
    if labels != list(range(1, roi_count + 1)):
        raise ValueError("Cell Edit identity label IDs are not contiguous")
    if any(not value for value in cell_uids) or len(cell_uids) != len(set(cell_uids)):
        raise ValueError("Cell Edit identity records repeat or omit Cell UIDs")
    if any(value < 1 for value in original_ids) or len(original_ids) != len(
        set(original_ids)
    ):
        raise ValueError(
            "Cell Edit identity records repeat or omit Original Astrocyte IDs"
        )
    normalized_paths: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    for key in ("whole", "soma", "processes"):
        path = Path(str(mask_paths.get(key, ""))).resolve()
        expected_hash = str(mask_hashes.get(key, "")).lower()
        if not path.is_file() or cell_edit_sha256(path) != expected_hash:
            raise ValueError(f"Cell Edit response {key} label mask failed integrity")
        normalized_paths[key] = str(path)
        normalized_hashes[key] = expected_hash
    normalized.update(
        {
            "roi_count": roi_count,
            "label_mask_paths": normalized_paths,
            "label_mask_sha256": normalized_hashes,
        }
    )
    return normalized


def build_fiji_identity_records(
    *,
    result_identities: dict[int, Any],
    request_identity_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt transaction UIDs to Fiji's distinct Original-ID lineage contract."""

    prior_by_uid = {
        str(record["cell_uid"]): dict(record)
        for record in request_identity_records
    }
    if len(prior_by_uid) != len(request_identity_records):
        raise ValueError("Request identity records repeat a Cell UID")
    used_original_ids = {
        int(record["original_id"]) for record in request_identity_records
    }
    next_original_id = max(used_original_ids, default=0) + 1
    parent_child_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for display_id, identity in sorted(result_identities.items()):
        cell_uid = str(identity.cell_uid)
        parent_uid = (
            "" if getattr(identity, "parent_uid", None) is None
            else str(identity.parent_uid)
        )
        owner_uid = (
            "" if getattr(identity, "owner_nucleus_uid", None) is None
            else str(identity.owner_nucleus_uid)
        )
        prior = prior_by_uid.get(cell_uid)
        if prior is not None:
            original_id = int(prior["original_id"])
            source_lineage = [int(value) for value in prior["lineage"]]
        elif parent_uid and parent_uid in prior_by_uid:
            parent = prior_by_uid[parent_uid]
            child_index = parent_child_counts.get(parent_uid, 0)
            parent_child_counts[parent_uid] = child_index + 1
            if child_index == 0:
                original_id = int(parent["original_id"])
            else:
                while next_original_id in used_original_ids:
                    next_original_id += 1
                original_id = next_original_id
                used_original_ids.add(original_id)
                next_original_id += 1
            source_lineage = [int(value) for value in parent["lineage"]]
        else:
            source_uids = tuple(str(value) for value in identity.lineage)
            source_records = [
                prior_by_uid[value]
                for value in source_uids
                if value in prior_by_uid
            ]
            if not source_records:
                raise ValueError(
                    f"Result Cell UID has no transport lineage: {cell_uid}"
                )
            original_id = min(int(value["original_id"]) for value in source_records)
            source_lineage = sorted(
                {
                    int(source_id)
                    for value in source_records
                    for source_id in value["lineage"]
                }
            )
        if original_id in {
            int(value["original_id"]) for value in records
        }:
            while next_original_id in used_original_ids:
                next_original_id += 1
            original_id = next_original_id
            used_original_ids.add(original_id)
            next_original_id += 1
        records.append(
            {
                "label_id": int(display_id),
                "original_id": original_id,
                "cell_uid": cell_uid,
                "parent_uid": parent_uid,
                "lineage": source_lineage,
                "owner_nucleus_id": owner_uid,
            }
        )
    if [record["label_id"] for record in records] != list(
        range(1, len(records) + 1)
    ):
        raise ValueError("Result identities are not consecutively displayed")
    return records


class CellEditRequestService:
    """Single-flight filesystem broker for Fiji-to-Python Cell Edit requests."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        dispatcher: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    ) -> None:
        config = manifest.get("cell_edit")
        if not isinstance(config, dict):
            raise ValueError("Manifest has no Cell Edit configuration")
        self.manifest = manifest
        self.config = config
        self.enabled_actions = tuple(
            str(value).lower() for value in config.get("enabled_actions", ())
        )
        self.request_dir = Path(str(config["request_dir"]))
        self.response_dir = Path(str(config["response_dir"]))
        self.cancel_dir = Path(str(config["cancel_dir"]))
        self.dispatcher = dispatcher
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="project-leap-cell-edit",
        )
        self._active: tuple[dict[str, Any], Future] | None = None
        self._handled: set[str] = set()
        self._lock = threading.Lock()

    def _response_path(self, request_id: str) -> Path:
        return self.response_dir / f"{request_id}.json"

    def _cancel_path(self, request_id: str) -> Path:
        return self.cancel_dir / f"{request_id}.cancel"

    def _execute(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self.dispatcher(request, self.manifest)
        return validate_cell_edit_response(result, request=request)

    def _publish_failure(
        self,
        *,
        request: dict[str, Any],
        status: str,
        reason: str,
    ) -> None:
        atomic_write_cell_edit_json(
            self._response_path(str(request["request_id"])),
            {
                "schema_version": CELL_EDIT_SCHEMA_VERSION,
                "request_id": request["request_id"],
                "action": request["action"],
                "state_revision": request["state_revision"],
                "state_token": request["state_token"],
                "status": status,
                "reason": reason,
            },
        )

    def poll(self) -> None:
        with self._lock:
            if self._active is not None:
                request, future = self._active
                if not future.done():
                    return
                self._active = None
                request_id = str(request["request_id"])
                try:
                    response = future.result()
                    if self._cancel_path(request_id).exists():
                        self._publish_failure(
                            request=request,
                            status="cancelled",
                            reason="Cell Edit was cancelled; ROI state was not changed.",
                        )
                    else:
                        atomic_write_cell_edit_json(
                            self._response_path(request_id),
                            response,
                        )
                except BaseException as error:
                    self._publish_failure(
                        request=request,
                        status="error",
                        reason=f"Cell Edit worker failed: {error}",
                    )
                self._handled.add(request_id)

            if self._active is not None:
                return
            for path in sorted(self.request_dir.glob("*.json")):
                request_id = path.stem
                if request_id in self._handled:
                    continue
                payload: dict[str, Any] = {}
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    request = validate_cell_edit_request(
                        payload,
                        enabled_actions=self.enabled_actions,
                    )
                except BaseException as error:
                    if _CELL_EDIT_REQUEST_ID.fullmatch(request_id):
                        fallback = {
                            "request_id": request_id,
                            "action": str(payload.get("action", "unknown")),
                            "state_revision": int(payload.get("state_revision", -1)),
                            "state_token": str(payload.get("state_token", "")),
                        }
                        self._publish_failure(
                            request=fallback,
                            status="error",
                            reason=f"Invalid Cell Edit request: {error}",
                        )
                    self._handled.add(request_id)
                    continue
                if self._cancel_path(request_id).exists():
                    self._publish_failure(
                        request=request,
                        status="cancelled",
                        reason="Cell Edit was cancelled before calculation started.",
                    )
                    self._handled.add(request_id)
                    continue
                self._active = (request, self.executor.submit(self._execute, request))
                break

    def cancel_active(self) -> None:
        with self._lock:
            if self._active is None:
                return
            request, future = self._active
            self._cancel_path(str(request["request_id"])).touch(exist_ok=True)
            future.cancel()

    def close(self) -> None:
        self.cancel_active()
        self.executor.shutdown(wait=False, cancel_futures=True)
