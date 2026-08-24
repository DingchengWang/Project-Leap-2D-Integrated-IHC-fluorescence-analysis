from __future__ import annotations

import base64
import hashlib
import json
import uuid
import zlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CELL_EDIT_PROTOCOL_VERSION = 1
_CELL_UID_NAMESPACE = uuid.UUID("1c25d78b-f594-46fb-a14d-df013ac90f35")
_VALID_OPERATIONS = frozenset(("split", "merge", "enlarge"))


class CellEditError(RuntimeError):
    """Base class for cell-edit transaction failures."""


class CellEditValidationError(CellEditError):
    """Raised when a triplet or edit violates a scientific invariant."""


class StaleCellEditError(CellEditError):
    """Raised when a proposal no longer matches the current Fiji state."""


class CellEditProtocolError(CellEditError):
    """Raised when serialized transaction data is incomplete or corrupt."""


def _nonempty_text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise CellEditValidationError(f"{field_name} must not be empty")
    return text


def _deduplicate(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        text = _nonempty_text(value, "lineage entry")
        if text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


@dataclass(frozen=True)
class CellIdentity:
    """Stable biological identity independent of the displayed Fiji number."""

    cell_uid: str
    parent_uid: Optional[str] = None
    owner_nucleus_uid: Optional[str] = None
    lineage: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cell_uid = _nonempty_text(self.cell_uid, "cell_uid")
        parent_uid = (
            None
            if self.parent_uid is None
            else _nonempty_text(self.parent_uid, "parent_uid")
        )
        owner_uid = (
            None
            if self.owner_nucleus_uid is None
            else _nonempty_text(self.owner_nucleus_uid, "owner_nucleus_uid")
        )
        lineage = _deduplicate(self.lineage or (cell_uid,))
        object.__setattr__(self, "cell_uid", cell_uid)
        object.__setattr__(self, "parent_uid", parent_uid)
        object.__setattr__(self, "owner_nucleus_uid", owner_uid)
        object.__setattr__(self, "lineage", lineage)

    def to_protocol_dict(self) -> Dict[str, Any]:
        return {
            "cell_uid": self.cell_uid,
            "parent_uid": self.parent_uid,
            "owner_nucleus_uid": self.owner_nucleus_uid,
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_protocol_dict(cls, value: Mapping[str, Any]) -> "CellIdentity":
        try:
            return cls(
                cell_uid=value["cell_uid"],
                parent_uid=value.get("parent_uid"),
                owner_nucleus_uid=value.get("owner_nucleus_uid"),
                lineage=tuple(value.get("lineage", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CellEditProtocolError(
                f"Invalid cell identity record: {value!r}"
            ) from exc


def make_initial_cell_identity(
    *,
    identity_namespace: str,
    original_display_id: int,
    owner_nucleus_uid: Optional[str] = None,
) -> CellIdentity:
    """Create a reproducible UID while importing the initial Fiji label set."""

    if int(original_display_id) < 1:
        raise CellEditValidationError("original_display_id must be positive")
    seed = f"initial:{identity_namespace}:{int(original_display_id)}"
    cell_uid = str(uuid.uuid5(_CELL_UID_NAMESPACE, seed))
    return CellIdentity(
        cell_uid=cell_uid,
        owner_nucleus_uid=owner_nucleus_uid,
        lineage=(cell_uid,),
    )


def make_split_child_identity(
    parent: CellIdentity,
    *,
    owner_nucleus_uid: str,
    child_index: int,
    edit_nonce: str,
) -> CellIdentity:
    """Create one deterministic child identity for a manual Split."""

    if int(child_index) not in (1, 2):
        raise CellEditValidationError("Split child_index must be 1 or 2")
    owner_uid = _nonempty_text(owner_nucleus_uid, "owner_nucleus_uid")
    nonce = _nonempty_text(edit_nonce, "edit_nonce")
    seed = f"split:{parent.cell_uid}:{nonce}:{int(child_index)}:{owner_uid}"
    child_uid = str(uuid.uuid5(_CELL_UID_NAMESPACE, seed))
    lineage = _deduplicate((*parent.lineage, parent.cell_uid))
    return CellIdentity(
        cell_uid=child_uid,
        parent_uid=parent.cell_uid,
        owner_nucleus_uid=owner_uid,
        lineage=lineage,
    )


def make_merged_cell_identity(
    sources: Sequence[CellIdentity],
    *,
    owner_nucleus_uid: Optional[str],
    edit_nonce: str,
) -> CellIdentity:
    """Create a deterministic identity for a Merge result."""

    if len(sources) < 2:
        raise CellEditValidationError("Merge requires at least two identities")
    nonce = _nonempty_text(edit_nonce, "edit_nonce")
    source_uids = tuple(sorted(identity.cell_uid for identity in sources))
    seed = f"merge:{nonce}:{','.join(source_uids)}"
    merged_uid = str(uuid.uuid5(_CELL_UID_NAMESPACE, seed))
    lineage = _deduplicate(
        value
        for identity in sources
        for value in (*identity.lineage, identity.cell_uid)
    )
    return CellIdentity(
        cell_uid=merged_uid,
        owner_nucleus_uid=owner_nucleus_uid,
        lineage=lineage,
    )


def _freeze_label_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise CellEditValidationError(f"{name} labels must be a 2D array")
    if not np.issubdtype(array.dtype, np.integer):
        raise CellEditValidationError(f"{name} labels must use an integer dtype")
    if array.size and int(array.min()) < 0:
        raise CellEditValidationError(f"{name} labels must not be negative")
    if array.size and int(array.max()) > np.iinfo(np.uint32).max:
        raise CellEditValidationError(f"{name} labels exceed uint32 capacity")
    result = np.ascontiguousarray(array, dtype=np.uint32)
    result.setflags(write=False)
    return result


def _positive_ids(array: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(value) for value in np.unique(array) if int(value) > 0)


def _validate_raw_triplet(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    *,
    require_contiguous: bool,
) -> Tuple[int, ...]:
    if whole_labels.shape != soma_labels.shape or whole_labels.shape != process_labels.shape:
        raise CellEditValidationError(
            "Whole, Soma, and Processes labels must have the same shape"
        )
    whole_ids = _positive_ids(whole_labels)
    soma_ids = _positive_ids(soma_labels)
    process_ids = _positive_ids(process_labels)
    if not whole_ids:
        raise CellEditValidationError("The cell-edit state must contain at least one cell")
    if whole_ids != soma_ids or whole_ids != process_ids:
        raise CellEditValidationError(
            "Whole, Soma, and Processes must contain exactly the same display IDs"
        )
    if require_contiguous and whole_ids != tuple(range(1, len(whole_ids) + 1)):
        raise CellEditValidationError(
            "Displayed Astrocyte IDs must be consecutive starting at 1"
        )
    if np.any((soma_labels > 0) & (soma_labels != whole_labels)):
        raise CellEditValidationError(
            "A Soma pixel is outside or assigned to a different Whole cell"
        )
    if np.any((process_labels > 0) & (process_labels != whole_labels)):
        raise CellEditValidationError(
            "A Processes pixel is outside or assigned to a different Whole cell"
        )
    if np.any((soma_labels > 0) & (process_labels > 0)):
        raise CellEditValidationError("Soma and Processes overlap")
    reconstructed = np.where(soma_labels > 0, soma_labels, process_labels)
    if not np.array_equal(reconstructed, whole_labels):
        raise CellEditValidationError(
            "Soma and Processes must form an exact partition of Whole"
        )
    return whole_ids


def _canonical_identity_json(
    identities: Mapping[int, CellIdentity],
) -> bytes:
    records = [
        {
            "display_id": int(display_id),
            **identity.to_protocol_dict(),
        }
        for display_id, identity in sorted(identities.items())
    ]
    return json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _state_content_hash(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identities: Mapping[int, CellIdentity],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"ProjectLeap2D-CellEditState-v1\0")
    digest.update(np.asarray(whole_labels.shape, dtype="<u8").tobytes())
    for name, array in (
        ("whole", whole_labels),
        ("soma", soma_labels),
        ("processes", process_labels),
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array, dtype="<u4").tobytes(order="C"))
    digest.update(_canonical_identity_json(identities))
    return digest.hexdigest()


@dataclass(frozen=True)
class CellEditState:
    """Immutable, validated Whole/Soma/Processes state."""

    whole_labels: np.ndarray = field(repr=False)
    soma_labels: np.ndarray = field(repr=False)
    process_labels: np.ndarray = field(repr=False)
    identities: Mapping[int, CellIdentity]
    version: int = 0
    state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        whole = _freeze_label_array(self.whole_labels, "Whole")
        soma = _freeze_label_array(self.soma_labels, "Soma")
        processes = _freeze_label_array(self.process_labels, "Processes")
        ids = _validate_raw_triplet(
            whole, soma, processes, require_contiguous=True
        )
        try:
            identities = {
                int(display_id): (
                    identity
                    if isinstance(identity, CellIdentity)
                    else CellIdentity.from_protocol_dict(identity)
                )
                for display_id, identity in self.identities.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise CellEditValidationError("Invalid identities mapping") from exc
        if tuple(sorted(identities)) != ids:
            raise CellEditValidationError(
                "Identity display IDs must match the triplet label IDs exactly"
            )
        cell_uids = [identity.cell_uid for identity in identities.values()]
        if len(set(cell_uids)) != len(cell_uids):
            raise CellEditValidationError("Cell UIDs must be unique")
        version = int(self.version)
        if version < 0:
            raise CellEditValidationError("State version must not be negative")
        identities_proxy = MappingProxyType(dict(sorted(identities.items())))
        state_hash = _state_content_hash(
            whole, soma, processes, identities_proxy
        )
        object.__setattr__(self, "whole_labels", whole)
        object.__setattr__(self, "soma_labels", soma)
        object.__setattr__(self, "process_labels", processes)
        object.__setattr__(self, "identities", identities_proxy)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "state_hash", state_hash)

    @property
    def cell_count(self) -> int:
        return len(self.identities)

    def with_version(self, version: int) -> "CellEditState":
        return CellEditState(
            whole_labels=self.whole_labels,
            soma_labels=self.soma_labels,
            process_labels=self.process_labels,
            identities=self.identities,
            version=version,
        )

    def identity_by_uid(self, cell_uid: str) -> CellIdentity:
        matches = [
            identity
            for identity in self.identities.values()
            if identity.cell_uid == cell_uid
        ]
        if len(matches) != 1:
            raise CellEditValidationError(
                f"Unknown or non-unique Cell UID: {cell_uid}"
            )
        return matches[0]

    def display_id_for_uid(self, cell_uid: str) -> int:
        matches = [
            display_id
            for display_id, identity in self.identities.items()
            if identity.cell_uid == cell_uid
        ]
        if len(matches) != 1:
            raise CellEditValidationError(
                f"Unknown or non-unique Cell UID: {cell_uid}"
            )
        return matches[0]

    def mask_for_uid(self, compartment: str, cell_uid: str) -> np.ndarray:
        arrays = {
            "whole": self.whole_labels,
            "soma": self.soma_labels,
            "processes": self.process_labels,
        }
        try:
            array = arrays[compartment]
        except KeyError as exc:
            raise CellEditValidationError(
                f"Unknown compartment: {compartment}"
            ) from exc
        return array == self.display_id_for_uid(cell_uid)

    def to_protocol_dict(self, *, include_labels: bool = True) -> Dict[str, Any]:
        result = {
            "schema_version": CELL_EDIT_PROTOCOL_VERSION,
            "state_revision": self.version,
            "state_hash": self.state_hash,
            "shape": list(self.whole_labels.shape),
            "identities": [
                {
                    "display_id": display_id,
                    **identity.to_protocol_dict(),
                }
                for display_id, identity in self.identities.items()
            ],
            "labels": {},
        }
        for name, array in (
            ("whole", self.whole_labels),
            ("soma", self.soma_labels),
            ("processes", self.process_labels),
        ):
            result["labels"][name] = _encode_label_array(
                array, include_data=include_labels
            )
        return result

    @classmethod
    def from_protocol_dict(cls, value: Mapping[str, Any]) -> "CellEditState":
        _require_protocol_version(value)
        try:
            identities = {
                int(record["display_id"]): CellIdentity.from_protocol_dict(record)
                for record in value["identities"]
            }
            labels = value["labels"]
            state = cls(
                whole_labels=_decode_label_array(labels["whole"]),
                soma_labels=_decode_label_array(labels["soma"]),
                process_labels=_decode_label_array(labels["processes"]),
                identities=identities,
                version=int(value["state_revision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CellEditProtocolError("Invalid cell-edit state payload") from exc
        expected_hash = value.get("state_hash")
        if expected_hash is not None and state.state_hash != expected_hash:
            raise CellEditProtocolError(
                "Serialized state hash does not match its labels and identities"
            )
        return state


def renumber_label_triplet(
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identities: Mapping[int, CellIdentity],
    *,
    display_order: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, CellIdentity], Dict[int, int]]:
    """Renumber an otherwise-valid triplet to consecutive display IDs."""

    whole = _freeze_label_array(whole_labels, "Whole")
    soma = _freeze_label_array(soma_labels, "Soma")
    processes = _freeze_label_array(process_labels, "Processes")
    old_ids = _validate_raw_triplet(
        whole, soma, processes, require_contiguous=False
    )
    order = tuple(old_ids if display_order is None else (int(v) for v in display_order))
    if len(order) != len(set(order)) or set(order) != set(old_ids):
        raise CellEditValidationError(
            "display_order must contain every old display ID exactly once"
        )
    if set(int(value) for value in identities) != set(old_ids):
        raise CellEditValidationError(
            "Identity IDs must match the unnumbered triplet IDs"
        )
    old_to_new = {old_id: index + 1 for index, old_id in enumerate(order)}

    def remap(array: np.ndarray) -> np.ndarray:
        result = np.zeros(array.shape, dtype=np.uint32)
        for old_id, new_id in old_to_new.items():
            result[array == old_id] = new_id
        return result

    renumbered_identities = {
        old_to_new[old_id]: identities[old_id] for old_id in order
    }
    return (
        remap(whole),
        remap(soma),
        remap(processes),
        renumbered_identities,
        old_to_new,
    )


def _uid_mapping(state: CellEditState) -> Dict[str, CellIdentity]:
    return {identity.cell_uid: identity for identity in state.identities.values()}


def _assert_unaffected_cells_unchanged(
    base: CellEditState,
    result: CellEditState,
    unaffected_uids: Iterable[str],
) -> None:
    for cell_uid in unaffected_uids:
        if base.identity_by_uid(cell_uid) != result.identity_by_uid(cell_uid):
            raise CellEditValidationError(
                f"Unedited identity changed unexpectedly: {cell_uid}"
            )
        for compartment in ("whole", "soma", "processes"):
            if not np.array_equal(
                base.mask_for_uid(compartment, cell_uid),
                result.mask_for_uid(compartment, cell_uid),
            ):
                raise CellEditValidationError(
                    f"Unedited {compartment} geometry changed: {cell_uid}"
                )


def _new_result_state(
    base: CellEditState,
    *,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identities: Mapping[int, CellIdentity],
) -> CellEditState:
    return CellEditState(
        whole_labels=whole_labels,
        soma_labels=soma_labels,
        process_labels=process_labels,
        identities=identities,
        version=base.version + 1,
    )


@dataclass(frozen=True)
class CellEditProposal:
    proposal_id: str
    operation: str
    base_version: int
    base_hash: str
    result_state: CellEditState
    source_cell_uids: Tuple[str, ...]
    result_cell_uids: Tuple[str, ...]
    audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        proposal_id = _nonempty_text(self.proposal_id, "proposal_id")
        operation = str(self.operation).strip().lower()
        if operation not in _VALID_OPERATIONS:
            raise CellEditValidationError(f"Unsupported edit operation: {operation}")
        base_version = int(self.base_version)
        if base_version < 0:
            raise CellEditValidationError("base_version must not be negative")
        base_hash = _nonempty_text(self.base_hash, "base_hash")
        sources = _deduplicate(self.source_cell_uids)
        results = _deduplicate(self.result_cell_uids)
        if not sources or not results:
            raise CellEditValidationError(
                "A proposal must record source and result Cell UIDs"
            )
        try:
            audit = json.loads(json.dumps(dict(self.audit), sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise CellEditValidationError("Proposal audit must be JSON-safe") from exc
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "base_version", base_version)
        object.__setattr__(self, "base_hash", base_hash)
        object.__setattr__(self, "source_cell_uids", sources)
        object.__setattr__(self, "result_cell_uids", results)
        object.__setattr__(self, "audit", MappingProxyType(audit))

    def to_protocol_dict(self, *, include_labels: bool = True) -> Dict[str, Any]:
        return {
            "schema_version": CELL_EDIT_PROTOCOL_VERSION,
            "proposal_id": self.proposal_id,
            "operation": self.operation,
            "base_state_revision": self.base_version,
            "base_state_hash": self.base_hash,
            "source_cell_uids": list(self.source_cell_uids),
            "result_cell_uids": list(self.result_cell_uids),
            "audit": dict(self.audit),
            "result_state": self.result_state.to_protocol_dict(
                include_labels=include_labels
            ),
        }

    @classmethod
    def from_protocol_dict(cls, value: Mapping[str, Any]) -> "CellEditProposal":
        _require_protocol_version(value)
        try:
            return cls(
                proposal_id=value["proposal_id"],
                operation=value["operation"],
                base_version=int(value["base_state_revision"]),
                base_hash=value["base_state_hash"],
                result_state=CellEditState.from_protocol_dict(value["result_state"]),
                source_cell_uids=tuple(value["source_cell_uids"]),
                result_cell_uids=tuple(value["result_cell_uids"]),
                audit=value.get("audit", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CellEditProtocolError("Invalid cell-edit proposal payload") from exc


def _proposal(
    operation: str,
    base: CellEditState,
    result: CellEditState,
    *,
    source_cell_uids: Sequence[str],
    result_cell_uids: Sequence[str],
    proposal_id: Optional[str],
    audit: Optional[Mapping[str, Any]],
) -> CellEditProposal:
    return CellEditProposal(
        proposal_id=proposal_id or str(uuid.uuid4()),
        operation=operation,
        base_version=base.version,
        base_hash=base.state_hash,
        result_state=result,
        source_cell_uids=tuple(source_cell_uids),
        result_cell_uids=tuple(result_cell_uids),
        audit={} if audit is None else audit,
    )


def propose_split(
    base: CellEditState,
    *,
    source_cell_uid: str,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identities: Mapping[int, CellIdentity],
    proposal_id: Optional[str] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> CellEditProposal:
    """Validate a Split result before it can mutate Fiji state."""

    source_uid = _nonempty_text(source_cell_uid, "source_cell_uid")
    base.identity_by_uid(source_uid)
    result = _new_result_state(
        base,
        whole_labels=whole_labels,
        soma_labels=soma_labels,
        process_labels=process_labels,
        identities=identities,
    )
    if result.cell_count != base.cell_count + 1:
        raise CellEditValidationError("Split must increase the cell count by exactly one")
    base_uids = set(_uid_mapping(base))
    result_uids = set(_uid_mapping(result))
    if source_uid in result_uids:
        raise CellEditValidationError(
            "Split must replace the parent UID with two child UIDs"
        )
    added_uids = result_uids - base_uids
    if result_uids - added_uids != base_uids - {source_uid} or len(added_uids) != 2:
        raise CellEditValidationError(
            "Split must preserve all unaffected cells and create exactly two children"
        )
    children = [result.identity_by_uid(uid) for uid in sorted(added_uids)]
    if any(child.parent_uid != source_uid for child in children):
        raise CellEditValidationError(
            "Both Split children must reference the selected parent UID"
        )
    if any(source_uid not in child.lineage for child in children):
        raise CellEditValidationError(
            "Both Split children must retain the parent in their lineage"
        )
    owners = [child.owner_nucleus_uid for child in children]
    if any(owner is None for owner in owners) or len(set(owners)) != 2:
        raise CellEditValidationError(
            "Split children require two distinct owner nucleus UIDs"
        )
    unaffected = base_uids - {source_uid}
    _assert_unaffected_cells_unchanged(base, result, unaffected)
    return _proposal(
        "split",
        base,
        result,
        source_cell_uids=(source_uid,),
        result_cell_uids=tuple(sorted(added_uids)),
        proposal_id=proposal_id,
        audit=audit,
    )


def propose_merge(
    base: CellEditState,
    *,
    source_cell_uids: Sequence[str],
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identities: Mapping[int, CellIdentity],
    proposal_id: Optional[str] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> CellEditProposal:
    """Validate an exact geometric Merge of two or more cells."""

    sources = _deduplicate(source_cell_uids)
    if len(sources) < 2:
        raise CellEditValidationError("Merge requires at least two source cells")
    for source_uid in sources:
        base.identity_by_uid(source_uid)
    result = _new_result_state(
        base,
        whole_labels=whole_labels,
        soma_labels=soma_labels,
        process_labels=process_labels,
        identities=identities,
    )
    expected_count = base.cell_count - len(sources) + 1
    if result.cell_count != expected_count:
        raise CellEditValidationError(
            "Merge must replace all selected cells with exactly one result cell"
        )
    base_uids = set(_uid_mapping(base))
    result_uids = set(_uid_mapping(result))
    source_set = set(sources)
    if result_uids & source_set:
        raise CellEditValidationError("Merged source UIDs must not remain active")
    added_uids = result_uids - base_uids
    if len(added_uids) != 1 or result_uids - added_uids != base_uids - source_set:
        raise CellEditValidationError(
            "Merge must preserve unaffected cells and create one merged UID"
        )
    merged_uid = next(iter(added_uids))
    merged_identity = result.identity_by_uid(merged_uid)
    if not source_set.issubset(set(merged_identity.lineage)):
        raise CellEditValidationError(
            "Merged identity lineage must contain every source Cell UID"
        )
    unaffected = base_uids - source_set
    _assert_unaffected_cells_unchanged(base, result, unaffected)
    expected_whole = np.logical_or.reduce(
        [base.mask_for_uid("whole", uid) for uid in sources]
    )
    expected_soma = np.logical_or.reduce(
        [base.mask_for_uid("soma", uid) for uid in sources]
    )
    if not np.array_equal(result.mask_for_uid("whole", merged_uid), expected_whole):
        raise CellEditValidationError("Merge Whole must equal the source Whole union")
    if not np.array_equal(result.mask_for_uid("soma", merged_uid), expected_soma):
        raise CellEditValidationError("Merge Soma must equal the source Soma union")
    return _proposal(
        "merge",
        base,
        result,
        source_cell_uids=sources,
        result_cell_uids=(merged_uid,),
        proposal_id=proposal_id,
        audit=audit,
    )


def propose_enlarge(
    base: CellEditState,
    *,
    source_cell_uid: str,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identities: Mapping[int, CellIdentity],
    proposal_id: Optional[str] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> CellEditProposal:
    """Validate a Soma enlargement with synchronized Whole expansion."""

    source_uid = _nonempty_text(source_cell_uid, "source_cell_uid")
    base.identity_by_uid(source_uid)
    result = _new_result_state(
        base,
        whole_labels=whole_labels,
        soma_labels=soma_labels,
        process_labels=process_labels,
        identities=identities,
    )
    if result.cell_count != base.cell_count:
        raise CellEditValidationError("Enlarge must not change the cell count")
    base_by_uid = _uid_mapping(base)
    result_by_uid = _uid_mapping(result)
    if base_by_uid != result_by_uid:
        raise CellEditValidationError("Enlarge must preserve every Cell identity")
    unaffected = set(base_by_uid) - {source_uid}
    _assert_unaffected_cells_unchanged(base, result, unaffected)
    old_whole = base.mask_for_uid("whole", source_uid)
    new_whole = result.mask_for_uid("whole", source_uid)
    old_soma = base.mask_for_uid("soma", source_uid)
    new_soma = result.mask_for_uid("soma", source_uid)
    if np.any(old_whole & ~new_whole):
        raise CellEditValidationError("Enlarge must not remove existing Whole pixels")
    if np.any(old_soma & ~new_soma):
        raise CellEditValidationError("Enlarge must not remove existing Soma pixels")
    if not np.any(new_soma & ~old_soma):
        raise CellEditValidationError("Enlarge must add at least one Soma pixel")
    added_whole = new_whole & ~old_whole
    if np.any(added_whole & ~new_soma):
        raise CellEditValidationError(
            "Every Whole pixel added by Enlarge must simultaneously belong to Soma"
        )
    return _proposal(
        "enlarge",
        base,
        result,
        source_cell_uids=(source_uid,),
        result_cell_uids=(source_uid,),
        proposal_id=proposal_id,
        audit=audit,
    )


def commit_cell_edit(
    current_state: CellEditState,
    proposal: CellEditProposal,
) -> CellEditState:
    """Return the proposal result only if the base state still matches."""

    if current_state.version != proposal.base_version:
        raise StaleCellEditError(
            "Cell edit rejected because the Fiji state revision changed"
        )
    if current_state.state_hash != proposal.base_hash:
        raise StaleCellEditError(
            "Cell edit rejected because the Fiji ROI content changed"
        )
    if proposal.result_state.version != current_state.version + 1:
        raise CellEditValidationError(
            "Committed state revision must increase by exactly one"
        )
    return proposal.result_state


@dataclass(frozen=True)
class CommittedCellEdit:
    proposal_id: str
    operation: str
    before_state: CellEditState
    committed_hash: str
    committed_version: int


class CellEditLedger:
    """In-memory atomic commit and LIFO Revert state for the Fiji session."""

    def __init__(self, initial_state: CellEditState) -> None:
        if not isinstance(initial_state, CellEditState):
            raise CellEditValidationError("initial_state must be a CellEditState")
        self._current_state = initial_state
        self._undo_stack: List[CommittedCellEdit] = []

    @property
    def current_state(self) -> CellEditState:
        return self._current_state

    @property
    def can_revert(self) -> bool:
        return bool(self._undo_stack)

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    def commit(self, proposal: CellEditProposal) -> CellEditState:
        result = commit_cell_edit(self._current_state, proposal)
        record = CommittedCellEdit(
            proposal_id=proposal.proposal_id,
            operation=proposal.operation,
            before_state=self._current_state,
            committed_hash=result.state_hash,
            committed_version=result.version,
        )
        self._undo_stack.append(record)
        self._current_state = result
        return result

    def revert(self) -> CellEditState:
        if not self._undo_stack:
            raise CellEditValidationError("There is no Cell Edit action to revert")
        record = self._undo_stack[-1]
        if (
            self._current_state.version != record.committed_version
            or self._current_state.state_hash != record.committed_hash
        ):
            raise StaleCellEditError(
                "Revert rejected because the Fiji state changed after the commit"
            )
        restored = record.before_state.with_version(self._current_state.version + 1)
        self._undo_stack.pop()
        self._current_state = restored
        return restored


def _encode_label_array(
    array: np.ndarray, *, include_data: bool
) -> Dict[str, Any]:
    canonical = np.ascontiguousarray(array, dtype="<u4")
    raw = canonical.tobytes(order="C")
    payload = {
        "encoding": "zlib+base64+uint32le",
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    if include_data:
        payload["data"] = base64.b64encode(zlib.compress(raw, level=6)).decode(
            "ascii"
        )
    return payload


def _decode_label_array(value: Mapping[str, Any]) -> np.ndarray:
    try:
        if value["encoding"] != "zlib+base64+uint32le":
            raise CellEditProtocolError("Unsupported label encoding")
        if "data" not in value:
            raise CellEditProtocolError("Serialized label data is missing")
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) != 2 or any(item < 0 for item in shape):
            raise CellEditProtocolError("Serialized label shape is invalid")
        raw = zlib.decompress(base64.b64decode(value["data"], validate=True))
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * 4
        if len(raw) != expected_bytes or len(raw) != int(value["byte_count"]):
            raise CellEditProtocolError("Serialized label byte count is invalid")
        if hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise CellEditProtocolError("Serialized label checksum failed")
        return np.frombuffer(raw, dtype="<u4").reshape(shape).copy()
    except CellEditProtocolError:
        raise
    except (KeyError, TypeError, ValueError, zlib.error) as exc:
        raise CellEditProtocolError("Invalid serialized label array") from exc


def _require_protocol_version(value: Mapping[str, Any]) -> None:
    try:
        version = int(value["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CellEditProtocolError("Missing cell-edit schema version") from exc
    if version != CELL_EDIT_PROTOCOL_VERSION:
        raise CellEditProtocolError(
            f"Unsupported cell-edit schema version: {version}"
        )


def protocol_json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def protocol_json_loads(value: str) -> Dict[str, Any]:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise CellEditProtocolError("Invalid cell-edit JSON") from exc
    if not isinstance(payload, dict):
        raise CellEditProtocolError("Cell-edit JSON root must be an object")
    return payload
