from __future__ import annotations

from typing import Any

import numpy as np


def validate_cell_edit_label_triplet(
    *,
    whole_labels: np.ndarray,
    soma_labels: np.ndarray,
    process_labels: np.ndarray,
    identity_records: list[dict[str, Any]],
) -> None:
    arrays = {
        "whole": np.asarray(whole_labels),
        "soma": np.asarray(soma_labels),
        "processes": np.asarray(process_labels),
    }
    if len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("Cell Edit label masks do not share one image shape")
    if any(value.ndim != 2 for value in arrays.values()):
        raise ValueError("Cell Edit label masks must be two-dimensional")
    if any(not np.issubdtype(value.dtype, np.integer) for value in arrays.values()):
        raise ValueError("Cell Edit label masks must use integer IDs")
    if any(np.any(value < 0) for value in arrays.values()):
        raise ValueError("Cell Edit label masks contain negative IDs")
    expected_ids = list(range(1, len(identity_records) + 1))
    for key, labels in arrays.items():
        observed = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
        if observed != expected_ids:
            raise ValueError(
                f"Cell Edit {key} IDs are not complete and contiguous: {observed}"
            )
    whole = arrays["whole"]
    soma = arrays["soma"]
    processes = arrays["processes"]
    if np.any((soma > 0) & (soma != whole)):
        raise ValueError("Cell Edit Soma extends outside or across Whole ownership")
    if np.any((processes > 0) & (processes != whole)):
        raise ValueError("Cell Edit Processes extend outside or across Whole ownership")
    if np.any((soma > 0) & (processes > 0)):
        raise ValueError("Cell Edit Soma and Processes overlap")
    if not np.array_equal(
        whole,
        np.where(soma > 0, soma, processes),
    ):
        raise ValueError("Cell Edit Soma and Processes do not exactly partition Whole")
    labels = [int(record.get("label_id", 0)) for record in identity_records]
    if labels != expected_ids:
        raise ValueError("Cell Edit identity labels are not complete and ordered")
    uids = [str(record.get("cell_uid", "")).strip() for record in identity_records]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("Cell Edit Cell UIDs are missing or repeated")
    original_ids = [
        int(record.get("original_id", 0)) for record in identity_records
    ]
    if any(value < 1 for value in original_ids) or len(original_ids) != len(
        set(original_ids)
    ):
        raise ValueError("Cell Edit Original Astrocyte IDs are missing or repeated")


def validate_cell_edit_delta(
    *,
    action: str,
    before_whole: np.ndarray,
    before_soma: np.ndarray,
    before_processes: np.ndarray,
    after_whole: np.ndarray,
    after_soma: np.ndarray,
    after_processes: np.ndarray,
    selected_id: int,
) -> None:
    normalized_action = str(action).strip().lower()
    if normalized_action not in {"split", "enlarge"}:
        raise ValueError(f"Unsupported Cell Edit delta action: {action!r}")
    before = tuple(
        np.asarray(value)
        for value in (before_whole, before_soma, before_processes)
    )
    after = tuple(
        np.asarray(value)
        for value in (after_whole, after_soma, after_processes)
    )
    if any(left.shape != right.shape for left, right in zip(before, after)):
        raise ValueError("Cell Edit changed the image dimensions")
    selected_id = int(selected_id)
    if selected_id < 1:
        raise ValueError("Cell Edit selected ID must be positive")
    before_ids = sorted(
        int(value) for value in np.unique(before[0]) if int(value) > 0
    )
    after_ids = sorted(
        int(value) for value in np.unique(after[0]) if int(value) > 0
    )
    if normalized_action == "enlarge":
        if after_ids != before_ids:
            raise ValueError("Enlarge changed the Astrocyte ID set")
        if np.any((before[1] == selected_id) & (after[1] != selected_id)):
            raise ValueError("Enlarge removed part of the selected Soma")
    else:
        if len(after_ids) != len(before_ids) + 1:
            raise ValueError("Split did not increase the Astrocyte count by exactly one")
    protected = (before[0] > 0) & (before[0] != selected_id)
    for left, right, name in zip(before, after, ("Whole", "Soma", "Processes")):
        if not np.array_equal(left[protected], right[protected]):
            raise ValueError(f"Cell Edit changed another Astrocyte {name} assignment")
