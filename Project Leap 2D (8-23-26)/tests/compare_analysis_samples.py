#!/usr/bin/env python3
"""Read-only comparator for designated analysis sample roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


TRIPLET_KEYS = ("whole_labels", "soma_labels", "process_labels")


def positive_ids(value: np.ndarray) -> set[int]:
    return {int(item) for item in np.unique(value) if int(item) > 0}


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_plan(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("comparison plan must be a JSON object")
    return value


def triplet_errors(label: str, arrays: Mapping[str, np.ndarray]) -> list[str]:
    missing = [name for name in TRIPLET_KEYS if name not in arrays]
    if missing:
        return [f"{label} is missing arrays: {missing}"]
    whole, soma, processes = (arrays[name] for name in TRIPLET_KEYS)
    errors: list[str] = []
    if not (whole.shape == soma.shape == processes.shape):
        errors.append(f"{label} compartment shapes differ")
        return errors
    if whole.ndim != 2:
        errors.append(f"{label} compartment arrays are not two-dimensional")
    if not (whole.dtype == soma.dtype == processes.dtype):
        errors.append(f"{label} compartment dtypes differ")
    if any(
        not np.issubdtype(value.dtype, np.integer)
        for value in (whole, soma, processes)
    ):
        errors.append(f"{label} compartment arrays do not use integer IDs")
    if any(np.any(value < 0) for value in (whole, soma, processes)):
        errors.append(f"{label} compartment arrays contain negative IDs")
    if np.any((soma > 0) & (processes > 0)):
        errors.append(f"{label} Soma and Processes overlap")
    reconstructed = np.where(soma > 0, soma, processes)
    if not np.array_equal(reconstructed, whole):
        errors.append(f"{label} Whole != Soma union Processes with matching IDs")
    ids = positive_ids(whole)
    if positive_ids(soma) != ids:
        errors.append(f"{label} Soma IDs do not match Whole IDs")
    if positive_ids(processes) != ids:
        errors.append(f"{label} Processes IDs do not match Whole IDs")
    return errors


def exact_errors(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> list[str]:
    errors = [
        *triplet_errors("reference", reference),
        *triplet_errors("candidate", candidate),
    ]
    if set(reference) != set(candidate):
        errors.append(
            "zero-delta array keys differ: "
            f"reference_only={sorted(set(reference) - set(candidate))}, "
            f"candidate_only={sorted(set(candidate) - set(reference))}"
        )
        return errors
    for name in sorted(reference):
        if reference[name].shape != candidate[name].shape:
            errors.append(
                f"zero-delta array shape changed: {name} "
                f"{reference[name].shape} -> {candidate[name].shape}"
            )
            continue
        if reference[name].dtype != candidate[name].dtype:
            errors.append(
                f"zero-delta array dtype changed: {name} "
                f"{reference[name].dtype} -> {candidate[name].dtype}"
            )
            continue
        if not np.array_equal(reference[name], candidate[name]):
            errors.append(f"zero-delta array changed: {name}")
    return errors


def normalized_mapping(plan: Mapping[str, object]) -> dict[int, int]:
    raw = plan.get("unchanged_id_map", {})
    if not isinstance(raw, dict):
        raise ValueError("unchanged_id_map must be a JSON object")
    return {int(old): int(new) for old, new in raw.items()}


def unchanged_cell_errors(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    mapping: Mapping[int, int],
) -> list[str]:
    errors: list[str] = []
    for old_id, new_id in sorted(mapping.items()):
        for name in TRIPLET_KEYS:
            old_mask = reference[name] == old_id
            new_mask = candidate[name] == new_id
            if not np.array_equal(old_mask, new_mask):
                errors.append(
                    f"untargeted cell changed in {name}: {old_id} -> {new_id}"
                )
    return errors


def mapping_coverage_errors(
    *,
    action: str,
    reference_ids: set[int],
    candidate_ids: set[int],
    selected_reference_id: int,
    selected_candidate_ids: set[int],
    mapping: Mapping[int, int],
) -> list[str]:
    expected_old = reference_ids - {selected_reference_id}
    expected_new = candidate_ids - selected_candidate_ids
    errors: list[str] = []
    if set(mapping) != expected_old:
        errors.append(
            f"{action} unchanged_id_map does not cover every untargeted "
            f"reference cell: expected={sorted(expected_old)}, "
            f"observed={sorted(mapping)}"
        )
    values = list(mapping.values())
    if len(values) != len(set(values)):
        errors.append(f"{action} unchanged_id_map repeats candidate IDs")
    if set(values) != expected_new:
        errors.append(
            f"{action} unchanged_id_map does not cover every untargeted "
            f"candidate cell: expected={sorted(expected_new)}, "
            f"observed={sorted(set(values))}"
        )
    return errors


def split_errors(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    plan: Mapping[str, object],
) -> list[str]:
    errors = [
        *triplet_errors("reference", reference),
        *triplet_errors("candidate", candidate),
    ]
    if errors:
        return errors
    ref_ids = positive_ids(reference["whole_labels"])
    cand_ids = positive_ids(candidate["whole_labels"])
    if len(cand_ids) != len(ref_ids) + 1:
        errors.append(
            f"Split must increase cell count by one: {len(ref_ids)} -> {len(cand_ids)}"
        )
    try:
        target = int(plan["target_reference_id"])
        raw_children = plan["child_candidate_ids"]
        if not isinstance(raw_children, (list, tuple)):
            raise TypeError
        children = tuple(int(value) for value in raw_children)
    except (KeyError, TypeError, ValueError):
        errors.append(
            "Split comparison plan requires integer target_reference_id and "
            "two child_candidate_ids"
        )
        return errors
    if len(children) != 2 or len(set(children)) != 2:
        errors.append("Split requires exactly two distinct child_candidate_ids")
        return errors
    if target not in ref_ids:
        errors.append(f"Split target is absent from reference: {target}")
    if not set(children).issubset(cand_ids):
        errors.append(f"Split children are absent from candidate: {children}")
        return errors
    parent = reference["whole_labels"] == target
    for child in children:
        if not np.any(candidate["soma_labels"] == child):
            errors.append(f"Split child has no Soma: {child}")
        if not np.any(parent & (candidate["whole_labels"] == child)):
            errors.append(
                f"Split child has no overlap with the selected parent: {child}"
            )
    child_union = np.isin(candidate["whole_labels"], children)
    if not np.any(parent & child_union):
        errors.append("Split children have no overlap with the selected parent")
    try:
        mapping = normalized_mapping(plan)
    except (TypeError, ValueError):
        errors.append("Split unchanged_id_map is invalid")
        return errors
    errors.extend(
        mapping_coverage_errors(
            action="Split",
            reference_ids=ref_ids,
            candidate_ids=cand_ids,
            selected_reference_id=target,
            selected_candidate_ids=set(children),
            mapping=mapping,
        )
    )
    errors.extend(
        unchanged_cell_errors(
            reference,
            candidate,
            mapping,
        )
    )
    return errors


def enlarge_errors(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    plan: Mapping[str, object],
) -> list[str]:
    errors = [
        *triplet_errors("reference", reference),
        *triplet_errors("candidate", candidate),
    ]
    if errors:
        return errors
    ref_ids = positive_ids(reference["whole_labels"])
    cand_ids = positive_ids(candidate["whole_labels"])
    if len(cand_ids) != len(ref_ids):
        errors.append(
            f"Enlarge must preserve cell count: {len(ref_ids)} -> {len(cand_ids)}"
        )
    try:
        old_id = int(plan["target_reference_id"])
        new_id = int(plan.get("target_candidate_id", old_id))
    except (KeyError, TypeError, ValueError):
        errors.append(
            "Enlarge comparison plan requires an integer target_reference_id"
        )
        return errors
    old_whole = reference["whole_labels"] == old_id
    new_whole = candidate["whole_labels"] == new_id
    old_soma = reference["soma_labels"] == old_id
    new_soma = candidate["soma_labels"] == new_id
    if np.any(old_whole & ~new_whole):
        errors.append("Enlarge removed pixels from the selected Whole")
    if np.any(old_soma & ~new_soma):
        errors.append("Enlarge removed pixels from the selected Soma")
    if not np.any(new_soma & ~old_soma):
        errors.append("Enlarge did not add any Soma pixels")
    if np.any(new_soma & ~new_whole):
        errors.append("Enlarge added Soma pixels outside the updated Whole")
    try:
        mapping = normalized_mapping(plan)
    except (TypeError, ValueError):
        errors.append("Enlarge unchanged_id_map is invalid")
        return errors
    errors.extend(
        mapping_coverage_errors(
            action="Enlarge",
            reference_ids=ref_ids,
            candidate_ids=cand_ids,
            selected_reference_id=old_id,
            selected_candidate_ids={new_id},
            mapping=mapping,
        )
    )
    errors.extend(
        unchanged_cell_errors(
            reference,
            candidate,
            mapping,
        )
    )
    return errors


def gfap_only_errors(candidate: Mapping[str, np.ndarray]) -> list[str]:
    errors = triplet_errors("candidate", candidate)
    if errors:
        return errors
    whole = candidate["whole_labels"]
    soma = candidate["soma_labels"]
    if not positive_ids(whole):
        errors.append("GFAP-only result contains no Astrocyte instances")
    for cell_id in sorted(positive_ids(whole)):
        if not np.any(soma == cell_id):
            errors.append(f"GFAP-only cell has no Soma: {cell_id}")
    return errors


def compare(
    *,
    profile: str,
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray] | None,
    plan: Mapping[str, object],
) -> list[str]:
    if profile in {"mature_zero_delta", "scientific_zero_delta"}:
        if reference is None:
            return [f"{profile} requires a reference NPZ"]
        return exact_errors(reference, candidate)
    if profile == "split":
        if reference is None:
            return ["split requires a reference NPZ"]
        return split_errors(reference, candidate, plan)
    if profile == "enlarge":
        if reference is None:
            return ["enlarge requires a reference NPZ"]
        return enlarge_errors(reference, candidate, plan)
    if profile == "gfap_only":
        return gfap_only_errors(candidate)
    return [f"unknown comparison profile: {profile}"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only scientific-array comparator for analysis samples"
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=(
            "mature_zero_delta",
            "scientific_zero_delta",
            "split",
            "enlarge",
            "gfap_only",
        ),
    )
    parser.add_argument("--candidate-npz", required=True, type=Path)
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument("--plan-json", type=Path)
    args = parser.parse_args()

    candidate = load_arrays(args.candidate_npz)
    reference = load_arrays(args.reference_npz) if args.reference_npz else None
    plan = load_plan(args.plan_json)
    errors = compare(
        profile=args.profile,
        candidate=candidate,
        reference=reference,
        plan=plan,
    )
    if errors:
        print(f"FAIL: {len(errors)} comparison error(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {args.profile} array contract satisfied")
    if args.profile == "gfap_only":
        print("NOTE: full-resolution visual biological review is still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
