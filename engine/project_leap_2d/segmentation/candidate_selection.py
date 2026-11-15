# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def rank_candidates(masks: list[np.ndarray], rows: list[dict]) -> int:
    count = len(masks)
    if count == 0:
        raise ValueError("No candidate masks to rank")
    if count == 1:
        rows[0].update({"mean_candidate_iou": 1.0, "auto_selection_score": 1.0, "auto_selected": True})
        return 0

    mean_iou = np.zeros(count, dtype=np.float64)
    mask_areas = np.asarray(
        [np.count_nonzero(mask) for mask in masks],
        dtype=np.int64,
    )
    intersection_buffer = np.empty_like(masks[0], dtype=bool)
    for left in range(count):
        for right in range(left + 1, count):
            np.logical_and(
                masks[left],
                masks[right],
                out=intersection_buffer,
            )
            intersection = int(np.count_nonzero(intersection_buffer))
            union = int(mask_areas[left] + mask_areas[right] - intersection)
            iou = intersection / union if union else 1.0
            mean_iou[left] += iou
            mean_iou[right] += iou
    mean_iou /= count - 1

    def rank01(values: list[float], higher_is_better: bool = True) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        ranks = np.asarray([(array < value).mean() + 0.5 * (array == value).mean() for value in array])
        return ranks if higher_is_better else 1.0 - ranks

    def raw_metric(row: dict, key: str) -> float:
        return float(row.get(f"_raw_{key}", row[key]))

    coverage_rank = rank01([raw_metric(row, "structural_signal_coverage") for row in rows])
    precision_rank = rank01([raw_metric(row, "structural_precision") for row in rows])
    unsupported_rank = rank01(
        [raw_metric(row, "unsupported_wide_fraction") for row in rows],
        higher_is_better=False,
    )
    soma_counts = np.asarray(
        [float(row["soma_supported_components"]) for row in rows],
        dtype=np.float64,
    )
    soma_consensus_cap = float(np.percentile(soma_counts, 75))
    soma_rank = rank01(np.minimum(soma_counts, soma_consensus_cap).tolist())
    unanchored_rank = rank01(
        [raw_metric(row, "unanchored_area_fraction") for row in rows],
        higher_is_better=False,
    )
    edge_rank = rank01(
        [raw_metric(row, "edge_proximity_area_fraction") for row in rows],
        higher_is_better=False,
    )
    border_burden_rank = rank01(
        [raw_metric(row, "border_removed_area_fraction") for row in rows],
        higher_is_better=False,
    )
    preserved_border_rank = rank01(
        [raw_metric(row, "border_preserved_complete_area_fraction") for row in rows],
        higher_is_better=False,
    )
    score = (
        0.27 * mean_iou
        + 0.10 * coverage_rank
        + 0.15 * precision_rank
        + 0.09 * unsupported_rank
        + 0.20 * soma_rank
        + 0.08 * unanchored_rank
        + 0.02 * edge_rank
        + 0.01 * border_burden_rank
        + 0.04 * preserved_border_rank
    )
    best = int(np.argmax(score))
    for index, row in enumerate(rows):
        row.update(
            {
                "mean_candidate_iou": round(float(mean_iou[index]), 6),
                "_raw_mean_candidate_iou": float(mean_iou[index]),
                "auto_selection_score": round(float(score[index]), 6),
                "_raw_auto_selection_score": float(score[index]),
                "auto_selected": index == best,
            }
        )
    return best

def rank_production_candidates(masks: list[np.ndarray], rows: list[dict]) -> int:
    eligible = [index for index, row in enumerate(rows) if not row.get("error")]
    if not eligible:
        errors = "\n".join(
            f"candidate {row.get('candidate', index + 1)}: {row.get('error')}"
            for index, row in enumerate(rows)
        )
        raise RuntimeError(f"All ROI candidates failed or used an exception fallback:\n{errors}")
    for index, row in enumerate(rows):
        if index not in eligible:
            row.update(
                {
                    "mean_candidate_iou": 0.0,
                    "auto_selection_score": -1.0,
                    "auto_selected": False,
                    "selection_eligible": False,
                }
            )
    eligible_masks = [masks[index] for index in eligible]
    eligible_rows = [rows[index] for index in eligible]
    best_eligible_position = rank_candidates(eligible_masks, eligible_rows)
    for row in eligible_rows:
        row["selection_eligible"] = True
    return eligible[best_eligible_position]

def weighted_rank01(
    values: list[float],
    weights: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if array.shape != weights.shape or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Invalid candidate ranking weights")
    weights /= float(weights.sum())
    ranks = np.asarray(
        [
            float(weights[array < value].sum())
            + 0.5 * float(weights[array == value].sum())
            for value in array
        ],
        dtype=np.float64,
    )
    return ranks if higher_is_better else 1.0 - ranks

def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    ordered_values, inverse = np.unique(
        np.asarray(values, dtype=np.float64),
        return_inverse=True,
    )
    ordered_weights = np.zeros(len(ordered_values), dtype=np.float64)
    np.add.at(ordered_weights, inverse, np.asarray(weights, dtype=np.float64))
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    target = float(np.clip(percentile / 100.0, 0.0, 1.0)) * float(
        ordered_weights.sum(dtype=np.float64)
    )
    position = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])

def cluster_near_duplicate_candidates(
    indices: np.ndarray,
    pairwise_iou: np.ndarray,
) -> list[np.ndarray]:
    parent = {int(index): int(index) for index in indices}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    integer_indices = [int(index) for index in indices]
    for left_position, left in enumerate(integer_indices):
        for right in integer_indices[left_position + 1 :]:
            if pairwise_iou[left, right] >= NEAR_DUPLICATE_CANDIDATE_IOU:
                union(left, right)
    clusters: dict[int, list[int]] = {}
    for index in integer_indices:
        clusters.setdefault(find(index), []).append(index)
    return [
        np.asarray(clusters[root], dtype=np.int64)
        for root in sorted(clusters)
    ]

def family_z_balance_structure(
    rows: list[dict],
    pairwise_iou: np.ndarray,
) -> tuple[dict[tuple[str, str], list[np.ndarray]], np.ndarray]:
    cells: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["candidate_family"]), str(row["z_mode"]))
        cells.setdefault(key, []).append(index)
    clustered_cells: dict[tuple[str, str], list[np.ndarray]] = {}
    candidate_weights = np.zeros(len(rows), dtype=np.float64)
    cell_weight = 1.0 / max(len(cells), 1)
    for cell_key in sorted(cells):
        clusters = cluster_near_duplicate_candidates(
            np.asarray(cells[cell_key], dtype=np.int64),
            pairwise_iou,
        )
        clustered_cells[cell_key] = clusters
        cluster_weight = cell_weight / max(len(clusters), 1)
        for cluster in clusters:
            # A near-duplicate cluster contributes one fixed vote.  Its
            # lowest-index member is the deterministic representative, so
            # appending an exact or near duplicate cannot change the metric
            # reference distribution or amplify that candidate family.
            candidate_weights[int(cluster[0])] = cluster_weight
    candidate_weights /= float(candidate_weights.sum(dtype=np.float64))
    return clustered_cells, candidate_weights

def rank_candidates_family_balanced(
    masks: list[np.ndarray],
    rows: list[dict],
    *,
    expected_families: tuple[str, ...] = EXPECTED_CANDIDATE_FAMILIES,
    expected_z_interval_count: int = EXPECTED_Z_INTERVAL_COUNT,
) -> int:
    count = len(masks)
    if count == 0:
        raise ValueError("No candidate masks to rank")
    if count == 1:
        rows[0].update(
            {
                "mean_candidate_iou": 1.0,
                "_raw_mean_candidate_iou": 1.0,
                "auto_selection_score": 1.0,
                "_raw_auto_selection_score": 1.0,
                "auto_selected": True,
            }
        )
        return 0

    pairwise_iou = np.eye(count, dtype=np.float64)
    mask_areas = np.asarray(
        [np.count_nonzero(mask) for mask in masks],
        dtype=np.int64,
    )
    intersection_buffer = np.empty_like(masks[0], dtype=bool)
    for left in range(count):
        for right in range(left + 1, count):
            np.logical_and(
                masks[left],
                masks[right],
                out=intersection_buffer,
            )
            intersection = int(np.count_nonzero(intersection_buffer))
            union = int(mask_areas[left] + mask_areas[right] - intersection)
            iou = intersection / union if union else 1.0
            pairwise_iou[left, right] = iou
            pairwise_iou[right, left] = iou

    clustered_cells, candidate_weights = family_z_balance_structure(
        rows,
        pairwise_iou,
    )
    observed_families = {str(row["candidate_family"]) for row in rows}
    observed_z_modes = {str(row["z_mode"]) for row in rows}
    if observed_families != set(expected_families):
        raise RuntimeError(
            "Family-Z candidate balance is missing a predefined candidate family"
        )
    if expected_z_interval_count <= 0:
        raise ValueError("Expected Z-interval count must be positive")
    expected_cell_count = len(expected_families) * expected_z_interval_count
    if (
        len(observed_z_modes) != expected_z_interval_count
        or len(clustered_cells) != expected_cell_count
    ):
        raise RuntimeError(
            f"Family-Z candidate balance requires all {expected_cell_count} "
            "predefined cells"
        )
    mean_iou = np.zeros(count, dtype=np.float64)
    for index in range(count):
        cell_means: list[float] = []
        for clusters in clustered_cells.values():
            cluster_means: list[float] = []
            for cluster in clusters:
                if np.any(cluster == index):
                    continue
                cluster_means.append(
                    float(pairwise_iou[index, int(cluster[0])])
                )
            if cluster_means:
                cell_means.append(float(np.mean(cluster_means, dtype=np.float64)))
        mean_iou[index] = float(np.mean(cell_means, dtype=np.float64))

    def raw_metric(row: dict, key: str) -> float:
        return float(row.get(f"_raw_{key}", row[key]))

    coverage_rank = weighted_rank01(
        [raw_metric(row, "structural_signal_coverage") for row in rows],
        candidate_weights,
    )
    precision_rank = weighted_rank01(
        [raw_metric(row, "structural_precision") for row in rows],
        candidate_weights,
    )
    unsupported_rank = weighted_rank01(
        [raw_metric(row, "unsupported_wide_fraction") for row in rows],
        candidate_weights,
        higher_is_better=False,
    )
    soma_counts = np.asarray(
        [float(row["soma_supported_components"]) for row in rows],
        dtype=np.float64,
    )
    soma_consensus_cap = weighted_percentile(
        soma_counts,
        candidate_weights,
        75.0,
    )
    soma_rank = weighted_rank01(
        np.minimum(soma_counts, soma_consensus_cap).tolist(),
        candidate_weights,
    )
    unanchored_rank = weighted_rank01(
        [raw_metric(row, "unanchored_area_fraction") for row in rows],
        candidate_weights,
        higher_is_better=False,
    )
    edge_rank = weighted_rank01(
        [raw_metric(row, "edge_proximity_area_fraction") for row in rows],
        candidate_weights,
        higher_is_better=False,
    )
    border_burden_rank = weighted_rank01(
        [raw_metric(row, "border_removed_area_fraction") for row in rows],
        candidate_weights,
        higher_is_better=False,
    )
    preserved_border_rank = weighted_rank01(
        [raw_metric(row, "border_preserved_complete_area_fraction") for row in rows],
        candidate_weights,
        higher_is_better=False,
    )
    score = (
        0.27 * mean_iou
        + 0.10 * coverage_rank
        + 0.15 * precision_rank
        + 0.09 * unsupported_rank
        + 0.20 * soma_rank
        + 0.08 * unanchored_rank
        + 0.02 * edge_rank
        + 0.01 * border_burden_rank
        + 0.04 * preserved_border_rank
    )
    best = int(np.argmax(score))
    for index, row in enumerate(rows):
        row.update(
            {
                "mean_candidate_iou": round(float(mean_iou[index]), 6),
                "_raw_mean_candidate_iou": float(mean_iou[index]),
                "auto_selection_score": round(float(score[index]), 6),
                "_raw_auto_selection_score": float(score[index]),
                "auto_selected": index == best,
            }
        )
    return best

def challenger_dominates_incumbent(
    challenger: dict,
    incumbent: dict,
) -> tuple[bool, dict[str, object]]:
    def raw(row: dict, key: str) -> float:
        return float(row.get(f"_raw_{key}", row[key]))

    higher_metrics = (
        "structural_signal_coverage",
        "structural_precision",
    )
    lower_metrics = (
        "unsupported_wide_fraction",
        "unanchored_area_fraction",
        "edge_proximity_area_fraction",
        "border_removed_area_fraction",
        "border_preserved_complete_area_fraction",
    )
    comparisons: dict[str, bool] = {
        "same_z_interval": (
            int(challenger["z_start_0based"])
            == int(incumbent["z_start_0based"])
            and int(challenger["z_end_0based_inclusive"])
            == int(incumbent["z_end_0based_inclusive"])
            and str(challenger["projection"]) == str(incumbent["projection"])
        )
    }
    for key in higher_metrics:
        comparisons[f"{key}_not_lower"] = raw(challenger, key) >= raw(
            incumbent, key
        )
    for key in lower_metrics:
        comparisons[f"{key}_not_higher"] = raw(challenger, key) <= raw(
            incumbent, key
        )
    comparisons["soma_supported_components_equal"] = int(
        challenger["soma_supported_components"]
    ) == int(incumbent["soma_supported_components"])
    comparisons["incomplete_border_components_not_higher"] = int(
        challenger["final_incomplete_border_touching_components"]
    ) <= int(incumbent["final_incomplete_border_touching_components"])
    comparisons["family_balanced_score_margin"] = float(
        challenger["_raw_auto_selection_score"]
    ) >= (
        float(incumbent["_raw_auto_selection_score"])
        + CHALLENGER_MIN_SCORE_MARGIN
    )
    passed = bool(all(comparisons.values()))
    return passed, {
        "passed": passed,
        "comparisons": comparisons,
    }

def rank_pre_distribution_baseline_candidates(
    masks: list[np.ndarray],
    rows: list[dict],
    *,
    morphology_baseline_count: int = 30,
    expected_z_interval_count: int = EXPECTED_Z_INTERVAL_COUNT,
) -> tuple[int, dict[str, object]]:
    if len(masks) != len(rows):
        raise ValueError("Candidate mask and row counts do not match")
    if morphology_baseline_count <= 0 or morphology_baseline_count > len(masks):
        raise ValueError("Invalid morphology-baseline candidate count")
    morphology_baseline_errors = [
        row["candidate"]
        for row in rows[:morphology_baseline_count]
        if row.get("error")
    ]
    if morphology_baseline_errors:
        raise RuntimeError(
            "Frozen morphology baseline is incomplete; failed candidates: "
            f"{morphology_baseline_errors}"
        )

    morphology_baseline_rows = [
        dict(row) for row in rows[:morphology_baseline_count]
    ]
    morphology_baseline_position = rank_production_candidates(
        masks[:morphology_baseline_count],
        morphology_baseline_rows,
    )
    for index, morphology_baseline_row in enumerate(morphology_baseline_rows):
        rows[index]["morphology_baseline_mean_candidate_iou"] = (
            morphology_baseline_row.get(
                "mean_candidate_iou"
            )
        )
        rows[index]["morphology_baseline_auto_selection_score"] = (
            morphology_baseline_row.get("auto_selection_score")
        )
        rows[index]["_raw_morphology_baseline_mean_candidate_iou"] = (
            morphology_baseline_row.get("_raw_mean_candidate_iou")
        )
        rows[index]["_raw_morphology_baseline_auto_selection_score"] = (
            morphology_baseline_row.get("_raw_auto_selection_score")
        )
        rows[index]["morphology_baseline_auto_selected"] = (
            index == morphology_baseline_position
        )

    eligible = [index for index, row in enumerate(rows) if not row.get("error")]
    if not eligible:
        raise RuntimeError(f"All {len(rows)} ROI candidates failed")
    for index, row in enumerate(rows):
        row["selection_eligible"] = index in eligible
        if index not in eligible:
            row.update(
                {
                    "mean_candidate_iou": 0.0,
                    "auto_selection_score": -1.0,
                    "auto_selected": False,
                }
            )
    if len(eligible) != len(rows):
        challenger_position = morphology_baseline_position
        chosen_position = morphology_baseline_position
        rows[morphology_baseline_position].update(
            {
                "mean_candidate_iou": morphology_baseline_rows[
                    morphology_baseline_position
                ]["mean_candidate_iou"],
                "_raw_mean_candidate_iou": morphology_baseline_rows[
                    morphology_baseline_position
                ]["_raw_mean_candidate_iou"],
                "auto_selection_score": morphology_baseline_rows[
                    morphology_baseline_position
                ]["auto_selection_score"],
                "_raw_auto_selection_score": morphology_baseline_rows[
                    morphology_baseline_position
                ]["_raw_auto_selection_score"],
            }
        )
        guard = {
            "passed": False,
            "reason": (
                "candidate_error_fail_closed_to_"
                "morphology_baseline_incumbent"
            ),
            "comparisons": {},
        }
    else:
        rank_candidates_family_balanced(
            masks,
            rows,
            expected_families=PRE_DISTRIBUTION_BASELINE_CANDIDATE_FAMILIES,
            expected_z_interval_count=expected_z_interval_count,
        )
        incumbent_z_key = (
            int(rows[morphology_baseline_position]["z_start_0based"]),
            int(rows[morphology_baseline_position]["z_end_0based_inclusive"]),
            str(rows[morphology_baseline_position]["projection"]),
        )
        same_z_candidates = [
            index
            for index, row in enumerate(rows)
            if (
                int(row["z_start_0based"]),
                int(row["z_end_0based_inclusive"]),
                str(row["projection"]),
            )
            == incumbent_z_key
        ]
        challenger_position = max(
            same_z_candidates,
            key=lambda index: (
                float(rows[index]["_raw_auto_selection_score"]),
                -index,
            ),
        )
        if challenger_position == morphology_baseline_position:
            chosen_position = morphology_baseline_position
            guard = {
                "passed": True,
                "reason": (
                    "family_z_balanced_rank_retained_"
                    "morphology_baseline_incumbent"
                ),
                "comparisons": {},
            }
        else:
            passed, guard = challenger_dominates_incumbent(
                rows[challenger_position],
                rows[morphology_baseline_position],
            )
            guard["reason"] = (
                "same_z_challenger_dominated_incumbent"
                if passed
                else "challenger_rejected_by_non_regression_guard"
            )
            chosen_position = (
                challenger_position if passed else morphology_baseline_position
            )

    for index, row in enumerate(rows):
        row["family_balanced_challenger"] = index == challenger_position
        row["auto_selected"] = index == chosen_position
    morphology_baseline_candidate = rows[morphology_baseline_position]["candidate"]
    challenger_candidate = rows[challenger_position]["candidate"]
    selected_candidate = rows[chosen_position]["candidate"]
    details = {
        "morphology_baseline_incumbent_candidate": morphology_baseline_candidate,
        "family_balanced_challenger_candidate": challenger_candidate,
        "selected_candidate": selected_candidate,
        "guard": guard,
    }
    rows[chosen_position].update(
        {
            "selection_guard_reason": str(guard["reason"]),
            "morphology_baseline_incumbent_candidate": (
                morphology_baseline_candidate
            ),
            "morphology_baseline_incumbent_score": morphology_baseline_rows[
                morphology_baseline_position
            ].get("auto_selection_score"),
            "family_balanced_challenger_candidate": challenger_candidate,
            "family_balanced_challenger_score": rows[challenger_position].get(
                "auto_selection_score"
            ),
        }
    )
    return chosen_position, details

def rank_position_production_candidates(
    masks: list[np.ndarray],
    rows: list[dict],
) -> tuple[int, dict[str, object]]:
    """Rank one Z position's 6 + 6 + 6 production candidates.

    Candidate agreement, percentile references, and incumbent/challenger guards
    are deliberately confined to this single physical Z interval.  The best
    row is marked with ``position_auto_selected``; ``auto_selected`` remains
    reserved for the later comparison between the five position-best candidates.
    """
    morphology_count = 6
    pre_distribution_count = 12
    complete_count = 18
    if len(masks) != len(rows):
        raise ValueError("Candidate mask and row counts do not match")
    if len(rows) != complete_count:
        raise ValueError(
            "Position-local production ranking requires exactly 18 candidates"
        )
    expected_modules = (
        *("morphology_baseline",) * 6,
        *("structural_refinement",) * 6,
        *("distributional_threshold",) * 6,
    )
    observed_modules = tuple(str(row.get("candidate_module", "")) for row in rows)
    if observed_modules != expected_modules:
        raise ValueError(
            "Position-local candidates must be ordered as 6 Morphology Baseline, "
            "6 Structural Refinement, and 6 Distributional Threshold candidates"
        )
    z_keys = {
        (
            int(row["z_start_0based"]),
            int(row["z_end_0based_inclusive"]),
            str(row["projection"]),
            str(row["z_mode"]),
        )
        for row in rows
    }
    if len(z_keys) != 1:
        raise ValueError(
            "Position-local ranking cannot compare candidates from different Z intervals"
        )
    candidate_ids = [row["candidate"] for row in rows]
    if len(set(candidate_ids)) != complete_count:
        raise ValueError("Position-local candidate identifiers must be unique")
    position_by_candidate = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }

    pre_distribution_rows = [
        dict(row) for row in rows[:pre_distribution_count]
    ]
    pre_distribution_position, pre_distribution_details = (
        rank_pre_distribution_baseline_candidates(
            masks[:pre_distribution_count],
            pre_distribution_rows,
            morphology_baseline_count=morphology_count,
            expected_z_interval_count=1,
        )
    )
    morphology_candidate = pre_distribution_details[
        "morphology_baseline_incumbent_candidate"
    ]
    morphology_position = position_by_candidate[morphology_candidate]
    pre_distribution_candidate = pre_distribution_rows[
        pre_distribution_position
    ]["candidate"]
    pre_distribution_full_position = position_by_candidate[
        pre_distribution_candidate
    ]

    morphology_fields = (
        "morphology_baseline_mean_candidate_iou",
        "morphology_baseline_auto_selection_score",
        "_raw_morphology_baseline_mean_candidate_iou",
        "_raw_morphology_baseline_auto_selection_score",
        "morphology_baseline_auto_selected",
    )
    for index, row in enumerate(rows):
        row["pre_distribution_baseline_auto_selected"] = (
            index == pre_distribution_full_position
        )
        row["position_auto_selected"] = False
        row["auto_selected"] = False
        if index < pre_distribution_count:
            baseline_row = pre_distribution_rows[index]
            row["pre_distribution_baseline_frozen_mean_candidate_iou"] = (
                baseline_row.get("mean_candidate_iou")
            )
            row["pre_distribution_baseline_frozen_auto_selection_score"] = (
                baseline_row.get("auto_selection_score")
            )
            row["_raw_pre_distribution_baseline_frozen_mean_candidate_iou"] = (
                baseline_row.get("_raw_mean_candidate_iou")
            )
            row[
                "_raw_pre_distribution_baseline_frozen_auto_selection_score"
            ] = baseline_row.get("_raw_auto_selection_score")
            for field_name in morphology_fields:
                if field_name in baseline_row:
                    row[field_name] = baseline_row[field_name]

    failed_candidates = [row["candidate"] for row in rows if row.get("error")]
    if failed_candidates:
        challenger_position = pre_distribution_full_position
        chosen_position = pre_distribution_full_position
        for index, row in enumerate(rows):
            row["selection_eligible"] = (
                index < pre_distribution_count and not bool(row.get("error"))
            )
            if index < pre_distribution_count:
                baseline_row = pre_distribution_rows[index]
                for field_name in (
                    "mean_candidate_iou",
                    "_raw_mean_candidate_iou",
                    "auto_selection_score",
                    "_raw_auto_selection_score",
                ):
                    if field_name in baseline_row:
                        row[field_name] = baseline_row[field_name]
            else:
                row.update(
                    {
                        "mean_candidate_iou": 0.0,
                        "_raw_mean_candidate_iou": 0.0,
                        "auto_selection_score": -1.0,
                        "_raw_auto_selection_score": -1.0,
                    }
                )
        guard: dict[str, object] = {
            "passed": False,
            "reason": (
                "candidate_error_fail_closed_to_"
                "pre_distribution_baseline_incumbent"
            ),
            "comparisons": {},
            "failed_candidates": failed_candidates,
        }
    else:
        for row in rows:
            row["selection_eligible"] = True
        rank_candidates_family_balanced(
            masks,
            rows,
            expected_families=EXPECTED_CANDIDATE_FAMILIES,
            expected_z_interval_count=1,
        )
        challenger_position = max(
            range(complete_count),
            key=lambda index: (
                float(rows[index]["_raw_auto_selection_score"]),
                -index,
            ),
        )
        if challenger_position == pre_distribution_full_position:
            chosen_position = pre_distribution_full_position
            guard = {
                "passed": True,
                "reason": (
                    "family_balanced_rank_retained_"
                    "pre_distribution_baseline_incumbent"
                ),
                "comparisons": {},
            }
        else:
            passed, guard = challenger_dominates_incumbent(
                rows[challenger_position],
                rows[pre_distribution_full_position],
            )
            guard["reason"] = (
                "same_z_challenger_dominated_"
                "pre_distribution_baseline_incumbent"
                if passed
                else (
                    "challenger_rejected_by_"
                    "pre_distribution_baseline_non_regression_guard"
                )
            )
            chosen_position = (
                challenger_position
                if passed
                else pre_distribution_full_position
            )

    challenger_candidate = rows[challenger_position]["candidate"]
    selected_candidate = rows[chosen_position]["candidate"]
    for index, row in enumerate(rows):
        row["family_balanced_challenger"] = index == challenger_position
        row["position_auto_selected"] = index == chosen_position
        row["auto_selected"] = False
    details = {
        "morphology_baseline_incumbent_candidate": morphology_candidate,
        "pre_distribution_baseline_incumbent_candidate": (
            pre_distribution_candidate
        ),
        "family_balanced_challenger_candidate": challenger_candidate,
        "position_selected_candidate": selected_candidate,
        "pre_distribution_baseline_selection": pre_distribution_details,
        "guard": guard,
    }
    rows[chosen_position].update(
        {
            "position_selection_guard_reason": str(guard["reason"]),
            "morphology_baseline_incumbent_candidate": morphology_candidate,
            "morphology_baseline_incumbent_score": pre_distribution_rows[
                morphology_position
            ].get(
                "morphology_baseline_auto_selection_score",
                pre_distribution_rows[morphology_position].get(
                    "auto_selection_score"
                ),
            ),
            "pre_distribution_baseline_incumbent_candidate": (
                pre_distribution_candidate
            ),
            "pre_distribution_baseline_incumbent_score_frozen": (
                pre_distribution_rows[pre_distribution_position].get(
                    "auto_selection_score"
                )
            ),
            "pre_distribution_baseline_incumbent_score_complete_pool": rows[
                pre_distribution_full_position
            ].get("auto_selection_score"),
            "family_balanced_challenger_candidate": challenger_candidate,
            "family_balanced_challenger_score": rows[challenger_position].get(
                "auto_selection_score"
            ),
        }
    )
    return chosen_position, details
