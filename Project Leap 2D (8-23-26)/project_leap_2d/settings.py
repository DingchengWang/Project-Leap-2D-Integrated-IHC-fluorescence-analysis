# This functional source module is assembled into one shared runtime.
from __future__ import annotations

PRODUCT_DISPLAY_NAME = "Project Leap 2D"

ANALYSIS_CORE_NAME = "Project Leap 2D Analysis Core"

PIPELINE_NAME = ANALYSIS_CORE_NAME

TERMINAL_RULE = "/" * 112

_RUN_STARTED_AT: float | None = None

DEFAULT_INPUT_DIR = Path.home() / "Desktop" / "IHC IMAGE"

DEFAULT_OUT_DIR = DEFAULT_INPUT_DIR

WHOLE_OVERLAY_FILENAME = "IHC_2D_Whole_Astrocyte_Overlay.png"

SOMA_OVERLAY_FILENAME = "IHC_2D_Astrocyte_Soma_Overlay.png"

PROCESS_OVERLAY_FILENAME = "IHC_2D_Astrocyte_Processes_Overlay.png"

REPORT_FILENAME = "IHC_2D_Analysis_Report.txt"

WORKBOOK_FILENAME = "IHC_2D_Fluorescence_Results.xlsx"

DEBUG_WHOLE_OVERLAY_FILENAME = "IHC_2D_DEBUG_Whole_Overlay.png"

DEBUG_SOMA_OVERLAY_FILENAME = "IHC_2D_DEBUG_Soma_Overlay.png"

DEBUG_PROCESS_OVERLAY_FILENAME = "IHC_2D_DEBUG_Processes_Overlay.png"

DEBUG_REPORT_FILENAME = "IHC_2D_DEBUG_Report.txt"

DEBUG_STATE_FILENAME = "IHC_2D_DEBUG_Compartment_State.npz"

STRUCTURAL_CHANNELS = ("eGFP", "GFAP")

MEASUREMENT_CHANNELS = ("KCNN1", "KCNN2", "KCNN3", "KCNJ10")

CHANNEL_PATTERNS = {
    "DAPI": re.compile(r"(?<![a-z0-9])dapi(?![a-z0-9])", re.IGNORECASE),
    "eGFP": re.compile(r"(?<![a-z0-9])(?:e?gfp)(?![a-z0-9])", re.IGNORECASE),
    "GFAP": re.compile(r"(?<![a-z0-9])gfap(?![a-z0-9])", re.IGNORECASE),
    "KCNN1": re.compile(r"(?<![a-z0-9])(?:kcnn1|sk1)(?![a-z0-9])", re.IGNORECASE),
    "KCNN2": re.compile(r"(?<![a-z0-9])(?:kcnn2|sk2)(?![a-z0-9])", re.IGNORECASE),
    "KCNN3": re.compile(r"(?<![a-z0-9])(?:kcnn3|sk3)(?![a-z0-9])", re.IGNORECASE),
    "KCNJ10": re.compile(
        r"(?<![a-z0-9])(?:kcnj10|kir4(?:[._ -]?1)?)(?![a-z0-9])",
        re.IGNORECASE,
    ),
}

AGE_PROFILE_PATTERNS = {
    "neonatal": re.compile(r"(?<![a-z0-9])neonatal(?![a-z0-9])", re.IGNORECASE),
    "mature": re.compile(r"(?<![a-z0-9])mature(?![a-z0-9])", re.IGNORECASE),
}

AGE_PROFILE_THRESHOLD = 0.55

_CELLPOSE_MODEL = None

_CELLPOSE_DEVICE = None

_CELLPOSE_WORKING_MAX_SIDE = None

_CELLPOSE_EFFECTIVE_BATCH_SIZE: int | None = None

_CELLPOSE_MASK_CACHE: dict[tuple, tuple[np.ndarray, str]] = {}

CELLPOSE_BATCH_SIZE = 32

CELLPOSE_BATCH_FALLBACKS = (16, 8)

CELLPOSE_PRIOR_MIN_AREA_PX = 55

CANDIDATE_CPU_WORKERS = 12

DAPI_INVENTORY_CPU_WORKERS = 12

REVIEW_MERGE_MAX_SOMA_GAP_UM = 1.0

_EFFECTIVE_CANDIDATE_CPU_WORKERS = CANDIDATE_CPU_WORKERS

_EFFECTIVE_DAPI_INVENTORY_CPU_WORKERS = DAPI_INVENTORY_CPU_WORKERS

NEAR_DUPLICATE_CANDIDATE_IOU = 0.995

CHALLENGER_MIN_SCORE_MARGIN = 0.01

MORPHOLOGY_BASELINE_CANDIDATE_COUNT = 30

STRUCTURAL_REFINEMENT_CANDIDATE_COUNT = 30

DISTRIBUTIONAL_THRESHOLD_CANDIDATE_COUNT = 30

PRE_DISTRIBUTION_BASELINE_CANDIDATE_COUNT = (
    MORPHOLOGY_BASELINE_CANDIDATE_COUNT
    + STRUCTURAL_REFINEMENT_CANDIDATE_COUNT
)

TOTAL_CANDIDATE_COUNT = (
    PRE_DISTRIBUTION_BASELINE_CANDIDATE_COUNT
    + DISTRIBUTIONAL_THRESHOLD_CANDIDATE_COUNT
)

EXPECTED_Z_INTERVAL_COUNT = 5

PRE_DISTRIBUTION_BASELINE_PROFILES_PER_Z = 12

EXPECTED_PROFILES_PER_Z = 18

PRE_DISTRIBUTION_BASELINE_CANDIDATE_FAMILIES = (
    "process_sensitivity",
    "balanced_adaptive",
    "precision",
    "strict_merge",
    "channel_consensus",
    "topology_continuity",
)

EXPECTED_CANDIDATE_FAMILIES = (
    *PRE_DISTRIBUTION_BASELINE_CANDIDATE_FAMILIES,
    "distributional_threshold",
)

_BRANCH_FEATURE_CACHE: dict[
    tuple,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
] = {}

_CACHE_LOCK = threading.RLock()

_DAPI_NUCLEI_CACHE: dict[tuple, np.ndarray] = {}

_FULL_PERCENTILE_CACHE: dict[
    tuple, tuple[weakref.ReferenceType[np.ndarray], float]
] = {}

_FULL_SUM_CACHE: dict[tuple, tuple[weakref.ReferenceType[np.ndarray], float]] = {}

_TOP_HAT_CACHE: dict[tuple, tuple[np.ndarray, float]] = {}

_NORMALIZED_PROJECTION_CACHE: dict[
    tuple, tuple[weakref.ReferenceType[np.ndarray], np.ndarray]
] = {}

_CANDIDATE_BASE_CACHE: dict[tuple, "CandidateBaseResult"] = {}

_CANDIDATE_BASE_LOCKS: dict[tuple, threading.Lock] = {}

_CACHE_KEY_LOCKS: dict[tuple, threading.Lock] = {}

_DISTRIBUTION_MODEL_CACHE: dict[tuple, "Log1pGMMThreshold"] = {}

_DISTRIBUTION_MODEL_FAILURES: dict[tuple, str] = {}

_DISTRIBUTION_DIAGNOSTIC_CACHE: dict[tuple, dict[str, object]] = {}

_RUNTIME_TIMINGS: dict[str, object] = {
    "cellpose_model_init_seconds": 0.0,
    "cellpose_inference_events": [],
    "candidate_postprocess_seconds": [],
    "candidate_total_seconds": [],
    "candidate_stage_wall_seconds": 0.0,
    "rank_candidates_seconds": 0.0,
    "compartment_split_seconds": 0.0,
}

DAPI_FRAGMENT_WORKLOAD_LIMITS: DapiFragmentWorkloadLimits | None = (
    DapiFragmentWorkloadLimits(
        policy_version="DAPI-Fragment-Workload-Guard-v1",
        max_parent_fragments=2_600,
        max_total_fragments=7_500,
        max_parent_voxel_comparisons=27_000_000_000,
        max_total_voxel_comparisons=60_000_000_000,
        max_parent_result_payload_bytes_lower_bound=4 * 1024**3,
    )
)
