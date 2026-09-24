# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the best Whole Astrocyte ROI from 90 candidates, split Soma and Processes, "
            "open six compartment views in Fiji, and measure one raw KCNN/KCNJ10 projection."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--disable-cellpose",
        action="store_true",
        help="Debug fallback: disable Cellpose-SAM and use deterministic structural processing.",
    )
    parser.add_argument(
        "--skip-fiji",
        action="store_true",
        help="Debug only: stop after ROI selection and write a Python preview overlay.",
    )
    parser.add_argument(
        "--fiji-auto-continue",
        action="store_true",
        help="Skip the pre-measurement review dialog for controlled automated validation.",
    )
    parser.add_argument(
        "--fiji-timeout-minutes",
        type=float,
        default=120.0,
        help="Maximum time to wait for the Fiji display and measurement workflow.",
    )
    parser.add_argument(
        "--fiji-launcher",
        type=Path,
        default=None,
        help="Optional explicit Fiji launcher path.",
    )
    parser.add_argument(
        "--dapi-fragment-workload-preflight-only",
        action="store_true",
        help=(
            "Validation only: reconstruct and count 3D DAPI fragments without "
            "submitting fragment jobs, loading the measurement stack, or launching Fiji."
        ),
    )
    parser.add_argument(
        "--dapi-fragment-workload-json",
        type=Path,
        default=None,
        help=(
            "Optional DAPI fragment workload diagnostic JSON path. Required with "
            "--dapi-fragment-workload-preflight-only."
        ),
    )
    return parser.parse_args(argv)
