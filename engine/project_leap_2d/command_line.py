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
    return parser.parse_args(argv)
