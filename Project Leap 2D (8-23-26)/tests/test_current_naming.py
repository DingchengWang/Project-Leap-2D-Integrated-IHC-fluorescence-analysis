from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "project_leap_2d"
PUBLISHABLE_TEXT_SUFFIXES = {
    ".command",
    ".groovy",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
}
RUNTIME_DIRECTORY_NAMES = {"Original Image", "Result", "Runtime"}
HISTORICAL_NAME_PATTERNS = (
    (
        "numbered Stage label",
        re.compile(
            r"(?<![A-Za-z0-9])stage[\s_-]*\d+(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "standalone version label",
        re.compile(r"(?<![A-Za-z0-9_])V\d+(?![A-Za-z0-9_])"),
    ),
    (
        "Legacy sample label",
        re.compile(
            r"(?<![A-Za-z0-9])legacy(?:[\s_-]*#?[\s_-]*\d+)?"
            r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "numbered Soma label",
        re.compile(
            r"(?<![A-Za-z0-9])soma[\s_-]*\d+[A-Za-z]?"
            r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "numbered Whole label",
        re.compile(
            r"(?<![A-Za-z0-9])whole[\s_-]*\d+(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "numbered Extension label",
        re.compile(
            r"(?<![A-Za-z0-9])extension[\s_-]*(?:v[\s_-]*)?\d+"
            r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "versioned development identifier",
        re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?:before|finalize|ihc|round\d+|r\d+)_v\d+"
            r"(?=$|[^A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "versioned pair identifier",
        re.compile(
            r"(?<![A-Za-z0-9])v\d+_pair(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "retired generic identifier",
        re.compile(
            r"\b(?:analysis_core_version|pipeline_version|functional_display)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "person-suffixed product name",
        re.compile(r"\bProject Leap 2D-[A-Za-z]"),
    ),
)
DATED_PRODUCT_NAME = re.compile(
    r"\bProject Leap 2D \(\d{1,2}-\d{1,2}-\d{2}\)"
)


class CurrentNamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from project_leap_2d.runtime_loader import load_runtime

        cls.runtime = load_runtime()

    def test_runtime_uses_current_product_and_core_names(self) -> None:
        self.assertEqual(self.runtime.PRODUCT_DISPLAY_NAME, "Project Leap 2D")
        self.assertEqual(
            self.runtime.ANALYSIS_CORE_NAME,
            "Project Leap 2D Analysis Core",
        )
        self.assertEqual(
            self.runtime.PIPELINE_NAME,
            "Project Leap 2D Analysis Core",
        )

    def test_candidate_catalog_uses_functional_module_names(self) -> None:
        rng = np.random.default_rng(20260823)
        stacks = {
            "eGFP": rng.integers(
                0,
                4096,
                size=(11, 24, 20),
                dtype=np.uint16,
            ),
            "GFAP": rng.integers(
                0,
                4096,
                size=(11, 24, 20),
                dtype=np.uint16,
            ),
        }
        specs = self.runtime.complete_candidate_specs(11, stacks)
        self.assertEqual(len(specs), 90)
        self.assertEqual(
            {
                self.runtime.candidate_module_display_name(spec)
                for spec in specs
            },
            {
                "Morphology Baseline",
                "Structural Refinement",
                "Distributional Threshold",
            },
        )
        self.assertTrue(
            all(
                spec.name.startswith(
                    (
                        "morphology_baseline_auto",
                        "structural_refinement_auto",
                        "distributional_threshold_auto",
                    )
                )
                for spec in specs
            )
        )

    def test_report_describes_linked_geometry_edits(self) -> None:
        source = (
            CODE_ROOT / "reporting" / "analysis_report.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "ROI Review: linked Delete, Soma Merge, Whole Split, and Soma Enlarge",
            source,
        )
        self.assertIn("update Whole/Soma/Processes together; Revert is LIFO.", source)
        self.assertNotIn("ROI geometry is read-only", source)

    def test_publishable_text_excludes_historical_identifiers(self) -> None:
        current_dated_product_name = PROJECT_ROOT.name
        failures: list[str] = []
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in PUBLISHABLE_TEXT_SUFFIXES
            ):
                continue
            relative = path.relative_to(PROJECT_ROOT)
            if RUNTIME_DIRECTORY_NAMES.intersection(relative.parts):
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for label, pattern in HISTORICAL_NAME_PATTERNS:
                for match in pattern.finditer(text):
                    failures.append(
                        f"{relative}: {label}: {match.group(0)!r}"
                    )
            for match in DATED_PRODUCT_NAME.finditer(text):
                if match.group(0) != current_dated_product_name:
                    failures.append(
                        f"{relative}: outdated dated product name: "
                        f"{match.group(0)!r}"
                    )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
