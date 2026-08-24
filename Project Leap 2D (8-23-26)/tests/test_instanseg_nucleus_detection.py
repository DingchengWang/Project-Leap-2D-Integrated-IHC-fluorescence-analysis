from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_leap_2d.nuclei import instanseg_nucleus_detection as adapter


INSTANSEG_FIXTURE_ENV = "PROJECT_LEAP_INSTANSEG_FIXTURE_DIR"
OFFICIAL_FIXTURE_SHA256 = {
    "test-input.npy": (
        "a68dc9bf2a3c4fb18804757bd13772777792959e8ab690882f2711f4d399cc24"
    ),
    "test-output_instance_segmentation.npy": (
        "f4e154d1bc3c3605945e6fe736ecd6e66e0708e74168bf0b6ab787108e59c1ab"
    ),
}


def resolve_official_fixture_root() -> Path | None:
    raw_value = os.environ.get(INSTANSEG_FIXTURE_ENV)
    if raw_value is None or not raw_value.strip():
        return None
    fixture_root = Path(raw_value.strip())
    if not fixture_root.is_absolute():
        raise RuntimeError(
            f"{INSTANSEG_FIXTURE_ENV} must be an absolute path"
        )
    if not fixture_root.is_dir():
        raise RuntimeError(
            f"{INSTANSEG_FIXTURE_ENV} is not an existing directory: "
            f"{fixture_root}"
        )
    missing = [
        filename
        for filename in OFFICIAL_FIXTURE_SHA256
        if not (fixture_root / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{INSTANSEG_FIXTURE_ENV} is missing required file(s): "
            f"{', '.join(missing)}"
        )
    return fixture_root


class InstanSegNucleusDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resource_dir = (
            PROJECT_ROOT / "project_leap_2d" / "resources" / "models"
        )
        cls.model_path = (
            cls.resource_dir / "instanseg_single_channel_nuclei.pt"
        )

    def tearDown(self) -> None:
        adapter.clear_instanseg_model_cache()

    def test_pinned_model_and_metadata_are_valid(self) -> None:
        metadata = adapter.validate_instanseg_model_resources()
        self.assertEqual(metadata["name"], "single_channel_nuclei")
        self.assertEqual(
            hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            adapter.INSTANSEG_MODEL_SHA256,
        )

    def test_integrity_failure_is_clear_and_precedes_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.pt"
            corrupt.write_bytes(b"not the pinned model")
            with self.assertRaisesRegex(
                RuntimeError,
                "integrity check failed",
            ):
                adapter.get_instanseg_model(corrupt)

    def test_normalization_declared_by_model(self) -> None:
        plane = np.arange(10_000, dtype=np.float32).reshape(100, 100)
        normalized = adapter.normalize_instanseg_dapi_plane(plane)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
        self.assertFalse(
            adapter.normalize_instanseg_dapi_plane(
                np.ones((32, 32), dtype=np.uint16)
            ).any()
        )

    def test_official_fixture_runs_without_instanseg_package(self) -> None:
        fixture_root = resolve_official_fixture_root()
        if fixture_root is None:
            self.skipTest(
                f"Official InstanSeg fixture is unavailable; set "
                f"{INSTANSEG_FIXTURE_ENV} to its absolute directory"
            )
        for filename, expected_sha256 in OFFICIAL_FIXTURE_SHA256.items():
            observed_sha256 = hashlib.sha256(
                (fixture_root / filename).read_bytes()
            ).hexdigest()
            self.assertEqual(
                observed_sha256,
                expected_sha256,
                f"Official InstanSeg fixture integrity check failed: {filename}",
            )
        input_tensor = np.load(
            fixture_root / "test-input.npy",
            allow_pickle=False,
        )
        expected = np.load(
            fixture_root / "test-output_instance_segmentation.npy",
            allow_pickle=False,
        )
        dapi = input_tensor[0, 0][None, :, :]
        result = adapter.detect_instanseg_nuclei(
            dapi,
            0.5,
            0.5,
            model_path=self.model_path,
        )
        observed = result.labels_zyx[0]
        self.assertEqual(observed.shape, expected.shape[-2:])
        self.assertEqual(observed.dtype, np.int32)
        self.assertEqual(result.instance_counts, (25,))
        expected_foreground = expected[0, 0] > 0
        observed_foreground = observed > 0
        self.assertLessEqual(
            int(np.count_nonzero(expected_foreground ^ observed_foreground)),
            16,
        )

    def test_fixture_directory_unset_or_empty_disables_comparison(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(INSTANSEG_FIXTURE_ENV, None)
            self.assertIsNone(resolve_official_fixture_root())
        for empty_value in ("", "   "):
            with self.subTest(value=empty_value), mock.patch.dict(
                os.environ,
                {INSTANSEG_FIXTURE_ENV: empty_value},
            ):
                self.assertIsNone(resolve_official_fixture_root())

    def test_fixture_directory_must_be_absolute(self) -> None:
        with mock.patch.dict(
            os.environ,
            {INSTANSEG_FIXTURE_ENV: "relative/fixture"},
        ):
            with self.assertRaisesRegex(RuntimeError, "absolute path"):
                resolve_official_fixture_root()

    def test_fixture_directory_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with mock.patch.dict(
                os.environ,
                {INSTANSEG_FIXTURE_ENV: str(missing)},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "not an existing directory",
                ):
                    resolve_official_fixture_root()

    def test_fixture_directory_requires_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture_root = Path(directory)
            (fixture_root / "test-input.npy").write_bytes(b"placeholder")
            with mock.patch.dict(
                os.environ,
                {INSTANSEG_FIXTURE_ENV: str(fixture_root)},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "test-output_instance_segmentation.npy",
                ):
                    resolve_official_fixture_root()

    def test_fixture_directory_accepts_complete_placeholder_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture_root = Path(directory)
            for filename in OFFICIAL_FIXTURE_SHA256:
                (fixture_root / filename).write_bytes(b"placeholder")
            with mock.patch.dict(
                os.environ,
                {INSTANSEG_FIXTURE_ENV: str(fixture_root)},
            ):
                self.assertEqual(resolve_official_fixture_root(), fixture_root)

    def test_crop_z_selection_physical_resize_and_batching(self) -> None:
        class FakeModel:
            def __call__(self, tensor):
                import torch

                values = torch.zeros_like(tensor)
                values[:, :, 4:-4, 4:-4] = 1.0
                return values

        stack = np.zeros((4, 80, 60), dtype=np.uint16)
        stack[1, 20:60, 10:50] = 1000
        stack[3, 20:60, 10:50] = 1000
        config = adapter.InstanSegNucleusConfig(batch_size=2)
        with mock.patch.object(
            adapter,
            "get_instanseg_model",
            return_value=FakeModel(),
        ):
            result = adapter.detect_instanseg_nuclei(
                stack,
                0.25,
                0.5,
                z_indices=(3, 1),
                crop_bounds_yx=(10, 70, 5, 55),
                config=config,
            )
        self.assertEqual(result.labels_zyx.shape, (2, 60, 50))
        self.assertEqual(result.z_indices, (3, 1))
        self.assertEqual(result.crop_bounds_yx, (10, 70, 5, 55))
        self.assertEqual(result.source_shape_zyx, (4, 80, 60))
        self.assertEqual(result.instance_counts, (1, 1))

    def test_blank_planes_do_not_load_model(self) -> None:
        stack = np.zeros((3, 40, 50), dtype=np.uint16)
        with mock.patch.object(
            adapter,
            "get_instanseg_model",
            side_effect=AssertionError("blank input must not load model"),
        ):
            result = adapter.detect_instanseg_nuclei(stack, 0.5, 0.5)
        self.assertFalse(result.labels_zyx.any())
        self.assertEqual(result.instance_counts, (0, 0, 0))

    def test_invalid_requests_have_clear_messages(self) -> None:
        stack = np.zeros((3, 40, 50), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "positive micrometer"):
            adapter.detect_instanseg_nuclei(stack, 0.0, 0.5)
        with self.assertRaisesRegex(ValueError, "contains duplicates"):
            adapter.detect_instanseg_nuclei(
                stack,
                0.5,
                0.5,
                z_indices=(1, 1),
            )
        with self.assertRaisesRegex(ValueError, "outside the source"):
            adapter.detect_instanseg_nuclei(
                stack,
                0.5,
                0.5,
                crop_bounds_yx=(0, 41, 0, 50),
            )


if __name__ == "__main__":
    unittest.main()
