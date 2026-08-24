"""Pinned InstanSeg adapter for DAPI nucleus-candidate segmentation.

This module deliberately uses only the exported TorchScript model.  It does
not import the InstanSeg Python package and it does not decide whether a
nucleus belongs to an astrocyte.  The returned 2D instance labels are
proposals for the project's existing 3D inventory and ownership logic.
"""

from __future__ import annotations

import hashlib
import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage import transform


INSTANSEG_MODEL_NAME = "single_channel_nuclei"
INSTANSEG_MODEL_VERSION = "0.1.0"
INSTANSEG_MODEL_SHA256 = (
    "118066231fb7753ffb048bcd2164186397bcc5c8fc4ad7f826efe928b515794c"
)
INSTANSEG_MODEL_PIXEL_SIZE_UM = 0.5
INSTANSEG_MODEL_FILENAME = "instanseg_single_channel_nuclei.pt"
INSTANSEG_METADATA_FILENAME = "instanseg_single_channel_nuclei.json"


@dataclass(frozen=True)
class InstanSegNucleusConfig:
    """Inference-only settings for the pinned single-channel nuclei model."""

    model_pixel_size_um: float = INSTANSEG_MODEL_PIXEL_SIZE_UM
    lower_percentile: float = 0.1
    upper_percentile: float = 99.9
    batch_size: int = 4
    min_model_side: int = 32
    max_model_pixels: int = 4_194_304
    device: str = "cpu"


@dataclass(frozen=True)
class InstanSegNucleusResult:
    """Slice-wise DAPI instances in the requested source crop.

    Labels are independent between Z planes.  They must be linked and
    biologically validated by the existing 3D nucleus logic before use.
    """

    labels_zyx: np.ndarray
    z_indices: tuple[int, ...]
    crop_bounds_yx: tuple[int, int, int, int]
    source_shape_zyx: tuple[int, int, int]
    source_pixel_size_yx_um: tuple[float, float]
    model_pixel_size_um: float
    model_sha256: str
    instance_counts: tuple[int, ...]


_INSTANSEG_MODEL_CACHE: dict[tuple[str, str], object] = {}
_INSTANSEG_MODEL_CACHE_LOCK = threading.RLock()
_INSTANSEG_INFERENCE_LOCK = threading.RLock()


def instanseg_model_resource_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "models"


def instanseg_model_path() -> Path:
    return instanseg_model_resource_dir() / INSTANSEG_MODEL_FILENAME


def instanseg_metadata_path() -> Path:
    return instanseg_model_resource_dir() / INSTANSEG_METADATA_FILENAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_instanseg_model_resources(
    model_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> dict:
    """Validate pinned model bytes and their machine-readable provenance."""

    resolved_model = Path(model_path) if model_path is not None else instanseg_model_path()
    resolved_metadata = (
        Path(metadata_path) if metadata_path is not None else instanseg_metadata_path()
    )
    if not resolved_model.is_file():
        raise RuntimeError(
            "InstanSeg nucleus model is missing. Restore the validated Project "
            f"Leap 2D package: {resolved_model}"
        )
    observed_sha = _sha256_file(resolved_model)
    if observed_sha != INSTANSEG_MODEL_SHA256:
        raise RuntimeError(
            "InstanSeg nucleus model integrity check failed. Expected "
            f"{INSTANSEG_MODEL_SHA256}, found {observed_sha}."
        )
    if not resolved_metadata.is_file():
        raise RuntimeError(
            "InstanSeg nucleus model metadata is missing. Restore the validated "
            f"Project Leap 2D package: {resolved_metadata}"
        )
    try:
        metadata = json.loads(resolved_metadata.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"InstanSeg nucleus model metadata is unreadable: {exc}"
        ) from exc
    expected = {
        "name": INSTANSEG_MODEL_NAME,
        "version": INSTANSEG_MODEL_VERSION,
        "sha256": INSTANSEG_MODEL_SHA256,
        "pixel_size_um": INSTANSEG_MODEL_PIXEL_SIZE_UM,
    }
    mismatched = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "InstanSeg nucleus model metadata does not match the pinned runtime "
            f"contract: {mismatched}"
        )
    return metadata


def clear_instanseg_model_cache() -> None:
    """Clear cached InstanSeg model references for test isolation."""

    with _INSTANSEG_MODEL_CACHE_LOCK:
        _INSTANSEG_MODEL_CACHE.clear()


def get_instanseg_model(
    model_path: Path | str | None = None,
    *,
    device: str = "cpu",
):
    """Lazily load the checksum-pinned TorchScript model."""

    if device != "cpu":
        raise ValueError(
            "InstanSeg nucleus inference is currently validated on CPU only."
        )
    resolved_model = (
        Path(model_path) if model_path is not None else instanseg_model_path()
    ).resolve()
    # Validation precedes the cache lookup so replaced/corrupted package bytes
    # cannot silently keep using a previously loaded model.
    validate_instanseg_model_resources(model_path=resolved_model)
    cache_key = (str(resolved_model), device)
    with _INSTANSEG_MODEL_CACHE_LOCK:
        cached = _INSTANSEG_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is unavailable; InstanSeg nucleus inference cannot run."
            ) from exc
        try:
            model = torch.jit.load(str(resolved_model), map_location=device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"InstanSeg nucleus model could not be loaded: {exc}"
            ) from exc
        _INSTANSEG_MODEL_CACHE[cache_key] = model
        return model


def normalize_instanseg_dapi_plane(
    plane: np.ndarray,
    *,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
) -> np.ndarray:
    """Apply the percentile scaling declared by the published BioImage model."""

    image = np.asarray(plane, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"DAPI plane must be 2D; found shape={image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError("DAPI plane contains NaN or infinite values.")
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError(
            "InstanSeg normalization percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )
    low, high = np.percentile(
        image,
        (float(lower_percentile), float(upper_percentile)),
    )
    dynamic = float(high - low)
    if dynamic <= max(np.finfo(np.float32).eps * max(abs(float(high)), 1.0), 1e-6):
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - float(low)) / (dynamic + 1e-6), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _validate_instanseg_request(
    dapi_zyx: np.ndarray,
    pixel_height_um: float,
    pixel_width_um: float,
    z_indices: Iterable[int] | None,
    crop_bounds_yx: tuple[int, int, int, int] | None,
    config: InstanSegNucleusConfig,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int, int, int]]:
    stack = np.asarray(dapi_zyx)
    if stack.ndim != 3:
        raise ValueError(f"DAPI stack must be ZYX; found shape={stack.shape}.")
    if stack.size == 0:
        raise ValueError("DAPI stack is empty.")
    if not np.isfinite(pixel_height_um) or float(pixel_height_um) <= 0.0:
        raise ValueError("DAPI pixel height must be a positive micrometer value.")
    if not np.isfinite(pixel_width_um) or float(pixel_width_um) <= 0.0:
        raise ValueError("DAPI pixel width must be a positive micrometer value.")
    if config.device != "cpu":
        raise ValueError(
            "InstanSeg nucleus inference is currently validated on CPU only."
        )
    if not np.isfinite(config.model_pixel_size_um) or config.model_pixel_size_um <= 0:
        raise ValueError("InstanSeg model pixel size must be positive.")
    if not (
        0.0
        <= config.lower_percentile
        < config.upper_percentile
        <= 100.0
    ):
        raise ValueError("Invalid InstanSeg normalization percentiles.")
    if config.batch_size < 1:
        raise ValueError("InstanSeg batch size must be at least 1.")
    if config.min_model_side < 32:
        raise ValueError("InstanSeg minimum model side must be at least 32 pixels.")
    if config.max_model_pixels < config.min_model_side**2:
        raise ValueError("InstanSeg model pixel budget is too small.")

    if z_indices is None:
        selected_z = tuple(range(int(stack.shape[0])))
    else:
        selected_z = tuple(int(value) for value in z_indices)
        if not selected_z:
            raise ValueError("At least one DAPI Z plane must be selected.")
        if len(set(selected_z)) != len(selected_z):
            raise ValueError("DAPI Z plane selection contains duplicates.")
        if any(value < 0 or value >= stack.shape[0] for value in selected_z):
            raise ValueError(
                f"DAPI Z plane selection is outside 0..{stack.shape[0] - 1}."
            )

    height, width = int(stack.shape[1]), int(stack.shape[2])
    if crop_bounds_yx is None:
        crop = (0, height, 0, width)
    else:
        crop = tuple(int(value) for value in crop_bounds_yx)
        if len(crop) != 4:
            raise ValueError("DAPI crop must be (y0, y1, x0, x1).")
        y0, y1, x0, x1 = crop
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(
                f"DAPI crop {crop} is outside the source YX shape {(height, width)}."
            )
    return stack, selected_z, crop


def _resize_dapi_for_instanseg(
    plane: np.ndarray,
    pixel_height_um: float,
    pixel_width_um: float,
    config: InstanSegNucleusConfig,
) -> tuple[np.ndarray, tuple[int, int]]:
    source_shape = (int(plane.shape[0]), int(plane.shape[1]))
    model_shape = (
        max(
            1,
            int(
                round(
                    source_shape[0]
                    * float(pixel_height_um)
                    / config.model_pixel_size_um
                )
            ),
        ),
        max(
            1,
            int(
                round(
                    source_shape[1]
                    * float(pixel_width_um)
                    / config.model_pixel_size_um
                )
            ),
        ),
    )
    if int(np.prod(model_shape)) > config.max_model_pixels:
        raise RuntimeError(
            "InstanSeg nucleus request is too large for the validated memory "
            f"budget ({model_shape[0]}x{model_shape[1]} model pixels). Use a "
            "smaller local crop."
        )
    if model_shape == source_shape:
        resized = np.asarray(plane, dtype=np.float32)
    else:
        resized = transform.resize(
            np.asarray(plane, dtype=np.float32),
            model_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)
    return resized, model_shape


def _pad_instanseg_plane(
    plane: np.ndarray,
    minimum_side: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    pad_y = max(0, int(minimum_side) - int(plane.shape[0]))
    pad_x = max(0, int(minimum_side) - int(plane.shape[1]))
    if pad_y == 0 and pad_x == 0:
        return plane, (0, 0)
    return (
        np.pad(plane, ((0, pad_y), (0, pad_x)), mode="edge"),
        (pad_y, pad_x),
    )


def _run_instanseg_batch(model, batch_bchw: np.ndarray) -> np.ndarray:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is unavailable; InstanSeg nucleus inference cannot run."
        ) from exc
    try:
        with _INSTANSEG_INFERENCE_LOCK:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sparse CSR tensor support is in beta state.*",
                    category=UserWarning,
                )
                with torch.inference_mode():
                    output = model(torch.from_numpy(batch_bchw))
        values = output.detach().to("cpu").numpy()
    except Exception as exc:
        raise RuntimeError(f"InstanSeg nucleus inference failed: {exc}") from exc
    if (
        values.ndim != 4
        or values.shape[0] != batch_bchw.shape[0]
        or values.shape[1] != 1
        or values.shape[2:] != batch_bchw.shape[2:]
    ):
        raise RuntimeError(
            "InstanSeg nucleus model returned an unexpected shape: "
            f"{values.shape}; expected {(batch_bchw.shape[0], 1, *batch_bchw.shape[2:])}."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("InstanSeg nucleus model returned invalid label values.")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-4):
        raise RuntimeError("InstanSeg nucleus model returned non-integer labels.")
    return rounded.astype(np.int32, copy=False)


def detect_instanseg_nuclei(
    dapi_zyx: np.ndarray,
    pixel_height_um: float,
    pixel_width_um: float,
    *,
    z_indices: Iterable[int] | None = None,
    crop_bounds_yx: tuple[int, int, int, int] | None = None,
    config: InstanSegNucleusConfig | None = None,
    model_path: Path | str | None = None,
) -> InstanSegNucleusResult:
    """Segment DAPI nuclei slice-wise at the model's physical resolution.

    The returned array contains only the requested Z planes and YX crop.  Its
    first axis follows ``z_indices`` exactly.  Blank planes are skipped without
    invoking the model.  Nonblank planes are batched to improve CPU use.
    """

    active_config = config or InstanSegNucleusConfig()
    stack, selected_z, crop = _validate_instanseg_request(
        dapi_zyx,
        pixel_height_um,
        pixel_width_um,
        z_indices,
        crop_bounds_yx,
        active_config,
    )
    y0, y1, x0, x1 = crop
    source_crop_shape = (y1 - y0, x1 - x0)
    prepared: list[np.ndarray] = []
    model_shapes: list[tuple[int, int]] = []
    active_positions: list[int] = []
    for result_position, z_index in enumerate(selected_z):
        resized, model_shape = _resize_dapi_for_instanseg(
            stack[z_index, y0:y1, x0:x1],
            float(pixel_height_um),
            float(pixel_width_um),
            active_config,
        )
        normalized = normalize_instanseg_dapi_plane(
            resized,
            lower_percentile=active_config.lower_percentile,
            upper_percentile=active_config.upper_percentile,
        )
        padded, _ = _pad_instanseg_plane(
            normalized,
            active_config.min_model_side,
        )
        model_shapes.append(model_shape)
        prepared.append(padded)
        if np.any(normalized):
            active_positions.append(result_position)

    output_planes = [
        np.zeros(source_crop_shape, dtype=np.int32) for _ in selected_z
    ]
    if active_positions:
        model = get_instanseg_model(model_path, device=active_config.device)
        grouped: dict[tuple[int, int], list[int]] = {}
        for position in active_positions:
            grouped.setdefault(tuple(prepared[position].shape), []).append(position)
        for padded_shape in sorted(grouped):
            positions = grouped[padded_shape]
            for batch_start in range(0, len(positions), active_config.batch_size):
                batch_positions = positions[
                    batch_start : batch_start + active_config.batch_size
                ]
                batch = np.stack(
                    [prepared[position] for position in batch_positions],
                    axis=0,
                )[:, None, :, :].astype(np.float32, copy=False)
                batch_labels = _run_instanseg_batch(model, batch)
                for batch_index, result_position in enumerate(batch_positions):
                    model_h, model_w = model_shapes[result_position]
                    labels_at_model_scale = batch_labels[
                        batch_index,
                        0,
                        :model_h,
                        :model_w,
                    ]
                    if labels_at_model_scale.shape == source_crop_shape:
                        restored = labels_at_model_scale
                    else:
                        restored = transform.resize(
                            labels_at_model_scale,
                            source_crop_shape,
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        )
                    output_planes[result_position] = np.rint(restored).astype(
                        np.int32,
                        copy=False,
                    )

    labels_zyx = np.stack(output_planes, axis=0)
    instance_counts = tuple(
        int(np.unique(plane[plane > 0]).size) for plane in labels_zyx
    )
    return InstanSegNucleusResult(
        labels_zyx=labels_zyx,
        z_indices=selected_z,
        crop_bounds_yx=crop,
        source_shape_zyx=tuple(int(value) for value in stack.shape),
        source_pixel_size_yx_um=(
            float(pixel_height_um),
            float(pixel_width_um),
        ),
        model_pixel_size_um=float(active_config.model_pixel_size_um),
        model_sha256=INSTANSEG_MODEL_SHA256,
        instance_counts=instance_counts,
    )
