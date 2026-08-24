# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def get_cellpose_model():
    global _CELLPOSE_MODEL, _CELLPOSE_DEVICE
    from cellpose import models
    import torch

    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL, _CELLPOSE_DEVICE

    np.random.seed(0)
    torch.manual_seed(0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model_init_started = time.perf_counter()
    _CELLPOSE_MODEL = models.CellposeModel(
        gpu=(device.type != "cpu"),
        pretrained_model="cpsam_v2",
        device=device,
        use_bfloat16=False,
    )
    _RUNTIME_TIMINGS["cellpose_model_init_seconds"] = float(
        time.perf_counter() - model_init_started
    )
    _CELLPOSE_DEVICE = str(device)
    return _CELLPOSE_MODEL, _CELLPOSE_DEVICE

def synchronize_mps() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass

def clear_mps_cache() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()

def is_recoverable_cellpose_batch_error(exc: Exception) -> bool:
    message = repr(exc).lower()
    resource_markers = (
        "out of memory",
        "failed to allocate",
        "allocation failed",
        "resource exhausted",
        "insufficient memory",
        "recommended max working set size",
    )
    return any(marker in message for marker in resource_markers)

def run_cellpose_mask(struct: np.ndarray, spec: TestSpec, cache_key: tuple) -> tuple[np.ndarray, str]:
    global _CELLPOSE_WORKING_MAX_SIDE
    model, device_name = get_cellpose_model()
    if cache_key in _CELLPOSE_MASK_CACHE:
        return _CELLPOSE_MASK_CACHE[cache_key]

    sizes: list[int] = []
    if _CELLPOSE_WORKING_MAX_SIDE is not None:
        sizes.append(_CELLPOSE_WORKING_MAX_SIDE)
    else:
        sizes.append(spec.cellpose_max_side)
        for fallback in [1536, 1280, 1024]:
            if fallback < spec.cellpose_max_side:
                sizes.append(fallback)

    last_error = ""
    for max_side in sizes:
        try:
            print_terminal_event(
                "Cellpose-SAM inference started | "
                f"device={device_name} | max_side={max_side} | "
                f"diameter={spec.cellpose_diameter} | cellprob={spec.cellpose_cellprob}"
            )
            synchronize_mps()
            inference_started = time.perf_counter()
            try:
                mask, note = run_cellpose_mask_at_size(struct, spec, model, device_name, max_side)
            except Exception as exc:
                synchronize_mps()
                events = _RUNTIME_TIMINGS["cellpose_inference_events"]
                assert isinstance(events, list)
                events.append(
                    {
                        "seconds": float(time.perf_counter() - inference_started),
                        "device": str(device_name),
                        "max_side": int(max_side),
                        "diameter": float(spec.cellpose_diameter),
                        "cellprob": float(spec.cellpose_cellprob),
                        "success": False,
                        "error": repr(exc),
                    }
                )
                raise
            synchronize_mps()
            inference_seconds = time.perf_counter() - inference_started
            events = _RUNTIME_TIMINGS["cellpose_inference_events"]
            assert isinstance(events, list)
            events.append(
                {
                    "seconds": float(inference_seconds),
                    "device": str(device_name),
                    "max_side": int(max_side),
                    "diameter": float(spec.cellpose_diameter),
                    "cellprob": float(spec.cellpose_cellprob),
                    "success": True,
                }
            )
            print_terminal_event(
                f"Cellpose-SAM inference {len(events):02d} completed | "
                f"device={device_name} | max_side={max_side} | "
                f"batch={_CELLPOSE_EFFECTIVE_BATCH_SIZE} | "
                f"time={float(inference_seconds):.3f} s"
            )
            _CELLPOSE_WORKING_MAX_SIDE = max_side
            result = (mask, note)
            _CELLPOSE_MASK_CACHE[cache_key] = result
            return result
        except Exception as exc:
            last_error = repr(exc)
            continue
    raise RuntimeError(f"Cellpose-SAM failed at all attempted sizes: {last_error}")

def run_cellpose_mask_at_size(
    struct: np.ndarray,
    spec: TestSpec,
    model,
    device_name: str,
    max_side: int,
) -> tuple[np.ndarray, str]:
    global _CELLPOSE_EFFECTIVE_BATCH_SIZE
    scale = min(1.0, max_side / max(struct.shape))
    if scale < 1.0:
        small = transform.resize(
            struct,
            (int(round(struct.shape[0] * scale)), int(round(struct.shape[1] * scale))),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
    else:
        small = struct.astype(np.float32, copy=False)
    img = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    eval_kwargs = {
        "channels": [0, 0],
        "diameter": max(8.0, spec.cellpose_diameter * scale),
        "cellprob_threshold": spec.cellpose_cellprob,
        "flow_threshold": 0.4,
        "min_size": max(20, int(CELLPOSE_PRIOR_MIN_AREA_PX * scale * scale)),
    }
    batch_sizes: list[int] = []
    starting_batch = (
        _CELLPOSE_EFFECTIVE_BATCH_SIZE
        if _CELLPOSE_EFFECTIVE_BATCH_SIZE is not None
        else CELLPOSE_BATCH_SIZE
    )
    for batch_size in (starting_batch, *CELLPOSE_BATCH_FALLBACKS):
        if (
            batch_size > 0
            and batch_size <= starting_batch
            and batch_size not in batch_sizes
        ):
            batch_sizes.append(batch_size)
    last_batch_error: Exception | None = None
    for attempt_index, effective_batch_size in enumerate(batch_sizes):
        try:
            masks, flows, styles = model.eval(
                img,
                batch_size=effective_batch_size,
                **eval_kwargs,
            )[:3]
            break
        except Exception as batch_error:
            last_batch_error = batch_error
            if (
                attempt_index + 1 >= len(batch_sizes)
                or not is_recoverable_cellpose_batch_error(batch_error)
            ):
                raise
            clear_mps_cache()
            print(
                f"Cellpose-SAM batch_size={effective_batch_size} failed; retrying "
                f"the same input at batch_size={batch_sizes[attempt_index + 1]}: "
                f"{batch_error!r}",
                flush=True,
            )
    else:
        raise RuntimeError(f"Cellpose-SAM batch evaluation failed: {last_batch_error!r}")
    _CELLPOSE_EFFECTIVE_BATCH_SIZE = int(effective_batch_size)
    if masks is None:
        return np.zeros_like(struct, dtype=bool), f"cellpose_cpsam_v2_{device_name}_{max_side}_empty"
    mask_small = np.asarray(masks) > 0
    if scale < 1.0:
        mask = transform.resize(
            mask_small.astype(np.uint8),
            struct.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(bool)
    else:
        mask = mask_small.astype(bool)
    return (
        mask.astype(bool),
        f"cellpose_cpsam_v2_{device_name}_{max_side}_batch{effective_batch_size}",
    )
