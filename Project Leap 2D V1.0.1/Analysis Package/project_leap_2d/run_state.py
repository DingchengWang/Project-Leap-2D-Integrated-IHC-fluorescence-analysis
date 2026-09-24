# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def runtime_elapsed_seconds() -> float:
    if _RUN_STARTED_AT is None:
        return 0.0
    return float(time.perf_counter() - _RUN_STARTED_AT)

def print_terminal_stage(title: str, detail: str | None = None) -> None:
    print(f"\n{TERMINAL_RULE}", flush=True)
    print(f"{title} | elapsed={runtime_elapsed_seconds():.3f} s", flush=True)
    if detail:
        print(detail, flush=True)
    print(TERMINAL_RULE, flush=True)

def print_terminal_event(message: str) -> None:
    print(f"[elapsed={runtime_elapsed_seconds():.3f} s] {message}", flush=True)

def physical_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = pages * page_size
    return total if total > 0 else None

def select_candidate_cpu_workers(requested: int = CANDIDATE_CPU_WORKERS) -> int:
    """Select a stable preflight worker count without retrying completed jobs."""
    cpu_cap = max(1, min(int(requested), 12, int(os.cpu_count() or 1)))
    total_memory = physical_memory_bytes()
    if total_memory is None:
        memory_cap = cpu_cap
    else:
        gib = total_memory / float(1024**3)
        if gib >= 32.0:
            memory_cap = 12
        elif gib >= 24.0:
            memory_cap = 8
        elif gib >= 16.0:
            memory_cap = 6
        else:
            memory_cap = 4
    return max(1, min(cpu_cap, memory_cap))

def print_runtime_timing_summary(fiji_details: dict) -> None:
    events = _RUNTIME_TIMINGS["cellpose_inference_events"]
    candidate_times = _RUNTIME_TIMINGS["candidate_postprocess_seconds"]
    assert isinstance(events, list)
    assert isinstance(candidate_times, list)
    print_terminal_stage(
        "09 | FINAL RUNTIME SUMMARY",
        "Timing is displayed in Terminal only; no timing log is retained in the analysis folder.",
    )
    print(
        f"  Cellpose model initialization: "
        f"{float(_RUNTIME_TIMINGS['cellpose_model_init_seconds']):.3f} s "
        "(excluded from inference and morphology/postprocess timings)",
        flush=True,
    )
    if events:
        for index, event in enumerate(events, start=1):
            status = "ok" if event.get("success", False) else "failed"
            print(
                f"  Cellpose inference {index:02d}: {float(event['seconds']):.3f} s "
                f"({event['device']}, max_side={event['max_side']}, {status})",
                flush=True,
            )
        print(
            f"  Cellpose inference total: {sum(float(event['seconds']) for event in events):.3f} s "
            "(model initialization excluded; cache hits excluded)",
            flush=True,
        )
    else:
        print("  Cellpose inference: no uncached model.eval call", flush=True)
    if candidate_times:
        values = np.asarray(candidate_times, dtype=np.float64)
        print(
            f"  Candidate morphology/postprocess worker compute total: "
            f"{float(values.sum()):.3f} s; "
            f"median={float(np.median(values)):.3f} s; max={float(values.max()):.3f} s",
            flush=True,
        )
        print(
            f"  Candidate evaluation elapsed wall time: "
            f"{float(_RUNTIME_TIMINGS['candidate_stage_wall_seconds']):.3f} s",
            flush=True,
        )
    print(
        f"  rank_candidates: {float(_RUNTIME_TIMINGS['rank_candidates_seconds']):.3f} s",
        flush=True,
    )
    print(
        f"  Soma/Processes split: {float(_RUNTIME_TIMINGS['compartment_split_seconds']):.3f} s",
        flush=True,
    )
    startup = fiji_details.get("fiji_startup_seconds")
    measurement = fiji_details.get("measurement_seconds")
    review_wait = fiji_details.get("review_wait_seconds")
    if startup is not None:
        print(f"  Fiji startup to six-window review-ready: {float(startup):.3f} s", flush=True)
    if measurement is not None:
        print(f"  Fiji native measurement (Whole, Processes, Soma): {float(measurement):.3f} s", flush=True)
    if review_wait is not None:
        print(
            f"  Human review/decision wait excluded from performance comparison: {float(review_wait):.3f} s",
            flush=True,
        )
    print(f"  Complete pipeline elapsed: {runtime_elapsed_seconds():.3f} s", flush=True)
