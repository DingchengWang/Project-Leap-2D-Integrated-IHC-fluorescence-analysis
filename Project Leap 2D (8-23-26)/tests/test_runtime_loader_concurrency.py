from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(script: str) -> dict[str, object]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with tempfile.TemporaryDirectory() as matplotlib_cache:
        environment["MPLCONFIGDIR"] = matplotlib_cache
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
    return json.loads(completed.stdout.strip())


class RuntimeLoaderConcurrencyTests(unittest.TestCase):
    def test_sixteen_concurrent_callers_share_one_fully_loaded_runtime(self) -> None:
        observed = _run_isolated(
            f"""
            import builtins
            import json
            import sys
            import threading
            from concurrent.futures import ThreadPoolExecutor

            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            from project_leap_2d import runtime_loader as loader

            sys.modules.pop(loader._RUNTIME_NAME, None)
            code_root = loader.Path(loader.__file__).resolve().parent
            expected = {{
                str(code_root / relative): relative
                for relative in loader.MODULE_ORDER
            }}
            compile_counts = {{relative: 0 for relative in loader.MODULE_ORDER}}
            compile_lock = threading.Lock()
            original_compile = builtins.compile

            def counted_compile(source, filename, mode, *args, **kwargs):
                relative = expected.get(str(filename))
                if relative is not None:
                    with compile_lock:
                        compile_counts[relative] += 1
                return original_compile(source, filename, mode, *args, **kwargs)

            builtins.compile = counted_compile
            try:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    runtimes = list(executor.map(lambda _index: loader.load_runtime(), range(16)))
            finally:
                builtins.compile = original_compile

            first = runtimes[0]
            print(json.dumps({{
                "object_count": len({{id(runtime) for runtime in runtimes}}),
                "module_is_registered": sys.modules.get(loader._RUNTIME_NAME) is first,
                "ready": bool(getattr(first, "_PROJECT_LEAP_RUNTIME_READY", False)),
                "loaded_files": len(first._PROJECT_LEAP_LOADED_FILES),
                "module_order": len(loader.MODULE_ORDER),
                "compile_counts": compile_counts,
                "cache_identity_shared": all(
                    runtime._CELLPOSE_MASK_CACHE is first._CELLPOSE_MASK_CACHE
                    and runtime._RUNTIME_TIMINGS is first._RUNTIME_TIMINGS
                    for runtime in runtimes
                ),
            }}, sort_keys=True))
            """
        )
        self.assertEqual(observed["object_count"], 1)
        self.assertTrue(observed["module_is_registered"])
        self.assertTrue(observed["ready"])
        self.assertTrue(observed["cache_identity_shared"])
        self.assertEqual(observed["loaded_files"], observed["module_order"])
        self.assertTrue(
            all(count == 1 for count in observed["compile_counts"].values())
        )

    def test_failed_load_leaves_no_half_initialized_runtime_and_can_retry(self) -> None:
        observed = _run_isolated(
            f"""
            import builtins
            import json
            import sys

            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            from project_leap_2d import runtime_loader as loader

            sys.modules.pop(loader._RUNTIME_NAME, None)
            code_root = loader.Path(loader.__file__).resolve().parent
            failure_path = str(code_root / loader.MODULE_ORDER[3])
            original_compile = builtins.compile
            failure_type = None

            def failing_compile(source, filename, mode, *args, **kwargs):
                if str(filename) == failure_path:
                    raise RuntimeError("injected runtime assembly failure")
                return original_compile(source, filename, mode, *args, **kwargs)

            builtins.compile = failing_compile
            try:
                loader.load_runtime()
            except Exception as exc:
                failure_type = type(exc).__name__
            finally:
                builtins.compile = original_compile

            registered_after_failure = loader._RUNTIME_NAME in sys.modules
            runtime = loader.load_runtime()
            print(json.dumps({{
                "failure_type": failure_type,
                "registered_after_failure": registered_after_failure,
                "retry_registered": sys.modules.get(loader._RUNTIME_NAME) is runtime,
                "retry_ready": bool(getattr(runtime, "_PROJECT_LEAP_RUNTIME_READY", False)),
                "retry_loaded_files": len(runtime._PROJECT_LEAP_LOADED_FILES),
                "module_order": len(loader.MODULE_ORDER),
            }}, sort_keys=True))
            """
        )
        self.assertEqual(observed["failure_type"], "RuntimeError")
        self.assertFalse(observed["registered_after_failure"])
        self.assertTrue(observed["retry_registered"])
        self.assertTrue(observed["retry_ready"])
        self.assertEqual(observed["retry_loaded_files"], observed["module_order"])


if __name__ == "__main__":
    unittest.main()
