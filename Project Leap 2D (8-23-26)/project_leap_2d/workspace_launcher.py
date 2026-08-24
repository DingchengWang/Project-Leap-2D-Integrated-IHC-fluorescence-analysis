from __future__ import annotations

import sys
from pathlib import Path

from .runtime_loader import load_runtime
from .runtime_attributes import temporary_runtime_attributes
from .workspace.input_cleanup import InputSnapshot
from .workspace.input_cleanup_recovery import (
    move_inputs_to_macos_trash_recoverable,
    recover_input_trash_transaction,
)
from .workspace.pending_results import archive_result_root_files
from .workspace.publication_recovery import (
    publish_with_publication_journal,
    recover_publication_transaction,
)
from .workspace.workspace_preflight import (
    project_workspace_lock,
    require_nonempty_original_image,
    validate_workspace_layout,
)


def _contains_path_override(argv: list[str]) -> bool:
    return any(
        value in {"--input-dir", "--output-dir"}
        or value.startswith("--input-dir=")
        or value.startswith("--output-dir=")
        for value in argv
    )


def _run_workspace(project_root: Path, forwarded: list[str]) -> int:
    original_image = project_root / "Original Image"
    result = project_root / "Result"
    runtime_root = project_root / "Runtime"
    validate_workspace_layout(project_root)

    with project_workspace_lock(runtime_root):
        recover_input_trash_transaction(
            original_image=original_image,
            runtime_root=runtime_root,
        )
        recover_publication_transaction(
            result_root=result,
            runtime_root=runtime_root,
        )
        require_nonempty_original_image(original_image)
        archive_result_root_files(result, runtime_root)

        runtime = load_runtime()
        from .analysis_workflow import run_analysis_workflow

        captured_inputs: dict[Path, InputSnapshot] = {}
        production_published = False
        original_discover = runtime.discover_channel_paths
        original_publish = runtime.publish_output_bundle

        def tracked_discover(input_dir: Path):
            selected, ignored = original_discover(input_dir)
            captured_inputs.update(
                {
                    path.resolve(): InputSnapshot.capture(path.resolve())
                    for path in selected.values()
                }
            )
            return selected, ignored

        def tracked_publish(*args, **kwargs):
            nonlocal production_published
            result_value = publish_with_publication_journal(
                publish_callback=original_publish,
                call_args=args,
                call_kwargs=kwargs,
                result_root=result,
                runtime_root=runtime_root,
            )
            production_published = True
            return result_value

        with temporary_runtime_attributes(
            runtime,
            discover_channel_paths=tracked_discover,
            publish_output_bundle=tracked_publish,
        ):
            return_code = int(
                run_analysis_workflow(
                    runtime,
                    [
                        "--input-dir",
                        str(original_image),
                        "--output-dir",
                        str(result),
                        *forwarded,
                    ]
                )
            )

        if return_code != 0 or not production_published:
            return return_code

        try:
            trash_path = move_inputs_to_macos_trash_recoverable(
                original_image=original_image,
                runtime_root=runtime_root,
                snapshots=tuple(captured_inputs.values()),
            )
        except Exception as exc:
            print(
                "\nANALYSIS COMPLETED, BUT SOURCE IMAGES WERE NOT MOVED TO TRASH\n"
                "The validated Result files were kept. The Original Image files "
                "were retained or restored.\n"
                f"Reason: {exc}",
                flush=True,
            )
            return 3
        print(f"Source images moved to macOS Trash: {trash_path}", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if _contains_path_override(forwarded):
        raise ValueError(
            "The packaged launcher always uses its own Original Image and Result "
            "folders; --input-dir and --output-dir are not accepted here."
        )
    project_root = Path(__file__).resolve().parents[1]
    return _run_workspace(project_root, forwarded)
