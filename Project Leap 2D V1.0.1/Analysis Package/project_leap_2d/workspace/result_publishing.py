# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def publish_output_bundle(
    *,
    staged_files: dict[str, Path],
    final_files: dict[str, Path],
    run_dir: Path,
) -> None:
    if set(staged_files) != set(final_files):
        raise ValueError("Staged and final output keys do not match")
    token = uuid.uuid4().hex
    temporary: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    existed: dict[str, bool] = {}
    for key, final_path in final_files.items():
        temporary[key] = (
            final_path.parent / f"temporary_IHC_{token}_{key}.tmp"
        )
        backups[key] = run_dir / f"previous_{key}.backup"
        existed[key] = final_path.exists()
        if existed[key]:
            shutil.copy2(final_path, backups[key])
        shutil.copy2(staged_files[key], temporary[key])
    try:
        for key in staged_files:
            os.replace(temporary[key], final_files[key])
    except Exception:
        for key, final_path in final_files.items():
            if existed[key] and backups[key].exists():
                shutil.copy2(backups[key], final_path)
            elif not existed[key]:
                final_path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
