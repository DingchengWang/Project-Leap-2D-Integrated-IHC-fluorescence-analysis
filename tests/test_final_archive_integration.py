"""Exercise the final release ZIP and real maintenance shell chain in isolation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

WORK = Path(__file__).resolve().parents[1]
ASSET = "Project-Leap-2D-V1.0.1"
PACKAGE = "Project Leap 2D V1.0.1"
REPO = "DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def put(path: Path, data: str | bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    path.chmod(mode)

def main() -> None:
    archive = WORK / "artifacts" / f"{ASSET}.zip"
    public_manifest = WORK / "artifacts" / f"{ASSET}.manifest.json"
    manifest = json.loads(public_manifest.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["version"] == "1.0.1"
    assert manifest["package_name"] == PACKAGE
    with tempfile.TemporaryDirectory(prefix="leap-final-archive-") as temporary:
        root = Path(temporary)
        extracted = root / "extracted"
        subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(extracted)], check=True)
        dist = extracted / f"{PACKAGE} Distribution"
        project = root / "中文 工作包"
        shutil.copytree(dist / PACKAGE, project)
        for relative, entry in manifest["files"].items():
            p = project / relative
            assert sha(p) == entry["sha256"] and stat.S_IMODE(p.stat().st_mode) == entry["mode"], relative
        assert len(manifest["files"]) == 95
        expected_top = {"Sample Image", "Result", "Analysis Package", "README", "Run Analysis.command", "Repair.command", "Manual Command.txt"}
        assert {p.name for p in project.iterdir()} == expected_top
        import_probe = """import importlib.util, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
names = {'project_leap_2d': '__init__.py', 'project_leap_2d.fiji_review.cell_edit_worker': 'fiji_review/cell_edit_worker.py'}
for name, relative in names.items():
    spec = importlib.util.find_spec(name)
    assert spec is not None and Path(spec.origin).resolve() == root/'Analysis Package'/'project_leap_2d'/relative, name
assert not any(name in sys.modules for name in ('numpy', 'torch', 'cv2'))
print('Canonical package and isolated worker module resolve without model imports.')
"""
        probe = subprocess.run([sys.executable, "-B", "-S", "-c", import_probe, str(project)],
            cwd=project, env={**os.environ, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONPATH": str(project/"Analysis Package"), "PYTHONDONTWRITEBYTECODE":"1"},
            text=True, capture_output=True, timeout=15)
        assert probe.returncode == 0, probe.stderr
        assert not any(p.name in {"tests", "audit", "__pycache__", ".pytest_cache", ".DS_Store"} or p.suffix in {".pyc", ".log"} for p in project.rglob("*"))
        sentinels = {
            "Sample Image/keep.tif": b"isolated input sentinel, not an image",
            "Result/keep.txt": b"isolated result sentinel",
            "Analysis Package/Run State/recovery/keep.json": b'{"isolated_state": true}',
        }
        for relative, data in sentinels.items(): put(project / relative, data)
        support = root / "隔离 Support"
        contract = (project / "Analysis Package/project_leap_2d/maintenance/environment_contract.txt").read_text().strip()
        release = support / "Releases" / f"environment-{contract[:16]}"
        runtime = release / "Managed Python/cpython-3.9.25-macos-aarch64-none"
        recorder = '#!/bin/sh\nprintf "%s\\n" "$@" >> "$FIXTURE_MAINTENANCE_CALLS"\nexit 0\n'
        put(runtime / "bin/python3.9", recorder, 0o755)
        put(runtime / "lib/libpython3.9.dylib", b"isolated library placeholder")
        python = release / "Environment/bin/python3"
        put(python, recorder, 0o755)
        fiji = release / "Fiji/Fiji.app/Contents/MacOS/fiji-macos-arm64"
        put(fiji, "#!/bin/sh\nexit 97\n", 0o755)
        models = release / "Models"
        put(models / "cpsam_v2", b"isolated model placeholder")
        state = {
            "status": "ready", "schema_version": "2", "platform": "macos-arm64",
            "environment_contract_id": contract, "project_version": "older-compatible-version",
            "python_executable": str(python), "fiji_launcher": str(fiji), "cellpose_models_path": str(models),
        }
        state_bytes = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode()
        put(support / "installation_state.json", state_bytes)
        assets = []
        for path in (public_manifest, archive):
            assets.append({"name": path.name, "state": "uploaded", "size": path.stat().st_size,
                           "digest": "sha256:" + sha(path),
                           "browser_download_url": f"https://github.com/{REPO}/releases/download/v1.0.1/{path.name}"})
        put(root / "release.json", json.dumps({"tag_name": "v1.0.1", "draft": False, "assets": assets}))
        transport = root / "curl-fixture"
        put(transport, f'''#!{sys.executable}
import json, os, pathlib, shutil, sys
args=sys.argv[1:]
root=pathlib.Path(os.environ["FIXTURE_ROOT"])
url=args[-1]
mapping=json.loads((root/"url-map.json").read_text())
if url not in mapping: raise SystemExit(95)
with (root/"requests.jsonl").open("a") as log: log.write(json.dumps({{"url":url}})+"\\n")
shutil.copyfile(mapping[url],args[args.index("--output")+1])
''', 0o755)
        urls = {f"https://api.github.com/repos/{REPO}/releases/tags/v1.0.1": str(root / "release.json")}
        urls.update({asset["browser_download_url"]: str(path) for asset, path in zip(assets, (public_manifest, archive))})
        put(root / "url-map.json", json.dumps(urls))
        entry = project / "Repair.command"
        text = entry.read_text()
        assert text.count("'/usr/bin/curl'") == 1
        entry.write_text(text.replace("'/usr/bin/curl'", repr(str(transport))))
        put(project / "Analysis Package/project_leap_2d/settings.py", b"# deliberate isolated corruption\n")
        (project / "Analysis Package/project_leap_2d/fiji_review/resources/processes_trim.groovy").unlink()
        (project / "Run Analysis.command").chmod(0o644)
        tmp = root / "temporary"
        tmp.mkdir()
        env = {**os.environ, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(root), "ZDOTDIR": str(root),
               "PROJECT_LEAP_SUPPORT_DIR": str(support), "TMPDIR": str(tmp),
               "FIXTURE_ROOT": str(root), "FIXTURE_MAINTENANCE_CALLS": str(root / "maintenance.calls"),
               "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PROJECT_LEAP_REPAIR_LOCK_HELD", None)
        completed = subprocess.run(["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
                                    "/bin/sh", str(entry)], env=env, text=True, capture_output=True, timeout=60)
        assert completed.returncode == 0, (completed.returncode, completed.stdout, completed.stderr)
        for relative, expected in manifest["files"].items():
            p = project / relative
            assert sha(p) == expected["sha256"] and stat.S_IMODE(p.stat().st_mode) == expected["mode"], relative
        for relative, data in sentinels.items(): assert (project / relative).read_bytes() == data, relative
        assert (support / "installation_state.json").read_bytes() == state_bytes
        calls = (root / "maintenance.calls").read_text().splitlines()
        assert calls.count("content") == 1 and calls.count("smoke") == 1, calls
        assert not (support / "Source Repair State v1.0.1").exists()
        assert not list(tmp.iterdir())
        requests = [json.loads(line)["url"] for line in (root / "requests.jsonl").read_text().splitlines()]
        assert len(requests) == 3 and requests[-1].endswith(".zip"), requests
        result = {"passed": True, "archive_sha256": sha(archive), "manifest_sha256": sha(public_manifest),
                  "program_files_verified_after_recovery": 95, "top_level_seven_items_verified": True, "fresh_python_package_and_worker_resolution": True,
                  "restored": ["Repair.command (fixture transport edit)", "Analysis Package/project_leap_2d/settings.py", "Analysis Package/project_leap_2d/fiji_review/resources/processes_trim.groovy", "Run Analysis.command executable permission"],
                  "existing_input_result_runtime_data_preserved": True, "original_installation_state_preserved": True,
                  "real_maintenance_shell_chain_reached": True, "content_then_smoke_invocations": ["content", "smoke"],
                  "project_python_and_environment_components": "Temporary shell recorders, no actual interpreter/model/Fiji execution",
                  "network": "OS-denied; local release API/manifest/ZIP transport fixture", "temporary_recovery_files_remaining": 0}
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
