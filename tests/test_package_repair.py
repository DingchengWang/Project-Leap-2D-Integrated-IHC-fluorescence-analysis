"""Isolated source-repair contract tests; never use network or installed Support.

The real entry/helper are copied into a deliberately small fixture release.
Only those copies replace /usr/bin/curl with an offline fixture transport, and
the environment maintenance entry is a recorder. All fixture digests are rebuilt.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile


WORK = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "Project Leap 2D V1.0.1"
CANDIDATE = WORK / PACKAGE_NAME
MAINTENANCE = "Analysis Package/project_leap_2d/maintenance"
HELPER = f"{MAINTENANCE}/repair_package.pl"
MANAGE = f"{MAINTENANCE}/manage_environment.command"
ZIP_NAME = "Project-Leap-2D-V1.0.1.zip"
MANIFEST_NAME = "Project-Leap-2D-V1.0.1.manifest.json"
REPOSITORY = "DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis"
DOWNLOAD_URL = f"https://github.com/{REPOSITORY}/releases/download/v1.0.1/"
ARCHIVE_ROOT = f"{PACKAGE_NAME} Distribution/{PACKAGE_NAME}/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_file(path: Path, content: bytes | str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    path.chmod(mode)


def wait_for(path: Path, process: subprocess.Popen | None = None, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process is not None and process.poll() is not None:
            raise AssertionError(f"process exited {process.returncode} before {path.name}")
        time.sleep(0.01)
    raise AssertionError(f"timeout waiting for {path.name}")


def instrument_replacement_failure(source: str) -> str:
    """Fixture-only failure at the replacement boundary; production has no hook."""
    needle = '    rename($tmp, $p) or stop("Cannot atomically replace $p");'
    if source.count(needle) != 1:
        raise AssertionError("Production atomic replacement changed; review fixture injection.")
    injection = r'''    if ($p eq "$project/Analysis Package/project_leap_2d/settings.py" && !-e "$ENV{FIXTURE_ROOT}/replace-failed-once") {
        open(my $mark, '>', "$ENV{FIXTURE_ROOT}/replace-failed-once") or die $!;
        print {$mark} "injected immediately before atomic rename\n"; close($mark);
        unlink($tmp) or die $!;
        stop('Fixture destination replacement failure.');
    }
'''
    return source.replace(needle, injection + needle)


def instrument_pause_after_first_write(source: str) -> str:
    """Pause a fixture helper after a real write so SIGKILL is deterministic."""
    needle = '            print "Recovered program file: $rel\\n";'
    if source.count(needle) != 1:
        raise AssertionError("Production publication loop changed; review fixture injection.")
    injection = r'''
            if ($rel eq 'Analysis Package/VERSION' && !-e "$ENV{FIXTURE_ROOT}/source-write-paused") {
                open(my $mark, '>', "$ENV{FIXTURE_ROOT}/source-write-paused") or die $!;
                print {$mark} "first real replacement completed\n"; close($mark);
                while (!-e "$ENV{FIXTURE_ROOT}/source-write-resume") { select(undef, undef, undef, 0.01); }
            }
'''
    return source.replace(needle, needle + injection)


def instrument_pause_during_retired_record_cleanup(source: str) -> str:
    """Interrupt a fixture after one retired backup is deleted, before the rest."""
    needle = "    rename($journal_dir, $completed) or stop('The completed source recovery record could not be retired.');"
    if source.count(needle) != 1:
        raise AssertionError("Production journal retirement changed; review fixture injection.")
    injection = r'''
    unlink("$completed/file-0") or die 'Fixture expected a retired backup';
    open(my $mark, '>', "$ENV{FIXTURE_ROOT}/retired-record-paused") or die $!;
    print {$mark} "$completed\n"; close($mark);
    while (!-e "$ENV{FIXTURE_ROOT}/retired-record-resume") { select(undef, undef, undef, 0.01); }
'''
    return source.replace(needle, needle + injection)


TRANSPORT = r'''import json, os, pathlib, shutil, sys, time
args = sys.argv[1:]
destination = pathlib.Path(args[args.index('--output') + 1])
url = args[-1]
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
with (root / 'transport.jsonl').open('a') as handle:
    handle.write(json.dumps({'url': url, 'destination': str(destination)}) + '\n')
if url.endswith('/releases/tags/v1.0.1'):
    source = root / 'release.json'
elif url.endswith('/Project-Leap-2D-V1.0.1.manifest.json'):
    source = root / 'manifest.json'
elif url.endswith('/Project-Leap-2D-V1.0.1.zip'):
    source = root / 'release.zip'
else:
    sys.stderr.write('Fixture rejected an unexpected network request: ' + url + '\n')
    sys.exit(96)
if os.environ.get('FIXTURE_PAUSE_CURL') == '1' and source.name == 'release.zip':
    (root / 'curl-paused').write_text(str(os.getpid()))
    while not (root / 'curl-resume').exists():
        time.sleep(.01)
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, destination)
'''


ENVIRONMENT_RECORDER = r'''import fcntl, hashlib, json, os, pathlib, stat, time
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
project = pathlib.Path(os.environ['FIXTURE_PROJECT'])
expected = json.loads((root / 'expected.json').read_text())
invalid = []
for relative, record in expected.items():
    path = project / relative
    if (not path.is_file() or path.is_symlink() or
        hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256'] or
        stat.S_IMODE(path.stat().st_mode) != record['mode']):
        invalid.append(relative)
inherited = {}
lock_identity = {}
for fd in (8, 9):
    try:
        descriptor_stat = os.fstat(fd)
        inherited[str(fd)] = True
        lock_identity[str(fd)] = [descriptor_stat.st_dev, descriptor_stat.st_ino]
    except OSError:
        inherited[str(fd)] = False
support = pathlib.Path(os.environ['PROJECT_LEAP_SUPPORT_DIR'])
with (support / 'Environment Usage Lock').open('a') as handle:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        support_blocked = False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BlockingIOError:
        support_blocked = True
record = {'invalid_source': invalid, 'inherited': inherited,
          'lock_identity': lock_identity,
          'support_lock_held': support_blocked,
          'state': json.loads((support / 'installation_state.json').read_text())}
with (root / 'environment.jsonl').open('a') as handle:
    handle.write(json.dumps(record) + '\n')
if os.environ.get('FIXTURE_PAUSE_ENVIRONMENT') == '1':
    (root / 'environment-paused').write_text('ready')
    while not (root / 'environment-resume').exists():
        time.sleep(.01)
raise SystemExit(97 if invalid or not inherited['9'] or not support_blocked else
                 int(os.environ.get('FIXTURE_ENVIRONMENT_EXIT', '0')))
'''


class Fixture:
    def __init__(self, *, helper_transform=None):
        self.temporary = tempfile.TemporaryDirectory(prefix="leap-source-repair-")
        self.root = Path(self.temporary.name)
        self.project = self.root / "已搬移 work package" / PACKAGE_NAME
        self.support = self.root / "Isolated Support"
        self.project.mkdir(parents=True)
        self.support.mkdir()
        (self.root / "Temp").mkdir()
        (self.root / "Home").mkdir()
        self.transport = self.root / "offline-curl"
        write_file(self.transport, f"#!{sys.executable}\n" + TRANSPORT, 0o755)
        self.environment = {
            **os.environ,
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PROJECT_LEAP_SUPPORT_DIR": str(self.support),
            "FIXTURE_ROOT": str(self.root),
            "FIXTURE_PROJECT": str(self.project),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(self.root / "Temp"),
            "HOME": str(self.root / "Home"),
            "ZDOTDIR": str(self.root / "Home"),
        }
        self.environment.pop("PROJECT_LEAP_REPAIR_LOCK_HELD", None)
        self.processes = []
        self.capture_handles = []
        self.allowed_orphan_temp = set()
        self.payload: dict[str, tuple[bytes, int]] = {}
        for relative in ("Repair.command", HELPER):
            raw = (CANDIDATE / relative).read_text()
            if relative == HELPER and helper_transform is not None:
                raw = helper_transform(raw)
            raw = raw.replace("/usr/bin/curl", str(self.transport))
            self.payload[relative] = (raw.encode(), 0o755 if relative.endswith('.command') else 0o644)
        launcher_name = next(name for name in ("Run Analysis.command", "Run.command", "run_project_leap_2d.command")
                             if (CANDIDATE / name).exists())
        self.launcher_name = "Run Analysis.command"
        self.payload[self.launcher_name] = ((CANDIDATE / launcher_name).read_bytes(), 0o755)
        self.payload.update({
            "Analysis Package/VERSION": (b"1.0.1\n", 0o644),
            "README/README EN.md": (b"Fixture documentation.\n", 0o644),
            "README/README CN.md": ("隔离测试说明。\n".encode(), 0o644),
            "Manual Command.txt": (b"./Run Analysis.command\n", 0o644),
            "Analysis Package/project_leap_2d/settings.py": (b"VALUE = 'fixture-original'\n", 0o644),
            f"{MAINTENANCE}/environment_doctor.py": (b"# Fixture doctor; never executed.\n", 0o644),
            f"{MAINTENANCE}/environment_contract.txt": ((CANDIDATE / MAINTENANCE / "environment_contract.txt").read_bytes(), 0o644),
            MANAGE: ((f"#!/bin/zsh\nexec '{sys.executable}' - <<'PYTHON'\n" + ENVIRONMENT_RECORDER + "\nPYTHON\n").encode(), 0o755),
        })
        for relative, (data, mode) in self.payload.items():
            write_file(self.project / relative, data, mode)
        self.protected = {
            "Sample Image/user input.tif": b"fake source pixels; not a real TIFF",
            "Result/prior result.txt": b"accepted user result",
            "Analysis Package/Run State/keep user diagnostics.txt": b"retained runtime evidence",
        }
        for relative, data in self.protected.items():
            write_file(self.project / relative, data)
        workspace_lock = self.project / "Analysis Package/Run State/locks/workspace.lock"
        write_file(workspace_lock, b"fixture workspace lock metadata\n")
        lock_stat = workspace_lock.stat()
        self.workspace_lock_identity = [lock_stat.st_dev, lock_stat.st_ino]
        python_bin = self.support / "Environment/bin/python3"
        fiji = self.support / "Fiji/fiji"
        models = self.support / "Models"
        write_file(python_bin, "#!/bin/sh\nprintf unexpected > \"$FIXTURE_ROOT/project-python-invoked\"\nexit 93\n", 0o755)
        write_file(fiji, "#!/bin/sh\nexit 94\n", 0o755)
        write_file(models / "cpsam_v2", b"fixture-model-presence-only")
        self.state = {
            "schema_version": "2", "status": "ready", "platform": "macos-arm64",
            "environment_contract_id": self.payload[f"{MAINTENANCE}/environment_contract.txt"][0].decode().strip(),
            "project_version": "older-compatible-project-version",
            "python_executable": str(python_bin), "fiji_launcher": str(fiji),
            "cellpose_models_path": str(models),
        }
        self.state_path = self.support / "installation_state.json"
        self.original_state_bytes = json.dumps(self.state, indent=2).encode() + b"\n"
        write_file(self.state_path, self.original_state_bytes)
        self.publish()

    def publish(self, *, manifest_change=None, archive_change=None, release_change=None):
        self.expected = {
            relative: {"sha256": sha256(data), "size": len(data), "mode": mode}
            for relative, (data, mode) in sorted(self.payload.items())
        }
        write_file(self.root / "expected.json", json.dumps(self.expected, sort_keys=True))
        self.manifest = {
            "schema_version": 1, "version": "1.0.1", "package_name": PACKAGE_NAME,
            "files": self.expected.copy(),
        }
        if manifest_change:
            manifest_change(self.manifest)
        manifest_bytes = json.dumps(self.manifest, sort_keys=True).encode()
        write_file(self.root / "manifest.json", manifest_bytes)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for relative, (data, mode) in sorted(self.payload.items()):
                item = zipfile.ZipInfo(ARCHIVE_ROOT + relative)
                item.create_system = 3
                item.external_attr = (stat.S_IFREG | mode) << 16
                item.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(item, data)
            if archive_change:
                archive_change(zipped)
        zip_bytes = archive.getvalue()
        write_file(self.root / "release.zip", zip_bytes)
        release = {"tag_name": "v1.0.1", "draft": False, "prerelease": False, "assets": [
            {"name": name, "state": "uploaded", "digest": "sha256:" + sha256(data),
             "browser_download_url": DOWNLOAD_URL + name}
            for name, data in ((ZIP_NAME, zip_bytes), (MANIFEST_NAME, manifest_bytes))
        ]}
        if release_change:
            release_change(release)
        write_file(self.root / "release.json", json.dumps(release))

    def close(self):
        for process in self.processes:
            # Kill known descendant groups even when a parent already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
        for handle in self.capture_handles:
            handle.close()
        self.temporary.cleanup()

    def start(self, *, command=None, extra_env=None):
        index = len(self.processes)
        stdout = (self.root / f"process-{index}.stdout").open("w+")
        stderr = (self.root / f"process-{index}.stderr").open("w+")
        self.capture_handles.extend((stdout, stderr))
        process = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
             *(command or [str(self.project / "Repair.command")])], cwd=self.project,
            env={**self.environment, **(extra_env or {})}, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        self.processes.append(process)
        return process, stdout, stderr

    def finish(self, running, timeout=15):
        process, stdout, stderr = running
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise
        stdout.flush(); stderr.flush()
        stdout.seek(0); stderr.seek(0)
        return subprocess.CompletedProcess(process.args, process.returncode, stdout.read(), stderr.read())

    def run(self, **kwargs):
        return self.finish(self.start(**kwargs))

    def records(self, filename):
        path = self.root / filename
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def source_snapshot(self):
        snapshot = {}
        for relative in self.payload:
            path = self.project / relative
            if path.is_file() and not path.is_symlink():
                snapshot[relative] = (sha256(path.read_bytes()), stat.S_IMODE(path.stat().st_mode))
            elif path.is_symlink():
                snapshot[relative] = ("link", os.readlink(path))
            elif path.exists():
                snapshot[relative] = ("directory",)
            else:
                snapshot[relative] = None
        return snapshot


class PackageRepairTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = Fixture(**kwargs)
        self.addCleanup(fixture.close)
        return fixture

    def assert_protected(self, fixture):
        for relative, data in fixture.protected.items():
            self.assertEqual((fixture.project / relative).read_bytes(), data, relative)
        self.assertFalse((fixture.root / "project-python-invoked").exists())

    def assert_success(self, fixture, completed):
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for relative, (data, mode) in fixture.payload.items():
            path = fixture.project / relative
            self.assertEqual(path.read_bytes(), data, relative)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode, relative)
        self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
        records = fixture.records("environment.jsonl")
        self.assertEqual(len(records), 1, completed.stdout + completed.stderr)
        self.assertEqual(records[0]["invalid_source"], [])
        self.assertTrue(records[0]["inherited"]["8"])
        self.assertTrue(records[0]["inherited"]["9"])
        self.assertEqual(records[0]["lock_identity"]["8"], fixture.workspace_lock_identity)
        support_stat = (fixture.support / "Environment Usage Lock").stat()
        self.assertEqual(records[0]["lock_identity"]["9"], [support_stat.st_dev, support_stat.st_ino])
        self.assertTrue(records[0]["support_lock_held"])
        self.assertEqual(records[0]["state"], fixture.state)
        self.assertFalse((fixture.support / "Source Repair State v1.0.1").exists())
        self.assertEqual(set((fixture.root / "Temp").iterdir()), fixture.allowed_orphan_temp)
        self.assertFalse((fixture.project / "Runtime").exists())
        self.assertFalse((fixture.project / "Original Image").exists())
        self.assert_protected(fixture)

    def test_healthy_source_skips_zip_and_calls_environment_under_inherited_lock(self):
        fixture = self.fixture()
        self.assert_success(fixture, fixture.run())
        urls = [row["url"] for row in fixture.records("transport.jsonl")]
        self.assertFalse(any(url.endswith(".zip") for url in urls), urls)

    def test_source_and_missing_maintenance_and_execute_mode_are_repaired_before_environment(self):
        fixture = self.fixture()
        write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"damaged source\n")
        shutil.rmtree(fixture.project / MAINTENANCE)
        (fixture.project / fixture.launcher_name).chmod(0o644)
        write_file(fixture.project / "Analysis Package/VERSION", b"damaged version\n")
        shutil.rmtree(fixture.project / "README")
        (fixture.project / "Manual Command.txt").unlink()
        Path(fixture.state["python_executable"]).unlink()
        self.assert_success(fixture, fixture.run())
        self.assertTrue(any(row["url"].endswith(".zip") for row in fixture.records("transport.jsonl")))

    def test_remote_metadata_and_archive_failures_leave_local_source_and_state_unchanged(self):
        cases = ("manifest_digest", "manifest_version", "release_tag", "zip_digest")
        for case in cases:
            with self.subTest(case=case):
                fixture = self.fixture()
                write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"damaged source\n")
                if case == "manifest_version":
                    fixture.publish(manifest_change=lambda value: value.update(version="1.0.1.6"))
                elif case == "release_tag":
                    fixture.publish(release_change=lambda value: value.update(tag_name="v1.0.1.6"))
                else:
                    target = MANIFEST_NAME if case == "manifest_digest" else ZIP_NAME
                    def break_digest(value):
                        for item in value["assets"]:
                            if item["name"] == target:
                                item["digest"] = "sha256:" + "0" * 64
                    fixture.publish(release_change=break_digest)
                before = fixture.source_snapshot()
                completed = fixture.run()
                self.assertNotEqual(completed.returncode, 0, case)
                self.assertEqual(fixture.source_snapshot(), before, case)
                self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
                self.assertEqual(fixture.records("environment.jsonl"), [])
                self.assert_protected(fixture)

    def test_unsafe_archive_paths_and_symlinks_are_rejected_before_source_write(self):
        for case in ("parent", "absolute", "symlink"):
            with self.subTest(case=case):
                fixture = self.fixture()
                write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"damaged source\n")
                outside = fixture.root / "escaped-from-zip"
                def append_unsafe(zipped):
                    if case == "parent":
                        zipped.writestr(ARCHIVE_ROOT + "../../../../escaped-from-zip", b"escape")
                    elif case == "absolute":
                        zipped.writestr(str(outside), b"escape")
                    else:
                        link = zipfile.ZipInfo(ARCHIVE_ROOT + "Analysis Package/project_leap_2d/unsafe_link")
                        link.create_system = 3
                        link.external_attr = (stat.S_IFLNK | 0o777) << 16
                        zipped.writestr(link, str(outside))
                fixture.publish(archive_change=append_unsafe)
                before = fixture.source_snapshot()
                completed = fixture.run()
                self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(fixture.source_snapshot(), before)
                self.assertFalse(outside.exists())
                self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
                self.assertEqual(fixture.records("environment.jsonl"), [])
                self.assert_protected(fixture)

    def test_existing_support_or_workspace_lock_blocks_repair_before_transport(self):
        for kind in ("support", "workspace"):
            with self.subTest(kind=kind):
                fixture = self.fixture()
                lock = fixture.support / "Environment Usage Lock" if kind == "support" else fixture.project / "Analysis Package/Run State/locks/workspace.lock"
                lock.parent.mkdir(parents=True, exist_ok=True)
                marker = fixture.root / "lock-held"
                holder = fixture.start(command=[sys.executable, "-c",
                    "import fcntl,pathlib,sys,time; f=open(sys.argv[1],'a'); "
                    "fcntl.flock(f.fileno(),fcntl.LOCK_EX); pathlib.Path(sys.argv[2]).write_text('held'); time.sleep(20)",
                    str(lock), str(marker)])
                wait_for(marker, holder[0])
                before = fixture.source_snapshot()
                completed = fixture.run()
                self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(fixture.records("transport.jsonl"), [])
                self.assertEqual(fixture.records("environment.jsonl"), [])
                self.assertEqual(fixture.source_snapshot(), before)
                self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)

    def test_running_repair_blocks_second_repair_and_normal_analysis(self):
        fixture = self.fixture()
        running = fixture.start(extra_env={"FIXTURE_PAUSE_ENVIRONMENT": "1"})
        wait_for(fixture.root / "environment-paused", running[0])
        before_requests = len(fixture.records("transport.jsonl"))
        second = fixture.run()
        analysis = fixture.run(command=[str(fixture.project / fixture.launcher_name)])
        self.assertNotEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertNotEqual(analysis.returncode, 0, analysis.stdout + analysis.stderr)
        self.assertEqual(len(fixture.records("transport.jsonl")), before_requests)
        self.assertFalse((fixture.root / "project-python-invoked").exists())
        write_file(fixture.root / "environment-resume", b"resume")
        self.assert_success(fixture, fixture.finish(running))

    def test_failed_replacement_rolls_back_partial_writes_and_preserves_recovery_record(self):
        fixture = self.fixture(helper_transform=instrument_replacement_failure)
        write_file(fixture.project / "Analysis Package/VERSION", b"pre-existing damaged version\n", 0o640)
        write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"pre-existing damaged module\n", 0o600)
        before = fixture.source_snapshot()
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Fixture destination replacement failure", completed.stderr)
        self.assertIn("Recovered program file: Analysis Package/VERSION", completed.stdout)
        self.assertEqual(fixture.source_snapshot(), before)
        self.assertEqual(json.loads(fixture.state_path.read_text())["status"], "repairing")
        journal_dir = fixture.support / "Source Repair State v1.0.1"
        self.assertEqual((journal_dir / "installation_state.original").read_bytes(), fixture.original_state_bytes)
        self.assertEqual(json.loads((journal_dir / "journal.json").read_text())["phase"], "prepared")
        self.assertEqual(fixture.records("environment.jsonl"), [])
        self.assert_protected(fixture)
        self.assert_success(fixture, fixture.run())

    def test_sigkill_after_partial_source_write_is_recovered_on_next_repair(self):
        fixture = self.fixture(helper_transform=instrument_pause_after_first_write)
        write_file(fixture.project / "Analysis Package/VERSION", b"pre-existing damaged version\n")
        write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"pre-existing damaged module\n")
        running = fixture.start()
        wait_for(fixture.root / "source-write-paused", running[0])
        self.assertEqual((fixture.project / "Analysis Package/VERSION").read_bytes(), fixture.payload["Analysis Package/VERSION"][0])
        self.assertEqual((fixture.project / "Analysis Package/project_leap_2d/settings.py").read_bytes(), b"pre-existing damaged module\n")
        self.assertEqual(json.loads(fixture.state_path.read_text())["status"], "repairing")
        os.killpg(running[0].pid, signal.SIGKILL)
        self.assertEqual(fixture.finish(running).returncode, -signal.SIGKILL)
        fixture.allowed_orphan_temp = set((fixture.root / "Temp").iterdir())
        self.assertEqual(len(fixture.allowed_orphan_temp), 1)
        self.assertEqual(fixture.records("environment.jsonl"), [])
        blocked = fixture.run(command=[str(fixture.project / fixture.launcher_name)])
        self.assertNotEqual(blocked.returncode, 0)
        self.assertFalse((fixture.root / "project-python-invoked").exists())
        recovered = fixture.run()
        self.assertIn("Restoring the pre-repair program files", recovered.stdout)
        self.assert_success(fixture, recovered)

    def test_cancelled_download_stops_child_and_cleans_staging_without_source_writes(self):
        fixture = self.fixture()
        write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"damaged source\n")
        before = fixture.source_snapshot()
        running = fixture.start(extra_env={"FIXTURE_PAUSE_CURL": "1"})
        paused = fixture.root / "curl-paused"
        wait_for(paused, running[0])
        child_pid = int(paused.read_text())
        os.kill(running[0].pid, signal.SIGTERM)
        completed = fixture.finish(running)
        self.assertNotEqual(completed.returncode, 0)
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)
        self.assertEqual(fixture.source_snapshot(), before)
        self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
        self.assertEqual(fixture.records("environment.jsonl"), [])
        self.assertEqual(list((fixture.root / "Temp").iterdir()), [])
        self.assertFalse((fixture.support / "Source Repair State v1.0.1").exists())
        self.assert_protected(fixture)

    def test_sigkill_during_retired_record_cleanup_does_not_block_next_healthy_repair(self):
        fixture = self.fixture(helper_transform=instrument_pause_during_retired_record_cleanup)
        write_file(fixture.project / "Analysis Package/project_leap_2d/settings.py", b"damaged source\n")
        running = fixture.start()
        paused = fixture.root / "retired-record-paused"
        wait_for(paused, running[0])
        retired = Path(paused.read_text().strip())
        self.assertTrue(retired.is_dir())
        self.assertFalse((retired / "file-0").exists())
        self.assertTrue((retired / "journal.json").is_file())
        self.assertFalse((fixture.support / "Source Repair State v1.0.1").exists())
        self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
        os.killpg(running[0].pid, signal.SIGKILL)
        self.assertEqual(fixture.finish(running).returncode, -signal.SIGKILL)
        fixture.allowed_orphan_temp = set((fixture.root / "Temp").iterdir())
        self.assertEqual(fixture.records("environment.jsonl"), [])
        archive_count = sum(row["url"].endswith(".zip") for row in fixture.records("transport.jsonl"))
        self.assert_success(fixture, fixture.run())
        self.assertEqual(sum(row["url"].endswith(".zip") for row in fixture.records("transport.jsonl")), archive_count)

    def test_bootstrap_and_trusted_helper_reject_data_paths_in_program_manifest(self):
        for entry in ("bootstrap", "helper"):
            for relative in (
                "Sample Image/user input.tif", "Result/prior result.txt",
                "Analysis Package/Run State/keep user diagnostics.txt",
            ):
                with self.subTest(entry=entry, protected_path=relative):
                    fixture = self.fixture()
                    def include_data(manifest):
                        manifest["files"][relative] = {
                            "sha256": sha256(b"unwanted replacement"), "size": 20, "mode": 0o644,
                        }
                    fixture.publish(manifest_change=include_data)
                    before = fixture.source_snapshot()
                    if entry == "bootstrap":
                        completed = fixture.run()
                        self.assertIn("Unsafe or conflicting program path in manifest", completed.stderr)
                        self.assertFalse(any(row["url"].endswith(".zip") for row in fixture.records("transport.jsonl")))
                    else:
                        # Call the trusted helper itself with real inherited
                        # kernel locks, bypassing bootstrap validation entirely.
                        work = fixture.root / "Temp/direct-helper"
                        work.mkdir()
                        journal = fixture.support / "Source Repair State v1.0.1/journal.json"
                        write_file(journal, b"must not be parsed or rewritten\n")
                        wrapper = (
                            "import fcntl,os,sys; project,support,helper,manifest,work=sys.argv[1:]; "
                            "paths=((support+'/Environment Usage Lock',9),"
                            "(project+'/Analysis Package/Run State/locks/workspace.lock',8)); "
                            "\nfor path,target in paths:\n"
                            " fd=os.open(path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600); "
                            "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); "
                            "os.dup2(fd,target,inheritable=True)\n"
                            "os.environ['PROJECT_LEAP_REPAIR_LOCK_HELD']=paths[0][0]\n"
                            "os.execv('/usr/bin/perl',['/usr/bin/perl',helper,'--project',project,"
                            "'--source','','--manifest',manifest,'--work',work,'--support',support])"
                        )
                        completed = fixture.run(command=[
                            sys.executable, "-c", wrapper, str(fixture.project), str(fixture.support),
                            str(fixture.project / HELPER), str(fixture.root / "manifest.json"), str(work),
                        ])
                        self.assertIn("Unsafe program path in recovery manifest", completed.stderr)
                        self.assertEqual(journal.read_bytes(), b"must not be parsed or rewritten\n")
                        self.assertEqual(fixture.records("transport.jsonl"), [])
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(fixture.source_snapshot(), before)
                    self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
                    self.assertEqual(fixture.records("environment.jsonl"), [])
                    self.assert_protected(fixture)


def candidate_fingerprints():
    relatives = ["Repair.command", HELPER]
    relatives.extend(name for name in ("Run Analysis.command", "Run.command", "run_project_leap_2d.command")
                     if (CANDIDATE / name).exists())
    return {relative: {"sha256": sha256((CANDIDATE / relative).read_bytes()),
                       "mode": stat.S_IMODE((CANDIDATE / relative).stat().st_mode)}
            for relative in relatives}


def run_and_record():
    before = candidate_fingerprints()
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PackageRepairTests))
    after = candidate_fingerprints()
    text = output.getvalue()
    print(text, end="")
    report = {
        "status": "PASS" if result.wasSuccessful() and before == after else "FAIL",
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "candidate_unchanged_during_tests": before == after,
        "candidate_files": after,
        "test_file_sha256": sha256(Path(__file__).read_bytes()),
        "fixture_scope": "TemporaryDirectory only; OS network deny; offline curl, synthetic program/data/model-presence files, environment recorder; no real downloads/install/doctor/analysis",
        "fault_injection": "Only copied fixture helpers: failure before one atomic rename and deterministic pause after one real replacement; fixture manifests and ZIP digests rebuilt",
        "sigkill_temp_cleanup": "Killed-process temporary staging is confined to fixture TMPDIR and removed by TemporaryDirectory cleanup, not claimed to be recovered by production code",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    run_and_record()
