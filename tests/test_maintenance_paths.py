from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[1]
PACKAGE = WORK / "Project Leap 2D V1.0.1"
MAINTENANCE_RELATIVE = Path("Analysis Package/project_leap_2d/maintenance")
MAINTENANCE = PACKAGE / MAINTENANCE_RELATIVE
# These fixed component contracts were retained in the V1.0.1 release. Their
# literal hashes make the test independent of an old development package.
FIXED_COMPONENT_SHA256 = {
    "component_manifest.sh": "36b6e2a421f301e17ee3a4d42240d7f70ea8642ec1c72009ce3c5762f1631974",
    "requirements_macos_arm64.lock.txt": "48c59884b99fc75622d207b18704bc8014377eb2c3df03cb931cf6f3102aa536",
    "environment_contract.txt": "d93a370168848e0caddc39207c6689c976873c3251f000f1a7f4213879bd8acb",
    "python_wheel_integrity.json": "5c2987fccb41781cc0bc35d563027e8fd44ec5fdf873db47fb664301fde4d52d",
    "managed_python_integrity.json": "f4b612f2e410142f53c93d8099069affb915686cdd01bc8c5a0cdb748973c987",
    "fiji_tree_integrity.json": "dd4d945b5db14642d0bf734979fe41afc50d64034fa681c0414b806f9cf0a7e5",
}
RESOURCE_KEYS = {
    "environment_installer.sh": "PROJECT_LEAP_ENVIRONMENT_INSTALLER_SHA256",
    "environment_doctor.sh": "PROJECT_LEAP_ENVIRONMENT_DOCTOR_SHELL_SHA256",
    "environment_doctor.py": "PROJECT_LEAP_ENVIRONMENT_DOCTOR_SHA256",
    "component_manifest.sh": "PROJECT_LEAP_COMPONENT_MANIFEST_SHA256",
    "requirements_macos_arm64.lock.txt": "PROJECT_LEAP_REQUIREMENTS_LOCK_SHA256",
    "environment_contract.txt": "PROJECT_LEAP_ENVIRONMENT_CONTRACT_SHA256",
    "python_wheel_integrity.json": "PROJECT_LEAP_PYTHON_WHEEL_INTEGRITY_SHA256",
    "managed_python_integrity.json": "PROJECT_LEAP_MANAGED_PYTHON_INTEGRITY_SHA256",
    "fiji_tree_integrity.json": "PROJECT_LEAP_FIJI_TREE_INTEGRITY_SHA256",
}
NETWORK_DENY = [
    "/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)"
]


class MaintenancePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="leap-layout-maintenance-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "移动后的 工作包"
        self.maintenance = self.project / MAINTENANCE_RELATIVE
        self.maintenance.mkdir(parents=True)
        self.support = self.root / "独立 Support"
        self.env = {
            **os.environ, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(self.root), "ZDOTDIR": str(self.root),
            "PROJECT_LEAP_SUPPORT_DIR": str(self.support),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.env.pop("PROJECT_LEAP_REPAIR_LOCK_HELD", None)

    def run_shell(self, args):
        return subprocess.run(
            NETWORK_DENY + args, env=self.env, text=True,
            capture_output=True, timeout=10,
        )

    def test_bootstrap_passes_working_root_and_separate_code_directory(self):
        shutil.copytree(MAINTENANCE, self.maintenance, dirs_exist_ok=True)
        captured = self.root / "installer-arguments"
        installer = self.maintenance / "environment_installer.sh"
        installer.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$@" > ' + shlex.quote(str(captured)) + "\n"
        )
        installer.chmod(0o755)
        # Refresh only this isolated copy after replacing its installer with a
        # recorder. No development-only refresh tool or baseline is required.
        def replace_digest(path, key, digest):
            self.assertTrue(path.resolve().is_relative_to(self.root))
            updated, count = re.subn(
                rf'(?m)^{re.escape(key)}="[0-9a-f]{{64}}"$',
                f'{key}="{digest}"', path.read_text(),
            )
            self.assertEqual(count, 1)
            path.write_text(updated)

        manifest = self.maintenance / "installer_integrity_manifest.sh"
        bootstrap = self.maintenance / "bootstrap_macos.sh"
        entry = self.maintenance / "manage_environment.command"
        replace_digest(manifest, "PROJECT_LEAP_ENVIRONMENT_INSTALLER_SHA256", hashlib.sha256(installer.read_bytes()).hexdigest())
        manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        replace_digest(bootstrap, "INSTALLER_INTEGRITY_MANIFEST_SHA256", manifest_digest)
        replace_digest(entry, "EXPECTED_BOOTSTRAP_SHA256", hashlib.sha256(bootstrap.read_bytes()).hexdigest())
        replace_digest(entry, "EXPECTED_INTEGRITY_MANIFEST_SHA256", manifest_digest)
        result = self.run_shell(["/bin/sh", str(self.maintenance / "bootstrap_macos.sh")])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(captured.read_text().splitlines(), [
            str(self.support), str(self.project), str(self.maintenance),
            (MAINTENANCE / "environment_contract.txt").read_text().strip(),
        ])
        self.assertFalse(self.support.exists())

    def test_installer_reads_nested_version_cache_and_workspace_lock(self):
        for name in ("component_manifest.sh", "requirements_macos_arm64.lock.txt"):
            shutil.copyfile(MAINTENANCE / name, self.maintenance / name)
        (self.project / "Analysis Package" / "VERSION").write_text("nested-fixture-version\n")
        state = self.project / "Analysis Package" / "Run State"
        (state / "locks").mkdir(parents=True)
        (state / "locks" / "workspace.lock").write_text(f"pid={os.getpid()} fixture\n")
        source = (MAINTENANCE / "environment_installer.sh").read_text()
        # Execute only declarations and the existing read-only PID probe. The
        # installer's dispatch, deep checks, downloads and writes are omitted.
        declarations = source[:source.index("\nstate_file_string()")]
        start = source.index("analysis_is_running() {")
        probe = source[start:source.index("\nrequire_analysis_idle()", start)]
        script = self.root / "installer-path-probe.sh"
        script.write_text(
            declarations + "\n" + probe
            + '\nprintf "%s\\n" "$PROJECT_VERSION" "$DOCTOR_CACHE_DIR"\nanalysis_is_running\n'
        )
        result = self.run_shell([
            "/bin/sh", str(script), str(self.support), str(self.project),
            str(self.maintenance), "fixture-contract",
        ])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "nested-fixture-version", str(state / "matplotlib"),
        ])
        self.assertFalse(self.support.exists())

    def test_doctor_reads_model_and_groovy_below_analysis_package(self):
        code = self.project / "Analysis Package" / "project_leap_2d"
        models = code / "resources" / "models"
        models.mkdir(parents=True)
        payload = b"isolated-model-fixture"
        (models / "instanseg_single_channel_nuclei.pt").write_bytes(payload)
        (models / "instanseg_single_channel_nuclei.json").write_text(
            json.dumps({"sha256": hashlib.sha256(payload).hexdigest()})
        )
        groovy = code / "fiji_review" / "resources" / "astrocyte_roi_reviewer.groovy"
        groovy.parent.mkdir(parents=True)
        groovy.write_text("fixture resource\n")
        spec = importlib.util.spec_from_file_location(
            "layout_fixture_doctor", MAINTENANCE / "environment_doctor.py"
        )
        doctor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(doctor)
        doctor.require_project_resources(self.project)

    def test_fixed_environment_contract_and_complete_integrity_chain(self):
        for name, expected in FIXED_COMPONENT_SHA256.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((MAINTENANCE / name).read_bytes()).hexdigest(), expected)
        contract_bytes = (
            b"schema=2\n" + (MAINTENANCE / "component_manifest.sh").read_bytes()
            + b"\n--dependency-lock--\n"
            + (MAINTENANCE / "requirements_macos_arm64.lock.txt").read_bytes()
        )
        self.assertEqual(
            hashlib.sha256(contract_bytes).hexdigest(),
            (MAINTENANCE / "environment_contract.txt").read_text().strip(),
        )

        def digest_assignment(filename, key):
            matches = re.findall(rf'(?m)^{re.escape(key)}="([0-9a-f]{{64}})"$', (MAINTENANCE / filename).read_text())
            self.assertEqual(len(matches), 1, (filename, key))
            return matches[0]

        manifest = "installer_integrity_manifest.sh"
        for name, key in RESOURCE_KEYS.items():
            with self.subTest(resource=name):
                self.assertEqual(digest_assignment(manifest, key), hashlib.sha256((MAINTENANCE / name).read_bytes()).hexdigest())
        manifest_digest = hashlib.sha256((MAINTENANCE / manifest).read_bytes()).hexdigest()
        self.assertEqual(digest_assignment("bootstrap_macos.sh", "INSTALLER_INTEGRITY_MANIFEST_SHA256"), manifest_digest)
        self.assertEqual(digest_assignment("manage_environment.command", "EXPECTED_INTEGRITY_MANIFEST_SHA256"), manifest_digest)
        self.assertEqual(
            digest_assignment("manage_environment.command", "EXPECTED_BOOTSTRAP_SHA256"),
            hashlib.sha256((MAINTENANCE / "bootstrap_macos.sh").read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
