from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 production environment
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
INSTALLATION = ROOT / "Installation" / "macOS"


def write_owned_installing_marker(
    marker: Path,
    *,
    contract: str,
    release: Path,
) -> None:
    marker.write_text(
        f"environment_contract_id={contract}\nrelease_dir={release}\n",
        encoding="utf-8",
    )


def write_fake_managed_runtime(release: Path) -> None:
    runtime = (
        release
        / "Managed Python"
        / "cpython-3.9.25-macos-aarch64-none"
    )
    python = runtime / "bin" / "python3.9"
    library = runtime / "lib" / "libpython3.9.dylib"
    python.parent.mkdir(parents=True, exist_ok=True)
    library.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    library.write_bytes(b"test-only-libpython")
    python.chmod(0o755)


class MacOSInstallationContractTests(unittest.TestCase):
    def test_pyproject_declares_validated_runtime_and_package_data(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            document = tomllib.load(handle)
        project = document["project"]
        dependencies = set(project["dependencies"])
        self.assertEqual(project["requires-python"], "==3.9.*")
        for expected in {
            "cellpose==4.2.1.1",
            "numpy==1.26.4",
            "openpyxl==3.1.5",
            "torch==2.8.0",
            "torchvision==0.23.0",
        }:
            self.assertIn(expected, dependencies)
        package_data = document["tool"]["setuptools"]["package-data"][
            "project_leap_2d"
        ]
        self.assertIn("resources/models/*.pt", package_data)
        self.assertIn("resources/models/*.json", package_data)
        self.assertIn("fiji_review/resources/*.groovy", package_data)

    def test_component_manifest_pins_uv_cellpose_and_fiji(self) -> None:
        text = (INSTALLATION / "component_manifest.sh").read_text(encoding="utf-8")
        self.assertIn('PROJECT_LEAP_UV_VERSION="0.11.16"', text)
        self.assertRegex(
            text,
            r'PROJECT_LEAP_UV_SHA256="[0-9a-f]{64}"',
        )
        self.assertRegex(
            text,
            r'PROJECT_LEAP_CELLPOSE_MODEL_SHA256="[0-9a-f]{64}"',
        )
        self.assertIn(
            'PROJECT_LEAP_FIJI_URL="https://downloads.imagej.net/fiji/archive/'
            'latest/20260718-0417/fiji-latest-macos-arm64-jdk.zip"',
            text,
        )
        self.assertIn(
            'PROJECT_LEAP_FIJI_SHA256="'
            'e66a395160b5affc0c2328accb4782918703918c4b7391a79cfc7300299fea72"',
            text,
        )
        self.assertIn(
            'PROJECT_LEAP_FIJI_LAUNCHER_RELATIVE="'
            'Fiji/Fiji.app/Contents/MacOS/fiji-macos-arm64"',
            text,
        )

    def test_integrity_manifests_cover_all_locked_wheels_python_and_fiji_java(
        self,
    ) -> None:
        from Installation.macOS import environment_doctor

        locked_names = {
            re.sub(r"[-_.]+", "-", line.split("==", 1)[0]).lower()
            for line in (
                INSTALLATION / "requirements_macos_arm64.lock.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
        wheels = json.loads(
            (INSTALLATION / "python_wheel_integrity.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(wheels["schema_version"], 1)
        self.assertEqual(
            wheels["algorithm"],
            "canonical-wheel-record-sha256-v2",
        )
        self.assertEqual(
            set(wheels["distributions"]),
            locked_names,
        )
        self.assertEqual(len(locked_names), 39)
        normalized_environment_files = 0
        for row in wheels["distributions"].values():
            self.assertRegex(
                row["canonical_record_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                row["hashed_entries"] + 1,
                row["record_entries"],
            )
            normalized_environment_files += row[
                "normalized_environment_files"
            ]
        self.assertEqual(normalized_environment_files, 19)

        fiji = json.loads(
            (INSTALLATION / "fiji_tree_integrity.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(fiji["schema_version"], 1)
        self.assertEqual(
            fiji["source_zip_sha256"],
            "e66a395160b5affc0c2328accb4782918703918c4b7391a79cfc7300299fea72",
        )
        self.assertEqual(fiji["file_count"], 1202)
        self.assertGreater(fiji["symlink_count"], 0)
        self.assertGreater(fiji["total_file_bytes"], 800_000_000)
        self.assertEqual(
            fiji["excluded_mutable_relative_paths"],
            [".checksums", "db.xml.gz"],
        )

        managed_python = json.loads(
            (INSTALLATION / "managed_python_integrity.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(managed_python["schema_version"], 1)
        self.assertEqual(
            managed_python["runtime"],
            "cpython-3.9.25-macos-aarch64-none",
        )
        self.assertEqual(
            managed_python["algorithm"],
            "normalized-python-runtime-sha256-v4",
        )
        self.assertGreater(managed_python["file_count"], 2_000)
        self.assertGreater(managed_python["total_file_bytes"], 40_000_000)
        self.assertEqual(
            managed_python["excluded_relative_paths"],
            [".lock", ".temp"],
        )
        self.assertEqual(
            managed_python["excluded_directory_names"],
            [],
        )

        for name, expected in (
            environment_doctor.TRUSTED_INTEGRITY_MANIFEST_SHA256.items()
        ):
            observed = hashlib.sha256(
                (INSTALLATION / name).read_bytes()
            ).hexdigest()
            self.assertEqual(observed, expected, name)

    def test_bootstrap_authenticates_the_offline_integrity_checker_bundle(
        self,
    ) -> None:
        bootstrap = (INSTALLATION / "bootstrap_macos.sh").read_text(
            encoding="utf-8"
        )
        integrity_manifest_path = (
            INSTALLATION / "installer_integrity_manifest.sh"
        )
        expected_manifest = re.search(
            r'INSTALLER_INTEGRITY_MANIFEST_SHA256="([0-9a-f]{64})"',
            bootstrap,
        )
        self.assertIsNotNone(expected_manifest)
        self.assertEqual(
            hashlib.sha256(integrity_manifest_path.read_bytes()).hexdigest(),
            expected_manifest.group(1),
        )

        integrity_manifest = integrity_manifest_path.read_text(
            encoding="utf-8"
        )
        expected_files = {
            "PROJECT_LEAP_ENVIRONMENT_INSTALLER_SHA256":
                "environment_installer.sh",
            "PROJECT_LEAP_ENVIRONMENT_DOCTOR_SHELL_SHA256":
                "environment_doctor.sh",
            "PROJECT_LEAP_ENVIRONMENT_DOCTOR_SHA256":
                "environment_doctor.py",
            "PROJECT_LEAP_COMPONENT_MANIFEST_SHA256":
                "component_manifest.sh",
            "PROJECT_LEAP_REQUIREMENTS_LOCK_SHA256":
                "requirements_macos_arm64.lock.txt",
            "PROJECT_LEAP_ENVIRONMENT_CONTRACT_SHA256":
                "environment_contract.txt",
            "PROJECT_LEAP_PYTHON_WHEEL_INTEGRITY_SHA256":
                "python_wheel_integrity.json",
            "PROJECT_LEAP_MANAGED_PYTHON_INTEGRITY_SHA256":
                "managed_python_integrity.json",
            "PROJECT_LEAP_FIJI_TREE_INTEGRITY_SHA256":
                "fiji_tree_integrity.json",
        }
        for variable, filename in expected_files.items():
            match = re.search(
                rf'{variable}="([0-9a-f]{{64}})"',
                integrity_manifest,
            )
            self.assertIsNotNone(match, variable)
            observed = hashlib.sha256(
                (INSTALLATION / filename).read_bytes()
            ).hexdigest()
            self.assertEqual(observed, match.group(1), filename)

        for filename in expected_files.values():
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    isolated_installation = (
                        Path(directory) / "Installation" / "macOS"
                    )
                    shutil.copytree(INSTALLATION, isolated_installation)
                    with (isolated_installation / filename).open("ab") as handle:
                        if filename == "component_manifest.sh":
                            handle.write(
                                b'\n/bin/touch '
                                b'"$PROJECT_DIR/untrusted-source-ran"\n'
                            )
                        else:
                            handle.write(b"\n")
                    support = Path(directory) / "Support Must Stay Absent"
                    environment = os.environ.copy()
                    environment["PROJECT_LEAP_SUPPORT_DIR"] = str(support)
                    completed = subprocess.run(
                        [
                            "/bin/sh",
                            str(
                                isolated_installation
                                / "bootstrap_macos.sh"
                            ),
                            "--dry-run",
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        f"{filename} is damaged",
                        completed.stderr,
                    )
                    self.assertFalse(support.exists())
                    self.assertFalse(
                        (
                            Path(directory)
                            / "untrusted-source-ran"
                        ).exists()
                    )

        entrypoint = (INSTALLATION / "install_macos.command").read_text(
            encoding="utf-8"
        )
        for variable, filename in (
            ("EXPECTED_BOOTSTRAP_SHA256", "bootstrap_macos.sh"),
            (
                "EXPECTED_INTEGRITY_MANIFEST_SHA256",
                "installer_integrity_manifest.sh",
            ),
        ):
            match = re.search(
                rf'{variable}="([0-9a-f]{{64}})"',
                entrypoint,
            )
            self.assertIsNotNone(match, variable)
            self.assertEqual(
                hashlib.sha256((INSTALLATION / filename).read_bytes())
                .hexdigest(),
                match.group(1),
            )

        for filename in (
            "bootstrap_macos.sh",
            "installer_integrity_manifest.sh",
        ):
            with self.subTest(entrypoint_resource=filename):
                with tempfile.TemporaryDirectory() as directory:
                    isolated_installation = (
                        Path(directory) / "Installation" / "macOS"
                    )
                    shutil.copytree(INSTALLATION, isolated_installation)
                    with (isolated_installation / filename).open("ab") as handle:
                        handle.write(b"\n")
                    support = Path(directory) / "Support Must Stay Absent"
                    environment = os.environ.copy()
                    environment["PROJECT_LEAP_SUPPORT_DIR"] = str(support)
                    completed = subprocess.run(
                        [
                            "/bin/zsh",
                            str(
                                isolated_installation
                                / "install_macos.command"
                            ),
                            "--dry-run",
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(f"{filename} is damaged", completed.stderr)
                    self.assertFalse(support.exists())

    def test_offline_content_verifiers_detect_runtime_wheel_and_fiji_mutation(
        self,
    ) -> None:
        from Installation.macOS import environment_doctor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            release = root / "Release"
            site_packages = (
                release
                / "Environment"
                / "lib"
                / "python3.9"
                / "site-packages"
            )
            dist_info = site_packages / "demo-1.0.dist-info"
            dist_info.mkdir(parents=True)
            module = site_packages / "demo.py"
            metadata = dist_info / "METADATA"
            module.write_text("VALUE = 1\n", encoding="utf-8")
            metadata.write_text(
                "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
                encoding="utf-8",
            )

            def record_row(path: Path, relative: str) -> str:
                value = path.read_bytes()
                digest = base64.urlsafe_b64encode(
                    hashlib.sha256(value).digest()
                ).rstrip(b"=").decode("ascii")
                return f"{relative},sha256={digest},{len(value)}"

            record = dist_info / "RECORD"
            record.write_text(
                "\n".join(
                    (
                        record_row(module, "demo.py"),
                        record_row(
                            metadata,
                            "demo-1.0.dist-info/METADATA",
                        ),
                        "demo-1.0.dist-info/RECORD,,",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            lock = root / "requirements.lock"
            lock.write_text("demo==1.0\n", encoding="utf-8")
            distribution = environment_doctor.importlib.metadata.Distribution.at(
                dist_info
            )
            expected = {
                "distributions": {
                    "demo": {
                        "version": "1.0",
                        "record_path": "demo-1.0.dist-info/RECORD",
                        "canonical_record_sha256": (
                            environment_doctor.canonical_wheel_record_sha256(
                                [
                                    tuple(
                                        record_row(module, "demo.py").split(
                                            ",",
                                            2,
                                        )
                                    ),
                                    tuple(
                                        record_row(
                                            metadata,
                                            "demo-1.0.dist-info/METADATA",
                                        ).split(",", 2)
                                    ),
                                    (
                                        "demo-1.0.dist-info/RECORD",
                                        "",
                                        "",
                                    ),
                                ]
                            )
                        ),
                        "record_entries": 3,
                        "hashed_entries": 2,
                        "normalized_environment_files": 0,
                    }
                }
            }
            with mock.patch.object(
                environment_doctor.importlib.metadata,
                "distributions",
                return_value=[distribution],
            ):
                environment_doctor.require_wheel_file_integrity(
                    lock,
                    release,
                    expected,
                )
                module.write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "digest changed"):
                    environment_doctor.require_wheel_file_integrity(
                        lock,
                        release,
                        expected,
                    )
                record.write_text(
                    "\n".join(
                        (
                            record_row(module, "demo.py"),
                            record_row(
                                metadata,
                                "demo-1.0.dist-info/METADATA",
                            ),
                            "demo-1.0.dist-info/RECORD,,",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "canonical wheel RECORD changed",
                ):
                    environment_doctor.require_wheel_file_integrity(
                        lock,
                        release,
                        expected,
                    )

            managed_python = release / "Managed Python"
            standard_library = (
                managed_python
                / "cpython-3.9.25-macos-aarch64-none"
                / "lib"
                / "python3.9"
            )
            standard_library.mkdir(parents=True)
            multiprocessing = standard_library / "multiprocessing.py"
            multiprocessing.write_text("TRUSTED = True\n", encoding="utf-8")
            (standard_library / "_sysconfigdata.py").write_text(
                f"prefix = {str(managed_python)!r}\n",
                encoding="utf-8",
            )
            (managed_python / ".lock").write_bytes(b"")
            observed = environment_doctor.managed_tree_fingerprint(
                managed_python,
                {".lock", ".temp"},
                set(),
                label="Managed Python",
                normalization_root=managed_python,
                normalization_token="${MANAGED_PYTHON_ROOT}",
                strip_libpython_signature=True,
            )
            managed_manifest = {
                **observed,
                "runtime": "cpython-3.9.25-macos-aarch64-none",
                "excluded_relative_paths": [".lock", ".temp"],
                "excluded_directory_names": [],
            }
            verified_signature = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="",
                stderr="",
            )
            with mock.patch.object(
                environment_doctor.subprocess,
                "run",
                return_value=verified_signature,
            ):
                environment_doctor.require_managed_python_tree_integrity(
                    release,
                    managed_manifest,
                )
                cache = standard_library / "__pycache__"
                cache.mkdir()
                (cache / "multiprocessing.pyc").write_bytes(b"unexpected cache")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "integrity changed",
                ):
                    environment_doctor.require_managed_python_tree_integrity(
                        release,
                        managed_manifest,
                    )

                shutil.rmtree(cache)
                multiprocessing.write_text(
                    "TRUSTED = False\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "integrity changed",
                ):
                    environment_doctor.require_managed_python_tree_integrity(
                        release,
                        managed_manifest,
                    )

            fiji = release / "Fiji"
            (fiji / "jars").mkdir(parents=True)
            core = fiji / "jars" / "core.jar"
            core.write_bytes(b"trusted")
            (fiji / ".checksums").write_text("mutable", encoding="utf-8")
            observed = environment_doctor.fiji_tree_fingerprint(
                fiji,
                {".checksums", "db.xml.gz"},
            )
            fiji_manifest = {
                **observed,
                "excluded_mutable_relative_paths": [
                    ".checksums",
                    "db.xml.gz",
                ],
            }
            environment_doctor.require_fiji_tree_integrity(
                release,
                fiji_manifest,
            )
            (fiji / ".checksums").write_text("changed", encoding="utf-8")
            environment_doctor.require_fiji_tree_integrity(
                release,
                fiji_manifest,
            )
            core.write_bytes(b"damaged")
            with self.assertRaisesRegex(RuntimeError, "integrity changed"):
                environment_doctor.require_fiji_tree_integrity(
                    release,
                    fiji_manifest,
                )

    def test_canonical_wheel_record_is_environment_path_independent(
        self,
    ) -> None:
        from Installation.macOS import environment_doctor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fingerprints = []
            raw_hashes = []
            for environment_root in (
                root / "short" / "Environment",
                (
                    root
                    / "a-much-longer-project-leap-support-directory"
                    / "Releases"
                    / "environment-contract"
                    / "Environment"
                ),
            ):
                script = environment_root / "bin" / "cellpose"
                script.parent.mkdir(parents=True)
                value = (
                    "#!/bin/sh\n"
                    f"exec '{environment_root}/bin/python3' \"$0\" \"$@\"\n"
                ).encode("utf-8")
                script.write_bytes(value)
                raw_digest = base64.urlsafe_b64encode(
                    hashlib.sha256(value).digest()
                ).rstrip(b"=").decode("ascii")
                row, normalized = (
                    environment_doctor.canonical_wheel_record_row(
                        package_path="../../../bin/cellpose",
                        recorded_hash=raw_digest,
                        recorded_size=len(value),
                        resolved_path=script.resolve(strict=True),
                        environment_root=environment_root.resolve(strict=True),
                    )
                )
                self.assertTrue(normalized)
                fingerprints.append(
                    environment_doctor.canonical_wheel_record_sha256(
                        [
                            row,
                            (
                                "cellpose-4.2.1.1.dist-info/RECORD",
                                "",
                                "",
                            ),
                        ]
                    )
                )
                raw_hashes.append(raw_digest)
            self.assertNotEqual(raw_hashes[0], raw_hashes[1])
            self.assertEqual(fingerprints[0], fingerprints[1])

            script.write_bytes(
                script.read_bytes().replace(b"exec ", b"exec env ")
            )
            altered_value = script.read_bytes()
            altered_digest = base64.urlsafe_b64encode(
                hashlib.sha256(altered_value).digest()
            ).rstrip(b"=").decode("ascii")
            altered_row, normalized = (
                environment_doctor.canonical_wheel_record_row(
                    package_path="../../../bin/cellpose",
                    recorded_hash=altered_digest,
                    recorded_size=len(altered_value),
                    resolved_path=script.resolve(strict=True),
                    environment_root=environment_root.resolve(strict=True),
                )
            )
            self.assertTrue(normalized)
            altered_fingerprint = (
                environment_doctor.canonical_wheel_record_sha256(
                    [
                        altered_row,
                        (
                            "cellpose-4.2.1.1.dist-info/RECORD",
                            "",
                            "",
                        ),
                    ]
                )
            )
            self.assertNotEqual(altered_fingerprint, fingerprints[-1])

    def test_integrity_manifest_tampering_is_rejected_before_use(self) -> None:
        from Installation.macOS import environment_doctor

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            names = (
                "requirements_macos_arm64.lock.txt",
                "python_wheel_integrity.json",
                "managed_python_integrity.json",
                "fiji_tree_integrity.json",
            )
            for name in names:
                shutil.copy2(INSTALLATION / name, copied / name)
            lock = copied / "requirements_macos_arm64.lock.txt"
            environment_doctor.load_integrity_manifests(
                lock,
                expected_fiji_zip_sha256=(
                    "e66a395160b5affc0c2328accb4782918703918c4b7391a79cfc7300299fea72"
                ),
            )
            managed_manifest = copied / "managed_python_integrity.json"
            managed_manifest.write_text(
                managed_manifest.read_text(encoding="utf-8").replace(
                    '"file_count": 2346',
                    '"file_count": 2345',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "trusted SHA-256"):
                environment_doctor.load_integrity_manifests(
                    lock,
                    expected_fiji_zip_sha256=(
                        "e66a395160b5affc0c2328accb4782918703918c4b7391a79cfc7300299fea72"
                    ),
                )

    def test_managed_tree_normalization_is_path_independent_and_rejects_root_link(
        self,
    ) -> None:
        from Installation.macOS import environment_doctor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "short"
            second = root / "a-much-longer-managed-python-installation-root"
            for managed in (first, second):
                runtime = managed / "cpython-3.9.25-macos-aarch64-none"
                runtime.mkdir(parents=True)
                (runtime / "_sysconfigdata.py").write_text(
                    f"prefix = {str(managed)!r}\n",
                    encoding="utf-8",
                )
                (managed / "cpython-3.9-macos-aarch64-none").symlink_to(
                    runtime,
                    target_is_directory=True,
                )

            def fingerprint(managed: Path) -> dict[str, int | str]:
                return environment_doctor.managed_tree_fingerprint(
                    managed,
                    set(),
                    set(),
                    label="Managed Python",
                    normalization_root=managed,
                    normalization_token="${MANAGED_PYTHON_ROOT}",
                )

            self.assertEqual(fingerprint(first), fingerprint(second))
            linked_root = root / "linked-root"
            linked_root.symlink_to(first, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "not a managed directory"):
                fingerprint(linked_root)

    def test_virtual_environment_chain_and_inventory_reject_unregistered_code(
        self,
    ) -> None:
        from Installation.macOS import environment_doctor

        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "Release"
            managed = release / "Managed Python"
            runtime_name = "cpython-3.9.25-macos-aarch64-none"
            runtime = managed / runtime_name
            runtime_python = runtime / "bin" / "python3.9"
            runtime_python.parent.mkdir(parents=True)
            runtime_python.write_bytes(b"python")
            alias = managed / "cpython-3.9-macos-aarch64-none"
            alias.symlink_to(runtime, target_is_directory=True)

            environment = release / "Environment"
            bin_dir = environment / "bin"
            site_packages = (
                environment / "lib" / "python3.9" / "site-packages"
            )
            bin_dir.mkdir(parents=True)
            site_packages.mkdir(parents=True)
            (bin_dir / "python").symlink_to(alias / "bin" / "python3.9")
            (bin_dir / "python3").symlink_to("python")
            (bin_dir / "python3.9").symlink_to("python")
            (environment / "pyvenv.cfg").write_text(
                "\n".join(
                    (
                        f"home = {alias / 'bin'}",
                        "implementation = CPython",
                        "uv = 0.11.16",
                        "version_info = 3.9.25",
                        "include-system-site-packages = false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            environment_doctor.require_virtual_environment_files(
                release,
                runtime_name,
            )
            trusted_environment_files = {}
            for name, content in (
                ("_virtualenv.pth", b"import _virtualenv"),
                ("_virtualenv.py", b"# fixed virtual-environment bootstrap\n"),
            ):
                path = site_packages / name
                path.write_bytes(content)
                trusted_environment_files[
                    path.relative_to(environment).as_posix()
                ] = (hashlib.sha256(content).hexdigest(), len(content))

            inventory_patch = mock.patch.object(
                environment_doctor,
                "TRUSTED_VIRTUAL_ENVIRONMENT_FILES",
                trusted_environment_files,
            )
            with inventory_patch:
                environment_doctor.require_environment_inventory(
                    release,
                    set(),
                )
                for name in ("sitecustomize.py", "unexpected.pth"):
                    unexpected = site_packages / name
                    unexpected.write_text(
                        "raise SystemExit(1)\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Unregistered Python environment path",
                    ):
                        environment_doctor.require_environment_inventory(
                            release,
                            set(),
                        )
                    unexpected.unlink()

                virtualenv_path = site_packages / "_virtualenv.pth"
                virtualenv_path.write_text(
                    "import altered_virtualenv",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "virtual-environment bootstrap changed",
                ):
                    environment_doctor.require_environment_inventory(
                        release,
                        set(),
                    )
            with mock.patch.object(
                environment_doctor.sys,
                "prefix",
                str(environment),
            ):
                with mock.patch.object(
                    environment_doctor.sys,
                    "base_prefix",
                    str(alias),
                ):
                    with mock.patch.object(
                        environment_doctor.sys,
                        "executable",
                        str(bin_dir / "python3"),
                    ):
                        environment_doctor.require_running_virtual_environment(
                            release,
                            runtime_name,
                        )

            config = environment / "pyvenv.cfg"
            config.write_text(
                config.read_text(encoding="utf-8").replace(
                    "include-system-site-packages = false",
                    "include-system-site-packages = true",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "pyvenv.cfg changed"):
                environment_doctor.require_virtual_environment_files(
                    release,
                    runtime_name,
                )

    def test_fiji_archive_extracts_its_own_top_level_directory(self) -> None:
        text = (INSTALLATION / "environment_installer.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '/usr/bin/ditto -x -k "$FIJI_ARCHIVE" "$RELEASE_DIR"',
            text,
        )
        self.assertIn(
            '/usr/bin/tar -xzf "$FIJI_ARCHIVE" -C "$RELEASE_DIR"',
            text,
        )
        self.assertNotIn(
            '/usr/bin/ditto -x -k "$FIJI_ARCHIVE" "$RELEASE_DIR/Fiji"',
            text,
        )
        self.assertNotIn(
            '/usr/bin/tar -xzf "$FIJI_ARCHIVE" -C "$RELEASE_DIR/Fiji"',
            text,
        )
        self.assertIn(
            'test -x "$FIJI_LAUNCHER"',
            text,
        )

    def test_uv_python_is_confined_to_the_support_release(self) -> None:
        text = (INSTALLATION / "environment_installer.sh").read_text(
            encoding="utf-8"
        )
        install_dir_position = text.index(
            'export UV_PYTHON_INSTALL_DIR="$RELEASE_DIR/Managed Python"'
        )
        install_position = text.index(
            '"$UV_BIN" --no-progress python install --no-bin '
            '"$PROJECT_LEAP_PYTHON_VERSION"'
        )
        find_position = text.index(
            '"$UV_BIN" python find'
        )
        self.assertLess(install_dir_position, install_position)
        self.assertLess(install_position, find_position)
        self.assertIn("--python-preference only-managed", text)
        self.assertNotIn('UV_PYTHON_BIN_DIR=', text)
        self.assertIn(
            '"$check_managed_python" \\\n'
            "    -B -I -S \\",
            text,
        )
        self.assertIn(
            '"$check_python" \\\n'
            "    -B -I \\",
            text,
        )
        cache_cleanup = text[
            text.index("  cache_found=0")
            :text.index('  /bin/mkdir -p "$DOCTOR_CACHE_DIR"')
        ]
        self.assertIn('"$check_release/Managed Python"', cache_cleanup)
        self.assertNotIn('"$check_release/Environment"', cache_cleanup)

    def test_environment_contract_matches_manifest_lock_and_schema(self) -> None:
        import hashlib

        manifest = (INSTALLATION / "component_manifest.sh").read_bytes()
        lock_file = (
            INSTALLATION / "requirements_macos_arm64.lock.txt"
        ).read_bytes()
        payload = (
            b"schema=2\n"
            + manifest
            + b"\n--dependency-lock--\n"
            + lock_file
        )
        expected = hashlib.sha256(payload).hexdigest()
        observed = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.assertEqual(observed, expected)

    def test_dependency_lock_is_exact_and_has_no_duplicate_names(self) -> None:
        lock_file = INSTALLATION / "requirements_macos_arm64.lock.txt"
        observed: set[str] = set()
        for raw_line in lock_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            self.assertEqual(line.count("=="), 1, line)
            name, version = line.split("==", 1)
            canonical = re.sub(r"[-_.]+", "-", name).lower()
            self.assertNotIn(canonical, observed)
            self.assertTrue(version)
            observed.add(canonical)
        self.assertTrue(
            {
                "cellpose",
                "imagecodecs",
                "numpy",
                "opencv-python-headless",
                "openpyxl",
                "scikit-image",
                "scipy",
                "torch",
                "torchvision",
            }.issubset(observed)
        )

    def test_bootstrap_dry_run_does_not_create_support_or_use_pipe_to_shell(
        self,
    ) -> None:
        bootstrap = INSTALLATION / "bootstrap_macos.sh"
        installer = INSTALLATION / "environment_installer.sh"
        combined = (
            bootstrap.read_text(encoding="utf-8")
            + installer.read_text(encoding="utf-8")
        )
        self.assertNotRegex(combined, r"curl[^\n]*\|\s*(?:sh|bash|zsh)")

        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support Must Stay Absent"
            environment = os.environ.copy()
            environment["PROJECT_LEAP_SUPPORT_DIR"] = str(support)
            completed = subprocess.run(
                ["/bin/sh", str(bootstrap), "--dry-run"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Fiji: fixed archive and SHA-256 present", completed.stdout)
            self.assertNotIn("BLOCKED", completed.stdout)
            self.assertFalse(support.exists())

    def test_run_launcher_reuses_ready_contract_across_project_versions(self) -> None:
        launcher = ROOT / "run_project_leap_2d.command"
        launcher_text = launcher.read_text(encoding="utf-8")
        self.assertNotIn(str(Path.home()), launcher_text)
        self.assertIn("installation_state.json", launcher_text)
        self.assertIn("CELLPOSE_LOCAL_MODELS_PATH", launcher_text)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", launcher_text)

        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            python_bin = support / "Environment" / "bin" / "python3"
            fiji_bin = support / "Fiji" / "fiji"
            model_dir = support / "Models"
            quick_counter = support / "deep-check-counter"
            python_bin.parent.mkdir(parents=True)
            fiji_bin.parent.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            python_bin.write_text(
                f"#!/bin/sh\nprintf x >> {str(quick_counter)!r}\nexit 91\n",
                encoding="utf-8",
            )
            fiji_bin.write_text("#!/bin/sh\nexit 92\n", encoding="utf-8")
            python_bin.chmod(0o755)
            fiji_bin.chmod(0o755)
            (model_dir / "cpsam_v2").write_bytes(b"model-present")
            state = {
                "schema_version": "2",
                "status": "ready",
                "platform": "macos-arm64",
                "environment_contract_id": (
                    INSTALLATION / "environment_contract.txt"
                ).read_text(encoding="utf-8").strip(),
                "project_version": "an-older-project-version",
                "python_executable": str(python_bin),
                "fiji_launcher": str(fiji_bin),
                "cellpose_models_path": str(model_dir),
            }
            (support / "installation_state.json").write_text(
                json.dumps(state, indent=2),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PROJECT_LEAP_SUPPORT_DIR"] = str(support)
            started = time.perf_counter()
            completed = subprocess.run(
                [str(launcher), "--check-environment"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            elapsed = time.perf_counter() - started
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("environment is ready", completed.stdout)
            self.assertLess(elapsed, 0.1)

            doctor = subprocess.run(
                [
                    str(INSTALLATION / "environment_doctor.sh"),
                    "--quick",
                    str(support),
                    str(ROOT),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(doctor.returncode, 0, doctor.stderr)
            self.assertIn("environment is ready", doctor.stdout)
            self.assertFalse(quick_counter.exists())

    def test_ready_state_recovers_residual_installing_marker(self) -> None:
        installer = INSTALLATION / "environment_installer.sh"
        installer_text = installer.read_text(encoding="utf-8")
        cleanup_position = installer_text.rindex(
            '/bin/rm -rf "$RELEASE_DIR/Downloads" "$RELEASE_DIR/Cache"'
        )
        publish_position = installer_text.rindex(
            '/bin/mv -f "$STATE_NEXT" "$STATE_FILE"'
        )
        marker_position = installer_text.rindex(
            '/bin/rm -f "$RELEASE_DIR/INSTALLING"'
        )
        self.assertLess(cleanup_position, publish_position)
        self.assertLess(publish_position, marker_position)

        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            release = support / "Releases" / release_id
            python_bin = release / "Environment" / "bin" / "python3"
            fiji_bin = (
                release
                / "Fiji"
                / "Fiji.app"
                / "Contents"
                / "MacOS"
                / "fiji-macos-arm64"
            )
            model_dir = release / "Models"
            deep_counter = support / "deep-check-counter"
            python_bin.parent.mkdir(parents=True)
            fiji_bin.parent.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            write_fake_managed_runtime(release)
            python_bin.write_text(
                f"#!/bin/sh\nprintf x >> {str(deep_counter)!r}\nexit 0\n",
                encoding="utf-8",
            )
            fiji_bin.write_text("#!/bin/sh\nexit 92\n", encoding="utf-8")
            python_bin.chmod(0o755)
            fiji_bin.chmod(0o755)
            (model_dir / "cpsam_v2").write_bytes(b"model-present")
            marker = release / "INSTALLING"
            marker.write_text("", encoding="utf-8")
            state = {
                "schema_version": "2",
                "status": "ready",
                "platform": "macos-arm64",
                "environment_contract_id": contract,
                "project_version": "an-older-project-version",
                "python_executable": str(python_bin),
                "fiji_launcher": str(fiji_bin),
                "cellpose_models_path": str(model_dir),
            }
            support.mkdir(parents=True, exist_ok=True)
            (support / "installation_state.json").write_text(
                json.dumps(state, indent=2),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(installer),
                    str(support),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Checking the installed", completed.stdout)
            self.assertIn("passed the full integrity check", completed.stdout)
            self.assertEqual(deep_counter.read_text(encoding="utf-8"), "x")
            self.assertFalse(marker.exists())
            self.assertTrue(release.is_dir())
            self.assertTrue((support / "installation_state.json").is_file())
            sentinel = release / "healthy-environment-sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            state_path = support / "installation_state.json"
            state_before = state_path.read_bytes()
            state_mtime_before = state_path.stat().st_mtime_ns

            repeated = subprocess.run(
                [
                    "/bin/sh",
                    str(installer),
                    str(support),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("passed the full integrity check", repeated.stdout)
            self.assertEqual(deep_counter.read_text(encoding="utf-8"), "xx")
            self.assertEqual(state_path.read_bytes(), state_before)
            self.assertEqual(state_path.stat().st_mtime_ns, state_mtime_before)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_launcher_and_installer_share_one_environment_lock(self) -> None:
        launcher = ROOT / "run_project_leap_2d.command"
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            python_bin = support / "Environment" / "bin" / "python3"
            fiji_bin = support / "Fiji" / "fiji"
            model_dir = support / "Models"
            invocation = support / "python-invoked"
            python_bin.parent.mkdir(parents=True)
            fiji_bin.parent.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            python_bin.write_text(
                "#!/bin/sh\n"
                'test -f "$PROJECT_LEAP_USAGE_LOCK_FILE" || exit 81\n'
                "test -e /dev/fd/9 || exit 82\n"
                f"printf invoked > {str(invocation)!r}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fiji_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_bin.chmod(0o755)
            fiji_bin.chmod(0o755)
            (model_dir / "cpsam_v2").write_bytes(b"model-present")
            state = {
                "schema_version": "2",
                "status": "ready",
                "platform": "macos-arm64",
                "environment_contract_id": contract,
                "project_version": "1.0.0",
                "python_executable": str(python_bin),
                "fiji_launcher": str(fiji_bin),
                "cellpose_models_path": str(model_dir),
            }
            (support / "installation_state.json").write_text(
                json.dumps(state, indent=2),
                encoding="utf-8",
            )
            environment = {**os.environ, "PROJECT_LEAP_SUPPORT_DIR": str(support)}

            launched = subprocess.run(
                [str(launcher)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(launched.returncode, 0, launched.stderr)
            self.assertEqual(invocation.read_text(encoding="utf-8"), "invoked")
            lock_file = support / "Environment Usage Lock"
            self.assertTrue(lock_file.is_file())
            holder_ready = support / "holder-ready"
            holder = subprocess.Popen(
                [
                    "/bin/sh",
                    "-c",
                    (
                        'exec 9>>"$1"; '
                        "/usr/bin/lockf -s -t 0 9 || exit 91; "
                        'printf ready >"$2"; '
                        "/bin/sleep 5"
                    ),
                    "sh",
                    str(lock_file),
                    str(holder_ready),
                ],
                cwd=ROOT,
            )
            for _ in range(100):
                if holder_ready.exists():
                    break
                time.sleep(0.01)
            self.assertTrue(holder_ready.exists())
            invocation.unlink()
            try:
                blocked = subprocess.run(
                    [str(launcher)],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            finally:
                holder.terminate()
                holder.wait(timeout=5)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn(
                "installation or analysis is already using",
                blocked.stderr,
            )
            self.assertFalse(invocation.exists())
            self.assertTrue(lock_file.is_file())

    def test_entrypoint_needs_no_manual_stale_lock_cleanup(self) -> None:
        entrypoint = (ROOT / "project_leap_2d" / "__main__.py").read_text(
            encoding="utf-8"
        )
        launcher = (ROOT / "run_project_leap_2d.command").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_release_analysis_environment_lock", entrypoint)
        self.assertIn("/usr/bin/lockf -s -t 0 9", launcher)

    def test_corrupt_owned_release_repairs_once_and_rolls_back_on_failure(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = root / "Support"
            release = support / "Releases" / release_id
            python_bin = release / "Environment" / "bin" / "python3"
            fiji_bin = (
                release
                / "Fiji"
                / "Fiji.app"
                / "Contents"
                / "MacOS"
                / "fiji-macos-arm64"
            )
            model_dir = release / "Models"
            python_bin.parent.mkdir(parents=True)
            fiji_bin.parent.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            write_fake_managed_runtime(release)
            # A broken interpreter/native dependency may exit outside the
            # doctor's normal 20/21/22 protocol. Owned damage still receives
            # the same single transactional repair attempt.
            python_bin.write_text("#!/bin/sh\nexit 139\n", encoding="utf-8")
            fiji_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_bin.chmod(0o755)
            fiji_bin.chmod(0o755)
            (model_dir / "cpsam_v2").write_bytes(b"damaged-model")
            sentinel = release / "previous-environment-sentinel"
            sentinel.write_text("preserve", encoding="utf-8")
            state = {
                "schema_version": "2",
                "status": "ready",
                "platform": "macos-arm64",
                "environment_contract_id": contract,
                "project_version": "1.0.0",
                "python_executable": str(python_bin),
                "fiji_launcher": str(fiji_bin),
                "cellpose_models_path": str(model_dir),
            }
            state_path = support / "installation_state.json"
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            state_before = state_path.read_bytes()

            isolated_installation = root / "Installation"
            shutil.copytree(INSTALLATION, isolated_installation)
            manifest = isolated_installation / "component_manifest.sh"
            manifest_text = manifest.read_text(encoding="utf-8")
            manifest_text = re.sub(
                r'PROJECT_LEAP_UV_URL="[^"]+"',
                'PROJECT_LEAP_UV_URL="https://127.0.0.1:9/uv.tar.gz"',
                manifest_text,
                count=1,
            )
            manifest.write_text(manifest_text, encoding="utf-8")

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(isolated_installation / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(isolated_installation),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(
                completed.stdout.count("Starting one automatic repair"),
                1,
            )
            self.assertIn("uv download failed", completed.stderr)
            self.assertIn(
                "previous Project Leap 2D environment was restored",
                completed.stderr,
            )
            self.assertEqual(state_path.read_bytes(), state_before)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertFalse((release / "INSTALLING").exists())
            self.assertFalse(
                (support / "Releases" / f"{release_id} Repair Backup").exists()
            )
            self.assertFalse(
                (support / f"Repair State {release_id}").exists()
            )

    def test_interrupted_installing_release_is_rebuilt_from_scratch(
        self,
    ) -> None:
        installer_text = (
            INSTALLATION / "environment_installer.sh"
        ).read_text(encoding="utf-8")
        download_start = installer_text.index("download_verified()")
        download_end = installer_text.index("\n}\n", download_start)
        download_function = installer_text[download_start:download_end]
        self.assertNotIn("--continue-at", download_function)
        self.assertNotRegex(download_function, r"(?:^|\s)-C(?:\s|$)")
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = root / "Support"
            release = support / "Releases" / release_id
            release.mkdir(parents=True)
            write_owned_installing_marker(
                release / "INSTALLING",
                contract=contract,
                release=release,
            )
            (release / "old-partial-sentinel").write_text(
                "must-not-resume",
                encoding="utf-8",
            )

            isolated_installation = root / "Installation"
            shutil.copytree(INSTALLATION, isolated_installation)
            manifest = isolated_installation / "component_manifest.sh"
            manifest_text = manifest.read_text(encoding="utf-8")
            manifest_text = re.sub(
                r'PROJECT_LEAP_UV_URL="[^"]+"',
                'PROJECT_LEAP_UV_URL="https://127.0.0.1:9/uv.tar.gz"',
                manifest_text,
                count=1,
            )
            manifest.write_text(manifest_text, encoding="utf-8")

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(isolated_installation / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(isolated_installation),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("uv download failed", completed.stderr)
            self.assertFalse(release.exists())
            self.assertFalse((support / "installation_state.json").exists())
            self.assertFalse(
                (support / "Releases" / f"{release_id} Repair Backup").exists()
            )
            self.assertFalse(
                (support / f"Repair State {release_id}").exists()
            )

    def test_empty_release_from_pre_marker_power_loss_restarts_cleanly(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = root / "Support"
            release = support / "Releases" / release_id
            release.mkdir(parents=True)

            isolated_installation = root / "Installation"
            shutil.copytree(INSTALLATION, isolated_installation)
            manifest = isolated_installation / "component_manifest.sh"
            manifest.write_text(
                re.sub(
                    r'PROJECT_LEAP_UV_URL="[^"]+"',
                    'PROJECT_LEAP_UV_URL="https://127.0.0.1:9/uv.tar.gz"',
                    manifest.read_text(encoding="utf-8"),
                    count=1,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(isolated_installation / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(isolated_installation),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("uv download failed", completed.stderr)
            self.assertNotIn("unverified release directory", completed.stderr)
            self.assertFalse(release.exists())
            self.assertTrue((support / "Environment Usage Lock").is_file())

    def test_interrupted_automatic_repair_restores_previous_release(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            release = support / "Releases" / release_id
            backup = support / "Releases" / f"{release_id} Repair Backup"
            repair_state = support / f"Repair State {release_id}"

            python_bin = backup / "Environment" / "bin" / "python3"
            fiji_bin = (
                backup
                / "Fiji"
                / "Fiji.app"
                / "Contents"
                / "MacOS"
                / "fiji-macos-arm64"
            )
            model_dir = backup / "Models"
            python_bin.parent.mkdir(parents=True)
            fiji_bin.parent.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            write_fake_managed_runtime(backup)
            python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fiji_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_bin.chmod(0o755)
            fiji_bin.chmod(0o755)
            (model_dir / "cpsam_v2").write_bytes(b"previous-model")
            (backup / "previous-environment-sentinel").write_text(
                "restore-me",
                encoding="utf-8",
            )

            canonical_python = release / "Environment" / "bin" / "python3"
            canonical_fiji = (
                release
                / "Fiji"
                / "Fiji.app"
                / "Contents"
                / "MacOS"
                / "fiji-macos-arm64"
            )
            canonical_models = release / "Models"
            state = {
                "schema_version": "2",
                "status": "ready",
                "platform": "macos-arm64",
                "environment_contract_id": contract,
                "project_version": "1.0.0",
                "python_executable": str(canonical_python),
                "fiji_launcher": str(canonical_fiji),
                "cellpose_models_path": str(canonical_models),
            }
            support.mkdir(parents=True, exist_ok=True)
            state_path = support / "installation_state.json"
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

            release.mkdir(parents=True)
            write_owned_installing_marker(
                release / "INSTALLING",
                contract=contract,
                release=release,
            )
            (release / "replacement-partial-sentinel").write_text(
                "discard-me",
                encoding="utf-8",
            )
            repair_state.mkdir()
            (repair_state / "environment_contract_id").write_text(
                contract + "\n",
                encoding="utf-8",
            )
            (repair_state / "release_dir").write_text(
                str(release) + "\n",
                encoding="utf-8",
            )
            (repair_state / "backup_dir").write_text(
                str(backup) + "\n",
                encoding="utf-8",
            )
            (repair_state / "installation_state.json").write_bytes(
                state_path.read_bytes()
            )

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(INSTALLATION / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "restored after an interrupted repair",
                completed.stdout,
            )
            self.assertIn("passed the full integrity check", completed.stdout)
            self.assertEqual(
                (release / "previous-environment-sentinel").read_text(
                    encoding="utf-8"
                ),
                "restore-me",
            )
            self.assertFalse(
                (release / "replacement-partial-sentinel").exists()
            )
            self.assertFalse(backup.exists())
            self.assertFalse(repair_state.exists())

    def test_unowned_repair_backup_is_preserved_and_blocks_installation(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            release = support / "Releases" / release_id
            backup = support / "Releases" / f"{release_id} Repair Backup"
            release.mkdir(parents=True)
            backup.mkdir(parents=True)
            unknown_marker = release / "INSTALLING"
            unknown_marker.write_text("", encoding="utf-8")
            release_sentinel = release / "release-sentinel"
            backup_sentinel = backup / "backup-sentinel"
            release_sentinel.write_text("release", encoding="utf-8")
            backup_sentinel.write_text("backup", encoding="utf-8")

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(INSTALLATION / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "unverified automatic-repair record exists",
                completed.stderr,
            )
            self.assertEqual(
                release_sentinel.read_text(encoding="utf-8"),
                "release",
            )
            self.assertEqual(
                backup_sentinel.read_text(encoding="utf-8"),
                "backup",
            )
            self.assertTrue(unknown_marker.exists())

    def test_second_installer_cannot_clean_the_active_installation(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            release = support / "Releases" / release_id
            release.mkdir(parents=True)
            marker = release / "INSTALLING"
            sentinel = release / "active-installation-sentinel"
            write_owned_installing_marker(
                marker,
                contract=contract,
                release=release,
            )
            sentinel.write_text("keep-running", encoding="utf-8")
            lock_file = support / "Environment Usage Lock"
            holder_ready = support / "holder-ready"
            holder = subprocess.Popen(
                [
                    "/bin/sh",
                    "-c",
                    (
                        'exec 9>>"$1"; '
                        "/usr/bin/lockf -s -t 0 9 || exit 91; "
                        'printf ready >"$2"; '
                        "/bin/sleep 5"
                    ),
                    "sh",
                    str(lock_file),
                    str(holder_ready),
                ],
                cwd=ROOT,
            )
            for _ in range(100):
                if holder_ready.exists():
                    break
                time.sleep(0.01)
            self.assertTrue(holder_ready.exists())

            try:
                completed = subprocess.run(
                    [
                        "/bin/sh",
                        str(INSTALLATION / "environment_installer.sh"),
                        str(support),
                        str(ROOT),
                        str(INSTALLATION),
                        contract,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            finally:
                holder.terminate()
                holder.wait(timeout=5)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "another installation or analysis is already using",
                completed.stderr,
            )
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "keep-running",
            )
            self.assertTrue(marker.exists())
            self.assertTrue(lock_file.is_file())

    def test_only_one_installer_holds_the_kernel_environment_lock(self) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            release = support / "Releases" / release_id
            python_bin = release / "Environment" / "bin" / "python3"
            fiji_bin = (
                release
                / "Fiji"
                / "Fiji.app"
                / "Contents"
                / "MacOS"
                / "fiji-macos-arm64"
            )
            model_dir = release / "Models"
            python_bin.parent.mkdir(parents=True)
            fiji_bin.parent.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            write_fake_managed_runtime(release)
            python_bin.write_text(
                "#!/bin/sh\n/bin/sleep 1\nexit 0\n",
                encoding="utf-8",
            )
            fiji_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_bin.chmod(0o755)
            fiji_bin.chmod(0o755)
            (model_dir / "cpsam_v2").write_bytes(b"model-present")
            state = {
                "schema_version": "2",
                "status": "ready",
                "platform": "macos-arm64",
                "environment_contract_id": contract,
                "project_version": "1.0.0",
                "python_executable": str(python_bin),
                "fiji_launcher": str(fiji_bin),
                "cellpose_models_path": str(model_dir),
            }
            (support / "installation_state.json").write_text(
                json.dumps(state, indent=2),
                encoding="utf-8",
            )
            command = [
                "/bin/sh",
                str(INSTALLATION / "environment_installer.sh"),
                str(support),
                str(ROOT),
                str(INSTALLATION),
                contract,
            ]
            processes = [
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            results = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=5)
                results.append((process.returncode, stdout, stderr))

            self.assertEqual(sorted(item[0] for item in results), [0, 1])
            winner = next(item for item in results if item[0] == 0)
            loser = next(item for item in results if item[0] != 0)
            self.assertIn("passed the full integrity check", winner[1])
            self.assertIn(
                "already using",
                loser[2],
            )
            lock_file = support / "Environment Usage Lock"
            self.assertTrue(lock_file.is_file())
            unlocked = subprocess.run(
                [
                    "/bin/sh",
                    "-c",
                    'exec 9>>"$1"; /usr/bin/lockf -s -t 0 9',
                    "sh",
                    str(lock_file),
                ],
                check=False,
            )
            self.assertEqual(unlocked.returncode, 0)
            self.assertTrue(release.is_dir())

    def test_unverified_installing_marker_is_preserved(self) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        release_id = f"environment-{contract[:16]}"
        with tempfile.TemporaryDirectory() as directory:
            support = Path(directory) / "Support"
            release = support / "Releases" / release_id
            release.mkdir(parents=True)
            marker = release / "INSTALLING"
            sentinel = release / "unknown-partial-sentinel"
            marker.write_text("", encoding="utf-8")
            sentinel.write_text("preserve", encoding="utf-8")

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(INSTALLATION / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unverified release directory", completed.stderr)
            self.assertTrue(marker.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_symbolic_link_support_directory_is_rejected_without_changes(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "real-support"
            target.mkdir()
            support_link = root / "linked-support"
            support_link.symlink_to(target, target_is_directory=True)

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(INSTALLATION / "environment_installer.sh"),
                    str(support_link),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must not be a symbolic link", completed.stderr)
            self.assertEqual(list(target.iterdir()), [])
            self.assertTrue(support_link.is_symlink())

    def test_state_symlink_and_predictable_old_next_file_are_never_followed(
        self,
    ) -> None:
        contract = (INSTALLATION / "environment_contract.txt").read_text(
            encoding="utf-8"
        ).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = root / "Support"
            support.mkdir()
            sentinel = root / "sentinel"
            sentinel.write_text("do-not-overwrite", encoding="utf-8")
            state = support / "installation_state.json"
            state.symlink_to(sentinel)
            old_next = support / "installation_state.json.next-12345"
            old_next.symlink_to(sentinel)

            completed = subprocess.run(
                [
                    "/bin/sh",
                    str(INSTALLATION / "environment_installer.sh"),
                    str(support),
                    str(ROOT),
                    str(INSTALLATION),
                    contract,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("state path is not a regular file", completed.stderr)
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "do-not-overwrite",
            )
            self.assertTrue(state.is_symlink())
            self.assertTrue(old_next.is_symlink())
            installer_text = (
                INSTALLATION / "environment_installer.sh"
            ).read_text(encoding="utf-8")
            self.assertNotIn("installation_state.json.next-$$", installer_text)
            self.assertIn("mktemp -d", installer_text)


if __name__ == "__main__":
    unittest.main()
