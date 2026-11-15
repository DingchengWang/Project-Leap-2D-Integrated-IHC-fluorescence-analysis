#!/usr/bin/env python3
"""Rebuild or verify the small task-local Java preferences resource."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "engine/project_leap_2d/fiji_review/resources"
SOURCE = RESOURCES / "java/org/projectleap/desktop/TaskPreferencesFactory.java"
DESTINATION = RESOURCES / "desktop-preferences.jar"


def build(jdk_home: Path, check: bool = False) -> dict:
    javac = jdk_home / "bin/javac"
    with tempfile.TemporaryDirectory(prefix="leap-java-build-", dir="/private/tmp") as temporary:
        directory = Path(temporary)
        classes = directory / "classes"
        classes.mkdir()
        command = [str(javac), "--release", "8", "-Xlint:-options", "-g:none", "-d", str(classes), str(SOURCE)]
        subprocess.run(command, check=True)
        members = sorted(path for path in classes.rglob("*.class"))
        expected = {"org/projectleap/desktop/TaskPreferencesFactory.class",
                    "org/projectleap/desktop/TaskPreferencesFactory$Store.class"}
        if {path.relative_to(classes).as_posix() for path in members} != expected:
            raise RuntimeError("Unexpected Java class set")
        artifact = directory / "desktop-preferences.jar"
        # Fixed entry metadata and stored bytes make the artifact reproducible.
        with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_STORED) as archive:
            manifest = zipfile.ZipInfo("META-INF/MANIFEST.MF", date_time=(2020, 1, 1, 0, 0, 0))
            archive.writestr(manifest, b"Manifest-Version: 1.0\nCreated-By: Project Leap Desktop build\n\n")
            for path in members:
                entry = zipfile.ZipInfo(path.relative_to(classes).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
                archive.writestr(entry, path.read_bytes())
        content = artifact.read_bytes()
        if check:
            if DESTINATION.read_bytes() != content:
                raise RuntimeError("Packaged Java preferences JAR differs from rebuilt source")
        else:
            DESTINATION.write_bytes(content)
        return {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "jar": str(DESTINATION), "jar_sha256": hashlib.sha256(content).hexdigest(),
                "class_members": sorted(expected), "javac_arguments": command,
                "check_only": check, "matches_source": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jdk-home", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.jdk_home is None:
        choices = sorted({path.resolve() for path in (ROOT / "runtime/seed/Fiji/java").glob("**/bin/javac")})
        if len(choices) != 1:
            parser.error("Set --jdk-home to the isolated JDK containing bin/javac")
        args.jdk_home = choices[0].parent.parent
    print(json.dumps(build(args.jdk_home.resolve(), args.check), indent=2))
