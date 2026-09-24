"""Fixed-version release transport contracts using only local test fixtures."""
from __future__ import annotations

import json
import os
import signal
import sys
import unittest

from test_package_repair import Fixture, MANIFEST_NAME, ZIP_NAME, wait_for, write_file


class ReleaseFetchTests(unittest.TestCase):
    def fixture(self):
        fixture = Fixture()
        self.addCleanup(fixture.close)
        return fixture

    def assert_stopped_unchanged(self, fixture, fragment):
        before = fixture.source_snapshot()
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(fragment, completed.stderr)
        self.assertEqual(fixture.source_snapshot(), before)
        self.assertEqual(fixture.state_path.read_bytes(), fixture.original_state_bytes)
        self.assertEqual(fixture.records("environment.jsonl"), [])
        return completed

    def test_fixed_tag_and_public_downloads_need_no_user_credentials(self):
        fixture = self.fixture()
        raw = fixture.transport.read_text()
        raw = raw.replace("args = sys.argv[1:]", "args = sys.argv[1:]\nassert not any('Authorization' in item or 'Bearer' in item for item in args)\nassert '--proto' in args and '--proto-redir' in args")
        fixture.transport.write_text(raw)
        completed = fixture.run()
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        requests = fixture.records("transport.jsonl")
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0]["url"].endswith("/releases/tags/v1.0.1"))
        self.assertTrue(requests[1]["url"].endswith("/" + MANIFEST_NAME))
        self.assertNotIn("latest", json.dumps(requests))
        self.assertNotIn("/main/", json.dumps(requests))

    def test_unpublished_release_stops_without_environment_changes(self):
        fixture = self.fixture()
        write_file(fixture.transport, "#!/bin/sh\nexit 22\n", 0o755)
        self.assert_stopped_unchanged(fixture, "may not yet be published")

    def test_release_identity_and_assets_are_strict(self):
        def wrong_tag(release): release["tag_name"] = "v9.9.9"
        def draft(release): release["draft"] = True
        def missing(release): release["assets"].pop()
        def duplicate(release): release["assets"].append(release["assets"][0].copy())
        def unready(release): release["assets"][0]["state"] = "starter"
        def no_digest(release): release["assets"][0].pop("digest")
        def wrong_digest_format(release): release["assets"][0]["digest"] = "md5:" + "0" * 32
        def wrong_host(release): release["assets"][0]["browser_download_url"] = "https://example.org/payload.zip"
        def latest(release): release["assets"][0]["browser_download_url"] = release["assets"][0]["browser_download_url"].replace("/v1.0.1/", "/latest/")
        for change, message in (
            (wrong_tag, "requested published version"), (draft, "requested published version"),
            (missing, "missing or ambiguous"), (duplicate, "missing or ambiguous"),
            (unready, "incomplete"), (no_digest, "valid SHA-256"),
            (wrong_digest_format, "valid SHA-256"), (wrong_host, "fixed GitHub asset"),
            (latest, "fixed GitHub asset"),
        ):
            with self.subTest(change=change.__name__):
                fixture = self.fixture()
                fixture.publish(release_change=change)
                self.assert_stopped_unchanged(fixture, message)
                self.assertEqual(len(fixture.records("transport.jsonl")), 1)

    def test_manifest_download_digest_is_checked_before_parsing(self):
        fixture = self.fixture()
        (fixture.root / "manifest.json").write_bytes(b"not JSON")
        self.assert_stopped_unchanged(fixture, "manifest failed its SHA-256")
        self.assertEqual(len(fixture.records("transport.jsonl")), 2)

    def test_manifest_identity_and_file_metadata_are_strict(self):
        cases = (
            (lambda m: m.update(schema_version=2), "does not describe"),
            (lambda m: m.update(version="9.9.9"), "does not describe"),
            (lambda m: m.update(package_name="another package"), "does not describe"),
            (lambda m: m["files"].pop("Analysis Package/VERSION"), "omits a required"),
            (lambda m: m["files"].pop("Run Analysis.command"), "omits a required"),
            (lambda m: m["files"]["Analysis Package/VERSION"].update(sha256="broken"), "Invalid program file metadata"),
            (lambda m: m["files"]["Analysis Package/VERSION"].update(size=-1), "Invalid program file metadata"),
            (lambda m: m["files"]["Analysis Package/VERSION"].update(mode=0o4755), "Invalid program file metadata"),
        )
        for index, (change, message) in enumerate(cases):
            with self.subTest(case=index):
                fixture = self.fixture()
                fixture.publish(manifest_change=change)
                self.assert_stopped_unchanged(fixture, message)
                self.assertEqual(len(fixture.records("transport.jsonl")), 2)

    def test_manifest_cannot_address_user_data_or_escape_package(self):
        for relative in ("Sample Image/example.tif", "Result/out.csv", "Analysis Package/Run State/cache", "Analysis Package/Run State/locks/workspace.lock", "../outside", "/tmp/outside", "Analysis Package/project_leap_2d/../outside", "Analysis Package/Run State", "Analysis Package/project_leap_2d\\outside", "Analysis Package/user-data.txt"):
            with self.subTest(relative=relative):
                fixture = self.fixture()
                fixture.publish(manifest_change=lambda m: m["files"].update({relative: {"sha256": "0" * 64, "size": 0, "mode": 0o644}}))
                self.assert_stopped_unchanged(fixture, "Unsafe or conflicting program path")
                self.assertEqual(len(fixture.records("transport.jsonl")), 2)

    def test_zip_digest_is_checked_before_unpacking(self):
        fixture = self.fixture()
        (fixture.project / "Analysis Package/VERSION").write_text("damaged\n")
        (fixture.root / "release.zip").write_bytes(b"not a ZIP")
        self.assert_stopped_unchanged(fixture, "ZIP failed its SHA-256")
        requests = fixture.records("transport.jsonl")
        self.assertEqual(len(requests), 3)
        self.assertTrue(requests[-1]["url"].endswith("/" + ZIP_NAME))

    def test_terminating_only_entry_pid_releases_download_and_locks(self):
        fixture = self.fixture()
        write_file(fixture.transport, f"#!{sys.executable}\n" + "import os, pathlib, time\nroot=pathlib.Path(os.environ['FIXTURE_ROOT'])\n(root/'download-pid').write_text(str(os.getpid()))\ntime.sleep(30)\n", 0o755)
        running = fixture.start()
        process = running[0]
        wait_for(fixture.root / "download-pid", process)
        child_pid = int((fixture.root / "download-pid").read_text())
        process.send_signal(signal.SIGTERM)
        completed = fixture.finish(running, timeout=5)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Repair interrupted", completed.stderr)
        with self.assertRaises(ProcessLookupError): os.kill(child_pid, 0)
        write_file(fixture.transport, "#!/bin/sh\nexit 22\n", 0o755)
        retry = fixture.run()
        self.assertIn("may not yet be published", retry.stderr)
        self.assertNotIn("Another analysis", retry.stderr)
        self.assertEqual(list((fixture.root / "Temp").iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
