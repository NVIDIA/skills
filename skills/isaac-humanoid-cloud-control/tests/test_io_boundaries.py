# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for HTTP boundaries, ambiguous submissions, and inventory files."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import http_transport
import submit_navigation_mission as mission


class HTTPBoundary(unittest.TestCase):
    def test_unsupported_urls_rejected_before_connection(self):
        with patch.object(http_transport.http.client, "HTTPConnection") as connect:
            for url in ("file:///etc/hosts", "ftp://example.org/data",
                        "http://user:pass@example.org/", "http://example.org/#fragment"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    with http_transport.open_http(url, timeout=1):
                        self.fail("invalid URL accepted")
            connect.assert_not_called()

    def test_redirect_rejected_and_error_body_preserved(self):
        with patch.object(http_transport.http.client, "HTTPConnection") as connect:
            response = connect.return_value.getresponse.return_value
            response.status = 302
            response.reason = "Found"
            response.headers = {"Location": "file:///etc/hosts"}
            response.read.return_value = b"redirect rejected"
            with self.assertRaises(urllib.error.HTTPError) as error:
                with http_transport.open_http("http://localhost:8050/route", timeout=2):
                    self.fail("redirect accepted")
            self.assertEqual(error.exception.read(), b"redirect rejected")
            connect.return_value.request.assert_called_once_with("GET", "/route", body=None, headers={})
            connect.return_value.close.assert_called_once()

    def test_https_uses_tls_and_preserves_post(self):
        with patch.object(http_transport.http.client, "HTTPSConnection") as connect:
            connect.return_value.getresponse.return_value.status = 200
            request = http_transport.urllib.request.Request(
                "https://example.org:8443/mission?id=one", data=b"{}", method="POST")
            with http_transport.open_http(request, timeout=2):
                pass
            connect.assert_called_once_with("example.org", port=8443, timeout=2)
            connect.return_value.request.assert_called_once_with("POST", "/mission?id=one", body=b"{}", headers={})
            connect.return_value.close.assert_called_once()


class SubmissionBoundary(unittest.TestCase):
    def test_ambiguous_post_queries_record_without_resubmitting(self):
        calls = []
        @contextlib.contextmanager
        def transport(target, *, timeout):
            calls.append(target)
            if isinstance(target, mission.urllib.request.Request):
                raise TimeoutError("solver still running")
            if "/robot/" in target:
                value = {"name": "g1_0", "lifecycle": "ALIVE", "status": {
                    "online": True, "state": "IDLE", "position_initialized": True}}
            else:
                value = {"name": "test-mission", "status": {"state": "RUNNING"}}
            yield io.BytesIO(json.dumps(value).encode())
        argv = ["submit", "--host", "localhost", "--x", "1", "--y", "2",
                "--mission-id", "test-mission", "--validated-free-space", "--submit"]
        with patch.object(sys, "argv", argv), patch.object(mission, "open_http", transport), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mission.main(), 0)
        self.assertEqual(sum(isinstance(c, mission.urllib.request.Request) for c in calls), 1)
        self.assertTrue(calls[-1].endswith("/mission/test-mission"))

    def test_invalid_host_rejected_before_network(self):
        with patch.object(mission, "open_http") as transport:
            for host in ("user@example.org", "example.org?query", "example.org#fragment"):
                with patch.object(sys, "argv", ["submit", "--host", host, "--x", "1", "--y", "2"]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    mission.main()
            transport.assert_not_called()


class InventoryBoundary(unittest.TestCase):
    def run_inventory(self, **values):
        omni = MagicMock()
        stage = omni.usd.get_context.return_value.get_stage.return_value
        stage.Traverse.return_value = []
        stage.GetRootLayer.return_value.identifier = "test-stage"
        pxr = MagicMock()
        pxr.UsdGeom.GetStageMetersPerUnit.return_value = 1.0
        with patch.dict(sys.modules, {"omni": omni, "omni.usd": omni.usd, "pxr": pxr}), contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(ROOT / "scripts/scene_inventory.py"), init_globals=values)

    def test_default_creates_unique_private_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tempfile, "tempdir", directory):
            self.run_inventory()
            self.run_inventory()
            files = list(Path(directory).glob("humanoid-inventory-*.json"))
            self.assertEqual(len(files), 2)
            for path in files:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(json.loads(path.read_text())["stage_identifier"], "test-stage")

    def test_explicit_symlink_does_not_overwrite_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "keep.json"
            target.write_text("keep")
            link = Path(directory) / "inventory.json"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                self.run_inventory(output_path=str(link))
            self.assertEqual(target.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
