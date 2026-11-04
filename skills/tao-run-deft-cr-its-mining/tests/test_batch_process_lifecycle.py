#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real CLI/torchrun-like subprocess trees, not mocked process calls."""

import json
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from batch_cosmos_embed_text import Cancellation, prepare  # noqa: E402
from native_process import run_child, send_signal, stop_and_reap  # noqa: E402
from validate_cosmos_embed_output import check_completion, validate_completion  # noqa: E402


FAKE_CLI = r'''
import json, os, pathlib, signal, subprocess, sys, time
import numpy as np
import yaml

def record_pid(root, name):
    pid = os.getpid()
    start = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    (root / f"{name}.pid").write_text(str(pid))
    (root / f"{name}.start").write_text(start)

if len(sys.argv) > 1 and sys.argv[1] == "leaf":
    root = pathlib.Path(sys.argv[2])
    record_pid(root, "leaf")
    if os.environ.get("FAKE_MODE") == "ignore":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "leaf.ready").touch()
    while True:
        time.sleep(0.05)

if len(sys.argv) > 1 and sys.argv[1] == "worker":
    root = pathlib.Path(sys.argv[2])
    record_pid(root, "worker")
    def stop(sig, _):
        print("SHUTDOWN_DIAGNOSTIC:" + "\x1e" * 200000, flush=True)
        if os.environ.get("FAKE_MODE") != "ignore":
            sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    subprocess.Popen([sys.executable, __file__, "leaf", str(root)], start_new_session=True)
    while not (root / "leaf.ready").exists():
        time.sleep(0.01)
    print("LIVE_WITHOUT_NEWLINE", end="", flush=True)
    (root / "ready").touch()
    while True:
        time.sleep(0.05)

spec = yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text())
root = pathlib.Path(sys.argv[-1]).parent
number = int(root.name.removeprefix("batch_"))
out = pathlib.Path(spec["results_dir"]) / "inference"
out.mkdir(parents=True)
texts = spec["inference"]["query"]["input_texts"]
np.save(out / "text_embeddings.npy", np.ones((len(texts), 3)))
(out / "text_embeddings.json").write_text(json.dumps({
    "checkpoint": spec["inference"]["checkpoint"], "npy_file": "text_embeddings.npy",
    "results": [{"text": text, "npy_row": i} for i, text in enumerate(texts)],
}))
mode = os.environ.get("FAKE_MODE", "success")
if number == 2 and mode == "fail":
    sys.exit(7)
if number == 2 and mode in {"block", "ignore", "orphan"}:
    record_pid(root, "cli")
    worker = subprocess.Popen([sys.executable, __file__, "worker", str(root)], start_new_session=True)
    if mode == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if mode == "orphan":
        while not (root / "ready").exists():
            time.sleep(0.01)
        sys.exit(130)
    worker.wait()
sys.exit(130)
'''


@unittest.skipUnless(sys.platform == "linux", "native container worker ownership requires Linux")
class BatchProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        cli = self.bin / "cosmos-embed1"
        cli.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
        cli.chmod(0o755)
        self.spec = self.make_spec("first")

    def make_spec(self, name):
        spec = self.root / f"{name}.yaml"
        spec.write_text(yaml.safe_dump({
            "results_dir": str(self.root / name),
            "inference": {"checkpoint": "/selected/model", "mode": "text", "num_gpus": 1,
                          "query": {"input_texts": ["duplicate", "b", "duplicate", "c", "tail"]}},
        }))
        prepare(spec, 2)
        return spec

    def launch(self, mode, spec=None, blocked_logs=False):
        spec = spec or self.spec
        log = self.root / f"{spec.stem}-{mode}.log"
        with log.open("wb") as output:
            child = subprocess.Popen(
                [sys.executable, str(SCRIPTS / "batch_cosmos_embed_text.py"),
                 "run", "--inference-spec", str(spec)],
                env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_MODE": mode},
                stdout=subprocess.PIPE if blocked_logs else output,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        def cleanup():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            # A failing test must not leak even a deliberately detached fixture.
            for path in (self.root / spec.stem).glob("batches/*/*.pid"):
                try:
                    send_signal(int(path.read_text()), path.with_suffix(".start").read_text(), signal.SIGKILL)
                except (ProcessLookupError, FileNotFoundError):
                    pass
            if child.stdout is not None:
                child.stdout.close()
        self.addCleanup(cleanup)
        return child, log

    def wait_for(self, predicate, child, log):
        deadline = time.monotonic() + 20
        while not predicate():
            if child.poll() is not None or time.monotonic() > deadline:
                self.fail(log.read_text()[-6000:])
            time.sleep(0.02)

    def assert_reaped(self):
        for path in (self.root / "first").glob("batches/*/*.pid"):
            self.assertFalse(Path(f"/proc/{path.read_text()}").exists(), str(path))

    def cancel_case(self, sig, mode="block"):
        child, log = self.launch(mode)
        batch = self.root / "first/batches/batch_002"
        self.wait_for(lambda: (batch / "ready").exists()
                      and "LIVE_WITHOUT_NEWLINE" in log.read_text(), child, log)
        self.assertIn("LIVE_WITHOUT_NEWLINE", (batch / "container-child.log").read_text())
        self.assertEqual(len(json.loads((self.root / "first/batch-progress.json").read_text())), 1)
        os.killpg(child.pid, sig)
        if mode == "ignore":
            time.sleep(0.1)
            os.killpg(child.pid, signal.SIGINT)  # A second signal must not bypass cleanup.
        self.assertEqual(child.wait(timeout=15), 128 + sig, log.read_text()[-3000:])
        self.assert_reaped()
        self.assertFalse((self.root / "first/inference/completion_validation.json").exists())
        self.assertFalse((self.root / "first/batches/batch_003/container-child.log").exists())
        self.assertEqual(json.loads((self.root / "first/batch-cancellation.json").read_text())["signal"], sig)
        for code in (0, 130):
            with self.assertRaisesRegex(ValueError, "canceled"):
                validate_completion(self.spec, code)
        for content in (log.read_text(), (batch / "container-child.log").read_text()):
            self.assertIn("SHUTDOWN_DIAGNOSTIC:", content)
            # Other workers may interleave traceback bytes in the shared pipe.
            self.assertEqual(content.count("\x1e"), 200000)
        return child, log

    def test_sigterm_cleans_detached_worker_and_preserves_shutdown_output(self):
        self.cancel_case(signal.SIGTERM)

    def test_sigint_is_cancellation_not_native_teardown_success(self):
        self.cancel_case(signal.SIGINT)

    def test_stalled_platform_logs_do_not_block_cancellation(self):
        child, log = self.launch("block", blocked_logs=True)
        batch = self.root / "first/batches/batch_002"
        self.wait_for(lambda: (batch / "ready").exists(), child, log)
        os.killpg(child.pid, signal.SIGTERM)
        self.assertEqual(child.wait(timeout=12), 143)
        self.assert_reaped()
        content = (batch / "container-child.log").read_text()
        self.assertIn("SHUTDOWN_DIAGNOSTIC:", content)
        self.assertEqual(content.count("\x1e"), 200000)
        self.assertIn("live bytes omitted", content)

    def test_unsupported_pidfds_fail_before_launch(self):
        with patch("native_process.open_pidfd", side_effect=OSError("fixture seccomp denial")):
            with patch("native_process.subprocess.Popen") as popen:
                with self.assertRaisesRegex(OSError, "fixture seccomp denial"):
                    run_child([sys.executable, "-c", "pass"], stdout=io.BytesIO(),
                              stderr=subprocess.STDOUT, start_new_session=True,
                              cancellation=Cancellation())
                popen.assert_not_called()

    def test_missing_proc_children_support_fails_before_launch(self):
        with patch("native_process.Path.exists", return_value=False):
            with patch("native_process.subprocess.Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "CONFIG_PROC_CHILDREN"):
                    run_child([sys.executable, "-c", "pass"], stdout=io.BytesIO(),
                              stderr=subprocess.STDOUT, start_new_session=True,
                              cancellation=Cancellation())
                popen.assert_not_called()

    def test_direct_cli_is_stopped_when_descendant_discovery_degrades(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        try:
            stop_and_reap(child, lambda: {}, lambda *_args, **_kwargs: False, signal.SIGTERM)
            self.assertEqual(child.returncode, -signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_python_without_pidfd_open_uses_real_libc_api(self):
        with patch("native_process.os.pidfd_open", None, create=True), \
                patch("native_process.signal.pidfd_send_signal", None, create=True):
            result = run_child([sys.executable, "-c", "pass"], stdout=io.BytesIO(),
                               stderr=subprocess.STDOUT, start_new_session=True,
                               cancellation=Cancellation())
        self.assertEqual(result.returncode, 0)

    def test_preexisting_child_is_not_owned_or_signaled(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
        try:
            with patch("native_process.subprocess.Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "dedicated process"):
                    run_child([sys.executable, "-c", "pass"], stdout=io.BytesIO(),
                              stderr=subprocess.STDOUT, start_new_session=True,
                              cancellation=Cancellation())
                popen.assert_not_called()
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=5)

    def test_repeated_cancellation_escalates_and_reaps_uncooperative_workers(self):
        self.cancel_case(signal.SIGTERM, "ignore")

    def test_native_exit_reaps_orphaned_session_without_container_cleanup(self):
        child, log = self.launch("orphan")
        self.assertEqual(child.wait(timeout=20), 0, log.read_text()[-3000:])
        self.assert_reaped()
        self.assertEqual(check_completion(self.spec)["observed_count"], 5)

    def test_failure_after_progress_requires_fresh_results_and_recovers(self):
        child, log = self.launch("fail")
        self.assertEqual(child.wait(timeout=20), 7, log.read_text()[-3000:])
        self.assertEqual(len(json.loads((self.root / "first/batch-progress.json").read_text())), 1)
        self.assertFalse((self.root / "first/inference").exists())
        evidence = (self.root / "first/batches/batch_001/container-child.log").read_bytes()
        retry, retry_log = self.launch("success")
        self.assertNotEqual(retry.wait(timeout=20), 0)
        self.assertIn("already attempted", retry_log.read_text())
        self.assertEqual((self.root / "first/batches/batch_001/container-child.log").read_bytes(), evidence)
        fresh = self.make_spec("recovery")
        recovered, recovered_log = self.launch("success", fresh)
        self.assertEqual(recovered.wait(timeout=20), 0, recovered_log.read_text()[-3000:])
        self.assertEqual(check_completion(fresh)["observed_count"], 5)

    def test_signal_during_popen_creation_still_reaps_child(self):
        cancellation = Cancellation()
        real_popen = subprocess.Popen
        launched = []
        def launch(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            launched.append(child)
            cancellation.request(signal.SIGTERM, None)
            return child
        with patch("native_process.subprocess.Popen", side_effect=launch):
            result = run_child([sys.executable, "-c", "import time; time.sleep(60)"],
                               stdout=io.BytesIO(), stderr=subprocess.STDOUT,
                               start_new_session=True, cancellation=cancellation)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(Path(f"/proc/{launched[0].pid}").exists())

    def test_log_failure_is_not_success_even_after_child_exits(self):
        class BrokenLog(io.BytesIO):
            def write(self, value):
                raise OSError("fixture disk full")
        real_popen = subprocess.Popen
        def launch(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            child.wait(timeout=5)  # Force the already-exited path with buffered output.
            return child
        with patch("native_process.subprocess.Popen", side_effect=launch):
            with self.assertRaisesRegex(OSError, "fixture disk full"):
                run_child([sys.executable, "-c", "print('native output')"],
                          stdout=BrokenLog(), stderr=subprocess.STDOUT,
                          start_new_session=True, cancellation=Cancellation())


if __name__ == "__main__":
    unittest.main()
