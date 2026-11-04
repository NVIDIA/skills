#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
VERIFY = SCRIPT_DIR / "verify_vlm_captioning_base_url.py"
PREPARE = SCRIPT_DIR / "prepare_paidf_config.py"
HANDOFF = SCRIPT_DIR / "write_paidf_handoff.py"


class VlmCaptioningModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            body = b'{"object":"list","data":[]}\n'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


VLM_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), VlmCaptioningModelsHandler)
VLM_THREAD = threading.Thread(target=VLM_SERVER.serve_forever, daemon=True)
VLM_THREAD.start()
VLM_ENDPOINT = f"http://127.0.0.1:{VLM_SERVER.server_port}/v1"


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def run_script(script: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=check,
        text=True,
        capture_output=True,
    )


def write_prompt(path: Path) -> None:
    path.write_text("Describe the video in one paragraph.\n", encoding="utf-8")


def prepare(input_jsonl: Path, output_dir: Path, media_dir: Path, *extra_args: str) -> None:
    prompt_file = output_dir.parent / "caption_prompt.txt"
    write_prompt(prompt_file)
    run_script(
        PREPARE,
        "--input-jsonl",
        str(input_jsonl),
        "--output-dir",
        str(output_dir),
        "--vlm-captioning-endpoint",
        VLM_ENDPOINT,
        "--media-dir",
        str(media_dir),
        "--paidf-num-gpus",
        "4",
        "--caption-prompt-file",
        str(prompt_file),
        *extra_args,
    )


class PaidfCosmosPredictTest(unittest.TestCase):
    def test_verify_vlm_captioning_base_url(self) -> None:
        result = run_script(
            VERIFY,
            "--vlm-captioning-endpoint",
            VLM_ENDPOINT,
        )
        self.assertIn("VLM captioning base URL passed /models preflight probe", result.stdout)

    def test_verify_vlm_captioning_base_url_fails(self) -> None:
        result = run_script(
            VERIFY,
            "--vlm-captioning-endpoint",
            "http://127.0.0.1:9/v1",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VLM captioning base URL did not pass the /models preflight probe", result.stderr)

    def test_prepare_is_deterministic_and_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            input_jsonl = root / "media.jsonl"
            output_dir = root / "paidf"
            media_a = media_dir / "a.mp4"
            media_b = media_dir / "b.mp4"
            write_jsonl(
                input_jsonl,
                [
                    {"id": "sample-a", "media_path": str(media_a)},
                    {"id": "sample-b", "media_path": str(media_b)},
                ],
            )

            prepare(input_jsonl, output_dir, media_dir)
            first_config = (output_dir / "config.yaml").read_text(encoding="utf-8")
            first_map = (output_dir / "path_map.jsonl").read_text(encoding="utf-8")

            prepare(input_jsonl, output_dir, media_dir)
            self.assertEqual(first_config, (output_dir / "config.yaml").read_text())
            self.assertEqual(first_map, (output_dir / "path_map.jsonl").read_text())
            self.assertIn(f'url: "{VLM_ENDPOINT}"', first_config)
            self.assertIn('model: "Qwen/Qwen3-VL-235B-A22B-Instruct"', first_config)
            self.assertIn(f'rgb: "{media_a}"', first_config)
            self.assertIn("num_processes: 4", first_config)
            self.assertIn("Describe the video in one paragraph.", first_config)

    def test_prepare_uses_required_paidf_num_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            input_jsonl = root / "media.jsonl"
            output_dir = root / "paidf"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])

            prepare(input_jsonl, output_dir, media_dir, "--paidf-num-gpus", "2")

            self.assertIn("num_processes: 2", (output_dir / "config.yaml").read_text())

    def test_handoff_writes_one_row_per_input_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            input_jsonl = root / "media.jsonl"
            output_dir = root / "paidf"
            generated_jsonl = output_dir / "generated_videos.jsonl"
            failed_jsonl = output_dir / "failed_videos.jsonl"
            media_path = media_dir / "a.mp4"
            write_jsonl(
                input_jsonl,
                [
                    {"id": "sample-a", "media_path": f"{media_dir}/../media/a.mp4"},
                    {"id": "sample-b", "media_path": str(media_path)},
                ],
            )

            prepare(input_jsonl, output_dir, media_dir)
            path_map_lines = (output_dir / "path_map.jsonl").read_text().splitlines()
            self.assertEqual(len(path_map_lines), 1)
            mapping = json.loads(path_map_lines[0])
            Path(mapping["host_generated_video_path"]).touch()

            run_script(
                HANDOFF,
                "--input-jsonl",
                str(input_jsonl),
                "--path-map",
                str(output_dir / "path_map.jsonl"),
                "--generated-jsonl",
                str(generated_jsonl),
                "--failed-jsonl",
                str(failed_jsonl),
            )
            rows = [json.loads(line) for line in generated_jsonl.read_text().splitlines()]
            self.assertEqual(
                set(rows[0]),
                {"id", "original_media_path", "generated_video_path"},
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["generated_video_path"], rows[1]["generated_video_path"])
            self.assertEqual(rows[0]["original_media_path"], str(media_path))
            self.assertEqual(failed_jsonl.read_text(), "")

    def test_handoff_writes_failed_rows_when_generated_video_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            input_jsonl = root / "media.jsonl"
            output_dir = root / "paidf"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])
            prepare(input_jsonl, output_dir, media_dir)

            run_script(
                HANDOFF,
                "--input-jsonl",
                str(input_jsonl),
                "--path-map",
                str(output_dir / "path_map.jsonl"),
                "--generated-jsonl",
                str(output_dir / "generated_videos.jsonl"),
                "--failed-jsonl",
                str(output_dir / "failed_videos.jsonl"),
            )
            self.assertEqual((output_dir / "generated_videos.jsonl").read_text(), "")
            rows = [
                json.loads(line)
                for line in (output_dir / "failed_videos.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                set(rows[0]),
                {"id", "original_media_path", "expected_generated_video_path", "error"},
            )
            self.assertEqual(rows[0]["id"], "sample-a")
            self.assertEqual(rows[0]["error"], "missing_generated_video")

    def test_prepare_rejects_media_path_outside_media_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_root = root / "media"
            media_root.mkdir()
            prompt_file = root / "caption_prompt.txt"
            write_prompt(prompt_file)
            input_jsonl = root / "media.jsonl"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": "/other/a.mp4"}])

            result = run_script(
                PREPARE,
                "--input-jsonl",
                str(input_jsonl),
                "--output-dir",
                str(root / "paidf"),
                "--vlm-captioning-endpoint",
                VLM_ENDPOINT,
                "--media-dir",
                str(media_root),
                "--paidf-num-gpus",
                "4",
                "--caption-prompt-file",
                str(prompt_file),
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("is not under --media-dir", result.stderr)

    def test_prepare_rejects_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            prompt_file = root / "caption_prompt.txt"
            write_prompt(prompt_file)
            input_jsonl = root / "media.jsonl"
            write_jsonl(input_jsonl, [{"id": "sample-a"}])

            result = run_script(
                PREPARE,
                "--input-jsonl",
                str(input_jsonl),
                "--output-dir",
                str(root / "paidf"),
                "--vlm-captioning-endpoint",
                VLM_ENDPOINT,
                "--media-dir",
                str(media_dir),
                "--paidf-num-gpus",
                "4",
                "--caption-prompt-file",
                str(prompt_file),
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required field 'media_path'", result.stderr)

    def test_prepare_requires_media_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_file = root / "caption_prompt.txt"
            write_prompt(prompt_file)
            input_jsonl = root / "media.jsonl"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(root / "media" / "a.mp4")}])

            result = run_script(
                PREPARE,
                "--input-jsonl",
                str(input_jsonl),
                "--output-dir",
                str(root / "paidf"),
                "--vlm-captioning-endpoint",
                VLM_ENDPOINT,
                "--paidf-num-gpus",
                "4",
                "--caption-prompt-file",
                str(prompt_file),
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--media-dir", result.stderr)

    def test_prepare_requires_paidf_num_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            prompt_file = root / "caption_prompt.txt"
            write_prompt(prompt_file)
            input_jsonl = root / "media.jsonl"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])

            result = run_script(
                PREPARE,
                "--input-jsonl",
                str(input_jsonl),
                "--output-dir",
                str(root / "paidf"),
                "--vlm-captioning-endpoint",
                VLM_ENDPOINT,
                "--media-dir",
                str(media_dir),
                "--caption-prompt-file",
                str(prompt_file),
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--paidf-num-gpus", result.stderr)

    def test_prepare_rejects_non_positive_paidf_num_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            prompt_file = root / "caption_prompt.txt"
            write_prompt(prompt_file)
            input_jsonl = root / "media.jsonl"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])

            result = run_script(
                PREPARE,
                "--input-jsonl",
                str(input_jsonl),
                "--output-dir",
                str(root / "paidf"),
                "--vlm-captioning-endpoint",
                VLM_ENDPOINT,
                "--media-dir",
                str(media_dir),
                "--paidf-num-gpus",
                "0",
                "--caption-prompt-file",
                str(prompt_file),
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--paidf-num-gpus must be >= 1", result.stderr)

    def test_prepare_requires_caption_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            input_jsonl = root / "media.jsonl"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])

            result = run_script(
                PREPARE,
                "--input-jsonl",
                str(input_jsonl),
                "--output-dir",
                str(root / "paidf"),
                "--vlm-captioning-endpoint",
                VLM_ENDPOINT,
                "--media-dir",
                str(media_dir),
                "--paidf-num-gpus",
                "4",
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--caption-prompt-file", result.stderr)

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "root can write through read-only file mode bits",
    )
    def test_prepare_fails_early_for_unwritable_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            prompt_file = root / "caption_prompt.txt"
            write_prompt(prompt_file)
            input_jsonl = root / "media.jsonl"
            output_dir = root / "paidf"
            config_path = output_dir / "config.yaml"
            output_dir.mkdir()
            config_path.write_text("existing\n", encoding="utf-8")
            config_path.chmod(0o400)
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])

            try:
                result = run_script(
                    PREPARE,
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-dir",
                    str(output_dir),
                    "--vlm-captioning-endpoint",
                    VLM_ENDPOINT,
                    "--media-dir",
                    str(media_dir),
                    "--paidf-num-gpus",
                    "4",
                    "--caption-prompt-file",
                    str(prompt_file),
                    check=False,
                )
            finally:
                config_path.chmod(0o600)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PAIDF config file is not writable", result.stderr)

    def test_prepare_makes_output_dirs_container_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / "media"
            media_dir.mkdir()
            input_jsonl = root / "media.jsonl"
            output_dir = root / "paidf"
            write_jsonl(input_jsonl, [{"id": "sample-a", "media_path": str(media_dir / "a.mp4")}])

            prepare(input_jsonl, output_dir, media_dir)

            for relative_path in (
                "generated/videos",
                "generated/metadata",
                "captions",
            ):
                mode = (output_dir / relative_path).stat().st_mode
                self.assertTrue(
                    mode & stat.S_IWOTH,
                    f"{relative_path} should be writable by container users",
                )


if __name__ == "__main__":
    unittest.main()
