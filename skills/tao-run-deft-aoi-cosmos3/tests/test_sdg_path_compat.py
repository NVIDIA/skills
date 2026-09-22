#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import emit_sdg_sharegpt  # noqa: E402


PROMPT = "Compare the AOI image with the golden reference."
STEM = "IC+bridge_00000.png"


def _image(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path.resolve()


def _csv(
    path: pathlib.Path, fieldnames: list[str], row: dict[str, str]
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return path


class SdgPathCompatibilityTests(unittest.TestCase):
    def test_paidf_101_repo_relative_output_and_image_filename_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = pathlib.Path(temporary) / "paidf-anomalygen"
            relative_output = pathlib.Path("results/nvpcb_qa/original")
            sdg_dir = repo_root / relative_output
            generated = _image(sdg_dir / "reconstructed_image" / STEM)
            source = _image(
                repo_root / "data/nvpcb/IC/clean_image/source_board.jpg"
            )
            sdg_csv = _csv(
                sdg_dir / "SDG_result.csv",
                ["output_filename", "image_filename", "guardrail_pass"],
                {
                    "output_filename": str(
                        relative_output / "reconstructed_image" / STEM
                    ),
                    "image_filename": str(source),
                    "guardrail_pass": "1",
                },
            )

            records, _ = emit_sdg_sharegpt.emit_records(
                sdg_csv,
                media_root=repo_root,
                prompt=PROMPT,
                relative=False,
            )

            self.assertEqual(records[0]["images"], [str(generated), str(source)])

    def test_documented_output_relative_pair_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sdg_dir = pathlib.Path(temporary) / "sdg"
            generated = _image(sdg_dir / "reconstructed_image" / STEM)
            clean = _image(sdg_dir / "original_image" / STEM)
            sdg_csv = _csv(
                sdg_dir / "SDG_result.csv",
                ["reconstructed_image", "original_image"],
                {
                    "reconstructed_image": f"reconstructed_image/{STEM}",
                    "original_image": f"original_image/{STEM}",
                },
            )

            records, _ = emit_sdg_sharegpt.emit_records(
                sdg_csv,
                media_root=sdg_dir,
                prompt=PROMPT,
                relative=False,
            )

            self.assertEqual(records[0]["images"], [str(generated), str(clean)])

    def test_missing_101_source_uses_resolved_pair_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = pathlib.Path(temporary) / "paidf-anomalygen"
            relative_output = pathlib.Path("results/nvpcb_qa/original")
            sdg_dir = repo_root / relative_output
            generated = _image(sdg_dir / "reconstructed_image" / STEM)
            clean = _image(sdg_dir / "original_image" / STEM)
            sdg_csv = _csv(
                sdg_dir / "SDG_result.csv",
                ["output_filename", "image_filename"],
                {
                    "output_filename": str(
                        relative_output / "reconstructed_image" / STEM
                    ),
                    "image_filename": "/container/data/missing-source.jpg",
                },
            )

            records, _ = emit_sdg_sharegpt.emit_records(
                sdg_csv,
                media_root=repo_root,
                prompt=PROMPT,
                relative=False,
            )

            self.assertEqual(records[0]["images"], [str(generated), str(clean)])

    def test_explicit_sdg_root_resolves_a_moved_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo_root = root / "paidf-anomalygen"
            relative_output = pathlib.Path("results/nvpcb_qa/original")
            generated = _image(
                repo_root / relative_output / "reconstructed_image" / STEM
            )
            source = _image(repo_root / "data/clean/source.jpg")
            sdg_csv = _csv(
                root / "staged/SDG_result.csv",
                ["output_filename", "image_filename"],
                {
                    "output_filename": str(
                        relative_output / "reconstructed_image" / STEM
                    ),
                    "image_filename": str(source),
                },
            )

            records, summary = emit_sdg_sharegpt.emit_records(
                sdg_csv,
                media_root=repo_root,
                prompt=PROMPT,
                relative=False,
                sdg_root=repo_root.resolve(),
            )

            self.assertEqual(records[0]["images"][0], str(generated))
            self.assertEqual(summary["sdg_root"], str(repo_root.resolve()))

    def test_failure_lists_every_attempted_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo_root = root / "paidf-anomalygen"
            relative_output = pathlib.Path("results/nvpcb_qa/original")
            sdg_dir = repo_root / relative_output
            override = root / "override"
            raw = relative_output / "reconstructed_image" / STEM
            sdg_csv = _csv(
                sdg_dir / "SDG_result.csv",
                ["output_filename", "image_filename"],
                {
                    "output_filename": str(raw),
                    "image_filename": "/container/data/missing-source.jpg",
                },
            )

            with self.assertRaisesRegex(ValueError, "attempted paths:") as raised:
                emit_sdg_sharegpt.emit_records(
                    sdg_csv,
                    media_root=repo_root,
                    prompt=PROMPT,
                    relative=False,
                    sdg_root=override.resolve(),
                )

            message = str(raised.exception)
            expected = (
                (sdg_dir / raw).resolve(),
                (repo_root / raw).resolve(),
                (override / raw).resolve(),
            )
            for attempted in expected:
                self.assertIn(str(attempted), message)

    def test_missing_clean_lists_source_and_pair_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sdg_dir = pathlib.Path(temporary) / "sdg"
            _image(sdg_dir / "reconstructed_image" / STEM)
            source = pathlib.Path(temporary) / "missing-source.jpg"
            sdg_csv = _csv(
                sdg_dir / "SDG_result.csv",
                ["output_filename", "image_filename"],
                {
                    "output_filename": f"reconstructed_image/{STEM}",
                    "image_filename": str(source),
                },
            )

            with self.assertRaisesRegex(ValueError, "attempted paths:") as raised:
                emit_sdg_sharegpt.emit_records(
                    sdg_csv,
                    media_root=sdg_dir,
                    prompt=PROMPT,
                    relative=False,
                )

            message = str(raised.exception)
            self.assertIn(str(source.resolve()), message)
            self.assertIn(
                str((sdg_dir / "original_image" / STEM).resolve()), message
            )

    def test_future_schema_failure_names_available_columns(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "available columns:.*future_output_path.*attempted paths: none",
        ):
            emit_sdg_sharegpt._resolve_pair(
                {"future_output_path": "new/location.png"},
                sdg_dir=pathlib.Path("/tmp/sdg"),
                row_number=2,
            )


if __name__ == "__main__":
    unittest.main()
