#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _docker_blocks(path: pathlib.Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        "\n".join(lines[index : index + 6])
        for index, line in enumerate(lines)
        if line.lstrip().startswith("docker run")
    ]


class ApplicationDockerIdentityTests(unittest.TestCase):
    def _assert_mapped_identity(self, block: str) -> None:
        self.assertIn("--user", block)
        self.assertIn("$(id -u):$(id -g)", block)
        self.assertIn("USER=", block)
        self.assertIn("LOGNAME=", block)
        self.assertIn("HOME=/tmp", block)
        self.assertIn("/etc/passwd:/etc/passwd:ro", block)
        self.assertIn("/etc/group:/etc/group:ro", block)

    def test_all_anomalygen_launches_map_the_host_identity(self) -> None:
        reference = SKILL_ROOT / "references/tao-generate-anomalies.md"
        launch_blocks = _docker_blocks(reference)
        self.assertEqual(len(launch_blocks), 5)
        for block in launch_blocks:
            self._assert_mapped_identity(block)
        text = reference.read_text(encoding="utf-8")
        self.assertIn(".tao-write-probe", text)
        self.assertIn("download_anomalygen_checkpoints", text)

    def test_consumer_inference_maps_identity_after_write_probe(self) -> None:
        reference = SKILL_ROOT / "references/prepare-for-inference.md"
        launch_blocks = _docker_blocks(reference)
        self.assertEqual(len(launch_blocks), 1)
        self._assert_mapped_identity(launch_blocks[0])
        text = reference.read_text(encoding="utf-8")
        self.assertIn(".tao-write-probe", text)
        self.assertIn("owned by the submitting host user", text)

    def test_visual_changenet_overlay_maps_every_action(self) -> None:
        reference = (SKILL_ROOT / "references/visual-changenet.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("VCN_IDENTITY_ARGS", reference)
        self.assertIn('--user "$(id -u):$(id -g)"', reference)
        self.assertIn('LOGNAME="$(id -un)"', reference)
        self.assertIn("/etc/passwd:/etc/passwd:ro", reference)
        self.assertIn("train, evaluate, inference, export, and quantize", reference)
        self.assertIn("including resumed actions", reference)

    def test_direct_fallback_carries_runtime_identity(self) -> None:
        reference = (SKILL_ROOT / "references/scripts-and-agents.md").read_text(
            encoding="utf-8"
        )
        self.assertIn('docker run --user "$(id -u):$(id -g)"', reference)
        self.assertIn('USER="$(id -un)"', reference)
        self.assertIn('LOGNAME="$(id -un)"', reference)
        self.assertIn("HOME=/tmp", reference)

    def test_cosmos3_maps_or_delegates_every_writable_launch(self) -> None:
        applications = SKILL_ROOT.parent
        cosmos3_root = applications / "tao-run-deft-aoi-cosmos3"
        direct_launches = []
        for reference in sorted((cosmos3_root / "references").glob("*.md")):
            direct_launches.extend(
                (reference.relative_to(cosmos3_root), block)
                for block in _docker_blocks(reference)
            )
        for reference, block in direct_launches:
            with self.subTest(reference=reference):
                self._assert_mapped_identity(block)

        cosmos_reason = (cosmos3_root / "references/cosmos-reason.md").read_text(
            encoding="utf-8"
        )
        cosmos_mining = (
            cosmos3_root / "references/tao-mine-aoi-images.md"
        ).read_text(encoding="utf-8")
        cosmos_anomalygen = (
            cosmos3_root / "references/tao-generate-anomalies.md"
        ).read_text(encoding="utf-8")

        self.assertEqual(direct_launches, [])
        self.assertIn("CR3_IDENTITY_ARGS", cosmos_reason)
        self.assertIn('--user "$(id -u):$(id -g)"', cosmos_reason)
        self.assertIn('LOGNAME="$(id -un)"', cosmos_reason)
        self.assertIn("/etc/passwd:/etc/passwd:ro", cosmos_reason)
        self.assertIn("Train, Proxy evaluate", cosmos_reason)
        self.assertIn("including resumed actions", cosmos_reason)
        self.assertIn(".tao-write-probe", cosmos_reason)
        self.assertIn("--user $(id -u):$(id -g)", cosmos_mining)
        self.assertIn('LOGNAME="$(id -un)"', cosmos_mining)
        self.assertIn("/etc/passwd:/etc/passwd:ro", cosmos_mining)
        self.assertIn("never launch mining as root", cosmos_mining.lower())
        self.assertIn(
            "skills/applications/tao-run-deft-aoi/references/"
            "tao-generate-anomalies.md",
            cosmos_anomalygen,
        )
        self.assertIn("Do not duplicate them here", cosmos_anomalygen)

    def test_other_deft_variants_already_map_writable_launches(self) -> None:
        applications = SKILL_ROOT.parent
        pas_runner = (
            applications / "tao-run-deft-pas/scripts/run_deft_container.py"
        ).read_text(encoding="utf-8")
        object_detection = (
            applications
            / "tao-run-deft-object-detection/references/pipeline-and-state.md"
        ).read_text(encoding="utf-8")

        self.assertIn('"--user",', pas_runner)
        self.assertIn('f"{os.getuid()}:{os.getgid()}"', pas_runner)
        self.assertIn(
            'Every TAO invocation in this skill passes `--user "$(id -u):$(id -g)"`',
            object_detection,
        )


if __name__ == "__main__":
    unittest.main()
