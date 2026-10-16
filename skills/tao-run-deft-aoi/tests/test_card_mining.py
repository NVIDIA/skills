# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_deft_run
import prepare_card_mining


class CardMiningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.rd = self.workspace / "results/run_fixture"
        self.rd.mkdir(parents=True)
        self.state = {"config": {"mining_filter": {
            "metric": "cosine", "min_similarity": 0.9, "top_k_per_target": 5,
            "history_aware": {"enabled": True, "history_file": str(self.rd / "mining_history.json")},
        }}}
        (self.rd / "deft_state.json").write_text(json.dumps(self.state))

    def candidates(self, iteration, names, scores=None):
        root = self.rd / iteration / "mining_filter"
        root.mkdir(parents=True)
        images = self.workspace / "pool"
        images.mkdir(exist_ok=True)
        paths = []
        for name in names:
            image = images / (name + "_SolderLight.jpg")
            image.write_bytes(b"fixture-image")
            paths.append(str(image))
        pd.DataFrame({"filepath": paths, "max_cosine_similarity": scores or [0.99] * len(paths)}
                     ).to_parquet(root / "mined.parquet", index=False)
        return root

    def prepare(self, iteration):
        return prepare_card_mining.prepare(self.rd, self.workspace, iteration)

    def test_history_excludes_prior_selection_and_audit_accepts_artifacts(self):
        self.candidates("iter1", ["a", "b", "low"], [0.99, 0.99, 0.5])
        self.prepare("iter1")
        root = self.candidates("iter2", ["b", "c", "c"])
        summary = self.prepare("iter2")
        self.assertEqual(summary["selected_count"], 1)
        self.assertEqual(summary["already_mined_count"], 1)
        csv = pd.read_csv(root / "mining_pool.csv")
        self.assertEqual(csv["object_name"].tolist(), ["c"])
        self.assertEqual(pd.read_csv(root / "knn_summary.csv")["kept_count"].tolist(), [2])
        info = {
            "mining_candidate_parquet": str(root / "mining_candidates.parquet"),
            "mining_mined_parquet": str(root / "mined_filtered.parquet"),
            "mining_history": str(self.rd / "mining_history.json"),
            "mining_history_summary": str(root / "mining_history_summary.json"),
            "mining_mined_count": 1,
        }
        errors = []
        audit_deft_run._mining_summary_proof(root / "knn_summary.csv", 2, "summary", errors)
        audit_deft_run._mining_history_proof("iter2", info, self.state, 2, 1, errors)
        self.assertEqual(errors, [])

    def test_resume_does_not_change_history_or_candidate_hashes(self):
        root = self.candidates("iter1", ["a"])
        first = self.prepare("iter1")
        before = (self.rd / "mining_history.json").read_bytes()
        candidate_before = (root / "mining_candidates.parquet").read_bytes()
        self.assertEqual(self.prepare("iter1"), first)
        self.assertEqual((self.rd / "mining_history.json").read_bytes(), before)
        self.assertEqual((root / "mining_candidates.parquet").read_bytes(), candidate_before)

    def test_card_commit_command_passes_novel_count_and_history_evidence(self):
        self.candidates("iter1", ["a"])
        self.prepare("iter1")
        root = self.candidates("iter2", ["a", "b"])
        self.prepare("iter2")
        skill = Path(__file__).resolve().parents[1]
        commands = re.findall(r"\x60{3}bash\n(.*?)\x60{3}", (skill / "cards/50-mining.md").read_text(), re.S)
        scripts = self.workspace / "fake-skill/scripts"
        scripts.mkdir(parents=True)
        # Parse the real card command with the current commit helper's parser,
        # without committing a fabricated training run.
        (scripts / "commit_stage.py").write_text(
            f"import sys, json\nsys.path.insert(0, {str(skill / 'scripts')!r})\n"
            "import commit_stage\n"
            "print(json.dumps(vars(commit_stage._parser().parse_args()), default=str))\n"
        )
        result = subprocess.run(
            ["bash", "-c", commands[-1]], capture_output=True, text=True, timeout=20,
            env={"PATH": os.environ["PATH"], "DPY": sys.executable,
                 "SKILL_ROOT": str(scripts.parent), "RD": str(self.rd),
                 "ITER": "iter2", "STAGE_T0": "0"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args["mining_count"], 1)  # two pre-history candidates, one novel image
        self.assertEqual(args["mining_candidates"], str(root / "mining_candidates.parquet"))
        self.assertEqual(args["mining_history"], str(self.rd / "mining_history.json"))
        self.assertEqual(args["mining_history_summary"], str(root / "mining_history_summary.json"))

    def test_empty_novel_selection_retains_csv_headers(self):
        self.candidates("iter1", ["a"])
        self.prepare("iter1")
        root = self.candidates("iter2", ["a"])
        self.assertEqual(self.prepare("iter2")["selected_count"], 0)
        csv = pd.read_csv(root / "mining_pool.csv")
        self.assertEqual(len(csv), 0)
        self.assertEqual(list(csv), ["input_path", "golden_path", "label", "object_name"])

    def test_missing_image_fails_before_history_commit(self):
        self.candidates("iter1", ["missing"])
        (self.workspace / "pool/missing_SolderLight.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.prepare("iter1")
        self.assertFalse((self.rd / "mining_history.json").exists())

    def test_changed_candidates_cannot_rewrite_committed_history(self):
        root = self.candidates("iter1", ["a"])
        self.prepare("iter1")
        before = (self.rd / "mining_history.json").read_bytes()
        frame = pd.read_parquet(root / "mined.parquet")
        frame["max_cosine_similarity"] = 0.1
        frame.to_parquet(root / "mined.parquet")
        with self.assertRaisesRegex(ValueError, "candidate evidence changed"):
            self.prepare("iter1")
        self.assertEqual((self.rd / "mining_history.json").read_bytes(), before)

    def test_distance_is_not_treated_as_cosine_similarity(self):
        root = self.candidates("iter1", ["a"])
        frame = pd.read_parquet(root / "mined.parquet")
        frame.rename(columns={"max_cosine_similarity": "distance"}).to_parquet(root / "mined.parquet")
        with self.assertRaisesRegex(ValueError, "cosine similarity"):
            self.prepare("iter1")


if __name__ == "__main__":
    unittest.main()
