# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for capacity_report.py — no GPU/DeepStream required."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

# Load capacity_report.py as a module without requiring a package install.
_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "capacity_report.py"
_spec = importlib.util.spec_from_file_location("capacity_report", _SCRIPT)
cr = importlib.util.module_from_spec(_spec)
sys.modules["capacity_report"] = cr
_spec.loader.exec_module(cr)


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(dedent(content).lstrip("\n"))
    return p


class TestParseMicrobench(unittest.TestCase):
    """parse_microbench handles current + legacy schema and skips garbage."""

    def test_current_schema(self):
        tmp = Path(tempfile.mkdtemp(prefix="capreport_"))
        csv = _write(tmp, "m.csv", """
            B,fps_aggregate,fps_per_stream
            1,769.04,769.04
            2,1493.16,746.58
            4,1498.00,374.50
        """)
        rows = cr.parse_microbench(csv)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["B"], 1)
        self.assertAlmostEqual(rows[0]["fps_aggregate"], 769.04, places=2)
        self.assertAlmostEqual(rows[2]["fps_aggregate"], 1498.00, places=2)

    def test_legacy_schema(self):
        # Legacy script emitted (B, fps_per_batch, fps_aggregate) where the
        # `fps_per_batch` column was actually aggregate fps (the bug we fixed).
        # parse_microbench must accept it: fps_per_batch -> fps_aggregate.
        tmp = Path(tempfile.mkdtemp(prefix="capreport_"))
        csv = _write(tmp, "m.csv", """
            B,fps_per_batch,fps_aggregate
            1,763.65,763.6
            2,1483.90,2967.8
        """)
        rows = cr.parse_microbench(csv)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["fps_aggregate"], 763.65, places=2)
        self.assertAlmostEqual(rows[1]["fps_aggregate"], 1483.90, places=2)

    def test_skips_garbage_rows(self):
        tmp = Path(tempfile.mkdtemp(prefix="capreport_"))
        csv = _write(tmp, "m.csv", """
            B,fps_aggregate,fps_per_stream
            1,769.04,769.04
            ?,not_a_number,oops
            4,1498.00,374.50
        """)
        rows = cr.parse_microbench(csv)
        self.assertEqual(len(rows), 2)


class TestParseDmon(unittest.TestCase):
    """parse_dmon extracts max(SM/mem/dec) from a `nvidia-smi dmon -s mu` log."""

    def test_extracts_maxes(self):
        tmp = Path(tempfile.mkdtemp(prefix="capreport_"))
        log = _write(tmp, "dmon.txt", """
            # gpu     fb   bar1   ccpm     sm    mem    enc    dec    jpg    ofa
            # Idx     MB     MB     MB      %      %      %      %      %      %
                0   100      0      0     30      5      0     20      0      0
                0   100      0      0     82     61      0    100      0      0
                0   100      0      0     45      3      0     50      0      0
        """)
        m = cr.parse_dmon(log)
        self.assertEqual(m["sm_max"], 82)
        self.assertEqual(m["mem_max"], 61)
        self.assertEqual(m["dec_max"], 100)

    def test_filters_by_gpu_id(self):
        # Multi-GPU log: GPU 0 has high values, GPU 1 has low values.
        # Filtering by gpu_id=1 must NOT pick up GPU 0's peaks.
        tmp = Path(tempfile.mkdtemp(prefix="capreport_"))
        log = _write(tmp, "dmon.txt", """
            # gpu     fb   bar1   ccpm     sm    mem    enc    dec    jpg    ofa
            # Idx     MB     MB     MB      %      %      %      %      %      %
                0   100      0      0     90     85      0    100      0      0
                1   100      0      0     20     10      0     30      0      0
                0   100      0      0     95     70      0     90      0      0
                1   100      0      0     25     15      0     40      0      0
        """)
        m = cr.parse_dmon(log, gpu_id=1)
        self.assertEqual(m["sm_max"], 25)
        self.assertEqual(m["mem_max"], 15)
        self.assertEqual(m["dec_max"], 40)
        # No filter -> aggregates across all GPUs (= max from either GPU)
        m_all = cr.parse_dmon(log)
        self.assertEqual(m_all["sm_max"], 95)
        self.assertEqual(m_all["dec_max"], 100)

    def test_handles_empty(self):
        tmp = Path(tempfile.mkdtemp(prefix="capreport_"))
        log = _write(tmp, "dmon.txt", "# no data\n")
        m = cr.parse_dmon(log)
        self.assertEqual(m, {"sm_max": 0, "mem_max": 0, "dec_max": 0})


class TestClassifyBound(unittest.TestCase):
    """classify_bound applies the boundedness-rules.md decision tree."""

    def test_decode_strong(self):
        # B=1 -> 763, B=2 -> 1493 (ratio 0.51 < 0.55) AND NVDEC at 100% =>
        # two independent decode signals => high confidence.
        microbench = [
            {"B": 1, "fps_aggregate": 763.0},
            {"B": 2, "fps_aggregate": 1493.0},
            {"B": 4, "fps_aggregate": 1498.0},
            {"B": 16, "fps_aggregate": 1486.0},
        ]
        dmon = {"sm_max": 46, "mem_max": 32, "dec_max": 100}
        out = cr.classify_bound(microbench, dmon, target_fps=30)
        self.assertEqual(out["bound"], "DECODE_BOUND")
        self.assertEqual(out["confidence"], "high")

    def test_compute_bound(self):
        # FPS plateaus immediately (compute is the limit even at low B);
        # SM at saturation, DRAM low, NVDEC barely engaged.
        microbench = [
            {"B": 1, "fps_aggregate": 1500.0},
            {"B": 2, "fps_aggregate": 1530.0},  # ratio 0.98 — no decode signal
            {"B": 4, "fps_aggregate": 1545.0},
            {"B": 8, "fps_aggregate": 1547.0},
        ]
        dmon = {"sm_max": 95, "mem_max": 40, "dec_max": 30}
        out = cr.classify_bound(microbench, dmon, target_fps=30)
        self.assertEqual(out["bound"], "COMPUTE_BOUND")

    def test_memory_bw_bound(self):
        # DRAM saturated, SM low — memory-BW wall.
        microbench = [
            {"B": 1, "fps_aggregate": 800.0},
            {"B": 2, "fps_aggregate": 820.0},
            {"B": 4, "fps_aggregate": 825.0},
        ]
        dmon = {"sm_max": 55, "mem_max": 92, "dec_max": 40}
        out = cr.classify_bound(microbench, dmon, target_fps=30)
        self.assertEqual(out["bound"], "MEMORY_BW_BOUND")
        self.assertEqual(out["confidence"], "high")

    def test_inconclusive(self):
        # No strong signals.
        microbench = [
            {"B": 1, "fps_aggregate": 500.0},
            {"B": 2, "fps_aggregate": 700.0},  # ratio 0.71 -> no decode
        ]
        dmon = {"sm_max": 50, "mem_max": 30, "dec_max": 50}
        out = cr.classify_bound(microbench, dmon, target_fps=30)
        self.assertEqual(out["bound"], "INCONCLUSIVE")

    def test_zero_fps_no_division_error(self):
        # If a microbench iteration produces fps=0 (pipeline failed to
        # start, probe never fired, etc.) the classifier must not crash
        # with ZeroDivisionError. It should fall through to UNKNOWN /
        # INCONCLUSIVE based on remaining signals.
        microbench = [
            {"B": 1, "fps_aggregate": 100.0},
            {"B": 2, "fps_aggregate": 0.0},   # bad row
            {"B": 4, "fps_aggregate": 0.0},   # bad row
        ]
        dmon = {"sm_max": 0, "mem_max": 0, "dec_max": 0}
        # Must not raise.
        out = cr.classify_bound(microbench, dmon, target_fps=30)
        self.assertIn(out["bound"], {"INCONCLUSIVE", "UNKNOWN_BOTTLENECK"})

    def test_zero_ceiling_fps_no_division_error(self):
        # Highest-B row has fps=0 — plateau check must not crash dividing
        # by ceiling_fps=0.
        microbench = [
            {"B": 1, "fps_aggregate": 800.0},
            {"B": 2, "fps_aggregate": 1500.0},
            {"B": 4, "fps_aggregate": 0.0},   # corrupt last row
        ]
        dmon = {"sm_max": 50, "mem_max": 30, "dec_max": 50}
        out = cr.classify_bound(microbench, dmon, target_fps=30)
        # Should still classify based on the B=1/B=2 ratio without crashing
        self.assertIn(out["bound"], {"DECODE_BOUND", "INCONCLUSIVE",
                                      "UNKNOWN_BOTTLENECK"})


class TestComputeCapacity(unittest.TestCase):
    """compute_capacity treats the measured peak as authoritative."""

    def test_measurement_is_authoritative(self):
        microbench = [
            {"B": 1, "fps_aggregate": 769.0},
            {"B": 2, "fps_aggregate": 1493.0},
            {"B": 4, "fps_aggregate": 1498.0},
        ]
        gpu = {"name": "RTX A6000", "compute_cap": "8.6"}
        out = cr.compute_capacity(microbench, target_fps=30, codec="h264",
                                  source_res="1080p", gpu=gpu,
                                  bound="DECODE_BOUND")
        # n_overall = floor(1498 / 30) = 49
        self.assertEqual(out["n_overall"], 49)
        # Theoretical decode ceiling reported alongside
        self.assertGreater(out["decode_ceiling_theoretical"], 0)
        # Realization < 100% — measurement realized only part of theoretical
        self.assertLess(out["decode_realization_pct"], 100)
        # Dominant ceiling propagated from bound
        self.assertEqual(out["dominant"], "decode")

    def test_unknown_gpu_uses_safe_defaults(self):
        microbench = [{"B": 1, "fps_aggregate": 100.0}]
        out = cr.compute_capacity(microbench, target_fps=30, codec="h264",
                                  source_res="1080p", gpu={},
                                  bound="UNKNOWN_BOTTLENECK")
        self.assertEqual(out["nvdec_count"], 2)  # safe default
        # Default arch bucket is the modern one
        self.assertEqual(out["arch_bucket"], "ada_hopper_blackwell")


class TestArchBucketLookup(unittest.TestCase):
    """_arch_bucket_from_compute_cap chooses the right NVDEC throughput bucket."""

    def test_modern_caps(self):
        for cc in ("8.0", "8.6", "8.9", "9.0", "10.0"):
            self.assertEqual(cr._arch_bucket_from_compute_cap(cc),
                             "ada_hopper_blackwell")

    def test_old_caps(self):
        for cc in ("7.5", "7.0", "6.1"):
            self.assertEqual(cr._arch_bucket_from_compute_cap(cc),
                             "turing_or_older")

    def test_unknown_defaults_to_modern(self):
        self.assertEqual(cr._arch_bucket_from_compute_cap(""),
                         "ada_hopper_blackwell")
        self.assertEqual(cr._arch_bucket_from_compute_cap("not_a_cap"),
                         "ada_hopper_blackwell")


class TestNvdecCountLookup(unittest.TestCase):
    """_nvdec_count_for_gpu matches by substring."""

    def test_known_gpus(self):
        # Counts per NVIDIA's Video Encode/Decode Support Matrix.
        cases = [
            ("NVIDIA RTX A6000", 2),
            ("NVIDIA L40S", 3),
            ("NVIDIA L4", 4),
            ("NVIDIA H100 80GB SXM", 7),
            ("NVIDIA T4", 2),                 # T4 has 2 NVDECs (Turing 4th gen)
            ("NVIDIA A100-SXM4-80GB", 5),
            ("NVIDIA A40", 2),                # A40 has 2 (matrix), not 3
            ("Tesla V100-SXM2", 1),           # Volta — single NVDEC
            ("NVIDIA Orin AGX", 2),
        ]
        for name, expected in cases:
            self.assertEqual(cr._nvdec_count_for_gpu(name), expected, msg=name)

    def test_unknown_gpu_safe_default(self):
        self.assertEqual(cr._nvdec_count_for_gpu("Made-up GPU 9001"), 2)
        self.assertEqual(cr._nvdec_count_for_gpu(""), 2)


if __name__ == "__main__":
    unittest.main()
