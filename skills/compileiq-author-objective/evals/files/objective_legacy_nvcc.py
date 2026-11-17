"""Tune the ptxas controls for my reduction kernel with CompileIQ.

Objective adapted from an older wiki page. It ran fine with normalize=False;
since switching to normalize=True the first evaluation fails.
"""
import subprocess
import tempfile
from pathlib import Path

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import SearchConfiguration

INVALID_SCORE = float("inf")

SOURCE_CU = Path(__file__).parent / "reduction.cu"
ARCH = "sm_100"


def run_and_time(exe: Path) -> float:
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    # The benchmark prints "elapsed_ms=<float>" on its last line.
    return float(out.stdout.strip().splitlines()[-1].split("=")[1])


def objective(config_blob):
    tmpdir = tempfile.mkdtemp(prefix="ciq_")
    acf_path = Path(tmpdir) / "controls.acf"
    with open(acf_path, "wb") as f:
        f.write(bytes.fromhex(config_blob))

    exe = Path(tmpdir) / "reduction"
    subprocess.run(
        ["nvcc", f"-arch={ARCH}", "-O3",
         "-Xptxas", f"--apply-controls={acf_path}",
         str(SOURCE_CU), "-o", str(exe)],
        check=True,
    )
    return run_and_time(exe)


if __name__ == "__main__":
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(),
        search_config=SearchConfiguration(
            problem_type="min", generations=10, pool_size=16, normalize=True
        ),
        dump_results="results.csv",
    )
    results = tuner.start(num_workers=1)
    best = results.get_best_result()
    with open("best.acf", "wb") as f:
        f.write(bytes.fromhex(best["params"]))
