"""Overnight ptxas tuning run for the reduction kernel.

Target box: one node with 2 x H100 (CUDA_VISIBLE_DEVICES=0,1).
Expected wall time: roughly 10 hours.
"""
import os
import subprocess
import tempfile

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import ProblemType, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config


def lock_clocks():
    # Lock clocks so every evaluation sees the same frequency.
    subprocess.run(["sudo", "nvidia-smi", "-pm", "1"], check=True)
    subprocess.run(
        ["sudo", "nvidia-smi", "--lock-gpu-clocks=1980,1980", "--lock-memory-clocks=2619,2619"],
        check=True,
    )


def objective(config: str) -> float:
    lock_clocks()
    workdir = tempfile.mkdtemp(prefix="ciq_")
    acf_path = os.path.join(workdir, "candidate.acf")
    exe_path = os.path.join(workdir, "reduction")
    save_compiler_config(acf_path, config)
    subprocess.run(
        [
            "nvcc",
            f"-Xptxas=--apply-controls={acf_path}",
            "-arch=sm_90",
            "reduction.cu",
            "-o",
            exe_path,
        ],
        check=True,
    )
    out = subprocess.run([exe_path, "--trials", "50"], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())  # median latency in ms


if __name__ == "__main__":
    config = SearchConfiguration(
        problem_type=ProblemType.MIN,
        generations=25,
        pool_size=32,
    )
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version="13.3"),
        search_config=config,
    )
    results = tuner.start(num_workers=8)
    print(results.get_best_result())
