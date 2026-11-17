"""CompileIQ objective for my Triton softmax kernel.

Problem: every candidate scores the same as the baseline (within noise) for
the whole search, so I suspect the ACF never takes effect.
"""
import os
import tempfile

import torch
import triton
import triton.language as tl

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * n_cols + offs, num / tl.sum(num, axis=0), mask=mask)


def run_softmax(x):
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    softmax_kernel[(n_rows,)](out, x, n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols))
    return out


def objective(config):
    if isinstance(config, dict) and not config:
        os.environ.pop("PTXAS_OPTIONS", None)
    else:
        acf = tempfile.NamedTemporaryFile(suffix=".acf", delete=False)
        save_compiler_config(acf.name, config)
        os.environ["PTXAS_OPTIONS"] = f"--apply-controls={acf.name}"

    x = torch.randn(4096, 2048, device="cuda", dtype=torch.float16)
    try:
        return triton.testing.do_bench(lambda: run_softmax(x), warmup=25, rep=100)
    except Exception:
        return INVALID_SCORE


if __name__ == "__main__":
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(),
        search_config=SearchConfiguration(problem_type="min", generations=8, pool_size=24),
        dump_results="results.csv",
    )
    results = tuner.start(num_workers=1)
    save_compiler_config("best.acf", results.get_best_result()["params"])
