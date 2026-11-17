"""Nightly CompileIQ tuning for our inference kernels.

Each kernel gets its own search space; the driver loop below runs one
Search() per entry and dumps results to results/<kernel>.csv.
"""

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import (
    LocalSearchSpaceBin,
    NvccSearchSpace,
    PtxasSearchSpace,
)

from our_kernels import objectives  # one objective function per kernel
from our_kernels.config import SEARCH_CONFIG

# The 'att' variant is attribute-based register allocation, which helps any
# register-heavy kernel, so we use it for the big dense kernels.
KERNEL_SEARCH_SPACES = {
    # Dense GEMM with fused bias + GELU epilogue
    "fused_gemm_bias_gelu": PtxasSearchSpace(version="13.3", variant="att"),
    # RMSNorm over the hidden dimension
    "rmsnorm": PtxasSearchSpace(version="13.3", variant="att"),
    # Grouped-query attention decode step (FlashInfer-style batch decode)
    "gqa_decode": PtxasSearchSpace(version="13.3", variant="default"),
    # Multi-head latent attention prefill; we want NVCC-level knobs here
    "mla_prefill": NvccSearchSpace(version="13.3", variant="att"),
    # Re-search starting from the winner in the Helion booster pack
    "paged_kv_copy": LocalSearchSpaceBin("packs/helion/paged_kv_copy_best.acf"),
}


def main() -> None:
    for name, search_space in KERNEL_SEARCH_SPACES.items():
        tuner = Search(
            objective_function=getattr(objectives, name),
            search_space=search_space,
            search_config=SEARCH_CONFIG,
            dump_results=f"results/{name}.csv",
        )
        tuner.start()


if __name__ == "__main__":
    main()
