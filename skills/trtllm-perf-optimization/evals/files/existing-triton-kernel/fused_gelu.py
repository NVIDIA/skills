"""Fused bias + GELU (tanh approximation) Triton kernel used in our MLP block.

Currently tuned on H100 with the default (PTX) Triton backend. Correctness test:
tests/test_fused_gelu.py (compares against torch.nn.functional.gelu(x + bias, approximate="tanh")).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n_cols"],
)
@triton.jit
def _bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    v = x + b
    # tanh(z) = 2 * sigmoid(2z) - 1
    z = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    y = 0.5 * v * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))
    tl.store(out_ptr + row * n_cols + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and x.shape[-1] == bias.shape[0]
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    out = torch.empty_like(x)
    grid = lambda meta: (n_rows, triton.cdiv(n_cols, meta["BLOCK_SIZE"]))
    _bias_gelu_kernel[grid](x, bias, out, n_rows, n_cols)
    return out
