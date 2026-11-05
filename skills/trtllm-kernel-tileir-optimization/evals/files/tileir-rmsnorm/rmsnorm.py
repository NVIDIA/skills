"""RMSNorm forward used in our decoder layers (bf16 activations, hidden size 4096-8192).

Tuned for Hopper with the PTX backend; verified against torch on H100.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["n_cols"],
)
@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, eps,
                       BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    x2d = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2d)
    n_rows, n_cols = x2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rmsnorm_fwd_kernel[(n_rows,)](x2d, weight, out, x2d.stride(0), out.stride(0), n_cols, eps,
                                  BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)
