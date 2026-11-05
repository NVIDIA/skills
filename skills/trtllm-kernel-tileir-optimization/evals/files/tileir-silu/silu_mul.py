"""Fused SiLU-and-multiply (SwiGLU activation) for bf16 activations. Tuned on H100."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_mul_kernel(x_ptr, g_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    y = g * tl.sigmoid(g) * x
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(x, gate):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    silu_mul_kernel[grid](x, gate, out, n)
    return out
