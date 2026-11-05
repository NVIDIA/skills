"""CuTe DSL kernel: out = 2 * sigmoid(x) for fp16 2-D tensors (gating activation).

Target: H100 (SM90), nvidia-cutlass-dsl 4.x. Exports kernel_fn / reference_fn / get_inputs
for our verify and benchmark harness.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def scaled_sigmoid_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    if idx >= m * n:
        return
    a = gA[(None, (idx // n, idx % n))].load()
    s = cute.math.sigmoid(a)
    gC[(None, (idx // n, idx % n))] = s * 2


@cute.jit
def scaled_sigmoid_host(mA: cute.Tensor, mC: cute.Tensor):
    vec = 16 // (mA.element_type.width // 8)
    gA = cute.zipped_divide(mA, (1, vec))
    gC = cute.zipped_divide(mC, (1, vec))
    threads = 256
    tiles = cute.size(gA.shape[1])
    scaled_sigmoid_kernel(gA, gC).launch(grid=((tiles + threads - 1) // threads, 1, 1),
                                         block=(threads, 1, 1))


def kernel_fn(x):
    out = torch.empty_like(x)
    scaled_sigmoid_host(from_dlpack(x), from_dlpack(out))
    return out


def debug_single_tile(x):
    """Used by a teammate's unit test to poke one tile without the host wrapper."""
    out = torch.empty_like(x)
    scaled_sigmoid_kernel(from_dlpack(x), from_dlpack(out))
    return out


def reference_fn(x):
    return 2 * torch.sigmoid(x)


def get_inputs():
    return [torch.randn(4096, 8192, dtype=torch.float16, device="cuda")]
