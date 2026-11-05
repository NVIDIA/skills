"""Small Triton kernels used by our fp16 inference stack.

Reported problems: one kernel fails to compile, outputs are wrong at the tail of
tensors whose size is not a multiple of the block, the split-K GEMM is wrong
after autotuning, the roll op is wrong for some shifts, and the dropout wrapper
is slow.
"""
import torch
import triton
import triton.language as tl


# --- 1. bias + tanh-GELU (fp16 in, fp16 out) --------------------------------------------
@triton.jit
def bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_elements, n_cols, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    b = tl.load(b_ptr + offsets % n_cols)
    y = x + b
    inner = 0.7978845608 * (y + 0.044715 * y * y * y)
    t = (tl.exp(2.0 * inner) - 1.0) / (tl.exp(2.0 * inner) + 1.0)
    tl.store(out_ptr + offsets, 0.5 * y * (1.0 + t))


def bias_gelu(x, bias):
    out = torch.empty_like(x)
    n = x.numel()
    bias_gelu_kernel[(triton.cdiv(n, 1000),)](x, bias, out, n, x.shape[-1], 1000)
    return out


# --- 2. dropout ----------------------------------------------------------------------------
@triton.jit
def dropout_kernel(x_ptr, out_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    keep = tl.rand(seed, offsets) > p
    tl.store(out_ptr + offsets, tl.where(keep, x / (1.0 - p), 0.0), mask=mask)


def dropout(x, p=0.1):
    out = torch.empty_like(x)
    n = x.numel()
    seed = torch.randint(0, 2**31 - 1, (1,), device=x.device).item()
    grid = lambda meta: triton.cdiv(n, meta["BLOCK_SIZE"])
    dropout_kernel[grid](x, out, n, p, seed, BLOCK_SIZE=1024)
    return out


# --- 3. split-K GEMM: C (fp32, zero-initialised) += A (fp16) @ B (fp16) ---------------------
@triton.jit
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 8}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
def splitk_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc.to(tl.float32), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def splitk_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT_K"])
    splitk_matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1))
    return c


# --- 4. roll a 1-D tensor right by `shift` (0 <= shift < n), like torch.roll ----------------
@triton.jit
def roll_kernel(x_ptr, out_ptr, n, shift, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    src = (offs - shift) % n
    tl.store(out_ptr + offs, tl.load(x_ptr + src, mask=mask), mask=mask)


def roll(x, shift):
    out = torch.empty_like(x)
    n = x.numel()
    roll_kernel[(triton.cdiv(n, 1024),)](x, out, n, shift, BLOCK_SIZE=1024)
    return out
