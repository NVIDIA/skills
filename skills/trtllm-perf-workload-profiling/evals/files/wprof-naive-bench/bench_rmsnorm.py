"""Benchmark for an eager-mode RMSNorm. Results vary a lot between runs."""
import time

import torch


def rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def main():
    times = []
    for i in range(20):
        x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = rmsnorm(x, weight)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"rmsnorm: {sum(times) / len(times):.3f} ms")


if __name__ == "__main__":
    main()
