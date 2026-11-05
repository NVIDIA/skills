# Batched GEMM microbenchmark from our attention-projection study.
# Target GPU: H100 SXM. The region of interest is bracketed with
# cudaProfilerStart/cudaProfilerStop so profilers can capture only the timed loop.
import torch

B, M, N, K = 16, 1024, 4096, 4096
DTYPE = torch.bfloat16
WARMUP, ITERS = 10, 50


def main():
    torch.manual_seed(0)
    a = torch.randn(B, M, K, device="cuda", dtype=DTYPE)
    b = torch.randn(B, K, N, device="cuda", dtype=DTYPE)

    for _ in range(WARMUP):
        torch.bmm(a, b)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(ITERS):
        c = torch.bmm(a, b)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(tuple(c.shape))


if __name__ == "__main__":
    main()
