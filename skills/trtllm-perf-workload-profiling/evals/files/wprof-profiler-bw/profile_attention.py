"""Profile a naive attention block and report softmax bandwidth vs. the GPU peak.

On our PyTorch 2.5 environment this script dies with AttributeError before it
prints the bandwidth numbers.
"""
import torch
from torch.profiler import ProfilerActivity, profile

B, H, S, D = 16, 32, 2048, 128
ITERS = 20


def attention_block(q, k, v):
    scores = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main():
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for _ in range(10):
        attention_block(q, k, v)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            attention_block(q, k, v)
        torch.cuda.synchronize()

    events = prof.key_averages()
    print(events.table(sort_by="self_cuda_time_total", row_limit=15))
    total_gpu = sum(e.self_cuda_time_total for e in events)
    softmax_time = sum(e.cuda_time_total for e in events if "softmax" in e.key)
    print(f"softmax share of GPU time: {100 * softmax_time / total_gpu:.1f}%")

    props = torch.cuda.get_device_properties(0)
    peak_bw = 2 * props.mem_clock_rate * (props.bus_width / 8) / 1e9
    print(f"{props.name}: {props.total_mem / 2**30:.0f} GiB, peak {peak_bw:.0f} GB/s")

    # softmax reads and writes the fp16 score matrix once per call
    bytes_moved = 2 * (B * H * S * S) * 2 * ITERS
    achieved = bytes_moved / softmax_time / 1e9
    print(f"softmax achieved {achieved:.0f} GB/s ({100 * achieved / peak_bw:.1f}% of peak)")


if __name__ == "__main__":
    main()
