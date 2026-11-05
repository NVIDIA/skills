# Round 0 baseline run

- Model: Llama-3.1-8B-Instruct (text-only), TensorRT-LLM PyTorch backend
- Hardware: 4x B200, TP=4
- Load: ISL 1024 / OSL 1024, concurrency 128, 4,100 forward steps in the profiled window
- Launch: `EXTRA_SUFFIX=round0_baseline bash profile.sh` (our wrapper around the benchmark)
- Profiler env used for this run:

```bash
export TLLM_LINE_PROFILER_ENABLED=True
export TLLM_LINE_PROFILER_PATH=$WORKDIR/lp_results/profile_round0_baseline.txt
# TLLM_LINE_PROFILER_FUNCTIONS not set (default function set only)
```

- CPU affinity: all ranks launched under `numactl --cpunodebind=0 --membind=0`
  (GPUs 0-3 sit on NUMA node 0).
- Benchmark result: output throughput 3,150 tok/s, mean TPOT 37.9 ms.
- `nvidia-smi dmon` GPU utilization during decode: ~55%. Doubling concurrency to 256
  raised throughput by only 6%.
