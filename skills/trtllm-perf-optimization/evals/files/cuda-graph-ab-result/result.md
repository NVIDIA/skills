# Optimization attempt: CUDA Graph capture of the decode loop

- File changed: `serve_decode.py` (decode loop of our in-house PyTorch serving script)
- Before the change: `backup_file("serve_decode.py")` was called; baseline was measured.
- Implementation: perf-torch-cuda-graph-specialist captured the per-token decode step
  (static input buffers, graphs captured for batch sizes 1, 2, 4, 8, 16, 32).
- Prerequisites checked by the specialist: no host-device syncs inside the captured
  region, static shapes per captured batch size, H100 (graphs supported).
- Measurement by perf-profiling-specialist, identical config, 5 repetitions each,
  medians reported:

| Metric | Before | After |
|---|---|---|
| Throughput | 1,520 tok/s | 1,402 tok/s |
| p50 per-token latency | 41.8 ms | 45.2 ms |

- Correctness: generated tokens identical to the baseline on 200 evaluation prompts.
- Profiler note from the specialist: batches of 17-32 requests now pad to the 32-size
  graph; the workload's typical in-flight batch is 18-20.
