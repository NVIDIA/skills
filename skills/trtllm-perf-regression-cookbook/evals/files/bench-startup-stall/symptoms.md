# Nightly perf QA: multi-GPU throughput cases "regressed"

QA comparison (candidate nightly vs previous good nightly, same cluster, same configs):

- 14 multi-GPU cases (TP4/TP8, several models) report -31% to -100% throughput.
- All 22 single-GPU cases are within +/-2%.

## One affected case (candidate wheel), harness stdout

```
[2026-10-14 02:14:05] perf harness: launching trtllm-bench throughput (tp=8)
[2026-10-14 02:14:06] [TRT-LLM] [I] Starting trtllm-bench, model=deepseek-r1-fp4, tp=8
[2026-10-14 02:14:07] [llmapi] start MpiSession with 8 workers
[2026-10-14 02:44:08] perf harness: no stdout for 1800 s (_STALL_TIMEOUT=1800s); sending SIGKILL to process group
[2026-10-14 02:44:09] perf harness: parsed 0 of 3 iterations; reporting partial result
```

`nvidia-smi` snapshot taken at 02:30 on that node: 8 `python` worker processes,
each using 4 MiB of GPU memory, GPU-Util 0% on all 8 GPUs. No Python traceback,
no MPI abort.

## A serve-mode case

`deepseek-r1-fp4-serve`: `/health did not become ready within 3600s`. The "baseline"
wheel QA used for this serve comparison was built one day after commit 3 below landed;
when QA re-ran the case on 2026-10-15, the same `/health` timeout happened with
**both** the candidate wheel and that baseline wheel.

## Launch chain used by the perf harness

bash -> pytest (`tests/integration/defs/perf/`) -> `subprocess.Popen(server_cmd, ...)`
-> `trtllm-llmapi-launch` -> `mpirun` -> worker processes.

## Commit subjects between the previous good nightly and the candidate

1. `[None][perf] Add 128x256 tile config to FP4 grouped GEMM`
2. `[None][fix] Fix MoE routing NaN for top-k=8`
3. `[None][feat] Pass IPC HMAC key through file descriptor`
4. `[None][chore] Bump flashinfer to the next minor release`
5. `[None][perf] Tune autotuner buckets for MLA decode`
6. `[None][test] Update perf test waives`
7. `[None][docs] Refresh deployment guide for DeepSeek-R1`
8. `[None][fix] Handle empty stop-word lists`
9. `[None][feat] Add NVFP4 KV-cache option for Llama models`
