# Disaggregated gen_only perf-sanity metric moves between reps

Cases: `disagg_upload-gen_only-*` (TensorRT-LLM disaggregated serving, generation
servers measured in isolation). Gating metric: `mean_gen_worker_per_iter_device_step_time`,
parsed from each `gen_server_{i}.log` after the benchmark client returns. Each gen
server runs exactly 2,000 decode iterations per rep.

| Rep | Reported metric | Iterations the parser used (expected 2,000) |
|---|---|---|
| 1 | 8.41 ms | 1,996 |
| 2 | 8.39 ms | 2,000 |
| 3 | 7.12 ms | 1,140 |
| 4 | 8.44 ms | 2,000 |
| 5 | 6.95 ms | 870 |

No errors are raised; the low reps simply report a smaller number.

## How the log is written and read

- Each gen server is started as `srun ... &> gen_server_{i}.log`, so all TP ranks of
  that server write into one file. The log directory is on NFS; the parser runs on
  the benchmark node, not on the gen node. Gen workers stay alive after the client
  returns (they wait on a `benchmark_status` file).
- Parser today (`parse_gen_worker_device_step_time`): poll the log slice until the
  number of usable lines is non-zero and unchanged across two consecutive reads
  (`settle_timeout=90 s`, `poll_interval=2 s`), then average.
- History: before this settle-poll was added, the same cases intermittently failed
  with `missing 'prev_device_step_time' in gen_server_*.log`; the poll made those
  failures disappear.

## Proposed fix (from a teammate)

Raise `settle_timeout` to 300 s and `poll_interval` to 5 s.
