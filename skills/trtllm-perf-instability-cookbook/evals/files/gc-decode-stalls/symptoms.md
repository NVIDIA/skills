# Run-to-run throughput spread on the AutoDeploy backend

Model: 30B-A3B hybrid Mamba/MoE model, FP8, TensorRT-LLM served through the
AutoDeploy backend (`backend: _autodeploy`), TP=4 on 4x H100, ISL/OSL 1024/1024,
concurrency 1. Same config, same nodes, same container for all reps. We did not set
any GC-related option in our YAML.

| Rep | Output tok/s/user | ITL p50 | ITL p99 |
|---|---|---|---|
| 1 | 341 | 2.9 ms | 610 ms |
| 2 | 289 | 2.9 ms | 1,050 ms |
| 3 | 336 | 2.9 ms | 640 ms |
| 4 | 297 | 2.9 ms | 980 ms |
| 5 | 344 | 2.9 ms | 590 ms |
| 6 | 286 | 2.9 ms | 1,020 ms |

Spread: 286-344 tok/s/user (~17% of the max) on identical config.

## What the traces show

- nsys: every 20-40 s there is a 0.5-1.0 s window with no GPU kernels on any rank;
  outside those windows decode steps look identical across reps.
- `py-spy dump --pid <executor pid>` taken during one of these windows: the main
  thread is inside `collect` (gc) under `_executor_loop`, not in a kernel launch or
  a collective.
- We printed `gc.get_threshold()` from inside the executor process:
  - AutoDeploy backend run: `(700, 10, 10)`
  - same model through the regular PyTorch backend: `(20000, 10, 10)`, and those runs
    show no such stalls.
- Earlier hypothesis in the team: "the model is host-bound, it does not scale with TP".
