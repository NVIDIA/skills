# Host-optimization loop on `PyTorchModelEngine._prepare_tp_inputs` - round 3

Workload (identical for every round): Llama-3.1-8B-Instruct, TP=4 on 4x B200,
ISL 1024 / OSL 1024, concurrency 128, 4,100 forward steps profiled, CPU affinity
`numactl --cpunodebind=0 --membind=0`.

## Round 3 change (`model_engine.py`, applied as one commit)

The round-3 analysis table proposed:

| Option | Description | Est. savings | Risk |
|---|---|---|---|
| A | Cache `self.mapping.has_cp_helix()` before the loop | ~1.5 s | Low |
| B | A + drop the `first_beam = 0` assignment, compare `beam == 0` | ~1.9 s | Low |
| C | B + `beam_width = ... if _use_beam_search else 1` to skip the attribute chain when beam search is off | ~3 s | Low |

Option C was applied. Unit tests `tests/unittest/_torch/pyexecutor/test_model_engine.py`
passed after the change.

## Results

| Run (EXTRA_SUFFIX) | `_prepare_tp_inputs` total | Output throughput | Mean TPOT |
|---|---|---|---|
| round2_eliminate_redundant_iter (before) | 31.10 s | 3,705 tok/s | 32.3 ms |
| round3_cache_invariants (after) | 31.85 s | 3,688 tok/s | 32.5 ms |

Line-level profiles: `before_round2.txt` and `after_round3.txt` in this directory.
