# Tests

Unit tests for `scripts/capacity_report.py`. They do **not** require a GPU or
DeepStream — pure-Python checks of the parser, classifier, and capacity-derivation logic.

## Run

See [`README.md`](../README.md#testing) in the skill root for the standard test command.

## What's covered

19 tests grouped by `TestCase` class:

| Class | Test | Behaviour |
|---|---|---|
| `TestParseMicrobench` | `test_current_schema` | Parses `B,fps_aggregate,fps_per_stream` rows correctly. |
| | `test_legacy_schema` | Falls back to the legacy `B,fps_per_batch,fps_aggregate` columns (treats `fps_per_batch` as the aggregate, since the legacy script emitted aggregate fps under that name). |
| | `test_skips_garbage_rows` | Ignores rows with non-numeric values without raising. |
| `TestParseDmon` | `test_extracts_maxes` | Extracts max(SM%), max(mem%), max(dec%) from a `nvidia-smi dmon -s mu` log. |
| | `test_filters_by_gpu_id` | When `gpu_id` is given, only rows for that GPU index are aggregated. |
| | `test_handles_empty` | Empty / header-only logs return zeros without raising. |
| `TestClassifyBound` | `test_decode_strong` | B=1→B=2 fps ratio < 0.55 + NVDEC peak ≥ 90% ⇒ DECODE_BOUND, high confidence. |
| | `test_compute_bound` | Per-batch FPS plateaus immediately + SM% high + DRAM% low ⇒ COMPUTE_BOUND. |
| | `test_memory_bw_bound` | DRAM% high with SM% low ⇒ MEMORY_BW_BOUND. |
| | `test_inconclusive` | No strong signals ⇒ INCONCLUSIVE. |
| | `test_zero_fps_no_division_error` | Microbench rows with `fps_aggregate=0` don't crash the classifier with `ZeroDivisionError`. |
| | `test_zero_ceiling_fps_no_division_error` | Highest-B row with fps=0 skips the plateau-flatness check rather than dividing by zero. |
| `TestComputeCapacity` | `test_measurement_is_authoritative` | `n_overall = peak_measured // target_fps` regardless of the theoretical NVDEC-table value; theoretical is reported alongside as a sanity check. |
| | `test_unknown_gpu_uses_safe_defaults` | When `nvidia-smi` is unavailable, fall back to NVDEC=2, ada_hopper_blackwell arch. |
| `TestArchBucketLookup` | `test_modern_caps` | Compute cap ≥ 8 → `ada_hopper_blackwell`. |
| | `test_old_caps` | Compute cap < 8 → `turing_or_older`. |
| | `test_unknown_defaults_to_modern` | Empty / non-numeric input → `ada_hopper_blackwell` (safe default). |
| `TestNvdecCountLookup` | `test_known_gpus` | Substring match on GPU name returns the correct NVDEC count for known cards (T4, A6000, A40, A100, V100, L4, L40S, H100, Orin). Specific names listed before generic ones (`L40S` before `L4`, `A100` before `A10`). |
| | `test_unknown_gpu_safe_default` | Unknown GPU name falls back to NVDEC=2. |

## When tests fail

The classifier returning the wrong bound type is usually the most informative failure —
review `references/boundedness-rules.md` for whether the trigger thresholds need adjusting,
or whether the test's input represents an edge case that the rules haven't yet codified.
