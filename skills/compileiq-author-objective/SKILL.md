---
name: compileiq-author-objective
description: >
  Use when writing the objective_function= passed to Search(). Covers the two
  legal signatures (compiler-only str vs mixed list), the baseline-knockout
  branch, per-eval cache busting, framework-specific --apply-controls
  injection (raw PTXAS, NVCC, Triton, Helion, cuTeDSL/FA4, FlashInfer),
  correctness-before-timing, INVALID_SCORE handling, and the Debug-pack
  O0/O3 ACF-injection canary that must pass before launching a search.
  Triggers on "objective function", "apply-controls", "INVALID_SCORE",
  "save_compiler_config", "baseline knockout", "BASELINE_CONFIG", "every config
  returns the same score", "TypeError fromhex".
when_to_use: |
  - About to write or modify the function passed as objective_function=.
  - Search runs but every config returns an identical score (canary fails).
  - Getting TypeError around fromhex or bytes (legacy pattern).
  - Search hangs on individual configs (timeout/correctness handling missing).
license: CC-BY-4.0 AND Apache-2.0
metadata:
  version: "1.2.0"
  author: CompileIQ Team <compileiq@nvidia.com>
  domain: compiler-optimization
  tags: [compileiq, cuda, ptxas, compiler-optimization, autotuning]
allowed-tools: Bash Read
paths: ["**/*.py", "**/*.cu", "**/*.cuh"]
---

# compileiq-author-objective

The objective function is where ~80% of CompileIQ user errors happen. This
skill tells you the exact shape it must have for current CompileIQ, how to
inject `--apply-controls` for each supported compile path, and how to verify
the whole pipeline works before paying for a full search.

For paste-ready full-file templates per framework, see
`references/templates.md`.

## When

- Writing a brand-new objective function.
- Migrating an older objective off the legacy `bytes.fromhex(config_blob)` pattern.
- Diagnosing "every config returns the same score" or "TypeError: fromhex".

## The two legal signatures

| Shape of `search_space=` | Objective signature | What `config` is |
|---|---|---|
| Single provider, e.g. `PtxasSearchSpace()` | `def objective(config: str) -> float` | A hex string. Pass it straight to `save_compiler_config(acf_path, config)`. |
| List, e.g. `[{"k": ss.choice(...)}, PtxasSearchSpace()]` | `def objective(mixed: list) -> float` | A list of the same length. Unpack: `user_space, ptxas_config = mixed`, but check for `BASELINE_CONFIG` first (see below). |

Mixed-space results keep the same list shape in `best["params"]`. Unpack it
before saving the ACF, for example:
`user_space, ptxas_config = best["params"]`. (Pattern reference:
[`examples/compilers/triton_example/mixed_triton.py`](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/triton_example/mixed_triton.py).)

For multi-objective, return `tuple[float, ...]` of length `num_objectives`.

## Canonical imports

```python
from compileiq.types import INVALID_SCORE, BASELINE_CONFIG
from compileiq.utils.helpers import save_compiler_config
```

`INVALID_SCORE` is CompileIQ's sentinel — return it on any failure (compile,
hang, wrong answer, exception). Do **not** redefine it as `float('inf')`.

`BASELINE_CONFIG` is the empty-dict sentinel (`{}`) CompileIQ passes for a
baseline run. With `normalize=True` the first task of the search is always a
baseline run; without normalization it can still appear when knockout
removes every parameter. It is `{}` whatever the search-space shape, including
mixed (list) spaces.

`save_compiler_config(path, hex_str)` writes the binary blob to disk; it
handles the `bytes.fromhex` internally (`compileiq/utils/helpers.py:128-137`).
Users never need to touch `fromhex` themselves.

## Self-contained for IsoMultiProcessWorker and Ray

Heavy library imports (torch, triton, helion, cute) go **inside** the function
so the process `IsoMultiProcessWorker` spawns — or the remote Ray task — can
re-import them in a clean state. Cheap module-level constants (paths, regexes)
are fine.

## Per-eval cache busting (non-negotiable)

```python
import os, tempfile
# Set at the start of each evaluation, inside the objective's process.
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["HELION_SKIP_CACHE"]     = "1"
os.environ["TRITON_CACHE_DIR"]      = tempfile.mkdtemp(prefix="ciq_triton_")
```

If the benchmark runs in a subprocess instead, prefix its command with the same
three variables (for example `TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR=... python bench.py`).

For FlashInfer, additionally confirm the prebuilt cubin cache packages are
absent — `flashinfer_cubin` and `flashinfer_jit_cache`. See
[`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md) for the import-time check.

## Per-framework `--apply-controls` injection

| Target | Injection |
|---|---|
| Raw PTXAS (you have a `.ptx` file) | `ptxas --apply-controls candidate.acf kernel.ptx -arch=sm_100 -o kernel.cubin` |
| NVCC source (CUDA `.cu`) | `nvcc --apply-controls candidate.acf -arch=sm_100 kernel.cu -o exe`, which applies the ACF to every stage it lists (nvcc and ptxas). This form needs a Blackwell or later target (`-arch=sm_100` and up): for earlier targets such as `sm_90`, nvcc rejects `--apply-controls` and the build fails, even for a ptxas-only ACF. There, use `nvcc -Xptxas --apply-controls=candidate.acf ...`, which reaches ptxas only and silently ignores any nvcc-stage controls in the ACF. |
| Triton kernel | launch keyword argument `ptx_options=f"--apply-controls={acf_path}"` on the kernel launch (Triton 3.4+) plus `TRITON_ALWAYS_COMPILE=1`, `os.environ["TRITON_PTXAS_PATH"] = shutil.which("ptxas")`, and `os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = shutil.which("ptxas")` when Blackwell-specific PTXAS selection may apply. Use this inside objectives; see the `PTXAS_OPTIONS` gotcha below. |
| Helion | Helion's official ACF API. See `https://helionlang.com/examples/acfs/softmax_acf.html`. Always set `HELION_SKIP_CACHE=1`. |
| cuTeDSL / FA4 (TVM-FFI) | `cute.compile(..., options=f"{existing_options} --ptxas-options '--apply-controls {acf_path}'")`. If you can't reach the call site, patch `CompileCallable.__call__` to splice in the option string. |
| FlashInfer | `FLASHINFER_EXTRA_CUDAFLAGS="--ptxas-options=--apply-controls=$ACF_FILE"` (see [`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md)). |

## Baseline knockout branch

```python
def objective(config):
    if isinstance(config, dict) and not config:   # config == BASELINE_CONFIG
        return measure_without_acf()              # establish baseline run
    # config is a hex string: apply the ACF
    ...
```

Mixed (list) search spaces get the same `{}`, so check before unpacking, and
measure your default user parameters with no ACF:

```python
def objective(mixed):
    if mixed == BASELINE_CONFIG:
        return measure(DEFAULT_USER_CONFIG, acf_path=None)
    user_space, ptxas_config = mixed               # safe to unpack now
    ...
```

Without this check, unpacking `{}` raises `ValueError` on the first task when
`normalize=True`.

## Correctness-before-timing (mandatory)

The optimizer rewards whatever you measure. If you only measure
latency, the algorithm will happily reward configs that compile faster by
producing wrong answers. **Always** verify against a reference first:

```python
if not torch.allclose(actual, reference, atol=1e-2, rtol=0):
    return INVALID_SCORE
return triton.testing.do_bench(lambda: kernel(...), warmup=100, rep=1000, return_mode="mean")
```

(Pattern from [`examples/compilers/triton_example/mixed_triton.py`](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/triton_example/mixed_triton.py).)

## Catch everything → return INVALID_SCORE

```python
try:
    ...
except (subprocess.TimeoutExpired, RuntimeError, FileNotFoundError, ValueError, OSError) as e:
    return INVALID_SCORE
```

When in doubt, catch broadly. CompileIQ's workers do catch an exception that
escapes the objective, print the traceback, and record `INVALID_SCORE` with a
warning, so one bad config does not stop the search. Catching it yourself is
still better: you control cleanup (temp ACFs, build dirs), and you can tell
expected failures (compile error, wrong answer) from bugs in the objective.
What *does* stop a search: under `IsoMultiProcessWorker`, a child process
that dies abnormally (for example a segfault, or a return shape that doesn't
match `num_objectives`) raises `RuntimeError`.

## Pre-search canary (mandatory before tuner.start())

Two cheap calls that catch ~90% of "every score is the same" bugs:

```python
# Shape check — does the objective even run?
sample = tuner.sample(1)[0]
score = objective(sample)
print(f"sample run: {score}")
assert isinstance(score, (int, float)) and score == score   # not NaN

# ACF-injection canary using the Debug pack (downloaded once)
from compileiq.utils.helpers import load_compiler_config
O0_HEX = load_compiler_config("booster-pack-debug/ptxas_opt0.acf")
O3_HEX = load_compiler_config("booster-pack-debug/ptxas_opt3.acf")

baseline = objective({})                  # BASELINE_CONFIG path
score_O0 = objective(O0_HEX)              # mixed space: objective([default_user_params, O0_HEX])
score_O3 = objective(O3_HEX)              # mixed space: objective([default_user_params, O3_HEX])

assert score_O0 > baseline * 1.05, (
    f"O0 should regress (got {score_O0} vs baseline {baseline}). "
    "ACF is NOT reaching PTXAS — fix the cache-bust."
)
assert abs(score_O3 - baseline) / baseline < 0.05, (
    f"O3 should match baseline (got {score_O3} vs {baseline})."
)
print("ACF injection canary PASSED — safe to start the search.")
```

If either assertion fails, **stop** and fix the cache-bust before launching
the search. Otherwise every generation's score is measurement noise on a stale
binary.

## Self-test

A 3-line "smoke" objective inside the SKILL author's repo, used to verify the
scaffolding before plugging in a real kernel:

```python
def smoke_objective(config):
    return 1.0   # constant; useful to verify Search() shape, not measurement
```

Drop it into the `Search(...)` call and run 2 generations; if that completes
and `results.get_best_result()` returns a dict, your scaffold is correct.

## Gotchas

- **Don't inject Triton ACFs through `PTXAS_OPTIONS` inside an objective.**
  Triton 3.6+ reads `PTXAS_OPTIONS` once, when its NVIDIA backend is
  imported, as the default for `ptx_options`; earlier versions ignore it.
  Changing `os.environ["PTXAS_OPTIONS"]` between evaluations therefore does
  nothing. Pass `ptx_options=` on the kernel call instead. The env var is only
  safe as a prefix on a whole benchmark process
  (`PTXAS_OPTIONS=... python bench.py`, Triton 3.6+).
- **Triton caches on the ACF *path*, not its contents.** Rewriting the same
  ACF file each evaluation hits the cache and reuses the old binary unless
  `TRITON_ALWAYS_COMPILE=1` is set.
- **Mixed search spaces require list unpacking.** If you pass
  `search_space=[user_dict, PtxasSearchSpace()]`, your objective must accept
  a list, not a string. Results keep that list in `best["params"]`; unpack it
  before saving the compiler config.
- **Don't redefine `INVALID_SCORE`.** Import it from `compileiq.types`. If
  you redefine it locally as `float('inf')`, the value happens to work today
  but is not guaranteed to in future releases.
- **`config_blob` is no longer a parameter name.** The old skill set used
  `def objective(config_blob)` and called `bytes.fromhex(config_blob)`. Both
  are stale. Use `def objective(config)` and `save_compiler_config(path, config)`.

## Next

- Sizing `SearchConfiguration` and picking a Worker: `compileiq-run-search`.
- After the search: `compileiq-validate-result`.
- If something's wrong: `compileiq-debug`.
