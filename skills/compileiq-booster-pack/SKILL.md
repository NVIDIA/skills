---
name: compileiq-booster-pack
description: >
  Use BEFORE running a full CompileIQ search. Walks through downloading a
  Booster Pack from NVIDIA/CompileIQ GitHub Releases, applying ACF candidates
  one at a time to the user's compiler (raw PTXAS, NVCC, Triton, Helion,
  FlashInfer), and keeping only candidates that compile, pass correctness,
  and beat the no-ACF baseline. Includes the mandatory Debug-pack O0/O3
  ACF-injection canary that proves the ACF is reaching PTXAS. Triggers on
  "booster pack", "ACF", "apply-controls", "speed up without searching",
  "helion fp8", "flashinfer batch decode", "debug pack".
when_to_use: |
  - Workload is close to a published pack (Helion FP8 quant / causal
    depthwise conv / Gated DeltaNet fwd; FlashInfer BatchDecode is a known
    related workload).
  - User wants a fast shortcut before paying for a full CompileIQ search.
  - User asks "is there a known-good config for X?"
  Don't use when:
  - User lacks a stable baseline, correctness check, or benchmark setup.
  - Workload has nothing in common with available packs; skip to
    compileiq-run-search.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  version: "1.2.0"
  author: CompileIQ Team <compileiq@nvidia.com>
  domain: compiler-optimization
  tags: [compileiq, cuda, ptxas, compiler-optimization, autotuning]
allowed-tools: Bash Read
paths: ["**/*.cu", "**/*.cuh", "**/*.acf", "**/*.py", "**/*.sh"]
---

# compileiq-booster-pack

Try curated `.acf` candidates *before* running a full CompileIQ search.
A Booster Pack is a zip of ACFs that NVIDIA validated against a specific
workload family. They are not guaranteed speedups; treat every candidate
as workload-specific and validate it on your own benchmark.

Authoritative narrative: [`docs/booster_packs.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/booster_packs.md), [`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md).

## When

| If this is true | Use this path |
|---|---|
| Workload is close to a Booster Pack's intended workload, compiler, GPU, and validation context. | Try the Booster Pack first. |
| Workload differs materially or no pack candidate helps. | Run a full CompileIQ search (`compileiq-run-search`). |
| Baseline, correctness check, compiler path, or benchmark setup are not in place. | Wait. Fix those before applying any ACF. |

## Available packs (today)

| Pack | Workloads it was validated against | Notes |
|---|---|---|
| `booster-pack-helion.zip` | Helion FP8 Quantization, Causal Depthwise Convolution, Gated DeltaNet Forward | Has shown benefit on FlashInfer `BatchDecodeWithPagedKVCacheWrapper`; related attention workloads worth testing. |
| `booster-pack-debug.zip` | Diagnostic ACFs (`O0`, `O3`, others that disable or alter selected optimizations) | Not for speed; for **debugging**. Use the O0/O3 canary below before trusting any other pack. |

The public release shape is documented in [`docs/booster_packs.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/booster_packs.md). Read each
candidate's `compiler_stages` from its pack manifest and use every listed stage.
Only an NVCC build targeting Blackwell or later (`nvcc --apply-controls`) reaches
the `nvcc` stage; `-Xptxas --apply-controls` drops nvcc-stage controls without
a warning. Triton,
raw PTXAS, FlashInfer (`--ptxas-options`) and cuTeDSL only reach `ptxas`, so
in those setups use the ACFs whose stages are `["ptxas"]`.
There is no runtime download API today. Don't invent one.

## Steps

### 0. Pre-flight: the O0/O3 ACF-injection canary (mandatory first step)

The most common silent failure when applying ACFs is a framework cache (Triton,
Helion, FlashInfer's `flashinfer_cubin`/`flashinfer_jit_cache`, NVCC build
cache) serving a stale binary that ignored the ACF. The Debug pack has two
ACFs with predictable, opposite-direction signatures:

- **`ptxas_opt0.acf`**: forces unoptimized PTXAS compilation. Applied → expect a **measurable
  regression** (often 2-10x slower) vs. baseline.
- **`ptxas_opt3.acf`**: forces the default PTXAS optimization level. Applied → expect to
  **match baseline** (the no-ACF default is already `-O3`).

`run-benchmark.sh` stands for your benchmark: it should apply the ACF it is
given exactly the way your objective will (see the injection table below) and
print the time in ms.

```bash
# Baseline
T_BASE_MS=$(./run-benchmark.sh)

# O0 must regress
T_O0_MS=$(./run-benchmark.sh --acf booster-pack-debug/ptxas_opt0.acf)

# O3 must match baseline
T_O3_MS=$(./run-benchmark.sh --acf booster-pack-debug/ptxas_opt3.acf)

python -c "
import sys
base, o0, o3 = $T_BASE_MS, $T_O0_MS, $T_O3_MS
if o0 < base * 1.05:
    print('FAIL: O0 did not regress; ACF is NOT reaching PTXAS. Fix the cache-bust.')
    sys.exit(1)
if abs(o3 - base) / base > 0.05:
    print(f'WARN: O3 differs from baseline by >5%; baseline may not be -O3 or framework caching differs.')
print('PASS: ACF injection is wired up correctly.')
"
```

If this fails, **stop**. Fix the cache-bust before trying any real pack candidate:

- Triton: `export TRITON_ALWAYS_COMPILE=1`, unique `TRITON_CACHE_DIR` per eval.
- Helion: `export HELION_SKIP_CACHE=1`.
- FlashInfer: confirm `flashinfer_cubin` and `flashinfer_jit_cache` packages are absent ([`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md)).
- Raw nvcc: clean the build dir between candidates.

### 1. Download

Browse `https://github.com/NVIDIA/CompileIQ/releases`, find the latest tag
matching `booster-packs-*`, and download the relevant pack zip plus the
top-level `booster-pack-catalog.json`.

```bash
BOOSTER_TAG="$(gh release list -R NVIDIA/CompileIQ --limit 100 --json tagName,isDraft \
  --jq '.[] | select(.isDraft == false) | select(.tagName | startswith("booster-packs-")) | .tagName' \
  | head -n 1)"
echo "Using $BOOSTER_TAG"
gh release download "$BOOSTER_TAG" -R NVIDIA/CompileIQ -p 'booster-pack-helion.zip' -p 'booster-pack-catalog.json' -D ./packs
unzip ./packs/booster-pack-helion.zip -d ./packs
cat ./packs/booster-pack-helion/booster-pack-manifest.json
```

Always read the per-pack manifest before applying: it lists the intended
workload, compiler version, GPU target, validation context, and known caveats.
For a reproducible rerun, set `BOOSTER_TAG` to the exact tag printed above.

### 2. Apply one ACF at a time

| Target | Injection |
|---|---|
| Raw PTXAS | `ptxas -v -arch=sm_100 --apply-controls candidate.acf kernel.ptx` |
| NVCC (CUDA source) | `nvcc --apply-controls candidate.acf -arch=sm_100 kernel.cu -o exe` (applies the ACF to every stage it lists, nvcc and ptxas). Needs a Blackwell or later target (`-arch=sm_100` and up); for earlier targets such as `sm_90`, nvcc rejects `--apply-controls` and the build fails. There, use `nvcc -Xptxas --apply-controls=candidate.acf ...`, which reaches ptxas only and silently ignores nvcc-stage controls. |
| Triton | Launch keyword argument `ptx_options="--apply-controls=candidate.acf"` on the kernel launch (Triton 3.4+) with `TRITON_ALWAYS_COMPILE=1`. On Triton 3.6+ you can instead prefix the whole run: `PTXAS_OPTIONS="--apply-controls=candidate.acf" TRITON_ALWAYS_COMPILE=1 python bench.py`; Triton reads it once at startup, and versions before 3.6 ignore it. |
| Helion | Helion's official ACF API + `HELION_SKIP_CACHE=1` (see `helionlang.com/examples/acfs/softmax_acf.html`). |
| FlashInfer | `FLASHINFER_EXTRA_CUDAFLAGS="--ptxas-options=--apply-controls=$ACF_FILE" python bench.py` (see [`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md)). |

Apply exactly one ACF per run. If it fails to compile, hangs, crashes, returns
wrong answers, or regresses, **reject** that candidate and move to the next.

### 3. Validate every candidate

- Compare against a known-good reference (correctness, not just speed).
- Test multiple input shapes when shape matters.
- Use compile and runtime timeouts to bound runaway candidates.
- Run multiple performance trials if the benchmark is noisy.
- Record the reproducibility checklist below (one row per candidate).

### 4. Reproducibility log

For every candidate you accept or reject, append a row to
`booster-pack-log.csv` with:

- ACF filename (and sha256)
- Manifest / release version
- Benchmark command
- GPU model + driver version
- CTK version
- `nvcc` and `ptxas` paths + versions
- Framework version or commit
- Input shape
- Baseline result (mean ± std)
- Candidate result (mean ± std)
- Correctness status
- Decision: `KEPT` or `REJECTED:<reason>`

This is the same checklist [`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md) recommends.
The `scripts/apply_one_acf.sh` helper does most of this automatically.

## Self-test

```bash
bash scripts/apply_one_acf.sh --self-test
```

Dry-runs a "baseline vs baseline" comparison (no ACF applied to either side)
and confirms the helper correctly reports "NOT a real improvement". Catches
misconfigured script invocations before they pollute the reproducibility log.

## Gotchas

- **Pack name is not a hard boundary.** Helion Pack helps some FlashInfer
  cases ([`docs/booster_packs.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/booster_packs.md)); test before assuming.
- **Booster Packs are not search-space inputs.** Don't try to feed an ACF
  through `PtxasSearchSpace(...)`; packs are already-generated `.acf`
  candidate bundles, not inputs to `PtxasSearchSpace` or `NvccSearchSpace`.
- **Force recompilation.** If you can't *prove* a recompile happened between
  candidates, don't trust the measurement. See the cache-bust hints under the
  pre-flight canary.

## Next

- If no pack candidate helps your workload, go to `compileiq-run-search` for a
  full CompileIQ search over `PtxasSearchSpace()`.
- For attention workloads specifically, also see
  `compileiq-search-space` (variant="att").
