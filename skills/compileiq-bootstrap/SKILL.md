---
name: compileiq-bootstrap
description: >
  Use when starting a fresh CompileIQ project, hitting a hang on the first evaluation, or before
  running any other compileiq-* skill. Verifies CUDA 13.3+, ptxas, GPU access,
  that `from compileiq.ciq import Search` and friends resolve, and that
  `PtxasSearchSpace().retrieve()` returns a real path. Documents the env vars
  that control timeouts, caching, and search-space mirroring. Triggers on
  "set up compileiq", "compileiq doesn't work", "first eval hangs", "where do
  search spaces come from", "air-gapped compileiq".
when_to_use: |
  - First time a user opens a CompileIQ project.
  - Search() hangs on its first evaluation (almost always a non-BLAS cause now).
  - Before invoking any other compileiq-* skill in this session.
  Don't use when: the env is already validated in this session.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  version: "1.2.0"
  author: CompileIQ Team <compileiq@nvidia.com>
  domain: compiler-optimization
  tags: [compileiq, cuda, ptxas, compiler-optimization, autotuning]
allowed-tools: Bash Read
paths: ["**/*.py", "**/*.cu", "**/*.acf", "**/*.yaml", "**/*.yml"]
---

# compileiq-bootstrap

Validate that a host can run CompileIQ. No installation of system libraries
(that's distro-specific and out of scope). No state files. Just a checklist
that either passes or prints the precise next fix.

## When

- First-time setup on a new host or container.
- `Search().start()` hangs on the first evaluation.
- After upgrading CUDA, the `compileiq` wheel, or the Python venv.

## Steps

### 1. CUDA toolkit 13.3+

CompileIQ's `--apply-controls` mechanism requires CUDA 13.3 or later
(see [`examples/compilers/nvbench_example/optimize_reduction.py`](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/nvbench_example/optimize_reduction.py)).

```bash
nvcc --version | grep -E "release (13\.([3-9]|[1-9][0-9])|(1[4-9]|[2-9][0-9])\.[0-9]+)"
ptxas --version | grep -E "V(13\.([3-9]|[1-9][0-9])|(1[4-9]|[2-9][0-9])\.[0-9]+)"
```

If either command exits non-zero, install or upgrade the CUDA toolkit and put
`/usr/local/cuda/bin` on `PATH` and `/usr/local/cuda/lib64` on
`LD_LIBRARY_PATH`.

### 2. GPU is visible

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

CompileIQ targets compute capability 9.0 (Hopper / H100) and 10.0 (Blackwell /
B200) most aggressively, but works on any GPU PTXAS supports for the chosen
arch.

### 3. CompileIQ imports resolve

One shot that covers everything callers will need:

```bash
python -c "
from compileiq.ciq import Search
from compileiq.types import INVALID_SCORE, BASELINE_CONFIG, WorkerTypes, ProblemType, SearchConfiguration
from compileiq.search_spaces.compilers import PtxasSearchSpace, NvccSearchSpace, LocalSearchSpaceBin
from compileiq.utils.helpers import save_compiler_config, load_compiler_config
from compileiq.worker import MultiProcessWorker, IsoMultiProcessWorker, RayWorker, AsyncWorker
print('imports OK')
"
```

If this fails, run `pip install compileiq` (or `pip install -e .` from a
source checkout) and re-run.

### 4. Search-space resolution round-trip

This catches network or air-gapped issues that otherwise surface as
hangs deep inside the first evaluation:

```bash
python -c "
from compileiq.search_spaces.compilers import PtxasSearchSpace
p = PtxasSearchSpace().retrieve()
assert p.exists() and p.stat().st_size > 0, p
print(f'resolved: {p}')
"
```

The first run downloads from GitHub releases and caches under
`~/.cache/compileiq/<tag>/`. Subsequent runs hit the cache.

### 5. Env vars to know

| Variable | Default | What it controls |
|---|---|---|
| `CIQ_KEEP_CACHE` | unset (`0`) | Set to `1`/`true`/`yes` to keep `.cache` files after a run for post-mortem replay. |
| `CIQ_PROCESS_MODE` | `forkserver` | `forkserver`, `fork`, or `spawn`. Switch to `spawn` if `forkserver` fails on a constrained host. `IsoMultiProcessWorker` defaults to `fork` regardless. |
| `CIQ_SEARCH_SPACES_DIR` | unset | Path to a local mirror containing `manifest.json` plus the referenced `.bin` files. Set this on air-gapped hosts to skip network. |
| `CIQ_SEARCH_SPACES_REPO` | `NVIDIA/CompileIQ` | Override the GitHub repo that release-backed search-space resolution queries. Useful for staging or forks. |
| `CIQ_SS_TAG_PREFIX` | `search-spaces-` | Tag prefix the resolver uses when `tag="latest"`. Rarely needs changing. |
| `CIQ_SS_LATEST_TTL_DAYS` | `7` | Days to cache the tag that `latest` resolved to before re-checking GitHub. `0` re-resolves every run. |

## Self-test

```bash
bash scripts/check_env.sh
```

Exits 0 if every step passes. Exits with a non-zero count of failures and
prints, for each failure, the precise command the user should run next.

## Gotchas

- **If the first eval hangs:**
  1. Mirror search spaces locally and set
     `CIQ_SEARCH_SPACES_DIR=/path/to/mirror`.
  2. If still hangs, switch process mode: `CIQ_PROCESS_MODE=spawn`.
- **Don't install system libraries in this skill.** Distro-specific package
  installs need root and break in containers and on shared clusters. If
  `nvcc`, `ptxas`, or a Python interpreter is missing, surface the error
  and let the user decide how to install.

## Next

- For attention workloads: `compileiq-search-space` (variant="att" by default).
- Before paying for a full search: try `compileiq-booster-pack`.
- Writing an objective function: `compileiq-author-objective`.
