---
name: compileiq-search-space
description: >
  Use when picking the search_space= argument for Search(). Covers the three
  provider classes (PtxasSearchSpace, NvccSearchSpace, LocalSearchSpaceBin),
  how to pin a version/variant/tag, the attention-focused 'att' variant for
  attention kernels (FlashAttention, GQA, MHA, MLA, FlashInfer Batch Decode),
  air-gapped mirroring via CIQ_SEARCH_SPACES_DIR, and custom user-defined
  search spaces built from compileiq.search_spaces.base primitives. Triggers
  on "search space", "PtxasSearchSpace", "NvccSearchSpace", "air-gapped
  compileiq", "offline compileiq", "CIQ_SEARCH_SPACES_DIR", "attention
  variant", "ptxas att".
when_to_use: |
  - About to instantiate Search() and need to pick a search_space.
  - Network-restricted host that can't reach github.com/releases.
  - Have a one-off .bin and want to use it directly.
  - Want to pin a specific search-space release for reproducibility.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  version: "1.2.0"
  author: CompileIQ Team <compileiq@nvidia.com>
  domain: compiler-optimization
  tags: [compileiq, cuda, ptxas, compiler-optimization, autotuning]
allowed-tools: Bash Read
paths: ["**/*.py", "**/*.yaml", "**/*.yml"]
---

# compileiq-search-space

CompileIQ ships compiler search spaces as release-backed binary blobs that
the providers in `compileiq.search_spaces.compilers` fetch on demand. This
skill covers the three provider classes, the variants they expose today,
how to use them offline, and how to define a custom search space for
non-compiler tuning.

## When

- Choosing a search space for a new project.
- Pinning a specific release for a paper or production deployment.
- Working on an air-gapped or corporate-firewalled host.
- Stress-testing a one-off `.bin` you have on hand.

## The three provider classes

Reference: `compileiq/search_spaces/compilers.py:66-102`.

```python
from compileiq.search_spaces.compilers import (
    PtxasSearchSpace,
    NvccSearchSpace,
    LocalSearchSpaceBin,
)

# Default — auto-fetch latest PTXAS 13.3 default variant from GitHub releases
ss = PtxasSearchSpace()

# Pinned for reproducibility
ss = PtxasSearchSpace(version="13.3", variant="default", tag="search-spaces-YYYY.MM.DD")

# **Attention** workloads (FlashAttention / GQA / MHA / MLA / FlashInfer Batch Decode)
ss = PtxasSearchSpace(version="13.3", variant="att")

# NVCC variant for full-pipeline tuning
ss = NvccSearchSpace(version="13.3")

# Single .bin you already have on disk — skips manifest + network
ss = LocalSearchSpaceBin("/path/to/ptxas13.3_search_space.bin")

# Pass to Search
from compileiq.ciq import Search
tuner = Search(objective_function=..., search_space=ss, search_config=...)
```

## Variants available today

From `release/search-spaces/manifest-source.yaml`:

| Compiler | Version | Variant | File | When to use |
|---|---|---|---|---|
| `ptxas` | `13.3` | `default` | `ptxas13.3_search_space.bin` | Generic PTXAS tuning; the right starting point for most kernels. |
| `ptxas` | `13.3` | **`att`** | `ptxas13.3_att_search_space.bin` | **Attention workloads.** `att` is short for *attention*, not "attribute". This variant is curated for FlashAttention, GQA, MHA, MLA, FlashInfer Batch Decode, and similar attention kernels. Prefer this whenever the kernel is attention-shaped. |
| `nvcc` | `13.3` | `default` | `nvcc13.3_search_space.bin` | Full-compiler (front-end + back-end) tuning when you want NVCC-level knobs, not just PTXAS. |

To enumerate variants in the latest release at any time:

```bash
gh release view --json assets --jq '.assets[].name' -R NVIDIA/CompileIQ <tag>
```

## Air-gapped / offline mirror

Pre-download the manifest plus all `.bin` files on a connected host, then point
CompileIQ at the local mirror via `CIQ_SEARCH_SPACES_DIR`:

```bash
# On a connected host. Search-space releases use dated tags
# (search-spaces-YYYY.MM.DD); there is no literal "latest" tag, so resolve the
# newest published one the same way CompileIQ does.
SS_TAG="$(gh release list -R NVIDIA/CompileIQ --limit 100 --json tagName,isDraft,isPrerelease \
  --jq '.[] | select(.isDraft == false and .isPrerelease == false) | select(.tagName | startswith("search-spaces-")) | .tagName' \
  | head -n 1)"
echo "Using $SS_TAG"
mkdir -p /shared/ciq-search-spaces
gh release download "$SS_TAG" -R NVIDIA/CompileIQ -D /shared/ciq-search-spaces

# Move the directory to the air-gapped host (rsync, scp, sneaker-net, …)

# On the air-gapped host
export CIQ_SEARCH_SPACES_DIR=/shared/ciq-search-spaces
python -c "from compileiq.search_spaces.compilers import PtxasSearchSpace; print(PtxasSearchSpace().retrieve())"
```

Other env-var knobs:

- `CIQ_SEARCH_SPACES_REPO` (default `NVIDIA/CompileIQ`): override the GitHub
  repo the resolver queries — useful for staging or forks.
- `CIQ_SS_TAG_PREFIX` (default `search-spaces-`): tag prefix used when
  resolving `tag="latest"`. Rarely needs changing.
- `CIQ_SS_LATEST_TTL_DAYS` (default `7`): how long the tag that `latest`
  resolved to is cached on disk before CompileIQ asks GitHub again. So a new
  search-space release can take up to a week to be picked up; set `0` to
  re-resolve on every run, or pin `tag=` for reproducibility. If GitHub is
  unreachable, CompileIQ falls back to the cached tag with a warning.

## Cache location

Resolved binaries are cached at `~/.cache/compileiq/<tag>/<sha256_prefix>_<filename>`.
The resolver verifies cached files by sha256; a corrupted entry is re-downloaded
on the next call. Safe to wipe — wiping just costs one re-download.

## Custom search spaces (non-compiler tuning)

For hyperparameter tuning, autotuner knobs, or any user-defined space, pass
a dict (or list-of-dicts) directly instead of a provider. Primitives live in
`compileiq/search_spaces/base.py`:

```python
import compileiq.search_spaces.base as ss

search_space = {
    "block_size":  ss.choice([64, 128, 256, 512]),
    "unroll":      ss.range(start=1, end=8, step=1),
    "use_shmem":   ss.literal(True, knockout_prob=0.5),
    "lr":          ss.log_sampling(start=1e-5, end=1e-1, total=20),
}
```

| Primitive | Purpose |
|---|---|
| `choice([...])` | Sample uniformly from a discrete list. |
| `range(start, end, step)` | Range-like sampling (also supports float steps). |
| `literal(value, knockout_prob=...)` | Constant value; `knockout_prob` lets the GA disable this parameter. |
| `log_sampling(start, end, total)` | Logarithmic distribution between `start` and `end` with `total` discrete buckets. |

Mixed user + compiler search space (list shape):

```python
search_space = [
    {"config_idx": ss.range(0, len(CONFIGS) - 1)},   # user-defined
    PtxasSearchSpace(version="13.3"),                # compiler-side
]
```

When the search space is a list, the objective receives a list of the same
length — see `compileiq-author-objective` for the unpacking pattern.

## Self-test

```bash
# Default variant resolves
python -c "
from compileiq.search_spaces.compilers import PtxasSearchSpace
p = PtxasSearchSpace().retrieve()
assert p.exists() and p.stat().st_size > 0, p
print(f'default OK: {p}')
"

# Attention variant resolves
python -c "
from compileiq.search_spaces.compilers import PtxasSearchSpace
p = PtxasSearchSpace(version='13.3', variant='att').retrieve()
assert p.exists() and p.stat().st_size > 0, p
print(f'att OK: {p}')
"

# Misconfigured air-gap mirror produces a clear error
CIQ_SEARCH_SPACES_DIR=/nonexistent python -c "
from compileiq.search_spaces.compilers import PtxasSearchSpace
try:
    PtxasSearchSpace().retrieve()
    print('UNEXPECTED: should have failed')
except Exception as e:
    print(f'expected failure: {type(e).__name__}')
"
```

## Gotchas

- **`att` is *attention*, not "attribute".** Earlier docs and a test fixture
  mislabeled the variant as "attribute-based register allocation" — corrected
  in this same branch. When recommending the variant to a user, say
  "attention" explicitly.
- **Providers vs Booster Packs are different things.** Providers return one
  binary search space the search engine consumes during a search; Booster
  Packs are pre-built `.acf` candidates to apply *outside* a search. Don't
  feed a `.acf` from a Booster Pack to `PtxasSearchSpace(...)`.
- **`LocalSearchSpaceBin` doesn't validate `.bin` contents.** It only
  validates that the file exists. A malformed file will fail downstream when
  the engine tries to parse it; the resulting error message points at the
  resolver, not the provider.

## Next

- Writing the objective function that consumes the search space: `compileiq-author-objective`.
- Running the search: `compileiq-run-search`.
