---
name: earth2studio-create-datasource
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags:
    - earth2studio
    - earth2
    - python
    - data-source
    - forecast-source
    - integration
description: >
  Create and validate Earth2Studio data source wrappers (DataSource,
  ForecastSource, DataFrameSource, ForecastFrameSource) from remote stores.
  Do NOT use for fetching data with existing sources, model inference, or
  installation tasks.
argument-hint: URL or description of remote data store (optional)
---

# Create and Validate Data Source

## Purpose

End-to-end workflow for implementing a new Earth2Studio data source wrapper
that connects a remote data store (S3, GCS, Azure, HTTP, HuggingFace) to
Earth2Studio's async data fetching infrastructure — from analysis through
implementation, testing, validation, and PR submission.

## Prerequisites

- Earth2Studio dev environment with `uv` (`uv run python` must work)
- Git configured with fork (`origin`) and upstream (`upstream`) remotes
- Access to the target remote data store (credentials if private)
- Python 3.10+

## Workspace

Use the directory containing `pyproject.toml`. For Harbor evals, write to
`/workspace/output/` preserving paths. Never read `evals/targets/`.

## Instructions

> **Python Environment:** Always use `uv run python` or the local `.venv`.
> Never use the system Python directly.

Follow every step in order.

> **[CONFIRM] gates:** Only Step 1 (Source Type) and Step 12 (Sanity-Check
> Plots) require explicit user approval. All other `[CONFIRM]` markers are
> advisory — present decisions inline and proceed without blocking.
>
> **Deliverables first:** Write the source file and test file (Steps 6–7)
> before extended exploration, documentation, registration, CHANGELOG, or PR
> work. Skip Steps 8–14 when the user asks for implementation only.
>
> **Before you finish:** Run verification commands in the repo root so results
> appear in the session log:
>
> ```bash
> uv run pytest test/data/test_<source>.py -x
> make format && make lint
> ```
>
> **Be concise:** Avoid long architecture reports; summarize decisions in a
> few sentences and move on to file writes.
>
> **Hangs or User Feedback** If agent becomes stuck or user provides a
> correction during this skills use, conservatively review relevant part of
> the skill and improve. Be concise.
>
> **One source type per invocation.** Invoke again for companion types.

### Reference Files

Load these on demand during the relevant steps:

| File | Content | Load at |
|---|---|---|
| `references/implementation-guide.py` | Skeleton source with FILL comments | Steps 3–10 |
| `references/testing-guide.py` | Test skeleton with FILL comments | Step 11 |
| `references/validation-guide.md` | Plot templates, PR body template, Greptile handling | Steps 12–14 (optional, for templates) |

---

### Workflow Overview

```text
Step 0: Obtain reference → Step 1: Determine type → Step 2: Dependencies
→ Step 3: Add deps → Step 4: Create lexicon → Step 5: Update vocab/schema
→ Step 6: Create skeleton → Step 7: Implement source → Step 8: Register
→ Step 9: Documentation → Step 10: CHANGELOG → Step 11: Tests
→ Step 12: Validate & plots (user confirms) → Step 13: PR + sanity comment
→ Step 14: Greptile review
```

---

### Step 0 — Obtain Remote Data Store Reference

If `$ARGUMENTS` is provided, use it (URL → WebFetch; file path → read).

If empty, ask:

> Please provide a URL, API documentation link, or description of the
> remote data store. This will be used to understand storage format,
> access pattern, variable inventory, temporal/spatial resolution.

---

### Step 1 — Determine Source Type

| Protocol | Returns | Has `lead_time`? | Use |
|---|---|---|---|
| **DataSource** | `xr.DataArray` | No | Gridded analysis/reanalysis |
| **ForecastSource** | `xr.DataArray` | Yes | Gridded forecast |
| **DataFrameSource** | `pd.DataFrame` | No | Sparse/station obs |
| **ForecastFrameSource** | `pd.DataFrame` | Yes | Sparse forecast obs |

Key factors: gridded vs sparse → DataArray vs DataFrame; analysis vs forecast → Source vs ForecastSource.

#### [CONFIRM — Source Type]

Present recommended type with justification. Ask for confirmation.

---

### Step 2 — Examine Remote Store & Propose Dependencies

**Analyze:** storage backend, file format, authentication, access pattern,
temporal/spatial resolution, variable inventory.

**Prefer fsspec:**

| Backend | Preferred | Avoid |
|---|---|---|
| AWS S3 | `s3fs` (core dep) | `boto3` directly |
| GCS | `gcsfs` (core dep) | `google-cloud-storage` |
| Azure | `adlfs` | `azure-storage-blob` |
| HTTP | `fsspec` (core dep) | `requests` |
| HuggingFace | `huggingface_hub` (core dep) | custom scripts |

Only fall back to dedicated libraries when fsspec cannot access the store.

Check `pyproject.toml` — only propose packages not already present.
Core deps include: `s3fs`, `gcsfs`, `fsspec`, `zarr`, `netCDF4`, `h5py`,
`pygrib`, `huggingface-hub`, `pandas`, `pyarrow`.

#### [CONFIRM — Dependencies & Access Pattern]

Present: backend, fsspec filesystem, new packages (with license), auth method.

---

### Step 3 — Add Dependencies

> **Load `references/implementation-guide.py` from here through Step 10.**

If new packages needed:

1. `uv add --extra data <package>`
2. `uv lock`
3. Add optional dependency imports using `OptionalDependencyFailure` pattern

---

### Step 4 — Create Lexicon Class

Create `earth2studio/lexicon/<source_name>.py` with:

- `metaclass=LexiconType`
- `VOCAB: dict[str, str]` mapping E2S names → remote keys
- `get_item(cls, val)` returning `tuple[str, Callable]`
- Use `::` separator for structured keys

Map remote variables against `E2STUDIO_VOCAB` (282 entries in
`earth2studio/lexicon/base.py`).

#### [CONFIRM — Lexicon & Variable Mapping]

Present: class name, key format, full mapping table, modifiers, reference URL.

---

### Step 5 — Update E2STUDIO_VOCAB / SCHEMA (if needed)

- New vocab: surface = descriptive abbrev; pressure = `{name}{level}`
- New schema fields: DataFrame sources only, check `E2STUDIO_SCHEMA`

#### [CONFIRM — Vocabulary & Schema Updates]

Skip if no updates needed.

---

### Step 6 — Create Skeleton Data Source File

Follow canonical method ordering:

1. Class constants
2. SCHEMA
3. `__init__`
4. `_async_init`
5. `__call__`
6. `fetch`
7. `_create_tasks`
8. `fetch_wrapper`
9. `fetch_array`
10. `_validate_time`
11. Helpers
12. `cache` property
13. `available` classmethod

Use async task dataclass pattern for parallel execution.

#### [CONFIRM — Skeleton]

Present: class name, file path, skeleton code, task dataclass.

---

### Step 7 — Implement the Data Source & Tests

> **The test file is a co-equal deliverable.** Create `test/data/test_<filename>.py`
> alongside the source. Add `test_<source>_call_mock` for async sources.

**Sync sources:** Use `prep_data_inputs`/`prep_forecast_inputs`, direct `__call__`.

**Async sources:** See `references/implementation-guide.py` for required patterns:
`_sync_async`, `managed_session`, `gather_with_concurrency`, `async_retry`,
pure async I/O, `try/finally` cleanup. Constructor params: `cache=True`,
`verbose=True`, `async_timeout=600`, `async_workers=16`, `retries=3`.
DataFrame sources add `time_tolerance`.

---

### Step 8 — Register the Source

- `earth2studio/data/__init__.py` — alphabetical import
- `earth2studio/lexicon/__init__.py` — alphabetical import
- Verify `pyproject.toml` deps

---

### Step 9 — Update Documentation

- Add to correct RST file (`datasources_analysis.rst` / `_forecast.rst` / `_dataframe.rst`)
- Class docstring: Parameters, Warning (download size), Note (reference URLs), Badges (last)
- All public methods: NumPy-style docstrings

---

### Step 10 — Update CHANGELOG.md

Add entry under the current unreleased version. See
`references/implementation-guide.py` REGISTRATION CHECKLIST for the format.

One line per source. Do NOT add separate lexicon entries.

---

### Step 11 — Verify Style & Expand Tests

Run `make format && make lint && make license`. Load `references/testing-guide.py`
for test skeletons. Required tests: `test_<source>_fetch` (slow), `_cache` (slow),
`_call_mock`, `_exceptions`, `_available`. Target 90%+ coverage with `--slow`.

#### [CONFIRM — Tests]

Present test file, functions, coverage.

---

### Step 12 — Validate Variables & Sanity-Check

1. Validate all lexicon vars against real data (run script, do NOT commit)
2. Remove variables with < 10% valid data
3. Create sanity-check plot (gridded or sparse template)
4. Tell user the plot path and ask for visual confirmation

#### [CONFIRM — Sanity-Check Plots]

User MUST visually inspect plots. Do not proceed without confirmation.

---

### Step 13 — Branch, Commit & Open PR

1. Create branch `feat/data-source-<name>`
2. Commit (do NOT add sanity-check script/images)
3. Push to fork
4. `gh pr create --repo NVIDIA/earth2studio`
5. **Immediately post sanity-check validation as PR comment** with:
   - Variable coverage table (name, count, range, unit)
   - Data validation summary (regions, storms/stations, time range)
   - Key findings (physically reasonable values, conversions verified)
   - Full validation script in `<details>` block
   - Image placeholder: `<!-- Drag and drop sanity-check image here -->`

#### [CONFIRM — Ready to Submit]

Verify all steps complete before creating PR.

---

### Step 14 — Automated Code Review

1. Poll for Greptile review (5 min timeout)
2. Categorize feedback (bug/style/perf/docs/suggestion/false-positive)
3. Present triage table to user
4. Implement accepted fixes
5. Respond to PR comments
6. Push

#### [CONFIRM — Review Triage]

User approves which comments to address.

---

## Examples

```text
User: Add a data source for the NOAA GFS analysis on S3
Agent: [loads skill, proceeds through Steps 0–14]
```

---

## Limitations

- One source type per invocation
- Requires network access for validation (Step 12)
- SPDX license headers required in all files

---

## Reminders

**DO:** `uv run python`, `loguru.logger`, alphabetical order in `__init__.py`/RST/CHANGELOG,
canonical method ordering, async utilities (`managed_session`, `gather_with_concurrency`,
`async_retry`), pure async I/O, reference URLs in docstrings, `try/finally` cleanup.

**AVOID:** `asyncio.to_thread`, bare `tqdm.gather`, xarray for loading, full file downloads.

**NEVER:** `loop.set_default_executor()`, commit secrets, commit sanity-check scripts/images.
