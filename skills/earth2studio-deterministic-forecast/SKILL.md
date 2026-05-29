---
name: earth2studio-deterministic-forecast
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - inference
    - forecast
    - deterministic
description: >
  Build deterministic forecast scripts with Earth2Studio (model, data source,
  IO, inference). Do NOT use for ensemble, diagnostics, data-only fetch, or
  install.
---

# Earth2Studio Deterministic Forecast Skill

## Purpose

Guide users through building deterministic (single-member) weather forecast
inference scripts with Earth2Studio. Covers model selection, data source
compatibility, IO backend choice, nsteps calculation, and generating a
complete script following `earth2studio.run.deterministic`.

## Prerequisites

- Earth2Studio installed (`pip install earth2studio` or `uv add earth2studio`)
- CUDA-capable GPU with sufficient VRAM for the chosen model
- Network access for model weight download and data fetching
- Python 3.10+

## Instructions

You are helping a user build a deterministic forecast inference script
using Earth2Studio. The script follows the structure of
`earth2studio.run.deterministic` — a pipeline that takes a prognostic
model, fetches initial conditions from a data source, steps the model
forward, and writes output to an IO backend.

## Core principle: live docs drive every recommendation

Model availability, data source APIs, and IO backends change between
releases. Before recommending any component, fetch the relevant live
doc page to confirm it exists and check its current interface.

Live doc references (fetch only what the current step requires):

| Component | URL |
|-----------|-----|
| Prognostic models | <https://nvidia.github.io/earth2studio/modules/models_px.html> |
| Data sources (analysis) | <https://nvidia.github.io/earth2studio/modules/datasources_analysis.html> |
| Data sources (forecast) | <https://nvidia.github.io/earth2studio/modules/datasources_forecast.html> |
| IO backends | <https://nvidia.github.io/earth2studio/modules/io.html> |
| `run.deterministic` source | <https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/run.py> |
| Lexicon (variable compat) | <https://github.com/NVIDIA/earth2studio/tree/main/earth2studio/lexicon> |

## Interaction protocol

### Step 1. Understand forecast requirements

Ask the user (cap at 3 questions, skip what's already answered):

1. **Time horizon** — how far ahead? Hours (nowcast), days
   (medium-range), weeks/months (seasonal)?
2. **Variables of interest** — what do they want to predict?
   (temperature, wind, geopotential, precipitation, etc.)
3. **Region** — global or regional (e.g. CONUS for HRRR-based models)?
4. **Hardware** — what GPU / VRAM do they have? (filters model choices)

### Step 2. Select prognostic model

Fetch the prognostic models page. Filter candidates by:

- **Time horizon** → model class badge (NWC, MR, S2S, CM)
- **Region** → region badge (Global, NA, etc.)
- **VRAM** → rec VRAM badge
- **Variables** → check model's `input_coords`/`output_coords` against what the user needs

Present 2–4 candidate models with tradeoffs (resolution, speed, accuracy, VRAM). Let the user choose.

Once selected, note the model's:

- Required input variables (from `input_coords["variable"]`)
- Time step size (from `output_coords["lead_time"]`)
- These determine `nsteps` and constrain which data sources work

### Step 3. Select data source

The data source must provide the model's required input variables. Fetch
the analysis data source page (or forecast source page if comparing
against operational forecasts).

Verify compatibility:

1. Fetch the candidate source's lexicon from
   `earth2studio/lexicon/<source>.py`
2. Confirm all variables in the model's `input_coords["variable"]`
   exist as keys in the source's VOCAB

Present viable options. Common pairings:

- Global models (AIFS, Pangu, GraphCast, SFNO, etc.) →
  GFS, ARCO, CDS, WB2ERA5, IFS
- Regional models (StormCast, HRRR-based) → HRRR
- Historical/research runs → ARCO, CDS, WB2ERA5, NCAR_ERA5

Let the user choose. Confirm the initialization time(s) they want to forecast from.

### Step 4. Select IO backend

Present the available IO backends (fetch the IO page to confirm current list):

| Backend | Best for |
|---------|----------|
| ZarrBackend | Large outputs, chunked storage, recommended default |
| AsyncZarrBackend | Same as Zarr but async writes for performance |
| NetCDF4Backend | Compatibility with legacy tools |
| XarrayBackend | In-memory, small runs, interactive exploration |
| KVBackend | Key-value dict, debugging |

Recommend ZarrBackend unless the user has a specific reason for another. Ask where they want output saved.

### Step 5. Determine nsteps

Calculate `nsteps` from:

- User's desired forecast horizon (e.g. 5 days)
- Model's time step (e.g. 6 hours for most global models)
- `nsteps = forecast_hours / model_step_hours`

Confirm with the user: *"For a 5-day forecast with a 6-hour time step, that's 20 steps. Correct?"*

### Step 6. Generate the inference script

Write a complete Python script following the `earth2studio.run.deterministic` pattern. The script structure:

```python
import datetime
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.px import <ModelClass>
from earth2studio.data import <DataSourceClass>
from earth2studio.io import <IOBackendClass>
from earth2studio.run import deterministic

# 1. Initialize model
model = <ModelClass>.load_model(<ModelClass>.load_default_package())

# 2. Initialize data source
data = <DataSourceClass>()

# 3. Initialize IO backend
io = <IOBackendClass>("<output_path>")

# 4. (Optional) Subselect output variables/coords
output_coords = OrderedDict({
    "variable": np.array(["t2m", "u10m", ...]),  # only save these
})

# 5. Run deterministic forecast
io = deterministic(
    time=["YYYY-MM-DDTHH:MM:SS"],
    nsteps=<N>,
    prognostic=model,
    data=data,
    io=io,
    output_coords=output_coords,  # optional
    device=torch.device("cuda"),
)

# 6. Post-run: inspect results
print("Forecast complete. Output at: <output_path>")
```

**Before writing the script**, fetch the specific model's doc page
to confirm:

- The correct class import path
- How to load the model (`load_model` + `load_default_package()`
  is the standard pattern but verify)
- Any model-specific constructor arguments

Also fetch the data source's doc page to confirm constructor arguments
(some need cache paths, tokens, etc.).

### Step 7. Explain the script and next steps

After delivering the script, explain:

- How to change the forecast time (just edit the `time` list)
- How to run multiple initializations (add more entries to `time`)
- How to subset output variables via `output_coords`
- Where the output is saved and how to read it back
  (e.g. `xr.open_zarr(...)`)
- If they want to add diagnostics on top, point them to the
  `diagnostic` workflow pattern

## Ownership and out-of-scope

**Owns:** prognostic model selection for deterministic forecasts, data
source compatibility verification, IO backend selection, nsteps
calculation, generating the complete inference script following
`earth2studio.run.deterministic` structure.

**Does not own:** ensemble workflows, diagnostic model chaining,
data-only fetch (earth2studio-data-fetch),
installation (earth2studio-install), model training or fine-tuning,
custom model development.

## Examples

**Typical invocation:**
> "Run a 5-day global forecast with Pangu-Weather starting from
> today's GFS analysis, saving output to Zarr."

The skill walks through Steps 1-7: confirms requirements, selects Pangu24,
pairs with GFS data source, picks ZarrBackend, calculates nsteps=5 (24h steps),
generates the script, and explains how to inspect results.

## Limitations

- Only deterministic (single-member) forecasts; use ensemble workflow for
  probabilistic runs
- Cannot train or fine-tune models — inference only
- Model weights require first-time download (several GB depending on model)
- Regional models (e.g. StormCast) require matching regional data sources
- GPU required; CPU-only inference is not supported for most models

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError` on variable | Lexicon missing variable | Check compat; pick different source |
| `OutOfMemoryError` | VRAM exceeded | Use smaller model or free cache |
| `FileNotFoundError` package | Weights not cached | Call `load_default_package()` first |
| `TimeoutError` data fetch | API slow/unreachable | Retry or use cached source |
| `ValueError: nsteps` | Horizon < model step | Increase horizon or finer model |
