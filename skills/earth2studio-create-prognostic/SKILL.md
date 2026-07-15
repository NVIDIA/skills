---
name: earth2studio-create-prognostic
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags: [earth2studio, prognostic-model, python]
description: >
  Create Earth2Studio prognostic (time-stepping forecast) model wrappers.
  Do NOT use for diagnostic models, data sources, or installation.
argument-hint: URL or local path to reference inference script (optional)
---

## Quick Start Checklist

**Do these steps IN ORDER. Do not skip any step.**

- [ ] Read this SKILL.md completely first
- [ ] Get reference script (Step 0)
- [ ] Create `earth2studio/models/px/<name>.py` with triple inheritance
- [ ] Create `test/models/px/test_<name>.py` with mock tests
- [ ] Run: `uv run pytest test/models/px/test_<name>.py -v`
- [ ] Add/update model extra, install docs, API docs, and changelog (Steps 1-2, 9)
- [ ] Run: `make format && make lint`

> **⚠️ CRITICAL:** Always use `uv run` for Python commands:
> - ✅ `uv run pytest ...` / `uv run python ...`
> - ❌ `pytest ...` / `python ...` (missing dependencies)
>
> **Stuck or wrong output:** Do not keep retrying the same fix. Follow
> [Self-Improvement](#self-improvement) to patch this skill before continuing.

## Purpose

Implement a prognostic model wrapper connecting third-party ML weather models
to Earth2Studio. Prognostic models time-integrate forward—given initial state,
they predict future states by stepping through time (e.g., 6-hour increments).

## Workspace

| Context | Location |
|---------|----------|
| Harbor eval | Write to `/workspace/output/earth2studio/models/px/...` |
| Harbor + `--copy-repo` | Full checkout at `/workspace/repo` |
| Local clone | Directory with `pyproject.toml` |

**Never read `evals/targets/`** — grader references only.

### Reference Files

Load on demand during the matching step:

| File | Content | Load at |
|------|---------|---------|
| `references/skeleton-template.py` | Full model skeleton with FILL comments | Steps 3–6 |
| `references/method-templates.py` | Canonical method implementations | Steps 4–6 |
| `references/testing-guide.py` | Test skeleton and mock patterns | Step 7 |
| `references/validation-guide.md` | Comparison scripts, PR, code review | Steps 10–11 |

---

## Workflow Steps

### Step 0 — Get Reference Script

If `$ARGUMENTS` provided, use it. Otherwise ask:
> Please provide a reference inference script URL/path.

### Step 1 — Analyze & Propose Dependencies

Analyze: packages, architecture, I/O shapes, time step, resolution, checkpoint.

Propose `pyproject.toml` group (alphabetical, add to `all`). Every
prognostic model must have an optional dependency extra, even when no packages
are required:
```toml
model-name = ["package1>=version", "package2"]
# or, when no additional packages are required:
model-name = []
```

**[CONFIRM]** Present dependencies and ask user to approve.

### Step 2 — Add Dependencies

Edit `pyproject.toml`: add the model extra alphabetically, even if it is
empty, and update the `all` aggregate.

### Step 3 — Create Model File

**File:** `earth2studio/models/px/<lowercase>.py`

**Required inheritance (all three):**
```python
class ModelName(torch.nn.Module, AutoModelMixin, PrognosticMixin):
```

**Required imports:**
```python
import numpy as np
import torch
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticMixin
from earth2studio.models.utils import create_coords_from_lat_lon, handshake_dim
from earth2studio.lexicon import E2STUDIO_VOCAB
from earth2studio.utils import check_optional_dependencies
from loguru import logger
```

**SPDX header (required at top of every .py file):**
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
```

**Canonical method order:**
1. `__init__` 2. `input_coords` 3. `output_coords` (@batch_coords)
4. `load_default_package` 5. `load_model` 6. `to` (optional)
7. Private methods 8. `__call__` (@batch_func) 9. `_default_generator`
10. `create_iterator`

### Step 4 — Implement Coordinates

**input_coords rules:**
- `batch`: `np.empty(0)`
- `time`: `np.empty(0)` (dynamic)
- `lead_time`: starts at `np.timedelta64(0, "h")`
- `lat`: 90 to -90 (north to south); this is the public Earth2Studio convention even if the source model uses the opposite order
- `lon`: 0 to 360
- If a checkpoint/model core expects south-to-north latitude, flip tensors internally before/after the core model; do not expose flipped latitude in `input_coords` or `output_coords`
- Map variables to `E2STUDIO_VOCAB` (282 entries in `earth2studio/lexicon/base.py`)

**output_coords:** Use `handshake_dim`/`handshake_coords` for input validation, then increment `lead_time`. Prefer a shared coordinate-check helper and call it from `output_coords`, `__call__`, and iterator setup before model execution.

### Step 5 — Implement Forward Pass

**`__call__`:** @batch_func decorated, shape (batch, time, lead_time, var, lat, lon).
Reshape to model format → call model → reshape back.

**`create_iterator`:** MUST yield initial condition first (step 0).
Use `front_hook`/`rear_hook` for perturbation injection.

### Step 6 — Implement Model Loading

**`load_default_package`:** Lock HuggingFace URLs: `hf://org/repo@commit`

**`load_model`:** Use `package.resolve()`, `map_location="cpu"`, `eval()` mode,
decorate with `@check_optional_dependencies()`.

### Step 7 — Write Tests

**File:** `test/models/px/test_<name>.py`

**Required tests:**
| Function | Purpose |
|----------|---------|
| `test_<model>_call` | Single forward pass (parametrize device/time) |
| `test_<model>_iter` | Iterator produces sequence |
| `test_<model>_exceptions` | Invalid coords raise errors |
| `test_<model>_package` | Real weights (`@pytest.mark.package`) |

Create `PhooModelName` dummy matching interface for mock tests.

**Run tests:**
```bash
uv run pytest test/models/px/test_<name>.py -m "not package" -v
uv run pytest test/models/px/test_<name>.py::test_<model>_package --package -v
```

Do not omit the package test. If arbitrary random inputs are not physically
valid for the real checkpoint, use a stable model-appropriate synthetic input
while still loading real weights and running a forward pass.

### Step 8 — Register Model (if requested)

- Add to `earth2studio/models/px/__init__.py` (alphabetical)
- Verify deps in pyproject.toml

### Step 9 — Documentation

- Add to `docs/modules/models_px.rst` (alphabetical). This is required for
  every new prognostic model so the API docs include the generated page.
- Add to `docs/userguide/about/install.md` (alphabetical tab) for the
  model extra, even when the extra is empty. Include model-specific notes plus
  both `pip install earth2studio[model-name]` and
  `uv add earth2studio --extra model-name` instructions.
- Update `CHANGELOG.md` under `### Added`. This is required for every new
  prognostic model.

**Format and lint:**
```bash
make format && make lint && make license
```

### Step 10 - Validation (if requested)

Follow `references/validation-guide.md`. Create uncommitted vanilla, E2S,
comparison, and sanity-check scripts; do not commit generated outputs or images.
Use PR-safe placeholders for plots so the user can upload images manually.

**[CONFIRM]** User must visually inspect plots before proceeding.

### Step 11 - PR (if requested)

Follow `references/validation-guide.md` and use:
- `references/pr-body-template.md`
- `references/pr-comment-template.md`

Before creating the PR, verify `pyproject.toml` has the model extra, the
`all` extra includes it, install docs include both pip and uv commands, and
`docs/modules/models_px.rst` plus `CHANGELOG.md` are updated.

Do not include machine names, absolute paths, device inventory, or uploaded image
links in PR text. Use plot placeholders instead.

---

## Examples

### Simple Identity Model
```text
User: Create IdentityModel - returns input unchanged, 6h step, 181x360, vars: t2m, u10m, v10m, msl

Agent: [reads SKILL.md, creates identity.py with triple inheritance,
        creates test_identity.py, runs pytest, runs make format && lint]
```

### External Model (Pangu)
```text
User: Add Pangu-Weather wrapper
      GitHub: https://github.com/198808xc/Pangu-Weather

Agent: [reads SKILL.md, fetches inference.py, creates pangu.py,
        creates test_pangu.py, runs pytest]
```

---

## Key Patterns

### Coordinate Template
```python
@property
def input_coords(self) -> CoordSystem:
    return CoordSystem({
        "batch": np.empty(0),
        "time": np.empty(0),
        "lead_time": np.array([np.timedelta64(0, "h")]),
        "variable": np.array(["t2m", "u10m", ...]),
        # Public Earth2Studio convention is north-to-south latitude.
        "lat": np.linspace(90, -90, 181),
        "lon": np.linspace(0, 359, 360),
    })

@batch_coords()
def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
    output = input_coords.copy()
    output["lead_time"] = input_coords["lead_time"] + np.timedelta64(6, "h")
    return output
```

### Iterator Template
```python
def create_iterator(self, x, coords):
    yield x, coords  # Initial condition (step 0)
    while True:
        x, coords = self.front_hook(x, coords)
        x, coords = self(x, coords)
        x, coords = self.rear_hook(x, coords)
        yield x, coords
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `OptionalDependencyFailure` | `uv add --optional <group> <pkg>` |
| Coordinate handshake fails | Check `handshake_dim` indices match dim position |
| Iterator wrong shapes | Debug reshape logic with random input |
| `ModuleNotFoundError: pytest` | Use `uv run pytest` not `pytest` |

---

## Reminders

**DO:**
- Use `uv run python` for ALL Python commands
- Use `loguru.logger`, never `print()`
- Inherit `torch.nn.Module + AutoModelMixin + PrognosticMixin`
- Yield initial condition first in `create_iterator`
- Use `front_hook()`/`rear_hook()` in `_default_generator`
- Include SPDX header in every .py file

**DON'T:**
- Create general base classes for reuse
- Commit API keys or comparison scripts
- Read from `evals/targets/`
