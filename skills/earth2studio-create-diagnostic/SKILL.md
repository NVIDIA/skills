---
name: earth2studio-create-diagnostic
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags: [earth2studio, diagnostic-model, python]
description: >
  Create Earth2Studio diagnostic model wrappers for single-step data
  transformations, including simple derived diagnostics, packaged AutoModel
  diagnostics, and generative or diffusion diagnostics. Do NOT use for
  prognostic time-stepping models, data sources, or installation.
argument-hint: URL or local path to reference inference script (optional)
---

## Quick Start Checklist

Do these steps in order. Do not skip ahead. Before editing, read this
SKILL.md and load the relevant reference files for the model type. After
implementation, run the focused pytest command before saying tests pass. If
tests cannot run, report the exact command and failure instead of claiming
success.

- [ ] Read this SKILL.md completely first
- [ ] Get the reference script, repo, paper, or model documentation (Step 0)
- [ ] Classify the diagnostic as simple, AutoModel, or generative (Step 1)
- [ ] Propose dependency extras before editing dependency files (Step 1)
- [ ] Create `earth2studio/models/dx/<name>.py` with diagnostic-only APIs
- [ ] Create `test/models/dx/test_<name>.py` with mock tests
- [ ] Run: `uv run pytest test/models/dx/test_<name>.py -m "not package" -v`
- [ ] Add/update model extra, install docs, API docs, and changelog when required
- [ ] Run: `make format && make lint && make license`

Critical command rule: always use `uv run` for Python commands:

- Use `uv run pytest ...` and `uv run python ...`
- Do not use bare `pytest` or `python` in repo workflows

If the generated model is wrong, do not keep retrying the same fix. Follow
[Self-Improvement](#self-improvement), patch this skill or its references, then
continue with the corrected workflow.

## Purpose

Implement a diagnostic model wrapper connecting third-party or derived ML
transforms to Earth2Studio. Diagnostic models transform data at a single time
point: input fields in, output fields out, no forecast integration.

## Prerequisites

- Earth2Studio installed via `uv` with dev dependencies (`uv sync --all-extras`)
- Python 3.10+ environment
- Reference inference script, repo, paper, or model documentation
- Checkpoint source and license information for packaged models

## Limitations

- Handles single-step transformations only
- Does not support time-stepping forecast models; use `earth2studio-create-prognostic`
- Real package tests can require network access to NGC, HuggingFace, S3, or other registries
- Generative validation can require GPU and fixed seeds for meaningful comparison

## Diagnostic Model Types

| Type | Inheritance | Dependency extra | Example |
|------|-------------|------------------|---------|
| Simple derived diagnostic | `torch.nn.Module` only | Usually none | `Identity`, wind speed |
| Packaged AutoModel diagnostic | `torch.nn.Module, AutoModelMixin` | Required, even if empty | `PrecipitationAFNO` |
| Generative diagnostic | `torch.nn.Module, AutoModelMixin` | Required, even if empty | `CorrDiff` |

## Workspace

| Context | Location |
|---------|----------|
| Harbor eval | Write to `/workspace/output/earth2studio/models/dx/...` |
| Harbor + `--copy-repo` | Full checkout at `/workspace/repo` |
| Local clone | Directory with `pyproject.toml` |

Never read `evals/targets/`; those files are grader references only.

### Reference Files

Load these files on demand during the matching workflow:

| File | Content | Load at |
|------|---------|---------|
| `references/skeleton-template.py` | Full diagnostic skeletons for simple, AutoModel, and generative wrappers | Steps 3-6 |
| `references/method-templates.py` | Focused coordinate, loading, forward, and device method snippets | Steps 4-6 |
| `references/testing-guide.py` | Mock, package, exception, sample, and seed test patterns | Step 7 |
| `references/validation-guide.md` | Reference comparison, plots, PR hygiene, and review follow-up | Steps 10-11 |
| `references/pr-body-template.md` | PR body template | Step 11 |
| `references/pr-comment-template.md` | Validation comment template | Step 11 |

## Instructions

### Step 0 - Get Reference Material

If `$ARGUMENTS` provides a URL or local path, use it. Otherwise ask:

> Please provide a reference inference script, repository, paper, or model documentation.

Capture the reference model's input variables, output variables, tensor shapes,
normalization, grid, checkpoint source, dependency requirements, and license.

### Step 1 - Analyze Type and Propose Dependencies

Classify the requested diagnostic before editing files:

| If the model... | Then use... |
|-----------------|-------------|
| Computes a derived quantity with no checkpoint | Simple diagnostic |
| Loads weights from `Package` or an external checkpoint | AutoModel diagnostic |
| Produces multiple samples, diffusion outputs, VAE samples, or stochastic super-resolution | Generative diagnostic |

Dependency policy:

- Simple derived diagnostics usually do not need a `pyproject.toml` extra.
- AutoModel and generative diagnostics must have a named optional dependency extra, even if the list is empty.
- Add the extra alphabetically under `[project.optional-dependencies]` and include it in the `all` aggregate.
- Use the model-extra name in `OptionalDependencyFailure("model-extra")` and `@check_optional_dependencies()`.

Present the proposed dependency extra and ask the user to approve before editing
`pyproject.toml`:

```toml
model-name = ["package1>=version", "package2"]
# or, when the packaged diagnostic needs no extra runtime packages:
model-name = []
```

### Step 2 - Add Dependencies

After approval, edit `pyproject.toml`:

- Add the extra alphabetically.
- Update the `all` aggregate.
- Prefer minimum supported versions from the reference package documentation.
- Do not add broad unpinned Git dependencies unless the reference model requires them and the user approves.

### Step 3 - Create Model File

File: `earth2studio/models/dx/<lowercase>.py`

Use the repo-standard SPDX/license header shown in existing model files.

Simple diagnostic imports commonly include:

```python
from collections import OrderedDict
import numpy as np
import torch
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem
```

Packaged and generative diagnostics commonly also include:

```python
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils.imports import OptionalDependencyFailure, check_optional_dependencies
from loguru import logger
```

Canonical method order:

1. `__init__`
2. `input_coords`
3. `output_coords` decorated with `@batch_coords()`
4. `__str__` if useful
5. `load_default_package` for AutoModel/generative diagnostics
6. `load_model` for AutoModel/generative diagnostics
7. `to` only when non-PyTorch state must move devices
8. Private/support methods
9. `__call__` decorated with `@torch.inference_mode()` and `@batch_func()`

Avoid shared base classes or broad abstractions unless the wrapper naturally has
multiple closely related variants where a small base class reduces duplication.

### Step 4 - Implement Coordinates

Diagnostic input coordinates usually use this public Earth2Studio order:

1. `batch`: `np.empty(0)` and first in the `OrderedDict`
2. `variable`: input variable names using Earth2Studio vocabulary names
3. `lat`: public latitude convention north-to-south, usually `90` to `-90`
4. `lon`: public longitude convention `0` to `360`, endpoint normally false

No diagnostic wrapper should expose `lead_time`. If a diagnostic needs validity
time metadata, document it as per-sample metadata in `coords["time"]`; do not make
it a tensor dimension unless an existing dx pattern requires it.

`output_coords` must validate inputs with `handshake_dim` and `handshake_coords`.
Then update output variables and, when needed, output lat/lon resolution.
Generative diagnostics must add a `sample` dimension after `batch`.

### Step 5 - Implement Forward Pass

Use a single-step `__call__`; never create an iterator. Validate coordinates
before model execution, then return `(output_tensor, output_coords)`.

```python
@torch.inference_mode()
@batch_func()
def __call__(self, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
    output_coords = self.output_coords(coords)
    x = (x - self.center) / self.scale
    out = self.core_model(x)
    return out, output_coords
```

For generative diagnostics, loop over the batch dimension and generate
`number_of_samples` per input item. Use explicit seeds for reproducibility when
the reference implementation supports seeded sampling.

### Step 6 - Implement Model Loading

For packaged diagnostics:

- `load_default_package` should lock HuggingFace URLs to a commit (`hf://org/repo@commit`) or NGC/S3 versions to an immutable release.
- `load_model` should call `package.resolve(...)`, load checkpoints on CPU first, set modules to `eval()`, and disable gradients where appropriate.
- Use `weights_only=False` only when loading a pickled full PyTorch object is required.
- Decorate optional model classes and `load_model` with `@check_optional_dependencies()`.
- Use `loguru.logger` for useful loading messages; do not use `print()` inside `earth2studio/`.

### Step 7 - Write Tests

File: `test/models/dx/test_<name>.py`

Required tests:

| Function | Purpose |
|----------|---------|
| `test_<model>_call` | Forward pass with mock or simple model |
| `test_<model>_exceptions` | Invalid coordinate order, values, or variables raise errors |
| `test_<model>_package` | Real weights with `@pytest.mark.package` for AutoModel/generative diagnostics |

Generative diagnostics also require sample-count and deterministic-seed tests.
Use `references/testing-guide.py`. Create a `Phoo<ModelName>` dummy that matches
the real core model's interface and produces deterministic output.

Run focused tests:

```bash
uv run pytest test/models/dx/test_<name>.py -m "not package" -v
uv run pytest test/models/dx/test_<name>.py::test_<model>_package --package -v
```

Do not omit package tests for packaged models. If arbitrary random inputs are not
physically valid for the real checkpoint, build a stable model-appropriate input
while still loading real weights and running a forward pass.

### Step 8 - Register Model

For public models, update `earth2studio/models/dx/__init__.py` alphabetically.
Skip registration only when the user explicitly wants an internal or experimental
file that should not be exported.

### Step 9 - Documentation

For public models:

- Add to `docs/modules/models_dx.rst` alphabetically so API docs include the generated page.
- Add to `docs/userguide/about/install.md` if a model extra exists. Include model notes plus both `pip install earth2studio[model-name]` and `uv add earth2studio --extra model-name` instructions.
- Update `CHANGELOG.md` under `### Added`.

Format and lint:

```bash
make format && make lint && make license
```

### Step 10 - Validation (if requested)

Follow `references/validation-guide.md`. Create uncommitted vanilla,
Earth2Studio, comparison, and sanity-check scripts. Do not commit generated
outputs, checkpoints, images, or local validation scripts.

For generative diagnostics, fix seeds and compare matching samples or report
statistical/tolerance-based agreement when exact equality is impossible.
Ask the user to visually inspect plots before proceeding.

### Step 11 - PR (if requested)

Follow `references/validation-guide.md` and use:

- `references/pr-body-template.md`
- `references/pr-comment-template.md`

Before creating the PR, verify dependency extras, `all`, install docs, API docs,
changelog, tests, and validation artifacts are consistent. Do not include machine
names, hostnames, absolute paths, cache paths, device inventory, or uploaded image
links in PR text. Use plot placeholders for manual image upload.

## Examples

### Simple Diagnostic

```text
User: Create a diagnostic that computes wind speed from u10m and v10m.
Agent: Reads SKILL.md, classifies as simple, creates windspeed.py with only
       torch.nn.Module, writes call and exception tests, runs focused pytest.
```

### AutoModel Diagnostic

```text
User: Add a precipitation estimator from this reference script.
Agent: Reads SKILL.md and references, proposes dependency extra, creates a
       torch.nn.Module + AutoModelMixin wrapper, writes mock/package tests,
       updates docs/changelog/dependencies, and runs validation commands.
```

### Generative Diagnostic

```text
User: Wrap this diffusion super-resolution model.
Agent: Classifies as generative, adds sample output coordinates, supports seed
       handling, writes sample and deterministic-seed tests, and prepares seeded
       validation comparisons.
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `OptionalDependencyFailure` | Install with `uv sync --extra <model-extra>` or fix the extra name |
| Coordinate handshake fails | Check `OrderedDict` order and `handshake_dim` indices |
| Wrong output shape | Verify `output_coords` lengths match returned tensor shape |
| `ModuleNotFoundError: pytest` | Use `uv run pytest`, not bare `pytest` |
| Package test fails on random input | Use a stable physically plausible input while still loading real weights |

## Reminders

Do:

- Use `uv run python` and `uv run pytest` for all Python commands.
- Use `@batch_coords()` on `output_coords`.
- Use `@torch.inference_mode()` and `@batch_func()` on `__call__`.
- Keep `batch` as the first coordinate with `np.empty(0)` in `input_coords`.
- Validate coordinates with `handshake_dim()` and `handshake_coords()`.
- Add `sample` in generative `output_coords`.
- Include the repo-standard SPDX/license header in every Python file.
- Use `loguru.logger`, never `print()`, inside `earth2studio/`.

Do not:

- Inherit from `PrognosticMixin`.
- Include `lead_time` coordinates.
- Create `create_iterator`.
- Create general base classes for a single wrapper without a clear multi-variant need.
- Commit API keys, credentials, validation scripts, plots, or generated outputs.
- Read from `evals/targets/`.

## Self-Improvement

If this skill produces incorrect outputs, update it before continuing:

1. Identify the issue in the generated code or workflow.
2. Edit `SKILL.md` or the relevant file in `references/` to fix the guidance.
3. Run focused validation for the changed skill files.
4. Commit the skill fix separately when working in a branch that expects commits.
5. Continue the model implementation with the corrected workflow.
