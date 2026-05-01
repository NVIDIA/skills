---
name: testing
description: Testing reference for Megatron Bridge — unit and functional test layout, tier semantics (L0/L1/L2/flaky), script conventions, running tests locally, adding/moving/disabling tests, and pytest conventions.
when_to_use: Adding, running, moving, or disabling tests; debugging a test failure; choosing the right test tier; understanding L0 vs L1 vs L2; handling flaky tests; 'add a test', 'which tier', 'functional test layout'.
---

# Testing

## Directory Layout

```
tests/
  unit_tests/          # fast, isolated, no GPU required
  functional_tests/
    launch_scripts/
      h100/
        active/        # H100 tests that run in CI automatically
        flaky/         # H100 tests quarantined from blocking CI
      gb200/
        active/        # GB200 tests that run in CI automatically
        flaky/         # GB200 tests quarantined from blocking CI
```

Unit tests are independent of the launch script layout. Functional test
scripts are named `{Tier}_{Description}.sh` (e.g., `L0_Launch_training.sh`).

## Tier Semantics

| Tier | Trigger | Blocking |
|---|---|---|
| L0 | Every PR, every push to `main`, schedule | Yes — PR cannot merge if L0 fails |
| L1 | Push to `main`, schedule, PRs with `needs-more-tests` label | Yes |
| L2 | Schedule and `workflow_dispatch` only | Yes (when triggered) |
| flaky | `workflow_dispatch` with `test_suite=all` only | No — failures are informational |

H100 and GB200 each have independent L0/L1/L2/flaky jobs. Moving a script to
`flaky/` removes it from blocking CI on that hardware target only.

**Prefer unit tests over functional tests.** CI GPU resources are limited;
every functional test slot has a real cost.

## Running Tests Locally

### Unit Tests

No GPU required:

```bash
uv run pytest tests/unit_tests/ -x -v
```

Or inside Docker:

```bash
docker run --rm --gpus all -v $(pwd):/workdir/ -w /workdir/ megatron-bridge \
  uv run pytest tests/unit_tests/
```

### Functional Tests

Run the corresponding launch script directly on a GPU node:

```bash
bash tests/functional_tests/launch_scripts/h100/active/L0_Launch_training.sh
```

## Adding a Unit Test

1. Place the file under `tests/unit_tests/<domain>/test_<name>.py`.
2. Mark it: `@pytest.mark.unit`.
3. Keep configs tiny: small hidden dims, 1-2 layers, short sequences.
4. Run locally: `uv run python -m pytest tests/unit_tests/<your_test>.py`

**No foreign `setattr` on config dataclasses.** When applying overrides via
`setattr(config_obj, key, value)`, always guard first:

```python
if not hasattr(config_obj, key):
    raise ValueError(f"Config has no field '{key}'")
setattr(config_obj, key, value)
```

Setting a non-existent attribute silently creates a phantom field — the test
passes but the recipe fails for a real user.

## Adding a Functional Test

1. Create the script under `tests/functional_tests/launch_scripts/{h100,gb200}/active/`.
2. Start the file with a timeout header:
   ```bash
   # CI_TIMEOUT=<minutes>
   ```
3. Name it `{Tier}_{CamelDescription}.sh` — the tier prefix controls which CI matrix includes it.
4. Make it executable: `chmod +x <file>`.
5. Functional tests must use **at most 2 GPUs**.

No workflow file changes needed — the matrix is generated dynamically by
scanning the directory.

## Moving a Test to Flaky

```bash
# H100
git mv tests/functional_tests/launch_scripts/h100/active/L0_Foo.sh \
       tests/functional_tests/launch_scripts/h100/flaky/L0_Foo.sh

# GB200 (if the test also exists there)
git mv tests/functional_tests/launch_scripts/gb200/active/L0_Foo.sh \
       tests/functional_tests/launch_scripts/gb200/flaky/L0_Foo.sh
```

Flaky tests still run on manual dispatches (`test_suite=all`) so failures
remain visible. Move back to `active/` once the underlying issue is fixed.

## Removing a Test

Delete the script file and commit. No other changes required.

## Pytest Conventions

- Use pytest fixtures for common setup.
- Available markers: `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`.
- Functional tests are capped at **2 GPUs**. Set `CUDA_VISIBLE_DEVICES` explicitly for multi-GPU tests.
- Use `uv run python -m pytest`, never bare `pytest`.

## CI Job Reference

| GitHub Actions job | Hardware | Directory scanned |
|---|---|---|
| `cicd-functional-tests-l0` | H100 | `h100/active/L0_*.sh` |
| `cicd-functional-tests-l1` | H100 | `h100/active/L1_*.sh` |
| `cicd-functional-tests-l2` | H100 | `h100/active/L2_*.sh` |
| `cicd-functional-tests-flaky` | H100 | `h100/flaky/L*.sh` |
| `cicd-functional-tests-gb200-l0` | GB200 | `gb200/active/L0_*.sh` |
| `cicd-functional-tests-gb200-l1` | GB200 | `gb200/active/L1_*.sh` |
| `cicd-functional-tests-gb200-l2` | GB200 | `gb200/active/L2_*.sh` |
| `cicd-functional-tests-gb200-flaky` | GB200 | `gb200/flaky/L*.sh` |

Hardware runners: H100 uses `nemo-ci-{azure,aws}-gpu-x2`; GB200 uses `nemo-ci-gcp-gpu-x2`.

## Code Anchors

| Component | Path |
|---|---|
| Matrix generation (H100) | @.github/workflows/cicd-main.yml job `generate-test-matrix` |
| Matrix generation (GB200) | @.github/workflows/cicd-main.yml job `generate-gb200-test-matrix` |
| Test runner action | @.github/actions/test-template/action.yml |
| Launch scripts root | `tests/functional_tests/launch_scripts/` |
