---
name: test-system
description: Reference for the Megatron Bridge CI test system — directory layout, tier semantics, script conventions, and how to add, move, or disable tests.
---

# Megatron Bridge CI Test System

## Directory Layout

```
tests/functional_tests/launch_scripts/
  h100/
    active/   # H100 tests that run in CI automatically
    flaky/    # H100 tests quarantined from blocking CI
  gb200/
    active/   # GB200 tests that run in CI automatically
    flaky/    # GB200 tests quarantined from blocking CI
```

Scripts are named `{Tier}_{Description}.sh` (e.g., `L0_Launch_training.sh`).

Unit tests live under `tests/unit_tests/` and are independent of this layout.

## Tier Semantics

| Tier | Trigger | Blocking |
|------|---------|---------|
| L0 | Every PR, every push to `main`, schedule | Yes — PR cannot merge if L0 fails |
| L1 | Push to `main`, schedule, or PRs with `needs-more-tests` label | Yes |
| L2 | Schedule and `workflow_dispatch` only | Yes (when triggered) |
| flaky | `workflow_dispatch` with `test_suite=all` only | No — failures are informational |

H100 and GB200 each have their own L0/L1/L2/flaky jobs. Moving a test to flaky
removes it from blocking CI on both hardware targets independently.

## Script Conventions

Every launch script must start with:

```bash
# CI_TIMEOUT=<minutes>
```

This is parsed by the matrix generator to set the job timeout. If the header is
absent, the default is 30 minutes.

The tier prefix in the filename (`L0_`, `L1_`, `L2_`) controls which matrix the
script is included in. The matrix generator globs `{tier}_*.sh` from
`{h100,gb200}/active/`.

## Adding a New Test

1. Create the script under the appropriate `active/` directory (or both if the
   test should run on both H100 and GB200).
2. Start the file with `# CI_TIMEOUT=<minutes>`.
3. Name the file `{Tier}_{CamelDescription}.sh`.
4. Make it executable: `chmod +x <file>`.

No workflow changes are needed — the matrix is generated dynamically by
scanning the directory.

## Moving a Test to Flaky

Use `git mv` to relocate the script from `active/` to `flaky/`:

```bash
# H100
git mv tests/functional_tests/launch_scripts/h100/active/L0_Foo.sh \
       tests/functional_tests/launch_scripts/h100/flaky/L0_Foo.sh

# GB200 (if the test also exists there)
git mv tests/functional_tests/launch_scripts/gb200/active/L0_Foo.sh \
       tests/functional_tests/launch_scripts/gb200/flaky/L0_Foo.sh
```

Flaky tests still run on manual dispatches (`test_suite=all`) so failures
remain visible — they just don't block PRs. Move back to `active/` once the
underlying issue is fixed.

## Removing a Test

Delete the script file and commit. No other changes required.

## CI Job Reference

| GitHub Actions job | Hardware | Directory scanned |
|--------------------|----------|-------------------|
| `cicd-functional-tests-l0` | H100 | `h100/active/L0_*.sh` |
| `cicd-functional-tests-l1` | H100 | `h100/active/L1_*.sh` |
| `cicd-functional-tests-l2` | H100 | `h100/active/L2_*.sh` |
| `cicd-functional-tests-flaky` | H100 | `h100/flaky/L*.sh` |
| `cicd-functional-tests-gb200-l0` | GB200 | `gb200/active/L0_*.sh` |
| `cicd-functional-tests-gb200-l1` | GB200 | `gb200/active/L1_*.sh` |
| `cicd-functional-tests-gb200-l2` | GB200 | `gb200/active/L2_*.sh` |
| `cicd-functional-tests-gb200-flaky` | GB200 | `gb200/flaky/L*.sh` |

Hardware runners: H100 uses `nemo-ci-{azure,aws}-gpu-x2`; GB200 uses
`nemo-ci-gcp-gpu-x2`.

## Code Anchors

| Component | Path |
|-----------|------|
| Matrix generation (H100) | `.github/workflows/cicd-main.yml` job `generate-test-matrix` |
| Matrix generation (GB200) | `.github/workflows/cicd-main.yml` job `generate-gb200-test-matrix` |
| Test runner action | `.github/actions/test-template/action.yml` |
| Launch scripts root | `tests/functional_tests/launch_scripts/` |
