---
name: i4h-catheter-navigation-smoke
version: "0.7.0"
description: Run CPU-only fluorosim smoke tests (imports, preprocessing, CLI parsers). Use when asked to smoke-test catheter navigation in CI or without a GPU.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - smoke-test
    - ci
---

# i4h Catheter Navigation - Smoke Tests

## Purpose

Run CPU-only unit smoke tests for fluorosim components and entrypoint parsers,
safe for CI without a GPU. Exercises imports, CT->mu preprocessing, parser
sanity, and conditional viewport import checks.

## Base Code

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/catheter_navigation" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/catheter_navigation" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Run

### Step 1 - run tests

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
WF_ROOT="${REPO_ROOT}/workflows/catheter_navigation"
RUN_DIR="${WF_ROOT}/runs/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${WF_ROOT}/runs/.latest"

python3 -m unittest workflows/catheter_navigation/tests/test_fluorosim_smoke.py \
  2>&1 | tee "${RUN_DIR}/logs/smoke.log"
```

## Expected Output

```text
Ran N tests in ...s
OK
```

Parser error lines on stderr from negative CLI tests are **expected** - not failures.

## Verify

```bash
grep -E "Ran [0-9]+ tests" "${RUN_DIR}/logs/smoke.log"
grep "OK" "${RUN_DIR}/logs/smoke.log"
```

## Prerequisites

- Python 3 with workflow dependencies installed (for example via
  `workflows/catheter_navigation/requirements.txt`).
- No GPU required.

## Limitations

- Does not exercise GPU Slang rendering or the interactive viewport.
- For GPU validation, use [[i4h-catheter-navigation-render-drr]].

## Troubleshooting

- **Error:** import failures - Fix: run [[i4h-catheter-navigation-setup]] and
  ensure the active venv has catheter workflow dependencies installed.
- **Error:** unexpected test count - Fix: inspect log; this can change when smoke
  coverage is added/removed on a branch.

## Final Response

Report pass/fail, test count, and log path. If passed, note GPU stages still need separate verification.
