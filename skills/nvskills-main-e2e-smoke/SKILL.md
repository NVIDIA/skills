---
name: nvskills-main-e2e-smoke
description: Runs a deterministic local check for the production NVSkills workflow. Use when asked to verify the NVIDIA/skills and nvskills-ci main workflow integration.
license: Apache-2.0
metadata:
  author: "NVSkills CI"
  tags:
    - ci
    - smoke-test
    - signing
---

# NVSkills Main Workflow End-to-End Smoke Test

Run a harmless local check that confirms this temporary smoke-test skill was discovered and invoked.

## Instructions

1. Run `python3 scripts/check.py`.
2. Confirm that the command prints `NVSKILLS_MAIN_E2E_OK`.
3. Return the exact output and report the check as passed.
4. Do not modify files, access the network, or run unrelated commands.

## Examples

- "Use the NVSkills main workflow smoke-test skill."
- "Verify the NVIDIA/skills and nvskills-ci main workflow integration."

## Error Handling

- If the script is missing, report the missing path and stop.
- If the output differs from the expected value, report the actual output and fail the check.
