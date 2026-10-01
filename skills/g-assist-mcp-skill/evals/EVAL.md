<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# Evaluating `g-assist-mcp-skill`

This is one of three doc-style variants of the same RISE capability
(`rise-mcp-skill`, `g-assist-mcp-skill`, `rise-mcp-skill-3`). The three share an
identical task set so their scores can be compared directly: same prompts, same
expected behaviours, only the skill document under test differs. Read the
comparison as an A/B on how the doc style affects discoverability and
correctness — do not diverge the datasets between variants.

## Why the dataset is split in two

This skill drives `gassist_*` tools that only exist on a Windows PC with an NVIDIA
RTX GPU, a current driver, and a running RISE MCP server. A CI runner has none of
those, so a single dataset would either fail everywhere or test nothing.

| File | Cases | Environment |
|---|---|---|
| `evals.json` | 16 (11 positive, 5 negative) | Any runner. No NVIDIA hardware, driver, or G-Assist server needed. |
| `evals-hardware.json` | 31 (hw-001 through hw-031) | RTX testbed with the G-Assist server connected. |

`evals.json` is the CI-gated suite. It is not a reduced-value subset: with no
`gassist_*` tools present, a runner reproduces exactly the conditions in which this
skill is most likely to cause harm — an agent that invents a tool name, claims a
change it never made, tries to bypass a confirmation prompt, or starts modifying
driver files to "fix" a missing server. Those are the behaviours `evals.json`
measures, and the absence of tools is the point rather than a limitation.

`evals-hardware.json` covers what only real hardware can show: confirmation
gating, the keep-or-revert prompt on a modeset, live enum values, and the
false-success behaviour where a failed call still returns `isError: false`. Run
it on the RTX testbed and record which table in `BENCHMARK.md` the numbers came
from.

## Running the CI suite

```bash
uv tool install --python 3.13 "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"
export SKILL_EVAL_LLM_PROVIDER=nv_build
export NVIDIA_API_KEY='nvapi-...'

skillevaluator validate --external skills/g-assist-mcp-skill
skillevaluator run skills/g-assist-mcp-skill --dataset evals/evals.json
```

Validate with `--external` — the external profile is the one that applies to a
published skill, and it relaxes fields that only internal skills carry.

## Running the hardware suite

On the testbed, with the `rise` server configured in the host:

```bash
skillevaluator run skills/g-assist-mcp-skill --dataset evals/evals-hardware.json
```

Two things to know before you trust a run:

**Some cases change real settings.** `hw-002`, `hw-003`, and `hw-005` alter
V-Sync, the display mode, and the refresh rate. Run them on a machine you are
willing to have reconfigured, and expect `hw-003` to leave the display reverted
rather than changed, since that is the designed outcome when nobody answers the
keep prompt.

**Confirmation-gated cases need an elicitation-capable host.** Without it the
engine refuses the call, which is correct behaviour but measures the refusal path
rather than the approval path. Run the gated cases on a host that supports
elicitation, and never invent a `confirm`/`force` argument or an environment
variable to make them pass unattended — that bypasses the consent the case exists
to verify, so a green result would be meaningless.

## What a case is checking

Every case sets `expected_script` to `null`. This skill ships no `scripts/`
directory; its whole surface is instructions, so script execution is not a signal
here and a non-null value would be a false expectation.

The `expected_behavior` lists are deliberately heavy on negatives — "did not
invent a tool name", "did not include a confirm argument", "did not report
success because the call returned without an error". The failure modes for this
skill are things an agent does *too readily*, so a case that only asserts the
happy path would pass while the real risk went unmeasured.

## Extending the dataset

`skillevaluator create-eval-dataset` can seed new cases, but run it with
`--dry-run` first: `--force` overwrites this directory and the hand-written
negative assertions above would be lost.

When you add a case, keep three rules. Do not name the skill in the `question` — a
prompt that says "use g-assist-mcp-skill to..." measures instruction-following, not
discovery. Put it in the file matching its environment; a hardware-dependent case
in `evals.json` breaks CI for everyone. And add the same case to the other two
variants so the A/B comparison stays valid.

Keep the CI suite inside the NVCARPS one-hour runner limit. If it grows past
that, move the overflow to a scheduled job and say so in `BENCHMARK.md`.
