# Accuracy mode

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

Read this for `scripts/nimbench.py accuracy`: the test sets, the Gym image, and how a verdict is made.

## What a run does
For each test set the script runs `gym eval run --benchmark SET` in a NeMo Gym container. NeMo Gym
needs Python 3.13.14 or newer, so it never runs on this host. Gym turns its requests into chat
completions, which a NIM serves. The script then reads the rollout lines and the aggregate metrics
file, checks them, and scores the set.

| setting | default |
|---|---|
| test sets | `gsm8k`, `ifeval`, `mmlu_pro` (`--sets`) |
| items | every item; `mmlu_pro` is cut to 2,000. `--limit N` sets all, `--full` removes every cut |
| sampling | `--temperature 0`, `--max-tokens 4096`, `--repeats 1` answer per item |
| image | `--gym-image NAME@sha256:...`; a tag is refused (`image_not_pinned`) |

Gym marks these test sets experimental. Its repository has a Dockerfile (`docker/Dockerfile`), whose
entrypoint is `gym`: build it once, or take your team's build, record its digest and pin that.
Docker is required (`docker_missing` otherwise). The endpoint key goes in as
`+policy_api_key=${oc.env:NAME}`: a reference, never the value.

## Vision-language models
`--vlm` makes the default test sets the vision ones. Gym 0.7.0 has two that fit a single question
and answer:

| test set | what it asks | notes |
|---|---|---|
| `labbench2_vlm` | scientific figure, table and protocol questions | data licence CC-BY-SA-4.0 |
| `gdp_pdf` | professional reasoning over PDFs, a pass rate over criteria | scores are partial credit |

Both are graded by an **LLM judge**: a second model that sees the prompts, the images and the answers.
So you must name it with `--judge-endpoint`, `--judge-model` and, for a key, `--judge-api-key-env`.
It is part of the plan, and the approval covers it.

- The judge must not be a model being measured. The same endpoint and model as the candidate or the
  baseline is refused (`self_judging`). One judge grades both sides, and it is in the manifest, so a
  baseline with another judge is refused.
- The judge's endpoint is checked before any load. Check where its data goes: Gym's own default judge for
  these sets is an outside service, which is why the skill makes you name one.
- Vision sets keep no rollouts: they hold the images. `scores.json` keeps the rewards.
- Not supported: `hle_diamond` (Gym lets the model under test judge itself, which is unfair between
  endpoints) and the agent sets such as `osworld` (many steps, no single answer).
- These sets are far from the usual VLM benchmarks (MMMU, ChartQA, DocVQA), which Gym 0.7.0 does not
  have. Say so when you report a vision score.

## What counts
- The rollout count must match the aggregate file: `task_count` tasks times `--repeats`.
- At most 5% of answers may hit the output token limit. More makes the score invalid
  (`answers_truncated`): a reasoning model needs a larger `--max-tokens`.
- A set that already finished under the same plan id is skipped on a rerun.
- A failed set is listed, and other sets still run after ordinary scoring or cleanup failures.
  If the Gym container cannot be confirmed stopped after a timeout, do not start further sets
  or the baseline until it is confirmed stopped. With a baseline, a set that has no
  `scores.json` on either side (or only one left by an earlier plan) gets `no verdict`.

## The score and the verdict
- One score is the share of items answered right, with its 95% range: Wilson for right-or-wrong
  rewards, a normal range for partial credit. GSM8K (1,319 items) at 90% has a range of about ±1.6.
- With a baseline, both sides answer the same items. The verdict uses the paired difference
  (candidate minus baseline, in points) and its 95% range:

| the whole range is | verdict |
|---|---|
| above minus the margin (default 1.0, `--margin`) | `pass` |
| below minus the margin | `fail` |
| on both sides of it | `inconclusive`: not proven |

- Under 30 paired items (`MIN_VERDICT_ITEMS` in `scripts/nb_stats.py`, for example `--limit 10`)
  the verdict is always `inconclusive`: the normal range needs about 30 items to hold.
- When every item moved by the same amount (often: both sides gave the same answers), the spread is
  zero and proves nothing. The range is then the difference ± 300 / items points (the rule of
  three): ±10 at 30 items, ±0.3 at 1,000.
- `--margin` changes the verdict, so it is part of the plan id: a new margin needs a new approval.

- At 1,319 items the paired range is about ±1.2 points, wider than a 1-point margin. Expect
  `inconclusive` for small drops. More items, or `--repeats 4` with `--temperature 0.7`, narrows it.
  Repeats at temperature 0 give the same answers and add nothing.

## Errors
| code | exit | fix |
|---|---|---|
| `image_not_pinned` | 2 | pin the image: `name@sha256:<64 hex>` |
| `judge_required`, `self_judging` | 2 | name a judge model that is not a model measured |
| `docker_missing` | 3 | install Docker |
| `endpoint_unreachable`, `model_not_served` | 3 | check the URL and the served names |
| `tool_failed` | 4 | follow `errors[].fix`: restore output-folder access after cleanup failure, check the Docker client after startup failure, or read a saved log when named; if a client may still be running, ask the user for approval before stopping it and rerun only after the user confirms it has stopped |
| `answers_truncated` | 4 | raise `--max-tokens`, then run again |
| `literal_secret`, `bad_input` | 2 | a key goes in a variable; fix the flag |
| `not_approved` | 5 | run `--dry-run` again and ask the user |
| `baseline_mismatch` | 5 | the images, sets, items, temperature or token limit differ |

Cleanup failures keep any earlier timeout or scoring error. The failed set has no successful
score, and comparisons require it to appear in the run's successful-set summary. Scores are
saved only after vision rollout cleanup succeeds; cached vision scores must pass that cleanup
again before reuse. Other sets continue after ordinary scoring or cleanup failures. If the
Gym container cannot be confirmed stopped after a timeout, further sets and the baseline do
not run until it is stopped.
