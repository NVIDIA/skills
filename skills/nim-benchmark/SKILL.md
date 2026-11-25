---
name: nim-benchmark
description: Measures a running NIM endpoint, text or vision-language, for speed with AIPerf and for accuracy with NeMo Gym, against an optional baseline, behind an approved plan id. Use when asked to benchmark a NIM, measure its latency or throughput, or check its accuracy.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  version: 0.1.0
  tags:
    - nim
    - benchmark
    - aiperf
    - accuracy
    - nemo-gym
---

# NIM benchmark

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

## Hard rules
1. Never send load or start a container before the user approved the plan id that `--dry-run` printed. The guard hook (`references/hook.md`) blocks a direct `aiperf`, `gym` or `docker run`.
2. Never put a key on a command line: give `--api-key-env NAME`, the name of a variable. The script refuses a value (`literal_secret`).
3. Never show a score without its 95% range, and never call `inconclusive` a pass: say "not proven".
4. Trust the files the script checked, not an exit code. A run counts only when the script counted it.
5. Ask for a baseline once, at intake. With one, give its verdict. Without one, give the numbers and no verdict.
6. Never use a baseline the script refused (`baseline_mismatch`). Report without a verdict, or measure the baseline again with this setup.
7. This skill only measures. Never deploy, tune or change a NIM with it. Never let a model judge its own answers: the judge for vision sets is another model (`self_judging`).
8. Pin what you run: AIPerf 0.13.0, and the NeMo Gym image by digest, never a tag like `latest`.

`SKILL_DIR` is the folder that contains this SKILL.md. `scripts/X.py` below means `python3 "<SKILL_DIR>/scripts/X.py"`. Scripts print one JSON object; follow its `next` and each `errors[].fix`. Run them from any folder; `--out` names the run folder.

## When not to use
- Choosing a NIM image and profile, or checking that a model fits: `nim-select`.
- Deploying a NIM, or troubleshooting a deployment: `nim-deploy`.
- Changing runtime settings to make a NIM faster: `nim-tune`. It hands accuracy checks to this skill.
- Predicting latency or a layout on paper, with no running NIM: out of scope; say so.

## Inputs
| name | required | how to get | ask if missing |
|---|---|---|---|
| endpoint | yes | the user, an OpenAI-compatible URL | "Which endpoint URL serves the NIM?" |
| model | yes | `/v1/models` of the endpoint | "Which served model name?" |
| mode | yes | the user: speed (`perf`), accuracy (`accuracy`) or both | "Speed, accuracy, or both?" |
| model kind | yes | the user: a text model, or a vision-language model (VLM) that reads images | "Does the model read images?" |
| baseline | no | the user: another endpoint URL, or a finished run folder | "Do you have a baseline to compare against?" |
| tokenizer | `perf` | the user, a Hugging Face id that matches the model | "Which tokenizer matches the model?" |
| client | `perf` | `docker` (default), `local` (aiperf on this host) or `kubernetes` (a pod that has aiperf) | "Where may the load client run?" |
| images | VLM `perf` | the user: images per request, and their size (default 512 x 512) | "How many images per request, and how large?" |
| judge | VLM `accuracy` | the user: the URL and served name of a judge model that is not the model measured | "Which model should judge the answers? It sees the prompts, images and answers." |
| gym image | `accuracy` | the user, NeMo Gym pinned as `name@sha256:...` | "Which NeMo Gym image, pinned by digest?" |
| key variable | if the endpoint needs a key | the user, the NAME of a variable | "Which variable holds the API key?" |
| run folder | yes | `runs/RUN`, one folder for the whole benchmark | "Where should the results go?" |

## Steps
| step | command | on success | on failure |
|---|---|---|---|
| intake | Record the inputs; ask about the baseline | check | ask for the missing inputs together |
| check | `scripts/nimbench.py check --endpoint URL --model NAME` | plan | `endpoint_unreachable`, `model_not_served`: follow the fix |
| plan perf | `scripts/nimbench.py perf --endpoint URL --model NAME --tokenizer TOK --out runs/RUN --dry-run`; for a VLM add `--images-per-request N` | approve | fix each `errors[].fix` |
| plan accuracy | `scripts/nimbench.py accuracy --endpoint URL --model NAME --gym-image IMAGE --out runs/RUN --dry-run`; for a VLM add `--vlm --judge-endpoint URL --judge-model NAME` | approve | `image_not_pinned`, `docker_missing`, `judge_required`, `self_judging`: follow the fix |
| approve | Show the user the plan and `facts.plan_id`: the endpoints (and the judge, which sees their data), what runs and how many runs; wait for yes | run | stop |
| run | The approved planning command with `--dry-run` replaced by `--approved PLAN_ID`; keep every baseline option unchanged | report | On failure below |
| compare | `scripts/nimbench.py compare runs/RUN --baseline runs/OLD` for two finished folders | report | `baseline_mismatch`: rule 6 |
| report | Read `runs/RUN/report.md`; show the numbers, the ranges and each verdict | done | On failure below |

If a baseline was given, include `--baseline URL_OR_FOLDER` during both planning and execution. For a URL baseline, keep `--baseline-model NAME` and `--baseline-api-key-env VARIABLE` unchanged: changing them changes the plan id and needs a new approval. These two options have no effect on a folder baseline; omit them when comparing saved runs.

Run `perf` and `accuracy` into the same `--out` for "both": one manifest, one report. Each needs its own approval, because each has its own plan id.

For a Kubernetes performance client, add `--client kubernetes --namespace NS --pod POD`
to both `check` and `perf`. Endpoint and baseline probes run inside that pod, so in-cluster
Service names work without a host route to the cluster. The pod needs `python3` and AIPerf;
API key variables must be set in the pod. See `references/perf.md`.

## What the modes do
| mode | tool | what it measures | defaults |
|---|---|---|---|
| `perf` | AIPerf 0.13.0 in a client container, pod or this host | TTFT and ITL (average and p99), request time, tokens/s at each load | 1024 in and 256 out tokens, loads 8, 32 and 128, 3 repeats, fixed seeds, a fresh cache salt per run; a VLM also gets new random images per request |
| `accuracy` | NeMo Gym, in its own Docker container | the score of each test set, with its 95% range | text: `gsm8k`, `ifeval`, `mmlu_pro` (2,000 items); VLM (`--vlm`): `labbench2_vlm`, `gdp_pdf`; temperature 0; `--full` runs every item |

Read `references/perf.md` for the workload, clients, images and AIPerf's known bugs. Read `references/accuracy.md` for the test sets, the judge for vision sets, the paired verdict and why answers get cut off.

## Verdicts
| result | meaning | what you say |
|---|---|---|
| speed `better` or `worse` | the repeat ranges do not overlap and the change is over 2% | the change in percent, per load |
| speed `within_noise` or `same` | ranges overlap, or the change is under 2% | no real difference was measured |
| accuracy `pass` | the whole paired range is above minus the margin (default 1 point) | no drop beyond the margin |
| accuracy `fail` | the whole paired range is below minus the margin | accuracy dropped; do not promote |
| accuracy `inconclusive` | the range straddles the margin, or the set has under 30 items | not proven; offer more items or sampled repeats |
| no verdict | no baseline, an incomplete load, or a test set that failed | numbers only |

## Outputs
| file | holds |
|---|---|
| `manifest.json` | tool versions and images, workload, seeds, test sets, items hash, served models, plan ids |
| `perf/cLOAD-rREPEAT/` | the AIPerf export file and console log of one run |
| `accuracy/SET/` | Gym rollouts, the aggregate metrics file and `scores.json`; vision sets keep no rollouts, because they hold the images |
| `baseline/` | the same files for a baseline endpoint run in this session |
| `compare.json`, `report.md` | the verdicts, and the report with numbers only: no prompt and no answer |

## On failure
1. Read `errors[].code`, `message` and `fix`, or the saved log named in the fix.
2. Look the code up in `references/perf.md` or `references/accuracy.md`.
3. Retry once with the fix. The same plan id resumes an accuracy run and skips finished sets.
4. Still failing: stop. Show the user the error and what you tried.

## Done means
- `report.md` exists, and every score in it has a 95% range.
- A speed result has 3 counted repeats per load, or the load is listed as incomplete.
- Each verdict is `pass`, `fail`, `better`, `worse`, `same` or `within_noise`, or the report says why there is none.
- Nothing the run started is left running.

## Common mistakes
| mistake | do instead |
|---|---|
| Run `aiperf profile` by hand for a quick number | `scripts/nimbench.py perf`: seeds, repeats, a cache salt and checks come with it |
| Trust a run because AIPerf exited 0 | AIPerf can exit 0 with a short export; only the script's count matters |
| Call a 0.5-point drop on GSM8K a pass or a fail | The paired range is about 1.2 points wide at 1,319 items: say `inconclusive` |
| Compare a run with last month's folder | The manifest must match; a different AIPerf version or length is refused |
| Pass the API key as `--api-key-env sk-...` | Put the key in a variable, and pass only the variable's name |
| Benchmark a shared production endpoint at load 128 | Ask the endpoint owner first; start with `--loads 8` |
| Let the model under test judge the vision sets | Name a different judge model; the script refuses the same endpoint and model |
| Compare a text-only speed run with a VLM run | The image settings must match; a different image size or count is refused |

## References
- Read `references/perf.md` for performance mode: loads, clients, the AIPerf bugs and the error codes.
- Read `references/accuracy.md` for accuracy mode: test sets, the Gym image, verdict maths and the error codes.
- Read `references/hook.md` when installing the skill, or when the guard denied a command.
