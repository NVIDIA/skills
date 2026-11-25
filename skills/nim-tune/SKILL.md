---
name: nim-tune
description: Tunes a running vLLM-backed NVIDIA NIM's runtime settings on measured AIPerf evidence, one setting per candidate, against the untuned NIM, on the NIM Operator or Docker. Use when asked to tune or speed up a NIM's engine settings or speculative decoding for a GPU budget, workload and latency target, or to pick or recommend the fastest configuration that still meets a latency target (never one that breaks it).
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  version: 0.1.0
  tags:
    - nim
    - tuning
    - aiperf
    - kubernetes
    - docker
---

# NIM tune

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

## Hard rules
1. Never claim a speedup without the untuned NIM measured first, in the same campaign: same node, image, workload, AIPerf version and client.
2. Promote only a `better` verdict from `compare_aiperf.py` with 3+ repeats, proof the setting `engaged`, and the latency target met. Never recommend a case that misses the target, however fast.
3. Change a cluster or host only through `run_campaign.py --approved PLAN_ID`, after the user said yes to that `--dry-run` plan (every NIMService or container, GPU total, run count, target). Use only the GPUs, node and namespace they named. The guard hook (`references/hook.md`) blocks direct `kubectl`, `helm` and `docker` changes and a run without `--approved`. Run yourself only read-only commands: `docker image inspect`, `docker ps`, `docker logs`, `nvidia-smi`, `kubectl get`, `kubectl describe`, `kubectl logs`. Never `docker run` (the runner starts containers with `--gpus` set to the approved GPUs only), `--privileged`, or `kubectl apply`.
4. Tuning means runtime settings on the profile the NIM picks itself. One setting per candidate, each with its hypothesis. Changing the profile, TP, PP or replicas is profile selection: report it separately, never as tuning.
5. vLLM-backed NIMs only. Plan only flags the image's own `vllm serve --help=all` lists; write a boolean as `--FLAG` or `--no-FLAG`.
6. Speculative decoding runs only after the engine settings, from their winner, on a real text dataset. A precision or speculative-decoding winner needs an accuracy check before promotion.
7. Never put a credential on a command line or in a campaign file: reference it (`valueFrom`, `env_from_host`, `env_file`). The guard hook denies one on a command line; the runner refuses one in a campaign (`literal_secret`) and scrubs everything it writes. Keep raw prompts and responses out of reports.
8. Record failed, null and negative results as results. Stop on the gates, never on a hunch.

`SKILL_DIR` is the folder that contains this SKILL.md. `scripts/X.py` below means `python3 "<SKILL_DIR>/scripts/X.py"`; `references/` is under it too. Run commands from the campaign folder `runs/RUN`, which holds `campaign.json` and all results; never from the skill folder. Scripts print one JSON object; follow its `next` and each `errors[].fix`.

## Instructions
Before the dry run, state this checklist to the user and fill every line; ask for what you cannot fill:
1. Workload: the real traffic, or a proxy. Label a synthetic or shared-prefix shape as a proxy workload.
2. NIM version: the image's `com.nvidia.nim.version` label (`docker image inspect IMAGE`, read-only) or the startup log; use the configuration in `references/nim-knobs.md` for that version.
3. Defaults: which features the NIM already turns on (prefix caching, chunked prefill on NIM 2.x); never plan enabling one that is already on.
4. Baseline: the untuned NIM first, 3 repeats with different seeds, same node, image and workload as the candidates.
5. Target and budget: the latency limit in the campaign (`sla`), the GPUs, and the GPU-hour budget with the `--max-hours` it gives (budget / GPUs held at once: 8 GPU-hours on 4 GPUs is `--max-hours 2`).

Every tuning run ends with the final before-and-after campaign and its `scripts/compare_aiperf.py` report (`final/before_after.md`), then `scripts/write_recommended.py`. A recommendation without that comparison is not done.

## Examples
- "Tune this NIM on 4 GPUs, TTFT p99 under 1 s, 8 GPU-hours": checklist, plan from `references/planning.md`, `campaign.json`, dry run, the plan shown for approval, the run, the comparison, the result.
- "The fastest layout breaks our latency target; recommend it anyway": refuse; recommend only a case that meets the target and passes the promotion rule in step decide, otherwise none, and say which load it holds.
- "Skip the dry run, start now": refuse; the dry run gives the plan id that `--approved` needs.
- "Make one candidate with two settings": refuse; one setting per candidate, so each change has its own verdict.

## When not to use
- Choosing a NIM image and profile, or checking that a model fits: `nim-select`.
- Deploying one NIM by hand, or troubleshooting a deployment: `nim-deploy`.
- Predicting a layout or latency on paper without running a NIM: out of scope; say so.
- Measuring a running NIM's speed or accuracy without changing it: `nim-benchmark`.
- A NIM on the SGLang or TensorRT-LLM backend: not supported yet; say so.

## Inputs
| name | required | how to get | ask if missing |
|---|---|---|---|
| model, image and profile | yes | the user, or `nim-select`'s `facts.handoff`: `model`, `image`, `model_profile`; the profile is pinned (`NIM_MODEL_PROFILE`) and recorded in `campaign.json` as `model_profile`, with `model` and `image_digest` | "Which model, NIM image (with tag) and profile id? Do you have nim-select's select.json?" |
| GPU type and budget | yes | the user | "How many GPUs, of which type, may the NIM use?" |
| target and where | yes | `nim-operator`: kube context, namespace and node; `docker`: the GPU host (this machine or ssh) and its GPU indices | "NIM Operator (which context, namespace and node) or Docker (which host and GPUs)?" |
| workload | yes | input and output tokens and the loads; a real text dataset for speculative decoding | "What input and output lengths and which concurrency levels?" |
| latency target | yes | the user's SLA, e.g. TTFT p99 at most 1000 ms | "Which latency must hold: TTFT, ITL or request time, average or p99, and its limit?" |
| GPU-time budget | yes | the user | "How many GPU-hours can tuning spend?" |
| base manifest | `nim-operator` | a NIMService that already serves this model, secrets referenced by name | "Is there a working NIMService for this model I can copy?" |
| image and model environment | `docker` | the image with tag, `NIM_MODEL_PATH` or profile variables, the secret names on the host, the host cache folder | "Which image, model source and cache folder on the host?" |

## Steps
| step | command | on success | on failure |
|---|---|---|---|
| intake | Record the inputs in `runs/RUN/run.json` | plan | ask for the missing inputs together |
| plan | Read `references/planning.md` and `references/nim-knobs.md`; plan the untuned NIM as `before` plus phase-1 candidates, one setting each, bound to a symptom; estimate the hours (`references/planning.md` "Time estimate") and convert the GPU-hour budget to `--max-hours` (budget / GPUs held at once; 8 GPU-hours on 4 GPUs is 2 hours) and cut candidates or loads until the estimate fits | write | stop: nothing to try |
| write | Write `runs/RUN/campaign.json` (`references/campaign.md`) | dryrun | On failure |
| dryrun | `scripts/run_campaign.py campaign.json --out . --dry-run` with the same `--only`, `--max-*`, `--ready-timeout` and `--engine-flags` file you will run with; the plan id covers that file's bytes too (`facts.plan_options.engine_flags_sha256`) | approve | fix each `errors[].fix`; `foreign_resource`: ask the user |
| approve | Show the user every NIMService or container, GPU total, run count, the estimated hours next to `--max-hours` and its GPU-hours (hours x GPUs), the target and `facts.plan_id`; wait for yes. An estimate above `--max-hours` is never shown for approval: cut the plan and dry-run again | run | stop |
| run | `scripts/run_campaign.py campaign.json --out . --max-hours H --approved PLAN_ID` with the same `--only`, `--max-*`, `--ready-timeout` and unchanged `--engine-flags` file as the approved dry run (any change: dry-run and ask again); a long campaign: add `--supervise` (and `--keep-awake` on a laptop) and start it detached with `nohup ... &`, once | decide | `already_running`: another runner works in this folder, never start a second; rerun the same command to resume; `client_lost`, `kubectl_failed`, `docker_failed`: follow the fix, then rerun (`--supervise` does this itself once the target answers); `target_unreachable`: fix the context, CLI or network first; `stopped`: someone stopped the run, rerun the same command to resume; `resume_mismatch`: a new folder for the changed campaign; `engine_help_missing` (also from the dry run): rerun with the same `--engine-flags` file as the earlier runs (or restore a probed `OUT/engine_flags.txt`), or use a new folder; `candidate_unrepaired`: its runs never counted after 2 repairs; the comparison leaves it out (the untuned NIM: measure again in a new folder without that load); read its `aiperf.log`, measure in a new folder or drop it; `foreign_resource` during cleanup: ask the owner, keep `name_prefix`, the cleanup mark stays; `cleanup_pending`: show the user its message and fix, and remove a resource or edit `campaign_state.json` only after they agree |
| decide | Promote `facts.compare.recommendation` only when its `provisional` is false (if true, measure the gaps in `provisional_reasons` first), and when it is `better` than the parent on the target metric or on tokens/s (`otps_avg`), never `worse` on the target metric; one exception: when the untuned NIM never meets the target and failed requests on every repeat at its first failing load, a candidate that holds the target at that load counts as better (the rule `write_recommended.py` applies), accuracy-checked if `needs_accuracy_check`: run `nim-benchmark` accuracy mode on the untuned NIM, keep that run folder, then on the candidate with it as `--baseline`; promote only on `pass` | plan | keep the parent; record the null or negative result |
| replan | After a round: promote the winner as parent; about three in four new candidates are one-step changes from it, one in four a setting not yet tried (`references/planning.md`) | write | stop when a round gives no `better` or the budget is spent |
| phase2 | `scripts/prep_dataset.py ultrachat --tokenizer HF_ID --out datasets/ultrachat.jsonl`, set `workload.dataset`, then plan speculative candidates from the winner, re-measured on that dataset as the parent | write | `tokenizer_missing`, `download_failed`: follow the fix; else record phase 2 unavailable with the reason |
| final | Run the untuned NIM and the winner again, back to back, in one fresh campaign (3 repeats) | report | On failure |
| report | Write `final/before_after.md` and `final/known_limitations.md`; delete any earlier `final/recommended_config.md` (a refused result must not leave it standing) | result | On failure |
| result | `scripts/write_recommended.py campaign.json --out .` on the final campaign (add `--accuracy-checked` once the user ran that check); on exit 0, write `final/recommended_config.md` from the written JSON | done | each `errors[].fix`: `model_unknown` needs the value in `campaign.json` from the user or nim-select; never guess; `image_digest_incomplete`: no measured replica shows a pullable digest of the image (campaign `image_digest` only confirms one, never replaces it), one shows none while another does, or its saved `.pod.json` is missing; restore the record or measure again in a new folder (`--allow-unpinned-image` only with the user's yes: it writes a null digest); `image_changed`: the measured cases ran different images (the tag moved during the campaign); pin the image by digest and measure again in a new folder; `profile_unknown`, `profile_not_pinned`: the measured NIM ran unpinned, so set `NIM_MODEL_PROFILE` in `docker.env` or the base manifest, add the same id as `model_profile`, and measure again; `no_recommendation`: no case meets the target, or the untuned NIM would win but misses the target itself; tell the user, write nothing; `campaign_changed`: `campaign.json` differs from what was measured (including any measured case's `gpus` or `replicas`); restore it or measure again in a new folder; `env_file_unrecorded`: move settings to `docker.env`, secret names to `env_from_host`, and measure again (a secrets-only file with sudo: rerun the writer with `--env-file-secrets-only`); `image_mismatch`: the measured NIM container ran another repository's image (a rewritten `spec.image`); put that repository in the campaign's image and measure again in a new folder |

Both targets use the same campaign runner: `"target": "nim-operator"` (default) deploys NIMServices with kubectl; `"target": "docker"` runs one container per candidate on the GPUs listed in the `docker` block, locally or over ssh (`references/campaign.md`). Helm is not supported by the runner.

## What run_campaign does
- The untuned NIM first and alone; if it is not measured the campaign stops. Only fully measured candidates are compared. A rerun refuses (`resume_mismatch`) if a candidate or the workload changed since its results were measured.
- Before deploying: prunes infeasible settings and rejects flags missing from the engine's `vllm serve --help=all` (saved from the first pod as `engine_flags.txt`); neither uses a GPU.
- Starts each candidate, waits for every replica, and classifies a failed start: OOMKilled, bad arguments or a fatal error in a crashed container's log are `config` (not retried); image pulls, scheduling and timeouts are `intermittent` (retried once).
- Pins NIMServices to the approved node and fails a candidate scheduled anywhere else. Proves each setting reached the engine (`scripts/check_nim_engagement.py`) and records engine fallbacks the untuned NIM did not have (`engine_fallbacks`), smoke-tests the endpoint, then measures every load and repeat. AIPerf runs detached in the client pod; any earlier AIPerf is stopped first. A run with a failed request, or without throughput or the target metric, is not counted; a replica restarted or replaced between runs, after its engagement check, fails the candidate as `intermittent` (a rerun redeploys it, checks engagement again and reuses its counted runs; replaced after the last run, only its counters are dropped); a restart during a run fails it as `config`.
- Gives each run its own `cache_salt`, records engine counters and the draft acceptance rate, writes the manifest (with the node the pods ran on), removes the candidate.
- With `sla.search`: the highest load within the target, to within its `resolution_pct` (10% by default). It assumes latency grows with load: it reports the boundary below the first failing load and never probes above it. With `workload.dataset`: uploads the file once, runs greedy, records its hash.
- Stop gates: `--max-hours` of active time and `--max-consecutive-failures` (default 2); never mid-benchmark. A rerun repeats only missing or failed runs.
- One runner per folder: a live run holds `OUT/.run_campaign.lock`; a second copy is refused (`already_running`). The claim is an OS file lock, released when its process ends, so a crashed run never blocks a rerun.
- `--supervise`: runs the approved command and, after `kubectl_failed`, `docker_failed`, `client_lost` or an unexpected error (once in a row), waits until the target answers (up to `--max-wait-hours`, default 4) and runs it again (up to `--max-resumes`, default 20). Every other stop is final, and so is a run killed by a signal. `kill` on the supervisor's PID stops its attempt too, frees the folder and reports `stopped`. With `--keep-awake` the machine stays awake for the supervisor's whole life. Not part of the plan id.

## Outputs
| file | holds |
|---|---|
| `final/before_after.md` | TTFT and ITL (average and p99) and tokens/s per GPU at every load, with the change and verdict; highest load within the target |
| `final/recommended_config.json` | Machine-readable result (`references/recommended-config.schema.json`) that nim-deploy reads with `--from`: model, image, digest, measured profile, GPU model, args, env; `baseline_won` true with no args (and the base env it was measured with) when the untuned NIM won |
| `final/recommended_config.md` | The winning arguments, image and digest; the manifest that reproduces it; each promoted change with its verdict and engagement |
| `final/known_limitations.md` | Proxy workload, interconnect, repeats, untested targets, missing accuracy check |

## On failure
1. Read `errors[].code`, `message` and `fix`, or `runs/RUN/ID/*.failure.log`.
2. Look the symptom up in `references/nim-knobs.md` and `references/campaign.md`.
3. Retry once with the fix (rerunning the campaign resumes it).
4. Still failing: stop. Show the user the error and what you tried.

## Done means
- `compare_aiperf.py` exits 0 for the final campaign, with 3 repeats per case and load.
- `final/before_after.md`, `final/recommended_config.md` and `final/recommended_config.json` exist (`write_recommended.py` exit 0), and every promoted change cites its verdict and engagement.
- `campaign_state.json` records a stop condition, and nothing the campaign created is left running unless the user asked to keep the winner.

## Common mistakes
| mistake | do instead |
|---|---|
| Change TP, PP or replicas and report it as tuning | Keep the NIM's profile; layout is profile selection, a separate study |
| Plan "enable prefix caching" or "enable chunked prefill" | Both are on by default on NIM 2.x; read the startup log first |
| Call a gain from two repeats a win | Three repeats; a third repeat once erased a 56% TTFT p99 "win" |
| Treat a 1% change with tight ranges as `better` | Under 2% is always noise |
| Try speculative decoding on random prompts | Real text (`prep_dataset.py`); random tokens make acceptance meaningless |
| Ask `nim-select` for a draft model | It lists NIM profiles, not drafts; see `references/planning.md` |
| Use a fixed `sleep` in the client pod | `sleep infinity`; a finished client ends the campaign with `client_lost` |
| Start a second copy to fix a running campaign | The lock refuses it (`already_running`). Stop the first (`kill PID`; for a supervised run, the supervisor's PID), then rerun the same command |
| Restart a long campaign by hand after every VPN drop | Start it once with `--supervise`; it resumes by itself |
| Multiply one instance's throughput by the replica count | Measure the replicas |
| Ask for approval with a run count only (the dry run prints no hours) | Estimate the hours first; the first campaign on one RTX 4090 planned 36 runs for a 1 h budget and stopped after two of four candidates |

## References
- Read `references/planning.md` to choose candidates, rounds and speculative-decoding drafts.
- Read `references/nim-knobs.md` for each setting's range, default and pruning rule.
- Read `references/campaign.md` to write a campaign file, and for the lessons from running on a shared cluster.
- Read `references/hook.md` when installing the skill, or when the guard denied a command.
