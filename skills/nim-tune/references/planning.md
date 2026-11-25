# Planning tuning candidates

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

How `nim-tune` decides which runtime settings deserve GPU time. Planning never deploys or
benchmarks anything. Read `nim-knobs.md` alongside it for each setting's range and default.

## Rules
- Keep the profile, TP, PP and replica count the untuned NIM chose. Changing them is profile
  selection, not tuning; do it only when the user asks, and report it apart.
- Phase 1 is engine settings. Phase 2 (speculative decoding) starts only from the phase-1 winner,
  only on a real text dataset.
- Each candidate changes one setting against its parent. Split any bundled change.
- Plan only flags listed in `runs/RUN/engine_flags.txt` (the campaign saves the image's
  `vllm serve --help=all`), never one already at that value. Before the first campaign, plan from
  the flags in the launch line and the documented defaults, and mark them unverified.
- Predictions (from a simulator or any model) only order candidates; never record one as a result.
- A precision or speculative-decoding candidate needs an accuracy check before promotion.

## Steps
| step | what to do |
|---|---|
| baseline | Plan the untuned NIM as candidate `before`: no arguments, the NIM's own profile |
| defaults | From the untuned NIM's startup log, record each setting's effective value: batched tokens, max sequences, memory fraction, KV-cache dtype, chunked prefill, prefix caching, max model length, CUDA graph sizes |
| phase 1 | For each setting in `nim-knobs.md` "Runtime candidates", plan one or two values on each side of the effective default, bound to a symptom in the evidence |
| prune | Drop what `nim-knobs.md` "Pruning before GPU time" rejects; record the reason |
| order | Sort by expected effect on the target metric, then GPU cost; cut at the budget (as `--max-hours`) and list the rest as deferred |
| phase 2 | From the winner: `ngram` first (no draft), then EAGLE3 if a draft for the exact model exists, then draft lengths 1, 3, 5, 7; the winner re-measured on the dataset is the parent |

## Time estimate
`--dry-run` prints the run count, not the hours. Estimate them before approval:

hours = candidates x (startup + loads x repeats x minutes per run) / 60

This is elapsed time, and `--max-hours` limits elapsed (active) time. GPU-hours = hours x the
GPUs held at once (`gpus` x `replicas` of the candidate that runs). Example: 2 hours on 4 GPUs is
8 GPU-hours. The user's budget is in GPU-hours, so convert it before approval:
`--max-hours` = GPU-hour budget / GPUs held at once (8 GPU-hours on 4 GPUs: `--max-hours 2`).
Compare the estimated hours with that `--max-hours`, and show the user both numbers.

- Startup is the engine load: about 1 minute with the model cached, much more on the first start
  (an image pull and a model download: about 11 minutes for the untuned NIM below).
- Minutes per run is not fixed, and low loads are the slowest because one request at a time
  leaves the GPU idle. **[run]** gpt-oss-20b on one RTX 4090 (Docker, 1000 in / 200 out, 3
  repeats): about 4 minutes per run at concurrency 1 and about 1 minute at concurrency 16. The
  untuned NIM took about 32 minutes and each later candidate about 20, so 4 candidates did not
  fit a 1 hour budget and the third was cut off.
- When the run time is unknown, measure it: run the untuned NIM alone
  (`run_campaign.py ... --max-candidates 1 --approved PLAN_ID`) and read the run times from
  `campaign.log`. Leave out its first start, which includes the pull.
- Over the budget: cut candidates (keep the ones with the strongest evidence), drop the
  slowest load, or ask the user for more hours. List what was cut as `deferred`.
- Leave about 10% of the budget as margin. The runner never stops mid-benchmark, and a candidate
  stopped by the time budget is left out of every comparison.

## Candidate fields
Each candidate goes into the campaign file's `candidates` list (`campaign.md`) with `id`, `gpus`
(per replica, the untuned NIM's), `args` or `env`, and `why` (the hypothesis and the evidence). A
candidate that changes only an environment variable the launch line cannot show also needs
`engagement_pattern`. Keep a plan note per candidate with its expected effect and metric, its
accuracy check if any, and its status: `planned`, `deferred` or `pruned` with a reason.

## Search rounds
After a measured round, promote the winner as the new parent. Plan about three in four new
candidates as one-step changes from it, and one in four from a setting not yet tried. Keep the
parent in every round. Never re-plan a candidate that failed on its configuration; re-plan an
intermittent failure once. Stop when a round gives no `better` result, or the budget is spent.

## Speculative-decoding drafts
Neither `nim-select` nor the NIM Metadata API lists draft models: a profile records precision,
backend, TP, PP, memory, LoRA and tested GPUs only. So:
- Try `ngram` (prompt lookup) first. It needs no draft model.
- For EAGLE3 or EAGLE, search for a draft trained for the exact model (for example on Hugging
  Face). Record its repository, pinned revision, license and training data. Prefer one not
  trained on the evaluation dataset, or evaluate on a split it never saw.
- The NIM pod must be able to download the draft (its Hugging Face token reference and network).
- MTP only when the checkpoint has MTP heads.
- The draft acceptance rate explains a result; the target metric decides.

## Common mistakes
| mistake | do instead |
|---|---|
| Change TP, PP or replicas and call it tuning | Keep the NIM's profile; that is profile selection |
| Propose a flag from memory or a blog post | Check `engine_flags.txt` from the image |
| Raise the memory fraction when the KV cache is far larger than the load needs | Skip it; it helps only when memory is tight |
| Sweep the KV-cache data type on its own | Pair it with the checkpoint's weight precision, and plan an accuracy check |
| Plan speculative decoding on random synthetic prompts | Use a real text dataset |
