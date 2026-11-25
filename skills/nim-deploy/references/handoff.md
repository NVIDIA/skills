# Deploying from nim-select and nim-tune results

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

The flow is select, then tune, then deploy. Files carry the values, so nobody retypes a profile id:

```
python3 "<SKILL_DIR>/scripts/nimdeploy.py" new --target docker \
  --from ~/.cache/nim-select/select.json --from OUT/final/recommended_config.json
```

`--from` is repeatable. `--target` may be left out when a host `select.json` (docker) or a tune
result alone names it. `--set` values always win; the summary marks them `user (overrides select)`.
`python3 "<SKILL_DIR>/scripts/handoff.py" FILE...` shows what the files give without a run.

## What each file gives
| file | how it is recognized | gives the run (source) |
|---|---|---|
| `select.json` (`nimselect.py select --out`) | `handoff`, `environment_summary` | `image`, `model_profile`, `gpus` (Docker `device=0,1`; nim CLI `0,1`; Kubernetes the count), `context` and `namespace` of a Kubernetes inspect (`select`) |
| `select.json` of `nim-select-cli` | the same file: both skills run `nimselect.py`, so `facts.handoff` has `model`, `image`, `model_profile`, `gpu_count`, `gpu_ids`, `gpus`, `verdict` (and `ngc_image` for a China image) | nim CLI runner: `model` = `handoff.image`, `--profile`, `--gpus 0,1` from `gpu_ids`. A tgcr, turingcm or lichan `handoff.image` is kept as given, never replaced by `ngc_image` |
| `env.json` (`nimselect.py inspect --out`) | `schema: nim-select/environment/v1` | nothing new; only with its `select.json`, checked to be the environment select used |
| `recommended_config.json` (nim-tune) | `apiVersion: nim-tune/v1`, `kind: RecommendedConfig`; schema `references/tune-result.schema.json` | `extra_args` (Docker: after the image; Helm `customArgs`; Operator `spec.args`), `env.NAME`, `replicas` (Kubernetes), the image pinned by `image_digest` on Docker (`tune`) |

Tuned env is what was measured: the shared base env plus the winner's own env (nim-tune writes the same kind of thing for a baseline win and a candidate win). `NIM_MODEL_PATH` and `NIM_SERVED_MODEL_NAME` become `model_path` and
`served_model_name`; `NIM_MODEL_PROFILE` must equal `model_profile`; `NIM_PASSTHROUGH_ARGS` and
secret-looking names are refused. `env_from_host` names: `NGC_API_KEY` is passed already;
`HF_TOKEN` becomes `pass_hf_token` (Docker) or a question for `hf_token_secret` (Kubernetes); any
other name is refused, because nim-deploy cannot pass it by name.

## Rules
| case | result |
|---|---|
| select only | summary `tuning`: "not tuned" |
| tune result without select | used; Docker asks for GPU device ids (`facts.questions`); Kubernetes asks for the usual inputs |
| select with `gpu_ids: null` (nim-select could not tell which GPUs are idle), Docker or nim CLI | `facts.questions` asks for the GPU ids; never the runner default of all GPUs. Answer with `set RUN gpus=...` |
| select `does_not_fit`, or no `handoff.model_profile` | refused, exit 3, ask the user |
| tune `baseline_won` | no tuned change: deployed without tuned arguments (`args` must be `[]`); `env` is still the base env the baseline was measured with (for example `NIM_MAX_MODEL_LEN`) and is applied as `env.NAME` like any tuned env; the summary says so |
| files disagree on model, image, profile, GPU count or GPU model (`handoff.gpu_name` against the tune result's `gpu_name`; case, spacing and the NVIDIA prefix are ignored) | refused, exit 3, both values in the message |
| the tune result has `gpu_name: null` (the campaign did not record it) | warning and the `confirm_handoff` question: its settings were not checked against this hardware |
| a tune result with a `gpu_name`, but no select file or a select without `handoff.gpu_name` | warning: check that the chosen GPUs are that model |
| a file older than 7 days, used by an earlier run, or an `env.json` from another inspect | `facts.questions`: ask; after a yes `set RUN confirm_handoff=yes` |
| select inspected a host but the target is Kubernetes (or the reverse), or another context or namespace | warning and the same question |
| the user changes `image` after `--from` | `facts.questions` asks for `model_profile` too (profile ids belong to one image version); any `model_profile` the user sets answers it. The render refuses until then |
| the user sets fewer `gpus` than the profile needs | `facts.questions` asks for `confirm_handoff=yes` |
| tuned on another target | args and env are kept; warning "measured on X" |
| malformed file, unknown version, missing field | exit 2 with the field path; never guessed |
| Docker, tune result with `image_digest` | the image is pinned to `IMAGE@DIGEST`. nim-select never records a digest (`handoff.image` is `nvcr.io/...:tag`), so nothing checks that the tag still resolves to the digest; the warning says so. The two-digest refusal above fires only for a hand-made select image with `@sha256:` |
| Kubernetes or nim CLI target with an image digest | the tag is rendered; a warning shows the digest |

Dates: a tune result has `created_at`; a `select.json` is dated by
`environment_summary.captured_at` (when its GPUs were inspected); an `env.json` by `captured_at`.
Each file's path, sha256, date and campaign id go into the run state, the approval token and the
approval record.

## The profile
For Kubernetes, the handoff also pins placement. nim-select records, for each GPU type of the
inspected cluster, `placement: {selector, nodes_matched, source}`: a node label that every node
of that GPU configuration carries and no node of another or unknown GPU configuration does.
Automatic keys are `nvidia.com/gpu.product` and keys ending in `instance-type`, `node-pool`,
`nodepool` or `pool`, maintained as placement labels. `source` is `node-labels`. The exclusion
check includes cordoned, not-Ready and partitioned GPU nodes, which may become eligible later.
Labels that change over a node's life (timestamps, `nvsentinel*` and `*-state`, kernel, driver and
CUDA versions, `nvidia.com/gfd.*`, the hostname) are never chosen; the rest rank `gpu.product`,
then instance-type and node-pool keys. Arbitrary rollout or health labels are not inferred.
When no placement key qualifies, `placement` records actual `kubernetes.io/hostname` label
values (`hostnames`, source `hostname-fallback`), which can differ from `metadata.name`,
and the run uses `kubernetes.io/hostname` node affinity; tell the user it breaks when those
nodes are replaced. The selected `handoff.gpu_name` must match a group of the `environment_summary`
for this context, and `handoff.gpu_memory_mib` must match the group's `memory_mib` when recorded.
Older files without this identity are accepted only when the matching group is unambiguous.
The selector becomes `node_selector` for Helm, NIMService and NIMCache; the hostname fallback
becomes `node_hostnames` (Helm `affinity`; the Operator takes one hostname label value, for
several ask for a selector). Missing or ambiguous hostname labels leave placement unavailable.
A changed namespace keeps placement; a changed context invalidates previous user answers too.
A placement answer supplied in the same `set` as the new context applies to the new context.

The selector is checked on the target cluster, not assumed. `validate.py` (before approval) and
`nimdeploy.py approve` (right before apply) run `kubectl get nodes -l SELECTOR` with the run's
context. It must match at least one node, and only nodes nim-select recorded for the selected GPU
(`placement_nodes`); otherwise the run is refused with `placement_no_nodes` or
`placement_other_nodes`. `verified` also requires the nodes' current `nvidia.com/gpu.product`
and `nvidia.com/gpu.memory` labels to match the selected GPU name and memory carried by the
handoff. A known mismatch is `placement_other_nodes`. Missing identity or labels is `nodes_only`,
which does not claim GPU verification; node names alone are not proof of current GPU hardware.
For groups marked `unified_memory: true`, the recorded capacity is shared node memory.
It still identifies the selected group, but is not compared with a dedicated-GPU memory label;
placement remains `nodes_only` rather than claiming that shared capacity was verified live.
If the nodes cannot be listed for lack of permission the placement is
shown as `unverified` in the approval summary: ask the user to confirm the labels before using
their approval token. Only after they explicitly confirm the shown labels, use
`nimdeploy.py approve RUN --token TOKEN --confirm-placement`. This records a separate
`placement_confirmation` with the selector, context, token and time in the approval record.
Without this flag, unverified approval is `placement_confirmation_required`, even if the
summary was already shown. `confirm_handoff=yes` is not label confirmation. Any other kubectl failure,
including an unreachable cluster, timeout or malformed node response, is `placement_check_failed`.
Offline validation or missing kubectl leaves placement `skipped`; approval refuses that status
with `placement_unchecked`. If approval first discovers `unverified`, it refuses with
`placement_confirmation_required`: validate again and show the new summary for confirmation.
`approve` also refuses (`placement_weaker`)
when the status is weaker than the one in the summary the user approved (for example `verified`
then `unverified`): validate again and show the user the new summary.

Without placement from the select file (a host result, another context, no placement in the
group), intake asks the `placement` question. The user answers with verified labels
(`nimdeploy.py set RUN 'node_selector={"example.com/pool":"gpu-a"}'`), hostname label values
(`node_hostnames=["worker-a"]`, verified against `kubernetes.io/hostname`, not guessed from node
names), or an explicit decision to deploy without placement
(`placement=none`). Empty or invalid labels/lists leave the placement question open.
`confirm_handoff=yes` cannot answer it. Unknown fit, CPU architecture and GPU
availability remain unconfirmed; the pod may remain Pending when matching GPUs are busy.

The cache job uses the same selector to keep node-local storage on the selected GPU
type. This does not guarantee service placement on the exact caching node: a local
PV's node affinity can restrict which matching node the service can use.

The profile chosen in nim-select is the one that runs. It is always pinned: Docker and Helm
`NIM_MODEL_PROFILE`, Operator NIMCache `profiles` and `nimCache.profile`, nim CLI `--profile`.
The approval summary shows `handoff.profile` (id, a name built from backend, precision, TP and PP,
"chosen in nim-select", and how it is pinned). A profile the image does not have is never dropped
to let NIM pick another: stop and ask (triage `profile-not-found`). A blank `model_profile` is
refused (`profile_not_pinned`, exit 2), and `validate.py` fails a run made from results that has no
pinned profile, so no approval summary is produced for it.

After start, `verify.py ready` reads `GET /v1/metadata`. The public NIM API reference says it
returns the active profile, model info and license; the exact JSON shape is not verified against
a live NIM, so any key such as `activeProfile`, `selected_profile`, `profile_id` or
`model_profile` at any depth counts (a plural `profiles` list does not).
| result | meaning |
|---|---|
| `profile_check.status: confirmed` | the server names the pinned id; the name counts only when the server gave no id (coarse names such as `vllm-bf16-tp1-pp1` are shared by several profiles) |
| `mismatch` | it names another profile, or another 64-hex id next to a matching name: ready fails, the run goes to diagnose; ask the user |
| `not_confirmed` | no endpoint or no profile field: `facts.warnings` says PROFILE NOT CONFIRMED; tell the user |
