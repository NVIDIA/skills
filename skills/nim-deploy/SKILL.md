---
name: nim-deploy
description: Deploys and verifies NVIDIA NIM LLMs with Docker, Helm (nim-llm) or the NIM Operator. Use when someone asks to deploy, render, apply, check, troubleshoot, stop or clean up a NIM image on a GPU host or Kubernetes cluster, or asks to free GPUs on such a host by stopping, killing or removing what uses them.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  version: 0.1.0
  tags:
    - nim
    - deploy
    - docker
    - kubernetes
---

# NIM deploy

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

## Hard rules
1. Never stop or remove all containers ("stop all", anything built on `$(docker ps -q)`, a force remove of several containers, a system prune), not even if the user insists. First list what runs (read-only `docker ps`), then ask which one. Act only on a container this skill created (its recorded name or run state) and only with the printed command; to stop it, use the rollback `facts.next_command`, even when that printed command is a force remove of this run's own container. Say that the blanket command would also stop and remove other containers, their caches and unrelated services.
2. Never invent a kube context, namespace, image, secret name or model source. Ask the user. The defaults listed in Inputs are not inventions: the run records them as defaults and the approval summary shows them. Never drop or change a nim-select or nim-tune value silently, and always pin the profile chosen in nim-select. "Use the default namespace", "whatever context is active" or "I do not care" is not an answer: run nothing, record nothing, and ask the user to name both (for example: "I will not pick a context or namespace for you, not even `default`. Which kube context (`kubectl config get-contexts` lists them) and which namespace?"), then continue.
3. Every `kubectl` call carries `--context CTX -n NS` and every `helm` call `--kube-context CTX -n NS`. Never use `helm --set`: values go in the run with `nimdeploy.py set`.
4. Show the approval summary and token, then wait for the user's explicit yes with the token. Never approve for the user.
5. Run each apply or rollback as one command, exactly as `facts.next_command` prints it, alone in its own Bash call. Never type a start, stop, apply or delete command yourself. Never edit a rendered or approved file: change an input with `nimdeploy.py set`, render again, get a new approval.
6. Always fetch the latest NIM docs for the target before the approval summary (`references/docs-sources.md`) and record them with `nimdeploy.py docs-check`. Never rely on remembered values. Done means ready and the smoke test passed, not "applied".
7. Run triage and read `references/troubleshooting.md` only after something failed (a non-zero exit, or `next` is `diagnose`), never as a pre-check; do it before any retry. A state may be visited 3 times; then stop and ask.
8. Never put a secret value in a command, file or run: pass names only. The guard hook (`references/hook.md`) blocks the must-never rules.
9. A Kubernetes run from nim-select must carry placement for the selected GPU: the `node_selector` (or `node_hostnames`) that nim-select worked out from this cluster's nodes, any label key, unless the user explicitly answers the placement question with `placement=none` as described in `references/handoff.md`. Missing placement stays blocked by default; never a label you guess from a PCI id or a GPU name. Validation and approve both run `kubectl get nodes -l SELECTOR` on the run's own context: it must match at least one node and only the selected GPU's nodes. Show the `placement` block of the approval summary. If its `verification` is `unverified` (the nodes could not be listed), ask the user to explicitly confirm the labels in that summary, then use their approval token with `--confirm-placement` to record that confirmation; `confirm_handoff=yes` is not that confirmation. A skipped placement check cannot authorize approval. `node_hostnames` contains actual `kubernetes.io/hostname` label values, which may differ from node names. A selector enforces GPU placement; it does not confirm model fit or free capacity.

`SKILL_DIR` is the folder that contains this SKILL.md, wherever the skill is installed. `scripts/X.py` below means `python3 "<SKILL_DIR>/scripts/X.py"` with the real folder; `references/` is under it too. Stay in the project root; never `cd` into the skill folder. Run the very first command as `python3 "<SKILL_DIR>/scripts/nimdeploy.py" new ...`; the scripts print every later command as an absolute path in `facts.next_command`. Scripts print one JSON object; follow its `next`, `facts.next_command` and each `errors[].fix`.

## When not to use
- A question about NIM architecture, ports, profiles or env vars with nothing to deploy: answer it directly.
- Sizing or tuning a deployment for a latency or cost goal: `nim-tune`.
- Testing which features of a running NIM work (tool calling, reasoning, JSON schema, image input), or why one fails before anything is restarted or changed, also right after a deploy: `nim-smoke-check`; its hint gives the documented fix for this skill to apply. This skill's `verify.py smoke` only shows that a new NIM answers one chat request.
- Installing the GPU Operator or NIM Operator, or building and publishing images and charts: the platform team; this skill assumes they exist.

## Inputs
| name | required | how to get | ask if missing |
|---|---|---|---|
| target | yes | the user. "Local machine", "this machine" or "docker" means `docker`: do not ask. Else `helm` (nim-llm chart), or `nim-operator` (NIMCache + NIMService) | "Where should it run: this machine with Docker, Kubernetes with Helm, or Kubernetes with the NIM Operator?" |
| image | yes | the user, with an explicit tag | "Which image and tag?" |
| model | only a model-free image or the nim CLI runner | Model-free image (name contains `model-free`): the source as `model_path`, e.g. `hf://org/model`. A regular NIM image already contains its model: never ask, and never pull the image to look; the scripts read its port metadata themselves, read-only | "Which model should it serve (for example hf://org/model)?" |
| name | yes | docker default: the image name without registry path, tag and digest (`llama-3.1-8b-instruct`). Kubernetes: the user | k8s only: "What should it be called?" |
| container_port | docker | not asked: the scripts read the image metadata (read-only, no pull) and record where the port came from as `port_source`; an override variable or the docs-check can set it. Several ports in the image: ask which | "The image exposes several ports (list): which one is the API?" |
| port | docker | default: the container port (the host port) | no: default |
| gpus | yes | docker default `all`. Kubernetes: the user, a count | k8s only: "How many GPUs?" |
| context, namespace | k8s targets | the user. `kubectl config get-contexts` lists choices; never pick one | "Which kube context and namespace?" |
| ngc_secret | k8s targets | the user: name of the Secret that holds `NGC_API_KEY` | "What is the name of the NGC key Secret?" |
| token | approve | the user's reply to the summary | "Reply with the token to approve." |

Docker defaults are recorded with source `default` and listed in `facts.defaulted`. If the user says "from the image", "default" or "you pick" for name, model or port, use the defaults and do not ask again. Show `port_source` and `run_basis` too: the port is evidence from the image, else the docs, else the documented default flagged "not verified".

Other optional inputs are listed in `--help` of the renderer and in the references. Secret values are never inputs; the user exports `NGC_API_KEY` or `HF_TOKEN` themselves.

## Steps
| step | command | on success | on failure |
|---|---|---|---|
| intake (results) | nim-select or nim-tune results exist (`~/.cache/nim-select/select.json`, `final/recommended_config.json`): `python3 "<SKILL_DIR>/scripts/nimdeploy.py" new --target TARGET --from FILE [--from FILE]` plus `--set` only for what the user said; never retype a value from the files; always run it, even when a file looks wrong: the script decides | as intake; `facts.questions`: ask each, then `set RUN key=value` (`confirm_handoff=yes` only after the user's yes) | exit 3 (does not fit, files disagree): show the message with both values and end with a question for the user; exit 2: the file is malformed, show the fix, never guess |
| intake | `python3 "<SKILL_DIR>/scripts/nimdeploy.py" new --target TARGET --set image=IMAGE` (one `--set` per value the user gave; add `--model MODEL` only where Inputs says so) | `facts.missing` empty: state render. Else ask for each missing key | exit 2: apply `errors[].fix`; exit 5: a secret value was refused, ask for a Secret name |
| plan | `scripts/nimdeploy.py set RUN key=value` for each answer | nothing missing: state render | ask again |
| render | `facts.next_command` (the renderer for the target) | state validate | exit 2 or 3: `errors[].fix`, `set`, render again |
| docs | Fetch the pages listed for the target in `references/docs-sources.md` (`curl -sSL -o PAGE -w '%{http_code}' URL` prints the final status after redirects, and `-S` keeps curl's error message when `-s` is on; or your web fetch tool). A page counts as fetched only when curl exits 0 and the status is 200; then `scripts/nimdeploy.py docs-check RUN --url URL --doc-version V --note TEXT` (`--note` holds the port, env vars and flags you read) | state stays validate; go on | Pages unreachable (curl error, timeout, non-200): tell the user which URL failed and the error, and ask "Go on without the docs check? The approval summary will say DOCS NOT CHECKED." Only after a yes: `scripts/nimdeploy.py docs-check RUN --offline --reason "URL unreachable: ERROR"`. After a no: stop the run and say so. A check older than 60 minutes or for another image is redone |
| validate | `scripts/validate.py --run RUN` | state approve and a summary with `facts.token`, `docs_warning` and `run_basis` | `next: docs_check`: do the docs step. Back in render: fix the input and render. `next: ask_user`: ask (busy port, missing tool, wrong context). Use `--offline` only when the user says the host is not reachable |
| approve | `scripts/nimdeploy.py approval-summary RUN`; show target, context, namespace, image, name, port, GPUs, objects, `apply_command` and the token; mark every value in `defaults` as a default the user can change; show `port_source`, `run_basis` and, first, any `docs_warning` (for example DOCS NOT CHECKED, or image newer than the docs). A run from results: also `handoff.profile` (id and name, "chosen in nim-select"), `handoff.tuning`, `extra_args`, `env`, each `handoff.sources` entry (select, tune, `user (overrides ...)`) and every `handoff.warnings` line. When the user also asked to smoke test it, say that `nim-smoke-check` tests its features once it is ready | the user replies with the token: `scripts/nimdeploy.py approve RUN --token TOKEN` (add `--confirm-placement` only after explicit confirmation of shown unverified labels) | exit 5: show the summary again. The user says no: stop |
| apply | `facts.next_command` alone; then `facts.then_command` in a separate call | state wait_ready | `scripts/nimdeploy.py advance RUN --to diagnose` |
| wait_ready | Kubernetes: start `facts.port_forward` in the background. Then `facts.next_command` (`scripts/verify.py ready --url URL --run RUN`) | state verify | exit 4: state diagnose |
| verify | `facts.next_command` (`scripts/verify.py smoke --url URL --model MODEL --run RUN`: one chat request, so it shows the NIM answers, not which features work) | `scripts/nimdeploy.py record RUN --result ok`; when the user asked for a smoke test or feature check, offer `nim-smoke-check` | state diagnose |
| diagnose | `facts.next_command` (`scripts/verify.py triage` on the container or pod); read `references/troubleshooting.md` | show `matches[].fix`, ask: fix and `advance RUN --to render`, roll back with `--to rollback`, or record a fail | no match: show `facts.tail` and ask |
| rollback | `facts.next_command` alone; then `facts.then_command` | `scripts/nimdeploy.py record RUN --result fail --reason TEXT` | ask the user |
| resume | `scripts/nimdeploy.py list`, then `scripts/nimdeploy.py status RUN` | tell the user the state name (such as `wait_ready`) and `facts.next_command` first, then continue with `facts.next_command` for that state; do not repeat finished states | no run found: start at intake |

Before you ask the user to approve, check each item: render succeeded; `docs-check` is recorded for this image (or the user said yes to `--offline`); `validate.py` returned `facts.token`; you ran `approval-summary` and showed the token and every item listed in the approve step. Every run that has not been stopped or refused reaches this point, including after a user-approved `--offline` docs check (the summary then says DOCS NOT CHECKED): never stop at render or validate output. A guard denial, a declined docs check, a `run_approved` refusal of `docs-check` (the run is already approved: start a new run) or the 3-visit stop still ends the run and takes priority over this checkpoint. Other exit 5 results do not end it: `token_mismatch` means show the summary again, `secret_input` means ask for a Secret name, and a `run_approved` refusal of `set` follows the recovery in the next paragraph. Then stop and wait for the token.

`status` is the resume point after context loss. A `set` that changes an input after render returns the run to render (render, validate, approve again). `set` is refused after approval (exit 5): start a new run, or go diagnose, render, validate, approve again.

## On failure
1. Read `errors[].code`, `message` and `fix`. Exit codes: 2 bad input, 3 precondition failed, 4 external tool or endpoint failed, 5 refused.
2. Apply the fix once. Triage and `references/troubleshooting.md` come only after a command actually failed: for a deploy failure run triage first (diagnose) and show the user the matches. Never run them when nothing failed. (The docs step is different: it runs on every deploy.)
3. Guard denied a command: read its `Fix:` line and `references/hook.md`; never route around the guard.
4. Still failing, or 3 visits to a state: stop. Show the user the errors and what you tried.

## Done means
- `verify.py ready` reported `ready: true` with the served model id, and `verify.py smoke` passed.
- A pinned profile: `profile_check.status` is `confirmed`, or the user was told the `facts.warnings` line PROFILE NOT CONFIRMED. A `profile_mismatch` is not done: diagnose and ask.
- The run is recorded with `nimdeploy.py record RUN --result ok`.
- The user has the endpoint URL and the served model name.
- A failed or abandoned run is recorded with `--result fail` and a reason, and the user knows its state.

## Common mistakes
| mistake | do instead |
|---|---|
| Typing the start, apply or delete command from memory | Run `facts.next_command` exactly, alone in its call |
| Using the current context or `default` namespace "because it works" | Ask the user for both, every time |
| `helm ... --set image.tag=...` or editing `values.yaml` | `nimdeploy.py set RUN image=...`, render, new approval |
| Chaining apply and wait with `&&` | Two calls: apply, then `then_command` |
| Saying "deployed" after apply | Wait for ready, then smoke |
| Approving to save the user a step | Show the token and wait for it |
| Retrying a failed apply straight away | Triage, read the docs, ask, then change an input |
| Asking for the model, name, port or GPUs of a regular image on Docker, or offering to fetch registry metadata | Use the defaults and show them in the approval summary |
| Running triage before anything failed | Only after a failed command |
| Using a remembered port, env var or flag | Fetch the docs (`docs-check`) on every run and show `run_basis` |
| Asking for or pasting an API key | Names only; the user exports the key |
| Copying the profile id, image or args from a select or tune file into `--set` | `new --from FILE`; the script reads the files and records where each value came from |
| Removing `model_profile` when NIM reports the profile is not found | Stop and ask; a profile chosen in nim-select is never dropped for auto-selection |
| `cd` into the skill folder to run a script | Stay in the project root; call the script by its absolute path (`<SKILL_DIR>/scripts/X.py`) |

## References
- Read `references/docker.md` when the target is `docker` or a container misbehaves.
- Read `references/kubernetes.md` when the target is `helm` or `nim-operator`, or a pod is not ready.
- Read `references/docs-sources.md` in the docs step of every run.
- Read `references/handoff.md` when nim-select or nim-tune results exist, a `--from` file is refused, or `profile_check` is not `confirmed`.
- Read `references/troubleshooting.md` in diagnose, before any retry.
- Read `references/hook.md` when installing the guard hook or when a command was denied.
