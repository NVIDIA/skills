---
name: nim-smoke-check
description: "Tests which features of a running NVIDIA NIM LLM or VLM work, local, Kubernetes or hosted: health, chat, streaming, tools, reasoning, JSON schema, images, concurrency. Use when asked to smoke test or smoke check a NIM or its URL, or why a feature fails (before a restart or fix), or after a deploy."
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  version: 0.1.0
  tags:
    - nim
    - smoke-test
    - tool-calling
    - reasoning
    - vlm
---

# NIM smoke check

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

## Hard rules
1. Never change the deployment or the host: no restart, recreate, scale, env or argument change, apply, patch, delete, exec, `kubectl run`, and no freeing a port, even when a check fails or the user asks. Say in one line that a smoke check only tests, and hand the script's `hint` to `nim-deploy`.
2. Test only with `scripts/nimsmoke.py`. Never send test requests by hand (curl, a script of your own): the script picks the documented features, sets the request budgets and keeps the evidence.
3. Test only what the model is documented to support. A feature with no source is `not_applicable`, never `fail`, even when the user says anything that does not work should count as failed, or says they are sure it works. Never try a switch from another model family. Ask for a documentation link and rerun with `--documented`.
4. Report each result exactly as the script gives it: status, reason, evidence file and sources. Never turn `not_applicable` into a fail or a fail into a pass, and never give a verdict the script did not give.
5. Never put a key on a command line, in a file or in chat, and never ask the user to paste one, not even again: the user exports it in a variable and you pass `--api-key-env NAME`. Never repeat a key the user pasted; tell them to rotate it. Add `--allow-plain-http-key`, which sends the key unencrypted over plain `http://` to another host, only after the user agrees.
6. Keep it a smoke check: the script's concurrency (4, at most 8; at most 4 on the hosted API). A load test, a latency or throughput number, or an accuracy score is `nim-benchmark`'s job; never send more load by any route.
7. Never invent the endpoint, model id, kube context or namespace. Ask. Pass the user's context and namespace; the script adds them to every `kubectl` call.
8. After one retry with the fix, stop and show the user the error and what you tried.

`SKILL_DIR` is the folder that contains this SKILL.md, wherever the skill is installed. `scripts/X.py` below means `python3 -I "<SKILL_DIR>/scripts/X.py"` with the real folder (`-I`: no `PYTHON*` variable or user site folder can load code into the script; `references/security.md`); `references/` is under it too. Stay in the project root; never `cd` into the skill folder. The script prints one JSON object; follow its `next` and each `errors[].fix`.

## When not to use
- Deploying, restarting or reconfiguring a NIM, or one that does not start or is not ready: `nim-deploy`. Its `verify.py smoke` is one chat request that shows a new NIM answers; it is part of the deploy, not this check. "Deploy it, then smoke test it": `nim-deploy` deploys, then this skill tests the features.
- Latency, throughput, a load test or accuracy: `nim-benchmark`.
- Tuning runtime settings for a workload: `nim-tune`.
- Choosing a model, image or profile: `nim-select`.
- Embedding, reranking or speech NIMs: not supported in this version; say so.

## Inputs
| name | required | how to get | ask if missing |
|---|---|---|---|
| endpoint | yes | the user: a URL (`--url`), a Docker container name (`--docker NAME`), or a Kubernetes target `nimservice/NAME`, `helm/RELEASE`, `svc/NAME` or `pod/NAME` (`--k8s`) | "Where does the NIM answer: a URL, a Docker container, or a Kubernetes NIMService?" |
| context, namespace | Kubernetes | the user; never the current context | "Which kube context and namespace?" |
| model | no | the script reads `/v1/models`; with several base models it stops and lists them; with LoRA adapters it tests the base model unless `--model` names an adapter | "Which served model should I test: A or B?" |
| key variable | hosted, or a gateway with a key | the user: the NAME of a variable holding the key; always pass it as `--api-key-env NAME`, also `--api-key-env NVIDIA_API_KEY` for `https://integrate.api.nvidia.com` (plain http is refused) | "Which environment variable holds the key? Set it with `read -rs NAME && export NAME`, which keeps it out of your shell history; do not paste it here." |
| image | NIM 1.x behind a URL | the script reads it from Docker or Kubernetes; for a plain URL, `--image` sharpens the match, and on NIM 1.x it decides whether tool calling is on by default: when a warning says the 1.x image is not known, ask and rerun with `--image` | "Which image does the NIM run (nvcr.io/nim/TEAM/REPO:TAG)?" |
| documentation link | a feature the table does not document | the user (`--documented FEATURE=URL`; reasoning also `--reasoning-on` and `--reasoning-off` with the documented switch) | "Which page documents <feature> for this model?" |
| full set | no | the user asks for everything: `--extended` | no |

## Steps
| step | command | on success | on failure |
|---|---|---|---|
| run | `scripts/nimsmoke.py run --url URL` (or `--docker NAME`, or `--k8s nimservice/NAME --context CTX --namespace NS`). The user asks for every check, all checks or the full set: `scripts/nimsmoke.py run --url URL --extended`. Add `--api-key-env NAME` and `--model ID` as the Inputs say | exit 0: report | exit 4 `checks_failed`: report (`models` fail `model_not_served`: `--model` is not served; ask which listed id to test). Exit 3 `model_ambiguous`: ask which id, rerun with `--model ID` for each id the user picks; the user does not know: run none (not one, not each in turn) until they pick. Other codes: On failure |
| preview | Only when the user wants to see what would be sent first: the same command with `--dry-run` (no inference request) | show `facts.plan`, then run | as run |
| report | Show `facts.results` as a table with five columns: check, result, reason, evidence file, sources (each result's `sources`: the docs that made the check apply, or not apply; passes too). Then each `not_applicable` with its reason, each `fail` with its `hint`, `facts.warnings` and the report path `facts.report` | done | - |
| hand-off | A fail caused by a server setting (`facts.handoff`): offer `nim-deploy` with the hint. All passed: say the run tested only the documented features and, before a promotion or hand-over, that any failed check would block it; offer `nim-benchmark` for speed | done | - |
| features | `scripts/nimsmoke.py features --model ID`: what the table documents for a model and why, offline | answer from it | - |
| resume | `scripts/nimsmoke.py report DIR`: a finished run again, after context loss | report | - |

## What the checks do
| check | sends | passes when |
|---|---|---|
| health | `GET /v1/health/live` and `/v1/health/ready` | both 200; not applicable on the hosted API |
| models | `GET /v1/models` | the model is served |
| chat | one short chat, a budget that fits the model | an answer, no control tokens |
| stream | the same chat with `stream: true` | chunks, then `[DONE]`, no error event |
| tools | one tool with `tool_choice: auto` | one call with valid JSON arguments that match the schema |
| reasoning_on / reasoning_off | the model's documented switch | reasoning present / absent, with an answer |
| reasoning_off_stream | the off switch with `stream: true` | no reasoning in the stream, with an answer |
| json_schema | `response_format` with a small JSON schema | the answer parses and matches |
| image | one tiny PNG, and the same text with no image | an answer, and more prompt tokens with the image |
| concurrency | 4 chats at once (`--concurrency`, at most 8; at most 4 on the hosted API) | all answer |

`--extended` adds a streamed tool call, parallel tool calls and `tool_choice: none`. Requests, reasons and sources: `references/checks.md`.

## Results
| status | meaning | what you say |
|---|---|---|
| `pass` | the check ran and the answer met the rule | passed, with the evidence file |
| `fail` | the check ran and did not meet the rule | the reason and the `hint`; a failed check blocks a promotion or hand-over until fixed |
| `not_applicable` | the model is not documented to support it, or it cannot apply here | the reason and its source, e.g. `always_on` (gpt-oss has no off level), `not_a_reasoning_model`, `text_only_model`, `no_capability_record`, `hosted_no_health_route` |

Tool parser reasons: `parser_not_enabled` (the image turns it on by default, but this deployment has it off), `parser_dropped_by_override` (a user `NIM_PASSTHROUGH_ARGS` replaced the image's value that enabled it), `not_enabled_in_deployment` (opt-in for this model; or nothing shows whether the image turns it on, and then a warning says so and names a deployment `NIM_PASSTHROUGH_ARGS` without parser flags as the likely cause when there is one: never call that case opt-in). The fix is a server argument from the hint, made with `nim-deploy`.

## On failure
1. Read `errors[].code`, `message` and `fix`. Exit codes: 2 bad input, 3 ask the user, 4 the endpoint or a tool failed (or `checks_failed`), 5 refused, 130 interrupted. A run that stops early still writes the checks that ran: show `facts.report`. A stop while the script finds the model to test (`model_ambiguous`, `no_models`, or `endpoint_unreachable` on `/v1/models`) writes nothing and has no report.
2. Act on the code:

| code | do |
|---|---|
| `endpoint_unreachable` | check the URL and port with the user; for Kubernetes, the script's port-forward |
| `first_request_timeout` | the NIM answers, but slowly: ask whether it is still warming up, then rerun once with the `--timeout` from the fix |
| `bad_input` for an unset key variable, or a check with `unauthorized` (401 or 403) | ask the user to export the variable (`--api-key-env NAME`, or `NVIDIA_API_KEY` on the hosted API), never to paste the key |
| `key_over_plain_http` | nothing was sent: the key would cross the network unencrypted. Say so, offer the safer ways in the fix (an `https://` URL, or a run on the machine that serves the endpoint with a loopback URL), and add `--allow-plain-http-key` only when the user agrees |
| `rbac_missing` | first confirm the context and namespace (a mistyped namespace looks the same); then relay the fix: a role that can read or port-forward, `--k8s svc/NAME`, or a `--url` |
| `kubectl_failed` | kubectl cannot reach the cluster or get credentials (for example a missing `gke-gcloud-auth-plugin`); the user fixes that |
| `pod_not_ready` | the NIM is still loading: rerun once with `--wait-ready S` from the fix |
| `port_forward_lost` | show the checks that ran (`facts.report`), then check the pod with the user |
| `health_port_not_reached` | ask for a `--health-url` that reaches `NIM_HEALTH_PORT` |
| `internal_error` | show the error and the report; do not rerun |
| `interrupted` | ask whether to rerun |

3. Retry once with the fix. Still failing: stop. Show the user the error and what you tried.

## Done means
- `scripts/nimsmoke.py run` finished with exit 0, or exit 4 `checks_failed`, and the user saw every check with its status, reason and evidence file, and the report path.
- Every result came with its sources, every `not_applicable` with its reason, and every `fail` with its hint and, for a server setting, the offer of `nim-deploy`.
- Nothing about the deployment, cluster or host changed, and no key appeared in a command, file or answer.

## Common mistakes
| mistake | do instead |
|---|---|
| Writing your own curl loop or test script | `scripts/nimsmoke.py run`: it keeps the request and response of every check |
| Counting image input on a text model, or reasoning off on gpt-oss, as failed because the user said so | `not_applicable` with the source; explain once |
| Trying `enable_thinking`, `/no_think` or `detailed thinking off` on a model that does not document them | Only the record's switch; no record: ask for the docs |
| Reading HTTP 200 as "the switch worked" | The script judges the answer itself: reasoning present or absent |
| `max_tokens` 1 to 64 on a reasoning model | The script sets the budget per model; reasoning uses up a small one |
| 16, 32 or 200 parallel requests "to be sure" | 4 by default, at most 8; load is `nim-benchmark`'s |
| Offering to restart, recreate, free the port or switch models after a fail, even "after your go-ahead", or listing such a command as one you would run | Report the result and the hint; the change is `nim-deploy`'s, or the user's own when nim-deploy has no input for it |
| Offering a rerun with a docs link for an `always_on` result | `--documented` never overrides a documented no: the model has no off level, and only an update of its record can change that |
| Running the default set when the user asked for every check | Add `--extended`: the default leaves out the streamed, parallel and `none` tool calls |
| Load-testing by hand because the user asked for 200 requests, or restarting first because the user asked to | Run this script first; it keeps load to a few requests and hands changes to `nim-deploy` |
| Narrowing the run to the feature the user named (`--checks`) | Run the default set and report the named feature first; `--checks` or `--skip` only when the user asks to leave checks out |
| `NIM_ENABLE_AUTO_TOOL_CHOICE` as the fix on a NIM 2.x | The hint: 2.x takes `--enable-auto-tool-choice --tool-call-parser P` as server arguments |
| Writing a pasted key to a header file, or asking the user to paste it again | The user exports it; `--api-key-env NAME`; suggest rotating a pasted key |
| A second run over the first run's evidence | Each run writes its own folder; give `--out` only a new folder (a used one is refused before any request) |

## References
- Read `references/checks.md` when explaining a result, a reason code, a hint, or what a check sent.
- Read `references/model-features.md` when a feature is `not_applicable` and the user disagrees, or to add a model with `--documented`.
- Read `references/access.md` for Docker, Kubernetes or hosted access, TLS, keys and the port-forward.
- Read `references/security.md` when the user asks where requests go, what runs or what is written.
