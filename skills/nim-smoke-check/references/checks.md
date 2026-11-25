# Checks: what each sends, when it passes, why it fails

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

`scripts/nsc_checks.py` runs the checks. `scripts/nsc_features.py` decides first, per check, `test`
or `not_applicable` (`references/model-features.md`). A check that runs sends the request below,
with the switch, sampling and `max_tokens` of its plan entry.

Each result has:

| field | holds |
|---|---|
| `status` | `pass`, `fail` or `not_applicable` |
| `reason` | a code from the tables below; `ok` on every pass |
| `message` | one short line; never a full prompt or answer |
| `evidence` | the evidence files of the check, `evidence/<check>-<n>.json` |
| `sources` | the public pages behind the expectation |
| `hint` | the documented server switch, when a server setting is the likely cause; the change is `nim-deploy`'s |
| `known_issues` | release-note anchors for this model and check |
| `warnings` | findings that are not a fail |
| `attempts` | 1, or 2 after the one retry a check allows; `health`: the number of probes |

On stdout each result keeps `id`, `status`, `reason`, `message`, `evidence`, `sources` and `hint`;
`facts.warnings` holds the run's warnings and every check's, as `<check>: <text>`, then the known
issues of each check that failed on how the model answered (its message points there; a fail to
reach the endpoint, such as `unauthorized` or `timeout`, does not). `report.md` gives each
not-applicable check with its own message, sources and hint. Text in it from the endpoint (a
model id, a message) is one line with its Markdown characters escaped, so it cannot change the
report's structure.

## Rules for every check

| rule | what the script does | source |
|---|---|---|
| URL | trims a trailing `/` and `/v1`, then appends `/v1/...`; the hosted base is `https://integrate.api.nvidia.com/v1` | https://docs.api.nvidia.com/nim/reference/llm-apis |
| headers | `Content-Type: application/json` on every POST (vLLM answers another type with 400); `Accept: text/event-stream` for streams, else `application/json`; `Authorization: Bearer` only with `--api-key-env`, never sent on to a redirect, and over plain `http://` only to this machine unless `--allow-plain-http-key` (`key_over_plain_http`) | https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/utils/api_utils.py, https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/api_router.py |
| request id | `X-Request-Id: nsc-<run>-<check>-<n>`; NIM echoes it and passes it to the engine, so evidence matches server logs | https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/logging-and-observability.html#x-request-id |
| request fields | only `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stream_options`, `tools`, `tool_choice`, `parallel_tool_calls` (only `true`, only in `tools_parallel`), `response_format`, `chat_template_kwargs` and `reasoning_effort` (both only from a documented switch). vLLM accepts unknown fields silently, so a wrong field gives a false result, not an error | https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/engine/protocol.py |
| routes | GET `/v1/health/live`, `/v1/health/ready`, `/health`, `/v1/models` (hosted, with a model id: `/v1/models/ID`), and on a self-hosted NIM `/v1/version` and `/v1/metadata`, which are recorded as evidence (`probe_version`, `probe_metadata`) and never judged; POST `/v1/chat/completions`. No route that changes state (LoRA load, `/sleep`, `/reset_*`, profiling) | https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/api-reference.html#inference-endpoints |
| timeout | per chat request `60 s + max_tokens × 0.05 s`, at most 600 s; `--timeout S` gives every chat request S seconds instead (up to 3600: more for a slow NIM, or less); GETs 30 s, or S when lower. Slow profiles can exceed client timeouts, and the docs say: "Set the client request timeout to accommodate the expected generation time" | https://docs.nvidia.com/nim/large-language-models/2.1.5/troubleshooting/requests.html#requests-exceed-the-client-timeout |
| retries | only these, once each: no reasoning (`reasoning_on`), `finish_reason` `length` (`reasoning_on`, twice the budget, at most 16384, never less than the first), plain text instead of a call (`tools`, then one named diagnostic), and `tools_stream` where the plan says (gpt-oss). Never after HTTP 429 | the sources of each check below |
| answers | extra fields are accepted: "Clients should accept additional fields". A `tool_calls` of `[]`, `null` or missing means no call | https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html#send-a-chat-completion-request |
| errors | the status, content type and body are kept; vLLM sends `{"error": {"message", "type", "param", "code"}}`, SGLang `{"object": "error", ...}`, the hosted API text or problem+json | https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/exception_handling/error_response.py |

### Never sent

| field | why | source |
|---|---|---|
| `nvext` | removed in NIM 2.0 | https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/1.x-migration-guide.html#api-and-inference-requests |
| `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` | removed in vLLM 0.12; vLLM 0.31 answers 200 with unconstrained output | https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/engine/protocol.py |
| `structured_outputs` | inkling ignores it | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#known-issues |
| `chat_template`, also inside `chat_template_kwargs` | 400 unless the server trusts request templates | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#sglang-0-5-21 |
| `stop` | gpt-oss can lose the answer or keep the stop string; GLM-5.3-Flash matches it in the reasoning | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-gpt-oss-20b-chat-stop-in-reasoning |
| `min_p`, `logit_bias` | 400 on nemotron-3-super profiles with speculative decoding | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#known-issues |
| `mm_processor_kwargs`, `media_io_kwargs` | 400 unless the server trusts them | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#vllm-0-31-0 |
| `tool_choice: "required"` | NIM 1.10 and later dropped it; it uses guided decoding, which proves a parsable call, not a working parser | https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html#model-support-and-functionality |

### Reasons any request can give

| reason | when |
|---|---|
| `endpoint_unreachable` | no answer: connection refused, unknown host, reset |
| `timeout` | no answer, or no complete answer, within the request timeout |
| `tls_error` | the TLS handshake or the certificate check failed; pass `--ca-file` for a private CA (verification is never turned off) |
| `request_failed` | the request could not be sent (a bad header) |
| `redirect_refused` | a 3xx answer; redirects are never followed, so a key never reaches another host |
| `connection_lost` | the connection broke inside the body, or fewer bytes came than `Content-Length` |
| `response_too_large` | the body passed the read limit (8 MB JSON, 2 MB stream) |
| `unauthorized` | 401 or 403: no key (pass `--api-key-env NAME`), or the key was refused |
| `rate_limited` | 429: the endpoint's limit, not the model's; never retried. Hosted rate limits "vary per model" (https://forums.developer.nvidia.com/raw/300317) |
| `model_not_served` | 404 ``The model `X` does not exist.`` (https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/engine/serving.py) |
| `model_end_of_life` | hosted 410: the model "has reached its end of life" |
| `http_error` | any other status than 200 |
| `not_json` | a 200 that is not JSON, such as a proxy page, JSON nested deeper than 64 levels, or JSON with `NaN`, `Infinity` or a number too large for a float, which JSON does not have (such an answer is kept as text, and never ends the run) |
| `bad_response_shape` | JSON that is not a chat completion as documented (no `choices[0].message`, no `usage`, a malformed tool call) |
| `unexpected_finish_reason` | `finish_reason` is not `stop` where the check needs `stop` |

## health

| | |
|---|---|
| sends | `GET {health}/v1/health/live`, then `GET {health}/v1/health/ready`, with no body; `{health}` is `--health-url` when `NIM_HEALTH_PORT` moves the health routes, else the URL; the key goes with them only when `{health}` has the URL's scheme, host and port. `GET /health` when ready answers 404. `--wait-ready S` polls every 5 s; by default it probes once. When the wait ends and health still fails with `not_ready` or `not_live`, no other check runs: the run ends as `checks_failed` (exit 4), health keeps its evidence, and the other checks are in `not_run` and a warning. `--wait-ready` with health left out of `--checks` is `bad_input`; on the hosted API it is not applied (a warning) |
| pass | both answer 200, judged by status only (2.x and VLM 1.x bodies differ); or `/health` answers 200 when the NIM routes answer 404 (a warning: vLLM's own route says the engine is alive, not that the model is ready) |
| not applicable | `hosted_no_health_route`: the hosted API answers 404 on the health routes (seen 2026-11-11). `health_route_not_served`: ready and `/health` both 404 on a user URL; a gateway may expose only the inference routes (a warning). `health_port_not_reached`: `NIM_HEALTH_PORT` moved the routes to a port the container does not publish or the port-forward does not reach; nothing is sent (a warning: pass `--health-url`). With `--wait-ready` that case stops the run instead (`health_port_not_reached`, exit 3), as the readiness it asks for cannot be read |

| fail reason | when |
|---|---|
| `not_live` | live answers anything but 200, 404, 401, 403 or 429 |
| `unauthorized` | live or ready answers 401 or 403: a gateway in front of the NIM wants a key (pass `--api-key-env NAME`), or refused the one sent; a health base on another host or port gets no key, and the message says so (pass a `--health-url` on the URL's host and port, or leave health out) |
| `rate_limited` | live or ready answers 429 |
| `not_ready` | ready (or `/health`) answers 503 or another 5xx: the model is still downloading or loading |
| `health_timeout` | a health route did not answer in 30 s |

Traps:
- A ready GET with a JSON body can break the next `/v1/metrics` call on Kimi K3.1 Pro; the check sends no body
  (https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-kimi-k3-1-pro-readiness-metrics).
- Under load, ready can take longer than 5 s (mistral-small-4); no readiness probe runs during `concurrency`
  (https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-mistral-small-4-119b-health-ready-long-generation-load).
- Readiness "alone does not prove that a particular request is making progress"; the chat checks do that.

Sources: https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html#ready-endpoint,
https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html#verify-health-endpoints,
https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/api-reference.html#management-endpoints,
https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/environment-variables.html,
https://docs.nvidia.com/nim/large-language-models/2.1.5/resources/support-and-faq.html.

## models

| | |
|---|---|
| sends | `GET /v1/models`; hosted with a model id: `GET /v1/models/ID`, which needs no key (the hosted list is a catalog subset) |
| chooses | the requested id (`--model`) when it is served; with no `--model`, the one base model (a card whose `parent` is null: LoRA adapters name their base as `parent`); else it stops with `model_ambiguous`. A `--model` that is not served is never swapped for another model: however many base models are served, `models` fails with `model_not_served` (the served ids in the message) and no POST check runs |
| pass | 200, a JSON list with the chosen id; the card's `id`, `root`, `parent`, `owned_by` (`vllm`, `sglang`: a hint to the backend) and `max_model_len` are recorded |
| not applicable | never |

| fail reason | when |
|---|---|
| `model_not_served` | `--model` is not in `/v1/models` (one base model served), or the hosted API answers 404 for it; no POST check runs |
| `model_end_of_life` | hosted 410 |
| `http_error`, `not_json`, `bad_response_shape` | the list cannot be read (also: no card has a text `id`; a card whose `id` is not text is left out with a warning); the other checks still run with the requested id |

Errors that stop the run: `model_ambiguous` and `no_models` (exit 3, ask the user), `endpoint_unreachable` (exit 4). These write no
results and no report, so the rerun can use the same `--out`. The served id is
`NIM_SERVED_MODEL_NAME` when set; the docs say to read it, not assume it.

Sources: https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html#find-the-model-name,
https://docs.nvidia.com/nim/large-language-models/2.1.5/advanced-use-cases/finetune-lora.html,
https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/models/serving.py.

## chat

| | |
|---|---|
| sends | one user message "Reply with the single word OK.", `temperature` 0, `max_tokens` 64 (1024 for a model that reasons by default or has no record); no reasoning switch: it sees the model as a plain client does |
| pass | 200; `object` `chat.completion`; `choices[0].message.role` `assistant`; text in `content` or in a reasoning field; `usage.completion_tokens` > 0; no control tokens in `content`. The message says what came back: an answer, or reasoning only (only with `finish_reason` `length`) |
| warnings | `content` empty with `finish_reason` `length`: reasoning used the budget; the answer is not "OK" |
| not applicable | only an out-of-scope family |

| fail reason | when |
|---|---|
| `empty_completion` | no text anywhere, or `usage.completion_tokens` is 0 |
| `control_tokens_in_content` | `<\|channel\|>`, `<\|message\|>`, `<\|start\|>`, `<\|end\|>`, `<\|constrain\|>`, `<\|close\|>`, `<\|open\|>` or `<\|sep\|>` in `content` |
| `bad_response_shape` | not a chat completion, or no `usage.completion_tokens` |
| `answer_missing` | reasoning but no answer, with `finish_reason` `stop`: a client that reads `content` gets nothing. All the text in `reasoning` or `reasoning_content` means the reasoning parser put the answer in the reasoning field; the hint names the server fix (`--reasoning-parser`, `nim-deploy`'s job; no hint on the hosted API) |

Traps:
- `max_tokens` 1 (an older smoke) proves only that the server answers; on a reasoning model the budget goes to
  reasoning and `content` stays empty
  (https://docs.nvidia.com/nim/large-language-models/2.0.13/get-started/advanced/get-started-nemotron-3.5-super-vl.html#known-issues).
- Control tokens leak on gpt-oss LoRA profiles and on Kimi K3.1 Pro without its parsers
  (https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-gpt-oss-20b-lora-reasoning-format,
  https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/advanced/model-specific-deployment-considerations.html#kimi-k3-1-pro).

Sources: https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/api-reference.html#chat-completions,
https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html#send-a-chat-completion-request.

## stream

| | |
|---|---|
| sends | the chat request with `stream: true` and `stream_options: {"include_usage": true}` (vLLM defaults it to false), `Accept: text/event-stream` |
| reads | `data: <json>` events; lines that start with `:` are keep-alive comments and are skipped; stops at `data: [DONE]`; joins `delta.content`, `delta.reasoning` and `delta.reasoning_content` |
| pass | 2 or more `chat.completion.chunk` objects, text, a chunk with `finish_reason`, and `data: [DONE]`; the time to the first byte and the total time are recorded |
| warnings | no usage chunk although `include_usage` was sent; `content` empty with `finish_reason` `length`: reasoning used the budget |
| not applicable | only where a NIM source says the model cannot stream (Nemotron Parse, out of scope) |

| fail reason | when |
|---|---|
| `stream_ignored` | 200 with `application/json`, a single body, a stream that ended with no chunk objects, or the whole answer in one chunk |
| `timeout`, `connection_lost` | the stream started (200, `text/event-stream`, maybe keep-alive comments) but no chunk came within the request timeout, or the connection broke before the first chunk |
| `stream_error_event` | a `data: {"error": ...}` event; the status stays 200 because it comes after the headers |
| `stream_incomplete` | the stream ended or timed out before `[DONE]` |
| `empty_stream` | chunks and `[DONE]`, but no text |
| `bad_response_shape` | no chunk carries a `finish_reason` |
| `answer_missing` | as in chat: reasoning but no answer, with `finish_reason` `stop` |

Sources: https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html#streaming,
https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/serving.py,
https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/utils/sse_keep_alive.py,
https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/generate/base/protocol.py.

## tools (T-AUTO)

| | |
|---|---|
| sends | one `get_weather(city: string, unit: celsius\|fahrenheit)` tool, required `city` and `unit`; "What is the weather in Paris right now? Use celsius. Call the get_weather tool."; `tool_choice: "auto"` (always sent: NIM 1.x needs it); the family's mode switches; `temperature` 0 unless the family says otherwise; `max_tokens` 1024, or 4096 when the family keeps reasoning on during tool calls |
| pass | a call whose name is `get_weather`, whose `arguments` is a JSON string of an object that matches the schema, with `city` matching /paris/i and `unit` `celsius`. `finish_reason` is recorded, not required (`tool_calls` for auto, `stop` for a named call) |
| warnings | a call without `id`; argument keys the tool does not define |
| not applicable | not documented for the model and NIM generation (`not_documented`, `not_documented_for_model`); NIM LLM 1.x (not a VLM NIM's own 1.x releases) with pipeline parallelism above 1 (`pipeline_parallel_unsupported`; when the profile's parallelism cannot be read, as a 1.x `/v1/metadata` gives only a profile id, the check runs and the plan warns); no record (`no_capability_record`); not enabled here (`not_enabled_in_deployment`, below) |

Validation, in this order (each step stops at the first failure):

| step | fail reason |
|---|---|
| 1. HTTP 200; vLLM's 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` or `tool_choice=... requires --tool-call-parser to be set` means no tool parser here | `parser_not_enabled`, `parser_dropped_by_override` or `not_enabled_in_deployment` (below); `http_error` |
| 2. a call was made | `tool_call_in_content`, `tool_call_in_reasoning`, `no_tool_call`, `model_did_not_call_tool_with_auto` |
| 3. the right function (`type` `function` when given) | `wrong_function`, `bad_response_shape` |
| 4. `arguments` is a string | `arguments_not_json_string` |
| 5. it parses to an object | `invalid_json_arguments`, `double_encoded_arguments` (a JSON string inside a JSON string), `arguments_not_object` |
| 6. it matches the schema (`type`, `required`, `enum`) | `schema_mismatch` |
| 7. the values match the prompt | `argument_values_wrong` |
| 8. `finish_reason` | recorded only |

No parser in the deployment (D5), from the plan entry:

| plan says | result | example |
|---|---|---|
| the image turns the parser on by default (the record's `default_on`, or the image's own `NIM_PASSTHROUGH_ARGS`); a NIM 1.x `default_on` counts only for the model's own image, as 1.x enables tool calling by itself only in LLM-specific containers | `fail` `parser_not_enabled` | nemotron-3.5-super-vl on 2.x; Llama 3.1 in `nvcr.io/nim/meta/llama-3.1-8b-instruct:1.15.0` |
| the deployment's `NIM_PASSTHROUGH_ARGS` replaced an image default that set it (on vLLM both `--enable-auto-tool-choice` and `--tool-call-parser`; an override that keeps only the parser dropped it too) | `fail` `parser_dropped_by_override` | a tuned `--max-num-seqs N` that dropped the parser flags |
| opt-in for this model | `not_applicable` `not_enabled_in_deployment` | Kimi K3.1 Pro; 1.x variant images whose container-variants section says `TOOL_CALL_PARSER: Set to 1 to turn tool calling on` (Nemotron 3 Nano, Qwen3-32B, Qwen3 Next 80B A3B Thinking, the Nemotron Nano 9B v2 DGX Spark image) |
| unknown default | `not_applicable` `not_enabled_in_deployment` and a warning; the message never says opt-in. When the deployment sets `NIM_PASSTHROUGH_ARGS` without both tool flags and the image default cannot be read (the image is not on this machine), the warning names that override as the likely cause. On NIM 1.x with the image unknown (a `--url` run without `--image`) or another image (the multi-LLM NIM `nvcr.io/nim/nvidia/llm-nim`, which serves the same ids), the warning says so; pass `--image` | Llama 3.1 on 2.x; Llama 3.1 on 1.x behind a URL; Nemotron 3 Nano Omni on VLM NIM 1.7 (no VLM page names a switch) |

Each gets the documented switch as `hint`: 2.x vLLM `--enable-auto-tool-choice --tool-call-parser <parser>`, 2.x SGLang
`--tool-call-parser <parser>`, 1.x `NIM_ENABLE_AUTO_TOOL_CHOICE=1` and `NIM_TOOL_CALL_PARSER`, 1.x variants `TOOL_CALL_PARSER=1`.
A user `NIM_PASSTHROUGH_ARGS` replaces the whole image default. Changing it is `nim-deploy`'s job.

No call:

| answer | result |
|---|---|
| call markup in `content`: `<TOOLCALL>`, `[TOOL_CALLS]`, `<tool_call>`, `<\|python_tag\|>` or `"name": "get_weather"` | `tool_call_in_content`: the model called, the parser did not take it out (missing or wrong for this model); hint attached |
| call markup in the reasoning | `tool_call_in_reasoning`: calls are parsed from `content` only |
| plain text | the same request once more; still text: one named call (`tool_choice: {"type": "function", ...}`) as a diagnostic only where the record's `named` is `yes` or absent; `no` (gpt-oss, Qwen3-Next-Thinking) and `unknown` (Mistral Small 4, whose forced tool calls can answer in text) send none. A valid named call: `model_did_not_call_tool_with_auto`; else `no_tool_call`. On SGLang a missing parser gives 200 with text, so the message says the parser may be missing |

A named or `required` call never makes this check pass: guided decoding gives "a validly-parsable function call - not a
high-quality one".

Traps:

| trap | what the check does | source |
|---|---|---|
| HTTP 200 does not mean a call was made | checks the list itself | https://docs.nvidia.com/nim/large-language-models/2.1.5/troubleshooting/requests.html#a-forced-tool-call-is-missing |
| `arguments` must be a JSON string (OpenAI shape) | `arguments_not_json_string` otherwise | https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/generate/base/protocol.py |
| double-serialized arguments (fixed in Nemotron 3 Super 1.8.2-variant); `llama3_json` arrays as strings | parse, then validate the schema | https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html#all-current-known-issues-for-specific-models, https://docs.vllm.ai/en/v0.31.0/features/tool_calling/ |
| Llama-Nemotron calls tools with thinking off | the record's mode: system `detailed thinking off` | https://docs.nvidia.com/nim/large-language-models/1.15.0/function-calling.html |
| Nemotron 3 needs `enable_thinking: true` and `force_nonempty_content: true` with tools | the record's mode | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/main/README.md |
| a budget too small for reasoning plus the call gives 400 or a cut call | 4096 for families that reason during tool calls | https://docs.nvidia.com/nim/vision-language-models/2.1.2-variant/function-calling.html |
| sampling: Nemotron 3 Nano 0.6 / 0.95; Mistral 0.15; Qwen3 with thinking never greedy | the record's `sampling` | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/README.md, https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/README.md |
| gpt-oss supports `auto` only | no named diagnostic | https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html |
| Mistral tool-call ids in history must be 9 alphanumerics | single-turn requests only | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-mistral-small-4-119b-mistral-tokenizer-requests |
| forced calls can return empty `tool_calls` (qwen3.5) or text (mistral-small-4) | the main check uses `auto`; attached as known issues | https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-qwen3-5-122b-a10b-forced-tool-choice-empty-tool-calls, https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-mistral-small-4-119b-forced-tool-choice |
| models of 8B or fewer call tools "highly inaccurately" on 1.x | still a fail; the known issue is attached | https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html#model-support-and-functionality |

Sources: https://docs.nvidia.com/nim/large-language-models/2.1.5/advanced-use-cases/tool-calling-and-mcp.html#enable-tool-calling,
https://docs.nvidia.com/nim/large-language-models/2.1.5/advanced-use-cases/tool-calling-and-mcp.html#error-auto-tool-choice-requires-configuration-vllm-backend,
https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/advanced-configuration.html#passthrough-arguments-nim-passthrough-args,
https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/renderers/online_renderer.py#L203-L231,
https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html,
https://docs.nvidia.com/nim/large-language-models/1.15.0/nim-container-variants.html.

## tools_stream, tools_parallel, tools_none (`--extended`)

| check | sends | pass | fail reasons | not applicable |
|---|---|---|---|---|
| `tools_stream` | the T-AUTO request with `stream: true` | `delta.tool_calls` joined by `index` (id, type and name from the first chunk that has them; `arguments` pieces joined) give a valid call | the stream reasons, then the T-AUTO validation reasons; no named diagnostic | streaming calls documented as unsupported (`not_documented_for_model`): Super v1.5, Nano 4B and Nano 9B v2 parsers have no streaming |
| `tools_parallel` | "Get the current weather for Paris and for Tokyo, both in celsius. Call get_weather once for each city." with `parallel_tool_calls: true` (NIM 1.x defaulted it to false) | 2 or more valid calls, whose cities match /paris/i and /tokyo/i as in `tools` ("Paris, France" counts) | `too_few_calls`, the validation reasons, `argument_values_wrong` for other cities | not documented (`not_documented`, `not_documented_for_model`); `information_only` where the sources disagree (Llama 3.x) |
| `tools_none` | the tool with `tool_choice: "none"` and "What is 2 + 2? Answer with the number only." | no `tool_calls` and an answer; call markup in `content` is a warning: "the API only guarantees that tool_calls in the response is empty" | `tool_called_with_none`, `empty_answer` (qwen3.8-27b without `--exclude-tools-when-tool-choice-none`) | as `tools` |

`tools_stream` retries once only where the plan says (gpt-oss: an intermittent Harmony parser error with streaming
on 1.x). The parser rules of `tools` apply to all three; `tool_choice: "none"` needs no parser on vLLM.

Sources: https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html,
https://docs.nvidia.com/nim/vision-language-models/2.1.2-variant/function-calling.html,
https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/420ba7d28211abf116b8b103ab700d92619daf98/llama_nemotron_toolcall_parser_no_streaming.py,
https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/6533e8de2c68e4536bf7c411d7a3ce5734111476/nemotron_toolcall_parser_no_streaming.py,
https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html#all-current-known-issues-for-specific-models,
https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#known-issues.

## Reading reasoning (all reasoning checks)

| where | shape | how it is read |
|---|---|---|
| `reasoning` | `message.reasoning` (vLLM 0.18 and later; NIM 2.1.5 ships vLLM 0.31) | the field |
| `reasoning_content` | `message.reasoning_content` (SGLang, older vLLM) | the field; never counted twice when a delta has both |
| `inline` | `<think>X</think>Y` in `content` | X is reasoning, Y the answer |
| `inline_close_only` | `X</think>Y`: the template wrote `<think>` into the prompt (Nano 9B v2, Nemotron 3, DeepSeek-R1) | X is reasoning, Y the answer |
| `unclosed` | `<think>X` with no closing tag | cut off: reasoning with no answer |
| `none` | no field, no tags; `None`, `""`, white space and an empty `<think></think>` all count as no reasoning | the content is the answer |

Never from `usage.reasoning_tokens` (always 0 on SGLang) or from prose such as "Let me think". The place is recorded
even on a pass: inline reasoning is invisible to clients that read `message.reasoning`.

Sources: https://docs.vllm.ai/en/v0.31.0/features/reasoning_outputs/,
https://docs.sglang.io/docs/advanced_features/separate_reasoning,
https://docs.nvidia.com/nim/large-language-models/2.1.5/advanced-use-cases/custom-parsers-and-templates.html,
https://docs.nvidia.com/nim/large-language-models/1.15.0/reasoning-model.html#reasoning-mode.

## reasoning_on (R-ON)

| | |
|---|---|
| sends | "A shop sells pens for 3 dollars each. Ana buys 4 pens and pays with a 20 dollar bill. How much change does she get? Think it through, then give only the number." with the family's on switch (a system prompt, `chat_template_kwargs` or `reasoning_effort`; gpt-oss `"low"`), its on sampling (none: the server's defaults), `max_tokens` 4096 (GLM-5.3-Flash 8192), `stream: false` |
| pass | 8 or more characters of reasoning, `finish_reason` `stop`, and an answer outside the reasoning; the answer 8 is recorded, not required |
| warnings | reasoning inline in `content` (with the reasoning-parser hint); reasoning in another place than the record expects |
| not applicable | `not_a_reasoning_model`, `no_documented_switch`, `reasoning_not_detectable` (Phi-4-mini: no `<think>` markers), `not_enabled_in_deployment` (Kimi K3.1 Pro without `--reasoning-parser`), `no_capability_record` |

| fail reason | when |
|---|---|
| `switch_rejected` | a 4xx on the documented switch (401, 403, 404, 410 and 429 keep their own reasons) |
| `reasoning_missing` | no reasoning, after one retry of the same request (DeepSeek-R1 and Llama-Nemotron sometimes skip thinking) |
| `reasoning_not_finished` | `finish_reason` `length`, after one retry with twice the budget, at most 16384 (never less than the first) |
| `answer_missing` | reasoning, `finish_reason` `stop`, no answer (the gpt-oss stop-string symptom) |
| `control_tokens_in_content` | as `chat` |
| `unexpected_finish_reason` | another `finish_reason` than `stop` or `length` |

Traps:

| trap | what the check does | source |
|---|---|---|
| a small budget cuts the reasoning; with a pre-filled template the cut text has no tag and looks like an answer | 4096, then one retry with twice the budget | https://docs.nvidia.com/nim/large-language-models/2.0.13/get-started/advanced/get-started-nemotron-3.5-super-vl.html#known-issues |
| defaults differ by family and a server can change them (`--default-chat-template-kwargs`) | always sends the explicit switch | https://docs.vllm.ai/en/v0.31.0/features/reasoning_outputs/ |
| a switch the template does not read changes nothing (DeepSeek V3.x reads `thinking`, not `enable_thinking`) | only the record's switch | https://docs.sglang.io/docs/advanced_features/separate_reasoning |
| greedy decoding with thinking loops (Qwen3: "DO NOT use greedy decoding"; DeepSeek-R1: 0.6, no system prompt) | the record's on sampling | https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/README.md, https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/README.md |
| a trivial prompt gives an empty think block | a two-step arithmetic prompt | https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html#all-current-known-issues-for-specific-models |
| Mistral Small 4 answers 400 to any `chat_template_kwargs` | its record uses `reasoning_effort` | https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/mistral-small-4-119b-2603/api.html#reasoning |
| GLM-5.3-Flash needs at least 8192 tokens | the record's `max_tokens` | https://docs.nvidia.com/nim/large-language-models/2.1.4/get-started/advanced/get-started-glm-5.3-flash.html |

## reasoning_off (R-OFF) and reasoning_off_stream

| | |
|---|---|
| sends | "What is the capital of France? Answer with one word." with the family's off switch, `temperature` 0 (the cards recommend greedy for reasoning off) or the record's off sampling, `max_tokens` 256. `reasoning_off_stream` sends the same with `stream: true` |
| pass | `finish_reason` `stop`, no reasoning (absent, null, empty, or an empty think block), an answer in `content`; "Paris" is recorded, not required |
| not applicable | `always_on` (DeepSeek-R1 and distills, Qwen3-Thinking, gpt-oss, GLM-5.3-Flash, Muse Glimmer), `no_documented_switch` (inkling, Kimi K3.1 Pro, DeepSeek-V4-Pro: reasoning is on by default and no source names an off switch; `--reasoning-off` from the docs is used), `not_a_reasoning_model`, `no_prefilled_think_block` (`reasoning_off_stream` only), `no_capability_record` |

| fail reason | when |
|---|---|
| `answer_in_reasoning_field` | `content` empty and the reasoning field holds a short answer-like text (12 words or fewer, no `</think>`, names Paris): the reasoning parser routed the answer into the field |
| `reasoning_present` | any non-empty reasoning; for Llama-Nemotron Nano the message cites the model card's thinking with thinking off |
| `switch_rejected` | a 4xx on the documented switch |
| `empty_answer` | no reasoning and no answer |
| `unexpected_finish_reason` | not `stop`; `length` on a one-word answer suggests thinking in `content` |
| stream reasons | `reasoning_off_stream` only: as `stream` |

Why the streaming twin: for templates that pre-fill `<think></think>` with reasoning off (Super v1.5, Nano 9B v2,
Nano 12B VL, Nemotron 3, Qwen3), the misrouted answer depends on the code path; NVIDIA's public `nano_v3` parser fixes it
only in the non-streaming path.

Sources: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/nano_v3_reasoning_parser.py,
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/chat_template.jinja,
https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/420ba7d28211abf116b8b103ab700d92619daf98/README.md,
https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/6533e8de2c68e4536bf7c411d7a3ce5734111476/README.md,
https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/blob/54641c1611fcff44fa4865626462445e0a153fc7/README.md,
https://docs.nvidia.com/nim/large-language-models/1.15.0/reasoning-model.html#no-think-prompt,
https://docs.nvidia.com/nim/large-language-models/1.15.0/reasoning-model.html#chat-template-keyword-arguments.

## json_schema

| | |
|---|---|
| sends | "Introduce the product NIM, release 2, in one short sentence." (no JSON hint and no schema key, so a server that ignores `response_format` answers in prose and fails) with `response_format: {"type": "json_schema", "json_schema": {"name": "smoke", "schema": ...}}`; the schema has `name` (string) and `version` (integer), both required, `additionalProperties: false`; `temperature` 0; `max_tokens` 256. A family that reasons by default gets its off switch, or its own switch (DeepSeek-V4-Pro `reasoning_effort: "low"`); when it has no off switch, 4096 tokens or the record's reasoning budget when it is larger (GLM-5.3-Flash and Kimi K3.1 Pro 8192), with the record's on switch when it only lowers the effort (gpt-oss and GLM-5.3-Flash `reasoning_effort: "low"`) |
| pass | 200, `finish_reason` `stop`, and the answer outside any reasoning parses to an object that matches the schema; other values than "NIM" and 2 are a warning |
| not applicable | where a NIM source says the model or image does not support it (`not_documented_for_model`): the 1.x DGX Spark images of Llama 3.1 8B, Nemotron Nano 9B v2 and Qwen3-32B, and Qwen3 Next 80B A3B Thinking, whose release notes or container-variant sections list guided decoding as not supported; on NIM 1.x, a profile with pipeline parallelism above 1 (`pipeline_parallel_unsupported`) |

| fail reason | when |
|---|---|
| `truncated_json` | `finish_reason` `length` (glm-5.2 and inkling can emit white space until `max_tokens`) |
| `json_only_in_reasoning` | `content` empty and the JSON is in the reasoning |
| `invalid_json` | the answer is empty or does not parse (a fenced block too: the constraint was not applied) |
| `schema_mismatch` | not an object, a missing key, a wrong type, or a key the schema forbids |
| `http_error` | for example a schema the backend cannot compile, or inkling's 500 for a `response_format` that is not an object |

Never `json_object` (it allows "empty objects", the outlines backend rejects it on vLLM 0.31, and on mistral-small-4 it can
return an array); never `guided_json`. The schema avoids keywords SGLang 0.5.21 rejects (`if`, `then`, `else`, `not`,
`multipleOf`, `uniqueItems`, `contains`).

Sources: https://docs.vllm.ai/en/v0.31.0/features/structured_outputs/,
https://docs.sglang.io/docs/advanced_features/structured_outputs,
https://docs.nvidia.com/nim/large-language-models/1.15.0/structured-generation.html,
https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-glm-5-2-json-schema-whitespace,
https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-inkling-json-schema-whitespace-runaway,
https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-mistral-small-4-119b-json-object-array,
https://docs.nvidia.com/nim/large-language-models/2.0.13/get-started/advanced/get-started-deepseek-v4-pro-0813.html#structured-json-output.

## image

| | |
|---|---|
| sends | one user message with a text part "What is the main color of this image? Answer with one word." and one `image_url` part: an inline 64x64 solid red PNG data URL the script builds (about a hundred bytes; no network access needed from the container). `max_tokens` 32, or the off switch; for a VLM that reasons by default and has no off switch, 1024 tokens or the record's reasoning budget when it is larger (inkling and Muse Glimmer 4096, GLM-5.3-Flash and Kimi K3.1 Pro 8192), with the lighter effort as for `json_schema`. Then the text-only twin: the same text, no image, `max_tokens` 1 |
| pass | 200, an answer, and more `usage.prompt_tokens` with the image than without (one image plus a short question was 579 prompt tokens in a VLM example); an answer that names red is only recorded |
| not applicable | `text_only_model` (never send an image to a text-only model), `text_only_profile` (for example deepseek-v4.1-flash's FP8 profiles on B200, GB200 and H200), `not_documented`, `no_capability_record`. When the active profile's id is unknown (no `/v1/metadata`, the hosted API, a profile name only), the image is sent, and the 400 the docs give for the text-only profiles, "At most 0 image(s) may be provided in one prompt.", is `text_only_profile` too; any other 400 is `http_error` |

| fail reason | when |
|---|---|
| `image_ignored` | equal (or fewer) prompt tokens with the image: it was dropped |
| `empty_answer` | no answer to the image question |
| `http_error` | the image request or its twin failed (inkling answers 500, not 400, for bad image URLs) |
| `bad_response_shape` | no `usage.prompt_tokens` to compare |

One image only: Llama 3.2 Vision takes "only one image per request", and nemotron-3.5-content-safety answers 500 for two
or more. The evidence keeps the image's SHA-256 and size, never its data.

Sources: https://docs.nvidia.com/nim/large-language-models/2.1.5/advanced-use-cases/multimodal-input.html,
https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/support-matrix.html#vlms,
https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/support-matrix.html#deepseek-v4-1-flash,
https://docs.nvidia.com/nim/vision-language-models/2.1.2-variant/introduction.html#model-support,
https://docs.nvidia.com/nim/vision-language-models/1.7.0/api-reference.html,
https://docs.nvidia.com/nim/vision-language-models/1.1.1/release-notes.html#limitations,
https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html#issue-2-1-5-inkling-invalid-image-url-500.

## concurrency

| | |
|---|---|
| sends | N copies of the chat request at the same moment from N threads (`--concurrency`, default 4, at most 8; at most 4 on the hosted API, whose rate limits apply), `max_tokens` 64, with the family's off switch when there is one; never a reasoning-on prompt; no readiness probe while they run |
| pass | all N answer 200 with text; each latency and the wall time are recorded |
| warnings | through a kubectl port-forward every request reaches one pod, so the Service's load balancing is not tested (https://kubernetes.io/docs/reference/kubectl/generated/kubectl_port-forward/); answers with reasoning but no `content` and `finish_reason` `length`: reasoning used the budget |
| not applicable | only an out-of-scope family |

| fail reason | when |
|---|---|
| `rejected_503` | 503: a queue limit is set (vLLM `--max-num-queued-reqs`, `--max-num-queued-tokens`; SGLang `--max-queued-requests`) |
| `gateway_error` | 502 or 504 from a proxy or gateway |
| `rate_limited` | 429; never retried |
| `empty_completion` | 200 with no text |
| `answer_missing` | as in chat: reasoning but no answer, with `finish_reason` `stop` |
| `http_error` | any other status |

More than 8, or more than 4 on the hosted API, is `concurrency_too_high` (exit 2, before any request): a load or latency
test is `nim-benchmark`'s job. Llama-3.3-Nemotron-Super-49B
v1 "may crash" above 15 concurrent requests with thinking on; the check never turns thinking on.

Sources: https://docs.vllm.ai/en/v0.31.0/cli/serve/, https://docs.vllm.ai/en/v0.31.0/usage/metrics/,
https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/logging-and-observability.html#design-principle,
https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html#all-current-known-issues-for-specific-models.

## Evidence

One private file per request, `evidence/<check>-<n>.json` under the run folder (0700 folder, 0600 file, written
atomically, every string masked for keys, before any text is cut):

| key | holds |
|---|---|
| `request` | method, URL, headers (`Authorization` as `Bearer ***`, other credential headers `***`), body; inline image data replaced by `{"omitted", "media_type", "sha256", "bytes"}` |
| `status`, `content_type`, `latency_ms`, `ttfb_ms` (streams), `error`, `request_id`, `response_bytes` | the answer |
| `response` / `response_text` / `response_raw` | the JSON body, a text body, or the raw event stream cut to 20,000 bytes |
| `finish_reason`, `usage` | read from the body or the joined stream |
| `parsed` | what the check read: the role of the request (`attempt`, `retry`, `named_diagnostic`, `text_only_twin`, `request 2 of 4`), the reasoning place, reasoning, answer, tool calls |
| `cut` | the path of each text cut to 2,000 characters, with its full length |

A list or object nested deeper than 128 levels is replaced by `"[nested deeper than 128 levels: not kept]"`,
in evidence and in every printed or saved object, so no answer can exhaust Python's recursion limit.

## Reason codes

Check results (`status` is the one a check gives with the reason):

| code | status | checks | meaning |
|---|---|---|---|
| `ok` | pass | all | the check passed |
| `endpoint_unreachable` | fail | all | no answer: the connection failed |
| `timeout` | fail | all | no complete answer within the request timeout |
| `tls_error` | fail | all | the TLS handshake or the certificate check failed |
| `request_failed` | fail | all | the request could not be sent |
| `redirect_refused` | fail | all | a redirect, never followed |
| `connection_lost` | fail | all | the connection broke in the middle of the answer |
| `response_too_large` | fail | all | the answer passed the read limit |
| `unauthorized` | fail | all | 401 or 403: no key, or the key was refused |
| `rate_limited` | fail | all | 429: the endpoint's limit; never retried |
| `http_error` | fail | all | another status than 200 |
| `not_json` | fail | all but health | a 2xx that is not JSON |
| `bad_response_shape` | fail | all but health | the JSON is not a chat completion as documented |
| `model_not_served` | fail | models, POST checks | the endpoint does not serve the model (`--model` not served: no POST check runs) |
| `model_end_of_life` | fail | models, POST checks | hosted 410 |
| `unexpected_finish_reason` | fail | reasoning_on, reasoning_off, json_schema | `finish_reason` is not `stop` |
| `not_live` | fail | health | live did not answer 200 |
| `not_ready` | fail | health | ready (or `/health`) did not answer 200 |
| `health_timeout` | fail | health | a health route did not answer in time |
| `health_route_not_served` | not_applicable | health | neither ready nor `/health` is served |
| `health_port_not_reached` | not_applicable | health | `NIM_HEALTH_PORT` moved the health routes to a port this run cannot reach |
| `hosted_no_health_route` | not_applicable | health | the hosted API has no health route |
| `no_models` | error, exit 3 | models | `/v1/models` lists no model |
| `empty_completion` | fail | chat, concurrency | no generated text, or zero completion tokens |
| `control_tokens_in_content` | fail | chat, reasoning_on | raw control tokens in `content` |
| `rejected_503` | fail | concurrency | a queue limit rejected a request |
| `gateway_error` | fail | concurrency | 502 or 504 from a proxy or gateway |
| `stream_ignored` | fail | stream checks | the answer to `stream: true` is not a stream of chunks |
| `stream_error_event` | fail | stream checks | an error event after HTTP 200 |
| `stream_incomplete` | fail | stream checks | the stream ended before `[DONE]` |
| `empty_stream` | fail | stream | the stream carried no text |
| `parser_not_enabled` | fail | tools checks | no tool parser here, but the image or the docs turn it on by default |
| `parser_dropped_by_override` | fail | tools checks | `NIM_PASSTHROUGH_ARGS` replaced the image default that set it |
| `not_enabled_in_deployment` | not_applicable | tools checks, reasoning_on | documented, but not enabled here: opt-in, or the image default is unknown (a warning) |
| `no_tool_call` | fail | tools checks | no call (`tools`: after one retry and the named diagnostic) |
| `model_did_not_call_tool_with_auto` | fail | tools | no call with `auto`, but a named call works |
| `tool_call_in_content` | fail | tools checks | call markup in `content` |
| `tool_call_in_reasoning` | fail | tools checks | call markup in the reasoning |
| `wrong_function` | fail | tools checks | the call names another function |
| `arguments_not_json_string` | fail | tools checks | `arguments` is not a string |
| `invalid_json_arguments` | fail | tools checks | `arguments` does not parse |
| `double_encoded_arguments` | fail | tools checks | `arguments` is JSON inside a JSON string |
| `arguments_not_object` | fail | tools checks | `arguments` is JSON but not an object |
| `schema_mismatch` | fail | tools checks, json_schema | the value does not match the schema |
| `argument_values_wrong` | fail | tools checks | the arguments do not match the prompt |
| `too_few_calls` | fail | tools_parallel | fewer than 2 calls |
| `tool_called_with_none` | fail | tools_none | a call with `tool_choice: "none"` |
| `empty_answer` | fail | tools_none, reasoning_off, image | no answer text |
| `switch_rejected` | fail | reasoning_on, reasoning_off | a 4xx on the documented switch |
| `reasoning_missing` | fail | reasoning_on | no reasoning, after one retry |
| `reasoning_not_finished` | fail | reasoning_on | `finish_reason` `length`, after one retry with twice the budget |
| `answer_missing` | fail | chat, stream, concurrency, reasoning_on | reasoning but no answer, with `finish_reason` `stop` |
| `reasoning_present` | fail | reasoning_off | reasoning with reasoning off |
| `answer_in_reasoning_field` | fail | reasoning_off | the answer landed in the reasoning field |
| `invalid_json` | fail | json_schema | the answer does not parse as JSON |
| `truncated_json` | fail | json_schema | `finish_reason` `length` |
| `json_only_in_reasoning` | fail | json_schema | the JSON is in the reasoning |
| `image_ignored` | fail | image | no more prompt tokens with the image |

Plan reasons (`scripts/nsc_features.py`), given before any request:

| code | decision | meaning |
|---|---|---|
| `core_check` | test | health, models, chat and concurrency run for every chat NIM |
| `engine_feature` | test | streaming and JSON schema are engine features, tested for every chat NIM |
| `documented` | test | a source documents the feature for this model |
| `documented_conflict` | test | the sources disagree; both are listed |
| `documented_by_agent` | test | `--documented FEATURE=URL`, for a model with no record or a silent one |
| `hosted_no_health_route` | not_applicable | the hosted API has no health route |
| `out_of_scope` | not_applicable | embedding, reranking, speech, document parsers, safety classifiers |
| `no_capability_record` | not_applicable | no record matches the model and no `--documented` URL was given for the check's feature |
| `not_documented` | not_applicable | no source documents it |
| `not_documented_for_model` | not_applicable | a source says the model does not support it |
| `not_a_reasoning_model` | not_applicable | the model does not reason |
| `always_on` | not_applicable | the model always reasons; there is no off switch |
| `no_documented_switch` | not_applicable | no documented switch for this direction |
| `no_prefilled_think_block` | not_applicable | the streaming twin of reasoning off is not needed |
| `reasoning_not_detectable` | not_applicable | reasoning cannot be told apart from the answer |
| `text_only_model` | not_applicable | the model takes text only |
| `text_only_profile` | not_applicable | the active profile takes text only |
| `information_only` | not_applicable | parallel tool calls: the sources disagree |
| `pipeline_parallel_unsupported` | not_applicable | NIM LLM 1.x with pipeline parallelism above 1 (never a vlm record) |
| `not_enabled_in_deployment` | not_applicable | not enabled in this deployment: opt-in, or the image default is unknown |

Errors that stop a run (`errors[0].code`):

| code | exit | when |
|---|---|---|
| `model_ambiguous` | 3 | no `--model` and several base models are served; ask which one, and test none when the user does not know |
| `no_models` | 3 | `/v1/models` lists no model |
| `health_port_not_reached` | 3 | `--wait-ready` with `NIM_HEALTH_PORT` on a port this run cannot reach; ask for `--health-url` |
| `endpoint_unreachable` | 4 | `/v1/models` gets no answer, or the first chat request gets none (refused, reset): the run stops there (state `unreachable`); check the URL and the port. Not when the script's kubectl port-forward exited during that request: the next check restarts the forward, and its request counts as the first |
| `first_request_timeout` | 4 | the NIM answered `/v1/models`, but the first chat request got no complete answer within its budget: the run stops there (state `timeout`), unless the port-forward exited during it, as for `endpoint_unreachable`. The NIM is reachable but slow: the fix suggests a larger `--timeout`, the budget of every chat request, up to 3600 s |
| `port_forward_lost`, `pod_not_ready`, `kubectl_failed` | 4, 3, 4 | the port-forward stopped during the checks and cannot be restarted (`references/access.md`). The checks that ran keep `results.json`, `report.md` and their evidence; the others are listed in `not_run` and in a warning, and the message names the folder |
| `concurrency_too_high` | 2 | `--concurrency` above 8, or above 4 on the hosted API |
| `key_over_plain_http` | 2 | a key would go over plain `http://` to a host that is not this machine (not loopback), without `--allow-plain-http-key`: the run stops before any request and sends nothing (`references/access.md`). With the flag, only once the user agrees, the key is sent and the run keeps a warning in `facts.warnings`, also when it stops early |
| `bad_input` | 2 | a bad flag or value, such as `--concurrency 0`, `--wait-ready` with health left out, `http://` to the hosted API, a `--api-key-env` variable that is not set, a key shorter than 6 characters (too short to be masked), or a hosted run of chat checks with no key |
| `checks_failed` | 4 | the run finished and at least one check failed (results in `facts.results`); also when `--wait-ready` ends with health still `not_ready` or `not_live`, and then no other check ran (`not_run`) |
| `interrupted` | 130 | Ctrl-C (SIGINT): one JSON object (state `interrupted`, next `ask_user`); the checks that ran keep `results.json`, `report.md` and their evidence, the rest are in `not_run` |
| `internal_error` | 4 | an error no check expected (a bug, or an answer no parser handles): one JSON object, never a bare traceback (state `internal_error`); the checks that ran keep their files as above; the Python traceback is on stderr, masked |
