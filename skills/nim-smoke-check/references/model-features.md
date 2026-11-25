# Model features: what the table holds and how it decides

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

No route of a running NIM reports which features its model supports: `/v1/models`, `/v1/metadata`
and `/v1/version` carry no tool, reasoning or image fields. The NIM Metadata API documents a
`features` object, but it was empty in every document fetched on 2026-11-11. So the skill ships a
curated table, `scripts/model_features.json`, with one record per model family. Every value that
decides whether a check runs cites a public source. `scripts/nsc_features.py` loads the table,
finds the record of the target and turns it into a plan: `test` or `not_applicable` per check,
with a reason code and the sources. A feature that no source documents is `not_applicable`,
never `fail`.

## The table

| key | holds |
|---|---|
| `schema_version` | `1` |
| `reviewed` | the date of the last review, `YYYY-MM-DD` |
| `docs` | the doc versions the table was checked against: NIM LLM 2.1.5, VLM 2.1.2-variant, LLM 1.x 1.15.0, vLLM 0.31.0, SGLang 0.5.21 |
| `engine` | sources for the checks every chat NIM gets: `health`, `models`, `chat`, `stream`, `json_schema`, `concurrency`, plus `image_text_only`, `tool_parser_flags` and `pipeline_parallel_1x` |
| `families` | the records |

A family record:

| field | values | meaning |
|---|---|---|
| `id`, `name` | text | `id` is what `--family` takes |
| `match` | `repos`, `served_ids`, `patterns` | how a target finds the record (Matching, below) |
| `status` | `supported`, `eos`, `out_of_scope` | `eos`: End of Support on NGC; `out_of_scope`: only health and models run |
| `kind` | `llm`, `vlm` | an `llm` never gets the image check |
| `backend` | `vllm`, `sglang`, `either`, `null` | picks the server switch in hints |
| `reasoning` | object | below |
| `tools` | object | below |
| `image` | `documented`, `max_images`, `text_only_profiles`, `sources` | `text_only_profiles` lists profile ids that take text only; their documented HTTP 400 ("At most 0 image(s) may be provided in one prompt.") makes the image check `text_only_profile` when the active profile's id is unknown |
| `json_schema`, `stream` | `documented`: `engine` or `no`; optional `switch` | `engine` means the check runs like for every chat NIM |
| `server_switches` | `tools`, `reasoning` → `2.x-vllm`, `2.x-sglang`, `1.x` | the documented flag a hint names |
| `known_issues` | `url`, `affects` (check ids), `text` | release-note anchors attached to results |
| `overrides` | `ids` or `patterns`, then any field above | per-model differences inside a family |
| `out_of_scope_reason`, `out_of_scope_sources` | text, sources | why a family is out of scope |
| `notes` | text | context; never decides anything |

`reasoning`:

| field | values | meaning |
|---|---|---|
| `default` | `none`, `on`, `off`, `always`, `prompt`, `unknown` | `none`: not a reasoning model; `always`: cannot be turned off; `prompt`: set by the prompt |
| `on`, `off` | a switch | what the reasoning checks send |
| `sampling_on`, `sampling_off` | `temperature`, `top_p`, `top_k` | from the model card; off defaults to temperature 0 |
| `max_tokens` | number | budget of reasoning on (default 4096); for a model that reasons by default (`on`, `always`) and has no off switch, also the least `json_schema` and `image` get |
| `where` | `reasoning`, `reasoning_content`, `inline`, `inline_close_only` | where reasoning is expected |
| `prefills_off` | true, false | the template pre-fills the think block when off; adds the streaming twin of reasoning off |
| `off_reason` | `always_on`, `no_documented_switch` | why reasoning off is not applicable |
| `detectable` | false | reasoning cannot be told apart from the answer |
| `parser_default_on` | false | reasoning on runs only when the deployment shows `--reasoning-parser` |

`tools`:

| field | values | meaning |
|---|---|---|
| `documented` | `yes`, `no`, `unknown`, `conflict` | see the conflict rules |
| `default_on` | true, false, null, or `{"1.x": ..., "2.x": ...}` | whether the model's own NIM image turns the tool parser on by itself. A 1.x true counts only when the image is known and is one of the record's (`match.repos` or `patterns`): 1.x enables tool calling by itself only in LLM-specific containers, and the multi-LLM NIM serves the same ids |
| `parser` | text | the documented parser name, for information |
| `named`, `none` | `yes`, `no`, `unknown` | absent means the engine default (yes). The named diagnostic is sent only for `named` `yes` or absent: it is optional, so `unknown` skips it too. `none: no` makes `tools_none` not applicable; `none: unknown` still runs it |
| `parallel` | `yes`, `no`, `unknown`, `info` | `info`: the sources disagree, so parallel calls are information only |
| `stream` | `yes`, `no`, `unknown` | `no` makes `tools_stream` not applicable |
| `mode` | list of switches | sent with every tool request, for example `detailed thinking off` |
| `sampling`, `max_tokens`, `stream_retry` | | request settings |

A source is `{"tier", "url", "doc_version", "quote"}`. The quote is verbatim text from the page, at
most 300 characters. A quote from a table is the row's cells joined with ` | `. ` ... ` joins two
parts of the same page, in page order.

A switch is `{"method", "key", "value"}`. `--reasoning-on` and `--reasoning-off` take the same
switches as text:

| text | sends |
|---|---|
| `system=/no_think` | a first system message `/no_think` |
| `user_suffix= /no_think` | ` /no_think` added to the last user message (the leading space is kept) |
| `kwargs.enable_thinking=false` | `chat_template_kwargs: {"enable_thinking": false}`; the value is read as JSON |
| `effort=low` | `reasoning_effort: "low"` |
| `default` | nothing |

`kwargs.chat_template` is refused: NIM answers it with HTTP 400.

## Source tiers

| tier | source | how it is used |
|---|---|---|
| A | NIM docs for the NIM version: support matrix, per-model guides, model-specific considerations, release notes (support changes, known issues), the 1.15.0 supported-models table, the VLM site pages | decides; an explicit no for this version wins |
| A | NIM Metadata API `features` for the exact tag (`tool_calling`, `parallel_tool_calling`) | decides when present, but its yes does not override a no in the record; absent means unknown; empty today |
| B | the image's default `NIM_PASSTHROUGH_ARGS` against the deployment's effective value and container arguments | live signal (`tier_b`): parser on by default, or dropped by an override |
| C | the hosted endpoint spec of build.nvidia.com | information only; the table does not use it |
| D | upstream hints: model cards, NGC descriptions, vLLM and SGLang docs | a D yes alone never makes tools or image testable; a D no can support not applicable; reasoning switches and sampling may come from the model card when the NIM docs are silent |
| agent | `--documented FEATURE=URL`, and `--reasoning-on` / `--reasoning-off` | for a model with no record, or a feature the record leaves unknown; never overrides a no |

Read the versioned HTML pages (`.../large-language-models/2.1.5/<page>.html`). The `.html.md`
twins under `/latest/` still serve 2.0.13 text, and versioned twins return 404.
`NIM_PASSTHROUGH_ARGS` set in a deployment replaces the image's value; it does not append. Tier B
shows which parser flags that removed. Tier B counts tool calling as on only with both
`--enable-auto-tool-choice` and `--tool-call-parser`, as vLLM needs both for `tool_choice` `auto`.
Container arguments come last, and `--no-enable-auto-tool-choice` there turns it off.
On the hosted spec, `showReasoningToggle` and `defaultEnabled` are playground settings, and
`structuredOutput: false` is not a no.

## Conflict rules

| case | result | example in the table |
|---|---|---|
| a tier-A no for this version | `not_applicable` (`not_documented_for_model`), citing it, even when C or D say yes | Llama 3.2 Vision tools |
| a yes from tier A or B, and no tier-A no | test | Gemma 3 1B tools: the NIM table says yes, the upstream template has no tools section |
| a yes and a no in one tier | prefer the more specific source (per profile over per model), then the newer version | Mistral 7B v0.3 parallel calls: 1.15.0 No wins over 1.8 Yes |
| still unclear | `documented: "conflict"`: test, and report both sources | DeepSeek V3.1 Terminus tools |
| two sources against one | the record follows the majority and lists the other under `conflicts` | Qwen2.5 tools: not applicable |
| sources disagree on parallel calls | `parallel: "info"`: information only | Llama 3.1 |
| silent everywhere | `unknown`: `not_applicable` (`not_documented`) | Cosmos Reason2 tools |
| documented as impossible to turn off | reasoning off `not_applicable` (`always_on`) | gpt-oss, DeepSeek-R1, GLM-5.3-Flash, Muse Glimmer |
| reasons by default, and no source names an off switch or says there is none | reasoning off `not_applicable` (`no_documented_switch`); a `--reasoning-off` switch from the docs is used | inkling ("enable_thinking ... do not disable reasoning" names no switch), Kimi K3.1 Pro, DeepSeek-V4-Pro 0813 |
| a 1.x `-variant` image has its own section on the container variants page (1.15.0, or the 1.14.0 page that 1.15.0 points to) | that section wins for the image: `TOOL_CALL_PARSER: Set to 1` makes tool calling opt-in (`default_on` false, and that variable is the hint); guided decoding listed as not supported makes `json_schema` `not_applicable`. When the standard image serves the same id, an override keyed on the variant's repository holds these values | Nemotron Nano 9B v2 and Qwen3-32B DGX Spark images, Qwen3 Next 80B A3B Thinking, Nemotron 3 Nano 1.x |
| documented, but the parser is off in this deployment | the parser table below | |

Never invent a quote. When no page states a value, set it to `unknown` and explain it in
`notes`; the check is then not applicable.

## How a plan is made

| check | test when | not applicable (reason) |
|---|---|---|
| health, models | always (health: self-hosted) | hosted API (`hosted_no_health_route`) |
| chat, stream, json_schema, concurrency | always | `out_of_scope`; a documented no (`not_documented_for_model`); NIM LLM 1.x (not a VLM NIM's own 1.x releases) with pipeline parallelism above 1 for json_schema (`pipeline_parallel_unsupported`; when no profile name shows `pp<N>`, as a 1.x `/v1/metadata` gives only the profile id, the check runs and the plan warns once) |
| tools | documented yes or conflict, the Metadata API, or an agent URL | `no_capability_record`, `not_documented`, `not_documented_for_model`, `pipeline_parallel_unsupported`, `out_of_scope` |
| tools_stream, tools_none | tools is tested and the record does not say no | as tools, or `not_documented_for_model` |
| tools_parallel | tools is tested and `parallel: yes`, or `--documented tools_parallel=URL`: that URL is enough on its own for a model with no record or with tool calling `unknown`, and it does not make the other tool checks run | as tools, or `information_only` |
| reasoning_on | a reasoning model with an on switch (or reasoning on by default) | `no_capability_record`, `not_a_reasoning_model`, `reasoning_not_detectable`, `no_documented_switch`, `not_enabled_in_deployment`, `out_of_scope` |
| reasoning_off | a documented off switch | `always_on`, `no_documented_switch`, as reasoning_on |
| reasoning_off_stream | reasoning off is tested and `prefills_off` is true | `no_prefilled_think_block`, as reasoning off |
| image | `kind: vlm`, documented, and the active profile is not text only (a profile whose id is unknown is tested, with the reply above and a warning) | `no_capability_record`, `text_only_model`, `text_only_profile`, `not_documented`, `out_of_scope` |

An `out_of_scope` family gets `out_of_scope` on every check except health and models. A test entry
has a reason too: `core_check`, `engine_feature`, `documented`, `documented_conflict` or
`documented_by_agent`. `json_schema`, `image` and `concurrency` send the family's off switch when
it has one, so they test the feature and not the reasoning; `chat` and `stream` send no switch, so
they see the model as a plain client does (64 tokens, or 1024 when the model may reason or has no
record).
When the model reasons by default and has no off switch, `json_schema` and `image` get at least
the record's reasoning `max_tokens` (inkling and Muse Glimmer 4096, GLM-5.3-Flash and Kimi K3.1
Pro 8192), and the record's on switch when it only lowers the effort (`reasoning_effort` `low`:
gpt-oss, GLM-5.3-Flash).

When vLLM answers a tool request with HTTP 400 `"auto" tool choice requires
--enable-auto-tool-choice and --tool-call-parser to be set`, no tool parser runs in this
deployment. The tools entry's `if_parser_missing` says what to report then. The hint always names
the documented switch and says that changing the deployment is nim-deploy's job.

| what the sources say | result |
|---|---|
| tier B: the deployment's `NIM_PASSTHROUGH_ARGS` replaced an image default that had the parser | `fail`, `parser_dropped_by_override` |
| the record (`default_on: true`; on NIM 1.x only for the record's own image) or tier B: the image turns the parser on by default | `fail`, `parser_not_enabled` |
| the record: opt-in (`default_on: false`) | `not_applicable`, `not_enabled_in_deployment` |
| unknown | `not_applicable`, `not_enabled_in_deployment`, with a warning (never called opt-in); when the deployment sets `NIM_PASSTHROUGH_ARGS` without both tool flags and the image default cannot be read, the warning names that override as the likely cause. A 1.x `default_on: true` with the image unknown, or with another image (the multi-LLM NIM), is unknown too, and the warning says why |
| the hosted API | `fail`, `parser_not_enabled` (no deployment hint) |

## When a feature is not applicable and the user disagrees

| reason | what to do |
|---|---|
| `no_capability_record` | No record matches. If the model was renamed (`NIM_SERVED_MODEL_NAME`), pass `--image` or `--family ID` (a family that matches none of the target's ids gets a warning). Otherwise ask for the page that documents the feature for this model, and rerun with `--documented FEATURE=URL` (`tools`, `tools_parallel`, `reasoning`, `image`). For reasoning, also pass `--reasoning-on` and `--reasoning-off` with the switch the page gives. |
| `not_documented` | The record has no source for the feature. As above: a page, then `--documented`. |
| `no_documented_switch` | Ask for the switch from the docs; pass it with `--reasoning-on` or `--reasoning-off`. |
| `not_documented_for_model`, `text_only_model`, `not_a_reasoning_model`, `always_on`, `reasoning_not_detectable` | A source says no. Show it. `--documented` does not override it. A newer NIM page that says yes means the record needs an update (below). |
| `not_enabled_in_deployment` | The docs call the parser opt-in, or do not say (then a warning says so), and this deployment does not turn it on. The hint names the server argument; the change is nim-deploy's job. On NIM 1.x behind a `--url`, ask which image the NIM runs and rerun with `--image`: in the model's own image a missing parser is a fail. |
| `text_only_profile` | The active profile takes text only; the support matrix lists the profiles that take images. A profile change is nim-deploy's job. |
| `information_only` | The sources disagree, so the check does not run. |

## Models without a record

These NIMs are documented (the 2.1.5 support matrix, or the per-family tables the table was built
from) but have no record yet, so their model-specific checks are `no_capability_record`. Treat
them as above: ask the user for the page that documents the feature and pass
`--documented FEATURE=URL` (and `--reasoning-on` / `--reasoning-off` for reasoning). The list
holds the gaps the last review found, not every NIM.

| model | served id | why there is no record yet |
|---|---|---|
| DeepSeek-V4-Flash-0731 (2.1.5 LLM; also a 1.11.0-variant) | `deepseek-ai/deepseek-v4-flash-0731` | only the 1.x variant is documented (thinking through the `thinking` kwarg; no structured output while DSpark speculative decoding is on), both images serve this id, and no 2.x page gives its features |
| DeepSeek-V4-Flash (1.10.0-variant) | `deepseek-ai/deepseek-v4-flash` | documented only as a 1.x variant; not reviewed yet |
| Nemotron 3.5 Lightning (2.1.5 LLM) | `nvidia/nemotron-3.5-lightning-30b-a3b` | no NIM page documents its tool calling or reasoning |
| Qwen3.8-Flash-Next (vLLM Model-Free NIM) | `Qwen/Qwen3.8-Flash-Next` | the 2.1.5 release notes give the image's parser flags, but no page documents its reasoning switch or image input |
| Step 5 Preview (vLLM Model-Free NIM) | `stepfun-ai/Step-5-Preview` | its guide documents text and image input but no reasoning switch; not reviewed yet |
| Muse Spark Open (SGLang Model-Free NIM) | `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` | not reviewed yet (its 2.0.13 guide documents the `reasoning_effort` levels) |
| Mistral Large 4 (Model-Free NIM) | `mistralai/Mistral-Large-4-1T-A52B` | not reviewed yet |
| Llama 4 Scout 17B 16E Instruct (VLM NIM) | `meta/llama-4-scout-17b-16e-instruct` | tool calling came with a later VLM release ("Tool calling is now supported"), so it depends on the tag; not reviewed yet |
| StarCoder2 7B (2.1.5 LLM) | `bigcode/starcoder2-7b` | a code model; not reviewed yet |

When one of them gets a record, remove its row: `tests/test_nsc_features.py` checks that every
id listed here matches no record.

## Matching

1. `--family ID` picks the record. It is meant for a served id that `NIM_SERVED_MODEL_NAME`
   renamed from a model of that family. When none of the family's ids and patterns matches the
   served id, its root or the image, the run warns (`facts.warnings`): the family's verdicts and
   sources then hold only if the NIM really runs one of its models. Otherwise rerun without
   `--family` and pass `--documented FEATURE=URL` from the model's own docs.
2. The image's NGC repository: `nvcr.io/nim/TEAM/REPO:TAG` (or the same path on a mirror) becomes
   `nim/TEAM/REPO`. It comes first because `NIM_SERVED_MODEL_NAME` can rename the served model.
3. The served model id from `/v1/models`, then its `root`.

Each step tries the exact ids, then the patterns. Case does not matter, and `.`, `-` and `_` count
as the same character, so write patterns in lower case with `-`, anchored with `^` and `$`. An
exact id belongs to one family, and no pattern may match another family's ids: `load_table`
refuses the table otherwise. Overrides whose `ids` or `patterns` match the image, the served id or
the root are applied to a copy of the record. An override that sets a feature's main value
(`documented`, or `default` for reasoning) replaces that feature's sources; otherwise its sources
are added. A `null` in an override removes that key, `sources` and `conflicts` too. An override's
`ids`, when present, is a non-empty list of model ids; leave it out when the override has
`patterns` only.

On NIM 1.x the image decides some features: the model's own container turns tool calling on by
itself, the multi-LLM NIM (`nvcr.io/nim/nvidia/llm-nim`) serves the same ids with tool calling
opt-in, and variants such as the DGX Spark one differ in tool calling and structured output. A
1.x plan with no image (a `--url` run without `--image`) therefore warns, and asks for `--image`.
These 1.x rules, and the pipeline parallelism one, are for NIM LLM 1.x only: a VLM NIM numbers
its own releases (VLM NIM 1.7.0), so a 1.x tag on a vlm record skips them.

## Add or update a record

1. Read the NIM docs for the version the user runs (tier A): the support matrix (LLMs or VLMs
   section, the Input column), the model's guide or model-specific considerations, and the
   release notes for that model. For 1.x NIMs, read the 1.15.0 supported-models table and the
   VLM site.
2. Copy each quote exactly, at most 300 characters, with the URL and its anchor, the doc version
   and the tier.
3. Add the record, or change it, in `scripts/model_features.json`. Put a model that differs from
   its family under `overrides`.
4. Validate from the project root, with the real skill folder for `<SKILL_DIR>` (SKILL.md):
   `python3 -I "<SKILL_DIR>/scripts/nsc_features.py" --check` must print `"ok": true`, and the
   tests must pass:
   `python3 -m pytest -q -p no:cacheprovider -o addopts="" "<SKILL_DIR>/tests/test_nsc_features.py"`.
5. Add the model's served id and NGC repository to the matching tests, and a plan test when the
   record has a new kind of decision.

A record from the table, trimmed: a text model whose tool calling the 1.15.0 table documents.

```json
{
  "id": "mistral-7b-v0.3",
  "name": "Mistral 7B Instruct v0.3",
  "match": {"repos": ["nim/mistralai/mistral-7b-instruct-v0.3"],
            "served_ids": ["mistralai/mistral-7b-instruct-v0.3"],
            "patterns": ["^mistralai/mistral-7b-instruct-v0-3$"]},
  "status": "eos",
  "kind": "llm",
  "backend": "either",
  "reasoning": {"default": "none"},
  "tools": {"documented": "yes", "default_on": {"1.x": true}, "parser": "mistral",
            "parallel": "no", "stream": "unknown",
            "sources": [{"tier": "A",
                         "url": "https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html",
                         "doc_version": "1.15.0",
                         "quote": "Mistral 7B Instruct V0.3 | Mistral | mistralai/mistral-7b-instruct-v0.3 | 1.0, 1.1, 1.1.2, 1.3, 1.8.4, 1.12.0 | Yes | Yes | No | No"}]},
  "image": {"documented": "no"}
}
```

## Review per NIM release

The table is reviewed for every NIM release. Update `reviewed` and `docs`, then check:
- the new release notes: support changes, new known issues (with their anchors), fixed issues;
- the support matrix: new models, models moved between the LLM and VLM sections, new Input
  columns;
- new per-model guides and model-specific considerations, for switches and parser defaults;
- models that reached End of Support (`status: eos`);
- that the quotes are still on their pages.
