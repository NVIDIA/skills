# vLLM Tool-Call Schema Checks

Use this when a Nemo Gym dataset sends `responses_create_params.tools` to a
vLLM/OpenAI-compatible endpoint and failures happen before meaningful model
generation.

## Failure Profile

Symptoms:

- Gym reports nested 500s from agent/model servers.
- vLLM returns 400s mentioning grammar, guided decoding, Outlines, JSON Schema,
  `Unknown format`, or `Unsupported JSON Schema`.
- Direct endpoint probes fail before the model returns a tool call.
- The verifier is never reached, or the rollout dies at the policy model call.

Common causes:

- tool schemas contain JSON Schema constructs that are valid in general but not
  supported by the vLLM/Outlines tool grammar path
- boolean schema nodes in model-facing tool parameters, especially object
  closure like `additionalProperties: false`
- `format` annotations such as `uri`
- invalid entries inside `properties` maps
- stale materialized inputs preserve an older bad tool schema after regenerating
  source data

The key distinction: this is a model-serving schema compatibility issue, not a
tool execution issue. Do not debug sandbox/tool execution if the request fails
while compiling the tool grammar.

## Static Check

If the repo has a local checker for its own row contract, prefer that. If it
does not, use the portable script bundled with this skill:

```bash
python scripts/check_tool_call_jsonl.py \
    -i PATH_TO_TOOL_CALL_DATA.jsonl
```

Expected output:

```text
Errors: 0
```

These checks are offline and do not prove the endpoint can compile every
schema, but they catch the high-value failures before running expensive
rollouts.

Optional row summaries can help compare a failing subset with the full data:

```bash
python scripts/check_tool_call_jsonl.py \
    -i PATH_TO_TOOL_CALL_DATA.jsonl \
    --summary-key tool_schema_mode \
    --summary-key num_tools
```

## Endpoint Preflight

When static checks pass but vLLM may still reject the grammar, run a direct
endpoint preflight from the active repo. The useful pattern is: preserve
`tools`, preserve the same Responses-to-Chat conversion path used by the model
wrapper, replace the expensive document with a tiny prompt, and send a tiny
`max_tokens` request.

Treat incomplete generation parse errors after grammar compilation separately
from schema/compiler errors. A short `max_tokens=1` request can produce
incomplete tool-call JSON while still proving the schema compiled.

## How To Classify Results

- Static checker fails: fix data generation or the model-facing tool schema.
- Static checker passes, endpoint preflight fails with grammar/schema error:
  inspect the exact provider error and add a targeted compatibility transform
  in data generation.
- Endpoint preflight passes, Gym rollout fails before verifier: inspect
  Responses-to-Chat conversion and model wrapper logs.
- Gym reaches verifier and then fails reward: debug verifier/data semantics,
  not vLLM schema compilation.
