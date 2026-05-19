# Pivot Row Contract

## Required Top-Level Fields

For the single-step tool-use verifier family, each pivot row should contain:

```json
{
  "responses_create_params": {},
  "expected_action": {},
  "agent_ref": {"type": "responses_api_agents", "name": "dataset_agent_name"}
}
```

Other fields are optional metadata. They can be useful for traceability, filtering, and debugging,
but they are not part of the `single_step_tool_use_with_argument_comparison` resource-server input
contract.

## responses_create_params

The target object should be a minimal non-null Responses API request body:

```json
{
  "input": [],
  "tools": [],
  "parallel_tool_calls": false,
  "tool_choice": "auto"
}
```

Rules:

- `input` is the model-call prefix before the pivot action.
- `tools` is the full tool list available at that state.
- `tool_choice` should be `"auto"`. Avoid `"required"` for this workflow because it can trigger
  structured decoding paths in inference engines such as vLLM.
- omit optional fields whose value would be null, especially `metadata`.
- include `model` only if the local agent/model path expects row-level model names.

## expected_action

Supported action types for `single_step_tool_use_with_argument_comparison`:

- `function_call`: one tool call with `name` and JSON-string `arguments`.
- `message`: expected assistant text response.

For tool actions:

- every expected tool name should appear in `responses_create_params.tools`.
- every `arguments` value should decode as JSON.

`expected_action` is singular. If a source assistant action contains more than one tool call, filter
it out of the pivot dataset and keep it only in a skipped-row audit if it needs review.

## agent_ref

Use row-level routing unless the target launcher intentionally overrides it:

```json
{"type": "responses_api_agents", "name": "example_single_step_tool_use_with_argument_comparison_agent"}
```

The `agent_ref.name` should match the config agent block that consumes the dataset. The matching
resource-server block is usually the agent name with `_agent` replaced by `_resources_server`, but
always verify against the actual config.

## Optional Provenance

Provenance is recommended but not required by the verifier. Keep enough metadata to debug a pivot
without reopening the full source artifact when practical:

- task id or sample id
- source uuid, if present
- source trajectory/rollout id, if present
- source reward and clean-positive marker, if available
- pivot item index or assistant-action index, if meaningful
- depth or assistant depth, if meaningful
- original metadata that is useful and safe to carry

## Compatibility Checks

A pivot row is not done until it passes:

- JSONL parse and fixed row-shape checks.
- expected-action schema validation.
- Responses request schema validation when Gym models are available.
- agent/config alignment checks.
