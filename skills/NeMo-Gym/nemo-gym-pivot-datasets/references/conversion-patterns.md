# Conversion Patterns

## Normalize By Meaning

Source artifacts differ in field names, nesting, and whether model outputs are stored as chat
messages, Responses API items, completion objects, or custom rollout records. Do not make those
containers the skill's contract. For each source, first identify:

- the state before the target assistant action
- the target assistant action
- the tools available at that state
- the local reward/verifier target
- the task and trajectory provenance

Once those are identified, build the Gym pivot row from those semantic pieces.

## Model Input

`responses_create_params.input` should represent exactly the context the policy should see before
producing the pivot action.

Common mappings:

- system/developer/user turns become Responses `message` input items.
- prior assistant text becomes completed assistant message output items.
- prior assistant reasoning can be preserved as completed reasoning items when the target agent/model path accepts it.
- prior tool results become `function_call_output` items with the matching `call_id`.
- prior assistant tool calls become completed `function_call` items with `name`, `arguments`, `call_id`, and `status`.

The pivot action itself must not be included in the input prefix.

## Tools

Preserve the tool schema visible at the pivot state. For function tools, prefer the normalized
shape accepted by the target agent/resource server:

```json
{
  "type": "function",
  "strict": true,
  "name": "tool_name",
  "description": "Tool description",
  "parameters": {"type": "object", "properties": {}, "required": []}
}
```

Do not drop tools just because a pivot's expected action uses only one of them. The policy should
see the same action space that existed in the original state.

## Expected Action

For `single_step_tool_use_with_argument_comparison`, use these canonical action shapes:

```json
{"type": "function_call", "name": "tool_name", "arguments": "{\"x\": 1}"}
```

```json
{"type": "message", "content": "final assistant text"}
```

Keep `arguments` as a JSON string for function calls. Validate that each argument string decodes as
JSON before writing the row.

`expected_action` is singular. If a source assistant action contains more than one tool call, filter
it out of the pivot dataset and keep it only in a skipped-row audit if it needs review.

## Provenance

Every converter should preserve enough optional provenance to debug a bad pivot without reopening
the full source artifact when practical:

- source task id and source row id
- source trajectory id or rollout id
- source reward and source success marker when available
- pivot message index and assistant-action index
- trajectory length and pivot depth
- original uuid or a generated stable uuid if the source lacks one
- original metadata, unless it contains fields the target schema rejects

Use a dataset-specific provenance object rather than flattening every source field into the top
level when carrying metadata would help later analysis.

## Skips And Metrics

Write skipped-row audit artifacts when rows are rejected. Track at least:

- no expected action
- malformed JSON arguments
- missing tool schema
- unsupported action type
- assistant actions excluded because they contain more than one tool call

Write count-and-percent summaries for action type, tool name, depth, source reward group, and any
selection filters. This catches silent data-shape changes before training.
