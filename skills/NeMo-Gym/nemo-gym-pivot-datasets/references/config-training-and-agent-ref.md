# Config, Training, And Agent Ref

## Use The Pivot JSONL Directly

The pivot-format JSONL is the dataset. A Gym config should point a train or eval dataset entry
directly at the generated JSONL:

```yaml
example_single_step_tool_use_with_argument_comparison_agent:
  responses_api_agents:
    tool_simulation_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_single_step_tool_use_with_argument_comparison_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: /path/to/pivot.jsonl
        num_repeats: 1
        license: TBD
```

Place the config beside the generated pivot dataset when the user asks for a self-contained pivot
bundle.

## agent_ref Alignment

Each row should carry:

```json
{"type": "responses_api_agents", "name": "example_single_step_tool_use_with_argument_comparison_agent"}
```

The config must define that same agent name, or the launcher must intentionally override it. If the
dataset row and YAML agent name disagree, the row can validate as JSON but still route to the wrong
resource server or model path.

Use the dataset-specific prefix in `agent_ref.name`, not a generic name, when multiple pivot
datasets share the same resource-server family.

## Resource-Server Knobs

For `single_step_tool_use_with_argument_comparison`, the commonly tuned config block is:

```yaml
tool_call_comparator_config:
  word_count_similarity_threshold: 0.1
  floating_point_comparison_threshold: 1.0e-6
```

Knobs:

- `word_count_similarity_threshold`: for string arguments with at least two words on both sides,
  split on whitespace, lowercase tokens, count token multiplicities, and compute
  `intersection_count / (expected_word_total + actual_word_total)`. Lower is more permissive;
  higher is stricter. This is a word-overlap threshold in the current code, not a standard IoU.
- `floating_point_comparison_threshold`: absolute tolerance for float argument comparison.

Use `tool_choice: "auto"` in pivot rows for this workflow. Avoid `tool_choice: "required"` because
some inference engines, including vLLM paths, can treat it as a structured decoding request rather
than ordinary tool-choice behavior.

Keep each pivot row to one `function_call` expected action or one `message` expected action.

## Pivot Selection For Training

PivotRL-style training benefits from informative local states. Prefer:

- clean source trajectories with positive reward for the demonstrated pivots.
- tasks whose source trajectory group has mixed rewards when that grouping is available.
- candidate pivots that remain mixed under local on-policy rollout from the downstream or initial policy.

Default profiling rule:

1. Sample at least 8 local rollouts per candidate pivot when feasible.
2. Score with the target verifier.
3. Keep reward groups with both 0 and 1.
4. Discard all-1 groups because they are already easy.
5. Discard all-0 groups because they are usually too hard, underspecified, or mismatched to the verifier.
6. If data is abundant, drop the easiest retained pivots first and keep harder mixed-reward pivots.

Source-task mixed reward is useful but not required. A source model and downstream policy can be far
apart, so downstream local reward variance is the stronger selection signal when it is available.

Reference: [Yi et al., "PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost"
(arXiv:2603.21383v1, 2026)](https://arxiv.org/html/2603.21383v1).

## Minimum Acceptance Criteria

Before training:

- sampled rows pass the target resource-server request models.
- `agent_ref.name` matches the config.
- expected tool names exist in `responses_create_params.tools`.
- no `responses_create_params.metadata: null` rows are present.
- optional provenance is present when needed for debugging, filtering, or later analysis.
- selection metrics show how many candidates were kept, skipped, all-success, all-failure, and mixed.
