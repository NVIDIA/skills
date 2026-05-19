# Reward Profiling Quick Start

Substitute environment-specific config paths, input data, model endpoint, and output paths.

## Minimal Flow

```bash
CONFIG_PATHS="your_model_config_paths,your_env_config_paths"

POLICY_MODEL_NAME="your_policy_model_name"
POLICY_BASE_URL="your_policy_base_url"
POLICY_ENDPOINT_KEY="your_policy_endpoint_key"

DATA_JSONL="/path/to/your_input.jsonl"
ROLLOUTS_JSONL="/path/to/your_rollouts.jsonl"
MATERIALIZED_JSONL="${ROLLOUTS_JSONL%.jsonl}_materialized_inputs.jsonl"

AGENT_NAME="your_agent_name"
NUM_REPEATS=2
NUM_SAMPLES_IN_PARALLEL=8

ng_run "+config_paths=[$CONFIG_PATHS]" \
    +policy_model_name="$POLICY_MODEL_NAME" \
    +policy_base_url="$POLICY_BASE_URL" \
    +policy_api_key="$POLICY_ENDPOINT_KEY" &
NG_RUN_PID=$!
trap 'kill "$NG_RUN_PID" 2>/dev/null || true' EXIT
./scripts/wait_for_servers.sh "$NG_RUN_PID"

agent_args=()
if [[ -n "$AGENT_NAME" ]]; then
    agent_args=(+agent_name="$AGENT_NAME")
fi

ng_collect_rollouts \
    "${agent_args[@]}" \
    +input_jsonl_fpath="$DATA_JSONL" \
    +output_jsonl_fpath="$ROLLOUTS_JSONL" \
    +num_repeats="$NUM_REPEATS" \
    +num_samples_in_parallel="$NUM_SAMPLES_IN_PARALLEL"

ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    ++rollouts_jsonl_fpath="$ROLLOUTS_JSONL"
```

If rows already contain `agent_ref`, leave `AGENT_NAME` empty. Passing `+agent_name` supplies a default for rows without one.

## Partial Rollouts

By default, `ng_reward_profile` expects every materialized input row to have a matching rollout row. If collection stopped early, profile the completed rollouts with:

```bash
ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    ++rollouts_jsonl_fpath="$ROLLOUTS_JSONL" \
    ++allow_partial_rollouts=True
```

Partial profiling writes rows only for original input tasks with at least one completed rollout. The command prints how many input tasks were complete, partial, or dropped because they had no rollout.
