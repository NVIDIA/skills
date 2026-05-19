# Diagnostic Snippets

Scope: packageable Nemo Gym debugging commands. Substitute paths before running.

Prefer commands that read logs and JSONL files without mutating state. Do not remove caches or outputs until the stale-cache hypothesis is supported and the user agrees.

## Find the First Useful Error

```bash
rg -n -C 4 "Traceback|Exception|ERROR|ValueError|KeyError|TypeError|Validation|Pydantic|Unprocessable|Verifier|verify|sandbox|Ray|vLLM|ready|died|timeout" PATH_TO_LOG | head -240
```

Shutdown `CancelledError` stacks are often secondary. Prefer the first verifier/resource-server exception before shutdown.

## Boundary Diagnostic Logs

Use this when request-boundary debug is enabled or when a log may contain Gym boundary markers.

```bash
rg -n -C 3 "\\[rollout_collection\\] /run failed|Request info:|Response content:|Request kwargs:|Hit an exception in|/run failed|/v1/responses failed" PATH_TO_LOG | head -240
```

If the useful inner body is still absent, read `request-boundary-visibility.md`.

## Output Shape

```bash
wc -l PATH_TO_OUTPUT.jsonl PATH_TO_MATERIALIZED_INPUTS.jsonl PATH_TO_REWARD_PROFILE.jsonl 2>/dev/null
stat -c '%y %s %n' PATH_TO_SOURCE.jsonl PATH_TO_MATERIALIZED_INPUTS.jsonl PATH_TO_OUTPUT.jsonl 2>/dev/null
```

Interpretation:

- materialized inputs exist, rollout output missing: collection may not have started or output path changed
- rollout output has very few rows: early verifier/runtime failure or cancellation
- materialized inputs older than source data: suspect stale cache
- profiling/metrics missing while rollouts exist: profiling step or reward profile command failed

## JSONL Shape Summary

```bash
python - <<'PY'
import json
from collections import Counter
p = "PATH_TO_JSONL"
key_counts = Counter()
type_counts = Counter()
examples = {}
with open(p) as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        key_counts.update(obj.keys())
        for k, v in obj.items():
            type_counts[(k, type(v).__name__)] += 1
            examples.setdefault((k, type(v).__name__), (i, repr(v)[:300]))
        if i >= 1000:
            break
print("keys", key_counts.most_common(40))
print("types", type_counts.most_common(60))
for (k, typ), (line, ex) in list(examples.items())[:20]:
    print(k, typ, "line", line, ex)
PY
```

## Compare Source and Materialized Rows

```bash
python - <<'PY'
import json
for p in ["PATH_TO_SOURCE.jsonl", "PATH_TO_MATERIALIZED_INPUTS.jsonl", "PATH_TO_OUTPUT.jsonl"]:
    print("\\n", p)
    try:
        with open(p) as f:
            for i, line in zip(range(2), f):
                obj = json.loads(line)
                print("row", i, sorted(obj.keys()))
                for k in ["agent_ref", "responses_create_params", "verifier_type", "_ng_task_index", "_ng_rollout_index", "response", "reward"]:
                    if k in obj:
                        print(k, type(obj[k]).__name__, repr(obj[k])[:500])
    except FileNotFoundError:
        print("missing")
PY
```

## Server Readiness Search

```bash
rg -n "All .* servers ready|Uvicorn running|/models|Started Ray cluster|Connected to Ray|Loaded .* tools|Sandbox ready|readiness timeout|connection_error|not ready" PATH_TO_LOG
```

If readiness succeeds and verifier schema errors follow, stop debugging Ray/vLLM and inspect data/config.

## Per-File Log Overview

```bash
find PATH_TO_LOG_DIR -maxdepth 2 -type f | sort | sed -n '1,200p'
wc -l PATH_TO_LOG_DIR/* 2>/dev/null | sort -n | tail -40
```

Use this to find the log that actually contains the first failure. Driver logs often show summaries; worker/resource logs often show the root exception.

## Rollout Reward and Usage Probe

```bash
python - <<'PY'
import json
from collections import Counter
p = "PATH_TO_ROLLOUTS.jsonl"
rewards = Counter()
usage_keys = Counter()
missing_usage = 0
with open(p) as f:
    for i, line in enumerate(f, 1):
        row = json.loads(line)
        rewards[repr(row.get("reward"))] += 1
        usage = (row.get("response") or {}).get("usage")
        if not usage:
            missing_usage += 1
        else:
            usage_keys.update(usage.keys())
print("reward values", rewards.most_common(20))
print("usage keys", usage_keys.most_common(20))
print("missing usage", missing_usage)
PY
```

## Materialized Identity Probe

```bash
python - <<'PY'
import json
from collections import Counter
p = "PATH_TO_MATERIALIZED_INPUTS.jsonl"
tasks = Counter()
rollouts = Counter()
with open(p) as f:
    for line in f:
        row = json.loads(line)
        key = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
        tasks[row.get("_ng_task_index")] += 1
        rollouts[key] += 1
print("num tasks", len(tasks))
print("repeat count histogram", Counter(tasks.values()).most_common())
print("duplicate task/rollout keys", [k for k, c in rollouts.items() if c > 1][:20])
PY
```

If the target repo uses different global key names, inspect `nemo_gym/global_config.py` and substitute them.
