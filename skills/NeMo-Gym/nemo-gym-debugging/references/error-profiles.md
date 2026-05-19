# Error Profiles

Scope: detailed Nemo Gym failure classification. Use this file after a run fails or behaves suspiciously.

## Read Order

Start with the first fatal error in time order, not the loudest shutdown stack. Then classify the layer:

- Slurm/container/filesystem
- Ray process layout
- model serving/vLLM
- Gym config/Hydra
- resource server/verifier
- data/schema/materialized cache
- judge model
- tool/sandbox
- throughput/backpressure
- profiling/postprocess

If every visible error is a nested "Hit an exception in ... calling an inner server" 500 and the inner model/verifier/provider body is missing, use `request-boundary-visibility.md`. Start with existing request debug before adding temporary boundary logs.

## Slurm, Container, Filesystem

Symptoms:

- job exits before Ray or Gym logs appear
- `srun` reports container/image/mount errors
- logs are missing, empty, or written in an unexpected directory
- permissions or path-not-found errors appear before Python startup

Likely causes:

- wrong container image path
- missing mount for data, cache, output, or sandbox temp directory
- log directory not created before job start
- submit environment did not export expected variables
- base path differs between submit host and container

Evidence to collect:

- Slurm stdout/stderr
- first 100 lines of driver/head/worker logs
- resolved command and environment exports, if logged
- `stat`/`ls` on data, output, and log dirs

Next actions:

- fix image/mount/log/export before touching Gym code
- make log paths explicit and env-controlled in launchers
- avoid CPU-only submit paths when the run requires model GPUs

## Ray Layout

Symptoms:

- head starts but workers never join
- worker logs show connection refused to Ray head
- resources are fewer than expected
- driver waits indefinitely for placement/resources

Likely causes:

- wrong head node address
- worker `srun` excluded or included wrong nodes
- container networking mismatch
- GPUs not visible inside worker containers
- single-node vs multi-node logic mismatch

Evidence to collect:

- head log readiness line
- worker logs around Ray start
- resource summary from the driver, if available
- Slurm node allocation

Next actions:

- verify head and worker launch commands separately
- distinguish "no worker needed" single-node behavior from failed multi-node registration
- do not debug env data until Ray resources are sane

## vLLM or Policy Model Serving

Symptoms:

- `/models` never becomes available
- rollout requests fail with connection errors
- model logs show OOM, tokenizer load failure, unsupported dtype, or port conflict
- very low throughput with no verifier errors

Likely causes:

- model path or served model name mismatch
- insufficient GPU memory or wrong tensor/pipeline parallel settings
- port already in use
- policy base URL points to the wrong port/node
- `num_samples_in_parallel` too high for available serving capacity

Evidence to collect:

- vLLM startup logs
- model readiness probes
- policy base URL/model name in Gym config
- first failed OpenAI-compatible request error

Next actions:

- confirm `/v1/models` before running large collection
- tune concurrency only after readiness is stable
- separate serving bottlenecks from verifier/judge/tool bottlenecks

## Hydra or Config Composition

Symptoms:

- `ng_run` or `ng_collect_rollouts` fails before launching servers
- errors mention missing config keys, unknown overrides, or interpolation failures
- wrong agent/resource server starts despite expected data

Likely causes:

- config path typo
- composition order override collision
- branch uses different config schema than the script assumes
- launcher passes stale extra args
- explicit `agent_name` conflicts with row-level routing expectations

Evidence to collect:

- full command line
- composed config if the repo can print it
- target YAML files and extra overrides
- CLI help for the target checkout

Next actions:

- reduce to a minimal `ng_run "+config_paths=[...]"` command
- add overrides back one at a time
- trust target checkout code over copied command templates

## Data Schema or Materialized Cache

Symptoms:

- verifier errors mention missing fields, wrong types, invalid enum values, or parser failures
- source data was fixed but failures still mention old bad values
- materialized inputs contain different data than source JSONL
- output row count resumes from an older partial run

Likely causes:

- JSONL does not match verifier request model
- row-level `responses_create_params` has wrong shape
- row-level `agent_ref` missing or wrong
- stale `*_materialized_inputs.jsonl`
- `resume_from_cache=True` reused partial output after schema/config changes

Evidence to collect:

- first failing row, redacted if needed
- first few source/materialized/output rows
- modification times of source, materialized, and output files
- Pydantic validation error, if available

Next actions:

- validate rows against the request model
- rerun on a clean output path after schema/config changes
- if data owner must fix it, send the log, schema error, and a minimal row sample

## Resource Server or Verifier Runtime

Symptoms:

- Gym servers are ready, model requests complete, but verification returns 500/422
- traceback is inside a resource server or verifier
- rollouts are written until a specific row triggers failure

Likely causes:

- verifier assumes a field absent from data
- verifier cannot parse model output
- external dependency missing in container
- env-specific scoring code bug
- stale cache sends old schema to new verifier

Evidence to collect:

- first verifier traceback
- request body shape, if logged
- sample source/materialized row for the failing task
- verifier request/response models

Next actions:

- decide whether it is data, verifier code, or container dependency
- do not change model serving unless model calls are failing before verification

## Judge Model

Symptoms:

- policy rollouts complete but verifier waits or fails during judge calls
- errors mention judge base URL, judge server name, parse errors from judge output, or judge timeout
- throughput is much lower than direct-verifier envs

Likely causes:

- judge server not started or not registered
- judge base URL/server name mismatch
- judge prompt expects fields missing from data
- judge output parser too strict
- judge model capacity lower than policy rollout concurrency

Evidence to collect:

- judge server logs and readiness
- resource server config for judge client
- first judge request/response error
- data fields consumed by judge prompt

Next actions:

- verify judge readiness independently
- tune concurrency against judge throughput
- classify parser failures separately from infra failures

## Tool or Sandbox

Symptoms:

- tool env starts but first tool call fails
- sandbox logs missing or never show readiness, when a sandbox is expected
- errors mention unknown tool, tool timeout, code execution failure, missing package, or session restore
- only tool-integrated envs fail while direct envs work

Likely causes:

- tool execution ownership is unclear: agent, resource server, or sandbox
- sandbox image does not contain expected runtime, when a sandbox is expected
- sandbox command was not launched on the nodes that need it, when a sandbox is expected
- tool spec omitted or mismatched with agent
- per-node ports/mounts not exported
- concurrency overwhelms sandbox
- stale materialized inputs preserve old tool specs

Evidence to collect:

- sandbox startup logs
- tool loading logs
- first tool call request/error
- data row `responses_create_params.tools`
- relevant agent/resource config

Next actions:

- check whether the env actually needs a sandbox
- check tool names and ownership before changing model settings
- check sandbox readiness before data/schema work only when a sandbox is configured
- reduce concurrency to separate correctness from saturation

## Throughput and Saturation

Symptoms:

- no correctness errors, but jobs run slowly
- vLLM logs show sustained high request concurrency
- GPU utilization high but output advances steadily
- resource server or judge logs show queues/timeouts under load

Likely causes:

- expected saturation from high `num_samples_in_parallel`
- verifier/judge/tool bottleneck
- policy model context length or generation length dominates latency
- filesystem logging overhead

Evidence to collect:

- rollout completion rate
- model server throughput logs
- verifier/judge/tool latency or timeout lines
- output JSONL growth over time

Next actions:

- if output is steady and errors absent, do not "fix" saturation
- tune concurrency per env; GPU count alone is not the concurrency setting
- reduce concurrency for debugging, increase only after correctness is established

## Profiling or Postprocess

Symptoms:

- rollout JSONL exists but profiling output is missing or empty
- profile command fails on missing task/rollout keys
- token usage metrics are missing or all null
- materialized or rollout counts do not match expected repeats

Likely causes:

- profiling command used wrong input paths
- rollouts were collected without repeated samples
- materialized inputs and rollouts are from different runs
- results are out of order and code assumes zip alignment without sorting/joining
- responses do not include `usage`

Evidence to collect:

- line counts for source, materialized, rollout, and profiling files
- sample rows showing task/rollout keys
- reward and usage field summary
- target `nemo_gym/reward_profile.py`

Next actions:

- join by `(task index, rollout index)`
- keep missing usage explicit
- profile from clean materialized/output pairs
