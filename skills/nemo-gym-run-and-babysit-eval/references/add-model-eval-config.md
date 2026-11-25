---
name: add-model-eval-config
description: >
  Onboard a model for NeMo Gym evaluations and compose the complete eval config: the vLLM policy
  service, its wiring as the policy, the reasoning toggle, and each benchmark's own config from
  Gym's `benchmarks/<name>/`, then validate and smoke it. Use when a new model or checkpoint has to
  be evaluated in Gym and no eval config exists yet, including a quantized export (NVFP4, FP8) of a
  model that also runs in BF16. run-and-babysit-eval follows this skill itself when it is asked to
  run a model that has no config yet, and runs the config this skill writes. Not for adding a
  benchmark (use add-benchmark). Triggered by: "add a model", "new checkpoint", "serve this model in
  Gym", "model config", "vllm config", "enable thinking", "reasoning parser", "tool call parser",
  "compare two models fairly", "eval config", "where does the benchmark config come from", "add an
  NVFP4 variant", "serve the quantized checkpoint with the BF16 settings".
---

# Add a Model Config to Gym

The Gym pages are the reference; when they and this skill disagree, the pages win:
[Configuration](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/configuration.mdx) (the `services`
and `driver` fields),
[Multi-node vLLM](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/how-to/multi-node-vllm.mdx) and
[vLLM model server](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/model-server/vllm.mdx) (the `vllm_model`
fields, `chat_template_kwargs`, `extra_body`).

A model in a Gym evaluation has two parts:

- **The deployment**: a `vllm` service in the submit config, started with `vllm serve`, and named
  as the policy with `driver.policy_model`.
- **The client**: the model-type config Gym talks to it with (`driver.policy_model_type`, the
  `--model-type` of `gym eval run`). `openai_model` is the default; `vllm_model` adds reasoning
  extraction, `chat_template_kwargs`, `extra_body` and token information.

## 1. Read the model card first

First list the benchmarks the config will run: several settings depend on them. Each setting has one
owner:

| Setting | Comes from | Goes to |
| --- | --- | --- |
| **From the model card** | | |
| Sampling (`temperature`, `top_p`, `top_k`, `min_p`, ...) | The card, else the checkpoint's `generation_config.json` | The server default for what a benchmark leaves unset: `--generation-config auto`, or `--override-generation-config` in `extra_args` where the card and the file disagree. Never the client's `sampling_overrides`; the benchmark's own values win (section 3). |
| Chat template | The checkpoint's own; another only when the card names it | `--chat-template <path>` in `extra_args`, only in that case (section 2). |
| Reasoning on or off, and the switch's name | The card, and the template (`chat_template.jinja`) | `chat_template_kwargs` on the client, unless the benchmark sets it (section 3). |
| Output budget | Never the card: the benchmark's | Nothing to set; record the card's value (section 3). |
| **From the model and the planned benchmarks** | | |
| Reasoning parser | The model family; needed when it reasons | `--reasoning-parser` in `extra_args`, `uses_reasoning_parser` on the client (sections 2, 3). |
| Tool-call parser | The template's tool-call format; needed when any planned benchmark uses tools: its dataset rows carry `responses_create_params.tools`, or its `benchmarks/<name>/` config runs a tool-using agent (a coding, terminal or search agent) | `--enable-auto-tool-choice --tool-call-parser <p>` in `extra_args`. Without it, do not run a tool-using benchmark: tool calls come back as plain text and score about zero (and `tool_choice: auto` without `--enable-auto-tool-choice` is rejected). The smoke checks the tool calls (section 6). |
| Context length | The model's limit (card or `config.json`) and the planned benchmarks' longest prompt plus output budget | `--max-model-len` in `extra_args`: at least what the benchmarks need, at most the model's limit. |
| Multimodal limits | The planned benchmarks' most images (or other media) per request | vLLM's `--limit-mm-per-prompt` in `extra_args` (check its syntax in the image's vLLM). A text-only model does not run a vision benchmark. |
| **From whoever deploys the model** | | |
| Parallelism (tensor, pipeline, expert), `number_of_instances`, `--max-num-seqs`, `--gpu-memory-utilization`, nodes and GPUs per node | Their `services.policy` block or the serving params of the request | Use them as given; ask when none are given. This skill does not choose them, and they do not change the outputs (section 8). |

Collect before writing anything: the context length, the recommended sampling settings, whether
the model reasons (and which vLLM reasoning parser its family uses), its tool-call format (and the
vLLM tool-call parser), whether its chat template has a reasoning switch and what it is called, and
whether its repository ships a `generation_config.json`.

Without a model card, read the checkpoint itself: `chat_template.jinja` (or `chat_template` in
`tokenizer_config.json`) for the reasoning switch and the tool-call format, `generation_config.json`
for sampling, and `config.json` or `tokenizer_config.json` for the context length. The parser names
come from the serving image: `vllm serve --help=all` lists the choices of `--tool-call-parser` and
`--reasoning-parser`. Pick the tool-call parser whose format the template emits, and say in your
notes which parser you chose and why.

If the benchmarks use tools, also check that the served template renders them. It needs a `tools` branch
(`grep -c tools chat_template.jinja` is above 0) that renders the request's `tools`, the assistant's tool calls and the
tool results. A template without one drops the tools without any error: the model answers in free text, the agent
sees no tool call, and every rollout ends in a format error and scores 0.0.

## 2. Write the policy service

Write everything site-specific (images, paths, account, partition, hosts) as `${oc.env:VAR}` and
export the variables before you submit. Fill the angle-bracket values first: a placeholder does not
validate. This is the smallest config that resolves and runs:

```yaml
services:
  policy:
    type: vllm
    container: ${oc.env:SERVING_IMAGE}
    model: /checkpoint # or <org>/<model> from the Hub
    served_model_name: <model name>
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    number_of_instances: 1 # data-parallel replicas
    trust_remote_code: false
    node_pool: gpu
    health_check:
      timeout_seconds: 1800 # the default 60 s is shorter than most models take to load
    extra_args: >-
      --max-model-len 65536
      --max-num-seqs 256
      --enable-auto-tool-choice --tool-call-parser <parser>
      --reasoning-parser <parser>
      --generation-config auto
    env:
      HF_HOME: lit:/cache/huggingface
      HF_TOKEN: runtime:HF_TOKEN
    mounts:
      - ${oc.env:CHECKPOINT_DIR}:/checkpoint:ro
      - ${oc.env:CACHE_DIR}:/cache

driver:
  container: ${oc.env:GYM_DRIVER_IMAGE} # Python 3.13.14 or newer; the default python:3.12 is older
  policy_model: policy
  policy_model_type: vllm_model
  gym_install:
    repo: https://github.com/NVIDIA-NeMo/gym
    ref: <full Gym commit>
  benchmarks:
    <benchmark>:
      run:
        config_paths: [benchmarks/<benchmark>/config.yaml]

job:
  output_path: ${oc.env:OUTPUT_DIR}

compute:
  cluster:
    type: slurm
    account: ${oc.env:SLURM_ACCOUNT}
    hostname: ${oc.env:SLURM_LOGIN_HOST} # null when you submit on the login node itself
    forward_env: [HF_TOKEN] # runtime: credentials, forwarded when submitting over SSH
    walltime: "04:00:00"
    node_pools:
      gpu:
        partition: ${oc.env:SLURM_PARTITION}
        nodes: 1
        gpus_per_node: 4

otel:
  enabled: false # on by default; it needs the collector in the job's environment
```

A deployment derived and validated elsewhere (for example by the team that quantized or serves the
checkpoint) comes as a `services.policy` block: the model, the container, the parallelism, `extra_args`,
`env`, `mounts` and `pre_command`. That block is the deployment spec. Use it as it is, add only your
site's values (the node pool and the cluster), and write the rest of the config around it (sections 3
and 4). Still read the model card (section 1) for the client settings and as a check: report a flag that
disagrees with the card to the block's owner instead of changing it.

`driver.gym_install.ref` is the Gym the job installs and scores the run with: the full 40-character
hash of a commit reachable from public Gym's `main` or a release tag, with your benchmark; never a
branch, and a tag resolved to its commit. Recovery (`gym eval resubmit --recover`) reads
the run with that commit. The CLI you submit with may be newer than the pinned commit; the job
still installs the pinned one. Install the CLI as run-and-babysit-eval (section 1) describes. Site `#SBATCH`
directives (a QoS, a node to avoid) go in `compute.cluster.extra_args`, without leading dashes:
`{qos: <qos>, exclude: <node>}`.

| What | Where |
| --- | --- |
| Model path or name | `model`. A local checkpoint also needs a `mounts` entry. |
| Served name | `served_model_name`; Gym passes it as `policy_model_name`. |
| Tensor, pipeline parallel | `tensor_parallel_size`, `pipeline_parallel_size` (one instance). |
| Data parallel | `number_of_instances`. Gym picks the vLLM data-parallel, multi-node or Ray Serve layout from it and the node count (see Multi-node vLLM). |
| Expert parallel | vLLM's own flag in `extra_args`, for example `--enable-expert-parallel`. |
| Max model length, max sequences | `--max-model-len`, `--max-num-seqs` in `extra_args`. |
| Tool-call and reasoning parsers | `--enable-auto-tool-choice --tool-call-parser <p>`, `--reasoning-parser <p>` in `extra_args`. |
| Chat template | The model's own by default; `--chat-template <path>` in `extra_args` to replace it. |
| Generation config | `--generation-config auto` takes the model's `generation_config.json` defaults for any sampling value a request leaves unset; `vllm` takes vLLM's own. |
| Environment | `env`, each value prefixed `lit:`, `host:VAR` or `runtime:VAR`; shell statements that `env` cannot express go in `pre_command`. |

`extra_args` is passed to `vllm serve` verbatim: check each flag against the vLLM version in the
image. Gym does not validate them.

Each instance uses `tensor_parallel_size × pipeline_parallel_size` GPUs, and all instances together
must fit the node pool (`nodes × gpus_per_node`). More instances raise throughput for many
rollouts; they do not change the outputs. A model with no reasoning output runs without
`--reasoning-parser` and sets `uses_reasoning_parser: false` on the client (section 3). Weight
quantization (`--quantization`, a quantized checkpoint), `--dtype` and `--kv-cache-dtype` change
the outputs and the score: they are part of the model config, not a memory setting (section 8).

Two readiness limits: the job waits `health_check.timeout_seconds` for the policy service, and
the driver then waits `server_spinup_timeout_seconds` (under the benchmark's `run`, 600 s by
default) for the Gym servers it starts for the benchmark. A benchmark whose servers start slowly
ends the chain typed `server_spinup_timeout`; run-and-babysit-eval says how to raise the limit on a saved run.

## 3. The reasoning toggle

A model whose chat template has a reasoning switch takes it through `chat_template_kwargs` on the
`vllm_model` client. The key name is model-specific; check the template.

```yaml
driver:
  benchmarks:
    <benchmark>:
      run:
        policy_model:
          responses_api_models:
            vllm_model:
              uses_reasoning_parser: true # true when vLLM runs a reasoning parser for this model
              chat_template_kwargs:
                enable_thinking: false
```

A benchmark's `run` mapping becomes `+key=value` overrides, applied after every config file, so
these values win over the benchmark's and the model type's own. The same keys work on the command
line: `+policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=false`.

- Set `uses_reasoning_parser` on the client to match the server: true when vLLM runs a reasoning
  parser for this model. The shipped `vllm_model` config sets it to true, so a model served without
  `--reasoning-parser` needs `false` here.
- A benchmark whose own config already declares the policy model fails validation with "at most 1
  item" when another model type is composed beside it: set `driver.policy_model_type: ""` for it.
- Some benchmark configs set `chat_template_kwargs` (or `extra_body`) themselves. That is a
  benchmark setting: leave it as the benchmark sets it, because an override would replace it. The
  shipped `vllm_model` configs leave these keys unset, so a benchmark's value survives in either
  composition order (section 4).
- A request's own `chat_template_kwargs` is used only with `forward_request_chat_template_kwargs:
  true` (see the vLLM model server page).

What the client sends, and what it leaves to the server:

- Gym's agents speak the Responses API; `vllm_model` turns each call into a chat-completions
  request (`is_responses_native: false` by default) and the tool calls back into Responses items,
  so the served template and tool-call parser decide whether tools work.
- Sampling (`temperature`, `top_p`, the output budget `max_output_tokens`) comes from the
  benchmark and its data. Any value they leave unset falls to the server: the model's
  `generation_config.json` with `--generation-config auto`, vLLM's defaults otherwise. A product
  deployment's sampling defaults are not evaluation settings. Leave the client's
  `sampling_overrides` unset for evaluation: it replaces the benchmark's sampling on every request.
- The output budget is the benchmark's. A model card's recommended output length, or a rule derived from
  it, is not an evaluation setting: record the card's value in your notes and keep the benchmark's budget,
  so runs of different models stay comparable. Answers the budget cuts off are the model's result under
  the benchmark's protocol. A benchmark that sets no budget leaves the output bounded by the window:
  record `--max-model-len` with the results.
- When the output budget alone exceeds `--max-model-len`, the client lowers it to fit the window
  (`refit_output_budget_to_window`, on by default) and records it in the model call's capture. A
  window that is too short therefore shows up as truncated answers, not as an error.
- System prompts come from the benchmark. A template that rejects the `developer` role needs
  `replace_developer_role_with_system: true` on the client; check the template before setting it.

## 4. Compose the evaluation config

Sections 2 and 3 write the model's half of the submit config. The other half, what is evaluated and
how it is graded, comes from Gym: every benchmark ships its own evaluation config, and the submit
config points at it instead of restating it.

| Part of the submit config | What it holds | Where it comes from |
| --- | --- | --- |
| `services.<name>` | The vLLM deployment of the model. | Section 2. |
| `driver.policy_model`, `driver.policy_model_type` | Which service is the policy, and the client Gym talks to it with. | Sections 2 and 3. Submit fills in `policy_base_url`, `policy_model_name` and `policy_api_key` in every benchmark's `run`. |
| `driver.gym_install.ref` | The Gym commit the job installs, which holds the benchmark and scores the run. | Section 2. |
| `driver.benchmarks.<benchmark>.prepare` | Overrides for `gym eval prepare`, which builds the benchmark's data in the job before it runs. | The benchmark's config path. |
| `driver.benchmarks.<benchmark>.run` | Overrides for `gym eval run`: the benchmark's config, the dataset type to run, and the model settings of section 3. | The benchmark's config path and its dataset's `type`; your section 3 keys. |
| `driver.benchmarks.<benchmark>.resumable` | Resuming across allocations (`max_retries`, `max_walltime`). | run-and-babysit-eval. |
| `job`, `compute`, `otel` | The output directory, the cluster and its node pools, telemetry. | Your site (section 2). |

### Where a benchmark's evaluation config lives

`gym list benchmarks` prints every selectable benchmark; the name it prints is the `--benchmark` value,
variant included (`ifeval`, `aalcr/config_v1_1`, `livecodebench/v6_2408_2505/config`). A directory that
only groups variants is not selectable itself. Each benchmark is a directory `benchmarks/<benchmark>/` in
the Gym commit that `driver.gym_install.ref` pins. Most hold one `config.yaml`. Some hold one config per
harness or setting, `benchmarks/<benchmark>/<flavor>.yaml`, which `gym eval run --benchmark
<benchmark>/<flavor>` loads, and a few hold variants in subdirectories,
`benchmarks/<benchmark>/<variant>/`. A variant directory often holds only a `manifest.yaml` whose
`config_path`, relative to it, names a YAML beside it: AA-LCR v1.1 is
`benchmarks/aalcr/config_v1_1/manifest.yaml` with `config_path: ../config_v1_1.yaml`, and `--benchmark
aalcr/config_v1_1` loads `benchmarks/aalcr/config_v1_1.yaml`. Name the YAML file itself, never the
directory. The benchmark's README names the config to use. Most directories also have these files:

- The config (`config.yaml` or a flavor file) is the evaluation config. Through its own `config_paths`
  it chains to its servers' configs (usually one resources server,
  `resources_servers/<server>/configs/<server>.yaml`, and for some a judge, sandbox or second server),
  and it declares the benchmark's agent, its dataset (`type: benchmark`, the prepared `jsonl_fpath`, the
  `prepare_script`, and a `prompt_config` where the benchmark has one) and the grading settings it
  changes on the server.
- `manifest.yaml` describes the benchmark: its servers, its datasets with their `num_repeats` (the
  config repeats it only where the benchmark sets it there), the reward range, and `canonical_split`,
  the split of the source data it measures. That is the source's name for it
  (`test`, `train`, ...), not Gym's `split`.
- The dataset's `prepare_script` (usually `prepare.py`; a variant may name its own, such as
  `prepare_v1_1.py`) builds the dataset file the config reads. Its output is not checked in, so a
  benchmark whose dataset has a `prepare_script` needs a `prepare` block, or the job finds no data. A
  benchmark without one ships its data or builds its tasks another way; its README says how.
- `README.md` names the source data, how it is graded, and the benchmark's own commands.

Read them at the pinned commit, not on a newer branch: the benchmark's prompts, grading and prepare
code are part of what the run measures. The commit pins the prepare code, not data the prepare script
downloads: check the URL in the prepare script, and compare runs' materialized inputs.

The key under `driver.benchmarks` is the job's label for the benchmark, not the `--benchmark` selector,
and has no slash: for `aalcr/config_v1_1` use a label such as `aalcr_v1_1`, with
`benchmarks/aalcr/config_v1_1.yaml` in both `config_paths` lists.

Do not copy the benchmark's config into the submit config; name it. Put the file that
`gym eval run --benchmark <value>` resolves (`benchmarks/<benchmark>/config.yaml`, or the flavor or
variant file) in `run.config_paths`, put the same path under `prepare.config_paths` to prepare its data,
and set `run.split` to the `type` of the dataset the config declares, `benchmark` for an evaluation. Do
not copy `canonical_split` into `split`: `gym eval run` accepts only `train`, `validation` and
`benchmark` there. `gym env validate --benchmark <value>` (section 5) exits 2 for a name that does
not resolve; `--resolve-only` does not open the files in `run.config_paths`, so it passes either way.
A submitted job loads the model type's config, then the benchmark's config, so the benchmark wins a
key both set. `gym eval run --benchmark <value> --model-type <type>` loads them the other way round,
so there the model type wins: the shipped `vllm_model` sets `uses_reasoning_parser: true`, which then
replaces a benchmark's `false`. The `run` keys come last, as `+key=value` overrides, and win over both
files. So the model's settings from section 3 go in the same `run` mapping as the benchmark keys,
while the benchmark's own settings (prompts, sampling, `num_repeats`, grading) stay as Gym ships them.

Override scalars only, and record each as part of the run; section 8 says which a fair comparison keeps.
Of the benchmark's own settings only `num_repeats` may change, and only upward: more repeats measure the
same protocol with a smaller standard error, while fewer repeats or a `limit` make the run a smoke
(run-and-babysit-eval, section 6). A `run` key replaces a list as a whole: a key under a
benchmark's `datasets` drops the dataset's `jsonl_fpath`, `type` and `prepare_script`. Never override a
dataset list or a dataset's paths, and keep `split` and `config_paths` as this section sets them.

### Benchmarks that need more than the model

A benchmark graded by a judge model, one that searches the web, and one whose agent runs in a sandbox
need services and credentials of their own. The benchmark's config chains their configs itself (for
example `benchmarks/simpleqa/` chains a judge config, `benchmarks/arxivmath_0526/` another,
`benchmarks/swe_atlas_qa/` a sandbox provider, and `benchmarks/gdpval/` a sandbox provider, judges and a
search tool), so naming the benchmark's file is still enough. What the submit config must add is the
environment and the config values they read:

- Read the benchmark's README, and every `${oc.env:NAME}` and bare `${name}` in its config and the
  files it chains to.
- A bare `${name}` is a config value the benchmark expects from you, which `driver.env` does not set:
  `benchmarks/swe_atlas_qa/` needs `judge_base_url`, `judge_model_name` and `harness_bundle`, as its
  README lists. Set each as a key under both `prepare` and `run`, since both load the config.
- Give each `${oc.env:NAME}` a value in the job: `driver.env` as `host:NAME` (read from your shell at submit) or
  `runtime:NAME` together with `compute.cluster.forward_env`, as `HF_TOKEN` in section 2. A credential
  never goes in as a literal.
- Do not rely on a default. Several judge keys default to `dummy`: the run starts, the judge calls fail
  or grade nothing useful, and only the run's health shows it.

### Before you write values

Ask for what you do not know instead of inventing it: the cluster, its login host, the account and
partition, the serving and driver images, the checkpoint path and the output directory. Write each as
`${oc.env:VAR}` (section 2) and name the variables to export.

### Example: IFEval

`benchmarks/ifeval/` needs no judge and no credentials. Its `config.yaml` chains to
`resources_servers/instruction_following/configs/instruction_following.yaml`, and its `prepare.py`
downloads the prompts from the upstream default branch, so the Gym commit does not pin the data: compare
the runs' materialized inputs when you compare two runs. Its block is:

```yaml
driver:
  benchmarks:
    ifeval:
      prepare:
        config_paths: [benchmarks/ifeval/config.yaml]
      run:
        config_paths: [benchmarks/ifeval/config.yaml]
        split: benchmark
      resumable:
        max_retries: 3
        max_walltime: "08:00:00"
```

The rest is section 2. A policy that reasons by default and is served with `--reasoning-parser` (Qwen3,
for example) needs no section 3 keys, because the shipped `vllm_model` client already expects the parser;
a model that needs them adds them to the same `run` mapping. In the `--dry-run` script the job runs
`gym eval prepare '+config_paths=[benchmarks/ifeval/config.yaml]'`, then `gym eval run --model-type
vllm_model` with the same `+config_paths`, `+split=benchmark` and the three policy keys submit filled in.
Validate it (section 5) and smoke it (section 6), then hand the file to run-and-babysit-eval.

## 5. Validate without GPUs

```bash
# The submit config: schema, policy wiring, services and compute.
gym eval submit --config submit.yaml --resolve-only

# The job scripts, including the exact `vllm serve` command line.
gym eval submit --config submit.yaml --dry-run

# Each Gym server block against its server's config class, composed as `gym eval run` composes it.
gym env validate --server-types responses_api_models,responses_api_agents \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --config benchmarks/<benchmark>/config.yaml --allow-job-time-values
```

The two `--config` files are in the submitted job's order (section 4), model type first;
`--benchmark` with `--model-type` loads them the other way round.

`--allow-job-time-values` stands in for values that only exist at job time, such as the policy URL
that submit injects. `gym env validate` exits 0 with no invalid finding, 1 with one, and 2 when the
config does not compose; success prints nothing. It checks the benchmark's and the model type's own
server configs, not your submit config: the `run` overrides of sections 2 and 3 appear only in the
`--dry-run` job script, as `+key=value` tokens. Read them there. A key rendered twice (a dotted key
beside a nested block at the same path) fails the job on the cluster, after the GPUs are allocated.

## 6. Smoke it

Submit the model with one benchmark and a small task count before a full evaluation
(`gym eval submit --config submit.yaml`; it prints the run's `GYM_JOB_ID`):

```yaml
driver:
  benchmarks:
    <benchmark>:
      run:
        limit: 5
        num_repeats: 1
```

`limit` counts tasks; the run log prints the tasks and the rows they become. It takes the first tasks of
the input, which can all come from one part of the benchmark, so a smoke's score is never reported as the
benchmark's. A score on part of a benchmark needs a named, versioned subset (a list of task ids, or a
variant the benchmark ships) and is reported under that name; with no such subset, report no benchmark
score. Bound the smoke to
one allocation with the walltime and `resumable: {max_retries: 0, max_walltime: <the walltime>}`;
when it completes, it cancels the chain's pending follow-up and sentinel itself (see run-and-babysit-eval, section 2).
Then follow the [run-and-babysit-eval](../SKILL.md) skill: `gym eval resubmit "$GYM_JOB_ID"
<benchmark> --recover --inspect` and `gym eval health-check` (run-and-babysit-eval, section 6). Look for answers that end mid-thought (the output token
budget or `--max-model-len` is too small), reasoning that lands in the answer (the reasoning parser
or `uses_reasoning_parser` is wrong), and tool calls that arrive as plain text (the tool-call
parser is wrong). For a tool-using benchmark also count the tool calls: `ng_perf.num_tool_calls` in `rollouts.jsonl`
must be above 0 on the rows that need tools, and a request's `prompt_tokens` must exceed the token count of its
messages alone (equal means the template dropped the tools). In the inspection's output,
`gate_task.evaluation_health.model.kinds` and
`gate_task.failures.response_failures.by_cause` (`truncated`, `empty`, `malformed`) are the model's
own faults: they count in the score, never fail health, and point at the deployment (output
budget, `--max-model-len`, parsers). Answers that never stop and run to the output budget point
at an end-of-turn token that the template and the model's generation config disagree on.
`UNVERIFIED` with `trial_counts.unobserved_reason` naming the benchmark's server is about that server's grading record, not about your deployment: report it and
change nothing for it.

## 7. Read the resolved config

`gym eval submit --resolve-only` prints the config as it will run; after submission, the run
directory's `resolved-config.yaml` is the record. Diff two of them to compare two runs: the config
file may have changed since, and overrides mean nothing without it.

## 8. A fair comparison

| Keep identical between the models you compare | May change with the deployment |
| --- | --- |
| The benchmark config: data, split, prompts, task limits. | Tensor, pipeline, data and expert parallelism; `number_of_instances`. |
| Sampling settings the benchmark sets, and `num_repeats`. | `--max-num-seqs`, `--gpu-memory-utilization`, client concurrency. |
| Weight quantization, `--dtype` and `--kv-cache-dtype` (or record each difference as part of the model; a quantized export: section 9). | |
| The judge model and the grading config. | Ports, container image, node layout. |
| The benchmark's token budgets and timeouts. | `--max-model-len`, as long as it holds the benchmark's longest prompt plus its output budget. |
| A reasoning switch the benchmark sets, and the Gym commit that scores the run. | |

A deployment that is too slow shows up as timeouts and truncated answers, which count as model
failures under the benchmark's rules. Make the deployment faster; do not raise the evaluation's
limits.

A model's own chat template, parsers and generation-config defaults are part of the model config,
not the deployment: record them with the results, because they change the outputs. Gym's resume
check treats the same split: a setting named as operational (timeouts, retries, ports,
concurrency, paths, tracking) resumes with a warning, and any other changed setting refuses the
resume and is named.

## 9. A quantized variant

A quantized export (NVFP4, FP8 or another format) of a model you also evaluate in BF16 is served with the
BF16 config, except for the fields quantization requires. Anything else that moves makes the comparison
with BF16 invalid. The quantization settings belong to the export: they come from its `config.json`, its
`hf_quant_config.json` and the serving recipe that ships with it (a `vllm serve` command or its release
notes), and whoever produced the export owns them. A deployment fix goes into the BF16 config, so both
runs get it; never only into the quantized one.

1. Check that the export came from the BF16 checkpoint before you write anything: `config.json` matches
   the BF16 one outside `quantization_config`, and a few tensors the export keeps unquantized
   (`exclude_modules` in `hf_quant_config.json`, such as the embeddings or `lm_head`) are byte-identical
   to the BF16 checkpoint's. Also compare `generation_config.json`, including whether it exists: with
   `--generation-config auto` its values are the defaults for every sampling setting a request leaves unset,
   so a different file changes the sampling too. If the two differ, stop and ask the export's owner to
   align them; do not override the benchmark's sampling to compensate. Record what you compared.
2. Copy the BF16 config and change only these fields, each with a comment naming its source:
   - `model` (and its mount): the export.
   - `served_model_name`: the model's name without its precision suffix plus the export's actual
     precision, such as `<model>-nvfp4` or `<model>-fp8`, never `<model>-bf16-nvfp4`, so the two runs never
     merge.
   - The KV-cache and state-cache dtypes (`--kv-cache-dtype`, a hybrid model's Mamba cache dtypes), only
     where the export's serving recipe sets them. With no serving recipe, add none and ask the export's
     owner for it. Never copy cache flags from another model's config: an uncited flag is a guess.
3. Keep everything else as in BF16: the benchmark settings (section 8, left column), the parallelism, the
   parsers, the chat template, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, the image,
   `env`, the client and `driver.gym_install.ref`. Do not add `--quantization`: vLLM detects a quantized
   export. If the export ships its own chat template, diff it against the BF16 one, serve the BF16 one and
   report the difference to the export's owner. A slower load raises `health_check.timeout_seconds` in the
   BF16 config, so both runs keep one limit.
4. Resolve both (`gym eval submit --config <file> --resolve-only`) and diff them. Only these may differ:
   the fields of step 2, each benchmark's `run.policy_model_name` (submit derives it from the served name),
   and the MLflow fields of steps 5 and 6 (the precision tag, the `quantization_run_*` tags and the experiment name).
   The `pair_id` and every other serving, benchmark and scoring setting stay identical. Run both at the same
   Gym commit with the same `num_repeats`, and keep the policy timeout on the model (run-and-babysit-eval,
   section 6). Repeat the diff after any change to the BF16 config.
5. Tag both runs so they pair in MLflow: `mlflow_run_tags` with the precision and an id the two runs share,
   for example `{precision: nvfp4, pair_id: <id>}`. The served name is not the pairing tag. Put the tags
   where the BF16 config already tracks the run: under `driver.benchmarks.<name>.run` for live tracking, or
   in the MLflow export it runs after the rollouts (`job.export.mlflow`, or a benchmark's
   `driver.benchmarks.<name>.export.mlflow`). The keys are the same in both places. Merge them into the
   existing tags and do not add a second exporter. Tags alone do not turn tracking on (it needs the tracking
   URI, experiment and run name): if the BF16 config does not track, say that the pair is not recorded in
   MLflow.
6. Carry the export's provenance. When the quantization run was tracked (an MLflow run, for example) and
   the requester gives its name, id or URL, add them as the tags `quantization_run_name`,
   `quantization_run_id` and `quantization_run_url`, in the same place as step 5, and use the experiment
   the requester names as `mlflow_experiment_name`. The evaluation run stays on your own tracking server and
   keeps its own run name; the tags only link it to the quantization run. Do not copy that run's tracking URI
   or experiment id. With no such run, there is nothing to carry.

The smoke (section 6) of the quantized config reads the same fields as the BF16 one: a difference in the
model's own faults (`evaluation_health.model.kinds`, `failures.response_failures.by_cause`) between the two
smokes belongs to the export or its serving recipe, and goes to the export's owner.

## What this skill does not do

- It does not submit, watch, resume or recover a run, or judge its health. That is
  [run-and-babysit-eval](../SKILL.md), which takes the config this skill writes.
- It does not add a benchmark to Gym or change one; that is add-benchmark. A benchmark that has no
  directory under `benchmarks/` at the pinned commit cannot be composed here.

## Exit checks

- [ ] Each benchmark's `run.config_paths` and `prepare.config_paths` name the config file
  `--benchmark` resolves at the pinned commit, and `run.split` is its dataset's `type` (`benchmark`).
- [ ] `--resolve-only` and `gym env validate --server-types` pass, and the `--dry-run` job script
  shows each `run` override once, the partition and `--gpus-per-node`.
- [ ] The smoke run completes, and its health and answers look right.
- [ ] The resolved config shows the intended served name, parallelism, parsers and reasoning switch.
- [ ] Nothing in the "keep identical" column differs from the run you compare against.
- [ ] A quantized variant's resolved config differs from the BF16 one only where section 9, step 4
  allows, and both runs carry the pairing tags on the MLflow route they use (and the `quantization_run_*`
  tags when the quantization run was tracked).
