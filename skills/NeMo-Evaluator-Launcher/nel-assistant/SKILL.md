---
name: nel-assistant
description: Interactive config wizard for NeMo Evaluator Launcher (NEL). Use when the user wants to create a new evaluation config from scratch, set up an evaluation from existing configs, or modify a NEL config (deployment, tasks, multi-node, interceptors). ALWAYS triggers on mentions of creating configs, setting up evaluations, configuring models for evaluation, or modifying NEL YAML files. Do NOT use for monitoring, debugging, or analyzing already-running evaluations.
license: Apache-2.0
---

## NeMo Evaluator Launcher Assistant

You're an expert in NeMo Evaluator Launcher! Guide the user through creating production-ready YAML configurations, running evaluations, and monitoring progress via an interactive workflow specified below.

### Workflow

```
Config Generation Progress:
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```

**Step 1: Check if nel is installed**

Test that `nel` is installed with `nel --version`.

If not, instruct the user to `pip install nemo-evaluator-launcher`.

**Step 2: Build the base config file**

Prompt the user with "I'll ask you 5 questions to build the base config we'll adjust in the next steps". Guide the user through the 5 questions using AskUserQuestion:

1. Execution:
  - Local
  - SLURM
2. Deployment:
  - None (External)
  - vLLM
  - SGLang
  - NIM
  - TRT-LLM
3. Auto-export:
  - None (auto-export disabled)
  - MLflow
  - wandb
4. Model type
  - Base
  - Chat or Reasoning
5. Benchmarks:
  Allow for multiple choices in this question.
  - If Model type = Base:
    1. General Knowledge
    2. Coding
    3. Long Context
    4. Multilingual
  - If Model type = Chat or Reasoning:
    1. Core Reasoning
    2. Agentic
    3. Long Context
    4. Multilingual

DON'T ALLOW FOR ANY OTHER OPTIONS, only the ones listed above under each category (Execution, Deployment, Auto-export, Model type, Benchmarks). YOU HAVE TO GATHER THE ANSWERS for the 5 questions before you can build the base config.

When you have all the answers, run the script to build the base config:

```bash
nel skills build-config --execution <local|slurm> --deployment <none|vllm|sglang|nim|trtllm> --model_type <base|chat_reasoning> --benchmarks <general_knowledge|coding|core_reasoning|agentic|long_context|multilingual> [--export <none|mlflow|wandb>] [--output <OUTPUT>]
```

Where `--output` depends on what the user provides:

- Omit: Uses current directory with auto-generated filename
- Directory: Writes to that directory with auto-generated filename
- File path (*.yaml): Writes to that specific file

It never overwrites existing files.

**Step 3: Configure model path and parameters**

Ask for model path. Determine type:

- Checkpoint path (starts with `/` or `./`) → set `deployment.checkpoint_path: <path>` and `deployment.hf_model_handle: null`
- HF handle (e.g., `org/model-name`) → set `deployment.hf_model_handle: <handle>` and `deployment.checkpoint_path: null`

Use WebSearch to find model card (HuggingFace, build.nvidia.com). Read it carefully, the FULL text, the devil is in the details. Extract ALL relevant configurations:

- Sampling params (`temperature`, `top_p`)
- Context length (`deployment.extra_args: "--max-model-len <value>"`)
- TP/DP settings (to set them appropriately, AskUserQuestion on how many GPUs the model will be deployed)
- Reasoning config (if applicable):
  - reasoning on/off: use either:
    - `adapter_config.custom_system_prompt` (like `/think`, `/no_think`) and no `adapter_config.params_to_add` (leave `params_to_add` unrelated to reasoning untouched)
    - `adapter_config.params_to_add` for payload modifier (like `"chat_template_kwargs": {"enable_thinking": true/false}`) and no `adapter_config.custom_system_prompt` and `adapter_config.use_system_prompt: false` (leave `custom_system_prompt` and `use_system_prompt` unrelated to reasoning untouched).
  - If a task override contains `{"chat_template_kwargs": {"enable_thinking": false}, "skip_special_tokens": false}`, replace it with the model-specific payload from the model card that disables reasoning.
  - For pure-chat models, remove `adapter_config.params_to_add` completely if the model card does not define a reasoning toggle.
  - reasoning effort (if it's configurable, AskUserQuestion what reasoning effort they want)
  - higher `max_new_tokens`
  - etc.
- Deployment-specific `extra_args` for vLLM/SGLang (look for the vLLM/SGLang deployment command)
- Deployment-specific vLLM/SGLang versions (by default we use latest docker images, but you can control it with `deployment.image` e.g. vLLM above `vllm/vllm-openai:v0.11.0` stopped supporting `rope-scaling` arg used by Qwen models)
- ARM64 / non-standard GPU compatibility: The default `vllm/vllm-openai` image only supports common GPU architectures. For ARM64 platforms or GPUs with non-standard compute capabilities (e.g., NVIDIA GB10 with sm_121), use NGC vLLM images instead:
  - Example: `deployment.image: nvcr.io/nvidia/vllm:26.01-py3`
  - AskUserQuestion about their GPU architecture if the model card doesn't specify deployment constraints
- Tool-calling requirements:
  - If the selected benchmarks include `agentic`, you MUST configure tool calling end-to-end.
  - For self-deployment, extract the exact tool-calling flags/settings from the model card (for example vLLM/SGLang tool parser flags) and apply them.
  - For external endpoints, confirm the endpoint already supports tool calling before proceeding.
- Any preparation requirements (e.g., downloading reasoning parsers, custom plugins):
  - If the model card requires downloading files or running setup steps before deployment or evaluation, use `deployment.pre_cmd` or `evaluation.pre_cmd` for non-local execution.
  - In `pre_cmd` script:
    - Use `curl` instead of `wget` as it's more widely available in Docker containers. Example: `pre_cmd: curl -L -o reasoning_parser.py https://huggingface.co/.../reasoning_parser.py`
    - Always use `--no-cache-dir` when installing Python packages to avoid cross-device link errors in Docker containers (the pip cache and temp directories may be on different filesystems). Example: `pre_cmd: pip3 install --no-cache-dir flash-attn --no-build-isolation`
  - For local execution, do NOT rely on `pre_cmd`. Run the preparation steps yourself on the host first, then mount the resulting files/directories into the container if needed.
  - Short mount examples:
    - deployment: `execution.mounts.deployment: {"/absolute/path/to/reasoning_parser.py": "/vllm-workspace/reasoning_parser.py"}`
    - evaluation: `execution.mounts.evaluation: {"/absolute/path/to/hf_cache": "/root/.cache/huggingface"}`
- Env vars:
  - Use `deployment.env_vars` for deployment-side settings, `evaluation.env_vars` for evaluation-wide settings, and `evaluation.tasks[].env_vars` for task-specific overrides.
  - Supported value types: `host:VAR_NAME` = read the value from the host env var `VAR_NAME`; `lit:value` = use the literal value directly; `runtime:VAR_NAME` = resolve `VAR_NAME` only at runtime inside the execution environment.
- Any other model-specific requirements

Remember to check `evaluation.nemo_evaluator_config` and `evaluation.tasks.*.nemo_evaluator_config` overrides too for parameters to adjust (e.g. disabling reasoning)!

Present findings, explain each setting, ask user to confirm or adjust. If no model card found, ask user directly for the above configurations.

**Step 4: Fill in remaining missing values**

- Find all remaining `???` missing values in the config.
- Ask the user only for values that couldn't be auto-discovered from the model card (e.g., SLURM hostname, account, output directory, MLflow/wandb tracking URI). Don't propose any defaults here. Let the user give you the values in plain text.
- Ask the user if they want to change any other defaults e.g. execution partition or walltime (if running on SLURM) or add MLflow/wandb tags (if auto-export enabled).

**Step 5: Confirm tasks (iterative)**

Show tasks in the current config. Loop until the user confirms the task list is final:

1. Tell the user: "Run `nel ls tasks` to see all available tasks".
2. Ask if they want to add/remove tasks or add/remove/modify task-specific parameter overrides.
   To add per-task `nemo_evaluator_config` as specified by the user, e.g.:
   ```yaml
   tasks:
     - name: <task>
       nemo_evaluator_config:
         config:
           params:
             temperature: <value>
             max_new_tokens: <value>
             ...
   ```
3. Apply changes.
4. Show updated list and ask: "Is the task list final, or do you want to make more changes?"

**Step 6: Advanced - Multi-node**

There are two multi-node patterns. Ask the user which applies:

**Pattern A: Multi-instance (independent instances with HAProxy)**

Only if model >120B parameters or user wants more throughput. Explain: "Each node runs an independent deployment instance. HAProxy load-balances requests across all instances."

```yaml
execution:
    num_nodes: 4       # Total nodes
    num_instances: 4   # 4 independent instances → HAProxy auto-enabled
```

**Pattern B: Multi-node single instance (Ray TP/PP across nodes)**

When a single model is too large for one node and needs pipeline parallelism across nodes. Use `vllm_ray` deployment config:

```yaml
defaults:
  - deployment: vllm_ray   # Built-in Ray cluster setup (replaces manual pre_cmd)

execution:
    num_nodes: 2           # Single instance spanning 2 nodes

deployment:
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
```

**Pattern A+B combined: Multi-instance with multi-node instances**

For very large models needing both cross-node parallelism AND multiple instances:

```yaml
defaults:
  - deployment: vllm_ray

execution:
    num_nodes: 4       # Total nodes
    num_instances: 2   # 2 instances of 2 nodes each → HAProxy auto-enabled

deployment:
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
```

**Common Confusions**

- **`num_instances`** controls independent deployment instances with HAProxy. **`data_parallel_size`** controls DP replicas *within* a single instance.
- Global data parallelism is `num_instances x data_parallel_size` (e.g., 2 instances x 8 DP each = 16 replicas).
- With multi-instance, `parallelism` in task config is the total concurrent requests across all instances, not per-instance.
- `num_nodes` must be divisible by `num_instances`.

**Step 7: Advanced - Interceptors**

- Tell the user they should see: https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html .
- DON'T provide any general information about what interceptors typically do in API frameworks without reading the docs. If the user asks about interceptors, only then read the webpage to provide precise information.
- If the user asks you to configure some interceptor, then read the webpage of this interceptor and configure it according to the `--overrides` syntax but put the values in the YAML config under `evaluation.nemo_evaluator_config.config.target.api_endpoint.adapter_config` (NOT under `target.api_endpoint.adapter_config`) instead of using CLI overrides.
  By defining `interceptors` list you'd override the full chain of interceptors which can have unintended consequences like disabling default interceptors. That's why use the fields specified in the `CLI Configuration` section after the `--overrides` keyword to configure interceptors in the YAML config.

**Documentation Errata**

- The docs may show incorrect parameter names for logging. Use `max_logged_requests` and `max_logged_responses` (NOT `max_saved_*` or `max_*`).

**Step 8: Run the evaluation**

Print the following commands to the user. Propose to execute them in order to confirm the config works as expected before the full run.

**Important**: Ensure required environment variables are available. Ask the user to provide `HF_TOKEN`, even if they are not using a gated model (like Llama) or dataset (like GPQA), to reduce Hugging Face rate limiting errors. Remind the user to get access to GPQA, if it's in the config ("Please, click request access for GPQA-Diamond: https://huggingface.co/datasets/Idavidrein/gpqa"), and ask them to put missing tokens or keys (e.g. `HF_TOKEN`, `NVIDIA_API_KEY`, `api_key_name` from the config) in a `.env` file in the project root. NEL automatically reads `.env` — no need to source it manually.

```bash
# If using pre_cmd or post_cmd:
export NEMO_EVALUATOR_TRUST_PRE_CMD=1
```

1. **Dry-run** (validates config without running):
   ```
   nel run --config <config_path> --dry-run
   ```

2. **Test with limited samples** (quick validation run):
   ```
   nel run --config <config_path> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10
   ```

3. **Re-run a single task** (useful for debugging or re-testing after config changes):
   ```
   nel run --config <config_path> -t <task_name>
   ```
   Combine with `-o` for limited samples: `nel run --config <config_path> -t <task_name> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10`

4. **Full evaluation** (production run):
   ```
   nel run --config <config_path>
   ```

After the dry-run, check the output from `nel` for any problems with the config. If there are no problems, propose to first execute the test run with limited samples and then execute the full evaluation. If there are problems, resolve them before executing the full evaluation.

**Monitoring Progress**

After job submission, you can monitor progress using:

1. **Check job status:**
   ```bash
   nel status <invocation_id>
   nel info <invocation_id>
   ```

2. **Stream logs** (Local execution only):
   ```bash
   nel logs <invocation_id>
   ```
   Note: `nel logs` is not supported for SLURM execution.

3. **Inspect logs via SSH** (SLURM workaround):

   When `nel logs` is unavailable (SLURM), use SSH to inspect logs directly:

   First, get log locations:
   ```bash
   nel info <invocation_id> --logs
   ```

   Then, use SSH to view logs:

   **Check server deployment logs:**
   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/server-<slurm_job_id>-*.log"
   ```
   Shows vLLM server startup, model loading, and deployment errors (e.g., missing wget/curl).

   **Check evaluation client logs:**
   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/client-<slurm_job_id>.log"
   ```
   Shows evaluation progress, task execution, and results.

   **Check SLURM scheduler logs:**
   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/slurm-<slurm_job_id>.log"
   ```
   Shows job scheduling, health checks, and overall execution flow.

   **Search for errors:**
   ```bash
   ssh <username>@<hostname> "grep -i 'error\|warning\|failed' <log path from `nel info <invocation_id> --logs`>/*.log"
   ```

---

**Advanced workflow**: For more detailed run monitoring, debugging failed evaluations, and post-run analysis, see the `launching-evals` skill.

---

Direct users with issues to:

- **GitHub Issues:** https://github.com/NVIDIA-NeMo/Evaluator/issues
- **GitHub Discussions:** https://github.com/NVIDIA-NeMo/Evaluator/discussions

Now, copy this checklist and track your progress:

```
Config Generation Progress:
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```
