# PyTorch Recipe Selection

PyTorch identifies the training framework; it does not determine the federated
workflow. Choose the recipe from the user's FL intent and aggregation,
state-exchange, privacy, and site-role requirements.

## Recipe Discovery, Algorithms, And Selection

Recipe discovery, the algorithm guide (FedAvg, FedProx, FedOpt, SCAFFOLD,
Cyclic, Swarm, FedEval), catalog-based selection rules, the HE-not-supported
rule, and non-FedAvg recipe rules are shared across the PyTorch recipe family.
Follow
`../../nvflare-shared/references/pytorch-family-recipe-selection.md` for all of
them before constructing `job.py`. This file covers only the plain-PyTorch
`job.py` construction details.

## Standard FedAvg Fast Path

For a normal PyTorch-to-FedAvg conversion, keep the `job.py` recipe construction
small and portable. Run `recipe show fedavg-pt --format json` first and follow
`../../nvflare-shared/references/pytorch-family-recipe-construction.md`. The
example below contains only the workflow-specific constructor values; add
optional recipe keywords and decomposers only as directed by that capability
profile:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv

model_args = {"input_size": input_size, "num_classes": num_classes}
recipe_model = {"class_path": "model.ModelClass", "args": model_args}

recipe_kwargs = dict(
    name=job_name,
    min_clients=num_clients,
    num_rounds=num_rounds,
    model=recipe_model,
    train_script="client.py",
    train_args=train_args,
)
# Add only capability-gated keywords confirmed by recipe show, including
# key_metric when the selected execution path delivers that metric to the server.
recipe = FedAvgRecipe(**recipe_kwargs)
# Apply per-site configuration, then capability-gated decomposers, if required.

env = SimEnv(num_clients=num_clients, workspace_root=workspace_root)
run = recipe.execute(env)
```

Use a recipe model dict with the same constructor arguments used by the
client-side model whenever reconstruction depends on any constructor value:

```python
model={"class_path": "model.ModelClass", "args": model_args}
```

Prefer `class_path` at recipe construction time; `path` is the normalized key
used in exported job config, and recipes accept it as an alias.

The maintained `hello-pt` example uses the in-process recipe defaults. The
conversion profile may intentionally differ by selecting tensor-native
transport and, independently, enabling the server-side disk-offload
optimization when the selected recipe exposes those capabilities. The shared
construction reference is the single source of truth for their prerequisites,
ordering, and decomposer-registration rules.

The server-side recipe model and the client-side training model must construct
the same architecture. If the model constructor needs dimensions, class counts,
dropout settings, embedding sizes, or other architecture arguments, pass the
same values on both sides. Prefer a small shared constant, JSON/config file, or
explicit `train_args` values over hard-coded divergent defaults.

Follow the shared construction reference for `key_metric` capability, exact
client-metric matching, and lower-is-better metric direction.

Use these portable imports when writing custom Job API code:

```python
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.sim_env import SimEnv
```

Do not inspect large NVFLARE modules to recover these imports unless validation
shows that the installed version differs.

Do not infer that `per_site_config` is required only because recipe metadata
mentions it. For standard FedAvg with the same `client.py`, `model.py`, and
training arguments on all clients, leave `per_site_config` unset and let the
recipe deploy the executor to all clients. Use `per_site_config` only when at
least one site needs a different `train_script`, `train_args`, command,
external-process setting, framework/exchange setting, or launch behavior.

For non-FedAvg workflows, use the matching recipe from the catalog (see the
shared reference above) and keep the PyTorch Client API exchange aligned with
that recipe's expected task names, metadata, and parameter format.

## Recipe Capabilities And Execution Mode

Follow
`../../nvflare-shared/references/pytorch-family-recipe-construction.md` for
every selected recipe. It owns the capability checks for tensor settings,
decomposer registration, best-model naming, and process-based execution mode;
do not copy the FedAvg constructor above to a non-FedAvg recipe.

## Export Behavior

Export handling is shared across algorithms and frameworks. Follow
`../../nvflare-shared/references/conversion-workflow.md` for `--export`, `--export-dir`, and
local command-line parser behavior.
