# PyTorch-Family Recipe Selection

This reference is shared by every PyTorch-family conversion skill: plain
PyTorch, PyTorch Lightning, and Hugging Face Trainer. Lightning and Hugging Face
Trainer are PyTorch-family training frameworks, not separate recipe families, so
they select from the same catalog. The training framework identifies the
client-side conversion; it does not determine the federated workflow. Choose the
recipe from the user's FL intent and aggregation, state-exchange, privacy, and
site-role requirements.

## Discover Recipes

Run the local recipe catalog before creating or updating `job.py`:

```bash
nvflare recipe list --framework pytorch --format json
```

Use the returned recipe metadata as the source of truth for recipe names,
modules, classes, algorithms, aggregation mode, state exchange, privacy and
privacy-compatibility metadata, and optional dependencies.

After selecting a candidate recipe, inspect its parameters:

```bash
nvflare recipe show <recipe-name> --format json
```

Prefer recipe CLI metadata before Python introspection. If Python introspection
is still needed, do not guess top-level exports: the base `Recipe` class lives in
`nvflare.recipe.spec`, not `nvflare.recipe`.

When writing `job.py`, import the recipe class from the exact `module` and
`class` returned by `recipe show`; do not substitute a package-level or
top-level import. For `fedavg-pt`, import
`nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe`, not
`nvflare.recipe.FedAvgRecipe`.

## Quick Algorithm Guide

If the user does not know which FL algorithm they need, explain the choices in
plain language before editing `job.py`:

- FedAvg: the default starting point for most horizontal FL jobs. Each site
  trains locally, sends model weights or updates, and the server averages them.
  Use this when the user simply asks to federate normal training.
- FedAvg with HE: homomorphic encryption for protected aggregation is NOT
  supported by the conversion skills — it needs a provisioned deployment
  environment beyond conversion scope (see the HE rule in Selection Rules). If
  the user asks for HE or encrypted aggregation, report it as unsupported and
  route to provisioning/deployment.
- FedProx: FedAvg-style training with a proximal term in the client loss to
  improve stability when site data or compute behavior is very different.
- FedOpt: server-side optimizer variants such as FedAdam, FedYogi, or FedAdagrad.
  Use when the user wants server optimizer control or better behavior than plain
  averaging on heterogeneous data.
- SCAFFOLD: adds control variates to reduce client drift on non-IID data. Use
  when the user specifically asks for SCAFFOLD or drift mitigation.
- Cyclic: sends the model through clients sequentially instead of aggregating
  updates on the server. Use when the requested workflow is client-to-client or
  cyclic weight transfer.
- Swarm Learning: peer/client-parent aggregation topology instead of a normal
  server-centered FedAvg topology. Use when the user asks for swarm learning.
- FedEval: evaluation-only. Use when the user wants to distribute a checkpoint
  to sites and collect metrics without federated training updates.

For deeper background, see the algorithm papers for
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[FedProx](https://arxiv.org/abs/1812.06127),
[FedOpt](https://openreview.net/forum?id=LkFG3lB13U5),
[SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html), and
[Swarm Learning](https://www.nature.com/articles/s41586-021-03583-3). For
Cyclic recipes, use the local catalog and
`nvflare recipe show <selected-cyclic-recipe> --format json`.

## Selection Rules

- Use `fedavg-pt` for standard horizontal federated training where clients train
  the same model locally and the server aggregates model weights or weight diffs
  across rounds. This is the default fast path.
- Use `fedeval-pt` for evaluation-only jobs that send a checkpoint to sites and
  collect metrics without local training updates.
- For any other algorithm requirement, do not assume a recipe name:
  select the catalog recipe whose metadata matches the request, confirmed with
  `nvflare recipe list --framework pytorch --format json` and
  `nvflare recipe show <recipe> --format json`. Match on the catalog fields
  `recipe list` exposes on every entry: `algorithm`, `aggregation`,
  `state_exchange`, and `privacy` (these are also the `--filter` keys).
  `privacy_compatible` is present only on recipes that declare it (such as HE
  recipes) and is not a `--filter` key; confirm it with `recipe show` on the
  candidate recipe. Do not match on `topology`; use topology wording in the
  request as intent that maps to exposed catalog fields, recipe semantics, or
  required role parameters.
- Homomorphic encryption (HE) / encrypted aggregation is NOT supported by the
  conversion skills, and it is safety-critical not to fake it. HE recipes such as
  `fedavg-he-pt` (`FedAvgRecipeWithHE`) reject `SimEnv` and require a provisioned
  deployment environment (`HEBuilder` startup kits with `PocEnv`/`ProdEnv`),
  which is outside conversion scope per `conversion-workflow.md`. When the user
  requests HE or encrypted aggregation, state that HE is not supported by
  conversion, route the request to provisioning/deployment, and ask (interactive)
  or fail closed (unattended). Do not generate an HE `job.py`, and never silently
  substitute a non-HE recipe such as `fedavg-pt` to make the request or a
  simulation pass — that would drop the encryption requirement.
- The same rule covers every other privacy-protection request: differential
  privacy, privacy filters, and other NVFLARE `Filter`-based protection are NOT
  supported by the conversion skills either (see `conversion-workflow.md`). Do
  not treat a privacy request as recipe selection: report it as unsupported,
  route it to provisioning/deployment, and ask or fail closed rather than
  generating an unprotected job with only a disclaimer.
- Current names are examples to verify against the catalog, not an authoritative
  mapping: `fedprox-pt` (FedProx / proximal loss), `fedopt-pt` (server-side optimizer
  variants such as FedAdam / FedYogi / FedAdagrad), `scaffold-pt` (SCAFFOLD
  control variates / client-drift mitigation), `cyclic-pt` (sequential
  client-to-client transfer), `swarm-pt` (swarm / peer aggregation topology).
- Ask the user before choosing when the requested FL workflow is not clear.

## Non-FedAvg Recipe Rules

The FedAvg fast path is not a universal job template. When the user asks for
FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm Learning, FedEval, or a
topology-specific workflow (HE / encrypted aggregation is not supported — see the
HE rule in Selection Rules):

- use `nvflare recipe show <recipe-name> --format json` for the selected recipe;
- supply parameters marked `"required": true`;
- leave optional parameters at defaults unless the user request, source code, or
  validation result requires them;
- keep generated source names consistent with the loading skill and runtime
  locations consistent with `runtime-output-guidance.md`;
- keep shared generated files on all clients unless the recipe semantics or user
  request require site-specific roles, scripts, arguments, or launch settings;
- ask before choosing when recipe intent or topology is ambiguous.

For recipes with topology roles, such as cyclic ordering, swarm roles, vertical
data ownership, or label-owner style workflows, do not collapse the topology into
standard FedAvg. Preserve the selected recipe's required role or site parameters
and report any missing user input before validation.

## Framework-Specific Job Construction

This reference covers recipe discovery, algorithm choice, and selection rules.
For the framework-specific `job.py` construction — the fast-path recipe code,
model config, portable imports, per-site deployment, and execution mode — follow
the loading skill's own recipe reference (the recipe-selection reference for
plain PyTorch; the Lightning conversion reference for Lightning; the Hugging
Face conversion reference for Trainer workflows). For common PyTorch-family
recipe parameters such as `key_metric`, `launch_external_process`,
`server_expected_format`, and `enable_tensor_disk_offload`, follow
`pytorch-family-recipe-construction.md`. Export handling is shared: follow
`conversion-workflow.md` for `--export`, `--export-dir`, and local command-line
parser behavior.
