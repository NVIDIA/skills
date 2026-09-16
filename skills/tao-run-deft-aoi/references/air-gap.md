# Air-Gap and Offline Execution

Read this before Pre-Flight. This file owns global network-mode resolution and
overrides fetch, login, credential, and package-install instructions in other
stage references.

## Activation

Enable global air-gap mode when any condition is true:

1. `AIR_GAPPED=1` is present in the process environment.
2. The user explicitly requests air-gapped or offline execution.
3. The harness reports restricted networking.

Otherwise use the network-enabled path. Never attempt a network command to
infer the mode. Record both the resolved mode and activation source in the
Pre-Flight Summary.

`HF_HUB_OFFLINE=1` disables HuggingFace access only. It does not activate
global air-gap mode or disable other registries and services. Stage-specific
offline flags may still force cache-only execution after a permitted networked
bootstrap without changing the global mode.

## Air-Gap Contract

In global air-gap mode:

- Initialize state with `--network-mode airgap` and its activation source.
  After initialization, run external commands through `"$PYTHON" scripts/deft_exec.py`;
  it re-reads `execution_policy`, injects offline variables, and adds
  `--pull=never` to direct Docker/Podman runs.
- Do not run package managers, registry login or pull, downloads, APIs, or
  network probes.
- Convert every fetch/pull instruction into a local presence check. A missing
  image, model, dataset, or required host dependency is a hard stop.
- Use `bash scripts/deft_python.sh` to select host Python. On `ModuleNotFoundError`, retry
  through this orchestrator's launcher and hard-stop if no complete interpreter
  exists. Never run `pip`, even as a probe or after a failed bare-Python call; a
  package-manager attempt invalidates the air-gap run and requires a fresh run
  directory rather than an explanation after the fact.
- Use only images already present in the local Docker daemon and assets at the
  documented staged paths.
- Pass `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to stages that consume
  HuggingFace-backed caches.
- Never print, grep, or expose credential values from the process environment.
  Record `NGC_KEY` and `HF_TOKEN` as `N/A (offline)` when local assets satisfy
  the run, never use them for a network action, and never request a credential
  value.

The normal Pre-Flight approval gate still applies. Air-gap mode changes asset
resolution and network behavior, not user authorization or stage ordering.
Never read `references/network-bootstrap.md` in this mode.

## AnomalyGen staged set

When AnomalyGen will run, its base cache and Guardrail safety model must be fully pre-staged, and the base cache must be verified offline with the container's own check before SDG.

## Pre-Flight Evidence

Include these rows in the Summary:

| Field | Required evidence |
|---|---|
| Network mode | `airgap` or `network-enabled` |
| Activation source | `AIR_GAPPED`, user request, harness restriction, or default |
| Container images | Local image inspection result or post-approval pull plan |
| Models and datasets | Staged local path or post-approval fetch plan |
| Credentials | `N/A (offline)` or presence-only status for required networked actions |
