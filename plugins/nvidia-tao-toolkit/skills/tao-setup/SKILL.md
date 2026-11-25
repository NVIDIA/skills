---
name: tao-setup
description: One-time session setup and orchestration map for the TAO skill bank. Run this first when the TAO skills were installed individually (e.g. from a public skills catalog) so the session gets the cross-skill discovery flow, credential checks, and host preflight that the bundled plugin hook would otherwise inject automatically. Trigger phrases include "set up TAO skills", "TAO session setup", "prepare TAO environment", "TAO getting started".
license: Apache-2.0
compatibility: Requires bash and Python 3.10+ with PyYAML (the shared helper scripts import it). Docker plus the NVIDIA container toolkit are needed by most downstream TAO skills but are only checked (not installed) here.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- setup
- orchestration
- discovery
---

# TAO Setup

One-time session bootstrap for the TAO skill bank. TAO skills are standalone —
each model, data, and platform skill carries its own pinned container image and
instructions — but multi-skill workflows chain them (data prep, train,
evaluate, deploy). This skill provides the session-level pieces that make that
chaining work when skills are installed individually: the discovery flow, the
credential conventions, and the host preflight.

When the full skill bank is installed as a plugin from this repository, a
SessionStart hook injects this guidance automatically and you do not need to
run this skill. When skills were installed one-by-one from a skills catalog,
run this skill first.

## Quick Start

Individual skill installs do not include the shared `scripts/` helpers. The
bank that matches these skills is the `nvidia-tao-toolkit` plugin folder: the
folder that holds `skills/`, `scripts/`, `catalog/` and `versions.yaml`, with
every skill directly under `skills/`. Reuse a plugin-provided
`TAO_SKILL_BANK_PATH` when present. When `TAO_SKILL_BANK_PATH` is not already
set (no session hook exported it, as in Codex), set it before the check to the
plugin folder that contains this skill, the folder two levels above this
skill's folder, when that folder passes the check. The first lines of the
block do that once `<this-skill-folder>` is replaced with the folder that
holds this `SKILL.md`.

A checkout of https://github.com/NVIDIA-TAO/tao-skill-bank is a bank only when
it contains `plugins/nvidia-tao-toolkit`; the bank is then that folder, never the
root of the checkout, which keeps its skills in layer folders. When a checkout
has no such folder, do not use it: with user approval, install the complete
`nvidia-tao-toolkit` plugin from the catalog these skills came from. Verify the
bank before using any shared helper, and stop if the check fails.

```bash
# TAO_SKILL_BANK_PATH not set? Set it to the plugin folder that contains this skill: two levels above this skill's folder.
skill_dir="<this-skill-folder>"   # the folder that holds this skill's SKILL.md
if [ -z "${TAO_SKILL_BANK_PATH:-}" ] && [ -f "$skill_dir/../../catalog/skill-layers.json" ]; then
  TAO_SKILL_BANK_PATH="$(cd -P "$skill_dir/../.." && pwd -P)"
fi
export TAO_SKILL_BANK_PATH="${TAO_SKILL_BANK_PATH:-$HOME/tao-skill-bank/plugins/nvidia-tao-toolkit}"
for item in versions.yaml scripts/resolve_tao_model.py scripts/resolve_tao_image.py scripts/check_tao_launch_preflight.py scripts/redact_secrets.py scripts/tao_job_record.py scripts/skill_layers.py catalog/skill-layers.json; do
  [ -e "$TAO_SKILL_BANK_PATH/$item" ] || { echo "Incomplete TAO skill bank: missing $item under $TAO_SKILL_BANK_PATH. Set TAO_SKILL_BANK_PATH to the nvidia-tao-toolkit plugin folder that contains this skill (two levels above this skill's folder); a repository checkout without plugins/nvidia-tao-toolkit is not a bank." >&2; exit 1; }
done

set -a; source /path/to/.env; set +a   # omit if already exported

# 1. Host preflight — the shared helper scripts need PyYAML for python3, and
#    most TAO skills dispatch docker containers on a GPU host.
python3 -c 'import yaml' 2>/dev/null && echo "OK: PyYAML" || echo "MISSING: PyYAML (python3 cannot import yaml; the shared helper scripts need it). Install it with: python3 -m pip install pyyaml"
docker info > /dev/null && echo "OK: docker" || echo "MISSING: docker"
nvidia-smi > /dev/null && echo "OK: GPU" || echo "MISSING: NVIDIA GPU/driver"

# 2. Credential presence check — names only, never print values.
for v in NGC_KEY HF_TOKEN WANDB_API_KEY ACCESS_KEY SECRET_KEY S3_BUCKET_NAME S3_ENDPOINT_URL BREV_API_TOKEN; do
  [ -n "${!v:-}" ] && echo "SET:   $v" || echo "unset: $v"
done

# 3. NGC registry login (needed for nvcr.io image pulls). Key goes over
#    stdin — never as an argv flag, where it lands in the process table.
[ -n "${NGC_KEY:-}" ] && printf '%s' "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# 4. Token-efficient execution kit (recommended): preflight + scaffold
#    ~/.tao-kit/kit.env. Non-invasive; see skills/tao-token-efficient-execution.
bash "${TAO_SKILL_BANK_PATH}/skills/tao-token-efficient-execution/scripts/install.sh"
```

If the check prints `MISSING: PyYAML`, the machine is not ready: the listing,
resolver, and launch-preflight helpers under `scripts/` cannot start without
it. Install PyYAML into the Python that `python3` runs (in a virtual
environment when the system Python refuses the install), rerun the check, and
do not report setup as complete before it prints `OK: PyYAML`.

If Docker or the NVIDIA host runtime is missing, use the
`tao-setup-nvidia-gpu-host` skill — it checks and (with approval) installs
NVIDIA driver 580 or newer, CUDA Toolkit 13.0 or newer, and NVIDIA Container
Toolkit 1.19.0 or newer, and can install Docker itself on Debian/RHEL/SUSE-family
hosts. These are TAO-wide minimums. If the selected model's
`references/skill_info.yaml` declares `runtime_requirements.gpu_host`, pass
those model-specific minimums to the host setup skill instead.

## Credentials

Load a user-approved env file with `set -a; source /path/to/.env; set +a` in the
same bash call as the command that consumes the variable. This skill never
creates a credentials file for you; the one credential write here is step 3's
`docker login`, which stores an nvcr.io token in `~/.docker/config.json`.

- `NGC_KEY` — nvcr.io image pulls (most skills)
- `HF_TOKEN` — gated HuggingFace weights (several model skills)
- `WANDB_API_KEY` — experiment tracking (optional)
- `ACCESS_KEY` / `SECRET_KEY` / `S3_BUCKET_NAME` / `S3_ENDPOINT_URL` — S3 I/O
- `BREV_API_TOKEN` — Brev platform dispatch

## Token-efficient execution (recommended for repeated workflows)

Step 4 of the Quick Start installs the execution kit
(`skills/tao-token-efficient-execution`): compiled **card packs** run
application workflows as fresh per-stage sessions instead of one long
conversation (measured 74-91% cheaper, 4-5x lower peak context, and the only
mode validated on small execution models). Setup only scaffolds the config;
the routing rule applies at launch time: **when a card pack exists for the
requested workflow (`skills/<skill>/cards/`), the `pi` harness
is installed, and the user asked for the card-pack route, use its `driver.sh`
instead of in-conversation stage execution; otherwise run the stages in the
conversation.** The `tao-launch-workflow` gate still applies in full — the
driver launch is the side-effecting action it reviews. Packs ship today for
`tao-run-deft-aoi` and `tao-run-automl`; the kit's authoring prompt compiles
packs for other workflows.

## Discovery flow (how TAO skills chain)

1. **Read the task skill.** Model skills (`tao-train-*`, `tao-finetune-*`)
   own network specifics; data skills (`tao-generate-*`, `tao-analyze-*`,
   `tao-mine-*`, …) own transforms; application skills (`tao-run-automl`,
   `tao-run-deft-aoi`, …) compose model + data + platform into workflows.

2. **Read the skill's `references/skill_info.yaml`** (when present) for the
   structured contract: `container_image` (a pinned URI), or
   `backend_contracts.<backend>.container_image` for a multi-backend frontend;
   per-action `command`, `mode`, `config_format`, `inputs`, `outputs`, and
   optional `runtime_requirements.gpu_host`. Model runtime requirements
   override the TAO-wide platform defaults for that workflow.

3. **Pick an execution platform and read its skill** for mounts, env vars,
   and resource conventions: `tao-run-on-docker` conventions apply to any
   local `docker run`; `tao-run-on-slurm`, `tao-run-on-kubernetes`, and
   `tao-run-on-brev` cover managed dispatch; `tao-run-on-virtualenv` runs a
   Python script docker-free in a local venv. Externally installed platform
   skills (e.g. kratos) join as peers — no registration needed.
   The platforms are equal-class peers — if the user has not chosen, ask;
   never default silently. Every platform skill implements the same
   **four-verb consumer contract** (`submit`/`status`/`logs`/`cancel`) over its
   native CLI (`docker`/`kubectl`/`ssh`+`sbatch`/`brev exec`) — there is no
   `nvidia-tao-sdk`.

4. **Construct the spec as nested dicts** (`{"train": {"num_epochs": 12}}`,
   never flat dotted keys), confirm with the user, then **execute the four
   verbs**: `tao-launch-workflow` drives the shared launch gate;
   `scripts/tao_job_record.py open` mints the job id and binds `results_dir`
   *before* launch (record-then-launch); the platform skill runs `submit`; then
   monitor with `status`/`logs`, mapping native states to the fixed vocabulary
   `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`.

## Conventions all TAO skills follow

- **Confirm before side effects.** `docker run`, job submission, pushes, and
  file mutations outside the working directory need user confirmation first.
  Installing a missing Python package prerequisite is the one exception:
  install it by default and report what was installed.
- **Never ask for credentials in chat** and never print credential values or
  the contents of a credentials file; name the missing variable so the user can
  export it or add it to an env file you then source.
- **Container images are pinned per skill.** Each skill carries the exact
  image URI it was validated against; do not swap tags silently. Offer
  overrides only when the skill documents an override path.
- **Runtime requirements are layered.** Platform skills own the default host
  requirements and the check/install mechanism. A model may override only the
  minimum versions it has validated by declaring `runtime_requirements.gpu_host`
  in `references/skill_info.yaml`; pass those values to the shared host setup
  check rather than changing the defaults for unrelated models.
- **Execution is SDK-free.** Job tracking (`scripts/tao_job_record.py`),
  S3/data staging (`tao-data-io`, storage tiers A/B/C), and multi-node (the
  SLURM/K8s templates + `scripts/nccl_allreduce_probe.py`) are built into the
  bank — no `nvidia-tao-sdk`. The one exception is AutoML search
  (`tao-run-automl`), which uses the `nvidia-tao-automl` wheel and its
  transitive SDK.

## Optional: agent identity for Codex and Cursor

In Claude Code the session-start hook of this plugin supplies the TAO agent
identity, and nothing more is needed. Codex and Cursor do not load the
`AGENTS.md` at the root of the plugin folder; they read the `AGENTS.md` at the
root of the project the user works in. With user approval, add the identity
to that file as a marked block. The helper creates the file when it is
missing, leaves every other byte of an existing file alone, and never copies
over a file or makes a backup. Run it in the project. When
`TAO_SKILL_BANK_PATH` is not set (no session hook exported it, as in Codex),
replace `<this-skill-folder>` first, as in Quick Start:

```bash
skill_dir="<this-skill-folder>"   # the folder that holds this skill's SKILL.md
helper="${TAO_SKILL_BANK_PATH:-$skill_dir/../..}/scripts/tao_agent_identity.py"
[ -f "$helper" ] || { echo "No $helper. Set TAO_SKILL_BANK_PATH to the plugin folder first, see Quick Start." >&2; exit 1; }
python3 "$helper" add
```

`add` again updates the block in place, `status` reports it, and `remove`
deletes it; `--dry-run` shows what would change, and `--project DIR` names a
project other than the current one. Do not copy the identity over
`~/.codex/AGENTS.md` or any other file. Do not register another TAO
marketplace or install another TAO plugin for this either: the skills are
already installed, and a second copy of the plugin makes Codex list every
skill twice.
