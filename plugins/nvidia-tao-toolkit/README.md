# NVIDIA TAO Toolkit plugin (`nvidia-tao-toolkit`)

Agent skills for NVIDIA TAO: train, fine-tune, evaluate, export and deploy
vision and multimodal models, prepare and generate training data, run AutoML
and application workflows, and launch the jobs on Docker, Kubernetes, SLURM or
Brev. The plugin works in Claude Code, Codex and Cursor.

An agent uses a skill by reading its `SKILL.md` and then running the commands
that file documents, in most skills a TAO container started with `docker run`.
Nothing in this folder is a service. Apart from the session-start hook
described below, nothing here runs until an agent or a user starts it.

## What is in this folder

| Path | Content |
|---|---|
| `skills/<skill>/` | One folder per skill: its `SKILL.md` and the references, scripts, schemas and templates that the skill uses. |
| `scripts/` | Helper scripts that skills call: the listings of models and capabilities, the resolvers for container images, models and `versions.yaml` keys, the launch preflight, the job record, the secret lint, the multi-node probe, and `tao_agent_identity.py`. |
| `versions.yaml` | The container image tags and Python wheel pins that skills and the resolvers read. |
| `catalog/` | `skill-layers.json`, which records the layer of every skill (applications, core, data, models or platform), and the manifests that describe a whole layer. |
| `templates/` | Job templates for Kubernetes and SLURM. |
| `hooks/` | The session-start hook, `session_start.sh`, and its registration, `hooks.json`. |
| `AGENTS.md` | The agent guidance that the hook adds to a session. |
| `evals/evals.json` | The task file that NVIDIA's plugin verification evaluates the plugin with. It is not a Claude Code eval suite. |
| `plugin.json` and the `.<host>-plugin/` folders | The plugin manifests: Agent Plugins 1.0.0 at the root, and one folder for each host: `.claude-plugin/`, `.cursor-plugin/` and, unless the build left it out, `.codex-plugin/`. |
| `LICENSE` | The Apache License 2.0. |

This folder is generated from the repository
https://github.com/NVIDIA-TAO/tao-skill-bank and is not edited by hand.

## Requirements

- `bash`, which a host finds through `/usr/bin/env` when it starts the
  session-start hook, with `awk`, `basename`, `cat`, `dirname`, `grep`, `head`,
  `tail` and `wc`. The hook uses nothing else. Without `awk` it adds nothing to
  a session.
- Python 3.10 or later for the helper scripts.
- PyYAML for the helper scripts that read YAML:
  `python3 -m pip install pyyaml`.
- Docker with the NVIDIA Container Toolkit, an NVIDIA GPU and its driver, only
  to run jobs on this machine: most skills run training, evaluation, inference
  and data generation in TAO containers. Reading a skill, planning and the
  listings need none of this. The `tao-setup-nvidia-gpu-host` skill checks a
  host and, after the user's approval, installs what is missing.
- What one skill needs beyond this, such as `kubectl`, a login to a SLURM
  cluster or further Python packages, is in the `compatibility` line of its
  `SKILL.md`.

## How each host uses the plugin

In Claude Code and Codex, install the plugin with the host's plugin commands
from the marketplace or catalog that lists it. The Cursor CLI was tested with
this folder loaded directly: `cursor-agent --plugin-dir <this folder>`. The
skills are the same in the three hosts. What differs is what a session knows
about TAO before it opens a skill.

| | Claude Code | Codex | Cursor |
|---|---|---|---|
| A skill is listed as | `nvidia-tao-toolkit:<skill>` | `nvidia-tao-toolkit:<skill>` | `<skill>` |
| Session-start hook | runs | does not run | runs in the Cursor CLI |
| Agent identity (`AGENTS.md`) | added by the hook | not added; optional block in the project's `AGENTS.md` | added by the hook in the Cursor CLI |
| `TAO_SKILL_BANK_PATH` | exported by the hook | not set | not set; the hook states the folder |

- **Claude Code** runs the hook at the start of every session, in every
  project where the plugin is enabled.
- **Codex** lists no hook for this folder, because the root `plugin.json`
  declares the Agent Plugins schema. A Codex session therefore starts without
  the identity and without `TAO_SKILL_BANK_PATH`; the setup and preflight
  blocks of the skills set the variable to this folder, which is two levels
  above a skill's own folder.
- **Cursor**: the Cursor CLI runs the same hook and adds what it prints to the
  session. The hook exports nothing there, so it states this folder in its
  text.

Codex and Cursor read the `AGENTS.md` at the root of the project a session
works in, not the one in this folder. To give such a session the identity, run
this in the project, with the user's approval:

```bash
python3 <this folder>/scripts/tao_agent_identity.py add
```

It adds the identity to the project's `AGENTS.md` as a block between two
marker lines, or creates that file, and leaves the user's own text as it is;
`status` reports the block and `remove` deletes it. Where the hook already
adds the identity, the block repeats it. The `tao-setup` skill describes the
same step.

## What runs at session start

`hooks/hooks.json` registers one script, `hooks/session_start.sh`, as two
commands. Each prints one JSON object, and the host adds the text in it to the
session:

- the text of `AGENTS.md`;
- the path of this folder;
- the names, never the values, of the credential variables below that are set;
- when `docker` is not on `PATH`, the read-only command that checks the host;
- when the `pi` harness is installed, the skills that have a card pack for it.

In Claude Code the script also appends one line that exports
`TAO_SKILL_BANK_PATH` to the environment file of the session. It makes no
network call, installs nothing and starts no container.

## Credentials

Skills read credentials from environment variables. Export what a workflow
needs before the agent starts, or keep the values in an env file of your own
that the agent sources with your approval. `AGENTS.md` tells the agent never to
ask for a value in chat and never to print one, and the hook reports names
only.

| Variable | Read for |
|---|---|
| `NGC_KEY` | pulling TAO container images from `nvcr.io` |
| `HF_TOKEN` | gated Hugging Face models and datasets |
| `BREV_API_TOKEN` | the Brev platform skill, `tao-run-on-brev` |
| `ACCESS_KEY`, `SECRET_KEY`, `S3_BUCKET_NAME`, `S3_ENDPOINT_URL` | S3-compatible object storage |
| `WANDB_API_KEY` | experiment logging to Weights & Biases |

## Licence

The plugin is licensed under the Apache License 2.0; the text is in `LICENSE`.
Each skill states its licence in the `license` field of its `SKILL.md`.

## Reporting issues

Report a problem at https://github.com/NVIDIA-TAO/tao-skill-bank/issues.
Report a security vulnerability the way
https://github.com/NVIDIA-TAO/tao-skill-bank/blob/main/SECURITY.md describes,
not in a public issue.
