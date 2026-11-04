---
name: ambient-healthcare-agent-with-nemotron-voice-agent
description: Customize NVIDIA Nemotron Voice Agent's Generic Pipecat example for healthcare appointment, five-field patient intake, or custom tool-calling workflows without a separate backend.
license: CC-BY-4.0 AND Apache-2.0
allowed-tools: Read Grep Glob Edit Write Bash Env WebFetch
metadata:
  author: "Jin Li <jinl@nvidia.com>"
  team: healthcare-tme
  tags: ambient-healthcare,nemotron-voice-agent,pipecat,tool-calling,healthcare
  version: "1.0.0"
---

# Ambient Healthcare Agent with Nemotron Voice Agent

## Purpose

Customize `src/examples/generic/` in a user-selected NVIDIA Nemotron Voice Agent (NVA) checkout. Use this skill for appointment making, five-field patient intake, or a developer-defined healthcare workflow that fits one Pipecat pipeline with an LLM prompt, OpenAI-style tool schemas, and Python handlers.

This skill is owned by Healthcare TME. It is not maintained or endorsed by the NVA team, and it does not live in the NVA repository. For ordinary NVA deployment or non-healthcare voice work, follow the public NVA documentation instead.

Read [the user-experience flowchart](references/user-experience-flowchart.md) when a visual overview of the gated workflow or bundled scenario state machines would help.

## Requirements

- A compatible checkout of `https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent`
- Python 3.10+ with PyYAML, Git, and Docker Compose
- Network access and credentials for the user-selected LLM, ASR, and TTS services
- Permission to write only to the NVA checkout path the user explicitly supplies
- Human review before sending fictional healthcare-style histories to any model endpoint

Treat the skill loader's installed directory as `SKILL_DIR`. Resolve bundled `scripts/` and `references/` from that directory, not from the current working directory.

## Required Welcome

Begin a new positive workflow with this phrase verbatim:

```text
Welcome to the NVIDIA Nemotron Voice Agent (NVA for short). We will customize NVA for creating ambient healthcare agents.
Now I will make a fresh clone of the Nemotron Voice Agent repository, and this will be the directory we work out of. Where would you like me to clone the repo to? Please provide a path.
If you already have a clone of the repository somewhere, please point me to the path.
```

Then stop. Do not inspect files, search for clones, reuse a path from earlier context, run tools, or choose a default. Set `NVA_ROOT` only from a path the user provides or confirms after this welcome.

## Instructions

### 1. Resolve and validate the NVA checkout

If the user requests a fresh clone, clone only to their exact destination and only with network and write permission:

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent.git "$NVA_ROOT"
```

If cloning fails, report the observed reason and ask the user to either grant the session the required access via `/permissions` and request a retry, or manually clone the NVA repository and provide its path. Do not retry until the user grants access or supplies a checkout path. For either a fresh or existing checkout, require these markers:

```bash
test -f "$NVA_ROOT/docker-compose.yml" && \
test -f "$NVA_ROOT/examples_registry.yaml" && \
test -d "$NVA_ROOT/src/examples/generic"
```

An invalid path is a hard stop. Do not modify any repository before this check passes.

After the markers pass, create `$NVA_ROOT/.env` by copying `$NVA_ROOT/.env.example` when the target does not already exist. Preserve an existing `.env` and never print its contents. If `.env` is absent and the template is missing, report that setup failure and stop. Do this immediately after validating a fresh clone or an existing checkout, before asking the user to choose hosted services.

### 2. Inspect compatibility and service defaults

Run:

```bash
python3 "$SKILL_DIR/scripts/inspect_nva_generic_defaults.py" \
  --nva-root "$NVA_ROOT" --check-compatibility
```

Stop on compatibility errors. Report the inspected prompt plus the LLM, ASR, and TTS key, display name, model/server, base URL when present, and catalog source. Never substitute release-specific defaults from memory.

The preflight checks Python structure and required capabilities rather than an NVA version number or one exact source string. It also discovers deployment skills from both `skills/*/SKILL.md` and `.agents/skills/*/SKILL.md`. A passing check is not permission to guess through an unknown layout: stop when syntax or a required semantic hook is ambiguous.

### 3. Select the runtime and verify setup access

Explain that the default is the compatible public NVIDIA AI Endpoint entries in the NVA cloud catalog. Before any inference, let the user choose public NVIDIA endpoints, NVA-managed local NIMs, existing NIM endpoints, or a mixed layout. Never silently fall back to public endpoints after opt-out. In the same message as these choices, give the user the actual absolute path to `$NVA_ROOT/.env` and tell them to fill in its `NVIDIA_API_KEY=` entry if they choose public NVIDIA AI Endpoints. Explain that the key authenticates access to those endpoints. Direct the user to the public NVA deployment documentation for credential setup; never ask them to paste a secret into chat or display the file contents. State:

```text
The NVIDIA_API_KEY is required to utilize public NVIDIA AI Endpoints. With this key configured, I will be running live tests while customizing and standing up a Nemotron Voice Agent application.
```

Before applying a healthcare overlay:

1. Run `python3 "$SKILL_DIR/scripts/verify_docker_compose_access.py"` and require all checks to pass. If Docker Compose access is blocked by session permissions, report the observed failure, ask the user to grant the required access via `/permissions`, and stop until the user requests a retry.
2. Use the public NVA deployment instructions to identify one recipe and the selected runtime's credential and endpoint requirements. Do not start the unmodified Generic recipe or run inference at this stage.

If repository compatibility or Docker Compose access fails, report the exact failed gate and stop before applying an overlay. Missing credentials or unavailable endpoints do not prevent overlay and static validation, but they block the later authenticated service checks, live validation, and handoff. Do not change runtime modes silently or claim the app is ready.

### 4. Obtain informed scenario selection

After checkout compatibility and Docker Compose access pass, restate the resolved runtime and services, then ask exactly:

```text
What type of voice agent application would you like to create? We have two example default use cases, appointment making and patient intake, or you could tell me your own use case.
```

Offer `appointment-making example`, `patient-intake example`, and `customize your own use case`. In the same message, replace the destination placeholder below with the actual resolved LLM display name and base URL or host:

```text
If you choose either example, I will apply its customization and send its bundled fictional conversation histories to <resolved destination> for live validation during setup and before handoff. The appointment example includes the fictional patient Jordan Patel, date of birth 1979-09-24, and appointment details. The patient-intake example includes the fictional patient Maya Chen, date of birth 1988-04-12, symptoms, current medications, and pharmacy details.
```

Wait for selection. Selecting a preset after this disclosure authorizes only the disclosed fixture and destination. Ask again if either changes.

### 5. Apply a preset

For a selected preset, apply the bundled overlay without additional design questions:

```bash
python3 "$SKILL_DIR/scripts/apply_generic_agent_template.py" \
  --nva-root "$NVA_ROOT" \
  --scenario-dir "$SKILL_DIR/references/appointment-making"
```

Use `references/patient-intake` for patient intake. The applier must preserve current LLM/ASR/TTS defaults, set the scenario prompt as the Generic default, patch only supported insertion points, and remain idempotent.

The applier reruns compatibility with the selected scenario before writing. To inspect that gate separately, pass `--scenario appointment-making`, `--scenario patient-intake`, or `--scenario custom` together with `--check-compatibility`. Scenario checks must cover every scenario-specific hook, including deterministic session startup for both examples.

Appointment making installs SQLite support, initializes `data/appointment-making/appointment_schedule.sqlite`, and creates or merges `docker-compose.override.yml`; it must not edit the base Compose file. It also queues the exact fixed opening greeting once at session start and suppresses the model-generated intro. Patient intake collects name, date of birth, symptoms, current medications, and preferred pharmacy, and installs deterministic turn/speech guards plus generated unit tests. Preserve its shared conversation state, earliest-missing-field question, exactly-once direct tool responses, one-time welcome, silent empty tool transitions, interruption forwarding, and interruptible welcome. These are code-backed safety invariants, not prompt-only suggestions. Read `references/example-design-choices.md` when implementation detail is needed.

### 6. Design a custom workflow

For `customize your own use case`, first ask what the conversation should accomplish; which fields are required, optional, or sensitive; what must be known and confirmed before each tool call; what each tool should read, write, and return; and which representative histories demonstrate message content and tool timing.

Use `references/custom/guide.md` and its templates. Present the proposed expected-conversation artifact, resolved destination, and data fields to the user. Do not implement or transmit it until the user approves that exact artifact and destination.

### 7. Validate, start, and test voice

Run static and scenario tests after applying the overlay. Check the selected runtime's credentials, then start the customized NVA recipe using its public deployment instructions; build when the source changes require it. Require the app and selected LLM/ASR/TTS health and authentication checks to pass. If a credential, startup, or service check fails, report the exact failed gate and stop before live validation. Never start the unmodified Generic recipe.

For an approved preset or custom fixture, export the selected endpoint credential in the process environment without displaying it, then run:

```bash
python3 "$SKILL_DIR/scripts/run_expected_conversation.py" \
  --nva-root "$NVA_ROOT" --live-test-approved
```

The runner checks tool timing, arguments, status, and result identifiers. Review the actual assistant message for semantic alignment with `expected_next_message_content`; do not claim success if a deterministic or semantic check fails.

After any validation-driven change, rebuild or restart the selected recipe as directed by the public NVA documentation and recheck its services. Complete one real microphone-to-ASR-to-LLM/tool-to-TTS round trip. Preset handoff is blocked until static tests, approved live histories, service health, and voice validation pass.

### 8. Handoff

Report the NVA path and commit; compatibility/default inspection and catalog sources; runtime mode, recipe, and repository/Docker/authentication health gates; modified files, prompt key, tool names, and data paths; static, live-history, service-health, and voice results; and the verified UI URL. If anything is incomplete, name it as pending or failed rather than saying the application is ready.

## Available Scripts

| Script | Purpose | Main arguments |
|---|---|---|
| `inspect_nva_generic_defaults.py` | Resolve defaults and validate supported patch capabilities | `--nva-root`, `--check-compatibility`, optional `--scenario` |
| `apply_generic_agent_template.py` | Apply a preset or custom overlay idempotently | `--nva-root`, `--scenario-dir` or explicit artifact paths |
| `run_expected_conversation.py` | Run an authorized live LLM conversation contract | `--nva-root`, `--live-test-approved`, optional endpoint overrides |
| `verify_docker_compose_access.py` | Test Docker daemon, Compose, and disposable startup | optional image and timeout flags |
| `nva_generic_defaults.py` | Shared inspection library imported by other scripts | library module; do not invoke directly |

Invoke scripts with `python3` as shown. Agent runtimes that expose a `run_script` facility may use it with the same argument vector.

## Examples

- “Customize NVA Generic for appointment scheduling” → use this skill and start with the exact welcome.
- “Build patient intake directly in the Pipecat Generic example” → use this skill.
- “Deploy ordinary NVA Generic” → do not use this skill; follow NVA deployment documentation.
- “Create a FastAPI/LangGraph healthcare backend” → use a backend-oriented skill instead.

## Limitations

- The bundled examples are demonstrations, not clinical decision support or production records systems.
- The skill does not diagnose, triage, recommend treatment, or replace privacy/security review.
- Upstream NVA changes can invalidate required capabilities; syntax-aware compatibility and the selected scenario check must both pass.
- Live calls can transmit approved fictional fixture content and incur endpoint charges.
- Voice validation requires interactive audio hardware and cannot be inferred from text-only tests.

## Troubleshooting

| Failure | Action |
|---|---|
| Path is not an NVA checkout | Ask for a valid explicit path; do not search the workspace |
| Compatibility preflight fails | Stop and report the missing layout, default, catalog, or insertion point |
| Authentication/service health fails | Ask the user to correct the selected endpoint configuration; do not change modes silently |
| Docker access/startup fails | Report daemon, permission, network, or image-pull failure and stop |
| Overlay application fails | Preserve the checkout, report the exact patch point, and do not hand-edit around the guard |
| Live history differs from expectation | Correct prompt/tool behavior, rerun static tests, then rerun the approved history |

See `references/deployment-modes.md`, `references/example-design-choices.md`, and the scenario directories for deeper implementation detail.
