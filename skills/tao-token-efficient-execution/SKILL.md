---
name: tao-token-efficient-execution
description: Token-efficient execution kit for repeated TAO agentic workflows. Compiles a skill into stage cards once (with a strong model), then executes them forever in fresh, small headless sessions coordinated by a bash driver — measured 67-91% token reduction and 3-5x lower peak context vs one long conversation. Use when a TAO workflow will be run repeatedly, when token cost matters, when the target model has a small context window, or on trigger phrases like "token efficient", "reduce token cost", "card pack", "stage cards", "run this workflow on a small model", "compile this skill into cards".
license: Apache-2.0
compatibility: Requires bash, jq, Python 3.10+, and a headless agent harness with tool interception (see adapters/ for supported versions). The shipped card packs additionally need docker and the prerequisites of their application skill.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tokenomics
- orchestration
- efficiency
- cards
---

# TAO Token-Efficient Execution

Run a staged TAO workflow as a series of **fresh headless agent sessions** —
one stage card per session — instead of one long conversation. State lives on
disk, never in chat history, so no session ever rereads the run's past.

Measured on real workflows (same tasks, same outcomes, billed tokens):

| Workflow | One long conversation | Card execution | Peak context |
|---|---|---|---|
| DEFT AOI loop | 9.6M | 2.5M (-74%) | 242k → 59k |
| AutoML (VCN classify) | 5.8M | 0.5M (-91%) | 198k → 42k |
| HuggingFace finetune | 1.7M | 0.5M (-67%) | 119k → 42k |

The mechanism: an agent API is stateless, so a long conversation resends its
entire history with every message — by mid-run that is ~100k+ tokens of skill
text, spec edits, and training logs reread on every tool call, of which the
current step needs ~3k. Cards make each step's 3k explicit and drop the rest.

## The three layers

1. **Core (harness-agnostic)** — stage cards (markdown, env-var contract), a
   bash driver (stage routing + job waiting), and on-disk state contracts
   (`references/CONTRACTS.md`).
2. **Harness adapters** (`adapters/`) — guard + recorder implemented per
   harness: hooks for Claude Code, extensions for Pi. Same logic, ~150 lines
   each. Guards block known dead ends before tokens are burned; the recorder
   appends executed commands to `commands.log` so later sessions reuse
   instead of re-derive.
3. **Card packs (per application skill)** — compiled cards living next to the
   skill they were authored from, e.g.
   `skills/applications/tao-run-deft-aoi/cards/` and
   `skills/applications/tao-run-automl/cards/`. Each pack ships its own
   `driver.sh` and `README.md` (version pin, required env, how to run).

Cards shrink the model's job; they don't remove it. The driver owns routing
and waiting (deterministic), helper scripts own state commits (deterministic),
and the model executes one pre-planned stage: run verbatim commands, read a
gate table, decide pass/fail from real output, fill in the one dynamic value.
If a stage ever becomes 100% deterministic, graduate it out of the card into
the driver or a helper script.

## Quick start — run a shipped card pack

```bash
# plugin installs export TAO_SKILL_BANK_PATH automatically; standalone users set it first:
export TAO_SKILL_BANK_PATH=/path/to/tao-skill-bank

# one-time: scaffold ~/.tao-kit/kit.env and check harness prerequisites
bash "$TAO_SKILL_BANK_PATH/skills/core/tao-token-efficient-execution/scripts/install.sh"

# edit ~/.tao-kit/kit.env: set WS (and VENV for the AutoML pack) and MODEL;
# export your provider's API key in your shell — never put it in kit.env.
# then launch a pack driver detached (leave stderr attached: config errors print immediately):
nohup bash "$TAO_SKILL_BANK_PATH/skills/applications/tao-run-automl/cards/driver.sh" > /dev/null &

# monitor
tail -f ~/.tao-kit/automl/driver.log
```

The driver waits while jobs run (activity-based, not process-based), fires one
fresh session per stage, halts on committed errors (no auto-retry — an
operator decides), and exits 0 only after verified completion. DEFT prepares
its inference handoff and passes the completion audit; AutoML requires its
DONE marker with no trailing failure. Round-cap exhaustion exits nonzero.

Starting a pack driver is side-effecting work: complete the
`tao-launch-workflow` gate (platform, credentials, image, datasets, launch
review) before launching it, exactly as for in-conversation execution. Pi guards are
not an OS sandbox: use an isolated trusted worker, not a workstation containing
unrelated credentials (see `adapters/pi/README.md`).

## Quick start — author a pack for another workflow

One authoring session with a **strong** model compiles the skill into cards
(measured: ~10 minutes, one-time cost; every later run executes them):

1. Read `references/authoring_prompt.md` (the round-0 contract) and
   `references/authoring_prompt_v2.md` (rules distilled from measured
   small-model failures: one command per step, mechanical state gates, exact
   artifact paths, explicit termination, bounded self-repair).
2. Fill in run parameters — everything concrete for the target machine.
   Whatever you leave out, execution sessions will burn tokens rediscovering.
3. The authoring session reads the skill + inspects the workspace + writes
   cards into `skills/applications/<skill>/cards/`. It must NOT run the
   workflow.
4. Copy a shipped pack's `driver.sh` and edit its three config points: the
   stage→card routing table, the `working()` predicate, and the paths block.
   (`templates/driver.template.sh` is the harness-portable skeleton.)

`references/CARD_AUTHORING.md` reconstructs a real authoring session play by
play.

## Model floor (measured, execute-side)

Authoring always needs a strong model. Execution was measured across sizes:

| Model class | Card execution | Notes |
|---|---|---|
| Frontier (e.g. Claude) | flawless | reference bar: 0 guard fires, clean labels |
| ~400B MoE | near-flawless | correct results; a few guard saves |
| ~35B MoE | workflow-dependent | AutoML clean end-to-end; DEFT completed but with sloppy stage labels |
| ~30B dense w/o tool discipline | below floor | fabricated results, loops; guards catch but cannot fix |

Raw-skill execution (no cards) did not complete honestly on ANY of the small
models measured — aborts, protocol violations, or falsely declared success.
Cards are what make small-model execution possible, not a minor optimization.

**Report, don't patch:** when an execution model fails a card, report it and
stop (the driver's no-auto-retry halt enforces this). Never edit cards or
state to cover a model's mistakes; only genuine card-authoring defects get
fixed, explicitly.

## Files

| Path | What |
|---|---|
| `references/CONTRACTS.md` | progress log, STAGE_DONE, commands.log, env contract |
| `references/authoring_prompt.md` / `_v2.md` | card compilation prompts |
| `references/CARD_AUTHORING.md` | a real authoring session, annotated |
| `templates/driver.template.sh` | driver skeleton (three config points) |
| `templates/kit.env.template` | per-host config (`~/.tao-kit/kit.env`) |
| `adapters/pi/`, `adapters/claude-code/` | guard + recorder per harness |
| `scripts/install.sh` | prerequisite check + kit.env scaffold |
| `scripts/analyze_usage.py` | per-session token accounting from session JSONL |
| `scripts/smoke_test.sh` | no-GPU adapter test (guard, recorder, accounting) |
