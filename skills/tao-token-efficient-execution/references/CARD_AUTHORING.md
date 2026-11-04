# How cards get generated — the real authoring session, play by play

Cards are not written by hand. One headless session with the authoring prompt
(`authoring_prompt.md` in this directory, plus the `authoring_prompt_v2.md`
rules) compiles the skill into cards. This page reconstructs exactly what
that session did in the AutoML experiment, from its transcript — the result
is the pack now shipped at `skills/applications/tao-run-automl/cards/`.

**Session stats: ~10 minutes, 112k output tokens, one time per workflow.**
That is the entire compilation cost. Every execution session afterward runs
the cards it produced.

## Input: the authoring prompt

Two parts (see `authoring_prompt.md`):

- **Run parameters** — everything made concrete for this machine: training and
  validation CSV paths, image directory, backbone checkpoint, the exact
  container image (with "LOCAL ONLY, no pulls"), algorithm, rec count, epochs,
  GPU count, metric, and which approvals are pre-granted.
- **Card contract** — the exact list of card files to write, what each must
  do, and the rules (fresh sessions have no memory so cards carry all
  knowledge; check commands.log before deriving; detached launches end the
  turn; be terse).

Plus one hard constraint: *do NOT run the workflow in this session — only
read and write cards.*

## What the session actually did (from the transcript)

**1. Read the skills — 9 files:**

```
skills/applications/tao-run-automl/SKILL.md
skills/models/tao-train-visual-changenet/SKILL.md
skills/platform/tao-run-on-docker/SKILL.md
skills/applications/tao-run-automl/references/automl-runner-configuration.md
skills/applications/tao-run-automl/references/automl-preflight-concepts.md
skills/models/tao-train-visual-changenet/references/skill_info.yaml
skills/models/tao-train-visual-changenet/references/spec_template_evaluate.yaml
skills/models/tao-train-visual-changenet/eval.config
<workspace>/specs/baseline_spec.yaml
```

This is the expensive knowledge-ingestion step that a standard harness pays
on *every* run, and the kit pays exactly once.

**2. Inspected the real environment — 12 commands.** Listed the workspace and
dataset directories, checked the KPI data head, grepped the skill references
for the runner's state contract (`state.json`, recommendations, best config),
verified the train schema and wandb settings, and probed which spec fields
the search space could touch. This is where cards become *environment
specific*: paths, image names, and schema quirks get baked in as facts
instead of instructions to go look them up.

**3. Wrote the 4 cards:**

```
cards/00-preflight.md      verify package + docker + GPU, model gate, scaffold run dir
cards/10-baseline-eval.md  build eval spec, launch baseline eval DETACHED, idempotent
cards/20-launch-recs.md    write the (pre-approved) launch review, start the HPO
                           runner detached so it drives all 4 recs itself
cards/30-interpret.md      when the runner is done: harvest metrics, pick best,
                           write results.md, mark DONE
```

**4. Self-checked** — the last command syntax-validated the bash blocks it had
just written (`bash -n` on the extracted code fences).

## Output: what a compiled card looks like

Open the shipped `skills/applications/tao-run-automl/cards/30-interpret.md`
next to this page. Notice:

- Every command is copy-paste executable for the machine it was authored on —
  no "figure out the path", no skill references.
- STEP 0 is a **keep-waiting branch**: if the runner is still working, append
  the previous ok line and end the turn. That line was added *by an execution
  session* when reality disagreed with the card — cards are maintained by the
  runs themselves (bounded edits), not by humans.
- The failure branch enumerates known failure modes and their fixes, so a
  future session recovers instead of re-investigating.

## Why this is the cost win

The skills read in step 1 total tens of thousands of tokens and their
interpretation costs real reasoning. The session wrote ~20.6k chars of cards;
the shipped pack totals ~16.1k chars (~4k tokens) after trimming, and its
execution is mechanical. Compilation happens once; the binary runs forever —
and because execution needs almost no reasoning and peaks at ~42k context, it
runs on much smaller models than the one that authored the cards (measured
end-to-end down to ~35B-class; see the Model floor table in the kit SKILL.md).
