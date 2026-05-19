---
name: "skill-card-generator"
version: "1.0.1"
description: "Reads an existing agent skill directory and produces a governance skill card plus a review table. Use only when a skill directory exists and a skill card needs to be generated or updated; do NOT use for model cards, agent cards, container cards, system cards, or general documentation."
license: ""
compatibility: "Any agent that can run Python scripts and write files"
permissions:
  - file_read
  - file_write
  - shell
metadata:
  tags:
    - skill-card
    - governance
    - documentation
    - trustworthy-ai
  domain: documentation
---

# Generate Skill Card

**Skill directory to analyze**: $ARGUMENTS

## Purpose

Produces a filled skill card from a target skill's source files and the surrounding repo. The card is rendered deterministically from a Jinja template driven by a JSON context you author. A separate review table flags every inferred or human-required field.

## When to use

- A skill directory exists and needs a governance card
- An existing card is out of date after changes to the skill
- Before submitting a skill for legal/safety review
- Do NOT use for model cards, agent cards, container cards, system cards, or free-form documentation.

## Prerequisites

- Python 3 is available for script execution.
- `jinja2` is installed before rendering with `scripts/render_card.py`.
- The target path is an existing skill directory that contains `SKILL.md`.
- You have permission to read the target skill and write the generated card beside it.
- Run all bundled scripts from this skill directory or pass absolute script paths.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/discover_assets.py` | Reads the target skill directory, gathers local repo signals, and prints the source context needed to assemble the card JSON. | `<skill_directory>` |
| `scripts/render_card.py` | Validates a context JSON file and renders the final skill card from the Jinja template. | `--context <context.json> --template <skill-card.md.j2> --out <output.md>` |
| `scripts/validate_submission.py` | Checks a rendered skill card for unresolved VERIFY or SELECT markers before submission. | `<rendered-card.md>` |

## Workflow

### Step 1 — Resolve the target

If `$ARGUMENTS` is provided, use it. Otherwise default to the current working directory. The target should be a skill directory (typically `<repo>/.agents/skills/<name>/` or `.claude/skills/<name>/`).

Resolve the target path before writing files. Reject path traversal input such as `..`, and do not write outside the target skill directory unless the user explicitly confirms the output path.

### Step 2 — Run the discovery script

```
python3 <skill-dir>/scripts/discover_assets.py <target>
```

For runtimes that expose script helpers, use an explicit script call such as:

```
run_script("scripts/discover_assets.py", args=["<target>"])
```

The output contains, in order:
- Discovery report (file roles)
- Extracted file contents from the skill directory
- Extracted file contents from the repo root (README, evaluation docs)
- **Structured signal summary** (primary input for your context)
- Style guide (verbatim from `references/style-guide.md`)
- Jinja template (verbatim from `references/skill-card.md.j2`)

All inputs you need are in this one output — do not issue additional Read calls for source files.

### Step 3 — Build the context JSON

Read the style guide and build a context object for this skill. Every field is defined there. Key rules:

- The **signal summary** is your first stop for each field — frontmatter, license, version.
- When the summary doesn't cover a field, read the raw extracted file contents.
- When neither source supports a field, choose the honest default the style guide specifies; use `HUMAN-REQUIRED` placeholders only as a last resort.
- Do not leave `use_case` empty.
- Set `owner.verify: true` whenever ownership is inferred or defaulted (see style guide for when `verify: false` is appropriate). Set `license_verify: true` unless the license identifier was extracted verbatim from a documentation file.

Write the context to a temp file, e.g. `/tmp/<skill-name>-context.json`.

### Step 4 — Render the card

This step writes a markdown file. By default, write only to `<target>/<skill-name>-skill-card.md` so the generated card stays inside the analyzed skill directory.

```
python3 <skill-dir>/scripts/render_card.py \
  --context /tmp/<skill-name>-context.json \
  --template <skill-dir>/references/skill-card.md.j2 \
  --out <target>/<skill-name>-skill-card.md
```

For runtimes that expose script helpers, use:

```
run_script("scripts/render_card.py", args=[
  "--context", "/tmp/<skill-name>-context.json",
  "--template", "<skill-dir>/references/skill-card.md.j2",
  "--out", "<target>/<skill-name>-skill-card.md"
])
```

The script validates the context against a minimal schema and refuses to render if required fields are missing or typed wrong. Fix reported errors before proceeding.

### Step 5 — Self-verify

Before finishing:
- Cross-field consistency checks from the style guide must pass.
- The rendered card should not contain unrendered `{{ ... }}` or `{% ... %}` fragments.
- Run the submission validator before handing off the card:

```
run_script("scripts/validate_submission.py", args=["<target>/<skill-name>-skill-card.md"])
```

## Limitations

- The generated card is a draft and requires human review for inferred ownership, license, deployment, risk, and evaluation fields.
- The discovery script samples long files to control context size, so unusually large repositories may require targeted follow-up review.
- The renderer validates the JSON shape, not the real-world truth of governance claims.
- The skill reads local files, runs Python scripts, and writes a markdown output file.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `ERROR: jinja2 not installed` | The renderer dependency is missing. | Install `jinja2`, then rerun `scripts/render_card.py`. |
| `Context validation failed` | The context JSON is missing required keys or uses the wrong type. | Fix the reported keys using `references/style-guide.md`, then render again. |
| `STOP: No skill definition file found` | The target directory does not contain a parseable `SKILL.md`. | Point `$ARGUMENTS` at the root of an existing skill directory. |
| Unresolved VERIFY or SELECT markers remain | Human review has not removed generated review markers. | Resolve each marked field or canned entry, then rerun `scripts/validate_submission.py`. |


## Files in this skill

- `SKILL.md` — this file (orchestration)
- `references/style-guide.md` — per-context-field guidance (the substantive instructions)
- `references/skill-card.md.j2` — Jinja template (exact card layout)
- `references/catalog/limitations.json` — canned technical-limitations catalog (buffet list)
- `references/catalog/risks.json` — canned risk-management catalog (buffet list)
- `scripts/discover_assets.py` — discovery + signal extraction
- `scripts/render_card.py` — Jinja renderer with context validation and catalog injection
- `scripts/validate_submission.py` — pre-submission marker validator
