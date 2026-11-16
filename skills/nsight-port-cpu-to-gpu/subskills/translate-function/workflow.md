# Subskill: Translate One Function

Translate a **single** function from `porting_plan.json` through
phases 4a–4e.

## Resolve Flavor Pack

Read `{source_dir}/.nsight-ai-port/config.json`:

```text
framework_pack = flavors/{to_lang}/{framework}/
```

Output goes in `{parent}/{framework}/`; GPU builds use
`USE_{FRAMEWORK}=1`. The guard to emit is `config.json:include_guard`
— it defaults to `USE_{FRAMEWORK}` but may have been overridden.

Load flavor docs as needed:

- `framework_pack/translation.md` — the translation guide (layout,
  scaffold, accelerate, verify, integrate).
- `framework_pack/patterns.md` — accelerate step (directives / kernels)
- `framework_pack/build-integration.md` — when present (build wiring)
- [verification-guide.md](../../references/verification-guide.md)

### Cross-language: two stages (`from_lang != to_lang`)

If `config.json:from_lang != to_lang`, this function is translated in
**two stages** (see router SKILL.md "Cross-Language Porting").

```text
transition     = transitions/{from_lang}-to-{to_lang}.md  # Stage A
framework_pack = flavors/{to_lang}/{framework}/           # Stage B
```

- **Stage A** — follow that transition file (Stage A1–A5):
  translate the function to `{to_lang}` on the CPU and verify it
  against the original via the interop test. On success set
  `phases.language_translate: true` and commit. Stage B's reference
  is then the verified **`{function}_{to_lang}_host`** body — never
  the `{function}_{to_lang}` dispatcher, which forwards to the GPU
  once its guard is enabled.
- **Stage B** — run the framework-pack steps below **on that
  emitted `{to_lang}` file** instead of the original. The GPU
  guard nests inside it; the original caller's
  `USE_{TO_LANG}_TRANSLATION` dispatch is untouched.

If `from_lang == to_lang`, skip Stage A and run only the framework pack.

## Required Inputs

| Input | Source |
|-------|--------|
| `source_dir` | `config.json:source_dir` |
| `file`, `function` | Plan entry |
| `accelerator_chunks` | Plan entry (optional) |
| `config.json`, `progress.json`, `porting_plan.json` | `.nsight-ai-port/` |

## Prerequisites

- Device utilities done if plan lists `used_by` this function.
- Refactor tasks for this function complete.
- Target **externally callable** — else STOP for refactor.

## Steps

This subskill **is** Phase 4 of the orchestrator workflow. For
cross-language ports, first complete **Stage A** via the transition
file (above), then continue with the framework pack. Execute the
framework stage by following
`framework_pack/translation.md` — that guide enumerates phases
4a–4e (build-system exploration, scaffold, accelerate, verify,
integrate) and is authoritative for each. **Follow it, including any
further file it directs you to read** (for Fortran it routes to the
sibling/inline variant per `config.json:translation_mode`).

Do not re-derive the steps here; read the guide and work through its
phases in order, committing and updating `progress.json` as it
instructs. The guide's integrate phase does the completion
bookkeeping (`commit_hash`, `translated_functions`,
`last_working_commit`); do not repeat it here.

## Deferred to the router

Do not reinterpret these here:

- **Engagement level** —
  [Approval Gates vs. Engagement Level](../../SKILL.md#approval-gates-vs-engagement-level)
- **Attempt budget** — [Fault Tolerance](../../SKILL.md#fault-tolerance)
