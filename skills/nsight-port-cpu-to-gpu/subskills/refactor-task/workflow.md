# Subskill: Refactor Task

One entry from `porting_plan.json["refactor_tasks"]`. No GPU syntax.

## Resolve docs

Phase 2 edits **source** code. Always read the language guide:

```text
flavors/{from_lang}/refactoring.md
```

When `from_lang == to_lang`, also read the framework addendum
`flavors/{to_lang}/{framework}/refactoring.md`. Skip the addendum
on a cross-language port — Stage A/B handle the target language.

## Required Inputs

| Input | Source |
|-------|--------|
| `task` | One refactor_tasks[] object |
| `build_correctness`, `run_correctness` | `config.json` |

## Steps

1. Read task and source.
2. Structural change only (no accelerator directives / CUDA / Kokkos).
3. Run the configured `build_correctness` then `run_correctness`
   (from `config.json`) — both must pass.
4. Git commit; update plan if `new_functions` created.
   After the last task in this round's `refactor_tasks[]`,
   set `refactor_complete: true`.

## Deferred to the router

Do not reinterpret these here:

- **Engagement level** —
  [Approval Gates vs. Engagement Level](../../SKILL.md#approval-gates-vs-engagement-level)
- **Attempt budget** — [Fault Tolerance](../../SKILL.md#fault-tolerance)
