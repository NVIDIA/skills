# Subskill: Translate Device Utilities (One Group)

Phase 3 device utilities for one `{parent}/{framework}/` group.

## Resolve Flavor Pack

```text
pack = flavors/{to_lang}/{framework}/
```

Read `pack/device-utils.md` (Phase 3) and `pack/patterns.md` for
device-routine / `__device__` / Kokkos patterns.

## Required Inputs

| Input | Source |
|-------|--------|
| `utilities[]` | Subset of `device_utilities` sharing same parent dir |
| `config.json` | `.nsight-ai-port/config.json` |

## Steps

1. Write or extend `{parent}/{framework}/device_utils.*` per flavor.
2. Write `{parent}/{framework}/test/test_device_utils/`.
3. Hook build; verify with `USE_{FRAMEWORK}=1`.
4. Git commit; return to orchestrator.
   After the last group in this round's `device_utilities[]`,
   set `device_utils_complete: true`.

## Deferred to the router

Do not reinterpret these here:

- **Engagement level** —
  [Approval Gates vs. Engagement Level](../../SKILL.md#approval-gates-vs-engagement-level)
- **Attempt budget** — [Fault Tolerance](../../SKILL.md#fault-tolerance)
