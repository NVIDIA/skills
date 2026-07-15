---
name: jetson-quick-start
description: >-
  Entry skill for Jetson / IGX BSP customization. Asks one core
  click-to-select setup questionnaire and passes prefilled answers to
  downstream setup skills.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags: [setup, bootstrap, download]
  domain: setup
---

# Quick Start — BSP customization entrypoint

`jetson-quick-start` is a dispatcher and intake form. It does not download,
extract, materialize sources, bind documents, or derive carrier files. It
chooses the setup path, gathers core answers, then hands those answers to
the skills that own the actual work.

## Modes

Do not ask setup mode in a separate prompt. Include it as the first field
in the core quick-start questionnaire, with choices shown in this fixed
order:

1. `Auto Setup`
2. `Guided Setup`
3. `Use Existing Workspace`

Offer `cancel` separately as an exit action, not as a setup mode.

| Mode | Use when | Planned skills |
|---|---|---|
| `Auto Setup` | Fresh workspace, network available | target setup, `/jetson-download-bsp`, `/jetson-init-image`, `/jetson-init-source`, `/jetson-link-docs`, `/jetson-generate-kb`, optional `/jetson-derive-carrier` |
| `Guided Setup` | User already has paths, repos, or archives | target setup, `/jetson-init-image`, `/jetson-init-source`, `/jetson-link-docs`, `/jetson-generate-kb`, optional `/jetson-derive-carrier` |
| `Use Existing Workspace` | BSP image, source tree, and docs are already prepared | target setup, workspace verification, `/jetson-generate-kb`, optional `/jetson-derive-carrier`; route to init skills only for missing prerequisites |

Target setup means keep the active profile, switch via `/jetson-set-target`,
or create one via `/jetson-init-target`.

## Procedure

### Print Disclaimer

Before opening the core questionnaire or inspecting setup inputs, print
this disclaimer block exactly once:

```text
================================================================================
DISCLAIMER
These skills help automate Jetson BSP setup and customization, but they do not
replace NVIDIA official documentation or engineering review. Review generated
plans, commands, diffs, and commit messages before accepting them.

Flashing can erase device storage or leave a target temporarily unbootable.
Keep backups and verify the active target, BSP release, and hardware setup
before deploy steps.
================================================================================
```

### Disclaimer Acceptance Gate

Immediately after printing the disclaimer, open a separate one-question
`AskUserQuestions` disclaimer form and wait for the user's submitted
answer before doing anything else. Do not collect disclaimer acceptance
with a plain-text chat reply, inferred default, prior run state, active
profile, cached answer, or any other non-UI signal.

The form must contain exactly one question:

1. `accept_disclaimer` — prompt: "Accept the disclaimer and continue
   quick-start?" Choices: `accept_disclaimer` and `cancel`.

Treat only a submitted `accept_disclaimer` choice as acceptance. The
disclaimer form is separate from the core questionnaire and does not
replace any of the four routing-critical core questions.

If the form cannot be opened, is unavailable, is cancelled, returns no
submitted answer set, or returns any answer other than
`accept_disclaimer`, stop immediately and report that quick-start cannot
continue until the disclaimer is accepted through `AskUserQuestions`.

### Mandatory Questionnaire Gate

After the disclaimer has been accepted, `jetson-quick-start` MUST open the
core `AskUserQuestions` questionnaire and wait for the user's submitted
answers before doing any setup dispatch. If the questionnaire cannot be
opened, is unavailable, is cancelled, or returns no submitted answer set,
stop immediately and report that quick-start cannot continue without the
questionnaire.

Do not continue by using existing active profiles, workspace state,
tarball names, cached answers, release compatibility, or "obvious"
defaults. Do not call downstream skills until this gate has a submitted
`mode` answer and normalized `quick_start_prefill`.

### Survey State

Inspect `target-platform/active_target.yml` and existing
`target-platform/*.yaml` profiles. If an active profile exists, show
`reference_devkit.name`, optional `custom_carrier.name`, and active
`flash_config`. Do not infer platform identity from tarballs, repo names,
or document titles.

Read [`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md)
and parse every row in the product / chip / SKU / flash-config table.
This catalogue is the source for the full `active_platform`
questionnaire list. If the list is incomplete, update the catalogue;
do not duplicate or patch the platform list inside quick-start. Do not read
`jetson-init-target/SKILL.md` or run the init-target flow during
quick-start preprocessing; downstream target creation remains owned by
`jetson-init-target`.

Keep this survey lightweight so the core questionnaire appears quickly:
read target profile pointers/summaries, perform cheap path existence
checks when useful, and fetch only lightweight Jetson Linux release
metadata from the official archive needed to populate `bsp_release`
choices. Parse all current release sections, not only the section that
appears to match the selected platform. Do not scan large BSP / source /
document trees, inspect archives, generate KB content, or run setup tools
before the user submits the core questionnaire. Prepare the possible plans
for all three modes; choose the actual downstream plan only after the form
is submitted.

### Print Platform Reference List

Before opening the `AskUserQuestions` UI, print the full platform list
parsed from `bsp-platforms-catalogue.md` with stable numeric indexes for
this run:

```text
Available platforms:
T234 - Orin
  1. Jetson AGX Orin 64GB | CVM P3701-0005 | CVB P3737-0000 | jetson-agx-orin-devkit.conf
  ...
T264 - Thor
  N. Jetson AGX Thor T5000 | CVM P3834-0008 | CVB P4071-0000 | jetson-agx-thor-devkit.conf
```

Use these indexes only for the immediately following questionnaire.
Do not auto-select an index, even when there is an active profile or a
single obvious match.

### Ask Core Questionnaire

Generate one click-to-select core questionnaire for the whole setup run
using the `AskUserQuestions` UI. Do not collect quick-start answers with
plain-text prompts, inline chat questions, inferred defaults, or manual
summary confirmation.

This is a hard gate: do not skip it, do not use a one-profile shortcut,
and do not auto-select `mode`, `active_platform`, `bsp_release`, or
`custom_carrier` from `active_target.yml`, filenames, tarballs, repo
names, release compatibility, workspace state, cached answers, or prior
runs. An existing active profile may be shown as an option, but it is not
selected until the user submits the `AskUserQuestions` form.

Because `AskUserQuestions` can ask at most four questions, the core form
must contain only routing-critical fields:

1. `mode` — `Auto Setup`, `Guided Setup`, `Use Existing Workspace`, or
   `cancel`.
2. `active_platform` — use the printed full platform list; follow the
   active-platform question rules below.
3. `bsp_release` — recent global concrete release candidate shortcuts
   generated from the official Jetson Linux archive for this run, sorted
   newest first by dotted numeric order, plus `skip`. Preserve each
   official release token exactly as listed; concrete tokens may have two
   or three numeric components, such as `38.2`, `38.2.1`, `38.4`, or
   `R36.4.4`. Do not normalize `38.2` to `38.2.0`, and do not infer a
   missing patch component. Do not hard-code or cache candidates. If
   metadata fetch is slow or fails, still show `skip` and a typed concrete
   release option. Do not platform-filter this list in quick-start; include
   at least the newest concrete row from each current major release line
   found in the archive. If parsing returns only one major release line,
   treat the release list as incomplete and fall back to `skip` plus typed
   input instead of showing a partial candidate list.
4. `custom_carrier` — if the selected/active profile already has a
   custom carrier, offer `keep_existing_custom_carrier`,
   `no_custom_carrier`, `add_custom_carrier`, and `skip`; otherwise offer
   `no_custom_carrier`, `add_custom_carrier`, and `skip`.

The user submits this `AskUserQuestions` form once. Only after that
submission may `mode`, active platform, BSP release, and custom-carrier
intent decide which downstream skills run and which prefilled answers are
consumed or ignored. If there is no submitted form result, stop; never
continue with inferred answers. `add_custom_carrier` is only a routing
intent; carrier name, ID, SKU, revision, and custom flash config remain
downstream-owned.

Render the form using the shared
[`User input prompt style`](../../context/bsp-customization-workflow.md#user-input-prompt-style)
and normalize submitted UI answers into `quick_start_prefill`. Follow the
shared
[`quick_start_prefill` contract](../../context/bsp-customization-workflow.md#quick_start_prefill-contract).
Active-platform question rules:

- Print the full indexed platform list before the questionnaire.
- Explicit choices are only the first row from each chip-family group,
  labeled with full-list index and product name, for example
  `Index 1 — Jetson AGX Orin 64GB`.
- For all other platforms, the user chooses `Type something` and enters
  the numeric index from the printed list. `Type something` must be the
  tool's built-in freeform row, not a custom explicit choice.
- Prompt text: "Pick a platform shortcut, or choose Type something and
  enter the numeric index from the full platform list printed above. Type
  skip to skip platform selection and let downstream setup ask later."
- Only platform shortcuts may be explicit choices. Do not add `skip`,
  `Other`, or custom manual-entry choices; `skip` must be typed through
  the built-in `Type something` path. Do not paraphrase `Type something`
  as `Other` in prompt text.
- Do not resolve flash-config variants here; pass the selected catalogue
  row to `jetson-init-target`, which owns validation and flash-config
  selection.

Each non-`skip` answer must be explicit and directly consumable by the
owning downstream skill; use `skip` for anything unknown. A BSP release
answer must be a concrete version token with an optional leading `R` and
two or three numeric components (for example `R38.4`, `38.2`, `38.2.1`,
or `36.4.4`). Family placeholders such as `R38.x` or "pin a version
later" are not acceptable; use `skip` instead. Preserve the user's/source
token exactly when passing `quick_start_prefill.download.bsp_release`.

Image paths, source paths, document paths, carrier details, repository
overrides, toolchain choices, and document bindings are non-core setup
details. Leave them to the owning downstream skill. Downstream setup
skills should use documented defaults without asking when safe, and ask
only on conflicts, overwrites, incompatible releases, missing artifacts,
or user-requested overrides.

For Auto Setup, `/jetson-download-bsp` owns listing supported BSP releases
for the selected platform and warning on platform/BSP incompatibility.
Quick-start may forward a requested global candidate or typed release, but
it does not validate support or override compatibility decisions itself.

### Build Prefill Bundle

Normalize only explicit non-`skip` answers and keep them in memory:

```yaml
quick_start_prefill:
  mode: Auto Setup | Guided Setup | Use Existing Workspace
  target: { active_platform, custom_carrier }
  download: { bsp_release }
```

Omit any skipped field and omit owner subsets with no core answer. Validate
only enough to avoid obvious misrouting, such as blank mode or a BSP
placeholder. Downstream defaults are skill behavior, not prefill fields.
Do not invent missing identity values and do not write profile blocks from
quick-start.

### Invoke Downstream Skills

Build the downstream plan from `quick_start_prefill.mode`, target state,
and prerequisite checks. Before invoking downstream skills, print a
non-blocking execution plan summary with the selected mode, target
platform intent, requested BSP release or downstream release-selection
handoff, custom-carrier intent, planned downstream skills in order, and
non-core details that downstream skills may still ask for. Do not ask to
approve this summary; continue dispatching unless the user chose `cancel`
in the core questionnaire. This does not bypass downstream blocking
validation gates.

Pass each downstream skill its relevant `quick_start_prefill` subset plus
the top-level `mode`.
The downstream skill must use valid prefilled answers, ask again for
missing, invalid, ambiguous, or incompatible required inputs, and own any
mutation of its profile block: `reference_devkit:`, `custom_carrier:`,
`bsp_image:`, `source:`, or `documents:`.

For `Use Existing Workspace`, verify:

- `<bsp_image.root_path or workspace/Image>/Linux_for_Tegra/` exists.
- `Linux_for_Tegra/rootfs/etc/nv_tegra_release` exists, proving
  `apply_binaries.sh` ran; otherwise route to `/jetson-init-image`.
- `<source.root_path or workspace/Source>/Linux_for_Tegra/` exists and is
  a git repo.
- recorded `documents:` paths exist when present.

Finish with executed skills, skipped skills, remaining downstream questions,
and whether the workspace is ready for `customize-*`.

### Suggest I/O customization next steps

After the dispatch summary, print a non-blocking "Next steps —
I/O customization" list **only when the active profile's `documents:`
block has at least one carrier-board-related slot bound via
`/jetson-link-docs`**. The qualifying slots are:

- `documents.carrier_board_spec`
- `documents.carrier_schematic`
- `documents.ref_devkit_pinmux_xls`
- `documents.custom_carrier_schematic`
- `documents.custom_carrier_pinmux_xls`

If none of those are bound, skip this step — the user has not yet
provided the carrier-board files these skills depend on, and the right
next step is `/jetson-link-docs`, not a customize-* skill. Say so in
one line ("No carrier-board docs bound — run `/jetson-link-docs` to
register the carrier schematic / pinmux xlsx before I/O customization")
and stop.

When at least one qualifying slot is bound, print the list below.
Adapting a board to its on-board peripherals typically starts with
pinmux and UPHY, then branches per-controller:

- `/jetson-customize-pinmux` — per-pin SFIO / direction / pull state
  (consumes `documents.custom_carrier_pinmux_xls` or
  `documents.ref_devkit_pinmux_xls`).
- `/jetson-customize-uphy` — UPHY lane allocation via `ODMDATA`
  (consumes `documents.module_design_guide` and
  `documents.custom_carrier_schematic` / `documents.carrier_schematic`).
- `/jetson-customize-pcie` — per-controller PCIe wiring
  (consumes `documents.module_design_guide`,
  `documents.custom_carrier_schematic` / `documents.carrier_schematic`,
  and `documents.custom_carrier_pinmux_xls` /
  `documents.ref_devkit_pinmux_xls`).
- `/jetson-customize-usb` — USB2 / USB3 SS port enable / disable
  (consumes `documents.module_design_guide`,
  `documents.custom_carrier_schematic` / `documents.carrier_schematic`,
  and `documents.custom_carrier_pinmux_xls` /
  `documents.ref_devkit_pinmux_xls`).
- `/jetson-customize-mgbe` — Multi-Gigabit Ethernet PHY wiring
  (consumes `documents.module_design_guide`,
  `documents.custom_carrier_schematic` / `documents.carrier_schematic`,
  and `documents.custom_carrier_pinmux_xls` /
  `documents.ref_devkit_pinmux_xls`).
- `/jetson-customize-camera` — CSI / MIPI / GMSL sensor bring-up
  (consumes `documents.module_design_guide` and
  `documents.custom_carrier_schematic` / `documents.carrier_schematic`).

For each suggested skill, flag any required `documents.*` slot that is
not yet bound as "run `/jetson-link-docs` first to bind <field>" rather
than hiding the skill — the user may want to fix the binding and retry.
Do not invoke the customize-* skills here; quick-start ends at the
suggestion.

## Purpose

Single entrypoint that gathers the four routing-critical answers
(`mode`, `active_platform`, `bsp_release`, `custom_carrier`) and
dispatches the right downstream Setup skills. Not a one-stop installer
— it never downloads, extracts, materializes, or binds anything on its
own.

## Prerequisites

- `bsp-platforms-catalogue.md` reachable (used to build the platform
  list).
- Submitted `accept_disclaimer` answer from the disclaimer
  `AskUserQuestions` form.
- `AskUserQuestions` UI available — quick-start refuses to continue
  without submitted disclaimer and core questionnaire forms.
- Network access only required for the Auto Setup branch
  (`/jetson-download-bsp` is the actual fetcher).

## Limitations

- Asks at most four core questions; any non-core value (image paths,
  source paths, document paths, carrier IDs, toolchain) is left to the
  owning downstream skill.
- Does not validate platform / BSP-release compatibility — that is
  `/jetson-download-bsp`'s job.
- Does not write to `target-platform/*.yaml`; profile mutation stays
  with `/jetson-init-target`, `/jetson-init-image`, etc.

## Troubleshooting

- **"questionnaire cancelled" exit** — the form must be submitted; rerun
  `/jetson-quick-start` and complete the four questions.
- **"disclaimer not accepted" exit** — rerun `/jetson-quick-start`, read
  the disclaimer, and choose `accept_disclaimer` in the disclaimer form
  when ready to continue.
- **Active profile not auto-selected** — by design; even with one
  matching profile, the user must pick it explicitly in the form.
- **Unknown `bsp_release` typed** — pass it as-is; `/jetson-download-bsp`
  validates the release against the official archive.

## Gotchas

- Ask the core questionnaire once. Downstream skills ask for non-core
  values, blanks, invalid values, or ambiguous choices.
- Keep `quick_start_prefill` in memory unless the user asks to save it.
- Never construct NVIDIA artifact URLs here; `/jetson-download-bsp` follows
  links from the selected Jetson Linux archive release.
- `platform_template.yaml` has no User Guide / Release Notes slots. Store
  those files or URLs in the final summary or KB, not as invented profile
  fields.

## References

- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md)
- [`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md)
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml)
- [`../jetson-download-bsp/SKILL.md`](../jetson-download-bsp/SKILL.md)
- Jetson Linux archive: <https://developer.nvidia.com/embedded/jetson-linux-archive>
