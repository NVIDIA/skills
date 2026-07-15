# jetson-customize-pinmux — Detailed Procedure

Detailed step-by-step procedure for the `jetson-customize-pinmux` skill.
The SKILL.md keeps a short summary; this file holds the full prose.

## Step 1 — Resolve active target + XLSM

Resolve the active profile per `target-platform-contract.md`
(in `../../context/`).

**Refuse conditions:**

| Condition | Reason |
|---|---|
| No active profile (`active: NA`) | Run `/jetson-init-target` first. |
| `custom_carrier:` missing | Pinmux customization is carrier-specific. |
| `<source.root_path>/Linux_for_Tegra/.git` missing | Run `/jetson-init-source` first. |
| Carrier conf fork missing from overlay tracker | Run `/jetson-derive-carrier` first. |
| No `documents.{custom_carrier_pinmux_xls,ref_devkit_pinmux_xls}` AND no `.xlsm` found under `documents.root_path` | Prompt for an absolute path; abort if user doesn't provide one. |

**Identifiers from the profile:**

| Var | Source |
|---|---|
| `<chip>` | `reference_devkit.module.id` → catalogue (T234 / T264) |
| `<soc>` | `264` (Thor) / `234` (Orin) |
| `<custom-carrier>` | kebab-cased `custom_carrier.name` |
| `<custom-flash-conf>` | `custom_carrier.flash_config` |

**Resolve `<xlsm>` path** (precedence order):

1. `documents.custom_carrier_pinmux_xls` if set.
2. `documents.ref_devkit_pinmux_xls` if set.
3. Single `.xlsm` under `<documents.root_path>/` (one-level scan).
4. Prompt for absolute path; record in run-state but **do not** write
   back to the profile.

## Step 2 — Probe — parse the XLSM, build the pinmap

The bundled script writes its own KB into a per-skill scratch
directory under `target-platform/`:

```
<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/
├── session.json          # shim for modify_pinmux.py --kb-dir
└── pinmap/
    └── <custom-carrier>.json
```

**Generate the session shim first** (one-time per run, before any
script call) — see Step 7 for the schema. Then run:

```bash
KB=<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux
python3 <skill-base>/scripts/modify_pinmux.py probe \
    --xlsm <resolved xlsm path> \
    --carrier-name <custom-carrier> \
    --kb-dir "$KB"
```

`probe` writes `<KB>/pinmap/<custom-carrier>.json` and updates
`<KB>/session.json` with `xlsm_path` + `carrier_name`. **No DTSIs are
written yet.**

If exactly one `.xlsm` matched the search in Step 1, ask the user to
confirm (don't auto-bind).

## Step 3 — Lookup — resolve a user query to one or more pinmap rows

The user does NOT need to know the DT pin name. They type whatever
they have on hand:

- CVM ball (`H10`)
- Verilog name (`PEX_L4_RST_N`)
- Carrier signal name (`PCIE4_RST_L`, `CAM0_I2C_SCL`, `GPIO47`)
- DT pin name (`pex_l4_rst_n_pd1`)

```bash
python3 <skill-base>/scripts/modify_pinmux.py lookup \
    --kb-dir "$KB" --query "<X>"
```

Case-insensitive match across `dt_pin_name` / `pin`, `cvm_pin` /
`ball`, `verilog_name`, `signal_name`, and any token in
`evidence[].net`. The output prints supported SFIO list (with
`(suggested)` on the SFIO matching `customer_usage`), default
direction, default initial-state, customer-usage note, and a
`configurable: yes/no` line.

**Match-count handling:**

- **0 matches** — script prints up to 5 closest substring candidates
  ("did you mean…?") and exits non-zero. Surface to user.
- **1 match** — proceed to Step 4.
- **>1 matches** — list all candidates and require the user to
  disambiguate by DT pin name on the next call. Never silently pick.

## Step 4 — Set-pin — capture the per-pin Q1–Q6 flow

**HARD GATE — ask the user, do not infer.** Per-pin SFIO / direction /
state / pull / drive / open-drain are the load-bearing choices of this
skill. The model MUST drive Q1–Q6 through `AskUserQuestion` (one
prompt per pin, options sourced from `sfio_options` + the XLSM
defaults shown by `lookup`). No-stop / quiet-mode / "resume"
instructions DO NOT bypass this gate — they cover prereqs and
defaults, not the primary user decision. If the user declines to
answer, stop the run with no edits committed; never substitute a
silent default like "keep as customer_usage suggests".

Run the question flow from `questions.json` for each pin the user
picked in Step 3 (any of the input forms listed in Step 3 are valid;
`lookup` resolves them uniformly). Q1–Q3 are always asked; Q4–Q6 are
gated on `configurable: yes`.

**Q1–Q3 (always asked):**

1. **`sfio`** — single-select from the supported list. `(suggested)`
   tags the XLSM `customer_usage` match. Validate membership. If the
   list has only one entry, still show it and ask — never silently
   auto-select.
2. **`direction`** — `input` / `output` / `bidirectional` / `unused`.
   `tristate` and `e_input` in the DTSI are **derived** from
   direction (`unused` → tristate=ENABLE; `input` /
   `bidirectional` → enable-input=ENABLE) — never asked separately.
3. **`initial_state`** — `low` / `high` / `hi-z` / `n/a`. For
   non-output pins the only valid value is `n/a`. Maps to XLSM
   `gpio_init_val` (`0`→`low`, `1`→`high`, `z`→`hi-z`).

**Q4–Q6 (only when `configurable: yes`):**

4. **`pull`** — `none` / `pull-up` / `pull-down`. Pre-fill from
   `default_pull` (XLSM `pupd`). Resistor is silicon-fixed at 20 kΩ.
5. **`drive_type`** — `normal` (1X) / `high` (2X). Pre-fill from
   `default_drv_type` (`drv_type`). **Hint:** QSPI SCK / IO SFIOs
   (`qspi*_sck` / `qspi*_io`) default to `high` (2X); SDMEM pads
   showing `DEF_1X` raw → `normal`.
6. **`open_drain`** — `disable` / `enable`. Pre-fill from
   `default_open_drain` (`e_io_od`). **Hint:** XLSM sets
   `e_io_od=Enable` for I2C (`i2c*_clk` / `i2c*_dat`), HDMI CEC
   (`hdmi_cec`), PCIe CLKREQ / RST (`pe*_clkreq_l` / `pe*_rst_l`),
   SHUTDOWN_N, and DPAUX-routed I2C (`dp_aux_*`). When the chosen
   SFIO matches one of these and pre-fill is `enable`, surface as
   required.

`loopback` (`e_lpbk`) is **not asked** unless the user explicitly
requests it (only QSPI0 / QSPI2 SCK have `e_lpbk=Enable` on Thor).

Informational only (shown by `lookup`, not settable): `por_state`
(`pu`/`pd`/`z`/`0`/`1`), `pull_strength` (20 kΩ on `BD*` pads;
`N/A` on DPAUX).

After Q1–Q3 (or Q1–Q6) for each pin, call:

```bash
python3 <skill-base>/scripts/modify_pinmux.py set-pin \
    --kb-dir "$KB" \
    --pin <dt_pin_name> \
    --sfio <sfio> \
    --direction <direction> \
    --initial-state <initial_state> \
    [--pull <pull>] [--drive-type <drv>] [--open-drain <od>]
```

The call validates the SFIO is in the supported list (rejects
otherwise) and appends / merges-by-pin into the run-state's
`pin_edits[]`. Repeat Steps 3 + 4 for every pin to modify.

## Step 5 — Generate — emit the three DTSIs in one shot

After **all** pin edits are collected, run:

```bash
OUT=<source.root_path>/Linux_for_Tegra/bootloader
python3 <skill-base>/scripts/modify_pinmux.py generate \
    --kb-dir "$KB" \
    --out-dir "$OUT"
```

This applies the captured `pin_edits` to the XLSM data and writes:

- `tegra<soc>-mb1-bct-pinmux-<carrier-key>.dtsi` — per-pin blocks
  setting `nvidia,function = "<sfio>"`, `nvidia,enable-input`,
  `nvidia,tristate`. Closing brace carries `// custom-bsp: pinmux`
  marker (idempotency).
- `tegra<soc>-mb1-bct-gpio-<carrier-key>.dtsi` — per `sfio=gpio` pin,
  appends / updates entries under the matching `gpio@<addr>`
  controller's `default { ... }`: `input` → `gpio-input`;
  `output + low` → `gpio-output-low`; `output + high` →
  `gpio-output-high`. Controller node (`gpio@ac300000` for Thor MAIN,
  etc.) is selected from the `GPIOn_*` bank prefix in the pinmap row's
  `sfio` list. Pins lacking a `gpio=…` entry in their SFIO list are
  rejected at `set-pin` time.
- `tegra<soc>-mb1-bct-padvoltage-<carrier-key>.dtsi` — no per-pin
  edits in the current flow; rewritten from XLSM data.

`<carrier-key>` matches the rename pattern `/jetson-derive-carrier`
chose (`<module-id-num>-xxxx-<custom-id>-<custom-sku>` or similar) —
the **exact filenames** referenced from the carrier conf's
`PINMUX_CONFIG=` / `GPIOINT_CONFIG=` / `PMC_CONFIG=` lines. Generate
writes to those exact paths so the carrier conf needs no edits here.

**Resolve `<carrier-key>` at runtime — DO NOT pass the bare carrier
name.** The script's `--carrier-name` is used verbatim as the filename
suffix; passing the kebab-cased custom-carrier name (e.g. `ken`)
produces `tegra<soc>-mb1-bct-pinmux-ken.dtsi`, which the carrier conf
does NOT reference and which are **orphan files at flash time**.
Instead, derive `<carrier-key>` from the active flash-conf fork:

```bash
CONF=<source.root_path>/Linux_for_Tegra/$(readlink \
    <source.root_path>/Linux_for_Tegra/<custom_carrier.flash_config>)
PINMUX_REF=$(grep '^PINMUX_CONFIG=' "$CONF" | cut -d'"' -f2)
# e.g. "tegra264-mb1-bct-pinmux-p3834-xxxx-p6767-0002.dts"
CARRIER_KEY=$(echo "$PINMUX_REF" \
    | sed -E "s|^tegra[0-9]+-mb1-bct-pinmux-(.*)\.dtsi?$|\1|")
# CARRIER_KEY = "p3834-xxxx-p6767-0002"
```

Pass `--carrier-name "$CARRIER_KEY"` to every `probe` / `generate`
call. With the key derived this way, `generate` overwrites the
derive-carrier `.dtsi` forks in place, and the carrier conf's
`PINMUX_CONFIG=` / `PMC_CONFIG=` references resolve correctly with
no conf edit. The kebab-cased custom-carrier name remains the right
input for SCRATCH PATHS only (the per-skill scratch directory under
`target-platform/`, and the pinmap JSON filename
`<custom-carrier>.json`).

**Sanity-check the carrier `.dts` wrapper's `#include` line (REQUIRED).**
`/jetson-derive-carrier` forks the `.dts` wrapper at
`bootloader/generic/BCT/tegra<soc>-mb1-bct-{pinmux,padvoltage}-<CARRIER_KEY>.dts`
but its `#include "..."` may still point at the upstream devkit `.dtsi`
(e.g. `tegra264-mb1-bct-pinmux-p3834-xxxx-p4071-0000.dtsi`). After
`generate` writes the new carrier-key `.dtsi` to `bootloader/` root, the
wrapper must include the new file — otherwise the carrier conf still
sources the upstream devkit pinmux at flash time and your edits are
silently ignored. For each wrapper that needs it:

```c
/* before */
#include "tegra264-mb1-bct-pinmux-p3834-xxxx-p4071-0000.dtsi"
/* after */
#include "tegra264-mb1-bct-pinmux-<CARRIER_KEY>.dtsi" /* custom-bsp: pinmux */
```

**Use a bare basename, not a filesystem-relative path.** The BCT build
runs `cpp -I bootloader/` so the resolver picks up sibling `.dtsi`
files regardless of where the `.dts` wrapper lives in
`bootloader/generic/BCT/`. Authoring `../../tegra264-mb1-bct-…dtsi`
works but breaks the convention used by every other BCT include in the
tree. Stage the wrapper edit and roll it into the same commit as the
three DTSIs in Step 6.

## Step 6 — Commit the three DTSIs in the overlay tracker

The three DTSIs `generate` produced under
`<source.root_path>/Linux_for_Tegra/bootloader/` are a **single
logical edit**, so they get committed as one batched pair per the
overlay-tracker commit-batching rule in
`bsp-customization-workflow.md` (`#commit-batching-in-the-overlay-tracker`,
under `../../context/`):

- **Pristine import (one commit, skip if unnecessary).** If any of
  the three DTSIs lack a prior pristine import commit (rare —
  `/jetson-derive-carrier` should have created them), import the
  pre-edit copies via XLSM regeneration and commit all newly
  imported files in a single pristine commit. If all three already
  have pristine imports from `/jetson-derive-carrier`, skip this.
- **Customization (one commit, always).** `git add` all three DTSIs
  **and any carrier `.dts` wrapper whose `#include` was re-pointed in
  Step 5's sanity-check**, then commit **once** with a single
  `jetson-customize-pinmux:` message that names every pin changed plus
  a one-line summary per DTSI.

Never produce three separate per-DTSI customization commits — the
batched single commit is what the workflow's commit-batching rule
mandates for multi-file logical edits in the overlay tracker.

**Preview gate (required, per commit).** Before running `git commit`
for either the pristine import or the customization, render `git
diff --staged --name-status` plus the proposed commit message to
the operator and obtain accept / edit / cancel per the workflow
"Commit message preview gate" section (in `bsp-customization-workflow.md`).
On **cancel**, leave the index staged and exit without committing —
do not auto-commit.

## Step 7 — Run-state JSON sidecar + session shim

Two artifacts:

**User-facing sidecar** at
`<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux.json`:

```json
{
  "generator": "jetson-customize-pinmux",
  "generator_version": "0.1.0",
  "generated_at": "<ISO-8601 UTC timestamp>",
  "active_profile": "<profile-stem>.yaml",
  "bsp_version": "<bsp_image.version>",
  "xlsm_path": "<resolved absolute path>",
  "carrier_name": "<custom-carrier>",
  "pin_edits": [
    {
      "pin": "pex_l4_rst_n_pd1",
      "ball": "H10",
      "verilog_name": "PEX_L4_RST_N",
      "signal_name": "PCIE4_RST_L",
      "sfio": "gpio",
      "direction": "output",
      "initial_state": "high",
      "pull": "pull-up",
      "drive_type": "normal",
      "open_drain": "disable",
      "configurable": true,
      "evidence": "user",
      "reason": "M.2 NVMe slot CLKREQ reset"
    }
  ],
  "dtsis_written": [
    "<source.root_path>/Linux_for_Tegra/bootloader/tegra<soc>-mb1-bct-pinmux-<carrier-key>.dtsi",
    "<source.root_path>/Linux_for_Tegra/bootloader/tegra<soc>-mb1-bct-gpio-<carrier-key>.dtsi",
    "<source.root_path>/Linux_for_Tegra/bootloader/tegra<soc>-mb1-bct-padvoltage-<carrier-key>.dtsi"
  ],
  "warnings": [],
  "notes": [],
  "commit_shas": {
    "overlay_tracker": "<short SHA>"
  }
}
```

**Script-internal `session.json` shim** at
`<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/session.json`:

```json
{
  "platform": "thor" | "orin",
  "carrier_short": "<custom-carrier>",
  "carrier_name": "<custom-carrier>",
  "reference_conf": "<reference_devkit.flash_config>",
  "xlsm_path": "<resolved absolute path>",
  "cloned_pinmux_dtsi": "<source.root_path>/Linux_for_Tegra/bootloader/tegra<soc>-mb1-bct-pinmux-<carrier-key>.dtsi",
  "cloned_gpio_dtsi":   "<source.root_path>/Linux_for_Tegra/bootloader/tegra<soc>-mb1-bct-gpio-<carrier-key>.dtsi",
  "pinmux": {
    "pin_edits": [],
    "notes": [],
    "warnings": []
  }
}
```

**Atomic write** for both: write `.tmp` sibling then rename. The
`session.json` shim is the script's working state and is regenerated
each run from the user-facing sidecar + active profile.

**Idempotency contract:** re-runs read the user-facing sidecar to
surface prior `pin_edits` as a starting list; the user can add,
modify, or remove entries. If no edits change between runs, Steps 5
+ 6 emit `(no change)` and skip the rewrite.

## Step 8 — Summary

Print one block:

```
jetson-customize-pinmux: <N> pin edit(s) emitted to <carrier-key> DTSIs; commit <short SHA>.
```

| Field | Value |
|---|---|
| Active profile | `target-platform/<profile-stem>.yaml` |
| Custom carrier | `<custom_carrier.name>` (`<custom-id>-<custom-sku>`) |
| Pinmux XLSM | `<xlsm path>` |
| Pin edits | one row per `pin_edits[]` entry: `pin / ball / SFIO / direction / initial / (pull, drv, od when configurable)` |
| DTSIs written | three paths under `<source.root_path>/Linux_for_Tegra/bootloader/` |
| Commit SHA | `<source.root_path>/Linux_for_Tegra` → `<short SHA>` (single commit covering all three DTSIs) |
| Run-state sidecar | `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux.json` |
| Session shim | `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/session.json` |
| Warnings | newline-separated `warnings[]` (e.g. pins skipped by `commit`, missing per-pin blocks) |
| Notes | newline-separated `notes[]` |

**Next step (interactive prompt chain):**

After the summary table is printed, drive the downstream chain via
sequential `AskUserQuestion` prompts. Never auto-invoke; each prompt
needs an explicit `yes`. On `no` (or any abort), print the remaining
manual run-chain and exit.

In order:

1. **Customize any other I/O before build?** — offer
   `/jetson-customize-uphy`, `/jetson-customize-pcie`,
   `/jetson-customize-mgbe`, `/jetson-customize-usb`,
   `/jetson-customize-camera`, `/jetson-customize-clocks`,
   `/jetson-customize-fan`, `/jetson-customize-nvpmodel`,
   `/jetson-customize-memory`, and `no` (proceed). On any non-`no`
   pick, invoke the chosen sub-skill inline; when it returns, re-ask
   this same question (loop) until the user picks `no`. Then continue
   to (2).
2. **Build & promote?** — on `yes` invoke `/jetson-build-source` (skip
   if no source-tree repo was touched this session — pinmux-only edits
   land in the overlay tracker), then on success invoke
   `/jetson-promote-image`.
3. **Flash the board?** — only offer if (2) ran and succeeded; on
   `yes` invoke `/jetson-flash-image`.
4. **Validate on the DUT?** — only offer if (3) ran and succeeded; on
   `yes` invoke `/jetson-validate-image`.
