# jetson-customize-uphy — Full Procedure

Detailed steps for the UPHY lane allocation skill. The SKILL.md keeps a
short summary; this file is the source of truth for the eight steps.

## Step 1 — Resolve target + open source-of-truth docs

Resolve per the target-platform contract (`context/target-platform-contract.md`).
**Refuse** if no active profile, no `custom_carrier:`, no
`Linux_for_Tegra/.git`, or no forked carrier conf in the tracker.

| Var | Source |
|---|---|
| chip | `reference_devkit.module.id` -> catalogue |
| custom-flash-conf | `custom_carrier.flash_config` |
| carrier-conf | `readlink <source.root_path>/Linux_for_Tegra/<custom-flash-conf>` |

Resolve docs from the profile. Resolution precedence per doc:

| Doc | Required when | Resolution precedence |
|---|---|---|
| Adaptation Guide | always | `documents.adaptation_guide` -> `documents.bsp_developer_guide` (if local HTML mirror) -> web fetch -> prompt |
| Carrier schematic | `custom_carrier:` present | `documents.custom_carrier_schematic` — **REQUIRED, refuse on miss (no prompt fallback)** |
| Carrier pinmux XLSM | `custom_carrier:` present | `documents.custom_carrier_pinmux_xls` — **REQUIRED, refuse on miss (no prompt fallback)** |
| Module Design Guide | recommended | `documents.module_design_guide` -> prompt |
| SoC TRM | recommended | `documents.soc_tech_ref_manual` -> prompt |

**Refuse hard when `custom_carrier:` is present and either
`documents.custom_carrier_schematic` or
`documents.custom_carrier_pinmux_xls` is missing from the active
profile.** Routing decisions on a custom carrier cannot be guessed; the
skill must not fall back to "unknown = keep + warn" because that
diverges BPMP from kernel-DT. Tell the user to register both via
`/jetson-link-docs` (or hand-edit the profile) and re-run. Reference-
devkit-only profiles (no `custom_carrier:` block) skip this check.

**Web-fetch fallback for the Adaptation Guide.** Derive the URL from
`bsp_image.version` (major-minor only) and the chip family from
`reference_devkit.module.id`:

```
https://docs.nvidia.com/jetson/archives/r<major>.<minor>/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/Jetson<Family>AdaptationBringUp.html#configure-the-uphy-lane
```

- major.minor from `bsp_image.version` (e.g. `38.4.0` -> `38.4`).
- Family = Thor for T264 (`p3834`); OrinAGX / OrinNX / OrinNano for T234
  per `reference_devkit.module.id` against the catalogue. Unknown module
  id -> record a warning and fall through to prompt rather than guessing.
- The `#configure-the-uphy-lane` anchor pins the section Step 2 needs.
- If the URL 404s, record a warning and fall through to the prompt —
  never fabricate alternate URLs.
- **URL construction is template-only.** Use the template above verbatim;
  no other path shape (e.g. `Jetson<Family>Series.html`) is permitted.
  On 404, prompt — do not retry with a guessed path.

**Refusal rules.** Refuse when (a) no Adaptation Guide source resolves,
or (b) `custom_carrier:` is present and the carrier schematic or pinmux
XLSM is missing. Module Design Guide and SoC TRM are recommended;
proceed with warnings if missing.

## Step 2 — Locate "Configure the UPHY Lane"

Open the Adaptation Guide source resolved in Step 1:

- Local PDF -> `Read` with `pages:` ranged to the section (binary-search by ToC).
- Local HTML mirror -> open page containing `#configure-the-uphy-lane`.
- Web fallback -> `WebFetch` against the URL derived in Step 1; extract
  every documented `uphy0-config-N` and (Thor) `uphy1-config-N` index
  plus lane-to-controller mapping. Cache page text in memory for the
  rest of the run; do not re-fetch per option.

Cross-check against Module Design Guide UPHY tables + SoC TRM UPHY block
diagram; cite all three (or the live URL) in Step 8 `sources[]`.

**No hard-coded tables.** Every index, lane assignment, and controller
list is discovered at runtime. If skill prose conflicts with the Guide,
**Guide wins** — record in `notes[]`.

## Step 3 — Cross-reference the carrier schematic

Identify physically-routed UPHY-fed controllers. Build:

| Receptacle / device | Controller | Lane width | Schematic page |
|---|---|---|---|
| (e.g. M.2 NVMe slot) | `pcie@C5` | x4 | 12 |
| (e.g. 10G RJ-45) | `mgbe2` | 1xSERDES | 27 |

**Cite net names** (`MGBE2_TX_P/N`, `PEX5_LN0+-`, `USB_SS0_RX+-`,
`UFS_RX+-/TX+-` — anchored to Module Design Guide UPHY-pinout). Every
UPHY-fed controller MUST resolve to a definite `routed=yes` or
`routed=no` from the schematic — there is no `unknown` state, because
Step 1 mandates the schematic when `custom_carrier:` is present. A
class with zero matching nets is `routed=no` -> `status="disabled"` in
Step 7, regardless of UPHY allocation.

## Step 4 — Enumerate matching UPHY options

For UPHY0 (and on Thor, UPHY1), enumerate **every** index the Guide
documents. Per option capture index + token, lane assignments,
schematic-routing (yes/no/partial), and reference-conf default (grep
`^ODMDATA=` in the reference-devkit conf).

**Never collapse to "default + Customize".** Surface every index so the
user sees the menu + routing consequences side-by-side. On conflicts
prefer the Guide and list in `notes[]`.

## Step 5 — Ask the user which config (HARD GATE)

**Core decision of the skill. NEVER skip.** No-stop / quiet-mode
policies cover prereqs and defaults, not the primary user decision.

**Print first, then ask.** Before opening `AskUserQuestion`, print one
compact markdown table per UPHY surface (UPHY0, and on Thor UPHY1):
columns `Index | Token | Lane summary | Stock?`. One row per index from
Step 4 — no extra prose, no provenance dump.

Build `AskUserQuestion` choices from Step-4 indices — **never
hard-coded**, **never silently default**. One question per UPHY surface:

1. **UPHY0** — 1-of-N. Default = reference-conf ODMDATA. Label options
   `<token> -- <lane summary>`.
2. **UPHY1** *(Thor only)* — same shape.
3. **Carrier-routing confirmation** *(only when `custom_carrier:` is
   present)* — if any controller is UPHY-allocated but `routed=no` per
   the schematic, surface the conflict and ask: keep (overlay disables
   the controller) vs. pick a different UPHY config. **There is no
   `unknown` option** — the mandatory schematic from Step 1 always
   yields a definite yes/no.

Persist answers to a JSON sidecar (see `run-state-sidecar.md`).

**Emit the user-selected index verbatim.** Write the `uphyX-config-N`
returned by `AskUserQuestion` into ODMDATA unchanged. If the menu
omitted the BPMP-dumped index, re-fetch the Adaptation Guide so every
documented index appears and re-ask.

## Step 6 — Edit carrier flash-conf fork (ODMDATA, atomic)

Locate the forked conf:

```bash
OVL=<source.root_path>/Linux_for_Tegra
CONF=$OVL/$(readlink $OVL/<custom_carrier.flash_config>)
```

This skill owns **every** ODMDATA token for the run. One commit, one
`ODMDATA="..."` line, all tokens. Sub-skills MUST NOT touch ODMDATA.

**Mandatory table walk before emitting (HARD GATE).** Build the full
Step-7 per-controller allocation table FIRST — every UPHY-fed
controller on the SoC (Thor: `pcie@C0..C5`, `mgbe0..3`, `ufshci`,
`usb_ss0..2`). For each row compute `BPMP-stock` (from the dtc dump)
vs `Desired` (from chosen `uphyX-config-N` lane map). Emit one token
per row where they differ. **No row may be skipped** — picking only
the controllers you happen to notice (e.g. PCIe deltas while
forgetting MGBE-del or UFS) is the documented MGBE-FMON reboot-loop
bug. If the table isn't built, do not emit ODMDATA.

Tokens emitted, in this order:

1. **UPHY surface tokens** — every chosen `uphyX-config-N` (Thor:
   both `uphy0` and `uphy1`, even when one matches the guide default).
2. **Per-controller status / speed tokens** — emit one token per row
   where plan-state differs from BPMP-stock (Step 6 BPMP-stock diff
   below). Match-rows get no token.
   - PCIe: `pcie@N_status=okay|disabled`.
   - MGBE: `mgbeN-speed-<rate>` on allocate, `mgbeN-speed-del` on
     disable (FMON safety — required even when UPHY didn't allocate
     MGBE).
   - USB SS: per-port tokens when the SoC's ODMDATA grammar exposes
     them.
3. **`UPHY_CONFIG=""` clear** when `uphy0-config-6` is selected.

Preserve unrelated existing tokens (`pcie@4_clk-scheme=1` etc.).
Append the `# custom-bsp: uphy` trailing marker.

**`UPHY_CONFIG` handling for `uphy0-config-6`** (REQUIRED — wrong here
bricks cold boot). `t<soc>.conf.common` sets a default
`UPHY_CONFIG=<carrier-dtsi>`. Many carrier confs ALSO have a local
`UPHY_CONFIG="...";` line that re-overrides it. Bash last-assignment
wins, so **both** must be neutralized:

1. Append `UPHY_CONFIG=""; # custom-bsp: uphy` immediately after
   `source "${LDK_DIR}/t<soc>.conf.common";`.
2. Comment any later uncommented `^UPHY_CONFIG=` line in the conf
   (`# custom-bsp: uphy (overridden by clear above for uphy0-config-6)`).

Without (2), the local override silently wins -> `UPHY_CONFIG`
non-empty -> StMM `BPMP firmware is not ready` -> BL31 `ASSERT
plat_setup.c:726` -> cold-boot reboot loop. (RCM/flash boot doesn't
exercise StMM, so flash appears to succeed.) For any other config,
leave existing `UPHY_CONFIG=` lines alone.

`git add` + customization commit; message names chosen config(s) +
tokens. Commit directly — no accept/edit/cancel preview gate. Print
the staged files + final commit message after the commit lands.

**ODMDATA token grammar — wrong shapes brick boot:**

| Class | Form | Examples |
|---|---|---|
| `/uphy` top-level | dashed end-to-end | `uphy0-config-N`, `mgbeN-speed-del`, `mgbeN-speed-<rate>` |
| `/pcie/pcie@N` sub-node | `<node>_<key>=<value>` underscore | `pcie@N_status=okay`, `pcie@N_max-link-speed=4` |

Never mix namespaces — a single rejected token drops the WHOLE
`ODMDATA="..."` line.

**Three wrong forms — DO NOT use:**

| Wrong form | Failure |
|---|---|
| `mgbe0_speed-del` (underscore between controller and field) | Wrong namespace; whole line dropped. Use `mgbe0-speed-del`. |
| `mgbe0-speed-0` (dashed top-level "value") | Rejected; whole line dropped. `wait-for-device failed`; BPMP DTB unchanged. |
| `mgbe0_speed=0` | Sets literal `0` -> FMON armed on dead PHY -> BL31 SError reboot loop. |

**Missing token > wrong token.** Only emit if the BPMP DTB schema shows
the field AND the emission flips a state you want changed. Inspect:

```bash
BPMP_DTB="<bsp_image.root_path>/Linux_for_Tegra/bootloader/$(grep '^BPFDTB_FILE=' $CONF | cut -d'"' -f2)"
dtc -I dtb -O dts "$BPMP_DTB" | sed -n '/^\t\(pcie\|uphy\|mgbe\|ufs\) /,/^\t};/p'
```

Properties absent from the BPMP DTB (e.g. PCIe `num-lanes`, `pcie-mode`)
**cannot** be expressed via ODMDATA — they belong on the overlay
surface owned by the dispatched per-controller skills in Step 7.

## Step 7 — Build the per-controller allocation table and dispatch

ODMDATA was committed atomically in Step 6. Step 7 builds the
per-controller allocation table (used by both Step 6's token
emission and the sub-skills' overlay emission), snapshots K-stock
from the reference kernel DTB, and dispatches sub-skills for the
kernel-DT overlay surface only.

1. **Enumerate every UPHY-fed controller.** Thor: `pcie@C0..C5`,
   `mgbe0..3`, `ufshci`, xusb_padctl `usb_ss0..2`. Orin: derive from
   SoC TRM UPHY block diagram.

2. **Snapshot stock state from the reference DTBs.**

   Decompile both DTBs once:

   - Kernel DTB: `<bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/tegra<soc>-*-nv.dtb`
     -> `K-stock` per node.
   - BPMP DTB: `<bsp_image.root_path>/Linux_for_Tegra/bootloader/$(grep '^BPFDTB_FILE=' $CONF | cut -d'"' -f2)`
     -> `BPMP-stock` per node (`/pcie/pcie@N/status`, `/uphy/mgbeN-speed`).

   Resolve authoritative kernel-DT node addresses from the decompiled
   kernel DTB — TRM lists register bases but DT `reg` may differ; DTB
   is ground truth.

3. **Compute the per-controller allocation table** (print it before
   dispatching). One row per UPHY-fed controller:

   | Class | Instance | Node path (kernel DTB) | Allocated by chosen cfg? | BPMP-stock | K-stock | Schematic-routed? | Desired K state |
   |---|---|---|---|---|---|---|---|

   `Allocated by chosen cfg?` is derived from the chosen
   `uphy0-config-N` / `uphy1-config-N` lane mapping out of the
   Adaptation Guide. `Schematic-routed?` is always `yes` or `no` (no
   `unknown` — Step 1 enforces the schematic when `custom_carrier:` is
   present). `Desired K state` is:

   - `okay` when the controller is allocated AND `routed=yes`,
   - `disabled` when unallocated OR `routed=no`.

   This rule has no fallback branch and never freezes `Desired K state`
   to `K-stock` — BPMP-vs-kernel divergence is the failure mode the
   mandatory schematic exists to prevent.

4. **Dispatch per-controller skills.** Group rows by class and hand off
   — kernel-DT overlay surface only:

   | Sub-skill | Rows passed | Owns overlay fragments |
   |---|---|---|
   | `/jetson-customize-pcie` | `pcie@C0..N` | `pcie@<addr>` `status` / `num-lanes` / `max-link-speed` |
   | `/jetson-customize-mgbe` | `mgbe0..N` | `ethernet@<addr>` `status` + PHY plumbing |
   | `/jetson-customize-usb`  | `usb_ss0..N` | xusb_padctl `usb_ss<N>` `status` + companion cascade |

   See Step 6 for ODMDATA ownership rules. Dispatch is automatic —
   invoke each applicable sub-skill inline without prompting the
   operator. **Never stop, ask, or defer** between sub-skills citing
   "minimal changes" or context length; the ODMDATA commit alone is
   incomplete without the K-DT overlays and leaves BPMP-vs-kernel
   divergent — the exact failure mode the skill exists to prevent.

   UFS handling stays inline in this skill (no UFS sub-skill); emit
   `ufshci@<addr>` overlay fragment only when the K-stock vs
   `Desired K state` delta is non-empty.

5. **Pre-commit contract enforced on every sub-skill invocation
   (HARD GATE).** Each sub-skill MUST:

   - Re-read its controller's K-stock from the reference kernel DTB
     (do not trust the table values blindly — verify before emitting).
   - **Skip overlay emission when K-stock already matches
     `Desired K state`** (no fragment, no commit).
   - When K-stock disagrees, append exactly one fragment per controller
     to the composite overlay
     (`<source.root_path>/bsp_sources/hardware/nvidia/<chip>/nv-public/overlay/<chip>-<custom-id>-<custom-sku>+<module-id>-<module-sku>.dts`)
     with marker `uphy:<class><id>`. On re-run, delete every fragment
     matching that marker before appending so a diff-only set remains.
   - Never touch `ODMDATA` (see Step 6).

**Dry-run guard (fail-closed before dispatch):** walk the table for
every controller; refuse to dispatch if any row's `Desired K state`
contradicts the chosen `uphyX-config-N` allocation (impossible
combinations indicate a misread Adaptation Guide).

## Step 8 — Summary

**Headline** (mandatory first line):

```
jetson-customize-uphy: <uphy0-config-N>[ + <uphy1-config-M>] written to <CONF basename>; dispatched <N> sub-skill(s).
```

**Breakdown:** active profile, reference devkit, custom carrier
(id-sku), carrier flash-conf fork path, ODMDATA tokens this skill
applied (uphyX-config-N only), `UPHY_CONFIG=""` clear applied? (yes
only for `uphy0-config-6`), conflicting `UPHY_CONFIG=` lines
commented? (yes/n/a), sidecar path (new/refreshed/unchanged),
`sources[]`, `warnings[]` / `notes[]`.

**Choices table (mandatory):**

| UPHY surface | Chosen config | Lane summary | UPHY_CONFIG clear |
|---|---|---|---|
| UPHY0 | `uphy0-config-N` | <one-line lane mapping> | yes / no |
| UPHY1 *(Thor)* | `uphy1-config-M` | <one-line lane mapping> | n/a |

**Dispatch result table (mandatory):**

| Sub-skill | Rows passed | Overlay fragments emitted | Skipped (K-stock matched) | Commit SHA |
|---|---|---|---|---|
| `/jetson-customize-pcie` | … | … | … | … |
| `/jetson-customize-mgbe` | … | … | … | … |
| `/jetson-customize-usb`  | … | … | … | … |

**Changes summary table (mandatory — one row per touched file):**

| Repo | Path | Change | Commit SHA | Description |
|---|---|---|---|---|
| `Linux_for_Tegra` | `<carrier-conf>` | `~modified` | `<sha>` | ODMDATA + UPHY_CONFIG clear |
| `bsp_sources` | `<composite-overlay .dts>` | `~modified` / `+added` | `<sha>` | <sub-skill marker(s)> fragments |
| … | … | … | … | … |

**Next step (interactive prompt chain) — HARD GATE:**

After the summary tables are printed, you MUST drive the downstream
chain via sequential `AskUserQuestion` prompts. Each prompt needs an
explicit `yes`. On `no` (or any abort), print the remaining manual
run-chain and exit. **Never substitute a printed "Next step: …" /
"Next:" / "Then run …" line for the prompts** — printing the chain
as prose instead of asking is a skill bug, not a shortcut. This
gate is distinct from the intra-skill commit/dispatch flow (which
is auto, no prompts).

In order:

1. **Customize any other I/O before build?** — offer
   `/jetson-customize-pinmux`, `/jetson-customize-camera`,
   `/jetson-customize-pcie`, `/jetson-customize-mgbe`,
   `/jetson-customize-usb`, `/jetson-customize-clocks`,
   `/jetson-customize-fan`, `/jetson-customize-nvpmodel`,
   `/jetson-customize-memory`, and `no` (proceed). On any non-`no`
   pick, invoke the chosen sub-skill inline; when it returns, re-ask
   this same question (loop) until the user picks `no`. Then continue
   to (2).
2. **Build & promote?** — on `yes` invoke `/jetson-build-source`, then
   on success invoke `/jetson-promote-image`.
3. **Flash the board?** — only offer if (2) ran and succeeded; on
   `yes` invoke `/jetson-flash-image`.
4. **Validate on the DUT?** — only offer if (3) ran and succeeded; on
   `yes` invoke `/jetson-validate-image`.

Idempotent re-runs (no diff) emit `(no change)` and skip this chain.
