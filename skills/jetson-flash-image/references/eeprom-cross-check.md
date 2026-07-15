# EEPROM cross-check vs. active profile

Detailed reference for `../SKILL.md`'s "Preflight checks" step (EEPROM cross-check sub-aspect): how to
read the DUT's EEPROM in recovery mode, map the printed labels to
profile dispatch inputs, and reconcile EEPROM values against the
active profile.

## Read the EEPROM

From `<bsp_image.root_path>/Linux_for_Tegra/`:

```bash
sudo ./nvautoflash.sh --print_boardid
```

`nvautoflash.sh --print_boardid` is a read-only preflight helper that
walks the recovery channel and prints the module + baseboard EEPROM;
it does **not** flash. The actual flash tool (`flash.sh` /
`l4t_initrd_flash.sh`) is selected by the skill's "Select `<boot-dev>` and flash flow" matrix.

Sample output:

```
--- Reading board information succeeded.
--- Parsing chip_info.bin information succeeded.
Chip SKU(00:00:00:A0) ramcode(00:00:00:0C)
Parsing module EEPROM:
--- Parsing board ID (3834) succeeded.
--- Parsing board version (RC1) succeeded.
--- Parsing board SKU (0008) succeeded.
--- Parsing board REV (C.1) succeeded.
Parsing baseboard EEPROM:
--- Parsing baseboard ID (4071) succeeded.
jetson-agx-thor-devkit found.
```

## Label-to-dispatch-input mapping

| Printed label    | Dispatch input  |
|------------------|-----------------|
| board SKU        | `board_sku`     |
| board version    | `board_FAB`     |
| board REV        | `board_revision`|
| board ID         | `board_id`      |
| baseboard ID     | `baseboard_id`  |

A label printed with empty parentheses (e.g. `Parsing board SKU ()
succeeded.`) is the **empty** case — that is a valid value, not an
error.

## Empty / unset values are valid

**Empty / unset values are valid and not a refusal trigger.**
`board_FAB` is typically empty for production revisions; some
pre-production or unprogrammed EEPROMs return empty for `board_sku`
as well. The per-board conf dispatch treats empty as "production base
SKU" and uses default artifacts.

## Reconciliation table

For each dispatch input, reconcile the EEPROM value against the
active block's corresponding field (`module.sku` / `module.revision`)
per the table:

| EEPROM     | Profile                    | Action |
|------------|----------------------------|--------|
| non-empty  | non-empty, equal           | accept silently |
| non-empty  | non-empty, differ          | **refuse** with a printed diff |
| non-empty  | empty / `Any` / missing    | accept; prompt once to persist EEPROM value into profile |
| empty      | non-empty                  | accept; show "EEPROM: (empty) / Profile: `<value>`" in the "Confirm resolution" step — flash.sh will use base-SKU defaults regardless of what the profile asserts |
| empty      | empty / `Any` / missing    | accept; flash.sh will use base-SKU defaults |

Refusal trigger is a *real non-empty disagreement*, never a missing
value. The cross-check remains the primary defense against the
wrong-target / wrong-SKU class of failures whenever both sides
supply enough information to disagree.

## Don't override profile from the detected-board line

The detected-board summary line (e.g. `jetson-agx-thor-devkit
found.`) is informational — do **not** use it to override the
profile's `reference_devkit.name`. The reconciliation that matters is
the per-field table above; the summary line just names the BSP's
best guess from board+baseboard IDs.
