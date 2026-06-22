#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DTSI block editing helpers for modify_pinmux (pinmux + gpio default-state)."""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from _pinmux_common import SOC_PREFIX


# ===== pinmux DTSI block =================================================

_PULL_VAL = {"none": "TEGRA_PIN_PULL_NONE",
             "pull-up": "TEGRA_PIN_PULL_UP",
             "pull-down": "TEGRA_PIN_PULL_DOWN"}
_DRV_VAL = {"normal": "TEGRA_PIN_1X_DRIVER",
            "high": "TEGRA_PIN_2X_DRIVER"}
_OD_VAL = {"disable": "TEGRA_PIN_DISABLE",
           "enable": "TEGRA_PIN_ENABLE"}
_LPBK_VAL = {"disable": "TEGRA_PIN_DISABLE",
             "enable": "TEGRA_PIN_ENABLE"}


@dataclass
class PinPatchSpec:
    """Required fields for patching a single pinmux block."""
    pin: str
    sfio: str
    direction: str
    initial_state: str


@dataclass
class PinPatchOpts:
    """Optional pinmux block overrides (None means 'leave at XLSM default')."""
    pull: str | None = None
    drive_type: str | None = None
    open_drain: str | None = None
    loopback: str | None = None


def _patch_function(block: str, sfio: str) -> str:
    """Set or insert nvidia,function in a pinmux block."""
    new_block, n = re.subn(
        r'(nvidia,function\s*=\s*)"[^"]*"',
        rf'\1"{sfio}"', block, count=1)
    if n == 0:
        new_block = re.sub(
            r'(nvidia,pins\s*=\s*"[^"]*";\s*)',
            lambda mm: mm.group(1) + f'\n\t\t\t\tnvidia,function = "{sfio}";\n\t\t\t\t',
            new_block, count=1)
    return new_block


def _patch_optional(block: str, opts: PinPatchOpts) -> str:
    """Re-drive pull/drv-type/e-io-od/e-lpbk when fields are set."""
    if opts.pull is not None and opts.pull in _PULL_VAL:
        block, _ = re.subn(
            r'(nvidia,pull\s*=\s*)<[^>]*>',
            rf'\1<{_PULL_VAL[opts.pull]}>', block, count=1)
    if opts.drive_type is not None and opts.drive_type in _DRV_VAL:
        block, _ = re.subn(
            r'(nvidia,drv-type\s*=\s*)<[^>]*>',
            rf'\1<{_DRV_VAL[opts.drive_type]}>', block, count=1)
    if opts.open_drain is not None and opts.open_drain in _OD_VAL:
        block, _ = re.subn(
            r'(nvidia,e-io-od\s*=\s*)<[^>]*>',
            rf'\1<{_OD_VAL[opts.open_drain]}>', block, count=1)
    if opts.loopback is not None and opts.loopback in _LPBK_VAL:
        block, _ = re.subn(
            r'(nvidia,e-lpbk\s*=\s*)<[^>]*>',
            rf'\1<{_LPBK_VAL[opts.loopback]}>', block, count=1)
    return block


def _insert_marker(block: str) -> str:
    """Insert the `// custom-bsp: pinmux` marker idempotently."""
    if "// custom-bsp: pinmux" in block:
        return block
    return re.sub(
        r"(\n)([ \t]*)(\}\;)",
        lambda mm: (mm.group(1) + mm.group(2) + "\t// custom-bsp: pinmux"
                    + mm.group(1) + mm.group(2) + mm.group(3)),
        block, count=1)


def patch_pinmux_block(text: str, spec: PinPatchSpec,
                       opts: PinPatchOpts) -> tuple[str, bool]:
    """Patch the pinmux block for `spec.pin` in `text`. Returns (new_text, found).

    Idempotent. Re-drives nvidia,function, nvidia,enable-input, nvidia,tristate,
    and optional pull/drv-type/e-io-od/e-lpbk overrides from opts.
    """
    block_re = re.compile(
        rf"(^[ \t]*{re.escape(spec.pin)}[ \t]*\{{[^}}]*?^[ \t]*\}};)",
        re.MULTILINE | re.DOTALL,
    )
    m = block_re.search(text)
    if not m:
        return text, False

    block = m.group(1)
    block = _patch_function(block, spec.sfio)

    enable_input = spec.direction in ("input", "bidirectional")
    tristate = (spec.direction == "unused") or (spec.initial_state == "hi-z")
    ei_val = "TEGRA_PIN_ENABLE" if enable_input else "TEGRA_PIN_DISABLE"
    ts_val = "TEGRA_PIN_ENABLE" if tristate else "TEGRA_PIN_DISABLE"

    block, _ = re.subn(
        r'(nvidia,enable-input\s*=\s*)<[^>]*>',
        rf'\1<{ei_val}>', block, count=1)
    block, _ = re.subn(
        r'(nvidia,tristate\s*=\s*)<[^>]*>',
        rf'\1<{ts_val}>', block, count=1)

    block = _patch_optional(block, opts)
    block = _insert_marker(block)

    return text[:m.start()] + block + text[m.end():], True


# ===== gpio default-state DTSI ==========================================

_GPIO_NODE_BY_KIND = {
    "thor": {"MAIN": "gpio@ac300000", "AON": "gpio@e8300000"},
    "orin": {"MAIN": "gpio@2200000", "AON": "gpio@c2f0000"},
}


def gpio_macro_for(platform: str, bank: str, idx: int, kind: str) -> str:
    """Return the TEGRA*_*_GPIO(BANK, IDX) macro string for a gpio pin."""
    soc = SOC_PREFIX.get(platform, "tegra264")
    family = "TEGRA264" if soc == "tegra264" else "TEGRA234"
    macro_kind = "AON_GPIO" if kind == "AON" else "MAIN_GPIO"
    return f"{family}_{macro_kind}({bank}, {idx})"


def _sublist_for(direction: str, initial_state: str) -> str | None:
    """Pick the gpio-input / gpio-output-* sub-list for a direction+state."""
    if direction == "input":
        return "gpio-input"
    if direction == "output" and initial_state == "low":
        return "gpio-output-low"
    if direction == "output" and initial_state == "high":
        return "gpio-output-high"
    return None


def _resolve_gpio_target(platform: str, edit: dict, warnings: list[str]
                         ) -> tuple[str, str, str] | None:
    """Resolve (node, sublist, macro) for one edit. Returns None on skip."""
    gp = edit.get("_gpio")
    pin_name = edit.get("pin")
    if not gp:
        warnings.append(
            f"pin {pin_name!r}: no gpio bank/idx parsed; skipped from gpio DTSI"
        )
        return None
    bank, idx, kind = gp
    node = _GPIO_NODE_BY_KIND.get(platform, {}).get(kind)
    if not node:
        warnings.append(
            f"pin {pin_name!r}: unknown gpio controller "
            f"kind={kind} for platform={platform}; skipped"
        )
        return None
    sublist = _sublist_for(edit.get("direction"), edit.get("initial_state"))
    if sublist is None:
        warnings.append(
            f"pin {pin_name!r}: direction={edit.get('direction')!r} "
            f"initial_state={edit.get('initial_state')!r} has no matching "
            f"gpio-init sub-list; skipped"
        )
        return None
    macro = gpio_macro_for(platform, bank, idx, kind)
    return node, sublist, macro


def _append_one_gpio(text: str, platform: str, edit: dict,
                     warnings: list[str]) -> str:
    """Append a single gpio-init entry. Returns updated text; mutates warnings."""
    target = _resolve_gpio_target(platform, edit, warnings)
    if target is None:
        return text
    node, sublist, macro = target
    if re.search(rf"\b{re.escape(macro)}\b", text):
        return text

    node_re = re.compile(
        rf"({re.escape(node)}\s*\{{.*?\bdefault\b\s*\{{.*?{sublist}\s*=\s*<)"
        rf"([^>]*)(>;)",
        re.DOTALL,
    )
    m = node_re.search(text)
    if not m:
        warnings.append(
            f"pin {edit.get('pin')!r}: no `{node} {{ default {{ {sublist} = "
            f"<...>; ...` block in gpio DTSI; skipped"
        )
        return text
    body = m.group(2)
    macro_line_marker = f"{macro} // custom-bsp: pinmux"
    if macro_line_marker in body:
        return text
    new_body = body.rstrip()
    if not new_body.endswith("\n"):
        new_body += "\n"
    new_body += f"\t\t\t\t{macro_line_marker}\n\t\t\t\t"
    return text[:m.start()] + m.group(1) + new_body + m.group(3) + text[m.end():]


def patch_gpio_block(text: str, platform: str,
                     edits: list[dict]) -> tuple[str, list[str]]:
    """Append/update gpio-init entries for `edits` whose sfio == 'gpio'.

    Returns (new_text, warnings). Idempotent.
    """
    warnings: list[str] = []
    for edit in edits:
        if edit.get("sfio") != "gpio":
            continue
        text = _append_one_gpio(text, platform, edit, warnings)
    return text, warnings


# ===== file resolution ==================================================

def resolve_dtsi(l4t_dir: pathlib.Path, session: dict,
                 kind: str) -> pathlib.Path | None:
    """Resolve the cloned pinmux or gpio DTSI for the carrier."""
    explicit_key = f"cloned_{kind}_dtsi"
    explicit = session.get(explicit_key)
    if explicit:
        p = pathlib.Path(explicit)
        if p.exists():
            return p

    platform = session.get("platform") or "thor"
    soc = SOC_PREFIX.get(platform, "tegra264")
    carrier = session.get("carrier_short") or ""
    if not carrier:
        return None

    boot = l4t_dir / "bootloader"
    patterns = [
        f"{soc}-mb1-bct-{kind}-*-{carrier}.dtsi",
        f"{soc}-mb1-bct-{kind}-*{carrier}*.dtsi",
    ]
    seen: list[pathlib.Path] = []
    for pat in patterns:
        for p in boot.glob(pat):
            if p not in seen:
                seen.append(p)
        if seen:
            break

    if not seen:
        return None
    if len(seen) > 1:
        return None
    return seen[0]
