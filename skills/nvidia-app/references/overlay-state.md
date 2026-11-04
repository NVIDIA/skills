# Overlay Status, Statistics, and Current Game

Use this reference for `nvapp_overlay_get_status`, `nvapp_overlay_configure_stats`, `nvapp_overlay_configure_filters`, and `nvapp_overlay_get_current_game_info`.

## Overall status

Call `nvapp_overlay_get_status` with `{}` when the user asks for Overlay status or when a true toggle requires the current state.

[mcp-tool-contract.md](mcp-tool-contract.md) is the canonical source for the returned field set, its types and enums, the fields that are read-only within this skill, and the field-to-tool toggle mapping. Read the contract there instead of a second copy of the table.

This file covers only the operational routing:

- Reading excluded capability state does not authorize changing it.
- Do not use status as a routine pre-read for an explicit target state or as a verification call after a mutation.

## Statistics Overlay

Use `nvapp_overlay_configure_stats` with exactly one argument:

```json
{ "enable": true }
```

or:

```json
{ "enable": false }
```

The tool controls visibility only. It does not configure position, layout, opacity, font, logging, displayed metrics, or output paths.

For a true "toggle the Statistics Overlay" request, call status once and invert `statsOverlayEnabled`. If that field is missing, ask whether to show or hide it before making a mutation call.

Report the mutation's returned human-readable message; do not invent structured success fields or issue a verification read.

If the request also asks for an unsupported presentation change, explain the limitation and obtain confirmation before applying visibility alone.

## RTX Dynamic Vibrance

Use `nvapp_overlay_configure_filters` only for RTX Dynamic Vibrance. Explicit enable or disable requests go directly to the mutation tool. For a true toggle request, call status once, invert `rtxDvc` when present, and ask for the target state when it is absent. Do not issue a verification read.

## Current game

Use `nvapp_overlay_get_current_game_info` with no arguments. Treat this as a current runtime lookup, not a library, play-history, or game-settings query. Report only returned fields; when `running` is false, do not invent game identity.

## Access and readiness

- `nvapp_overlay_get_status` and `nvapp_overlay_get_current_game_info` require read-only access.
- `nvapp_overlay_configure_stats` and `nvapp_overlay_configure_filters` require standard access or higher.
- All three still require the In-Game Overlay-ready event.

An MCP connection can succeed while these tools return `overlay_not_available`. Report that NVIDIA In-Game Overlay is not ready and do not claim that a read or mutation ran.
