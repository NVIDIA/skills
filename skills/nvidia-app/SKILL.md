---
name: nvidia-app
description: >
  NVIDIA App MCP: drivers, games, laptops, overlay. Check drivers, manage and
  optimize games, configure laptop features.
metadata:
  author: "NVIDIA CORPORATION <info@nvidia.com>"
  tags:
    - nvidia-app
    - in-game-overlay
  domain: system-tools
  team: nvidia-app
  version: "1.1.8"
---

# NVIDIA App MCP Overlay

## Purpose

Use this skill for one product integration: operating NVIDIA App through its local MCP server. It has two routed modes:

- General NVIDIA App operations cover application listing and launch, driver status and release notes, per-game optimization, and laptop features.
- In-Game Overlay operations use the workflow below.

The Overlay workflow supports:

- Read high-level Overlay status.
- Start or stop gameplay recording.
- Enable, disable, or save Instant Replay.
- Enable or disable Highlights.
- Capture a screenshot to the Overlay gallery.
- Show or hide the Statistics Overlay.
- Enable or disable the RTX Dynamic Vibrance filter.
- Identify the currently running game.

Connection setup supports both modes but does not authorize additional actions.

## When to use this skill

Use this skill when the user asks for one of the supported application, driver, game-optimization, laptop-feature, Overlay, or connection workflows.

Do not select this skill for Desktop Capture, other Overlay filter mutations, NVIDIA Broadcast, OBS, Xbox Game Bar, or other capture software.

## Inputs

- **Required:** A user request for a supported NVIDIA App query, mutation, or connection diagnostic.
- **Optional:** An explicit target state, game or program identifier, tool-specific values, and client connection details. Obtain authentication tokens only from the protected discovery source described in [references/connection.md](references/connection.md); never accept or expose them as ordinary tool arguments.

Resolve information in this order: explicit user instructions, fresh live MCP schema or state, current client connection configuration, then the applicable self-contained skill reference for static contract details. Never treat a previous tool result as current state.

## Prerequisites

- Windows host with NVIDIA App installed (version 11.0.9.5xx or above) and its NvContainer-hosted MCP plugin running.
- Standard NVIDIA App executable path: `%ProgramFiles%\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA App.exe`.
- MCP server access is enabled when NVIDIA App exposes a master toggle.
- NVIDIA In-Game Overlay is enabled, running, and ready before any `nvapp_overlay_` call.
- The MCP client supports Streamable HTTP over loopback or connects through the installed stdio bridge.

MCP server readiness and Overlay readiness are separate. A successful MCP connection does not establish that Overlay tools are ready.

## Instructions

1. **Classify the request.** Route general NVIDIA App requests through live tool discovery. Route `nvapp_overlay_` requests through the Overlay workflow. Read-only Overlay status can include fields for excluded features, but that does not authorize changing them.
2. **Register `nvidia-app` in the current MCP client, or fall back explicitly.** The client is registered only when this session's live tool catalog advertises `nvapp_` tools; an open loopback port or a working stdio bridge is not registration. If those tools are absent, attempt registration per [references/connection.md](references/connection.md) before the first `nvapp_` call. Registration is the default path because it persists across turns and sessions. A reachable HTTP endpoint is not a reason to skip the attempt.
3. **Use the direct connection only as a declared fallback.** If the user declines the configuration change, the client cannot be modified or reloaded, or registration succeeds but `nvapp_` tools still do not appear in this session, connect directly to the loopback HTTP endpoint (or the stdio bridge) and continue with the same documented tool contracts. Say which path you used and that registering `nvidia-app` would make access persistent. Do not silently prefer the direct path.
4. **Discover general tools.** For a non-Overlay request, inspect `tools/list` on whichever connection you established, then read [references/general-tools.md](references/general-tools.md) before selecting or calling one of its seven documented public tools. Select only a returned tool whose description, schema, and annotations most narrowly match the user's in-scope intent, and follow any narrower live schema and annotations.
5. **Resolve an Overlay target state.** For an explicit start, stop, enable, disable, show, or hide request, call the mapped mutation directly. For a true toggle with no target state, read status once and invert only the corresponding Boolean.
6. **Validate the arguments.** For general tools, follow [references/general-tools.md](references/general-tools.md) and any narrower live schema. For Overlay tools, use exactly the mapped fields below. Never invent unsupported arguments.
7. **Execute in request order.** Fulfill compound requests by issuing each supported operation as its own tool call in the user's order. Make only the calls needed for the requested operations. Respect the access tier and any consent decision; never raise access or enable another feature as a workaround.
8. **Interpret the result literally.** Report only returned fields and outcomes. Overlay mutations return a human-readable message; report that message without inventing structured success fields. If a mutation message is ambiguous, state that the outcome is not independently confirmed instead of making a follow-up Overlay status call solely to verify it.

## Connecting




The default HTTP endpoint is:

```text
http://127.0.0.1:13508/mcp
```

The installed stdio bridge is:

```text
%ProgramFiles%\NVIDIA Corporation\NVIDIA App\McpServer\NvAppMcpServer.exe
```

The stdio executable relays to the persistent server; it does not start or enable that server.

## General NVIDIA App operations

Read [references/general-tools.md](references/general-tools.md) before selecting or calling a general NVIDIA App tool. It contains the seven supported tool names, exact arguments, enums, conditional schemas, and prerequisite checks.

Use only a documented tool advertised by live `tools/list`. The packaged instructions are self-contained and do not depend on repository-internal schema sources.

## Overlay operations

Use only the tools in this table and the authorized arguments shown here:

| User intent | Tool | Arguments |
|---|---|---|
| Read overall Overlay status | `nvapp_overlay_get_status` | `{}` |
| Start recording | `nvapp_overlay_capture` | `{ "action": "toggle_recording", "enable": true }` |
| Stop and save recording | `nvapp_overlay_capture` | `{ "action": "toggle_recording", "enable": false }` |
| Enable Instant Replay | `nvapp_overlay_capture` | `{ "action": "toggle_instant_replay", "enable": true }` |
| Disable Instant Replay | `nvapp_overlay_capture` | `{ "action": "toggle_instant_replay", "enable": false }` |
| Save Instant Replay | `nvapp_overlay_capture` | `{ "action": "save_instant_replay" }` |
| Enable Highlights | `nvapp_overlay_capture` | `{ "action": "toggle_highlights", "enable": true }` |
| Disable Highlights | `nvapp_overlay_capture` | `{ "action": "toggle_highlights", "enable": false }` |
| Capture a screenshot | `nvapp_overlay_capture` | `{ "action": "capture_screenshot" }` |
| Show Statistics Overlay | `nvapp_overlay_configure_stats` | `{ "enable": true }` |
| Hide Statistics Overlay | `nvapp_overlay_configure_stats` | `{ "enable": false }` |
| Enable RTX Dynamic Vibrance | `nvapp_overlay_configure_filters` | `{ "filter": "rtx_dvc", "enable": true }` |
| Disable RTX Dynamic Vibrance | `nvapp_overlay_configure_filters` | `{ "filter": "rtx_dvc", "enable": false }` |
| Get current game | `nvapp_overlay_get_current_game_info` | `{}` |

Never invoke `toggle_desktop_capture`. For `nvapp_overlay_configure_filters`, use only the `rtx_dvc` filter value.

## Capture and privacy

`toggle_recording`, `toggle_instant_replay`, `save_instant_replay`, `toggle_highlights`, and `capture_screenshot` record screen content and write media files to the user's system. Recordings and screenshots persist in the NVIDIA App Overlay gallery until the user deletes them, and this skill cannot choose their format, resolution, or destination.

Treat every one of them as a privacy-affecting operation:

- An explicit user request to start, stop, save, enable, or capture is the authorization. Make that single mapped call without an additional confirmation prompt, then report the returned message so the user knows capture began or media was written.
- Never initiate capture the user did not ask for. Do not start recording as a side effect of another request or as a workaround for a failed operation.
- Ask first when the request is ambiguous, when a true toggle has no target state and the status field is missing, or when an unsupported qualifier such as a format or destination applies.
- Enabling Instant Replay or Highlights leaves background capture running until it is disabled. Say so when you enable either one.
- Do not enable Desktop Capture to make a capture succeed. Content outside a supported game stays out of scope for this skill.

## Decision rules

- If this session's MCP catalog has no `nvapp_` tools, attempt `nvidia-app` registration before connecting directly. Once registration is refused or has failed, a direct HTTP or stdio connection is allowed for the rest of the session without re-asking.
- Whichever connection you use, honor the same tool names, schemas, access tier, and Overlay restrictions. The fallback changes the transport only, never the authorized surface.
- For a true request to toggle recording, Instant Replay, Highlights, Statistics Overlay, or RTX Dynamic Vibrance, call `nvapp_overlay_get_status` once and invert only `recordingActive`, `instantReplayEnabled`, `highlightsEnabled`, `statsOverlayEnabled`, or `rtxDvc`, respectively.
- If the required status field is absent, ask whether to enable or disable instead of guessing. Never convert an omitted field to `false`.
- `nvapp_overlay_capture` accepts one action, not an action list.
- Do not add `enable` to `save_instant_replay` or `capture_screenshot`.
- Do not omit `enable` from `toggle_recording`, `toggle_instant_replay`, `toggle_highlights`, or `nvapp_overlay_configure_stats`.
- Pass both `filter: "rtx_dvc"` and `enable` to `nvapp_overlay_configure_filters`; never invent another filter value.
- For a mixed request, execute independent supported operations in order and explain or skip unsupported operations; an unsupported sibling request must not block a fully specified supported action.
- When an unsupported qualifier modifies an otherwise supported state-changing operation, explain the limitation and ask whether to proceed with that supported subset before making its mutation call.
- `running: false` means no current game was identified. Do not invent `gameName` or `processId`. Report `fullscreen` only when it is returned.

## Output Format

- Follow the user's requested format; otherwise give a concise human-readable result.
- For reads, report only structured fields that are present and distinguish an omitted field from `false` or `null`.
- For failures, state that the operation failed and preserve the returned error code, retryability, and useful guidance.
- Never expose authentication material. When protocol-level detail is needed, show only the request or response shape and replace tokens, authorization headers, and other secrets with placeholders such as `<redacted-token>`.

## Limitations

- Installed NVIDIA App versions and access levels can expose different general tools and output contracts; live `tools/list` is authoritative.
- MCP readiness does not guarantee Overlay readiness, and a successful mutation message does not prove more than the returned result.
- The Overlay workflow cannot mutate Desktop Capture or filters other than RTX Dynamic Vibrance, choose screenshot format or destination, or configure Statistics Overlay layout and styling.
- Missing fields are unknown, not implicit false values, and unavailable tools or backend timeouts must not be replaced with inferred data.

## Examples

### Get driver status

For a request such as "Show my NVIDIA driver status," first check whether the current MCP client advertises `nvapp_` tools. If it does not, offer to register `nvidia-app`; if the user declines or registration is not possible, connect directly to the loopback endpoint and say so. Then call `tools/list` on that connection and confirm `nvapp_client_get_driver_status` is advertised with an empty-object input schema. If it is unavailable, explain that the connected NVIDIA App version or access level does not expose this workflow; do not substitute a similarly named tool without validating its live schema.

Invoke:

```json
{
  "name": "nvapp_client_get_driver_status",
  "arguments": {}
}
```

Use the returned structured fields literally:

- Report `systemType`, `installedDriver.version`, optional `installedDriver.channel`, and `installedDriver.availableActions`.
- Report `preferredChannel`.
- For each item in `recommendedDrivers`, report `name`, `version`, `channel`, `releaseDate`, `isUpdateAvailable`, and `availableActions`.
- If `previouslyInstalledDriver` is `null`, state that NVIDIA App reports no rollback driver is available. Otherwise, report its `name`, `version`, `releaseDate`, and `availableActions`.

This tool is read-only. Do not invoke a driver mutation; none is included in this seven-tool scope.

### Start recording with an explicit target state

Call the mutation directly because the requested target state is already known:

```json
{ "name": "nvapp_overlay_capture", "arguments": { "action": "toggle_recording", "enable": true } }
```

Do not read status before the call or issue a verification read afterward. The explicit request is the authorization, so no extra confirmation prompt is needed, but this call starts screen capture and writes a file; report the returned message and see [Capture and privacy](#capture-and-privacy).

### Toggle Instant Replay without a target state

First call `nvapp_overlay_get_status` with `{}`. If `instantReplayEnabled` is present, call `nvapp_overlay_capture` once with `enable` set to its opposite. If the field is absent, ask whether to enable or disable Instant Replay and make no mutation yet.

### Handle an unsupported screenshot qualifier

For “Take a PNG screenshot in `D:\Shots`,” explain that the tool cannot select the format or destination. Ask whether to capture to the Overlay gallery. After confirmation, call:

```json
{ "name": "nvapp_overlay_capture", "arguments": { "action": "capture_screenshot" } }
```

## Troubleshooting

| Error or symptom | Likely cause | Response |
|---|---|---|
| `service_disabled` | Persistent MCP server access is disabled | Ask the user to enable MCP server access in NVIDIA App Settings; do not repeatedly spawn bridges. |
| `overlay_not_available` | MCP is connected but In-Game Overlay is not ready | Ask the user to enable or start In-Game Overlay, then retry only if requested. |
| `overlay_timeout` | Overlay did not answer before the deadline | Preserve `retriable`; retry at most once when the result allows retrying and the retry remains within the request, then report the failure and ask before trying again. |
| `access_level_restricted` | Current MCP access ceiling blocks the tool | Report the restriction; do not raise access implicitly. |
| `isError: true` or another backend/validation failure | The requested operation did not succeed | Preserve the message and error code; do not infer data or report success. |

Overlay mutations return a human-readable message rather than structured success fields. Report that message literally and do not infer additional state. If saving Instant Replay reports that it is disabled, ask before enabling it. If capture requires Desktop Capture, report the limitation rather than enabling Desktop Capture.

## References

- Read [references/connection.md](references/connection.md) for client configuration, ports, discovery and authentication, protocol lifecycle, readiness, and connection troubleshooting.
- Read [references/general-tools.md](references/general-tools.md) for live discovery, schema use, routing, and result handling for non-Overlay NVIDIA App operations.
- Read [references/overlay-capture.md](references/overlay-capture.md) for recording, Instant Replay, Highlights, screenshots, unsupported qualifiers, access, and capture errors.
- Read [references/overlay-state.md](references/overlay-state.md) for status routing, true toggles, Statistics Overlay visibility, RTX Dynamic Vibrance, and current-game lookup.
- Read [references/mcp-tool-contract.md](references/mcp-tool-contract.md) for the canonical status field set and when exact Overlay schemas, outputs, access tiers, or error forms are needed.
