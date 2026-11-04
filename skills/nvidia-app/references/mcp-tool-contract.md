# NVIDIA App Overlay MCP Contract

## Scope

This reference documents only the Overlay functionality authorized for the `nvidia-app` skill:

```text
nvapp_overlay_get_status

nvapp_overlay_capture
  - toggle_recording
  - toggle_instant_replay
  - save_instant_replay
  - capture_screenshot
  - toggle_highlights

nvapp_overlay_configure_stats
nvapp_overlay_configure_filters
  - rtx_dvc
nvapp_overlay_get_current_game_info
```

The live `nvapp_overlay_capture` schema advertises `toggle_highlights`, which is authorized by this skill workflow. `toggle_desktop_capture` is not advertised and remains prohibited. `nvapp_overlay_configure_filters` supports only `rtx_dvc`; other filter values are not authorized. `nvapp_overlay_get_status` may report state for excluded capabilities, but that read access does not extend the mutation scope.

## Access and readiness

The MCP server derives access tiers from each tool's schema:

| Tool | Derived tier | Minimum role | Reason |
|---|---|---|---|
| `nvapp_overlay_get_status` | Safe | `readonly` | `readOnlyHint: true` |
| `nvapp_overlay_get_current_game_info` | Safe | `readonly` | `readOnlyHint: true` |
| `nvapp_overlay_configure_stats` | Normal | `standard` | State-changing and non-destructive; use only for an explicit user-requested visibility state |
| `nvapp_overlay_configure_filters` | Normal | `standard` | State-changing and non-destructive; use only for an explicit user-requested RTX Dynamic Vibrance state |
| `nvapp_overlay_capture` | Normal | `standard` | State-changing and non-destructive; use only for an explicit user-requested authorized action |

The global NVIDIA App MCP access-level ceiling can further hide or reject tools even when a client role would otherwise permit them.

Access availability does not authorize an unrequested mutation. Make a state-changing call only when it directly implements the user's explicit request. If the target state or supported subset is ambiguous, ask before calling.

The server checks the ShadowPlay Overlay-ready event before dispatching any Overlay tool, including both read-only tools. When the Overlay is disabled or not running, it returns:

```json
{
  "errorCode": "overlay_not_available",
  "retriable": true
}
```

If the dispatched Overlay call does not answer within the server deadline, it returns `overlay_timeout` with `retriable: true`. A call blocked by the configured access ceiling returns `access_level_restricted` with `retriable: false`.

Do not change the user's MCP access level or NVIDIA App configuration as an implicit recovery action.

## `nvapp_overlay_get_status`

### Input contract

```json
{}
```

### Output contract

The tool returns a best-effort snapshot containing any available subset of these fields:

```json
{
  "overlayView": "open",
  "recordingActive": false,
  "instantReplayEnabled": true,
  "desktopCapture": false,
  "highlightsEnabled": true,
  "statsOverlayEnabled": true,
  "statsLogging": false,
  "micMode": "off",
  "notificationsEnabled": true,
  "rtxDvc": false,
  "rtxHdr": true,
  "sharpening": false,
  "freestyle": false
}
```

Field meanings:

| Field | Type or enum | Meaning |
|---|---|---|
| `overlayView` | `open` or `closed` | Current high-level Overlay view state |
| `recordingActive` | Boolean | Manual recording state |
| `instantReplayEnabled` | Boolean | Instant Replay enabled state |
| `desktopCapture` | Boolean | Desktop Capture state; read-only within this skill |
| `highlightsEnabled` | Boolean | Highlights state; used to resolve a true toggle |
| `statsOverlayEnabled` | Boolean | Statistics Overlay visibility |
| `statsLogging` | Boolean | Statistics logging state; read-only within this skill |
| `micMode` | `on`, `off`, `always_on`, or `push_to_talk` | Normalized microphone mode |
| `notificationsEnabled` | Boolean | Global Overlay notification state |
| `rtxDvc` | Boolean | RTX Dynamic Vibrance state |
| `rtxHdr` | Boolean | RTX HDR state; read-only within this skill |
| `sharpening` | Boolean | Sharpening state; read-only within this skill |
| `freestyle` | Boolean | Whether a Freestyle game-filter slot is selected |

The schema intentionally declares no required fields because individual underlying reads are optional and undefined values are omitted. Never interpret a missing field as `false`.

Use this tool to answer status questions. For a true toggle request that omits the desired state, read one of these fields and set its opposite:

```text
recordingActive       -> toggle_recording
instantReplayEnabled  -> toggle_instant_replay
highlightsEnabled     -> toggle_highlights
statsOverlayEnabled   -> nvapp_overlay_configure_stats
rtxDvc                -> nvapp_overlay_configure_filters
```

If the necessary field is missing, ask for the target state. Do not call status before an explicit enable/disable request or after a mutation merely to verify it.

## `nvapp_overlay_capture`

### Input contract

```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": [
        "toggle_recording",
        "toggle_instant_replay",
        "save_instant_replay",
        "capture_screenshot",
        "toggle_highlights"
      ]
    },
    "enable": {
      "type": "boolean"
    }
  },
  "required": ["action"]
}
```

This is intentionally narrower than the server's live schema.

### Argument rules

| Action | `enable` rule | Effect |
|---|---|---|
| `toggle_recording` | Required Boolean | `true` starts recording; `false` stops and saves it |
| `toggle_instant_replay` | Required Boolean | Enables or disables Instant Replay |
| `save_instant_replay` | Must be absent | Saves the currently buffered Instant Replay |
| `capture_screenshot` | Must be absent | Captures a screenshot into the Overlay gallery |
| `toggle_highlights` | Required Boolean | Enables or disables Highlights |

Each invocation accepts one action. An array of actions is invalid.

### Result data

The tool returns a human-readable message for state-changing actions and does not advertise a structured output schema. Report the returned message literally. Do not invent `accepted`, `alreadyInDesiredState`, or other structured success fields, and do not add a follow-up status call solely for confirmation.

### Capture-specific conditions

- Starting recording can return a human-readable message that recording is unavailable when no supported game is running and Desktop Capture is unavailable.
- Stopping recording saves an active recording. If nothing is recording, report the returned message rather than inferring a structured state.
- Saving Instant Replay can return a message that it is disabled. If it says Instant Replay is disabled, ask before enabling it; for every other outcome, report the returned message literally without relabeling it or inferring state.
- A screenshot can return a message that capture is unavailable when neither a supported game nor permitted Desktop Capture is available.
- `invalid_params`, `unsupported_capture_action`, and `overlay_command_failed` identify validation or execution failures.

Do not enable Instant Replay or Desktop Capture as an unrequested workaround.

## `nvapp_overlay_configure_stats`

### Input contract

```json
{
  "type": "object",
  "properties": {
    "enable": {
      "type": "boolean"
    }
  },
  "required": ["enable"],
  "additionalProperties": false
}
```

Pass exactly one field. The tool shows or hides the Statistics Overlay and returns a human-readable message. It does not advertise a structured output schema.

It does not configure Overlay position, layout, opacity, font, displayed metrics, logging, or output paths.

Possible UI-handler errors are `invalid_params` and `overlay_command_failed`, in addition to server-level readiness and access errors.

## `nvapp_overlay_configure_filters`

### Input contract

```json
{
  "type": "object",
  "properties": {
    "filter": {
      "type": "string",
      "enum": ["rtx_dvc"]
    },
    "enable": {
      "type": "boolean"
    }
  },
  "required": ["filter", "enable"],
  "additionalProperties": false
}
```

Pass both fields. The tool enables or disables RTX Dynamic Vibrance and returns a human-readable message. It does not authorize RTX HDR, sharpening, Freestyle, or another filter value.

For a true toggle request, read `rtxDvc` once from `nvapp_overlay_get_status` and pass its opposite as `enable`. If `rtxDvc` is absent, ask for the target state instead of guessing. Do not issue a verification read.

Possible UI-handler errors are `invalid_params` and `overlay_command_failed`, in addition to server-level readiness and access errors.

## `nvapp_overlay_get_current_game_info`

### Input contract

```json
{}
```

### Output contract

When a game is identified:

```json
{
  "running": true,
  "gameName": "Example Game",
  "processId": 12345,
  "fullscreen": true
}
```

When no game is identified:

```json
{
  "running": false
}
```

Only `running` is unconditional. `fullscreen` is an optional Boolean and must be reported only when returned. Do not infer missing game information. A lookup failure returns `overlay_command_failed`.

## Result handling

Read tools may include structured data. Mutation tools return a human-readable message. Failures may also include `errorCode` and `retriable`.

- Prefer structured fields when explaining read state or the selected game; for mutations, report the returned message.
- Preserve meaningful human-readable failure guidance.
- Retry only when the returned result says retry may succeed and retrying remains within the user's request.
- Do not report success for `isError: true`, `status: error`, a missing required result, or an access/readiness failure.
