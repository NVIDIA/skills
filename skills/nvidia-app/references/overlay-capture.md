# Overlay Capture Workflows

Use this reference for recording, Instant Replay, Highlights, and screenshot requests handled by `nvapp_overlay_capture`.

## Authorized actions

| Intent | Arguments |
|---|---|
| Start recording | `{ "action": "toggle_recording", "enable": true }` |
| Stop and save recording | `{ "action": "toggle_recording", "enable": false }` |
| Enable Instant Replay | `{ "action": "toggle_instant_replay", "enable": true }` |
| Disable Instant Replay | `{ "action": "toggle_instant_replay", "enable": false }` |
| Save Instant Replay | `{ "action": "save_instant_replay" }` |
| Enable Highlights | `{ "action": "toggle_highlights", "enable": true }` |
| Disable Highlights | `{ "action": "toggle_highlights", "enable": false }` |
| Capture screenshot | `{ "action": "capture_screenshot" }` |

Use no other capture actions through this skill. In particular, never invoke `toggle_desktop_capture`.

## State and call-count rules

- Explicit start, stop, enable, or disable requests go directly to `nvapp_overlay_capture`; do not pre-read status.
- For a true request to toggle recording, Instant Replay, or Highlights without a target state, call `nvapp_overlay_get_status` once and invert `recordingActive`, `instantReplayEnabled`, or `highlightsEnabled`, respectively.
- If the required field is absent, ask whether to enable or disable instead of guessing.
- Each invocation accepts one action, not an action list.
- Do not make a follow-up status call solely to verify a capture mutation.
- Mutation results are human-readable messages. Report the returned message without inventing structured success fields.

## Action semantics

### Recording

Recording availability and returned-message behavior follow the conditions below.

Do not invent a separate save-recording action, output path, filename, duration, codec, resolution, or frame rate.

### Instant Replay

Saving Instant Replay saves the currently configured buffer. The tool cannot select a duration for the save request.

- If the returned message says Instant Replay is disabled, ask whether the user wants to enable it; do not enable it implicitly.
- For every other save outcome, report the returned message literally without relabeling it or inferring state.

Do not add `enable` to `save_instant_replay`.

### Highlights

`toggle_highlights` requires a Boolean target state in the tool call. Use `enable: true` to enable Highlights and `enable: false` to disable it. Do not invent game-specific event, duration, storage, or quality arguments.

### Screenshots

A successful screenshot message indicates that the screenshot was saved to the Overlay gallery. The tool cannot select the file path, filename, format, or resolution. Do not add `enable` to `capture_screenshot`.

Screenshot capture can fail when neither a supported game nor permitted Desktop Capture is available. A Desktop Capture condition is not permission to enable Desktop Capture.

## Unsupported qualifiers

When the request combines a supported capture with an unsupported qualifier:

1. Explain exactly which requirement cannot be satisfied.
2. Make no state-changing call yet.
3. Ask whether to proceed with the supported reduced operation.
4. After confirmation, send only the supported arguments.

Examples:

- "Save the last five minutes" — explain that the tool saves the configured Instant Replay buffer and ask whether to save that buffer.
- "Take a PNG screenshot in `D:\\Shots`" — explain that format and path are not selectable and ask whether to capture to the Overlay gallery.

## Access and errors

`nvapp_overlay_capture` is a Normal-tier tool available at standard MCP access or higher. Make a call only for an explicit user-requested authorized action. Do not raise access, route through another application, or enable an excluded capability to work around a failure.

Report the returned human-readable message for capture outcomes. For readiness, access, validation, or execution failures, preserve any returned `overlay_not_available`, `overlay_timeout`, `access_level_restricted`, `invalid_params`, `unsupported_capture_action`, or `overlay_command_failed` code. Retry only when the result marks the error retriable and retrying remains within the request.
