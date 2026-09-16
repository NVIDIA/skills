# NVIDIA Broadcast MCP Tool Contract

Last reviewed: 2026-08-14

This document covers the extended-tier MCP tools exposed by the NVIDIA Broadcast gateway.
Examples use representative values so reviewers can see where IDs and returned data appear.

## Tool TOC

1. [`get_broadcast_state`](#1-get_broadcast_state)
2. [`set_effects`](#2-set_effects)
3. [`describe_effects`](#3-describe_effects)
4. [`submit_file_processing`](#4-submit_file_processing)
5. [`get_file_processing_jobs`](#5-get_file_processing_jobs)
6. [`wait_for_processing_update`](#6-wait_for_processing_update)
7. [`control_file_processing_jobs`](#7-control_file_processing_jobs)
8. [`list_devices`](#8-list_devices)
9. [`set_active_device`](#9-set_active_device)
10. [`set_camera_resolution`](#10-set_camera_resolution)

## Result and Error Forms

- Successes return the payload **both** as `structuredContent` and as JSON text in
  `content`. Every tool publishes an `outputSchema` describing that payload, so you can
  validate it rather than parsing text defensively.
- Tool errors return concise text in `content` plus details in `structuredContent.error` —
  schema rejections included, so one parse path covers every failure.
- `error.code` tells you whether to retry. `-32602` means your arguments were wrong and
  will stay wrong. `-32603` with `retryable: true` means the app was busy or not ready, and
  the same call may succeed shortly.
- JSON-RPC protocol failures use a top-level `error` instead.

## Public Effect IDs

These are the public IDs agents use in requests and receive in responses.

```json
{
  "camera": {
    "virtual_key_light": { "warmth": 0.65 },
    "background_blur": { "strength": 0.75, "mode": "quality" },
    "background_replace": {
      "imagePath": "C:\\Users\\User\\Pictures\\background.png"
    },
    "background_remove": {},
    "vignette": { "strength": 0.35, "faceTracking": true },
    "video_noise_removal": {},
    "video_super_resolution": {
      "outputWidth": 1920,
      "outputHeight": 1080,
      "mode": "quality"
    },
    "video_frame_generation": { "multiplier": 2, "model": "balanced" },
    "eye_contact": {},
    "auto_frame": { "zoom": 0.4 }
  },
  "microphone": {
    "noise_removal": { "strength": 0.8 },
    "room_echo_removal": { "strength": 0.6 },
    "studio_voice": { "micProfile": "full" }
  },
  "speaker": {
    "speaker_noise_removal": { "strength": 0.7 },
    "speaker_echo_removal": { "strength": 0.5 }
  }
}
```

Where these IDs appear:

| Location                       | Example                                     |
| ------------------------------ | ------------------------------------------- |
| `set_effects` input            | `effects[].effectId: "background_blur"`     |
| `submit_file_processing` input | `effects[].effectId: "video_noise_removal"` |
| `describe_effects` output      | `effects[].id: "studio_voice"`              |
| `get_broadcast_state` output   | `camera.background_blur.enabled`            |
| `set_effects` output           | `state.microphone.noise_removal.strength`   |
| File job output                | `effects: ["auto_frame"]`                   |
| Skipped file effects           | `skippedEffects: ["background_blur"]`       |

The actual accepted `effectId` set is injected at runtime from effects available on the user's machine.
The current Studio Voice capability advertises `micProfile` values `default`, `bright`, `full`, and
`warm`; always treat the capability-aware tool schema/descriptor enum as authoritative.

## Public Device Shapes

Device tools use stable public IDs and resolution fields. Resolution is always specified with
`width`, `height`, and optional `frameRate` — never with internal indices.

```json
{
  "cameras": [
    {
      "id": "camera-uuid-logitech-brio",
      "name": "Logitech BRIO",
      "active": true,
      "currentResolution": {
        "width": 1920,
        "height": 1080,
        "frameRate": 60,
        "label": "1080p"
      },
      "availableResolutions": [
        { "width": 1920, "height": 1080, "frameRate": 30, "label": "1080p" },
        { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" },
        { "width": 3840, "height": 2160, "frameRate": 30, "label": "4K" }
      ]
    }
  ],
  "microphones": [
    {
      "id": "mic-uuid-blue-yeti",
      "name": "Blue Yeti",
      "active": true
    }
  ],
  "speakers": [
    {
      "id": "spk-uuid-realtek",
      "name": "Speakers (Realtek Audio)",
      "active": true
    }
  ]
}
```

Where these fields appear:

| Location                      | Example                                          |
| ----------------------------- | ------------------------------------------------ |
| `list_devices` output         | `cameras[].id`, `cameras[].availableResolutions` |
| `set_active_device` input     | `deviceId: "camera-uuid-logitech-brio"`          |
| `set_active_device` output    | `device.id`, `resolution.width`                  |
| `set_camera_resolution` input | `deviceId`, `width`, `height`, `frameRate`       |
| Failure recovery              | `state.cameras`, `availableDevices`              |

Device IDs come from a fresh `list_devices` call. Do not reuse IDs from an earlier session.

Device setters have two distinct failure forms. Requests rejected before an operation
starts—for example, an unknown device ID or unavailable/ambiguous resolution—return a
gateway tool error (`isError:true` with `structuredContent.error`). If an accepted device
switch or resolution change is attempted but cannot be confirmed, the tool returns a
normal result with `success:false` plus recovery state such as `availableDevices`, `actual`,
or `state.cameras`.

## Shared File Job Example

File-processing tools return this shape.

```json
{
  "id": "mcp-file-job-1720431234567-a1b2c3",
  "batchId": "mcp-file-batch-1720431234567-a1b2c3",
  "status": "completed",
  "progress": 1,
  "sourcePath": "C:\\Users\\User\\Videos\\input.mp4",
  "outputPath": "C:\\Users\\User\\Videos\\input_2026-07-14_10_30_00.mp4",
  "mediaKind": "video",
  "effects": ["auto_frame", "video_noise_removal", "noise_removal"],
  "createdAt": 1720431234567
}
```

## File Status Updates

`wait_for_processing_update` is the status channel for file batches: the first
call returns a snapshot, and a call with its `cursor` waits until a job status or the queue
pause state changes. Progress-only updates do not wake it, so one batch creates little
traffic even when it contains many files.

Aborting `wait_for_processing_update` stops waiting only; use
`control_file_processing_jobs` to cancel media work.

## 1. `get_broadcast_state`

Description exposed:

```text
Read current NVIDIA Broadcast effect state. Use for later current-state questions; do not answer them from prior tool results.
```

Annotations:

```json
{
  "title": "Read Broadcast state",
  "readOnlyHint": true,
  "destructiveHint": false,
  "openWorldHint": false
}
```

Input example:

```json
{
  "effectType": "all"
}
```

Output example:

```json
{
  "camera": {
    "background_blur": {
      "enabled": true,
      "strength": 0.75,
      "mode": "quality"
    },
    "eye_contact": {
      "enabled": false
    },
    "video_frame_generation": {
      "enabled": true,
      "multiplier": 2,
      "model": "balanced"
    }
  },
  "microphone": {
    "noise_removal": {
      "enabled": true,
      "strength": 0.8
    },
    "studio_voice": {
      "enabled": false,
      "micProfile": "full"
    }
  },
  "speaker": {
    "speaker_noise_removal": {
      "enabled": false,
      "strength": 0.7
    }
  },
  "warnings": [
    {
      "code": "device_not_selected",
      "deviceType": "camera",
      "affectedEffects": ["background_blur", "video_frame_generation"],
      "message": "No camera is selected. The listed effects are configured as enabled but are not currently applied."
    }
  ]
}
```

`warnings` is omitted when empty. A warning is returned only when a section has enabled effects but
no selected endpoint. In that case, `enabled` remains the preserved configuration; it does not mean
the affected effects are currently processing. The warning is MCP response metadata only and does
not create an app notification or telemetry event.

## 2. `set_effects`

Description exposed:

```text
Change Live effects or restore those suspended by Files processing. Batch changes in one call.
```

Annotations:

```json
{
  "title": "Change Broadcast effects",
  "readOnlyHint": false,
  "destructiveHint": true,
  "idempotentHint": true,
  "openWorldHint": false
}
```

`set_effects` accepts exactly one of these input forms:

- `effects` changes one or more named effects.
- `action: "restore_previous_live"` restores the app-owned Live snapshot suspended by Files
  processing.

The two forms are mutually exclusive.

Effects input example:

```json
{
  "effects": [
    {
      "effectId": "background_blur",
      "enabled": true,
      "params": {
        "strength": 0.75,
        "mode": "quality"
      }
    },
    {
      "effectId": "noise_removal",
      "enabled": true,
      "params": {
        "strength": 0.8
      }
    },
    {
      "effectId": "video_frame_generation",
      "enabled": true,
      "params": {
        "multiplier": 2,
        "model": "balanced"
      }
    },
    {
      "effectId": "studio_voice",
      "enabled": true,
      "params": {
        "micProfile": "full"
      }
    }
  ]
}
```

Output example. Only exceptions are listed — `success: true` with no `failed` or `skipped`
means every requested effect applied, so there is nothing per-effect to read:

```json
{
  "success": true,
  "state": {
    "camera": {
      "background_blur": {
        "enabled": true,
        "strength": 0.75,
        "mode": "quality"
      },
      "video_frame_generation": {
        "enabled": true,
        "multiplier": 2,
        "model": "balanced"
      }
    },
    "microphone": {
      "noise_removal": {
        "enabled": true,
        "strength": 0.8
      },
      "studio_voice": {
        "enabled": true,
        "micProfile": "full"
      }
    },
    "speaker": {}
  }
}
```

Restore input example:

```json
{
  "action": "restore_previous_live"
}
```

Restore output example:

```json
{
  "success": true,
  "event": "live_effects_restored",
  "state": {
    "camera": {
      "background_blur": { "enabled": true, "strength": 0.75 }
    },
    "microphone": {
      "noise_removal": { "enabled": true, "strength": 0.8 }
    },
    "speaker": {}
  }
}
```

`live_effects_restored` means Broadcast paused unfinished file work, kept it queued, and
restored its exact saved Live snapshot without requiring effect IDs from the agent. `no_change`
means no suspended snapshot was available or it was already restored. Both outcomes include the
authoritative resulting `state` for that response; no follow-up read is needed just to verify it. If
the user later asks for current state, call `get_broadcast_state` because effects can change outside
MCP. This action does not accept a consent flag and does not resume the file queue.

An ordinary `effects` request while Files processing owns the pipeline restores the saved Live
snapshot first, then applies the requested delta.

Before every effect operation, whether `enabled` is `true` or `false`, Broadcast checks that the
effect's camera, microphone, or speaker has a selected endpoint that is still present in the current
device list. The check is sequential and immediately precedes that effect's mutation, so a device
unplugged during a batch is still caught. A missing selection returns `reason: "device_not_selected"`,
`deviceType`, and `requiresUserInput: true` for that effect; effects already applied earlier in the
non-transactional batch remain applied. Existing configured effect state is preserved while the
endpoint is unavailable. If no requested effect applies because its devices are not selected, the tool
returns error code `-33201`.

When enabling `background_replace` with `params.imagePath`, the local image or video file must already
exist. Broadcast checks the path before changing the effect or its saved image selection. A missing
source returns `reason: "file_not_found"`, the rejected `imagePath`, and `requiresUserInput: true`; the
previous background remains selected. If no requested effect applies because its background sources
are missing, `set_effects` returns tool error `-32602` with a suggested fix instead of a success-shaped
result.

Output example with a skipped incompatible effect. `background_blur` applied and is
therefore absent — only what did **not** happen is named:

```json
{
  "success": false,
  "state": {
    "camera": {
      "background_blur": {
        "enabled": true,
        "strength": 0.75,
        "mode": "quality"
      },
      "background_replace": {
        "enabled": false
      }
    },
    "microphone": {},
    "speaker": {}
  },
  "skipped": [
    {
      "effectId": "background_replace",
      "conflictsWith": ["background_blur"]
    }
  ]
}
```

Output example where an effect could not be applied at all:

```json
{
  "success": false,
  "state": { "camera": {}, "microphone": {}, "speaker": {} },
  "failed": [{ "effectId": "noise_removal", "reason": "timeout" }]
}
```

`reason` is `timeout`, `unavailable`, `device_not_selected`, `file_not_found`, or `failed`. Retry
`timeout` and `unavailable`; the device and file reasons require corrected user input, while `failed`
is generic. If NO requested effect applied you get a tool error rather than a result.

`params` are ignored when `enabled:false`. To change params while keeping an effect
disabled, enable it with the new params and then disable it.

## 3. `describe_effects`

Description exposed:

```text
Describe effects. Use only when the user asks what an effect does.
```

Annotations:

```json
{
  "title": "Describe effects",
  "readOnlyHint": true,
  "destructiveHint": false,
  "openWorldHint": false
}
```

Input example:

```json
{
  "section": "microphone"
}
```

Output example:

```json
{
  "effects": [
    {
      "id": "noise_removal",
      "label": "Noise removal",
      "section": "microphone",
      "params": [
        {
          "name": "strength",
          "type": "number",
          "min": 0,
          "max": 1,
          "description": "Effect strength, 0-1."
        }
      ],
      "description": "Reduces background noise",
      "accessible": true
    },
    {
      "id": "studio_voice",
      "label": "Studio voice",
      "section": "microphone",
      "params": [
        {
          "name": "micProfile",
          "type": "enum",
          "enum": ["default", "bright", "full", "warm"],
          "description": "Studio Voice mic profile."
        }
      ],
      "description": "Enhances quality to simulate a high-end recording studio",
      "guidance": "Not recommended to use with games or GPU intensive apps.",
      "badges": ["high GPU usage"],
      "accessible": true
    }
  ]
}
```

## 4. `submit_file_processing`

Description exposed:

```text
Submit one local media file. Reuse the returned batchId on related submissions, then call wait_for_processing_update once for the batch.
```

Annotations:

```json
{
  "title": "Process media file",
  "readOnlyHint": false,
  "destructiveHint": true,
  "idempotentHint": false,
  "openWorldHint": false
}
```

Input example:

```json
{
  "_version": 1,
  "batchId": "<omit for first file; reuse its returned value for later files>",
  "inputPath": "C:\\Users\\User\\Videos\\input.mp4",
  "effects": [
    {
      "effectId": "auto_frame",
      "params": { "zoom": 0.4 }
    },
    {
      "effectId": "video_noise_removal"
    },
    {
      "effectId": "video_frame_generation",
      "params": {
        "multiplier": 2,
        "model": "balanced"
      }
    },
    {
      "effectId": "noise_removal",
      "params": { "strength": 0.8 }
    },
    {
      "effectId": "studio_voice",
      "params": { "micProfile": "full" }
    }
  ],
  "outputFolder": "C:\\Users\\User\\Videos"
}
```

Output example:

```json
{
  "id": "mcp-file-job-1720431234567-a1b2c3",
  "batchId": "mcp-file-batch-1720431234567-a1b2c3",
  "status": "queued",
  "progress": 0,
  "sourcePath": "C:\\Users\\User\\Videos\\input.mp4",
  "outputPath": "C:\\Users\\User\\Videos\\input_2026-07-14_10_30_00.mp4",
  "mediaKind": "video",
  "effects": [
    "auto_frame",
    "video_noise_removal",
    "video_frame_generation",
    "noise_removal"
  ],
  "skippedEffects": ["studio_voice"],
  "createdAt": 1720431234567
}
```

`noise_removal` appears before `studio_voice`, so it wins and Studio Voice is skipped as a
conflict. Reversing their order applies Studio Voice and skips noise removal. A request
where every file effect would be skipped is rejected instead of creating a job.

## 5. `get_file_processing_jobs`

Description exposed:

```text
Get one file-processing job or list recent jobs.
```

Annotations:

```json
{
  "title": "List file-processing jobs",
  "readOnlyHint": true,
  "destructiveHint": false,
  "openWorldHint": false
}
```

Input example to list jobs:

```json
{
  "_version": 1,
  "status": "in_progress",
  "limit": 50,
  "offset": 0
}
```

Output example for list:

```json
{
  "jobs": [
    {
      "id": "mcp-file-job-1720431234567-a1b2c3",
      "batchId": "mcp-file-batch-1720431234567-a1b2c3",
      "status": "in_progress",
      "progress": 0.42,
      "sourcePath": "C:\\Users\\User\\Videos\\input.mp4",
      "outputPath": "C:\\Users\\User\\Videos\\input_2026-07-14_10_30_00.mp4",
      "mediaKind": "video",
      "effects": ["auto_frame"],
      "createdAt": 1720431234567
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

Input example to get one job:

```json
{
  "_version": 1,
  "jobId": "mcp-file-job-1720431234567-a1b2c3"
}
```

Output example for one job:

```json
{
  "id": "mcp-file-job-1720431234567-a1b2c3",
  "batchId": "mcp-file-batch-1720431234567-a1b2c3",
  "status": "in_progress",
  "progress": 0.42,
  "sourcePath": "C:\\Users\\User\\Videos\\input.mp4",
  "outputPath": "C:\\Users\\User\\Videos\\input_2026-07-14_10_30_00.mp4",
  "mediaKind": "video",
  "effects": ["auto_frame"],
  "createdAt": 1720431234567
}
```

## 6. `wait_for_processing_update`

This is the sole status wait for one file or many files sharing a `batchId`.

Description exposed:

```text
Return a batch snapshot immediately without cursor; with cursor, wait until a job status or queue pause state changes. Progress-only updates do not wake it.
```

Annotations:

```json
{
  "title": "Wait for file-processing updates",
  "readOnlyHint": true,
  "destructiveHint": false,
  "openWorldHint": false
}
```

First call returns an immediate snapshot:

```json
{
  "_version": 1,
  "batchId": "mcp-file-batch-1720431234567-a1b2c3"
}
```

If the batch is not terminal or paused, pass `cursor` unchanged to wait for the next
job-status or queue-pause change. Progress-only changes do not wake the wait.

```json
{
  "_version": 1,
  "batchId": "mcp-file-batch-1720431234567-a1b2c3",
  "cursor": "<cursor from previous result>",
  "timeoutMs": 120000
}
```

A mixed terminal batch is explicit and cannot be mistaken for success:

```json
{
  "batchId": "mcp-file-batch-1720431234567-a1b2c3",
  "cursor": "<next cursor>",
  "status": "mixed",
  "endedBecause": "batch_ended",
  "summary": {
    "total": 3,
    "queued": 0,
    "inProgress": 0,
    "completed": 2,
    "failed": 0,
    "cancelled": 1
  },
  "jobs": [
    { "id": "job-1", "status": "completed" },
    { "id": "job-2", "status": "completed" },
    { "id": "job-3", "status": "cancelled", "reason": "cancelled_by_user" }
  ]
}
```

Batch `status` is `queued`, `in_progress`, `paused_by_user`, `completed`, `failed`,
`cancelled`, or `mixed`. `endedBecause` says why this wait returned: `snapshot`,
`status_changed`, `batch_ended`, or `timeout`. Always inspect every job. A cancelled job
with `reason: "cancelled_by_user"` is terminal and must never be retried automatically. If
the user explicitly asks after learning about the cancellation, call `retry` with that job's
exact `jobIds`; do not use `all`, resubmit, or create a replacement.

When status is `paused_by_user`, ask in the agent conversation before resuming. After an
explicit yes, call ordinary queue-wide `resume`. The MCP tool has no consent flag.
Aborting the wait stops waiting only; it never cancels the media work.

## 7. `control_file_processing_jobs`

Pause or resume the shared file-processing queue, cancel jobs, or retry eligible failed jobs.

Annotations:

```json
{
  "title": "Control file-processing jobs",
  "readOnlyHint": false,
  "destructiveHint": true,
  "idempotentHint": false,
  "openWorldHint": false
}
```

Pause and resume are queue-wide and omit `jobIds` and `all`:

```json
{
  "_version": 1,
  "action": "pause"
}
```

Cancel and retry use exactly one of `jobIds` or `all: true`:

```json
{
  "_version": 1,
  "action": "retry",
  "jobIds": ["mcp-file-job-1720431234567-a1b2c3"]
}
```

Results report a notification-like enum plus the jobs actually affected:

```json
{
  "event": "jobs_requeued",
  "jobIds": ["mcp-file-job-1720431234567-a1b2c3"],
  "jobs": [
    {
      "id": "mcp-file-job-1720431234567-a1b2c3",
      "batchId": "mcp-file-batch-1720431234567-a1b2c3",
      "status": "queued",
      "progress": 0
    }
  ]
}
```

`event` is `queue_paused`, `queue_resumed`, `jobs_cancelled`, `jobs_requeued`, or
`no_change`. Retry applies to failed jobs with `recovery: "retry"`. A cancelled job is eligible
only when it is named by exact `jobIds`, which asserts that the agent already received an explicit
user request; `all: true` never includes cancellations. If retry returns `no_change`, report it and
stop; do not submit replacement jobs. Retrying while the user-paused queue is stopped only
requeues eligible jobs and does not resume processing or stop live effects.

A stopped queue never resumes merely because a file was submitted. When
`wait_for_processing_update` reports `paused_by_user`, the agent asks the user. Only after
an explicit yes should it call `{ "_version": 1, "action": "resume" }`. Resume executes
immediately and the tool does not accept `userConsent` or any other consent flag.

## 8. `list_devices`

Description exposed:

```text
List devices by kind. Call before switching devices or when the user asks what is available.
```

Annotations:

```json
{
  "title": "Broadcast: List devices",
  "readOnlyHint": true,
  "destructiveHint": false,
  "openWorldHint": false
}
```

Input example:

```json
{
  "kind": "camera"
}
```

Output example:

```json
{
  "cameras": [
    {
      "id": "camera-uuid-logitech-brio",
      "name": "Logitech BRIO",
      "active": true,
      "currentResolution": {
        "width": 1920,
        "height": 1080,
        "frameRate": 60,
        "label": "1080p"
      },
      "availableResolutions": [
        { "width": 1920, "height": 1080, "frameRate": 30, "label": "1080p" },
        { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" },
        { "width": 3840, "height": 2160, "frameRate": 30, "label": "4K" }
      ]
    },
    {
      "id": "camera-uuid-integrated",
      "name": "Integrated Camera",
      "active": false,
      "currentResolution": null,
      "availableResolutions": [
        { "width": 1280, "height": 720, "frameRate": 30, "label": "720p HD" }
      ]
    }
  ]
}
```

Output example when scoped to all kinds:

```json
{
  "cameras": [
    {
      "id": "camera-uuid-logitech-brio",
      "name": "Logitech BRIO",
      "active": true,
      "currentResolution": {
        "width": 1920,
        "height": 1080,
        "frameRate": 60,
        "label": "1080p"
      },
      "availableResolutions": [
        { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" }
      ]
    }
  ],
  "microphones": [
    {
      "id": "mic-uuid-blue-yeti",
      "name": "Blue Yeti",
      "active": true
    }
  ],
  "speakers": [
    {
      "id": "spk-uuid-realtek",
      "name": "Speakers (Realtek Audio)",
      "active": true
    }
  ]
}
```

Output example during early init:

```json
{
  "cameras": [],
  "note": "No devices detected yet. The app may still be initializing — retry in a moment."
}
```

Camera `availableResolutions` reflects the current Video Super Resolution / Video Frame
Generation mode. If the active mode has no matching synthetic rows for a camera, the
tool falls back to that camera's highest native resolution rows instead of returning an
empty list.

## 9. `set_active_device`

Description exposed:

```text
Switch the active camera, microphone, or speaker. For cameras, optionally set resolution in the same call.
```

Annotations:

```json
{
  "title": "Broadcast: Switch device",
  "readOnlyHint": false,
  "destructiveHint": true,
  "idempotentHint": true,
  "openWorldHint": false
}
```

Input example to switch microphone:

```json
{
  "kind": "microphone",
  "deviceId": "mic-uuid-blue-yeti"
}
```

Output example on success:

```json
{
  "success": true,
  "message": "Switched microphone to Blue Yeti",
  "device": {
    "id": "mic-uuid-blue-yeti",
    "name": "Blue Yeti"
  },
  "deviceConfirmed": true
}
```

Input example to switch camera and resolution in one call:

```json
{
  "kind": "camera",
  "deviceId": "camera-uuid-logitech-brio",
  "width": 1920,
  "height": 1080,
  "frameRate": 60
}
```

Output example on success with resolution:

```json
{
  "success": true,
  "message": "Switched camera to Logitech BRIO",
  "device": {
    "id": "camera-uuid-logitech-brio",
    "name": "Logitech BRIO"
  },
  "deviceConfirmed": true,
  "resolution": {
    "confirmed": true,
    "width": 1920,
    "height": 1080,
    "frameRate": 60
  }
}
```

Output example when width/height applied but frame rate differed:

```json
{
  "success": true,
  "message": "Switched camera to Logitech BRIO",
  "device": {
    "id": "camera-uuid-logitech-brio",
    "name": "Logitech BRIO"
  },
  "deviceConfirmed": true,
  "resolution": {
    "confirmed": true,
    "width": 1920,
    "height": 1080,
    "frameRate": 30,
    "requestedFrameRate": 60
  },
  "frameRateMismatch": true
}
```

Output example when device is already active:

```json
{
  "success": true,
  "alreadyActive": true,
  "message": "Blue Yeti is already the active microphone — no change made.",
  "device": {
    "id": "mic-uuid-blue-yeti",
    "name": "Blue Yeti"
  }
}
```

Output example when device switched but requested resolution was not applied:

```json
{
  "success": false,
  "message": "Switched camera to Logitech BRIO, but the requested resolution was not applied. The camera may not support it — see resolution.actual or call list_devices for availableResolutions.",
  "device": {
    "id": "camera-uuid-logitech-brio",
    "name": "Logitech BRIO"
  },
  "deviceConfirmed": true,
  "resolution": {
    "confirmed": false,
    "requested": {
      "width": 3840,
      "height": 2160,
      "frameRate": 60
    },
    "actual": {
      "width": 1920,
      "height": 1080,
      "frameRate": 60
    }
  },
  "state": {
    "cameras": [
      {
        "id": "camera-uuid-logitech-brio",
        "name": "Logitech BRIO",
        "active": true,
        "currentResolution": {
          "width": 1920,
          "height": 1080,
          "frameRate": 60,
          "label": "1080p"
        },
        "availableResolutions": [
          { "width": 1920, "height": 1080, "frameRate": 30, "label": "1080p" },
          { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" }
        ]
      }
    ]
  }
}
```

Output example when switch failed:

```json
{
  "success": false,
  "message": "Logitech BRIO is present but did not become active. Please choose from the available cameras below.",
  "deviceConfirmed": false,
  "requested": {
    "id": "camera-uuid-logitech-brio",
    "name": "Logitech BRIO"
  },
  "actual": {
    "id": "camera-uuid-integrated",
    "name": "Integrated Camera"
  },
  "availableDevices": [
    {
      "id": "camera-uuid-logitech-brio",
      "name": "Logitech BRIO",
      "active": false
    },
    {
      "id": "camera-uuid-integrated",
      "name": "Integrated Camera",
      "active": true
    }
  ],
  "state": {
    "cameras": [
      {
        "id": "camera-uuid-logitech-brio",
        "name": "Logitech BRIO",
        "active": false,
        "currentResolution": null,
        "availableResolutions": [
          { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" }
        ]
      },
      {
        "id": "camera-uuid-integrated",
        "name": "Integrated Camera",
        "active": true,
        "currentResolution": {
          "width": 1280,
          "height": 720,
          "frameRate": 30,
          "label": "720p HD"
        },
        "availableResolutions": [
          { "width": 1280, "height": 720, "frameRate": 30, "label": "720p HD" }
        ]
      }
    ]
  }
}
```

Error example when multiple frame rates exist and `frameRate` was omitted:

```json
{
  "code": -32602,
  "message": "1920x1080 is available at multiple frame rates (30fps, 60fps). Ask the user which frame rate to use — do not pick a default.",
  "data": {
    "requiresUserInput": true,
    "reason": "ambiguous_frame_rate",
    "width": 1920,
    "height": 1080,
    "frameRateOptions": [
      { "width": 1920, "height": 1080, "frameRate": 30, "label": "1080p" },
      { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" }
    ]
  }
}
```

## 10. `set_camera_resolution`

Description exposed:

```text
Change resolution on the active camera. If switching cameras too, use set_active_device with resolution instead.
```

Annotations:

```json
{
  "title": "Broadcast: Set camera resolution",
  "readOnlyHint": false,
  "destructiveHint": true,
  "idempotentHint": true,
  "openWorldHint": false
}
```

Input example:

```json
{
  "deviceId": "camera-uuid-logitech-brio",
  "width": 1280,
  "height": 720,
  "frameRate": 30
}
```

The target must already be the selected camera and must still be present in the current device list.
When no camera is selected, the tool returns `-32602` with `reason: "device_not_selected"`,
`deviceType: "camera"`, and `requiresUserInput: true`; select a camera with `set_active_device`
before retrying.

Output example on success:

```json
{
  "success": true,
  "message": "Camera resolution confirmed: 1280x720 @ 30fps",
  "resolution": {
    "width": 1280,
    "height": 720,
    "frameRate": 30
  }
}
```

Output example when width/height applied but frame rate differed:

```json
{
  "success": true,
  "message": "Camera resolution set to 1920x1080, but at 30fps instead of the requested 60fps.",
  "frameRateMismatch": true,
  "resolution": {
    "width": 1920,
    "height": 1080,
    "frameRate": 30
  }
}
```

Output example when already at requested resolution:

```json
{
  "success": true,
  "alreadySet": true,
  "message": "Camera is already at 1920x1080 @ 60fps — no change made.",
  "resolution": {
    "width": 1920,
    "height": 1080,
    "frameRate": 60
  }
}
```

Output example when resolution was not applied:

```json
{
  "success": false,
  "message": "Requested 3840x2160 but camera is at 1920x1080. The camera may not support this resolution — see availableResolutions in state.",
  "requested": {
    "width": 3840,
    "height": 2160,
    "frameRate": 60
  },
  "actual": {
    "width": 1920,
    "height": 1080,
    "frameRate": 60
  },
  "state": {
    "cameras": [
      {
        "id": "camera-uuid-logitech-brio",
        "name": "Logitech BRIO",
        "active": true,
        "currentResolution": {
          "width": 1920,
          "height": 1080,
          "frameRate": 60,
          "label": "1080p"
        },
        "availableResolutions": [
          { "width": 1920, "height": 1080, "frameRate": 30, "label": "1080p" },
          { "width": 1920, "height": 1080, "frameRate": 60, "label": "1080p" }
        ]
      }
    ]
  }
}
```

Error example when camera is not active:

```json
{
  "code": -32602,
  "message": "Camera 'Integrated Camera' is not the active camera. Switch to it first using set_active_device, then change resolution."
}
```
