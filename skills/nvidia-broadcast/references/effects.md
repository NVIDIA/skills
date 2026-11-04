# Effect reference

The **authoritative** list of effectIds and their params is whatever the running app reports:
the MCP `instructions` (sent on connect) and the `set_effects` `effectId` enum. Treat the names
below as a guide; always use the exact ids/params from the live server. Effect ids and parameters
mirror the app's UI; internal representations (HDRI image paths, raw scales) are mapped for you.

Effects are grouped by section: `camera`, `microphone`, `speaker`. Read current values with
`get_broadcast_state` (optionally `effectType: "camera"`); it returns the same public shape:
`{ section: { effectId: { enabled, ...params } } }`.

## Camera

| effectId | what it does | params |
|---|---|---|
| `background_blur` | Blur the background | `strength` 0-1; `mode` "performance"\|"quality" |
| `background_replace` | Replace the background with an image or video | `imagePath` (see below) |
| `background_remove` | Remove the background | toggle |
| `virtual_key_light` | Relight you as if by a key light | `warmth` 0-1 (0 cool -> 1 warm) |
| `vignette` | Darken frame edges | `strength` 0-1; `faceTracking` bool |
| `video_noise_removal` | Reduce low-light video noise | toggle |
| `video_super_resolution` | Upscale the video | `outputWidth` and/or `outputHeight` (see note); `mode` "performance"\|"quality" |
| `video_frame_generation` | Increase output frame rate | `multiplier` 2\|4; `model` "performance"\|"balanced"\|"quality" |
| `eye_contact` | Adjust gaze toward the camera | toggle |
| `auto_frame` | Auto-zoom/track to keep you framed | `zoom` 0-1 |

## Microphone

| effectId | what it does | params |
|---|---|---|
| `noise_removal` | Background noise removal | `strength` 0-1 |
| `room_echo_removal` | Room echo / reverb removal | `strength` 0-1 |
| `studio_voice` | Enhance voice quality | `micProfile` capability enum |

## Speaker

| effectId | what it does | params |
|---|---|---|
| `speaker_noise_removal` | Noise removal on incoming audio | `strength` 0-1 |
| `speaker_echo_removal` | Echo removal on incoming audio | `strength` 0-1 |

Speaker effects are valid for live use but **not supported for file processing**.

## Describing Effects

Call **`describe_effects`** (optionally with `effectId` or `section`) for richer detail per effect:
label, description, guidance, **badges** (e.g. "beta", "high GPU usage"), whether it's currently
available, and each param with its range. You do **not** need this to call `set_effects` (the params
are already in the server instructions); use it to explain effects or weigh tradeoffs.

## Notes

- Numeric params can be normalized ranges (for example `strength` 0-1) or fixed
  numeric enum values (for example `multiplier`);
  modes/enums are exact strings. The gateway maps these to internal scales and
  representations.
- `video_super_resolution` sizing: pass `outputWidth`, `outputHeight`, or both. The output
  must preserve the source's aspect ratio, upscale it, keep the **short** edge at 2160 or
  below, and stay within 4096 per dimension; anything else is rejected. Popular tiers
  (720/1080/1440/2160) name the short edge, so for a portrait source the height is the long
  edge — a 1080x1920 source at the 2160 tier takes `outputHeight` 3840 (or `outputWidth` 2160).
- Omit `params` to just toggle an effect on/off.
- `params` are ignored when `enabled:false`, including stale cached values. To change params
  while leaving an effect disabled, enable it with the new params and then disable it.
- Treat the capability-aware Studio Voice `micProfile` enum as authoritative for both live and file
  processing. The current capability advertises Default, Bright, Full, and Warm.
- `background_replace.imagePath` must be a **local absolute path** ending in `.png`, `.jpg`,
  `.jpeg`, `.mp4`, `.mov` or `.mkv`. UNC (`\\server\share\...`), device (`\\.\...`),
  extended-length (`\\?\...`) and relative paths are rejected, as is anything over 259
  bytes. Ask the user for a real local file rather than guessing a path.
- `set_effects` reports EXCEPTIONS: `{ success, state, failed?, skipped? }`. When `success` is true
  neither list is present, because every requested effect applied. Trust the returned `state`
  instead of issuing a follow-up `get_broadcast_state`.
- When `success` is false, `failed[]` gives `{ effectId, reason }` where reason is `timeout`,
  `unavailable`, `device_not_selected`, `file_not_found` or `failed`. Retry `timeout` and
  `unavailable`; `device_not_selected` and `file_not_found` need corrected user input and carry
  `requiresUserInput: true` (plus `deviceType` or the rejected `imagePath`); `failed` is a generic
  failure. The batch is ordered but not transactional, so effects absent from both lists applied.
- If NO effect applied you get a tool error rather than a result.

## Incompatibilities

Some effects are mutually exclusive. The server advertises the exact incompatible groups (by public
id) in its `instructions`. The gateway also enforces it: `set_effects` skips an effect that would
conflict with what's already on and reports it in `skipped[]` as
`{ effectId, conflictsWith }`; always check `skipped` and the returned `state`.
