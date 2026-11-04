# General NVIDIA App MCP Tools

Use this reference for the seven public NVIDIA App tools when live discovery advertises them.

## Discovery workflow

Call `tools/list` before execution.

- Use only the seven tools documented below.
- Follow these call contracts plus any narrower schema returned by the live server.
- If a documented tool is absent, report it unavailable for the current build or session.
- Never expose or invoke an unlisted private tool.

## Public tool contracts

| Tool | What it does | Valid arguments |
|---|---|---|
| `nvapp_client_get_applications` | Lists NVIDIA App-discovered games and applications with OPS details. Optional inputs filter results; returned `game` values identify applications for game-scoped tools. | `{}`, `{ "game": "<exact game>" }`, or `{ "game": "<exact game>", "preset": "<preset>" }` |
| `nvapp_client_launch_application` | Launches a game or application returned by `nvapp_client_get_applications`. | `{ "game": "<exact game>" }` |
| `nvapp_client_get_driver_status` | Returns the installed driver, compatible recommendations, and previous-driver information. | `{}` |
| `nvapp_client_get_driver_release_notes` | Returns release highlights for the installed driver or a compatible recommended driver flavor. | `{ "driver": "<driver selector>" }` |
| `nvapp_client_set_game_optimization` | Applies a game optimization preset or restores defaults for a supported game. | `{ "game": "<exact game>", "preset": "<preset>" }` or `{ "game": "<exact game>", "preset": "<preset>", "powerSource": "<power source>" }` |
| `nvapp_client_get_laptop_features` | Returns Battery Boost and Whisper Mode settings globally or for a game. | `{}` or `{ "game": "<scope>" }` |
| `nvapp_client_set_laptop_feature` | Sets Battery Boost or Whisper Mode globally or for a game. | Exactly one legal feature shape listed below |

Argument rules:

- `nvapp_client_get_applications` - both `game` and `preset` may be omitted. When `preset` is supplied, `game` must also be supplied. Valid presets are `Balanced`, `Performance`, `Recommended`, and `Quality`.
- `nvapp_client_get_driver_release_notes` - `driver`: `INSTALLED`, `GAME_READY`, `STUDIO`, `NVIDIA_RECOMMENDED`, `LEADING_EDGE`, or `CONSERVATIVE`.
- `nvapp_client_set_game_optimization` - `preset`: `Performance` favors frame rate, `Balanced` selects the optimization slider midpoint, `Recommended` uses NVIDIA's per-game recommendation, `Quality` favors image quality, and `Revert` restores defaults.
- `nvapp_client_set_game_optimization` - optional `powerSource`: `AC` or `Battery`. `AC` means the system is connected to external power, while `Battery` means the system is running on battery power. It defaults to `AC` when omitted.
- `nvapp_client_get_laptop_features` - `game`: omit it, or use `global` or `base`, for global scope. Any other value must be an exact application identifier.
- `nvapp_client_set_laptop_feature` - global Whisper Mode `fanVolume`: `Balanced`, `Quiet`, or `Quieter`.

Legal `nvapp_client_set_laptop_feature` argument shapes:

```json
{ "batteryBoost": { "enabled": true } }
{ "whisperMode": { "enabled": true, "fanVolume": "Quiet" } }
{ "game": "Exact App", "batteryBoost": { "useGlobal": true } }
{ "game": "Exact App", "whisperMode": { "useGlobal": true } }
{ "game": "Exact App", "batteryBoost": { "useGlobal": false, "enabled": true } }
{ "game": "Exact App", "whisperMode": { "useGlobal": false, "enabled": true } }
```

For global laptop mutations, omit `game` or use `global` or `base`; do not send `useGlobal`. Application scope requires `useGlobal`. When `useGlobal` is `true`, omit `enabled` and `fanVolume`; when it is `false`, `enabled` is required. Application-scoped Whisper Mode does not accept `fanVolume`. Send exactly one feature object per call.

## Required prerequisite checks

- Before `nvapp_client_launch_application`, obtain the current eligible applications from a successful `nvapp_client_get_applications` call. The launch `game` must exactly match one of its returned `game` values. Treat free-form text, including an `Other` response, only as a selection request; never pass it directly to the launch tool. If it does not match, do not call the launch tool; offer to rescan or report that the application is unavailable.
- Use the exact `game` returned by `nvapp_client_get_applications` for optimization and application-scoped laptop calls. Do not substitute a path, executable, command line, or guessed display name.
- For "optimize all games/apps" requests, direct the user to use Optimize All from the NVIDIA App Graphics page instead of calling `nvapp_client_set_game_optimization` repeatedly.
- Before optimization, require `opsSupported: true` and inspect the selected power profile's optional `unavailableReason`.
  - When omitted, require a non-`Revert` preset to appear in `availablePresets`.
  - When present, report the returned reason literally and do not call the setter. A desktop-resolution reason already contains the recommended resolution.
- `unavailableReason` blocks every optimization mutation for that profile, including `Revert`. An empty `availablePresets` with this reason is a temporary prerequisite failure, not evidence that the game lacks OPS support. Do not change the display resolution implicitly.
- Before setting a laptop feature, read its current scope and require `supported: true`.
- `Revert` affects only the requested game and power source.

## Overlay routing boundary

When the selected tool has the `nvapp_overlay_` prefix, return to the Overlay workflow in `SKILL.md` and read the applicable Overlay reference. The Overlay workflow authorizes `toggle_highlights`, but does not authorize `toggle_desktop_capture` or any `nvapp_overlay_configure_filters` value other than `rtx_dvc`.

## Result handling

- Do not report success for `isError: true` or another returned failure state.
- Preserve machine-readable error codes exactly as returned.
- Use only structured fields that are present; do not invent omitted values.
- Retry only when the response indicates that retrying can succeed and the retry remains within the user's request.
- Do not raise the MCP access level, change NVIDIA App settings, or switch to another application as an implicit workaround.
- Preserve `desktop_resolution_unsupported` and `game_configuration_unavailable` setter errors and report their returned message literally. The desktop-resolution error includes the recommended resolution directly. Do not report the mutation as successful.
