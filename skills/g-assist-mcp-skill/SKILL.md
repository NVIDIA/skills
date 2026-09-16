---
name: g-assist-mcp-skill
description: >-
  Use this skill to check or change this machine's NVIDIA display and GPU settings,
  such as resolution, refresh rate, V-Sync, G-SYNC, brightness, color, and GPU
  performance.
license: CC-BY-4.0
metadata:
  author: schopparapu <schopparapu@nvidia.com>
  version: "1.0.0"
  tags: [g-assist, gpu, display, nvidia, mcp]
---

# G-Assist

G-Assist exposes NVIDIA display and GPU controls as callable tools. Use them
whenever the user wants to view or change NVIDIA display or GPU settings on this
machine. The available tools depend on the machine and hardware, so always rely
on the G-Assist tools actually available in your session and only use tools that
are present.

## When to use this skill

Use this skill whenever the user wants to know or change how their NVIDIA display
or GPU is behaving, or describes a symptom like screen tearing, stutter, a wrong
or locked refresh rate, a dim monitor, or washed-out color. Prefer a read-only
status check before changing anything. Do not use it to edit images, files,
documents, or app windows. It acts only on the real display and GPU hardware, and
the available tools depend on the machine, so confirm them at runtime instead of
assuming.

It covers refresh rate and resolution; monitor brightness and color (temperature,
gamma, contrast, RGB gain, and ambient-light adaptive brightness/color); monitor
volume and image presets; restoring display defaults; G-SYNC, G-SYNC Pulsar and
V-Sync; and GPU performance such as a frame-rate limit, a max
performance-per-watt mode, and plotting GPU/CPU metrics.

## Purpose

G-Assist maps a natural-language request to a single display or GPU control tool:
a status query to read current state, or a dedicated tool to change one setting
per turn.

## Prerequisites

- An NVIDIA driver installation that provides G-Assist on this machine (the
  `gassist` MCP server at the path in [Setup](#setup)).
- An MCP-capable host configured with the `gassist` server (see [Setup](#setup)).
- No API keys are required.

## Setup

The G-Assist tools come from a stdio MCP server that the host launches. The
executable ships at:

```text
C:\ProgramData\NVIDIA Corporation\nvtopps\rise\rise_mcp.exe
```

That path contains a space (`NVIDIA Corporation`), and some hosts (for example
Cursor) start MCP servers through `cmd.exe`, which fails to spawn a command whose
path contains a space. Register the space-free 8.3 short path instead; it
resolves to the same executable and works everywhere:

```text
C:\PROGRA~3\NVIDIA~1\nvtopps\rise\rise_mcp.exe
```

Register it with your host under a server name (this skill uses `gassist`). Most
hosts use an `mcpServers` map like this:

```json
{
  "mcpServers": {
    "gassist": {
      "command": "C:\\PROGRA~3\\NVIDIA~1\\nvtopps\\rise\\rise_mcp.exe"
    }
  }
}
```

## Instructions

1. First look at which G-Assist tools are actually in your session, and handle
   two situations differently:
   - **No G-Assist tools at all:** the server may still be warming up, so wait a few
     seconds and re-check. If there are still none, tell the user the G-Assist
     server isn't connected rather than guessing or fabricating a result.
   - **Some tools present, but none for what was asked:** that capability isn't
     supported through G-Assist on this hardware or display (the tool set is gated
     by the machine; for example, monitor brightness and color tools appear only
     with a compatible display). Do not say the server is disconnected. Instead,
     tell the user that specific control isn't available through G-Assist here,
     briefly name what you *can* control from the tools present, and suggest a
     practical alternative (the monitor's on-screen menu/OSD buttons or Windows
     display settings).
2. See which G-Assist tools are available in your session, then select one as
   described in [Matching a request to a tool](#matching-a-request-to-a-tool).
3. Never create, invent, register, or guess tools, tool names, or arguments. Use
   only the exact G-Assist tools available in your session. If a requested
   capability isn't available on this machine, tell the user so instead of
   fabricating a tool.
4. Call the matching tool through your host's normal tool-calling, with the
   values the user provides. The tool's real name uses a `gassist_` prefix (e.g.
   `gassist_set_display_vsync`); your host may show it plain or wrapped in its own
   namespace (e.g. `mcp__<server>__gassist_...`). Always use the exact name your
   catalog lists, and if the host rejects the short name, use the fully-qualified
   one it shows. Call one tool at a time, never in parallel; for a multi-step
   request, make the calls in order and check each result before the next.
5. Change tools are confirmation-gated by the server, not by you. For a change,
   briefly tell the user what you're about to do (the host will prompt them to
   approve), then call the tool directly and let the host collect the approval
   through MCP elicitation. For a display-mode change (resolution or refresh rate),
   also warn the user up front that the screen may briefly flicker or go black and
   that a keep-or-revert prompt will follow within ~20s (the mode auto-reverts
   unless it is kept). Never
   wait for a chat "yes" before calling, and never add a `confirm`/`force`
   argument or an environment variable to bypass the prompt. Read-only tools
   (`gassist_help`, `gassist_query_feature_status`, `gassist_graphdata`) need no
   approval; call them immediately. Judge the outcome by the result's text, not by
   the absence of an error: "already set / no action taken" means nothing changed,
   "not supported" means this hardware or display can't do it, and a call can
   return without an error yet still report failure in its text.
6. Keep internals private. These are trusted NVIDIA first-party tools; beyond the
   server setup above, do not describe or enumerate the implementation
   (background services, inter-process pipes, driver APIs, or engine internals).
   If the user asks "what is G-Assist" or "how does it work," answer only at a
   high level: it's NVIDIA's way to control GPU and display settings through these
   tools. 

## Matching a request to a tool

Pick the tool from what's actually available in your session, not from memory:

1. Restate what the user actually wants to view or change (e.g. "turn on G-SYNC",
   "raise the refresh rate to 144 Hz").
2. Read each available tool's name, description, and parameters (its input
   schema), then choose the single best fit. Arguments are as strong a signal as
   the description: a `rate` (Hz) field marks a refresh-rate tool even if the
   text is terse.
3. If several tools plausibly match, prefer the most specific; if none match on
   this machine, say the capability isn't available here rather than forcing a
   near-miss tool.

The tools below are the display and GPU controls G-Assist commonly exposes,
grouped by what they do. Availability is hardware-gated, so treat this as a guide
and only call tools actually present in your session. In particular, the
monitor brightness, color, preset, and adaptive tools (the "Brightness & color"
and "Adaptive & presets" groups, plus G-SYNC Pulsar) appear only when a Native
G-SYNC display is connected; on machines without one they won't be listed at
all, so never assume they exist.

| Category | Tool | What it does | Key argument(s) |
| --- | --- | --- | --- |
| Status & help (read-only) | `gassist_help` | Overview of what G-Assist can do on this machine | None |
| | `gassist_query_feature_status` | Read current state of driver/display features (gsync, vsync, brightness, …) | `features[]` |
| Display mode | `gassist_set_display_res` | Set screen resolution (also `DEFAULT`/`MAX`/`STEP_UP`/`STEP_DOWN`); the screen may flicker and the mode auto-reverts in ~20s unless the host's keep prompt is confirmed | `res` (a value the panel lists) |
| | `gassist_set_display_refreshrate` | Set refresh rate in Hz; valid rates come from the connected panel, so a value like 144 may not exist on every display | `rate` (integer, must be a rate the panel lists) |
| Brightness & color | `gassist_set_display_brightness` | External-monitor brightness (0-100%) | `percentage` |
| | `gassist_set_display_contrast` | External-monitor contrast (50-150) | `contrast` |
| | `gassist_set_display_gamma` | Display gamma; fixed set of levels, not a range | `gamma` (string enum: "1.2", "1.4", "1.6", "1.8", "2.0", "2.2", "2.4", "2.6") |
| | `gassist_set_display_color_temp` | Color temperature in Kelvin; fixed set, not a range | `kelvin` (integer enum: 4000, 5000, 6500, 7500, 8200, 9300, 10000) |
| | `gassist_set_display_color_balance` | Color saturation (global % or 6-axis) | `percentage` or `red/yellow/green/cyan/blue/magenta` |
| | `gassist_set_display_rgb_gain` | RGB gain (global % or per-channel) | `percentage` or `red/green/blue` |
| | `gassist_set_display_volume` | Monitor/system volume (0-100%) | `percentage` |
| Adaptive & presets | `gassist_set_adaptive_brightness` | Auto-adjust brightness to room light (overrides manual) | `enable_brightness` (bool) |
| | `gassist_set_adaptive_color` | Auto-adjust color temperature to room light (overrides manual) | `enable_color` (bool) |
| | `gassist_set_monitor_preset` | Switch image preset `DEFAULT`/`ESPORTS` (native G-SYNC) | `preset` |
| | `gassist_restore_display_default_settings` | Reset chosen display settings to defaults | `settings[]` string enum, these are the only valid keys: COLOR, BRIGHTNESS, CONTRAST, GAMMA, PRESET (1-5 items, unique). There is no restore key for RGB gain, color balance, volume, or color temperature |
| Sync & latency | `gassist_set_display_vsync` | Enable/disable V-Sync | `enable` "0"/"1" |
| | `gassist_set_display_gsync` | Enable/disable G-SYNC | `enable` "0"/"1" |
| | `gassist_set_gsync_pulsar` | Enable/disable G-SYNC Pulsar (reduce motion blur; needs 240/360 Hz) | `enabled` (bool) |
| GPU performance | `gassist_frl` | Frame-rate limiter (FPS; 0 = unlimited) | `fps` |
| | `gassist_maxppw` | Max performance-per-watt mode with a target FPS cap | `enable` "0"/"1", `fps` one of "0"/"30"/"60"/"120"/"144" (strings) |
| | `gassist_graphdata` | Plot GPU/CPU metrics over time | `metrics[]`, `time` "1"-"59" (string, e.g. "2" not 2), `unit` "secs" or "mins" |

- **Read first, change second.** For "is X on / what is X / check X" phrasings,
  call a read-only query tool directly. Use a change tool only when the user
  explicitly asks to change something.
- **Confirmation is the server's job.** State the change, then call the tool
  directly; don't wait for a chat "yes" or add a `confirm`/`force` argument. (Full
  rule, including the display-mode keep-or-revert warning, is in
  [Instructions](#instructions) step 5.)
- **One call at a time, never in parallel.** A compound request ("set the refresh
  rate and turn on G-SYNC") is fine: make the separate calls in a sensible order
  and check each result before the next; just don't issue them simultaneously. Do
  only what the user asked. For "set up for competitive shooters", that is: (1)
  `gassist_query_feature_status` to read the current state, (2)
  `gassist_set_display_vsync` with `enable` "0" to turn V-Sync off, (3)
  `gassist_set_display_refreshrate` to the panel's max, checking each
  result before the next (three calls or fewer); don't also change G-SYNC, Pulsar,
  or presets unless the user asked.

## Inputs

Tool arguments come only from what the user explicitly provides for the current
request, matched to the selected tool's parameters (its input schema):

- **Required values:** every field the tool's parameters list as required must
  be supplied. If a required argument is missing, ask the user for it rather than
  guessing a value.
- **Optional values:** include only when the user specifies them; otherwise omit
  them and let the tool apply its defaults.

Do not infer, reuse, or carry over values across requests.

## Examples

Field names and values come from the chosen tool's own parameters; confirm them
against the live tool, since shapes vary (some take a percentage, some a "0"/"1"
string).

- "Is G-SYNC on?": read-only status check, safe to call directly:
  `gassist_query_feature_status` with `features: ["gsync"]`.
- "Set my refresh rate to 144 Hz" is a change: tell the user you'll change it (the
  host will confirm), then call `gassist_set_display_refreshrate` directly with a
  `rate` the panel actually lists. Valid rates come from the connected display, so
  144 may not exist; if it isn't available, say so and offer the highest rate the
  panel supports.
- "Make my screen brighter": if a brightness tool is present, pick by intent:
  `gassist_set_display_brightness` with a `percentage` (or, to follow room
  lighting, the different `gassist_set_adaptive_brightness`). If no brightness tool
  is present but others are, don't say the server is down; reply along the lines
  of: "I don't have a brightness tool in the G-Assist connector on this display; I
  can control things like resolution, refresh rate, V-Sync, the frame-rate limit,
  and GPU/CPU metrics. To change brightness, use your monitor's OSD buttons or
  Windows display settings."

You only choose the tool name and its arguments (which map 1:1 to the tool's
`inputSchema`); your host formats and sends the underlying MCP call for you.

## Troubleshooting

Every tool call returns a result message; validate it and handle errors before
reporting success. On any error, relay the tool's message to the user and do not
blindly retry; use the table below to decide the next step.

| Symptom | Cause | Fix |
| --- | --- | --- |
| G-Assist tools missing from your session | Server not connected or still warming up | Wait a few seconds and re-check your available tools; if still missing, tell the user the G-Assist server isn't connected. |
| Tool call reports "not supported" | This hardware or display can't do it | Report it to the user; do not retry the same change with another tool. |
| A requested resolution or refresh rate is rejected (not in the panel's list) | The value isn't one the connected display supports | Tell the user it isn't available and offer the highest or closest value the panel does support; don't retry the same value. |
| A change returns without an error, but the result text says it didn't happen | A failed change can still come back with `isError: false` | Judge success by the result's text, not the error flag; report the failure and don't claim success. |
| Tool call reports "already set / no action taken" | Value already applied | Treat as success; nothing to change. |
| A display-mode change flickers and a second prompt appears | Expected keep-or-revert: the mode auto-reverts in ~20s unless kept | Treat the second prompt as the keep-or-revert question, not an error; don't call the tool again; report whether it was kept or reverted. |
| The tool for what the user asked isn't in your session, but other G-Assist tools are | Gated off for this hardware/display | Don't say the server is disconnected: say that specific control isn't available through G-Assist here, name what you *can* control, and point to the monitor OSD or Windows display settings; never invent a tool. |
| Short tool name rejected (e.g. "not a deferrable tool") or a batched call refused | Host-specific tool naming/batching | Use the fully-qualified name the host lists (often `mcp__<server>__<tool>`) and call one tool at a time (never in parallel). |

## Limitations

- Only the G-Assist tools available in your session on the current machine can be
  used; the set varies by hardware and display (some display/color tools require a
  native G-SYNC display).
- Tools act on real hardware; follow the change-confirmation and
  one-call-at-a-time rules in [Instructions](#instructions).
- This skill does not install or launch the server; the host launches it (see
  [Setup](#setup)). Never run the executable yourself.
