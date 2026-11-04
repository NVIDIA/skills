---
name: nvidia-broadcast
description: Use when controlling NVIDIA Broadcast through MCP to apply effects, process local media, select devices, or change camera resolution, or when Broadcast is missing or too old to expose the gateway and the user wants it installed; not for Broadcast app settings outside the MCP gateway.
metadata:
  author: "NVIDIA Broadcast Team <broadcast-skills-dev@nvidia.com>"
  version: "1.0.0"
  license: CC-BY-4.0 AND Apache-2.0
  tags:
    - nvidia-broadcast
    - mcp
    - camera
    - microphone
    - audio-effects
    - video-effects
    - studio-mode
  domain: media
---

# NVIDIA Broadcast

## Hard rules

1. **Launching:** start Broadcast only by running
   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<skill_dir>\scripts\launch_broadcast.ps1"`
   with the `terminal` tool. When running inside WSL, follow *Launch and version-check through
   interop* in `references/connection.md` instead. Except for the security-policy fallback below
   and that WSL interop procedure, never invoke
   `NVIDIA Broadcast.exe` directly — not via `terminal`, `execute_code`, `subprocess`, or
   `Start-Process`.

   **Security-policy fallback:** use this only if security policy blocks the launcher script.
   Confirm that `NVIDIA Broadcast.exe` is not already running, then run:
   `powershell.exe -NoProfile -Command "Start-Process -FilePath ([Environment]::ExpandEnvironmentVariables('%ProgramFiles%\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe')) -ArgumentList '--launch-hidden' -WindowStyle Hidden"`
   with the `terminal` tool.

2. **Command tools:** follow *Which tool to run commands with* below.

3. **Never read effect state from disk** — not `AppSetting.json`, not the registry, not logs, not
   the UI. Those are saved settings, not live state. Exception: `gateway.json` may be read and
   decoded only to obtain the gateway `port` for rule 4; it is connection configuration, not effect
   state.

4. **Connecting to the gateway — exactly these steps, no other path:**
   1. Before launching or registering the gateway, check the standard executable path and read its
      `ProductVersion` without launching it. If the executable is missing or its version is 2.2.x
      or older, stop the connection sequence and follow *Installing or updating NVIDIA Broadcast*.
      Do not return the restart-only response below.
   2. Check Broadcast is running: `tasklist /FI "IMAGENAME eq NVIDIA Broadcast.exe"` via `terminal`.
      If Broadcast is installed but not running, launch it immediately (rule 1); do not ask the user
      to start it manually. A `gateway.json` left from an earlier run proves nothing about whether
      the gateway is live.
   3. Whether already running or just launched, retry for up to 60 seconds: reread and base64-decode
      `%APPDATA%\nvidia-broadcast\gateway.json` → `{"port": <number>}`. Use the port from the latest
      successful read. Decode `gateway.json` within the command and output only the port. Never print
      its base64 contents because credential redaction may mask them.
   4. Register `http://127.0.0.1:<port>/gateway` with your MCP client as a Streamable HTTP server.
      You are the MCP client — register it yourself.
   5. Verify the server reports its name as `nvidia-broadcast` via the MCP client's server info.
      If it does not, treat NVIDIA Broadcast as unavailable, use the restart response below, and stop.
   6. Check whether the NVIDIA Broadcast MCP tools required for the user's request are available.

   If the required tools are available, the connection is ready; proceed with the requested
   operation.

   **Stop if any of these occurs — report and go no further:**
   - Registration fails, or
   - the server name is not `nvidia-broadcast`, or
   - after registering, any required NVIDIA Broadcast MCP tool is not in your tool list.

   Reply only: *"NVIDIA Broadcast MCP tools are unavailable in this session. Please restart your MCP
   client and retry."* Then stop. Do not add rationale, gateway URLs, port details, configuration
   instructions, or post-restart steps. Do not substitute another source or guess which effects are on.

5. **Only after completing rule 4, if any required NVIDIA Broadcast MCP tool is still not in your
   tool list, that is the final answer. Stop.** Its initial absence is not a reason to skip launch
   or registration.
   Do not try to make it work. Specifically, never:
   - Write, generate, or run any code that speaks MCP or HTTP to the gateway —
     `urllib`, `http.client`, `socket`, `requests`, `httpx`, `curl`,
     `Invoke-WebRequest`, or anything else. **A correct, spec-compliant client is
     still a violation.** The prohibition is on you speaking the protocol at all,
     not on which library you choose.
   - Search for, install, or run an MCP client binary — `which mcp`, `mcp exec`,
     `npx @modelcontextprotocol/cli`, `pip install mcp[cli]`.
   - Scan ports, hunt the filesystem for an endpoint, or delegate to another agent
     or session.
   Use exactly the restart response from rule 4, then stop. A state read obtained by any other
   route is not a valid substitute, and a guessed effect list is simply wrong.

6. **Recover from MCP connection failures.** If any NVIDIA Broadcast tool call fails because the
   MCP connection was closed, refused, reset, disconnected, or otherwise became unavailable, repeat
   the connection sequence in rule 4:
   1. Check whether `NVIDIA Broadcast.exe` is running.
   2. If it is not running, launch it using rule 1. If it is already running, do not launch another
      instance.
   3. Reread `gateway.json`, obtain the current port, and register the gateway again.
   4. Verify the server reports its name as `nvidia-broadcast`. If it does not, use the short
      restart response from rule 4 and stop.
   5. Check whether the NVIDIA Broadcast MCP tools required for the user's request are available.

   If the tools are still unavailable, use the short restart response from rule 4 and stop. After
   reconnecting, retry a read-only or idempotent tool call at most once. Do not automatically repeat
   a non-idempotent request when the connection closed after submission because it may already have
   been accepted.

   Do not run this recovery sequence for normal gateway or tool errors such as invalid arguments,
   `ambiguous_frame_rate`, `device_not_selected`, `file_not_found`, rate limiting, or an effect
   operation failure.

## Purpose

NVIDIA Broadcast applies AI effects to your camera, microphone, and speaker, and can apply the same effects to local media files. You drive it over a local MCP gateway that NVIDIA Broadcast runs on loopback. It also lets you check and change AI effects, select Studio Voice microphone profiles, manage camera, microphone, and speaker devices, and change camera resolution.

## Prerequisites

- Windows host with NVIDIA Broadcast installed. **2.2.x and older have no MCP gateway** and cannot be driven by this skill at all. When it is missing or too old, you may **offer** to install it from NVIDIA's update service — see *Installing or updating NVIDIA Broadcast* below. Never install without asking.
- Standard executable path: `%ProgramFiles%\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe`.
- MCP gateway config readable at `%APPDATA%\nvidia-broadcast\gateway.json`.
- The current agent runtime acts as the MCP client and connects through Streamable HTTP over loopback.
- **Running inside WSL:** the gateway lives on the Windows side, so both the endpoint and
  `gateway.json` are reached differently — see *If you are running inside WSL* below.

## Inputs

- `set_effects`: either `effects[]` with 1-32 unique effect IDs (each `effectId` + `enabled`, optional `params`), or `action: "restore_previous_live"`.
- `submit_file_processing`: `_version: 1`, absolute `inputPath`, 1-32 unique `effects[]`; optional `batchId`, `outputFolder`, `outputPath`. There is no overwrite input — a fresh collision is rejected, while resume/retry replaces only that job's own partial output.
- `list_devices`: optional `kind`. `set_active_device`: `kind` + fresh `deviceId`, plus optional camera `width`, `height`, `frameRate`. `set_camera_resolution`: fresh camera `deviceId`, `width`, `height`, optional `frameRate`.
- Do not reuse device IDs from earlier sessions, and do not guess frame rates. Supply `frameRate` only when the user explicitly specified or selected it; the camera's current frame rate is not the user's choice for a new request.

## Which tool to run commands with
Use `terminal` for shell commands and `execute_code` for Python. **Never call `browser_exec`** —
its `code` parameter makes it look like `execute_code`, but it runs that code inside a browser
session and opens Chrome. No task in this skill uses a browser.

## Connecting

### MCP client identity

**In this skill, “MCP client” means the current agent runtime executing this skill—not a separate
application, CLI, profile, agent, or session.** This definition applies throughout the skill.

Do not conclude that no MCP client exists merely because Broadcast tools were not preloaded. Do not
search client configuration files, invoke another client such as Claude Code, use a browser,
browser-based MCP connection, browser automation, or browser-based remote debugging, delegate to
another agent or session, or ask whether another MCP client is installed.

**Streamable HTTP is the default and recommended transport.** Register the gateway with your
MCP client and let the client speak the protocol. Do not hand-roll HTTP requests — it is a normal
Streamable HTTP MCP server, and your client already handles protocol version and transport. Add it
to your MCP client configuration:

```json
{
  "mcpServers": {
    "nvidia-broadcast": {
      "type": "http",
      "url": "http://127.0.0.1:18100/gateway"
    }
  }
}
```

Or, for clients that register servers from the command line:

```bash
<mcp-client> mcp add --transport http nvidia-broadcast http://127.0.0.1:<port>/gateway
```

Substitute the client's own command for `<mcp-client>` and the resolved gateway port for
`<port>`; subcommand and flag names vary by client, but the three values being registered are the
same ones in the entry above.

`18100` is the default port and is correct on almost every install. No API key or header is
required for loopback MCP clients. This URL and port are for the client's configuration, not for
connecting directly.
### If the default port does not work

The gateway takes the first free port in `18100`-`18109` and publishes the one it actually
bound in `%APPDATA%\nvidia-broadcast\gateway.json` — base64, decoding to `{ "port": <number> }`.
Re-read that file on every retry rather than caching the port; a restart can move it, which is
the one case where a registered URL goes stale.

1. Before launching or registering the gateway, read the executable's `ProductVersion` without
   launching it.
   **If the executable is not there, NVIDIA Broadcast is not installed.** **If it is 2.2.x or
   older, that build has no MCP gateway** and no amount of retrying will produce one. Either way,
   stop the connection sequence, tell the user what you found, and **offer to install or update
   it** — see *Installing or updating NVIDIA Broadcast* below. If they decline, or if installing is
   genuinely impossible from here — no Windows interop, no PowerShell — point them to
   <https://www.nvidia.com/broadcast-app/> and stop. **An earlier failed attempt is not one of those
   reasons**, and neither is a download that had to be retried.
2. Check whether `NVIDIA Broadcast.exe` is running. If it is not, launch it by running the bundled
   launcher script — the primary launch command. Use the security-policy fallback in rule 1 under
   Hard rules only if policy blocks the script:
   ```bash
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<skill_dir>\scripts\launch_broadcast.ps1"
   ```
   `<skill_dir>` is the absolute path in the `skill_dir` field of the `skill_view` result you
   already have. **Run it with the `terminal` tool.** Never with `browser_exec` — it takes a `code`
   parameter just like `execute_code`, but it runs that code inside a browser session and will open
   Chrome. Nothing in this skill ever needs a browser.
   Except for the security-policy fallback in rule 1 under Hard rules, do not write your own launch command or invoke
   `NVIDIA Broadcast.exe` directly with `subprocess.run`, `check_output`, `Start-Process -Wait`, or
   any command that waits for or captures output from the Broadcast process. Never launch a second
   copy when the process is already running.
3. Whether Broadcast was already running or was just launched, retry for up to 60 seconds: reread
   and decode `gateway.json`, then register `http://127.0.0.1:<port>/gateway` through the MCP client.
   If nothing answers, use the restart response in hard rule 4 and stop.

   **Discovery is exactly these two places and nothing else:** `gateway.json` and ports
   `18100`-`18109`. Do not search the filesystem for gateway or config files, and do not read
   environment variables looking for an endpoint or credential — the loopback gateway publishes
   its port only in `gateway.json` and needs no key. When both places come up empty, stop endpoint
   discovery.

   **Probe only through your MCP client.** Register the candidate URL and let the client connect;
   a failed registration is the probe result. Never reach the gateway from a shell: no command-line
   web clients, no throwaway request scripts, and no scanning the port range yourself — see
   *Safety and scope* for the full list of prohibited alternatives. If your
   client cannot register MCP servers at runtime, follow *If your client cannot register MCP
   servers* below. Not being able to probe never licenses shell tooling as a substitute.

### If your client cannot register MCP servers at runtime

Some clients only load MCP servers from a config file at startup. That is a **client configuration
problem, not a Broadcast problem**, and it is never a reason to fall back to "just turn it on in the
app yourself" without checking anything. Do the version check in step 1 first — it reads the
filesystem and needs no MCP client — then report whichever of these you actually found:

- **Broadcast is missing, or is 2.2.x or older.** Offer to install or update it exactly as below.
  Installing needs no MCP client, and it is worth doing before any config change.
- **A 2.3+ build is installed.** Nothing is wrong with Broadcast. Tell the user to add the gateway
  to their MCP client's configuration and restart the client, quoting the entry from *Connecting*
  with the port you resolved. Say plainly that you cannot drive Broadcast until they do.

**Name which of the two it is.** "The gateway isn't available to this session" on its own leaves the
user unable to tell whether the app is missing, too old, or merely unconfigured — and sends them off
to click through the UI when one config line, or an install, would have fixed it for good.

### If you are running inside WSL

The gateway runs in the Windows Broadcast process, so before registering anything establish that
`127.0.0.1` in the distro really is Windows loopback, and read the port through the mounted Windows
drive. Full procedure in `references/connection.md`; in short:

1. **Detect WSL** from `/proc/sys/kernel/osrelease` — it contains `microsoft` or `WSL`. Do not
   read environment variables to decide this.
2. **Check `wslinfo --networking-mode`.** On `mirrored` — or a WSL 1 distro — loopback is shared,
   so register `http://127.0.0.1:<port>/gateway` exactly as on Windows. On `nat` (the default) the
   gateway is **unreachable from the distro**: stop, tell the user, and offer mirrored networking,
   an MCP client running on Windows, or a tunnel terminating on Windows loopback. **Never
   substitute another address for loopback** — not the nameserver in `/etc/resolv.conf`, the
   default route from `ip route`, nor `$(hostname).local`; nothing is published off loopback and
   the gateway rejects every non-loopback `Host` by design.
3. **Go through Windows interop** for the rest: `wslpath "$(cmd.exe /c 'echo %APPDATA%' | tr -d
   '\r')"` to reach `gateway.json`, and `powershell.exe -NoProfile -Command '...'` to launch,
   version-check, and install. If the Windows drive is not mounted or `powershell.exe` is missing, say so and
   stop — do not search the Linux filesystem for `gateway.json` and do not look for a Linux-side
   substitute; ask the user to start Broadcast on Windows.

See `references/connection.md` for the version-check command, endpoint discovery, startup,
access, WSL specifics, and rate limits.

On connect, the server's `instructions` already list the **exact effectIds available
right now, grouped by section**, and the `set_effects` schema constrains `effectId` to
that live set. So you can act immediately — you do **not** need a discovery call first.

## Installing or updating NVIDIA Broadcast

Only when the version check above found **no executable at the standard path**, or a
**2.2.x-or-older** build. Never as a workaround for any other failure, and never as a side effect
of an unrelated request. Full procedure, commands and failure modes in `references/installation.md`.

1. **Ask what is available.** `GET https://ota.nvidia.com/release/available?product=rtxb&channel=OFFICIAL&version=$installed&cpuArchType=$arch`,
   where `$installed` is the installed `ProductVersion` (`0.0.0.0` when Broadcast is not installed)
   and `$arch` is `aarch64` or `x86_64`, read from the registry as shown next — never chosen or
   defaulted. Substitute both values; never send the names themselves.
   **Check both before sending** — `$installed` a dotted version number, `$arch` exactly `aarch64` or
   `x86_64`. The service never rejects a bad value: it answers `[]`, which reads as "nothing newer".
   Read the architecture from the registry, **not** the process — `$env:PROCESSOR_ARCHITECTURE` says
   `AMD64` inside x64 emulation on Windows on ARM, which fetches the wrong installer. Use
   `$native = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" -ErrorAction Stop).PROCESSOR_ARCHITECTURE`,
   map `AMD64` or `x86`→`x86_64` and `ARM64` or `ARM`→`aarch64` into `$arch`, and **throw on anything else — never default to one**.
   Those two values are the only ones the service accepts, and each serves a different installer.
   The reply is a **list of applicable builds, newest first**
   and already filtered to versions above the one you sent — take the first element. An empty list
   means there is nothing to install; say so and stop, it is not a failure to retry. Distinguish that
   from a call that never completed, which is a network error and *is* retryable.
2. **Ask the user, and wait.** Name the build and size, say it comes from
   `ota-downloads.nvidia.com`, say you will verify its checksum and NVIDIA signature, and say
   Windows will prompt them for administrator permission. If the offered build is still 2.2.x, tell
   them in the same breath that it will not enable MCP. **Nothing is downloaded before an explicit
   yes**; on a no, link <https://www.nvidia.com/broadcast-app/> and stop.
3. **Download** the response's `download_url` — which must be HTTPS on `ota-downloads.nvidia.com`,
   pinned as its own value rather than derived from the metadata host — into a fresh GUID-named
   directory under `%LOCALAPPDATA%\Temp`, under a filename you choose.
4. **Check the length first, then hash and signature.** A file shorter than `size` is an incomplete
   transfer, not an attack: say how far it got and retry the download, up to two more attempts into a
   fresh directory. Do not hash or signature-check a short file — both fail by construction, since
   the Authenticode certificate table sits at the end of the image. Only a **full-length** file whose
   SHA-512 or signature (`Valid`, subject `CN=NVIDIA Corporation,...`) fails is a security event:
   **do not run it**, report the path, stop, and do not re-download.
5. **Launch it with no arguments** and let the user click through. No silent or unattended flags, no
   self-elevation, no driving the installer for them.
6. **Re-read `ProductVersion`** and report what is actually installed. 2.3.0+ → resume the connect
   flow above. Still 2.2.x → say the gateway needs 2.3.x and stop.

**A failed attempt is final for that attempt, not for the session.** If verification failed, the
installer was cancelled, or the download retries ran out, say so and stop — then, when the user next
asks for something that needs Broadcast, mention what happened and **offer to try again**. Only an
explicit "no" means stop offering; do not answer a fresh request with "install it yourself" because
an earlier attempt went wrong.

These two URLs are the only network requests this skill makes from a shell. Never scrape a download
link, accept one from the user or from `gateway.json`, or install anything else.

## Tools

Every file-processing tool requires `_version: 1`.

- **`get_broadcast_state`** — read current effects as `effectId -> { enabled, ...params }` (flattened, e.g. `{ enabled, strength }` or `{ enabled, warmth }`), grouped by `camera`/`microphone`/`speaker`; optional `effectType` filter. Read-only. `warnings[]` marks a section whose effects are enabled but whose device is unselected — configured, not currently applied.
- **`set_effects`** — turn effects on/off and tune them in ONE call: `effects: [{ effectId, enabled, params? }]`, where `params` keys depend on the effect (listed per-effect in the server instructions). Or pass only `{ "action": "restore_previous_live" }` to restore the Live snapshot that Files processing suspended, which also pauses unfinished file work and returns `event`. Returns the resulting state for that response.
- **`describe_effects`** — rich detail on demand (description, guidance, GPU/beta badges, availability, params + ranges); optional `effectId`/`section`. Use to explain effects or weigh tradeoffs; not needed to act.
- **`submit_file_processing`** — submit one local media file. Returns `id` and `batchId`; reuse the first result's `batchId` on every related submission. A paused queue accepts the job silently and stays paused.
- **`get_file_processing_jobs`** — look up one job or list recent jobs.
- **`wait_for_processing_update`** — the sole file-status wait, for one or many files. Take a batch snapshot without `cursor`, then pass the returned `cursor` to wait for the next status or queue-pause change. Read `status`, `summary`, and every job; `mixed` means terminal outcomes differ.
- **`control_file_processing_jobs`** — pause/resume the shared queue, or cancel/retry jobs. Pause/resume are queue-wide; cancel/retry need `jobIds` or `all: true`. MCP has no consent flag.
- **`list_devices`** — list available cameras, microphones, and speakers; `kind: "camera"|"microphone"|"speaker"|"all"` (default `"all"`). Returns the active device, and for cameras the current resolution plus `availableResolutions` (`label`, `width`, `height`, `frameRate`) for the current Video Super Resolution / Video Frame Generation mode.
- **`set_active_device`** — switch the active camera, microphone, or speaker with `kind` plus a fresh `deviceId` from `list_devices`. For cameras, add `width`+`height`+optional `frameRate` to switch device and resolution in one call. Camera switches restart the stream automatically (allow 2–5 seconds).
- **`set_camera_resolution`** — change the active camera's resolution with `deviceId` plus `width`+`height`+optional `frameRate`; include `frameRate` only after the user explicitly specifies or selects it. When FPS is unspecified, omit `frameRate` so the gateway can accept a unique mode or return the available choices for an ambiguous size.

Invalid requests — including unavailable or ambiguous resolutions — return a gateway tool error
before anything changes. How to read the success-path fields (`success`, `alreadyActive`,
`alreadySet`, `frameRateMismatch`, `failed[]`, `skipped[]`) is under **Instructions**.

Effect ids and params mirror the app's UI (e.g. "Virtual key light" → `virtual_key_light` with `warmth`). The exact ids + each effect's params are in the server `instructions`. See `references/effects.md`.

For file processing, nest every effect's options under `params`, exactly as with `set_effects`.
Frame Generation accepts `multiplier` 2|4 and `model` "performance"|"balanced"|"quality".

## Studio Voice microphone profiles

Set the Studio Voice profile through `params.micProfile` in either `set_effects` or
`submit_file_processing`. Read the capability-aware enum from the tool schema or server
instructions; the current supported profiles are `default`,
`bright`, `full`, and `warm`. Do not send `flat`, `colored`, or `thin` when they are absent
from the advertised enum.

## Examples

Complete tool calls — single and combined effects, per-effect params, background replace, frame
generation, Studio Voice profiles, `restore_previous_live`, state reads, device switches and
resolution changes — are in `references/examples.md`. Load it when you need the exact JSON shape.

## Worked flows

Multi-step tasks have an expected shape. Follow it exactly.

### Several files as one batch

1. Submit the **first** file with no `batchId` — the response creates one:
   ```json
   { "name": "submit_file_processing", "arguments": { "_version": 1,
     "inputPath": "C:\\Users\\me\\Videos\\a.mp4",
     "effects": [{ "effectId": "background_blur", "params": { "strength": 0.6 } }] } }
   ```
   Keep the returned `batchId`.
2. Submit **every other file with that same `batchId`**.
3. Take **one** snapshot for the whole batch — omit `cursor`:
   ```json
   { "name": "wait_for_processing_update", "arguments": { "_version": 1, "batchId": "<batchId>" } }
   ```
4. While `status` is `queued` or `in_progress`, call it again with the same `batchId` plus the
   `cursor` the previous response returned, and an optional `timeoutMs`.
5. Stop at a terminal `status` — `completed`, `failed`, `cancelled`, or `mixed` — then report `summary` and name which files landed in which outcome from `jobs`. **`mixed` is not success.**

Do **not** wait once per file, and do **not** poll `get_file_processing_jobs` in a loop.

### Switch camera and set its resolution

1. List first — IDs and available modes change between turns:
   ```json
   { "name": "list_devices", "arguments": { "kind": "camera" } }
   ```
   Take the camera's `id`, then pick a row from **that camera's** `availableResolutions` rather than assuming a size is supported. If the size you want appears at more than one `frameRate`, ask the user which — do not choose for them. The current FPS does not count as an explicit user choice.
2. After the user specifies or selects an FPS, switch device **and** resolution in ONE `set_active_device` call, passing `kind`, the `deviceId` from step 1, `width`, `height`, and the chosen `frameRate`. If FPS was not specified and the size has only one advertised rate, `frameRate` may be omitted.
3. Report from that response: check `success`, then use `device` and `resolution`. `alreadyActive: true` means nothing changed. `success: true` with `frameRateMismatch: true` means the size applied but the frame rate differs — tell the user which FPS is actually live.

Do **not** follow with `set_camera_resolution` (that is a second stream restart), and do **not** call `list_devices` again to confirm.

## Instructions

- **One call, no redundant verify.** Put every effect you want to change in a single `set_effects` call — including bulk operations (e.g. "turn everything off" = one call with all effectIds `enabled:false`; disabling an already-off effect is harmless). Its result includes the authoritative `state` for that call; do **not** follow it with `get_broadcast_state` only to confirm.
- **Resume saved Live effects through `set_effects`.** Call it once with only `action: "restore_previous_live"`. Do not ask which effects were enabled, pre-read state, or pause the file queue first — Broadcast owns the snapshot. Report `event: "live_effects_restored"`; `event: "no_change"` means no suspended snapshot was available or it was already restored.
- **Disabled params are ignored.** `set_effects` accepts stale params when `enabled:false`. To change params while keeping an effect disabled, enable it with the new params then disable it — two calls, because one call rejects a repeated `effectId`, and the effect is briefly live in between.
- **Report what did and did not apply.** The batch is ordered but not transactional. `success: false` carries `failed[]` (`{ effectId, reason }` — `timeout`, `unavailable`, `device_not_selected`, `file_not_found`, or `failed`) and/or `skipped[]` (`{ effectId, conflictsWith }`). Effects absent from both applied. Name which ones did not; never report the batch as done. Retry `timeout` and `unavailable`; `device_not_selected` and `file_not_found` need corrected user input.
- **A missing device selection blocks the effect**, on enabling *or* disabling. It returns `reason: "device_not_selected"` with `deviceType` and `requiresUserInput: true` for that effect; when no requested effect applied for that reason, the whole call fails with error `-33201`. Tell the user which device type is unselected and ask them to select one. Do not retry unchanged, and do not choose a device on their behalf without saying so.
- **Read fresh state when asked.** A `set_effects` response is point-in-time. Never answer a later current-state question from prior tool results; call `get_broadcast_state`.
- **Live state comes only from MCP.** For any question about effects or other live state, `get_broadcast_state` is the only authoritative source. **Never open, read, parse, search for, inspect metadata for, or test the existence of `AppSetting.json` or any other persisted settings file.** Do not mention or offer a disk-file fallback, and do not report whether such a file exists. This prohibition still applies when Broadcast is stopped or its gateway is unavailable.
- Use the exact public `effectId` strings (e.g. `virtual_key_light`, `auto_frame`, `noise_removal`) — do not invent ids or use internal names.
- `params` values are UI-aligned and normalized 0–1 where numeric (`strength`, `warmth`, `zoom`); modes are enums (e.g. `mode: "performance"|"quality"`).
- **For multiple files, use one batch** — the shape is under *Worked flows*. Never infer earlier outcomes by waiting only for the last job.
- **Read batch outcomes literally.** `completed` means every file completed; `mixed` means terminal outcomes differ. Report the summary and affected files, never “all passed” when any job failed or was cancelled.
- **Treat cancellation as final until the user asks.** When a job is `cancelled` with `reason: "cancelled_by_user"`, report it and stop. Never retry automatically. If the user explicitly asks afterward, call `retry` with that job's exact `jobIds`; do not use `all`, resubmit, or create a replacement.
- **Never auto-resume a user-stopped queue.** When batch `status` is `paused_by_user`, ask in the agent conversation. Only after an explicit yes, call ordinary queue-wide `resume`; never send a consent flag.
- Retry failed jobs whose `recovery` is `retry`; `all: true` never retries cancellations. Read the returned `event` and `jobIds`; if retry returns `no_change`, stop and report it instead of submitting replacements.
- Every tool error carries concise text plus `structuredContent.error`, schema rejections included, so one parse path covers all of them; JSON-RPC protocol failures use a top-level `error` instead. When `data.requiresUserInput` is true, **stop and ask the user** — `ambiguous_frame_rate` (with `data.frameRateOptions`) means ask which fps, `device_not_selected` means ask them to select that device, `file_not_found` means ask for a real local path. Never substitute a value of your own.
- **Read `error.retryable` before giving up.** `-32602` means your arguments were wrong and will stay wrong; fix them. `-32603` with `retryable: true` means the app was busy or not ready — the same call may well succeed shortly, so back off and retry rather than reporting failure.
- **Always call `list_devices` fresh** — every time the user asks what is available, and before every `set_active_device` or `set_camera_resolution` to get the current `deviceId`. Devices can be plugged or unplugged between turns; never reuse an ID from an earlier turn or session.
- **Never infer an unspecified frame rate.** The camera's current FPS, a previous setting, the first advertised mode, and 30 FPS are not permission to choose that rate. If `list_devices` shows multiple FPS values for the requested size, stop before a setter call, present those values, and ask the user to choose. If ambiguity is not known yet, call the setter with `width`+`height` and omit `frameRate`; when it returns `requiresUserInput: true` with `frameRateOptions`, present those options and ask. Supply `frameRate` only after the user explicitly specifies or selects it, then retry.
- **Prefer one `set_active_device` call with resolution over two calls.** Switching camera and changing resolution together saves one stream restart and is faster.
- **One call, no verify (devices).** After a successful `set_active_device` or `set_camera_resolution`, use `device` / `resolution` from that response — do **not** follow with `list_devices`. `alreadyActive: true` means the device was already selected and no stream restart happened; `alreadySet: true` means the camera was already at that resolution.
- **Check `success` in non-error results from both device tools.** `success: false` means the switch or width/height change did not fully apply, and the response carries what you need to recover without another `list_devices`: for `set_active_device`, `deviceConfirmed: true` with `resolution.confirmed: false` means the device switched but the resolution did not (read `resolution.actual` and `state`), while `deviceConfirmed: false` includes `availableDevices` and `state`; for `set_camera_resolution`, `actual` and `state.cameras` show what the camera set. If width/height applied but only FPS differed, both return `success: true` with `frameRateMismatch: true` — tell the user the applied FPS.
- **Use the file-status wait.** `wait_for_processing_update` wakes for job-status and queue-pause changes, not progress ticks.

## Safety and scope

- **Never inspect the environment.** Do not run `env`, `printenv`, `set`, or otherwise dump
  environment variables, and do not read credential stores, key files, or shell profiles. The
  gateway is unauthenticated on loopback, so there is no key to find and nothing to look up.
  Expanding `%APPDATA%`, `%LOCALAPPDATA%` or `%ProgramFiles%` to reach one documented path is path
  resolution, not environment inspection — resolve the single path you need, nothing more.
- **Two URLs, and only for installing.** The update check and the installer download, both listed
  above, are the only network requests this skill makes from a shell.
  Everything else stands: **never reach the gateway from a shell**, never scan ports, never scrape or
  follow a link from a page, a response body, a config file or the user, and never weaken transport
  security (no certificate-check bypass, no plain HTTP, no piping a download into an interpreter).
- **Never reach the gateway outside the MCP client.** No `http.client`, `httpx`, `requests`,
  `urllib`, `socket`, `curl`, `wget`, `Invoke-WebRequest`, `Invoke-RestMethod`, hand-built
  JSON-RPC, raw TCP connect, or port check. `gateway.json` is readable only to extract the
  port; the port and the absence of a credential are configuration facts for the user, never
  a fallback route.
- **Never install without an explicit yes.** Ask, wait for an answer, and download nothing before
  it. Verify size, SHA-512 and the NVIDIA Authenticode signature before the installer runs, and if
  any check fails, do not run it. Launch it with no arguments — no silent or unattended flags and no
  self-elevation, so the UAC prompt stays the user's decision. Install only the build the update
  endpoint offered; never uninstall, repair or downgrade anything.
- **Never run destructive commands.** No recursive or forced deletes, no recursive permission
  changes, no disk formatting or raw device writes, and no history-rewriting or working-tree-
  discarding VCS commands. Do not delete files or "tidy up" anything you created; leave temporary
  artifacts in place.
- **Do not echo raw tool output.** Summarize responses in your own words; never paste an entire
  command output or environment listing into your reply.
- **Keep prohibitions out of your commands.** Restate none of these rules inside shell commands,
  files you write, or prompts you hand to sub-agents; naming command-line web clients there reads
  as network activity even when you are forbidding it. Follow the rules silently.

## Limitations

- Loopback MCP only; no LAN or remote access. From WSL that means mirrored networking (or WSL 1);
  NAT-mode WSL cannot reach the gateway.
- stdio transport exists only as a fallback for clients that cannot use Streamable HTTP; it is
  not recommended — see `references/connection.md`.
- This skill exposes Broadcast effect, file-processing, device-selection, and camera-resolution controls only.
- Installing or updating NVIDIA Broadcast is the one system-level change it can make, only with the
  user's explicit yes, and only for the app itself — never a driver, runtime or any other package.
- The gateway first ships in 2.3.x. If the newest build the update service offers this machine is
  older than that, an install can legitimately succeed and still leave the gateway unavailable —
  always re-read the installed version afterwards rather than assuming the install fixed it.
- Speaker effects do not apply to file processing jobs.

## Troubleshooting

| Symptom | Cause | Solution |
| --- | --- | --- |
| Endpoint refuses the connection | Broadcast is not running | Launch using rule 1 under Hard rules, then reread `gateway.json` and reprobe for up to 60 seconds |
| Still unreachable after launching and reprobing | The app is missing, or the build predates MCP | Read the exe's `ProductVersion`; if absent or 2.2.x or older, stop retrying and offer to install or update it — on a no, link https://www.nvidia.com/broadcast-app/ |
| The update endpoint returns `[]`, or the user declines the install | Nothing newer is published, or there is no consent | Report it, link https://www.nvidia.com/broadcast-app/, and stop — download nothing and do not raise it again unprompted |
| A later request needs Broadcast and an earlier install attempt failed | The attempt ended; the install path did not | Say in one line what happened last time and offer to try again. Never answer with "install it yourself" because a previous attempt failed |
| The downloaded installer is shorter than `size` | Incomplete transfer — the most common failure in this flow | Not a security event. Report how far it got and retry the download, up to two more attempts in a fresh directory |
| A **full-length** installer fails SHA-512 or the NVIDIA signature check | Substituted or corrupted content | Do not run it. Report which check failed and where the file is, then stop — do not re-download it now, but offer a fresh attempt when the user next asks for something needing Broadcast |
| `ProductVersion` is still 2.2.x after installing | The build that was installed predates the gateway | Say the MCP gateway needs 2.3.x, link the download page, and stop probing |
| `127.0.0.1:<port>` refused, and you are inside WSL | NAT-mode WSL has its own loopback; Windows loopback is unreachable | Check `wslinfo --networking-mode`; on `nat` report it and offer mirrored networking, a Windows-side client, or a loopback-terminating tunnel — never swap in the host IP |
| Tool error `-33102` or `-33103` | Rate limited, or too many in-flight calls | Back off using the limit in `data`, then retry |
| `requiresUserInput: true` with `frameRateOptions` | That frame size exists at several frame rates | Ask the user which fps, then retry with `frameRate` |
| Error `-33201`, or `failed[].reason` is `device_not_selected` | That effect's camera, mic, or speaker is not selected | Name the device type and ask the user to select it; do not retry unchanged or pick one silently |
| Batch status is `paused_by_user` | The user stopped file processing | Ask in the agent conversation; after yes, call ordinary `resume` |
| Job status is `cancelled` | The user cancelled processing | Report cancellation and stop. If explicitly asked afterward, retry with that job's exact `jobIds`; never use `all` or resubmit |
| Batch status is `mixed` | Terminal outcomes differ | Report `summary` and per-job statuses; do not claim all files passed |

`references/troubleshooting.md` has the full table, including rejected effect IDs, `warnings[]`,
and `success: false` recovery.

## More detail (load only if needed)

- `references/effects.md` — categories, effectIds, strength semantics, incompatibilities.
- `references/examples.md` — the full call-example set.
- `references/file-processing.md` — file workflow, output paths, progress.
- `references/connection.md` — endpoint discovery, startup, access, rate limits.
- `references/installation.md` — consent, download, verification, and install of a missing or too-old build.
- `references/troubleshooting.md` — the full symptom table.
- `references/mcp-tool-contract.md` — complete public tool contract.
