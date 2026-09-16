# Connection & protocol

## Registering the server (do this first)

Streamable HTTP is the default and recommended transport. Register the loopback gateway URL with
your MCP client:

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

`18100` is the default and correct on almost every install. You need no credential: the gateway
binds to loopback only and grants extended access to any local client. The URL can change when
that port is occupied; see the next section.

### stdio compatibility fallback

Use stdio only when the MCP client does not support Streamable HTTP. In NVIDIA Broadcast, open
**Settings > AI Agent Resources** and copy the stdio command so a custom installation location is
handled correctly. Configure that exact copied absolute path as the client's `command` value. A
default installation places the executable at
`%ProgramFiles%\NVIDIA Corporation\NVIDIA Broadcast\NvBroadcastMCP.exe`, but do not hardcode a
drive letter or rely on environment-variable expansion in MCP configuration.
Do not use the bare executable name: Windows may select a different executable from the client's
working directory or `PATH`. NVIDIA Broadcast must already be running; the bridge does not launch
it and rereads the current gateway port for each request.

The rest of this document describes the wire protocol. You need it to **build** a client or
to debug one, not to use the gateway from a client that already exists.

## Discovering the endpoint (when the default fails)

The server takes the first free port in `18100`-`18109`, so it can differ when `18100` is
occupied, and a restart can move it — the one case where a registered URL goes stale. The
stdio bridge rereads the port for every message, so it is unaffected.

1. Read `%APPDATA%/nvidia-broadcast/gateway.json`. Its contents are **base64**; decode it and
   read the `port` field. The record describes the last gateway process, not a promise that
   anything is listening now.
2. The MCP endpoint is `http://127.0.0.1:<port>/gateway` (loopback only; the server binds
   to `127.0.0.1` and rejects non-loopback `Host`/`Origin`).
3. Probe the endpoint before starting an MCP session. Reread the config on every retry rather
   than caching the port.

Inside WSL both step 1 and step 2 change — see *Reaching the gateway from WSL* below.

## Confirm you are talking to NVIDIA Broadcast

A successful probe proves *something* answers on that port, not that it is this server. If
Broadcast is not running, another local process may hold the port — and sending it tool
arguments would hand that process file paths and settings.

Check that the server reports its name as `nvidia-broadcast` before acting on anything, and
treat a mismatch as "Broadcast is not available" rather than retrying. Your MCP client
surfaces this as the server info it read during connection.

## Starting Broadcast hidden

Starting the app is a real side effect on the user's machine, and `--launch-hidden` starts it
with no window, so nothing appears on screen to tell them it happened. **Say that you are
starting NVIDIA Broadcast before you do it**, and ask first if the user has not requested
something that clearly needs it. Launch only the verified executable at the standard path below —
never a path read from `gateway.json` or any other file.

Resolve the executable at its one fixed path and nowhere else. Do not search the filesystem or
`PATH` for it, do not read registry uninstall keys, Start-menu shortcuts, or running-process image
paths to locate it, do not look for a Microsoft Store or `WindowsApps` build, and do not search the
web. The path is identical on every install, so a missing file there is conclusive: Broadcast is not
installed - follow `installation.md` rather than looking elsewhere.

If the endpoint is unavailable:

1. Check whether `NVIDIA Broadcast.exe` is already running. If it is, do not launch another
   copy; wait for readiness and then report that agent integration is disabled or unavailable
   if the timeout expires.
2. If it is not running, resolve and verify the installed executable. The standard location is
   `%ProgramFiles%\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe`.
3. Launch NVIDIA Broadcast as a **detached GUI process** with `--launch-hidden`.

   Run the bundled launcher as a single `terminal` call. If security policy blocks it, use the
   fallback in rule 1 under Hard rules. Otherwise, do not write your own launch command or invoke
   `NVIDIA Broadcast.exe` directly with `subprocess.run`, `check_output`, `Start-Process -Wait`, or
   any command that waits for or captures output from the Broadcast process.



   ```bash
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<skill_dir>\scripts\launch_broadcast.ps1"
   ```
   `<skill_dir>` is the absolute path in the `skill_dir` field of the `skill_view` result you
   already have.

   Do not wait for the Broadcast process to exit. After the launcher returns, continue immediately
   to the gateway-readiness checks in step 4.
4. For up to 60 seconds, repeatedly reread and base64-decode `gateway.json`, then probe the
   resulting endpoint. Treat a responsive MCP endpoint—not process creation—as readiness.
5. On timeout, check whether the app is present and whether the build predates MCP before
   reporting anything. This reads the filesystem and does not launch anything:
   ```powershell
   $exe = "$env:ProgramFiles\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe"
   if (Test-Path $exe) { (Get-Item $exe).VersionInfo.ProductVersion } else { "not installed" }
   ```
   **If the executable is not there, NVIDIA Broadcast is not installed.** Tell the user the app
   is not installed on this machine and that the skill requires it. Stop probing and relaunching,
   then **offer to install it** and follow `installation.md` if they accept. If they decline,
   point them to <https://www.nvidia.com/broadcast-app/> and stop.

   **NVIDIA Broadcast 2.2.x and older ship no MCP gateway.** On those builds `gateway.json` is
   never created and nothing ever binds a port, so the timeout is permanent, not transient.
   Tell the user their installed version does not support MCP, and **offer to update it** the
   same way — `installation.md` covers the consent wording, the verification, and what to do when
   the update service has nothing newer to offer. Do not retry, relaunch, or look for an
   alternative transport.

   Neither case licenses anything more than that: nothing is downloaded before the user says yes,
   and no installer runs before it passes the checksum and NVIDIA-signature checks.
6. If the version is newer than 2.2.x, or cannot be read, the installed build is not the problem.
   Separate the two reasons you might be here:

   - **You were never able to probe**, because your client cannot register MCP servers at runtime.
     Nothing is wrong with Broadcast. Say so, and tell the user to add the gateway to their MCP
     client's configuration and restart the client. Do not report this as a Broadcast failure and do
     not fall back to talking them through the app's UI as if the app were at fault.
   - **The probe ran and genuinely failed.** Report that Broadcast could not initialize, or that
     agent integration is disabled.

   Either way, **do not offer to reinstall.** A 2.3+ build that will not start is not an install
   problem, and reinstalling over it is a privileged system change nobody asked for.

This is your job, not the bridge's: `NvBroadcastMCP.exe` never starts Broadcast, because a
client spawns it at startup and launching a camera app from that would surprise the user.

## Reaching the gateway from WSL

An agent inside WSL runs on the machine hosting Broadcast but in a different OS, with its own
filesystem root and — in the default networking mode — its own loopback interface. Nothing about
the gateway changes: it still binds `127.0.0.1` on Windows and still rejects every non-loopback
`Host` and `Origin`. What changes is whether `127.0.0.1` inside WSL *is* that interface, and how
you read the published port.

### 1. Detect WSL

`/proc/sys/kernel/osrelease` contains `microsoft` or `WSL` on a WSL kernel:

```bash
grep -qi 'microsoft\|wsl' /proc/sys/kernel/osrelease && echo wsl
```

Decide from that file, not from environment variables. A Linux machine that is *not* WSL on the
Broadcast host is out of scope for this skill: there is no gateway to reach, and you should say so
rather than looking for one.

### 2. Establish whether Windows loopback is reachable

```bash
wslinfo --networking-mode
```

- **`mirrored`** — WSL shares the Windows network interfaces, including loopback. Register
  `http://127.0.0.1:<port>/gateway` exactly as you would on Windows; the `Host` header is
  `127.0.0.1`, so the guards pass. WSL 1 distros share the Windows network stack for the same
  reason and behave the same way.
- **`nat`** (the default) — WSL has its own loopback, and a Windows service bound to `127.0.0.1`
  is not reachable from it. The gateway is unavailable until the user changes that, and no amount
  of retrying, reprobing, or relaunching Broadcast will help.
- **Command unavailable** (older WSL without `wslinfo`) — run the normal endpoint discovery and
  startup flow first, then try `127.0.0.1:<port>` through your MCP client. Conclude `nat` only once
  the Windows-side listener is confirmed up and WSL still cannot reach it; a single failure can
  equally mean Broadcast is not running yet or the port moved.

**Do not substitute another address for loopback.** The Windows host address from
`/etc/resolv.conf`, `ip route show default`, or `$(hostname).local` reaches the Windows network
stack but not a loopback-bound listener, and even if something answered there the request would be
refused for a non-loopback `Host`. Trying those addresses is not a fallback, it is scanning the
user's network for a service that is not published on it.

### 3. Read `gateway.json` through the mounted Windows drive

`%APPDATA%` is a Windows path and cannot be opened from WSL. Resolve it once, then read the same
file as on Windows:

```bash
APPDATA_WSL="$(wslpath "$(cmd.exe /c 'echo %APPDATA%' 2>/dev/null | tr -d '\r')")"
# then read "$APPDATA_WSL/nvidia-broadcast/gateway.json"
# (the translated path lands under the mounted Windows system drive)
```

The contents are base64 decoding to `{ "port": <number> }`, unchanged. Resolving `%APPDATA%` this
way is path translation for one documented location — it is not the environment inspection the
safety rules forbid, and there is still no credential anywhere to look for. If `wslpath` fails or the
translated path is not readable (drive mounting or interop disabled), stop and tell the user; do not
search the Linux filesystem for a gateway file.

### 4. Launch and version-check through interop

Run the PowerShell commands above unchanged, through Windows interop:

```bash
powershell.exe -NoProfile -Command 'Start-Process -FilePath "$env:ProgramFiles\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe" -ArgumentList "--launch-hidden" -WindowStyle Hidden'
```

The same applies to the `ProductVersion` check. Announce the launch first, exactly as on Windows.
If `powershell.exe` is not on `PATH`, Windows interop is disabled: ask the user to start Broadcast
on the Windows side instead of looking for a Linux-side substitute.

Installing goes through the same interop path — see the WSL notes in `installation.md`. Send the
whole download-verify-install sequence as a **single** `powershell.exe -NoProfile -Command`
invocation, because each call is a fresh process and the variables holding the download URL,
checksum and file path do not survive between them. Without interop there is no install path from
here either; ask the user to install on Windows rather than looking for a Linux-side installer.

### 5. What to tell the user in NAT mode

Report that the gateway is on Windows loopback, which NAT-mode WSL cannot reach, and offer the
choices — do not reconfigure their machine yourself:

1. Switch WSL to mirrored networking (Windows 11 22H2 or newer, WSL 2.0.0+): add to
   `%USERPROFILE%\.wslconfig`
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   then `wsl --shutdown` and reopen the distro.
2. Register the gateway from an MCP client running on Windows rather than inside WSL.
3. Use a tunnel they set up that terminates on **Windows loopback** while the WSL-side client
   still targets `127.0.0.1:<port>` — the `Host` header stays loopback, so the guards pass. Set
   one up only if the user explicitly asks for it.

Never ask the user to bind the gateway to another interface or to relax the loopback guards. That
is not configurable, and it is the control that keeps the unauthenticated surface off the network.

## Access

Connect over loopback and you get the full extended surface — effect control, device
control and file processing. There is no token to obtain and no scope to request.

**Known trade-off: this surface is unauthenticated.** There is no token because the gateway
binds loopback only — but that also means any local process able to reach the port gets exactly
the same full access you do, whether that is another MCP client, a compromised dependency, or
malware. The gateway cannot tell them apart from you, and a port-squatting process would receive
the file paths and device settings you send. Treat the endpoint as trustworthy only as far as the
local machine is trustworthy. Always complete the server-identity check above before sending
anything, and on shared or multi-user machines tell the user about this limitation rather than
treating loopback as private.

Mirrored-networking WSL widens "local" a little: the distro shares the host's loopback, so a
process inside it has the same unauthenticated access as a Windows one. Mention that when the user
is choosing how to connect from WSL.

All extended tools use standard MCP annotations and `openWorldHint:false`. Read-only tools
are `get_broadcast_state`, `describe_effects`, `get_file_processing_jobs`,
`wait_for_processing_update`, and `list_devices`. Effect/device setters are idempotent mutations; file submission and
queue control are non-idempotent mutations.

## File status and cancellation

Use `wait_for_processing_update` for one or many files. It wakes for job-status and
queue-pause changes, not progress ticks. Aborting the wait does not cancel media work; use
`control_file_processing_jobs` when the user asks to cancel it.

## Limits

Rate limits and a max in-flight cap protect the app. If you exceed them you
get a structured tool-error (codes `-33102` rate-limited, `-33103` too many in-flight)
with the limit in `data` — back off and retry. `set_effects` is throttled (avoid rapid
toggling); reads are generously bursted.
