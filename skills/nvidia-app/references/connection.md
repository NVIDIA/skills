# NVIDIA App MCP Connection

## Transport overview

NVIDIA App exposes one persistent MCP server from its NvContainer-hosted plugin and supports two client transports:

- Streamable HTTP over loopback.
- A stdio bridge executable that relays MCP messages to the server's per-user named pipe.

The stdio executable is a relay, not a standalone server. If NVIDIA App's MCP master toggle is disabled, launching the bridge does not enable or start the persistent server.

## Agent-side configuration

When a connection is needed and `nvidia-app` is not already registered with the current MCP client, first try to add it to that client using an available MCP-management capability or the client's documented configuration mechanism. Do not stop at showing configuration JSON when the agent can apply it directly.

- Inspect the existing client configuration first; reuse a valid `nvidia-app` entry and do not create a duplicate.
- Prefer the Streamable HTTP connection described below. Use the stdio bridge only when HTTP is unavailable or the current client requires or manages a local relay process.
- After adding or reloading the entry, call `tools/list` to verify the connection and tool discovery.
- If the agent cannot modify or reload its own MCP configuration, provide the exact configuration and state the specific action the user must perform. Do not guess an unverified configuration location or format.

### Direct-connection fallback

Connect directly to the endpoints below, without a client entry, when any of these holds:

- The user declines or defers the client configuration change.
- The client offers no way for the agent to add or reload an MCP entry.
- The entry was written, but `nvapp_` tools are still absent from the current session and the user does not want to restart or reload.

In that case, complete the user's request over Streamable HTTP (or the stdio bridge) using the same documented tools, schemas, and restrictions. Name the fallback in your reply and note that registering `nvidia-app` avoids repeating it.

- If connection or tool discovery still fails after the applicable configuration and reload attempts, instruct the user to verify that NVIDIA App is installed, that it is the latest available version (11.0.9.5xx or above), and that MCP server access is enabled in NVIDIA App Settings before retrying.

## HTTP connection

The compiled defaults are:

```text
Host:     127.0.0.1
Port:     13508
Endpoint: http://127.0.0.1:13508/mcp
```

Typical client configuration:

```json
{
  "mcpServers": {
    "nvidia-app": {
      "url": "http://127.0.0.1:13508/mcp"
    }
  }
}
```

The server registers `/mcp/` with http.sys and reports its client URL without the trailing slash.

### Port fallback

If the server fails to register port `13508`, it falls back in two tiers. It first tries the adjacent range next to the default, then the ephemeral range:

```text
Tier 1 (adjacent):  13509, 13510, 13511, 13512, 13513,
                    13514, 13515, 13516, 13517, 13518
Tier 2 (ephemeral): 49152, 49153, 49154, 49155, 49156,
                    49157, 49158, 49159, 49160, 49161
```

If other methods do not reveal the MCP port, use this discovery file as the fallback:

```text
%LOCALAPPDATA%\NVIDIA Corporation\NvAppMcpServer\server.json
```

Example shape:

```json
{
  "pipe": "<per-user named-pipe name>",
  "http": "http://127.0.0.1:13508/mcp/",
  "version": "1.0.0",
  "pid": 12345
}
```

Read `server.json` fresh and use its `http` value, which contains the active endpoint and fallback port. The `http` property is omitted when HTTP failed to start; in that case, use the stdio bridge.

If `server.json` cannot be found or read, follow the connection and tool discovery failure guidance above.

If the server cannot register any HTTP port, NVIDIA App continues in pipe-only degraded mode when the named-pipe server is available.

## Stdio connection

Installed bridge path:

```text
%ProgramFiles%\NVIDIA Corporation\NVIDIA App\McpServer\NvAppMcpServer.exe
```

Typical configuration:

```json
{
  "mcpServers": {
    "nvidia-app": {
      "command": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\McpServer\\NvAppMcpServer.exe",
      "args": []
    }
  }
}
```

The bridge resolves the current user's named-pipe name, connects with retry, and relays binary-safe stdio. If the pipe is unavailable and the MCP readiness event is not signaled, it reports `service_disabled` with guidance to enable MCP server access in NVIDIA App Settings.

## MCP protocol lifecycle

The server accepts two protocol eras:

- Modern `2026-07-28`: stateless HTTP requests; it does not emit `Mcp-Session-Id` or use standalone session-stream or session-close requests.
- Legacy `2025-11-25` and earlier supported versions (`2024-11-05`, `2025-03-26`): initialize creates a session, the response carries `Mcp-Session-Id`, subsequent requests echo it, and the client manages the notification stream and session closure.

Ordinary MCP clients should manage this lifecycle. Do not manually add legacy session headers to a modern connection.

## Readiness sequence

Connection success and Overlay success are separate:

1. NVIDIA App's MCP master toggle allows the persistent NvContainer plugin to run.
2. The named-pipe and, when available, HTTP transports start.
3. The MCP server readiness event is signaled.
4. NVIDIA In-Game Overlay independently signals its own readiness event.

The Overlay tools in this skill can return `overlay_not_available` even after the MCP connection succeeds. Non-Overlay NVIDIA App tools do not depend on Overlay readiness.

## Troubleshooting

| Symptom | Likely cause | Response |
|---|---|---|
| Neither transport connects | MCP master toggle is off or the server did not start | Ask the user to enable MCP server access in NVIDIA App Settings. |
| Port `13508` refuses connection | HTTP used a fallback port, HTTP startup failed, or the server is disabled | Use the discovery/runtime URL when available or use stdio. |
| Discovery file lacks `http` | HTTP failed and the server is in pipe-only mode | Use the stdio bridge. |
| Stdio returns `service_disabled` | Persistent server readiness event is not signaled | Enable MCP server access; do not repeatedly spawn bridge processes. |
| MCP connects but Overlay calls return `overlay_not_available` | In-Game Overlay is disabled or not running | Enable NVIDIA App Settings > Features > In-Game Overlay, then retry. |
