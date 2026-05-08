---
name: video-analytics
description: Query video analytics data and metrics from Elastic search via the VA-MCP server (port 9901). This includes incidents, alerts, sensor data, and metrics. Use for any question about violations, alerts, incidents, object counts, speeds, occupancy, or anything that requires looking up recorded events. This is the primary way to answer a question that requires incidents, alerts and other metrics such as people counts and violations.
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

# Video Analytics (VA-MCP)

Queries incidents, alerts, and metrics stored in Elasticsearch via MCP JSON-RPC at **port 9901**.

> **ALWAYS run the commands below yourself and relay results to the user. Do NOT guess or describe — actually execute and report back.**

---

## Deployment prerequisite

This skill reads from the Elasticsearch/VA-MCP stack brought up by the VSS **alerts** profile (either `verification` or `real-time` mode). Before any query:

1. Probe the VA-MCP endpoint:
   ```bash
   curl -sf --max-time 5 "http://${HOST_IP}:9901/mcp" >/dev/null 2>&1 || \
     curl -sf --max-time 5 "http://${HOST_IP}:9901/" >/dev/null
   ```

2. **If the probe fails**, ask the user:
   > *"The VSS `alerts` profile isn't running on `$HOST_IP` (VA-MCP unreachable). Which mode should I deploy — `verification` (CV) or `real-time` (VLM)?"*

   - Answer → hand off to the `/deploy` skill with `-p alerts -m <mode>`. Return here once it succeeds.
   - If the user declines → stop. No incidents/alerts/metrics to query without the alerts stack up.

   (If your caller has granted explicit pre-authorization to deploy
   autonomously — e.g. the request says "pre-authorized to deploy
   prerequisites", or you are running in a non-interactive evaluation
   harness with that permission — skip the confirmation and invoke
   `/deploy` directly. Default the mode to `verification` unless the
   request specifies otherwise.)

3. If the probe passes, proceed.

---

## REQUIRED: Two-Step Pattern (copy this exactly)

**Every query requires two shell commands run in sequence:**

```bash
# Step 1: initialize — get session ID from response HEADER
SESSION_ID=$(curl -si -X POST http://localhost:9901/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"cli","version":"1.0"}},"id":0}' \
  | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')

# Step 2: call the tool using the session ID in the header
curl -s -X POST http://localhost:9901/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"max_count":10}},"id":1}' \
  | grep '^data:' | sed 's/^data: //' | jq -r '.result.content[0].text'
```

> The session ID comes from the **response header** `mcp-session-id`, not the body.
> Skipping Step 1 always results in `Bad Request: Missing session ID`.

---

## Tool Reference

Replace the `-d` payload in Step 2 with any of the following.

### video_analytics__get_incidents

| Parameter | Type | Description |
|---|---|---|
| `source` | string | Sensor ID or place name (optional) |
| `source_type` | string | `sensor` or `place` |
| `start_time` | string | ISO 8601: `YYYY-MM-DDTHH:MM:SS.sssZ` |
| `end_time` | string | ISO 8601 |
| `max_count` | int | Max results (default: 10) |
| `includes` | list | Extra fields: `objectIds`, `info` |
| `vlm_verdict` | string | `confirmed`, `rejected`, or `unverified` |

```bash
# Recent incidents (all sensors)
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"max_count":10}},"id":1}'

# For a specific sensor
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"source":"<sensor-id>","source_type":"sensor","max_count":20}},"id":1}'

# Confirmed (VLM-verified) only
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"vlm_verdict":"confirmed","max_count":10}},"id":1}'
```

### video_analytics__get_incident

```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incident","arguments":{"id":"<incident-id>","includes":["objectIds","info"]}},"id":1}'
```

### video_analytics__get_sensor_ids

```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_sensor_ids","arguments":{}},"id":1}'
```

### video_analytics__get_places

```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_places","arguments":{}},"id":1}'
```

### video_analytics__get_fov_histogram

```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_fov_histogram","arguments":{"source":"<sensor-id>","source_type":"sensor","start_time":"<ISO>","end_time":"<ISO>","object_type":"Person","bucket_count":10}},"id":1}'
```

### video_analytics__analyze

`analysis_type`: `max_min_incidents`, `average_speed`, `avg_num_people`, `avg_num_vehicles`

```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__analyze","arguments":{"source":"<sensor-id>","source_type":"sensor","start_time":"<ISO>","end_time":"<ISO>","analysis_type":"avg_num_people"}},"id":1}'
```

### vst_sensor_list

```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"vst_sensor_list","arguments":{}},"id":1}'
```
