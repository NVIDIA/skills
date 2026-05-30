---
name: vss-generate-video-report
description: Use this skill when producing a VSS analysis report — Mode A per-clip VLM, Mode B incident-range via video-analytics. Not for real-time alerts or ad-hoc Q&A.
license: Apache-2.0
metadata:
  version: "3.2.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

# Report

Generate a video analysis report by routing to one of two backends — **never via** `POST /generate` on the VSS agent.

| Mode | Trigger | Backend |
|---|---|---|
| **A. Video clip** | "report on `<sensor>`", "report on this video", "analyze warehouse_01.mp4" | `/vss-manage-video-io-storage` → clip URL → **VLM chat/completions** |
| **B. Incident range** | "report on incidents from `<t1>` to `<t2>`", "report on alerts today", "what incidents happened on `<sensor>` last hour" | `/vss-query-analytics` → incident list → narrative report |

If the request is ambiguous (e.g. "report on `<sensor>`" with no time range and no incident wording), default to **Mode A**. Ask only if the user mentions both a sensor and a time range.

---

## When to Use

- "Generate a report for this video" / "for `<sensor-id>`" — **Mode A**
- "Create an analysis report on the uploaded video" — **Mode A**
- "Report on incidents from 12:31Z to 12:32Z" — **Mode B**
- "Summarize alerts on `<sensor>` between `<t1>` and `<t2>`" — **Mode B**

---

## Deployment prerequisite

**Mode A** needs the VSS **base** profile (VST + VLM NIM).
**Mode B** needs the VSS **alerts** profile (VA-MCP + Elasticsearch).

Probe:

```bash
# Mode A — VST + VLM reachability
curl -sf --max-time 5 "http://${HOST_IP}:30888/vst/api/v1/sensor/version" >/dev/null

# Mode B — VA-MCP
curl -sf --max-time 5 "http://${HOST_IP}:9901/" >/dev/null
```

If the probe fails, hand off to `/vss-deploy-profile` with `-p base` (Mode A) or `-p alerts` (Mode B). **Always** confirm the deploy with the user first.

---

## Browser-playable clip URL (always do this before embedding any clip in the report)

VST returns clip URLs using the agent-internal `${HOST_IP}:30888` host:port. Those work in-cluster (VLM frame pulls, agent backend) but the user's browser cannot reach them. The deploy layer already exports the browser-facing host:port as `$VSS_PUBLIC_HOST` / `$VSS_PUBLIC_PORT` (and scheme as `$VSS_PUBLIC_HTTP_PROTOCOL`) in every profile `.env` — Brev or bare-metal — so the rewrite is a one-liner:

```bash
BROWSER_CLIP_URL=$(echo "$RAW_URL" | sed -E "s|^https?://[^/]+|${VSS_PUBLIC_HTTP_PROTOCOL}://${VSS_PUBLIC_HOST}:${VSS_PUBLIC_PORT}|")
```

Apply it to **every clip URL surfaced in the rendered report** (Mode A Step 4 Clip URL row; Mode B per-incident clip sub-bullet). Leave the VLM `video_url` content block in Mode A Step 3 on the original internal URL — the VLM is in-cluster.

---

## Mode A — Report on a recorded video clip

**If the VSS `lvs` profile is deployed** — `curl -sf --max-time 5 "http://${HOST_IP}:38111/v1/ready"` returns HTTP 200 — run `/vss-summarize-video` to produce the summary, then paste its output into the report template in Step 4 and skip Steps 1–3 (the VLM-direct path). Run Steps 1–3 only when `/v1/ready` is non-200.

### Step 1 — Resolve the clip URL

Hand off to `/vss-manage-video-io-storage` to:

1. List sensors and confirm the named `<sensor-id>` exists (upload first if not).
2. Fetch `/storage/<streamId>/timelines` for the recorded range when the user did not supply `startTime` / `endTime`.
3. Request a clip URL:

   ```bash
   curl -s "http://${HOST_IP}:30888/vst/api/v1/storage/file/<streamId>/url?startTime=<startTime>&endTime=<endTime>&container=mp4&disableAudio=true" | jq -r .videoUrl
   ```

   That gives a direct `mp4` URL that the VLM can pull frames from. Bind it to `VIDEO_URL` (used in-cluster by the VLM in Step 3) **and** rewrite to `BROWSER_CLIP_URL` for the Step 4 report template using the one-liner from *Browser-playable clip URL* above — the user's browser cannot reach `$VIDEO_URL` directly.
   Mode A requires the selected VLM endpoint to be able to fetch `VIDEO_URL`.
   Local NIM/RT-VLM deployments normally can; remote endpoints generally cannot
   fetch `localhost`, private `HOST_IP`, or VST-internal URLs. If the live
   `VLM_ENDPOINT` is remote, surface that reachability requirement instead of
   making a chat request that will fail after `/v1/models` succeeds.

### Step 2 — Resolve VLM endpoint and model

The deploy may serve the VLM through either of two stacks. Both expose an OpenAI-compatible `chat/completions` API — pick whichever is live:

| Backend | Env vars | Typical host endpoint | Picked when |
|---|---|---|---|
| **NIM Cosmos** | `VLM_BASE_URL`, `VLM_NAME` | `${VLM_BASE_URL}/v1` (no trailing `/v1` on the env var; the agent appends it) | `VLM_MODE` ∈ {`local`, `local_shared`, `remote`} **and** `VLM_BASE_URL` is non-empty |
| **RT-VLM Cosmos** | `RTVI_VLM_BASE_URL`, `RTVI_VLM_MODEL_TO_USE` (model identifier on the RT-VLM side, e.g. `cosmos-reason2`) | `${RTVI_VLM_BASE_URL}/v1` — alerts default `http://${HOST_IP}:8018/v1`, base default `http://${HOST_IP}:30082/v1` (`RTVI_VLM_ENDPOINT`) | `VLM_MODE=none` **or** `VLM_BASE_URL` empty; also the only path for `warehouse` |

Read the live values off the running agent container — do not guess:

```bash
docker exec vss-agent env | grep -E '^(VLM_BASE_URL|VLM_NAME|VLM_MODE|RTVI_VLM_BASE_URL|RTVI_VLM_ENDPOINT|RTVI_VLM_MODEL_TO_USE)='
```

Selection rule:

```bash
if [ -n "${VLM_BASE_URL}" ] && [ "${VLM_MODE}" != "none" ]; then
  VLM_ENDPOINT="${VLM_BASE_URL%/}/v1"
  VLM_MODEL="${VLM_NAME}"
else
  VLM_ENDPOINT="${RTVI_VLM_ENDPOINT:-${RTVI_VLM_BASE_URL%/}/v1}"
  VLM_MODEL="${RTVI_VLM_MODEL_TO_USE}"
fi
```

Probe `/v1/models` before sending a chat request to confirm the chosen endpoint is alive and the model is loaded:

```bash
curl -sf --max-time 5 "${VLM_ENDPOINT}/models" | jq -r '.data[].id'
```

If the probe fails or the listed ids don't include `${VLM_MODEL}`, fall back to the other backend (or surface the error — never silently pick a model that isn't on the server).

### Step 3 — Call the VLM directly

Use the OpenAI-compatible `chat/completions` endpoint with a `video_url` content block — the same payload shape `video_understanding` builds in `src/vss_agents/tools/video_understanding.py` (`_build_vlm_messages`):

```bash
PROMPT='Describe in detail what happens in the video, with timestamps (start–end in seconds from clip start) for each segment or event. Cover scenes, objects, people, vehicles, and notable actions.'

# Cosmos Reason 2 reasoning prompt suffix — matches video_understanding.py for is_cosmos_reason2 + reasoning=true.
# Drop this suffix for non-cosmos-reason2 VLMs.
PROMPT="${PROMPT}

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."

curl -s -X POST "${VLM_ENDPOINT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<EOF | jq -r '.choices[0].message.content'
{
  "model": "${VLM_MODEL}",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": $(jq -Rs . <<< "${PROMPT}")},
        {"type": "video_url", "video_url": {"url": "${VIDEO_URL}"}}
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.0
}
EOF
```

If the VLM returns a `<think>…</think>` block (Cosmos Reason reasoning mode), keep only the text after `</think>` as the report body.

### Step 4 — Fill the Video Analysis Report template

```markdown
# Video Analysis Report

## Basic Information

| Field | Value |
|-------|-------|
| **Report Identifier** | vss_report_<YYYYMMDD_HHMMSS> |
| **Date of Analysis** | <YYYY-MM-DD> |
| **Time of Analysis** | <HH:MM:SS> |
| **Video Source** | <sensor_id or filename> |
| **Clip Range** | <startTime> – <endTime> |
| **Clip URL** | `<BROWSER_CLIP_URL>` (apply the `$VSS_PUBLIC_HOST:$VSS_PUBLIC_PORT` rewrite — NEVER paste the raw `HOST_IP:30888` URL here) |
| **VLM** | <VLM_MODEL (NIM or RT-VLM)> |
| **Analysis Request** | <user's request> |

## Analysis Results

<VLM output: timestamped caption / summary>
```

Return the rendered markdown to the user.

---

## Mode B — Report on incidents in a time range

### Step 1 — Resolve the time range and (optionally) sensor

- `start_time` / `end_time` must be ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SS.sssZ`). Resolve relative phrases ("last hour", "today") against the current host clock.
- If the user names a sensor, capture it as `source` + `source_type=sensor`. Otherwise leave both unset for an all-sensors query.

### Step 2 — Fetch incidents via `/vss-query-analytics`

Hand off to `/vss-query-analytics` (initialize → `tools/call`) with:

```json
{
  "name": "video_analytics__get_incidents",
  "arguments": {
    "source": "<sensor-id-or-omit>",
    "source_type": "sensor",
    "start_time": "<ISO>",
    "end_time": "<ISO>",
    "max_count": 100,
    "includes": ["objectIds", "info"]
  }
}
```

For each incident keep: `id`, `sensorId`, `timestamp`, `end`, `category`, `place.name`, `info.verdict`, `info.reasoning`, `objectIds`, and the clip URL (commonly `info.clip_url`, `clip_url`, or whichever clip-pointer field the response carries). **Apply the `$VSS_PUBLIC_HOST:$VSS_PUBLIC_PORT` rewrite (see *Browser-playable clip URL* above) to every clip URL before pasting it into the report** — the raw value is a `HOST_IP:30888` URL the user's browser cannot reach.

### Step 3 — Fill the Incident Range Report template

Group by sensor (or by category if no sensor scope), tally verdicts, list each incident as a bullet with timestamp / category / verdict / reasoning.

```markdown
# Incident Range Report

## Basic Information

| Field | Value |
|-------|-------|
| **Report Identifier** | vss_report_<YYYYMMDD_HHMMSS> |
| **Range** | <start_time> – <end_time> |
| **Scope** | <sensor_id> | all sensors |
| **Total Incidents** | <N> |
| **Confirmed / Rejected / Unverified** | <c> / <r> / <u> |

## Incidents

### <sensor_id_or_category>

- **<timestamp>** — <category> — verdict: **<confirmed|rejected|unverified>**
  - <info.reasoning (1–2 lines)>
  - clip: `<rewritten URL>` (omit row when the incident carries no clip URL — never paste a raw `HOST_IP:30888` URL)
  - objects: <objectIds joined>
- …

## Summary

<2–4 sentences synthesizing what dominates the range — top categories, sensors with the most confirmed incidents, any clusters in time.>
```

If `get_incidents` returns zero results, return a one-line report stating the range and scope produced no incidents — do not invent content and do not fall back to Mode A.

---

## Cross-Reference

- **`/vss-manage-video-io-storage`** — sensor list, timelines, and clip URL for Mode A Step 1.
- **`/vss-query-analytics`** — incident retrieval (and verdict / reasoning enrichment) for Mode B Step 2.
- **`/vss-ask-video`** — ad-hoc VLM Q&A on a single clip (not a structured report).
- **`/vss-summarize-video`** — used by Mode A to produce the summary body when the `lvs` profile is deployed; the report template (Step 4) is still filled here.

