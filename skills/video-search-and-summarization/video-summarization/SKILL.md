---
name: video-summarization
description: Summarize a video by calling the VLM NIM or the Long Video Summarization (LVS) microservice directly. For short videos (under 60s) call the VLM's OpenAI-compatible chat completions endpoint; for long videos (60s or longer) call the LVS microservice. Use when asked to summarize a video, describe what happens in a video, analyze a recording, call or debug LVS summarize/model/health/recommended-config/metrics endpoints, or configure and troubleshoot the LVS service that backs long-video summarization.
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

You are a video summarization assistant. You call the VLM NIM or the LVS
microservice **directly**. Always run `curl` commands yourself; never instruct the user to run them.

Primary video workflow query type: **"Summarize this video."** Direct LVS API
and service-ops requests are handled by the reference-routed sections below.

## Reference Map

Use these references only when the user asks for the relevant detail, or when
the core workflow below needs deeper LVS information:

- **LVS API details**: [`references/lvs-api.md`](references/lvs-api.md) for
  `/summarize`, `/v1/summarize`, health probes, `/models`,
  `/recommended_config`, `/metrics`, request fields, response shapes, and API
  gotchas.
- **LVS service configuration and ops**:
  [`references/deploy-lvs-service.md`](references/deploy-lvs-service.md) for
  the LVS service compose profile, ports, required env vars, logs, status,
  dry-runs, teardown, model/backend swaps, and service-level troubleshooting.
- **Extended LVS ops references**:
  [`references/lvs-environment-variables.md`](references/lvs-environment-variables.md),
  [`references/lvs-debugging.md`](references/lvs-debugging.md), and
  [`references/lvs.env.example`](references/lvs.env.example).

Do not load these references for routine short-video VLM summaries. Load
`lvs-api.md` for long-video LVS request details or direct LVS API requests.
Load `deploy-lvs-service.md` only for deployment, configuration, or service
operations.

## LVS API And Service Ops Requests

If the user asks to call or debug LVS endpoints directly, answer from
[`references/lvs-api.md`](references/lvs-api.md) instead of running the
end-to-end video summarization workflow. Examples: list LVS models, check
readiness, get recommended chunking config, inspect metrics, explain a 422
response, or build a `/summarize` request body.

If the user asks to configure, deploy, restart, tear down, or troubleshoot the
LVS service, prefer the `deploy` skill for full VSS profile deployment and use
[`references/deploy-lvs-service.md`](references/deploy-lvs-service.md) for
LVS-specific service details.

## Routing

Decide purely from video duration (fetch the timeline via the `vios`
skill, then do the math — see Step 1):

| Video duration | Backend | Endpoint |
|---|---|---|
| `< 60s` (short) | **VLM NIM** (OpenAI-compatible) | `POST ${VLM_BASE_URL}/v1/chat/completions` |
| `>= 60s` (long), LVS available | **LVS microservice** | `POST ${LVS_BACKEND_URL}/summarize` |
| `>= 60s`, LVS **not** reachable | **VLM NIM** + tell the user | `POST ${VLM_BASE_URL}/v1/chat/completions` |

Fallback message when LVS is unreachable for a long video (copy verbatim
into the response, before the summary):

> ⚠️ **Note:** Input video `<name>` is `<N>`s long.
> Long Video Summarization (LVS) is not deployed, so this summary was
> produced by the VLM alone. Deploy the `lvs` profile for higher-quality
> long-video summaries.

## Deployment Prerequisite For Summarization

The video summarization workflow requires the VSS **lvs** profile running on
the host at `$HOST_IP`. Before any summarization request:

1. Probe the LVS microservice:
   ```bash
   curl -sf --max-time 5 "http://${HOST_IP}:8000/docs" >/dev/null \
     && curl -sf --max-time 5 "http://${HOST_IP}:38111/v1/ready" >/dev/null
   ```
   (Port 38111 is LVS. HTTP 200 → ready; 503 → still warming, retry in a moment.)

2. **If the probe fails**, ask the user:
   > *"The VSS `lvs` profile isn't running on `$HOST_IP`. Shall I deploy it now using the `/deploy` skill with `-p lvs`?"*

   - If yes → hand off to the `/deploy` skill. Return here once it succeeds.
   - If no → stop. Long-video summarization without LVS falls back to VLM-only, which is a different (lower-quality) path — confirm with the user before substituting.

   (If your caller has granted explicit pre-authorization to deploy
   autonomously — e.g. the request says "pre-authorized to deploy
   prerequisites", or you are running in a non-interactive evaluation
   harness with that permission — skip the confirmation and invoke
   `/deploy` directly.)

3. If the probe passes, proceed.

For LVS-specific service status, compose profile, ports, logs, or environment
debugging, read [`references/deploy-lvs-service.md`](references/deploy-lvs-service.md).
The `deploy` skill remains canonical for full VSS profile deployment.

---

## Setup

**Endpoints (defaults for a local VSS deployment):**

- VLM NIM: `${VLM_BASE_URL}` — default `http://localhost:30082`
- LVS MS: `${LVS_BACKEND_URL}` — default `http://localhost:38111`
- VIOS: owned by the `vios` skill; refer there.

**Endpoint resolution order:**

1. If the env vars `VLM_BASE_URL` / `LVS_BACKEND_URL` are set, use them
   (strip a trailing `/v1` from `VLM_BASE_URL` — NIM exposes `/v1/...` and
   this skill appends it).
2. Otherwise use the defaults above.
3. If neither works, ask the user for the endpoints. Do not scan ports or
   read config files to guess them.

**Model name:** read `${VLM_NAME}` (default `nvidia/cosmos-reason2-8b`).
Both VLM and LVS requests use the same model name.

For full LVS endpoint schemas, optional request fields, response envelopes,
and error handling, read [`references/lvs-api.md`](references/lvs-api.md).

**Availability checks** (run both before routing):

**Readiness is determined by the HTTP status code only.** Do not parse
or inspect the response body — LVS's `/v1/ready` can legitimately return
`200` with an empty body. Do not treat empty stdout from `curl` as
"unavailable."

```bash
# VLM: 200 on /v1/models
vlm_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 \
  "${VLM_BASE_URL:-http://localhost:30082}/v1/models")
[ "$vlm_code" = "200" ] && echo "VLM OK" || echo "VLM not reachable (HTTP $vlm_code)"

# LVS: 200 on /v1/ready, with retry on 503 (warmup) for up to ~30s
LVS=${LVS_BACKEND_URL:-http://localhost:38111}
lvs_code=000
for i in $(seq 1 10); do
  lvs_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 "$LVS/v1/ready")
  case "$lvs_code" in
    200) echo "LVS OK"; break ;;
    503) sleep 3 ;;                 # warming up; keep polling
    *)   break ;;                   # any other code = not reachable, stop retrying
  esac
done
[ "$lvs_code" = "200" ] || echo "LVS not reachable (HTTP $lvs_code)"
```

**How to interpret the results:**

- `vlm_code = 200` and `lvs_code = 200` → normal routing (Step 2a for
  `<60s`, Step 2b for `>=60s`).
- `vlm_code != 200` → fail; summarization cannot run without the VLM.
- `vlm_code = 200`, `lvs_code != 200` → LVS is truly unavailable; use
  the VLM fallback path described above for long videos.
- A non-200 LVS code after the retry loop is the ONLY signal that LVS
  is unavailable. Empty stdout, missing JSON fields, or a "weird"
  response body are NOT "unavailable."

---

## Step 1 — Resolve the video to a clip URL (delegate to `vios`)

**Use the `vios` skill for all VIOS interactions** — it owns the
canonical curl recipes, parameter defaults, and delete/upload flows. Do not
fabricate URLs or hand-roll VIOS calls here; they will drift.

From `vios`, you need exactly three things for summarization:

1. **`streamId`** for the video (via `sensor/list` → `sensor/<id>/streams`,
   or directly from an upload response).
2. **Timeline** — `{startTime, endTime}` for the stream, ISO 8601 UTC.
   `endTime - startTime` is the duration that drives the routing decision
   below. Always compute; never assume.
3. **Temporary MP4 clip URL** — the `/storage/file/<streamId>/url` variant
   with `container=mp4`. The VLM and LVS both need an HTTP(S) URL they can
   `GET`; the `/url` variant is preferred over streaming bytes through the
   summarization client. Response field: `.videoUrl`.

Everything else (auth, error handling, upload, `disableAudio`, expiry, etc.)
is covered in the `vios` skill — refer users there if the VIOS step
fails.

---

## Step 2a — Short video (< 60s) → VLM direct

### HITL: confirm the VLM prompt first (REQUIRED — do not skip)

Full prompt-confirmation walk-through (questions to ask the user, examples, refusal handling) lives in [`references/hitl-prompts.md`](references/hitl-prompts.md). Always run this step before calling the VLM.
### Call the VLM

Once the user confirms a prompt, send it as the `text` part of the VLM
message. OpenAI-compatible chat completions with the video URL embedded in
the message content:

```bash
PROMPT='<confirmed_prompt_from_hitl>'

curl -s -X POST "${VLM_BASE_URL:-http://localhost:30082}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
        --arg model "${VLM_NAME:-nvidia/cosmos-reason2-8b}" \
        --arg text "$PROMPT" \
        --arg url "<clip_url_from_vios>" \
        '{
          model: $model,
          temperature: 0.0,
          max_tokens: 1024,
          messages: [{
            role: "user",
            content: [
              {type: "text", text: $text},
              {type: "video_url", video_url: {url: $url}}
            ]
          }]
        }')" | jq -r '.choices[0].message.content'
```

**Response:** standard OpenAI chat-completion envelope. The summary is in
`choices[0].message.content`.

**Cosmos-model notes:** Cosmos Reason 2 supports reasoning via
`<think>...</think><answer>...</answer>` blocks. Omit the reasoning
instructions if you want a plain summary. Frame sampling and pixel limits
are applied server-side; no client-side prep is required when you pass a
`video_url`.

---

## Step 2b — Long video (>= 60s) → LVS microservice direct

This section contains the narrow long-video summarization path. For advanced
LVS fields such as `media_info`, `schema`, structured output, chunk overlap,
live stream timestamps, metrics, or recommended config, read
[`references/lvs-api.md`](references/lvs-api.md).

### HITL: collect scenario and events first (REQUIRED — do not skip)

Full scenario/events collection walk-through lives in [`references/hitl-prompts.md`](references/hitl-prompts.md). Always run this step before calling LVS.
# Extract the summary and events in one pipe:
curl -s -X POST "${LVS_BACKEND_URL:-http://localhost:38111}/summarize" \
  -H "Content-Type: application/json" \
  -d @request.json \
  | jq -r '.choices[0].message.content' \
  | jq '{video_summary, events}'
```

If both `video_summary` and `events` come back empty, the clip probably
doesn't contain the requested events — re-run with different `events` or a
broader `scenario` rather than reporting "no content."

**Tuning:**

- `chunk_duration` (default `10`) — seconds per chunk. Smaller = finer
  timestamps, more VLM calls. Use `0` to send the whole video in one chunk.
- `num_frames_per_chunk` (default `20`) — frames sampled per chunk.
- `seed` (default `1`) — reproducibility; change or omit to get variety.

---

## End-to-end examples

Assume the `vios` skill has already given you `$CLIP` (clip URL) and
`$DURATION` (seconds) for the target video — those two values are the
contract from Step 1.

### Short video (`$DURATION < 60`)

**HITL (required, before the curl):** post the Step 2a message, wait for
`Submit` (or a `/generate` / `/refine` round-trip that ends in `Submit`),
then set `PROMPT` to the confirmed text. Do not run the curl below until
that confirmation has arrived.

```bash
PROMPT='Describe in detail what is happening in this video,
including all visible people, vehicles, equipments, objects,
actions, and environmental conditions.
OUTPUT REQUIREMENTS:
[timestamp-timestamp] Description of what is happening.
EXAMPLE:
[0.0s-4.0s] <description of the first event>
[4.0s-12.0s] <description of the second event>'

curl -s -X POST "${VLM_BASE_URL:-http://localhost:30082}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg url "$CLIP" --arg text "$PROMPT" \
        --arg model "${VLM_NAME:-nvidia/cosmos-reason2-8b}" '{
    model: $model,
    temperature: 0.0,
    max_tokens: 1024,
    messages: [{role:"user", content:[
      {type:"text", text:$text},
      {type:"video_url", video_url:{url:$url}}
    ]}]
  }')" | jq -r '.choices[0].message.content'
```

### Long video (`$DURATION >= 60`)

**HITL (required, before the curl):** post the Step 2b message and wait
for the user's reply. Substitute their values (or the `defaults` opt-in)
into `$SCENARIO`, `$EVENTS_JSON`, and `$OBJECTS_JSON` below. Do not run
the curl without that reply.

```bash
LVS=${LVS_BACKEND_URL:-http://localhost:38111}

# From HITL reply:
SCENARIO='warehouse monitoring'            # or whatever the user gave
EVENTS_JSON='["notable activity"]'         # jq-compatible JSON array
OBJECTS_JSON=''                            # '' to omit, else '["cars","trucks"]'

# Readiness = HTTP 200 on /v1/ready. Body may be empty — do not inspect it.
# Retry on 503 (warmup) for up to ~30s before concluding LVS is unavailable.
lvs_code=000
for i in $(seq 1 10); do
  lvs_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 "$LVS/v1/ready")
  case "$lvs_code" in 200) break ;; 503) sleep 3 ;; *) break ;; esac
done

if [ "$lvs_code" = "200" ]; then
  curl -s -X POST "$LVS/summarize" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg url "$CLIP" \
          --arg model "${VLM_NAME:-nvidia/cosmos-reason2-8b}" \
          --arg scenario "$SCENARIO" \
          --argjson events "$EVENTS_JSON" \
          --argjson objects "${OBJECTS_JSON:-null}" '{
      url: $url,
      model: $model,
      scenario: $scenario,
      events: $events,
      chunk_duration: 10,
      num_frames_per_chunk: 20,
      seed: 1
    } + (if $objects == null then {} else {objects_of_interest: $objects} end)')" \
    | jq -r '.choices[0].message.content' | jq '{video_summary, events}'
else
  echo "⚠️ Note: video is ${DURATION}s long. LVS returned HTTP $lvs_code; falling back to VLM."
  # Fall back to the short-video VLM flow above (which itself requires
  # the Step 2a HITL confirmation before calling the VLM).
fi
```

---

## Responses

- **VLM** returns an OpenAI chat-completion envelope; the summary string is
  `choices[0].message.content`.
- **LVS** returns the same envelope but `content` is a JSON string — run
  `jq -r '.choices[0].message.content' | jq` to reach `{video_summary, events}`.
- **Errors** from VLM/LVS surface as HTTP non-2xx plus JSON `{error: ...}`.
  `503` from LVS typically means it is still warming up — wait and retry
  `v1/ready`.

### Presenting the output to the user (IMPORTANT — do not rewrite)

The VLM and LVS responses are the final user-facing product. Surface
them with minimal transformation; do not paraphrase, re-voice, add
emojis, or re-format into bullets/tables that weren't in the source.

**Exactly one backend call, exactly one rendering.** A single confirmed
prompt (Step 2a) or a single confirmed scenario/events set (Step 2b)
corresponds to exactly one `POST /v1/chat/completions` or `POST
/summarize` request, and exactly one block of output to the user. Do
NOT fan out parallel calls to hedge (e.g., one call for "full scene"
plus another for "anomalies"), and do NOT render the same response
twice with different headers. If the user wants a second pass (e.g.,
"now with a safety-incident focus"), that's a new HITL round → a new
single call → a new single rendering.

**Header line format.** Start the response with exactly one header:

```
Summary of <video_name> (<duration>)
```

Use `<duration>` formatted as `Ns` for durations under 60 seconds (e.g.
`25s`) and `Mm Ss` for durations ≥60 seconds (e.g. `3m 30s`). Never
include the same header twice in different formats.

**LVS output:**

- **`video_summary`** (string) — render **verbatim** as the narrative
  summary. It is already a polished, tone-controlled "Observational
  Report"; the agent rewriting it loses fidelity (e.g., the model's
  neutral/formal voice becomes the agent's default voice, subtle
  phrasing gets smoothed out).
- **`events`** (list) — render each event with its `start_time`,
  `end_time`, `type`, and the full `description` verbatim. Pick a
  format that renders cleanly in the current client; you may use a
  table if the client renders them legibly, otherwise fall back to a
  per-event list. Do not shorten or paraphrase `description`.
- You MAY add a one-line header identifying the video (e.g.
  `**Summary of <name>** (<duration>, scenario: <scenario>)`) and a
  closing offer to re-run with different parameters. You MAY NOT
  summarize, reorder, or interpret the content itself.

**VLM output:** `choices[0].message.content` is already the full
assistant reply — render it verbatim. If the model produced
`<think>...</think><answer>...</answer>` blocks, strip the `<think>`
block and show the `<answer>` content (or the whole content if the
tags are absent).

**Fallback warning**, when applicable, goes **above** the LVS/VLM
output, not mixed into it.

## Tips

- **HITL is not optional.** Every summarization starts with the HITL
  message (Step 2a or 2b). Skipping it to "be efficient" is the single
  most common failure mode of this skill — do not do it.
- **LVS readiness = HTTP 200 on `/v1/ready`. Nothing else.** The body is
  often empty (`size=0`). Do NOT pipe the readiness check through
  `head`, `jq`, `grep`, or any other command — bash will report the
  pipeline's last exit code, not curl's, and an empty body will look
  identical to a real failure. Use the `curl -s -o /dev/null -w
  '%{http_code}'` pattern from *Setup → Availability checks* verbatim.
- **Delegate VIOS to `vios`.** Do not hand-roll clip-URL, timeline, or
  upload calls here — they'll drift from the canonical recipes.
- **Duration is authoritative.** Don't route on filename or user hints;
  compute from the timeline returned by `vios`.
- **`jq` twice for LVS.** First unwraps the OpenAI-style envelope, second
  parses the JSON string inside `content`.
- **Do not rewrite LVS / VLM output.** The `video_summary` from LVS and
  `choices[0].message.content` from VLM are the deliverables. Render
  them verbatim; don't paraphrase into your own voice or reformat. See
  *Responses → Presenting the output to the user*.
- **One call, one render.** One confirmed HITL → one backend request →
  one block of output. No parallel hedging, no duplicate renderings
  with different headers.

## Cross-reference

- **deploy** — bring up the `base` (VLM only) or `lvs` (VLM + LVS MS) profile
- **vios** (VIOS API) — upload videos, list streams, get clip URLs
- **video-search** — semantic search across the archive (different profile)
- **video-analytics** — query incidents/events from Elasticsearch
- **LVS API reference** — [`references/lvs-api.md`](references/lvs-api.md)
- **LVS service ops reference** — [`references/deploy-lvs-service.md`](references/deploy-lvs-service.md)
