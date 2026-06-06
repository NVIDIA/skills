---
name: vss-search-archive
description: Use to run top-level VSS fusion search on archived video, or to ingest video files / RTSP streams for search. Not for ad-hoc Q&A or live captioning.
license: Apache-2.0
metadata:
  author: "NVIDIA Video Search and Summarization team"
  version: "3.2.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---
## Purpose

Run the top-level VSS fusion search across archived video and ingest new clips / RTSP streams for search.

## Prerequisites

- Active VSS deployment reachable on `$HOST_IP` (see `vss-deploy-profile` and `references/`).
- NGC credentials in `$NGC_CLI_API_KEY` and `$NVIDIA_API_KEY` for any image pulls.
- `curl`, `jq`, and Docker available on the caller.

## Instructions

Follow the routing tables and step-by-step workflows below. Each section that ends in *workflow*, *quick start*, or *flow* is intended to be executed top-to-bottom. Detailed reference material lives in `references/` and helper scripts live in `scripts/` — call them via `run_script` when the skill points to a script by name.

## Examples

Worked end-to-end examples are kept under `evals/` (each `*.json` manifest contains a runnable scenario) and inline in the per-workflow `curl` blocks below. Run a Tier-3 evaluation with `nv-base validate <this-skill-dir> --agent-eval` to replay them.

## Limitations

- Requires the matching VSS profile / microservice to be deployed and reachable from the caller.
- NGC-hosted models and NIMs may be subject to rate-limits, GPU memory requirements, and license restrictions.
- Concurrency, GPU memory, and storage limits depend on the host hardware and the profile's compose file.

## Troubleshooting

- **Error**: REST call returns connection refused. **Cause**: target microservice not running. **Solution**: probe `/docs` or `/health`; redeploy via `vss-deploy-profile` or the matching `vss-deploy-*` skill.
- **Error**: HTTP 401/403 from NGC pulls. **Cause**: missing/expired `NGC_CLI_API_KEY`. **Solution**: `docker login nvcr.io` and re-export the key before retrying.
- **Error**: container OOM or model fails to load. **Cause**: insufficient GPU memory for the selected profile. **Solution**: switch to a smaller variant or free GPUs via `docker compose down`.

# Video Search Workflows

> **Alpha Feature** — not recommended for production use.

Search video archives by natural language using Cosmos Embed1 embeddings. Requires the search profile — deploy with the `vss-deploy-profile` skill (`-p search`). These videos sources can be ingested files or RTSP streams.

## When to Use

- "Find all instances of forklifts"
- "When did someone enter the restricted area?"
- "Show me people near the loading dock"
- "Search for vehicles between 8am and noon"
- Any natural-language search across video archives
- "Ingest `<file>` for search" / "upload this video for search"
- "Add this RTSP stream for search" / "register `<rtsp_url>` for search"

---

## Deployment prerequisite

This skill requires the VSS **search** profile running on the host at `$HOST_IP`. Before any request:

1. Probe the stack:
   ```bash
   curl -sf --max-time 5 "http://${HOST_IP}:8000/docs" >/dev/null \
     && curl -sf --max-time 5 "http://${HOST_IP}:9200/" >/dev/null
   ```
   (The second check confirms Elasticsearch is up — unique to the search profile.)

2. **If the probe fails**, ask the user:
   > *"The VSS `search` profile isn't running on `$HOST_IP`. Shall I deploy it now using the `/vss-deploy-profile` skill with `-p search`?"*

   - If yes → hand off to the `/vss-deploy-profile` skill. Return here once it succeeds.
   - If no → stop. Do not run this skill against a missing or wrong-profile stack.

   (If your caller has granted explicit pre-authorization to deploy
   autonomously — e.g. the request says "pre-authorized to deploy
   prerequisites", or you are running in a non-interactive evaluation
   harness with that permission — skip the confirmation and invoke
   `/vss-deploy-profile` directly.)

3. If the probe passes, proceed.

---

## Ingestion prerequisite (required before any `/generate`)

For a source to be searchable it must be ingested **through the VSS agent backend**, not through VIOS alone. The agent's ingest routes own the VIOS upload + RTVI-CV register + RTVI-embed pipeline as one transaction; a bare VIOS PUT only stores the bytes and never wires them into Elasticsearch.

Confirm the source exists in VIOS first (Mandatory workflow Step 2). If it is missing, ingest it with one of the recipes below before firing `/generate`. After ingest succeeds, the source appears in `sensor/list` under the name you provided and can be referenced from the natural-language query the agent forwards to its search-tool decomposer — you do NOT need to construct a structured `video_sources` payload yourself.

### File upload — universal three-step flow

```bash
# 1. Ask the agent for the chunked-upload URL
URL=$(curl -s -X POST "http://${HOST_IP}:8000/api/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{"filename": "<filename.mp4>"}' | jq -r .url)

# 2. Chunked POST the file to that VST URL (the UI streams chunks; from a shell,
#    a single multipart POST is fine). The final-chunk response carries sensorId.
SENSOR=$(curl -s -X POST "$URL" \
  -F "file=@/path/to/<filename.mp4>;type=video/mp4" | jq -r .sensorId)

# 3. Tell the agent the upload finished — this fans out to RTVI-CV + RTVI-embed
curl -s -X POST "http://${HOST_IP}:8000/api/v1/videos/${SENSOR}/complete" \
  -H "Content-Type: application/json" \
  -d '{"filename": "<filename.mp4>"}' | jq .
```

Wait for the `/complete` response (it returns `chunks_processed > 0` once embeddings land). Only then is the video searchable.

> The deprecated `PUT /api/v1/videos-for-search/{filename}` route is also wired in for legacy callers (single-shot, agent-driven), but its OpenAPI entry is flagged `deprecated`. Prefer the three-step flow above for new work.

### RTSP stream — single endpoint

```bash
curl -s -X POST "http://${HOST_IP}:8000/api/v1/rtsp-streams/add" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_url": "rtsp://<host>:<port>/<path>",
    "name": "<sensor-name>",
    "username": "",
    "password": "",
    "location": "",
    "tags": ""
  }' | jq .
```

The response shape is `{status, message, error}` — no `sensorId` (the agent keys the stream by the `name` you provided). On any step's failure earlier steps roll back. The `start_embedding_generation` step is fire-and-verify: a 2xx confirms the request was accepted and the embedding pipeline is running in the background, **not** that the stream is searchable yet. Search hits will start appearing only after enough chunks land in Elasticsearch — poll with a low-`top_k` query a few seconds in if you need a readiness signal.

---

## How Search Works

1. **Ingest** — Files come in through the agent's three-step universal flow; RTSP streams through `/api/v1/rtsp-streams/add`. Both routes hand the source to RTVI-CV (attribute detection) and RTVI-Embed (Cosmos Embed1) which generates vector embeddings for video segments.
2. **Index** — Embeddings are stored in Elasticsearch via the Kafka pipeline.
3. **Query** — Natural-language queries are embedded and matched against stored vectors by similarity.
4. **Results** — Timestamped video segments ranked by relevance, with clip playback links.

This search orchestrated by VSS agent can lead to 3 behaviors:
- Attribute-only: when the LLM decomposes the query and finds only appearance attributes with no action (e.g. "person wearing red jacket")
- Embed-only: when the query has no extractable attributes (e.g. "show me forklifts")
- Fusion: when the query has both an action and attributes (e.g., "person in red jacket running"), it runs embed search first, then reranks using attribute search

---

## Mandatory workflow

When using this skill, ALWAYS follow this high-level workflow:
1. **Resolve inputs from user instructions — HARD STOP if `$HOST_IP`
   is not explicitly provided.** See § Input resolution below. Do NOT
   default to `localhost`, `127.0.0.1`, the host the agent itself is
   running on, or any other guess. Do NOT issue a
   `POST http://.../generate` request until the user has supplied an
   endpoint. Respond to the user with a single question asking for
   `HOST_IP` / the VSS agent endpoint and wait.
2. **Resolve the source — HARD STOP before any `/generate` call.**
   If the user query references a specific video / sensor name
   (e.g. "the airport video", "warehouse_cam_3", "sample warehouse"),
   verify it's actually registered in VIOS **before** firing
   `POST .../generate`:

   ```bash
   curl -s "http://${HOST_IP}:30888/vst/api/v1/sensor/list" | jq '.[].name'
   ```

   Then:
   - **If the named source (or a clearly substring-matching name) IS in the list** → proceed to step 3. Forward the user's natural-language query verbatim — the agent's own search tool decomposer (`services/agent/src/vss_agents/tools/search.py`) extracts `video_sources` from the prose given the available sources, so the skill does NOT need to construct a structured `video sources` payload.
   - **If the named source is NOT in the list** → STOP. Do NOT fire `/generate` as a probe. Respond to the user with the registered source names and ask whether they meant one of those, want to ingest the missing source (point them at *Ingestion prerequisite* and run the matching file or RTSP recipe through the **agent backend**, not bare VIOS), or want to abandon the query. Wait for clarification.
   - **If the query names no specific source** ("find forklifts in the ingested videos", "search across all sources") → skip the substring check, but `sensor/list` must still return non-empty (otherwise no sources are ingested → HARD STOP).
3. Run the search(es) via approach chosen
4. Present the results to the user query. Format response as a professional inspection report but name it `Video Search Results`:
   — Use clear section headers
   - Organize findings individually with supporting detail, and close with a summary
   - Use tables where comparisons help. Write like a technical report, not a chat message.
   - If criteria results are non-null, then in addition to a column "Critic result" ("confirmed" | "rejected" | "skipped"), include a column "Criteria" with all the criteria for this search result ({criteria_n}: ✓ | ✗)
5. CRITICAL: Verify the results and explain this to the user concisely.
   If search fails, or returns unexpected results (i.e. videos that do not appear to match user query, zero matches, zero videos returned, error etc.), STOP. Do not proceed without reading [troubleshooting.md](references/troubleshooting.md) to iterate with feedback loops until proper results are found and presented like a professional inspection report.
6. Final verifications:
   - ALWAYS inform user that final and further verifications can be run. Present this as a `Verification Step`
   - ONLY IF user agrees, download screenshots using the `screenshot_url` of the best candidates (highest similarity scores) from the search hits (JSON results) to `/tmp`. Read them and verify if they correspond to the user query

## Input resolution

Infer these inputs only from the conversation or user query (no other files unless provided). If some cannot be inferred, ask the user immediately:
- $HOST_IP: where the VSS agent backend runs

---

## Gotchas

- ALWAYS step into the troubleshooting step of the workflow immediately if anything unexpected happens, read [troubleshooting.md](references/troubleshooting.md)
- Queries work best with **concrete visual descriptions** (objects, actions, locations). Augment user queries if needed to enhance the quality of the questions, expanding potential details
- The skill assumes video sources are **already ingested through the agent backend** (see *Ingestion prerequisite*). It MAY run the agent-backed ingest recipes when the user explicitly asks ("ingest `<file>` for search", "add `<rtsp_url>` for search"); it does NOT search the local filesystem for files the user didn't name, and it does NOT use the bare-VIOS PUT path (no embeddings get generated). Workflow step 2 still makes confirming "this source exists in VIOS" a hard precondition before `/generate`.
- Use `vss-query-analytics` skill to cross-reference search results with incident/alert data

---

## Search via REST API

Default to using this REST API approach, unless user specifies otherwise.

```bash
# Consider only ingested video file sources by default
curl -s -X POST http://${HOST_IP}:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "find all instances of forklifts"}' | jq .
```

### More Examples

```bash
# Search by object
curl -s -X POST http://${HOST_IP}:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "find vehicles in the parking lot"}' | jq .

# Search by action
curl -s -X POST http://${HOST_IP}:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "show me people running"}' | jq .

# Search by time context
curl -s -X POST http://${HOST_IP}:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "what happened at the entrance between 2pm and 3pm?"}' | jq .

# Consider only RTSP sources with `search_source_type` filter i.e. live camera streams
curl -s -X POST http://${HOST_IP}:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "find all instances of forklifts", "search_source_type": "rtsp"}' | jq .
```

### Advanced control knobs

If user query is ambiguous, user wants more guidance or when fine-grained control is needed, augment the user `input_message` by calling out explicitly certain options in plain-text and steering the agent in the desired direction. Available control axes: 

| Axes                 | Type      | Default | Description                                               |
|----------------------|-----------|---------|-----------------------------------------------------------|
| `video sources`      | string[]  | null    | Filter to specific cameras or sensor names                |
| `top k`              | int       | 10      | Max results
| `minimum similarity` | float     | 0.0     | Min similarity threshold; raise (e.g. 0.3) to filter noise|
| `critic usage`       | bool      | true    | VLM verifies each result and removes false positives      |
| `description`        | string    | null    | Filter by camera metadata (e.g. location, category) if metadata is available|

Pick and choose some of these tuning options. Adjust them as needed for the user’s situation and query. 
For examples of discovery modes leveraging these, see [discovery_modes.md](references/discovery_modes.md).

---

## Search via Agent UI

Open `http://${HOST_IP}:3000/` and type natural-language queries:

```
find all instances of forklifts
show me people near the loading dock
when did a truck arrive at the gate?
find someone wearing a red jacket
```

Results include timestamped clips with similarity scores.

bump:1
