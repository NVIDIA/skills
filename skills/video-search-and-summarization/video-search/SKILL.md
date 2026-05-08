---
name: video-search
description: Search video archives using natural language — find events, objects, actions, and people across recorded video using fusion search (Cosmos Embed1 semantic search + CV attribute search). Use when asked to search for something in video, find actions and events, locate objects and people, or query video archives. For these types of questions, default to this top-level fusion search unless user specifies otherwise. Requires the search profile to be deployed.
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

# Video Search Workflows

> **Alpha Feature** — not recommended for production use.

Search video archives by natural language using Cosmos Embed1 embeddings. Requires the search profile — deploy with the `deploy` skill (`-p search`). These videos sources can be ingested files or RTSP streams.

## When to Use

- "Find all instances of forklifts"
- "When did someone enter the restricted area?"
- "Show me people near the loading dock"
- "Search for vehicles between 8am and noon"
- Any natural-language search across video archives

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
   > *"The VSS `search` profile isn't running on `$HOST_IP`. Shall I deploy it now using the `/deploy` skill with `-p search`?"*

   - If yes → hand off to the `/deploy` skill. Return here once it succeeds.
   - If no → stop. Do not run this skill against a missing or wrong-profile stack.

   (If your caller has granted explicit pre-authorization to deploy
   autonomously — e.g. the request says "pre-authorized to deploy
   prerequisites", or you are running in a non-interactive evaluation
   harness with that permission — skip the confirmation and invoke
   `/deploy` directly.)

3. If the probe passes, proceed.

---

## How Search Works

1. **Ingest** — Videos are uploaded or streamed via VIOS. The RTVI-Embed service (Cosmos Embed1) generates vector embeddings for video segments.
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
2. Run the search(es) via approach chosen
3. Present the results to the user query. Format response as a professional inspection report but name it `Video Search Results`:
   — Use clear section headers
   - Organize findings individually with supporting detail, and close with a summary
   - Use tables where comparisons help. Write like a technical report, not a chat message.
4. CRITICAL: Verify the results and explain this to the user concisely. 
   If search fails, or returns unexpected results (i.e. videos that do not appear to match user query, zero matches, zero videos returned, error etc.), STOP. Do not proceed without reading [troubleshooting.md](references/troubleshooting.md) to iterate with feedback loops until proper results are found and presented like a professional inspection report.
5. Final verifications:
   - ALWAYS inform user that final and further verifications can be run. Present this as a `Verification Step`
   - ONLY IF user agrees, download screenshots using the `screenshot_url` of the best candidates (highest similarity scores) from the search hits (JSON results) to `/tmp`. Read them and verify if they correspond to the user query

## Input resolution

Infer these inputs only from the conversation or user query (no other files unless provided). If some cannot be inferred, ask the user immediately:
- $HOST_IP: where the VSS agent backend runs

---

## Gotchas

- ALWAYS step into the troubleshooting step of the workflow immediately if anything unexpected happens, read [troubleshooting.md](references/troubleshooting.md)
- Queries work best with **concrete visual descriptions** (objects, actions, locations). Augment user queries if needed to enhance the quality of the questions, expanding potential details
- User queries to do video search supposes video sources are already ingested. No need to search for them locally.
  Assume this unless the findings show the video source is not ingested yet
- Use `video-analytics` skill to cross-reference search results with incident/alert data

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

---

## Interact via Browser (agent-browser)

```bash
npx agent-browser --auto-connect open http://${HOST_IP}:3000
npx agent-browser --auto-connect wait --load networkidle
npx agent-browser --auto-connect snapshot -i
```

Find the chat input, enter a search query, and snapshot results.
