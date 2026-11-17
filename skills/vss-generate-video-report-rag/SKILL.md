---
name: vss-generate-video-report-rag
description: Generates VSS video summary reports with LVS HITL and optional Enterprise RAG document grounding. Trigger when the user asks for a frag/RAG-assisted video report, knowledge-enhanced analysis, or Enterprise RAG context in a video summary.
license: Apache-2.0
metadata:
  version: "3.3.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

# VSS Generate Video Report RAG — Video Analysis with Enterprise RAG

Generate video summary reports using the LVS profile's RAG-enabled agent config.
This skill adds Enterprise RAG document grounding and guided human-in-the-loop
(HITL) parameter collection on top of the VSS agent.

Always run `curl` commands yourself; never instruct the user to run them.

## Enable Enterprise RAG on the LVS Profile

The repository ships the RAG-enabled LVS agent config at
`deploy/docker/developer-profiles/dev-profile-lvs/vss-agent/configs/config_rag.yml`.
It is a superset of the default LVS config: regular caption retrieval remains
enabled, and `frag_retrieval` adds Enterprise RAG document grounding.

Use the normal `/vss-build-vision-ai` workflow for deployment. The source
`.env` and `overrides.env` remain read-only; initialize `generated.env` from
`overrides.env` and apply non-secret overrides there.
`generated.env` is ignored by the repository, but it is still a plaintext file:
do not commit it, paste it into logs, or store long-lived credentials there.
Prefer a vault, Docker secrets, or ephemeral shell environment variables for
API keys.

### Step 1: Configure the generated env file

```bash
REPO=${REPO:-$(git rev-parse --show-toplevel)}
cd "$REPO"
cp deploy/docker/developer-profiles/dev-profile-lvs/overrides.env \
  deploy/docker/developer-profiles/dev-profile-lvs/generated.env
```

Set these non-secret values in `generated.env`:
- `HOST_IP` — host IP (`hostname -I | awk '{print $1}'`)
- `VSS_AGENT_CONFIG_FILE=/vss-agent/deploy/docker/developer-profiles/dev-profile-lvs/vss-agent/configs/config_rag.yml`
- `RAG_SERVER_URL` — Enterprise RAG server HTTP endpoint (defaults to `http://rag-server:8081/v1`)
- `KNOWLEDGE_COLLECTION` — default Enterprise RAG collection for `frag_retrieval`

Keep sensitive values (`NGC_CLI_API_KEY`, `NVIDIA_API_KEY`, `RAG_API_KEY`) out
of `generated.env` and out of `resolved.yml`. Do not export them before running
`docker compose config > resolved.yml`, because Compose expands environment
variables into that file. Use a secret manager, an existing authenticated Docker
session, or a local override file that references an ephemeral shell variable at
`up` time.

### Step 2: Log in to NGC registry

Prefer an existing authenticated Docker session or a secret-managed login. If a login is required, use `--password-stdin` without printing token values:

```bash
read -rsp "NGC API key: " NGC_CLI_API_KEY
printf '%s\n' "$NGC_CLI_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
unset NGC_CLI_API_KEY
```

### Step 3: Deploy the LVS profile with the RAG config

Do not export `RAG_API_KEY` for the dry-run below. If the RAG server requires an
API key, create this untracked local override after `resolved.yml` is generated:

```bash
cat > rag-secret.override.yml <<'EOF'
services:
  vss-agent:
    environment:
      RAG_API_KEY: ${RAG_API_KEY:?Set RAG_API_KEY only for docker compose up}
EOF
```

```bash
REPO=${REPO:-$(git rev-parse --show-toplevel)}
cd "$REPO/deploy/docker"
docker compose \
  --env-file developer-profiles/dev-profile-lvs/.env \
  --env-file developer-profiles/dev-profile-lvs/generated.env \
  config > resolved.yml
uv run "$REPO/skills/vss-build-vision-ai/scripts/normalize_resolved_yml.py" \
  "$REPO/deploy/docker/resolved.yml"
docker compose \
  --env-file developer-profiles/dev-profile-lvs/.env \
  --env-file developer-profiles/dev-profile-lvs/generated.env \
  -f resolved.yml up -d
```

When `rag-secret.override.yml` is needed, use:

```bash
read -rsp "RAG API key: " RAG_API_KEY
RAG_API_KEY="$RAG_API_KEY" docker compose \
  --env-file developer-profiles/dev-profile-lvs/.env \
  --env-file developer-profiles/dev-profile-lvs/generated.env \
  -f resolved.yml -f rag-secret.override.yml up -d
unset RAG_API_KEY
```

### Step 4: Verify deployment

```bash
# Check containers are running
docker ps --format "table {{.Names}}\t{{.Status}}"

# Health check
curl -sf --max-time 5 "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/health" >/dev/null \
  && echo "VSS LVS RAG agent is running" \
  || echo "VSS LVS RAG agent is NOT reachable"
```

### Tear down

```bash
REPO=${REPO:-$(git rev-parse --show-toplevel)}
cd "$REPO/deploy/docker"
docker compose -f resolved.yml down
```

## When to Use

- User wants to generate a video summary or report using the RAG-enabled LVS pipeline
- User asks to analyze a video with Enterprise RAG knowledge context
- User mentions "frag", "enterprise RAG", or "knowledge-enhanced report"

## When NOT to Use

- Simple video understanding queries (use `video-understanding` skill)
- Direct LVS summarization without HITL (use `video-summarization` skill)
- Deployment tasks (use `deploy` skill)
- Real-time alerts (use `alerts` skill)

## Workflow: Generate an LVS Report with Enterprise RAG

### Step 1: List available videos

```bash
curl -sS -X POST "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What videos are available?"}]}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
```

A selected video is required before Step 2. If the user has not already named one, return the short list and stop; resume when the user supplies the video name.

### Step 2: Collect parameters from the user

Required user-provided parameters:

1. **Scenario** — scenario label for the video.
   Example: "warehouse monitoring", "traffic monitoring", "retail store activity"
2. **Events** — comma-separated event names to detect.
   Example: "accident, forklift stuck, workers not wearing PPE, person entering restricted area"
3. **Objects of Interest** — focus objects, or "skip".
   Example: "forklifts, pallets, workers"

If any required value is missing, return a concise missing-fields message and stop; resume the workflow when the user supplies the missing values.

There is no separate Enterprise RAG Query HITL prompt. Document grounding comes
from the RAG-enabled agent config exposing `frag_retrieval`; if the user wants
specific SOP, policy, or procedure context reflected in the report, capture that
context in the original report request or resolve it as a document-grounding
question before starting the HITL report flow.

### Step 3: Start the report (HTTP HITL)

Send a POST to `/v1/chat`. This returns HTTP 202 with an execution_id and the first
HITL prompt. Replace VIDEO_NAME with the chosen video:

```bash
curl -sS -X POST "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Generate a report for VIDEO_NAME using long video summarization"}]}'
```

The response contains:
- `execution_id` — save this, used in all subsequent requests
- `interaction_id` — identifies the current prompt
- `prompt.text` — the HITL prompt text
- `response_url` — the URL to POST the response to

### Step 4: Respond to HITL prompts

For each prompt, POST the user's parameter to the response_url.
Replace EXECUTION_ID, INTERACTION_ID, and the text value:

```bash
curl -sS -X POST \
  "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/executions/EXECUTION_ID/interactions/INTERACTION_ID/response" \
  -H "Content-Type: application/json" \
  -d '{"response": {"type": "text", "text": "USER_VALUE_HERE"}}'
```

Then poll for the next prompt:

```bash
curl -sS "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/executions/EXECUTION_ID"
```

The HITL prompts come in this order:
1. **Scenario** — respond with the scenario from Step 2
2. **Events** — respond with the events from Step 2
3. **Objects of Interest** — respond with the objects from Step 2, or "skip"
4. **Confirmation** — respond with empty string "" to confirm and start processing

Repeat the POST-then-poll cycle for each prompt.

### Step 5: Wait for completion

After the confirmation prompt, the system processes the video. This takes 3-5 minutes.
Keep polling until the status changes from "running" to "completed":

```bash
curl -sS "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/executions/EXECUTION_ID"
```

Set the expectation that processing usually takes 3-5 minutes, then poll every 30 seconds.

### Step 6: Present the results

When status is "completed", the response contains the full report with:
- Detected events with timestamps
- Narrative analysis summary
- Enterprise RAG context (if queried)
- PDF report download link (if available)

Present the report content to the user in a readable format.

## Error Handling

- If a deployment, health, or chat request fails, report the failing endpoint, HTTP status or command error, and the most useful next check. Do not continue into HITL without a valid `execution_id`, `interaction_id`, and `response_url`.
- If a HITL response is rejected or the next execution poll omits the expected prompt, stop and show the execution status plus any error payload instead of guessing the next prompt.
- If the execution status becomes `failed`, `cancelled`, or stays `running` without progress beyond the expected processing window, surface the status and recommend checking the `vss-agent` logs before retrying.
- If the final response lacks report text or a PDF link, return the available response fields and clearly state which output was missing.

## Quick Commands

### Simple chat query (non-report)

For simple questions that do NOT involve report generation:

```bash
curl -sS -X POST "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "YOUR_QUESTION_HERE"}]}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
```

## Notes

- LVS reports take 3-5 minutes for a ~3.5 minute video; set that expectation before polling
- Enterprise RAG requires a reachable RAG server with data already ingested in `KNOWLEDGE_COLLECTION`
- If objects are not needed, respond with "skip"
- The HITL response format is always: `{"response": {"type": "text", "text": "value"}}`
- HTTP HITL is enabled by default in the Docker Compose LVS deployment; set
  `HITL_ENABLED=false` to disable it. NemoClaw defaults HITL to false, so set
  `HITL_ENABLED=true` explicitly when that harness needs HITL.
- See also: `video-summarization`, `video-understanding`, `report`, `vios`, `deploy`

