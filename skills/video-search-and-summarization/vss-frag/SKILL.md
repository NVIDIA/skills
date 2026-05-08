---
name: vss-frag
description: "Generate video summary reports using the VSS video_search_frag extension with Long Video Summarization (LVS), Enterprise RAG knowledge retrieval, and human-in-the-loop parameter collection. Use when: user wants to generate a video summary, report, or analysis using the frag pipeline."
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

# VSS Frag — Video Analysis with Enterprise RAG

Generate video summary reports using the VSS `video_search_frag` extension.
This skill adds Enterprise RAG (Milvus) knowledge retrieval and guided
human-in-the-loop (HITL) parameter collection on top of the base VSS agent.

Always run `curl` commands yourself; never instruct the user to run them.

## Deploying the Frag Extension

The frag extension layers Enterprise RAG and HITL LVS tools on top of the base
VSS agent image. Deployment is a two-step Docker build followed by compose up.

> **Environment variables:** All commands use values from the `.env` file at
> `deployments/developer-workflow/dev-profile-lvs/.env`. Edit it before deploying.
> Key variables: `HOST_IP`, `VSS_AGENT_PORT` (default `8000`), `NGC_CLI_API_KEY`,
> `NVIDIA_API_KEY`, `ENTERPRISE_RAG_*`.

### Step 1: Configure the .env file

```bash
nano deployments/developer-workflow/dev-profile-lvs/.env
```

Set at minimum:
- `HOST_IP` — your machine's IP (`hostname -I | awk '{print $1}'`)
- `NGC_CLI_API_KEY` — from https://ngc.nvidia.com/
- `NVIDIA_API_KEY` — from https://build.nvidia.com/
- `VSS_AGENT_CONFIG_FILE=./configs/video_search_frag/config.yml`
- `ENTERPRISE_RAG_VDB_ENDPOINT` — your Milvus endpoint (e.g., `tcp://127.0.0.1:19530`)
- `ENTERPRISE_RAG_COLLECTION_NAMES` — your Milvus collection name

### Step 2: Log in to NGC registry

```bash
echo "$NGC_CLI_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Step 3: Build the base agent image

```bash
cd agent
docker build -f docker/Dockerfile -t vss-agent-base .
```

### Step 4: Build the frag extension image

```bash
docker compose \
  -f app/video_search_frag/docker-compose.yml \
  --env-file ../deployments/developer-workflow/dev-profile-lvs/.env \
  build
```

This produces `vss-agent-frag:latest` — the base agent extended with
`video_search_frag` (Enterprise RAG, HITL LVS, PDF report generation).

### Step 5: Deploy with docker compose

```bash
docker compose \
  -f app/video_search_frag/docker-compose.yml \
  -f ../deployments/agents/agent_ui/compose.yml \
  --env-file ../deployments/developer-workflow/dev-profile-lvs/.env \
  --profile bp_developer_lvs_2d \
  up -d
```

Two `-f` flags: the frag compose defines `vss-agent`, the UI compose defines
`metropolis-vss-ui`. They merge into a single deployment.

### Step 6: Verify deployment

```bash
# Check containers are running
docker ps --format "table {{.Names}}\t{{.Status}}"

# Health check
curl -sf --max-time 5 "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/health" >/dev/null \
  && echo "VSS frag agent is running" \
  || echo "VSS frag agent is NOT reachable"
```

### Tear down

```bash
docker compose \
  -f app/video_search_frag/docker-compose.yml \
  -f ../deployments/agents/agent_ui/compose.yml \
  --env-file ../deployments/developer-workflow/dev-profile-lvs/.env \
  --profile bp_developer_lvs_2d \
  down
```

### Rebuild after code changes

Always `down` then rebuild and `up` — never just `up -d` alone after changes.

```bash
docker compose \
  -f app/video_search_frag/docker-compose.yml \
  --env-file ../deployments/developer-workflow/dev-profile-lvs/.env \
  build

docker compose \
  -f app/video_search_frag/docker-compose.yml \
  -f ../deployments/agents/agent_ui/compose.yml \
  --env-file ../deployments/developer-workflow/dev-profile-lvs/.env \
  --profile bp_developer_lvs_2d \
  down

docker compose \
  -f app/video_search_frag/docker-compose.yml \
  -f ../deployments/agents/agent_ui/compose.yml \
  --env-file ../deployments/developer-workflow/dev-profile-lvs/.env \
  --profile bp_developer_lvs_2d \
  up -d
```

## When to Use

- User wants to generate a video summary or report using the frag pipeline
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

Show the user the video list and ask which one they want to analyze.

### Step 2: Collect parameters from the user

Ask the user for these four inputs one at a time:

1. **Scenario** — What type of scenario is the video about?
   Example: "warehouse monitoring", "traffic monitoring", "retail store activity"
2. **Events** — What events should be detected? Comma-separated.
   Example: "accident, forklift stuck, workers not wearing PPE, person entering restricted area"
3. **Objects of Interest** — What objects should the analysis focus on? Or "skip" to skip.
   Example: "forklifts, pallets, workers"
4. **Enterprise RAG Query** — An optional question to search the enterprise knowledge base
   for additional context to include in the report. Or "skip" to skip.
   Example: "What are the principles of STCC?"

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
curl -sS "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/executions/EXECUTION_ID" | python3 -m json.tool
```

The HITL prompts come in this order:
1. **Scenario** — respond with the scenario from Step 2
2. **Events** — respond with the events from Step 2
3. **Objects of Interest** — respond with the objects from Step 2, or "skip"
4. **Enterprise RAG Query** — respond with the query from Step 2, or "skip"
5. **Confirmation** — respond with empty string "" to confirm and start processing

Repeat the POST-then-poll cycle for each prompt.

### Step 5: Wait for completion

After the confirmation prompt, the system processes the video. This takes 3-5 minutes.
Keep polling until the status changes from "running" to "completed":

```bash
curl -sS "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/executions/EXECUTION_ID" | python3 -m json.tool
```

Tell the user to wait — this takes 3-5 minutes. Poll every 30 seconds.

### Step 6: Present the results

When status is "completed", the response contains the full report with:
- Detected events with timestamps
- Narrative analysis summary
- Enterprise RAG context (if queried)
- PDF report download link (if available)

Present the report content to the user in a readable format.

## Quick Commands

### Health check

```bash
curl -sS "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/health"
```

### Simple chat query (non-report)

For simple questions that do NOT involve report generation:

```bash
curl -sS -X POST "http://${HOST_IP}:${VSS_AGENT_PORT:-8000}/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "YOUR_QUESTION_HERE"}]}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
```

## Notes

- LVS reports take 3-5 minutes for a ~3.5 minute video — always tell the user to wait
- Enterprise RAG requires a Milvus vector database with data ingested
- If objects or rag_query are not needed, respond with "skip"
- The HITL response format is always: `{"response": {"type": "text", "text": "value"}}`
- `enable_interactive_extensions: true` must be set in the frag config for HTTP HITL to work
- See also: `video-summarization`, `video-understanding`, `report`, `vios`, `deploy`
