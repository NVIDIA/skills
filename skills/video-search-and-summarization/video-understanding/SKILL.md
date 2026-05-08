---
name: video-understanding
description: Call the vss agent to run video understanding on video to answer a text question. Use when the user asks about video content, or about visual details that cannot be answered from conversation history, search hits, or metadata alone.
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
---

# Video QnA using VLM through VSS Agent

Use this skill when you need details about the video which requires VLM to look at the video frames — for example the agent has **no** usable prior answer and needs a **fresh look at the pixels** for a specific clip.

---

## When to Use

- The user asks **what happens in the video**, what **objects / people / actions** appear, **colors**, **timing**, **safety**, or other **visual facts** that require watching the clip.
- The user asks for **details** that **cannot be answered** from existing messages, summaries, Elasticsearch/MCP results, or filenames alone—you need **model inference on the video**.
- Follow-up questions about **content details** after a coarse summary or after report generation.

Do **not** use this skill when a **database / MCP / prior tool output** already answers the question, unless the user explicitly wants **verification** against the video.

---

## Deployment prerequisite

This skill requires a VSS profile that serves the `video_understanding` tool — typically **base** (recommended) or **lvs**. Before any request:

1. Probe the VSS agent:
   ```bash
   curl -sf --max-time 5 "http://${HOST_IP}:8000/docs" >/dev/null
   ```

2. **If the probe fails**, ask the user:
   > *"No VSS profile is running on `$HOST_IP`. Shall I deploy `base` (recommended for per-clip VLM QnA) using the `/deploy` skill? If you prefer `lvs`, say so."*

   - If yes → hand off to `/deploy -p base` (or `-p lvs` if the user prefers). Return here once it succeeds.
   - If no → stop.

   (If your caller has granted explicit pre-authorization to deploy
   autonomously — e.g. the request says "pre-authorized to deploy
   prerequisites", or you are running in a non-interactive evaluation
   harness with that permission — skip the confirmation and invoke
   `/deploy -p base` directly. Prefer `base` unless the request names
   another profile.)

3. If the probe passes, proceed.

---

## Agent workflow

1. **Clip** — Identify **sensor id**, **filename**, or **URL** for one video segment. If ambiguous, ask the user.
2. Call vss agent with the sensor id and ask for it to call video_understanding tool to answer the user's question. The **sensor / file** name must be included in the input message to the agent.
3. Return the vss agent's answer back to the user.


## Query VSS agent (`/generate`)

```bash
# Set from deployment (compose / .env / host where vss-agent listens)
export VSS_AGENT_BASE_URL="http://localhost:8000"

curl -s -X POST "${VSS_AGENT_BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Call video_understanding tool to answer the following question about <sensor-id>: <user query>"}' | jq .
```

---

## Cross-Reference

- **vios** — VST storage/replay URLs so **`VIDEO_URL`** is valid for the VLM.
- **report** — timestamped **reports** via the **VSS agent** (`/generate`); this skill is **direct VLM** for ad-hoc **video Q&A**.
