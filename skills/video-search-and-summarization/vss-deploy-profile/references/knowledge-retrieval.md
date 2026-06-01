---
name: knowledge-retrieval
description: Configure, swap, or wire in the VSS agent's pluggable knowledge retrieval at deploy time. Use when enabling document grounding via a RAG Blueprint server (`frag_api` adapter), enabling LVS caption Q&A from Elasticsearch (`es_caption` adapter), or running both side-by-side on any profile.
---

# Knowledge Retrieval — Configure, Swap, Wire In

The agent has a pluggable retrieval surface registered as `_type: knowledge_retrieval`. A single Python tool, five backends — `frag_api` (HTTP rag-server), `es_caption` (Elasticsearch captions), `frag_lib` (in-process Milvus), `langchain` (embedded Chroma via LangChain), `llama_index` (embedded Chroma via LlamaIndex) — and any number of named instances per deployment. Today the repo ships two named instances:

| Tool name in YAML | `_type` | `backend` | Mainly for |
|---|---|---|---|
| `lvs_caption_retrieval` | `knowledge_retrieval` | `es_caption` | Follow-up Q&A on summarized RTSP live streams (no VLM re-run) |
| `frag_retrieval` | `knowledge_retrieval` | `frag_api` | Document grounding via a deployed NVIDIA RAG Blueprint rag-server |

You can register both, either, or neither — the YAML block name (`lvs_caption_retrieval`, `frag_retrieval`, your-name-here) is what the LLM sees as the tool; the `_type` just routes it to the right Python registration.

## Current config examples

| Profile | Default `config.yml` | `config_rag.yml` (opt-in via `VSS_AGENT_CONFIG_FILE`) |
|---|---|---|
| `base` | nothing | [`frag_retrieval`](../../../deploy/docker/developer-profiles/dev-profile-base/vss-agent/configs/config_rag.yml) |
| `lvs` | [`lvs_caption_retrieval`](../../../deploy/docker/developer-profiles/dev-profile-lvs/vss-agent/configs/config.yml) | [superset: `lvs_caption_retrieval` + `frag_retrieval`](../../../deploy/docker/developer-profiles/dev-profile-lvs/vss-agent/configs/config_rag.yml) |

- LVS profile has caption Q&A out of the box — no swap needed. Just deploy `lvs` and ask questions about a captioned stream.
- `config_rag.yml` is a superset of `config.yml` plus `frag_retrieval`. Swap to it when you want both stream caption Q&A and document grounding side-by-side; nothing is lost.

## When to Use

- "Enable document grounding (`frag_retrieval`) on `base`" → swap to `config_rag.yml`
- "Enable both stream caption Q&A and document grounding on `lvs`" → swap to `config_rag.yml`

## Customization

Three patches per retriever you want to add. They can be layered on top of any profile's `config.yml` — including profiles that already register one retriever and you want to add the other.

### Procedure

1. Read the target profile's existing `config.yml` to see which retrievers (if any) are already registered. Preserve them.
2. Pick the adapter(s) to add: `frag_api`, `es_caption`, or both.
3. For each retriever to add, apply Patches 1, 2, 3 below.
4. Save as a new file in the profile's `configs/` directory (convention: `config_rag.yml` if it adds frag), then repoint `VSS_AGENT_CONFIG_FILE` in the profile's `.env`.
5. Hand off to the `vss-deploy-profile skill` (see `../SKILL.md`) with the new config path.

### Patch 1 — register a function block

Add an entry under `functions:` with `_type: knowledge_retrieval`. Pick a tool name (the YAML block name) and a backend, then fill in [backend-specific fields](#backend-reference).

```yaml
functions:
  ...
  my_retriever:                    # your chosen tool name
    _type: knowledge_retrieval
    backend: frag_api              # or es_caption
    top_k: 5
    backend_config:
      # … fields per the backend reference below
```

### Patch 2 — expose it to the workflow

Append the tool name (whatever you used as the block name above) to `workflow.tool_names`:

```yaml
workflow:
  ...
  tool_names:
    - <existing tools>
    - my_retriever                 # add this
```

### Patch 3 — teach the routing agent when to call it

Add a routing rule inside the `workflow.prompt:` block, under `## Routing Rules:`. Backend-specific filter shapes are baked into the tool description automatically — you only need a short trigger rule:

- **`frag_api`** — *"Call for compliance / SOP / policy / procedure questions. Pass `filters` only when the user names an exact filename — never invent one."*
- **`es_caption`** — *"DEFAULT path for content questions about a captioned live stream. Pass `collection = <stream's friendly name>` and a `query` keyword. Add `filters={\"time_range\": {...}}` only when the user gives a time window. If `chunks=[]`, fall back to `lvs_stream_understanding`."*

## Backend reference

### `frag_api`

For document grounding via a deployed [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag).

| Field | Purpose | Default |
|---|---|---|
| `collection_name` | Default Milvus collection (used when caller passes empty) | env `KNOWLEDGE_COLLECTION` |
| `rag_url` | rag-server `/v1` endpoint | env `RAG_SERVER_URL` |
| `api_key` | Bearer token (optional) | env `RAG_API_KEY` |
| `verify_ssl`, `timeout` | HTTP client tuning | `true`, `300` |

**Per-query `filters`** — pass `filter_expr` only when the user names an exact filename:

```python
filters = {"filter_expr": 'content_metadata["filename"] == "<exact name>"'}
```

### `es_caption`

For LVS live-stream Q&A. Reads the same Elasticsearch instance the LVS pipeline writes to (Kafka → Logstash → ES). One ES index per stream (`default_<uuid_with_underscores>`); the adapter searches across all by default. Q&A on uploaded videos is not supported right now.

| Field | Purpose | Default |
|---|---|---|
| `elasticsearch_url` | ES base URL | env `ELASTIC_SEARCH_ENDPOINT` |
| `index` | Index pattern to search | `default_*` |
| `default_doc_type` | `doc_type` to retrieve when callers don't override | **`raw_events`** |
| `api_key` | ES API key (optional) | unset |
| `verify_ssl`, `timeout` | HTTP client tuning | `true`, `30` |

Three `doc_type` values are written per streamed video — `raw_events` (per VLM chunk), `structured_events` (merged batches), and `aggregated_summary` (one narrative per video). Override per-query via `filters.doc_type`.

**Per-query `filters`** — all keys optional. Pass `doc_type` to override the deployment default (see paragraph above).

| Key | Value |
|---|---|
| `time_range` | `{"start": <ISO 8601 \| epoch>, "end": <same>}` — either bound optional. See below. |
| `es_query` | Full ES query body — escape hatch that replaces the constructed query. |

**`collection_name`** — pass the stream's friendly name (or its `stream_id` / UUID). If omitted, the search runs across all streams in the configured index pattern.

## Additional backends (`frag_lib`, `langchain`, `llama_index`)

Three more backends are available behind opt-in Python packages (`vss-agents[frag_lib]`, `vss-agents[langchain]`, `vss-agents[llama_index]`). These are experimental — the default agent image does not install them. Enable one by passing `--extra <backend>` at image build time.

### Enabling one

1. In `services/agent/docker/Dockerfile`, add `--extra <backend>` to the `uv sync` line. For example, the default line:
   ```
   uv sync --frozen --no-dev --no-editable --link-mode copy
   ```
   becomes (for `frag_lib`):
   ```
   uv sync --frozen --no-dev --no-editable --link-mode copy --extra frag_lib
   ```
   Then rebuild the agent image with the same tag your deploy already uses.

2. Add a `config_<backend>.yml` to your profile's configs directory and set `VSS_AGENT_CONFIG_FILE` to point at it.

3. Bring the deployment up as usual.

### `frag_lib`

In-process equivalent of `frag_api`. Runs the full NVIDIA RAG Blueprint pipeline (retrieve → rerank → optional guardrails) inside the agent against a Milvus you supply. Same `filter_expr` shape as `frag_api`, so the LLM-facing contract is interchangeable.

| Field | Purpose | Default |
|---|---|---|
| `milvus_uri` | Milvus URI for the vector store | `nvidia-rag` default (`http://localhost:19530`) |
| `collection_name` | Default Milvus collection (used when caller passes empty) | `default` |
| `llm_base_url` | LLM NIM `server_url` override | unset (uses `nvidia-rag` default) |
| `llm_model_name` | LLM model name override | unset |
| `embedder_base_url` | Embedder NIM `server_url` override | unset |
| `embedder_model_name` | Embedder model name override | unset |
| `enable_citations` | Return citation metadata with each chunk | `true` |
| `enable_guardrails` | Run guardrails on the response | `false` |

**Per-query `filters`** — same shape as `frag_api`:

```python
filters = {"filter_expr": 'content_metadata["filename"] == "<exact name>"'}
```

### ChromaDB-backed adapters (`langchain`, `llama_index`)

Both adapters do in-process retrieval over an embedded ChromaDB persist
directory using the NVIDIA embedding NIM. The persist directory must be
pre-built (no in-adapter ingestion); mount it into the agent container at a
known path. They share the same field set and defaults:

| Field | Purpose | Default |
|---|---|---|
| `persist_dir` | Path to the Chroma persist directory inside the container | env `VSS_CHROMA_DIR`, then `/tmp/chroma_data` |
| `collection_name` | Default Chroma collection (used when caller passes empty) | `default` |
| `embed_model` | NVIDIA embedding model name | `nvidia/llama-nemotron-embed-vl-1b-v2` |
| `embed_base_url` | Embedding NIM base URL | `https://integrate.api.nvidia.com/v1` |
| `embed_api_key` | NVIDIA API key | env `NVIDIA_API_KEY` |

**Per-query `filters`** — filter pushdown is not supported for either adapter;
pass only `query`, `collection`, and `top_k`.

**Framework-specific dependencies:**

- `langchain` — routes through LangChain's `langchain_chroma` + the NVIDIA
  embedding wrapper.
- `llama_index` — routes through LlamaIndex's
  `llama-index-vector-stores-chroma` + `llama-index-embeddings-nvidia`.

Behavior is otherwise identical; pick the framework that matches the rest of
the caller's stack.

## Revert

To go back to the profile's default behaviour, set `VSS_AGENT_CONFIG_FILE` back to `config.yml` and redeploy.
