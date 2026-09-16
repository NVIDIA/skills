---
name: nemo-retriever-mcp
description: Use when a task needs to search or add documents through NeMo Retriever MCP.
license: Apache-2.0
---

# NeMo Retriever MCP

Use the NeMo Retriever MCP tools exposed under `retriever` instead
of the `retriever` CLI.

The server exposes MCP tools, not MCP resources. Do not call
`list_mcp_resources` to discover documents or decide whether Retriever is
available; an empty MCP resource list is expected.

## Workflow

- Use `query` to search documents already available to Retriever.
- Never ingest documents that are already searchable, including documents in a
  prebuilt index.
- Use `ingest_documents` only when required documents are not yet searchable
  and server-readable paths are explicitly available. Do not assume that paths
  visible to the agent are also readable by the Retriever server.
- After ingestion completes, use `query` to retrieve evidence from the added
  documents.
- Start with `top_k=5` and `format="hits"`. Request more only when the first
  successful response does not contain enough evidence to answer the task.
- Read evidence hits from the tool response and ground your answer in them.
- Preserve available document identifiers such as `document_id`, `source_id`,
  or `pdf_basename`, along with page numbers, when the task requests citations
  or structured output.
- If `query` fails, inspect the error and retry with corrected parameters or a
  shorter equivalent query. If it still fails, report the error before using
  another retrieval method.
