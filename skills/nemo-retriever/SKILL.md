---
name: nemo-retriever
description: Use when searching, extracting, ingesting, or querying a document collection with the NeMo Retriever 26.8.1 CLI, including local LanceDB indexes and deployed Retriever services. Use for PDFs, images, Office files, HTML, text, audio, and video; not for editing documents or web search.
license: Apache-2.0
---

# NeMo Retriever

Use the `retriever` CLI. Prefer it over hand-built retrieval
code.

## Install only when missing

Create a project-local Python environment:

```bash
uv venv .venv --python 3.12
export PATH="$PWD/.venv/bin:$PATH"
```

Install the package variant required by the workflow:

```bash
# Remote NIM or service client
uv pip install --python .venv/bin/python "nemo-retriever==26.8.1"

# Local GPU ingestion
uv pip install --python .venv/bin/python "nemo-retriever[local]==26.8.1"

# Local service using Hugging Face models
uv pip install --python .venv/bin/python \
  "nemo-retriever[service,local]==26.8.1"

# Local audio or video ingestion
uv pip install --python .venv/bin/python \
  "nemo-retriever[local,multimedia]==26.8.1"
```

Do not clone NeMo Retriever or install from a Git URL. If `retriever` is already
on `PATH`, use that installation.

## Local workflow

Build a local index:

```bash
retriever ingest <file-or-directory> \
  --lancedb-uri lancedb --table-name nemo-retriever
```

Query it:

```bash
retriever query "<question>" \
  --lancedb-uri lancedb --table-name nemo-retriever \
  --top-k 5 --format evidence
```

Use `retriever ingest batch` only for an explicitly requested Ray batch run.

## Service workflow

Use these forms for an already deployed Retriever service:

```bash
retriever ingest service <file-or-directory> \
  --service-url "$RETRIEVER_SERVICE_URL"

retriever query service "<question>" \
  --service-url "$RETRIEVER_SERVICE_URL" \
  --top-k 5 --format evidence
```

Set `NEMO_RETRIEVER_API_TOKEN` when the service requires Bearer authentication.
Do not pass local LanceDB flags to the service commands.

## Rules

- Use the existing index or service when one is provided; do not rebuild it.
- Use `retriever ingest --help`, `retriever query --help`, or the relevant
  `batch` / `service` help for options not shown here.
- Answer only from retrieved evidence; preserve source and page metadata when
  the task requests citations.
