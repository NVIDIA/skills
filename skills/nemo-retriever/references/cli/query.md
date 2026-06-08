# retriever query

Embed a text query and return the top-k nearest rows from a LanceDB table
previously written by `retriever ingest` (or any compatible pipeline).

If flags below look stale, re-check `retriever query --help`.

## When to use this

- You have already ingested documents and want to retrieve relevant
  chunks/primitives for a natural-language query.
- You want a one-shot CLI lookup — no service, no UI.

**Use a different command when:**

- You want recall metrics over a labelled query set → `retriever recall`.
- You want to grade end-to-end QA quality → `retriever eval`.
- You want a long-running query endpoint → `retriever service`.
- You want to compare two retrieval runs → `retriever compare`.

## Canonical invocations

Top-10 search against the default table:

```bash
<RETRIEVER_VENV>/bin/retriever query "what is in chart 1?"
```

Top-3, custom table:

```bash
<RETRIEVER_VENV>/bin/retriever query "average frequency ranges for tweeters" \
  --top-k 3 \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

## Inputs

- **Positional `QUERY`** — single text string. Required. Quote it in the shell
  to keep multi-word queries intact.

## Outputs

- JSON array on stdout, one object per hit, in retriever ranking order.
- The root CLI intentionally returns compact objects:
  - `source` — origin document path.
  - `page_number` — 1-indexed page when available.
  - `text` — retrieved primitive text, table text, chart text, or image caption.
- Internal scores, raw metadata, and bounding boxes are available from the Python
  `Retriever.query(...)` API, not the public root CLI output.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--top-k` | `10` | Final number of hits to return. Must be >= 1. |
| `--candidate-k` | unset | Wider pre-filter/pre-dedup candidate pool. When set, it must be >= `--top-k`; make it larger when `--page-dedup` or `--content-types` could reduce final hits. |
| `--page-dedup` | `false` | Collapse results to unique document pages. |
| `--content-types` | unset | Comma-separated content types to keep, such as `text,table` or `image,chart`; query-time values are normalized to canonical hit metadata types, `images` is accepted as an alias for captioned image rows, and untyped hits are excluded. |
| `--lancedb-uri` | `lancedb` | Must match what `ingest` wrote to. |
| `--table-name` | `nemo-retriever` | Must match what `ingest` wrote to. |

## Ranking interpretation

- The embedder (`llama-nemotron-embed-vl-1b-v2`) returns mean-pooled vectors;
  LanceDB ranks by L2 distance by default. The root CLI hides raw distance values;
  treat result order as ranking-only, not calibrated confidence.
- The query uses the **VL** variant of the embedder so text queries can match
  ingested image/chart embeddings as well as text. Expect mixed-modality hits
  in the result list.

## Common failure modes

- **Empty result array** — table is empty (no ingest run yet) or
  `--table-name` / `--lancedb-uri` don't match where ingest wrote.
- **`Table 'nemo-retriever' was not found`** — same root cause: wrong table/URI,
  or ingest hasn't been run.
- **First query is slow (~10–15s)** — vLLM startup for the query embedder.
  Subsequent queries in the same process are sub-second; one-shot CLI
  invocations always pay this cost.
- **Surprisingly low-relevance top hit** — for very short corpora, even
  unrelated queries return *something*. Broaden with `--candidate-k`, use
  `--page-dedup` for page diversity, or use `--content-types` for targeted
  table/chart/image-caption searches.

## Related

- [[ingest]] — populate the table this command reads.
- `retriever recall --help` — batch query → recall@k against ground truth.
- `retriever eval --help` — end-to-end QA evaluation.
