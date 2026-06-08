# Query turn — the WHOLE workflow


```bash
timeout 2000 <RETRIEVER_VENV>/bin/retriever query "<the user's question>" --top-k 10 --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --query-embed-backend hf --reranker-backend hf --rerank \
  | tee ./hits.json \
  | <RETRIEVER_VENV>/bin/python -c "import json,sys,os; [print(f'rank={i+1} page={h[\"page_number\"]} doc={os.path.basename(h[\"source\"])}') for i,h in enumerate(json.load(sys.stdin))]"
```

Run that **exactly** as a single pipeline — do not split it into `HITS=$(...)` + `echo "$HITS" | <RETRIEVER_VENV>/bin/python -c ...` (the assignment swallows stdout, the pipe sees nothing, you waste 3 bash calls recovering). Stdout is clean JSON (model-init logs are silenced at the CLI layer); leave stderr unredirected so real errors surface on the first call. The full hits land in `./hits.json` **in the current working directory** (not `/tmp` — a shared `/tmp` path gets clobbered when queries run in parallel). The summary above lists rank/page/doc — to read hit text for synthesizing `final_answer`, parse `./hits.json` directly. The top hit's text is one one-liner away: `<RETRIEVER_VENV>/bin/python -c "import json; print(json.load(open('./hits.json'))[0]['text'])"` (or `[i]` for the rank-(i+1) hit). Fetch only what you need — pulling all 10 hits' text into context inflates cached prompt size on every subsequent turn.

That's your FIRST tool call on every query turn. Do not Read, Glob, Grep, or list PDFs before this — those duplicate what `retriever query` already did.

`--query-embed-backend hf` and `--reranker-backend hf` run the query embedder and reranker via HuggingFace instead of vLLM: a single query then loads in ~20–30s (vLLM's batch engine cold-starts much slower and hogs GPU memory). Same model, same hits — just a faster, lighter cold start for one-off queries. (Ingest still uses vLLM for batch throughput.)

**No narration between tool calls.** Do not write "Let me search…", "I'll now analyze…", "The retriever returned…", or any other commentary. Every assistant token you emit between the `retriever query` Bash call and the `Write` of `./output.json` becomes input tokens (and cached input tokens) for every subsequent turn in this session — quadratic cost. Go straight from reading the summary to writing the JSON file. The only assistant text in a query turn should be the tool calls themselves.

Each hit has exactly three keys: `source` (the **full PDF path** — the doc_id is its basename, `os.path.basename(h["source"])[:-4]` to drop `.pdf`), `page_number` (int, **1-indexed**: the first page of a PDF is page `1`), and `text`. There is no `pdf_basename`, `metadata`, `pdf_page`, or `_distance` field — referencing those raises `KeyError`.

## Keyword/regex search across the corpus

If you need exact text matches that semantic `retriever query` may have skipped — e.g. "find every mention of 'mRNA-1273' across all PDFs" — use:

```bash
<RETRIEVER_VENV>/bin/python <skill_dir>/scripts/grep_corpus.py "<regex>" [--max-hits 50]
```

It scans the LanceDB table the retriever already built — no PDF re-extraction. Output is `<pdf>:p<page>:<type>:  ...<snippet>...` per hit; `NO_MATCH` if nothing. Counts against the same "one optional follow-up call" budget as the targeted text-extract (mutually exclusive — pick one).

Don't reach for `pdftotext`, `pdftohtml`, or `pdfgrep` — they're system tools that aren't guaranteed installed on the user's machine. The retriever venv bundles pdfium and `lancedb`; `grep_corpus.py` and `retriever pdf stage page-elements --method pdfium` cover the same use cases without that dependency.

## Compose your reply from the hits

- `final_answer`: synthesize from the top hits' `text`. Include the exact number / name / date / row / column the question asks for, plus the source PDF and 0-indexed page. One paragraph. No restating the question, no hedging caveats. If the chunks talk *around* the fact but don't state it, run ONE `<RETRIEVER_VENV>/bin/retriever pdf stage page-elements ./pdfs --method pdfium --json-output-dir /tmp/pdf_text --compact-json` and `Read` `/tmp/pdf_text/<top_pdf>.pdf.pdf_extraction.json` for the rank-1 page (or rank-2 if rank-1 is metadata) — that almost always surfaces the exact figure. Then synthesize. **If after both calls the asked-for fact still isn't in the evidence, write `final_answer` that says so explicitly** — e.g. "The retrieved pages do not state [X] for [entity]; the closest content is [Y]." Do NOT invent, extrapolate, or generate plausible-sounding content from adjacent material. A confidently-wrong answer scores worse than an honest "not in the retrieved pages".
- `ranked_retrieved`: one entry per hit in the order `retriever query` returned: `{"doc_id": "<pdf_basename without .pdf>", "page_number": <int>, "rank": <i+1>}`. Up to 10. Duplicate `(doc, page)` is fine. **Indexing:** the retriever's `page_number` is 1-indexed. If the task's output schema says 0-indexed (e.g. "first page is page 0"), emit `hit.page_number - 1`; if the task says 1-indexed or doesn't specify, emit `hit.page_number` as-is.

**Before writing `final_answer`, re-read the question.** If it lists multiple entities, years, or categories, your answer must address each one explicitly — even if for some of them the chunks say "not provided" or contain no data. Missing entities lose more judge points than imprecise numbers.

## Charts and images — the single biggest source of judge=2/3 trials

When `metadata.type` of a hit is `chart` or `image`, its `text` field is a model-generated transcription that frequently:

- reverses direction words (`increase`↔`decrease`, `rose`↔`fell`, `surge`↔`drop`), and
- rounds or misreads exact percentages (e.g. transcribing 12% as 20%).

If a question asks for an exact percentage or a directional claim **and the evidence is only a chart/image hit** (no `text`-type hit corroborates the same number or direction):

1. Run the targeted `<RETRIEVER_VENV>/bin/retriever pdf stage page-elements --method pdfium` text-extract on the rank-1 PDF (this counts as your second tool call) and look for the number in prose.
2. If prose confirms the chart number, assert it confidently.
3. If prose doesn't mention it, **quote the chart transcription verbatim with an explicit hedge in `final_answer`**: "The chart on page N indicates [verbatim phrase] (chart-derived, not verified against prose)." Do NOT restate the chart's number as a confident fact.

When both a chart hit and a text hit cover the same fact, always prefer the text hit's number.
After your reply, STOP. No print, no summary, no further tool calls.

## Non-semantic operations (use these, don't fall back to native tools)

**Page filter** — "what's on page N of doc.pdf" → filter LanceDB directly, no `Read`:

```bash
<RETRIEVER_VENV>/bin/python -c "import lancedb,json; df=lancedb.connect('./lancedb').open_table('nemo-retriever').to_pandas(); print('\n'.join(r['text'] for _,r in df.iterrows() if json.loads(r['source'])['source_name']=='APPLE_2022_10K.pdf' and json.loads(r['metadata'])['page_number']==14))"
```

**Verbatim quote with `[page]` citation** — quote retrieved chunks with `[page N]` markers in `final_answer`; don't paraphrase.

**Corpus-level aggregate** — "list distinct sources", "count chunks per source" → no `ls`/`grep`/`find`:

```bash
<RETRIEVER_VENV>/bin/python -c "import lancedb,json; from collections import Counter; df=lancedb.connect('./lancedb').open_table('nemo-retriever').to_pandas(); names=[json.loads(s)['source_name'] for s in df['source']]; print(sorted(set(names))); print(dict(Counter(names)))"
```

**Image / chart captioning** — when the user asks to *describe / caption* an image (prose summary, not OCR text): `retriever ingest` already produces chart/image-type hits whose `text` field is the model-generated caption (see "Charts and images" above). Workflow: ingest the image folder (`setup.md` image recipe), then `retriever query` with a topic-related question — the hits with `metadata.type=chart|image` carry the caption in `text`. Use that as `final_answer`. No separate captioning CLI command.
