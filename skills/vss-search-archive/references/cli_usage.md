# `vss search run` reference

One CLI for Compose and Kubernetes. Endpoints come from the deployment recorded
by `vss configure`; the command takes none. Run the `vss` on PATH — shipped in
the harness images, otherwise installed from this skill's checkout with
`uv tool install <checkout>/libs/vss/cli`. Bootstrap, exit codes, and the
common CLI rules live in [AGENTS.md](../../../../AGENTS.md); this reference covers
search-specific invocation and optional retrieval tuning only.

```bash
vss search run <path> [options]
```

Verify the entry point directly when preflight is uncertain:

```bash
vss search run --help
```

Do not invoke it through `docker exec`, `kubectl exec`, or a pod shell. Do not
manually call Elasticsearch, embedding, or search endpoints.

## The five paths

| command | fields | services required |
| --- | --- | --- |
| `run embed` | `--query` | Elasticsearch, RT-Embed |
| `run attribute` | `--attribute` (repeatable) | Elasticsearch, RT-CV |
| `run tag` | `--query` + `--video-source` (repeatable, optional) | Elasticsearch |
| `run fusion` | `--query` + `--video-source` (repeatable, optional) + optional `--attribute` / `--description` / `--min-cosine-similarity` | Elasticsearch, RT-Embed (RT-CV optional) |
| `run object` | `--object-id` (repeatable) | Elasticsearch, RT-CV |

Each path accepts only its own fields. `run embed` has no `--attribute`;
`run attribute` and `run object` have no `--query`; `run tag` has no
`--attribute`. A path whose services are absent exits 4 naming them, before any
request.

VST is not required by any path: it only mints `screenshot_url` media links and
resolves source names to stream ids. A deployment that exposes Elasticsearch
and the path's retrieval services but not VST still searches; hits return with an
empty `screenshot_url`, and a named `--video-source` that VST cannot resolve
narrows to an empty result rather than failing.

## Query controls

Shared by every path: `--source-type`, `--video-source` (repeatable),
`--timestamp-start`, `--timestamp-end`, `--top-k`, and `--original-query`.
When a caller decomposes the request, `--original-query` carries the exact
pre-decomposition user sentence to the critic; retrieval continues to use the
path-specific query, attributes, or object IDs.

```bash
# Embed-only
run embed --query "red forklift" --source-type video_file --top-k 10

# Time-bounded named-source search
run embed --query "person at entrance" --video-source entrance-camera \
  --timestamp-start "2025-01-01T14:00:00" --timestamp-end "2025-01-01T15:00:00"

# Tag-only (BM25 over VLM tag documents)
run tag --query "forklift loading pallet" --source-type video_file \
  --video-source warehouse_sample --top-k 10

# Fusion (tag + embed + optional attribute)
run fusion --query "person in white jacket running" --attribute "white jacket" \
  --source-type video_file --video-source warehouse_sample
```

`--video-source` is matched **literally** against the index for `embed`,
`attribute`, and `object` — the CLI does no name↔id resolution or VST
validation, so an unknown source silently returns nothing (not an error). `tag`
resolves a source name to its VST sensor ID (an already-id passes through);
`fusion` matches the sensor ID literally. See the `--video-source` table in
SKILL.md for which identifier each path takes; validating a named source
against `vss vios list` is the skill's job either way.

## Retrieval tuning

`--fusion-method weighted_rrf|rrf`, `--w-tag`, `--w-embed`, `--w-attribute`,
`--rrf-k`, `--rrf-w`, `--top-percent-filter`,
`--embed-confidence-threshold`, `--min-cosine-similarity`. At least one
provider weight must be positive; library defaults are `w_tag=0` (VLM tag leg
off by default), `w_embed=0.35`, `w_attribute=0.55`, `rrf_k=60`,
`rrf_w=0.5`, `fusion_method=rrf` (legacy embed + attribute RRF,
no tag leg). Opting into the VLM tag leg with `--w-tag > 0`
auto-selects `weighted_rrf` (the only method that fuses a tag leg);
an explicit `--fusion-method rrf --w-tag > 0` is an input error.
`--critic-eval-count N` caps how many retrieved hits the VLM critic verifies;
hits beyond the cap stay `unverified`. Omit to verify every hit (bounded by
`--top-k`). The critic is best-effort, fail-open, and already concurrent
(semaphore 5), so for small top-k it costs about one VLM round trip; the cap
bounds latency and remote-VLM cost on large result sets.

`--no-merge-adjacent` reports raw retrieval windows. By default contiguous
same-sensor windows merge into one result whose score is the mean of the merged
windows — expect fewer, longer results with averaged scores.

## Output and exits

JSON on stdout (`SearchOutput.data`). `--raw` compact, `--pretty` indented.

| exit | meaning |
| --- | --- |
| 0 | success |
| 1 | unexpected error; report failure, nothing actionable |
| 2 | invalid input (unknown flag, bad value); read the selected path's `--help` once before correcting the invocation |
| 3 | backend unreachable |
| 4 | configuration — not configured, foreign config, or a required service absent |
| 5 | not found: a searched index that is not the uploads anchor is missing (an absent anchor returns exit 0 with empty results) |
| 6 | partial: when `data` exists, report hits and the supplied limitation (e.g. `persisted: false`); without `data`, report the supplied failure. Never retry the job or an individual stage |
| 7 | timeout: the marker carries a job id for `status`/`get`; the work is gone, only the caller can decide to spend it again |

Search automatically attempts bounded visual verification through
`vss_core.search_core.critic` when `vss configure` discovered both VST and an RT-VLM model.
When those services are available, the critic attempts every returned hit,
unless `--critic-eval-count N` caps it to the first N.
Every hit the critic evaluated contains `critic_result.result`: `confirmed`,
`rejected`, or `unverified`. Verification is fail-open and never fails
retrieval: an inaccessible clip or a failed VLM call for an evaluated hit gives
`result: "unverified"`. `critic_result: null` means the hit was not evaluated:
no VST or RT-VLM route, an RT-VLM that fails the startup probe, a critic error,
a hit with no sensor or invalid bounds, or a hit past the cap. Both read as
unverified. `--critic-eval-count` is the only critic flag; deployment discovery
remains the single source of endpoints and model ids.

Only when every displayed hit is `unverified` may the host ask whether the user
wants them checked through the separate `vss-ask-video` workflow. If even one
hit is `confirmed` or `rejected`, do not offer or invoke that fallback.

Model ids come from `vss configure show`; the CLI never accepts an index. Bases
and family wildcards are pinned, and host-side ES checks use the family
wildcards. Never pass or infer an index and never read `ELASTIC_SEARCH_INDEX`; it
names only the embedding index and must not be reused as the behavior or raw
index.

Never provide secrets through CLI flags. Kubernetes Secret values are not read
by this command.

`vss search run` is read-only. For upload, registration, deletion, or repair,
use the `vss-manage-video-io-storage` skill (`vss vios add` / `vss vios delete`).

For live streams, `--source-type rtsp` selects the live family wildcard excluding
the fixed uploads anchor, regardless of ingestion order. `vss configure` records
routes not exposed through ingress as absent; a path requiring one exits 4.
Report missing required ingress for the operator to expose the
supported route and reconfigure, without port-forwarding or private endpoints.

Refresh `vss configure` after first ingestion. Before attribute/fusion retrieval,
inspect `services.elasticsearch.indices` in `vss configure show`. If no name
starts with `mdx-raw-`, refresh once using the recorded nonempty `base_url`.
Inspect again and disclose absent frame enrichment if that family remains
missing. Readiness belongs to source provisioning, not a search retry loop.

For attribute/fusion, this bounded inventory check uses recorded configuration
only and continues with a disclosed limitation when refresh cannot discover raw
indexes:

```bash
CONFIG_JSON=$(vss configure show) || exit $?
if ! printf '%s' "${CONFIG_JSON}" | jq -e \
  'any(.services.elasticsearch.indices[]?; startswith("mdx-raw-"))' >/dev/null; then
  RECORDED_ORIGIN=$(printf '%s' "${CONFIG_JSON}" |
    jq -er '.base_url | select(type == "string" and length > 0)') || exit 1
  vss configure --base-url "${RECORDED_ORIGIN}" >/dev/null || exit $?
  CONFIG_JSON=$(vss configure show) || exit $?
  if ! printf '%s' "${CONFIG_JSON}" | jq -e \
    'any(.services.elasticsearch.indices[]?; startswith("mdx-raw-"))' >/dev/null; then
    echo "Frame enrichment unavailable: raw index family remains absent" >&2
  fi
fi
```
