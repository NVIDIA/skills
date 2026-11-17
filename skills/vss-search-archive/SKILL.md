---
name: vss-search-archive
description: Use this skill to search archived VSS video or indexed live RTSP camera streams already registered in a configured deployment — by natural-language, similarity, attribute, object-ID, or lexical tag query — and for follow-up questions about displayed search results or their verification. For fresh clip Q&A use vss-ask-video; for full-recording summaries use vss-summarize-video; for source ingestion/deletion use vss-manage-video-io-storage; for deployment use vss-build-vision-ai. Route those requests to the named workflow without loading this skill, including when media, the CLI, or the deployment is unavailable. Not for live captioning.
license: Apache-2.0
metadata:
  author: "NVIDIA Video Search and Summarization team"
  version: "3.3.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
  # What a live deployment must expose for this skill to be usable, as the vss CLI
  # names it: a command group (search, summarize, vlm, vios, memory), "alerts"
  # (Alert Bridge), or "always" for a skill every VSS deployment gets. The
  # OpenClaw harness image ships and activates skills by it.
  vss-requires: "search"
---

# Search archived VSS video

## When to Use

- Search archived VSS video that is already registered in a configured deployment, by natural-language, similarity, attribute, object-ID, or lexical tag query.

Not for:

- A fresh visual question about a supplied local clip — `vss-ask-video`.
- Long-form summarization of a recording — `vss-summarize-video`.
- Deploying or changing a profile — `/vss-build-vision-ai`.
- Ingesting or deleting a source — use the deployment's source-management workflow.

Answer from the configured deployment through the installed `vss` CLI. Do not
fall back to raw REST when a CLI command fails.

> **Hard rule — use the `vss` CLI for retrieval.** `vss configure` has already
> pointed the CLI at the deployment, so every search action goes through
> `vss search run` and nothing reaches the deployment any other way.
>
> Four things follow:
>
> - use the `vss` on PATH (see Prerequisites); never `docker exec`, `kubectl
>   exec`, a pod shell, or a hand-built `/api/v1/search` call;
> - a named source is resolved with `vss vios list` before search — never
>   inferred from a display name, and never substituted when missing;
> - capture stdout and the exit status separately — never put the command
>   behind `if ! command`, which hides the real exit code, and never discard usable
>   exit-6 results;
> - a failing command is a finding: report the exit code rather than routing
>   around it, repairing the deployment, or retrying with broadened scope.

## Prerequisites

- A running VSS `search` profile with `vss configure` already run against its origin. Refresh configuration after first ingestion, once source provisioning has established readiness; lazy raw indexes enable frame enrichment for attribute/fusion.
- The `vss` CLI on PATH. The OpenClaw and Hermes harness images ship it; anywhere else, install it from the same checkout as this skill so the CLI and the skill match: `uv tool install <checkout>/libs/vss/cli`.

Bootstrap, exit codes, and common CLI rules live in [AGENTS.md](../../../AGENTS.md).

## Source management handoff

For an explicit request to ingest or delete a source, use
`vss-manage-video-io-storage` for `vss vios add` or `vss vios delete`, then
return to archive search once the source is available. The deployment's mounted
notification config owns consumer fan-out; the presence of an Agent tier does
not change the VIOS registration path. Check the requested receiver as enabled,
absent, or unknown under source management's policy contract. With an enabled
receiver, use fan-out; with confirmed absent lifecycle support, report that
limitation. Only an explicit tagging request with a confirmed absent streaming
tagging receiver and the provisioning prerequisites may use manual tagging.
With unknown policy, report unconfirmed fan-out, never unavailable indexing,
and do not start manual tagging.
If the user only asks to search a named source and it is missing, do not
ingest, switch videos, or run an unrestricted search — answer with the reply
[Search workflow step 1](#search-workflow) specifies for zero matches. If no
supported source management workflow is available, report that blocker.

## Search workflow

**1. Resolve a named source.** If the request names a file, camera, or sensor,
resolve it with `vss vios list` (it reads the origin `vss configure` recorded,
so it takes no endpoint). Accept an exact name or sensor ID, or one unambiguous
normalized match; stop on zero or multiple matches. The listing is
`{"count", "type", "sensors": [...]}`; preserve the matched entry's
`.sensors[].name` and `.sensors[].sensor_id`, and never infer an identifier from
the display name. Below, `.name` and `.sensor_id` mean those fields.

Stopping is not the whole answer — the reply has to hand the decision back. On
zero matches, the final reply states all three of:

- the requested name is not registered;
- the sources that *are* registered, from the listing;
- a request that the user clarify which source they meant, or explicitly ask
  for the missing one to be ingested.

The third is as required as the first two. Refusing to substitute is correct
but incomplete: a reply that reports the mismatch and then stops leaves the
user with no stated next step. Ingesting, switching to another video, or
dropping the source filter and searching everything are all still forbidden
(see [Source management handoff](#source-management-handoff)). On multiple
matches, name the candidates and ask which one.

**2. Choose one retrieval path.** Preserve the user's exact original sentence for
`--original-query` first — critic verification must receive that wording, while
retrieval may use the decomposed query, attributes, or object IDs. Then choose
exactly one path:

- `object` — explicit tracked object IDs;
- `tag` — explicit lexical tag/keyword intent (BM25 over indexed VLM tags), not semantic free text;
- `attribute` — detectable properties only, no action or relation;
- `fusion` — a detectable property combined with an action or relation;
- otherwise `embed` — semantic free text.

`--attribute` is for properties RT-CV detects on a subject (attire, PPE,
color-on-person), not object identity or an object's own color — keep `red
forklift` wholly in `--query`. `worker in a hard hat carrying a cone` has a
property (`hard hat`) and an action (`carrying a cone`): `run fusion`. Reserve
`embed` for genuinely attribute-free intent, and `tag` for keyword/tag queries
that name no detectable property.

The `--video-source` value differs by path while the CLI's paths need different
identifiers:

| path | `--video-source` takes |
| --- | --- |
| `embed` | preserved `.sensor_id` |
| `attribute` | preserved `.name` |
| `object` | preserved `.name` |
| `tag` | `.name` (the CLI resolves it to the VST sensor ID; an already-id passes through) |
| `fusion` | preserved `.sensor_id` (the tag leg accepts IDs too) |

For every path an unknown source yields an empty, narrowed result, not an error.

Before `attribute` or `fusion`, inspect `vss configure show`'s
`services.elasticsearch.indices`. If no entry starts with `mdx-raw-`, refresh
once with `vss configure --base-url` using that same record's nonempty
`base_url`, then inspect again. Stop on a configuration failure; if the raw
family is still absent, disclose that frame enrichment is unavailable and
continue the requested retrieval. Do not construct an origin or poll indexes.

**3. Invoke the CLI.** Use `--query` for embed/fusion/tag, repeatable
`--attribute` for attribute/fusion, and repeatable `--object-id` for object.
Use `--timestamp-start` / `--timestamp-end` for time bounds. Set `--source-type video_file` for an uploaded recording
or `rtsp` for a live stream. Source type selects the fixed uploads anchor or
the live family wildcard excluding that uploads anchor, independently of source
identity and ingestion order. Carry
the requested time bounds and `--top-k`, and
run `vss search run <path>` with no endpoint, index, model, or profile flag —
`vss configure` owns those. Build the invocation as a Bash array and capture
stdout and the exit status separately; never clear a previously resolved source
array or hide the exit code behind `if ! command`:

```bash
: "${SEARCH_PATH:?set embed|attribute|fusion|object|tag}"
: "${SOURCE_TYPE:?set video_file or rtsp}"
: "${ORIGINAL_QUERY:?set the exact pre-decomposition user question}"
TOP_K="${TOP_K:-3}"
# Set VIDEO_SOURCES from step 1 for a named source. Explicitly set it to () only
# for a request that was unrestricted from the start; never reset a resolved scope.
declare -p VIDEO_SOURCES >/dev/null 2>&1 || { echo "Set VIDEO_SOURCES before search" >&2; exit 1; }
: "${SOURCE_SCOPED:?set true for a resolved scope; false only when unrestricted}"
if [ "${SOURCE_SCOPED}" = true ] && [ "${#VIDEO_SOURCES[@]}" -eq 0 ]; then
  echo "Resolved source scope is empty; refusing an unrestricted search" >&2
  exit 1
fi
SEARCH_COMMAND=(vss search run "${SEARCH_PATH}" --source-type "${SOURCE_TYPE}" \
  --top-k "${TOP_K}" --original-query "${ORIGINAL_QUERY}" --raw)
for source in "${VIDEO_SOURCES[@]}"; do
  SEARCH_COMMAND+=(--video-source "${source}")
done
# Append only the selected path's fields and --timestamp-start/--timestamp-end.
if SEARCH_JSON=$("${SEARCH_COMMAND[@]}"); then
  STATUS=0
else
  STATUS=$?
fi
```

Read [CLI usage](references/cli_usage.md) only when tuning retrieval weights
(`--fusion-method`, `--w-tag`, etc.); do not open it for a standard search
invocation — the contract above is the whole invocation.

**4. Interpret the result by exit status.**

- Exit 0 — interpret `data` and `search_messages`.
- Exit 6 — partial (usually a persistence stage after retrieval failed; a critic failure is exit 0 with `search_messages`): report hits only when the payload contains `data`, disclosing the supplied limitation. Without `data`, report the supplied failure; do not claim retrieval succeeded. Do not rerun or retry an individual stage.
- Exit 2 — read `vss search run <path> --help` once; correct invalid flags or values only from that help, then stop if the corrected command fails.
- Other nonzero — report the typed failure (3 backend unreachable, 4 configuration or missing service, 5 not found) and stop.

Routes not exposed through ingress are recorded as absent by `vss configure`;
a search path requiring one exits 4. Report the missing capability and ask the
operator to expose its supported ingress and
refresh configuration. Never create a port-forward or use private endpoints.
After exit 5 following first ingestion, hand readiness back to source management
and refresh recorded configuration once when it is ready; never ingest implicitly.

An empty `data` array means zero retrieved candidates — a fact about retrieval,
not about the video; a threshold or embedding gap yields the same empty result
as a genuine absence, so do not describe what the footage contains or argue it
is not something you would expect there. If `search_messages` indicate degraded
retrieval, include that limitation. Do not retry, broaden scope, or use raw REST.

For each hit, read its `critic_result`:

- `confirmed`: the critic found all requested visual criteria in that clip.
- `rejected`: the critic found a visual criterion was not met.
- `unverified`: the critic attempted the hit but produced no usable verdict.
  This includes inaccessible media, a failed VLM call, and malformed or
  inconclusive output.
- `null`: the critic did not evaluate the hit (no VLM, a critic failure,
  bounds it could not check, or a hit past `--critic-eval-count`). Report
  it as `unverified`.

**5. Report each hit honestly.** For each hit, report the registered/display
source, bounded interval, retrieval score where present, the returned media
URL **if present**, and the exact `confirmed` / `rejected` / `unverified`
verdict. Retrieval score, filename, source ID, and media availability are not
visual proof. The CLI attempts critic verification by default and is fail-open:
a missing VLM or inaccessible media leaves a hit `unverified` and does not fail
retrieval. Do not inspect screenshots or call another verifier during this
first turn. A media URL may be empty when VST is unavailable; when present it
carries the scheme, host, and port of the origin `vss configure` recorded. Avoid
a mandated heading or raw JSON dump, and keep the reply implementation-neutral
— never expose a job ID, model or service name, deployment service address, CLI
flag, or a raw `sensor_id`; say "visual verification" and report its
verdict. The one exception: when the user asks which commands to run, show the
commands.

A media URL is returned data, not a service address: reproduce it whole,
including the source ID in its path; the `sensor_id` rule is about how you name
the source in prose. Report each hit's verdict, and its `critic_result.criteria_met`
when nonempty, without reading a verdict into the criteria.

**6. Offer a Verification Step only when the whole set is unverified.** If and
only if every displayed result in the nonempty set is `unverified`, offer a
`Verification Step` by asking whether the user wants the hits checked through
`vss-ask-video`. If any hit is
`confirmed` or `rejected`, offer nothing and hand off nothing. On explicit
confirmation, load [search-result verification](references/result_verification.md)
and hand off only the displayed, bounded hits, preserving the original question;
do not rerun search. Never hand off a partially verified result set.
