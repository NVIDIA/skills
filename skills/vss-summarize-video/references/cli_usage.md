# `vss summarize run` reference

One CLI for Compose and Kubernetes. The LVS origin and the Elasticsearch
holding unified memory come from the deployment recorded by `vss configure`;
the command takes no endpoint flags.

Run the `vss` on `PATH` — shipped in the harness images, otherwise installed
from this skill's checkout with `uv tool install <checkout>/libs/vss/cli`:

```bash
vss summarize run --help >/dev/null || exit 1
```

Do not run it through `docker exec`, `kubectl exec`, or a pod shell.

## Configure once

```bash
vss configure --base-url "${VSS_PUBLIC_URL}"   # probe + record ~/.vss/config.json
vss configure show                              # recorded services, models, indices
vss configure check                             # re-probe; exit 3 if a route went away
```

`VSS_PUBLIC_URL` is the ingress origin the operator gave you. If it is unset,
stop and ask; do not substitute `HOST_IP`, `localhost`, or a port. Never configure
against `:38111` directly: that is the LVS container port, and a deployment
recorded from it exposes no Elasticsearch for memory. Re-run `configure` after
any deployment change.

`summarize` needs the recorded `lvs` service. Persistence additionally needs
`elasticsearch`; `vss configure show` is what proves both are routed.

## The one job command

```bash
vss summarize run --url <video-url> --video-id <id> \
  --scenario "warehouse monitoring" --event "forklift activity" \
  --creation-time 2025-01-01T00:00:00.000Z
```

Exactly one of `--id` (media the deployment already ingested) or `--url` (a
video to fetch directly) is required. `--scenario` and at least one `--event`
are required because LVS requires them and answers 422 without them; there is
no default to invent.

One `run` is exactly one `POST /v1/summarize`. The command never retries, and
returns only when the result is final.

| flag | meaning |
|---|---|
| `--id` / `--url` | the media; exactly one |
| `--scenario` | use-case context, required |
| `--event` | event to detect; repeat for several, required |
| `--object-of-interest` | object to detect; repeatable |
| `--creation-time` | absolute media start, ISO-8601 UTC |
| `--model` | VLM id; defaults to what the deployment reports serving |
| `--prompt`, `--system-prompt` | VLM prompts |
| `--chunk-duration`, `--chunk-overlap-duration` | chunking, seconds |
| `--temperature`, `--top-p`, `--top-k`, `--max-tokens`, `--seed` | sampling |
| `--num-frames-per-chunk` | frames sampled per chunk |
| `--enable-audio/--no-enable-audio` | transcribe audio alongside video |
| `--no-enable-vlm-structured-output` | prose instead of summary + events |

Omitted fields are absent from the request rather than sent as null, so the
backend's own defaults apply. Do not pass `--model` unless the caller named
one: the default is the model `vss configure` recorded from LVS itself.

### `--creation-time` is what makes events persistable

LVS reports event times as numbers — offsets into the clip unless a
`creation_time` anchors it. Unified memory stores instants, so a run that
persists without `--creation-time` cannot write its events and degrades to
exit 6 with the summary still on stdout. Pass the media's absolute start time
whenever configured persistence is enabled. `2025-01-01T00:00:00.000Z` is the
conventional value for uploaded sample media with no real timestamp.

## Persistence

| flag | meaning |
|---|---|
| `--no-persist` | safe per-run opt-out from the configured persistence policy |
| `--write-memory-note/--no-write-memory-note` | per-run OpenClaw cache override |
| `--video-id` | `video_id` on the record; defaults to `--id`, required with `--url` |
| `--media-source` | `media_ref.source`, default `vst` |
| `--media-name` | `media_ref.name`, e.g. the original filename |
| `--request-timeout-seconds` | HTTP timeout, default 3600 |

Configure Elasticsearch, its authoritative index, the default persistence
policy, and optional OpenClaw Markdown cache once with `vss configure memory`.
There is no positive per-run `--persist` flag and no per-run
`--memory-index`. Query, identity, media, and model fields remain request
flags.

Persisting a `--url` summary without `--video-id` exits 2 before the
summarization runs, rather than after paying for it. Both edges are configured
to wait as long as this default does, so a summarization that runs for an hour
is not cut short by a 504 the CLI would have to record as a failed job.

The job is written twice — `submitted` before the VLM call, terminal after — so
a run that times out or dies still leaves a record to reconcile against.

## Output and exits

Output is compact by default: stdout contains two JSON documents. The first is
the operation result; the final line is the completion marker naming the job.
Read both, and do not treat prose on stderr as the result. Do not pass
`--pretty`, which indents the documents across many lines and breaks
first-line/final-line parsing. (`--help` renders the default as `(pretty)`; the
behaviour is compact.)

```json
{"job_id": "summarize-01K...", "summary": { ... LVS envelope ... },
 "persist": {"status": "complete", "index": "...", "group": "summary", "events": 3},
 "record": "closed"}
{"event":"vss_job_completed","group":"summary","job_id":"summarize-01K...",
 "status":"completed","persisted":true,"exit_hint":0}
```

The LVS response is nested under `summary` verbatim, so its own envelope is
unchanged: `summary.choices[0].message.content` is the JSON-encoded string
holding `video_summary` and `events`, and `summary.usage` still carries
`total_chunks_processed`. A run that fails after the job exists replaces those
keys with `status`/`record`/`error` but still names the `job_id`, so only exits
0 and 6 carry a summary to parse — reach for `summary` on any other exit and jq
fails on a missing field, burying the marker that explains what went wrong.

`record` is on every result body, and says what the `job_id` is now worth to a read:
`closed` when the record states the outcome, `absent` when nothing was persisted
(`--no-persist`), `stale` when the write could not land and the record therefore
still reads `submitted`, which is the one state `status` reports as running. An
exit 6 carries both it and `persist`, the write that failed.

A rejected flag or an unconfigured deployment is refused *before* a job exists,
so stdout is empty and the stderr diagnostic is the entire result. Exit 2 spans
both sides of that line: a flag the CLI rejects prints nothing, while a request
LVS rejects has already minted a job and does print a marker. Emptiness, not the
exit code, is what tells them apart — check stdout is non-empty before reading a
marker from it. That line also settles what comes next: an empty stdout submitted
nothing, so a corrected call is still this request's one `POST /v1/summarize`,
while a marker means the post already happened and was refused.

| exit | meaning | what to do |
|---|---|---|
| 0 | summarized, and persisted unless `--no-persist` | report the result |
| 2 | invalid input the CLI refused; nothing was submitted | fix the call, then run once |
| 2 | invalid input LVS refused; the marker names the job | report it; the submission is spent |
| 3 | LVS or the memory store unreachable, or a 5xx | infrastructure; report it |
| 4 | not configured, or the deployment lacks a required service | `vss configure` |
| 5 | unknown `job_id` on a read verb | disambiguate |
| 6 | partial: summary in hand, persistence failed | keep the summary; never re-run the job |
| 7 | timeout: the marker names the job, which is not resumable | reconcile with `status`/`get` |

Exits 6 and 7 both mean the summarization must not be repeated.

## Reading records back

Pure reads against the memory index — they start no summarization:

```bash
vss summarize get --job-id summarize-01K...      # one complete record
vss summarize status --job-id summarize-01K...   # reconcile a pending job
vss summarize list --since 2026-01-01T00:00:00Z --status completed
```

`--since` takes an ISO-8601 instant, not a duration: `1h` exits 2. `list` also
filters by `--sensor-id`, and answers `[]` rather than failing when nothing has
been written yet. There is no `recall` verb: fetching one
record by id *is* `get`, and querying recent ones *is* `list`. Records are
`nv.vss.memory/1.0`, so `get` answers with the same schema every VSS memory
writer uses.

Never pass an index or endpoint the CLI did not record, and never read
Elasticsearch directly.

These verbs are for records this skill did not just write: reconciling a job whose
outcome a timeout left unknown, or picking up one an earlier session left behind.
A run that exited 0 already reported its write in the marker's `persist` object,
so reading it back proves nothing new — and recalling memory in its own right
belongs to the memory skill, not this one.
