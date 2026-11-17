---
name: vss-summarize-video
description: Use when summarizing a recorded video through HITL-gated LVS, falling back to `vss vlm run` when LVS is not ready. Not for reports, archive search, or live RTSP captioning.
license: Apache-2.0
metadata:
  version: "3.3.0"
  author: "NVIDIA Video Search and Summarization team"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational"
  # What a live deployment must expose for this skill to be usable, as the vss CLI
  # names it: a command group (search, summarize, vlm, vios, memory), "alerts"
  # (Alert Bridge), or "always" for a skill every VSS deployment gets. The
  # OpenClaw harness image ships and activates skills by it. `a|b` = either one.
  vss-requires: "summarize|vlm"
---

# VSS Summarize Video

## Instructions

- Before readiness checks or video preparation, reject summarization of any
  stream (RTSP/RTSPS URL or registered camera), including recorded time windows:
  **Live-stream summarization / report generation isn't supported.** Then stop;
  do not ingest, extract clips, invoke inference/fallback, or deploy captioning.
  Reject an explicitly identified stream/camera even if its URL is absent.
  For a registered sensor name/id (not a local file path, uploaded file, or
  direct recorded-video URL), first check its current type with only
  `vss vios list --sensor <name>`; never reuse an earlier file classification.
  Reject a `stream`; if the type is unclear or the lookup fails, report that
  limitation and stop. Local/uploaded files follow the recorded-video workflow,
  including Stage 2 registration when absent; do not classify their paths as sensors.
- Execute the five workflow stages below in order.
- Run API commands yourself; do not tell the user to run them.
- Use the required references at their named decision points.

## Examples

Runnable scenarios live under `evals/`. The command implementations are in
[`references/end-to-end-example.md`](references/end-to-end-example.md).

## When to Use

Use when summarizing a recorded video through HITL-gated LVS, to produce one
polished narrative summary with timestamped events when LVS is available.

Do not use this skill for:

- Live RTSP captioning: use `vss-deploy-dense-captioning`.
- Incident or alert-window reports: use `vss-generate-video-report` Mode B.
- Archive search: use `vss-search-archive`.

## Required References

Load these files only as directed:

- [`references/end-to-end-example.md`](references/end-to-end-example.md): load
  before executing the recorded-video workflow. It contains the exact
  readiness, VIOS preparation, single-run summarize, and VLM fallback commands.
- [`references/cli_usage.md`](references/cli_usage.md): load before Stage 4.
  `vss summarize run` issues the summarize request and applies the operator's
  configured memory policy; this reference has its flags, exit codes, output
  shape, and read verbs.
- [`references/video-summarization-api.md`](references/video-summarization-api.md):
  load before constructing a live LVS operation **by hand** — a direct API
  question. Follow its **Runtime OpenAPI
  Discovery** procedure: the LVS schema is `/openapi.json` under
  `services.lvs.url`; the origin's own `/openapi.json` is the Agent's. The ordered
  workflow does not build a summarize payload; the CLI owns that.
- [`references/hitl-prompts.md`](references/hitl-prompts.md): load when
  collecting LVS scenario, events, and optional objects of interest.
- [`references/video-summarization-debugging.md`](references/video-summarization-debugging.md):
  load only when diagnosing a failed or empty response.
- [`references/video-summarization-deployment.md`](references/video-summarization-deployment.md):
  load only for deployment, configuration, logs, or service operations.
- [`references/video-summarization-environment-variables.md`](references/video-summarization-environment-variables.md)
  and `assets/video-summarization.env.example`: use when configuring the
  service environment.
- [`../vss-build-vision-ai/references/deployment_resolution.md`](../../vss-build-vision-ai/references/deployment_resolution.md):
  Kubernetes `VSS_PUBLIC_URL` contract and the `/lvs` mount, for deployment
  questions. The workflow itself reads its service URLs from `vss configure show`.
- [`references/deploy-lvs-service.md`](references/deploy-lvs-service.md): load
  when asked about LVS's own container image, GPU/CPU/storage sizing, or
  deployment contract as a peer service (heavier than
  `video-summarization-deployment.md`, which covers operating an already
  running deployment).
- [`references/integrate-lvs-service.md`](references/integrate-lvs-service.md):
  load when another agent or skill needs to integrate with LVS as a peer
  service — required peers, integration interfaces, API schema, and network
  requirements.

## Core Invariants

- Route by LVS readiness, never by video duration.
- HTTP 200 from `/v1/ready` selects LVS. Empty response bodies do not mean
  unavailable.
- Once LVS is selected, do not call a VLM `/v1/chat/completions` endpoint.
- Issue exactly one `vss summarize run` per recorded segment. One run is
  one `POST /v1/summarize`. Never retry, hedge, broaden events, or run a second
  backend automatically.
- Endpoints come from the deployment `vss configure` recorded. Never pass an
  endpoint, index, or model flag the caller did not name, and never replace a
  failed run with hand-rolled curl against `/v1/summarize`.
- Save the complete command and its stdout. Diagnose failures from those files,
  the run's own exit code, service logs, and non-mutating GET requests.
- Render `video_summary` and every returned event verbatim. Do not paraphrase,
  truncate descriptions, add fields, or fabricate `id`.
- When LVS is not ready, fall back to `vss vlm run` directly. Do not ask
  first, and do not offer to deploy LVS.

## Prerequisites

- The `lvs` profile, reachable through the origin recorded by `vss configure`.
- `curl` for the readiness probe only, and `jq` for reading CLI JSON. Capture
  stdout before piping it, or use `set -o pipefail`. Exit codes and the common
  CLI rules live in the repository root [`AGENTS.md`](../../../AGENTS.md).
- Network reachability from the LVS service to the final VIOS clip URL (Docker:
  from `vss-lvs`; Kubernetes: deploy must mint a URL the LVS pod can fetch).
- The `vss` CLI on `PATH`. The OpenClaw and Hermes harness images ship it; anywhere else, install it from the same checkout as this skill so the CLI and the skill match: `uv tool install <checkout>/libs/vss/cli`.
- One recorded deployment origin:

```bash
vss summarize run --help >/dev/null || exit 1
vss configure show
```

`vss configure show` fails when nothing is recorded. Then the only setup is
`vss configure --base-url "${VSS_PUBLIC_URL}"`, with the ingress origin the
operator gave you. If `VSS_PUBLIC_URL` is unset, stop and ask for that origin;
do not substitute `HOST_IP`, `localhost`, or a port.

Configure against the ingress origin, never `:38111` — that LVS container port
exposes no Elasticsearch, so a deployment recorded from it cannot persist.

The `vss-build-vision-ai` skill can deploy the profile.

## Limitations

- Direct VLM fallback cannot target LVS scenarios or events and is lower
  quality.
- Private VIOS URLs may be unreachable from remote VLM endpoints.
- One `vss summarize run` per recorded segment, with no automatic retry.
- Persistence needs a routed Elasticsearch. A deployment without one summarizes
  and reports the result unpersisted rather than failing the job.
- Both edges are configured to wait an hour, matching the CLI's own default, so
  a long summarization is not cut short by a 504 that would be recorded as a
  failed job. An Ingress the deployment overrides shorter still caps the wait.

## Recorded services

Every URL the recorded-video workflow and a direct API question touch is one
`vss configure` recorded. Read it from `vss configure show`; never assemble it
from `VSS_PUBLIC_URL`, `HOST_IP`, or a port (the one exception is the Stage 2
rewrite of a loopback `media_url` host to the host's routable IP), and ignore leftover `LVS_BACKEND_URL` / `VLM_BASE_URL` /
`RTVI_VLM_BASE_URL`. Do not use `kubectl port-forward`, Service DNS, NodePorts,
`docker exec`, or `docker inspect`, and do not scan ports or configuration
files for an endpoint.

| Service | Recorded as | Called by the workflow |
|---|---|---|
| LVS | `services.lvs.url` (the `/lvs` mount) | only `GET …/v1/ready`, the readiness probe; `vss summarize run` resolves the rest |
| VLM / RT-VLM | `services.rt_vlm.url` (the `/rtvi-vlm` mount) | only `GET …/v1/models`, a reachability check in the same probe; `vss vlm run` resolves the rest |

The readiness probe is the only direct HTTP the workflow makes, and it exists because the CLI
has no readiness verb: `vss configure` records LVS on liveness (`/lvs/v1/live`),
while routing needs HTTP 200 from `/v1/ready`. It is not a fallback. When a
`vss` command fails, report that failure; never repeat the work with curl
against `/v1/summarize` or `/v1/chat/completions`.

Do not treat the origin's `/openapi.json` as the LVS schema; on stock Ingress
that path is the Agent's.

## Routing

Probe LVS `/v1/ready` using the loop in the end-to-end reference. Readiness is
the HTTP status only: retry 503 warmup responses for about 30 seconds, and do
not inspect the body. No recorded `lvs` service counts as not ready.

| LVS result | Action |
|---|---|
| HTTP 200 | Use LVS for every video duration. |
| Anything else | Use the VLM fallback (`vss vlm run`) without asking. |

## Recorded Video Workflow

### Stage 1: Select the Backend

Load the end-to-end and CLI references. Run the LVS readiness probe before
preparing the clip.

The summarization model needs no discovery: `vss configure` recorded the id LVS
reports serving, and `vss summarize run` defaults to it on both Docker and
Kubernetes. The VLM fallback needs none either: `vss vlm run` defaults to the
model the deployment's RT-VLM reports. Pass `--model` only when the caller
named one, and read the recorded value from `vss configure show` when it has to
be reported.

A non-200 LVS readiness result after warmup is the only unavailability signal.
An empty summary, empty events, missing optional fields, or empty readiness
stdout must not trigger fallback.

### Stage 2: Prepare the Video Through VIOS

Use the `vss` CLI for every step; no VIOS REST calls, and do not invoke a
separate skill.

1. A named sensor goes straight to step 3. For a file, `vss vios list --sensor
   <stem>`; reuse the recording when present. VIOS names an uploaded sensor by
   its filename stem.
2. If absent and the exact local file is available, `vss vios add <file>`. It
   waits for the timeline; its default timestamp is `2025-01-01T00:00:00.000Z`.
3. `vss vios timeline --sensor <name>`, then for each segment `vss vios clip
   --sensor <name> --start-time <start> --end-time <end>`. Always pass the
   segment's bounds: a window may not span a gap, and an RTSP sensor has no
   default window. Pass `media_url` to `--url` as returned, except that a
   `localhost` / `127.0.0.1` host becomes the host's routable IP: `vss-lvs`
   cannot fetch loopback and rejects it.
4. If `warmed` is `false`, stop and report it. `warmed: true` shows only that
   the CLI host fetched the URL, not that LVS can.

Require the exact recording, full timeline, and fresh clip URL before
continuing. When the source file is available, compare VIOS timeline duration
with source duration.

If preparation fails, stop and report the missing prerequisite. Do not choose
an arbitrary `/tmp` video, alternate recording, local HTTP server, NvStreamer,
or RTSP source unless the user explicitly requested that source.

### Stage 3: Collect LVS Settings

When LVS is selected, load the HITL reference and collect `scenario`, `events`,
and optional `objects_of_interest` before the summarize run.

When the caller explicitly says to run autonomously without prompting and asks
for defaults or supplies no settings, use these values verbatim:

```text
scenario="activity monitoring"
events=["notable activity"]
```

This is the only HITL bypass. Do not infer defaults from filenames or sensor
names. Mention defaults in the final response and offer a separate rerun with
specific settings.

### Stage 4: Submit Once Through the CLI

Load the CLI reference. `vss summarize run` issues the summarize request on both
Docker and Kubernetes, and persists only when static memory policy enables it.
Do not build a `/v1/summarize` payload by hand, and do not fetch
`/openapi.json` to construct one — the CLI owns the request shape,
`vss configure` owns the endpoint, and `vss configure memory` owns persistence.

Use the invocation in the end-to-end reference. It passes the fresh VIOS URL
from Stage 2, the exact HITL values from Stage 3, `--chunk-duration 10`, and
`--seed 1`; repeat `--event` per event and add `--object-of-interest` only when
the caller provided objects. Pass no endpoint flag.

Do not pass `--persist` or `--memory-index`; those per-request controls do not
exist. The standard workflow also does not pass `--no-persist`, so the
operator's configured persistence default applies. When persistence is enabled,
the record needs two values:

- `--video-id`, required alongside `--url`. Use the recording's VIOS **sensor**
  id — the `sensor_id` from Stage 2's `vss vios clip` output — never the
  stream id. It becomes the record's sensor, which is what `list --sensor-id`
  and time-windowed recall key on. Without `--video-id` the run exits 2 before
  summarizing rather than after.
- `--creation-time`, the media's absolute start. LVS reports event times as
  offsets into the clip unless this anchors them, and unified memory stores
  instants — so without it the events cannot be written and the run degrades to
  exit 6. For uploaded sample media use the same `2025-01-01T00:00:00.000Z`
  Stage 2 uploaded with.

Do not pass `--num-frames-per-chunk` in the standard workflow. RT-VLM owns frame
sampling; unset fields are absent from the request, so the deployment's own
default applies.

The final line of stdout is one JSON object naming the job. Read that line and
the exit code; the prose on stderr is a diagnostic, not the result. A call
refused before a job exists prints no marker at all and stderr is the whole
result — check stdout is non-empty before parsing it. Emptiness, not the exit
code, is what says whether a job was created.

| exit | meaning | action |
|---|---|---|
| 0 | summarized; persistence followed configured policy | present the result and report `persisted` truthfully |
| 2 | a flag the CLI refused, before anything was submitted | fix the call, then run once |
| 2 | LVS rejected the request it was sent; the marker names a job closed as failed | report the failure with that `job_id` |
| 3 | LVS unreachable or returned 5xx | report it with the marker's `job_id` |
| 4 | deployment configuration is missing, or an explicit Markdown note lacks static sink configuration | no job, no marker — run the remediation command from stderr |
| 6 | summary produced; Elasticsearch or Markdown cache write failed | present the summary; report ES and Markdown outcomes separately |
| 7 | timed out | reconcile with `vss summarize get --job-id`; do not re-run |

Exits 6 and 7 both mean the summarization already happened. Never repeat the run
to obtain a different view of it, and never repeat it for diagnosis — a second
run requires a separate user request. The exit 2 that carries a marker is the
same story earlier: the request reached LVS and came back refused, so the one
submission this request had is spent and a corrected call belongs to a new
request. Once a job exists, every outcome names its `job_id`; use it rather than
re-running. A call refused before a job exists names
nothing, which is why empty stdout is the test.

The completion marker's `persisted` boolean is the authoritative Elasticsearch
outcome. The result body includes `persist` only when persistence was attempted
and reports its index and event count; optional Markdown status is separate
under `memory_note`. `record` says what the `job_id` is worth to a later read:
`closed`, `absent` when policy skipped persistence, or `stale` when a submitted
record could not be closed. Do not read the record back to confirm it, and never
read Elasticsearch directly — recalling memory is a separate skill's job. The
one read that belongs here is reconciling an exit 7, whose outcome is genuinely
unknown until `vss summarize get --job-id <job_id>` answers.

If `video_summary` and `events` are empty, inspect the same payload's
`summary.usage.total_chunks_processed`. A positive integer confirms processing;
zero or missing means processing was not confirmed. Do not claim "no
detections."

### VLM Fallback for Stages 3-4

Use the fallback when LVS remained unavailable after warmup; do not ask first.
Do not run LVS HITL, and never use fallback to repair or replace an LVS
response. Run one `vss vlm run --sensor <name> --start-time <start> --end-time
<end>` per recorded segment from `vss vios timeline --sensor <name>`, with the
default prompt in the end-to-end reference. The CLI resolves the clip and the
model itself; do not call `/v1/chat/completions` by hand.

Before the result, include:

> **Note:** Input video `<name>` is `<N>`s long. The video summarization
> service is not deployed, so this summary was produced by the VLM alone with
> a generic default prompt. Deploy the `lvs` profile for higher-quality
> summaries with scenario/events targeting.

A non-zero `vss vlm run` exit is the result to report; do not retry it.

### Stage 5: Present the Result

Start with exactly one header:

```text
Summary of <video_name> (<duration>)
```

Use `Ns` below 60 seconds and `Mm Ss` otherwise.

For LVS, the CLI nests the service's own envelope under `summary`: parse the
JSON string in `summary.choices[0].message.content` while preserving
`summary.usage`. Render `video_summary` verbatim, followed by every event in
service order. Preserve every returned field and the full `description`; use a
per-event list if a table would truncate text.

Close with the job's identity: the `job_id`, the completion marker's
`persisted` value, and any separate `memory_note` result. An absent `persist`
object with `persisted=false` means static policy chose stdout-only execution,
not a failure.

For VLM, render `choices[0].message.content` verbatim. For Cosmos output, omit
the `<think>...</think>` block and show the answer. Do not add emojis or
re-voice either backend's content.

## Troubleshooting

| Symptom | Action |
|---|---|
| `/v1/ready` remains 503 | Treat LVS as unavailable after the warmup loop. |
| Readiness stdout is empty | Use the HTTP status; a 200 body may be empty. |
| Summary and events are empty | Inspect saved `summary.usage.total_chunks_processed`; do not retry. |
| `vss` not found | Install it from this skill's checkout (`uv tool install <checkout>/libs/vss/cli`), or report the image problem. |
| Run exits 4 | Follow stderr: configure the deployment, or configure the Markdown sink requested explicitly. |
| Run exits 6 | A post-operation memory write failed. Present the summary and separate ES/Markdown status; do not re-run. |
| Run exits 7 | Timed out. `vss summarize get --job-id <id>`; do not re-run. |
| VLM returns `<think>` | Remove reasoning through `</think>` when rendering. |
| K8s `/openapi.json` looks like Agent | Expected — do not use it as LVS schema. |
| `/models` 404 / HTML | Probing the bare origin — use `services.lvs.url` + `/models` or `services.rt_vlm.url` + `/v1/models` from `vss configure show`. |

Use the debugging reference for deeper diagnostics and the deployment
reference for logs or configuration. `VSS_VIDEO_SUMMARIZATION_TAG` and
`VSS_RT_VLM_TAG` both default to `3.3.0` on x86/Jetson Thor; use the
`3.3.0-sbsa` variant on SBSA/DGX Spark/Grace.

## Direct API and Service Operations

For direct API questions such as models, readiness, recommended configuration,
metrics, schemas, or 422 responses, use the API reference instead of the
recorded-video workflow, with `services.lvs.url` from `vss configure show` as
the base. `/lvs` is a Prefix mount, so everything LVS serves is public under it
— `/v1/ready`, `/v1/summarize`, `/models`, `/metrics` — where the previous
Exact-path Ingress published only readiness and summarize. A direct API
question is never a substitute for a failed `vss` command. For deployment,
restart, teardown, backend selection, or service logs, prefer
`vss-build-vision-ai` and use the deployment reference.

## Cross-reference

- `vss-build-vision-ai`: deploy the `lvs` profile.
- `vss-manage-video-io-storage`: general VIOS administration outside this
  ordered workflow.
- `vss-search-archive`: search archived video.
- `vss-query-analytics`: query stored incidents and events.
- `vss-generate-video-report` and `vss-ask-video`: hand off here for recordings of 120 s or longer.
