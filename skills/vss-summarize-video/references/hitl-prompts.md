# Video Summarization — HITL Prompt Walkthroughs

### HITL: collect scenario and events first (REQUIRED — do not skip)

**Before any call to `POST /v1/summarize`, you MUST ask the user for
`scenario`, `events`, and `objects_of_interest`, and wait for their
response.** Do not call the video summarization service with defaults silently — if the user wants
defaults, they must say so explicitly (e.g., "use the generic
defaults").

You MAY reuse previously confirmed `scenario` / `events` /
`objects_of_interest` from earlier in the same chat **only if** the user
is asking to re-summarize the **same video** (same `streamId` / clip
URL) — in that case, remind the user which parameters you're about to
reuse and let them change them before calling. For any **different
video**, re-run the HITL from scratch.

Post the message as follows (literal template — fill the `{video_name}`
and `{duration}` placeholders):

> Before submitting **{video_name}** ({duration}s) to the video summarization service, please provide three
> parameters:
>
> 1. **`scenario`** — one-line context, e.g. `"warehouse monitoring"`,
>    `"traffic monitoring"`
> 2. **`events`** — a comma-separated list of events to surface, e.g.
>    `accident, pedestrian crossing`, `boxes falling, forklift stuck, accident`
> 3. **`objects_of_interest`** *(optional)* — things to track, e.g.
>    `cars, trucks, pedestrians` or `forklifts, pallets, workers`.
>    Leave blank if you don't want to specify any.
>
> Or reply `defaults` to use `scenario="activity monitoring"`,
> `events=["notable activity"]`, no objects. Reply `/cancel` to stop.

Only after the user replies with values (or `defaults`) may you build
and send the video summarization request.

**Required parameters:**

| Param | Type | Example |
|---|---|---|
| `scenario` | string (required) | `"activity monitoring"`, `"traffic monitoring"`, `"warehouse monitoring"` |
| `events` | list[string] (required) | `["notable activity"]`, `["accident", "pedestrian crossing"]` |
| `objects_of_interest` | list[string] (optional) | `["cars", "trucks", "pedestrians"]` |

If the user explicitly replies `defaults` to the HITL prompt above, use
`scenario="activity monitoring"` and `events=["notable activity"]`, and
mention in your response that you used generic defaults (offer to redo
with more specific parameters). **Do not apply defaults without that
explicit opt-in** — the HITL message is the gate.

**Defaults opt-in via the original query (autonomous mode).** When HITL
is bypassed (e.g. the caller said "run autonomously without prompting
for confirmation") and the original query contains the word `default`
or `defaults` for scenario/events, treat that as the same opt-in as a
HITL `defaults` reply: use `scenario="activity monitoring"` and
`events=["notable activity"]` **verbatim** - do not infer the scenario
from the video filename, sensor name, or any other context. In the
final reply, note that you used the generic defaults and offer to redo
with more specific parameters. The same rule applies if the original
query gives no scenario/events at all and HITL is bypassed - use the
canonical defaults rather than guessing.

**Request:**

The collected values become flags on `vss summarize run` (`SKILL.md` Stage 4,
`end-to-end-example.md`). The CLI issues the request, so no payload is built
here and no model is discovered: `vss configure` recorded both.

| HITL value | flag |
|---|---|
| scenario | `--scenario "<scenario>"` |
| each event | `--event "<event>"`, repeated |
| each object of interest | `--object-of-interest "<object>"`, repeated |

```bash
SUMMARIZE_OUT=/tmp/vss-summarize-video-run.json
vss summarize run \
  --url "<fresh_vios_clip_url_from_stage_2>" \
  --video-id "<resolved VIOS sensor id>" \
  --scenario "<scenario>" \
  --event "<event1>" --event "<event2>" \
  --creation-time "<media start, ISO-8601 UTC>" \
  --chunk-duration 10 --seed 1 > "$SUMMARIZE_OUT"
```

Execute exactly once. Do not repeat the run when it exits nonzero or the summary
is empty; read the saved stdout, the stderr diagnostic, and service logs
instead.

Omit `--object-of-interest` when the user provided none. Also omit frame
sampling flags in the standard workflow so RT-VLM uses the model-specific
deployment default; the deprecated `--num-frames-per-chunk` must not be used.

**Response shape:** the CLI nests LVS's OpenAI-style envelope under `summary`,
where `choices[0].message.content` is a **JSON string** — parse it to get the
actual summary and event list.

```bash
jq '{
  usage: (.summary.usage // {}),
  result: (.summary.choices[0].message.content | fromjson | {video_summary, events})
}' "$SUMMARIZE_OUT"
```

When both result fields are empty, report whether
`usage.total_chunks_processed` is positive. Zero or missing usage does not
prove that LVS processed the media; do not describe that case as "no events
detected."
