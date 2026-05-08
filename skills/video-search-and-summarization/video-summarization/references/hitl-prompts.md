# Video Summarization — HITL Prompt Walkthroughs

### HITL: confirm the VLM prompt first (REQUIRED — do not skip)

**Before any call to the VLM, you MUST show the default prompt to the
user verbatim and wait for their response.** Do not proceed on silence
and do not assume defaults.

You MAY reuse a confirmed prompt from earlier in the same chat **only
if** the user is asking to re-summarize the **same video** (same
`streamId` / clip URL) — in that case, remind the user what prompt
you're about to reuse and offer them the chance to change it before
calling. For any **different video**, re-run the HITL from scratch.

Post the message as follows (literal template — fill the `{video_name}`
placeholder):

> I'm about to summarize **{video_name}** with this VLM prompt. Reply
> `Submit` to use it as-is, paste replacement text, `/generate <desc>`
> to rewrite it from a description, `/refine <instr>` to tweak it, or
> `/cancel` to stop.
>
> ```
> <default VLM prompt below>
> ```

**Default VLM prompt** (copy verbatim from the base profile):

```
Describe in detail what is happening in this video,
including all visible people, vehicles, equipments, objects,
actions, and environmental conditions.
OUTPUT REQUIREMENTS:
[timestamp-timestamp] Description of what is happening.
EXAMPLE:
[0.0s-4.0s] <description of the first event>
[4.0s-12.0s] <description of the second event>
```

**User response handling:**

| User input | Effect |
|---|---|
| `Submit` (or empty) | Approve the current prompt and call the VLM |
| Any other free text | Treat as a full replacement prompt; echo it back and ask for `Submit` before calling |
| `/generate <description>` | You (the assistant) write a new prompt from the description, show it back, and wait for `Submit` |
| `/refine <instructions>` | You (the assistant) refine the current prompt per the instructions, show it back, and wait for `Submit` |
| `/cancel` | Cancel summarization |

Rules:

- You MAY call the VLM **only** after receiving `Submit` (or an empty
  confirmation) on a prompt that is currently visible in the chat.
- `/generate` and `/refine` are not terminal — they produce a new prompt
  that itself needs `Submit`.
- When handling `/generate` and `/refine`, preserve the
  `[Xs-Ys] <description>` output-format requirement from the default
  prompt.
- If the user just says "go" / "ok" / "yes" without having seen the
  prompt, show the prompt first, then wait for `Submit`.


### HITL: collect scenario and events first (REQUIRED — do not skip)

**Before any call to `POST /summarize`, you MUST ask the user for
`scenario`, `events`, and `objects_of_interest`, and wait for their
response.** Do not call LVS with defaults silently — if the user wants
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

> I'm about to send **{video_name}** ({duration}s) to LVS. I need three
> parameters first:
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
and send the LVS request.

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

**Request:**

```bash
curl -s -X POST "${LVS_BACKEND_URL:-http://localhost:38111}/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "<clip_url_from_vios>",
    "model": "'"${VLM_NAME:-nvidia/cosmos-reason2-8b}"'",
    "scenario": "<scenario>",
    "events": ["<event1>", "<event2>"],
    "chunk_duration": 10,
    "num_frames_per_chunk": 20,
    "seed": 1
  }' | jq .
```

Omit `objects_of_interest` if the user did not provide any. Include it as a
JSON array otherwise.

**Response shape:** OpenAI-style envelope. `choices[0].message.content` is a
**JSON string** — parse it to get the actual summary and event list.

```bash
