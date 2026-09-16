# Prompt Patterns

Use these patterns when authoring PAIDF auto-labeling prompts. Prompts should
improve evidence quality. They should not duplicate DAFT schema-writing logic
unless the user or a specific captioning/VQA service contract requires a
structured response.

## Universal Prompt Contract

Every production prompt should specify:

- Role and input scope: one image, one video window, whole clip, or text-only
  aggregation.
- Response style: plain prose by default; structured output only when explicitly
  requested.
- Evidence policy: visible evidence only; no hidden causes, no intent, no fault,
  no identities, no protected attributes.
- Uncertainty policy: say unclear/unknown when evidence is weak.
- Compactness: one or two sentences for captions unless the task asks for more.
- Grounding: timestamps for video, visible track IDs only when attached to the
  subject, and no invented IDs.
- Parser compatibility: when structured captioning is requested, include fields
  the captioning parser can extract, such as `caption`, `description`,
  `scene_description`, `event_summary`, or `chunks`.

## Image Caption Prompt

Use for `--image-prompt-file`. Images have no time axis.

Required qualities:

- State that the input is one still image.
- Ask for visible objects, layout, activity implied by the still frame, quality,
  and readable text.
- Forbid motion, future/past events, identity, private attributes, and hidden
  context.
- Return concise prose unless the user requests a specific structured captioning
  format.

## Video Dense Caption Prompt

Use for `--prompt-file`. The captioning service appends window start/end times
to the prompt, so the model should reason within that window.

Required qualities:

- Separate scene layout from event/activity evidence in the wording, even if the
  final response is plain prose.
- Ask the model to compare early/middle/late frames for temporal changes.
- Distinguish static conditions from observed transitions.
- Mention track IDs only when visible overlays are readable and attached to the
  described subject.
- Prefer uncertainty over forced verdicts.

If structured captioning is explicitly requested, keep it lightweight and focused
on evidence fields. The captioning parser extracts `event_summary` first, then
`caption`, `description`, `scene_description`, and finally
`chunks[].description`.

## VQA Evidence Prompt

Use VQA prompts and question banks for answerable visual evidence, not for broad
captioning. Good VQA output should include:

- Direct answer with an explicit unknown/not-visible option when evidence is
  missing.
- Short visual evidence phrase with timestamp/window reference when available.
- Confidence or visibility gating when downstream schema supports it.
- No forced yes/no answer when the evidence is occluded or outside the frame.

## LLM Evidence Aggregation Prompt

Use aggregation prompts to turn accumulated captions, tracks, and VQA evidence
into concise summaries for downstream services. Do not ask the prompt to write
DAFT schemas unless the `reasoning` stage contract explicitly requires it (DAFT
`task/` writing belongs to `reasoning`; the single `daft_export` stage is
retired).

For video event evidence:

- Preserve start/end time evidence from captions and VQA outputs.
- Use open-vocabulary, short labels only as evidence tags when useful.
- Include involved visible IDs only when upstream evidence grounded them.
- Avoid infrastructure, signal, lane, victim, responder, or causal claims unless
  directly visible in the evidence.

## Quality Checklist

Before committing a prompt, verify:

- The first paragraph tells the model exactly what media/context it receives.
- The requested response is concise and appropriate for the consuming service.
- The prompt separates visible observations from labels or verdicts.
- It has anti-fabrication rules specific to the domain.
- It tells the model how to handle unclear or empty inputs.
- It does not leak paths, endpoint names, secrets, or one-off experiment details.
