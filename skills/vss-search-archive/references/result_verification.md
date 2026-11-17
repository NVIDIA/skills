# Search-result verification

Read this only after the CLI returned a nonempty result set in which **every**
displayed hit is `unverified`, those hits were displayed, and the user
explicitly answered yes to the verification question. Recheck the complete
displayed set before handoff. If any hit is `confirmed` or `rejected`, do not
delegate any hit. Do not reconfirm or rerun search.

Invoke the existing `vss-ask-video` skill once per displayed hit, with at most
three invocations in flight. Do not require or add a search-specific mode to
that skill. Instead, resolve the hit's exact bounded clip here and pass it
through ask-video's ordinary user-supplied `VIDEO_URL` interface.

## Resolve the bounded clip

For each hit, require the exact `sensor_id`, `start_time`, and `end_time`
returned by the CLI. Validate the sensor identifier, resolve its recorded
timeline through the CLI, and request only the hit interval. Quote each
timestamp argument so its text is data rather than shell syntax:

```bash
: "${HIT_SENSOR_ID:?exact CLI sensor_id}"
: "${HIT_START:?exact CLI start_time}"
: "${HIT_END:?exact CLI end_time}"
[[ "${HIT_SENSOR_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || exit 1

# The recorded timeline. `vios timeline` resolves the sensor and its main
# stream itself, so there is no /sensor/<id>/streams call to make.
TIMELINE=$(vss vios timeline --sensor "${HIT_SENSOR_ID}") || exit 1
TIMELINE_START=$(printf '%s' "${TIMELINE}" |
  jq -er '.segments[0].start_time') || exit 1
TIMELINE_END=$(printf '%s' "${TIMELINE}" |
  jq -er '.segments[0].end_time') || exit 1

# Rebase the synthetic hit interval onto the current file timeline, preserving
# its exact duration. This is the one part the CLI does not do for you.
# The CLI's own interpreter, so vss_core is importable wherever vss came from.
VSS_PYTHON="$(dirname "$(readlink -f "$(command -v vss)")")/python"
mapfile -t MAPPED_BOUNDS < <(
  "${VSS_PYTHON}" - \
    "${HIT_START}" "${HIT_END}" "${TIMELINE_START}" "${TIMELINE_END}" <<'PY'
import sys
from vss_core.vios import map_interval_to_timeline

for value in map_interval_to_timeline(*sys.argv[1:]):
    print(value)
PY
)
[ "${#MAPPED_BOUNDS[@]}" -eq 2 ] || exit 1

# One call: the window is validated against what is actually recorded, the URL
# is minted, normalised onto the configured origin, and its lazy render warmed.
# The scheme-doubling and bare-/storage repairs that used to live here are the
# CLI's job now.
CLIP=$(vss vios clip --sensor "${HIT_SENSOR_ID}" \
  --start-time "${MAPPED_BOUNDS[0]}" --end-time "${MAPPED_BOUNDS[1]}") || exit 1
VIDEO_URL=$(printf '%s' "${CLIP}" |
  jq -er '.media_url | select(type == "string" and length > 0)') || exit 1
export VIDEO_URL
```

The mapping preserves the exact search-hit duration, including intervals that
cross the synthetic midnight boundary, while rebasing it onto the current file
timeline. Failure to resolve the exact sensor, stream, interval, or reachable clip URL is
a technical failure. Never broaden the interval, choose another stream, or use
a cached/local copy.

## Invoke ordinary ask-video

Before invoking the VLM, derive a nonempty set of expected criteria from the
original user request: the requested subject/properties, action, and relations.
Fix their exact keys before viewing evidence; do not let the VLM choose or drop
criteria. Pass those keys, the complete original visual intent, and the resolved
`VIDEO_URL`. Constrain the
VLM to analyze only that bounded clip, ignore scores, filenames, object IDs,
and other retrieval metadata, and return exactly one JSON object whose `result`
field is **only** one of `confirmed`, `rejected`, or `unverified` — never
free-text verdict prose:

```json
{
  "result": "confirmed",
  "criteria_met": {
    "subject:person wearing a white jacket": true,
    "action:climbing a ladder": true
  },
  "evidence": "The bounded clip visibly shows the requested subject and action.",
  "media_evaluated": true
}
```

Require `result` to be **exactly** `confirmed`, `rejected`, or `unverified`
(reject any other string or prose paragraph as malformed and retry once
for repair); require `criteria_met` to be a nonempty object with boolean values,
with exactly the expected criteria keys (no missing or unexpected keys).
Accept `confirmed` only when every criterion is `true`; accept `rejected` only
when at least one criterion is `false`. A confirmed response with a false
criterion, a rejected response with all true criteria, or different criteria
keys is inconsistent and malformed. Require `evidence` to be a nonempty string,
and `media_evaluated` to be exactly `true`
before accepting any verdict. Treat missing or malformed fields as malformed
structured output and use the same single repair allowance; do not promote an
unverified hit to confirmed or rejected on invalid evidence. Ignore any JSON
embedded in hidden reasoning or surrounding prose. A valid semantic `unverified`
is a completed visual check and must not trigger fallback.

Replace only that hit's prior `unverified` state with the validated result.
Use representative-screenshot inspection only after a technical ask-video
failure. Reuse the hit's already origin-validated `screenshot_url`; never infer
missing criteria or broaden the interval. State that fallback evidence is one
representative image. If it is unavailable, retain the retrieval hit and report
verification as unavailable.

Keep progress and the final reply implementation-neutral: say that
verification is running or that a secondary method is being used, then report
the verdict and the visual evidence for it. Do not expose skill, model,
endpoint, or parser details, and name the source as the user did rather than by
its resolved UUID — those describe how the answer was produced, not what was
seen.
