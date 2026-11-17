## End-to-end example

Use these implementations with the ordered stages in `SKILL.md`.

- [Resolve endpoints](#resolve-endpoints)
- [Probe readiness](#probe-readiness)
- [Prepare the video through VIOS](#prepare-the-video-through-vios)
- [Submit one summarize job](#submit-one-summarize-job)
- [Run the VLM fallback](#run-the-vlm-fallback)

Do not run a direct VLM fallback when LVS is ready, and do not rerun the
summarize job with broader events when the result is empty.

### Resolve endpoints

Run once before any probe. The service URLs are the ones `vss configure`
recorded (SKILL.md prerequisites) — the `/lvs` and `/rtvi-vlm` mounts on the
ingress origin, with **no** `/v1` suffix. Read them; do not rebuild them from
`VSS_PUBLIC_URL`, `HOST_IP`, a port, or leftover `LVS_BACKEND_URL` /
`VLM_BASE_URL` / `RTVI_VLM_BASE_URL`.

```bash
DEPLOYMENT=$(vss configure show) || exit $?
VIDEO_SUMMARIZATION_URL=$(printf '%s' "$DEPLOYMENT" | jq -r '.services.lvs.url // empty')
VLM=$(printf '%s' "$DEPLOYMENT" | jq -r '.services.rt_vlm.url // empty')
```

Only the readiness probe below uses these. The summarize request, the VIOS
preparation, and the VLM fallback take no endpoint: the CLI resolves them from
the same recorded deployment.

### Probe readiness

```bash
if [ -n "$VLM" ]; then
  vlm_code=$(curl -s -o /dev/null -w '%{http_code}' \
    --connect-timeout 3 --max-time 10 "$VLM/v1/models")
  [ "$vlm_code" = "200" ] || echo "VLM not reachable (HTTP $vlm_code)"
fi

# Readiness = HTTP 200 on /v1/ready. Body may be empty — do not inspect it.
# Retry on 503 (warmup) for up to ~30s before concluding the service is unavailable.
# No recorded lvs service is the same answer as not ready: take the VLM fallback.
video_sum_code=unrecorded
[ -n "$VIDEO_SUMMARIZATION_URL" ] && for i in $(seq 1 10); do
  video_sum_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 "$VIDEO_SUMMARIZATION_URL/v1/ready")
  case "$video_sum_code" in 200) break ;; 503) sleep 3 ;; *) break ;; esac
done

if [ "$video_sum_code" != "200" ]; then
  # Not a failure: LVS is not ready, so take the VLM fallback below without asking.
  echo "video summarization service not ready (HTTP $video_sum_code): use the VLM fallback" >&2
fi
```

### Prepare the video through VIOS

Use the `vss` CLI for every step; no VIOS REST calls and no `docker exec` /
`kubectl exec` probe. Replace `SOURCE_FILE` with the exact requested file; for
a named sensor, set `SENSOR_NAME` to it and start at `TIMELINE=`.

```bash
SOURCE_FILE=/path/to/video.mp4
FILENAME=$(basename "$SOURCE_FILE")
STEM="${FILENAME%.*}"   # VIOS names an uploaded sensor by its filename stem

LISTING=$(vss vios list --sensor "$STEM") || exit $?
if printf '%s' "$LISTING" | jq -e '.count > 0' >/dev/null; then
  SENSOR_NAME="$STEM"
else
  ADDED=$(vss vios add "$SOURCE_FILE") || exit $?
  SENSOR_NAME=$(printf '%s' "$ADDED" | jq -er '.name')
fi

# A window may not span a gap, and an RTSP sensor has no default window:
# clip each recorded segment with its own bounds, one summarize run per clip.
TIMELINE=$(vss vios timeline --sensor "$SENSOR_NAME") || exit $?
SEGMENT=$(printf '%s' "$TIMELINE" | jq -ec '.segments[0]')   # repeat per entry in .segments
CLIPPED=$(vss vios clip --sensor "$SENSOR_NAME" \
  --start-time "$(printf '%s' "$SEGMENT" | jq -r '.start_time')" \
  --end-time "$(printf '%s' "$SEGMENT" | jq -r '.end_time')") || exit $?
CLIP=$(printf '%s' "$CLIPPED" | jq -er '.media_url')
# A loopback CLI origin mints a loopback URL, which vss-lvs cannot fetch (and rejects).
if printf '%s' "$CLIP" | grep -Eq '^https?://(localhost|127\.0\.0\.1)[:/]'; then
  case "${HOST_IP:-}" in ""|localhost|127.*) HOST_ADDR=$(hostname -I | awk '{print $1}') ;; *) HOST_ADDR="$HOST_IP" ;; esac
  CLIP=$(printf '%s' "$CLIP" | sed -E "s#^(https?://)(localhost|127\.0\.0\.1)([:/])#\1${HOST_ADDR:?no host IP for the loopback clip URL}\3#")
fi
SENSOR_ID=$(printf '%s' "$CLIPPED" | jq -er '.sensor_id')
START_TIME=$(printf '%s' "$CLIPPED" | jq -er '.start_time')
[ "$(printf '%s' "$CLIPPED" | jq -r '.warmed')" = true ] || { echo "clip is cold: $CLIP" >&2; exit 1; }
```

### Submit one summarize job

Assume video preparation established `$CLIP`, and the SKILL.md prerequisites
put `vss` on `PATH` and ran `vss configure`. One `vss summarize run` is exactly
one `POST /v1/summarize`; the CLI resolves the LVS endpoint and the default
model from the recorded deployment, so no model discovery is needed here and no
payload is built by hand.

```bash
# HITL (required, before the run): collect the Stage 3 scenario/events and wait
# for the user's reply. Substitute their values (or the `defaults` opt-in) into
# $SCENARIO and the arrays below. Do not run the command without that reply.
SCENARIO='warehouse monitoring'            # or whatever the user gave
EVENTS=("notable activity")                # one array element per event
OBJECTS=()                                 # empty to omit

# The record's identity is the sensor, never the stream: `list --sensor-id` and
# time-windowed recall key on the sensor, so a summary filed under a stream id
# is one nothing can find again. Both preparation paths above resolved one.
VIDEO_ID="$SENSOR_ID"
[ -n "$VIDEO_ID" ] || {
  echo "no VIOS sensor id resolved; do not persist under a stream id"
  return 1 2>/dev/null || exit 1
}
# The clip's absolute start, from `vss vios clip` above -- never a constant.
# Without --creation-time event times are clip offsets memory cannot store.
CREATION_TIME="$START_TIME"
[ -n "$CREATION_TIME" ] || {
  echo "no VIOS timeline start resolved; event times would not be instants"
  return 1 2>/dev/null || exit 1
}
SUMMARIZE_OUT=/tmp/vss-summarize-video-run.json

SUMMARIZE_COMMAND=(
  vss summarize run
  --url "$CLIP"
  --video-id "$VIDEO_ID"
  --scenario "$SCENARIO"
  --creation-time "$CREATION_TIME"
  --chunk-duration 10
  --seed 1
)
for event in "${EVENTS[@]}"; do
  SUMMARIZE_COMMAND+=(--event "$event")
done
for object in "${OBJECTS[@]}"; do
  SUMMARIZE_COMMAND+=(--object-of-interest "$object")
done

# Exactly one run, ever. Keep stdout: its final line is the completion marker.
if "${SUMMARIZE_COMMAND[@]}" > "$SUMMARIZE_OUT"; then
  SUMMARIZE_EXIT=0
else
  SUMMARIZE_EXIT=$?
fi

# A call refused before a job is minted -- a rejected flag, no recorded
# deployment -- writes no marker, so the stderr diagnostic is the whole result
# and parsing stdout would replace it with a jq error. Test emptiness, not the
# exit code: an exit 2 from LVS rejecting the request does carry a marker, and
# the failure branch below reports it.
if [ ! -s "$SUMMARIZE_OUT" ]; then
  echo "no job was created (exit $SUMMARIZE_EXIT): fix the call or run vss configure, then run once"
  return 1 2>/dev/null || exit 1
fi
SUMMARIZE_RESULT=$(sed -n '1p' "$SUMMARIZE_OUT")
COMPLETION_MARKER=$(tail -1 "$SUMMARIZE_OUT")
JOB_ID=$(printf '%s\n' "$COMPLETION_MARKER" | jq -er '.job_id')
echo "exit=$SUMMARIZE_EXIT job=$JOB_ID"

# Only exits 0 and 6 carry a summary. Every other marker reports status,
# record, and error in its place, so it is the whole result: print it and stop
# rather than parsing a summary that is not there.
case "$SUMMARIZE_EXIT" in
  0) ;;
  6) echo "summary produced; an ES or Markdown write failed — present the result and both memory outcomes" ;;
  7)
    printf '%s\n' "$COMPLETION_MARKER" | jq -e '{job_id, status, persisted}'
    echo "timed out — reconcile once: ${VSS[*]} summarize get --job-id $JOB_ID"
    return 1 2>/dev/null || exit 1
    ;;
  *)
    printf '%s\n' "$COMPLETION_MARKER" | jq -e '{job_id, status, persisted}'
    echo "summarize failed (exit $SUMMARIZE_EXIT, job $JOB_ID)"
    echo "the marker means it was submitted: report this job, do not resubmit it"
    return 1 2>/dev/null || exit 1
    ;;
esac

# The LVS envelope is nested under .summary, otherwise unchanged.
printf '%s\n' "$SUMMARIZE_RESULT" | jq -e '{
  usage: (.summary.usage // {}),
  result: (.summary.choices[0].message.content | fromjson | {video_summary, events})
}'
```

The result and completion marker report separate memory outcomes, so nothing
needs to be read back:

```bash
# `.persist` is absent when static policy selected stdout-only execution.
printf '%s\n' "$SUMMARIZE_RESULT" | jq '{persist: (.persist // null), memory_note: (.memory_note // null)}'
printf '%s\n' "$COMPLETION_MARKER" | jq '{job_id, status, persisted, exit_hint}'
```

For any failure, inspect `$SUMMARIZE_OUT`, the stderr diagnostic, and service
logs. Never repeat the run to obtain a different view of the result — exits 6
and 7 mean the summarization already happened.

If both result fields are empty, use `summary.usage.total_chunks_processed` from
the same payload to report whether LVS processed any media. Do not infer "no
detections" when that value is zero or missing.

### Run the VLM fallback

Run this when LVS is not ready; do not ask first. `vss vlm run` resolves the
clip and the model itself, so there is nothing to discover by hand. One call
per recorded segment, since a window may not span a gap between segments:

```bash
PROMPT='Describe in detail what is happening in this video,
including all visible people, vehicles, equipment, objects,
actions, and environmental conditions.
OUTPUT REQUIREMENTS:
[timestamp-timestamp] Description of what is happening.'

TIMELINE=$(vss vios timeline --sensor "$SENSOR_NAME") || exit $?
printf '%s\n' "$TIMELINE" | jq -c '.segments[]' | while read -r SEGMENT; do
  vss vlm run --sensor "$SENSOR_NAME" --prompt "$PROMPT" \
    --start-time "$(printf '%s' "$SEGMENT" | jq -r '.start_time')" \
    --end-time "$(printf '%s' "$SEGMENT" | jq -r '.end_time')" || exit $?
done
```
