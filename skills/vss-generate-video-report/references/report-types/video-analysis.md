# Video analysis report (Mode A)

Numbered steps for Mode A, loaded on demand from [`SKILL.md`](../../SKILL.md) (`$SKILL_DIR/SKILL.md`); routing, the shared setup and the generic prerequisite probes live there (mode-specific gates are the steps flagged below) and the skeleton is defined in `SKILL.md` § Report types.

- **Trigger phrases:** `SKILL.md` § Examples — the Mode A rows.
- **Prerequisite gate:** `SKILL.md` § Runtime prerequisites — the *Mode-by-mode checklist* row(s) for Mode A and the long-video hard gate; apply `SKILL.md` § HITL prompt mode before Step 3.
- **Inputs to resolve:** Step 1 below.
- **Template:** Step 4 below.
- **Output contract:** `SKILL.md` § Instructions → *Output contract for evaluators*, the Mode A lines.
- **Failure modes:** `SKILL.md` § Error Handling, plus the rules stated in the steps below.

---

## Mode A — Report on a recorded video clip

Mode A routes by clip duration, not by LVS readiness. Step 1 measures the clip:
under 120 s, run Steps 1–3 (direct VLM). At 120 s or longer (the Long-video
rule), run `/vss-summarize-video`, which uses LVS when ready and `vss vlm run`
otherwise, then paste its output into the Step 4 template and skip Steps 2–3.
A base64 input is decoded to a file first (`base64 -d "$VIDEO_B64_FILE" >
<file>`) and handed off by that path.
That hand-off has no Step 3 prompt-approval loop: when HITL resolved `false` (or the caller asked for autonomous execution) invoke `/vss-summarize-video` with its explicit autonomous instruction and defaults (`scenario="activity monitoring"`, `events=["notable activity"]`) and state those defaults in the chat response; when HITL resolved `true`, its settings dialogue replaces the Step 3 approval.

### Step 1 — Resolve Mode A input (A1 clip URL or A2 local-file/base64)

Choose one path:

#### A1 — VST clip URL path

Hand off to `/vss-manage-video-io-storage` to:

1. List sensors and confirm the named `<sensor-id>` exists (upload first if not).
2. Fetch `/storage/<streamId>/timelines` for the recorded range when the user did not supply `startTime` / `endTime`.
3. Request a clip URL:

   ```bash
   # Resolves the sensor by name, mints the clip URL, normalises it, and warms the render.
   # Omit the window to take the whole recorded segment; the response echoes what it resolved.
   # `vss` on PATH (SKILL.md Runtime prerequisites); exit codes: AGENTS.md at the repo root
   # Fresh shell: paste the Endpoint resolution hand-off (SKILL.md) at the top of this block.
   case "${DEPLOYMENT_KIND:?paste the Endpoint resolution output at the top of this block}" in
     kubernetes) VSS_ORIGIN="${VSS_PUBLIC_URL:?kubernetes hand-off lacks VSS_PUBLIC_URL — re-run Endpoint resolution}" ;;
     docker)     VSS_ORIGIN="http://${HOST_IP:?docker hand-off lacks HOST_IP — re-run Endpoint resolution}:7777" ;;   # HAProxy host port
     *) echo "ERROR: DEPLOYMENT_KIND must be kubernetes or docker, got '${DEPLOYMENT_KIND}'" >&2; exit 1 ;;
   esac
   vss configure --base-url "${VSS_ORIGIN%/}" \
     || { echo "vss configure failed for ${VSS_ORIGIN} — is the VSS origin reachable?" >&2; exit 1; }   # once per deployment

   # Captured, not piped: `vss ... | jq` hides the CLI's exit code behind jq's,
   # so a failed command with empty stdout reads as an empty answer.
   CLIP=$(vss vios clip --sensor <sensor-name> [--start-time <startTime> --end-time <endTime>]) || {
     echo "vss vios clip failed for <sensor-name>" >&2; exit 1; }
   VIDEO_URL=$(printf '%s' "${CLIP}" | jq -er '.media_url | select(type=="string" and length>0)') \
     || { echo "vss vios clip returned no media_url for <sensor-name>" >&2; printf '%s\n' "${CLIP}" >&2; exit 1; }
   # Hand-off (shell-quoted, paste at the top of the Step 3 block and keep for Step 4): the clip URL and the
   # window the CLI resolved (the whole recorded segment when none was given).
   CLIP_START=$(printf '%s' "${CLIP}" | jq -r '.start_time // empty'); CLIP_END=$(printf '%s' "${CLIP}" | jq -r '.end_time // empty')
   [ -n "${CLIP_START}" ] && [ -n "${CLIP_END}" ] || { echo "vss vios clip returned no start_time / end_time for <sensor-name>" >&2; printf '%s\n' "${CLIP}" >&2; exit 1; }
   # Duration in seconds (fraction kept, ms precision) from the resolved window: epoch integer via GNU date -d or
   # BSD date -j, the ISO fraction re-attached. It feeds the Long-video rule and Step 3's frame sampling (Step 3
   # truncates for num_frames only) — never leave it to a default.
   _iso2s() { _f=$(printf '%s' "$1" | sed -nE 's/^[^.]*T[0-9:]+(\.[0-9]+).*$/\1/p'); _t=$(printf '%s' "$1" | sed -E 's/(\.[0-9]+)?(Z|\+00:00)$/Z/'); _i=$(date -u -d "$_t" +%s 2>/dev/null || date -j -u -f '%Y-%m-%dT%H:%M:%SZ' "$_t" +%s 2>/dev/null) && printf '%s%s' "$_i" "$_f"; }
   _s=$(_iso2s "${CLIP_START}"); _e=$(_iso2s "${CLIP_END}")
   for _v in "${_s:-x}" "${_e:-x}"; do printf '%s' "$_v" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "ERROR: could not parse the resolved window ${CLIP_START} / ${CLIP_END} as ISO-8601 UTC" >&2; exit 1; }; done
   # LC_ALL=C: awk formats and parses numbers per locale; a decimal comma would break the checks below and Step 3.
   CLIP_SECONDS=$(LC_ALL=C awk -v s="$_s" -v e="$_e" 'BEGIN{d=sprintf("%.3f", e-s); if (d ~ /\./) { sub(/0+$/,"",d); sub(/\.$/,"",d) }; print d}')
   case "$CLIP_SECONDS" in -*|0) echo "ERROR: resolved window is empty or reversed (${CLIP_START} -> ${CLIP_END})" >&2; exit 1 ;; esac
   LC_ALL=C awk -v d="$CLIP_SECONDS" 'BEGIN{exit !(d+0 < 120)}' || { echo "Clip is ${CLIP_SECONDS} s (120 s or longer): hand off to /vss-summarize-video (Long-video rule)" >&2; exit 1; }
   printf 'VIDEO_URL=%q\nCLIP_START=%q\nCLIP_END=%q\nCLIP_SECONDS=%q\n' "${VIDEO_URL}" "${CLIP_START}" "${CLIP_END}" "${CLIP_SECONDS}"
   ```

The block prints shell-quoted `VIDEO_URL=…`, `CLIP_START=…`, `CLIP_END=…` and `CLIP_SECONDS=…` assignments (signed clip URLs carry `&` and `?`, so paste the lines as printed, never the bare URL) and refuses a clip of 120 s or longer (Long-video rule). Each fenced block is a fresh shell: at the top of the Step 3 block paste those lines and the `VLM_BACKEND` / `VLM_ENDPOINT` / `VLM_MODEL` hand-off Step 2 printed — the Step 3 guards refuse to run without a video source and a VLM endpoint + model. Set `RAW_URL="$VIDEO_URL"` before applying the report-link rewrite for Step 4.

Remote VLM reachability guard (required):
- If the selected `VLM_ENDPOINT` is remote/non-local, do not assume it can fetch `VIDEO_URL` when `VIDEO_URL` points to localhost/private VST addresses (for example `127.0.0.1`, `localhost`, `HOST_IP`, `172.16-31.x`, `192.168.x`, `10.x`, or in-cluster/internal DNS).
- Before Step 3, explicitly warn and stop when this mismatch exists: remote VLM + internal-only `VIDEO_URL`.
- In that case, ask the user to choose one of:
  1. Use a local/in-cluster VLM endpoint that can reach VST internal URLs.
  2. Switch to Mode A A2 and send local-file/base64 bytes to the remote VLM.
  3. Expose a browser/publicly reachable clip URL and confirm the remote VLM can fetch it.

#### A2 — Local file on disk or base64 video path (no VST dependency)

If the user provides either:
- a local video file path on disk (where OpenClaw/agent is running), or
- a base64 video payload,
and a VLM endpoint, run Step 2 (it confirms caller-supplied values or discovers an endpoint, then prints the hand-off) and use that hand-off in Step 3.

Local file requirement (strict):
- `VIDEO_FILE` must point to a path that is directly readable from the runtime executing this skill (OpenClaw/agent host or container).
- The path cannot be browser-only client storage.
- If the file is only on a user's laptop/browser session and not on the runtime filesystem, ask the user to place it on the runtime disk (or provide base64 instead).

Bind:
- `VIDEO_FILE` = user-provided local path (if using file path input)
- `VIDEO_B64_FILE` = a file holding the base64 bytes (if using base64 input; no data-uri prefix). Never paste base64 into a shell block: write the user's payload to a file with the Write tool and pass the path.
- `VIDEO_MIME` = `video/mp4` unless user provided another valid mime type
- `VIDEO_DATA_URL` = `data:${VIDEO_MIME};base64,<payload>` — built inside the Step 3 block from `VIDEO_FILE` or `VIDEO_B64_FILE` (see the A2 note after that block)

If `VIDEO_FILE` is provided, the Step 3 block encodes it at runtime; never paste raw base64 into chat output or into a shell block.

For this path, set report `Clip URL` row to `N/A (local/base64 input)` unless a public playback URL is also available.

#### Long-video rule (required)

A clip of **120 seconds or longer** never takes the direct VLM path: hand off to
`/vss-summarize-video` and continue with its output in Step 4. The A1 Step 1
block and the Step 3 block both exit on `CLIP_SECONDS` of 120 or more — never a
default; that exit is the hand-off, not a failure.

### Step 2 — Resolve VLM endpoint and model

The deploy may serve the VLM through either of two stacks. Both expose an OpenAI-compatible `chat/completions` API — pick whichever is live:

| Backend | Discovery input | Typical host endpoint | Picked when |
|---|---|---|---|
| **Public Ingress RT-VLM** | `VSS_PUBLIC_URL` / `VLM_ENDPOINT` | `${VSS_PUBLIC_URL}/rtvi-vlm/v1` | Kubernetes / Helm, any profile, when `VSS_PUBLIC_URL` is set (preferred) |
| **NIM Cosmos** | Explicit `VLM_ENDPOINT`, or successful `/models` probe | `http://${HOST_IP}:30082/v1` | Docker: port 30082 responds with at least one model |
| **RT-VLM Cosmos** | Explicit `VLM_ENDPOINT`, or successful `/models` probe | `http://${HOST_IP}:8018/v1` | Docker: port 8018 responds with at least one model |

If the user already supplied a `VLM_ENDPOINT` + model id, paste them at the top of the block below **after** the *Endpoint resolution* hand-off lines (later lines win); the block then only confirms them.

One block, one shell — paste the *Endpoint resolution* hand-off lines (SKILL.md) at its top. It picks the public Ingress RT-VLM route on Kubernetes (do **not** probe `/vlm/v1`), probes the standard host endpoints on Docker only (same endpoint-selection contract as `/vss-ask-video`), confirms `/v1/models`, and prints the hand-off for Step 3:

```bash
case "${DEPLOYMENT_KIND:?paste the Endpoint resolution output (SKILL.md § Endpoint resolution) at the top of this block}" in
  kubernetes|docker) ;;
  *) echo "ERROR: DEPLOYMENT_KIND must be kubernetes or docker, got '${DEPLOYMENT_KIND}'" >&2; exit 1 ;;
esac
VLM_ENDPOINT="${VLM_ENDPOINT%/}"   # tolerate a trailing slash on a pasted / caller-supplied endpoint

# 1) Kubernetes: the public Ingress RT-VLM route — normally pasted from Endpoint resolution;
#    re-derived here only when that line is missing.
if [ -z "${VLM_ENDPOINT:-}" ] && [ "${DEPLOYMENT_KIND}" = "kubernetes" ] && [ -n "${VSS_PUBLIC_URL:-}" ]; then
  VLM_ENDPOINT="${VSS_PUBLIC_URL%/}/rtvi-vlm/v1"
fi

# 2) Docker only: probe the standard host endpoints. A caller-supplied VLM_MODEL must be served by the
#    candidate, otherwise keep looking (so :8018 is still tried when :30082 serves another model).
if [ -z "${VLM_ENDPOINT:-}" ] && [ "${DEPLOYMENT_KIND}" = "docker" ]; then
  for _endpoint in "http://${HOST_IP:-localhost}:30082/v1" "http://${HOST_IP:-localhost}:8018/v1"; do
    _models="$(curl -sf --max-time 5 "${_endpoint}/models")" || continue
    _model="$(printf '%s' "${_models}" | jq -er --arg m "${VLM_MODEL:-}" 2>/dev/null \
      'if $m == "" then (.data[0].id // empty) else (.data[]?.id | select(. == $m)) end' | head -n 1)" || continue
    [ -n "$_model" ] || continue
    VLM_ENDPOINT="${_endpoint}"
    VLM_MODEL="${_model}"
    break
  done
fi

[ -n "${VLM_ENDPOINT:-}" ] || {
  if [ "${DEPLOYMENT_KIND}" = "kubernetes" ]; then
    echo "ERROR: VLM_ENDPOINT and VSS_PUBLIC_URL are missing from the pasted Endpoint resolution output — re-paste it (or supply VLM_ENDPOINT)" >&2
  else
    echo "ERROR: no VLM serving ${VLM_MODEL:-any model} on ${HOST_IP:-localhost}:30082 or ${HOST_IP:-localhost}:8018; provide VLM_ENDPOINT and VLM_MODEL" >&2
  fi
  exit 1
}

# 3) Confirm the chosen endpoint is alive and the model is loaded.
_models="$(curl -sf --max-time 5 "${VLM_ENDPOINT}/models")" || {
  echo "ERROR: VLM endpoint is not reachable: ${VLM_ENDPOINT}" >&2
  exit 1
}
printf '%s' "${_models}" | jq -er '.data[].id' >&2   # served ids, on stderr so stdout stays paste-clean
[ -n "${VLM_MODEL:-}" ] ||
  VLM_MODEL="$(printf '%s' "${_models}" | jq -er '.data[0].id // empty')"
[ -n "${VLM_MODEL:-}" ] || { echo "ERROR: ${VLM_ENDPOINT}/models lists no model ids; provide VLM_MODEL" >&2; exit 1; }
# Never silently use an unknown model: the chosen / caller-supplied id must be one the endpoint serves.
printf '%s' "${_models}" | jq -e --arg m "$VLM_MODEL" 'any(.data[]?.id; . == $m)' >/dev/null || {
  echo "ERROR: VLM_MODEL '${VLM_MODEL}' is not served at ${VLM_ENDPOINT}; pick one of the ids listed above or use a *VLM selection when unclear* option (SKILL.md)" >&2
  exit 1
}

# Backend, derived once from the FINAL endpoint / model (a pasted VLM_BACKEND from an earlier run is
# ignored): the public /rtvi-vlm route and :8018 are RT-VLM (never send them NIM Cosmos kwargs), :30082 is
# NIM Cosmos, anything else follows the model id. Step 3 requires this value from the hand-off.
case "${VLM_ENDPOINT}" in
  */rtvi-vlm/*|*:8018/*) VLM_BACKEND="rtvlm" ;;
  *:30082/*)             VLM_BACKEND="nim_cosmos" ;;
  *) case "${VLM_MODEL}" in nvidia/cosmos*) VLM_BACKEND="nim_cosmos" ;; *) VLM_BACKEND="rtvlm" ;; esac ;;
esac

# Hand-off — paste these lines at the top of the Step 3 block (fresh shell).
printf 'VLM_BACKEND=%q\nVLM_ENDPOINT=%q\nVLM_MODEL=%q\n' "$VLM_BACKEND" "$VLM_ENDPOINT" "$VLM_MODEL"
```

If `VLM_MODEL` is empty, adopt the first id the endpoint advertises. If the probe fails or the listed ids don't include `${VLM_MODEL}`, either:
- try a discovered fallback endpoint, or
- ask the user to choose one of the *VLM selection when unclear* options (SKILL.md).

Never silently pick an unknown model.

### Step 3 — Call the VLM directly

Use the OpenAI-compatible `chat/completions` endpoint with a `video_url` content block — the same payload shape **and multimodal settings** `video_understanding` builds in `src/vss_agents/tools/video_understanding.py` (`_build_vlm_messages` + the Cosmos `base_vlm.bind(...)` call).

Use explicit `VIDEO_UNDERSTANDING_*` overrides when supplied; otherwise use
the base-profile defaults (`max_fps=2`, `max_frames=30`, `min_pixels=3136`,
`max_pixels=8388608`). Step 3 never reads the running vss-agent's config, so it does not fail when vss-agent is absent: the defaults (or the `VIDEO_UNDERSTANDING_*` overrides) apply.

```bash
# Default prompt — load from the skill tree (do NOT use a cwd-relative path).
# Set SKILL_DIR to the "Base directory for this skill" announced when this skill loads.
: "${SKILL_DIR:?Set SKILL_DIR to the loaded skill base directory (from the skill loader)}"
PROMPT_FILE="$SKILL_DIR/references/default-vlm-prompt.md"
[ -s "$PROMPT_FILE" ] || {
  echo "ERROR: missing or empty VLM prompt file: $PROMPT_FILE" >&2
  exit 1
}
DEFAULT_PROMPT="$(cat "$PROMPT_FILE")"
[ -n "$DEFAULT_PROMPT" ] || {
  echo "ERROR: DEFAULT_PROMPT is empty after reading $PROMPT_FILE" >&2
  exit 1
}

# Fresh shell: HITL state never arrives on its own. Set BOTH at the top of THIS block
# (values from SKILL.md § HITL prompt mode); the block refuses to run without them:
#   HITL_RESOLVED=true|false
#   HITL_PROMPT_FILE — required when HITL_RESOLVED=true: never paste prompt text into the shell.
#   Write the approved / edited / new text to a file with the Write tool (e.g. /tmp/vss-hitl-prompt.txt)
#   and set HITL_PROMPT_FILE=<that path>; the block reads it with cat, so nothing in the prompt is
#   ever interpreted by the shell.
: "${HITL_RESOLVED:?set HITL_RESOLVED=true|false at the top of this block (SKILL.md § HITL prompt mode)}"
case "$HITL_RESOLVED" in
  true)  [ -f "${HITL_PROMPT_FILE:-}" ] && FINAL_PROMPT=$(cat "$HITL_PROMPT_FILE") && [ -n "$(printf '%s' "$FINAL_PROMPT" | tr -d '[:space:]')" ] \
           || { echo "ERROR: HITL resolved true but HITL_PROMPT_FILE is unset, unreadable or blank — write the approved prompt to a file with the Write tool and set HITL_PROMPT_FILE" >&2; exit 1; } ;;
  false) FINAL_PROMPT="$DEFAULT_PROMPT" ;;
  *)     echo "ERROR: HITL_RESOLVED must be exactly true or false, got '$HITL_RESOLVED'" >&2; exit 1 ;;
esac

# FINAL_PROMPT was set by the HITL_RESOLVED case above: false -> the default prompt file;
# true -> the approved text from HITL_PROMPT_FILE (resolution order: video_report_gen.hitl_enabled,
# then HITL_ENABLED, then default false — SKILL.md § HITL prompt mode). Both branches are non-empty (true: whitespace-stripped check; false: the [ -n ] guard above).
PROMPT="$FINAL_PROMPT"

# Reasoning is OFF by default — matches the base-profile video_understanding config (`reasoning: false`).
# video_understanding.py uses config.reasoning unless the caller overrides it, so default to non-reasoning.
# Append the Cosmos Reason 2 reasoning suffix ONLY when the user explicitly asked for reasoning: set
# REASONING=true at the top of this block in that case (drop it for non-cosmos-reason2 VLMs). With reasoning
# off, the response has no <think> block.
if [ "${REASONING:-false}" = "true" ]; then
PROMPT="${PROMPT}

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."
fi

# Backend comes from the Step 2 hand-off (rtvlm | nim_cosmos); no second copy of the rule here.
: "${VLM_BACKEND:?paste the Step 2 hand-off (VLM_BACKEND / VLM_ENDPOINT / VLM_MODEL) at the top of this block}"
case "${VLM_BACKEND}" in rtvlm|nim_cosmos) ;; *) echo "ERROR: VLM_BACKEND must be rtvlm or nim_cosmos, got '${VLM_BACKEND}'" >&2; exit 1 ;; esac

# Multimodal settings — explicit overrides or base-profile defaults.
CFG_JSON='{"max_fps":2,"max_frames":30,"min_pixels":3136,"max_pixels":8388608}'

printf '%s' "${CFG_JSON}" | jq -e . >/dev/null || { echo "Invalid video_understanding config JSON"; exit 1; }
MAX_FPS="$(printf '%s' "${CFG_JSON}" | jq -r '.max_fps')"
MAX_FRAMES="$(printf '%s' "${CFG_JSON}" | jq -r '.max_frames')"
MIN_PIXELS="$(printf '%s' "${CFG_JSON}" | jq -r '.min_pixels')"
MAX_PIXELS="$(printf '%s' "${CFG_JSON}" | jq -r '.max_pixels')"
MAX_FPS="${VIDEO_UNDERSTANDING_MAX_FPS:-$MAX_FPS}"
MAX_FRAMES="${VIDEO_UNDERSTANDING_MAX_FRAMES:-$MAX_FRAMES}"
MIN_PIXELS="${VIDEO_UNDERSTANDING_MIN_PIXELS:-$MIN_PIXELS}"
MAX_PIXELS="${VIDEO_UNDERSTANDING_MAX_PIXELS:-$MAX_PIXELS}"

# CLIP_SECONDS comes from the Step 1 hand-off (A1) or the measured duration (A2: ffprobe, or the user's stated
# length) — a number of seconds, never an ISO timestamp and never a default. It re-checks the Long-video rule
# and drives num_frames = min(int(clip_seconds) * max_fps, max_frames), min 1 — matches video_understanding.py.
: "${CLIP_SECONDS:?paste CLIP_SECONDS from the Step 1 hand-off (A1) or set the measured duration in seconds (A2)}"
case "$CLIP_SECONDS" in ''|.*|*[!0-9.]*) echo "ERROR: CLIP_SECONDS must be a number of seconds, got '${CLIP_SECONDS}'" >&2; exit 1 ;; esac
LC_ALL=C awk -v d="$CLIP_SECONDS" 'BEGIN{exit !(d+0 > 0)}' || { echo "ERROR: CLIP_SECONDS must be greater than 0 s, got '${CLIP_SECONDS}' — re-measure the clip" >&2; exit 1; }
LC_ALL=C awk -v d="$CLIP_SECONDS" 'BEGIN{exit !(d+0 < 120)}' || { echo "Clip is ${CLIP_SECONDS} s (120 s or longer): hand off to /vss-summarize-video (Long-video rule)" >&2; exit 1; }
CLIP_SECONDS=$(LC_ALL=C awk -v s="$CLIP_SECONDS" 'BEGIN{printf "%d", s}')   # truncated here only, for num_frames (bash arithmetic is integer-only); both gates above ran on the precise value
NUM_FRAMES=$(( CLIP_SECONDS * MAX_FPS ))
[ "$NUM_FRAMES" -gt "$MAX_FRAMES" ] && NUM_FRAMES=$MAX_FRAMES
[ "$NUM_FRAMES" -lt 1 ] && NUM_FRAMES=1

# Only apply Cosmos mm/media kwargs on the NIM Cosmos path.
# RT-VLM mode uses its own server-side preprocessing and should not receive these kwargs.
MM_KWARGS=""
if [ "${VLM_BACKEND}" = "nim_cosmos" ]; then
  case "$VLM_MODEL" in
    *cosmos-reason2*) MM_KWARGS=", \"mm_processor_kwargs\": {\"size\": {\"shortest_edge\": ${MIN_PIXELS}, \"longest_edge\": ${MAX_PIXELS}}}, \"media_io_kwargs\": {\"video\": {\"num_frames\": ${NUM_FRAMES}}}" ;;
    *cosmos*)         MM_KWARGS=", \"mm_processor_kwargs\": {\"videos_kwargs\": {\"min_pixels\": ${MIN_PIXELS}, \"max_pixels\": ${MAX_PIXELS}}}, \"media_io_kwargs\": {\"video\": {\"num_frames\": ${NUM_FRAMES}}}" ;;
    *)                      MM_KWARGS="" ;;
  esac
fi

# Fresh shell: Step 2's VLM discovery does not arrive on its own — set both at the top of this block.
: "${VLM_ENDPOINT:?set VLM_ENDPOINT (Step 2) at the top of this block}" "${VLM_MODEL:?set VLM_MODEL (Step 2) at the top of this block}"
VLM_ENDPOINT="${VLM_ENDPOINT%/}"   # tolerate a trailing slash on a pasted / caller-supplied endpoint
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-4096}"   # the agent's nim/openai/vllm VLM profiles use 4096 (rtvi_vlm sets none); override only with a positive integer
case "$VLM_MAX_TOKENS" in ''|0*|*[!0-9]*) echo "ERROR: VLM_MAX_TOKENS must be a positive integer, got '${VLM_MAX_TOKENS}'" >&2; exit 1 ;; esac
# A1 sends the VST clip URL; A2 (Step 1) sends inline bytes — this one block serves both paths.
VIDEO_SRC="${VIDEO_DATA_URL:-${VIDEO_URL:?set VIDEO_URL (A1) or VIDEO_DATA_URL (A2) at the top of this block}}"
case "$VIDEO_SRC" in
  null|"") echo "ERROR: VIDEO_URL is null/empty — Step 1 clip resolution failed (no media_url)" >&2; exit 1 ;;
  data:*,) echo "ERROR: VIDEO_DATA_URL carries no base64 payload (empty or unreadable VIDEO_FILE / VIDEO_B64_FILE)" >&2; exit 1 ;;
esac

# Keep the raw response: classify transport / HTTP / API failures BEFORE treating anything as report text.
VLM_BODY=$(mktemp) || exit 1
trap 'rm -f "$VLM_BODY"' EXIT
CODE=$(curl -sS --connect-timeout 5 --max-time 120 -o "$VLM_BODY" -w '%{http_code}' -X POST "${VLM_ENDPOINT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": $(printf '%s' "${VLM_MODEL}" | jq -Rs .),
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": $(printf '%s' "${PROMPT}" | jq -Rs .)},
        {"type": "video_url", "video_url": {"url": $(printf '%s' "${VIDEO_SRC}" | jq -Rs .)}}
      ]
    }
  ],
  "max_tokens": ${VLM_MAX_TOKENS},
  "temperature": 0.0${MM_KWARGS}
}
EOF
) || { echo "VLM chat/completions: curl failed (transport error above)" >&2; cat "$VLM_BODY" >&2; exit 1; }
case "$CODE" in
  2??) ;;
  *) echo "VLM chat/completions failed: HTTP $CODE" >&2; cat "$VLM_BODY" >&2; exit 1 ;;
esac
# The body must be JSON; a truncated answer (finish_reason "length") is not a report: stop and raise VLM_MAX_TOKENS.
jq -e 'type == "object"' "$VLM_BODY" >/dev/null 2>&1 || { echo "VLM chat/completions returned a body that is not a JSON object" >&2; cat "$VLM_BODY" >&2; exit 1; }
jq -e '(try .choices[0].finish_reason catch null) != "length"' "$VLM_BODY" >/dev/null 2>&1 \
  || { echo "VLM answer truncated by max_tokens (finish_reason=length) — raise VLM_MAX_TOKENS" >&2; cat "$VLM_BODY" >&2; exit 1; }
# The report body is the content with every <think>…</think> reasoning block (Cosmos Reason reasoning mode)
# removed — also a leading block whose <think> the chat template injected — and any <answer> tags dropped, trimmed. Failures (surface per SKILL.md § Error Handling, never render): an error envelope, an
# empty body, null or list-typed content, a reasoning-only answer, or a <think> block left unclosed because
# max_tokens cut the answer off.
jq -er '.choices[0].message.content | select(type=="string")
        | gsub("<think>[\\s\\S]*?</think>"; "") | sub("^[\\s\\S]*?</think>"; "") | gsub("</?answer>"; "")
        | select(test("<think>") | not)
        | sub("^\\s+"; "") | sub("\\s+$"; "") | select(length>0)' "$VLM_BODY" 2>/dev/null \
  || { echo "VLM chat/completions returned no report text (error envelope, empty body, null / list-typed content, reasoning-only, an unclosed <think> block — truncated by max_tokens? — or the literal text <think> in the answer)" >&2; cat "$VLM_BODY" >&2; exit 1; }
```

For Mode A path A2 (inline bytes), run the same Step 3 block with `VIDEO_DATA_URL` (Step 1) set at its top instead of `VIDEO_URL`; the block sends whichever is set, so the HITL guard, prompt resolution, `CFG_JSON` and `MM_KWARGS` apply to A2 unchanged. Because the block is a fresh shell, build the data URL there too — for a local file: `[ -s "$VIDEO_FILE" ] || { echo "ERROR: VIDEO_FILE missing or empty: $VIDEO_FILE" >&2; exit 1; }; VIDEO_DATA_URL="data:${VIDEO_MIME:-video/mp4};base64,$(base64 < "$VIDEO_FILE" | tr -d '\n')"` (the `tr` strips the line wrapping GNU `base64` adds, which would otherwise corrupt the data URL) and set the duration the Long-video rule and frame sampling need: `CLIP_SECONDS=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO_FILE")` (ask the user for the length in seconds when `ffprobe` is unavailable); for user-supplied base64 written to `VIDEO_B64_FILE` (set `CLIP_SECONDS` to the length the user states, or measure the decoded file with `ffprobe`): `[ -s "$VIDEO_B64_FILE" ] || { echo "ERROR: VIDEO_B64_FILE missing or empty: $VIDEO_B64_FILE" >&2; exit 1; }; VIDEO_DATA_URL="data:${VIDEO_MIME:-video/mp4};base64,$(tr -d '[:space:]' < "$VIDEO_B64_FILE")"`.

> The kwargs block is backend-aware: on `nim_cosmos`, Reason2 variants (`nvidia/cosmos-reason2*`) use `mm_processor_kwargs.size{shortest_edge,longest_edge}` and other NIM Cosmos variants (`nvidia/cosmos*`) use `mm_processor_kwargs.videos_kwargs{min_pixels,max_pixels}`; both also send `media_io_kwargs.video.num_frames`. On `rtvlm`, no Cosmos kwargs are sent.

If the VLM returns a `<think>…</think>` block (Cosmos Reason reasoning mode), the block above already removes every closed block and prints the remaining text as the report body; it exits non-zero when nothing remains or when a block was left unclosed (answer truncated by `max_tokens`).

### Step 4 — Fill the Video Analysis Report template

Load the matching template from [`$SKILL_DIR/references/report-templates/video-analysis-report.md`](../report-templates/video-analysis-report.md). Treat the template as read-only — copy its structure **verbatim**, keeping its exact headings and `## Basic Information` pipe-table, and fill every placeholder. Fill all placeholders before returning markdown. Never leave template instructions, placeholder tokens (e.g. `<BROWSER_CLIP_URL>`, `<sensor_id>`, `<YYYY-MM-DD>`), or internal-only URLs in user output. Row values by path — a literal `N/A (<reason>)` is a concrete value, a placeholder token is not:
- `Clip URL`: A1 — the `BROWSER_CLIP_URL` printed by the rewrite block when non-empty; when it printed empty, `N/A (browser-playable URL unavailable)` and say why in the chat response that accompanies the report (never inside `## Analysis Results`). A2 — `N/A (local/base64 input)` unless a public playback URL exists. LVS path — the rewritten URL when a VST clip exists, else `N/A (LVS summary)`. Never use the raw `HOST_IP:30888` URL.
- `Clip Range`: A1 — `CLIP_START - CLIP_END` from the Step 1 hand-off (the template's `<startTime> - <endTime>`); LVS path — the window given to `/vss-summarize-video`; A2 — `N/A (local/base64 input)`.
- `VLM`: `VLM_MODEL` plus the backend kind in the template's form, `(NIM or RT-VLM)` (`nim_cosmos` → NIM, `rtvlm` → RT-VLM), for the VLM-direct path — never the endpoint URL; `LVS (/vss-summarize-video)` for the LVS path.
