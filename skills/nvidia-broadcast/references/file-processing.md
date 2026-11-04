# File processing workflow

Apply NVIDIA Broadcast effects to a local media file, asynchronously.

Supported inputs: `.mp4`, `.m4a`, `.mp3`, `.wav`. Output defaults to the Broadcast output
folder with a timestamped name; override with `outputFolder` or `outputPath`, which may
name any absolute local folder — the default is a default, not a boundary.

## Steps

1. **Submit** (note `_version: 1`):
   ```json
   {
     "name": "submit_file_processing",
     "arguments": {
       "_version": 1,
       "inputPath": "C:\\Users\\me\\Videos\\clip.mp4",
       "effects": [
         { "effectId": "background_blur", "params": { "strength": 0.6 } },
         { "effectId": "video_frame_generation", "params": { "multiplier": 2, "model": "balanced" } },
         { "effectId": "noise_removal" }
       ]
     }
   }
   ```
   Returns `{ id, batchId, status, progress, sourcePath, outputPath, mediaKind, effects,
   skippedEffects?, errorMessage?, recovery?, reason?, createdAt }`. For multiple files, omit
   `batchId` on the first submission, then pass its returned `batchId` on every related
   submission. Submit between 1 and 32 unique effects.

2. **Wait once per batch**, not once per file. First request an immediate snapshot:
   ```json
   { "name": "wait_for_processing_update", "arguments": { "_version": 1, "batchId": "<batchId>" } }
   ```
   If `status` is not terminal, pass the returned `cursor` unchanged to wait for the next job-status
   or queue-pause change:
   ```json
   { "name": "wait_for_processing_update", "arguments": { "_version": 1, "batchId": "<batchId>", "cursor": "<cursor>", "timeoutMs": 120000 } }
   ```
   Repeat while `status` is `queued` or `in_progress`. Progress-only changes deliberately
   do not wake the wait. Terminal batch statuses are `completed`, `failed`, `cancelled`,
   and `mixed`; report the per-status summary and do not describe `mixed` as passed.
   Cancellation is an explicit terminal user action: report `cancelled_by_user` and stop.
   If the user explicitly asks afterward, call `retry` with that job's exact `jobIds`; never
   retry automatically, use `all`, resubmit, or create a replacement job.

3. **Handle a paused queue without overriding the user.** A submission made while the
   user has stopped file processing is queued silently. A batch update then returns
   `status: "paused_by_user"` and a message telling the agent to ask before resuming.
   Ask in the agent conversation. Only after an explicit yes, call ordinary queue-wide
   `resume` once. MCP does not accept or verify a consent flag; consent is the agent's
   responsibility. If the user declines, leave every file queued.

4. **Control** processing with `control_file_processing_jobs`:
   - Pause and resume are queue-wide and never accept `jobIds` or `all`.
   - Cancel or retry jobs with `jobIds`, or use `all: true`.
   - Retry applies to failed jobs with `recovery: "retry"`. It also accepts a cancelled job
     only when the agent passes that job's exact `jobIds` after an explicit user request.
     `all: true` never includes cancellations. Read `event`, `jobIds`, and `jobs`; if `event`
     is `no_change`, stop rather than submitting replacements.
   - Aborting `wait_for_processing_update` stops waiting only; it never cancels media work.
     Use an explicit cancel action to cancel jobs.



## Notes

- **Speaker effects are rejected** for file processing.
- File effects use `{ effectId, params? }`; all effect options must be nested under `params`.
- `effects` requires 1-32 unique public effect IDs.
- For `video_frame_generation`, pass `params.multiplier` 2\|4 and `params.model`
  "performance"\|"balanced"\|"quality".
- For `background_replace`, pass a local absolute path as `params.imagePath`, ending in
  `.png`, `.jpg`, `.jpeg`, `.mp4`, `.mov` or `.mkv`. UNC, device and extended-length paths
  are rejected, as is anything over 259 bytes. Other path aliases are not part of the schema.
- `studio_voice` accepts `params.micProfile`; use only a value advertised by the capability-aware schema.
- An incompatible effect (e.g. a video effect on an audio-only file) is **skipped** when at
  least one requested effect applies — check `skippedEffects` on the returned job. A request
  where every effect would be skipped is rejected.
- Output extension is derived from input; processing an `.mp3` with audio effects yields `.m4a`.
- There is no overwrite input. A fresh submission rejects an existing destination and the
  agent must tell the user to choose another path or remove it. Resume/retry may replace
  only the same job's own partial output, tracked internally; they cannot overwrite an
  unrelated file. The input file is never replaceable.
- Tool errors use concise text plus `structuredContent.error` with the code, `retryable`,
  and `suggestedFix` — schema rejections included, so one parse path covers every failure.
  JSON-RPC protocol failures use a top-level `error` instead. Retry on `retryable: true`;
  a `-32602` will keep failing until you change the arguments.
