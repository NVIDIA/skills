# Iterate

Use for changes to an existing working agent. Preserve the working baseline, change one
layer, then repeat the spoken exchange.

If the agent is already broken, start with `operations/troubleshoot.md`. Do not mix a
repair with an enhancement.

## 1. Read the Current Project

Read the agent file, dependency file, `.env.example`, and any deployment instructions.
Identify the current framework, pipeline, model ids, endpoints, and platform before
proposing a change.

Never overwrite `.env` or read secrets into the response.

## 2. Route the Change

| Change | Reopen | Approval |
| --- | --- | --- |
| Prompt, persona, response style | `domain/agent-behavior.md` + framework docs MCP | no new intake table |
| Reasoning | `models/llm.md` | confirm when turning on |
| LLM, ASR, TTS, language, or voice | matching model file + `models/language-routing.md` when relevant | show changed row and dependent rows |
| Domain glossary, boosting, or pronunciation | `domain/speech-customization.md` | show and confirm every changed term |
| Transport or turn handling | selected framework file + docs MCP | confirm user-visible behavior |
| Logs or latency metrics | `operations/observability.md` + framework docs MCP | no new intake table |
| Cloud / self-hosted / GPU placement, or expected concurrency | `preflight.md` + the routed platform guide | confirm deployment row |
| Cascaded / Omni | `intake.md` + selected framework files | confirm all changed rows |
| Pipecat / LiveKit | `intake.md` + both framework files | full framework confirmation |
| Hardware platform | `preflight.md` + routed platform guide | rerun probe and deployment fit |

Do not repeat the full intake for a local change. Reopen only the affected row and
anything it changes downstream.

## 3. Revalidate Dependencies

- Re-query the framework documentation MCP before changing framework imports, classes,
  settings, or pipeline structure.
- Rediscover model ids and deployment instructions. Do not reuse a remembered id, image,
  profile, or tag.
- Recheck credentials only when the change adds a provider, framework, or model source.
- Keep healthy local services running unless their model, profile, tags, or endpoint must
  change.

## Cascading Changes

- Model swap: a Cascaded NIM reruns the LLM NIM matrix and `list-model-profiles`. Omni NIM
  reruns the VLM NIM sources in `frameworks/omni.md`. Cloud rechecks model id and API. The
  single-GPU stack reopens the vLLM model card and reconfirms the quantization variant for
  this GPU. Then revisit reasoning and stack fit.
- Concurrency change on a workstation: reopen `preflight.md` §4, because it can move the
  agent between the single-GPU stack and NIM. Confirm the Deployment row before touching
  anything, since that swap replaces every local service.
- ASR swap: keep streaming first and recheck language, tags, and deployment fit.
- TTS or language change: restart TTS when required, then query its voice-discovery API
  again.
- Cascaded to Omni:
  - copy current `nvidia_omni_multimodal_service.py` and
    `audio_only_smart_turn_strategy.py`
  - remove ASR from agent wiring and add the Omni turn and initial-mute strategies
  - replace local `llm` + `asr` Compose services with `omni`, retaining `tts`
  - recalculate Omni + TTS fit
- Omni to cascaded:
  - add streaming ASR and the documented text LLM service
  - remove Omni-only turn strategies and imports
  - replace local `omni` with `llm` + `asr`, retaining `tts`
  - remove unmodified copied Omni files when they are no longer imported
  - preserve and disclose any user-modified Omni file
- Deployment change: update code constants, generated Compose, and README through
  `output-contract.md`. `.env` remains secrets only.

## 4. Apply the Smallest Change

Prefer a documented runtime setting update when the framework supports it. Otherwise
restart only the affected agent or model service. Rebuild the pipeline when its framework
or shape changes.

Do not regenerate unrelated files, reset user edits, or replace working services.

## 5. Verify

1. Check readiness for every changed local service.
2. Update `scripts/smoke.sh` for the new shape and run it.
3. Start the agent.
4. Verify the changed behavior.
5. Complete the full spoken exchange in `operations/run.md`.
6. If verification fails, use `operations/troubleshoot.md` and keep the failure scoped to
   this change.

## Anti-Patterns

- Editing framework APIs from memory.
- Changing several layers before testing.
- Treating an enhancement as permission to rewrite the project.
- Changing an approved model or deployment silently.
- Declaring success without a spoken exchange.
