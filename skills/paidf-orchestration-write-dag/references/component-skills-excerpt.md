# Component Skills — Minimal Excerpt

**What this is.** `write-dag` depends on the `augmentation` and `auto-labeling` component skills for
config schema, prompt-authoring conventions, and troubleshooting. Those skills live outside this
repo and are not linked to `sdg-workflow` in any durable way (no submodule, no plugin registration)
— they're only reachable when a session happens to have them added as extra workspace directories.
This file is a **small, frozen excerpt** of the parts `write-dag` has actually needed so far, kept
here so the skill still works when they're not reachable.

**Staleness warning.** This is a snapshot, not a live reference. It will drift from the real repos.
If the two repos *are* reachable in your session, prefer calling the real skills
(`Skill(skill="augmentation")` / `Skill(skill="auto-labeling")`) over this file — they're
authoritative. Only fall back to this file when they're unavailable, and if what you need isn't
covered here, say so and ask rather than extrapolating beyond it.

---

## `augmentation` — config schema essentials

Configs are validated against a `PipelineConfig` Pydantic model with seven top-level sections:
`data`, `endpoints` (a **list**), `pipeline`, `captioning`, `augmentation`, `data_processing`,
`evaluators`.

### `endpoints` — BYOM registry (a list, not a mapping)

```yaml
endpoints:
  - id: vlm_qwen                 # optional; only needed to disambiguate 2+ same-role endpoints
    role: vlm                    # required
    url: "http://localhost:8000/v1"   # required
    model: "Qwen/Qwen3.6-27B-FP8"     # wire model string, default ""
    # adapter omitted -> role default
    api_key_env: VLM_API_KEY     # optional; name of env var holding the key, never the key itself
    timeout: 120                 # seconds
```

**Roles → default adapter:** `vlm`/`llm` → `openai.chat.completions`; `image_edit` →
`nim`; `video_transfer`/`video_predict` → `nim`; `image2video` → `openai.video.sync`.

**Known adapters:** `openai.chat.completions`, `openai.images.edits`, `openai.video.sync`,
`openai.video.async`, `nim`, `passthrough`.

**Model selection:** `augmentation.model.name` resolves to an endpoint by `id`, then by `role`,
then by a model-name→role map (`image-edit`→`image_edit`, `cosmos-transfer2.5`→`video_transfer`,
`cosmos-predict`→`video_predict`, `cosmos3-image2video`→`image2video`). Two endpoints sharing a role
must be disambiguated with `id`.

### Image-edit / IAA-style captioning block

```yaml
captioning:
  llm:
    text: "Change the person's clothing to a {top_outer_color} {top_outer_type}, ..."
    variables: { top_outer_color: ["red"], ... }        # full replace, not merge, per generation
    verification_options: { top_outer_color: [...allowed values...] }
evaluators:
  - attribute_verification:
      enabled: true
      exclude_variables: ["shoe_type", "shoe_color"]     # excluded from MCQ gate, still in prompt
```

`Template has placeholders {x} not found in variables` means every placeholder in `captioning.llm.text`
needs a matching key in `variables` — there is no partial/merge behavior, a caller must supply all of them.

### Serving requirement (IAA image-edit specifically)

`vllm-omni serve Qwen/Qwen-Image-Edit-2511 --omni` — the `--omni` flag is required to expose
`/v1/chat/completions`; without it the endpoint doesn't serve the chat-completions contract that
`adapter: openai.chat.completions` expects.

### Troubleshooting — top entries actually hit so far

| Error | Cause | Fix |
|---|---|---|
| `no endpoint matched selector '<name>'` | `model.name` doesn't match any endpoint `id`/`role` | Add a matching endpoint, or set `model.name` to a specific `id` |
| `selector '<name>' matches multiple endpoints with role '<role>'` | Two endpoints share a role | Give them `id`s, target one by `id` |
| `endpoint <id> resolves to unknown adapter '<x>'` | Typo'd/unsupported adapter | Use one of the known adapters listed above |
| `Template has placeholders {x} not found in variables` | Missing key in `captioning.llm.variables` | Supply every placeholder referenced in the text template |
| `401 Unauthorized` | Missing/wrong API key | Set the env var named by `api_key_env`; local endpoints need none |
| `404` on a chat/video route | Wrong adapter for the server's actual route | Match adapter to route: `/v1/chat/completions` vs `/v1/images/edits` vs `/v1/videos/sync` vs NIM `/v1/infer` |
| Hosted NVCF NIM times out / returns 202 | Generation outlives the hold-open window | `adapter: nim`, raise endpoint `timeout`; poller uses `NVCF-POLL-SECONDS`/`NVCF-REQID` automatically |

---

## `auto-labeling` — prompt/question-bank authoring essentials

### Workflow (from `references/prompt-authoring.md`)

1. Classify the prompt type: image caption, video dense caption, VQA evidence, or LLM evidence
   aggregation.
2. Write evidence-first — describe what's visible before any classification; never infer identity,
   intent, fault, or protected attributes. Explicit uncertainty handling required.
3. Put prompt files under the cookbook's `prompts/` tree, referenced via `stage_args` /
   `question_bank_file`, never inlined into shell commands.
4. For a **reasoning-capable model** (e.g. Gemini 3 Flash) on the `reasoning`/`visual_qa` stages,
   raise `max_tokens` generously (e.g. `32768`) — reasoning models spend part of the budget on
   internal thinking, so a low cap silently truncates/empties output.

### Question-bank JSON shape (real schema, shared by `visual_qa` and `reasoning`)

```json
{
  "name": "scene_qa",
  "version": "1.0",
  "questions": [
    {
      "id": "q_activity",
      "question": "What is the primary activity in the scene?",
      "options": ["A. Loading", "B. Unloading", "C. Idle", "D. Transit"],
      "aggregation": "any"
    }
  ]
}
```

Wired in via config, not inline:

```yaml
stage_args:
  captioning: --prompt-file ../prompts/dense_caption/scene_prompt.md
visual_qa:
  question_bank_file: question_bank.json
```

### Guardrails

- Evidence first, labels second — the model observes and describes before classification.
- Don't make output schema a prompt concern; the `reasoning` stage owns DAFT `task/` output shapes.
- Explicit rules against inferring identity, intent, fault, protected attributes, or hidden cause.
- Video prompts require timestamp-aware reasoning and track-ID grounding only when overlays are
  actually readable — don't assume track IDs are visible.
