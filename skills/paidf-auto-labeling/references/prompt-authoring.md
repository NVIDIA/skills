# PAIDF Auto-Labeling Prompt Authoring

Use this skill when a task needs high-quality prompts or question banks for a
PAIDF auto-labeling cookbook.

## Workflow

1. Classify the prompt type: image caption, video dense caption, VQA evidence,
   or LLM evidence aggregation.
2. Read [prompt-patterns.md](prompt-authoring/prompt-patterns.md) for evidence-first
   prompt structure and anti-fabrication rules.
3. Read [domain-evidence-checklists.md](prompt-authoring/domain-evidence-checklists.md)
   for the target use case before writing domain-specific language.
4. Read [question-bank-authoring.md](prompt-authoring/question-bank-authoring.md)
   before changing a question bank shared by the `visual_qa` and `reasoning`
   stages (field `visual_qa.question_bank_file`).
5. Put prompts under the cookbook `prompts/` tree and reference them through
   `workflow.nodes.<node>.args`, not inline shell commands. `stage_args` remains
   accepted only for compatibility.
6. Validate path references with the asset checks in the workflow-stage skill.

## Examples

Evidence-first dense-caption prompt (`prompts/dense_caption/scene_prompt.md`):

```markdown
You are annotating a video clip. Describe only what is visible.
- List visible actors, objects, and actions with approximate timestamps.
- Ground references to track IDs only when overlays are readable.
- If something is ambiguous, say "uncertain"; never infer intent, fault, or
  protected attributes.
```

Question bank entry (`question_bank.json`; real schema, shared by `visual_qa`
and `reasoning`):

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

Wire both through cookbook fields / `workflow.nodes.<node>.args`, never inline
shell (`stage_args` is compatibility-only):

```yaml
workflow:
  nodes:
    captioning:
      stage: captioning
      args: [--prompt-file, ../prompts/dense_caption/scene_prompt.md]
visual_qa:
  question_bank_file: question_bank.json
```

## Guardrails

- Write prompts for visible evidence, not labels first. The model should observe
  and describe before any classification is requested.
- Do not make output schemas a prompt concern by default. The `reasoning` stage
  owns DAFT `task/` output schemas (the single `daft_export` stage is retired);
  use structured captioning output only when the user or service contract
  explicitly asks for it.
- Include uncertainty handling and explicit rules against identity, intent, fault,
  protected-attribute, and hidden-cause inference.
- For video prompts, require timestamp-aware temporal reasoning and visible track
  ID grounding only when overlays are readable.
- Keep prompt files reusable. Domain details belong in the prompt; experiment
  paths, endpoints, and secrets belong in runtime config.
- When a prompt or question bank targets a **reasoning-capable model** (for
  example Gemini 3 Flash), note in the cookbook that `max_tokens` for the
  `reasoning` and `visual_qa` LLM substages must be raised to a generous positive
  value (e.g. `32768`). Reasoning models spend part of the budget on internal
  thinking, so a low cap truncates or empties otherwise well-authored outputs.
