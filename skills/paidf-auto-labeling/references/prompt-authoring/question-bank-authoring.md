# Question-Bank Authoring

Question banks are shared evidence contracts for the `visual_qa` and `reasoning`
stages through the field `visual_qa.question_bank_file`. Keep one bank per
scenario when the same evidence supports multiple task outputs.

## Expected Sections

Use these top-level arrays so validation and the `reasoning` stage can route each
family:

- `questions`: visual-question items used by VQA-style evidence generation.
- `open_qa`: free-form questions for DAFT `task/` output, written by the
  `reasoning` stage.
- `mcq_openended`: multiple-choice questions with options and evidence.
- `bcq_openended`: binary-choice questions with evidence.
- `temporal_localization`: queries that require start/end time evidence.
- `causal_linkage`: apparent event-pair or cause/effect questions only when the
  passage provides visible start/end or causal evidence; forbid inference beyond
  what is presented.

## Item Quality Rules

- Ask only questions that can be answered from pixels, captions, tracks, or
  accumulated scene evidence.
- Include an explicit unknown/not visible path when evidence can be occluded.
- Prefer visibility gates over forced answers. For example, only ask PPE color
  if the worker is visible enough to inspect clothing.
- Keep option sets mutually exclusive and collectively useful; avoid overlapping
  choices like `unsafe` and `possibly unsafe` unless the schema defines a
  confidence field.
- Use stable IDs and concise wording. Avoid domain jargon unless the prompt also
  teaches the visual cue.
- Do not encode private or protected attributes.

## Migration Pattern

When porting an older scenario, preserve the bank's domain coverage but remove
runtime-only settings. The new cookbook should reference the bank from
`visual_qa.question_bank_file`; the `reasoning` stage reads the same file through
its config sections. Do not reference the legacy `mcq_generation` question-bank
path, and do not fork identical question files for `visual_qa` and `reasoning`.
