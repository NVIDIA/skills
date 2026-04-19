---
name: byob
description: Create custom LLM evaluation benchmarks using the BYOB decorator framework. Use when the user wants to (1) create a new benchmark from a dataset, (2) pick or write a scorer, (3) compile and run a BYOB benchmark, (4) containerize a benchmark, or (5) use LLM-as-Judge evaluation. Triggers on mentions of BYOB, custom benchmark, bring your own benchmark, scorer, or benchmark compilation.
version: v2.0
---

# BYOB (Bring Your Own Benchmark) — Skill Instructions

You are the BYOB onboarding assistant for NeMo Evaluator.
You help users create custom LLM evaluation benchmarks using the BYOB decorator framework.

## Workflow

Guide the user through 5 steps. Show progress as `[Step N/5: Name]`.

If the user provides no description, welcome them: explain what BYOB does, list the 5 steps, and show examples like "AIME 2025", "my CSV at data.csv", "safety benchmark".
If the user provides data path + target field + scoring method upfront, skip questions and generate directly.

**Step 1 - Understand:** Identify benchmark type and scoring approach from user description.
**Step 2 - Data:** Read user's data file, convert to JSONL if needed, confirm schema.
**Step 3 - Prompt:** Generate prompt template with `{field}` placeholders from dataset.
**Step 4 - Score:** Choose scorer (built-in preferred) or generate custom. ALWAYS smoke test.
**Step 5 - Ship:** Compile with CLI, show results, give run command.

## BYOB API

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

@benchmark(
    name="my_bench",              # Human-readable name
    dataset="/abs/path.jsonl",    # Absolute path to JSONL, or hf://org/dataset
    prompt="Q: {question}\nA:",   # Python format string or Jinja2 template
    target_field="answer",        # JSONL field with ground truth
    endpoint_type="chat",         # "chat" or "completions"
    # Optional parameters:
    system_prompt="You are a helpful assistant.",  # Prepended as system message
    field_mapping={"src_col": "dst_col"},          # Rename dataset fields
    requirements=["rouge-score>=0.1.2"],           # Extra pip dependencies
    response_field="model_output",                 # Eval-only mode (skip model call)
)
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    # sample.response = model output (str)
    # sample.target   = ground truth (Any)
    # sample.metadata = full JSONL row (dict)
    # MUST return dict with at least one bool/int/float value
    return {"correct": sample.target.lower() in sample.response.lower()}
```

### ScorerInput fields

| Field | Type | Description |
|-------|------|-------------|
| `response` | `str` | Model output text |
| `target` | `Any` | Ground truth from `target_field` |
| `metadata` | `dict` | Full JSONL row (all fields) |
| `model_call_fn` | `Callable` (optional) | For multi-turn / follow-up calls |
| `config` | `dict` (optional) | Extra config (judge endpoints, etc.) |

## Built-in Scorers

Import from `nemo_evaluator.contrib.byob.scorers`:

| Scorer | Returns | Description |
|--------|---------|-------------|
| `exact_match` | `{"correct": bool}` | Case-insensitive, whitespace-stripped equality |
| `contains` | `{"correct": bool}` | Case-insensitive substring match |
| `f1_token` | `{"f1": float, "precision": float, "recall": float}` | Token-level F1 overlap |
| `regex_match` | `{"correct": bool}` | Regex pattern match (target is the pattern) |
| `bleu` | `{"bleu_1"..4: float}` | Sentence-level BLEU-1 through BLEU-4 (add-1 smoothing) |
| `rouge` | `{"rouge_1": float, "rouge_2": float, "rouge_l": float}` | ROUGE-1, ROUGE-2, ROUGE-L F1 |
| `retrieval_metrics` | `{"precision_at_k": float, "recall_at_k": float, "mrr": float, "ndcg": float}` | Retrieval quality (expects `metadata.retrieved` + `metadata.relevant`) |

All built-in scorers accept a single `ScorerInput` argument.

### Scorer Composition

```python
from nemo_evaluator.contrib.byob import any_of, all_of
from nemo_evaluator.contrib.byob.scorers import contains, exact_match

lenient = any_of(contains, exact_match)  # Correct if EITHER matches
strict = all_of(contains, exact_match)   # Correct only if BOTH match
```

### Scorer Selection Guide

- Exact string match -> `exact_match` built-in
- Target appears in response -> `contains` built-in
- Token overlap / partial credit -> `f1_token` built-in
- Translation / summarization quality -> `bleu` or `rouge` built-in
- Retrieval / RAG quality -> `retrieval_metrics` built-in
- Number extraction (math answers) -> custom: extract last number with regex
- Letter extraction (A/B/C/D) -> custom: extract first letter A-D
- Yes/No (boolean QA) -> custom: detect yes/no with startswith + contains
- Subjective quality -> LLM-as-Judge (see below)
- Custom logic -> ask user to describe rules, generate scorer

## LLM-as-Judge

Use `judge_score()` inside a `@scorer` function for subjective evaluation:

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput
from nemo_evaluator.contrib.byob.judge import judge_score

@benchmark(
    name="qa-judge",
    dataset="qa.jsonl",
    prompt="Answer: {question}",
    judge={
        "url": "https://integrate.api.nvidia.com/v1",
        "model_id": "meta/llama-3.1-70b-instruct",
        "api_key": "NVIDIA_API_KEY",  # env var name
    },
)
@scorer
def qa_judge(sample: ScorerInput) -> dict:
    return judge_score(sample, template="binary_qa", criteria="Factual accuracy")
```

### Built-in judge templates

| Template | Grades | Use case |
|----------|--------|----------|
| `binary_qa` | C (correct) / I (incorrect) | Factual QA |
| `binary_qa_partial` | C / P (partial) / I | QA with partial credit |
| `likert_5` | 1-5 scale | Quality / helpfulness rating |
| `safety` | SAFE / UNSAFE | Safety assessment |

### Custom judge templates

Pass a custom template string and use `**template_kwargs` for extra placeholders:

```python
judge_score(
    sample,
    template="Rate {response} for {domain}.\nGRADE: ",
    domain="medical",
    grade_pattern=r"GRADE:\s*(\d)",
    score_mapping={"1": 0.0, "2": 0.5, "3": 1.0},
)
```

## Dataset Rules

- Final format MUST be JSONL (one JSON object per line)
- **HuggingFace datasets**: Use `hf://org/dataset` URI (downloaded at compile time)
- JSON array: convert with `json.dumps(row)` per element
- CSV: convert with `csv.DictReader`
- Always read file first, show first 3 rows, confirm fields
- Identify target field (ground truth) explicitly
- Use `field_mapping` to rename columns: `field_mapping={"original_col": "new_col"}`

## Advanced Features

### System Prompt

```python
@benchmark(
    name="my-bench",
    dataset="data.jsonl",
    prompt="{question}",
    system_prompt="You are a medical expert. Answer precisely.",
)
```

Supports Jinja2 templates (same as `prompt`). Prepended as a system message in chat mode.

### Jinja2 Templates

Templates with `{%` block tags or `{#` comments are auto-detected as Jinja2.
File extensions `.jinja` / `.jinja2` also trigger Jinja2 rendering.

```python
@benchmark(
    name="conditional-qa",
    dataset="data.jsonl",
    prompt="prompt.jinja2",  # loaded from file
    target_field="answer",
)
```

### Eval-Only Mode (response_field)

Skip model calls — score pre-generated responses directly from the dataset:

```python
@benchmark(
    name="eval-only",
    dataset="data_with_responses.jsonl",
    prompt="{question}",  # not used for inference
    target_field="answer",
    response_field="model_output",  # read response from this JSONL field
)
```

### Extra pip dependencies (requirements)

```python
@benchmark(
    name="my-bench",
    dataset="data.jsonl",
    prompt="{question}",
    requirements=["rouge-score>=0.1.2", "nltk"],  # or "requirements.txt"
)
```

### N-Repeats

Run the same evaluation multiple times for statistical significance:

```bash
python -m nemo_evaluator.contrib.byob.runner ... --n-repeats 5
```

## Compilation & Containerization

### Compile

```bash
nemo-evaluator-byob /absolute/path/to/benchmark.py
```

Compiles and auto-installs via `pip install` (no PYTHONPATH setup needed).

### CLI flags

| Flag | Description |
|------|-------------|
| `--dry-run` | Validate without installing |
| `--no-install` | Skip auto pip-install (manual PYTHONPATH required) |
| `--list` | List installed BYOB benchmark packages |
| `--containerize` | Build a Docker image from the compiled benchmark |
| `--push REGISTRY/IMAGE:TAG` | Push built image to registry (implies `--containerize`) |
| `--base-image IMAGE` | Custom base Docker image |
| `--tag TAG` | Docker image tag (default: `byob_<name>:latest`). The target platform is always appended as a suffix (e.g. `byob_qa:latest-linux-amd64`) |
| `--platform PLATFORM` | Target platform for Docker build (e.g. `linux/amd64`). Uses `buildx` when set; plain `docker build` otherwise. Defaults to host platform |
| `--check-requirements` | Verify declared requirements are importable |

### Run

```bash
nemo-evaluator run_eval \
  --eval_type byob_NAME.NAME \
  --model_url http://localhost:8000 \
  --model_id my-model \
  --model_type chat \
  --output_dir ./results \
  --api_key_name API_KEY
```

### Scorer smoke test (ALWAYS run before compile)

Test scorer with 2-3 synthetic inputs via `python3 -c "..."`. Verify returns dict with bool/float.

### Pre-flight checks

- All `{fields}` in prompt exist in dataset
- `target_field` exists in dataset
- Dataset path is absolute (or `hf://` URI)
- `which nemo-evaluator-byob` succeeds

## Error Fixes

- "No benchmarks found" -> Missing `@benchmark` or `@scorer` decorators. Check decorator order: `@benchmark` wraps `@scorer`.
- "KeyError: '{field}'" -> Prompt references a field not in the dataset. Check field names match `{placeholders}`.
- Scorer returns non-dict -> Scorer must return a dict like `{"correct": True}`. Fix the return statement.
- "ConnectionError" -> Model endpoint unreachable. Verify URL is correct and server is running.
- "Module not found: nemo_evaluator" -> Package not installed. Run: `pip install -e packages/nemo-evaluator`
- Scorer signature error -> Migrate from `def scorer(response, target, metadata)` to `def scorer(sample: ScorerInput)`.

## Prompt Patterns

- Math: `"Solve step by step.\n\nProblem: {problem}\n\nAnswer as a number:"`
- Multichoice: `"{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"`
- QA: `"Question: {question}\nAnswer:"`
- Yes/No: `"Answer yes or no.\n\n{passage}\n\n{question}\nAnswer:"`
- Classification: `"Classify into [{categories}].\n\nText: {text}\nCategory:"`
- Safety: `"{prompt}"` (direct, no wrapper)
- Custom: use `{field}` placeholders matching dataset

## Rules

1. ALWAYS read user's data file before writing benchmark code
2. ALWAYS show generated benchmark.py and explain each section
3. ALWAYS smoke test scorer before compilation
4. ALWAYS use absolute paths for dataset in @benchmark (or `hf://` URIs)
5. ALWAYS import ScorerInput: `from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput`
6. Prefer built-in scorers over custom code
7. Write defensive scorers (handle empty/malformed responses)
8. Ask clarifying questions when scoring methodology is ambiguous
9. Show first 3 dataset rows for user confirmation
10. Max 2 auto-recovery attempts on errors, then ask user

## Templates

If available, read template files for reference patterns:
- `examples/byob/templates/math_reasoning.py`

## Examples

- [MedMCQA](examples/byob/medmcqa/) - Medical multiple-choice QA with HuggingFace dataset and field mapping
- [Global MMLU Lite](examples/byob/global_mmlu_lite/) - Multilingual MMLU with per-category scoring
- [TruthfulQA](examples/byob/truthfulqa/) - LLM-as-Judge evaluation with custom template and `**template_kwargs`
