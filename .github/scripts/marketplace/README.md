# Skill Metadata Generation

This repo generates two catalog-wide metadata files:

- `.github/scripts/marketplace/metadata.json` — canonical inventory of every
  catalog skill plus its taxonomy assignments (`product.primary`,
  `classification.category.primary`, `catalog.subdomain`, `audience`,
  `discovery.activity_tags`).
- `skills.sh.json` (repo root) — marketplace/index file derived from
  `metadata.json`, grouping skills by `catalog.subdomain`.

Both files are produced by
[`generate-skill-metadata.py`](./generate-skill-metadata.py) (this directory).

## Authoring rules

All marketplace artifacts (generated outputs, schemas, and the subdomain
config) live under `.github/scripts/marketplace/`:

- The generator never rewrites `SKILL.md` content. Skill `name` and
  `description` are copied verbatim from each `SKILL.md` frontmatter.
- All controlled values come from
  [`metadata.schema.json`](./metadata.schema.json). The schema is the single
  source of truth and is consumed directly at runtime; there is no derived
  taxonomy companion file to keep in sync.
- [`skills-sh.schema.json`](./skills-sh.schema.json) describes the structural
  contract for the generated `skills.sh.json`.
- Subdomain group titles, descriptions, and ordering live in
  [`skills-subdomains.json`](./skills-subdomains.json).
- Skills can be temporarily withheld from both outputs by adding them to
  [`metadata-exclusions.yaml`](./metadata-exclusions.yaml).
- Skill-to-product mapping is deterministic only when a skill is declared in
  one of the synced `components.d/*.yml` registries. Skills outside that set
  (e.g. catalog-only entries staged via direct PR) fall through to AI
  enrichment for `product.primary` like any other skill missing a mapping.

## Running the generator locally

Install the generator's dependencies (PyYAML, jsonschema, requests):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install pyyaml jsonschema requests
```

### `--check` (CI / pre-commit)

Verifies the checked-in `metadata.json` and `skills.sh.json` are byte-stable
against the current skill tree and pass schema/taxonomy validation. AI is
disabled in this mode, so any skill missing required fields is a hard fail
that must be fixed by regenerating.

```bash
python3 .github/scripts/marketplace/generate-skill-metadata.py --check --no-ai
```

### `--write` (default)

Regenerate and write the outputs. AI enrichment is used for skills that lack
required fields after deterministic carry-forward. Set the inference
configuration in the environment before running:

```bash
export INFERENCE_API_KEY=...        # secret in CI; .env locally
export INFERENCE_MODEL=...          # required when AI runs (no default)
export INFERENCE_API_URL=...        # optional; defaults to NVIDIA inference API

python3 .github/scripts/marketplace/generate-skill-metadata.py
```

### `--no-ai`

Refuse to call the inference API; fail with actionable errors when any skill
needs enrichment. Use this when iterating on the generator itself or when you
want a clean deterministic re-emit of an already-enriched catalog.

```bash
python3 .github/scripts/marketplace/generate-skill-metadata.py --no-ai
```

### `--report-only`

Print the change classification (added / removed / renamed / materially
changed / excluded / unchanged) against the checked-in `metadata.json` and
exit. Does not write or validate.

```bash
python3 .github/scripts/marketplace/generate-skill-metadata.py --report-only
```

## CI behavior

Defined in
[`.github/workflows/generate-skill-metadata.yml`](../../workflows/generate-skill-metadata.yml).

| Trigger | Action | AI? |
| --- | --- | --- |
| `pull_request` (paths: `skills/**`, metadata config, generator) | `--check --no-ai`; fails on drift or schema/taxonomy errors. | No |
| `workflow_dispatch` | Regenerate; open a PR if outputs changed. | Yes |
| `workflow_run` after `Sync Skills from Product Repos` | Regenerate; open a PR if outputs changed. | Yes |

On validation failure, a single tracking issue
(`[skill-metadata] generator validation failed`) is opened or updated with
the generator log, the workflow-run link, and `@jim-NVIDIA` /
`@jasonNVIDIA` mentioned. The next successful run closes the issue.

## Required repository configuration

| Kind | Name | Purpose |
| --- | --- | --- |
| Secret | `INFERENCE_API_KEY` | NVIDIA Inference API token used by the AI enrichment client. |
| Variable | `INFERENCE_MODEL` | Model slug passed to the inference API. The generator fails fast if this is unset and AI is needed. |
| Variable | `INFERENCE_API_URL` | (Optional) Override the inference endpoint. Defaults to `https://inference-api.nvidia.com/v1/chat/completions`. |

Locally the same names are read from `.env` (see `.env.example` if added) or
from the shell environment. `.env` is gitignored.

## How AI enrichment is constrained

The generator calls the inference API in one of two narrow modes per skill:

1. **Fill mode** — used for new skills (no baseline metadata) or skills whose
   carried-forward metadata is missing required fields. The model is asked to
   pick a value from the controlled vocabulary for each missing field.
2. **Amend mode** — used when a skill is `materially_changed` (its `name` or
   `description` in `SKILL.md` differs from the baseline) but the existing
   metadata still validates. The model is shown each existing value and asked
   whether to keep it (return verbatim) or change it because the new content
   warrants a different controlled value. The prompt explicitly biases toward
   preserving existing values; only clear mismatches should be amended.

Skills classified as `unchanged` never trigger an AI call — their metadata
flows through byte-identically. Running with `--no-ai` disables both modes;
in that case `materially_changed` skills keep their existing metadata as-is
(only `name` / `description` update) and any `added` skill that needs fill
mode causes a hard error.

Every request is bounded to a single skill and includes:
- the skill path, name, description, and frontmatter,
- matched `components.d/` data,
- the first 4 KiB of `skill-card.md` if present,
- the explicit list of requested fields and allowed values per field,
- (amend mode only) the existing value for each requested field.

The model must return a single JSON object whose keys are exactly the
requested fields. Any unexpected key, missing field, non-string value, or
`UNRESOLVED` value causes a hard validation failure. The generator never
writes a value that is not in the schema's controlled vocabulary; jsonschema
validation runs on every output before files are written.

## Adding a new subdomain

1. Add the slug to `catalog.subdomain.enum` in
   [`metadata.schema.json`](./metadata.schema.json).
2. Add a matching entry to
   [`skills-subdomains.json`](./skills-subdomains.json) with a `title` and
   `description`.
3. Re-run the generator to assign or migrate skills as needed.

## Adding a new product enum

1. Add the canonical product display name to `product.primary.enum` in the
   schema.
2. If a `components.d/<product>.yml` exists, ensure its `name:` matches the
   new enum entry exactly. The generator deterministically maps a component
   to `product.primary` only when the names are identical.
3. Re-run the generator.
