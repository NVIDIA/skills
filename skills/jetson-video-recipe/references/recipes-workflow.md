# Encoder recipe workflow

Use this workflow to turn one workload intent into a deterministic,
surface-neutral encoder recipe. Planning and structural validation do not use
media and never launch a codec operation.

## Contents

- [Owner and CLI](#owner-and-cli)
- [Intent contract](#intent-contract)
- [Recipe contract](#recipe-contract)
- [Workflow](#workflow)
- [Live compatibility](#live-compatibility)
- [Handoffs](#handoffs)

## Owner and CLI

The canonical recipe engine is
`jetson-video-recipe/scripts/recipes/recipe_model.py`, owned by this recipe
skill. Invoke its public CLI directly from the canonical installed skill path:

```text
python3 -I <recipe-skill>/scripts/recipes/recipe_model.py --help

python3 -I <recipe-skill>/scripts/recipes/recipe_model.py \
  plan --intent intent.json --output nvcodec-recipe.json

python3 -I <recipe-skill>/scripts/recipes/recipe_model.py \
  validate --recipe nvcodec-recipe.json

python3 -I <recipe-skill>/scripts/recipes/recipe_model.py \
  check-live --recipe nvcodec-recipe.json \
  --surface native --output recipe-live-native.json

python3 -I <recipe-skill>/scripts/recipes/recipe_model.py \
  check-live --recipe nvcodec-recipe.json \
  --environment nvcodec-environment.json \
  --capability-report nvcodec-encoder-capability.json \
  --surface pynvc --output recipe-live-pynvc.json
```

`--surface` accepts exactly `native` or `pynvc`. `--environment` is optional for
both live checks. With no environment, an exact projection remains valid but
its live classification is honestly `unknown` and includes non-mutating
remediation to `jetson-video-setup`. This does not change `plan` or `validate`.
If setup is not installed, tell the user to install that skill.

When an environment is supplied, the native check validates that selected
surface and never requires a capability report; its authenticated AppEncCuda
operation remains deferred. The Py check does not require a separate report:
the `capabilities` block of the schema-1.2 environment is the established
encoder-capability authority and is fully functional on its own.
`--capability-report` is an optional Py-only refinement — omit it and the
environment's own block is used; supply it only with a fresh environment to
which it is bound. A report supplied without an environment cannot establish
selected-surface readiness or selected-GPU identity. Omit it for native.
Use `--buffer-mode cpu` or `--buffer-mode gpu` for the selected PyNv execution
path; omission preserves the `gpu` default and its Torch requirement.
`check-live` exits `0` for a compatible projection or a native `unknown` whose
authenticated operation remains deferred, `2` for an unsupported projection or
other unresolved live evidence, and `3` for malformed input. Never describe the
native exit-`0` deferred case as compatible or ready.

This skill owns `scripts/recipes/recipe_model.py` and its catalog. Do not invoke
the engine from a copied path, insert another scripts directory into
`PYTHONPATH`, import it from a sibling skill, or recreate it in a consumer. A
missing entry point is an incomplete recipe-skill installation.

## Intent contract

The intent is a strict JSON object. Require only `use_case`, positive integer
`width`, and positive integer `height`. `use_case` is `conferencing`,
`live_streaming`, `vod`, `archival`, or `lossless`.

Leave omitted fields to the resolver: `codec` defaults to `h264`, `format` to
`NV12`, `fps` to `30`, GPU to `0`, and `output_requirement` to
`elementary_stream`. `frame_count` (or its `frames` alias) is optional for
planning but must be an exact positive integer before execution. Include only
caller-specified profile, preset, tuning, rate-control, bitrate, maximum
bitrate, VBV, GOP, B-frame, lookahead, AQ, temporal-AQ, multipass, CQ, or
constant-QP controls.

An unqualified “low latency” request does not select a use case. Ask whether it
means conferencing, live streaming, or another latency contract before
defaulting controls.

Recipe planning is media-free. Do not ask for a path or URL, probe media,
retrieve content, or create raw frames to produce this object.

## Recipe contract

The output is one `schema_version: "2.0"`, `kind: "nvcodec-recipe"` object. It
contains:

- the canonical `encoder_intent`;
- exact native and PyNvVideoCodec projections;
- per-surface projection status and any projection losses;
- caller-supplied and defaulted values with provenance;
- assumptions and rationale;
- a deterministic catalog identity.

The native and Python projections are representations of the same intent, not
independent recipes. Do not delete a requested control merely to make the
projections look identical.

Validation reconstructs the recipe from its recorded intent and catalog and
requires exact deterministic equality. Regenerate instead of editing the
artifact in place.

## Workflow

1. Normalize the user request into one intent JSON.
2. Run the public CLI's `plan` subcommand to create a fresh output.
3. Run its `validate` subcommand on the exact resulting file.
4. Return the recipe identity, assumptions, defaults, native projection, Python
   projection, and any unrepresentable controls.
5. Stop. Do not select media, build samples, encode, decode, benchmark, or
   claim the configuration ran.

Follow [recipes-knobs-and-constraints.md](recipes-knobs-and-constraints.md) for
accepted values, parser spellings, integer bounds, and surface limitations.

## Live compatibility

`check-live` can run with no setup evidence. It still replay-validates the exact
recipe, then returns `classification: unknown` plus a `jetson-video-setup`
remediation for an otherwise exact projection. It performs no probe, install,
repair, surface discovery, or recipe mutation. A capability report supplied by
itself does not resolve that result. A structurally unrepresentable projection
remains `unsupported` independently of live evidence.

- When supplied, `--environment` is the setup-probe environment, exactly
  `kind: "nvcodec-environment"`, `schema_version: "1.2"`, and `mode: "live"`.
  Read from it: `selected_gpu` and the matching `nvidia_smi.gpus[]` record for
  GPU identity, the selected native or Py surface, and — when Py is selected —
  Python readiness from
  `installation.python.packages.<name>.{status, version, requirement_satisfied}`,
  where `torch` additionally carries `cuda_build`, `cuda_available`, and
  `sample_readiness`.
- `--capability-report` is the optional capability-owned Py refinement,
  `kind: "nvcodec-encoder-capability-report"`, `schema_version: "1.0"` report.
  When one is supplied, encoder evidence is read from it: `authority`, `gpu`,
  `evidence_classification.encode`, `linked_nvenc_api`, `encode.<codec>`, and
  `documentation_crosscheck`. When it is omitted, that evidence comes from the
  environment's own established `capabilities` block, which is fully functional
  on its own.
  Its absence never fails a check. When present, it requires the environment
  and affects only `pynvc`; omit it for `native`.

Every supplied artifact is mandatory to validate and is never silently ignored
or replaced with an evidence-free fallback. Strict JSON or recipe input errors
fail as input errors; unavailable, wrong-schema, stale, mismatched, or incomplete
live authority remains closed as `unknown` or `unsupported` according to the
explicit evidence. In particular, a supplied invalid capability report does
not fall back to the environment's `capabilities` block.

A report is valid only for the environment it was produced from. It records
that artifact's path, size, raw-byte SHA-256, canonical-JSON SHA-256, schema,
mode, selected GPU, lexical interpreter, and authenticated Py identity. Each
field is compared in its own domain against the supplied environment; any
mismatch fails closed as `unknown`. This rejects an environment/report swap
and any change to the environment between report production and consumption.

Codec/API support and installation readiness are different questions with
different owners; both are in scope for a live compatibility check. An explicit
negative dependency readiness is `unsupported`; missing or malformed dependency
evidence is `unknown`. Both include non-mutating remediation to
`jetson-video-setup` when selected-surface prerequisites need setup work. If the
setup skill is absent, tell the user to install it.

`GetEncoderCaps` evidence is `status: "capability_reported"` with
`supported: null` and `operation_status: "not_tested"`; `capability_reported` is
never support. Final-answer precedence is API query, then official operation,
then NVIDIA documentation: an applicable `documentation_crosscheck.support` of
`No` is the final `unsupported` verdict for any codec and outranks a positive
query, and the discrepancy is recorded alongside it.

Pass artifacts as data; never import the producing skill's implementation. The
command classifies one projection from exact fields:

- `compatible`: requested controls are expressible and required positive fields
  are present;
- `unsupported`: an exact required field is explicitly negative or the surface
  cannot represent a requested control;
- `unknown`: authority, field, or current evidence is missing.

Compatibility is not readiness or operation proof. The later pipeline must
authenticate and run the selected official sample and independently decode the
exact output before claiming an encode succeeded.

Evaluate native and PyNv projections independently. For explicit `both`,
preserve both outcomes and every blocked reason.

## Handoffs

- `jetson-video-setup` optionally supplies the schema-1.2 environment artifact
  that resolves otherwise unknown selected-surface readiness.
  `jetson-video-capability` owns the optional encoder capability report that
  `check-live` may additionally consume, plus authenticated documentation
  evidence. A
  recommendation consumes those results when it needs a support statement.
- `jetson-video-benchmark` consumes the exact validated recipe artifact and
  holds all non-compared controls constant.
- `jetson-video-pipeline` consumes the exact validated recipe artifact for
  official sample execution and artifact handoff verification.

Every consumer receives the original canonical recipe path, byte size, and
SHA-256 and rehashes it before use. Consumers exchange versioned artifacts or
public-CLI results; they do not import this skill. A copied summary or rewritten
projection is not a valid handoff.
