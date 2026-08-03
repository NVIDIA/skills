# `hsb-ip-def` — HSB IP `HOLOLINK_def.svh` Skill

An Agent Skill following the [agentskills.io specification](https://agentskills.io/specification) for generating, validating, and explaining `HOLOLINK_def.svh` configuration headers for the NVIDIA Holoscan Sensor Bridge (HSB) FPGA IP.

This skill is **self-contained**: drop the `hsb-ip-def/` directory into any repo and it works without external example files or HSB IP source access at runtime.

## What this skill provides

- **Validator** (`scripts/validate_def.py`) — lints a `HOLOLINK_def.svh` against the rules in `references/validation-rules.md`. Emits structured JSON or human-readable text.
- **Generator** (`scripts/generate_def.py`) — emits a valid `HOLOLINK_def.svh` from one of 6 archetypes plus a YAML/JSON profile of overrides. Refuses to emit if the result fails validation.
- **Comparer** (`scripts/compare_defs.py`) — semantic diff between two `HOLOLINK_def.svh` files; ignores whitespace, comments, and equivalent syntactic forms.
- **References** (`references/*.md`) — authoritative per-macro spec, validation rule catalog, archetype samples, init_reg cookbook, port map, and advanced-macro coverage.
- **Bundled example metadata** (`assets/metadata/*.json`) — anonymized records used to maintain the example library.
- **`SKILL.md`** — the entrypoint the assistant reads when this skill activates. Contains the trigger phrases and workflow definitions.

## Quick start

```bash
# Validate an existing file
scripts/validate_def.py path/to/HOLOLINK_def.svh --text

# Generate a new file from an archetype
scripts/generate_def.py --archetype mid-bandwidth-baseline \
    --profile my-profile.yaml -o HOLOLINK_def.svh

# Diff two files semantically
scripts/compare_defs.py a.svh b.svh --text
```

The 6 archetypes:

- `high-bandwidth-single-sensor` — ≥200 MHz HIF, 512+ bit datapath, dual host
- `mid-bandwidth-baseline` — 64-bit datapath, dual host, 156.25 MHz HIF
- `gigabit-baseline` — 8-bit host @ 125 MHz (1 GbE), single sensor each direction
- `minimal-gateway` — 1 sensor, 1 host, no TX, 1500 MTU
- `ultra-minimal` — 8-bit datapath, TX-only
- `very-high-speed` — 322 MHz HIF, 512-bit datapath, single sensor

## Requirements

- Python >= 3.9
- `PyYAML` (system-installed in the original build environment; otherwise `pip install pyyaml`)

## Repository layout

```
hsb-ip-def/
├── SKILL.md                # entrypoint — frontmatter + 3 workflows
├── README.md               # this file
├── references/             # authoritative reference docs (loaded on demand)
├── assets/metadata/        # bundled anonymized example records + stats
└── scripts/                # validator, generator, comparer + lib/
```

## Refreshing the example metadata

The bundled `assets/metadata/corpus.json` and `assets/metadata/corpus-stats.json` are produced by `scripts/build_corpus_metadata.py` from a curated set of example `HOLOLINK_def.svh` files. They ship inside the skill as static data — **runtime usage of the skill never re-touches the source files**. The metadata is for maintaining the example library, not for warning that a legal user value differs from an example.

Re-run only when the example set intentionally changes:

```bash
scripts/build_corpus_metadata.py \
    /path/to/project1/top/HOLOLINK_def.svh \
    /path/to/project2/top/HOLOLINK_def.svh \
    ...
# regenerated assets/metadata/corpus.json + assets/metadata/corpus-stats.json
git add assets/metadata/ && git commit -m "Refresh example metadata"
```

The script anonymizes on the way in: UUIDs are redacted, project names and paths are stripped, only macro values survive.

## Documentation

| File | Purpose |
|---|---|
| [SKILL.md](SKILL.md) | Skill entrypoint — workflow definitions and trigger phrases |
| [references/macro-reference.md](references/macro-reference.md) | Authoritative per-macro spec with RTL line citations |
| [references/validation-rules.md](references/validation-rules.md) | Stable rule IDs (HD-Exxx/Wxxx/Ixxx) emitted by the validator |
| [references/archetypes.md](references/archetypes.md) | 6 archetypes with synthetic sample defs |
| [references/init-reg-cookbook.md](references/init-reg-cookbook.md) | `init_reg[]` address-space conventions and templates |
| [references/top-port-map.md](references/top-port-map.md) | `HOLOLINK_top` ports grouped by gating macro |
| [references/advanced-macros.md](references/advanced-macros.md) | Advanced IP macros (CDC tightening, RAM overrides, feature disables) |

## Public-doc baseline

The skill is written against the public HSB user guide:

- [docs/user_guide/ip_integration.md](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/release-2.6.0-EA/docs/user_guide/ip_integration.md)
- [docs/user_guide/port_description.md](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/release-2.6.0-EA/docs/user_guide/port_description.md)

Where the bundled references conflict with the public docs, follow the bundled references — conflicts are noted inline.

## License

Apache-2.0.
