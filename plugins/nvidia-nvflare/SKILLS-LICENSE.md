# NVIDIA FLARE skills licensing

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The NVIDIA FLARE skills and skills plugin use `Apache-2.0 AND CC-BY-4.0`:
Apache 2.0 applies to code, and Creative Commons Attribution 4.0 International
applies to documentation and other skill content. Both license texts accompany
the plugin: [Apache 2.0](LICENSE) and [CC BY 4.0](LICENSE-CC-BY-4.0).

## Scope

This policy covers NVIDIA-authored material in `skills/` and the corresponding
copies in `plugins/nvidia-nvflare/skills/`, including future additions. It also
covers the plugin-specific `PLUGIN.md` (distributed as `README.md`), plugin cards,
and plugin evaluation task content under `evals/` and in `regression-evals.json`
at the repository and plugin roots.

- **Apache-2.0:** executable source code, scripts, code templates, configuration,
  and code examples, including code snippets embedded in documentation. Code-only
  files keep their Apache-2.0 license identifiers.
- **CC-BY-4.0:** skill instructions, reference prose, tutorials, illustrative
  examples other than code, evaluation prompts and grading references, and
  NVIDIA-authored example datasets supplied with the skills or plugin.
- **Mixed files and packages:** use the metadata expression
  `Apache-2.0 AND CC-BY-4.0`; the respective license applies to each portion as
  described above. This includes `SKILL.md` files combining instructions and code.

The rest of the NVFlare repository, including SDK code and the repository-wide
`examples/` directory, retains its existing licensing. This policy does not
change third-party license terms or existing notices and attributions, and does
not license external datasets, models, user inputs, or unrelated generated
outputs. Preserve and follow any applicable third-party terms.

## Attribution and contributions

When sharing CC-BY-4.0 material, retain supplied attribution and copyright
notices, identify modifications, and include the license text or a link to it.
Identify the source as NVIDIA FLARE and retain the source file path and repository
reference where supplied: <https://github.com/NVIDIA/NVFlare>.

New skills and their examples follow this same code/content division. Retain
third-party notices, identify any exceptions explicitly, and follow the
repository's [contribution requirements](CONTRIBUTING.md).

Licensing changes do not establish evaluation or signing status. Generated
cards, benchmarks, and signatures must identify and verify the content of the
release they describe.
