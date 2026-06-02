# `metadata.json` `product.primary` vs `components.d/` `name:` audit

Auto-generated. One row per skill in `metadata.json`. Shows the actual
`product.primary` value vs the `name:` declared in whichever yml file
claims the skill. Sources: 110 metadata.json entries, 32 components.d files, schema enum size 343.

Status legend:
- `match` — yml `name:` equals `product.primary` exactly
- `mismatch` — yml claims the skill but `name:` differs from `product.primary`
- `not in any yml` — catalog-only skill (no components.d entry)
- `multi-yml` — claimed by more than one yml file (rare)

Concern column legend:
- empty — no concern; the `product.primary` value looks correct for this skill
- `wrong: …` — metadata.json picked a substantively wrong enum value; needs SME review
- `schema gap: …` — the right product name isn't in the schema enum yet; both files reasonable

| skill | `product.primary` (metadata.json) | claiming yml | yml `name:` | status | concern |
| --- | --- | --- | --- | --- | --- |
| `skills/tilegym-adding-cutile-kernel` | `"CUDA Tile"` | `tilegym.yml` | "TileGym" | **mismatch** | schema gap: `TileGym` missing from enum |
| `skills/cudaq-guide` | `"CUDA-Q"` | `cuda-q.yml` | "CUDA-Q" | match |  |
| `skills/accelerated-computing-cudf` | `"cuDF"` | `cudf.yml` | "cuDF" | match |  |
| `skills/cuopt-developer` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-install` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-numerical-optimization-api-c` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-numerical-optimization-api-cli` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-numerical-optimization-api-python` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-numerical-optimization-formulation` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-routing-api-python` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-routing-formulation` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-server-api-python` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-server-common` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-skill-evolution` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cuopt-user-rules` | `"cuOpt"` | `cuopt.yml` | "cuOpt" | match |  |
| `skills/cupynumeric-hdf5` | `"cuPyNumeric"` | `cupynumeric.yml` | "cuPyNumeric" | match |  |
| `skills/cupynumeric-install` | `"cuPyNumeric"` | `cupynumeric.yml` | "cuPyNumeric" | match |  |
| `skills/cupynumeric-migration-readiness` | `"cuPyNumeric"` | `cupynumeric.yml` | "cuPyNumeric" | match |  |
| `skills/cupynumeric-parallel-data-load` | `"cuPyNumeric"` | `cupynumeric.yml` | "cuPyNumeric" | match |  |
| `skills/dali-dynamic-mode` | `"DALI"` | `dali.yml` | "DALI" | match |  |
| `skills/deepstream-dev` | `"DeepStream SDK"` | `deepstream.yml` | "DeepStream" | **mismatch** |  |
| `skills/deepstream-import-vision-model` | `"DeepStream SDK"` | `deepstream.yml` | "DeepStream" | **mismatch** |  |
| `skills/dynamo-interconnect-check` | `"Dynamo"` | `dynamo.yml` | "Dynamo" | match |  |
| `skills/dynamo-recipe-runner` | `"Dynamo"` | `dynamo.yml` | "Dynamo" | match |  |
| `skills/dynamo-router-starter` | `"Dynamo"` | `dynamo.yml` | "Dynamo" | match |  |
| `skills/dynamo-troubleshoot` | `"Dynamo"` | `dynamo.yml` | "Dynamo" | match |  |
| `skills/earth2studio-data-fetch` | `"Earth2Studio"` | `earth2studio.yml` | "Earth2Studio" | match |  |
| `skills/earth2studio-deterministic-forecast` | `"Earth2Studio"` | `earth2studio.yml` | "Earth2Studio" | match |  |
| `skills/earth2studio-discover` | `"Earth2Studio"` | `earth2studio.yml` | "Earth2Studio" | match |  |
| `skills/earth2studio-install` | `"Earth2Studio"` | `earth2studio.yml` | "Earth2Studio" | match |  |
| `skills/holoscan-install-conda` | `"Holoscan"` | `holoscan-sdk.yml` | "Holoscan SDK" | **mismatch** |  |
| `skills/holoscan-install-container` | `"Holoscan"` | `holoscan-sdk.yml` | "Holoscan SDK" | **mismatch** |  |
| `skills/holoscan-install-debian` | `"Holoscan"` | `holoscan-sdk.yml` | "Holoscan SDK" | **mismatch** |  |
| `skills/holoscan-install-source` | `"Holoscan"` | `holoscan-sdk.yml` | "Holoscan SDK" | **mismatch** |  |
| `skills/holoscan-install-wheel` | `"Holoscan"` | `holoscan-sdk.yml` | "Holoscan SDK" | **mismatch** |  |
| `skills/holoscan-setup` | `"Holoscan"` | `holoscan-sdk.yml` | "Holoscan SDK" | **mismatch** |  |
| `skills/mcore-create-issue` | `"Megatron-Core"` | `megatron-core.yml` | "Megatron-Core" | match |  |
| `skills/mcore-linting-and-formatting` | `"Megatron-Core"` | `megatron-core.yml` | "Megatron-Core" | match |  |
| `skills/mcore-run-on-slurm` | `"Megatron-Core"` | `megatron-core.yml` | "Megatron-Core" | match |  |
| `skills/mcore-split-pr` | `"Megatron-Core"` | `megatron-core.yml` | "Megatron-Core" | match |  |
| `skills/mcore-testing` | `"Megatron-Core"` | `megatron-core.yml` | "Megatron-Core" | match |  |
| `skills/dicom-metadata-extract` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/dicom-series-preflight` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/dicom-series-to-volume` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-generate-ct-rflow` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-generate-mr` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-generate-mr-brain` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-generate-mr-brain-finetune` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-generate-vae-finetune` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-reason-cxr` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-segment-ct` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-segment-ct-finetune` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nv-segment-ctmr` | `"MONAI"` | `medical-ai-skills.yml` | "Medical AI Skills" | **mismatch** |  |
| `skills/nemo-data-designer-plugin` | `"NeMo"` | `nemo-platform.yml` | "NeMo Platform" | **mismatch** | wrong: `NeMo` is too generic; schema may lack a specific NeMo Data Designer value |
| `skills/nemo-evaluator-plugin` | `"NeMo"` | `nemo-platform.yml` | "NeMo Platform" | **mismatch** | wrong: should be `NeMo Evaluator` (more specific enum exists) |
| `skills/aiq-deploy` | `"NeMo Agent Toolkit"` | `aiq.yml` | "AIQ" | **mismatch** | wrong: SKILL.md says 'NVIDIA AI-Q Blueprint'; should be `AIQToolkit`, not `NeMo Agent Toolkit` |
| `skills/aiq-research` | `"NeMo Agent Toolkit"` | `aiq.yml` | "AIQ" | **mismatch** | wrong: SKILL.md says 'NVIDIA AI-Q Blueprint'; should be `AIQToolkit`, not `NeMo Agent Toolkit` |
| `skills/nemo-automodel-distributed-training` | `"NeMo Framework"` | `nemo-automodel.yml` | "NeMo AutoModel" | **mismatch** | schema gap: `NeMo AutoModel` missing from enum |
| `skills/nemo-automodel-launcher-config` | `"NeMo Framework"` | `nemo-automodel.yml` | "NeMo AutoModel" | **mismatch** | schema gap: `NeMo AutoModel` missing from enum |
| `skills/nemo-automodel-model-onboarding` | `"NeMo Framework"` | `nemo-automodel.yml` | "NeMo AutoModel" | **mismatch** | schema gap: `NeMo AutoModel` missing from enum |
| `skills/nemo-automodel-recipe-development` | `"NeMo Framework"` | `nemo-automodel.yml` | "NeMo AutoModel" | **mismatch** | schema gap: `NeMo AutoModel` missing from enum |
| `skills/nemo-mbridge-mlm-bridge-training` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-multi-node-slurm` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-activation-recompute` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-cpu-offloading` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-cuda-graphs` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-expert-parallel-overlap` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-hierarchical-context-parallel` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-megatron-fsdp` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-memory-tuning` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-moe-comm-overlap` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-moe-dispatcher-selection` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-moe-hardware-configs` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-moe-long-context` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-moe-optimization-workflow` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-moe-vlm-training` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-parallelism-strategies` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-sequence-packing` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-perf-tp-dp-comm-overlap` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-recipe-recommender` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-mbridge-resiliency` | `"NeMo Megatron Bridge"` | `nemo-mbridge.yml` | "NeMo MBridge" | **mismatch** |  |
| `skills/nemo-retriever` | `"NeMo Retriever"` | `nemo-retriever.yml` | "NeMo Retriever" | match |  |
| `skills/nemoclaw-user-agent-skills` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-configure-inference` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-configure-security` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-deploy-remote` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-get-started` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-manage-policy` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-manage-sandboxes` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-monitor-sandbox` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-overview` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemoclaw-user-reference` | `"NeMoClaw"` | `nemoclaw.yml` | "NemoClaw" | **mismatch** |  |
| `skills/nemotron-customize` | `"Nemotron"` | `nemotron.yml` | "Nemotron" | match |  |
| `skills/nemotron-retrieval-recipes` | `"Nemotron"` | `nemotron.yml` | "Nemotron" | match |  |
| `skills/digital-health-clinical-asr-build` | `"Nemotron for Digital Health"` | `digital-health-examples.yml` | "NVIDIA Digital Health Examples" | **mismatch** | wrong: ASR skill mapped to an LLM product; review (Riva / NeMo Speech?) |
| `skills/digital-health-clinical-asr-eval` | `"Nemotron for Digital Health"` | `digital-health-examples.yml` | "NVIDIA Digital Health Examples" | **mismatch** | wrong: ASR skill mapped to an LLM product; review (Riva / NeMo Speech?) |
| `skills/digital-health-clinical-asr-finetune` | `"Nemotron for Digital Health"` | `digital-health-examples.yml` | "NVIDIA Digital Health Examples" | **mismatch** | wrong: ASR skill mapped to an LLM product; review (Riva / NeMo Speech?) |
| `skills/digital-health-clinical-asr-setup` | `"Nemotron for Digital Health"` | `digital-health-examples.yml` | "NVIDIA Digital Health Examples" | **mismatch** | wrong: ASR skill mapped to an LLM product; review (Riva / NeMo Speech?) |
| `skills/physical-ai-neural-reconstruction` | `"NuRec"` | `manual-components.yml` | "Physical AI" | **mismatch** |  |
| `skills/omniverse-cad-to-simready` | `"Omniverse"` | `manual-components.yml` | "Physical AI" | **mismatch** |  |
| `skills/omniverse-realtime-viewer` | `"Omniverse"` | `manual-components.yml` | "Physical AI" | **mismatch** |  |
| `skills/omniverse-usd-performance-tuning` | `"Omniverse"` | `manual-components.yml` | "Physical AI" | **mismatch** |  |
| `skills/physical-ai-defect-image-generation` | `"Physical AI Dataset"` | `physical-ai-data-factory.yml` | "Physical AI" | **mismatch** |  |
| `skills/physical-ai-infrastructure-setup-and-resilient-scaling` | `"Physical AI Dataset"` | `manual-components.yml` | "Physical AI" | **mismatch** |  |
| `skills/physical-ai-video-data-augmentation` | `"Physical AI Dataset"` | `physical-ai-data-factory.yml` | "Physical AI" | **mismatch** |  |
| `skills/physicsnemo-discover` | `"PhysicsNeMo"` | `physicsnemo.yml` | "PhysicsNeMo" | match |  |
| `skills/rag-blueprint` | `"RAG"` | `rag-blueprint.yml` | "RAG Blueprint" | **mismatch** | schema gap: `RAG Blueprint` missing from enum |
| `skills/rag-eval` | `"RAG"` | `rag-blueprint.yml` | "RAG Blueprint" | **mismatch** | schema gap: `RAG Blueprint` missing from enum |
| `skills/rag-perf` | `"RAG"` | `rag-blueprint.yml` | "RAG Blueprint" | **mismatch** | schema gap: `RAG Blueprint` missing from enum |
| `skills/skill-card-generator` | `"Trustworthy AI"` | `skill-card-generator.yml` | "Skill Card Generator" | **mismatch** | wrong: `Trustworthy AI` is the SKILL.md author team, not the product |

## Skills declared in `components.d/` but missing from `metadata.json`

Could be excluded, in-flight, or awaiting AI enrichment.

| skill path (declared) | claiming yml | yml `name:` |
| --- | --- | --- |
| `skills/nemotron-speech` | `nemotron-speech.yml` | `"Nemotron Speech"` |
| `skills/tilegym-converting-cutile-to-julia` | `tilegym.yml` | `"TileGym"` |
| `skills/tilegym-converting-cutile-to-triton` | `tilegym.yml` | `"TileGym"` |
| `skills/tilegym-cutile-autotuning` | `tilegym.yml` | `"TileGym"` |
| `skills/tilegym-cutile-python` | `tilegym.yml` | `"TileGym"` |
| `skills/tilegym-improve-cutile-kernel-perf` | `tilegym.yml` | `"TileGym"` |
| `skills/tilegym-monkey-patch-kernels-to-transformers` | `tilegym.yml` | `"TileGym"` |
| `skills/vss-ask-video` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-deploy-dense-captioning` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-deploy-detection-tracking-2d` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-deploy-detection-tracking-3d` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-deploy-profile` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-deploy-video-embedding` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-generate-video-calibration` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-generate-video-report` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-manage-alerts` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-manage-video-io-storage` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-query-analytics` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-search-archive` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-setup-behavior-analytics` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-setup-video-analytics-api` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |
| `skills/vss-summarize-video` | `video-search-and-summarization.yml` | `"Video Search and Summarization"` |

## Tally

- match: 36 / 110
- mismatch: 74 / 110
- not in any yml (catalog-only): 0 / 110
- multi-yml: 0 / 110
- flagged as `wrong`: 9 / 110  (metadata.json value needs review)
- flagged as `schema gap`: 8 / 110  (schema enum missing the right value)


## Concerns: which mismatches look like the wrong _enum value_, not just the wrong _label_?

The 13 rename groups split into three categories. Use this to drive the team
discussion: not every mismatch should be fixed by changing `components.d`.

### A. `metadata.json` is canonical — fix `components.d` `name:` (6 groups, 32 skills)

These look right in `metadata.json`. The components.d label is just stale or
abbreviated. Trivial yml edits.

| `components.d/<file>` | current `name:` | should be | rationale |
| --- | --- | --- | --- |
| `deepstream.yml` | `"DeepStream"` | `"DeepStream SDK"` | Schema chooses the SDK-specific value; skills are about the SDK. |
| `holoscan-sdk.yml` | `"Holoscan SDK"` | `"Holoscan"` | Schema uses `"Holoscan"` as the umbrella enum; `"Holoscan SDK"` not in enum. |
| `nemo-mbridge.yml` | `"NeMo MBridge"` | `"NeMo Megatron Bridge"` | Pure abbreviation expansion. |
| `nemoclaw.yml` | `"NemoClaw"` | `"NeMoClaw"` | Casing fix only. |
| `physical-ai-data-factory.yml` | `"Physical AI"` | `"Physical AI Dataset"` | Skills are synthetic-dataset generation; `"Physical AI Dataset"` is the marketing name. |

### B. `metadata.json` looks WRONG — needs human verification (4 groups, 7 skills)

The `product.primary` value the generator (or upstream curation) landed on
doesn't actually fit the skill content. Needs a subject-matter call before
either side is changed.

| `components.d/<file>` | current `metadata.json` says | concern | suggested correct value |
| --- | --- | --- | --- |
| `aiq.yml` (`"AIQ"`) | `"NeMo Agent Toolkit"` | SKILL.md content explicitly says **"NVIDIA AI-Q Blueprint"**. Schema has both `"AIQToolkit"` and `"NeMo Agent Toolkit"` as **separate** products. AIQ is not NeMo. | `"AIQToolkit"` (or possibly `"AgentIQ"`) |
| `digital-health-examples.yml` (`"NVIDIA Digital Health Examples"`) | `"Nemotron for Digital Health"` | Skills are clinical **ASR** (audio speech recognition) — `digital-health-clinical-asr-{setup,build,eval,finetune}`. Nemotron is an LLM family; ASR is typically Riva or NeMo Speech. Mapping a speech skill to an LLM product is suspect. | TBD by team — likely a Riva or NeMo-Speech enum (may be a schema gap if no good fit exists) |
| `nemo-platform.yml` (`"NeMo Platform"`) | `"NeMo"` (too generic) | Skills are `nemo-evaluator-plugin` and `nemo-data-designer-plugin`. Schema has the more specific `"NeMo Evaluator"`. The generic `"NeMo"` loses information. | `"NeMo Evaluator"` for the evaluator skill; possible schema gap for Data Designer |
| `skill-card-generator.yml` (`"Skill Card Generator"`) | `"Trustworthy AI"` | The SKILL.md author email is `trustworthyaiprojects@nvidia.com`. The **author** is the Trustworthy AI team, but the skill itself is a generic governance card generator — it's not a Trustworthy AI product feature. AI enrichment likely confused author with product. | Schema gap — there's no fitting enum value; consider not mapping this catalog-utility skill or adding a `"Skill Card Generator"` enum |

### C. Schema is the lossy one (3 groups, 7 skills) — schema gap, not really anyone's fault

The components.d label is the more accurate product name, but the schema
enum doesn't contain it. Both files are doing the best they can. Fix is to
extend the schema if NVIDIA's official product taxonomy supports these.

| `components.d/<file>` | current `name:` | `metadata.json` settled on | gap |
| --- | --- | --- | --- |
| `nemo-automodel.yml` | `"NeMo AutoModel"` | `"NeMo Framework"` (broader) | Schema has no `"NeMo AutoModel"`. AutoModel is a real NeMo capability/product. |
| `rag-blueprint.yml` | `"RAG Blueprint"` | `"RAG"` (generic technique) | Schema has no `"RAG Blueprint"`. The skill is explicitly about NVIDIA's RAG Blueprint product, not generic RAG. |
| `tilegym.yml` | `"TileGym"` | `"CUDA Tile"` (the underlying tech) | Schema has no `"TileGym"`. CUDA Tile is the substrate; TileGym is the tool. |

### D. Borderline — `medical-ai-skills.yml` (`"Medical AI Skills"` → `"MONAI"`, 12 skills)

The components.d label is generic, and the metadata value `"MONAI"` is
plausible for the imaging skills (`dicom-*`, `nv-segment-*`, `nv-generate-mr*`,
`nv-generate-ct-*`). But "Medical AI Skills" being a single bucket is a smell
— if any non-MONAI skill ends up in this file later (e.g. healthcare LLM
work), the deterministic mapping will silently misclassify. Suggest splitting
this yml into a per-product file (`monai.yml`) and giving each future medical
skill its own components.d row.

---

## Suggested follow-ups for the team

1. **Mechanical (low risk)** — apply the 6 yml edits in §A. Zero metadata
   impact today; pure README hygiene + future-proofing if we ever wire the
   deterministic mapper back in.
2. **Editorial (needs SME input)** — review the 4 cases in §B with the
   relevant product owners (AI-Q team, Digital Health/ASR team, NeMo team,
   Trustworthy AI team) and decide whether to fix `metadata.json` (next
   regen run) or extend the schema enum.
3. **Schema additions** — propose adding `"NeMo AutoModel"`, `"RAG Blueprint"`,
   and `"TileGym"` to `metadata.schema.json`'s `product.primary.enum` if
   they're recognized NVIDIA products. Otherwise, accept the broader-bucket
   metadata values as-is and document why.
4. **Refactor `medical-ai-skills.yml`** into per-product files once the
   medical AI catalog grows beyond MONAI.
