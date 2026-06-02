# PRD: Skill Metadata Generation Automation

## Summary

Build a GitHub automation for the NVIDIA skills catalog that generates and validates two metadata files whenever skills change:

- `metadata.json`
- `skills.sh.json`

The automation scans the repository's `skills/` folder, maps every skill to its product/subdomain, enriches missing metadata when needed, validates the generated output against the approved schema and taxonomy, and alerts the team if validation fails while leaving the previous metadata in place.

This replaces an earlier fully-agentic local workflow. The new design should use deterministic code for inventory, parsing, change detection, generation, and validation. AI should only be called through a small, schema-constrained enrichment API for fields that cannot be confidently derived from repo content.

## Goals

- Generate a complete `metadata.json` entry for every skill in `skills/**/SKILL.md`.
- Generate or regenerate `skills.sh.json` from the same skill inventory, mapping each skill into its subdomain block using the completed `catalog.subdomain` value from `metadata.json`.
- Treat each skill `metadata` object as a flat string-to-string map.
- Preserve existing valid metadata for unchanged skills.
- Validate the generated `metadata.json` file against the approved metadata schema and taxonomy.
- Alert on CI issues when generated files are missing, stale, malformed, or schema-invalid.
- Open or update a GitHub issue on validation failure and mention `@jim-NVIDIA` and `@jasonNVIDIA`.
- Support AI-assisted metadata enrichment for fields that cannot be reliably derived from static files.

## References

- Example generated files: https://github.com/NVIDIA/skills/pull/90
- `skills.sh.json` customization guidance: https://www.skills.sh/docs/customize

## Non-Goals

- Do not rewrite skill content.
- Do not change the Agent Skills specification.
- Do not rely on manual edits to generated files as the primary workflow.
- Do not silently accept unknown taxonomy values.

## Inputs

- `skills/`: all cataloged skill directories.
- `components.d/*.yml`: product registry and catalog directory mapping.
- Existing skill-local metadata where present:
  - `SKILL.md` frontmatter
  - `skill-card.md`
  - `card.yaml`
- Approved metadata JSON schema.
- Approved taxonomy standard.
- `taxonomy-categories-and-values.json`: controlled values for MVP metadata fields.
- `skills-subdomains.json`: controlled subdomain slugs, display titles, descriptions, and group order.
- Example `metadata.json` and `skills.sh.json` files.
- Optional AI enrichment API configuration.

## Outputs

- `metadata.json`: canonical metadata inventory for all skills.
- `skills.sh.json`: marketplace/index file regenerated from the canonical inventory.
- CI validation report.
- GitHub issue on validation failure.

## Functional Requirements

1. Skill discovery
   - Recursively find every `SKILL.md` under `skills/`.
   - Treat the parent directory of each `SKILL.md` as one skill.
   - Exclude non-skill docs, examples, and nested references unless they contain a top-level `SKILL.md`.

2. Product and subdomain mapping
   - Map each skill to a product using the first path segment under `skills/`.
   - Resolve the product against `components.d/*.yml`.
   - Use the component name as the primary product unless the taxonomy requires a different canonical value.
   - Fail validation if a skill cannot be mapped to a known component or taxonomy entry.

3. Static metadata extraction
   - Parse `SKILL.md` frontmatter with a YAML-aware parser.
   - Correctly handle multiline YAML block scalars such as `description: |` and `description: >`.
   - Extract `name` and `description` exactly as authored in the current `SKILL.md`.
   - Do not rewrite, improve, normalize, or otherwise alter existing skill names or descriptions.
   - Parse other frontmatter fields such as `version` and `when_to_use` when available.
   - Use `skill-card.md` or `card.yaml` as supplemental metadata when present.
   - Derive stable defaults from the skill path only for fields explicitly allowed by the schema.

4. AI-assisted enrichment
   - Use an AI agent/API only for fields that are missing or require judgment, such as:
     - primary product
     - primary classification
     - intended audience
     - discovery activity text
     - other taxonomy-bound descriptive fields
   - Provide the AI agent with bounded context:
     - `SKILL.md`
     - available local skill-card/card metadata
     - component metadata
     - allowed taxonomy values
     - target JSON schema fragment
   - Require structured JSON output from the AI agent.
   - Validate AI output before merging it into generated metadata.
   - Never allow the AI agent to invent taxonomy values.
   - Require high confidence for `product.primary`; unresolved or ambiguous product mapping should fail validation with an actionable error.

5. Generation behavior
   - Regenerate both output files deterministically.
   - Sort products and skills consistently for stable diffs.
   - Preserve existing metadata values for unchanged skills unless the value is invalid under the current schema/taxonomy.
   - Do not retag existing skills for stylistic improvements.
   - Preserve no hand-edited content in generated files unless explicitly supported by the design.
   - The same input tree must produce byte-stable output.

6. Validation
   - Validate `metadata.json` against the approved schema.
   - Validate `skills.sh.json` against its expected schema or structural contract.
   - Validate taxonomy values against the approved taxonomy standard.
   - Validate inventory completeness:
     - every `skills/**/SKILL.md` appears in `metadata.json`
     - every `metadata.json` skill maps to an existing `SKILL.md`
     - every skill appears in `skills.sh.json` when required
   - Validate MVP metadata shape:
     - only approved MVP metadata keys are emitted
     - no metadata key uses a leading `nvidia.` prefix
     - every metadata value is a string
     - multi-value fields use comma-separated strings, not arrays
     - no arrays, objects, booleans, or nulls are emitted in a skill's `metadata` object
   - Fail CI on validation errors.

7. GitHub automation
   - Run when files under `skills/**` change.
   - Run on manual dispatch.
   - Optionally run after the existing skill sync workflow updates the catalog.
   - If generated files change, propose the changes in a pull request.
   - If validation fails, open or update an issue with:
     - validation summary
     - affected files/skills
     - workflow run link
     - requested owners: `@jim-NVIDIA` and `@jasonNVIDIA`
   - Avoid creating duplicate open issues for the same validation class.

8. Change detection
   - Compare current skill inventory to the best available baseline `metadata.json`.
   - For unchanged skills, carry forward existing valid metadata.
   - For new skills, generate all required MVP metadata fields.
   - For removed skills, remove them from both generated files.
   - For renamed or moved skills, use git history, path/name similarity, and content similarity to decide whether prior metadata can be carried forward.
   - For materially changed skills, re-evaluate only fields affected by changes to product, audience, classification, subdomain, or activity.
   - Report added, removed, renamed/moved, materially changed, and unresolved skills in CI output.

9. Temporary exclusions
   - Support a configurable exclusion list for temporarily embargoed skills.
   - Excluded skills must not appear in `metadata.json` or `skills.sh.json`.
   - Excluded skills should be reported separately and not counted as removed or unresolved.

10. `skills.sh.json` generation
   - Generate `skills.sh.json` from `metadata.json` plus `skills-subdomains.json`.
   - Use `skills-subdomains.json` as the source of truth for group order, display title, description, and allowed subdomain slugs.
   - Emit only non-empty groups.
   - Include `$schema`.
   - Set `notGrouped` to `bottom`.
   - Ensure every skill with `metadata["catalog.subdomain"]` appears exactly once.
   - Fail validation if a subdomain is missing, unknown, duplicated, or cannot be grouped cleanly.

## MVP Metadata Fields

Every generated `metadata.json` skill entry must include only these metadata fields unless the approved schema changes:

- `product.primary`
- `classification.category.primary`
- `catalog.subdomain`
- `audience`
- `discovery.activity_tags`

The checked-in schema is the source of truth for which fields are required versus recommended.

Controlled-value sources:

- `product.primary`, `classification.category.primary`, `audience`, and `discovery.activity_tags` must use values from `taxonomy-categories-and-values.json`.
- `catalog.subdomain` must use values from `skills-subdomains.json`.
- Multi-value `discovery.activity_tags` must be emitted as a comma-separated string.

Expected `metadata.json` shape:

```json
{
  "skills": [
    {
      "path": "skills/example-skill-directory",
      "name": "existing-skill-name",
      "description": "Existing description copied exactly from SKILL.md.",
      "metadata": {
        "product.primary": "Example Product",
        "classification.category.primary": "ai_and_machine_learning",
        "catalog.subdomain": "agentic-ai",
        "audience": "application_developer",
        "discovery.activity_tags": "configure,validate,debug"
      }
    }
  ]
}
```

Expected `skills.sh.json` shape:

```json
{
  "$schema": "https://skills.sh/schemas/skills.sh.schema.json",
  "notGrouped": "bottom",
  "groupings": [
    {
      "title": "Decision Optimization",
      "description": "Formulate and solve routing, scheduling, and numerical optimization problems using optimization solvers, APIs, services, and deployment workflows.",
      "skills": [
        "cuopt-routing-api-python"
      ]
    }
  ]
}
```

## Suggested Implementation

Shipped layout (the entire metadata pipeline — generator script, schemas,
configs, exclusions list, and the generated `metadata.json` itself — lives
under `.github/scripts/marketplace/`, co-located with upstream's existing
`marketplace/metadata.json`):

- Generator script:
  - `.github/scripts/marketplace/generate-skill-metadata.py`
- Schemas and configuration:
  - `.github/scripts/marketplace/metadata.schema.json` — canonical source of truth for controlled values; consumed directly at runtime.
  - `.github/scripts/marketplace/skills-sh.schema.json`
  - `.github/scripts/marketplace/skills-subdomains.json`
  - `.github/scripts/marketplace/metadata-exclusions.yaml`
- Generated outputs:
  - `.github/scripts/marketplace/metadata.json`
  - `skills.sh.json` (repo root, per upstream convention)
- Workflow:
  - `.github/workflows/generate-skill-metadata.yml`
- Local-usage docs:
  - `docs/metadata-generation.md`

The schemas encode required versus recommended fields. There is no derived
`taxonomy-categories-and-values.json` companion file: the generator extracts
controlled values directly from `metadata.schema.json` at runtime.

## AI Enrichment Contract

The enrichment layer should be optional and explicit:

- If all required fields can be derived locally, no API call is made.
- If fields require AI enrichment and the API is unavailable, fail with a clear validation error.
- Use an OpenAI-compatible API interface; the concrete provider, model, and endpoint configuration are TBD.
- Cache or persist enriched metadata only through regenerated output files.
- Require the AI response to match a strict JSON schema before use.
- Include enough error detail to let a maintainer fix the skill or taxonomy manually.
- The API should receive only the bounded context needed for the specific skill and missing fields.
- The API should return only the MVP fields requested by the generator.
- The generator, not the AI API, is responsible for final validation and file writing.
- If the API returns unknown values, extra keys, malformed JSON, or low-confidence assignments, the generator must fail instead of guessing.

## Acceptance Criteria

- Running the generator locally creates `metadata.json` and `skills.sh.json`.
- Running the generator twice with no input changes produces no diff.
- Removing a skill removes it from both generated files.
- Adding a skill adds it to both generated files.
- Schema-invalid generated output fails CI.
- Unknown taxonomy values fail CI.
- Missing component/subdomain mapping fails CI.
- Validation failure creates or updates a GitHub issue mentioning `@jim-NVIDIA` and `@jasonNVIDIA`.
- The workflow can run without AI calls for skills that already have complete metadata.
- AI-enriched fields are validated before being written.
- Existing valid metadata is preserved for unchanged skills.
- Existing `name` and `description` are always aligned exactly to current `SKILL.md` frontmatter.
- Multiline YAML descriptions are parsed correctly.
- Temporarily excluded skills do not appear in generated outputs and are reported separately.

## Open Questions

- What is the initial temporary exclusion list, and where should it be configured?
  - file: `.github/scripts/marketplace/metadata-exclusions.yaml` (co-located with the rest of the marketplace pipeline); lists skills by directory name to exclude from both metadata files.
- What OpenAI-compatible endpoint, model, and credentials should the workflow use for enrichment?
- example
  - curl --location 'https://inference-api.nvidia.com/v1/chat/completions' \  --header 'Content-Type: application/json' \  --header "Authorization: Bearer $INFERENCE_API_KEY" \  --data '{"model":"openai/openai/gpt-5.5","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello! Can you help me?"}],"max_tokens":1024}'
  - INFERENCE_API_KEY will be set in github secret and locally in .env file

## metadata.json schema
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://developer.nvidia.com/schemas/skills-mvp-metadata.schema.json",
  "title": "NVIDIA Skills MVP Metadata",
  "description": "Schema for the generated NVIDIA skills metadata.json file. Controlled values are derived from skills-mvp-taxonomy-and-subdomains.json.",
  "$comment": "Generated 2026-05-29 from skills-mvp-taxonomy-and-subdomains.json.",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "skills"
  ],
  "properties": {
    "skills": {
      "type": "array",
      "description": "One entry per non-embargoed current skill included in the metadata artifact.",
      "items": {
        "$ref": "#/$defs/skill"
      }
    }
  },
  "$defs": {
    "skill": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "path",
        "name",
        "description",
        "metadata"
      ],
      "properties": {
        "path": {
          "type": "string",
          "minLength": 1,
          "pattern": "^skills/.+",
          "description": "Repo-relative path to the skill directory."
        },
        "name": {
          "type": "string",
          "minLength": 1,
          "description": "Skill name copied exactly from SKILL.md frontmatter."
        },
        "description": {
          "type": "string",
          "minLength": 1,
          "description": "Skill description copied exactly from SKILL.md frontmatter."
        },
        "metadata": {
          "$ref": "#/$defs/metadata"
        }
      }
    },
    "metadata": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "product.primary",
        "classification.category.primary",
        "catalog.subdomain",
        "audience",
        "discovery.activity_tags"
      ],
      "properties": {
        "product.primary": {
          "type": "string",
          "enum": [
            "ACE",
            "Aerial",
            "Aerial CUDA Accelerated RAN",
            "Aerial Omniverse Digital Twin",
            "Aerial Resources",
            "AgentIQ",
            "AI Enterprise",
            "AI Workbench",
            "AIQToolkit",
            "AIStore",
            "Alpamayo",
            "AmgX",
            "Apex",
            "Attestation",
            "Audio Effects Microservice",
            "Audio Effects SDK",
            "Augmented Reality (AR) SDK",
            "Backend",
            "Base Command Manager",
            "Base Command Platform",
            "BioNeMo",
            "BioNeMo Blueprints",
            "BioNeMo Framework",
            "BioNeMo Open Models",
            "BlueField",
            "Brev",
            "CCCL-CUDA Core Compute Library",
            "Certification Programs",
            "Clara Holoscan SDK",
            "Clara Parabricks",
            "Client",
            "Cloud Functions",
            "Cloud Native Stack",
            "CloudXR",
            "CloudXR SDK",
            "Compute Sanitizer",
            "ConnectX",
            "Container Toolkit",
            "Cosmos",
            "CUB",
            "cuBLAS",
            "cuBLASMp",
            "cuBLASXt",
            "cuCIM",
            "cuCollections",
            "CUDA",
            "CUDA Libraries",
            "CUDA MATH API",
            "CUDA Python",
            "CUDA Tile",
            "CUDA Toolkit",
            "CUDA-Q",
            "CUDA-Q QEC",
            "CUDA-Q Solvers",
            "cuDF",
            "cuDNN",
            "cuDSS",
            "cuEquivariance",
            "cuFFT",
            "cuFFTDx",
            "cuGraph",
            "cuML",
            "cuMotion",
            "cuNumeric",
            "cuOpt",
            "cuPCL",
            "CUPTI",
            "CUPTI Python",
            "CuPy",
            "cuPyNumeric",
            "cuQuantum",
            "cuRAND",
            "cuSOLVER",
            "cuSPARSE",
            "cuSPARSELt",
            "cuSpatial",
            "cuTENSOR",
            "CUTLASS",
            "cuVS",
            "cuVSLAM",
            "cuXfilter",
            "CV-CUDA",
            "DALI",
            "Dask-CUDA",
            "Data Plane Development Kit (DPDK)",
            "Datacenter",
            "DCCPU",
            "DCGM",
            "DCGM Exporter",
            "Deep Learning",
            "Deep Learning Inference Accelerator (NVDLA)",
            "Deepops",
            "DeepStream 360 Smart Parking Application",
            "DeepStream LPR App",
            "DeepStream SDK",
            "DeepStream TAO Apps",
            "Deploy",
            "DGX",
            "DGX BasePOD",
            "DGX Cloud",
            "DGX Spark",
            "DGX SuperPOD",
            "DLSS",
            "DOCA",
            "DRIVE",
            "DRIVEOS",
            "DriveWorks",
            "Dynamo",
            "Dynamo-Triton (Previously Triton)",
            "Enroot",
            "Falcor",
            "FasterTransformer",
            "Federated Learning Active Runtime Environment (FLARE)",
            "Flang",
            "FlashInfer",
            "Fleet Command",
            "FME",
            "Frameview SDK",
            "GameWorks",
            "GDRCopy",
            "GeForce Now DevTools",
            "GeForce Now SDK",
            "GPU Operator",
            "GPUDirect Storage",
            "GRID",
            "HGX Platforms",
            "Holoscan",
            "Holoscan for Media",
            "HPC Benchmarks",
            "HPC Container Maker",
            "HPC SDK",
            "HPC-X",
            "HugeCTR",
            "IGX Orin",
            "In-Game Inferencing SDK",
            "IndeX",
            "Inference Xfer Library (NIXL)",
            "Isaac for Healthcare",
            "Isaac GROOT",
            "Isaac Lab",
            "Isaac Manipulator",
            "Isaac Perception",
            "Isaac ROS",
            "Isaac Ros Visual Slam",
            "Isaac Sim",
            "Ising",
            "Jetbot",
            "JetPack SDK",
            "Jetson",
            "Jitify",
            "JupyterLab NVIDIA Dashboard",
            "K8s Device Plug-in",
            "KAI Scheduler",
            "Kaolin",
            "Kaolin Library",
            "Learn OpenUSD",
            "Learning",
            "Legate",
            "libcu++",
            "Libnvidia-container",
            "License System",
            "Material Definition Language (MDL)",
            "MatX",
            "Maxine",
            "MDL SDK",
            "Megatron-Core",
            "Megatron-LM",
            "Mellotron",
            "Merlin",
            "Merlin HugeCTR",
            "Metropolis",
            "Mission Control",
            "MMS",
            "Model Analyzer",
            "MOJ",
            "MONAI",
            "MONAI Core",
            "MONAI Deploy App SDK",
            "MONAI Deploy Workflow Manager",
            "MONAI Label",
            "MONAI Toolkit",
            "Morpheus",
            "Mosaic",
            "Multi-GPU Programming Models",
            "Multi-Node NVLink",
            "Nano VDB",
            "NCCL",
            "NCCL-tests",
            "NeMo",
            "NeMo Agent Toolkit",
            "NeMo Curator",
            "NeMo Framework",
            "NeMo Guardrails",
            "NeMo Gym",
            "NeMo Megatron Bridge",
            "NeMo Megatron Launcher",
            "NeMo Microservices",
            "NeMo Retriever",
            "NeMo Retriever Extraction",
            "NeMo Retriever Ingest",
            "NeMo RL",
            "NeMoClaw",
            "Nemotron",
            "Nemotron for Digital Health",
            "Network Operator",
            "Networking",
            "Neural VDB",
            "Newton Physics Engine",
            "NGC",
            "NGC.cn",
            "NIM",
            "NIM for Healthcare",
            "NIM Operator",
            "NPP",
            "NPP+",
            "Nsight Aftermath",
            "Nsight Cloud",
            "Nsight Compute",
            "Nsight Copilot",
            "Nsight Deep Learning Designer",
            "Nsight DL Designer",
            "Nsight Graphics",
            "Nsight Perf SDK",
            "Nsight Python",
            "Nsight Systems",
            "Nsight Tegra VSE",
            "Nsight Test",
            "Nsight Visual Studio Code Edition",
            "Nsight Visual Studio Edition",
            "Nsight VS Integration",
            "NuRec",
            "NVAPI",
            "NVBench",
            "NVBlox",
            "nvCOMP",
            "NVFLare",
            "NVIGI SDK",
            "nvImageCodec",
            "nvJPEG",
            "nvJPEG2K",
            "nvmath-python",
            "NVPL",
            "NVRHI",
            "NVRTC",
            "NVSHMEM",
            "NVTabular",
            "nvTiff",
            "NVTX",
            "Omniverse",
            "Omniverse Kit",
            "Omniverse NuRec",
            "ONNX-TensorRT",
            "Opacity Micromap (OMM) SDK",
            "Open GPU Documentation",
            "Open GPU Kernel Modules",
            "OpenUSD",
            "OpenUSD Exchange SDK",
            "Optical Flow SDK",
            "Orbit",
            "Parabricks",
            "Path Tracing SDK",
            "PerfKit",
            "Physical AI Dataset",
            "PhysicsNeMo",
            "PhysX",
            "PVA",
            "PyTorch Geometric (PyG) Container",
            "PyTriton",
            "Quantum Open Models",
            "Raft",
            "RAG",
            "RAPIDS",
            "RAPIDS Accelerator for Apache Spark",
            "RAPIDS Accelerator for Spark",
            "Real-Time Denoisers (NRD) SDK",
            "Reflex",
            "Research Contributions",
            "Riva",
            "Rivermax",
            "RL games",
            "RMM",
            "ROS GEMs",
            "RTX branch of Unreal Engine (NvRTX)",
            "RTX CR-Character Rendering",
            "RTX Direct Illumination (RTXDI)",
            "RTX Kit",
            "RTX MG-Mega Geometry",
            "RTX NS-Neural Shading",
            "RTX NTC-Neural Text Compression",
            "RTX Remix",
            "RTX Texture Filtering",
            "RTX Texture Streaming",
            "RTXGI",
            "SDK Manager",
            "Settings",
            "Sionna",
            "Slang",
            "Spark Rapids",
            "Spectrum Switch SDK",
            "Spectrum-X",
            "STBN-SpatioTemporal BlueNoise",
            "StdExec",
            "Streamline",
            "System Profiler",
            "TAO",
            "TAO Toolkit",
            "TAO Toolkit API",
            "TensorRT",
            "TensorRT Edge-LLM",
            "TensorRT for RTX",
            "TensorRT-LLM",
            "Texture Tools (NVTT)3",
            "Texture Tools-Photoshop Plug in",
            "TimeLoop",
            "Tokkio",
            "Torch Harmonics",
            "Torch-TensorRT",
            "Torch2TRT",
            "Transformer Engine",
            "Transformers4Rec",
            "Triton Inference Server",
            "TRT Samples for Hackathon CN",
            "Trustworthy AI",
            "UCF",
            "UCX-Unified Communication X",
            "UFM",
            "VCR SDK",
            "vGPU",
            "Video Codec",
            "Video Effects SDK",
            "Video Search and Summarization (VSS)",
            "Video Technologies",
            "visRTX",
            "VK Mini Path Tracer",
            "vMaterials",
            "VPI",
            "VRWorks Graphics for HMD",
            "VSS",
            "Warp",
            "WaveWorks",
            "XCR",
            "YOLO DeepStream"
          ],
          "description": "Primary NVIDIA product or product surface."
        },
        "classification.category.primary": {
          "type": "string",
          "enum": [
            "ai_and_machine_learning",
            "accelerated_computing",
            "physical_ai",
            "infrastructure",
            "graphics_and_media",
            "developer_tools"
          ],
          "description": "Primary top-level product/category classification."
        },
        "catalog.subdomain": {
          "type": "string",
          "enum": [
            "training-ai",
            "agentic-ai",
            "inference-ai",
            "decision-optimization",
            "vision-ai",
            "gpu-development",
            "conversational-ai",
            "quantum-computing",
            "ai-factory",
            "ai-storage",
            "av-simulation",
            "cybersecurity",
            "data-science",
            "digital-twins",
            "extended-reality",
            "gaming",
            "gpu-virtualization",
            "infrastructure",
            "in-vehicle-computing",
            "networking",
            "physical-ai",
            "robotics",
            "robotics-simulation",
            "scientific-visualization",
            "simulation-modeling"
          ],
          "description": "Catalog subdomain slug used to generate skills.sh groupings."
        },
        "audience": {
          "type": "string",
          "pattern": "^(?:computational_biologist|student_non_university|research_non_academic|application_developer|solutions_architect|simulation_engineer|security_operations|production_operator|genomics_researcher|student_university|robotics_developer|quantum_researcher|business_executive|student_undergrad|security_engineer|research_academic|platform_engineer|hands_on_builder|bioinformatician|devops_engineer|data_scientist|it_operations|hpc_developer|data_engineer|ml_engineer|ai_engineer|consulting|marketing|developer|clinician|educator|designer|press|other)(?:,(?:computational_biologist|student_non_university|research_non_academic|application_developer|solutions_architect|simulation_engineer|security_operations|production_operator|genomics_researcher|student_university|robotics_developer|quantum_researcher|business_executive|student_undergrad|security_engineer|research_academic|platform_engineer|hands_on_builder|bioinformatician|devops_engineer|data_scientist|it_operations|hpc_developer|data_engineer|ml_engineer|ai_engineer|consulting|marketing|developer|clinician|educator|designer|press|other))*$",
          "description": "One or more Audience values as a comma-separated string with no spaces.",
          "examples": [
            "developer",
            "developer,application_developer"
          ]
        },
        "discovery.activity_tags": {
          "type": "string",
          "pattern": "^(?:troubleshoot|orchestrate|get_started|synthesize|contribute|transform|summarize|integrate|inference|fine_tune|configure|benchmark|validate|optimize|generate|evaluate|classify|recover|profile|package|operate|monitor|migrate|measure|inspect|extract|convert|compare|select|extend|deploy|assess|train|scale|debug|test)(?:,(?:troubleshoot|orchestrate|get_started|synthesize|contribute|transform|summarize|integrate|inference|fine_tune|configure|benchmark|validate|optimize|generate|evaluate|classify|recover|profile|package|operate|monitor|migrate|measure|inspect|extract|convert|compare|select|extend|deploy|assess|train|scale|debug|test))*$",
          "description": "One or more Activity Tags values as a comma-separated string with no spaces.",
          "examples": [
            "configure",
            "configure,validate,debug"
          ]
        }
      }
    }
  }
}
