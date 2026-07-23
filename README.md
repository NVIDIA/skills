<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# NVIDIA Agent Skills

**Official, NVIDIA-verified Agent Skills for Claude Code, Codex, and other coding agents.**

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue)](https://agentskills.io)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2B%20CC--BY--4.0-green.svg)](#license)

> 📖 **Docs:** [docs.nvidia.com/skills](https://docs.nvidia.com/skills) &nbsp;·&nbsp;
> 📺 **Livestream:** [From Vulnerable to Verified](https://www.youtube.com/watch?v=sVpKonYJ4D4&list=PL5B692fm6--vEL0FwctKghCpyEnBGAQJA&index=1) &nbsp;·&nbsp;
> 📝 **Blog:** [NVIDIA Verified Agent Skills: Capability Governance for AI Agents](https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/)

---

Skills are portable instruction sets that teach AI agents how to use NVIDIA software optimally: Physical AI and robotics workflows, simulation, CUDA-X libraries, RAG and AI Blueprints, and platform tools. This repository is a catalog: skills are maintained in their respective product repos, and mirrored here daily via an automated sync pipeline. Skills are being added continuously, so check back for updates. We are building this infrastructure in the open, and contributions are welcome. See the [Roadmap](#roadmap) for what is planned next.

---

## Quickstart

Install NVIDIA skills with the default [`skills` CLI](https://github.com/vercel-labs/skills) flow:

```bash
npx skills add nvidia/skills
```

The CLI runs through `npx` and prompts you to choose a skill and install destination. You do not need to clone this repo or copy skill folders by hand.

> **Requires a current `skills` CLI (v1.5.16 or newer).** Installing via `npx skills@latest add nvidia/skills` always uses the latest. On older CLIs (v1.5.15 and earlier), skills may install but not appear in Claude Code — see [Troubleshooting](docs/advanced-install.mdx#troubleshooting).

The skill is available the next time your agent loads skills and encounters a relevant task. For example, ask your agent to "solve a linear programming problem with cuOpt" and the skill guides it through the cuOpt Python API. In Claude Code, run `/reload-skills` to load newly installed skills in your current session.

### Install One Skill Without Prompts

Use this when you already know the skill name and want to skip prompts.

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api --yes
```

Replace `cuopt-numerical-optimization-api` with any skill name from the [Skill Catalog](#skill-catalog).

### Install for a Specific Agent

Use `--agent` to target a specific AI coding agent. Initially, we'll support common client targets, expanding the list over time. For the full list of clients supported by the spec, see the [`skills` CLI Supported Agents table](https://github.com/vercel-labs/skills#supported-agents).

**Claude Code**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api --agent claude-code
```

**Codex**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api --agent codex
```

**Snowflake CoCo**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api --agent cortex
```

**Cursor**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api --agent cursor
```

**Kiro**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api --agent kiro-cli
```

Use `--agent` more than once to install the same skill into multiple agents.

```bash
npx skills add nvidia/skills \
  --skill cuopt-numerical-optimization-api \
  --agent claude-code \
  --agent codex \
  --agent cursor \
  --agent kiro-cli
```

### Keep Skills Up to Date

New skills land continuously, and existing ones are revised, renamed, or consolidated as the catalog evolves. Refresh what you have installed with:

```bash
npx skills update
```

Run it interactively and the CLI also flags skills that were removed or merged upstream (for example, when several skills are consolidated into one) and offers to remove the stale local copies. Use `npx skills list` to see what is installed and `npx skills check` to preview what is out of date first.

### Browse the Catalog

Use this when you want to see available NVIDIA skills before installing anything.

```bash
npx skills add nvidia/skills --list
```

For non-interactive installs, global installs, agent-specific installs, updates, removals, and fallback manual copying, see [Advanced installation](docs/advanced-install.mdx).

---

## Skill Catalog

<!-- skills-table-start -->
| Product | Description | Skills |
|---------|-------------|--------|
| **AIQ** | NVIDIA AI-Q Blueprint - deploy local AI-Q services and run shallow or deep research workflows as agent skills. | [`aiq-research`](skills/aiq-research), [`aiq-deploy`](skills/aiq-deploy) |
| **CUDA-Q** | CUDA Quantum — onboarding guide for installation, test programs, GPU simulation, QPU hardware, and quantum applications. | [`cudaq-guide`](skills/cudaq-guide) |
| **cuDF** | Official NVIDIA-authored guidance for NVIDIA cuDF GPU DataFrames, pandas acceleration, dask-cuDF, ETL, joins, groupby, CSV/Parquet I/O, nullable semantics, and multi-GPU DataFrame workloads. | [`accelerated-computing-cudf`](skills/accelerated-computing-cudf) |
| **cuFOLIO** | GPU-accelerated Mean-CVaR portfolio optimization with NVIDIA cuOpt — CVaR optimization, efficient frontier, scenario generation, backtesting, and rebalancing. | [`cufolio`](skills/cufolio) |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | [`cuopt-install`](skills/cuopt-install), [`cuopt-multi-objective-exploration`](skills/cuopt-multi-objective-exploration), [`cuopt-numerical-optimization-api`](skills/cuopt-numerical-optimization-api), [`cuopt-numerical-optimization-formulation`](skills/cuopt-numerical-optimization-formulation), [`cuopt-routing-api-python`](skills/cuopt-routing-api-python), [`cuopt-server-api-python`](skills/cuopt-server-api-python) |
| **cuPyNumeric** | NumPy and SciPy on multi-node multi-GPU systems — skills to help with installing cuPyNumeric, migrating existing NumPy code, and doing parallel I/O | [`cupynumeric-hdf5`](skills/cupynumeric-hdf5), [`cupynumeric-install`](skills/cupynumeric-install), [`cupynumeric-migration-readiness`](skills/cupynumeric-migration-readiness), [`cupynumeric-parallel-data-load`](skills/cupynumeric-parallel-data-load) |
| **DALI** | GPU-accelerated data loading and processing with NVIDIA DALI. | [`dali-dynamic-mode`](skills/dali-dynamic-mode) |
| **Data Designer** | Build declarative synthetic dataset generation pipelines with NeMo Data Designer. | [`data-designer`](skills/data-designer) |
| **DeepStream** | Agentic skills for guided DeepStream development. | [`amc-run-sample-calibration`](skills/amc-run-sample-calibration), [`amc-run-video-calibration`](skills/amc-run-video-calibration), [`amc-setup-calibration-stack`](skills/amc-setup-calibration-stack), [`deepstream-dev`](skills/deepstream-dev), [`deepstream-generate-pipeline`](skills/deepstream-generate-pipeline), [`deepstream-import-vision-model`](skills/deepstream-import-vision-model), [`deepstream-profile-pipeline`](skills/deepstream-profile-pipeline), [`deepstream-sop`](skills/deepstream-sop) |
| **Digital Health** | Agent skills for the clinical ASR evaluation flywheel — term curation, synthetic clinical-speech benchmark generation, KER (Keyword Error Rate) scoring, and fine-tune guidance. | [`digital-health-clinical-asr-setup`](skills/digital-health-clinical-asr-setup), [`digital-health-clinical-asr-build`](skills/digital-health-clinical-asr-build), [`digital-health-clinical-asr-eval`](skills/digital-health-clinical-asr-eval), [`digital-health-clinical-asr-finetune`](skills/digital-health-clinical-asr-finetune) |
| **DOCA** | Teach AI agents to use the NVIDIA DOCA SDK on BlueField DPUs and ConnectX NICs — setup, libraries, services, tools, deployment, and debugging. | [`doca-bare-metal-deployment`](skills/doca-bare-metal-deployment), [`doca-bf3-deployment`](skills/doca-bf3-deployment), [`doca-bf4-deployment`](skills/doca-bf4-deployment), [`doca-collectx-deployment`](skills/doca-collectx-deployment), [`doca-container-deployment`](skills/doca-container-deployment), [`doca-debug`](skills/doca-debug), [`doca-hardware-safety`](skills/doca-hardware-safety), [`doca-programming-guide`](skills/doca-programming-guide), [`doca-public-knowledge-map`](skills/doca-public-knowledge-map), [`doca-setup`](skills/doca-setup), [`doca-structured-tools-contract`](skills/doca-structured-tools-contract), [`doca-upgrade`](skills/doca-upgrade), [`doca-version`](skills/doca-version), [`doca-aes-gcm`](skills/doca-aes-gcm), [`doca-argp`](skills/doca-argp), [`doca-comch`](skills/doca-comch), [`doca-common`](skills/doca-common), [`doca-compress`](skills/doca-compress), [`doca-devemu`](skills/doca-devemu), [`doca-dma`](skills/doca-dma), [`doca-dpa`](skills/doca-dpa), [`doca-dpdk-bridge`](skills/doca-dpdk-bridge), [`doca-erasure-coding`](skills/doca-erasure-coding), [`doca-eth`](skills/doca-eth), [`doca-flow`](skills/doca-flow), [`doca-flow-dpa-provider`](skills/doca-flow-dpa-provider), [`doca-gpi`](skills/doca-gpi), [`doca-gpunetio`](skills/doca-gpunetio), [`doca-mgmt`](skills/doca-mgmt), [`doca-pcc`](skills/doca-pcc), [`doca-pcc-ztr-rttcc-algo`](skills/doca-pcc-ztr-rttcc-algo), [`doca-rdma`](skills/doca-rdma), [`doca-rdmi`](skills/doca-rdmi), [`doca-rmax`](skills/doca-rmax), [`doca-sha`](skills/doca-sha), [`doca-sta`](skills/doca-sta), [`doca-telemetry`](skills/doca-telemetry), [`doca-telemetry-exporter`](skills/doca-telemetry-exporter), [`doca-urom`](skills/doca-urom), [`doca-verbs`](skills/doca-verbs), [`doca-argus`](skills/doca-argus), [`doca-dms`](skills/doca-dms), [`doca-firefly`](skills/doca-firefly), [`doca-urom-svc`](skills/doca-urom-svc), [`doca-bench`](skills/doca-bench), [`doca-bench-extension`](skills/doca-bench-extension), [`doca-caps`](skills/doca-caps), [`doca-comm-channel-admin`](skills/doca-comm-channel-admin), [`doca-dpa-hl-tracer`](skills/doca-dpa-hl-tracer), [`doca-flow-dpa-perf`](skills/doca-flow-dpa-perf), [`doca-flow-grpc-server`](skills/doca-flow-grpc-server), [`doca-flow-perf`](skills/doca-flow-perf), [`doca-flow-tune`](skills/doca-flow-tune), [`doca-gpunetio-ib-write-bw`](skills/doca-gpunetio-ib-write-bw), [`doca-gpunetio-ib-write-lat`](skills/doca-gpunetio-ib-write-lat), [`doca-pcc-counters`](skills/doca-pcc-counters), [`doca-sha-offload-engine`](skills/doca-sha-offload-engine), [`doca-socket-relay`](skills/doca-socket-relay), [`doca-spcx-cc`](skills/doca-spcx-cc), [`doca-telemetry-utils`](skills/doca-telemetry-utils) |
| **Dynamo** | NVIDIA Dynamo deployment bring-up on Kubernetes — pick and deploy recipes, start router modes, validate disagg NIXL/UCX/NCCL interconnect, and triage day-2 failures. | [`dynamo-interconnect-check`](skills/dynamo-interconnect-check), [`dynamo-recipe-runner`](skills/dynamo-recipe-runner), [`dynamo-router-starter`](skills/dynamo-router-starter), [`dynamo-troubleshoot`](skills/dynamo-troubleshoot) |
| **Earth2Studio** | Open-source deep-learning framework for exploring, building and deploying AI weather/climate workflows. | [`earth2studio-create-datasource`](skills/earth2studio-create-datasource), [`earth2studio-create-diagnostic`](skills/earth2studio-create-diagnostic), [`earth2studio-create-prognostic`](skills/earth2studio-create-prognostic), [`earth2studio-data-fetch`](skills/earth2studio-data-fetch), [`earth2studio-deterministic-forecast`](skills/earth2studio-deterministic-forecast), [`earth2studio-discover`](skills/earth2studio-discover), [`earth2studio-install`](skills/earth2studio-install) |
| **Holoscan SDK** | Install and set up the Holoscan SDK on any platform (container, Debian, Python, Conda, or source). | [`holoscan-install-debian`](skills/holoscan-install-debian), [`holoscan-install-source`](skills/holoscan-install-source), [`holoscan-install-wheel`](skills/holoscan-install-wheel), [`holoscan-install-conda`](skills/holoscan-install-conda), [`holoscan-install-container`](skills/holoscan-install-container), [`holoscan-setup`](skills/holoscan-setup) |
| **Holoscan Sensor Bridge** | Agent-ready skills for Holoscan Sensor Bridge devkit workflows, covering demo environment bring-up, FPGA flashing for Lattice and VB1940 hardware, example application execution, and QA test-plan automation. | [`hsb-setup`](skills/hsb-setup), [`hsb-flash`](skills/hsb-flash), [`hsb-app`](skills/hsb-app), [`hsb-test`](skills/hsb-test) |
| **Isaac for Healthcare Workflows** | Agent-ready skills for Isaac for Healthcare agentic and catheter-navigation workflows, covering task authoring, data pipelines, policy training and validation, CT-derived digital twins, DRR rendering, and interactive catheter simulation. | [`i4h-workflow`](skills/i4h-workflow), [`i4h-workflow-setup`](skills/i4h-workflow-setup), [`i4h-workflow-create`](skills/i4h-workflow-create), [`i4h-workflow-scene-edit`](skills/i4h-workflow-scene-edit), [`i4h-workflow-dataset-teleop`](skills/i4h-workflow-dataset-teleop), [`i4h-workflow-dataset-replay`](skills/i4h-workflow-dataset-replay), [`i4h-workflow-dataset-mimic`](skills/i4h-workflow-dataset-mimic), [`i4h-workflow-dataset-annotate`](skills/i4h-workflow-dataset-annotate), [`i4h-workflow-dataset-convert`](skills/i4h-workflow-dataset-convert), [`i4h-workflow-finetune`](skills/i4h-workflow-finetune), [`i4h-workflow-validate`](skills/i4h-workflow-validate), [`i4h-workflow-e2e`](skills/i4h-workflow-e2e), [`i4h-lerobot-viz`](skills/i4h-lerobot-viz), [`i4h-catheter-navigation`](skills/i4h-catheter-navigation), [`i4h-catheter-navigation-setup`](skills/i4h-catheter-navigation-setup), [`i4h-catheter-navigation-digital-twin`](skills/i4h-catheter-navigation-digital-twin), [`i4h-catheter-navigation-render-drr`](skills/i4h-catheter-navigation-render-drr), [`i4h-catheter-navigation-viewport`](skills/i4h-catheter-navigation-viewport), [`i4h-catheter-navigation-smoke`](skills/i4h-catheter-navigation-smoke), [`i4h-catheter-navigation-e2e`](skills/i4h-catheter-navigation-e2e) |
| **Jetson BSP** | Agentic skills for setting up and customizing an NVIDIA Jetson Linux Board Support Package (BSP) — pick a target, prepare image and sources, customize IO (camera, PCIe, USB, pinmux, clocks, and more), then promote, flash, and validate. | [`jetson-build-source`](skills/jetson-build-source), [`jetson-customize-camera`](skills/jetson-customize-camera), [`jetson-customize-clocks`](skills/jetson-customize-clocks), [`jetson-customize-fan`](skills/jetson-customize-fan), [`jetson-customize-mgbe`](skills/jetson-customize-mgbe), [`jetson-customize-nvpmodel`](skills/jetson-customize-nvpmodel), [`jetson-customize-pcie`](skills/jetson-customize-pcie), [`jetson-customize-pinmux`](skills/jetson-customize-pinmux), [`jetson-customize-uphy`](skills/jetson-customize-uphy), [`jetson-customize-usb`](skills/jetson-customize-usb), [`jetson-derive-carrier`](skills/jetson-derive-carrier), [`jetson-download-bsp`](skills/jetson-download-bsp), [`jetson-flash-image`](skills/jetson-flash-image), [`jetson-generate-kb`](skills/jetson-generate-kb), [`jetson-init-image`](skills/jetson-init-image), [`jetson-init-source`](skills/jetson-init-source), [`jetson-init-target`](skills/jetson-init-target), [`jetson-link-docs`](skills/jetson-link-docs), [`jetson-optimize-memory`](skills/jetson-optimize-memory), [`jetson-print-bsp-info`](skills/jetson-print-bsp-info), [`jetson-promote-image`](skills/jetson-promote-image), [`jetson-quick-start`](skills/jetson-quick-start), [`jetson-set-target`](skills/jetson-set-target), [`jetson-validate-image`](skills/jetson-validate-image) |
| **Jetson Device** | Device-side agent skills for working with a live NVIDIA Jetson after boot — diagnostics, memory auditing, headless setup, inference memory tuning, LLM serving and benchmarking, packaging guidance, and speculative decoding. | [`jetson-diagnostic`](skills/jetson-diagnostic), [`jetson-headless-mode`](skills/jetson-headless-mode), [`jetson-inference-mem-tune`](skills/jetson-inference-mem-tune), [`jetson-llm-benchmark`](skills/jetson-llm-benchmark), [`jetson-llm-serve`](skills/jetson-llm-serve), [`jetson-memory-audit`](skills/jetson-memory-audit), [`jetson-package`](skills/jetson-package), [`jetson-print-device-info`](skills/jetson-print-device-info), [`jetson-speculative-decoding`](skills/jetson-speculative-decoding) |
| **Medical AI Skills** | Agent-ready medical AI skills built on MONAI for DICOM handling, NVIDIA-hosted medical imaging model workflows, segmentation, synthesis, and evidence-oriented evaluation. | [`dicom-metadata-extract`](skills/dicom-metadata-extract), [`dicom-series-preflight`](skills/dicom-series-preflight), [`dicom-series-to-volume`](skills/dicom-series-to-volume), [`nv-generate-ct-rflow`](skills/nv-generate-ct-rflow), [`nv-generate-mr`](skills/nv-generate-mr), [`nv-generate-mr-brain`](skills/nv-generate-mr-brain), [`nv-generate-mr-brain-finetune`](skills/nv-generate-mr-brain-finetune), [`nv-generate-vae-finetune`](skills/nv-generate-vae-finetune), [`nv-reason-cxr`](skills/nv-reason-cxr), [`nv-segment-ct`](skills/nv-segment-ct), [`nv-segment-ct-finetune`](skills/nv-segment-ct-finetune), [`nv-segment-ctmr`](skills/nv-segment-ctmr) |
| **Megatron-Core** | Large-scale distributed training — model parallelism, pipeline parallelism, and mixed precision. | [`mcore-create-issue`](skills/mcore-create-issue), [`mcore-linting-and-formatting`](skills/mcore-linting-and-formatting), [`mcore-run-on-slurm`](skills/mcore-run-on-slurm), [`mcore-split-pr`](skills/mcore-split-pr), [`mcore-testing`](skills/mcore-testing) |
| **NeMo AutoModel** | NeMo AutoModel - PyTorch-native distributed training for LLMs/VLMs with Hugging Face support, recipes, launchers, and validation workflows. | [`nemo-automodel-distributed-training`](skills/nemo-automodel-distributed-training), [`nemo-automodel-launcher-config`](skills/nemo-automodel-launcher-config), [`nemo-automodel-model-onboarding`](skills/nemo-automodel-model-onboarding), [`nemo-automodel-recipe-development`](skills/nemo-automodel-recipe-development) |
| **NeMo MBridge** | NeMo MBridge - PyTorch-native bridge between Hugging Face and Megatron-Core for checkpoint conversion, training recipes, and NVIDIA GPU performance workflows. | [`nemo-mbridge-mlm-bridge-training`](skills/nemo-mbridge-mlm-bridge-training), [`nemo-mbridge-multi-node-slurm`](skills/nemo-mbridge-multi-node-slurm), [`nemo-mbridge-perf-activation-recompute`](skills/nemo-mbridge-perf-activation-recompute), [`nemo-mbridge-perf-cpu-offloading`](skills/nemo-mbridge-perf-cpu-offloading), [`nemo-mbridge-perf-cuda-graphs`](skills/nemo-mbridge-perf-cuda-graphs), [`nemo-mbridge-perf-expert-parallel-overlap`](skills/nemo-mbridge-perf-expert-parallel-overlap), [`nemo-mbridge-perf-hierarchical-context-parallel`](skills/nemo-mbridge-perf-hierarchical-context-parallel), [`nemo-mbridge-perf-megatron-fsdp`](skills/nemo-mbridge-perf-megatron-fsdp), [`nemo-mbridge-perf-memory-tuning`](skills/nemo-mbridge-perf-memory-tuning), [`nemo-mbridge-perf-moe-comm-overlap`](skills/nemo-mbridge-perf-moe-comm-overlap), [`nemo-mbridge-perf-moe-dispatcher-selection`](skills/nemo-mbridge-perf-moe-dispatcher-selection), [`nemo-mbridge-perf-moe-hardware-configs`](skills/nemo-mbridge-perf-moe-hardware-configs), [`nemo-mbridge-perf-moe-long-context`](skills/nemo-mbridge-perf-moe-long-context), [`nemo-mbridge-perf-moe-optimization-workflow`](skills/nemo-mbridge-perf-moe-optimization-workflow), [`nemo-mbridge-perf-moe-vlm-training`](skills/nemo-mbridge-perf-moe-vlm-training), [`nemo-mbridge-perf-parallelism-strategies`](skills/nemo-mbridge-perf-parallelism-strategies), [`nemo-mbridge-perf-sequence-packing`](skills/nemo-mbridge-perf-sequence-packing), [`nemo-mbridge-perf-tp-dp-comm-overlap`](skills/nemo-mbridge-perf-tp-dp-comm-overlap), [`nemo-mbridge-recipe-recommender`](skills/nemo-mbridge-recipe-recommender), [`nemo-mbridge-resiliency`](skills/nemo-mbridge-resiliency) |
| **NeMo Platform** | NeMo Platform brings NVIDIA NeMo libraries together under one CLI, Python SDK, and web UI | [`nemo-evaluator-plugin`](skills/nemo-evaluator-plugin), [`nemo-data-designer-plugin`](skills/nemo-data-designer-plugin) |
| **NeMo Relay** | Skills to help get started and use NeMo Relay - a runtime for instrumenting and controlling AI agents across harnesses, applications, and frameworks. | [`nemo-relay-install`](skills/nemo-relay-install), [`nemo-relay-get-started`](skills/nemo-relay-get-started), [`nemo-relay-instrument-calls`](skills/nemo-relay-instrument-calls), [`nemo-relay-instrument-context-isolation`](skills/nemo-relay-instrument-context-isolation), [`nemo-relay-instrument-typed-wrappers`](skills/nemo-relay-instrument-typed-wrappers), [`nemo-relay-plugin-adaptive-tuning`](skills/nemo-relay-plugin-adaptive-tuning), [`nemo-relay-plugin-build`](skills/nemo-relay-plugin-build), [`nemo-relay-plugin-observability`](skills/nemo-relay-plugin-observability), [`nemo-relay-migrate-from-flow`](skills/nemo-relay-migrate-from-flow), [`nemo-relay-debug-runtime-integration`](skills/nemo-relay-debug-runtime-integration) |
| **NeMo Retriever** | NeMo Retriever - deploy NeMo Retriever Library locally, extract information from corpus of data, and answer questions against the corpus. | [`nemo-retriever`](skills/nemo-retriever) |
| **NeMo-RL** | RLHF training on Ray — GRPO, DPO, and SFT for LLMs and VLMs with FSDP2 and Megatron-Core. | [`launch-nemo-rl`](skills/launch-nemo-rl), [`nemo-rl-auto-research`](skills/nemo-rl-auto-research), [`nemo-rl-brev-etiquette`](skills/nemo-rl-brev-etiquette), [`nemo-rl-docs`](skills/nemo-rl-docs), [`nemo-rl-session-memory`](skills/nemo-rl-session-memory) |
| **NemoClaw** | Secure agent sandboxing — run OpenClaw inside NVIDIA OpenShell with managed inference, policy management, remote deployment, sandbox monitoring. | [`nemoclaw-user-guide`](skills/nemoclaw-user-guide) |
| **Nemotron** | Author end-to-end model development, customization, evaluation, and deployment pipelines using the NVIDIA AI stack. | [`nemotron-customize`](skills/nemotron-customize), [`nemotron-retrieval-recipes`](skills/nemotron-retrieval-recipes), [`nemotron-policy-generator`](skills/nemotron-policy-generator) |
| **Nemotron Speech** | Deploy and operate NVIDIA Nemotron Speech (Riva) NIMs — ASR, TTS, and NMT, cloud-hosted via build.nvidia.com or self-hosted on your own GPU. | [`nemotron-speech`](skills/nemotron-speech), [`nemotron-asr-finetune`](skills/nemotron-asr-finetune) |
| **Physical AI** | Physical AI skills for simulation, synthetic data generation, training, validation and deployment and more. | [`omniverse-cad-to-simready`](skills/omniverse-cad-to-simready), [`omniverse-realtime-viewer`](skills/omniverse-realtime-viewer), [`omniverse-usd-performance-tuning`](skills/omniverse-usd-performance-tuning), [`physical-ai-infrastructure-setup-and-resilient-scaling`](skills/physical-ai-infrastructure-setup-and-resilient-scaling), [`physical-ai-neural-reconstruction`](skills/physical-ai-neural-reconstruction), [`physical-ai-defect-image-generation`](skills/physical-ai-defect-image-generation), [`physical-ai-video-data-augmentation`](skills/physical-ai-video-data-augmentation), [`physical-ai-people-attribute-search`](skills/physical-ai-people-attribute-search) |
| **PhysicsNeMo** | NVIDIA PhysicsNeMo - Open-source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods. | [`physicsnemo-discover`](skills/physicsnemo-discover) |
| **RAG Blueprint** | RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm. | [`rag-blueprint`](skills/rag-blueprint), [`rag-eval`](skills/rag-eval), [`rag-perf`](skills/rag-perf) |
| **Skill Card Generator** | Reads an agent skill's source files and produces a skill card plus a review table. Use when a skill directory exists and a governance card needs to be generated or updated. | [`skill-card-generator`](skills/skill-card-generator) |
| **TAO Toolkit** | NVIDIA TAO Toolkit - fine-tune and optimize 100+ pretrained vision AI models with your own data using low-code microservices, then export production-ready models for edge or cloud deployment. | [`tao-analyze-changenet-rca`](skills/tao-analyze-changenet-rca), [`tao-finetune-huggingface-model`](skills/tao-finetune-huggingface-model), [`tao-port-huggingface-model`](skills/tao-port-huggingface-model), [`tao-run-automl`](skills/tao-run-automl), [`tao-run-automl-deft-pipeline`](skills/tao-run-automl-deft-pipeline), [`tao-run-deft-aoi`](skills/tao-run-deft-aoi), [`tao-run-inference-service`](skills/tao-run-inference-service), [`tao-train-single-step`](skills/tao-train-single-step), [`tao-analyze-gaps-visual-changenet`](skills/tao-analyze-gaps-visual-changenet), [`tao-analyze-gaps-vlm-bcq`](skills/tao-analyze-gaps-vlm-bcq), [`tao-convert-dataset-format`](skills/tao-convert-dataset-format), [`tao-generate-image-grounding`](skills/tao-generate-image-grounding), [`tao-generate-referring-expressions`](skills/tao-generate-referring-expressions), [`tao-generate-video-reasoning-annotations`](skills/tao-generate-video-reasoning-annotations), [`tao-mine-aoi-images`](skills/tao-mine-aoi-images), [`tao-route-visual-changenet-samples`](skills/tao-route-visual-changenet-samples), [`tao-validate-dataset-format`](skills/tao-validate-dataset-format), [`tao-finetune-clip`](skills/tao-finetune-clip), [`tao-finetune-cosmos-embed`](skills/tao-finetune-cosmos-embed), [`tao-finetune-cosmos-reason`](skills/tao-finetune-cosmos-reason), [`tao-train-action-recognition`](skills/tao-train-action-recognition), [`tao-train-bevfusion`](skills/tao-train-bevfusion), [`tao-train-centerpose`](skills/tao-train-centerpose), [`tao-train-deformable-detr`](skills/tao-train-deformable-detr), [`tao-train-depth-anything-v2`](skills/tao-train-depth-anything-v2), [`tao-train-dino`](skills/tao-train-dino), [`tao-train-fast-foundation-stereo`](skills/tao-train-fast-foundation-stereo), [`tao-train-foundation-stereo`](skills/tao-train-foundation-stereo), [`tao-train-grounding-dino`](skills/tao-train-grounding-dino), [`tao-train-image-classification`](skills/tao-train-image-classification), [`tao-train-mask-auto-encoder`](skills/tao-train-mask-auto-encoder), [`tao-train-mask-auto-label`](skills/tao-train-mask-auto-label), [`tao-train-mask-grounding-dino`](skills/tao-train-mask-grounding-dino), [`tao-train-mask2former`](skills/tao-train-mask2former), [`tao-train-metric-learning-recognition`](skills/tao-train-metric-learning-recognition), [`tao-train-nvdinov2`](skills/tao-train-nvdinov2), [`tao-train-nvpanoptix3d`](skills/tao-train-nvpanoptix3d), [`tao-train-ocdnet`](skills/tao-train-ocdnet), [`tao-train-ocrnet`](skills/tao-train-ocrnet), [`tao-train-oneformer`](skills/tao-train-oneformer), [`tao-train-optical-inspection`](skills/tao-train-optical-inspection), [`tao-train-pointpillars`](skills/tao-train-pointpillars), [`tao-train-pose-classification`](skills/tao-train-pose-classification), [`tao-train-reid`](skills/tao-train-reid), [`tao-train-rtdetr`](skills/tao-train-rtdetr), [`tao-train-segformer`](skills/tao-train-segformer), [`tao-train-sparse4d`](skills/tao-train-sparse4d), [`tao-train-visual-changenet`](skills/tao-train-visual-changenet), [`tao-run-on-brev`](skills/tao-run-on-brev), [`tao-run-on-kubernetes`](skills/tao-run-on-kubernetes), [`tao-run-on-local-docker`](skills/tao-run-on-local-docker), [`tao-run-on-slurm`](skills/tao-run-on-slurm), [`tao-run-platform`](skills/tao-run-platform), [`tao-setup-nvidia-gpu-host`](skills/tao-setup-nvidia-gpu-host), [`tao-launch-workflow`](skills/tao-launch-workflow), [`tao-list-capabilities`](skills/tao-list-capabilities) |
| **TileGym** | Tile-based GPU programming — adding new kernels, cross-framework conversion, and performance optimization. | [`tilegym-adding-cutile-kernel`](skills/tilegym-adding-cutile-kernel), [`tilegym-converting-cutile-to-julia`](skills/tilegym-converting-cutile-to-julia), [`tilegym-converting-cutile-to-triton`](skills/tilegym-converting-cutile-to-triton), [`tilegym-cutile-autotuning`](skills/tilegym-cutile-autotuning), [`tilegym-cutile-python`](skills/tilegym-cutile-python), [`tilegym-improve-cutile-kernel-perf`](skills/tilegym-improve-cutile-kernel-perf), [`tilegym-monkey-patch-kernels-to-transformers`](skills/tilegym-monkey-patch-kernels-to-transformers) |
| **Video Search and Summarization** | VSS Blueprint — deploy profiles, search and summarize video, generate analysis reports, manage alerts and incidents, query VIOS sensors, and use the RTVI VLM microservice. | [`vss-ask-video`](skills/vss-ask-video), [`vss-deploy-dense-captioning`](skills/vss-deploy-dense-captioning), [`vss-deploy-detection-tracking-2d`](skills/vss-deploy-detection-tracking-2d), [`vss-deploy-detection-tracking-3d`](skills/vss-deploy-detection-tracking-3d), [`vss-deploy-profile`](skills/vss-deploy-profile), [`vss-deploy-video-embedding`](skills/vss-deploy-video-embedding), [`vss-generate-video-calibration`](skills/vss-generate-video-calibration), [`vss-generate-video-report`](skills/vss-generate-video-report), [`vss-manage-alerts`](skills/vss-manage-alerts), [`vss-manage-video-io-storage`](skills/vss-manage-video-io-storage), [`vss-query-analytics`](skills/vss-query-analytics), [`vss-search-archive`](skills/vss-search-archive), [`vss-setup-behavior-analytics`](skills/vss-setup-behavior-analytics), [`vss-setup-video-analytics-api`](skills/vss-setup-video-analytics-api), [`vss-summarize-video`](skills/vss-summarize-video) |
<!-- skills-table-end -->

---

## Getting Help & Contributing

**Where to file an issue depends on what's broken:**

- **Skill content issues** (a specific skill has a bug, missing functionality, or incorrect content) — file in the **source repo** for that product, using the per-product table below.
- **Catalog issues** (catalog README errors, sync workflow problems, distribution channels, signing/verification flow, docs in this repo) — file [here](../../issues/new/choose) using the catalog issue templates: **Bug Report**, **Feature Request**, or **Documentation Request or Correction**.
- **Questions or general discussion** — use [Discussions](../../discussions). The issue tracker is reserved for bug reports, feature proposals with a design, and documentation issues.
- **Security vulnerabilities** — follow the disclosure process in [SECURITY.md](SECURITY.md); do not open a public issue.

Per-product source repo links:

<!-- help-table-start -->
| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **AIQ** | [Issues](https://github.com/NVIDIA-AI-Blueprints/aiq/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/aiq/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/SECURITY.md) |
| **CUDA-Q** | [Issues](https://github.com/NVIDIA/cuda-quantum/issues) | [Discussions](https://github.com/NVIDIA/cuda-quantum/discussions) | [Contributing](https://github.com/NVIDIA/cuda-quantum/blob/main/Contributing.md) | [Security](https://github.com/NVIDIA/cuda-quantum/blob/main/SECURITY.md) |
| **cuDF** | [Issues](https://github.com/rapidsai/cudf/issues) | [Discussions](https://github.com/rapidsai/cudf/discussions) | [Contributing](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md) | [Security](https://github.com/rapidsai/cudf/blob/main/SECURITY.md) |
| **cuFOLIO** | [Issues](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO/blob/main/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
| **cuPyNumeric** | [Issues](https://github.com/nv-legate/cupynumeric/issues) | — | [Contributing](https://github.com/nv-legate/cupynumeric/blob/main/CONTRIBUTING.md) | — |
| **DALI** | [Issues](https://github.com/NVIDIA/DALI/issues) | — | [Contributing](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md) | — |
| **Data Designer** | [Issues](https://github.com/NVIDIA-NeMo/DataDesigner/issues) | [Discussions](https://github.com/NVIDIA-NeMo/DataDesigner/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/SECURITY.md) |
| **DeepStream** | [Issues](https://github.com/NVIDIA/DeepStream/issues) | — | [Contributing](https://github.com/NVIDIA/DeepStream/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/DeepStream/blob/main/SECURITY.md) |
| **Digital Health** | [Issues](https://github.com/NVIDIA/digital-health-examples/issues) | — | [Contributing](https://github.com/NVIDIA/digital-health-examples/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/digital-health-examples/blob/main/SECURITY.md) |
| **DOCA** | [Issues](https://github.com/NVIDIA-DOCA/doca-skills/issues) | — | [Contributing](https://github.com/NVIDIA-DOCA/doca-skills/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-DOCA/doca-skills/blob/main/SECURITY.md) |
| **Dynamo** | [Issues](https://github.com/ai-dynamo/dynamo/issues) | [Discussions](https://github.com/ai-dynamo/dynamo/discussions) | [Contributing](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md) | [Security](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) |
| **Earth2Studio** | [Issues](https://github.com/NVIDIA/earth2studio/issues) | [Discussions](https://github.com/NVIDIA/earth2studio/discussions) | [Contributing](https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md) | — |
| **Holoscan SDK** | [Issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues) | — | [Contributing](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/CONTRIBUTING.md) | [Security](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/SECURITY.md) |
| **Holoscan Sensor Bridge** | [Issues](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/issues) | — | [Contributing](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/CONTRIBUTING.md) | — |
| **Isaac for Healthcare Workflows** | [Issues](https://github.com/isaac-for-healthcare/i4h-workflows/issues) | — | [Contributing](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/CONTRIBUTING.md) | [Security](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/SECURITY.md) |
| **Jetson BSP** | [Issues](https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills/issues) | — | [Contributing](https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills/blob/main/SECURITY.md) |
| **Jetson Device** | [Issues](https://github.com/NVIDIA-AI-IOT/jetson-device-skills/issues) | — | [Contributing](https://github.com/NVIDIA-AI-IOT/jetson-device-skills/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-IOT/jetson-device-skills/blob/main/SECURITY.md) |
| **Medical AI Skills** | [Issues](https://github.com/NVIDIA-Medtech/medical-AI-skills/issues) | — | [Contributing](https://github.com/NVIDIA-Medtech/medical-AI-skills/blob/dev/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-Medtech/medical-AI-skills/blob/dev/SECURITY.md) |
| **Megatron-Core** | [Issues](https://github.com/NVIDIA/Megatron-LM/issues) | [Discussions](https://github.com/NVIDIA/Megatron-LM/discussions) | [Contributing](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) | — |
| **NeMo AutoModel** | [Issues](https://github.com/NVIDIA-NeMo/Automodel/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Automodel/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Automodel/blob/main/SECURITY.md) |
| **NeMo MBridge** | [Issues](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/SECURITY.md) |
| **NeMo Platform** | [Issues](https://github.com/NVIDIA-NeMo/nemo-platform/issues) | [Discussions](https://github.com/NVIDIA-NeMo/nemo-platform/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/SECURITY.md) |
| **NeMo Relay** | [Issues](https://github.com/NVIDIA/NeMo-Relay/issues) | [Discussions](https://github.com/NVIDIA/NeMo-Relay/discussions) | [Contributing](https://github.com/NVIDIA/NeMo-Relay/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NeMo-Relay/blob/main/SECURITY.md) |
| **NeMo Retriever** | [Issues](https://github.com/NVIDIA/NeMo-Retriever/issues) | [Discussions](https://github.com/NVIDIA/NeMo-Retriever/discussions) | [Contributing](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NeMo-Retriever/blob/main/SECURITY.md) |
| **NeMo-RL** | [Issues](https://github.com/NVIDIA-NeMo/RL/issues) | [Discussions](https://github.com/NVIDIA-NeMo/RL/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/RL/blob/main/SECURITY.md) |
| **NemoClaw** | [Issues](https://github.com/NVIDIA/NemoClaw/issues) | [Discussions](https://github.com/NVIDIA/NemoClaw/discussions) | [Contributing](https://github.com/NVIDIA/NemoClaw/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NemoClaw/blob/main/SECURITY.md) |
| **Nemotron** | [Issues](https://github.com/NVIDIA-NeMo/Nemotron/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Nemotron/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/SECURITY.md) |
| **Nemotron Speech** | [Issues](https://github.com/nvidia-riva/Nemotron-speech-skills/issues) | — | [Contributing](https://github.com/nvidia-riva/Nemotron-speech-skills/blob/main/CONTRIBUTING.md) | [Security](https://github.com/nvidia-riva/Nemotron-speech-skills/blob/main/SECURITY.md) |
| **Physical AI** | [Issues](https://github.com/NVIDIA/physical-ai-data-factory/issues) | — | [Contributing](https://github.com/NVIDIA/physical-ai-data-factory/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/physical-ai-data-factory/blob/main/SECURITY.md) |
| **PhysicsNeMo** | [Issues](https://github.com/NVIDIA/physicsnemo/issues) | [Discussions](https://github.com/NVIDIA/physicsnemo/discussions) | [Contributing](https://github.com/NVIDIA/physicsnemo/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/physicsnemo/blob/main/SECURITY.md) |
| **RAG Blueprint** | [Issues](https://github.com/NVIDIA-AI-Blueprints/rag/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/rag/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/SECURITY.md) |
| **Skill Card Generator** | [Issues](https://github.com/NVIDIA/Trustworthy-AI/issues) | — | [Contributing](https://github.com/NVIDIA/Trustworthy-AI/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Trustworthy-AI/blob/main/SECURITY.md) |
| **TAO Toolkit** | [Issues](https://github.com/NVIDIA-TAO/tao-skills-bank/issues) | [Discussions](https://github.com/NVIDIA-TAO/tao-skills-bank/discussions) | [Contributing](https://github.com/NVIDIA-TAO/tao-skills-bank/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-TAO/tao-skills-bank/blob/main/SECURITY.md) |
| **TileGym** | [Issues](https://github.com/NVIDIA/TileGym/issues) | — | [Contributing](https://github.com/NVIDIA/TileGym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TileGym/blob/main/SECURITY.md) |
| **Video Search and Summarization** | [Issues](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/main/SECURITY.md) |
<!-- help-table-end -->

For issues with **this catalog repo itself** (README, structure, listing a new product): [open an issue here](../../issues).

---

## Verifying Skills

Every published skill ships with a detached OMS signature (`skill.oms.sig`). The sync pipeline drops any skill missing the required artifacts before publishing, so every skill in the catalog carries:

- `SKILL.md` — the skill instructions consumed by the agent
- `skill-card.md` — skill identity and governance card
- `skill.oms.sig` — detached OMS signature (verifiable against `nv-agent-root-cert.pem`)
- A Tier-3 evaluation dataset — accepted at `evals/evals.json`, `evals/*.json`, `eval/*.json`, or `benchmark/evals.json`
- `BENCHMARK.md` — generated benchmark report capturing verifiable uplift data

Verify a skill against the NVIDIA trust anchor [`nv-agent-root-cert.pem`](nv-agent-root-cert.pem):

```bash
pip install model-signing
model_signing verify certificate SKILL_DIR \
  --signature SKILL_DIR/skill.oms.sig \
  --certificate_chain nv-agent-root-cert.pem \
  --ignore_unsigned_files
```

A successful verification confirms that the skill contents have not been modified since signing by NVIDIA.

See [Verify Signed Agent Skills](docs/signing-agent-skills.mdx) for signature layout, the trust pipeline, and policy options.

---

## Roadmap

- ✅ Public skills catalog with NVIDIA-verified skills across multiple products
- ✅ Automated sync pipeline with skills mirrored from product repos daily
- ✅ Security scanning for all published skills covering instruction safety and supply-chain integrity
- ✅ Skills signing so every published skill carries a verifiable NVIDIA signature
- ✅ Skills universal evaluation criteria and task-specific criteria
- ✅ Skill Card with machine-readable metadata for identity, provenance, quality, and behavioral boundaries
- ✅ Sync-time compliance gates — signature drift detection and missing-artifact enforcement
- ✅ Syndication to external marketplaces — Skills.sh, Codex plugin, Claude Code plugin, ClawHub, Hermes Hub
- 🔲 Syndication to additional MCP hubs and partner channels

---

## Repository Structure

```
NVIDIA/skills/
├── skills/                      # NVIDIA-verified skills (count grows continuously),
│   │                              synced from upstream product repos
│   ├── README.md                 # Browser-facing install guidance
│   ├── <product-prefix>-*/       # Flat layout — one dir per skill, product-prefixed
│   │                               # e.g. aiq-*, cufolio, cuopt-*, cupynumeric-*,
│   │                               # dali-*, deepstream-*, dicom-*, digital-health-*,
│   │                               # dynamo-*, earth2studio-*, holoscan-*, hsb-*,
│   │                               # jetson-*, launch-nemo-rl, mcore-*,
│   │                               # nemo-automodel-*, nemo-data-designer-plugin,
│   │                               # nemo-evaluator-plugin, nemo-mbridge-* (20 skills),
│   │                               # nemo-retriever, nemo-rl-* (4 skills),
│   │                               # nemoclaw-user-guide, nemotron-*, nemotron-speech,
│   │                               # nv-* (medical AI), physicsnemo-*, rag-*,
│   │                               # skill-card-generator, tao-*, tilegym-*,
│   │                               # vss-* (15 skills), accelerated-computing-cudf,
│   │                               # cudaq-guide
│   ├── omniverse-*/              # Physical AI — manually staged (see manual-components.yml)
│   └── physical-ai-*/            # Physical AI — manually staged
├── components.d/                # Product registry — one file per component, teams onboard here
│   ├── README.md                 # Schema and onboarding instructions
│   └── <product>.yml             # one file per registered product
├── plugins/                     # Packaged plugin distributions
│   └── nvidia-skills/            # Curated NVIDIA skills bundle (Claude Code, Codex)
├── plugins.d/                   # Plugin build registry — config for `build-plugins.py`
│   ├── README.md
│   ├── _defaults.yml
│   └── nvidia-skills.yml
├── .claude-plugin/              # Claude Code marketplace metadata
│   └── marketplace.json
├── .agents/plugins/             # Agent marketplace metadata (other clients)
│   └── marketplace.json
├── docs/                        # Long-form documentation (published via Fern)
│   ├── README.md                 # How to build the docs locally
│   ├── index.mdx
│   ├── advanced-install.mdx
│   ├── agent-skill-trust-pipeline.mdx
│   ├── release-checklist.mdx
│   ├── scanning-agent-skills.mdx
│   ├── signing-agent-skills.mdx
│   └── skill-cards.mdx
├── fern/                        # Fern docs site configuration
├── .github/
│   ├── workflows/                # Sync pipeline, plugin validation, DCO check, author verify
│   └── scripts/                  # regenerate-readme.sh, build-plugins.py,
│                                 # manual-components.yml (temp Physical AI catalog
│                                 # exception, removed after Computex 2026),
│                                 # marketplace/metadata.json (skill metadata sidecar)
├── nv-agent-root-cert.pem       # Trust anchor for OMS signature verification
├── skills.sh.json               # Skills.sh marketplace grouping config
├── CHANGELOG.md
├── CONTRIBUTING.md              # Contribution guidelines
├── SECURITY.md                  # Security reporting policy
├── CODE_OF_CONDUCT.md           # Community code of conduct
├── LICENSE-APACHE               # Apache 2.0 (source code)
└── LICENSE-CC-BY-4.0            # CC BY 4.0 (documentation/skills)
```

Skills are maintained in their respective product repos (see the **Source** column in the [Skill Catalog](#skill-catalog)) and synced to this repo daily. Products only appear under `skills/` after the sync pipeline confirms each skill carries:

- `skill.oms.sig` — detached OMS-format signature (verifiable against `nv-agent-root-cert.pem`)
- `skill-card.md` — skill identity and governance card
- A Tier-3 evaluation dataset — accepted at `evals/evals.json`, `evals/*.json`, `eval/*.json`, or `benchmark/evals.json`

When evaluation runs produce a `BENCHMARK.md`, it ships alongside the skill so consumers can see verifiable benchmark uplift data.

---

## Standards & Compatibility

This repository adheres to the [Agent Skills specification](https://agentskills.io/specification):

- Skills are portable directories with a `SKILL.md` file at their root.
- Metadata uses YAML frontmatter with required `name` and `description` fields.
- Skills follow a progressive disclosure model — lightweight metadata loads at startup, full instructions load on activation.
- Validate your skill using the [`skills-ref`](https://github.com/agentskills/agentskills/tree/main/skills-ref) reference library.

---

## License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This code is dual-licensed with documentation/skills under the CC-BY-4.0 AND source code under Apache-2.0 license terms. The full license texts can be found in [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0) respectively.
