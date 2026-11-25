# nvidia-tao-toolkit Plugin

**Best evaluated agent: Codex · 98.06/100 · PASS · Complete score coverage.** See [BENCHMARK.md](BENCHMARK.md) for both agents and their baseline comparisons.

## Description

NVIDIA TAO skills for training, fine-tuning, evaluation, export and deployment of vision and multimodal models; data preparation and generation; AutoML; and application workflows on Docker, Kubernetes, SLURM, Brev or Python virtual environments.

**Owner-declared readiness:** Commercial-use and external-use readiness declarations are not supplied in the package. The license and evaluation results are reported separately below.

## Owner

NVIDIA. The manifests identify NVIDIA as the author and link to [NVIDIA-TAO/tao-skill-bank](https://github.com/NVIDIA-TAO/tao-skill-bank).

### License/Terms of Use

**Package license:** [Apache-2.0](LICENSE), as declared in the [manifest](plugin.json). Component license exceptions are listed under Skills.

## Use Case

Developers and platform operators use the skills to select TAO models and workflows, prepare datasets, train or fine-tune models, evaluate and export results, run inference, and submit or monitor jobs on the selected execution platform. See the [package README](README.md) and [workflow guidance](AGENTS.md).

### Compatible Agents

| Agent | Declared minimum version | Basis and limits |
| --- | --- | --- |
| Claude Code | Not specified | Host manifest and README support; evaluation covers bundled skills. |
| Codex | Not specified | Host manifest and README support; evaluation covers bundled skills. |
| Cursor | Not specified | Host manifest and README support; no Cursor evaluation result is included. |

The [README](README.md) describes host-specific behavior: Claude Code and Cursor CLI run the session-start hook; Codex does not load that hook for this package's root Agent Plugins manifest. Whole-plugin loading and every host/version combination are outside the evaluation scope.

### Release Management

| Channel | Location | Version / release status |
| --- | --- | --- |
| GitHub source | [NVIDIA-TAO/tao-skill-bank](https://github.com/NVIDIA-TAO/tao-skill-bank) | Package version 0.1.19; release date not declared. |
| Agent marketplace/catalog | No listing URL declared | README describes installation through the consuming host's marketplace/catalog. |

### Deployment Geography for Use

Not declared in the package. The owner must provide any applicable geographic restrictions or deployment declaration.

## Plugin Package

**Package format:** Agent Plugins manifest, declared schema 1.0.0, with Claude Code, Codex and Cursor host manifests.

**Manifest locations:** [plugin.json](plugin.json), [.claude-plugin/plugin.json](.claude-plugin/plugin.json), [.codex-plugin/plugin.json](.codex-plugin/plugin.json), [.cursor-plugin/plugin.json](.cursor-plugin/plugin.json).

**Package unit:** Directory.

**Client extension namespaces:** None declared. The host manifest directories are standard host integration folders.

**Directory layout** (skill companion files are abbreviated):

```text
nvidia-tao-toolkit/
├── plugin.json
├── .claude-plugin/plugin.json
├── .codex-plugin/plugin.json
├── .cursor-plugin/plugin.json
├── AGENTS.md
├── README.md
├── LICENSE
├── BENCHMARK.md
├── plugin-card.md
├── hooks/
├── scripts/
├── catalog/
├── templates/
├── versions.yaml
├── evals/evals.json
└── skills/<skill>/SKILL.md
```

## Plugin Components

**Skills:** 82 bundled skills, exposed through the three declared hosts. **Agents:** No separate agent components are declared; `AGENTS.md` contains orchestration guidance. **MCP servers:** None declared in the package manifests. The Brev workflow can use an independently configured Brev skill or MCP server; it is not bundled here.

### Skills

**Component licenses:** `tao-run-deft-aoi`, `tao-run-deft-aoi-cosmos3` and `tao-run-deft-pas` declare `Apache-2.0 AND CC-BY-4.0`; the other 79 skills declare `Apache-2.0`.

Individual skill cards and signatures are not supplied in the package. Skill names link to their source; host support below is declared support, not independent verification of every host/version.

<details>
<summary>Bundled skills and signature status</summary>

| Skill | Declared hosts | Signature verification |
| --- | --- | --- |
| [paidf-cosmos-predict](skills/paidf-cosmos-predict/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-analyze-changenet-rca](skills/tao-analyze-changenet-rca/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-analyze-detection-kpi](skills/tao-analyze-detection-kpi/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-analyze-gaps-od-map](skills/tao-analyze-gaps-od-map/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-analyze-gaps-visual-changenet](skills/tao-analyze-gaps-visual-changenet/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-analyze-gaps-vlm-bcq](skills/tao-analyze-gaps-vlm-bcq/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-artifacts](skills/tao-artifacts/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-convert-dataset-format](skills/tao-convert-dataset-format/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-data-io](skills/tao-data-io/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-anomalygennext](skills/tao-finetune-anomalygennext/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-clip](skills/tao-finetune-clip/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-cosmos-embed](skills/tao-finetune-cosmos-embed/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-cosmos-reason](skills/tao-finetune-cosmos-reason/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-huggingface-model](skills/tao-finetune-huggingface-model/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-kumo-anomaly](skills/tao-finetune-kumo-anomaly/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-kumo-forecast](skills/tao-finetune-kumo-forecast/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-finetune-video-clip](skills/tao-finetune-video-clip/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-generate-anomalies](skills/tao-generate-anomalies/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-generate-image-embeddings](skills/tao-generate-image-embeddings/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-generate-image-grounding](skills/tao-generate-image-grounding/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-generate-od-defects](skills/tao-generate-od-defects/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-generate-referring-expressions](skills/tao-generate-referring-expressions/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-generate-video-reasoning-annotations](skills/tao-generate-video-reasoning-annotations/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-launch-workflow](skills/tao-launch-workflow/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-list-capabilities](skills/tao-list-capabilities/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-mine-aoi-images](skills/tao-mine-aoi-images/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-mine-nearest-neighbors](skills/tao-mine-nearest-neighbors/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-mine-od-images](skills/tao-mine-od-images/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-port-huggingface-model](skills/tao-port-huggingface-model/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-prepare-anomalygennext-inputs](skills/tao-prepare-anomalygennext-inputs/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-route-visual-changenet-samples](skills/tao-route-visual-changenet-samples/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-automl](skills/tao-run-automl/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-automl-deft-pipeline](skills/tao-run-automl-deft-pipeline/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-deft-aoi](skills/tao-run-deft-aoi/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-deft-aoi-cosmos3](skills/tao-run-deft-aoi-cosmos3/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-deft-cr-its-mining](skills/tao-run-deft-cr-its-mining/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-deft-object-detection](skills/tao-run-deft-object-detection/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-deft-od-aoi](skills/tao-run-deft-od-aoi/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-deft-pas](skills/tao-run-deft-pas/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-dinov3-ssl-deft](skills/tao-run-dinov3-ssl-deft/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-inference-service](skills/tao-run-inference-service/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-on-brev](skills/tao-run-on-brev/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-on-docker](skills/tao-run-on-docker/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-on-kubernetes](skills/tao-run-on-kubernetes/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-on-slurm](skills/tao-run-on-slurm/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-run-on-virtualenv](skills/tao-run-on-virtualenv/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-setup](skills/tao-setup/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-setup-nvidia-gpu-host](skills/tao-setup-nvidia-gpu-host/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-token-efficient-execution](skills/tao-token-efficient-execution/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-action-recognition](skills/tao-train-action-recognition/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-bevfusion](skills/tao-train-bevfusion/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-centerpose](skills/tao-train-centerpose/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-codetr](skills/tao-train-codetr/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-deformable-detr](skills/tao-train-deformable-detr/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-depth-anything-v2](skills/tao-train-depth-anything-v2/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-dino](skills/tao-train-dino/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-dinov3](skills/tao-train-dinov3/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-fast-foundation-stereo](skills/tao-train-fast-foundation-stereo/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-foundation-stereo](skills/tao-train-foundation-stereo/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-grounding-dino](skills/tao-train-grounding-dino/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-image-classification](skills/tao-train-image-classification/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-mask-auto-encoder](skills/tao-train-mask-auto-encoder/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-mask-auto-label](skills/tao-train-mask-auto-label/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-mask-grounding-dino](skills/tao-train-mask-grounding-dino/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-mask2former](skills/tao-train-mask2former/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-metric-learning-recognition](skills/tao-train-metric-learning-recognition/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-nvdinov2](skills/tao-train-nvdinov2/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-nvpanoptix3d](skills/tao-train-nvpanoptix3d/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-ocdnet](skills/tao-train-ocdnet/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-ocrnet](skills/tao-train-ocrnet/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-oneformer](skills/tao-train-oneformer/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-optical-inspection](skills/tao-train-optical-inspection/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-pointpillars](skills/tao-train-pointpillars/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-pose-classification](skills/tao-train-pose-classification/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-reid](skills/tao-train-reid/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-rtdetr](skills/tao-train-rtdetr/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-segformer](skills/tao-train-segformer/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-single-step](skills/tao-train-single-step/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-sparse4d](skills/tao-train-sparse4d/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-train-visual-changenet](skills/tao-train-visual-changenet/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-validate-dataset-format](skills/tao-validate-dataset-format/SKILL.md) | Claude Code, Codex, Cursor | Not verified |
| [tao-validate-recipe-transfer](skills/tao-validate-recipe-transfer/SKILL.md) | Claude Code, Codex, Cursor | Not verified |

</details>

### Hooks

[hooks/hooks.json](hooks/hooks.json) registers one SessionStart event with two commands:

| Command | Added session context | Other effects |
| --- | --- | --- |
| `session_start.sh identity` | First portion of `AGENTS.md`, including identity and discovery guidance. | No workflow is launched. |
| `session_start.sh context` | Remaining guidance, plugin path, names of configured credential variables, and available setup/card-pack hints. | In Claude Code, appends a `TAO_SKILL_BANK_PATH` export to the session environment file. |

The [hook source](hooks/session_start.sh) reads package guidance and checks tool/variable presence. It reports credential names, not values, and makes no network request, installs no package and starts no container. The [README](README.md) reports hook execution in Claude Code and Cursor CLI, and no hook execution in Codex for this package format. The hook adds context; it does not implement a launch-approval boundary.

### Commands, Rules, and Other Client-Specific Components

No standalone native command or rule components are declared. Shared [scripts](scripts) support model/capability discovery, image/version resolution, launch preflight, job records, secret lint/redaction, agent guidance management, SSH setup and distributed-training probes.

| Resource | Source |
| --- | --- |
| Kubernetes action job | [action-job.yaml.tmpl](templates/k8s/action-job.yaml.tmpl) |
| Kubernetes indexed job | [indexed-job.yaml.tmpl](templates/k8s/indexed-job.yaml.tmpl) |
| Kubernetes single-pod job | [single-pod-job.yaml.tmpl](templates/k8s/single-pod-job.yaml.tmpl) |
| SLURM multi-node job | [multinode.sbatch.tmpl](templates/slurm/multinode.sbatch.tmpl) |
| SLURM single-node job | [singlenode.sbatch.tmpl](templates/slurm/singlenode.sbatch.tmpl) |

### Bundled Dependencies

No package-level SBOM or third-party runtime dependency inventory is supplied. Runtime prerequisites and optional integrations are described below and in each skill's `compatibility` field. External containers, models and packages are not made part of this card's verification by being referenced.

## Requirements / Dependencies

**Requires API key or external credential:** Workflow-dependent. Reading skills and listing capabilities needs no target-service credential. Image/model downloads, object storage and execution backends can require credentials.

| Credential or configuration | Used for |
| --- | --- |
| `NGC_KEY` | Authenticated NVIDIA container/model access. |
| `HF_TOKEN` | Gated Hugging Face models or datasets. |
| `BREV_API_TOKEN` or an active Brev login | Brev instance access. |
| AWS/S3 credentials and endpoint configuration | Dataset staging and result upload; standard AWS variables are supported, with mappings from legacy TAO variable names. |
| SSH keys, `SLURM_USER`, `SLURM_HOSTNAME` | SLURM access over SSH. |
| Kubeconfig or an in-cluster service account | Kubernetes access; optional image-pull and per-job Secrets support credentials inside jobs. |
| `WANDB_API_KEY` | Optional Weights & Biases experiment logging. |

**Credential supplied by:** The user/operator or the administrator of the selected registry, model provider, storage service or execution platform. Credentials reach the session through environment variables or a user-approved environment file.

**Runtime environment:** Bash and the standard shell utilities listed in the [README](README.md); Python 3.10+ and PyYAML for the shared helpers. Container workflows require the selected model's compatible NVIDIA GPU/driver and Docker/NVIDIA Container Toolkit environment. Kubernetes needs an authenticated `kubectl` client and GPU-capable workers; SLURM needs SSH and scheduler access; Brev needs its CLI and an active login. The virtualenv platform uses a Python environment with the selected training dependencies. Model-specific requirements and image/wheel pins are in each skill and [versions.yaml](versions.yaml).

**Additional workflow dependencies:** Data staging can require `aws`, `s5cmd`, `boto3` and, for annotation-selective downloads, pandas/pyarrow. AutoML uses the `nvidia-tao-automl` package and its TAO SDK dependency; the other native-platform paths do not require that SDK. Brev provisioning may use a separately installed Brev skill or external MCP server. These are conditional requirements, not prerequisites for reading the skills.

**Network access required:** Depends on the selected workflow: NVIDIA documentation and `nvcr.io`/NGC, Hugging Face, GitHub for documented downloads, configured object-storage endpoints, SSH/SLURM hosts, Kubernetes API endpoints, Brev services and optional experiment logging. This is a workflow summary, not an exhaustive egress allowlist. The session-start hook itself does not use the network.

**Subprocess execution:** The host can run the SessionStart commands. User-requested workflows run shell/Python helpers, container commands, `kubectl`, SSH/SLURM commands or `brev exec` under the host's granted permissions. Setup and optional integration installation can change the execution environment after the applicable approval.

**Filesystem read:** Packaged skills, guidance, catalogs, versions and templates; selected datasets, checkpoints, configuration and logs; and approved credential/configuration files required by the chosen backend.

**Filesystem write:** Workflow specs, staged data, checkpoints, results and logs; job records under `${TAO_STATE_DIR}/jobs` or `~/.tao/jobs`; the Claude Code session environment file; and, when explicitly requested, the marked TAO guidance block in a project's `AGENTS.md`. Cluster resources and object-storage writes depend on the selected workflow. Setup/install workflows can also change system or host-agent files after approval.

Sources: [README](README.md), [agent guidance](AGENTS.md), [data I/O skill](skills/tao-data-io/SKILL.md), and the platform skills listed above.

## Capability and Permission Summary

**Operations:** Read package and workflow data; write specs, artifacts and job state; execute tools and workloads; access selected network services; submit, inspect and cancel jobs. The [Codex manifest](.codex-plugin/plugin.json) declares `Read`, `Write` and `Interactive` interface capabilities. Host permissions and the approved task bound effective authority.

**Operation scope:** TAO model/data/application workflows and the chosen compute/storage backends. No plugin-hosted inbound API or independent background service is declared; separately launched jobs and services can continue after a session ends.

**Changes host agent autonomy:** Yes. Session-start context and the optional project guidance block affect routing, planning and tool selection. They do not grant additional host permissions. Source instructions require confirmation before side-effecting launches and installations, with a documented exception for small Python prerequisites; these instructions are not proof that every host enforces the gate.

## Known Technical Limitations

- Host behavior differs: Codex does not receive the package's SessionStart context automatically, and the optional project guidance block must be managed separately.
- GPU, registry access, backend permissions, data visibility and model/container compatibility are workflow-specific; reading a skill does not establish execution readiness.
- The virtualenv platform requires Linux procfs/pidfd support for active job cancellation; other supported POSIX paths fail closed rather than signal unproven process IDs. See [the platform skill](skills/tao-run-on-virtualenv/SKILL.md).
- The evaluation covers bundled skills on Claude Code and Codex, not native plugin loading, hooks, every host/version, all deployment environments or production hardware. Static analysis coverage is partial.

## Known Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Generated guidance or selected models/data may be unsuitable for a task. | Review the chosen skill, references, model/data terms and produced artifacts for the intended use. |
| Launches, installs, uploads and infrastructure operations can change state or incur cost. | Review the selected target, resolved image, inputs/outputs and launch plan; use host permissions and the documented approval gate. |
| Credential handling and artifact upload can expose sensitive data. | Use approved credential sources, secret lint/redaction and upload exclusions; inspect output destinations and access controls. |
| Session-start or project guidance changes agent behavior. | Review the packaged guidance and hook; disable the plugin or remove the marked guidance block when that behavior is not wanted. |

## Disable and Revoke

**Disable / uninstall:** Use the consuming host's plugin controls and start a fresh session. If the optional TAO project-guidance block was added, [tao_agent_identity.py](scripts/tao_agent_identity.py) provides a `remove` command for that marked block. Removing the plugin does not cancel submitted jobs or delete outputs; cancel jobs through the chosen platform and its recorded backend identity.

**Revoke credentials:** Revoke or rotate credentials with the issuing registry, model provider, cloud/storage service or cluster administrator, and end sessions/processes retaining them. Removing the plugin alone does not revoke provider credentials.

**Measured time to effect:** Not measured.

## Data Flow and Security

**Logging and traceability:** Agent/host command output, backend logs and per-job JSON records. [tao_job_record.py](scripts/tao_job_record.py) records job identity, resolved result location, backend reference and status transitions; it applies secret redaction before writes. Retention and access controls are deployment-specific.

**Persisted data:** Job records under `${TAO_STATE_DIR}/jobs` or `~/.tao/jobs`, workflow specs, datasets/checkpoints, results and logs, plus approved backend credential sidecars/Secrets and optional project guidance. [tao-data-io](skills/tao-data-io/SKILL.md) selects mounted storage or configured upload destinations; ephemeral storage requires explicit handling to preserve results.

**Data sent to external services:** Requested datasets, model artifacts, job/configuration data, outputs and logs can move to the selected compute/storage services. Gated model downloads and optional experiment logging use their configured providers. Agent/model-provider data handling depends on the consuming host, not on a separate model API configured by this package.

**Database name / type:** No package-managed database is declared. The job-record helper uses JSON files; downstream services may use their own storage.

**Authorization / human approval:** The [agent guidance](AGENTS.md) and [launch workflow](skills/tao-launch-workflow/SKILL.md) require the selected platform, model/image, input/output locations and launch plan to be established before execution. The workflow includes preflight/data-visibility checks and command secret linting. These are documented workflow controls; enforcement depends on the host and execution path.

**Security controls:** Credential values must not be requested in chat or printed; the hook reports presence by variable name. The package includes [secret lint/redaction](scripts/redact_secrets.py), redacted atomic job-record writes and checks before launch. Kubernetes and SLURM document backend-specific Secret/sidecar handling. These controls do not establish security approval of the package or downstream services.

**Exposure categories:** Outbound registry/model/storage/service access; cluster/SSH/cloud control-plane access; automatic SessionStart context injection on the hosts described above; and user-requested jobs that may outlive the session. No bundled inbound MCP or API server is declared.

**Exceptions:** None declared in the supplied package; no suppressions were applied in the validation report.

## Testing Completed

### Evaluation results

| Agent | With-plugin quality score | Verdict | Score coverage |
| --- | --- | --- | --- |
| Claude Code | 94.71/100 | PASS | Partial |
| **Codex** | **98.06/100** | **PASS** | **Complete** |

Codex has the highest with-plugin quality score. Claude Code has the greatest baseline improvement, **+70.56 points**; Codex's lift is **+17.73 points**. Quality score and baseline improvement are separate measures.

All **556/556 planned executions completed**, including 139 with-plugin and 139 baseline executions per agent. Evaluation covers the bundled skills; the package contains 82 skills. Native plugin loading, hooks, display-name behavior and root instructions were outside this evaluation's scope. See [BENCHMARK.md](BENCHMARK.md) for dimension scores, grading coverage and paired comparisons.

### Static validation

| Check | Result |
| --- | --- |
| CLI package validation | FAIL |
| Profile / selected hosts | External / Claude and Codex |
| HIGH / CRITICAL records | 529 / 2 |
| Unique HIGH/CRITICAL findings | 355 |
| Scanner execution errors | 0 |
| Bandit | Completed |
| Analysis coverage | Partial |
| Suppressions applied | None |

Static validation used NVCARPS CLI 1.3.1435. Findings remain open in that report; the runtime evaluation PASS does not clear them or establish publication approval.

| Review / artifact | Supplied status |
| --- | --- |
| Bundled MCP server security review | No bundled MCP server is declared; optional external integrations are outside this package. |
| Dependency / SBOM review | Review evidence and package SBOM not supplied. |
| Network security review | Approval evidence not supplied. |
| Product security review | Approval evidence not supplied. |
| Signing / signature verification | Whole-plugin signature: [plugin.oms.sig](plugin.oms.sig). Verification uses the external certificate chain and package file inventory. Individual skill signatures are not supplied. |

## Plugin Version(s)

**Version (manifest):** `0.1.19` in all four manifests.

**Signing profile:** External — job `SKILLS_EXTERNAL`, scope `SIGNING_SKILLS_EXTERNAL`.

**Content digest:** Per-file SHA-256 inventory carried in [plugin.oms.sig](plugin.oms.sig).

**Signature / provenance:** Detached whole-plugin signature: [plugin.oms.sig](plugin.oms.sig). Evaluation exports and signature files are excluded from the signed inventory. Individual skill signatures are not supplied.

## Reference(s)

- [TAO skill bank source](https://github.com/NVIDIA-TAO/tao-skill-bank)
- [Package README](README.md), [agent guidance](AGENTS.md), and [session-start hook](hooks/session_start.sh)
- [Job records](scripts/tao_job_record.py), [project guidance helper](scripts/tao_agent_identity.py), and [data I/O](skills/tao-data-io/SKILL.md)
- [Evaluation result](https://gitlab-master.nvidia.com/ai_tools/verified-plugins-playground/-/blob/3d6cc6efebfadd6dddb999762165bae90b987833/plugins/nvidia-tao-toolkit/.eval-results/e068d566e59f4351acaa7e0196ceb8cb/eval-result.json) — NVCARPS 1.3.1430
- [Evaluation HTML report](https://gitlab-master.nvidia.com/ai_tools/verified-plugins-playground/-/raw/3d6cc6efebfadd6dddb999762165bae90b987833/plugins/nvidia-tao-toolkit/.eval-results/e068d566e59f4351acaa7e0196ceb8cb/evaluation-report.html) — NVCARPS 1.3.1430
- [Canonical cross-agent aggregate: 94.71/100, PASS, Partial coverage](https://gitlab-master.nvidia.com/ai_tools/verified-plugins-playground/-/blob/3d6cc6efebfadd6dddb999762165bae90b987833/plugins/nvidia-tao-toolkit/.eval-results/e068d566e59f4351acaa7e0196ceb8cb/overall-score.json) — NVCARPS 1.3.1430; the reported policy takes the minimum agent score. Codex's individual score is 98.06/100.
- [TAO source and evaluation submission](https://gitlab-master.nvidia.com/ai_tools/verified-plugins-playground/-/merge_requests/7)

## Ethical Considerations

A package-specific, owner-approved ethics statement is not supplied. The repository provides an [issue tracker](https://github.com/NVIDIA-TAO/tao-skill-bank/issues) for quality concerns and points to [security reporting instructions](https://github.com/NVIDIA-TAO/tao-skill-bank/blob/main/SECURITY.md) for vulnerabilities.
