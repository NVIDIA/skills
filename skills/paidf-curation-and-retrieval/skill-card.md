## Description: <br>
Use when operating PAIDF Curation and Retrieval or NVIDIA Cosmos Curator pipelines (split, filter, caption, embed, dedup, shard, image annotate) or PAIDF Data Mining nearest-neighbor matching on Curator embeddings. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and data engineers who operate GPU-accelerated video and image curation pipelines with NVIDIA Cosmos Curator and Data Mining to build training-ready datasets for physical AI. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Cloud Credentials, API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Capabilities and Key Matrix](references/capabilities.md) <br>
- [Calibration Config](references/calibration-config.md) <br>
- [Configuration Decision Tree](references/configuration-decision-tree.md) <br>
- [Context Understanding](references/context-understanding.md) <br>
- [Cosmos Curator](references/cosmos-curator.md) <br>
- [Curation-Retrieval Workflow](references/curation-retrieval-workflow.md) <br>
- [Data Mining](references/data-mining.md) <br>
- [Distribution Analysis](references/distribution-analysis.md) <br>
- [Distribution-Aware Curation](references/distribution-aware-curation.md) <br>
- [FFmpeg Sidecar](references/ffmpeg-sidecar.md) <br>
- [Gotchas](references/gotchas.md) <br>
- [Image Curation](references/image-curation.md) <br>
- [KPI Metrics](references/kpi-metrics.md) <br>
- [Restrictive Curation](references/restrictive-curation.md) <br>
- [Running Pipelines](references/running-pipelines.md) <br>
- [SAM3 Configuration](references/sam3-config.md) <br>
- [Video Curation](references/video-curation.md) <br>
- [Video-Lake Curation](references/video-lake-curation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML configuration] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (13 positive) across five dimensions in isolated sandbox pods, evaluated with and without the skill to measure uplift. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal completion and expected workflow adherence). <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 54% → 91% (+37 points) | 48% → 88% (+40 points) |
| Security | 96% → 100% (+4 points) | 85% → 96% (+12 points) |
| Correctness | 49% → 100% (+51 points) | 52% → 95% (+43 points) |
| Discoverability | 45% → 88% (+43 points) | 33% → 77% (+44 points) |
| Effectiveness | 49% → 88% (+39 points) | 44% → 86% (+42 points) |
| Efficiency | 32% → 78% (+46 points) | 27% → 86% (+58 points) |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
