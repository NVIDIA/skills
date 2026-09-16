## Description: <br>
Configure and run NVIDIA Cosmos Curator video and image curation pipelines (split, filter, caption, embed, dedup, shard, image annotate) and PAIDF Data Mining nearest-neighbor matching to turn raw collections into training-ready datasets for physical AI. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to configure and run NVIDIA Cosmos Curator video/image curation pipelines and PAIDF Data Mining nearest-neighbor retrieval, producing curated, training-ready datasets for physical AI applications. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Calibration Config](references/calibration-config.md) <br>
- [Capabilities](references/capabilities.md) <br>
- [Configuration Decision Tree](references/configuration-decision-tree.md) <br>
- [Context Understanding](references/context-understanding.md) <br>
- [Cosmos Curator](references/cosmos-curator.md) <br>
- [Curation Retrieval Workflow](references/curation-retrieval-workflow.md) <br>
- [Data Mining](references/data-mining.md) <br>
- [Distribution Analysis](references/distribution-analysis.md) <br>
- [Distribution-Aware Curation](references/distribution-aware-curation.md) <br>
- [FFmpeg Sidecar](references/ffmpeg-sidecar.md) <br>
- [Gotchas](references/gotchas.md) <br>
- [Image Curation](references/image-curation.md) <br>
- [KPI Metrics](references/kpi-metrics.md) <br>
- [Restrictive Curation](references/restrictive-curation.md) <br>
- [Running Pipelines](references/running-pipelines.md) <br>
- [SAM3 Config](references/sam3-config.md) <br>
- [Video Curation](references/video-curation.md) <br>
- [Video Lake Curation](references/video-lake-curation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (13 positive) across 2 agents, with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was selected, decoys were avoided, and the expected workflow executed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Checks tool-call productivity (50%) and token efficiency (50%), avoiding wasted skill and tool usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.7% | 87.8% |
| Security | 100.0% → 100.0% (±0.0 points) | 90.9% → 100.0% (+9.1 points) |
| Correctness | 33.9% → 100.0% (+66.1 points) | 40.9% → 93.9% (+53.0 points) |
| Discoverability | 83.9% | 68.5% |
| Effectiveness | 36.0% → 93.2% (+57.2 points) | 34.0% → 81.4% (+47.4 points) |
| Efficiency | 86.5% | 95.3% |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
