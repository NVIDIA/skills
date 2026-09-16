## Description: <br>
Host setup for TAO GPU backends that checks and, after user approval, installs minimum-compatible NVIDIA driver, CUDA Toolkit, and NVIDIA Container Toolkit versions for Docker/local-Docker and Kubernetes GPU worker hosts. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to check and install NVIDIA GPU runtime dependencies (driver, CUDA Toolkit, Container Toolkit, Docker) on Linux hosts before running TAO workflows on Docker or Kubernetes GPU backends. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [skill_info.yaml](references/skill_info.yaml) <br>
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) <br>
- [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) <br>
- [Docker Engine Install Guide](https://docs.docker.com/engine/install/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), evaluated in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal completion and expected workflow adherence, equally weighted). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage through quality routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `skill_efficiency`: Assesses routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Measures final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Evaluates whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 23% → 97% (+73 points) | 33% → 54% (+21 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 20% → 100% (+80 points) |
| Discoverability | 0% → 100% (+100 points) | 0% → 0% (±0 points) |
| Effectiveness | 17% → 83% (+67 points) | 43% → 70% (+27 points) |
| Efficiency | 0% → 100% (+100 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
