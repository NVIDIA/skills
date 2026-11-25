# NVIDIA FLARE plugin

`nvidia-nvflare` packages NVIDIA FLARE's existing agent skills for Agent Plugins,
Codex, Claude Code, and Cursor. The proposed initial plugin version is `1.0.0`;
it is independent of the NVFlare Python package and individual skill versions.
This packaging candidate is not an approved plugin release.

Verification CI generates the final Plugin Card and benchmark for the evaluated
bundle. Preparation checks below do not establish NVIDIA Verified status.

## Contents

| Directory under `skills/` | Workflow |
| --- | --- |
| `nvflare-orient` | Route open-ended NVIDIA FLARE questions |
| `nvflare-convert-pytorch` | Convert plain PyTorch training to federated jobs |
| `nvflare-convert-lightning` | Convert Lightning training to federated jobs |
| `nvflare-convert-huggingface` | Convert Hugging Face Trainer and TRL SFTTrainer workflows |
| `nvflare-diagnose-job` | Diagnose failed or stalled jobs |
| `nvflare-fed-stats` | Compute federated statistics |
| `nvflare-autofl` | Optimize an existing federated job |
| `nvflare-autofl-report` | Report a stopped or interrupted Auto-FL campaign |
| `nvflare-shared` | Required references and templates used by the other skills |

Install the complete bundle to preserve sibling references. `nvflare-shared` is
a support component, not a separate user workflow; some hosts may nevertheless
list it because they do not interpret NVIDIA's internal-status metadata.

There is no MCP server, hook, or automatic installer. The bundle contains the
complete skill directories, including evaluation fixtures and retained skill
reports and signatures. Evaluation fixtures are grading data, not runtime
instructions. Skill names, instructions, scripts, references, and evaluation
data are preserved by packaging.

The workflows require NVFlare 2.9.0 or later and the Python/framework dependencies
specified by the selected workflow. Installing the plugin does not install
NVFlare, download models, provision GPU resources, or grant permission to submit
jobs. Conversion and Auto-FL workflows can edit files or run jobs; use a scratch
workspace and local simulation for initial testing.

## Manifests and source of truth

The distributable plugin is committed under `plugins/nvidia-nvflare/`. The
existing repository `skills/` directories remain the source of truth; edit skills
there and regenerate the bundle. The root manifests, plugin-level `evals/evals.json` and `regression-evals.json`,
`evals/config.json`, plugin evaluation fixtures, and this guide are also packaging inputs. The paths below
are relative to the generated plugin directory.

| Target | Entry point | Skill discovery |
| --- | --- | --- |
| Agent Plugins 1.0.0 | `plugin.json` | Root `skills/` directory |
| Codex | `plugin.json` plus `.codex-plugin/plugin.json` | Portable discovery; compatibility manifest also declares `./skills/` |
| Claude Code | `.claude-plugin/plugin.json` | Default root `skills/` directory |
| Cursor | `plugin.json` or `.cursor-plugin/plugin.json` | Portable discovery or the native `./skills/` path |

Shared identity fields must agree in all four manifests. The Codex compatibility
manifest supplies the display label **NVIDIA FLARE**. There is no inline
`extensions.com.openai` override in the portable manifest.
It also declares `requires_local_executor: true`, because these workflows use
local Python programs and workspaces. The OpenAI marketplace entry is intended
for desktop execution.

The build copies real files and preserves executable permissions. It packages
only tracked skill files, the four manifests, this guide as `README.md`,
the plugin-level `evals/evals.json` and `regression-evals.json`, `evals/config.json`, evaluation fixtures, `LICENSE`, `LICENSE-CC-BY-4.0`,
`SKILLS-LICENSE.md`, and `CONTRIBUTING.md` with the DCO text.
The maintainer-authored root `plugin-card.md` is preparation material, excluded
from the bundle; verification CI supplies the final card. Repository development
settings and unrelated source code are excluded.
No component symlinks are used.

## Licensing

The plugin and mixed skill files declare `Apache-2.0 AND CC-BY-4.0`.
Code, scripts, configuration, and code examples use Apache 2.0. Skill instructions,
reference prose, non-code examples, and NVIDIA-authored evaluation content and
example datasets use CC BY 4.0. This scope includes future skills and their
examples; see [the scope and attribution policy](SKILLS-LICENSE.md).
Both full license texts are included. Other NVFlare code and documentation retain
their existing licensing, and third-party terms and attributions remain in force.

The whole-plugin signature is `plugin.oms.sig` at the plugin root. Member
signatures are stored as `skills/<name>/skill.oms.sig`. Verify each signature's
certificate chain and covered file inventory against the package contents.

## Build and validate

Run these commands from a checkout containing this packaging change, with Python
3.10 or later. The build itself uses only the standard library and Git. Choose a
new output directory for each archive. First regenerate and check the directory
that will be committed, evaluated, signed, and published:

```bash
python3 dev_tools/agent/package_plugin.py --sync
python3 dev_tools/agent/package_plugin.py --check
python3 dev_tools/agent/package_plugin.py --output /tmp/nvidia-nvflare-candidate
```

Commit the generated `plugins/nvidia-nvflare/` directory along with its source
changes. CI rejects missing, extra, or changed input files, executable-mode drift,
and symlinks. Only the root CLI export files and known verification outputs are
permitted in addition to generated inputs. These outputs are preserved unchanged
when archiving; the drift check does not validate their scores or signatures.

When inputs change, `--sync` refuses to carry forward existing evaluation or
signing outputs. Retain the old evidence outside the plugin, remove those outputs,
rebuild, commit, and run fresh evaluations and verification. Review and remove
obsolete bundle files explicitly; rebuilding does not silently delete them.
Use an external evaluation output directory so private raw traces are not mixed
into the distributable.

The output contains `nvidia-nvflare/`, `nvidia-nvflare-1.0.0.zip`, `SHA256SUMS`, and
`build-metadata.json`. The metadata records the source commit, skill tree, every
packaged file's SHA-256 digest, and the archive digest. ZIP entries use fixed
timestamps so identical inputs produce identical archives with the same toolchain.
No private keys, credentials, or local logs are required by the build.
During preparation, `packaging_matches_source_commit` is false if the manifests,
guide, evaluation suite, license, contribution guide, or builder have not been committed. Commit the reviewed packaging and
rebuild before using the source commit as provenance for a release.

The builder requires all nine skills and checks every bundled skill file against
the resource lists in the retained `skill.oms.sig` files. A mismatch, missing
resource, unrecorded skill file, symlink, or changed skill tree fails the build.
These digest comparisons establish content preservation; they do not verify the
signing certificate's trust, constitute a new evaluation, or sign the plugin.

The `Plugin Package` CI workflow runs on source and bundled skill changes. It
checks the committed directory against its sources, validates discovery settings,
builds twice to check reproducibility, and checks archive contents, permissions,
retained resource hashes, bundled documentation links, and skill lint. It retains
the candidate and check results as workflow artifacts. Run the same check locally
in an isolated Python 3.12 environment:

```bash
python3.12 -m venv /tmp/nvidia-nvflare-validation
/tmp/nvidia-nvflare-validation/bin/python -m pip install \
  'PyYAML==6.0.3' 'jsonschema==4.25.1'
curl --fail --location --silent --show-error --max-time 30 \
  https://agent-plugins.org/schemas/1.0.0/plugin.schema.json \
  --output /tmp/nvidia-nvflare-plugin.schema.json
/tmp/nvidia-nvflare-validation/bin/python \
  -m dev_tools.agent.plugin_validation.ci_check \
  --schema /tmp/nvidia-nvflare-plugin.schema.json \
  --output /tmp/nvidia-nvflare-package-checks
```

The checker verifies the downloaded schema's pinned SHA-256 digest before use.
Use a new check output directory outside the checkout. The builder and checker
are development tools excluded from the distributed plugin; they require no
provider credentials and do not run agents, sign, or publish content. A packaging
pass does not establish security-scan completeness or workflow quality.

## Host and evaluation checks

Load an extracted copy outside the repository through each host's documented
plugin mechanism, without duplicate installed skills. Confirm the display label,
all nine components, routing, and shared references. Record the host/model
versions, archive digest, and outcomes. See the host references below and the
retained check evidence for probes already completed.

### Evaluation

The default evaluation suite contains five plugin cases in `evals/evals.json`. These
cover a PyTorch FedAvg conversion, Lightning checkpoint evaluation, federated DICOM
statistics, a request combining Hugging Face conversion and federated statistics,
and an unrelated website request for which no NVFlare skill should activate.
The cases retain the prompts, routing, grading criteria, and fixture bytes used
in the broader regression suite. The two framework cases also stage their four
member project files under `evals/skills/` to preserve their workspace paths.

The plugin root's `regression-evals.json` retains the full 107-case member
projection for explicitly selected regression runs. CI checks all projected
cases, the five default cases, and their staged fixtures against the member
sources and the three plugin-specific revisions below. Both evaluation variants
receive the same fixtures and project-edit permissions.

The plugin runs `pytorch-convert-basic-v2` in the original conversion slot.
The signed `pytorch-convert-basic` asks for equal partitions of eight rows across
three sites. Asking which constraint to relax is appropriate for that prompt.
Its source has no validation loader. A successful simulation on 300 replacement
rows cannot establish conversion of the requested data.

The v2 case uses the signed member's `eval-pt` project, which draws eight
training rows and four separate validation rows and defines accuracy. The prompt fixes the data
seed, authorizes a 3/3/2 training split and requires each site to evaluate on all four
validation rows in every round. The four rows are a smoke check of metric plumbing. Preserve
historical results under their original IDs. A new comparison needs fresh skill
and baseline attempts on v2 with the same fixture, runtime and settings, plus
retained partition, validation and completion evidence for each attempt.

The plugin runs `lightning-eval-only-v2` in the original Lightning evaluation-only
slot. Both arms receive the member's Lightning project and the same synthetic CPU
state dictionary for its `LitNet` model at `evals/files/lightning-eval-only/litnet-eval.pt`.
The fixture was generated on CPU with PyTorch 2.15.0, `torch.manual_seed(17)`,
and `torch.save(dict(LitNet().state_dict()), "litnet-eval.pt")`. To verify compatibility,
load with `torch.load(path, map_location="cpu", weights_only=True)` and pass the
result to `LitNet().load_state_dict(..., strict=True)`. The case retains the member's
grading criteria for safe loading, one-shot evaluation, and per-site validation metrics.

In `fedstats-image-dicom-dependency-v2`, each of the three hospitals has one
synthetic chest X-ray DICOM file under `evals/files/fedstats-image-dicom-dependency/`.
Both arms receive the same workspace-relative inputs and the member's original
loader and histogram grading criteria.

The compute image needs NVFlare and the frameworks exercised by the tasks.
Select the same compatible image for skill and baseline attempts.

The bundled `evals/config.json` declares Git/curl, read-only access to the
sandbox's own `/proc` for NVFlare parent-process checks, and three Hugging Face
Xet origins in addition to the managed destinations. These grants apply to
both skill and baseline attempts; they do not mount the host's `/proc`.
Application dependencies belong in a compatible compute image selected by the
operator, not in the task prompts.

The default suite schedules 20 attempts for five cases, two agents, and both
plugin and baseline arms.
Inspect the plan, results, scores, security findings, and any capture or grading
errors for the selected suite. A structurally valid result does not establish a pass.
Never commit credentials or private logs.
Rebuild, re-evaluate, and re-sign after changing bundled inputs.

Loading probes do not establish conversions or full workflow coverage. Retain
complete reports and triage findings; an incomplete scan remains a release
blocker. Do not edit signed resources or suppress findings to obtain a pass.

## Release evidence checklist

- [ ] Review the exact package inventory, license notices, dependencies, and model
  or dataset references; retain third-party attribution where applicable.
- [ ] Obtain closed OSRB approval for the plugin's marketplace-distribution
  scope, linking the prior repository approval. Follow the program's current
  launch process and applicable IP review, attribution, and VP requirements.
- [ ] Complete native loading and workflow smoke tests for the supported hosts.
- [ ] Add an approved square marketplace icon and confirm each marketplace's
  presentation requirements and final installation entry. The current candidate
  has no icon or published marketplace installation command.
- [ ] Scan and evaluate the final plugin through the supported NVIDIA pipeline;
  retain tool versions, source revision, report links, complete configured
  agent/arm coverage, execution errors, canonical case IDs, and gate verdicts.
- [ ] Resolve or obtain a documented reviewer disposition for the Auto-FL report
  task-polarity concern tracked in [NVFlare #5294](https://github.com/NVIDIA/NVFlare/pull/5294).
  Its four canonical cases represent one positive and three target-relative
  negatives. A report labeling all four positive does not validate that distinction.
  The 107-case regression suite declares these expectations explicitly. Use the
  regression suite to check the Auto-FL report case routing.
  Verify the final benchmark and card together.
- [ ] Generate a Plugin Card and plugin-level signature for the exact final
  package. Existing per-skill signatures do not attest to the new manifests or
  the bundle as a whole. Verify the final published package against its attestation.
- [ ] Review and merge the packaging into the public repository, select the release
  revision/version, then add a `plugins:` entry in NVIDIA/skills
  `components.d/nvflare.yml` identifying the repository, `2.9` branch, plugin
  directory, and `catalog_dir` install name. Confirm resulting marketplace content
  and install behavior before announcing availability.

Preparation and local validation can proceed while approval is pending. Keep
unverified items open and avoid describing the candidate as NVIDIA Verified until
the required review, evaluation, signing, and publication steps are complete.

## Packaging references

Specifications consulted for this candidate on November 12, 2026:

- [Agent Plugins 1.0.0](https://agent-plugins.org/) and its
  [manifest schema](https://agent-plugins.org/schemas/1.0.0/plugin.schema.json)
- [Agent Skills specification](https://agentskills.io/specification)
- [Codex plugin packaging](https://developers.openai.com/plugins/build/plugins)
- [Claude Code plugin manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference)
- [Cursor plugin reference](https://cursor.com/docs/reference/plugins)
- [NVIDIA/skills plugin catalog conventions](https://github.com/NVIDIA/skills/blob/main/plugins.d/README.md)
- [NVIDIA FLARE catalog inventory](https://github.com/NVIDIA/skills/blob/main/components.d/nvflare.yml)

The NVIDIA plugin program's approval and marketplace requirements are separate
from the host schemas. Recheck submission guidance before publication.
