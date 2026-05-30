# VSS Skills

Skills for working with NVIDIA Video Search & Summarization (VSS). Each subdirectory under `skills/` is a self-contained skill following the [agentskills.io](https://agentskills.io/specification) specification, with `name`, `description`, `version`, and `license` declared in its `SKILL.md` frontmatter.

## Renamed in GA

The VSS 3.2 GA skill names replaced the pre-GA slash-command names:

| Pre-GA command | VSS 3.2 GA command |
|---|---|
| `/alerts` | `/vss-manage-alerts` |
| `/deploy` | `/vss-deploy-profile` |
| `/report` | `/vss-generate-video-report` |
| `/rt-vlm` | `/vss-deploy-dense-captioning` |
| `/video-analytics` | `/vss-query-analytics` |
| `/video-search` | `/vss-search-archive` |
| `/video-summarization` | `/vss-summarize-video` |
| `/video-understanding` | `/vss-ask-video` |
| `/vios` | `/vss-manage-video-io-storage` |
| `/vss-frag` | `/vss-generate-video-report-rag` |

## Catalog

| Skill | Description |
|---|---|
| [vss-manage-alerts](vss-manage-alerts/SKILL.md) | Skill to add, manage, and monitor alerts on streamed video. |
| [vss-deploy-profile](vss-deploy-profile/SKILL.md) | Skills to deploy, debug, or tear down any VSS profile using a docker compose-centric workflow. |
| [vss-deploy-dense-captioning](vss-deploy-dense-captioning/SKILL.md) | Skill for deploying and calling the RT-VLM dense captioning microservice API. |
| [vss-deploy-video-embedding](vss-deploy-video-embedding/SKILL.md) | Skill for deploying and operating the RT-Embed video embedding microservice API. |
| [vss-deploy-detection-tracking-2d](vss-deploy-detection-tracking-2d/SKILL.md) | Skill for deploying, operating, and calling the RTVI-CV perception microservice for 2D detection and tracking (warehouse 2d/3d, sparse4d, smartcity rtdetr/gdino). |
| [vss-generate-video-report](vss-generate-video-report/SKILL.md) | Skill to produce video analysis reports by querying the VSS agent's `/generate` endpoint. |
| [vss-generate-video-report-rag](vss-generate-video-report-rag/SKILL.md) | Skill to generate video summary reports with Enterprise RAG using the VSS frag/RAG pipeline. |
| [vss-query-analytics](vss-query-analytics/SKILL.md) | Skill for querying video analytics data and metrics from Elasticsearch via the VA-MCP server. |
| [vss-search-archive](vss-search-archive/SKILL.md) | Skills for searching video archives using natural language, multi-embedding fusion, and VLM critique. |
| [vss-summarize-video](vss-summarize-video/SKILL.md) | Skill for summarizing a video through chunking, dense captioning, and aggregation functions using the Long Video Summarization (LVS) microservice. |
| [vss-ask-video](vss-ask-video/SKILL.md) | Skill for using video understanding tool to answer text questions about video content using a VLM. |
| [vss-manage-video-io-storage](vss-manage-video-io-storage/SKILL.md) | Skill for video and stream management, recording timelines, clip extraction, snapshots (and more) using the Video IO and Storage microservices. |
| [vss-generate-video-calibration](vss-generate-video-calibration/SKILL.md) | Skill to run AutoMagicCalib (AMC) camera calibration on local MP4s, RTSP streams, or the bundled sample dataset, and to deploy the vss-auto-calibration microservice when needed. |
| [vss-deploy-detection-tracking-3d](vss-deploy-detection-tracking-3d/SKILL.md) | Skill to deploy and operate the RTVI-CV-3D stack (a.k.a. MV3DT / Multi-View 3D Tracking / RTVI-CV-MV3DT) on sample, custom video, or RTSP inputs — auto-chains to vss-generate-video-calibration when calibration is missing. |


Skills with `evals/*.json` specs are exercised automatically by the Skills Eval CI workflow on every PR that touches `skills/**`; legacy `eval/*.json` specs are still accepted for skills that have not moved yet. See [`.github/skill-eval/AGENTS.md`](../.github/skill-eval/AGENTS.md) for harness behavior.

## Install (recommended: ask your coding agent)

Open this repository in your coding agent (Claude Code, Codex, Cursor, or any other agentskills.io-compatible host) and paste the following prompt:

> Read `skills/README.md` and every `SKILL.md` file under `skills/`. For each skill in the catalog, install it for this host so I can invoke it from a shell or chat session. Use the host's standard skills directory:
>
> - Claude Code: `~/.claude/skills/<name>/`
> - Codex: `~/.codex/skills/<name>/`
> - Hosts that follow the agentskills.io universal path: `~/.agents/skills/<name>/`
>
> Symlink each skill folder rather than copying it so a `git pull` here keeps every install up to date. Skip skills that are already installed and pointing at this checkout. When you're done, list the skills you registered and which directory you used.

The agent will read the frontmatter of each `SKILL.md`, create the symlinks, and confirm what's installed. The skills become invokable in the next agent session.

### Single-skill install

To install skills individually, paste the following prompt:

> Install only `skills/<name>/` for this host the same way.

### Update

After `git pull`, the symlinks already point at the updated content — nothing to do unless skills were added or renamed. To pick up new skills use the following prompt:

> Re-read `skills/README.md` and add any new skills missing from this host's skills directory.

### Uninstall

To uninstall skills, paste the following prompt:

> Remove every VSS skill symlink you previously created under this host's skills directory.

## Source of truth

This `skills/` directory is the canonical source. Skills published to the public catalog at `github.com/nvidia/skills` are mirrored from here at sync time.
