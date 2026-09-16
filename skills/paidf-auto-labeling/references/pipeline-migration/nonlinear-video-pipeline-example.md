# Nonlinear Video Pipeline Example

Use this reference as an example pattern for source repos shaped like:

```text
raw videos
  -> tracking/crops
  -> video chunking
  -> chunk VLM annotation
  -> anomaly votes
  -> retrieval query generation
  -> visualizer
```

## Source Pipeline Concepts

Typical source steps map to capabilities like this:

| Source concept | UPA target | Migration decision |
|---|---|---|
| SAM tracking and crop export | `detection_and_tracking` | extend existing service if crop/PAS handoff is missing |
| 5-second video chunks | `media_chunking` | new generic service if not already available |
| Scene and dense captions over chunks | `captioning` | extend existing service with chunk input mode |
| PAS from track crops | `visual_qa` or generic extraction stage | choose based on whether it is question-answering or structured extraction |
| Multi-model anomaly vote | `anomaly_vote` or generic classification | new generic service if reusable |
| Retrieval query generation | `query_generation` | new generic service |
| HTML visualizer | optional report/export service | keep out of core annotation path unless productized |

## Cookbook Shape

Prefer dependency-shaped YAML over a new pipeline-branded service:

```yaml
pipeline: video

workflow:
  nodes:
    tracking:
      stage: detection_and_tracking
    chunking:
      stage: media_chunking
    captions:
      stage: captioning
      needs: [chunking]
    anomaly:
      stage: anomaly_vote
      needs: [chunking]
    queries:
      stage: query_generation
      needs: [tracking, captions, anomaly]
```

If the current runner cannot execute these stage names yet, call that out as a
runner registry gap instead of inventing source-repo-specific stage names.

## Sidecar Sketch

One possible sidecar layout:

```text
sidecars/detection_and_tracking/
  tracks.json
  crops/
  crop_manifest.json

sidecars/media_chunking/
  chunks.json
  chunks/
    chunk_000.mp4

sidecars/captioning/
  chunk_windows.json
  dense_captions.json

sidecars/anomaly_vote/
  votes.json

sidecars/query_generation/
  retrieval_queries.json
```

Downstream stages should discover these through `pipeline_state.json` or stable
stage output conventions, not hard-coded source-repo folder names.

## What Not To Do

- Do not port the whole source repo as one opaque container if individual
  primitives can become reusable UPA services.
- Do not split every source script into a pipeline-branded UPA service.
- Do not make workflow-runner responsible for the script's internal thread
  pools. Stage-local concurrency can remain inside the service.
- Do not require workflow-runner parallel scheduling for the first migration;
  use `workflow.nodes` for dependency intent and let platform schedulers own
  distributed execution when needed.
