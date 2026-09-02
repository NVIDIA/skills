# Video PAS reasoning run reference

Dry-run the shipped contract:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml --container-dry-run'
```

Expected node order:

```text
detection_and_tracking
-> captioning
-> event_verification_visual_qa
-> reasoning
-> person_attribute_visual_qa
-> person_attribute_search
-> training_export
```

The event VQA pass writes `visual_qa_event_verification/` evidence and DAFT
`mcq.json`, `bcq.json`, and `open_qa.json` with reasoning traces. The per-track
pass reads detection crops and writes
`visual_qa_per_track/{items.json,windows.normalized.json}` without flat QA files.
PAS consumes the tracks, captions, and per-track VQA sidecars and writes
`sidecars/person_attribute_search/{pas.json,chunk_queries.json}` plus the
configured contextual mirror. Training export writes the configured
TAO-VL-Reason dataset.

Required mounts include media/output, model cache, read-only SAM3 weights at
`/models/sam3`, both question banks, and the cookbook config. VLM settings go to
captioning and both Visual QA passes; LLM settings go to event verification,
reasoning, and PAS. PAS must not receive VLM settings.

After a passing preflight and explicit approval, remove
`--container-dry-run` to execute.
