# PAIDF Visual Attribute Search

Use this skill to operate one of the four checked-in Visual Attribute Search workflows. Read
[run-reference.md](event-and-person-attribute-search/run-reference.md) before generating commands.

The `event-and-person-attribute-search-service` is PAS-only. It consumes an
explicit attribute JSON or sidecars from explicit upstream workflow nodes and
uses only the text LLM for query generation. It does not caption media, run
Visual QA, detect people, copy crops, or synthesize tracks.

## Workflow

1. **Select one contract.** Match the input and requested output to exactly one
   cookbook:
   - `pipeline_image_attributes_pas.yaml`: one attribute JSON, LLM only.
   - `pipeline_image_multiview_pas.yaml`: one identity across multiple images;
     joint captioning + Visual QA, then PAS.
   - `pipeline_video_pas_reasoning.yaml`: tracking, event verification and
     reasoning, per-track PAS, then training export.
   - `pipeline_video_epas.yaml`: tracking, captions, anomaly evidence,
     per-track attributes, PAS query buckets, and contextual output.
2. **Collect required values.** Confirm input, output, and the endpoints required
   by the selected contract. Video runs also need a model cache and SAM3 weights;
   multi-view needs a representative image and local image-group directory.
   Never invent paths, URLs, models, or credentials. For real individuals, also
   confirm the documented consent or other legal basis, permitted purpose,
   retention period, and required privacy or legal approvals before proceeding.
3. **Create a local config.** Copy the tracked config to `*.local.yaml` and make
   only deployment-specific edits. Keep its node IDs, dependencies, question
   banks, artifact namespaces, and PAS source lists intact.
4. **Dry-run.** Run the selected config with `--container-dry-run`. Validate the
   exact node order, images, endpoint arguments, input/output mounts, path-valued
   node arguments, and that every PAS input has an upstream producer.
   Fix and retry at most three times; if it remains invalid, stop and report the
   failing contract.
5. **Preflight.** Verify local inputs and mounts exist, model endpoints respond,
   and required images are available or can be built. Do not print secrets.
6. **Execution gate.** A dry-run is read-only. Before a real Docker run, present
   the exact command, GPU use, builds/pulls, mounts, endpoints, and output writes,
   then require explicit user approval. If approval is not given, stop.
7. **Validate artifacts.** After a successful run, validate the producer and PAS
   artifacts listed in the run reference. Do not validate a partial output after
   a failed container.

## Sensitive-data disclaimer for expected artifacts

> Visual Attribute Search expected artifacts, including `person_attributes.json`,
> `bundle_queries.json`, Visual QA results, crops, tracks, captions, and derived
> query files, may contain personal data, biometric-adjacent attributes, or other
> PII about identifiable individuals. Before running image or video person
> attribute search on real people, identify and document consent or another valid
> legal basis and confirm compliance with all applicable laws, which may include
> the GDPR, CCPA/CPRA, and Illinois BIPA. Obtain qualified privacy or legal review
> when requirements are uncertain.
>
> Limit input and output directory access to authorized personnel and service
> accounts with least-privilege permissions; never use public or broadly shared
> storage. Encrypt output artifacts at rest. Define and enforce a purpose-limited
> retention schedule before execution, including secure deletion of source media,
> crops, sidecars, query bundles, backups, and derived copies when the retention
> period expires.

## Contract facts

- Attribute-only PAS uses the same JSON as the single `DataEntry.media_path` and
  `--attribute-json`. It requires no VLM or image processing.
- Multi-view PAS keeps `media_path` as one representative file and passes the
  group directory to captioning and Visual QA with `--max-group-images 0`. It
  produces one consolidated identity scene.
- Both video flows repeat `visual_qa`. Preserve distinct node IDs, question
  banks, sidecar paths, and `--state-artifacts-key` values.
- Only the video configs require detection/tracking and the SAM3 weights mount.
- The VLM serves captioning and Visual QA. The LLM serves reasoning when present
  and PAS query generation. PAS must not receive VLM arguments.
- Reasoning/thinking models may need a larger `max_tokens` value in the Visual QA
  and reasoning nodes; non-reasoning cookbook defaults should remain unchanged.

## Guardrails

- Always inspect a dry-run before execution.
- Require explicit approval before Docker execution or image builds.
- Forward credential names such as `NVIDIA_API_KEY`, never secret values.
- Do not collapse explicit producer nodes into PAS configuration and do not add
  `image_pas`, composite producer, or overwrite/reuse settings to the service.
  The absence of an `image_pas` block does not remove the privacy obligations for
  image-derived attributes or other person data in the pipeline.
- Do not change one Visual QA pass to write another pass's namespace.
