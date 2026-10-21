# GPU Footprint Derivation

Use this when filling in the two `GPU footprint` lines of the Step 7 summary in `SKILL.md`. Compute
both lines from the manifest itself — never state or imply external mode is 0 without checking.
There are two independent sources, and only one of them depends on the service mode:

1. **Task pods that do their own local model inference, in either mode.** For every entry in
   `components.tasks`, resolve its `deployment_profile` against `deployment.profiles`: a
   `k8s_gpu_task`-profiled (or otherwise `gpu:`-bearing) task claims 1 GPU per running pod
   *regardless of `external_services`*, because it's doing in-pod inference, not calling an
   endpoint — a `k8s_cpu_task`/`augmentation_task`-profiled one claims none. This is exactly why
   IAA's external mode is genuinely 0 GPU (its tasks — `augmentation`, `cosmos_post_processing`,
   `event_and_person_attribute_search` — are all CPU-profiled) while EVG's external mode is not
   (`detection_and_tracking`/`captioning`/`visual_qa` all run on `k8s_gpu_task`, 3 GPUs minimum
   before any endpoint is even considered).
2. **Endpoints deployed internally**, i.e. `external_services: false` for that service. Per
   replica: VLM / LLM / image-edit / Cosmos Transfer or Predict = 1 GPU; image2video = **2** GPUs
   (`gpu_count: 2`, `host_ipc: true`). `external_services: true` contributes 0 here.

Internal-mode total = source 1 + source 2 — the task-pod cost doesn't disappear in internal mode,
it's additive. Report both totals in the summary; don't make the user re-derive them from the
manifest, and don't reuse a number from a different pipeline's shape (IAA's "external = 0" does not
generalize to a pipeline that includes any `k8s_gpu_task`-profiled stage).
