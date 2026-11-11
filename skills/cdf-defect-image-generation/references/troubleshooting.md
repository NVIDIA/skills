# DIG troubleshooting

Start with the failed task instance, its map index and sanitized error. Diagnose
one existing run before deciding whether any retry is appropriate.

| Symptom | Check and response |
|---|---|
| DAG absent | Check manifest presence/override path and DAG import errors; the registered ID is `defect_image_generation_smt_dag_k8s`. Do not invent a mode-specific ID. |
| Payload rejected | Check API version, required collections for the mode, unique safe IDs, and top-level/nested service-setting agreement. |
| Full payload passes model but `generate_configs` fails | A full-mode `anomaly_input.class_by_component_type` mapping is validated during compilation; use the bundled checker before submission. |
| `image_generation` fails before rendering | The Simulation JobConfig rejects unknown or invalid fields. Compare the override with the CDF Simulation JobConfig reference, and check that board-specific template settings match the scene. |
| `anomaly_package` reports missing component type/mapping | Inspect the simulation/augmentation manifest and map every type in the batch, even for a defect variant targeting only one class. |
| Simulation output missing | Inspect the selected pass's `dataset_index.json`, actual files, and any `simulation_output.image` path, which starts with the pass name. A scene change can invalidate Artix-specific selectors. |
| Checkpoint download fails | Check HF access, the pinned checkpoint revision, PVC mount/writability, and disk space. Do not change model revisions silently. |
| Augmentation connection/model error | Check the effective image-edit endpoint and served model from compute, independent of AMG VLM configuration. External endpoints receive the deployment's NGC API key; one that needs a different credential requires a manifest change. |
| AnomalyGen rejects its configuration | `run_pipeline` rejects unknown keys and inconsistent combinations, such as `per_defect_counts` outside `mode: inference`. Compare the `anomaly_generation` override with AnomalyGen's `run_pipeline` config reference. |
| AnomalyGen reports missing assets | Inspect effective `staging.fetch`; list replacement may have removed required shared assets or prepared clean inputs. Override per run instead of changing global defaults. |
| AnomalyGen VLM auth or model error | Check that `amg_gen_model` and `amg_judge_model` each match an `amg_endpoints` model, that `base_url` includes its version path, and that the task credential is present. The key is written to `/work/endpoints.yaml` inside the task; do not print that file. |
| `validate_anomalygen_outputs`: export refers to an unexpected input | The staged inputs contained clean images other than the packaged goldens, for example inside the shared asset bank, such as the public PCB dataset's `clean_image/` folders. In full mode, stage reference assets only. |
| AnomalyGen GPU pod pending | Default AMG requires two GPUs on one node; also check two available Airflow pool slots, PVC placement and image pulls. External image-edit does not eliminate this requirement. |
| Internal service waiting | Check replica count, GPU availability, slot reservations, model-cache PVC and readiness. Lower replicas only within the user's workload intent. |
| Image-edit replicas restart with `Orchestrator did not become ready within <N>s` | Follow [model startup timeouts](../../cdf-orchestration-setup/references/model-startup-timeouts.md). The flags belong in `image_edit_service` `container_args` of the SMT K8s manifest; the checked-in manifest sets neither, so the 600-second default applies. Check each restarting replica's log rather than assuming one cause. Propose a timeout increase to the user. |
| Missing/invalid export manifest | Inspect raw exports and input mappings. Do not hand-author a manifest to bypass validation. |
| Fewer selected defects than requested | Compare raw exports to selected artifacts and input reuse. Full-mode selection is at most one export per clean input; requested counts are not coverage guarantees. |
| Later stages skipped | Confirm pipeline mode; simulation and sim2real deliberately skip the defect tail. |

Leave partial artifacts available for diagnosis. Before clearing tasks, changing
configuration, or rerunning, explain the identified cause and likely compute cost;
continue only within existing authorization. Never delete all namespace pods or
other runs' services. Query a known submission ID after a timeout before retrying.
