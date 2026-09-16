# Air-Gapped Cosmos3 DEFT AOI

Enable air-gap mode when `AIR_GAPPED=1` is present, the user explicitly asks
for offline/air-gapped execution, or the harness reports restricted
networking. Resolve this before dependency checks; never probe the network to
infer the mode.

Air-gap mode is valid only when every selected platform input is already
visible from the compute frame:

- Cosmos-RL and data-services images;
- Cosmos3 base model / tokenizer cache;
- Proxy, Benchmark, Mining JSON and all referenced images;
- the AnomalyGen image, its fine-tuned checkpoint (`ag_config.yaml` plus the
  iteration checkpoint), its dataset directory (`defect_spec.jsonl`,
  `semantic_segmentation_labels.json`, clean images, cad masks), and the Cosmos
  base-checkpoints cache — required only when the AnomalyGen stage will run;
- selected platform native CLI and GPU runtime;
- host Python with `pyarrow` and `yaml`.

In air-gap mode:

- initialize state with `--network-mode airgap` and its activation source;
  after initialization run local external commands through
  `"$PYTHON" scripts/deft_exec.py`, which injects offline variables and enforces
  `--pull=never` for direct Docker/Podman runs;
- do not run image pulls, package installs, Hugging Face downloads, S3 staging,
  or credential login. This explicitly prohibits `pip`, `pip3`, `uv`, `conda`,
  `apt`, and package-manager commands from an existing virtual environment,
  even as a probe or retry. This also includes the AnomalyGen post-gate
  bootstrap, whose checkpoint/dataset/base-cache fetchers must all be
  pre-staged instead;
- use `bash scripts/deft_python.sh` to select an already-provisioned interpreter; if
  no candidate provides `pyarrow` and `yaml`, report those imports and stop;
- keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set for AnomalyGen runs;
- leave both `HF_TOKEN` and its legacy alias `HUGGING_FACE_HUB_TOKEN` unset
  when local assets are sufficient; clearing only one still leaves a usable
  token in the environment for `huggingface_hub` to pick up;
- use storage tier A and verify every mount/path before the launch review;
- stop on a missing asset instead of substituting a model, image, evaluator, or
  reduced workflow.

The same user gate, job-record ordering, four verbs, state contract, frozen
Benchmark hash, and bare annotation contract still apply.
Never read `references/network-bootstrap.md` in this mode.

When AnomalyGen will run, its base cache and Guardrail safety model must be fully pre-staged, and the base cache must be verified offline with the container's own check before SDG.
