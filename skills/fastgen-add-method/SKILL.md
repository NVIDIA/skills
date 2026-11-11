---
name: fastgen-add-method
description: >-
  Implement and validate a new NVIDIA FastGen training method or objective (distillation, consistency, adversarial or fine-tuning variants) from a paper or reference code: FastGenModel subclass, method config, gradients, student sampling and tests. Use it whenever an algorithm is not expressible through the existing method configs.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Add a FastGen method

Apply [fastgen-develop](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-develop/SKILL.md)
and [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md). Read the
[method index](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/README.md) and the nearest maintained implementation,
the `FastGenModel` docstrings in [model.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/model.py), then the
[method contract](references/method-contract.md).

1. **Plan.** Resolve the algorithm version and its authoritative paper or code. Write down
   the state and time convention, target construction, detach boundaries, loss weighting
   and reduction, update order, required teacher or auxiliary networks and the inference
   rollout. Separate the method from the reference's architecture, dataset and
   engineering defaults, and record unresolved paper/code discrepancies.
2. **Reuse.** If existing options express the algorithm, hand off to
   [fastgen-configure](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md).
   Otherwise choose the narrowest compatible method extension. Share mechanisms only
   where semantics match; do not inherit losses or networks because they are defaults.
   Move objective math into
   [common_loss.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/common_loss.py) only when another caller shares it.
3. **Implement.** Extend the selected method, creating a `FastGenModel` subclass in the
   matching category only when needed. Follow the contract's model, optimizer, scheduler,
   batch and checkpoint interfaces; for a new method add a typed config under
   `fastgen/configs/methods/` and its export in
   [methods/__init__.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/__init__.py). Keep model and dataset names out
   of the method; experiment configs own them.
4. **Dependencies.** Trace the network operations and batch fields the method needs
   through the actual callers; use
   [fastgen-add-network](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/SKILL.md) for missing network
   capabilities and [fastgen-dataset](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md) for supervision.
   Declare the supported prediction types, samplers, modalities and causal modes.
5. **Configs.** Use [fastgen-configure](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) for a small
   supported experiment and the target recipe, including the method's student grid,
   logging and staged initialisation.
6. **Verify the math.** Compare each objective term's value **and gradients** with the reference on
   fixed tensors, times and noise before any expensive run; cover boundary times,
   disabled terms and detach boundaries. Add and run unit tests following the
   `tests/test_*model.py` pattern and use [fastgen-validate](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md)
   for sampling parity and checkpoints.
7. **Run and document.** Use [fastgen-train](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) for a
   bounded run covering every optimizer phase and the requested distributed mode; check
   that intended modules update, frozen targets stay fixed, validation preserves training
   state and resume restores the new state. Evaluate with
   [fastgen-inference](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md) at the target NFE, add the
   method to the README tables (they feed the documentation site).

Report method and config paths, the correspondence to the reference math, tests, runs
and supported combinations. For finite but ineffective training inspect gradient paths
and optimizer phases before sweeping hyperparameters, and separate unverified reference
claims from demonstrated results.
