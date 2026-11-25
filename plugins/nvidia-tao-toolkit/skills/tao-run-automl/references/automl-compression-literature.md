# AutoML For Distill, Prune, And Quantize

Literature-backed policy for non-train TAO AutoML actions. Reviewed 2026-06-02.

## Conclusion

The existing TAO AutoML algorithms are sufficient for first-class support across
`distill`, `prune`, and `quantize` when the action exposes a valid schema and a
metric can be measured. Bayesian/BFBO search fits small single-shot compression
spaces; Hyperband, ASHA, BOHB, DEHB, and PBT fit only when the action has a real
fidelity axis such as epochs, checkpoints, pruning retrain epochs, calibration
budget, or population generations. LLM, hybrid, and autoresearch can plan search
spaces, but must still delegate to schema-valid parameters and real metrics.

New algorithms are not a blocker for broad non-train action support. For best
accuracy/latency/size tradeoffs, the most valuable future additions are
compression-aware candidate priors and multi-objective/hardware-aware search,
not a replacement for the existing HPO stack.

## Action Policy

`distill` is train-like when it optimizes over epochs and writes checkpoints.
Search temperature, hard/soft loss weights, teacher checkpoint choice, learning
rate, augmentation, and schedule parameters with Bayesian/BFBO for small budgets
or budgeted algorithms when an epoch/rung axis exists.

`prune` is usually a single-shot or short retrain action. Use Bayesian/BFBO for
pruning threshold/ratio/granularity and retrain knobs. Use Hyperband/ASHA/BOHB/
DEHB only when the action schema includes a meaningful partial-fidelity budget,
for example retrain epochs or progressive sparsity stages. Prefer downstream
`eval_fn` when the prune job itself does not emit the task metric.

`quantize` is often a calibration/search action with a discrete or mixed
precision space. Use Bayesian/BFBO for calibration size, batch count, precision
mode, per-layer/group choices exposed by the schema, and QAT knobs. Use
budgeted algorithms only when lower-fidelity calibration/eval is representative
enough to rank candidates. Use `eval_fn` for post-quantize evaluate/inference
metrics and latency measurements.

## Evidence

- Knowledge distillation is a direct compression method: Hinton, Vinyals, and
  Dean show that ensemble knowledge can be compressed into a smaller deployable
  model and improve a production acoustic model. This supports treating distill
  as a training-like action whose loss, teacher, and schedule knobs are valid
  AutoML parameters. Source: https://arxiv.org/abs/1503.02531
- Hyperband frames HPO as adaptive resource allocation over configurations and
  reports large speedups over Bayesian baselines when a valid resource such as
  iterations, data, or features can be allocated. This supports budgeted
  algorithms only for non-train actions with a meaningful fidelity/resource
  field. Source: https://www.jmlr.org/papers/v18/16-558.html
- ASHA is designed for large-scale parallel HPO with aggressive early stopping,
  so it is appropriate for action jobs that produce comparable partial metrics
  and are cheap enough to stop early. Source: https://arxiv.org/abs/1810.05934
- BOHB combines Bayesian model guidance with Hyperband budgets and reports
  strong anytime and final performance across varied deep learning problems.
  This supports BOHB/DEHB for action spaces that have both search parameters
  and a valid budget axis. Source: https://proceedings.mlr.press/v80/falkner18a.html
- AMC uses reinforcement learning for model compression policy search and shows
  improved compression quality and mobile speedups. This is evidence for future
  compression-aware candidate generators or priors when TAO needs more than
  generic HPO over exposed prune knobs. Source:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.html
- HAQ uses hardware feedback in the quantization loop and shows that optimal
  mixed-precision policies differ by architecture, hardware, and resource
  constraint. This is evidence for future hardware-aware or multi-objective
  AutoML when optimizing quantized accuracy, latency, energy, and size together.
  Source:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html
- HAWQ uses Hessian sensitivity to choose mixed precision and fine-tuning order,
  showing that sensitivity-aware priors can improve quantization search quality.
  This argues for adding Hessian/sensitivity metadata as candidate priors rather
  than requiring a new optimizer before supporting quantize actions. Source:
  https://arxiv.org/abs/1905.03696
- Network Slimming demonstrates channel sparsity and pruning can reduce model
  size and compute while preserving comparable accuracy, supporting schema-level
  search over sparsity/pruning controls plus downstream task evaluation. Source:
  https://arxiv.org/abs/1708.06519

## Recommended Roadmap

1. Ship broad non-train support with existing algorithms, gated by
   `schemas/<action>.schema.json`, `spec_template_<action>.yaml`, and metric
   extraction/evaluation hooks.
2. Add optional multi-objective result handling for accuracy, latency, size,
   memory, and energy so compression workflows can report Pareto candidates
   instead of one scalar unless the user chooses a scalar objective.
3. Add compression-aware priors: pruning sensitivity, layer saliency, Hessian or
   activation sensitivity, and hardware measurements. Feed them into existing
   Bayesian/BFBO/LLM/hybrid candidate generation.
4. Add hardware-aware policy search only after the runner can collect reliable
   latency/energy/size signals per platform. RL/evolutionary compression search
   is useful, but only with that measurement loop in place.
