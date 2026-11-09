---
name: fastgen-validate
description: >-
  Validate NVIDIA FastGen integrations and training quality: numerical parity against reference implementations, seeded side-by-side media, gradient and trainability checks, checkpoint round trips and repository tests. Use it whenever a claim about correctness, parity or quality needs evidence.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Validate FastGen behavior

Apply [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md). This skill owns comparison
and acceptance evidence; [fastgen-inference](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md)
generates outputs and [fastgen-train](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) executes training.
Follow the [new-network checklist](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#validate-a-new-network)
for adapters, and the
[run contracts](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/references/run-contract.md#interpret-media-and-scalars-correctly)
when callback logs are the evidence.

1. **Manifest.** Freeze source and checkpoint revisions, environment, inputs, seeds and
   noise tensors, preprocessing, dtype, schedule, guidance, grid/NFE and output geometry;
   keep validation inputs held out from training. Fix tolerances and quality criteria
   before looking at results.
2. **Parity.** Compare upstream and FastGen at widening scope: normalised inputs and
   encoded conditioning and latents, the same noisy state and time through the denoiser,
   converted predictions and sampler steps, decoded outputs. Match the actual noise and
   time grids rather than seeds, since equal seeds do not imply equal RNG consumption.
   Prefer independently constructed upstream components over shared prepared ones, check
   that shared weights and buffers survive preparation, and treat precision differences
   as a separate experiment. Find the first differing intermediate before loosening
   tolerances; a loosened tolerance is a separately documented comparison that leaves
   the failure against the manifest's criteria on record.
3. **Media.** Generate matched seeded examples through both implementations across
   prompts and conditioning modes; present side-by-side galleries for human judgment and
   record judgments separately from metrics. Small galleries are not benchmarks.
4. **Trainability.** Check finite losses and gradients before clipping, intended
   parameter updates and frozen modules; run generation followed by backward under the
   real distributed wrapper, since inference can leave lazy state behind. Inspect
   per-iteration media and fixed-input outputs around a short SFT run, exercise
   auxiliary phases, save, reload and compare outputs, and verify
   [all saved training state](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/references/run-contract.md#resume-restores-state-not-the-exact-trajectory)
   when resume is claimed.
5. **Distillation quality.** Compare held-out teacher and student outputs at the target
   NFE with the actual checkpoint and sample count; use task metrics (for images the
   [FID script](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#fid-evaluation)), perceptual inspection and
   timing/memory measurements. Count network evaluations, not sampling intervals, when
   methods use guidance or multiple heads.
6. **Tests.** Reuse the patterns in [tests/](https://github.com/NVlabs/FastGen/tree/main/tests) (shared helpers, markers such as
   `large_model` and `integration`, tiny CPU models) and run `make lint` and the relevant
   tests as described in [CONTRIBUTING.md](https://github.com/NVlabs/FastGen/blob/main/CONTRIBUTING.md#continuous-integration)
   before claiming a code change is validated.

Keep a compact report with commands, manifests, metrics, media, tracker links, hardware
and explicit pass/fail/not-run states; keep large artifacts outside the source tree and
name repeated evaluations distinctly so they cannot replace the reference. Measure and
report reference nondeterminism separately.
