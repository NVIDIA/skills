# Answering Cost / Time / Data Questions

Part of the orchestrator's job is to answer planning questions the user raises at any stage. Give ranges with the
assumptions stated, and prefer a cheap measured experiment over a confident guess. Treat every number as an estimate to
validate on the user's data and hardware.

## How much data do I need?

Anchor to the customization ladder ([`path-selection.md`](path-selection.md)):

- **Word boosting / custom vocab / n-gram LM:** little or no transcribed audio; an LM needs domain **text**, not audio.
- **Fine-tune:** NIM guide recommends **100+ hours**; **~10 hours** is the floor and only works when **mixed with a
  larger dataset** to avoid catastrophic forgetting.
- **New language:** cross-language transfer from ~16 h; from scratch needs thousands of hours.

Always ask what fraction is **real vs synthetic** and whether the real target-domain audio is representative.

## Synthetic vs real?

- Prefer **real** target-domain audio; it is the usual bottleneck and the strongest signal.
- Use **synthetic** (TTS, e.g. Magpie) and augmentation to fill **measured** gaps (rare words, specific noise), not as a
  bulk substitute. Keep synthetic sources separately weighted so they can be ablated.
- Validate that synthetic data actually moves domain WER before scaling it up; generic synthetic audio can dilute the
  real-domain signal.

## How many hours / steps to hit X% WER?

There is no universal mapping from hours to WER. Give the honest method instead of a fabricated number:

- Establish the **baseline WER** and the **target**; the gap size and its cause (lexical vs acoustic) set expectations.
- Run a **small pilot** (a cheap rung or a short fine-tune) and read the WER-vs-data trend before promising a full
  run's outcome.
- Larger acoustic gaps and new languages need more data/steps; lexical/phrasing gaps often close with boosting/LM at a
  fraction of the cost.

## What will it cost / how long?

- Estimate GPU-hours from dataset hours, model size, epochs/steps, and precision; multiply by the instance's hourly cost.
- Cheap rungs (boosting/LM) are minutes–hours of CPU/setup. Fine-tunes and from-scratch training dominate cost.
- Run the GPU/OOM preflight (Research/Training sub-skill) to size batch/precision before quoting a wall-clock estimate.

## L40S or H100 (which GPU)?

- **H100** (high memory bandwidth/compute, more VRAM) suits large models, large batches, long context, and the fastest
  turnaround — often best throughput-per-run for full fine-tunes.
- **L40S** (lower cost, ample VRAM) suits smaller models, inference/eval, and cost-sensitive or parallel
  smaller jobs.
- Decide on model size, batch/precision (bf16), VRAM headroom from the OOM preflight, and cost target. When unsure,
  pilot on the cheaper GPU and scale up only if turnaround is the bottleneck. Verify current instance specs/pricing.

## Framing

Lead with the cheapest sufficient path and its rough cost, then present the full fine-tuning plan and cost as the
escalation. Consult the user before committing to expensive cycles, and record assumptions and measured pilot results in
the run ledger ([`../assets/experiment-ledger-template.md`](../assets/experiment-ledger-template.md)).
