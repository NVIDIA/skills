# Choosing a public Encodon checkpoint

Use this reference for checkpoint recommendations and performance questions.
Honor a requested checkpoint and resource constraints; supplied example metadata
is not evidence that it is the best model for a different task.

| User objective | Starting choice | Reason |
| --- | --- | --- |
| Strongest published translation-efficiency/expression results | [nvidia/NV-CodonFM-Encodon-1B-v1](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1), `encodon_1b` | Largest released random-mask model; strongest downstream results in the preprint |
| Codon-frequency-weighted masking | [nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1](https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1), also `encodon_1b` | Alternative masking objective; compare on task-specific validation data |
| Quick demonstration or restricted memory/latency | 80M (`encodon_80m`), or 600M (`encodon_600m`) | Resource tradeoff, not evidence of superior embedding quality |

The [CodonFM preprint, Figure 5 and accompanying text, page 11](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf)
compares random-forest regressors trained on frozen pretrained embeddings. It
reports the strongest downstream correlation and explained variance for the 1B
model. Cdwt embeddings depend less on simple features such as GC content and can
perform worse on tasks dominated by those features. Prefer random-mask 1B for
this benchmark-driven recommendation; Cdwt is not universally better.

Keep the reported metrics distinct: Figure 5A's caption specifies mean 10-fold
cross-validation **R²** for translation efficiency; Figure 5B specifies
**Spearman correlation** for mRFP expression. An R² value is not Pearson's r or
Spearman's rho. Cite the paper/figure for performance claims, not runner source
or checkpoint configuration, which contain no benchmark results. Quote a
number only after checking the corresponding metric, panel, and dataset; label
a visually estimated value as approximate.

For a new ~200-sequence labeled set, no numerical correlation is established by
these benchmarks. Recommend 1B as an evidence-based starting point, explain the
published advantage, and separate it from expected performance on the new assay.
Suggest cross-validation of a lightweight regressor on frozen features, grouping
related sequences to reduce leakage, and reporting uncertainty. Small training
sets motivate controlling regressor complexity; they do not by themselves
justify choosing the smallest frozen backbone. Do not promise `r ≈ 0.6` or infer
correlation by taking the square root of the paper's cross-validation R².

The [public repository's model table](https://github.com/NVIDIA-Digital-Bio/CodonFM#pre-trained-models)
lists 80M, 600M, 1B, and Cdwt-1B. Parser options alone do not establish that 5B or
10B weights are released. The `TE` checkpoint family refers to the accelerated
**Transformer Engine** implementation, not a translation-efficiency-trained
checkpoint. Keep the original public runner and its compatible checkpoints
together; changing to the accelerated recipe is a separate runtime choice.
