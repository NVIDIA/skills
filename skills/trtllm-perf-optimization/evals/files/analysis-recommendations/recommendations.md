# Output of the perf-analysis step (input for optimization planning)

Workload: GPT-style 1.3B decoder pre-training step, `train.py`, PyTorch eager,
8x H100 SXM with DDP, BF16 autocast, micro-batch 8, sequence length 2048.

## Summary
Training step is launch-overhead and memory bound; 39% MFU.

## Metrics
- Step time: 412 ms
- Throughput: 318 samples/s
- MFU: 39%
- Kernel launches per step: 9,640; mean kernel duration 21 us
- Launch overhead (CPU launch time not hidden by GPU work): 31% of step
- Attention (unfused softmax(QK^T)V path): 22% of step

## Findings
1. Thousands of small kernels per step; the CPU cannot launch them fast enough.
2. Attention uses the unfused path and materializes the full score matrix.
3. Bias-add + GeLU + dropout in every MLP run as three separate kernels.
4. All matmuls run in BF16.

## Recommendations (unordered, from the analysis)
1. Switch matmuls to FP8 (expected: +25% throughput; accuracy impact unknown).
2. Write a custom Triton kernel fusing bias + GeLU + dropout (expected: +6% throughput).
3. Enable FlashAttention for the attention block (expected: +12% throughput).
4. Apply CUDA Graph capture to the training step (expected: -20% step time).
