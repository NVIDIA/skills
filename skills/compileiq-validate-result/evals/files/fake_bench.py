#!/usr/bin/env python3
"""CPU-only stand-in for a GPU kernel benchmark.

Prints one 'mean: <ms>' line per run, like the real benchmark. When
PTXAS_OPTIONS contains --apply-controls (the ACF is being applied), the
simulated kernel is about 10% faster. Timings carry random noise.
"""
import os
import random

applied = "--apply-controls" in os.environ.get("PTXAS_OPTIONS", "")
mean_ms, noise_ms = (0.900, 0.004) if applied else (1.000, 0.008)

print("routine=layernorm M=4096 N=8192 dtype=bf16")
print(f"mean: {random.gauss(mean_ms, noise_ms):.5f}")
