# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The one timer both sides use, and the residue count every row carries.

Copy into `$WORKDIR/bench/`. The window matches `FoldingEngine.execute` with
`profile_inference=True` (measurement.md, Timing window): the batch is built and
moved to the GPU before it, and nothing after the second sync is inside it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

POLYMER_TYPES = frozenset({"protein", "rna", "dna"})


def time_model_forward(
    model: Any, device_batch: Any, runtime_args: Mapping[str, Any]
) -> tuple[Any, float, float, float]:
    """GPU-synced host clock around `model.forward` only.

    Args:
        model: Module whose `forward` is timed.
        device_batch: Feature batch, already on the GPU.
        runtime_args: Keyword arguments for `forward`, from the locked config.

    Returns:
        `(output, elapsed_s, peak_alloc_gb, peak_reserved_gb)`.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model(device_batch, **runtime_args)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    return output, elapsed_s, peak_alloc_gb, peak_reserved_gb


def residues_all_chains(item: Mapping[str, Any]) -> int:
    """Residues and nucleotides summed over every polymer chain of a spec item.

    Ligands are not residues. A polymer whose `chain_id` is a list counts once
    per copy. The result must equal the spec's `seq_len`.
    """
    total = 0
    for polymer in item["polymers"]:
        if polymer.get("polymer_type", "protein") not in POLYMER_TYPES:
            continue
        chain_id = polymer.get("chain_id")
        n_copies = len(chain_id) if isinstance(chain_id, list | tuple) else 1
        total += len(polymer["sequence"]) * max(n_copies, 1)
    return total
