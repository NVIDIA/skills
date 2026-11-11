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
# Toy message-passing model used by a model-wrapping evaluation task.
# It is an ordinary PyTorch module: it knows nothing about nvalchemi yet.
"""A tiny pair-distance network that predicts one scalar energy per graph.

forward(positions, atomic_numbers, batch_indices) -> energy tensor of
shape [num_graphs]. Pairwise distances are embedded with a fixed
exponential radial basis and summed with learnable coefficients, so the
model has a handful of trainable parameters but no message-passing
rounds. Positions are in angstrom and energies are arbitrary units.
"""

import torch
from torch import nn


class TinyPairModel(nn.Module):
    """Pair-distance energy model with an exponential radial basis."""

    def __init__(self, n_rbf: int = 8, cutoff: float = 5.0) -> None:
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.register_buffer(
            "centers", torch.linspace(0.5, cutoff, n_rbf).view(1, 1, n_rbf)
        )
        self.register_buffer("widths", torch.full((1, 1, n_rbf), 0.5 / n_rbf))
        self.element_embedding = nn.Embedding(16, n_rbf)
        self.pair_weight = nn.Linear(n_rbf, 1, bias=False)
        nn.init.ones_(self.pair_weight.weight)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return one energy per graph for the given atoms."""
        # pairwise distances within each graph
        dist = torch.cdist(positions, positions)  # [N, N]
        n = positions.shape[0]
        same_graph = batch_indices.view(n, 1) == batch_indices.view(1, n)
        eye = torch.eye(n, dtype=torch.bool, device=positions.device)
        pair_mask = same_graph & ~eye
        d = dist.unsqueeze(-1)
        rbf = torch.exp(-((d - self.centers) ** 2) / (2.0 * self.widths**2))
        pair_coeff = (
            self.element_embedding(atomic_numbers).unsqueeze(1)
            * self.element_embedding(atomic_numbers).unsqueeze(0)
        ).sum(-1, keepdim=True)  # [N, N, 1]
        pair_energy = (rbf * pair_coeff * self.pair_weight.weight.view(1, 1, -1)).sum(
            -1
        )
        energy_per_atom = (pair_energy * pair_mask).sum(-1) * 0.5  # [N]
        num_graphs = int(batch_indices.max().item()) + 1
        return torch.zeros(
            num_graphs, dtype=energy_per_atom.dtype, device=positions.device
        ).index_add_(0, batch_indices, energy_per_atom)
