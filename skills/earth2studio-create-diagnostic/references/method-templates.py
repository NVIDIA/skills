# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

"""Method templates for Earth2Studio diagnostic wrappers.

Use these snippets to implement one model. Keep public diagnostic coordinates in
single-step order: batch, variable, lat, lon. Generative diagnostics add sample
only in output coordinates.
"""

from collections import OrderedDict

import numpy as np
import torch
from loguru import logger

from earth2studio.models.auto import Package
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import CoordSystem

INPUT_VARIABLES = ["u10m", "v10m", "t2m", "msl"]
OUTPUT_VARIABLES = ["tp"]


def input_coords_template(self) -> CoordSystem:
    """Diagnostic public input coordinate order."""
    return OrderedDict(
        {
            "batch": np.empty(0),
            "variable": np.array(INPUT_VARIABLES),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )


def output_coords_template(self, input_coords: CoordSystem) -> CoordSystem:
    """Validate input coordinates and replace output variables."""
    target = self.input_coords()
    handshake_dim(input_coords, "variable", 1)
    handshake_dim(input_coords, "lat", 2)
    handshake_dim(input_coords, "lon", 3)
    handshake_coords(input_coords, target, "variable")
    handshake_coords(input_coords, target, "lat")
    handshake_coords(input_coords, target, "lon")

    output_coords = input_coords.copy()
    output_coords["variable"] = np.array(OUTPUT_VARIABLES)
    return output_coords


def generative_output_coords_template(self, input_coords: CoordSystem) -> CoordSystem:
    """Validate input and return output with a sample dimension."""
    target = self.input_coords()
    handshake_dim(input_coords, "variable", 1)
    handshake_dim(input_coords, "lat", 2)
    handshake_dim(input_coords, "lon", 3)
    handshake_coords(input_coords, target, "variable")
    handshake_coords(input_coords, target, "lat")
    handshake_coords(input_coords, target, "lon")

    return OrderedDict(
        {
            "batch": input_coords["batch"],
            "sample": np.arange(self.number_of_samples),
            "variable": np.array(self.output_variables),
            "lat": self.lat_output_numpy,
            "lon": self.lon_output_numpy,
        }
    )


def simple_call_template(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Example derived diagnostic forward pass."""
    output_coords = self.output_coords(coords)
    u = x[:, 0:1]
    v = x[:, 1:2]
    return torch.sqrt(u**2 + v**2), output_coords


def automodel_call_template(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Example packaged deterministic diagnostic forward pass."""
    output_coords = self.output_coords(coords)
    x = (x - self.center) / self.scale
    out = self.core_model(x)
    return out, output_coords


def generative_call_template(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Example generative diagnostic forward pass."""
    output_coords = self.output_coords(coords)
    out = torch.empty(
        [len(v) for v in output_coords.values()],
        device=x.device,
        dtype=x.dtype,
    )
    x = (x - self.in_center) / self.in_scale
    for batch_index in range(x.shape[0]):
        samples = []
        for sample_index in range(self.number_of_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + sample_index)
            samples.append(self._forward_one(x[batch_index], sample_index))
        out[batch_index] = torch.stack(samples, dim=0)
    return out, output_coords


def load_default_package_template(cls) -> Package:
    """Return an immutable package URL and stable cache name."""
    return Package(
        "hf://org/repo@commit-or-ngc-version",
        cache_options={
            "cache_storage": Package.default_cache("model_name"),
            "same_names": True,
        },
    )


@check_optional_dependencies()
def load_model_template(cls, package: Package) -> DiagnosticModel:
    """Resolve package assets, load on CPU first, and return the wrapper."""
    logger.info("Loading model-name diagnostic package")
    model_path = package.resolve("model.pt")
    center_path = package.resolve("center.npy")
    scale_path = package.resolve("scale.npy")

    core_model = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(core_model, torch.nn.Module):
        core_model.eval()
        core_model.requires_grad_(False)

    center = torch.as_tensor(np.load(center_path), dtype=torch.float32)
    scale = torch.as_tensor(np.load(scale_path), dtype=torch.float32)
    return cls(core_model=core_model, center=center, scale=scale)


def to_template(self, device: torch.device | str) -> DiagnosticModel:
    """Move non-PyTorch state only when inherited torch.nn.Module.to is insufficient."""
    torch.nn.Module.to(self, device)
    if hasattr(self, "non_torch_state"):
        self.non_torch_state = self.non_torch_state.to(device)
    return self
