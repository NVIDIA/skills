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

"""Diagnostic model wrapper skeletons.

Copy the relevant template only, then replace placeholders with the target
model's real variables, grid, package files, normalization, and forward logic.
Diagnostic wrappers are single-step transforms: no PrognosticMixin,
create_iterator, or lead_time coordinate.

Canonical method order:
1. __init__
2. input_coords
3. output_coords with @batch_coords()
4. __str__ if useful
5. load_default_package for packaged models
6. load_model for packaged models
7. to only for non-PyTorch state
8. private/support methods
9. __call__ with @torch.inference_mode() and @batch_func()
"""

from collections import OrderedDict

import numpy as np
import torch
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

# Optional dependency pattern for packaged diagnostics. Replace this block with
# real imports and the pyproject optional-extra name approved by the user.
try:
    OptionalCoreModel = None  # from optional_package import OptionalCoreModel
except ImportError:
    OptionalDependencyFailure("model-extra")
    OptionalCoreModel = None

INPUT_VARIABLES = ["u10m", "v10m", "t2m", "msl"]
OUTPUT_VARIABLES = ["tp"]


class SimpleDiagnostic(torch.nn.Module):
    """Simple derived diagnostic with no checkpoint or AutoModelMixin."""

    def __init__(self) -> None:
        super().__init__()

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(["u10m", "v10m"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        target = self.input_coords()
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target, "variable")
        handshake_coords(input_coords, target, "lat")
        handshake_coords(input_coords, target, "lon")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["ws10m"])
        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        u = x[:, 0:1]
        v = x[:, 1:2]
        return torch.sqrt(u**2 + v**2), output_coords


@check_optional_dependencies()
class AutoModelDiagnostic(torch.nn.Module, AutoModelMixin):
    """Packaged deterministic diagnostic with AutoModelMixin.

    Badges
    ------
    Use badges defined in docs/conf.py only, ordered as region, class, product,
    year, gpu. Example: region:global class:dx product:precip year:2026 gpu:40gb
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
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

    def __str__(self) -> str:
        return "model-name"

    @classmethod
    def load_default_package(cls) -> Package:
        return Package(
            "ngc://models/nvidia/modulus/model_name@v1.0",
            cache_options={
                "cache_storage": Package.default_cache("model_name"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> DiagnosticModel:
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

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center) / self.scale

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        x = self._normalize(x)
        out = self.core_model(x)
        return out, output_coords


@check_optional_dependencies()
class GenerativeDiagnostic(torch.nn.Module, AutoModelMixin):
    """Generative diagnostic that emits a sample dimension."""

    def __init__(
        self,
        residual_model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        lat_input: torch.Tensor,
        lon_input: torch.Tensor,
        lat_output: torch.Tensor,
        lon_output: torch.Tensor,
        input_variables: list[str],
        output_variables: list[str],
        number_of_samples: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.residual_model = residual_model
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_center", out_center)
        self.register_buffer("out_scale", out_scale)
        self.register_buffer("lat_input", lat_input)
        self.register_buffer("lon_input", lon_input)
        self.register_buffer("lat_output", lat_output)
        self.register_buffer("lon_output", lon_output)

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(self.input_variables),
                "lat": self.lat_input.detach().cpu().numpy(),
                "lon": self.lon_input.detach().cpu().numpy(),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
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
                "lat": self.lat_output.detach().cpu().numpy(),
                "lon": self.lon_output.detach().cpu().numpy(),
            }
        )

    @classmethod
    def load_default_package(cls) -> Package:
        return Package(
            "hf://org/repo@commit-or-ngc-version",
            cache_options={"cache_storage": Package.default_cache("model_name")},
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> DiagnosticModel:
        raise NotImplementedError("Load real generative weights and metadata here")

    def _generate_sample(self, x: torch.Tensor, sample_index: int) -> torch.Tensor:
        if self.seed is not None:
            torch.manual_seed(self.seed + sample_index)
        sample = self.residual_model(x.unsqueeze(0)).squeeze(0)
        return sample * self.out_scale + self.out_center

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        out = torch.empty(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=x.dtype,
        )
        x = (x - self.in_center) / self.in_scale
        for batch_index in range(x.shape[0]):
            samples = [
                self._generate_sample(x[batch_index], sample_index)
                for sample_index in range(self.number_of_samples)
            ]
            out[batch_index] = torch.stack(samples, dim=0)
        return out, output_coords
