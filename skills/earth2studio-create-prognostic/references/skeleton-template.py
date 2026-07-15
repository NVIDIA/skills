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

"""Prognostic Model Wrapper Skeleton Template.

This file demonstrates the required structure for Earth2Studio prognostic model
wrappers. Copy this template and replace TODO comments with implementations.

Method ordering is CANONICAL and must be preserved:
1. __init__ — constructor
2. input_coords — input coordinate system
3. output_coords — output coordinate system (decorated @batch_coords())
4. load_default_package — classmethod returning default Package
5. load_model — classmethod loading model from package
6. to — device management (optional, only if non-PyTorch state exists)
7. Private/support methods (e.g., _prepare_input, _normalize, etc.)
8. __call__ — single-step forward (decorated @batch_func())
9. _default_generator — batch-decorated generator (decorated @batch_func())
10. create_iterator — public time-integration entry point

See real examples:
- earth2studio/models/px/pangu.py
- earth2studio/models/px/aurora.py
- earth2studio/models/px/fcnv2.py
"""

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

# ---------------------------------------------------------------------------
# Optional dependency imports (try/except pattern)
# ---------------------------------------------------------------------------
try:
    # TODO: Import optional packages here
    # import model_package
    pass
except ImportError:
    OptionalDependencyFailure("model-name")  # TODO: Use your dependency group name
    # model_package = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TODO: Define variable list from E2STUDIO_VOCAB
VARIABLES = [
    # Surface variables
    "t2m",
    "u10m",
    "v10m",
    "msl",
    # Pressure level variables (examples)
    "z500",
    "t850",
    "u500",
    "v500",
]


class ModelName(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """One-line description of the model.

    Extended description of the model, its source, architecture,
    and any relevant details.

    Parameters
    ----------
    core_model : torch.nn.Module, optional
        Core model instance. None for simple models without external weights.
    center : torch.Tensor, optional
        Normalization center values.
    scale : torch.Tensor, optional
        Normalization scale values.

    Note
    ----
    For more information see: <link to paper/repo>

    Additional resources:
    - <link to checkpoint source>
    - <link to model documentation>

    Badges
    ------
    Use only badges defined in docs/conf.py
    (https://github.com/NVIDIA/earth2studio/blob/main/docs/conf.py). Order them
    as region(s), class, product(s), year, gpu. Do not add unsupported badges.
    region:global class:mrf product:wind product:temp year:2026 gpu:40gb
    """

    # =========================================================================
    # 1. CONSTRUCTOR
    # =========================================================================
    def __init__(
        self,
        core_model: torch.nn.Module | None = None,
        center: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = core_model
        # Register normalization parameters as buffers
        if center is not None:
            self.register_buffer("center", center)
        if scale is not None:
            self.register_buffer("scale", scale)
        # Device tracking buffer (required for all px models)
        self.register_buffer("device_buffer", torch.empty(0))
        # TODO: Store time step as timedelta
        self._time_step = np.timedelta64(6, "h")

    # =========================================================================
    # 2. INPUT COORDINATES
    # =========================================================================
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),  # MUST be first, MUST be np.empty(0)
                "time": np.empty(0),  # Dynamic time dimension
                "lead_time": np.array([np.timedelta64(0, "h")]),  # Initial lead time
                "variable": np.array(VARIABLES),
                # TODO: Set grid dimensions from reference. Keep public
                # Earth2Studio latitude north-to-south even if the core model
                # uses south-to-north internally.
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    # =========================================================================
    # 3. OUTPUT COORDINATES
    # =========================================================================
    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates with updated lead_time.

        Raises
        ------
        ValueError
            If input coordinates are invalid.
        """
        target_input_coords = self.input_coords()

        # Validate dimensions exist at correct indices
        # handshake_dim indices match position in OrderedDict:
        # batch=0, time=1, lead_time=2, variable=3, lat=4, lon=5
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "lon", 5)

        # Validate coordinate values match expected
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        output_coords = input_coords.copy()
        output_coords["lead_time"] = input_coords["lead_time"] + np.array(
            [self._time_step]
        )
        return output_coords

    # =========================================================================
    # 4. LOAD DEFAULT PACKAGE
    # =========================================================================
    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package on HuggingFace/NGC/S3.

        Returns
        -------
        Package
            Model package with checkpoint files.

        Warning
        -------
        NEVER commit credentials, API keys, or secrets to this file.
        If authentication is required, use environment variables:
        - HF_TOKEN for HuggingFace
        - NGC_API_KEY for NVIDIA NGC
        - AWS credentials via standard AWS environment variables
        """
        # TODO: Replace with actual checkpoint URL
        # Lock HuggingFace URLs to specific commit: hf://org/repo@commit
        return Package(
            "hf://organization/model-name@abc123def456",
            cache_options={
                "cache_storage": Package.default_cache("model_name"),
                "same_names": True,
            },
        )

    # =========================================================================
    # 5. LOAD MODEL
    # =========================================================================
    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic model from package.

        Parameters
        ----------
        package : Package
            Model package with checkpoint files.

        Returns
        -------
        PrognosticModel
            Loaded model instance.
        """
        # Resolve checkpoint files from package
        model_path = package.resolve("model.pt")
        center_path = package.resolve("center.npy")
        scale_path = package.resolve("scale.npy")

        # Load model (always to CPU first)
        core_model = torch.load(model_path, map_location="cpu", weights_only=False)
        core_model.eval()

        # Load normalization parameters
        center = torch.from_numpy(np.load(center_path))
        scale = torch.from_numpy(np.load(scale_path))

        return cls(core_model, center=center, scale=scale)

    # =========================================================================
    # 6. DEVICE MANAGEMENT (optional)
    # =========================================================================
    # NOTE: Only override .to() if there is non-PyTorch state to manage
    # (e.g., ONNX Runtime sessions, JAX device placement).
    # For pure PyTorch wrappers, inherited .to() is sufficient.
    #
    # def to(self, device: torch.device | str) -> PrognosticModel:
    #     """Move model to device.
    #
    #     Parameters
    #     ----------
    #     device : torch.device | str
    #         Target device.
    #
    #     Returns
    #     -------
    #     PrognosticModel
    #         Model on target device.
    #     """
    #     super().to(device)
    #     # Handle non-PyTorch state here
    #     return self

    # =========================================================================
    # 7. PRIVATE/SUPPORT METHODS
    # =========================================================================
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        return (x - self.center) / self.scale

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor."""
        return x * self.scale + self.center

    # =========================================================================
    # 8. SINGLE-STEP FORWARD
    # =========================================================================
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, time, lead_time, variable, lat, lon).
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinates one time step ahead.
        """
        target_input_coords = self.input_coords()
        handshake_dim(coords, "lead_time", 2)
        handshake_dim(coords, "variable", 3)
        handshake_dim(coords, "lat", 4)
        handshake_dim(coords, "lon", 5)
        handshake_coords(coords, target_input_coords, "lead_time")
        handshake_coords(coords, target_input_coords, "variable")
        handshake_coords(coords, target_input_coords, "lat")
        handshake_coords(coords, target_input_coords, "lon")

        # Move to device
        device = self.device_buffer.device
        x = x.to(device)

        # TODO: Reshape input tensor for the core model
        # If the core model uses south-to-north latitude, torch.flip the
        # internal tensor before the model and flip the output back before
        # returning. Public coords must remain 90 to -90.
        # E2S format: (batch, time, lead_time, variable, lat, lon)
        # Example reshape to (batch, variable, lat, lon):
        # x_model = x.squeeze(1).squeeze(1)  # Remove time and lead_time dims

        # Normalize
        # x_model = self._normalize(x_model)

        # Run forward pass
        with torch.no_grad():
            # y_model = self.model(x_model)
            pass

        # Denormalize
        # y_model = self._denormalize(y_model)

        # TODO: Reshape output back to E2S format
        # output = y_model.unsqueeze(1).unsqueeze(1)  # Add time and lead_time dims

        out_coords = self.output_coords(coords)
        # TODO: Return actual output instead of placeholder
        return x, out_coords

    # =========================================================================
    # 9. BATCH-DECORATED GENERATOR
    # =========================================================================
    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Batch-decorated generator for time integration.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each time step.
        """
        # MUST yield initial condition first (step 0)
        yield x, coords

        # Time integration loop (runs indefinitely)
        current_x = x
        current_coords = coords
        while True:
            # Apply front hook (for perturbation injection, etc.)
            current_x, current_coords = self.front_hook(current_x, current_coords)

            # Forward step
            current_x, current_coords = self.__call__(current_x, current_coords)

            # Apply rear hook (for post-processing, etc.)
            current_x, current_coords = self.rear_hook(current_x, current_coords)

            yield current_x, current_coords

    # =========================================================================
    # 10. PUBLIC ITERATOR ENTRY POINT
    # =========================================================================
    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create time-integration iterator.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each time step.
        """
        yield from self._default_generator(x, coords)
