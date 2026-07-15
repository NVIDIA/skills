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

"""Method Implementation Templates for Prognostic Model Wrappers.

This file contains fully-documented method templates with complete docstrings
following NumPy style. Use these as reference when implementing specific methods.

Key patterns:
- input_coords: Define coordinate system with batch, time, lead_time, variable, lat, lon
- output_coords: Validate inputs with handshake_*, increment lead_time
- __call__: Reshape E2S format → model format → run inference → reshape back
- create_iterator: Yield initial condition first, then loop with hooks
- load_model: Resolve from package, load to CPU, set eval mode
"""

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import CoordSystem

# =============================================================================
# COORDINATE TEMPLATES
# =============================================================================

# Example variable list (from E2STUDIO_VOCAB)
VARIABLES = ["t2m", "u10m", "v10m", "msl", "z500", "t850"]


def input_coords_template(self) -> CoordSystem:
    """Input coordinate system of the prognostic model.

    The coordinate system MUST follow this exact order:
    1. batch - np.empty(0), placeholder for batch dimension
    2. time - np.empty(0), filled at runtime
    3. lead_time - starts at 0h, incremented each step
    4. variable - model input variables from E2STUDIO_VOCAB
    5. lat - latitude array, 90 to -90 (north to south). Keep this
       public convention even when the source checkpoint expects south-to-north.
    6. lon - longitude array, 0 to 360

    Returns
    -------
    CoordSystem
        Coordinate system dictionary.
    """
    return OrderedDict(
        {
            "batch": np.empty(0),
            "time": np.empty(0),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(VARIABLES),
            # Common grid sizes:
            # 0.25° global: 721 x 1440
            # 0.5° global: 361 x 720
            # 1° global: 181 x 360
            # Public Earth2Studio convention is north-to-south latitude.
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )


def input_coords_with_history_template(self) -> CoordSystem:
    """Input coords for models requiring history (e.g., -6h and 0h).

    Some models need multiple time steps as input. Use negative timedeltas
    for history requirements.

    Returns
    -------
    CoordSystem
        Coordinate system with history in lead_time.
    """
    return OrderedDict(
        {
            "batch": np.empty(0),
            "time": np.empty(0),
            "lead_time": np.array(
                [
                    np.timedelta64(-6, "h"),  # History: 6h before init time
                    np.timedelta64(0, "h"),  # Current time
                ]
            ),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )


@batch_coords()
def output_coords_template(self, input_coords: CoordSystem) -> CoordSystem:
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

    # handshake_dim validates dimension exists at the expected index
    # Indices match position in OrderedDict:
    # batch=0, time=1, lead_time=2, variable=3, lat=4, lon=5
    handshake_dim(input_coords, "lead_time", 2)
    handshake_dim(input_coords, "variable", 3)
    handshake_dim(input_coords, "lat", 4)
    handshake_dim(input_coords, "lon", 5)

    # handshake_coords validates coordinate values match
    handshake_coords(input_coords, target_input_coords, "variable")
    handshake_coords(input_coords, target_input_coords, "lat")
    handshake_coords(input_coords, target_input_coords, "lon")

    output_coords = input_coords.copy()
    # Increment lead_time by model's time step
    output_coords["lead_time"] = input_coords["lead_time"] + np.array([self._time_step])
    return output_coords


# =============================================================================
# FORWARD PASS TEMPLATES
# =============================================================================


@batch_func()
def call_template(
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

    device = self.device_buffer.device
    x = x.to(device)

    # Common reshape patterns:
    #
    # Pattern 1: Model expects (batch, channel, lat, lon)
    # E2S: (batch, time, lead_time, variable, lat, lon)
    # Flatten batch*time*lead_time, variable becomes channel
    # batch_size = x.shape[0] * x.shape[1] * x.shape[2]
    # x_model = x.view(batch_size, x.shape[3], x.shape[4], x.shape[5])
    #
    # Pattern 2: Model expects (batch, time, channel, lat, lon)
    # x_model = x.squeeze(2)  # Remove lead_time dim
    #
    # Pattern 3: Model expects dict with named tensors
    # x_model = {"surface": x[..., :4, :, :], "pressure": x[..., 4:, :, :]}

    # If the source model expects a different latitude order, flip internally only.
    # Example for south-to-north model cores with public north-to-south coords:
    # x_model = torch.flip(x_model, dims=(-2,))

    # Normalize (if model requires)
    # x_model = (x_model - self.center) / self.scale

    with torch.no_grad():
        # y_model = self.model(x_model)
        pass

    # Denormalize (if model requires)
    # y_model = y_model * self.scale + self.center

    # Flip model output latitude back to public Earth2Studio order if needed.
    # y_model = torch.flip(y_model, dims=(-2,))

    # Reshape back to E2S format
    # output = y_model.view(x.shape[0], x.shape[1], x.shape[2], -1, x.shape[4], x.shape[5])

    out_coords = self.output_coords(coords)
    return x, out_coords  # Replace x with actual output


@batch_func()
def default_generator_template(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
    """Batch-decorated generator for time integration.

    IMPORTANT: Must yield initial condition first (step 0).
    Must use front_hook and rear_hook for perturbation support.

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
    # CRITICAL: Yield initial condition first (step 0)
    # This ensures the first output has lead_time = 0h
    yield x, coords

    current_x = x
    current_coords = coords
    while True:
        # front_hook: Apply perturbations before model step
        # Used by ensemble methods to inject noise
        current_x, current_coords = self.front_hook(current_x, current_coords)

        # Forward step
        current_x, current_coords = self.__call__(current_x, current_coords)

        # rear_hook: Post-processing after model step
        # Used for output transformations
        current_x, current_coords = self.rear_hook(current_x, current_coords)

        yield current_x, current_coords


def create_iterator_template(
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


# =============================================================================
# MODEL LOADING TEMPLATES
# =============================================================================


@classmethod
def load_default_package_template(cls) -> Package:
    """Default pre-trained model package.

    Returns
    -------
    Package
        Model package with checkpoint files.

    Notes
    -----
    Always lock HuggingFace URLs to a specific commit:
    - Good: hf://organization/model@abc123def456
    - Bad: hf://organization/model (may change)

    Supported URL schemes:
    - hf:// - HuggingFace Hub
    - ngc:// - NVIDIA NGC
    - s3:// - AWS S3
    - gs:// - Google Cloud Storage
    - Local paths

    Warning
    -------
    NEVER commit credentials, API keys, or secrets to this file.
    If authentication is required, use environment variables:
    - HF_TOKEN for HuggingFace
    - NGC_API_KEY for NVIDIA NGC
    - AWS credentials via standard AWS environment variables
    """
    return Package(
        "hf://organization/model-name@commit_hash",
        cache_options={
            "cache_storage": Package.default_cache("model_name"),
            "same_names": True,
        },
    )


@classmethod
@check_optional_dependencies()
def load_model_template(
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

    Notes
    -----
    Key patterns:
    1. Use package.resolve() to get local file paths
    2. Load with map_location="cpu" first
    3. Set model to eval() mode
    4. Use weights_only=False only if needed for custom classes
    """
    # Resolve files from package
    model_path = package.resolve("model.pt")

    # Load to CPU first (user moves to GPU with .to())
    core_model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,  # Set True if no custom classes
    )
    core_model.eval()

    # Load additional files if needed
    # config = json.loads(package.resolve("config.json").read_text())
    # center = torch.from_numpy(np.load(package.resolve("center.npy")))

    return cls(core_model)


def to_template(self, device: torch.device | str) -> PrognosticModel:
    """Move model to device.

    NOTE: Only implement this if you have non-PyTorch state to manage.
    For pure PyTorch models, the inherited .to() method is sufficient.

    Use cases for custom .to():
    - ONNX Runtime: Destroy session, recreate on new device
    - JAX: Handle device placement
    - Mixed backends: Coordinate multiple frameworks

    Parameters
    ----------
    device : torch.device | str
        Target device.

    Returns
    -------
    PrognosticModel
        Model on target device.
    """
    super().to(device)
    # Handle non-PyTorch state here
    # Example for ONNX:
    # self._destroy_onnx_session()
    # self._create_onnx_session(device)
    return self
