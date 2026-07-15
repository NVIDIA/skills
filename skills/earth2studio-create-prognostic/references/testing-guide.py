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

"""Testing Guide for Prognostic Model Wrappers.

This file provides test templates for Earth2Studio prognostic models.
Copy and adapt for your specific model.

Test file location: test/models/px/test_<model_name>.py

Required standard tests. Do not rename or omit these:
1. test_<model>_call - Single forward pass with mock model, parameterized over time and device where practical.
2. test_<model>_iter - Iterator produces the initial condition and advances lead_time for ensemble sizes 1 and 2.
3. test_<model>_exceptions - Invalid coordinates, variables, lead times, or dimension order raise errors.
4. test_<model>_package - Integration test with real weights using @pytest.mark.package and the repo --package option.

Use Random/fetch_data for mock call and iterator tests when possible. For package
tests, arbitrary Gaussian random fields may be physically invalid for some real
checkpoints; in that case construct a stable model-appropriate finite input
(e.g. normalization center or another documented neutral state), but still load
the real package and run the real forward path.

See real examples:
- test/models/px/test_pangu.py
- test/models/px/test_aurora.py

Use `uv run pytest` to execute tests
"""

import gc
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import ModelName  # TODO: Import your model
from earth2studio.utils import handshake_dim


# =============================================================================
# DUMMY MODEL (Phoo Pattern)
# =============================================================================


class PhooModelName(torch.nn.Module):
    """Dummy model that mimics the actual model's interface.

    The Phoo model should:
    1. Accept input in the same shape the real model expects
    2. Produce output in the same shape the real model produces
    3. Perform a simple, deterministic operation (e.g., x + 1)

    This allows testing the wrapper's reshape logic without real weights.
    """

    def __init__(self) -> None:
        super().__init__()
        # Add any parameters needed to match interface
        # self.dummy_param = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple deterministic forward pass.

        TODO: Adapt input/output shapes to match your model's interface.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor in model's native format.

        Returns
        -------
        torch.Tensor
            Output tensor (same shape as input for simple case).
        """
        # Simple deterministic operation for testing
        return x + 1


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create a dummy model package for testing.

    This fixture creates a temporary directory with:
    - model.pt: Serialized PhooModelName
    - Any additional files needed (config, normalization params, etc.)
    """
    tmp_path = tmp_path_factory.mktemp("model_data")

    # Save dummy model
    model = PhooModelName()
    torch.save(model, tmp_path / "model.pt")

    # TODO: Add additional files if needed
    # np.save(tmp_path / "center.npy", np.zeros(10))
    # np.save(tmp_path / "scale.npy", np.ones(10))

    return Package(str(tmp_path))


# =============================================================================
# MOCK TESTS (No real weights required)
# =============================================================================


class TestModelNameMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize(
        "time",
        [
            np.array(
                [
                    np.datetime64("1999-10-11T12:00"),
                    np.datetime64("2001-06-04T00:00"),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_model_call(self, test_package, time, device):
        """Test single forward pass with mock model.

        Validates:
        - Model loads from package
        - Forward pass produces correct output shape
        - Output coordinates are valid
        """
        model = ModelName.load_model(test_package)
        model = model.to(device)

        # Get input coordinates - Random only needs spatial coords (lat, lon)
        dc = model.input_coords()
        del dc["batch"]
        del dc["lead_time"]
        del dc["variable"]

        # Fetch random input data
        ds = Random(dc)
        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(ds, time, variable, lead_time, device=device)

        # Run forward pass
        out, out_coords = model(x, coords)

        # Validate output shape matches input
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

        # Validate output coordinates
        assert isinstance(out_coords, OrderedDict)
        assert coords["lat"][0] == 90
        assert coords["lat"][-1] == -90
        assert out_coords["lat"][0] == 90
        assert out_coords["lat"][-1] == -90
        # Standard models: time=0, lead_time=1, variable=2, lat=3, lon=4
        handshake_dim(out_coords, "variable", 2)

        # Validate lead_time was incremented
        # TODO: Adjust time step check for your model
        expected_lead = coords["lead_time"] + np.timedelta64(6, "h")
        np.testing.assert_array_equal(out_coords["lead_time"], expected_lead)

    @pytest.mark.parametrize("ensemble", [1, 2])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_model_iter(self, test_package, ensemble, device):
        """Test iterator produces correct sequence.

        Validates:
        - Iterator is iterable
        - First yield is initial condition (step 0)
        - Subsequent yields advance lead_time correctly
        - Ensemble dimension is preserved
        """
        model = ModelName.load_model(test_package)
        model = model.to(device)

        # Create input - Random only needs spatial coords (lat, lon)
        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        del dc["batch"]
        del dc["lead_time"]
        del dc["variable"]
        ds = Random(dc)
        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(ds, time, variable, lead_time, device=device)

        # Add ensemble dimension
        x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))
        coords["ensemble"] = np.arange(ensemble)
        coords.move_to_end("ensemble", last=False)

        # Create iterator
        iterator = model.create_iterator(x, coords)
        assert isinstance(iterator, Iterable)

        # Check initial condition (step 0)
        step0_x, step0_coords = next(iterator)
        assert step0_x.shape[0] == ensemble
        np.testing.assert_array_equal(
            step0_coords["lead_time"], np.array([np.timedelta64(0, "h")])
        )

        # Check subsequent steps
        for i, (step_x, step_coords) in enumerate(iterator):
            assert step_x.shape[0] == ensemble
            # TODO: Adjust time step for your model
            expected_lead = np.array([np.timedelta64((i + 1) * 6, "h")])
            np.testing.assert_array_equal(step_coords["lead_time"], expected_lead)
            if i >= 4:
                break

        # Cleanup
        del model
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "coords",
        [
            # Wrong variable name
            OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(["wrong_var"]),
                    "lat": np.linspace(90, -90, 10),
                    "lon": np.linspace(0, 360, 20),
                }
            ),
            # Wrong latitude convention: public model coords must be 90 to -90
            OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(["t2m"]),  # TODO: Replace with model variables
                    "lat": np.linspace(-90, 90, 10),
                    "lon": np.linspace(0, 360, 20),
                }
            ),
            # Wrong dimension order (if applicable)
            # OrderedDict({...}),
        ],
    )
    def test_model_exceptions(self, test_package, coords):
        """Test model raises on invalid coordinates.

        Validates:
        - Invalid variable names raise KeyError or ValueError
        - Wrong dimensions raise ValueError
        """
        model = ModelName.load_model(test_package)
        x = torch.randn(
            1,
            1,
            1,
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )
        with pytest.raises((KeyError, ValueError)):
            model(x, coords)


# =============================================================================
# PACKAGE/INTEGRATION TEST (Real weights required)
# =============================================================================


@pytest.mark.package
def test_model_package():
    """Integration test with real model weights.

    This test:
    1. Downloads real checkpoint via load_default_package
    2. Runs forward pass with random input
    3. Validates output shape and coordinates

    Marked with @pytest.mark.package to skip in normal test runs.
    Run with: uv run pytest test/models/px/test_<model_name>.py::test_<model>_package --package -v
    """
    # Load real model
    model = ModelName.load_model(ModelName.load_default_package())

    # Get input coordinates - Random only needs spatial coords (lat, lon)
    time = np.array([np.datetime64("2024-01-01T00:00")])
    dc = model.input_coords()
    del dc["batch"]
    del dc["lead_time"]
    del dc["variable"]

    # Fetch random data. If random fields are not valid for the real
    # checkpoint, replace x with a stable model-appropriate finite input
    # while preserving coords.
    ds = Random(dc)
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(ds, time, variable, lead_time)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)

    # Run forward pass
    out, out_coords = model(x, coords)

    # Validate output
    assert out.shape == x.shape
    assert isinstance(out_coords, OrderedDict)
    # Standard models: time=0, lead_time=1, variable=2, lat=3, lon=4
    handshake_dim(out_coords, "variable", 2)


# =============================================================================
# OPTIONAL: ADDITIONAL TEST PATTERNS
# =============================================================================


class TestModelNameAdditional:
    """Additional test patterns (optional)."""

    def test_model_dtype(self, test_package):
        """Test model preserves dtype."""
        model = ModelName.load_model(test_package)
        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        del dc["batch"]
        del dc["lead_time"]
        del dc["variable"]

        for dtype in [torch.float32, torch.float16]:
            ds = Random(dc)
            lead_time = model.input_coords()["lead_time"]
            variable = model.input_coords()["variable"]
            x, coords = fetch_data(ds, time, variable, lead_time)
            x = x.to(dtype)
            out, _ = model(x, coords)
            assert out.dtype == dtype

    def test_model_deterministic(self, test_package):
        """Test model produces deterministic output."""
        model = ModelName.load_model(test_package)
        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        del dc["batch"]
        del dc["lead_time"]
        del dc["variable"]
        ds = Random(dc)
        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(ds, time, variable, lead_time)

        out1, _ = model(x, coords)
        out2, _ = model(x, coords)
        torch.testing.assert_close(out1, out2)
