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

"""Tests for AdditiveModel prognostic model."""

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import AdditiveModel
from earth2studio.utils import handshake_dim


class PhooAdditiveModel(torch.nn.Module):
    """Dummy additive model for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.01


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create dummy package with model and normalization files."""
    tmp_path = tmp_path_factory.mktemp("model_data")
    model = PhooAdditiveModel()
    torch.save(model, tmp_path / "model.pt")
    np.save(tmp_path / "center.npy", np.zeros(4))
    np.save(tmp_path / "scale.npy", np.ones(4))
    return Package(str(tmp_path))


class TestAdditiveModelMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize(
        "time",
        [
            np.array([np.datetime64("2024-01-01T00:00")]),
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
    def test_additive_call(self, test_package, time, device):
        """Test single forward pass adds constant."""
        model = AdditiveModel.load_model(test_package)
        model = model.to(device)

        dc = model.input_coords()
        del dc["batch"]
        del dc["lead_time"]
        del dc["variable"]

        ds = Random(dc)
        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(ds, time, variable, lead_time, device=device)

        out, out_coords = model(x, coords)

        assert out.shape == x.shape
        assert isinstance(out_coords, OrderedDict)
        # output coords order: time, lead_time, variable, lat, lon
        handshake_dim(out_coords, "variable", 2)

        # Check that output is input + 0.01
        torch.testing.assert_close(out, x + 0.01)

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
    def test_additive_iter(self, test_package, ensemble, device):
        """Test iterator produces correct sequence with accumulating additions."""
        model = AdditiveModel.load_model(test_package)
        model = model.to(device)

        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        del dc["batch"]
        del dc["lead_time"]
        del dc["variable"]
        ds = Random(dc)
        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(ds, time, variable, lead_time, device=device)

        x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))
        coords["ensemble"] = np.arange(ensemble)
        coords.move_to_end("ensemble", last=False)

        iterator = model.create_iterator(x, coords)
        assert isinstance(iterator, Iterable)

        step0_x, step0_coords = next(iterator)
        assert step0_x.shape[0] == ensemble
        np.testing.assert_array_equal(
            step0_coords["lead_time"], np.array([np.timedelta64(0, "h")])
        )

        for i, (step_x, step_coords) in enumerate(iterator):
            assert step_x.shape[0] == ensemble
            expected_lead = np.array([np.timedelta64((i + 1) * 6, "h")])
            np.testing.assert_array_equal(step_coords["lead_time"], expected_lead)
            # Additive model accumulates: x + (i+1) * 0.01
            torch.testing.assert_close(step_x, x + (i + 1) * 0.01)
            if i >= 4:
                break

    @pytest.mark.parametrize(
        "coords",
        [
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
        ],
    )
    def test_additive_exceptions(self, test_package, coords):
        """Test model raises on invalid coordinates."""
        model = AdditiveModel.load_model(test_package)
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

    def test_additive_normalization_buffers(self, test_package):
        """Test normalization buffers are loaded from package."""
        model = AdditiveModel.load_model(test_package)
        assert hasattr(model, "center")
        assert hasattr(model, "scale")
        assert model.center.shape == (4,)
        assert model.scale.shape == (4,)

    def test_additive_grid_size(self):
        """Test grid size is 361x720 (0.5 degree)."""
        model = AdditiveModel.load_model(AdditiveModel.load_default_package())
        coords = model.input_coords()
        assert len(coords["lat"]) == 361
        assert len(coords["lon"]) == 720
