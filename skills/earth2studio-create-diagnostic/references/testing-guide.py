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

"""Testing guide for diagnostic model wrappers.

Copy the relevant patterns into `test/models/dx/test_<model_name>.py` and
replace helper placeholders before running tests. Generated tests should be
executable: do not leave skipped, placeholder, or NotImplementedError tests.

Required standard tests:
1. `test_<model>_call` for a mock or simple forward pass.
2. `test_<model>_exceptions` for invalid coordinates.
3. `test_<model>_package` with `@pytest.mark.package` for AutoModel and
   generative diagnostics.

Generative diagnostics also need sample-count and deterministic-seed coverage.
Run tests with `uv run pytest`.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.auto import Package
from earth2studio.models.dx import ModelName  # TODO: replace with the real model
from earth2studio.utils import handshake_dim


class PhooModelName(torch.nn.Module):
    """Dummy core model matching the real model interface."""

    def __init__(self, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.out_channels]


def make_coords(model: ModelName, batch: int) -> OrderedDict:
    """Build valid diagnostic coordinates from the model's public input coords."""
    input_coords = model.input_coords()
    return OrderedDict(
        {
            "batch": np.arange(batch),
            "variable": input_coords["variable"],
            "lat": input_coords["lat"],
            "lon": input_coords["lon"],
        }
    )


def make_input(model: ModelName, batch: int, device: str) -> torch.Tensor:
    """Create random input matching model.input_coords()."""
    input_coords = model.input_coords()
    shape = [
        batch,
        len(input_coords["variable"]),
        len(input_coords["lat"]),
        len(input_coords["lon"]),
    ]
    return torch.randn(*shape, device=device)


def assert_diagnostic_coord_order(coords: OrderedDict) -> None:
    """Assert standard diagnostic coordinate order."""
    handshake_dim(coords, "batch", 0)
    handshake_dim(coords, "variable", 1)
    handshake_dim(coords, "lat", 2)
    handshake_dim(coords, "lon", 3)


def assert_generative_coord_order(coords: OrderedDict) -> None:
    """Assert generative diagnostic output coordinate order."""
    handshake_dim(coords, "batch", 0)
    handshake_dim(coords, "sample", 1)
    handshake_dim(coords, "variable", 2)
    handshake_dim(coords, "lat", 3)
    handshake_dim(coords, "lon", 4)


@pytest.fixture(scope="class")
def test_package(tmp_path_factory) -> Package:
    """Create a package for load_model tests.

    Adapt filenames and metadata to match the real `load_model` implementation.
    """
    tmp_path = tmp_path_factory.mktemp("model_data")
    torch.save(PhooModelName(out_channels=1), tmp_path / "model.pt")
    np.save(tmp_path / "center.npy", np.zeros((4, 1, 1), dtype=np.float32))
    np.save(tmp_path / "scale.npy", np.ones((4, 1, 1), dtype=np.float32))
    return Package(str(tmp_path))


def load_mock_model(test_package: Package) -> ModelName:
    """Return the wrapper under test with mock weights.

    Replace this helper with the real constructor or `ModelName.load_model` call.
    """
    raise NotImplementedError("Replace load_mock_model with model-specific setup")


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_model_call(test_package, batch, device):
    """Forward pass with mock weights."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = load_mock_model(test_package).to(device)
    x = make_input(model, batch=batch, device=device)
    coords = make_coords(model, batch=batch)

    out, out_coords = model(x, coords)

    assert out.shape[0] == batch
    assert out.shape[-2:] == (len(out_coords["lat"]), len(out_coords["lon"]))
    if "sample" in out_coords:
        assert_generative_coord_order(out_coords)
        assert out.shape[1] == len(out_coords["sample"])
        assert out.shape[2] == len(out_coords["variable"])
    else:
        assert_diagnostic_coord_order(out_coords)
        assert out.shape[1] == len(out_coords["variable"])


@pytest.mark.parametrize(
    "bad_coords_builder",
    [
        lambda model, batch: OrderedDict(
            {
                "batch": np.arange(batch),
                "variable": np.array(["wrong_var"]),
                "lat": model.input_coords()["lat"],
                "lon": model.input_coords()["lon"],
            }
        ),
        lambda model, batch: OrderedDict(
            {
                "batch": np.arange(batch),
                "variable": model.input_coords()["variable"],
                "lon": model.input_coords()["lon"],
                "lat": model.input_coords()["lat"],
            }
        ),
        lambda model, batch: OrderedDict(
            {
                "batch": np.arange(batch),
                "variable": model.input_coords()["variable"],
                "lat": model.input_coords()["lat"][::-1],
                "lon": model.input_coords()["lon"],
            }
        ),
    ],
)
def test_model_exceptions(test_package, bad_coords_builder):
    """Invalid variables, coordinate order, or coordinate values should raise."""
    model = load_mock_model(test_package)
    x = make_input(model, batch=1, device="cpu")
    with pytest.raises((KeyError, ValueError)):
        model(x, bad_coords_builder(model, 1))


@pytest.mark.package
def test_model_package():
    """Real-weight package test for AutoModel and generative diagnostics.

    Run with:
    `uv run pytest test/models/dx/test_<model_name>.py::test_<model>_package --package -v`
    """
    model = ModelName.load_model(ModelName.load_default_package())
    x = make_input(model, batch=1, device="cpu")
    coords = make_coords(model, batch=1)
    out, out_coords = model(x, coords)

    assert torch.isfinite(out).all()
    assert out.shape[0] == 1
    if "sample" in out_coords:
        assert out.shape[1] == len(out_coords["sample"])
        assert_generative_coord_order(out_coords)
    else:
        assert_diagnostic_coord_order(out_coords)


@pytest.mark.parametrize("number_of_samples", [1, 3])
def test_model_samples(test_package, number_of_samples):
    """Generative diagnostics should expose and size the sample dimension."""
    model = load_mock_model(test_package)
    model.number_of_samples = number_of_samples
    x = make_input(model, batch=1, device="cpu")
    coords = make_coords(model, batch=1)

    out, out_coords = model(x, coords)

    assert "sample" in out_coords
    assert len(out_coords["sample"]) == number_of_samples
    assert out.shape[1] == number_of_samples
    assert_generative_coord_order(out_coords)


def test_model_deterministic_seed(test_package):
    """Same seed should reproduce samples when the sampler supports seeding."""
    model_a = load_mock_model(test_package)
    model_b = load_mock_model(test_package)
    model_a.seed = 42
    model_b.seed = 42

    x = make_input(model_a, batch=1, device="cpu")
    coords = make_coords(model_a, batch=1)
    out_a, _ = model_a(x, coords)
    out_b, _ = model_b(x, coords)

    torch.testing.assert_close(out_a, out_b)
