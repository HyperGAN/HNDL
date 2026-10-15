"""Coordinate geometry, endpoints, buffer persistence, and extra_repr for ``coordinate_grid``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.coordinate_grid import CoordinateGrid
from hndl.torch import build


@pytest.mark.parametrize("height,width", [(4, 6), (8, 8), (3, 5)])
def test_coordinate_grid_endpoints_and_axes(device, height, width):
    module = CoordinateGrid(height=height, width=width).to(device)
    x = torch.randn(3, 8, device=device)
    out = module(x)
    assert out.shape == (3, height, width, 2)
    assert not out.requires_grad

    # Axis 0 is horizontal x in [-1, 1] across columns
    torch.testing.assert_close(out[0, 0, :, 0], torch.linspace(-1, 1, width, device=device))
    # Axis 1 is vertical y in [-1, 1] across rows
    torch.testing.assert_close(out[0, :, 0, 1], torch.linspace(-1, 1, height, device=device))
    # Output identical across batches
    torch.testing.assert_close(out[0], out[1])


def test_coordinate_grid_direct_init_spatial_shape(device):
    module = CoordinateGrid(spatial_shape=(8, 8)).to(device)
    assert module.extra_repr()
    x = torch.randn(2, 4, device=device)
    out = module(x)
    assert out.shape == (2, 8, 8, 2)


def test_coordinate_grid_extra_repr():
    module = CoordinateGrid(height=4, width=6)
    assert "height=4" in module.extra_repr()
    assert "width=6" in module.extra_repr()


def test_coordinate_grid_built_plan(device):
    plan = resolve("coordinate_grid(4, 6)", input_shape=("B", 8), output_shape=("B", 4, 6, 2), input_dtype="int64")
    model = build(plan, device=device)
    x = torch.randint(0, 10, (2, 8), device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 4, 6, 2)


def test_coordinate_grid_invalid_bounds():
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("coordinate_grid(1, 5)", input_shape=("B", 3), output_shape=("B", 1, 5, 2))
