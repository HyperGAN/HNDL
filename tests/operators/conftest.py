"""Shared fixtures for the generic operator harness."""

import pytest
import torch

from hndl import Registry

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def operator_examples():
    """Every (operator, example) pair of the built-in catalog."""
    cases = []
    for spec in Registry.builtins().operators:
        for index, example in enumerate(spec.examples):
            marks = [pytest.mark.network] if example.network else []
            cases.append(pytest.param(spec, example, id=f"{spec.alias}-{index}", marks=marks))
    return cases


def example_input(example, batch, device, dtype):
    """A random input tensor matching an example's contract."""
    shape = (batch, *example.input_shape[1:])
    if example.input_dtype in ("int64", "int32"):
        return torch.randint(0, 8, shape, device=device, dtype=getattr(torch, example.input_dtype))
    if example.input_dtype == "bool":
        return torch.rand(shape, device=device) > 0.5
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=True)


def contract(example, **extra):
    kwargs = dict(input_shape=example.input_shape, output_shape=example.output_shape)
    if example.input_dtype is not None:
        kwargs["input_dtype"] = example.input_dtype
    kwargs.update(extra)
    return kwargs


def all_operators():
    return [pytest.param(spec, id=spec.alias) for spec in Registry.builtins().operators]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param
