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
            cases.append(pytest.param(spec, example, id=f"{spec.alias}-{index}"))
    return cases


def all_operators():
    return [pytest.param(spec, id=spec.alias) for spec in Registry.builtins().operators]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param
