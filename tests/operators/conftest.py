"""Shared fixtures for the operator tests."""

import pytest

from hndl.testing import available_devices

DEVICES = available_devices()


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param
