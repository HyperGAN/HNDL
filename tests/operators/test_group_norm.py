"""Group normalization numerics, mode invariance, and divisibility validation for ``group_norm``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.operators.group_norm import GroupNorm
from hndl.torch import build


@pytest.mark.parametrize("groups,channels,shape", [
    (2, 4, (2, 4, 8)),
    (4, 16, (2, 16, 8, 8)),
    (1, 8, (2, 8, 4, 4)),
])
def test_group_norm_parity_and_gradients(device, groups, channels, shape):
    module = GroupNorm(num_groups=groups, num_channels=channels, eps=1e-5, affine=True).to(device)
    x = torch.randn(*shape, device=device, requires_grad=True)

    actual = module(x)
    expected = F.group_norm(x, groups, module.weight, module.bias, 1e-5)
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gx, gw, gb = torch.autograd.grad(actual, (x, module.weight, module.bias), grad_outputs=grad_out)
    assert gx.shape == x.shape
    assert gw.shape == module.weight.shape
    assert gb.shape == module.bias.shape


def test_group_norm_train_eval_invariance(device):
    module = GroupNorm(num_groups=4, num_channels=8, eps=1e-5, affine=True).to(device)
    x = torch.randn(2, 8, 4, 4, device=device)
    module.train()
    out_train = module(x)
    module.eval()
    out_eval = module(x)
    torch.testing.assert_close(out_train, out_eval)


def test_group_norm_built_plan(device):
    plan = resolve("conv(16, kernel_size=3, padding=1)\ngroup_norm(4)\nrelu()",
                   input_shape=("B", 3, 8, 8), output_shape=("B", 16, 8, 8))
    model = build(plan, device=device)
    x = torch.randn(2, 3, 8, 8, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 16, 8, 8)


def test_group_norm_channels_not_divisible_by_groups():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("group_norm(4)", input_shape=("B", 10, 8, 8), output_shape=("B", 10, 8, 8))
