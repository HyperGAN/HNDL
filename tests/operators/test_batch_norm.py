"""Numerics, running statistics, inference and error codes for ``batch_norm``."""

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.torch import build

CASES = {
    2: ("batch_norm()\nlinear()", ("B", 6), ("B", 3), nn.BatchNorm1d),
    3: ("batch_norm()\nflatten()\nlinear()", ("B", 6, 5), ("B", 3), nn.BatchNorm1d),
    4: ("batch_norm()\nflatten()\nlinear()", ("B", 6, 4, 5), ("B", 3), nn.BatchNorm2d),
}
STATE_KEYS = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]


def norm_module(source, input_shape, output_shape, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=7)
    node = next(node for node in plan.nodes if node.op.startswith("batch_norm@"))
    return plan, model, model[node.id], node


@pytest.mark.parametrize("rank", sorted(CASES))
def test_matches_torch_batch_norm_in_train_and_eval(rank, device):
    source, input_shape, output_shape, reference_class = CASES[rank]
    _, _, module, _ = norm_module(source, input_shape, output_shape, device)
    reference = reference_class(input_shape[1]).to(device)
    assert list(module.state_dict()) == STATE_KEYS
    reference.load_state_dict(module.state_dict())
    with torch.no_grad():
        module.weight.normal_()
        module.bias.normal_()
    reference.load_state_dict(module.state_dict())

    x = torch.randn(8, *input_shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    torch.testing.assert_close(module(x), reference(mirror))
    module(x).square().mean().backward()
    reference(mirror).square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)
    torch.testing.assert_close(module.running_mean, reference.running_mean)
    torch.testing.assert_close(module.running_var, reference.running_var)

    module.eval()
    reference.eval()
    evaluated = module(x)
    torch.testing.assert_close(evaluated, reference(x))
    assert not torch.allclose(evaluated, module.train()(x))


@pytest.mark.parametrize("rank", sorted(CASES))
def test_normalizes_channels_of_axis_one(rank, device):
    source, input_shape, output_shape, _ = CASES[rank]
    _, _, module, _ = norm_module(source, input_shape, output_shape, device)
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.zero_()
    x = torch.randn(16, *input_shape[1:], device=device) * 3 + 2
    out = module(x)
    reduced = [axis for axis in range(x.dim()) if axis != 1]
    torch.testing.assert_close(out.mean(dim=reduced), torch.zeros(input_shape[1], device=device), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(dim=reduced, unbiased=False), torch.ones(input_shape[1], device=device),
                               atol=1e-3, rtol=0)


def test_running_statistics_update_once_per_forward(device):
    source, input_shape, output_shape, _ = CASES[4]
    _, _, module, _ = norm_module(source, input_shape, output_shape, device)
    x = torch.randn(8, *input_shape[1:], device=device)
    module(x)
    assert int(module.num_batches_tracked) == 1
    expected_mean = 0.9 * 0.0 + 0.1 * x.mean(dim=(0, 2, 3))
    expected_var = 0.9 * 1.0 + 0.1 * x.var(dim=(0, 2, 3), unbiased=True)
    torch.testing.assert_close(module.running_mean, expected_mean)
    torch.testing.assert_close(module.running_var, expected_var)
    module(x)
    assert int(module.num_batches_tracked) == 2
    module.eval()
    frozen = module.running_mean.clone()
    module(x)
    assert int(module.num_batches_tracked) == 2 and torch.equal(module.running_mean, frozen)


def test_untracked_statistics_have_no_buffers_and_ignore_mode(device):
    _, _, module, node = norm_module("batch_norm(track_running_stats=False)\nflatten()\nlinear()",
                                     ("B", 6, 4, 5), ("B", 3), device)
    assert node.args["track_running_stats"] is False
    assert list(module.state_dict()) == ["weight", "bias"]
    assert module.running_mean is None and module.running_var is None
    x = torch.randn(8, 6, 4, 5, device=device)
    trained = module(x)
    module.eval()
    torch.testing.assert_close(module(x), trained)


def test_affine_false_has_no_parameters(device):
    plan, model, module, node = norm_module("batch_norm(affine=False)\nflatten()\nlinear()",
                                            ("B", 6, 4, 5), ("B", 3), device)
    assert list(module.parameters()) == []
    assert module.weight is None and module.bias is None
    assert list(module.state_dict()) == ["running_mean", "running_var", "num_batches_tracked"]
    x = torch.randn(8, 6, 4, 5, device=device)
    reference = nn.BatchNorm2d(6, affine=False).to(device)
    torch.testing.assert_close(module(x), reference(x))


def test_channel_count_is_inferred_forward_and_backward():
    forward = resolve("conv(12, kernel_size=3, padding=1)\nbatch_norm()\nflatten()\nlinear()",
                      input_shape=("B", 3, 8, 8), output_shape=("B", 5))
    assert forward.nodes[1].args["num_features"] == 12
    backward = resolve("linear()\nbatch_norm(num_features=24)\nrelu()\nlinear()",
                       input_shape=("B", 16), output_shape=("B", 4))
    assert backward.nodes[0].args["out_features"] == 24
    assert backward.nodes[1].input_shapes["x"] == ("B", 24)
    sequence = resolve("batch_norm(0.001, momentum=0.5)\nflatten()\nlinear()",
                       input_shape=("B", 7, 9), output_shape=("B", 2))
    assert sequence.nodes[0].args == {"eps": 0.001, "momentum": 0.5, "affine": True,
                                      "track_running_stats": True, "num_features": 7}


def test_invalid_arguments_and_shapes_report_their_codes():
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("batch_norm(0.0)\nlinear()", input_shape=("B", 6), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("batch_norm(momentum=1.5)\nlinear()", input_shape=("B", 6), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("batch_norm(num_features=0)\nlinear()", input_shape=("B", 6), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("batch_norm(num_features=8)\nflatten()\nlinear()", input_shape=("B", 6, 4, 5), output_shape=("B", 3))


def test_wrong_input_rank_is_a_runtime_error(device):
    _, _, module, _ = norm_module(*CASES[4][:3], device)
    with pytest.raises(HNDLError, match="E_RUNTIME.*rank 4"):
        module(torch.randn(8, 6, 20, device=device))
