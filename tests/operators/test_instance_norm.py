"""Numerics, mode independence, inference and error codes for ``instance_norm``."""

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.torch import build

CASES = {
    3: ("instance_norm({})\nflatten()\nlinear()", ("B", 6, 5), ("B", 3), nn.InstanceNorm1d),
    4: ("instance_norm({})\nflatten()\nlinear()", ("B", 6, 4, 5), ("B", 3), nn.InstanceNorm2d),
}


def norm_module(source, input_shape, output_shape, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=11)
    node = next(node for node in plan.nodes if node.op.startswith("instance_norm@"))
    return plan, model, model[node.id], node


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("rank", sorted(CASES))
def test_matches_torch_instance_norm_in_train_and_eval(rank, affine, device):
    source, input_shape, output_shape, reference_class = CASES[rank]
    _, _, module, _ = norm_module(source.format(f"affine={affine}"), input_shape, output_shape, device)
    reference = reference_class(input_shape[1], affine=affine).to(device)
    if affine:
        with torch.no_grad():
            module.weight.normal_()
            module.bias.normal_()
        reference.load_state_dict(module.state_dict())

    x = torch.randn(4, *input_shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    for mode in (True, False):
        module.train(mode)
        reference.train(mode)
        torch.testing.assert_close(module(x), reference(mirror))
    module(x).square().mean().backward()
    reference(mirror).square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


@pytest.mark.parametrize("rank", sorted(CASES))
def test_normalizes_each_example_and_channel_over_positions(rank, device):
    source, input_shape, output_shape, _ = CASES[rank]
    _, _, module, _ = norm_module(source.format(""), input_shape, output_shape, device)
    scale = torch.arange(1, 5, device=device).reshape(4, *[1] * (len(input_shape) - 1)).float()
    x = torch.randn(4, *input_shape[1:], device=device) * scale + scale
    out = module(x)
    positions = tuple(range(2, x.dim()))
    torch.testing.assert_close(out.mean(dim=positions), torch.zeros(4, input_shape[1], device=device),
                               atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(dim=positions, unbiased=False), torch.ones(4, input_shape[1], device=device),
                               atol=1e-2, rtol=0)
    # Per-example normalization: rescaling one example leaves the others untouched.
    other = x.clone()
    other[0] *= 5.0
    torch.testing.assert_close(module(other)[1:], out[1:])


def test_train_and_eval_agree_and_no_running_statistics(device):
    _, _, module, _ = norm_module("instance_norm()\nflatten()\nlinear()", ("B", 6, 4, 5), ("B", 3), device)
    assert module.running_mean is None and module.running_var is None
    assert module.track_running_stats is False
    assert list(module.state_dict()) == []
    assert list(module.buffers()) == []
    x = torch.randn(4, 6, 4, 5, device=device)
    trained = module(x)
    module.eval()
    torch.testing.assert_close(module(x), trained)


def test_affine_adds_exactly_two_channel_parameters(device):
    plan, _, module, node = norm_module("instance_norm(affine=True)\nflatten()\nlinear()",
                                        ("B", 6, 4, 5), ("B", 3), device)
    assert node.args["affine"] is True and node.args["num_features"] == 6
    assert list(module.state_dict()) == ["weight", "bias"]
    assert [tuple(p.shape) for p in module.parameters()] == [(6,), (6,)]
    _, _, plain, _ = norm_module("instance_norm()\nflatten()\nlinear()", ("B", 6, 4, 5), ("B", 3), device)
    assert list(plain.parameters()) == []


def test_channel_count_is_inferred_forward_and_backward():
    forward = resolve("conv(12, kernel_size=3, padding=1)\ninstance_norm()\nflatten()\nlinear()",
                      input_shape=("B", 3, 8, 8), output_shape=("B", 5))
    assert forward.nodes[1].args["num_features"] == 12
    backward = resolve("deconv(kernel_size=4, stride=2, padding=1)\ninstance_norm(num_features=9)\nflatten()\nlinear()",
                       input_shape=("B", 4, 4, 4), output_shape=("B", 5))
    assert backward.nodes[0].args["out_channels"] == 9
    assert backward.nodes[1].input_shapes["x"] == ("B", 9, 8, 8)
    sequence = resolve("instance_norm(0.01)\nflatten()\nlinear()", input_shape=("B", 7, 9), output_shape=("B", 2))
    assert sequence.nodes[0].args == {"eps": 0.01, "affine": False, "num_features": 7}


def test_invalid_arguments_and_shapes_report_their_codes():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*rank-2"):
        resolve("instance_norm()\nlinear()", input_shape=("B", 6), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*rank-2"):
        resolve("flatten()\ninstance_norm()\nlinear()", input_shape=("B", 6, 4, 5), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("instance_norm(0.0)\nflatten()\nlinear()", input_shape=("B", 6, 5), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("instance_norm(num_features=-1)\nflatten()\nlinear()", input_shape=("B", 6, 5), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("instance_norm(num_features=8)\nflatten()\nlinear()", input_shape=("B", 6, 4, 5), output_shape=("B", 3))


def test_wrong_input_rank_is_a_runtime_error(device):
    _, _, module, _ = norm_module(CASES[4][0].format(""), *CASES[4][1:3], device)
    with pytest.raises(HNDLError, match="E_RUNTIME.*rank 4"):
        module(torch.randn(4, 6, 20, device=device))
