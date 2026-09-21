"""Custom port adapters and numerical evidence for the AdaIN-style fixture."""
# ruff: noqa: E402 -- the backend is optional in pure-core environments.

from dataclasses import replace
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from examples.adaptive_normalization import (
    AdaptiveNorm, MAPPING_CONFIG, SPLIT_CONFIG, make_registry, mapping_model, split_model,
)
from hndl import Argument, HNDLError, Registry, ResolvedPlan, ShapeRule, preserves_shape, resolve
from hndl.torch import build, network, register_torch


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def reference_norm(x, params, eps=1e-5):
    """Independent reference using PyTorch's instance-normalization kernel."""
    normalized = F.instance_norm(x, use_input_stats=True, eps=eps)
    gamma, beta = torch.chunk(params, 2, dim=1)
    return normalized * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constant", [False, True])
def test_fixture_values_and_first_second_derivatives(device, constant):
    x = torch.linspace(-0.8, 1.2, 16, dtype=torch.float64, device=device).reshape(2, 2, 2, 2)
    if constant:
        x = torch.full_like(x, 0.75)
    params = torch.linspace(-0.2, 0.3, 8, dtype=torch.float64, device=device).reshape(2, 4)
    weight = torch.linspace(-0.4, 0.8, 16, dtype=torch.float64, device=device).reshape_as(x)
    direction_x = torch.linspace(0.3, -0.7, x.numel(), dtype=x.dtype, device=device).reshape_as(x)
    direction_p = torch.linspace(-0.2, 0.5, params.numel(), dtype=x.dtype, device=device).reshape_as(params)

    results = []
    for operation in (AdaptiveNorm(), reference_norm):
        features = x.detach().clone().requires_grad_()
        style = params.detach().clone().requires_grad_()
        output = operation(features, style)
        loss = (output * weight).sum() + output.square().mean()
        gradients = torch.autograd.grad(loss, (features, style), create_graph=True)
        directional = (gradients[0] * direction_x).sum() + (gradients[1] * direction_p).sum()
        second = torch.autograd.grad(directional, (features, style))
        results.append((output, *gradients, *second))
        assert all(value.isfinite().all() for value in results[-1])
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_fixture_finite_difference_gradcheck_and_gradgradcheck(device):
    # Double precision is solely a numerical check of the custom implementation;
    # HNDL execution contracts remain float32.
    x = torch.tensor([[[[-0.7, 0.4], [1.1, -0.2]], [[0.3, -0.9], [0.8, 1.3]]]],
                     device=device, dtype=torch.float64, requires_grad=True)
    params = torch.tensor([[0.1, -0.2, 0.3, -0.1]], device=device, dtype=torch.float64, requires_grad=True)
    layer = AdaptiveNorm()
    assert torch.autograd.gradcheck(layer, (x, params), eps=1e-6, atol=2e-5, rtol=1e-3)
    assert torch.autograd.gradgradcheck(layer, (x, params), eps=1e-6, atol=5e-5, rtol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_mapping_fanout_reference_gradients_optimizer_and_reload(device):
    # Nonzero ordinary affine initialization exercises both feature/style paths.
    model = network(MAPPING_CONFIG, input_shape=("B", 128), output_shape=("B", 64, 4, 4),
                    registry=make_registry(), device=device, initialization_seed=11)

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = nn.Linear(128, 256)
            self.project = nn.Linear(256, 1024)
            self.style = nn.Linear(256, 128)

        def forward(self, x):
            w = F.relu(self.mapping(x))
            return reference_norm(self.project(w).reshape(-1, 64, 4, 4), self.style(w))

    expected = Reference().to(device)
    for name in ("mapping", "project", "style"):
        getattr(expected, name).load_state_dict(model[name].state_dict())
    visits = {name: 0 for name in ("mapping", "w", "project", "seed", "style", "norm")}
    handles = []
    for name in visits:
        def count(module, args, result, name=name):
            visits[name] += 1
        handles.append(model[name].register_forward_hook(count))
    x = torch.linspace(-1, 1, 256, device=device).reshape(2, 128).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    actual, ref = model(x), expected(x_ref)
    assert visits == dict.fromkeys(visits, 1)
    torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-5)
    for value in (actual, ref):
        (value.square().mean() + value.mean()).backward()
    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-7, rtol=5e-5)
    for name in ("mapping", "project", "style"):
        for p, q in zip(model[name].parameters(), getattr(expected, name).parameters()):
            assert p.grad.isfinite().all() and torch.count_nonzero(p.grad)
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=5e-5)
    for module in (model, expected):
        torch.optim.SGD(module.parameters(), lr=0.01).step()
    torch.testing.assert_close(model(x), expected(x_ref), atol=2e-6, rtol=2e-5)
    for handle in handles:
        handle.remove()
    assert not tuple(model["norm"].parameters()) and not tuple(model["norm"].buffers())
    assert model["style"].out_features == 128
    assert model["project"].out_features == 1024
    registry = make_registry()
    restored = build(ResolvedPlan.from_json(model.plan.to_json(), registry=registry),
                     registry=registry, device=device, initialization_seed=47)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x=x)["output"], model(x))


@pytest.mark.parametrize("device", DEVICES)
def test_example_zero_affine_normalizes_and_split_remainder_receives_gradients(device):
    mapped = mapping_model(device=device)
    assert torch.count_nonzero(mapped["style"].weight) == 0
    assert torch.count_nonzero(mapped["style"].bias) == 0
    z = torch.randn(2, 128, device=device)
    features = mapped["seed"](mapped["project"](mapped["w"](mapped["mapping"](z))))
    normalized = F.instance_norm(features, use_input_stats=True, eps=1e-5)
    torch.testing.assert_close(mapped(z), normalized, atol=2e-6, rtol=2e-5)
    assert not torch.allclose(mapped(z), features)
    before = mapped(z)
    mapped.eval()
    torch.testing.assert_close(mapped(z), before)

    split = split_model(device=device)
    z = torch.randn(2, 128, device=device, requires_grad=True)
    output = split(z)
    expected = reference_norm(split["project"](z[:, :64]).reshape(2, 32, 4, 4), z[:, 64:])
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    (output.square().mean() + output.mean()).backward()
    assert z.grad.isfinite().all()
    assert torch.count_nonzero(z.grad[:, :64]) and torch.count_nonzero(z.grad[:, 64:])
    assert split["project"].out_features == 512


def test_incompatible_custom_contracts_and_scalar_arguments_fail_before_build():
    for source, output_shape in [
        (SPLIT_CONFIG, ("B", 31, 4, 4)),
        (MAPPING_CONFIG.replace('linear(w, name="style")', 'linear(w, 127, name="style")'), ("B", 64, 4, 4)),
        (MAPPING_CONFIG.replace('name="norm"', 'name="norm", eps=0'), ("B", 64, 4, 4)),
    ]:
        with pytest.raises(HNDLError):
            resolve(source, input_shape=("B", 128), output_shape=output_shape, registry=make_registry())
    model = split_model()
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        model(torch.ones(0, 128))
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        model(torch.ones(2, 127))
    with pytest.raises(ValueError, match="style parameters"):
        AdaptiveNorm()(torch.ones(2, 3, 2, 2), torch.ones(1, 6))


def test_custom_named_unary_layer_is_a_sliceable_chain():
    registry = Registry.builtins()
    registry.register("shift", identity="example.shift", version=1,
                      input_ports=("features",), output_ports=("shifted",), shape=preserves_shape,
                      arguments={"amount": Argument(float, default=0.25)}, max_state_bytes=0)

    class Shift(nn.Module):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount
        def forward(self, features):
            return features + self.amount

    register_torch(registry, "shift", module=Shift, state_version=1)
    model = network('shift(0.5, name="offset"); relu()', input_shape=("B", 3), output_shape=("B", 3),
                    registry=registry, device="cpu")
    assert model[0] is model["offset"] and len(model) == 2
    assert model[:][0] is model[0]
    x = torch.tensor([[-2., 0., 1.]])
    torch.testing.assert_close(model(x), F.relu(x + 0.5))
    torch.testing.assert_close(model[:](x), model(x))
    assert "[B, 3]" in repr(model) and "input shape" in repr(model)


@pytest.mark.parametrize("result_kind", ["tuple", "dict"])
def test_custom_multioutput_binds_declared_port_order(result_kind):
    registry = Registry.builtins()
    registry.register("pair", identity="example.pair", version=1,
                      shape=ShapeRule(inputs={"features": ("B", "F")},
                                      outputs={"positive": ("B", "F"), "negative": ("B", "F")}),
                      max_state_bytes=0)

    class Pair(nn.Module):
        def forward(self, features):
            if result_kind == "tuple":
                return features, -features
            # Reverse insertion order; declared port order is authoritative.
            return {"negative": -features, "positive": features}

    register_torch(registry, "pair", module=Pair, state_version=1)
    model = network('a, b = pair(name="pair"); concat(a, b)', input_shape=("B", 3), output_shape=("B", 6),
                    registry=registry, device="cpu")
    x = torch.tensor([[1., 2., 3.]], requires_grad=True)
    torch.testing.assert_close(model(x), torch.cat((x, -x), dim=1))
    model(x).square().sum().backward()
    torch.testing.assert_close(x.grad, 4 * x)
    with pytest.raises(TypeError):
        model[0]


def test_custom_bound_applies_after_dtype_materialization_and_versions_stay_exact():
    registry = Registry.builtins()
    registry.register("buffered", identity="example.buffered", version=1,
                      shape=preserves_shape, max_state_bytes=2)

    class Buffered(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("value", torch.ones(1, dtype=torch.float16))
        def forward(self, x):
            return x

    register_torch(registry, "buffered", module=Buffered, state_version=1)
    plan = resolve("buffered()", input_shape=("B", 2), output_shape=("B", 2), registry=registry)
    with pytest.raises(HNDLError, match="E_RESOURCE.*materialized"):
        build(plan, device="cpu", registry=registry)
    registry.backends["example.buffered@1"] = replace(registry.backends["example.buffered@1"], state_version=2)
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        build(plan, device="cpu", registry=registry)


def test_example_runs_as_script():
    result = subprocess.run([sys.executable, "examples/adaptive_normalization.py", "--device", "cpu"],
                            check=True, capture_output=True, text=True, timeout=30)
    assert "Mapping fan-out" in result.stdout and "Split and remainder" in result.stdout
    assert result.stdout.count("finite input gradient: True") == 2
