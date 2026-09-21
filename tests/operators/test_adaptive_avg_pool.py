"""Numerics, bidirectional inference, and error codes for ``adaptive_avg_pool``."""

import contextlib

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.operators import adaptive_avg_pool as adaptive
from hndl.torch import DTYPES, build


def pool(source, input_shape, output_shape, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    return plan, build(plan, device=device)


def module_for(output_size, device):
    """The bare pooling module, so tests can feed it dtypes the plan does not carry."""
    height, width = output_size
    plan, model = pool(f"adaptive_avg_pool(({height}, {width}))", ("B", 2, 8, 8), ("B", 2, height, width), device)
    return model[plan.nodes[0].id]


@contextlib.contextmanager
def deterministic_algorithms():
    """Run the body under ``torch.use_deterministic_algorithms(True)``, restoring the setting."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("spatial,output_size", [
    ((28, 28), 1),
    ((15, 15), 4),
    ((9, 6), (3, 2)),
    ((7, 7), 7),
    ((5, 5), 8),
    ((4, 6), (8, 3)),
])
def test_matches_torch_forward_and_backward(spatial, output_size, device):
    height, width = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    source = f"adaptive_avg_pool({output_size})"
    _, model = pool(source, ("B", 5, *spatial), ("B", 5, height, width), device)
    x = torch.randn(3, 5, *spatial, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.adaptive_avg_pool2d(mirror, (height, width))
    assert tuple(actual.shape) == (3, 5, height, width)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_output_size_one_is_the_spatial_mean(device):
    _, model = pool("adaptive_avg_pool(1)", ("B", 4, 11, 13), ("B", 4, 1, 1), device)
    x = torch.randn(2, 4, 11, 13, device=device)
    torch.testing.assert_close(model(x=x)["output"], x.mean(dim=(2, 3), keepdim=True))


def test_output_larger_than_input_replicates_like_torch(device):
    """torch permits an output grid larger than the input; windows then repeat elements."""
    _, model = pool("adaptive_avg_pool(6)", ("B", 2, 3, 3), ("B", 2, 6, 6), device)
    x = torch.randn(2, 2, 3, 3, device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, F.adaptive_avg_pool2d(x, (6, 6)))
    torch.testing.assert_close(out[:, :, 0, 0], x[:, :, 0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_stays_in_the_plan_dtype(dtype):
    plan = resolve("adaptive_avg_pool(2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 2, 2), dtype=dtype)
    model = build(plan, device="cuda:0")
    x = torch.randn(2, 3, 8, 8, device="cuda:0", dtype=DTYPES[dtype])
    out = model(x=x)["output"]
    assert out.dtype == DTYPES[dtype]
    torch.testing.assert_close(out.float(), F.adaptive_avg_pool2d(x.float(), (2, 2)), rtol=2e-3, atol=2e-3)


def test_output_size_is_inferred_backward_from_the_output_contract():
    plan = resolve("adaptive_avg_pool()", input_shape=("B", 6, 9, 9), output_shape=("B", 6, 3, 3))
    node = plan.nodes[0]
    assert node.args["output_size"] == (3, 3)
    assert node.provenance["output_size"] == "inferred"
    assert node.output_shapes["out"] == ("B", 6, 3, 3)


def test_an_int_output_size_normalizes_to_a_pair_and_a_pair_is_kept():
    square = resolve("adaptive_avg_pool(2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 2, 2))
    assert square.nodes[0].args["output_size"] == (2, 2)
    rectangle = resolve("adaptive_avg_pool((2, 4))", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 2, 4))
    assert rectangle.nodes[0].args["output_size"] == (2, 4)


def test_channels_flow_backward_through_the_pool():
    plan = resolve("conv(kernel_size=3, padding=1)\nadaptive_avg_pool(2)",
                   input_shape=("B", 3, 8, 8), output_shape=("B", 12, 2, 2))
    assert plan.nodes[0].args["out_channels"] == 12
    assert plan.nodes[1].input_shapes["x"] == ("B", 12, 8, 8)


def test_input_spatial_extents_are_not_inferable_backward():
    """Every input extent maps to the requested output, so the inverse is unconstrained."""
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("reshape()\nadaptive_avg_pool(2)", input_shape=("B", 48), output_shape=("B", 3, 2, 2))


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("adaptive_avg_pool(0)", ("B", 3, 8, 8), ("B", 3, 1, 1), "E_ARGUMENT"),
    ("adaptive_avg_pool(-2)", ("B", 3, 8, 8), ("B", 3, 1, 1), "E_ARGUMENT"),
    ("adaptive_avg_pool((1, 2, 3))", ("B", 3, 8, 8), ("B", 3, 1, 1), "E_ARGUMENT"),
    ("adaptive_avg_pool(2.5)", ("B", 3, 8, 8), ("B", 3, 2, 2), "E_ARGUMENT"),
    ("adaptive_avg_pool(2)", ("B", 3, 8, 8), ("B", 3, 4, 4), "E_CONSTRAINT"),
    ("adaptive_avg_pool(2)", ("B", 16), ("B", 16), "E_CONSTRAINT"),
    ("adaptive_avg_pool(2)", ("B", 4, 8, 8), ("B", 4, 2), "E_CONSTRAINT"),
])
def test_invalid_arguments_and_shapes_report_their_codes(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_the_pool_has_no_parameters_and_is_stateless(device):
    plan, model = pool("adaptive_avg_pool(3)", ("B", 4, 9, 9), ("B", 4, 3, 3), device)
    module = model[plan.nodes[0].id]
    assert list(module.parameters()) == [] and list(module.buffers()) == []
    x = torch.randn(2, 4, 9, 9, device=device)
    model.train()
    trained = model(x=x)["output"]
    model.eval()
    torch.testing.assert_close(model(x=x)["output"], trained)


def test_the_repr_still_names_the_output_size(device):
    module = module_for((4, 4), device)
    assert repr(module) == repr(torch.nn.AdaptiveAvgPool2d((4, 4)))
    assert module.output_size == (4, 4)


# Determinism and higher-order gradients.

@pytest.mark.parametrize("spatial,output_size", [
    ((32, 32), (4, 4)),     # divisible
    ((15, 15), (4, 4)),     # ragged
    ((9, 6), (3, 2)),       # divisible
    ((5, 5), (8, 8)),       # ragged, k > n
])
def test_forward_and_backward_run_under_deterministic_algorithms(spatial, output_size, device):
    """The divisible and small-ragged paths never reach a nondeterministic kernel."""
    height, width = output_size
    _, model = pool(f"adaptive_avg_pool({output_size})", ("B", 3, *spatial), ("B", 3, height, width), device)
    x = torch.randn(2, 3, *spatial, device=device, requires_grad=True)
    with deterministic_algorithms():
        out = model(x=x)["output"]
        out.square().sum().backward()
    torch.testing.assert_close(out, F.adaptive_avg_pool2d(x.detach(), output_size))
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("spatial,output_size,falls_back", [
    ((32, 32), (4, 4), False),      # divisible: reshape and mean
    ((15, 15), (4, 4), False),      # ragged, 16 windows: explicit slicing
    ((3, 3), (6, 6), False),        # replicated, 36 windows: explicit slicing
    ((17, 17), (9, 9), True),       # ragged, 81 windows: past MAX_EXPLICIT_WINDOWS
])
def test_only_a_large_ragged_output_falls_back_to_the_torch_kernel(spatial, output_size, falls_back, monkeypatch):
    calls = []
    kernel = adaptive.F.adaptive_avg_pool2d

    def recording(*args, **kwargs):
        calls.append(args)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(adaptive.F, "adaptive_avg_pool2d", recording)
    module_for(output_size, "cpu")(torch.randn(1, 2, *spatial))
    assert bool(calls) is falls_back
    assert adaptive.MAX_EXPLICIT_WINDOWS == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="atomics only make the CUDA backward nondeterministic")
@pytest.mark.parametrize("spatial,output_size", [((32, 32), (4, 4)), ((15, 15), (4, 4))])
def test_repeated_cuda_backward_passes_are_bit_identical(spatial, output_size):
    height, width = output_size
    _, model = pool(f"adaptive_avg_pool({output_size})", ("B", 3, *spatial), ("B", 3, height, width), "cuda:0")
    x = torch.randn(4, 3, *spatial, device="cuda:0")
    seed = torch.randn(4, 3, height, width, device="cuda:0")

    def gradient():
        leaf = x.clone().requires_grad_()
        (model(x=leaf)["output"] * seed).sum().backward()
        return leaf.grad

    first = gradient()
    for _ in range(4):
        assert torch.equal(first, gradient())


@pytest.mark.parametrize("spatial,output_size", [((32, 32), (4, 4)), ((15, 15), (4, 4))])
def test_gradient_penalty_style_double_backward(spatial, output_size, device):
    """A WGAN-GP critic differentiates the pool's own backward: grad_outputs, create_graph, backward."""
    height, width = output_size
    _, model = pool(f"adaptive_avg_pool({output_size})", ("B", 3, *spatial), ("B", 3, height, width), device)
    base = torch.randn(2, 3, *spatial, device=device, requires_grad=True)
    mirror = base.detach().clone().requires_grad_()

    def penalty(leaf, pooled):
        gradient, = torch.autograd.grad(pooled, leaf, grad_outputs=torch.ones_like(pooled), create_graph=True)
        return ((gradient.flatten(1).norm(dim=1) - 1) ** 2).mean()

    actual = penalty(base, model(x=base.pow(3))["output"])
    expected = penalty(mirror, F.adaptive_avg_pool2d(mirror.pow(3), output_size))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(base.grad, mirror.grad)
    assert base.grad.abs().sum() > 0


@pytest.mark.parametrize("spatial,output_size", [((4, 4), (2, 2)), ((5, 5), (2, 2)), ((2, 2), (3, 3))])
def test_gradgradcheck_in_float64(spatial, output_size, device):
    module = module_for(output_size, device)
    x = torch.randn(2, 2, *spatial, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))
    assert torch.autograd.gradgradcheck(module, (x,))


@pytest.mark.parametrize("spatial,output_size", [
    ((32, 32), (4, 4)),     # divisible: reshape and mean
    ((8, 12), (8, 3)),      # divisible with a width-only reduction
    ((15, 15), (4, 4)),     # ragged: explicit windows
    ((9, 7), (2, 5)),       # ragged on both axes
    ((3, 3), (6, 6)),       # replicated: k > n
    ((17, 17), (9, 9)),     # ragged past MAX_EXPLICIT_WINDOWS: the torch fallback
])
def test_every_path_matches_functional_adaptive_avg_pool2d(spatial, output_size, device):
    height, width = output_size
    _, model = pool(f"adaptive_avg_pool({output_size})", ("B", 3, *spatial), ("B", 3, height, width), device)
    x = torch.randn(2, 3, *spatial, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.adaptive_avg_pool2d(mirror, output_size)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_a_non_contiguous_input_pools_like_a_contiguous_one(device):
    _, model = pool("adaptive_avg_pool(2)", ("B", 3, 8, 8), ("B", 3, 2, 2), device)
    x = torch.randn(2, 3, 8, 8, device=device).transpose(2, 3)
    assert not x.is_contiguous()
    torch.testing.assert_close(model(x=x)["output"], F.adaptive_avg_pool2d(x, (2, 2)))
