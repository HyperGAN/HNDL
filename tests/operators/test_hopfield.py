"""Numerics, retrieval behavior, inference, and error codes for ``hopfield``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build, network


def reference_hopfield(x, stored, beta, steps, normalize, eps=1e-5):
    """An independent retrieval loop written directly from the update rule."""
    xi = x
    if normalize:
        mean = xi.mean(dim=-1, keepdim=True)
        variance = xi.var(dim=-1, unbiased=False, keepdim=True)
        xi = (xi - mean) / torch.sqrt(variance + eps)
    for _ in range(steps):
        scores = beta * torch.matmul(xi, stored.transpose(0, 1))
        xi = torch.matmul(torch.softmax(scores, dim=-1), stored)
    return xi


@pytest.mark.parametrize("shape,patterns,beta,steps,normalize", [
    ((3, 16), 8, 1.0, 1, True),
    ((2, 12), 5, 2.5, 3, False),
    ((4, 7, 10), 6, 0.5, 2, True),
    ((2, 5, 9), 3, 4.0, 1, False),
])
def test_forward_matches_reference_on_random_shapes(device, shape, patterns, beta, steps, normalize):
    source = f"hopfield({patterns}, beta={beta}, steps={steps}, normalize={normalize})"
    contract = ("B",) + shape[1:]
    model = network(source, input_shape=contract, output_shape=contract, device=device, initialization_seed=7)
    x = torch.randn(*shape, device=device)
    expected = reference_hopfield(x, model["n0"].stored, beta, steps, normalize)
    torch.testing.assert_close(model(x), expected)


def test_stored_patterns_have_the_declared_shape_and_initial_scale(device):
    model = network("hopfield(64)", input_shape=("B", 128), output_shape=("B", 128),
                    device=device, initialization_seed=11)
    stored = model["n0"].stored
    assert tuple(stored.shape) == (64, 128)
    assert stored.requires_grad
    assert 0.5 < stored.std().item() * 128**0.5 < 2.0
    assert sum(p.numel() for p in model.parameters()) == 64 * 128


def test_a_stored_pattern_is_a_fixed_point_at_large_beta(device):
    model = network("hopfield(6, beta=40.0, steps=4, normalize=False)", input_shape=("B", 24),
                    output_shape=("B", 24), device=device, initialization_seed=3)
    # Separate the patterns so the softmax over well-spread scores saturates.
    with torch.no_grad():
        model["n0"].stored.copy_(torch.eye(6, 24, device=device) * 3.0)
    stored = model["n0"].stored.detach()
    query = stored[2].unsqueeze(0)
    retrieved = model(query)
    torch.testing.assert_close(retrieved, query, rtol=1e-4, atol=1e-4)
    # A noisy query falls into the same basin.
    noisy = query + 0.2 * torch.randn_like(query)
    torch.testing.assert_close(model(noisy), query, rtol=1e-3, atol=1e-3)


def test_small_beta_retrieves_the_mean_of_the_stored_patterns(device):
    model = network("hopfield(5, beta=0.0001, normalize=False)", input_shape=("B", 8),
                    output_shape=("B", 8), device=device, initialization_seed=5)
    stored = model["n0"].stored.detach()
    x = torch.randn(3, 8, device=device)
    torch.testing.assert_close(model(x), stored.mean(dim=0).expand(3, 8),
                               rtol=1e-3, atol=1e-3)


def test_steps_repeat_the_same_update_and_change_the_result(device):
    contract = ("B", 16)
    torch.manual_seed(101)
    x = torch.randn(4, 16, device=device)
    one = network("hopfield(7, beta=3.0)", input_shape=contract, output_shape=contract,
                  device=device, initialization_seed=13)
    three = network("hopfield(7, beta=3.0, steps=3)", input_shape=contract, output_shape=contract,
                    device=device, initialization_seed=13)
    stored = one["n0"].stored
    torch.testing.assert_close(stored, three["n0"].stored)
    # The query is normalized once; later updates consume the previous retrieval.
    chained = one(x)
    for _ in range(2):
        chained = torch.softmax(3.0 * chained @ stored.transpose(0, 1), dim=-1) @ stored
    torch.testing.assert_close(three(x), chained)
    assert (one(x) - three(x)).abs().max().item() > 1e-4


def test_a_sharp_memory_reaches_a_fixed_point_within_a_few_steps(device):
    contract = ("B", 20)
    torch.manual_seed(7)
    x = torch.randn(3, 20, device=device)
    outputs = []
    for steps in (3, 7):
        model = network(f"hopfield(5, beta=25.0, steps={steps}, normalize=False)", input_shape=contract,
                        output_shape=contract, device=device, initialization_seed=2)
        with torch.no_grad():
            model["n0"].stored.copy_(torch.eye(5, 20, device=device) * 3.0)
        outputs.append(model(x))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)


def test_one_step_equals_attention_with_the_patterns_as_keys_and_values(device):
    contract = ("B", 6, 12)
    model = network("hopfield(9, beta=2.0, normalize=False)", input_shape=contract, output_shape=contract,
                    device=device, initialization_seed=17)
    stored = model["n0"].stored
    x = torch.randn(2, 6, 12, device=device)
    attention = torch.nn.functional.scaled_dot_product_attention(
        x * 2.0, stored.expand(2, 9, 12), stored.expand(2, 9, 12), scale=1.0)
    torch.testing.assert_close(model(x), attention, rtol=1e-4, atol=1e-4)


def test_gradients_reach_the_stored_patterns(device):
    contract = ("B", 10)
    model = network("hopfield(4, beta=1.5, steps=2)", input_shape=contract, output_shape=contract, device=device,
                    initialization_seed=19)
    x = torch.randn(5, 10, device=device, requires_grad=True)
    model(x).square().mean().backward()
    stored = model["n0"].stored
    assert stored.grad is not None and stored.grad.isfinite().all()
    assert stored.grad.abs().max().item() > 0
    assert x.grad is not None and x.grad.isfinite().all()


def test_gradient_descent_pulls_a_pattern_toward_a_target(device):
    contract = ("B", 8)
    model = network("hopfield(3, beta=6.0, normalize=False)", input_shape=contract, output_shape=contract,
                    device=device, initialization_seed=23)
    target = torch.full((1, 8), 0.5, device=device)
    x = torch.randn(1, 8, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_normalize_rescales_the_query_before_the_first_update(device):
    contract = ("B", 16)
    x = torch.randn(3, 16, device=device)
    normalized = network("hopfield(6, beta=2.0)", input_shape=contract, output_shape=contract,
                         device=device, initialization_seed=29)
    plain = network("hopfield(6, beta=2.0, normalize=False)", input_shape=contract, output_shape=contract,
                    device=device, initialization_seed=29)
    torch.testing.assert_close(normalized["n0"].stored, plain["n0"].stored)
    assert (normalized(x) - plain(x)).abs().max().item() > 1e-4
    # With normalization the scale of the query does not matter.
    torch.testing.assert_close(normalized(x), normalized(x * 7.0),
                               rtol=1e-4, atol=1e-4)


def test_shapes_are_inferred_in_both_directions():
    forward = resolve("hopfield(8)\nlinear()", input_shape=("B", 32), output_shape=("B", 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 32)
    backward = resolve("linear()\nhopfield(8)", input_shape=("B", 32), output_shape=("B", 20))
    assert backward.nodes[0].args["out_features"] == 20
    assert backward.nodes[1].input_shapes["x"] == ("B", 20)
    sequence = resolve("linear()\nhopfield(5)", input_shape=("B", 6, 32), output_shape=("B", 6, 12))
    assert sequence.nodes[1].output_shapes["out"] == ("B", 6, 12)
    assert sequence.nodes[1].args == {"patterns": 5, "beta": 1.0, "steps": 1, "normalize": True}


def test_image_tensors_are_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*flatten"):
        resolve("hopfield(4)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 8, 8))
    ok = resolve("flatten()\nhopfield(4)", input_shape=("B", 3, 8, 8), output_shape=("B", 192))
    assert ok.nodes[1].output_shapes["out"] == ("B", 192)


@pytest.mark.parametrize("source", [
    "hopfield(0)",
    "hopfield(-2)",
    "hopfield(4, beta=0.0)",
    "hopfield(4, beta=-1.0)",
    "hopfield(4, steps=0)",
    "hopfield(4, steps=-3)",
    "hopfield(4.5)",
    "hopfield(4, normalize=1)",
])
def test_invalid_arguments_report_e_argument(source):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))


def test_missing_pattern_count_is_reported():
    with pytest.raises(HNDLError):
        resolve("hopfield()", input_shape=("B", 8), output_shape=("B", 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_tracks_the_float32_result(dtype):
    contract = ("B", 6, 16)
    plan = resolve("hopfield(8, beta=2.0, steps=2)", input_shape=contract, output_shape=contract, dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=31)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(2, 6, 16, device="cuda:0", dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype
    expected = reference_hopfield(x.float(), model["n0"].stored.float(), 2.0, 2, True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
