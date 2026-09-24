"""Equalized learning rate is runtime scaling, including nested projections."""
import math

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, ResolvedPlan
from hndl.torch import build, network
from hndl.operators._equalized import EqualLinear


@pytest.mark.parametrize('shape', [(2, 8), (2, 3, 8)])
@pytest.mark.parametrize('bias', [False, True])
def test_linear_forward_gradients_and_updated_weights(shape, bias):
    model = network(f'linear(5, equalized=True, bias={bias})',
                    device='cpu', input_shape=('B', *shape[1:]), output_shape=('B', *shape[1:-1], 5))
    layer = model['n0']
    if bias:
        with torch.no_grad():
            layer.bias.fill_(0.7)  # Bias must not receive the fan-in scale.
    x = torch.arange(math.prod(shape), dtype=torch.float32).sin().reshape(shape).requires_grad_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        y = model(x)
        expected = F.linear(x, layer.weight / math.sqrt(8), layer.bias)
        torch.testing.assert_close(y, expected)
        variables = (x, *model.parameters())
        actual_grad = torch.autograd.grad(y.square().mean(), variables, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().mean(), variables)
        for actual, reference in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(actual, reference)
        optimizer.zero_grad()
        y.square().mean().backward()
        optimizer.step()


def test_raw_initialization_and_default_backward_compatibility():
    model = network('linear(256, equalized=True)', device='cpu', input_shape=('B', 512), output_shape=('B', 256))
    layer = model['n0']
    assert layer.weight.var().item() == pytest.approx(1, rel=.025)
    assert layer.bias.count_nonzero() == 0
    assert not list(layer.buffers())
    for source in ('linear(16)', 'attention(4)', 'feed_forward(32)'):
        ordinary = network(source, device='cpu', input_shape=('B', 4, 16), output_shape=('B', 4, 16),
                           initialization_seed=7)
        disabled = network(source.replace(')', ', equalized=False)'),
                           device='cpu', input_shape=('B', 4, 16), output_shape=('B', 4, 16), initialization_seed=7)
        assert ordinary.state_dict().keys() == disabled.state_dict().keys()
        for name, value in ordinary.state_dict().items():
            assert torch.equal(value, disabled.state_dict()[name])
        x = torch.ones(2, 4, 16)
        torch.testing.assert_close(ordinary(x), disabled(x), rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['attention', 'feed_forward'])
def test_nested_projection_math_and_gradients(kind):
    source = ('attention(2, equalized=True, relative_position_bias=True, spatial_shape=(2,2))'
              if kind == 'attention' else 'feed_forward(12, equalized=True)')
    model = network(source, device='cpu', input_shape=('B', 4, 8), output_shape=('B', 4, 8))
    module = model['n0']
    x = torch.arange(64, dtype=torch.float32).cos().reshape(2, 4, 8).requires_grad_()

    def affine(x, layer):
        assert isinstance(layer, EqualLinear) and layer.equalized
        return F.linear(x, layer.weight / math.sqrt(layer.in_features), layer.bias)

    if kind == 'feed_forward':
        expected = affine(F.gelu(affine(x, module.up)), module.down)
    else:
        q, k, v = [affine(x, layer).reshape(2, 4, 2, 4).transpose(1, 2)
                   for layer in (module.q_proj, module.k_proj, module.v_proj)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(4) + module.position_bias()
        h = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 4, 8)
        expected = affine(h, module.o_proj)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    variables = (x, *model.parameters())
    a = torch.autograd.grad(actual.square().mean(), variables, retain_graph=True)
    b = torch.autograd.grad(expected.square().mean(), variables)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_plan_state_roundtrip_and_raw_initializer_override():
    source = '''linear(8, equalized=True, init={"weight": 1.0, "bias": 0.5})
attention(2, equalized=True)
feed_forward(16, equalized=True)'''
    model = network(source, device='cpu', input_shape=('B', 4, 8), output_shape=('B', 4, 8))
    assert torch.equal(model['n0'].weight, torch.ones_like(model['n0'].weight))
    restored = build(ResolvedPlan.from_json(model.plan.to_json()), device='cpu')
    restored.load_state_dict(model.state_dict(), strict=True)
    assert all(node.args['equalized'] for node in restored.plan.nodes)
    x = torch.ones(2, 4, 8)
    rng = torch.get_rng_state().clone()
    torch.testing.assert_close(restored(x=x)['output'], model(x), rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_equalized_linear_second_derivatives():
    layer = EqualLinear(3, 2).double()
    x = torch.tensor([[.1, .2, .3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(layer, (x,))


def test_spectral_norm_and_equalized_are_rejected_together():
    with pytest.raises(HNDLError, match='cannot be combined'):
        network('linear(8, equalized=True, spectral_norm=True)',
                device='cpu', input_shape=('B', 8), output_shape=('B', 8))
