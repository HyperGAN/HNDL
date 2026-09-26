"""Coordinate geometry, shared Fourier features, sampler gradients, and contracts."""
import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import network, build


def test_grid_axis_order_endpoints_dtype_batch_and_no_input_dependency():
    m = network('coordinate_grid(3, 5)', input_shape=("B", 7),
                output_shape=("B", 3, 5, 2), device='cpu').double()
    z = torch.randn(2, 7, dtype=torch.double, requires_grad=True)
    out = m(z)
    assert out.dtype == torch.double and not out.requires_grad
    torch.testing.assert_close(out[0, 0, :, 0], torch.linspace(-1, 1, 5, dtype=torch.double))
    torch.testing.assert_close(out[0, :, 0, 1], torch.linspace(-1, 1, 3, dtype=torch.double))
    assert torch.equal(out[0], out[1])
    assert len(list(m.parameters())) == 0
    assert m['n0'].state_dict()['grid'].shape == (1, 3, 5, 2)


def test_fourier_formula_coordinate_derivatives_roundtrip_and_no_forward_rng():
    m = network('fourier_features(7, scale=8.0)', input_shape=("B", 5, 2),
                output_shape=("B", 5, 14), device='cpu').double()
    x = torch.randn(2, 5, 2, dtype=torch.double, requires_grad=True)
    state = torch.get_rng_state().clone()
    result = m(x)
    phases = x @ m['n0'].frequencies.t()
    torch.testing.assert_close(result, torch.cat((phases.sin(), phases.cos()), dim=-1))
    assert torch.equal(torch.get_rng_state(), state)
    assert not list(m.parameters())
    assert torch.autograd.gradcheck(m, (x,))
    assert torch.autograd.gradgradcheck(m, (x,))
    replica = build(m.plan, device='cpu').double()
    replica.load_state_dict(m.state_dict())
    torch.testing.assert_close(replica(x=x)['output'], result, rtol=0, atol=0)


@pytest.mark.parametrize('padding', ['zeros', 'border', 'reflection'])
def test_sampler_matches_pytorch_for_image_and_coordinate_gradients(padding):
    m = network(f'grid_sample(image, coords, padding_mode="{padding}")',
                input_shape={'image': ('B', 2, 4, 5), 'coords': ('B', 3, 6, 2)},
                output_shape=('B', 2, 3, 6), device='cpu').double()
    a = torch.randn(2, 2, 4, 5, dtype=torch.double, requires_grad=True)
    g = (torch.randn(2, 3, 6, 2, dtype=torch.double) * 1.5).requires_grad_()
    actual = m(image=a, coords=g)
    expected = F.grid_sample(a, g, padding_mode=padding, align_corners=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    v = torch.randn_like(actual)
    got = torch.autograd.grad((actual*v).sum(), (a, g))
    want = torch.autograd.grad((expected*v).sum(), (a, g))
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('a_shape,b_shape', [((2, 3, 4), (2, 1, 4)),
                                          ((2, 1, 4), (2, 3, 1)),
                                          ((2, 3, 4, 5), (2, 3, 1, 1))])
def test_broadcast_mul_gradient_reductions_and_double_backward(a_shape, b_shape):
    output = ('B', *torch.broadcast_shapes(a_shape, b_shape)[1:])
    m = network('broadcast_mul(a, b)',
                input_shape={'a': ('B', *a_shape[1:]), 'b': ('B', *b_shape[1:])},
                output_shape=output, device='cpu').double()
    a = torch.randn(a_shape, dtype=torch.double, requires_grad=True)
    b = torch.randn(b_shape, dtype=torch.double, requires_grad=True)
    y = m(a=a, b=b)
    torch.testing.assert_close(y, a*b)
    v = torch.randn_like(y)
    ga, gb = torch.autograd.grad((y*v).sum(), (a, b), create_graph=True)
    torch.testing.assert_close(ga, (v*b).sum_to_size(a.shape))
    torch.testing.assert_close(gb, (v*a).sum_to_size(b.shape))
    assert torch.autograd.gradgradcheck(lambda a,b: m(a=a,b=b), (a,b), fast_mode=True)


def test_broadcast_mul_backward_shape_inference():
    plan = resolve('gate = mean(2, keepdim=True)\nh = linear(x)\nbroadcast_mul(h, gate)',
                   input_shape=('B', 6, 4), output_shape=('B', 6, 8))
    assert plan.nodes[1].args['out_features'] == 8


@pytest.mark.parametrize('source,ins,outs', [
    ('coordinate_grid(1, 5)', ('B', 3), ('B', 1, 5, 2)),
    ('fourier_features(0)', ('B', 3, 2), ('B', 3, 4)),
    ('fourier_features(2, scale=0.0)', ('B', 3, 2), ('B', 3, 4)),
    ('fourier_features(2)', ('B', 3, 2), ('B', 3, 5)),
    ('broadcast_mul(a,b)', {'a': ('B',3,4), 'b': ('B',3,5)}, ('B',3,4)),
    ('broadcast_mul(a,b)', {'a': ('B',3,4), 'b': ('B',4)}, ('B',3,4)),
    ('grid_sample(a,b)', {'a': ('B',3,4,4), 'b': ('B',8,8,3)}, ('B',3,8,8)),
])
def test_invalid_contracts(source, ins, outs):
    with pytest.raises(HNDLError):
        resolve(source, input_shape=ins, output_shape=outs)
