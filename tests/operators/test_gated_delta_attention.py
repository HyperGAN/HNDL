"""Numerics, inference and error codes for the ``gated_delta_attention`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.operators.gated_delta_attention import GatedDeltaAttention
from hndl.testing import check_operator
from hndl.torch import DTYPES, build, parameter_counts

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "gated_delta_attention")


def test_channel_decay_runs_and_preserves_shape(device):
    plan = resolve(
        'gated_delta_attention(4, decay_kind="channel", chunk_size=32)',
        input_shape=("B", 16, 64),
        output_shape=("B", 16, 64),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 16, 64, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 16, 64)
    assert out.isfinite().all()


def test_head_decay_runs_and_preserves_shape(device):
    plan = resolve(
        'gated_delta_attention(4, decay_kind="head", chunk_size=32)',
        input_shape=("B", 16, 64),
        output_shape=("B", 16, 64),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 16, 64, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 16, 64)
    assert out.isfinite().all()


def test_explicit_head_dim_and_conv_disabled(device):
    plan = resolve(
        "gated_delta_attention(2, head_dim=16, conv_size=0, chunk_size=16)",
        input_shape=("B", 12, 32),
        output_shape=("B", 12, 32),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 12, 32, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 12, 32)
    assert out.isfinite().all()


def test_parameter_counts_channel_decay():
    plan = resolve(
        'gated_delta_attention(4, decay_kind="channel", chunk_size=32, bias=False)',
        input_shape=("B", 16, 64),
        output_shape=("B", 16, 64),
    )
    assert parameter_counts(plan) == {"n0": 21888}


def test_parameter_counts_head_decay():
    plan = resolve(
        'gated_delta_attention(4, decay_kind="head", conv_size=0, chunk_size=32, bias=False)',
        input_shape=("B", 16, 64),
        output_shape=("B", 16, 64),
    )
    assert parameter_counts(plan) == {"n0": 19012}


def test_heads_not_dividing_d_reports_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divide"):
        resolve(
            "gated_delta_attention(5, head_dim=0)",
            input_shape=("B", 16, 64),
            output_shape=("B", 16, 64),
        )


@pytest.mark.parametrize("seq_len", [1, 7, 16, 35, 64, 73])
@pytest.mark.parametrize("decay_kind", ["channel", "head"])
def test_sequential_reference_parity_across_sequence_lengths(device, seq_len, decay_kind):
    plan = resolve(
        f'gated_delta_attention(2, decay_kind="{decay_kind}", chunk_size=16)',
        input_shape=("B", seq_len, 32),
        output_shape=("B", seq_len, 32),
    )
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    ref_fn = REGISTRY.get("gated_delta_attention").reference(module)

    x = torch.randn(2, seq_len, 32, device=device)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected)


def test_gradcheck_double(device):
    plan = resolve(
        "gated_delta_attention(2, head_dim=4, conv_size=2, chunk_size=4)",
        input_shape=("B", 6, 8),
        output_shape=("B", 6, 8),
    )
    model = build(plan, device=device)
    module = model[plan.nodes[0].id].double()
    x = torch.randn(2, 6, 8, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        'gated_delta_attention(4, decay_kind="channel", chunk_size=32)',
        input_shape=("B", 16, 64),
        output_shape=("B", 16, 64),
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 16, 64, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()


@pytest.mark.parametrize("decay_kind", ["channel", "head"])
def test_gated_delta_attention_empty_batch(device, decay_kind):
    """Verify that empty batch inputs (b=0) execute cleanly and propagate gradients."""
    layer = GatedDeltaAttention(heads=2, head_dim=16, chunk_size=32, decay_kind=decay_kind, D=32).to(device)
    x = torch.randn(0, 32, 32, device=device, requires_grad=True)
    out = layer(x)
    assert out.shape == (0, 32, 32)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (0, 32, 32)


@pytest.mark.parametrize("decay_kind", ["channel", "head"])
@pytest.mark.parametrize("a_val", [2.0, 5.0, 10.0])
def test_gated_delta_attention_backward_gradient_stability_under_strong_decay(device, decay_kind, a_val):
    """Verify that strong decay rates (A_log >= 2.0) produce 100% finite gradients."""
    layer = GatedDeltaAttention(
        heads=2, head_dim=16, chunk_size=16, conv_size=0, decay_kind=decay_kind, D=32
    ).to(device)
    with torch.no_grad():
        layer.A_log.fill_(a_val)
    x = torch.randn(2, 32, 32, device=device, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad)), f"NaN gradients detected for {decay_kind} with A_log={a_val}"
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), f"NaN param gradient in {name}"

