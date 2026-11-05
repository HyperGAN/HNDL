"""Packed storage keeps model/optimizer/checkpoint contracts and sparse grads."""

import copy

import pytest
import torch
from torch import nn

from hndl.operators._packing import clear_packed, prepare_packed, project
from hndl.operators.moe import MixtureOfExperts, _reference
from hndl.operators.swiglu import SwiGLU


def _module(kind):
    if kind == "swiglu":
        return SwiGLU(hidden=12, bias=True, D=8)
    return MixtureOfExperts(experts=4, hidden=12, top_k=2, activation="silu", D=8,
                            expert_kind=kind, bias=True)


@pytest.mark.parametrize("kind", ["swiglu", "mlp", "expert_swiglu"])
def test_optimizer_and_checkpoint_contract(kind):
    # expert_swiglu selects MoE's SwiGLU architecture.
    module = (_module(kind) if kind != "expert_swiglu" else
              MixtureOfExperts(4, 12, 2, "silu", D=8, expert_kind="swiglu"))
    reference = copy.deepcopy(module)
    parameters = tuple(module.parameters())
    keys = tuple(module.state_dict())
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    other_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    prepare_packed(module)
    assert tuple(module.state_dict()) == keys
    assert all(p is q for p, q in zip(parameters, module.parameters()))
    for _ in range(3):
        x = torch.randn(2, 3, 8)
        actual = module(x)
        expected = reference(x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for p, q in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-6)
        optimizer.step()
        other_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        other_optimizer.zero_grad(set_to_none=True)
    restored = copy.deepcopy(reference)
    restored.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), module(x))
    module.load_state_dict(reference.state_dict(), strict=True)
    torch.testing.assert_close(module(x), reference(x))


@pytest.mark.parametrize("kind", ["swiglu", "mlp"])
def test_deepcopy_cast_and_requires_grad_changes(kind):
    module = prepare_packed(_module(kind))
    x = torch.randn(2, 8)
    module(x)
    copied = copy.deepcopy(module).double()
    x64 = x.double().requires_grad_()
    actual = copied(x64)
    reference = copy.deepcopy(copied)
    clear_packed(reference)
    torch.testing.assert_close(actual, reference(x64))
    for parameter in copied.parameters():
        parameter.requires_grad_(False)
    assert copied(x64).requires_grad
    copied(x64).sum().backward()
    assert x64.grad is not None
    for parameter in copied.parameters():
        parameter.requires_grad_(True)
    copied(x64).sum().backward()
    assert all(p.grad is not None for p in copied.parameters())


def test_inference_preparation_then_training_and_direct_weight_updates():
    module = SwiGLU(12, True, D=8)
    with torch.inference_mode():
        prepare_packed(module)
        module(torch.randn(2, 8))
    x = torch.randn(2, 8, requires_grad=True)
    module(x).sum().backward()
    assert module.gate.weight.grad is not None
    with torch.no_grad():
        module.gate.weight.data.add_(0.25)
    expected = module.down(torch.nn.functional.silu(module.gate(x)) * module.up(x))
    torch.testing.assert_close(module(x), expected)


def test_no_parameter_stack_after_preparation(monkeypatch):
    module = prepare_packed(MixtureOfExperts(5, 12, 2, "silu", D=8, expert_kind="swiglu"))
    x = torch.randn(4, 8)
    expected = _reference(module)(x)
    def no_stack(*args, **kwargs):
        raise AssertionError("stacking parameters during prepared forward")
    monkeypatch.setattr(torch, "stack", no_stack)
    torch.testing.assert_close(module(x), expected)


class Projections(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 16)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, x):
        return project(self, x, ("q_proj", "k_proj", "v_proj"))


def test_qkv_fusion_and_projection_hooks():
    module = Projections()
    x = torch.randn(2, 3, 8, requires_grad=True)
    expected = module(x)
    prepare_packed(module)
    actual = module(x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    events = []
    module.k_proj.register_forward_hook(lambda *args: events.append("k"))
    module(x)
    assert events == ["k"]


@pytest.mark.parametrize("distinct", [False, True])
def test_tied_or_shared_storage_is_not_rebound(distinct):
    module = SwiGLU(8, False, D=8)
    module.up.weight = (nn.Parameter(module.gate.weight.detach()) if distinct else module.gate.weight)
    address = module.gate.weight.data_ptr()
    prepare_packed(module)
    assert module.gate.weight.data_ptr() == address == module.up.weight.data_ptr()
    x = torch.randn(2, 8)
    actual = module(x)
    expected = module.down(torch.nn.functional.silu(module.gate(x)) * module.up(x))
    torch.testing.assert_close(actual, expected)


def test_packing_twice_reuses_storage():
    module = prepare_packed(SwiGLU(12, True, D=8))
    pointers = tuple(p.data_ptr() for p in module.parameters())
    prepare_packed(module)
    assert tuple(p.data_ptr() for p in module.parameters()) == pointers


@pytest.mark.parametrize("packed", [False, True])
def test_sparse_moe_second_order_gradients(packed):
    module = MixtureOfExperts(3, 4, 2, "silu", D=2, expert_kind="swiglu", bias=False).double()
    if packed:
        prepare_packed(module)
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))
    assert torch.autograd.gradgradcheck(module, (x,))


@pytest.mark.parametrize("kind", ["swiglu", "mlp"])
def test_packed_fullgraph_compilation(kind):
    module = prepare_packed(_module(kind))
    x = torch.randn(3, 8, requires_grad=True)
    expected = module(x)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(p.grad is not None for p in module.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_moe_cuda_training():
    module = MixtureOfExperts(8, 32, 2, "silu", D=16, expert_kind="swiglu").cuda()
    reference = copy.deepcopy(module)
    prepare_packed(module)
    x = torch.randn(2, 5, 16, device="cuda", requires_grad=True)
    expected = _reference(reference)(x)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for p, q in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5)
