"""Packed storage keeps model/optimizer/checkpoint contracts and sparse grads."""

import copy

import pytest
import torch
from torch import nn

from hndl.operators._packing import clear_packed, prepare_packed, project
from hndl.operators.moe import MixtureOfExperts, _RoutedLinear, _reference
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


def test_untrusted_layout_detects_in_place_stride_change():
    module = prepare_packed(SwiGLU(8, False, D=8))
    x = torch.randn(2, 8)
    module.gate.weight.data = module.gate.weight.data.t()
    expected = module.down(torch.nn.functional.silu(module.gate(x)) * module.up(x))
    torch.testing.assert_close(module(x), expected)


def test_global_projection_hooks_are_preserved():
    from torch.nn.modules import module as torch_module
    module = prepare_packed(Projections())
    seen = []
    hook = torch_module.register_module_forward_hook(lambda layer, *args: seen.append(layer))
    try:
        module(torch.randn(2, 8))
    finally:
        hook.remove()
    assert module.q_proj in seen and module.k_proj in seen and module.v_proj in seen


@pytest.mark.parametrize("packed", [False, True])
def test_sparse_moe_second_order_gradients(packed):
    module = MixtureOfExperts(3, 4, 2, "silu", D=2, expert_kind="swiglu", bias=False).double()
    module.configure_dispatch("sparse")
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
    module.configure_dispatch("sparse")
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


@pytest.mark.parametrize("dispatch", ["auto", "sparse", "padded"])
def test_dispatch_choices_match_reference(dispatch):
    module = prepare_packed(MixtureOfExperts(5, 12, 2, "silu", D=8, expert_kind="swiglu"))
    module.configure_dispatch(dispatch, max_padding_bytes=0)
    x = torch.randn(4, 8)
    torch.testing.assert_close(module(x), _reference(module)(x), rtol=1e-5, atol=1e-6)


def test_dispatch_configuration_validation():
    module = _module("mlp")
    with pytest.raises(ValueError, match="dispatch"):
        module.configure_dispatch("missing")
    with pytest.raises(ValueError, match="nonnegative"):
        module.configure_dispatch(max_padding_bytes=-1)


def test_trusted_layout_registration_and_rebind_invalidation():
    from hndl.torch import network
    model = network("swiglu(12, bias=True)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
    prepare_packed(model, trusted=True)
    layer = model[0]
    x = torch.randn(2, 8)
    with torch.no_grad():
        model(x)
        layer.gate.weight = nn.Parameter(layer.gate.weight.detach().clone() + 0.2)
        expected = layer.down(torch.nn.functional.silu(layer.gate(x)) * layer.up(x))
        torch.testing.assert_close(layer(x), expected)
        # Direct storage edits require an explicit root rebind in trusted mode.
        layer.up.weight.data = layer.up.weight.detach().clone() - 0.3
        model._preparation_epoch = getattr(model, "_preparation_epoch", 0) + 1
        expected = layer.down(torch.nn.functional.silu(layer.gate(x)) * layer.up(x))
        torch.testing.assert_close(layer(x), expected)
    copied = copy.deepcopy(model)
    with torch.no_grad():
        copied[0].gate.weight.add_(0.1)
        expected = copied[0].down(torch.nn.functional.silu(copied[0].gate(x)) * copied[0].up(x))
        torch.testing.assert_close(copied[0](x), expected)


@pytest.mark.parametrize("parameters", [False, True])
def test_trusted_functional_call_uses_ephemeral_weights_without_rebinding(parameters):
    from hndl.torch import network
    module = network("swiglu(4, bias=True)", input_shape=("B", 2), output_shape=("B", 2), device="cpu")
    prepare_packed(module, trusted=True)
    x = torch.randn(3, 2)
    original = module(x)
    before = {name: parameter.data_ptr() for name, parameter in module.named_parameters()}
    replacements = {name: parameter.detach().clone() + 0.5 for name, parameter in module.named_parameters()}
    if parameters:
        replacements = {name: nn.Parameter(value) for name, value in replacements.items()}
    replacement_pointers = {name: value.data_ptr() for name, value in replacements.items()}
    actual = torch.func.functional_call(module, replacements, (x,))
    prefix = "nodes.n_n0."
    def linear(input, name):
        return torch.nn.functional.linear(input, replacements[prefix + name + ".weight"],
                                          replacements[prefix + name + ".bias"])
    expected = linear(torch.nn.functional.silu(linear(x, "gate")) * linear(x, "up"), "down")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module(x), original)
    assert {name: parameter.data_ptr() for name, parameter in module.named_parameters()} == before
    assert {name: value.data_ptr() for name, value in replacements.items()} == replacement_pointers


@pytest.mark.parametrize("frozen", [False, True])
def test_packed_projection_preserves_weight_version_checks(frozen):
    module = SwiGLU(4, False, D=2).double()
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)
    prepare_packed(module)
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    loss = module(x).sum()
    with torch.no_grad():
        module.gate.weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("frozen", [False, True])
def test_packed_experts_check_weights_used_for_input_gradients(frozen):
    module = MixtureOfExperts(3, 4, 2, "silu", D=2).double().configure_dispatch("padded")
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)
    prepare_packed(module)
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    loss = module(x).sum()
    with torch.no_grad():
        module.experts[0].up.weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


def test_selected_route_function_all_input_gradients_and_second_derivatives():
    inputs = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([2, 0, 2, 1])
    def run(x, w, b):
        return _RoutedLinear.apply(x, w, ids, b)
    assert torch.autograd.gradcheck(run, (inputs, weights, bias))
    assert torch.autograd.gradgradcheck(run, (inputs, weights, bias))


def test_functorch_input_grad_uses_differentiable_packing_fallback():
    module = prepare_packed(SwiGLU(4, True, D=2))
    reference = clear_packed(copy.deepcopy(module))
    x = torch.randn(3, 2)
    actual = torch.func.grad(lambda value: module(value).sum())(x)
    expected = torch.func.grad(lambda value: reference(value).sum())(x)
    torch.testing.assert_close(actual, expected)
