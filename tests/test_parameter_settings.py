"""Persisted construction settings, gradients, and exact parameter targets."""

import pytest
import torch
from torch import nn

from examples.adaptive_normalization import mapping_model
from hndl import HNDLError, Registry, ResolvedPlan, resolve, resolve_callable, ops
from hndl import truncated_normal, xavier_uniform
from hndl.torch import build, network


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_parameters_keep_input_gradients_and_match_pytorch_updates(device):
    source = '''
    linear(3, name="hidden", init={"weight": 0.5, "bias": 0.25}, trainable=False)
    tanh()
    linear(2, name="head", init={"weight": 0.2}, trainable={"bias": False})
    '''
    actual = network(source, input_shape=("B", 4), output_shape=("B", 2),
                     device=device, initialization_seed=19)
    expected = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).to(device)
    expected[0].load_state_dict(actual[0].state_dict())
    expected[2].load_state_dict(actual[2].state_dict())
    expected[0].requires_grad_(False)
    expected[2].bias.requires_grad_(False)
    assert torch.equal(actual[0].weight, torch.full_like(actual[0].weight, 0.5))
    assert torch.equal(actual[0].bias, torch.full_like(actual[0].bias, 0.25))
    assert torch.equal(actual[2].weight, torch.full_like(actual[2].weight, 0.2))
    assert actual.training and all(layer.training for layer in actual)
    assert [p.requires_grad for p in actual.parameters()] == [False, False, True, False]
    before = {name: parameter.detach().clone() for name, parameter in actual.named_parameters()}
    x = torch.tensor([[-0.2, 0.1, 0.3, 0.4], [0.5, -0.4, 0.2, 0.1]],
                     device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    actual(x).square().mean().backward()
    expected(x_ref).square().mean().backward()
    assert torch.count_nonzero(x.grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    for p, q in zip(actual.parameters(), expected.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad)
        else:
            assert p.grad is None and q.grad is None
    for model in (actual, expected):
        torch.optim.SGD(model.parameters(), lr=0.1).step()
    for (name, p), q in zip(actual.named_parameters(), expected.parameters()):
        torch.testing.assert_close(p, q)
        assert torch.equal(p, before[name]) == (not p.requires_grad)
    output = actual(x)
    actual.eval()
    assert not actual.training and all(not layer.training for layer in actual)
    assert [p.requires_grad for p in actual.parameters()] == [False, False, True, False]
    torch.testing.assert_close(actual(x), output)


@pytest.mark.parametrize("device", DEVICES)
def test_plan_roundtrip_restores_masks_and_constant_signed_zero(device):
    source = 'linear(2, init={"weight": -0.0, "bias": 0.125}, trainable={"weight": False})'
    model = network(source, input_shape=("B", 3), output_shape=("B", 2), device=device)
    plan = ResolvedPlan.from_json(model.plan.to_json())
    restored = build(plan, device=device, initialization_seed=123)
    assert not restored[0].weight.requires_grad and restored[0].bias.requires_grad
    assert torch.signbit(restored[0].weight).all()
    assert torch.count_nonzero(restored[0].weight) == 0
    assert torch.equal(restored[0].bias, torch.full_like(restored[0].bias, 0.125))
    with torch.no_grad():
        model[0].bias.add_(1)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x=torch.ones(2, 3, device=device))["output"],
                               model(torch.ones(2, 3, device=device)))
    assert not restored[0].weight.requires_grad
    # PyTorch state_dict holds tensors, not requires_grad flags.
    plain = network("linear(2)", input_shape=("B", 3), output_shape=("B", 2), device=device)
    plain.load_state_dict(model.state_dict())
    assert all(parameter.requires_grad for parameter in plain.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_settings_preserve_default_rng_and_explicit_rng_scope(device):
    torch.manual_seed(53)
    plain = network("linear(3)", input_shape=("B", 4), output_shape=("B", 3), device=device)
    cpu_after_plain = torch.get_rng_state().clone()
    cuda_after_plain = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    torch.manual_seed(53)
    configured = network('linear(3, init={"weight": 0}, trainable=False)',
                         input_shape=("B", 4), output_shape=("B", 3), device=device)
    assert torch.equal(torch.get_rng_state(), cpu_after_plain)
    if cuda_after_plain:
        assert all(torch.equal(a, b) for a, b in zip(cuda_after_plain, torch.cuda.get_rng_state_all()))
    torch.testing.assert_close(configured[0].bias, plain[0].bias)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    network('linear(3, init={"bias": 0}, trainable={"weight": False})',
            input_shape=("B", 4), output_shape=("B", 3), device=device, initialization_seed=72)
    assert torch.equal(cpu_before, torch.get_rng_state())
    if cuda_before:
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))


def custom_registry(factory):
    registry = Registry.builtins()

    @registry.operator("custom", identity="example.settings", summary="Test module.", shape="x[B, ...] -> out[B, ...]")
    class Custom(nn.Module):
        def __new__(cls):
            return factory()

    return registry


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(2, 2))
        self.block[0].weight.requires_grad_(False)
        self.register_buffer("statistics", torch.zeros(2))

    def forward(self, x):
        return self.block(x)


def test_nested_exact_targets_and_constructor_trainability_defaults():
    registry = custom_registry(Nested)
    default = network("custom()", input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    assert [p.requires_grad for p in default.parameters()] == [False, True]
    partial = network('custom(init={"block.0.weight": 0.5}, trainable={"block.0.bias": False})',
                      input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    assert [p.requires_grad for p in partial.parameters()] == [False, False]
    assert torch.equal(partial[0].block[0].weight, torch.full((2, 2), 0.5))
    assert torch.equal(partial[0].statistics, torch.zeros(2))
    unfrozen = network("custom(trainable=True)", input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    assert all(parameter.requires_grad for parameter in unfrozen.parameters())


@pytest.mark.parametrize("setting, code", [("init", "E_INITIALIZATION"), ("trainable", "E_TRAINABILITY")])
@pytest.mark.parametrize("target", ["statistics", "missing", "block.0.missing"])
def test_buffer_and_missing_parameter_targets_rejected(setting, code, target):
    value = "0" if setting == "init" else "False"
    with pytest.raises(HNDLError, match=code):
        network(f'custom({setting}={{"{target}": {value}}})', input_shape=("B", 2),
                output_shape=("B", 2), registry=custom_registry(Nested), device="cpu")


class Aliased(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.alias = self.weight

    def forward(self, x):
        return x * self.weight


def test_same_parameter_aliases_accept_matching_or_partial_settings():
    registry = custom_registry(Aliased)
    model = network('custom(init={"weight": 0.5, "alias": 0.5}, trainable={"alias": False})',
                    input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    assert model[0].weight is model[0].alias
    assert torch.equal(model[0].weight, torch.full((2,), 0.5))
    assert not model[0].weight.requires_grad
    x = torch.ones(1, 2, requires_grad=True)
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.5))


@pytest.mark.parametrize("source, code", [
    ('custom(init={"weight": 1, "alias": 2})', "E_INITIALIZATION"),
    ('custom(init={"alias": 2, "weight": 1})', "E_INITIALIZATION"),
    ('custom(init={"weight": 0.0, "alias": -0.0})', "E_INITIALIZATION"),
    ('custom(trainable={"weight": True, "alias": False})', "E_TRAINABILITY"),
    ('custom(trainable={"alias": False, "weight": True})', "E_TRAINABILITY"),
])
def test_alias_conflicts_fail_before_parameter_mutation(source, code):
    layer = Aliased()
    before = layer.weight.detach().clone()
    with pytest.raises(HNDLError, match=code):
        network(source, input_shape=("B", 2), output_shape=("B", 2),
                registry=custom_registry(lambda: layer), device="cpu")
    assert torch.equal(layer.weight, before) and layer.weight.requires_grad


class SharedStorage(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.view = nn.Parameter(self.weight.detach().view(2, 2))

    def forward(self, x):
        return x


def test_distinct_storage_aliases_reject_ambiguous_initialization_or_freezing():
    registry = custom_registry(SharedStorage)
    for source, code in [
        ('custom(init={"weight": 0})', "E_INITIALIZATION"),
        ('custom(init={"weight": 0, "view": 0})', "E_INITIALIZATION"),
        ('custom(trainable={"weight": False})', "E_TRAINABILITY"),
    ]:
        with pytest.raises(HNDLError, match=code):
            network(source, input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    allowed = network("custom(trainable=False)", input_shape=("B", 2), output_shape=("B", 2),
                      registry=registry, device="cpu")
    assert not any(parameter.requires_grad for parameter in allowed.parameters())


def test_untouched_shared_storage_preserves_constructor_flags():
    class Mixed(SharedStorage):
        def __init__(self):
            super().__init__()
            self.weight.requires_grad_(False)
    model = network("custom()", input_shape=("B", 2), output_shape=("B", 2),
                    registry=custom_registry(Mixed), device="cpu")
    assert [parameter.requires_grad for parameter in model.parameters()] == [False, True]


def test_initialization_rejects_parameter_storage_aliased_by_a_buffer():
    class BufferAlias(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("shadow", self.weight.detach())
        def forward(self, x):
            return x * self.weight
    layer = BufferAlias()
    with pytest.raises(HNDLError, match="E_INITIALIZATION.*buffer"):
        network('custom(init={"weight": 0})', input_shape=("B", 2), output_shape=("B", 2),
                registry=custom_registry(lambda: layer), device="cpu")
    assert torch.equal(layer.weight, torch.ones(2))
    assert torch.equal(layer.shadow, torch.ones(2))


def test_all_nodes_validate_before_any_parameter_settings_are_applied():
    layer = Aliased()
    registry = custom_registry(lambda: layer)
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        network('custom(init={"weight": 0}, trainable=False); linear(2, init={"missing": 0})',
                input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu", initialization_seed=63)
    assert torch.equal(layer.weight, torch.ones(2)) and layer.weight.requires_grad


@pytest.mark.parametrize("source, code", [
    ('custom(trainable=True)', "E_TRAINABILITY"),
    ('custom(init={"indices": 0})', "E_INITIALIZATION"),
])
def test_unsupported_parameter_dtype_fails_before_mutation(source, code):
    class IntegerParameter(nn.Module):
        def __init__(self):
            super().__init__()
            self.indices = nn.Parameter(torch.ones(2, dtype=torch.int64), requires_grad=False)
        def forward(self, x):
            return x
    with pytest.raises(HNDLError, match=code):
        network(source, input_shape=("B", 2), output_shape=("B", 2),
                registry=custom_registry(IntegerParameter), device="cpu")


def test_declared_adaptive_style_initialization_survives_plan_restoration():
    model = mapping_model()
    style = next(node for node in model.plan.nodes if node.id == "style")
    assert dict(style.initialization["overrides"]) == {"weight": 0.0, "bias": 0.0}
    restored = build(ResolvedPlan.from_json(model.plan.to_json()), device="cpu", initialization_seed=11)
    assert torch.count_nonzero(restored["style"].weight) == 0
    assert torch.count_nonzero(restored["style"].bias) == 0


def test_invalid_target_explicit_seed_restores_rng_on_failure():
    before = torch.get_rng_state().clone()
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        network('linear(2, init={"missing": 0})', input_shape=("B", 3), output_shape=("B", 2),
                device="cpu", initialization_seed=94)
    assert torch.equal(before, torch.get_rng_state())


def test_parameterless_operations_reject_named_settings_but_allow_all_freeze():
    model = network("relu(trainable=False)", input_shape=("B", 2), output_shape=("B", 2), device="cpu")
    torch.testing.assert_close(model(torch.tensor([[-1., 2.]])), torch.tensor([[0., 2.]]))
    plan = resolve('relu(init={"weight": 0})', input_shape=("B", 2), output_shape=("B", 2))
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        build(plan, device="cpu")


SCHEME_ORACLES = [
    ('xavier_uniform(gain=1.5)', lambda t: nn.init.xavier_uniform_(t, gain=1.5)),
    ('xavier_normal()', lambda t: nn.init.xavier_normal_(t, gain=1.0)),
    ('kaiming_uniform(a=0.25, mode="fan_out", nonlinearity="leaky_relu")',
     lambda t: nn.init.kaiming_uniform_(t, a=0.25, mode="fan_out", nonlinearity="leaky_relu")),
    ('kaiming_normal(nonlinearity="relu")',
     lambda t: nn.init.kaiming_normal_(t, a=0.0, mode="fan_in", nonlinearity="relu")),
    ('truncated_normal(mean=0.5, std=0.25, a=-1.0, b=1.0)',
     lambda t: nn.init.trunc_normal_(t, mean=0.5, std=0.25, a=-1.0, b=1.0)),
    ('normal(mean=-1.0, std=0.125)', lambda t: nn.init.normal_(t, mean=-1.0, std=0.125)),
    ('uniform(a=-0.5, b=0.5)', lambda t: nn.init.uniform_(t, a=-0.5, b=0.5)),
    ('orthogonal(gain=0.75)', lambda t: nn.init.orthogonal_(t, gain=0.75)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("call, apply", SCHEME_ORACLES, ids=[call.split("(")[0] for call, _ in SCHEME_ORACLES])
def test_scheme_initializers_match_torch_nn_init_applied_by_hand(call, apply, device):
    model = network(f'linear(4, init={{"weight": {call}}})', input_shape=("B", 8), output_shape=("B", 4),
                    device=device, initialization_seed=31)
    # The same RNG scope, the same construction order, the same torch routine.
    torch.manual_seed(31)
    with torch.device(device):
        reference = nn.Linear(8, 4)
    apply(reference.weight)
    assert torch.equal(model[0].weight, reference.weight)
    assert torch.equal(model[0].bias, reference.bias)
    assert model[0].weight.requires_grad and model[0].bias.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_scheme_and_constant_overrides_mix_in_one_node_and_across_a_graph(device):
    source = '''
    linear(4, name="body", init={"weight": kaiming_uniform(nonlinearity="relu"), "bias": 0.25})
    relu()
    linear(name="head", init={"weight": xavier_uniform(gain=1.5), "bias": normal(mean=2.0, std=0.5)},
           trainable={"bias": False})
    '''
    model = network(source, input_shape=("B", 8), output_shape=("B", 2), device=device,
                    initialization_seed=57)
    torch.manual_seed(57)
    with torch.device(device):
        body, head = nn.Linear(8, 4), nn.Linear(4, 2)
    nn.init.kaiming_uniform_(body.weight, a=0.0, mode="fan_in", nonlinearity="relu")
    with torch.no_grad():
        body.bias.fill_(0.25)
    nn.init.xavier_uniform_(head.weight, gain=1.5)
    nn.init.normal_(head.bias, mean=2.0, std=0.5)
    assert torch.equal(model["body"].weight, body.weight)
    assert torch.equal(model["body"].bias, body.bias)
    assert torch.equal(model["head"].weight, head.weight)
    assert torch.equal(model["head"].bias, head.bias)
    assert not model["head"].bias.requires_grad


def test_scheme_fills_are_reproducible_from_the_initialization_seed():
    source = 'linear(4, init={"weight": orthogonal(gain=1.5), "bias": uniform(a=-0.5, b=0.5)})'
    plan = ResolvedPlan.from_json(resolve(source, input_shape=("B", 8), output_shape=("B", 4)).to_json())
    first = build(plan, device="cpu", initialization_seed=404)
    second = build(plan, device="cpu", initialization_seed=404)
    other = build(plan, device="cpu", initialization_seed=405)
    for name in ("weight", "bias"):
        assert torch.equal(getattr(first[0], name), getattr(second[0], name))
        assert not torch.equal(getattr(first[0], name), getattr(other[0], name))
    # An explicit seed still leaves the caller's own RNG untouched.
    before = torch.get_rng_state().clone()
    build(plan, device="cpu", initialization_seed=404)
    assert torch.equal(before, torch.get_rng_state())


def test_constant_only_builds_consume_exactly_the_rng_they_always_did():
    kwargs = {"input_shape": ("B", 4), "output_shape": ("B", 3), "device": "cpu"}
    torch.manual_seed(11)
    plain = network("linear(3)", **kwargs)
    after_plain = torch.get_rng_state().clone()
    torch.manual_seed(11)
    constants = network('linear(3, init={"weight": 0})', **kwargs)
    assert torch.equal(torch.get_rng_state(), after_plain)
    torch.testing.assert_close(constants[0].bias, plain[0].bias)
    # A scheme draws inside the same scope, so it advances the caller's RNG.
    torch.manual_seed(11)
    network('linear(3, init={"weight": normal()})', **kwargs)
    assert not torch.equal(torch.get_rng_state(), after_plain)


@pytest.mark.parametrize("scheme", ["xavier_uniform()", "xavier_normal()", "kaiming_uniform()",
                                    "kaiming_normal()", "orthogonal()"])
def test_matrix_schemes_reject_parameters_with_fewer_than_two_dimensions(scheme):
    with pytest.raises(HNDLError, match="E_INITIALIZATION.*at least two dimensions"):
        network(f'linear(2, init={{"bias": {scheme}}})', input_shape=("B", 3), output_shape=("B", 2),
                device="cpu")

    class Scalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(()))

        def forward(self, x):
            return x * self.gamma

    with pytest.raises(HNDLError, match="E_INITIALIZATION.*at least two dimensions"):
        network(f'custom(init={{"gamma": {scheme}}})', input_shape=("B", 2), output_shape=("B", 2),
                registry=custom_registry(Scalar), device="cpu")


@pytest.mark.parametrize("scheme", ["normal(std=0.25)", "uniform(a=-1.0, b=1.0)",
                                    "truncated_normal(std=0.5)"])
def test_elementwise_schemes_accept_one_dimensional_parameters(scheme):
    model = network(f'linear(2, init={{"bias": {scheme}}})', input_shape=("B", 3), output_shape=("B", 2),
                    device="cpu", initialization_seed=8)
    assert model[0].bias.shape == (2,)
    assert torch.isfinite(model[0].bias).all()


def test_scheme_targets_are_validated_before_any_node_is_initialized():
    layer = Aliased()
    registry = custom_registry(lambda: layer)
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        network('custom(init={"weight": 0.5}); linear(2, init={"bias": xavier_uniform()})',
                input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu",
                initialization_seed=63)
    assert torch.equal(layer.weight, torch.ones(2))


def test_scheme_aliases_of_one_parameter_must_agree():
    registry = custom_registry(Aliased)
    with pytest.raises(HNDLError, match="E_INITIALIZATION.*Conflicting"):
        network('custom(init={"weight": normal(std=1.0), "alias": normal(std=2.0)})',
                input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    with pytest.raises(HNDLError, match="E_INITIALIZATION.*Conflicting"):
        network('custom(init={"weight": normal(), "alias": 0.0})',
                input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu")
    model = network('custom(init={"weight": uniform(a=-1.0, b=1.0), "alias": uniform(a=-1.0, b=1.0)})',
                    input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu",
                    initialization_seed=5)
    assert model[0].weight is model[0].alias
    assert ((model[0].weight >= -1) & (model[0].weight <= 1)).all()


def test_transgan_style_declaration_matches_its_native_python_equivalent():
    source = '''
    linear(16, name="stem", init={"weight": xavier_uniform(gain=1.0), "bias": 0.0})
    attention(4, name="attn", init={"o_proj.weight": truncated_normal(std=0.25)})
    linear(name="head", init={"weight": truncated_normal(std=0.25), "bias": 0.0})
    '''

    def author(x):
        ops.linear(16, name="stem", init={"weight": xavier_uniform(gain=1.0), "bias": 0.0})
        ops.attention(4, name="attn", init={"o_proj.weight": truncated_normal(std=0.25)})
        ops.linear(name="head", init={"weight": truncated_normal(std=0.25), "bias": 0.0})

    shapes = {"input_shape": ("B", 6, 8), "output_shape": ("B", 6, 4)}
    plan = resolve(source, **shapes)
    assert plan.semantic_digest == resolve_callable(author, **shapes).semantic_digest
    model = build(plan, device="cpu", initialization_seed=21)
    assert torch.count_nonzero(model["stem"].bias) == 0
    assert torch.count_nonzero(model["head"].bias) == 0
    assert model["attn"].o_proj.weight.std().item() < 1.0
    assert model(x=torch.ones(2, 6, 8))["output"].shape == (2, 6, 4)
