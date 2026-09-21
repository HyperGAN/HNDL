"""Numerical and integration checks for the optional PyTorch backend."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

import hndl
from hndl import HNDLError, Registry, preserves_shape
from hndl.torch import build, network, network_file, network_from_callable, register_torch


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _mlp(**kwargs):
    return network('linear(5, name="hidden"); relu(); linear(name="head")',
                   input_shape=("B", 4), output_shape=("B", 2), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_mlp_values_gradients_optimizer_and_reload(device):
    actual = _mlp(device=device, initialization_seed=41)
    expected = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2)).to(device)
    expected[0].load_state_dict(actual[0].state_dict())
    expected[2].load_state_dict(actual[2].state_dict())
    x = torch.tensor([[1., -2., 3., 0.5], [-1., 0.3, 0.7, 2.]], device=device, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    a, b = actual(x), expected(other_x)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, other_x.grad)
    for p, q in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.03) for m in (actual, expected)]
    for optimizer in optimizers:
        optimizer.step()
    for p, q in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p, q)
    restored = build(actual.plan, device=device, initialization_seed=8)
    restored.load_state_dict(actual.state_dict())
    torch.testing.assert_close(restored(x=x)["output"], actual(x))
    assert list(actual.state_dict()) == ["nodes.n_hidden.weight", "nodes.n_hidden.bias",
                                         "nodes.n_head.weight", "nodes.n_head.bias"]


def test_lookup_slices_share_real_layers_without_duplicate_registration():
    model = _mlp(device="cpu")
    names = tuple(model.state_dict())
    assert model[0] is model["hidden"]
    assert model[-1] is model["head"]
    assert list(model) == [model[0], model[1], model[2]]
    assert len(model) == 3
    features = model[:2]
    assert isinstance(features, nn.Sequential)
    assert features[0] is model[0]
    assert features[1] is model[1]
    with torch.no_grad():
        features[0].weight.fill_(0.5)
    assert torch.all(model[0].weight == 0.5)
    features.eval()
    assert not model[0].training
    assert tuple(model.state_dict()) == names
    with pytest.raises(KeyError):
        model["missing"]
    with pytest.raises(IndexError):
        model[3]
    with pytest.raises(TypeError):
        model[0] = nn.Identity()
    with pytest.raises(TypeError):
        model.nodes["n_hidden"] = nn.Identity()
    with pytest.raises(TypeError):
        del model.nodes["n_hidden"]
    with pytest.raises(TypeError):
        model.nodes = nn.ModuleDict()


def test_inspection_does_not_run_layers_or_draw_rng():
    model = _mlp(device="cpu")
    def forbid(*args):
        raise AssertionError("repr must not execute a layer")
    handle = model[0].register_forward_hook(forbid)
    before = torch.get_rng_state().clone()
    text = repr(model)
    handle.remove()
    assert "hidden" in text and "[B, 4]" in text and "[B, 5]" in text and "[B, 2]" in text
    assert torch.equal(before, torch.get_rng_state())


def test_default_rng_matches_handwritten_and_explicit_seed_isolates():
    torch.manual_seed(701)
    expected = nn.Linear(4, 5)
    after_expected = torch.get_rng_state().clone()
    torch.manual_seed(701)
    actual = network("linear(5)", input_shape=("B", 4), output_shape=("B", 5), device="cpu")
    torch.testing.assert_close(actual[0].weight, expected.weight)
    assert torch.equal(torch.get_rng_state(), after_expected)
    before = torch.get_rng_state().clone()
    one = _mlp(device="cpu", initialization_seed=81)
    two = _mlp(device="cpu", initialization_seed=81)
    assert torch.equal(before, torch.get_rng_state())
    for p, q in zip(one.parameters(), two.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_explicit_cuda_seed_restores_cpu_and_every_cuda_generator():
    cpu = torch.get_rng_state().clone()
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
    a = _mlp(device="cuda:0", initialization_seed=67)
    b = _mlp(device="cuda:0", initialization_seed=67)
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(cuda, torch.cuda.get_rng_state_all()))
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize("device", DEVICES)
def test_split_branches_execute_once_and_both_paths_have_gradients(device):
    source = '''
    left, right = split(2, name="parts")
    a = linear(left, 3, name="left")
    b = linear(right, 3, name="right")
    add(a, b, name="join")
    '''
    model = network(source, input_shape=("B", 6), output_shape=("B", 3),
                    device=device, initialization_seed=12)
    counts = {name: 0 for name in ("parts", "left", "right", "join")}
    handles = []
    for name in counts:
        def count(module, args, result, name=name):
            counts[name] += 1
        handles.append(model[name].register_forward_hook(count))
    x = torch.randn(2, 6, device=device, requires_grad=True)
    expected = nn.functional.linear(x[:, :2], model["left"].weight, model["left"].bias)
    expected = expected + nn.functional.linear(x[:, 2:], model["right"].weight, model["right"].bias)
    actual = model(x)
    assert counts == dict.fromkeys(counts, 1)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert torch.count_nonzero(x.grad[:, :2]) and torch.count_nonzero(x.grad[:, 2:])
    assert model["left"].weight.grad is not None and model["right"].weight.grad is not None
    with pytest.raises(TypeError):
        model[0]
    with pytest.raises(TypeError):
        model[:2]
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("device", DEVICES)
def test_convolution_reshape_normalization_and_flatten(device):
    source = '''
    linear()
    reshape(2)
    deconv(4, policy="up2")
    group_norm(2)
    leaky_relu(0.1)
    conv(3, kernel_size=3, padding=1)
    tanh()
    '''
    model = network(source, input_shape=("B", 5), output_shape=("B", 3, 8, 8),
                    device=device, initialization_seed=4)
    x = torch.randn(2, 5, device=device, requires_grad=True)
    expected = model[0](x).reshape(2, 2, 4, 4)
    for layer in list(model)[2:]:
        expected = layer(expected)
    result = model(x)
    torch.testing.assert_close(result, expected)
    result.square().mean().backward()
    assert x.grad.isfinite().all()
    flattened = network("flatten(); linear()", input_shape=("B", 3, 8, 8),
                        output_shape=("B", 2), device=device)
    assert flattened(result.detach()).shape == (2, 2)


def test_concat_and_unused_split_port():
    model = network("a, b = split(2); concat(b, a)", input_shape=("B", 5),
                    output_shape=("B", 5), device="cpu")
    x = torch.arange(10, dtype=torch.float32).reshape(2, 5).requires_grad_()
    torch.testing.assert_close(model(x), torch.cat((x[:, 2:], x[:, :2]), dim=1))
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    remainder = network("a, b = split(2); out = b", input_shape=("B", 5),
                        output_shape=("B", 3), device="cpu")
    torch.testing.assert_close(remainder(x), x[:, 2:])


def test_runtime_input_contracts_and_empty_identity():
    model = _mlp(device="cpu")
    for bad in (torch.randn(3, 3), torch.randn(0, 4), torch.randn(3, 4, dtype=torch.float64), None):
        with pytest.raises(HNDLError, match="E_RUNTIME"):
            model(bad)
    graph = build(model.plan, device="cpu")
    with pytest.raises(HNDLError, match="E_BINDING"):
        graph(wrong=torch.randn(1, 4))
    identity = network("", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)
    assert identity(x) is x
    assert len(identity) == 0


def test_custom_exact_registration_bounds_and_runtime_contract():
    registry = Registry.builtins()
    registry.register("silu", identity="example.silu", version=1, shape=preserves_shape, max_state_bytes=0)
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        register_torch(registry, "silu", module=nn.SiLU, state_version=2)
    register_torch(registry, "silu", module=nn.SiLU, state_version=1)
    with pytest.raises(HNDLError):
        register_torch(registry, "silu", module=nn.SiLU, state_version=1)
    model = network("linear(); silu()", input_shape=("B", 4), output_shape=("B", 3),
                    device="cpu", registry=registry)
    x = torch.randn(2, 4, requires_grad=True)
    torch.testing.assert_close(model(x), nn.functional.silu(model[0](x)))
    model(x).sum().backward()
    assert x.grad.isfinite().all()
    assert isinstance(model[1], nn.SiLU)

    class HiddenBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("data", torch.ones(1), persistent=False)
        def forward(self, x):
            return x
    registry.register("bad", identity="test.bad", version=1, shape=preserves_shape, max_state_bytes=0)
    register_torch(registry, "bad", module=HiddenBuffer, state_version=1)
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        network("bad()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)

    class WrongShape(nn.Module):
        def forward(self, x):
            return x[:, :1]
    registry.register("wrong", identity="test.wrong", version=1, shape=preserves_shape, max_state_bytes=0)
    register_torch(registry, "wrong", module=WrongShape, state_version=1)
    broken = network("wrong()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        broken(torch.randn(2, 4))


def test_file_and_callable_capture_once(tmp_path):
    path = tmp_path / "network.hndl"
    path.write_text("linear(3)", encoding="utf-8")
    file_model = network_file(path, input_shape=("B", 4), output_shape=("B", 3),
                              device="cpu", initialization_seed=9)
    calls = []
    def author(x):
        calls.append(x)
        hndl.ops.linear(3)
    captured = network_from_callable(author, input_shape=("B", 4), output_shape=("B", 3),
                                     device="cpu", initialization_seed=9)
    x = torch.randn(2, 4)
    torch.testing.assert_close(captured(x), file_model(x))
    repr(captured)
    captured(x)
    assert len(calls) == 1
