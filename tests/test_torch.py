"""Numerical and integration checks for the PyTorch backend."""

import copy
import pickle
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from torch import nn

import hndl
from hndl import HNDLError, Registry
from hndl.torch import build, network, network_file, network_from_callable
from hndl.types import ResolvedPlan


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


def test_reserved_module_attribute_name_and_explicit_cpu_index():
    model = network('linear(3, name="training")', input_shape=("B", 4),
                    output_shape=("B", 3), device="cpu:0")
    assert model[:][0] is model["training"]
    assert model(torch.ones(2, 4)).shape == (2, 3)


def test_inspection_does_not_run_layers_or_draw_rng():
    model = _mlp(device="cpu")
    def forbid(*args):
        raise AssertionError("repr must not execute a layer")
    handle = model[0].register_forward_hook(forbid)
    before = torch.get_rng_state().clone()
    text = repr(model)
    handle.remove()
    assert "hidden" in text and "[B, 4]" in text and "[B, 5]" in text and "[B, 2]" in text
    assert "input shape" in text and "output shape" in text and "linear@" not in text
    assert "input:x" not in text
    header, first, _, last = text.splitlines()[1:]
    assert first.index("hidden") == last.index("head") == header.index("name")
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
    assert "first=[B, 2]" in repr(model) and "rest=[B, 4]" in repr(model)
    assert "node:parts/rest" in repr(model)
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


def test_saved_concat_uses_numeric_port_order():
    model = network("a, b = split(1); concat(a,b,a,b,a,b,a,b,a,b,a,b)",
                    input_shape=("B", 3), output_shape=("B", 18), device="cpu")
    restored = build(ResolvedPlan.from_json(model.plan.to_json()), device="cpu")
    x = torch.tensor([[1., 2., 3.]])
    torch.testing.assert_close(restored(x=x)["output"], x.repeat(1, 6))


def test_state_reload_in_fresh_process(tmp_path):
    model = _mlp(device="cpu", initialization_seed=713)
    (tmp_path / "plan.json").write_text(model.plan.to_json())
    torch.save(model.state_dict(), tmp_path / "state.pt")
    script = '''
import pathlib, sys, torch
from hndl.types import ResolvedPlan
from hndl.torch import build
path = pathlib.Path(sys.argv[1])
plan = ResolvedPlan.from_json((path / "plan.json").read_text())
model = build(plan, device="cpu")
model.load_state_dict(torch.load(path / "state.pt", weights_only=True))
torch.save(model(x=torch.ones(2, 4))["output"], path / "output.pt")
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=30)
    torch.testing.assert_close(torch.load(tmp_path / "output.pt", weights_only=True),
                               model(torch.ones(2, 4)))


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


def test_direct_plan_tampering_rejected_before_module_allocation(monkeypatch):
    plan = hndl.resolve("linear(3)", input_shape=("B", 4), output_shape=("B", 3))
    forged = replace(plan, nodes=(replace(plan.nodes[0], args={**plan.nodes[0].args, "in_features": 5}),))
    from hndl.operators import linear

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid plan must fail before constructing a module")

    monkeypatch.setattr(linear.Linear, "__init__", forbidden)
    with pytest.raises(HNDLError, match="E_INTEGRITY|E_CONSTRAINT"):
        build(forged, device="cpu")
    monkeypatch.undo()
    with pytest.raises(HNDLError, match="E_RESOURCE.*before allocation"):
        build(plan, device="cpu", limits={"max_state_bytes": 1})
    assert build(plan, device="cpu").build_receipt["state_bytes"] == (4 * 3 + 3) * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_runtime_device_mismatch_and_module_moves():
    model = _mlp(device="cuda:0")
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        model(torch.ones(2, 4))
    model.cpu()
    assert model(torch.ones(2, 4)).shape == (2, 2)
    identity = network("", input_shape=("B", 4), output_shape=("B", 4), device="cpu").cuda(0)
    x = torch.ones(2, 4, device="cuda:0")
    assert identity(x) is x


def test_custom_operator_builds_with_shared_shape_inference():
    registry = Registry.builtins()

    @registry.operator("my_silu", identity="example.silu", summary="SiLU.", shape="x[B, ...] -> out[B, ...]")
    class SiLU(nn.SiLU):
        pass

    model = network("linear(); my_silu()", input_shape=("B", 4), output_shape=("B", 3),
                    device="cpu", registry=registry)
    x = torch.randn(2, 4, requires_grad=True)
    torch.testing.assert_close(model(x), nn.functional.silu(model[0](x)))
    model(x).sum().backward()
    assert x.grad.isfinite().all()
    assert isinstance(model[1], nn.SiLU)
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        build(model.plan, device="cpu", registry=Registry.builtins())


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


def _stateful(**kwargs):
    """An MLP whose batch norm gives the graph buffers as well as parameters."""
    return network('linear(5, name="hidden"); batch_norm(name="norm"); relu(); linear(name="head")',
                   input_shape=("B", 4), output_shape=("B", 2), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_deepcopy_gives_independent_state_and_shares_the_plan(device):
    model = _stateful(device=device, initialization_seed=23)
    model.eval()
    model["hidden"].bias.requires_grad_(False)
    x = torch.randn(3, 4, device=device)
    model.train()(x)  # move the batch-norm buffers away from their defaults
    model.eval()

    before = torch.get_rng_state().clone()
    clone = copy.deepcopy(model)
    assert torch.equal(before, torch.get_rng_state())

    assert type(clone) is type(model) and clone is not model
    assert clone.plan is model.plan and clone.build_receipt is model.build_receipt
    assert clone.training is False and repr(clone) == repr(model)
    torch.testing.assert_close(clone(x), model(x))

    names = tuple(model.state_dict())
    assert names == tuple(clone.state_dict()) and "nodes.n_norm.running_mean" in names
    originals = dict(model.named_parameters()) | dict(model.named_buffers())
    copies = dict(clone.named_parameters()) | dict(clone.named_buffers())
    for name, original in originals.items():
        assert copies[name] is not original and copies[name].data_ptr() != original.data_ptr()
        assert copies[name].requires_grad == original.requires_grad
        torch.testing.assert_close(copies[name], original)
    assert originals["nodes.n_hidden.bias"].requires_grad is False
    assert originals["nodes.n_hidden.weight"].requires_grad is True

    expected = model(x)
    with torch.no_grad():
        for tensor in (*clone.parameters(), *clone.buffers()):
            tensor.add_(1)
    torch.testing.assert_close(model(x), expected)
    assert not torch.allclose(clone(x), expected)


def test_deepcopy_of_a_branched_graph_keeps_the_architecture_locked():
    source = '''
    left, right = split(2, name="parts")
    a = linear(left, 3, name="left")
    b = linear(right, 3, name="right")
    add(a, b, name="join")
    '''
    model = network(source, input_shape=("B", 6), output_shape=("B", 3),
                    device="cpu", initialization_seed=12)
    clone = copy.deepcopy(model)
    x = torch.randn(2, 6)
    torch.testing.assert_close(clone(x), model(x))
    assert clone["left"] is not model["left"] and clone["left"] is clone.nodes["n_left"]
    with pytest.raises(TypeError, match="architecture is fixed"):
        clone.nodes.add_module("n_extra", nn.Identity())
    with pytest.raises(TypeError, match="architecture is fixed"):
        del clone.nodes["n_left"]
    with pytest.raises(TypeError, match="architecture is fixed"):
        clone.plan = model.plan
    with pytest.raises(TypeError):
        clone[0]


def test_deepcopy_keeps_lookup_moves_and_the_state_consistency_check():
    model = _stateful(device="cpu")
    clone = copy.deepcopy(model)
    assert len(clone) == 4 and clone[0] is clone["hidden"] and clone[-1] is clone["head"]
    assert list(clone) == [clone[i] for i in range(4)]
    assert isinstance(clone[:2], nn.Sequential)
    clone.to("cpu")
    assert clone._runtime_device == torch.device("cpu")
    x = torch.randn(2, 4)
    clone.eval()(x)
    object.__setattr__(clone, "_state_keys", clone._state_keys[:-1] + ((("ghost",), (), ()),))
    with pytest.raises(HNDLError, match="E_RUNTIME.*registered state"):
        clone(x)


def test_state_registered_during_forward_is_rejected():
    """The real bug the integrity check exists for: state appearing mid-forward."""
    registry = Registry.builtins()

    @registry.operator("grows_state", identity="tests.grows_state",
                       summary="Register a parameter during forward.",
                       shape="data[B, F] -> value[B, F]")
    class GrowsState(nn.Module):
        def forward(self, data):
            self.register_parameter("extra", nn.Parameter(torch.zeros(1)))
            return data

    model = network("grows_state()", input_shape=("B", 4), output_shape=("B", 4),
                    device="cpu", registry=registry)
    with pytest.raises(HNDLError, match="E_RUNTIME.*registered state"):
        model(torch.randn(2, 4))


def test_immutable_records_are_shared_and_modules_refuse_to_pickle(tmp_path):
    model = _mlp(device="cpu")
    plan = model.plan
    assert copy.deepcopy(plan) is plan and copy.copy(plan) is plan
    assert copy.deepcopy(plan.nodes[0]) is plan.nodes[0]
    operator = plan.registry.by_identity(plan.nodes[0].op)
    assert copy.deepcopy(operator) is operator
    assert copy.deepcopy({"plan": plan})["plan"] is plan

    shallow = copy.copy(model)
    assert shallow is not model and shallow.nodes is model.nodes and shallow.plan is model.plan
    torch.testing.assert_close(shallow(torch.ones(2, 4)), model(torch.ones(2, 4)))

    with pytest.raises(TypeError, match="cannot be pickled.*state_dict"):
        pickle.dumps(model)
    with pytest.raises(TypeError, match="cannot be pickled.*state_dict"):
        torch.save(model, tmp_path / "model.pt")


def _forward_or_skip(model, x):
    """Reduced precision is not implemented for every CPU kernel."""
    try:
        return model(x)
    except RuntimeError as error:  # pragma: no cover - host dependent
        pytest.skip(f"{x.dtype} is unsupported on this host: {error}")


@pytest.mark.parametrize("cast, dtype", [
    (lambda m: m.double(), torch.float64),
    (lambda m: m.half(), torch.float16),
    (lambda m: m.to(torch.bfloat16), torch.bfloat16),
    (lambda m: m.to(dtype=torch.float64), torch.float64),
])
def test_floating_point_casts_move_the_runtime_dtype_checks(cast, dtype):
    model = cast(_mlp(device="cpu"))
    assert all(p.dtype == dtype for p in model.parameters())
    result = _forward_or_skip(model, torch.randn(3, 4, dtype=dtype))
    assert result.dtype == dtype and result.shape == (3, 2)
    with pytest.raises(HNDLError, match=f"E_RUNTIME.*expected dtype {dtype}, got torch.float32"):
        model(torch.randn(3, 4))
    restored = model.float()
    assert restored is model
    assert model(torch.randn(3, 4)).dtype == torch.float32
    with pytest.raises(HNDLError, match="E_RUNTIME.*expected dtype torch.float32, got torch.float64"):
        model(torch.randn(3, 4, dtype=torch.float64))


def test_double_tracks_parameterless_graphs_and_device_only_moves():
    model = network("relu()", input_shape=("B", 4), output_shape=("B", 4), device="cpu").double()
    assert model(torch.randn(2, 4, dtype=torch.float64)).dtype == torch.float64
    with pytest.raises(HNDLError, match="E_RUNTIME.*expected dtype torch.float64"):
        model(torch.randn(2, 4))

    unmoved = _mlp(device="cpu").to("cpu")
    assert unmoved(torch.randn(2, 4)).dtype == torch.float32
    with pytest.raises(HNDLError, match="E_RUNTIME.*expected dtype torch.float32"):
        unmoved(torch.randn(2, 4, dtype=torch.float64))


def test_casts_leave_integer_and_declared_dtypes_alone():
    model = network('embedding(20, 6); linear(3, name="head")', input_shape=("B", 5),
                    output_shape=("B", 5, 3), input_dtype="int64", device="cpu").double()
    tokens = torch.randint(0, 20, (2, 5))
    assert model(tokens).dtype == torch.float64
    with pytest.raises(HNDLError, match="E_RUNTIME.*expected dtype torch.int64, got torch.int32"):
        model(tokens.to(torch.int32))
    with pytest.raises(HNDLError, match="E_RUNTIME.*expected dtype torch.int64, got torch.float64"):
        model(tokens.to(torch.float64))


def test_deepcopy_carries_and_isolates_the_runtime_dtype():
    model = _mlp(device="cpu")
    clone = copy.deepcopy(model.double())
    assert clone(torch.randn(2, 4, dtype=torch.float64)).dtype == torch.float64
    with pytest.raises(HNDLError, match="E_RUNTIME.*expected dtype torch.float64"):
        clone(torch.randn(2, 4))

    original = _mlp(device="cpu")
    copy.deepcopy(original).double()
    assert original(torch.randn(2, 4)).dtype == torch.float32
    assert original._input_dtypes["x"] == torch.float32 and model._input_dtypes["x"] == torch.float32
