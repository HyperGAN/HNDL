"""Prepared execution preserves PyTorch behavior and bounds replay ownership."""

import copy

import pytest
import torch
from torch import nn

from hndl import Registry, resolve
from hndl.errors import HNDLError
from hndl.torch import build, network


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("training", [True, False])
def test_generated_forward_backward_and_optimizer_share_parameters(device, training):
    source = network("linear(16); gelu(); linear(4)", input_shape=("B", 8),
                     output_shape=("B", 4), device=device)
    reference = copy.deepcopy(source)
    source.train(training)
    reference.train(training)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.05)
    prepared = source.prepare()
    assert list(prepared.parameters()) == list(source.parameters())
    assert prepared.state_dict().keys() == source.state_dict().keys()
    for _ in range(2):
        x = torch.randn(3, 8, device=device, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual, expected = prepared(x), reference(xr)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for p, pr in zip(prepared.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, pr.grad)
        optimizer.step()
        with torch.no_grad():
            for p in reference.parameters():
                p.add_(p.grad, alpha=-0.05)
        optimizer.zero_grad()
        reference.zero_grad()


def test_branches_named_bindings_multi_outputs_and_hooks():
    model = network("a, b = split(3); a = linear(a, 4); b = linear(b, 4); out = add(a, b); left = a",
                    input_shape=("B", 6), output_shape={"out": ("B", 4), "left": ("B", 4)},
                    device="cpu")
    calls = []
    handle = model["n1"].register_forward_hook(lambda *args: calls.append(1))
    prepared = model.prepare()
    x = torch.randn(2, 6)
    for _ in range(2):
        result = prepared(x=x)
        expected = model(x=x)
        assert result.keys() == expected.keys()
        for key in result:
            torch.testing.assert_close(result[key], expected[key])
    assert len(calls) == 4
    handle.remove()
    # Generated locals disappear after their last use, including split branches.
    assert "del " in prepared._export_module.source_code


def test_custom_dictionary_results_and_nested_checkpoint_roundtrip():
    registry = Registry.builtins()

    @registry.operator("mapping", identity="test.mapping", shape="x[B, D] -> out[B, D]",
                       summary="Return one named port.")
    class Mapping(nn.Module):
        def forward(self, x):
            return {"out": x * 2}

    model = network("mapping(); linear(3)", input_shape=("B", 4), output_shape=("B", 3),
                    registry=registry, device="cpu")
    prepared = model.prepare()
    x = torch.randn(2, 4)
    torch.testing.assert_close(prepared(x), model(x))
    torch.testing.assert_close(prepared(x), model(x))
    parent = nn.Sequential(prepared)
    state = copy.deepcopy(parent.state_dict())
    with torch.no_grad():
        model[-1].weight.zero_()
    parent.load_state_dict(state)
    torch.testing.assert_close(parent(x), model(x))
    assert parent.state_dict().keys() == nn.Sequential(model).state_dict().keys()


def test_graph_facade_return_conventions_and_example_bindings():
    plan = resolve("linear(3)", input_shape={"features": ("B", 4)}, output_shape=("B", 3))
    graph = build(plan, device="cpu")
    x = torch.randn(2, 4)
    prepared = graph.prepare({"features": x})
    assert isinstance(prepared(features=x), dict)
    assert isinstance(prepared(x), torch.Tensor)
    with pytest.raises(HNDLError, match="external tensor inputs"):
        prepared(typo=x)


def test_cast_rebind_and_state_replacement_invalidate_preparation():
    model = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    prepared = model.prepare()
    x = torch.randn(2, 4)
    prepared(x)
    prepared.double()
    assert model._runtime_dtype == torch.float64
    torch.testing.assert_close(prepared(x.double()), model(x.double()))
    replacement = nn.Linear(4, 4, dtype=torch.float64)
    model.nodes._modules["n_n0"] = replacement
    model.rebind_state()
    torch.testing.assert_close(prepared(x.double()), replacement(x.double()))
    with pytest.raises(HNDLError):
        prepared(torch.randn(2, 5, dtype=torch.float64))


def test_checkpointed_generated_training_preserves_gradients():
    plan = resolve("linear(8, name='hidden'); silu(); linear(4)",
                   input_shape=("B", 4), output_shape=("B", 4))
    source = build(plan, checkpoint=["hidden"], device="cpu")
    prepared = source.prepare()
    for _ in range(2):
        x = torch.randn(2, 4, requires_grad=True)
        prepared(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_input_changes_output_ownership_lru_and_weight_updates():
    model = network("linear(16); relu(); linear(4)", input_shape=("B", 8),
                    output_shape=("B", 4), device="cuda:0").eval()
    with torch.inference_mode():
        prepared = model.prepare(backend="cuda_graph", max_graphs=2)
        x = torch.randn(1, 8, device="cuda:0")
        y = prepared(x)
        snapshot = y.clone()
        x2 = x + 1
        torch.testing.assert_close(prepared(x2), model(x2))
        torch.testing.assert_close(y, snapshot)
        for batch in (2, 3):
            z = torch.randn(batch, 8, device="cuda:0")
            torch.testing.assert_close(prepared(z), model(z))
        assert prepared.cache_info["graphs"] == 2
        assert 0 < prepared.cache_info["bytes"] <= prepared.max_graph_bytes
        model[0].weight.add_(0.1)
        torch.testing.assert_close(prepared(x2), model(x2))
        # Assigning a new Parameter updates captured addresses on the next call.
        model[0].weight = nn.Parameter(model[0].weight.detach().clone())
        torch.testing.assert_close(prepared(x2), model(x2))
        assert prepared.cache_info["graphs"] == 1
        prepared.clear_cache()
        assert prepared.cache_info["bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_borrowed_outputs_and_memory_limit():
    model = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cuda:0").eval()
    x = torch.randn(2, 4, device="cuda:0")
    with torch.no_grad():
        prepared = model.prepare(backend="cuda_graph", clone_outputs=False)
        first = prepared(x)
        second = prepared(x + 1)
        assert first.data_ptr() == second.data_ptr()
        torch.testing.assert_close(second, model(x + 1))
        with pytest.raises(HNDLError, match="max_graph_bytes"):
            model.prepare(backend="cuda_graph", max_graph_bytes=1)(x)
    with pytest.raises(HNDLError, match="no_grad"):
        prepared(x)


def test_generated_fullgraph_compilation():
    model = network("linear(8); silu(); linear(4)", input_shape=("B", 4),
                    output_shape=("B", 4), device="cpu").eval()
    prepared = model.prepare()
    compiled = torch.compile(prepared, backend="eager", fullgraph=True)
    x = torch.randn(2, 4)
    torch.testing.assert_close(compiled(x), model(x))


def test_auto_training_executes_once_and_preserves_random_state():
    model = network("dropout(0.5); linear(4)", input_shape=("B", 4),
                    output_shape=("B", 4), device="cpu")
    prepared = model.prepare(backend="auto")
    x = torch.ones(3, 4)
    torch.manual_seed(17)
    expected = model(x)
    torch.manual_seed(17)
    actual = prepared(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model[-1].weight.grad is not None
    assert prepared.cache_info["graphs"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_cuda_tuning_fallback_and_gradient_mode_changes():
    model = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4),
                    device="cuda:0").eval()
    x = torch.randn(2, 4, device="cuda:0")
    prepared = model.prepare(backend="auto")
    with torch.inference_mode():
        torch.testing.assert_close(prepared(x), model(x))
        torch.testing.assert_close(prepared(x + 1), model(x + 1))
    # An eval model can still need gradients: never reuse its inference graph.
    actual = prepared(x.requires_grad_())
    actual.sum().backward()
    assert x.grad is not None
    with torch.no_grad():
        fallback = model.prepare(backend="auto", max_graph_bytes=1)
        torch.testing.assert_close(fallback(x.detach()), model(x.detach()))
        assert fallback.cache_info["graphs"] == 0


def test_compiled_preparation_tracks_rebind_and_external_training_mode():
    source = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    prepared = source.prepare()
    compiled = torch.compile(prepared, backend="eager", fullgraph=True)
    x = torch.randn(2, 4)
    compiled(x)
    replacement = nn.Linear(4, 4)
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.bias.fill_(3)
    source.nodes._modules["n_n0"] = replacement
    source.rebind_state()
    torch.testing.assert_close(compiled(x), source(x))
    source.eval()
    assert prepared._export_module.training is False
    assert prepared.training is False
    torch.testing.assert_close(compiled(x), source(x))


def test_deepcopy_preparation_has_independent_parameters_and_no_rng_draw():
    model = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    prepared = model.prepare()
    state = torch.random.get_rng_state().clone()
    cloned = copy.deepcopy(prepared)
    assert torch.equal(state, torch.random.get_rng_state())
    assert cloned.plan is prepared.plan
    assert len(model._prepared_dependents) == 1
    assert len(cloned.source._prepared_dependents) == 1
    x = torch.randn(2, 4)
    torch.testing.assert_close(cloned(x), prepared(x))
    with torch.no_grad():
        cloned[0].weight.zero_()
    assert cloned[0].weight.data_ptr() != prepared[0].weight.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_accounts_for_reserved_intermediate_workspace():
    source = network("linear(16384); gelu(); linear(1)", input_shape=("B", 64),
                     output_shape=("B", 1), device="cuda:0").eval()
    x = torch.randn(1024, 64, device="cuda:0")
    with torch.inference_mode():
        constrained = source.prepare(backend="cuda_graph", max_graph_bytes=16 * 1024**2)
        with pytest.raises(HNDLError, match="max_graph_bytes"):
            constrained(x)
        assert constrained.cache_info["graphs"] == 0
        ample = source.prepare(backend="cuda_graph", max_graph_bytes=256 * 1024**2)
        torch.testing.assert_close(ample(x), source(x))
        assert ample.cache_info["bytes"] >= 64 * 1024**2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_serializes_shared_buffers_across_streams():
    source = network("linear(32); silu(); linear(4)", input_shape=("B", 8),
                     output_shape=("B", 4), device="cuda:0").eval()
    x1 = torch.randn(16, 8, device="cuda:0")
    x2 = torch.randn_like(x1)
    with torch.inference_mode():
        prepared = source.prepare((x1,), backend="cuda_graph")
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        first.wait_stream(torch.cuda.current_stream())
        second.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(first):
            torch.cuda._sleep(3_000_000)
            y1 = prepared(x1)
        with torch.cuda.stream(second):
            y2 = prepared(x2)
        prepared.clear_cache()  # Also waits for in-flight replay and output clones.
        torch.testing.assert_close(y1, source(x1))
        torch.testing.assert_close(y2, source(x2))


def test_packed_preparation_optimizer_functional_cast_and_deepcopy():
    from hndl.operators._packing import clear_packed

    source = network("swiglu(32); linear(4)", input_shape=("B", 8),
                     output_shape=("B", 4), device="cpu")
    reference = copy.deepcopy(source)
    before = list(source.parameters())
    optimizer = torch.optim.SGD(before, lr=0.01)
    fast = source.prepare(pack=True)
    assert all(a is b for a, b in zip(before, fast.parameters()))
    for _ in range(2):
        x = torch.randn(3, 8, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual, expected = fast(x), reference(xr)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for p, pr in zip(source.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, pr.grad)
        optimizer.step()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(parameter.grad, alpha=-0.01)
        optimizer.zero_grad()
        reference.zero_grad()
    replacements = {name: value * 1.1 for name, value in source.named_parameters()}
    torch.testing.assert_close(torch.func.functional_call(fast, replacements, (x,)),
                               torch.func.functional_call(reference, replacements, (x,)))
    fast.double()
    reference.double()
    torch.testing.assert_close(fast(x.double()), reference(x.double()))
    cloned = copy.deepcopy(fast)
    clear_packed(reference)
    torch.testing.assert_close(cloned(x.double()), reference(x.double()))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(cloned.parameters(), fast.parameters()))


def test_compiled_named_outputs_use_plain_dictionaries():
    plan = resolve("left = linear(x,4); right = relu(left)", input_shape=("B", 4),
                   output_shape={"left": ("B", 4), "right": ("B", 4)})
    graph = build(plan, device="cpu")
    compiled = torch.compile(graph.prepare(), backend="eager", fullgraph=True)
    x = torch.randn(2, 4)
    actual = compiled(x=x)
    assert type(actual) is dict
    expected = graph(x=x)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_root_hooks_keep_original_module_identity_and_execute_once():
    source = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    calls = []

    def before(module, inputs):
        assert module is source
        calls.append("before")
        return (inputs[0] + 2,)

    def after(module, inputs, result):
        assert module is source
        calls.append("after")
        return result * 3

    source.register_forward_pre_hook(before)
    source.register_forward_hook(after)
    prepared = source.prepare()
    x = torch.randn(2, 4)
    expected = source(x)
    calls.clear()
    for _ in range(2):
        torch.testing.assert_close(prepared(x), expected)
    assert calls == ["before", "after", "before", "after"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_replay_pins_rotary_tables_after_cache_growth_and_eviction():
    source = network("attention(4,rope=True,causal=True)", input_shape=("B", 8, 64),
                     output_shape=("B", 8, 64), device="cuda:0").eval()
    # Different valid sequence sizes are normally supplied through separate
    # resolved contracts. A second model sharing the layer grows its table.
    other = network("attention(4,rope=True,causal=True)", input_shape=("B", 16, 64),
                    output_shape=("B", 16, 64), device="cuda:0").eval()
    other.nodes._modules["n_n0"] = source[0]
    other.rebind_state()
    x = torch.randn(2, 8, 64, device="cuda:0")
    with torch.inference_mode():
        expected = source(x)
        prepared = source.prepare(backend="cuda_graph")
        torch.testing.assert_close(prepared(x), expected)
        pointers = {t.untyped_storage().data_ptr() for tables in source[0]._rope_cache.values() for t in tables}
        other(torch.randn(2, 16, 64, device="cuda:0"))
        source[0]._rope_cache.clear()
        entry = next(iter(prepared._graphs.values()))
        retained = {t.untyped_storage().data_ptr() for t in entry[5]}
        assert pointers <= retained
        scratch = torch.full((2, 8, 64), 1e6, device="cuda:0")
        torch.testing.assert_close(prepared(x), expected)
        assert scratch.isfinite().all()
