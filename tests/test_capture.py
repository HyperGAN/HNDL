from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from torch import nn

from hndl import HNDLError, Registry, ops
from hndl.capture import capture_callable


def capture(fn, **kwargs):
    return capture_callable(fn, input_shape=("B", 128), output_shape=("B", 10), **kwargs)


def test_callable_once_current_sequence_and_empty_identity():
    calls = []

    def author(x):
        calls.append(x)
        ops.linear(64)
        ops.relu()
        ops.linear()

    graph = capture(author)
    assert len(calls) == 1
    assert [node.id for node in graph.nodes] == ["n0", "n1", "n2"]
    assert [node.inputs["x"] for node in graph.nodes] == ["input:x", "node:n0/out", "node:n1/out"]
    assert graph.output_ref == "node:n2/out"
    assert capture(lambda x: None).output_ref == "input:x"


def test_names_are_explicit_and_keyword_tensor_binding_is_supported():
    def author(x):
        saved = ops.linear(64, x=x, name="hidden")
        return ops.linear(saved, name="head")

    graph = capture(author)
    assert [node.id for node in graph.nodes] == ["hidden", "head"]
    assert graph.nodes[0].args["out_features"] == 64
    assert graph.nodes[1].inputs["x"] == "node:hidden/out"


def test_split_clears_current_and_helpers_restart_explicit_branches():
    def block(x):
        ops.linear(x, 10)
        return ops.relu()

    def author(x):
        first, rest = ops.split(64)
        return ops.add(block(first), block(rest))

    graph = capture(author)
    assert graph.nodes[1].inputs["x"] == "node:n0/first"
    assert graph.nodes[3].inputs["x"] == "node:n0/rest"
    assert dict(graph.nodes[-1].inputs) == {"a": "node:n2/out", "b": "node:n4/out"}

    def missing_current(x):
        first, rest = ops.split(64)
        del first, rest
        ops.linear(10)

    with pytest.raises(HNDLError, match="E_CURRENT"):
        capture(missing_current)
    with pytest.raises(HNDLError, match="E_CURRENT"):
        capture(lambda x: ops.split(64) and None)


def test_nested_arguments_bind_current_at_invocation():
    graph = capture(lambda x: ops.add(ops.linear(10), ops.linear(10)))
    assert graph.nodes[0].inputs["x"] == "input:x"
    assert graph.nodes[1].inputs["x"] == "node:n0/out"
    assert dict(graph.nodes[2].inputs) == {"a": "node:n0/out", "b": "node:n1/out"}


@pytest.mark.parametrize("fn,code", [
    (lambda x: ops.linear(x, x=x), "E_BINDING"),
    (lambda x: ops.add(x), "E_BINDING"),
    (lambda x: ops.add(), "E_BINDING"),
    (lambda x: ops.concat(x, x, input_count=2.0), "E_BINDING"),
    (lambda x: (x, x), "E_OUTPUT"),
    (lambda x: 3, "E_OUTPUT"),
    (lambda x: False, "E_OUTPUT"),
    (lambda x: bool(x), "E_CAPTURE"),
    (lambda x: int(x), "E_CAPTURE"),
    (lambda x: tuple(x), "E_CAPTURE"),
    (lambda x: x == x, "E_CAPTURE"),
    (lambda x: x + 1, "E_CAPTURE"),
    (lambda x: ops.linear(True), "E_ARGUMENT"),
    (lambda x: ops.relu(name=None), "E_NAME"),
    (lambda x: ops.relu(policy=None), "E_ARGUMENT"),
])
def test_invalid_capture_inputs_fail_without_fallback(fn, code):
    with pytest.raises(HNDLError, match=code):
        capture(fn)


def test_exception_cleanup_no_retry_and_nested_context_restoration():
    calls = []

    def broken(x):
        calls.append(1)
        ops.linear(64)
        raise RuntimeError("author failed")

    with pytest.raises(RuntimeError, match="author failed"):
        capture(broken)
    assert calls == [1]
    with pytest.raises(HNDLError, match="E_CAPTURE"):
        ops.relu()

    def outer(x):
        ops.linear(64)
        inner = capture(lambda y: ops.linear(32))
        assert inner.nodes[0].inputs["x"] == "input:x"
        return ops.relu()

    graph = capture(outer)
    assert len(graph.nodes) == 2
    assert graph.nodes[1].inputs["x"] == "node:n0/out"


def test_cross_capture_symbols_rejected_for_ports_and_outputs():
    saved = []

    def first(x):
        saved.append(x)
        return x

    capture(first)
    with pytest.raises(HNDLError, match="E_CAPTURE"):
        capture(lambda x: ops.relu(saved[0]))
    with pytest.raises(HNDLError, match="E_CAPTURE"):
        capture(lambda x: saved[0])


def test_concurrent_captures_keep_independent_current_tensors():
    barrier = Barrier(4)

    def run(width):
        def author(x):
            ops.linear(width)
            barrier.wait(timeout=5)
            return ops.relu()
        return capture(author)

    with ThreadPoolExecutor(max_workers=4) as executor:
        graphs = list(executor.map(run, [16, 32, 64, 128]))
    assert [graph.nodes[0].args["out_features"] for graph in graphs] == [16, 32, 64, 128]
    assert all(graph.nodes[1].inputs["x"] == "node:n0/out" for graph in graphs)


def test_registry_factories_require_selected_exact_identity():
    registry = Registry.builtins()

    @registry.operator("silu", identity="example.silu", summary="SiLU.", shape="x[B, ...] -> out[B, ...]")
    class SiLU(nn.SiLU):
        pass

    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        capture(lambda x: registry.ops.silu())
    graph = capture(lambda x: registry.ops.silu(), registry=registry)
    assert graph.nodes[0].op == "example.silu@1"


def test_variadic_concat_and_no_silent_missing_ports():
    def author(x):
        a, b = ops.split(64)
        return ops.concat(a, b, axis=1)

    graph = capture(author)
    assert dict(graph.nodes[-1].inputs) == {"x0": "node:n0/first", "x1": "node:n0/rest"}
    assert graph.nodes[-1].args["input_count"] == 2
    with pytest.raises(HNDLError, match="E_BINDING"):
        capture(lambda x: ops.concat(x0=x, x2=x))


def test_node_limits_and_generated_name_collisions():
    def author(x):
        ops.relu(name="n1")
        return ops.relu()

    with pytest.raises(HNDLError, match="E_NAME"):
        capture(author)
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture(lambda x: ops.relu(ops.relu()), limits={"max_nodes": 1})


def test_argument_origin_metadata_distinguishes_policy_and_defaults():
    graph = capture(lambda x: ops.deconv(3, policy="up2"))
    source = graph.nodes[0].source
    assert source["argument_origins"]["out_channels"] == "explicit"
    assert source["argument_origins"]["stride"] == "policy-selected"
    assert source["argument_origins"]["bias"] == "operator default"
    assert source["policy"] == "spatial.up2_transpose@1"
