"""Custom contracts must behave identically in both torch-free frontends."""

import subprocess
import sys

import pytest

from hndl import Argument, Dim, HNDLError, Registry, ShapeRule, resolve, resolve_callable
from hndl.capture import capture_callable
from hndl.config import capture_config


@pytest.fixture
def registry():
    result = Registry.builtins()
    result.register(
        "tagged", identity="tests.tagged", version=1, max_state_bytes=0,
        shape=ShapeRule(inputs={"data": ("B", "F")}, outputs={"value": ("B", "F")}),
        arguments={
            "count": Argument(int, minimum=1),
            "gain": Argument(float, default=1.0, minimum=0.0, exclusive_minimum=True),
            "enabled": Argument(bool, default=True),
            "label": Argument(str, default="a # literal"),
        },
    )
    result.register(
        "pair", identity="tests.pair", version=1, max_state_bytes=0,
        shape=ShapeRule(inputs={"data": ("B", "F")},
                        outputs={"original": ("B", "F"), "copy": ("B", "F")}),
    )
    result.register(
        "blend", identity="tests.blend", version=1, max_state_bytes=0,
        shape=ShapeRule(inputs={"left": ("B", "F"), "right": ("B", "F")},
                        outputs={"mixed": ("B", "F")}),
        arguments={"gain": Argument(float, default=1.0, minimum=0.0, exclusive_minimum=True)},
    )
    result.register(
        "adaptive_norm", identity="tests.adaptive_norm", version=1, max_state_bytes=0,
        shape=ShapeRule(
            inputs={"features": ("B", "C", "H", "W"), "params": ("B", Dim("C", scale=2))},
            outputs={"normalized": ("B", "C", "H", "W")},
        ),
        arguments={"eps": Argument(float, default=1e-5, minimum=0.0, exclusive_minimum=True)},
    )
    return result


def resolve_pair(source, author, registry, input_shape=("B", 64), output_shape=("B", 64)):
    kwargs = dict(registry=registry, input_shape=input_shape, output_shape=output_shape)
    configuration = resolve(source, **kwargs)
    native = resolve_callable(author, **kwargs)
    assert configuration.semantic_digest == native.semantic_digest
    return configuration, native


def test_scalar_positionals_follow_optional_named_tensor_port(registry):
    source = '''
    # The first value is a scalar argument; data is implicit current.
    tagged(2, 0.5, False, "text # not a comment", name="tag")
    relu()
    '''

    def author(x):
        registry.ops.tagged(2, 0.5, False, "text # not a comment", name="tag")
        registry.ops.relu()

    plan, _ = resolve_pair(source, author, registry)
    assert dict(plan.nodes[0].args) == {
        "count": 2, "gain": 0.5, "enabled": False, "label": "text # not a comment",
    }
    assert plan.nodes[0].inputs["data"] == "input:x"
    assert plan.nodes[1].inputs["x"] == "node:tag/value"
    assert set(plan.nodes[0].source["argument_origins"].values()) == {"explicit"}


def test_keyword_tensor_port_and_scalar_defaults_have_correct_origins(registry):
    plan, _ = resolve_pair(
        "tagged(3, data=x)", lambda x: registry.ops.tagged(3, data=x), registry,
    )
    origins = plan.nodes[0].source["argument_origins"]
    assert origins["count"] == "explicit"
    assert origins["gain"] == origins["enabled"] == origins["label"] == "operator default"
    assert "data" not in plan.nodes[0].args


@pytest.mark.parametrize("binding", ["positional", "keyword", "scalar_first"])
def test_multi_input_operator_scalar_binding_and_current_update(registry, binding):
    forms = {
        "positional": "blend(a, b, 0.25)",
        "keyword": "blend(a, right=b, gain=0.25)",
        "scalar_first": "blend(0.25, left=a, right=b)",
    }
    source = "a, b = split(32)\n" + forms[binding] + "\nrelu()"

    def author(x):
        a, b = registry.ops.split(32)
        if binding == "positional":
            registry.ops.blend(a, b, 0.25)
        elif binding == "keyword":
            registry.ops.blend(a, right=b, gain=0.25)
        else:
            registry.ops.blend(0.25, left=a, right=b)
        registry.ops.relu()

    plan, _ = resolve_pair(source, author, registry, output_shape=("B", 32))
    assert dict(plan.nodes[1].inputs) == {"left": "node:n0/first", "right": "node:n0/rest"}
    assert plan.nodes[1].args["gain"] == 0.25
    assert plan.nodes[2].inputs["x"] == "node:n1/mixed"


def test_custom_multi_output_order_tuple_storage_and_explicit_selection(registry):
    source = "parts = pair(name='copies')\n[a, b] = parts\nout = b"

    def author(x):
        parts = registry.ops.pair(name="copies")
        a, b = parts
        return b

    plan, _ = resolve_pair(source, author, registry)
    assert plan.nodes[0].outputs == ("original", "copy")
    assert plan.output_ref == "node:copies/copy"
    assert set(plan.nodes[0].output_shapes) == {"original", "copy"}


def test_multi_output_clears_current_until_explicit_branch_and_join(registry):
    source = "a, b = pair()\ntagged(a, 2)\nleft = relu()\nblend(left, b)"

    def author(x):
        a, b = registry.ops.pair()
        registry.ops.tagged(a, 2)
        left = registry.ops.relu()
        registry.ops.blend(left, b)

    plan, _ = resolve_pair(source, author, registry)
    assert plan.nodes[1].inputs["data"] == "node:n0/original"
    assert plan.nodes[-1].inputs["right"] == "node:n0/copy"


def test_nested_custom_calls_resolve_current_in_python_order(registry):
    plan, _ = resolve_pair(
        "blend(tagged(1), tagged(2))",
        lambda x: registry.ops.blend(registry.ops.tagged(1), registry.ops.tagged(2)),
        registry,
    )
    assert plan.nodes[1].inputs["data"] == "node:n0/value"
    assert dict(plan.nodes[2].inputs) == {"left": "node:n0/value", "right": "node:n1/value"}


def test_adaptive_rule_drives_both_frontends_backward_across_branches(registry):
    source = '''
    content, style = split(64)
    linear(content)
    features = reshape(32, 4, 4)
    params = linear(style)
    adaptive_norm(params=params, features=features, eps=0.001, name="norm")
    '''

    def author(x):
        content, style = registry.ops.split(64)
        registry.ops.linear(content)
        features = registry.ops.reshape(32, 4, 4)
        params = registry.ops.linear(style)
        registry.ops.adaptive_norm(params=params, features=features, eps=0.001, name="norm")

    plan, _ = resolve_pair(source, author, registry,
                           input_shape=("B", 128), output_shape=("B", 32, 4, 4))
    assert plan.nodes[1].args["out_features"] == 512
    assert plan.nodes[3].args["out_features"] == 64
    assert plan.output_ref == "node:norm/normalized"
    assert plan.nodes[-1].output_shapes["normalized"] == ("B", 32, 4, 4)


@pytest.mark.parametrize("source,author,code", [
    ("tagged()", lambda ops, x: ops.tagged(), "E_ARGUMENT"),
    ("tagged(True)", lambda ops, x: ops.tagged(True), "E_ARGUMENT"),
    ("tagged(2.0)", lambda ops, x: ops.tagged(2.0), "E_ARGUMENT"),
    ("tagged(1, gain=0)", lambda ops, x: ops.tagged(1, gain=0), "E_ARGUMENT"),
    ("tagged(1, gain=True)", lambda ops, x: ops.tagged(1, gain=True), "E_ARGUMENT"),
    ("tagged(1, enabled=1)", lambda ops, x: ops.tagged(1, enabled=1), "E_ARGUMENT"),
    ("tagged(1, label=3)", lambda ops, x: ops.tagged(1, label=3), "E_ARGUMENT"),
    ("tagged(x, 1, data=x)", lambda ops, x: ops.tagged(x, 1, data=x), "E_BINDING"),
    ("tagged(1, count=2)", lambda ops, x: ops.tagged(1, count=2), "E_ARGUMENT"),
    ("blend(x)", lambda ops, x: ops.blend(x), "E_BINDING"),
    ("blend()", lambda ops, x: ops.blend(), "E_BINDING"),
    ("blend(x, x, left=x)", lambda ops, x: ops.blend(x, x, left=x), "E_BINDING"),
    ("blend(x, x, nope=1)", lambda ops, x: ops.blend(x, x, nope=1), "E_ARGUMENT"),
])
def test_custom_port_and_scalar_errors_are_consistent(registry, source, author, code):
    kwargs = dict(registry=registry, input_shape=("B", 64), output_shape=("B", 64))
    with pytest.raises(HNDLError, match=code):
        resolve(source, **kwargs)
    with pytest.raises(HNDLError, match=code):
        resolve_callable(lambda x: author(registry.ops, x), **kwargs)


@pytest.mark.parametrize("source", ["pair()", "a, b = pair()\ntagged(1)"])
def test_custom_multi_output_requires_a_new_current_or_selected_output(registry, source):
    kwargs = dict(registry=registry, input_shape=("B", 64), output_shape=("B", 64))
    with pytest.raises(HNDLError, match="E_CURRENT"):
        resolve(source, **kwargs)

    def author(x):
        registry.ops.pair()
        if source != "pair()":
            registry.ops.tagged(1)

    with pytest.raises(HNDLError, match="E_CURRENT"):
        resolve_callable(author, **kwargs)


def test_custom_multi_output_arity_is_not_inferred_from_assignment(registry):
    with pytest.raises(HNDLError, match="E_OUTPUT_ARITY"):
        resolve("a, b, c = pair()", registry=registry,
                input_shape=("B", 64), output_shape=("B", 64))


def test_schema_failure_keeps_original_source_location(registry):
    with pytest.raises(HNDLError, match="E_ARGUMENT") as error:
        resolve("\n    tagged(1, gain=0)\n", registry=registry,
                input_shape=("B", 64), output_shape=("B", 64))
    assert (error.value.line, error.value.column) == (2, 5)


def test_custom_argument_schemas_do_not_expand_ast_permissions(registry):
    lookups = []

    def forbidden(*args):
        lookups.append(args)
        raise AssertionError("Operator looked up before whole-source AST validation")

    registry.get = forbidden
    with pytest.raises(HNDLError, match="E_SYNTAX"):
        capture_config('tagged(1)\ntagged(1, label=str(x))', registry=registry,
                       input_shape=("B", 64), output_shape=("B", 64))
    assert lookups == []


def test_invalid_custom_call_leaves_current_and_call_order_unchanged(registry):
    def author(x):
        first = registry.ops.tagged(1)
        with pytest.raises(HNDLError, match="E_BINDING"):
            registry.ops.blend(first)
        registry.ops.relu()

    graph = capture_callable(author, registry=registry,
                             input_shape=("B", 64), output_shape=("B", 64))
    assert [node.id for node in graph.nodes] == ["n0", "n1"]
    assert graph.nodes[1].inputs["x"] == "node:n0/value"


@pytest.mark.parametrize("alias", ["_blend", "_registry", "__call__"])
def test_private_aliases_cannot_create_config_native_namespace_mismatches(alias):
    registry = Registry.builtins()
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        registry.register(alias, identity="tests.private", version=1, max_state_bytes=0,
                          shape=ShapeRule(inputs={"data": ("B", "F")}, outputs={"value": ("B", "F")}))


def test_multi_port_registration_and_both_frontends_need_no_torch():
    script = '''
import builtins
real_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("Pure custom frontend imported torch")
    return real_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from hndl import Argument, Registry, ShapeRule, resolve, resolve_callable
registry = Registry.builtins()
registry.register("blend", identity="test.blend", version=1,
    shape=ShapeRule(inputs={"left": ("B", "F"), "right": ("B", "F")},
                    outputs={"mixed": ("B", "F")}),
    arguments={"gain": Argument(float, default=1.0)}, max_state_bytes=0)
kwargs = dict(registry=registry, input_shape=("B", 16), output_shape=("B", 16))
first = resolve("blend(x,x)", **kwargs)
second = resolve_callable(lambda x: registry.ops.blend(x,x), **kwargs)
assert first.semantic_digest == second.semantic_digest
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=15)
