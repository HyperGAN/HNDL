"""Custom operators: one decorator, a shape DSL, identical behavior in both frontends."""

from dataclasses import FrozenInstanceError
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hndl import Arg, HNDLError, Registry, ResolvedPlan, operator, resolve, resolve_callable
from hndl.capture import capture_callable
from hndl.config import capture_config
from hndl.operator import parse_shape
from hndl.registry import normalize_arguments
from hndl.resolver import resolve_graph
from hndl.torch import network
from hndl.types import Graph, Node


class Identity(nn.Module):
    def forward(self, *tensors):
        return tensors[0] if len(tensors) == 1 else tensors


@pytest.fixture
def registry():
    result = Registry.builtins()

    @result.operator("tagged", identity="tests.tagged", summary="Tag a tensor.", shape="data[B, F] -> value[B, F]",
                     args={
                         "count": Arg(int, min=1, help="A required count."),
                         "gain": Arg(float, 1.0, min=0.0, exclusive_min=True, help="Positive gain."),
                         "enabled": Arg(bool, True, help="Toggle."),
                         "label": Arg(str, "a # literal", help="Free text."),
                     })
    class Tagged(nn.Module):
        def __init__(self, count, gain, enabled, label):
            super().__init__()
            self.gain = gain

        def forward(self, data):
            return data * self.gain

    @result.operator("pair", identity="tests.pair", summary="Duplicate a tensor.",
                     shape="data[B, F] -> original[B, F], copy[B, F]")
    class Pair(nn.Module):
        def forward(self, data):
            return data, data.clone()

    @result.operator("blend", identity="tests.blend", summary="Blend two tensors.",
                     shape="left[B, F], right[B, F] -> mixed[B, F]",
                     args={"gain": Arg(float, 1.0, min=0.0, exclusive_min=True, help="Weight of left.")})
    class Blend(nn.Module):
        def __init__(self, gain):
            super().__init__()
            self.gain = gain

        def forward(self, left, right):
            return self.gain * left + right

    @result.operator("style_norm", identity="tests.style_norm", summary="AdaIN with custom port names.",
                     shape="features[B, C, H, W], params[B, 2*C] -> normalized[B, C, H, W]",
                     args={"eps": Arg(float, 1e-5, min=0.0, exclusive_min=True, help="Stability epsilon.")})
    class StyleNorm(nn.Module):
        def __init__(self, eps):
            super().__init__()
            self.eps = eps

        def forward(self, features, params):
            normalized = F.instance_norm(features, use_input_stats=True, eps=self.eps)
            gamma, beta = torch.chunk(params, 2, dim=1)
            return normalized * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

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
    assert dict(plan.nodes[0].args) == {"count": 2, "gain": 0.5, "enabled": False, "label": "text # not a comment"}
    assert plan.nodes[0].inputs["data"] == "input:x"
    assert plan.nodes[1].inputs["x"] == "node:tag/value"
    assert set(plan.nodes[0].source["argument_origins"].values()) == {"explicit"}


def test_keyword_tensor_port_and_scalar_defaults_have_correct_origins(registry):
    plan, _ = resolve_pair("tagged(3, data=x)", lambda x: registry.ops.tagged(3, data=x), registry)
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
    plan, _ = resolve_pair("blend(tagged(1), tagged(2))",
                           lambda x: registry.ops.blend(registry.ops.tagged(1), registry.ops.tagged(2)), registry)
    assert plan.nodes[1].inputs["data"] == "node:n0/value"
    assert dict(plan.nodes[2].inputs) == {"left": "node:n0/value", "right": "node:n1/value"}


def test_scaled_rule_drives_both_frontends_backward_across_branches(registry):
    source = '''
    content, style = split(64)
    linear(content)
    features = reshape(32, 4, 4)
    params = linear(style)
    style_norm(params=params, features=features, eps=0.001, name="norm")
    '''

    def author(x):
        content, style = registry.ops.split(64)
        registry.ops.linear(content)
        features = registry.ops.reshape(32, 4, 4)
        params = registry.ops.linear(style)
        registry.ops.style_norm(params=params, features=features, eps=0.001, name="norm")

    plan, _ = resolve_pair(source, author, registry, input_shape=("B", 128), output_shape=("B", 32, 4, 4))
    assert plan.nodes[1].args["out_features"] == 512
    assert plan.nodes[3].args["out_features"] == 64
    assert plan.output_ref == "node:norm/normalized"
    assert plan.nodes[-1].output_shapes["normalized"] == ("B", 32, 4, 4)
    model = network(source, input_shape=("B", 128), output_shape=("B", 32, 4, 4), registry=registry, device="cpu")
    x = torch.randn(2, 128, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad.isfinite().all() and torch.count_nonzero(x.grad[:, 64:])


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
        resolve("a, b, c = pair()", registry=registry, input_shape=("B", 64), output_shape=("B", 64))


def test_schema_failure_keeps_original_source_location(registry):
    with pytest.raises(HNDLError, match="E_ARGUMENT") as error:
        resolve("\n    tagged(1, gain=0)\n", registry=registry, input_shape=("B", 64), output_shape=("B", 64))
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

    graph = capture_callable(author, registry=registry, input_shape=("B", 64), output_shape=("B", 64))
    assert [node.id for node in graph.nodes] == ["n0", "n1"]
    assert graph.nodes[1].inputs["x"] == "node:n0/value"


def test_scaled_contradictions_have_node_and_config_source(registry):
    with pytest.raises(HNDLError, match="divisible") as caught:
        resolve('z1,z2=split(63)\nlinear(z1)\nf=reshape(32,4,4)\nstyle_norm(f,z2)',
                input_shape=("B", 128), output_shape=("B", 32, 4, 4), registry=registry)
    assert caught.value.node == "n3"
    assert caught.value.line == 4
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve('z1,z2=split(64)\nlinear(z1)\nf=reshape(16,4,4)\nstyle_norm(f,z2)',
                input_shape=("B", 128), output_shape=("B", 16, 4, 4), registry=registry)


def test_scaled_rules_work_forward_backward_and_symbols_are_per_node():
    registry = Registry.builtins()

    @registry.operator("double", identity="example.double", summary="Repeat features twice.",
                       shape="data[B, F] -> value[B, 2*F]")
    class Double(nn.Module):
        def forward(self, data):
            return torch.cat((data, data), dim=1)

    plan = resolve('double()\ndouble()', input_shape=("B", 3), output_shape=("B", 12), registry=registry)
    assert plan.nodes[0].output_shapes["value"] == ("B", 6)
    assert plan.nodes[1].output_shapes["value"] == ("B", 12)
    reverse = resolve('linear()\ndouble()', input_shape=("B", 3), output_shape=("B", 10), registry=registry)
    assert reverse.nodes[0].args["out_features"] == 5
    with pytest.raises(HNDLError, match="divisible"):
        resolve('linear()\ndouble()', input_shape=("B", 3), output_shape=("B", 9), registry=registry)
    model = network("double()", input_shape=("B", 3), output_shape=("B", 6), registry=registry, device="cpu")
    x = torch.tensor([[1., 2., 3.]])
    torch.testing.assert_close(model(x), x.repeat(1, 2))


def test_shared_symbol_infers_unknown_feature_channels_from_style_port(registry):
    @registry.operator("height", identity="example.height", summary="Require height 4.",
                       shape="x[B, C, 4, W] -> out[B, C, 4, W]")
    class Height(Identity):
        pass

    nodes = (
        Node("style", "linear@1", {"out_features": 128}, {"x": "input:x"}),
        Node("project", "linear@1", {}, {"x": "input:x"}),
        Node("seed", "reshape@1", {}, {"x": "node:project/out"}),
        Node("norm", "tests.style_norm@1", {}, {"features": "node:seed/out", "params": "node:style/out"},
             ("normalized",)),
        Node("height", "example.height@1", {}, {"x": "node:norm/normalized"}),
        Node("flatten", "flatten@1", {}, {"x": "node:height/out"}),
    )
    plan = resolve_graph(Graph(nodes, ("B", 128), ("B", 1024), "node:flatten/out"), registry)
    assert plan.nodes[2].args["shape"] == (64, 4, 4)
    assert plan.nodes[1].args["out_features"] == 1024


def test_multioutput_rule_and_literal_axis_constraints():
    registry = Registry.builtins()

    @registry.operator("wide_fixed", identity="example.wide_fixed", summary="Two outputs.",
                       shape="data[B, F] -> wide[B, 2*F], fixed[B, 3]")
    class WideFixed(nn.Module):
        def forward(self, data):
            return torch.cat((data, data), 1), data[:, :3]

    plan = resolve('wide,fixed=wide_fixed()\nout=wide', input_shape=("B", 5), output_shape=("B", 10), registry=registry)
    assert plan.nodes[0].outputs == ("wide", "fixed")
    assert plan.nodes[0].output_shapes["fixed"] == ("B", 3)


def test_ellipsis_shares_middle_axes_and_symbols_bind_to_arguments():
    registry = Registry.builtins()

    @registry.operator("project_last", identity="example.project_last", summary="Linear on the last axis.",
                       shape="x[B, ..., D_in] -> out[B, ..., D_out]",
                       args={"width": Arg(int, inferable=True, dim="D_out", min=1, help="Output width."),
                             "in_width": Arg(int, inferable=True, dim="D_in", min=1, positional=False,
                                             help="Input width.")})
    class ProjectLast(nn.Module):
        def __init__(self, width, in_width):
            super().__init__()
            self.linear = nn.Linear(in_width, width)

        def forward(self, x):
            return self.linear(x)

    plan = resolve("project_last()", input_shape=("B", 3, 8, 5), output_shape=("B", 3, 8, 7), registry=registry)
    assert dict(plan.nodes[0].args) == {"width": 7, "in_width": 5}
    plan = resolve("project_last(4)", input_shape=("B", 5), output_shape=("B", 4), registry=registry)
    assert plan.nodes[0].args["in_width"] == 5
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("project_last(4)", input_shape=("B", 2, 6, 5), output_shape=("B", 3, 6, 4), registry=registry)
    model = network("project_last(4)", input_shape=("B", 5), output_shape=("B", 4), registry=registry, device="cpu")
    assert model[0].linear.in_features == 5 and model[0].linear.out_features == 4


def test_constructor_receives_symbols_and_shapes_on_request():
    registry = Registry.builtins()
    seen = {}

    @registry.operator("probe", identity="example.probe", summary="Record construction inputs.",
                       shape="x[B, C, H, W] -> out[B, C, H, W]")
    class Probe(nn.Module):
        def __init__(self, *, C, H, input_shapes, output_shapes):
            super().__init__()
            seen.update(C=C, H=H, input_shapes=input_shapes, output_shapes=output_shapes)

        def forward(self, x):
            return x

    network("probe()", input_shape=("B", 3, 4, 5), output_shape=("B", 3, 4, 5), registry=registry, device="cpu")
    assert seen == {"C": 3, "H": 4, "input_shapes": {"x": ("B", 3, 4, 5)}, "output_shapes": {"out": ("B", 3, 4, 5)}}


def test_named_ports_and_builtin_like_argument_names_have_no_implicit_meaning():
    registry = Registry.builtins()

    @registry.operator("custom", identity="example.custom", summary="Odd names.",
                       shape="features[B, ...] -> value[B, ...]",
                       args={"dim": Arg(str, "channels", help="A string, not an axis."),
                             "bias": Arg(int, -2, help="An int, not a flag."),
                             "kernel_size": Arg(bool, False, help="A flag, not a size."),
                             "eps": Arg(float, -1.0, max=0, help="A non-positive number.")})
    class Custom(nn.Module):
        def __init__(self, dim, bias, kernel_size, eps):
            super().__init__()

        def forward(self, features):
            return features

    plan = resolve('linear()\ncustom()', input_shape=("B", 4), output_shape=("B", 8), registry=registry)
    assert plan.nodes[0].args["out_features"] == 8
    assert plan.nodes[1].args == {"dim": "channels", "bias": -2, "kernel_size": False, "eps": -1.0}
    assert plan.nodes[1].input_shapes == {"features": ("B", 8)}
    entry = registry.get("custom")
    with pytest.raises(TypeError):
        entry.args["dim"] = Arg(int, 1, help="No.")
    with pytest.raises(FrozenInstanceError):
        entry.args["dim"].default = "x"


@pytest.mark.parametrize("factory", [
    lambda: Arg(list, help="h"), lambda: Arg(int, True, help="h"),
    lambda: Arg(float, float("nan"), help="h"), lambda: Arg(float, float("inf"), help="h"),
    lambda: Arg(int, inferable=True, help="h").validate(1.5, "x"),
    lambda: Arg(int, 1, inferable=True, help="h"),
    lambda: Arg(float, min=3, max=2, help="h"),
    lambda: Arg(int, min=0.1, max=0.9, help="h"),
    lambda: Arg(float, min=1, max=1, exclusive_min=True, help="h"),
    lambda: Arg(float, exclusive_min=True, help="h"), lambda: Arg(bool, min=0, help="h"),
    lambda: Arg(str, "x" * 16_385, help="h"), lambda: Arg(str, "\ud800", help="h"),
    lambda: Arg(int, 2**63, help="h"), lambda: Arg(str, "a", choices=("b",), help="h"),
    lambda: Arg(float, dim="D", help="h"),
])
def test_invalid_scalar_schemas_reject_registration(factory):
    with pytest.raises(HNDLError, match="E_REGISTRY|E_ARGUMENT"):
        factory()


def test_required_scalar_types_bounds_and_default_expansion():
    registry = Registry.builtins()

    @registry.operator("custom", identity="example.custom", summary="Scalars.", shape="x[B, ...] -> out[B, ...]",
                       args={"count": Arg(int, min=1, help="Count."),
                             "gain": Arg(float, 1, min=0, exclusive_min=True, max=2, help="Gain."),
                             "enabled": Arg(bool, True, help="Flag."),
                             "label": Arg(str, "a # literal", help="Label.")})
    class Custom(nn.Module):
        def __init__(self, count, gain, enabled, label):
            super().__init__()

        def forward(self, x):
            return x

    entry = registry.get("custom")
    assert entry.required == ("count",)
    assert normalize_arguments(entry, (1,), {}) == {"count": 1, "gain": 1.0, "enabled": True, "label": "a # literal"}
    for args in ({}, {"count": True}, {"count": 0}, {"count": 1, "gain": 0},
                 {"count": 1, "gain": float("nan")}, {"count": 1, "gain": 3},
                 {"count": 1, "enabled": 1}, {"count": 1, "extra": 1}):
        with pytest.raises(HNDLError, match="E_ARGUMENT"):
            normalize_arguments(entry, (), args)


@pytest.mark.parametrize("shape", [
    "x[C, B] -> out[B, C]", "x[1, C] -> out[B, C]", "x[B, 0] -> out[B, C]", "x[B, 0*C] -> out[B, C]",
    "x[B, C, H, W, D] -> out[B, C]", "name[B, C] -> out[B, C]", "-> out[B, C]", "x[B, C]",
    "x[B, C] -> out[B, C] -> y[B]", "x[B, ..., ...] -> out[B]", "x[..., B] -> out[B, C]",
    "x[B, B] -> out[B, B]", "x[B, 2147483648] -> out[B, C]", "x[B] -> out[B]",
    ", ".join(f"p{i}[B, C]" for i in range(33)) + " -> out[B, C]",
    "x*, y -> out", "x -> out*, y", "x -> y, out*", "x -> out*[B, C]", "x[B, C]:float128 -> out[B, C]",
])
def test_invalid_or_excessive_shape_patterns_fail(shape):
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        parse_shape(shape)


def test_port_argument_conflicts_and_private_aliases_fail():
    registry = Registry.builtins()
    for field in ("x", "out", "name", "policy"):
        with pytest.raises(HNDLError, match="conflicts"):
            registry.operator("custom", summary="Bad.", shape="x[B, F] -> out[B, F]",
                              args={field: Arg(int, 1, help="Conflict.")})(nn.Identity)
    with pytest.raises(HNDLError, match="at most 64"):
        registry.operator("custom", summary="Bad.", shape="x[B, F] -> out[B, F]",
                          args={f"a{i}": Arg(int, 1, help="h") for i in range(65)})(nn.Identity)
    for alias in ("_custom", "_registry", "__call__"):
        with pytest.raises(HNDLError, match="E_REGISTRY"):
            registry.operator(alias, summary="Bad.", shape="x[B, F] -> out[B, F]")(nn.Identity)
    with pytest.raises(HNDLError, match="unknown shape symbol"):
        registry.operator("custom", summary="Bad.", shape="x[B, F] -> out[B, F]",
                          args={"width": Arg(int, inferable=True, dim="G", help="h")})(nn.Identity)


def test_custom_pattern_resource_limits_apply_before_build():
    registry = Registry.builtins()

    @registry.operator("huge", identity="example.huge", summary="Enormous scale.",
                       shape="x[B, F] -> out[B, 2147483647*F]")
    class Huge(Identity):
        pass

    with pytest.raises(HNDLError, match="E_RESOURCE"):
        resolve('huge()\nlinear()', input_shape=("B", 2), output_shape=("B", 1), registry=registry)


def test_restoration_rechecks_rule_incompatible_under_same_identity():
    def make(scale):
        registry = Registry.builtins()

        @registry.operator("twice", identity="example.twice", summary="Scale features.",
                           shape=f"x[B, F] -> out[B, {scale}*F]")
        class Twice(Identity):
            pass

        return registry

    registry = make(2)
    plan = resolve('twice()', input_shape=("B", 4), output_shape=("B", 8), registry=registry)
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        ResolvedPlan.from_json(plan.to_json(), registry=make(3))
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        ResolvedPlan.from_json(plan.to_json())
    assert json.loads(plan.to_json())["nodes"][0]["op"] == "example.twice@1"
    assert ResolvedPlan.from_json(plan.to_json(), registry=registry).semantic_digest == plan.semantic_digest


def test_module_level_operator_is_declared_but_not_registered_until_added():
    @operator("shift", identity="example.shift", summary="Add a constant.", shape="features[B, ...] -> shifted[B, ...]",
              args={"amount": Arg(float, 0.25, help="Offset.")})
    class Shift(nn.Module):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount

        def forward(self, features):
            return features + self.amount

    with pytest.raises(HNDLError, match="E_SYNTAX|E_OPERATOR"):
        resolve("shift()", input_shape=("B", 3), output_shape=("B", 3))
    registry = Registry.builtins()
    registry.add(Shift)
    model = network('shift(0.5, name="offset"); relu()', input_shape=("B", 3), output_shape=("B", 3),
                    registry=registry, device="cpu")
    assert model[0] is model["offset"] and len(model) == 2
    assert model[:][0] is model[0]
    x = torch.tensor([[-2., 0., 1.]])
    torch.testing.assert_close(model(x), F.relu(x + 0.5))
    assert "[B, 3]" in repr(model) and "input shape" in repr(model)


@pytest.mark.parametrize("result_kind", ["tuple", "dict"])
def test_custom_multioutput_binds_declared_port_order(result_kind):
    registry = Registry.builtins()

    @registry.operator("signs", identity="example.signs", summary="Positive and negative copies.",
                       shape="features[B, F] -> positive[B, F], negative[B, F]")
    class Signs(nn.Module):
        def forward(self, features):
            if result_kind == "tuple":
                return features, -features
            # Reverse insertion order; declared port order is authoritative.
            return {"negative": -features, "positive": features}

    model = network('a, b = signs(name="pair"); concat(a, b)', input_shape=("B", 3), output_shape=("B", 6),
                    registry=registry, device="cpu")
    x = torch.tensor([[1., 2., 3.]], requires_grad=True)
    torch.testing.assert_close(model(x), torch.cat((x, -x), dim=1))
    model(x).square().sum().backward()
    torch.testing.assert_close(x.grad, 4 * x)
    with pytest.raises(TypeError):
        model[0]


@pytest.mark.parametrize("reuse", ["module", "submodule", "parameter", "buffer", "storage"])
def test_custom_builder_rejects_accidental_sharing_between_nodes(reuse):
    registry = Registry.builtins()
    shared_module = nn.Identity()
    shared_tensor = torch.ones(4)
    shared_parameter = nn.Parameter(shared_tensor)

    @registry.operator("shared", identity="example.shared", summary="Shares state by mistake.",
                       shape="x[B, ...] -> out[B, ...]")
    class Layer(nn.Module):
        def __new__(cls):
            return shared_module if reuse == "module" else super().__new__(cls)

        def __init__(self):
            super().__init__()
            if reuse == "submodule":
                self.inner = shared_module
            elif reuse == "parameter":
                self.weight = shared_parameter
            elif reuse == "buffer":
                self.register_buffer("buffer", shared_tensor)
            elif reuse == "storage":
                self.weight = nn.Parameter(shared_tensor.view(4))

        def forward(self, x):
            raise AssertionError("Construction must reject sharing before any forward call")

    with pytest.raises(HNDLError, match="E_REGISTRY.*independent instance"):
        network("shared(); shared()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)


def test_internal_aliases_and_empty_storages_do_not_imply_cross_node_sharing():
    registry = Registry.builtins()

    @registry.operator("valid", identity="example.valid", summary="Aliased weight.", shape="x[B, ...] -> out[B, ...]")
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.alias = self.weight
            self.register_buffer("empty", torch.empty(0))

        def forward(self, x):
            return x * self.weight

    model = network("valid(); valid()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)
    assert model[0] is not model[1] and model[:][0] is model[0]
    torch.testing.assert_close(model(torch.ones(2, 4)), torch.ones(2, 4))


def test_runtime_contract_and_late_state_violations_are_reported():
    registry = Registry.builtins()

    @registry.operator("wrong", identity="test.wrong", summary="Breaks its contract.", shape="x[B, ...] -> out[B, ...]")
    class WrongShape(nn.Module):
        def forward(self, x):
            return x[:, :1]

    @registry.operator("late", identity="test.late", summary="Creates state late.", shape="x[B, ...] -> out[B, ...]")
    class LateParameter(nn.Module):
        def forward(self, x):
            self.weight = nn.Parameter(torch.ones(1))
            return x

    broken = network("wrong()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        broken(torch.randn(2, 4))
    late = network("late()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        late(torch.randn(2, 4))


def test_state_bound_applies_before_allocation():
    registry = Registry.builtins()

    @registry.operator("buffered", identity="example.buffered", summary="Holds a buffer.",
                       shape="x[B, ...] -> out[B, ...]")
    class Buffered(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("value", torch.ones(8))

        def forward(self, x):
            return x

    plan = resolve("buffered()", input_shape=("B", 2), output_shape=("B", 2), registry=registry)
    with pytest.raises(HNDLError, match="E_RESOURCE.*before allocation"):
        network("buffered()", input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu",
                limits={"max_state_bytes": 16})
    model = network("buffered()", input_shape=("B", 2), output_shape=("B", 2), registry=registry, device="cpu",
                    limits={"max_state_bytes": 32})
    assert model.build_receipt["state_bytes"] == 32
    assert plan.nodes[0].op == "example.buffered@1"


def fan_registry():
    """A registry holding an operator whose output port count is one of its arguments."""
    registry = Registry.builtins()

    def relation(s):
        for port in s.outputs:
            s.equal(port, "data")

    @registry.operator("fan", identity="tests.fan", summary="Copy a tensor once per named tag.",
                       shape="data[B, F] -> out*", outputs_from="tags", relation=relation,
                       shape_text="one output per tag, each shaped like data",
                       args={"tags": Arg("strs", (), positional=False,
                                         help="One output per tag, in the order written.")})
    class Fan(nn.Module):
        def __init__(self, tags):
            super().__init__()
            self.tags = tags

        def forward(self, data):
            if not self.tags:
                return data
            return tuple(data * (index + 1) for index in range(len(self.tags)))

    return registry


def test_variadic_outputs_expand_to_ordinal_ports_in_both_frontends():
    registry = fan_registry()
    source = 'a, b, c = fan(tags=("x", "y", "z"))\nconcat(a, b, c, axis=1)'

    def author(x):
        a, b, c = registry.ops.fan(tags=("x", "y", "z"))
        registry.ops.concat(a, b, c, axis=1)

    plan, _ = resolve_pair(source, author, registry, input_shape=("B", 4), output_shape=("B", 12))
    assert plan.nodes[0].outputs == ("out0", "out1", "out2")
    assert plan.nodes[0].args["tags"] == ("x", "y", "z")
    assert dict(plan.nodes[0].output_shapes) == {"out0": ("B", 4), "out1": ("B", 4), "out2": ("B", 4)}
    assert dict(plan.nodes[1].inputs) == {"x0": "node:n0/out0", "x1": "node:n0/out1", "x2": "node:n0/out2"}
    assert registry.get("fan").port_dtype("out2", "bfloat16") == "bfloat16"

    model = network(source, input_shape=("B", 4), output_shape=("B", 12), registry=registry, device="cpu")
    data = torch.arange(8.0).reshape(2, 4)
    torch.testing.assert_close(model(data), torch.cat((data, 2 * data, 3 * data), dim=1))


def test_a_one_entry_sequence_still_returns_a_tuple_and_clears_current():
    registry = fan_registry()
    source = 'only, = fan(tags=("x",))\nrelu(only)'

    def author(x):
        only, = registry.ops.fan(tags=("x",))
        registry.ops.relu(only)

    plan, _ = resolve_pair(source, author, registry, input_shape=("B", 4), output_shape=("B", 4))
    assert plan.nodes[0].outputs == ("out0",)
    assert plan.nodes[1].inputs["x"] == "node:n0/out0"
    with pytest.raises(HNDLError, match="E_CURRENT"):
        resolve('fan(tags=("x",))\nrelu()', input_shape=("B", 4), output_shape=("B", 4), registry=registry)


def test_an_empty_sequence_keeps_the_single_declared_output_port():
    registry = fan_registry()

    def author(x):
        registry.ops.fan()
        registry.ops.relu()

    plan, _ = resolve_pair("fan()\nrelu()", author, registry, input_shape=("B", 4), output_shape=("B", 4))
    assert plan.nodes[0].outputs == ("out",)
    assert plan.nodes[1].inputs["x"] == "node:n0/out"
    explicit = resolve("fan(tags=())\nrelu()", input_shape=("B", 4), output_shape=("B", 4), registry=registry)
    assert explicit.semantic_digest == plan.semantic_digest


def test_variadic_output_unpacking_must_match_the_requested_count():
    registry = fan_registry()
    for source in ('a, b = fan(tags=("x", "y", "z"))\nconcat(a, b, axis=1)',
                   'a, b, c, d = fan(tags=("x", "y", "z"))\nconcat(a, b, axis=1)'):
        with pytest.raises(HNDLError, match="E_OUTPUT_ARITY"):
            resolve(source, input_shape=("B", 4), output_shape=("B", 8), registry=registry)

    def author(x):
        a, b = registry.ops.fan(tags=("x", "y", "z"))
        registry.ops.concat(a, b, axis=1)

    with pytest.raises(ValueError):
        capture_callable(author, input_shape=("B", 4), output_shape=("B", 8), registry=registry)


def test_a_variadic_output_plan_round_trips_and_its_ports_are_rechecked():
    registry = fan_registry()
    plan = resolve('a, b = fan(tags=("x", "y"))\nconcat(a, b, axis=1)', input_shape=("B", 4),
                   output_shape=("B", 8), registry=registry)
    encoded = plan.to_json()
    assert '"tags":["x","y"]' in encoded and '"outputs":["out0","out1"]' in encoded
    restored = ResolvedPlan.from_json(encoded, registry=registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.nodes[0].outputs == ("out0", "out1")

    tampered = json.loads(encoded)
    tampered["nodes"][0]["args"]["tags"] = ["x", "y", "z"]
    with pytest.raises(HNDLError, match="E_INTEGRITY"):
        ResolvedPlan.from_json(json.dumps(tampered), registry=registry)
    graph = Graph((Node("n0", "tests.fan@1", {"tags": ("x", "y")}, {"data": "input:x"}, ("out", "extra")),),
                  ("B", 4), ("B", 4), "node:n0/out")
    with pytest.raises(HNDLError, match="E_BINDING.*output ports must be"):
        resolve_graph(graph, registry=registry)


@pytest.mark.parametrize("declaration", [
    {"shape": "data[B, F] -> out*"},
    {"shape": "data[B, F] -> out*", "outputs_from": "label"},
    {"shape": "data[B, F] -> out*", "outputs_from": "absent"},
    {"shape": "data[B, F] -> out[B, F]", "outputs_from": "tags"},
])
def test_variadic_output_declarations_are_checked_at_registration(declaration):
    registry = Registry.builtins()

    class Fanned(nn.Module):
        def __init__(self, tags, label):
            super().__init__()

        def forward(self, data):
            return data

    args = {"tags": Arg("strs", (), positional=False, help="Tags."),
            "label": Arg(str, "", positional=False, help="Label.")}
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        registry.operator("fan", identity="tests.fan", summary="Fan out.", args=args, **declaration)(Fanned)


def test_a_string_sequence_argument_validates_its_entries():
    entry = fan_registry().get("fan")
    assert normalize_arguments(entry, (), {"tags": ["x", "y"]}) == {"tags": ("x", "y")}
    for tags in ("xy", 3, ("x", 4), (b"x",), ["x"] * 33):
        with pytest.raises(HNDLError, match="E_ARGUMENT|E_RESOURCE"):
            normalize_arguments(entry, (), {"tags": tags})
