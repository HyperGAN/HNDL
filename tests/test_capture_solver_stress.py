"""Comprehensive Empirical Stress-Test Harness for Milestone 1 Challenger 2.

Covers:
- M-05: Falsy input_dtype ("", False, 0, (), []) across resolve and resolve_callable.
- M-06: Port names with Python keywords, "range", initializer schemes, and operator aliases across capture, resolve, resolve_callable, resolve_graph.
- M-07: Solver interval reporting on ambiguous conv/pool graphs (open vs collapsed intervals).
- L-02: Native Python capture emit errors preserving node context.
- L-03: Ellipsis underflow on various ranks and exact port attribution.
- L-04: conv_input_range, conv_transpose_input, conv_axis, and solver defensive bounds.
- L-05: Node and Example container types validation.
"""

import keyword
import pytest
from torch import nn

from hndl import Example, HNDLError, Registry, ops, resolve, resolve_callable
from hndl.capture import Capture
from hndl.relations import (
    conv_axis,
    conv_input_range,
    conv_output,
    conv_transpose_input,
    conv_transpose_output,
)
from hndl.resolver import DEFAULT_LIMITS, _Solver, resolve_graph
from hndl.settings import SCHEME_NAMES
from hndl.types import Graph, Node


# ==============================================================================
# M-05: Falsy input_dtype ("", False, 0, (), [])
# ==============================================================================

@pytest.mark.parametrize("falsy_dtype", ["", False, 0, (), []])
def test_falsy_input_dtype_in_resolve_single_input(falsy_dtype):
    """Verify falsy input_dtype raises E_SCHEMA in resolve with single input, never coerced to float32."""
    with pytest.raises(HNDLError) as exc:
        resolve("linear(4)", input_shape=("B", 4), output_shape=("B", 4), input_dtype=falsy_dtype)
    assert exc.value.code == "E_SCHEMA"
    assert "input_dtype must be one of" in exc.value.message


@pytest.mark.parametrize("falsy_dtype", ["", False, 0, (), []])
def test_falsy_input_dtype_in_resolve_named_input(falsy_dtype):
    """Verify falsy input_dtype raises E_SCHEMA in resolve with named input_shape."""
    with pytest.raises(HNDLError) as exc:
        resolve("linear(4)", input_shape={"x": ("B", 4)}, output_shape=("B", 4), input_dtype=falsy_dtype)
    assert exc.value.code == "E_SCHEMA"
    assert "input_dtype must be one of" in exc.value.message


@pytest.mark.parametrize("falsy_dtype", ["", False, 0, (), []])
def test_falsy_input_dtype_in_resolve_multi_inputs(falsy_dtype):
    """Verify falsy input_dtype raises E_SCHEMA in resolve with multiple named inputs."""
    with pytest.raises(HNDLError) as exc:
        resolve("add(a=x, b=y)", input_shape={"x": ("B", 4), "y": ("B", 4)},
                output_shape=("B", 4), input_dtype=falsy_dtype)
    assert exc.value.code == "E_SCHEMA"
    assert "input_dtype[x] must be one of" in exc.value.message


@pytest.mark.parametrize("falsy_dtype", ["", False, 0, (), []])
def test_falsy_input_dtype_in_resolve_callable(falsy_dtype):
    """Verify falsy input_dtype raises E_SCHEMA in resolve_callable, never coerced to float32."""
    with pytest.raises(HNDLError) as exc:
        resolve_callable(lambda x: ops.linear(x, 4), input_shape=("B", 4), output_shape=("B", 4), input_dtype=falsy_dtype)
    assert exc.value.code == "E_SCHEMA"
    assert "input_dtype must be one of" in exc.value.message


@pytest.mark.parametrize("falsy_dtype", ["", False, 0, (), []])
def test_falsy_input_dtype_in_capture_and_resolve_graph(falsy_dtype):
    """Verify falsy input_dtype in Capture finishes with invalid dtype and fails in resolve_graph."""
    with Capture(input_shape=("B", 4), output_shape=("B", 4), input_dtype=falsy_dtype) as cap:
        out = ops.linear(cap.input, 4)
        graph = cap.finish(out)
    assert graph.input_dtype == falsy_dtype
    with pytest.raises(HNDLError) as exc:
        resolve_graph(graph)
    assert exc.value.code == "E_SCHEMA"
    assert "input_dtype must be one of" in exc.value.message


@pytest.mark.parametrize("falsy_dtype", ["", False, 0, (), []])
def test_falsy_input_dtype_in_direct_graph_named_inputs(falsy_dtype):
    """Verify falsy input_dtype directly stored in Graph.named_inputs fails in resolve_graph."""
    node = Node("n0", "linear@1", {"out_features": 4}, {"x": "input:x"})
    graph = Graph((node,), ("B", 4), ("B", 4), "node:n0/out",
                  named_inputs={"x": {"shape": ("B", 4), "dtype": falsy_dtype}})
    with pytest.raises(HNDLError) as exc:
        resolve_graph(graph)
    assert exc.value.code == "E_SCHEMA"
    assert "input_dtype must be one of" in exc.value.message


# ==============================================================================
# M-06: Reserved port names (keywords, range, SCHEME_NAMES, aliases)
# ==============================================================================

# Exhaustive test of lowercase Python keywords that match identifier syntax
PYTHON_KEYWORDS = [kw for kw in keyword.kwlist if kw.islower()]
BUILTIN_SCHEMES = sorted(SCHEME_NAMES)
BUILTIN_ALIASES = ["linear", "conv", "relu", "gelu", "add", "concat", "batch_norm", "layer_norm"]


@pytest.mark.parametrize("kw", PYTHON_KEYWORDS)
def test_all_python_keywords_rejected_on_external_ports(kw):
    """Test all lowercase Python keywords raise E_SCHEMA across resolve, Capture, and resolve_callable."""
    # 1. resolve input port
    with pytest.raises(HNDLError) as exc_in:
        resolve("linear(4)", input_shape={kw: ("B", 4)}, output_shape=("B", 4))
    assert exc_in.value.code == "E_SCHEMA"
    assert "must not be Python keywords" in exc_in.value.message

    # 2. resolve output port
    with pytest.raises(HNDLError) as exc_out:
        resolve("linear(4)", input_shape=("B", 4), output_shape={kw: ("B", 4)})
    assert exc_out.value.code == "E_SCHEMA"
    assert "must not be Python keywords" in exc_out.value.message

    # 3. Capture input port
    with pytest.raises(HNDLError) as exc_cap:
        Capture(input_shape={kw: ("B", 4)}, output_shape=("B", 4))
    assert exc_cap.value.code == "E_SCHEMA"
    assert "must not be Python keywords" in exc_cap.value.message

    # 4. resolve_callable
    with pytest.raises(HNDLError) as exc_call:
        resolve_callable(lambda **kwargs: kwargs[kw], input_shape={kw: ("B", 4)}, output_shape={"out": ("B", 4)})
    assert exc_call.value.code == "E_SCHEMA"
    assert "must not be Python keywords" in exc_call.value.message


def test_range_port_name_rejected_on_all_frontends():
    """Test 'range' port name raises E_NAME across resolve, resolve_callable, Capture, and resolve_graph."""
    # 1. resolve input
    with pytest.raises(HNDLError) as exc1:
        resolve("linear(4)", input_shape={"range": ("B", 4)}, output_shape=("B", 4))
    assert exc1.value.code == "E_NAME"
    assert "collide with reserved loop iterator: range" in exc1.value.message

    # 2. resolve output
    with pytest.raises(HNDLError) as exc2:
        resolve("linear(4)", input_shape=("B", 4), output_shape={"range": ("B", 4)})
    assert exc2.value.code == "E_NAME"
    assert "collide with reserved loop iterator: range" in exc2.value.message

    # 3. Capture input
    with pytest.raises(HNDLError) as exc3:
        Capture(input_shape={"range": ("B", 4)}, output_shape=("B", 4))
    assert exc3.value.code == "E_NAME"
    assert "collide with reserved loop iterator: range" in exc3.value.message

    # 4. resolve_callable
    with pytest.raises(HNDLError) as exc4:
        resolve_callable(lambda **kwargs: kwargs["range"], input_shape={"range": ("B", 4)}, output_shape={"out": ("B", 4)})
    assert exc4.value.code == "E_NAME"
    assert "collide with reserved loop iterator: range" in exc4.value.message

    # 5. resolve_graph direct graph
    node = Node("n0", "linear@1", {"out_features": 4}, {"x": "input:range"})
    graph = Graph((node,), ("B", 4), ("B", 4), "node:n0/out", named_inputs={"range": {"shape": ("B", 4)}})
    with pytest.raises(HNDLError) as exc5:
        resolve_graph(graph)
    assert exc5.value.code == "E_NAME"
    assert "collide with reserved loop iterator: range" in exc5.value.message


@pytest.mark.parametrize("scheme", BUILTIN_SCHEMES)
def test_all_initializer_schemes_rejected_on_all_frontends(scheme):
    """Test every initializer scheme in SCHEME_NAMES raises E_NAME across all interfaces."""
    # 1. resolve input
    with pytest.raises(HNDLError) as exc1:
        resolve("linear(4)", input_shape={scheme: ("B", 4)}, output_shape=("B", 4))
    assert exc1.value.code == "E_NAME"
    assert f"collide with initializer schemes: {scheme}" in exc1.value.message

    # 2. resolve output
    with pytest.raises(HNDLError) as exc2:
        resolve("linear(4)", input_shape=("B", 4), output_shape={scheme: ("B", 4)})
    assert exc2.value.code == "E_NAME"
    assert f"collide with initializer schemes: {scheme}" in exc2.value.message

    # 3. Capture
    with pytest.raises(HNDLError) as exc3:
        Capture(input_shape={scheme: ("B", 4)}, output_shape=("B", 4))
    assert exc3.value.code == "E_NAME"
    assert f"collide with initializer schemes: {scheme}" in exc3.value.message

    # 4. resolve_graph
    node = Node("n0", "linear@1", {"out_features": 4}, {"x": f"input:{scheme}"})
    graph = Graph((node,), ("B", 4), ("B", 4), "node:n0/out", named_inputs={scheme: {"shape": ("B", 4)}})
    with pytest.raises(HNDLError) as exc4:
        resolve_graph(graph)
    assert exc4.value.code == "E_NAME"
    assert f"collide with initializer schemes: {scheme}" in exc4.value.message


@pytest.mark.parametrize("alias", BUILTIN_ALIASES)
def test_operator_aliases_rejected_on_all_frontends(alias):
    """Test operator alias port names raise E_NAME across all interfaces."""
    # 1. resolve input
    with pytest.raises(HNDLError) as exc1:
        resolve("linear(4)", input_shape={alias: ("B", 4)}, output_shape=("B", 4))
    assert exc1.value.code == "E_NAME"
    assert f"collide with operator aliases: {alias}" in exc1.value.message

    # 2. resolve output
    with pytest.raises(HNDLError) as exc2:
        resolve("linear(4)", input_shape=("B", 4), output_shape={alias: ("B", 4)})
    assert exc2.value.code == "E_NAME"
    assert f"collide with operator aliases: {alias}" in exc2.value.message

    # 3. Capture
    with pytest.raises(HNDLError) as exc3:
        Capture(input_shape={alias: ("B", 4)}, output_shape=("B", 4))
    assert exc3.value.code == "E_NAME"
    assert f"collide with operator aliases: {alias}" in exc3.value.message

    # 4. resolve_graph
    node = Node("n0", "linear@1", {"out_features": 4}, {"x": f"input:{alias}"})
    graph = Graph((node,), ("B", 4), ("B", 4), "node:n0/out", named_inputs={alias: {"shape": ("B", 4)}})
    with pytest.raises(HNDLError) as exc4:
        resolve_graph(graph)
    assert exc4.value.code == "E_NAME"
    assert f"collide with operator aliases: {alias}" in exc4.value.message


# ==============================================================================
# M-07: Solver interval reporting on ambiguous graphs
# ==============================================================================

def test_m07_ambiguous_conv_stride_1_2_reports_only_open_interval():
    """Conv with stride (1, 2): axis 2 collapses to (8, 8), axis 3 has open (7, 8). Reports only (7, 8)."""
    registry = Registry.builtins()

    def dummy_relation(s):
        s.rank("out", 4)
        s.axis("out", 1, 3)

    @registry.operator("gen_m07_1_2", identity="gen_m07_1_2", summary="test",
                       shape="x -> out", relation=dummy_relation)
    class GenM07_12(nn.Module):
        pass

    node0 = Node("g", "gen_m07_1_2@1", {}, {"x": "input:x"})
    node1 = Node("c", "conv2d@1", {"out_channels": 16, "kernel_size": 3, "stride": (1, 2), "padding": 1},
                 {"x": "node:g/out"})
    graph = Graph((node0, node1), ("B", 3), ("B", 16, 8, 4), "node:c/out")

    with pytest.raises(HNDLError) as exc:
        resolve_graph(graph, registry)
    assert exc.value.code == "E_AMBIGUOUS"
    msg = exc.value.message
    assert "; inverse allows (7, 8)" in msg
    assert "(8, 8)" not in msg


def test_m07_ambiguous_conv_stride_2_1_reports_only_open_interval():
    """Conv with stride (2, 1): axis 2 has open (7, 8), axis 3 collapses to (8, 8). Reports only (7, 8)."""
    registry = Registry.builtins()

    def dummy_relation(s):
        s.rank("out", 4)
        s.axis("out", 1, 3)

    @registry.operator("gen_m07_2_1", identity="gen_m07_2_1", summary="test",
                       shape="x -> out", relation=dummy_relation)
    class GenM07_21(nn.Module):
        pass

    node0 = Node("g", "gen_m07_2_1@1", {}, {"x": "input:x"})
    node1 = Node("c", "conv2d@1", {"out_channels": 16, "kernel_size": 3, "stride": (2, 1), "padding": 1},
                 {"x": "node:g/out"})
    graph = Graph((node0, node1), ("B", 3), ("B", 16, 4, 8), "node:c/out")

    with pytest.raises(HNDLError) as exc:
        resolve_graph(graph, registry)
    assert exc.value.code == "E_AMBIGUOUS"
    msg = exc.value.message
    assert "; inverse allows (7, 8)" in msg
    assert "(8, 8)" not in msg


def test_m07_ambiguous_pooling_stride_1_2_reports_only_open_interval():
    """AvgPool with stride (1, 2): axis 2 collapses to (8, 8), axis 3 has open (7, 8). Reports only (7, 8)."""
    registry = Registry.builtins()

    def dummy_relation(s):
        s.rank("out", 4)
        s.axis("out", 1, 3)

    @registry.operator("gen_m07_pool", identity="gen_m07_pool", summary="test",
                       shape="x -> out", relation=dummy_relation)
    class GenM07Pool(nn.Module):
        pass

    node0 = Node("g", "gen_m07_pool@1", {}, {"x": "input:x"})
    node1 = Node("p", "avg_pool2d@1",
                 {"kernel_size": (1, 1), "stride": (1, 2), "padding": (0, 0), "count_include_pad": True},
                 {"x": "node:g/out"})
    graph = Graph((node0, node1), ("B", 3), ("B", 3, 8, 4), "node:p/out")

    with pytest.raises(HNDLError) as exc:
        resolve_graph(graph, registry)
    assert exc.value.code == "E_AMBIGUOUS"
    msg = exc.value.message
    assert "; inverse allows (7, 8)" in msg
    assert "(8, 8)" not in msg


def test_m07_fully_collapsed_intervals_never_reported():
    """When both spatial axes collapse to (8, 8) but channels is left None, collapsed (8, 8) is never reported."""
    registry = Registry.builtins()

    def dummy_relation_no_ch(s):
        s.rank("out", 4)
        # channels left as None!

    @registry.operator("gen_no_ch", identity="gen_no_ch", summary="test",
                       shape="x -> out", relation=dummy_relation_no_ch)
    class GenNoCh(nn.Module):
        pass

    node0 = Node("g", "gen_no_ch@1", {}, {"x": "input:x"})
    node1 = Node("c", "conv2d@1", {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                 {"x": "node:g/out"})
    graph = Graph((node0, node1), ("B", 3), ("B", 16, 8, 8), "node:c/out")

    with pytest.raises(HNDLError) as exc:
        resolve_graph(graph, registry)
    assert exc.value.code == "E_AMBIGUOUS"
    assert "(8, 8)" not in exc.value.message
    assert "; inverse allows" not in exc.value.message


# ==============================================================================
# L-02: Native Python capture emit errors preserve node context
# ==============================================================================

def test_l02_node_context_preserved_across_emit_errors():
    """Verify that all error classes raised during callable emit attribute explicit node name."""
    # 1. Argument validation error
    with pytest.raises(HNDLError) as exc1:
        resolve_callable(lambda x: ops.linear(x, name="custom_linear_arg", out_features=-5),
                         input_shape=("B", 10), output_shape=("B", 20))
    assert exc1.value.node == "custom_linear_arg"
    assert exc1.value.code == "E_ARGUMENT"
    assert "node custom_linear_arg" in str(exc1.value)

    # 2. Duplicate binding error
    with pytest.raises(HNDLError) as exc2:
        resolve_callable(lambda x: ops.linear(x, x=x, name="custom_dup_binding"),
                         input_shape=("B", 10), output_shape=("B", 20))
    assert exc2.value.node == "custom_dup_binding"
    assert exc2.value.code == "E_BINDING"
    assert "node custom_dup_binding" in str(exc2.value)

    # 3. Variadic binding error (too few inputs)
    with pytest.raises(HNDLError) as exc3:
        resolve_callable(lambda x: ops.concat(x, name="custom_concat_variadic"),
                         input_shape=("B", 10), output_shape=("B", 20))
    assert exc3.value.node == "custom_concat_variadic"
    assert exc3.value.code == "E_BINDING"

    # 4. Unknown policy error
    with pytest.raises(HNDLError) as exc4:
        resolve_callable(lambda x: ops.linear(x, name="custom_bad_policy", policy=999),
                         input_shape=("B", 10), output_shape=("B", 20))
    assert exc4.value.node == "custom_bad_policy"
    assert exc4.value.code == "E_ARGUMENT"

    # 5. Invalid initialization scheme
    with pytest.raises(HNDLError) as exc5:
        resolve_callable(lambda x: ops.linear(x, name="custom_bad_init", init={"weight": "invalid_scheme"}),
                         input_shape=("B", 10), output_shape=("B", 20))
    assert exc5.value.node == "custom_bad_init"
    assert exc5.value.code == "E_INITIALIZATION"

    # 6. Unnamed node leaves node=None
    with pytest.raises(HNDLError) as exc6:
        resolve_callable(lambda x: ops.linear(x, out_features=-5),
                         input_shape=("B", 10), output_shape=("B", 20))
    assert exc6.value.node is None


# ==============================================================================
# L-03: Ellipsis underflow and exact port attribution
# ==============================================================================

def test_l03_ellipsis_underflow_blames_exact_port():
    """Verify that rank underflow on ellipsis patterns blames the exact port that underflowed."""
    registry = Registry.builtins()

    def dummy_unknown(s):
        pass

    @registry.operator("test_unknown_l03", identity="test_unknown_l03", summary="dummy",
                       shape="x -> out", relation=dummy_unknown)
    class DummyUnknownL03(nn.Module):
        pass

    @registry.operator("test_ellipsis_l03", identity="test_ellipsis_l03", summary="dummy",
                       shape="x[B, ..., C, D] -> out[B, ..., H, W]")
    class DummyEllipsisL03(nn.Module):
        pass

    # Case A: x has unknown shape (None), out has rank 2 ("B", 10).
    # Declared pattern requires rank >= 3.
    # Should blame 'out', NOT 'x'.
    n0 = Node("n0", "test_unknown_l03@1", {}, {"x": "input:x"})
    n1 = Node("n1", "test_ellipsis_l03@1", {}, {"x": "node:n0/out"})
    graph_a = Graph((n0, n1), ("B", 4), ("B", 10), "node:n1/out")

    with pytest.raises(HNDLError) as exc_a:
        resolve_graph(graph_a, registry)
    assert exc_a.value.code == "E_CONSTRAINT"
    assert "Rank of out is below the declared pattern rank" in exc_a.value.message
    assert "Rank of x is below the declared pattern rank" not in exc_a.value.message

    # Case B: x has rank 2 ("B", 10), out has unknown shape.
    # Should blame 'x', NOT 'out'.
    n2 = Node("n2", "test_ellipsis_l03@1", {}, {"x": "input:x"})
    graph_b = Graph((n2,), ("B", 10), ("B", 10, 4, 4), "node:n2/out")
    with pytest.raises(HNDLError) as exc_b:
        resolve_graph(graph_b, registry)
    assert exc_b.value.code == "E_CONSTRAINT"
    assert "Rank of x is below the declared pattern rank" in exc_b.value.message

    # Case C: Rank 3 input underflows pattern requiring rank >= 4
    @registry.operator("test_ellipsis_rank4_l03", identity="test_ellipsis_rank4_l03", summary="dummy",
                       shape="x[B, ..., C, D, E] -> out[B, ..., H, W]")
    class DummyRank4L03(nn.Module):
        pass

    n3 = Node("n3", "test_ellipsis_rank4_l03@1", {}, {"x": "input:x"})
    graph_c = Graph((n3,), ("B", 10, 20), ("B", 10, 4, 4), "node:n3/out")
    with pytest.raises(HNDLError) as exc_c:
        resolve_graph(graph_c, registry)
    assert exc_c.value.code == "E_CONSTRAINT"
    assert "Rank of x is below the declared pattern rank" in exc_c.value.message


# ==============================================================================
# L-04: conv_input_range, conv_transpose_input, and relations defensive bounds
# ==============================================================================

def test_l04_conv_input_range_bounds():
    """Test conv_input_range on boundary values."""
    # extent <= 0 must return (1, 0)
    assert conv_input_range(0, kernel=3, stride=1, padding=0, dilation=1) == (1, 0)
    assert conv_input_range(-1, kernel=3, stride=1, padding=0, dilation=1) == (1, 0)
    assert conv_input_range(-999, kernel=5, stride=2, padding=2, dilation=2) == (1, 0)

    # positive extent returns valid range
    assert conv_input_range(1, kernel=3, stride=1, padding=0, dilation=1) == (3, 3)
    assert conv_input_range(1, kernel=3, stride=2, padding=1, dilation=1) == (1, 2)


def test_l04_conv_transpose_input_bounds():
    """Test conv_transpose_input on boundary values."""
    # extent <= 0 must return None
    assert conv_transpose_input(0, kernel=3, stride=1, padding=0, dilation=1) is None
    assert conv_transpose_input(-1, kernel=3, stride=1, padding=0, dilation=1) is None
    assert conv_transpose_input(-100, kernel=3, stride=1, padding=0, dilation=1) is None

    # extent = 1 with kernel 4, stride 1, padding 0 -> would give -2 without guard -> must return None
    assert conv_transpose_input(1, kernel=4, stride=1, padding=0, dilation=1) is None

    # extent = 1 with kernel 3, stride 2, padding 0 -> numerator = -2 -> result = 0 < 1 -> must return None
    assert conv_transpose_input(1, kernel=3, stride=2, padding=0, dilation=1) is None

    # positive valid result
    assert conv_transpose_input(10, kernel=3, stride=1, padding=1, dilation=1) == 10


def test_l04_conv_axis_defensive_bounds():
    """Test conv_axis defensive bounds and error messages."""
    # Missing axis argument
    with pytest.raises(HNDLError) as exc1:
        conv_axis(None, kernel=3)
    assert exc1.value.code == "E_ARGUMENT"
    assert "conv_axis requires an axis argument" in exc1.value.message

    # NodeView s is None with negative or out-of-bounds axis
    with pytest.raises(HNDLError) as exc2:
        conv_axis(None, -1, kernel=3)
    assert exc2.value.code == "E_CONSTRAINT"
    assert "Axis -1 is out of bounds for 4D convolution" in exc2.value.message

    with pytest.raises(HNDLError) as exc3:
        conv_axis(None, 4, kernel=3)
    assert exc3.value.code == "E_CONSTRAINT"
    assert "Axis 4 is out of bounds for 4D convolution" in exc3.value.message

    with pytest.raises(HNDLError) as exc4:
        conv_axis(None, 2, kernel=3)
    assert exc4.value.code == "E_ARGUMENT"
    assert "NodeView s cannot be None" in exc4.value.message


def test_l04_solver_axis_and_interval_defensive_bounds():
    """Test _Solver.axis and _Solver.interval reject negative and out-of-bounds axes cleanly."""
    solver = _Solver(Graph((), ("B", 3, 8, 8), ("B", 3, 8, 8), "input:x"), (), {}, DEFAULT_LIMITS)
    ref = "test_tensor"
    solver.shapes[ref] = ["B", 3, 8, 8]

    # Negative axis must not mutate or alias
    solver.axis(ref, -1, 999)
    assert solver.shapes[ref] == ["B", 3, 8, 8]

    # Out-of-bounds axis must not raise IndexError
    solver.axis(ref, 10, 999)
    assert solver.shapes[ref] == ["B", 3, 8, 8]

    # Negative interval axis must be a no-op
    solver.interval(ref, -1, 5, 5)
    assert solver.shapes[ref] == ["B", 3, 8, 8]
    assert (ref, -1) not in solver.intervals


# ==============================================================================
# L-05: Node and Example container types validation
# ==============================================================================

@pytest.mark.parametrize("bad_args", [[1, 2], 123, "args", None])
def test_l05_node_invalid_args_raises_type_error(bad_args):
    """Test Node raises TypeError when args is not a Mapping."""
    with pytest.raises(TypeError, match="Node args must be a Mapping"):
        Node("n0", "linear@1", args=bad_args)


@pytest.mark.parametrize("bad_inputs", [["input:x"], 123, "inputs", None])
def test_l05_node_invalid_inputs_raises_type_error(bad_inputs):
    """Test Node raises TypeError when inputs is not a Mapping."""
    with pytest.raises(TypeError, match="Node inputs must be a Mapping"):
        Node("n0", "linear@1", inputs=bad_inputs)


@pytest.mark.parametrize("bad_outputs", ["out", 123, None, {"out": 1}, [1, 2], ("out", 123)])
def test_l05_node_invalid_outputs_raises_type_error(bad_outputs):
    """Test Node raises TypeError when outputs is a string or contains non-strings."""
    with pytest.raises(TypeError):
        Node("n0", "linear@1", outputs=bad_outputs)


@pytest.mark.parametrize("bad_shape", ["B, 4", 123, None, {"x": "B, 4"}, {"x": 123}, {}])
def test_l05_example_invalid_input_shape_raises_registry_error(bad_shape):
    """Test Example raises E_REGISTRY on invalid input_shape container types."""
    with pytest.raises(HNDLError) as exc:
        Example("linear(10)", input_shape=bad_shape, output_shape=("B", 10))
    assert exc.value.code == "E_REGISTRY"


@pytest.mark.parametrize("bad_shape", ["B, 4", 123, None, {"out": "B, 4"}, {"out": 123}, {}])
def test_l05_example_invalid_output_shape_raises_registry_error(bad_shape):
    """Test Example raises E_REGISTRY on invalid output_shape container types."""
    with pytest.raises(HNDLError) as exc:
        Example("linear(10)", input_shape=("B", 10), output_shape=bad_shape)
    assert exc.value.code == "E_REGISTRY"


def test_l05_example_multi_port_shape_dict_preserved():
    """Test Example preserves mapping type for multi-port shape specifications."""
    ex = Example("linear(10)", input_shape={"x": ("B", 4), "y": ("B", 6)},
                 output_shape={"out": ("B", 10)})
    assert isinstance(ex.input_shape, dict)
    assert ex.input_shape == {"x": ("B", 4), "y": ("B", 6)}
    assert ex.output_shape == {"out": ("B", 10)}
    assert ex.inputs == ex.input_shape
    assert ex.outputs == ex.output_shape


# ==============================================================================
# Mathematical Oracles & Property Fuzzing for Relations
# ==============================================================================

def test_relations_conv_input_range_coverage_oracle():
    """Oracle: for any extent >= 1, conv_output maps to out, and lower <= extent <= upper."""
    for extent in range(1, 50):
        for kernel in [1, 2, 3, 5]:
            for stride in [1, 2, 3]:
                for padding in [0, 1, 2]:
                    for dilation in [1, 2]:
                        out = conv_output(extent, kernel, stride, padding, dilation)
                        if out >= 1:
                            lo, hi = conv_input_range(out, kernel, stride, padding, dilation)
                            assert lo <= extent <= hi, f"Oracle failed: extent={extent}, out={out}, range=({lo}, {hi})"


def test_relations_conv_transpose_exact_invertibility_oracle():
    """Oracle: for any extent >= 1, conv_transpose_input(conv_transpose_output(extent)) == extent."""
    for extent in range(1, 50):
        for kernel in [1, 2, 3, 5]:
            for stride in [1, 2, 3]:
                for padding in [0, 1, 2]:
                    for dilation in [1, 2]:
                        for out_pad in range(stride):
                            out = conv_transpose_output(extent, kernel, stride, padding, dilation, out_pad)
                            if out >= 1:
                                recovered = conv_transpose_input(out, kernel, stride, padding, dilation, out_pad)
                                assert recovered == extent, f"Oracle failed: extent={extent}, recovered={recovered}"


def test_relations_extreme_values_and_empty_intervals():
    """Extreme bounds: huge extents, extreme paddings, and empty interval detection."""
    huge_extent = 10**9
    lo, hi = conv_input_range(huge_extent, kernel=3, stride=2, padding=1, dilation=1)
    assert lo > 0 and hi >= lo

    recovered = conv_transpose_input(huge_extent, kernel=3, stride=2, padding=1, dilation=1, output_padding=1)
    assert recovered is not None and recovered > 0

    # Large padding produces empty interval lower > upper
    lo_empty, hi_empty = conv_input_range(1, kernel=3, stride=1, padding=100)
    assert lo_empty > hi_empty

    # Solver rejects empty interval with E_CONSTRAINT
    solver = _Solver(Graph((), ("B", 3, 8, 8), ("B", 3, 8, 8), "input:x"), (), {}, DEFAULT_LIMITS)
    with pytest.raises(HNDLError) as exc:
        solver.interval("t", 2, 10, 5)
    assert exc.value.code == "E_CONSTRAINT"
    assert "Convolution inverse intervals are incompatible" in exc.value.message
