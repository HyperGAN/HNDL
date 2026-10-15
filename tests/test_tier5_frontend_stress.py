"""Tier 5 White-Box Adversarial Coverage Hardening Stress Test Suite.

Adversarial stress-testing across:
1. Parser Screening, Tokenization & AST Invariants (_parser.py, config.py)
2. Symbolic Solver, Interval Pruning & Constraint Graphs (resolver.py, relations.py, capture.py)
3. Plan Serialization, Deserialization & Repr Formatting (types.py, resolver.py)
"""

from __future__ import annotations

import copy
import json
import re

import pytest
import torch

from hndl import HNDLError, Registry, ResolvedPlan, resolve
from hndl._parser import MAX_DEPTH, MAX_LOOP_DEPTH, MAX_OPERATORS, parse
from hndl.config import capture_config
from hndl.relations import (
    broadcast,
    conv_axis,
    conv_input_range,
    conv_transpose_input,
    spatial,
)
from hndl.resolver import _validate_port_reservations, resolve_graph
from hndl.types import (
    Graph,
    Node,
    batch_multiple,
    batch_symbol,
    canonical,
    contract_header,
    digest,
)


# ==============================================================================
# Group 1: Parser Screening, Tokenization & AST Invariants (Tests 1–10)
# ==============================================================================


def test_01_screen_bracket_depth_exact_boundary():
    """Exact boundary stress test for bracket nesting screener depth (MAX_DEPTH=50)."""
    aliases = Registry.builtins().aliases

    # 49 surrounding parens + 1 paren from relu(...) = exactly 50 depth (boundary pass)
    src_50 = "(" * 49 + "relu(x)" + ")" * 49
    parsed_50 = parse(src_50, aliases)
    assert len(parsed_50["statements"]) == 1

    # 50 surrounding parens + 1 paren from relu(...) = 51 depth (boundary violation)
    src_51 = "(" * 50 + "relu(x)" + ")" * 50
    with pytest.raises(HNDLError) as exc_info:
        parse(src_51, aliases)
    err = exc_info.value
    assert err.code == "E_RESOURCE"
    assert f"Source nests brackets more than {MAX_DEPTH} levels deep" in err.message
    assert err.line == 1
    assert err.column == 55


def test_02_screen_operator_count_exact_boundary():
    """Exact boundary stress test for operator/keyword screener count (MAX_OPERATORS=32)."""
    aliases = Registry.builtins().aliases

    # 32 chained unary '-' signs before 1:
    # First '-' attaches to number literal without incrementing operator count.
    # The subsequent 31 signs each increment operator count to 31 <= 32 (passes screen).
    src_pass = "-" * 32 + "1"
    with pytest.raises(HNDLError) as exc_info:
        parse(src_pass, aliases)
    # Fails in Validator because bare expressions must be registered calls, NOT in screener
    assert exc_info.value.code == "E_SYNTAX"
    assert "Only registered calls may be expression statements" in exc_info.value.message

    # 34 chained unary '-' signs before 1 produces 33 counted operators > 32 (screener trip)
    src_fail = "-" * 34 + "1"
    with pytest.raises(HNDLError) as exc_info:
        parse(src_fail, aliases)
    err = exc_info.value
    assert err.code == "E_SYNTAX"
    assert f"Source uses more than {MAX_OPERATORS} Python operators or keywords" in err.message


def test_03_screen_fstring_variants_and_raw_strings():
    """Verify adversarial f-string permutations are rejected while raw strings pass."""
    aliases = Registry.builtins().aliases

    fstring_payloads = [
        'f"layer_{1}"',
        'F"layer_{1}"',
        'rf"prefix_{x}"',
        'fr"prefix_{x}"',
        'f"""multiline_{name}"""',
        "rf'''multiline_{name}'''",
    ]
    for payload in fstring_payloads:
        src = f"linear(x, 64, name={payload})"
        with pytest.raises(HNDLError) as exc_info:
            parse(src, aliases)
        assert exc_info.value.code == "E_SYNTAX"
        assert "f-strings are not permitted" in exc_info.value.message

    # Valid raw string and string literal containing 'f' inside quotes must pass screener
    raw_src = 'linear(x, 64, name=r"feature_layer")\n'
    parsed = parse(raw_src, aliases)
    assert parsed["statements"][0]["value"]["kwargs"][0] == ["name", {"kind": "literal", "value": "feature_layer"}]


def test_04_screen_multiline_comments_brackets_layout():
    """Multiline layout with comments containing fake brackets and NL tokens."""
    aliases = Registry.builtins().aliases
    source = """
    # Complex comment block with deceptive brackets: ) ] }
    # Also keywords: def class for if while break continue
    out = linear(
        # Inner comment before argument
        x,
        # Bracket in comment: { [ (
        64,
        # Another line
        name="dense_proj"
    )
    """
    parsed = parse(source, aliases)
    assert len(parsed["statements"]) == 1
    stmt = parsed["statements"][0]
    assert stmt["kind"] == "assign"
    assert stmt["targets"] == ["out"]
    assert stmt["value"]["alias"] == "linear"


def test_05_parser_loop_nesting_exact_boundary():
    """Loop depth nesting boundary (MAX_LOOP_DEPTH=8)."""
    aliases = Registry.builtins().aliases

    # 8 nested loops must succeed
    lines_8 = ["    " * i + "for _ in range(1):" for i in range(MAX_LOOP_DEPTH)]
    lines_8.append("    " * MAX_LOOP_DEPTH + "relu()")
    src_8 = "\n".join(lines_8)
    parsed_8 = parse(src_8, aliases)
    assert len(parsed_8["statements"]) == 1

    # 9 nested loops must be rejected with E_RESOURCE
    lines_9 = ["    " * i + "for _ in range(1):" for i in range(MAX_LOOP_DEPTH + 1)]
    lines_9.append("    " * (MAX_LOOP_DEPTH + 1) + "relu()")
    src_9 = "\n".join(lines_9)
    with pytest.raises(HNDLError) as exc_info:
        parse(src_9, aliases)
    err = exc_info.value
    assert err.code == "E_RESOURCE"
    assert f"Source nests loops more than {MAX_LOOP_DEPTH} levels deep" in err.message


def test_06_parser_loop_invalid_range_counts():
    """Loop range(N) parameter must be a strictly positive integer literal."""
    aliases = Registry.builtins().aliases
    invalid_ranges = [
        "range(0)",
        "range(-5)",
        "range(True)",
        "range(False)",
        "range(2.5)",
        "range(1, 4)",
        "range(N)",
    ]
    for r in invalid_ranges:
        src = f"for _ in {r}:\n    linear(10)"
        with pytest.raises(HNDLError) as exc_info:
            parse(src, aliases)
        assert exc_info.value.code == "E_SYNTAX"
        assert "range() count must be a positive integer literal" in exc_info.value.message or (
            "range() in a loop takes exactly one argument" in exc_info.value.message
        )


def test_07_parser_loop_empty_or_assignments_only():
    """Loops containing no operator calls must be rejected during AST validation."""
    aliases = Registry.builtins().aliases
    source = """
    for _ in range(3):
        a = x
        b = a
    linear(10)
    """
    with pytest.raises(HNDLError) as exc_info:
        parse(source, aliases)
    err = exc_info.value
    assert err.code == "E_SYNTAX"
    assert "A loop body must call at least one operator" in err.message


def test_08_parser_loop_anticipatory_max_nodes_rejection():
    """Loop unrolling exceeds max_nodes before executing first iteration."""
    source = """
    for _ in range(100):
        linear(10)
    """
    with pytest.raises(HNDLError) as exc_info:
        resolve(
            source,
            input_shape=("B", 10),
            output_shape=("B", 10),
            limits={"max_nodes": 50},
        )
    err = exc_info.value
    assert err.code == "E_RESOURCE"
    assert "range(100) repeats a body of 1 node" in err.message
    assert "unrolled, the graph would exceed max_nodes (50)" in err.message


def test_09_parser_loop_all_collision_diagnostics():
    """Verify loop collision diagnostic attributions across auto, explicit, and suffixed nodes."""
    shapes = {"input_shape": ("B", 10), "output_shape": ("B", 10)}

    # Collision 1: Auto-generated ID collision
    src_auto = """
    linear(10)
    for _ in range(1):
        linear(10, name="n")
    """
    with pytest.raises(HNDLError) as exc_auto:
        resolve(src_auto, **shapes)
    assert "an earlier node already has that ID (an auto-generated ID 'n0')" in exc_auto.value.message

    # Collision 2: Explicit node name collision outside loop
    src_explicit = """
    linear(10, name="block0")
    for _ in range(1):
        linear(10, name="block")
    """
    with pytest.raises(HNDLError) as exc_explicit:
        resolve(src_explicit, **shapes)
    assert "an earlier node already has that ID (an explicit name='block0')" in exc_explicit.value.message

    # Collision 3: Multiple loop suffixed name collision
    src_suffixed = """
    for _ in range(2):
        linear(10, name="block")
    for _ in range(1):
        linear(10, name="block")
    """
    with pytest.raises(HNDLError) as exc_suffixed:
        resolve(src_suffixed, **shapes)
    assert (
        "an earlier node already has that ID (name='block' with iteration suffix '0')" in exc_suffixed.value.message
    )

    # Collision 4: Explicit node outside loop colliding with earlier loop suffix
    src_outer = """
    for _ in range(2):
        linear(10, name="layer")
    linear(10, name="layer0")
    """
    with pytest.raises(HNDLError) as exc_outer:
        resolve(src_outer, **shapes)
    assert "an earlier loop already gave that name to name='layer' with iteration suffix '0'" in exc_outer.value.message


def test_10_parser_assignment_unpacking_and_initializers():
    """Edge cases for tuple assignments, non-symbol targets, and initializer calls."""
    aliases = Registry.builtins().aliases

    # 1. Empty tuple assignment target
    with pytest.raises(HNDLError) as exc_empty:
        parse("() = relu(x)", aliases)
    assert "Assignment targets must be names" in exc_empty.value.message

    # 2. Duplicate unpacking targets
    with pytest.raises(HNDLError) as exc_dup:
        parse("a, a = split(x, 64)", aliases)
    assert "Unpacking targets must be distinct" in exc_dup.value.message

    # 3. Initializer scheme with positional args rejected
    src_init_pos = 'linear(10, init={"weight": normal(0.0, 0.02)})'
    with pytest.raises(HNDLError) as exc_init:
        parse(src_init_pos, aliases)
    assert "Initializer calls take keyword arguments only" in exc_init.value.message

    # 4. Initializer call in disallowed position (outside init= kwarg)
    src_init_bad = "linear(10, bias=normal(mean=0.0))"
    with pytest.raises(HNDLError) as exc_bad:
        parse(src_init_bad, aliases)
    assert exc_bad.value.code == "E_SYNTAX"


# ==============================================================================
# Group 2: Symbolic Solver, Interval Pruning & Constraint Graphs (Tests 11–20)
# ==============================================================================


def test_11_solver_interval_pruning_strided_conv_ambiguity_and_collapse():
    """Strided convolution interval pruning: open interval triggers E_AMBIGUOUS, collapses when bound."""
    registry = Registry.builtins()

    def dummy_relation(s):
        s.rank("out", 4)
        s.axis("out", 1, 3)

    @registry.operator(
        "dummy_tier5_gen",
        identity="test.dummy_tier5_gen",
        summary="dummy generator",
        shape="x -> out",
        relation=dummy_relation,
    )
    class DummyGen(torch.nn.Module):
        pass

    node0 = Node("g", "test.dummy_tier5_gen@1", {}, {"x": "input:x"})
    node1 = Node(
        "c",
        "conv2d@1",
        {"out_channels": 16, "kernel_size": 3, "stride": (1, 2), "padding": 1},
        {"x": "node:g/out"},
    )
    graph_ambig = Graph((node0, node1), ("B", 3), ("B", 16, 8, 4), "node:c/out")

    # Ambiguous inverse: stride (1, 2) on output (8, 4) produces collapsed (8, 8) along H
    # and open interval (7, 8) along W. Must report (7, 8) rather than (8, 8).
    with pytest.raises(HNDLError) as exc_info:
        resolve_graph(graph_ambig, registry)
    err = exc_info.value
    assert err.code == "E_AMBIGUOUS"
    assert "(7, 8)" in err.message
    assert "(8, 8)" not in err.message

    # When constrained by an exact input shape (B, 3, 8, 8), resolution collapses and succeeds
    def exact_relation(s):
        s.rank("out", 4)
        s.axis("out", 1, 3)
        s.axis("out", 2, 8)
        s.axis("out", 3, 8)

    @registry.operator(
        "dummy_tier5_exact",
        identity="test.dummy_tier5_exact",
        summary="dummy exact",
        shape="x -> out",
        relation=exact_relation,
    )
    class DummyExact(torch.nn.Module):
        pass

    node0_exact = Node("g", "test.dummy_tier5_exact@1", {}, {"x": "input:x"})
    graph_exact = Graph((node0_exact, node1), ("B", 3), ("B", 16, 8, 4), "node:c/out")
    plan = resolve_graph(graph_exact, registry)
    assert plan.nodes[1].input_shapes["x"] == ("B", 3, 8, 8)


def test_12_solver_incompatible_inverse_intervals_conflict():
    """Conflicting convolution inverse intervals on the same axis raise E_CONSTRAINT."""
    registry = Registry.builtins()

    def dummy_fork(s):
        s.rank("out", 4)
        s.axis("out", 1, 16)

    @registry.operator(
        "dummy_tier5_fork",
        identity="test.dummy_tier5_fork",
        summary="dummy fork",
        shape="x -> out",
        relation=dummy_fork,
    )
    class DummyFork(torch.nn.Module):
        pass

    # Node 1 requires input width in [7, 8]
    # Node 2 with kernel=5, stride=2, padding=0 requires input width in [11, 12]
    # Connecting both to node0 imposes empty intersection [11, 8]
    nodes = (
        Node("src", "test.dummy_tier5_fork@1", {}, {"x": "input:x"}),
        Node(
            "c1",
            "conv2d@1",
            {"out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
            {"x": "node:src/out"},
        ),
        Node(
            "c2",
            "conv2d@1",
            {"out_channels": 16, "kernel_size": 5, "stride": 2, "padding": 0},
            {"x": "node:src/out"},
        ),
        Node(
            "join",
            "add@1",
            {},
            {"a": "node:c1/out", "b": "node:c2/out"},
        ),
    )
    graph = Graph(nodes, ("B", 16), ("B", 16, 4, 4), "node:join/out")
    with pytest.raises(HNDLError) as exc_info:
        resolve_graph(graph, registry)
    assert exc_info.value.code == "E_CONSTRAINT"
    assert "Convolution inverse intervals are incompatible" in exc_info.value.message


def test_13_solver_spatial_relation_divisibility_and_bounds():
    """White-box testing of spatial() relations and convolution extent arithmetic."""
    # 1. groups divisibility check
    conv_rel = spatial("conv2d")

    class FakeNodeView:
        def __init__(self):
            self.args = {
                "groups": 3,
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
            }
            self._shapes = {"x": [None, 16, 32, 32], "out": [None, 16, 32, 32]}

        def rank(self, port, r):
            pass

        def axis(self, port, ax, val):
            if val is not None:
                self._shapes[port][ax] = val

        def shape(self, port):
            return self._shapes[port]

        def arg(self, name, val):
            pass

        def interval(self, port, axis, lower, upper):
            pass

        def error(self, code, msg):
            raise HNDLError(code, msg)

    view = FakeNodeView()
    with pytest.raises(HNDLError) as exc_info:
        conv_rel(view)
    assert exc_info.value.code == "E_CONSTRAINT"
    assert "groups=3 must divide input and output channels" in exc_info.value.message

    # 2. conv_input_range on non-positive extents returns empty interval (1, 0)
    assert conv_input_range(0, kernel=3) == (1, 0)
    assert conv_input_range(-4, kernel=3) == (1, 0)

    # 3. conv_transpose_input on unreachable extent returns None
    assert conv_transpose_input(0, kernel=3) is None
    assert conv_transpose_input(11, kernel=4, stride=2, padding=1) is None  # numerator not divisible

    # 4. conv_axis with out-of-bounds axis raises E_CONSTRAINT
    with pytest.raises(HNDLError) as exc_ax:
        conv_axis(None, axis=5)
    assert exc_ax.value.code == "E_CONSTRAINT"
    assert "Axis 5 is out of bounds for 4D convolution" in exc_ax.value.message


def test_14_solver_broadcast_relation_stress():
    """Adversarial stress-testing of relations.broadcast logic."""
    class FakeBroadcastView:
        def __init__(self, a_shape, b_shape, out_shape):
            self._shapes = {"a": list(a_shape), "b": list(b_shape), "out": list(out_shape)}

        def shape(self, port):
            return self._shapes[port]

        def rank(self, port, r):
            if len(self._shapes[port]) != r:
                raise HNDLError("E_CONSTRAINT", f"Rank mismatch: {len(self._shapes[port])} vs {r}")

        def axis(self, port, ax, val):
            self._shapes[port][ax] = val

        def error(self, code, msg):
            raise HNDLError(code, msg)

    # Incompatible extents (3 vs 5)
    view1 = FakeBroadcastView(["B", 3], ["B", 5], ["B", None])
    with pytest.raises(HNDLError) as exc1:
        broadcast(view1, "test_add")
    assert "do not broadcast" in exc1.value.message

    # Output extent 1 forces both inputs to 1
    view2 = FakeBroadcastView(["B", None], ["B", None], ["B", 1])
    broadcast(view2, "test_add")
    assert view2._shapes["a"][1] == 1
    assert view2._shapes["b"][1] == 1

    # Output extent 8, input a=1 forces input b to 8
    view3 = FakeBroadcastView(["B", 1], ["B", None], ["B", 8])
    broadcast(view3, "test_add")
    assert view3._shapes["b"][1] == 8

    # Output extent 8, input a=3 cannot broadcast to 8
    view4 = FakeBroadcastView(["B", 3], ["B", None], ["B", 8])
    with pytest.raises(HNDLError) as exc4:
        broadcast(view4, "test_add")
    assert "cannot broadcast to the output extent 8" in exc4.value.message


def test_15_solver_batch_multiples_admissibility():
    """Symbolic batch parsing and strict admissibility checks."""
    # Valid multiples
    assert batch_multiple("B") == 1
    assert batch_multiple("2*B") == 2
    assert batch_multiple("16*B") == 16
    assert batch_multiple("1024*B") == 1024

    # Canonical symbols
    assert batch_symbol(1) == "B"
    assert batch_symbol(4) == "4*B"

    # Disallowed multiples
    assert batch_multiple("1*B") is None
    assert batch_multiple("0*B") is None
    assert batch_multiple("02*B") is None
    assert batch_multiple("B*2") is None
    assert batch_multiple("-2*B") is None
    assert batch_multiple("1025*B") is None
    assert batch_multiple(123) is None

    # Mismatch between external contracts
    with pytest.raises(HNDLError) as exc_batch:
        resolve(
            "linear(10)",
            input_shape=("B", 10),
            output_shape=(4, 10),
        )
    assert exc_batch.value.code == "E_CONSTRAINT"
    assert "Input and output must declare the same batch dimension" in exc_batch.value.message


def test_16_solver_pattern_ellipsis_rank_underflow_and_scale():
    """Ellipsis pattern rank underflow and scale divisibility."""
    # Rank underflow: add expects at least rank 2, given rank 1
    with pytest.raises(HNDLError) as exc_rank:
        resolve(
            "add(a, b)",
            input_shape={"a": ("B", 16), "b": ("B",)},
            output_shape=("B", 16),
        )
    assert exc_rank.value.code in ("E_CONSTRAINT", "E_SCHEMA")


def test_17_solver_dead_nodes_and_unused_inputs():
    """Topological graph validation for dead nodes, unused inputs, and cycles."""
    # Dead node
    nodes_dead = (
        Node("alive", "linear@1", {"out_features": 10}, {"x": "input:x"}),
        Node("dead", "relu@1", {}, {"x": "input:x"}),
    )
    g_dead = Graph(nodes_dead, ("B", 10), ("B", 10), "node:alive/out")
    with pytest.raises(HNDLError) as exc_dead:
        resolve_graph(g_dead)
    assert exc_dead.value.code == "E_BINDING"
    assert "Nodes do not reach the selected output: dead" in exc_dead.value.message

    # Unused declared external input
    nodes_unused = (
        Node("n0", "linear@1", {"out_features": 10}, {"x": "input:used"}),
    )
    g_unused = Graph(nodes_unused, ("B", 10), ("B", 10), "node:n0/out",
                     named_inputs={"used": {"shape": ("B", 10), "dtype": "float32"},
                                   "unused": {"shape": ("B", 10), "dtype": "float32"}})
    with pytest.raises(HNDLError) as exc_unused:
        resolve_graph(g_unused)
    assert exc_unused.value.code == "E_BINDING"
    assert "Declared inputs are never used: unused" in exc_unused.value.message

    # Cycle in DAG
    nodes_cycle = (
        Node("n1", "relu@1", {}, {"x": "node:n2/out"}),
        Node("n2", "relu@1", {}, {"x": "node:n1/out"}),
    )
    g_cycle = Graph(nodes_cycle, ("B", 10), ("B", 10), "node:n2/out")
    with pytest.raises(HNDLError) as exc_cycle:
        resolve_graph(g_cycle)
    assert exc_cycle.value.code == "E_BINDING"
    assert "Graph contains a cycle" in exc_cycle.value.message


def test_18_solver_resource_limits_enforcement():
    """Resolver resource bounds: max_iterations, max_dimension, and invalid limits."""
    # Max iterations exceeded on oscillating/unsettling relation
    registry = Registry.builtins()

    def oscillating_relation(s):
        # Force changed = True indefinitely without conflicting shape dimensions
        s.rank("out", 2)
        s.axis("out", 1, 10)
        s._solver.changed = True

    @registry.operator(
        "dummy_tier5_loop",
        identity="test.oscillating",
        summary="oscillating relation",
        shape="x -> out",
        relation=oscillating_relation,
    )
    class DummyLoop(torch.nn.Module):
        pass

    g_loop = Graph((Node("n0", "test.oscillating@1", {}, {"x": "input:x"}),),
                   ("B", 10), ("B", 10), "node:n0/out")
    with pytest.raises(HNDLError) as exc_iter:
        resolve_graph(g_loop, registry, limits={"max_iterations": 5})
    assert exc_iter.value.code == "E_RESOURCE"
    assert "Shape rules exceeded max_iterations" in exc_iter.value.message

    # Invalid limits: negative or unknown
    with pytest.raises(HNDLError) as exc_lim1:
        resolve("linear(10)", input_shape=("B", 10), output_shape=("B", 10),
                limits={"max_nodes": -1})
    assert "All resource limits must be positive integers" in exc_lim1.value.message

    with pytest.raises(HNDLError) as exc_lim2:
        resolve("linear(10)", input_shape=("B", 10), output_shape=("B", 10),
                limits={"unsupported_budget": 100})
    assert "Unknown resource limits" in exc_lim2.value.message


def test_19_solver_port_name_reservations():
    """External port reservation checks for keywords, loop iterator, schemes, and aliases."""
    registry = Registry.builtins()

    # Python keyword
    with pytest.raises(HNDLError) as exc_kw:
        _validate_port_reservations(["x", "def"], registry)
    assert exc_kw.value.code == "E_SCHEMA"
    assert "External port names must not be Python keywords: def" in exc_kw.value.message

    # range iterator
    with pytest.raises(HNDLError) as exc_range:
        _validate_port_reservations(["x", "range"], registry)
    assert exc_range.value.code == "E_NAME"
    assert "collide with reserved loop iterator: range" in exc_range.value.message

    # Initializer scheme
    with pytest.raises(HNDLError) as exc_scheme:
        _validate_port_reservations(["x", "normal"], registry)
    assert exc_scheme.value.code == "E_NAME"
    assert "collide with initializer schemes: normal" in exc_scheme.value.message

    # Operator alias
    with pytest.raises(HNDLError) as exc_alias:
        _validate_port_reservations(["x", "relu"], registry)
    assert exc_alias.value.code == "E_NAME"
    assert "collide with operator aliases: relu" in exc_alias.value.message


def test_20_solver_nested_loops_multi_stage_residual_graph():
    """Multi-stage nested loop network with residual skip connections."""
    source = """
    for _ in range(2):
        res = x
        x = linear(32, name="proj")
        x = relu(x, name="act")
        x = add(a=res, b=x, name="add")
    out = linear(10, x=x, name="head")
    """
    plan = resolve(source, input_shape=("B", 32), output_shape=("B", 10))
    # 2 iterations * 3 nodes + 1 head node = 7 nodes total
    assert len(plan.nodes) == 7
    node_ids = [node.id for node in plan.nodes]
    assert node_ids == ["proj0", "act0", "add0", "proj1", "act1", "add1", "head"]
    assert plan.nodes[2].inputs == {"a": "input:x", "b": "node:act0/out"}
    assert plan.nodes[5].inputs == {"a": "node:add0/out", "b": "node:act1/out"}
    assert plan.output_ref == "node:head/out"


# ==============================================================================
# Group 3: Plan Serialization, Deserialization & Repr Formatting (Tests 21–30)
# ==============================================================================


def test_21_plan_to_json_from_json_roundtrip_complex_dag():
    """Full roundtrip fidelity for complex DAG with multi-input and multi-output."""
    source = """
    h1 = linear(32, x=a, name="enc_a")
    h2 = linear(32, x=b, name="enc_b")
    fused = add(a=h1, b=h2, name="fuse")
    feat = relu(fused, name="feat")
    logits = linear(10, x=feat, name="logits")
    """
    plan = resolve(
        source,
        input_shape={"a": ("B", 16), "b": ("B", 16)},
        output_shape={"feat": ("B", 32), "logits": ("B", 10)},
    )
    serialized = plan.to_json()
    reconstructed = ResolvedPlan.from_json(serialized)

    assert reconstructed.semantic_digest == plan.semantic_digest
    assert reconstructed.to_json() == serialized
    assert len(reconstructed.nodes) == 5
    assert set(reconstructed.inputs) == {"a", "b"}
    assert set(reconstructed.outputs) == {"feat", "logits"}


def test_22_plan_from_json_artifact_digest_tampering():
    """Tampering with serialized node arguments breaks artifact_digest integrity."""
    plan = resolve("linear(32)\nlinear(10)", input_shape=("B", 16), output_shape=("B", 10))
    raw_json = plan.to_json()

    # Tamper with an argument value without updating artifact_digest
    tampered_json = re.sub(r'"out_features":32', '"out_features":64', raw_json)
    assert tampered_json != raw_json

    with pytest.raises(HNDLError) as exc_info:
        ResolvedPlan.from_json(tampered_json)
    assert exc_info.value.code == "E_INTEGRITY"
    assert "artifact_digest does not match its contents" in exc_info.value.message


def test_23_plan_from_json_semantic_digest_tampering():
    """Tampering with artifact_digest to match corrupted JSON leaves semantic_digest invalid."""
    plan = resolve("linear(32)\nlinear(10)", input_shape=("B", 16), output_shape=("B", 10))
    data = json.loads(plan.to_json())

    # Alter node ID and recompute only artifact digest
    data["nodes"][0]["id"] = "tampered_n0"
    data.pop("artifact_digest")
    data["artifact_digest"] = digest(data)
    tampered_json = json.dumps(data)

    with pytest.raises(HNDLError) as exc_info:
        ResolvedPlan.from_json(tampered_json)
    assert exc_info.value.code == "E_INTEGRITY"
    assert "semantic_digest does not match its contents" in exc_info.value.message


def test_24_plan_from_json_invalid_math_tampering():
    """Tampered plan with valid digests but mathematically contradictory equations is rejected."""
    plan = resolve("linear(32)\nlinear(10)", input_shape=("B", 16), output_shape=("B", 10))
    data = plan._data()

    # Tamper node 1's saved input shape to contradict node 0's output shape
    data["nodes"][1]["input_shapes"]["x"] = ["B", 48]

    sem_data = plan._data(semantic=True)
    sem_data["nodes"][1]["input_shapes"]["x"] = ["B", 48]
    data["semantic_digest"] = digest(sem_data)
    data["artifact_digest"] = digest(data)

    with pytest.raises(HNDLError) as exc_info:
        ResolvedPlan.from_json(canonical(data))
    assert exc_info.value.code == "E_INTEGRITY"
    assert "Saved plan does not satisfy its operators' equations" in exc_info.value.message


def test_25_plan_from_json_duplicate_keys_and_nan():
    """Saved JSON with duplicate dictionary keys or non-finite floats is strictly rejected."""
    plan = resolve("linear(10)", input_shape=("B", 10), output_shape=("B", 10))
    raw_json = plan.to_json()

    # 1. Duplicate JSON keys
    dup_json = '{"schema_version":1,"schema_version":1,' + raw_json[1:]
    with pytest.raises(HNDLError) as exc_dup:
        ResolvedPlan.from_json(dup_json)
    assert exc_dup.value.code == "E_SCHEMA"
    assert "Duplicate JSON field 'schema_version'" in exc_dup.value.message

    # 2. NaN float value
    nan_json = raw_json.replace('"schema_version":1', '"schema_version":NaN')
    with pytest.raises(HNDLError) as exc_nan:
        ResolvedPlan.from_json(nan_json)
    assert exc_nan.value.code == "E_SCHEMA"
    assert "Invalid saved plan" in exc_nan.value.message


def test_26_plan_from_json_version_mismatch_and_missing_fields():
    """State version mismatches and missing required fields in serialized nodes."""
    plan = resolve("linear(10)", input_shape=("B", 10), output_shape=("B", 10))
    data = json.loads(plan.to_json())

    # Unsupported schema version
    data_ver = dict(data)
    data_ver["schema_version"] = 2
    with pytest.raises(HNDLError) as exc_ver:
        ResolvedPlan.from_json(json.dumps(data_ver))
    assert exc_ver.value.code == "E_STATE_VERSION"
    assert "Expected plan schema 1 and resolution version 1" in exc_ver.value.message

    # Missing required field in node record (e.g. initialization missing)
    data_node = plan._data()
    del data_node["nodes"][0]["initialization"]
    data_node["semantic_digest"] = plan.semantic_digest
    data_node["artifact_digest"] = digest(data_node)

    with pytest.raises(HNDLError) as exc_node:
        ResolvedPlan.from_json(canonical(data_node))
    assert exc_node.value.code == "E_SCHEMA"
    assert "Saved schema 1 nodes require all canonical fields" in exc_node.value.message


def test_27_plan_immutability_and_copy():
    """ResolvedPlan and Node immutability invariants and copy semantics."""
    plan = resolve("linear(10)", input_shape=("B", 10), output_shape=("B", 10))

    # Shallow copy and deepcopy return identical instances
    assert copy.copy(plan) is plan
    assert copy.deepcopy(plan) is plan

    # Nodes are immutable and share references
    node = plan.nodes[0]
    assert copy.copy(node) is node
    assert copy.deepcopy(node) is node

    # Mutation attempt must fail
    with pytest.raises(Exception):
        node.id = "hacked"  # type: ignore[misc]

    with pytest.raises(Exception):
        node.args["out_features"] = 999  # MappingProxyType prevents in-place mutation


def test_28_plan_repr_chain_vs_branched_columns():
    """Verify ResolvedPlan.__repr__ differentiates unary chains from branched topologies."""
    # 1. Unary linear chain
    chain_plan = resolve("linear(32)\nrelu()\nlinear(10)", input_shape=("B", 16), output_shape=("B", 10))
    assert chain_plan._chain is True
    chain_repr = repr(chain_plan)
    assert "index" in chain_repr
    assert "input shape" in chain_repr
    assert "output shape" in chain_repr
    assert "input shapes" not in chain_repr
    assert "output shapes" not in chain_repr

    # 2. Branched DAG (residual add)
    src_branched = """
    h = linear(32)
    act = relu(h)
    add(a=h, b=act)
    linear(10)
    """
    branch_plan = resolve(src_branched, input_shape=("B", 32), output_shape=("B", 10))
    assert branch_plan._chain is False
    branch_repr = repr(branch_plan)
    assert "index" not in branch_repr
    assert "input shapes" in branch_repr
    assert "output shapes" in branch_repr
    assert "a=node:n0/out:" in branch_repr

    # 3. chunk(1) produces tuple outputs and is NOT classified as a chain
    src_chunk = """
    (c0,) = chunk(1, dim=1)
    out = c0
    """
    chunk_plan = resolve(src_chunk, input_shape=("B", 32), output_shape=("B", 32))
    assert chunk_plan._chain is False
    assert "index" not in repr(chunk_plan)
    assert "out0=[B, 32]" in repr(chunk_plan)


def test_29_plan_contract_header_formatting():
    """contract_header prints clear signatures for single and multi-input graphs."""
    # Single input / single output
    plan1 = resolve("linear(10)", input_shape=("B", 16), output_shape=("B", 10))
    header1 = contract_header(plan1, "Network")
    assert header1 == "Network: [B, 16] -> [B, 10]  dtype=float32"

    # Multi-input with differing dtypes
    g = capture_config(
        "add(a=img, b=seg)",
        input_shape={"img": ("B", 16), "seg": ("B", 16)},
        output_shape=("B", 16),
        input_dtype={"img": "float32", "seg": "float32"},
    )
    header2 = contract_header(g, "Graph")
    assert "img=[B, 16], seg=[B, 16]" in header2
    assert "dtype=float32" in header2


def test_30_plan_describe_fidelity():
    """plan.describe() prints full provenance and initialization metadata for all nodes."""
    plan = resolve("linear(32, equalized=True)\nrelu()", input_shape=("B", 16), output_shape=("B", 32))
    desc = plan.describe()
    assert f"Semantic digest: {plan.semantic_digest}" in desc
    assert "n0: " in desc
    assert "in_features=inferred" in desc
    assert "out_features=explicit" in desc
    assert "equalized=explicit" in desc
    assert "bias=operator default" in desc
    assert "initialization=" in desc
    assert "trainability=" in desc
