from dataclasses import replace

import pytest

from torch import nn

from hndl import Registry
from hndl.errors import HNDLError
from hndl.resolver import resolve_graph
from hndl.types import Graph, Node


def chain(operations, input_shape, output_shape):
    nodes = []
    ref = "input:x"
    for i, (op, args) in enumerate(operations):
        node = Node(f"n{i}", f"{op}@1", args, {"x": ref})
        nodes.append(node)
        ref = f"node:n{i}/out"
    return Graph(tuple(nodes), input_shape, output_shape, ref)


def generator(height, width):
    operations = [
        ("linear", {}), ("relu", {}), ("reshape", {"shape": [512]}),
        ("conv_transpose2d", {"out_channels": 256, "policy": "up2"}), ("relu", {}),
        ("conv_transpose2d", {"out_channels": 128, "policy": "up2"}), ("relu", {}),
        ("conv_transpose2d", {"out_channels": 64, "policy": "up2"}), ("relu", {}),
        ("conv2d", {"out_channels": 3, "kernel_size": 3, "padding": 1}), ("tanh", {}),
    ]
    return chain(operations, ("B", 128), ("B", 3, height, width))


@pytest.mark.parametrize("height,width,projection", [(32, 32, 8192), (64, 64, 32768), (32, 64, 16384)])
def test_generator_resolves_two_seed_axes_from_target(height, width, projection):
    plan = resolve_graph(generator(height, width))
    assert plan.nodes[0].args["out_features"] == projection
    assert plan.nodes[2].args["shape"] == (512, height // 8, width // 8)
    assert plan.nodes[3].source["policy"] == "spatial.up2_transpose@1"
    assert plan.nodes[3].provenance["stride"] == "policy-selected"
    assert plan.nodes[0].provenance["bias"] == "operator default"
    assert plan.nodes[0].provenance["out_features"] == "inferred"


def test_invalid_target_and_literal_projection_are_not_repaired():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve_graph(generator(30, 30))
    graph = generator(32, 32)
    graph = replace(graph, nodes=(replace(graph.nodes[0], args={"out_features": 128}), *graph.nodes[1:]))
    with pytest.raises(HNDLError, match="E_RESHAPE"):
        resolve_graph(graph)


def test_bare_reshape_infers_rank_from_downstream_and_elements_from_input():
    plan = resolve_graph(chain([("reshape", {}), ("linear", {})], ("B", 2, 4, 4), ("B", 10)))
    assert plan.nodes[0].args["shape"] == (32,)
    assert plan.nodes[1].args["in_features"] == 32


def test_conv_inverse_ambiguity_is_not_arbitrarily_resolved():
    graph = chain([("linear", {}), ("reshape", {"shape": (1,)}),
                   ("conv2d", {"out_channels": 1, "kernel_size": 3, "stride": 2, "padding": 1})],
                  ("B", 16), ("B", 1, 4, 4))
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve_graph(graph)


def test_split_remainder_backprop_and_unused_ports():
    split = Node("partition", "split@1", {"size": 64}, {"x": "input:x"}, ("first", "rest"))
    graph = Graph((split,), ("B", 160), ("B", 96), "node:partition/rest")
    plan = resolve_graph(graph)
    assert plan.nodes[0].output_shapes == {"first": ("B", 64), "rest": ("B", 96)}
    inferred = resolve_graph(replace(graph, nodes=(replace(split, args={}),)))
    assert inferred.nodes[0].args["size"] == 64
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve_graph(replace(graph, nodes=(replace(split, args={"size": 160}),)))


def test_concat_backward_width_and_add_shape_conflict():
    nodes = (
        Node("a", "linear@1", {"out_features": 4}, {"x": "input:x"}),
        Node("b", "linear@1", {}, {"x": "input:x"}),
        Node("join", "concat@1", {"input_count": 2}, {"x0": "node:a/out", "x1": "node:b/out"}),
    )
    plan = resolve_graph(Graph(nodes, ("B", 8), ("B", 10), "node:join/out"))
    assert plan.nodes[1].args["out_features"] == 6
    bad = (*nodes[:2], Node("join", "add@1", {}, {"a": "node:a/out", "b": "node:b/out"}))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve_graph(Graph(bad, ("B", 8), ("B", 10), "node:join/out"))


def test_group_norm_divisibility_and_flatten():
    good = chain([("group_norm", {"num_groups": 2}), ("flatten", {})], ("B", 4, 3, 3), ("B", 36))
    plan = resolve_graph(good)
    assert plan.nodes[0].args["num_channels"] == 4
    with pytest.raises(HNDLError, match="must divide"):
        resolve_graph(replace(good, nodes=(replace(good.nodes[0], args={"num_groups": 3}), good.nodes[1])))


def test_custom_shape_preserving_operator_has_bidirectional_relations():
    registry = Registry.builtins()

    @registry.operator("my_silu", identity="example.silu", summary="SiLU.", shape="x[B, ...] -> out[B, ...]")
    class SiLU(nn.SiLU):
        pass

    graph = chain([("linear", {}), ("example.silu", {})], ("B", 128), ("B", 10))
    plan = resolve_graph(graph, registry)
    assert plan.nodes[0].args["out_features"] == 10
    assert plan.nodes[1].input_shapes["x"] == ("B", 10)


def test_identity_and_resource_limits():
    identity = Graph((), ("B", 4), ("B", 4), "input:x")
    assert resolve_graph(identity).nodes == ()
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve_graph(replace(identity, output_shape=("B", 5)))
    graph = chain([("linear", {})] * 5000, ("B", 4), ("B", 4))
    with pytest.raises(HNDLError, match="max_nodes"):
        resolve_graph(graph)


def test_cycle_and_dead_nodes_rejected_and_topology_stable():
    a = Node("a", "relu@1", {}, {"x": "node:b/out"})
    b = Node("b", "relu@1", {}, {"x": "node:a/out"})
    with pytest.raises(HNDLError, match="cycle"):
        resolve_graph(Graph((a, b), ("B", 4), ("B", 4), "node:a/out"))
    b = replace(b, inputs={"x": "input:x"})
    plan = resolve_graph(Graph((a, b), ("B", 4), ("B", 4), "node:a/out"))
    assert [node.id for node in plan.nodes] == ["b", "a"]
    with pytest.raises(HNDLError, match="do not reach"):
        resolve_graph(Graph((a, b), ("B", 4), ("B", 4), "node:b/out"))


def test_huge_concat_input_count_rejected_before_materializing_ports():
    node = Node("join", "concat@1", {"input_count": 2**31 - 1}, {})
    with pytest.raises(HNDLError, match="max_edges"):
        resolve_graph(Graph((node,), ("B", 4), ("B", 4), "node:join/out"))


def test_resolve_graph_rejects_reserved_ports():
    # Operator alias port
    graph_alias = Graph(nodes=(), input_shape=("B", 4), output_shape=("B", 4), output_ref="input:linear",
                        named_inputs={"linear": {"shape": ("B", 4), "dtype": "float32"}},
                        named_outputs={"output": {"shape": ("B", 4), "ref": "input:linear"}})
    with pytest.raises(HNDLError, match="E_NAME") as exc:
        resolve_graph(graph_alias)
    assert "collide with operator aliases: linear" in exc.value.message

    # range port
    graph_range = Graph(nodes=(), input_shape=("B", 4), output_shape=("B", 4), output_ref="input:range",
                        named_inputs={"range": {"shape": ("B", 4), "dtype": "float32"}},
                        named_outputs={"output": {"shape": ("B", 4), "ref": "input:range"}})
    with pytest.raises(HNDLError, match="E_NAME") as exc:
        resolve_graph(graph_range)
    assert "collide with reserved loop iterator: range" in exc.value.message

    # scheme port
    graph_scheme = Graph(nodes=(), input_shape=("B", 4), output_shape=("B", 4), output_ref="input:normal",
                         named_inputs={"normal": {"shape": ("B", 4), "dtype": "float32"}},
                         named_outputs={"output": {"shape": ("B", 4), "ref": "input:normal"}})
    with pytest.raises(HNDLError, match="E_NAME") as exc:
        resolve_graph(graph_scheme)
    assert "collide with initializer schemes: normal" in exc.value.message


def test_ambiguous_interval_reporting_ignores_collapsed_axes():
    """[M-07] E_AMBIGUOUS reports open intervals (lo < hi) instead of collapsed ones (lo == hi)."""
    registry = Registry.builtins()

    def dummy_relation(s):
        s.rank("out", 4)
        s.axis("out", 1, 3)

    @registry.operator("dummy_m07_gen", identity="dummy_m07_gen", summary="dummy",
                       shape="x -> out", relation=dummy_relation)
    class DummyM07Gen(nn.Module):
        pass

    # Node g outputs [B, 3, None, None].
    # Node c is conv2d with stride (1, 2) and out_channels 16.
    # Output shape is [B, 16, 8, 4].
    # Axis 2 (stride 1) collapses to (8, 8) -> resolves to 8.
    # Axis 3 (stride 2) has interval (7, 8) -> remains None (ambiguous).
    node0 = Node("g", "dummy_m07_gen@1", {}, {"x": "input:x"})
    node1 = Node("c", "conv2d@1", {"out_channels": 16, "kernel_size": 3, "stride": (1, 2), "padding": 1},
                 {"x": "node:g/out"})
    graph = Graph((node0, node1), ("B", 3), ("B", 16, 8, 4), "node:c/out")

    with pytest.raises(HNDLError) as exc_info:
        resolve_graph(graph, registry)

    error_msg = str(exc_info.value)
    assert exc_info.value.code == "E_AMBIGUOUS"
    assert "; inverse allows (7, 8)" in error_msg
    assert "(8, 8)" not in error_msg


def test_patterns_ellipsis_underflow_blames_responsible_port():
    """[L-03] _Solver.patterns blames the port with underflowing rank, not the first ellipsis port."""
    registry = Registry.builtins()

    def noop(s):
        pass

    @registry.operator("gen_unknown_l03", identity="gen_unknown_l03", summary="dummy",
                       shape="x -> out", relation=noop)
    class GenUnknownL03(nn.Module):
        pass

    @registry.operator("test_ellipsis_l03", identity="test_ellipsis_l03", summary="test",
                       shape="x[B, ..., C, D] -> out[B, ..., H, W]")
    class DummyEllipsisL03(nn.Module):
        pass

    # n0 produces out with shape None (unknown rank)
    # n1 requires pattern rank >= 3 on both x and out
    # n1/out is fixed to rank 2 ("B", 10), which underflows pattern rank
    n0 = Node("n0", "gen_unknown_l03@1", {}, {"x": "input:x"})
    n1 = Node("n1", "test_ellipsis_l03@1", {}, {"x": "node:n0/out"})
    graph = Graph((n0, n1), ("B", 4), ("B", 10), "node:n1/out")

    with pytest.raises(HNDLError) as exc_info:
        resolve_graph(graph, registry)

    error_msg = str(exc_info.value)
    assert exc_info.value.code == "E_CONSTRAINT"
    assert "Rank of out is below the declared pattern rank" in error_msg
    assert "Rank of x is below the declared pattern rank" not in error_msg


def test_solver_axis_and_interval_defensive_bounds():
    """[L-04] _Solver.axis and _Solver.interval reject negative and out-of-bounds axes cleanly."""
    from hndl.resolver import _Solver, DEFAULT_LIMITS

    solver = _Solver(Graph((), ("B", 3, 8, 8), ("B", 3, 8, 8), "input:x"), (), {}, DEFAULT_LIMITS)
    ref = "test_tensor"
    solver.shapes[ref] = ["B", 3, 8, 8]

    # Negative axis must not mutate or alias the last element
    solver.axis(ref, -1, 99)
    assert solver.shapes[ref] == ["B", 3, 8, 8]

    # Out-of-bounds axis must not raise IndexError
    solver.axis(ref, 10, 99)
    assert solver.shapes[ref] == ["B", 3, 8, 8]

    # Interval on negative axis must be a no-op
    solver.interval(ref, -1, 5, 5)
    assert solver.shapes[ref] == ["B", 3, 8, 8]
    assert (ref, -1) not in solver.intervals


