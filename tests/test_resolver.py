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

    @registry.operator("silu", identity="example.silu", summary="SiLU.", shape="x[B, ...] -> out[B, ...]")
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
