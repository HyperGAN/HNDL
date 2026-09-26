"""Bounded ``for _ in range(N):`` loops in declarative configs.

A loop is unrolled while the config is read, so it must produce exactly the
graph the same statements written out by hand produce. Only each node's
source location differs: it points at the statement inside the loop and
records the iteration it was unrolled from.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, ResolvedPlan, ops, resolve, resolve_callable
from hndl.config import _parse, capture_config
from hndl.torch import network, network_from_callable

ROOT = Path(__file__).resolve().parents[1]
TOKENS = {"input_shape": ("B", 6, 16), "output_shape": ("B", 6, 16)}


def _without_locations(plan):
    """``to_json`` minus what legitimately differs: where each node was written."""
    data = json.loads(plan.to_json())
    del data["artifact_digest"]  # a digest over the locations too
    for node in data["nodes"]:
        for key in ("line", "column", "iterations"):
            node["source"].pop(key, None)
    return data


def _assert_identical(looped, unrolled, **contract):
    contract = contract or TOKENS
    a, b = resolve(looped, **contract), resolve(unrolled, **contract)
    assert [node.id for node in a.nodes] == [node.id for node in b.nodes]
    assert a.semantic_digest == b.semantic_digest
    assert _without_locations(a) == _without_locations(b)
    torch.manual_seed(0)
    first = network(looped, device="cpu", **contract)
    second = network(unrolled, device="cpu", **contract)
    assert list(first.state_dict()) == list(second.state_dict())
    assert {key: tuple(value.shape) for key, value in first.state_dict().items()} == \
        {key: tuple(value.shape) for key, value in second.state_dict().items()}
    return a


# Acceptance and the identity property ---------------------------------------

def test_a_chain_in_a_loop_is_the_same_plan_as_the_chain_written_out():
    plan = _assert_identical("for _ in range(3):\n    linear(16)\n    relu()\n",
                             "linear(16)\nrelu()\n" * 3)
    assert [node.id for node in plan.nodes] == [f"n{i}" for i in range(6)]


def test_rebinding_inside_a_loop_builds_residual_blocks():
    looped = textwrap.dedent("""
        h = x
        for _ in range(8):
            h = add(h, feed_forward(layer_norm(h), hidden=32))
        out = h
    """)
    unrolled = "h = x\n" + "h = add(h, feed_forward(layer_norm(h), hidden=32))\n" * 8 + "out = h\n"
    plan = _assert_identical(looped, unrolled)
    assert len(plan.nodes) == 24
    # Each add reads the previous block's add, not the block input.
    adds = [node for node in plan.nodes if node.op.startswith("add@")]
    assert adds[1].inputs["a"] == f"node:{adds[0].id}/out"


def test_nested_loops_multiply_out_in_order():
    looped = textwrap.dedent("""
        for _ in range(2):
            linear(16)
            for _ in range(3):
                relu()
            layer_norm()
    """)
    unrolled = ("linear(16)\n" + "relu()\n" * 3 + "layer_norm()\n") * 2
    plan = _assert_identical(looped, unrolled)
    assert len(plan.nodes) == 10


def test_a_named_node_in_a_loop_gets_the_iteration_suffix():
    looped = "for _ in range(4):\n    transformer_block(4, name='block')\nlayer_norm(name='final')\n"
    unrolled = "".join(f"transformer_block(4, name='block{i}')\n" for i in range(4)) + "layer_norm(name='final')\n"
    plan = _assert_identical(looped, unrolled)
    assert [node.id for node in plan.nodes] == ["block0", "block1", "block2", "block3", "final"]


def test_nested_names_append_every_loop_index_outermost_first():
    looped = textwrap.dedent("""
        for _ in range(2):
            linear(16, name="down")
            for _ in range(3):
                h = linear(16, name="res")
                relu()
    """)
    unrolled = "".join(f"linear(16, name='down{i}')\n"
                       + "".join(f"h = linear(16, name='res{i}_{j}')\nrelu()\n" for j in range(3))
                       for i in range(2))
    plan = _assert_identical(looped, unrolled)
    assert [node.id for node in plan.nodes if not node.id.startswith("n")] == [
        "down0", "res0_0", "res0_1", "res0_2", "down1", "res1_0", "res1_1", "res1_2"]


def test_loops_accept_every_top_level_statement_and_indentation():
    looped = textwrap.indent(textwrap.dedent("""
        h = x
        for _ in range(2):
            a, b = split(8)
            h = concat(relu(a), b)
            for _ in range(1):
                h = linear(h, 16)
            out = h
    """), "        ")
    unrolled = "h = x\n" + "a, b = split(8)\nh = concat(relu(a), b)\nh = linear(h, 16)\nout = h\n" * 2
    _assert_identical(looped, unrolled, input_shape=("B", 16), output_shape=("B", 16))


def test_the_parser_keeps_one_record_per_loop_and_counts_its_nodes():
    program = _parse("for _ in range(5):\n    for _ in range(3):\n        h = add(relu(), linear(8))\n",
                     Registry.builtins().aliases)
    (loop,) = program["statements"]
    assert (loop["kind"], loop["count"], loop["nodes"]) == ("for", 5, 45)
    assert loop["body"][0]["nodes"] == 9


def test_the_example_networks_match_their_hand_written_blocks():
    fixture = json.loads((ROOT / "tests/fixtures/plans_0_6_0/gpt_tiny.json").read_text(encoding="utf-8"))
    looped = (ROOT / "examples/networks/gpt_tiny.hndl").read_text(encoding="utf-8")
    assert "for _ in range(4):" in looped and 'name="block0"' in fixture["source"]
    contract = {"input_shape": ("B", 64), "output_shape": ("B", 64, 256), "input_dtype": "int64"}
    a, b = resolve(looped, **contract), resolve(fixture["source"], **contract)
    assert [node.id for node in a.nodes] == [node.id for node in b.nodes]
    assert [node.id for node in a.nodes][2:6] == ["block0", "block1", "block2", "block3"]
    assert a.semantic_digest == b.semantic_digest == json.loads(fixture["plan"])["semantic_digest"]
    assert _without_locations(a) == _without_locations(b)


# The native frontend -------------------------------------------------------

def test_a_python_loop_in_the_native_frontend_gives_the_same_plan():
    looped = textwrap.dedent("""
        h = x
        for _ in range(3):
            h = add(h, feed_forward(layer_norm(h), hidden=32), name="block")
            for _ in range(2):
                h = linear(h, 16, name="proj")
    """)

    def author(x):
        h = x
        for i in range(3):
            h = ops.add(h, ops.feed_forward(ops.layer_norm(h), hidden=32), name=f"block{i}")
            for j in range(2):
                h = ops.linear(h, 16, name=f"proj{i}_{j}")
        return h

    config, native = resolve(looped, **TOKENS), resolve_callable(author, **TOKENS)
    assert [node.id for node in config.nodes] == [node.id for node in native.nodes]
    assert config.semantic_digest == native.semantic_digest
    first = network(looped, device="cpu", **TOKENS)
    second = network_from_callable(author, device="cpu", **TOKENS)
    assert list(first.state_dict()) == list(second.state_dict())


# Rejections ----------------------------------------------------------------

@pytest.mark.parametrize("source, construct", [
    ("for i in range(3):\n    relu()", "loop target must be _"),
    ("for (a, b) in range(3):\n    relu()", "loop target must be _"),
    ("for _ in range(3, 4):\n    relu()", "exactly one argument"),
    ("for _ in range():\n    relu()", "exactly one argument"),
    ("for _ in range(stop=3):\n    relu()", "exactly one argument"),
    ("for _ in range(0):\n    relu()", "positive integer literal"),
    ("for _ in range(-2):\n    relu()", "positive integer literal"),
    ("for _ in range(True):\n    relu()", "positive integer literal"),
    ("for _ in range(2.0):\n    relu()", "positive integer literal"),
    ("for _ in range(n):\n    relu()", "positive integer literal"),
    ("for _ in range(2 * 3):\n    relu()", "positive integer literal"),
    ("for _ in [1, 2]:\n    relu()", "iterate over range(N)"),
    ("for _ in split(2):\n    relu()", "iterate over range(N)"),
    ("for _ in range(3):\n    relu()\nelse:\n    relu()", "for ... else"),
    ("for _ in range(3):\n    relu()\n    break", "break is not permitted"),
    ("for _ in range(3):\n    continue", "continue is not permitted"),
    ("while True:\n    relu()", "while loops are not permitted"),
    ("if x:\n    relu()", "if statements are not permitted"),
    ("for _ in range(2):\n    if x:\n        relu()", "if statements are not permitted"),
    ("linear(4 if x else 8)", "Conditional expressions"),
    ("out = [relu() for _ in range(3)]", "List comprehensions"),
    ("linear(out_features={1 for _ in range(3)})", "Set comprehensions"),
    ("linear(init={k: 1 for _ in range(3)})", "Dict comprehensions"),
    ("add(*(relu() for _ in range(3)))", "Star argument"),
    ("linear(sum(1 for _ in range(3)))", "Generator expressions"),
    ("for _ in range(3):\n    linear(_)", "_ is the loop placeholder"),
    ("for _ in range(3):\n    h = _", "_ is the loop placeholder"),
    ("for _ in range(3):\n    relu(name=_)", "_ is the loop placeholder"),
    ("relu(x=_)", "_ is the loop placeholder"),
    ("linear(range(3))", "range is only permitted as the loop iterator"),
    ("h = range(3)", "range is only permitted as the loop iterator"),
    ("linear(init={'weight': range(3)})", "range is only permitted as the loop iterator"),
    ("range = relu()", "range cannot be rebound"),
    ("h = range", "range is only permitted as the loop iterator"),
    ("h = x\nfor _ in range(1000000000):\n    h = x", "must call at least one operator"),
    ("for _ in range(3):\n    for _ in range(3):\n        out = x", "must call at least one operator"),
    ("async def f():\n    async for _ in range(3):\n        relu()", "outside the declarative Python subset"),
])
def test_everything_but_the_one_loop_form_is_rejected_by_name(source, construct):
    with pytest.raises(HNDLError, match="E_SYNTAX") as rejected:
        capture_config(source, **TOKENS)
    assert construct in rejected.value.message
    assert rejected.value.line is not None


def test_range_is_reserved_as_an_operator_alias():
    with pytest.raises(HNDLError, match="E_NAME.*range"):
        _parse("relu()", ("relu", "range"))


# Resource limits -----------------------------------------------------------

def _counting_registry(emitted):
    """Builtins whose lookups are recorded: the interpreter looks up each call it emits."""
    registry = Registry.builtins()
    lookup = registry.get

    def get(alias):
        emitted.append(alias)
        return lookup(alias)

    registry.get = get
    return registry


def test_a_huge_range_fails_before_anything_is_unrolled():
    emitted = []
    started = time.perf_counter()
    with pytest.raises(HNDLError, match="E_RESOURCE") as rejected:
        capture_config("linear(16)\nfor _ in range(1000000000):\n    relu()",
                       registry=_counting_registry(emitted), **TOKENS)
    assert time.perf_counter() - started < 1
    assert emitted == ["linear"]
    assert (rejected.value.line, rejected.value.column) == (2, 1)
    assert "range(1000000000) repeats a body of 1 node" in rejected.value.message
    assert "max_nodes (4096)" in rejected.value.message


def test_the_multiplied_out_count_uses_the_callers_max_nodes():
    capture_config("relu()\nfor _ in range(3):\n    relu()\n    relu()", limits={"max_nodes": 7}, **TOKENS)
    emitted = []
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture_config("relu()\nfor _ in range(3):\n    relu()\n    relu()", limits={"max_nodes": 6},
                       registry=_counting_registry(emitted), **TOKENS)
    assert emitted == ["relu"]


def test_deep_nesting_fails_fast_by_count_and_by_depth():
    def nest(depth, count):
        return "".join("    " * level + f"for _ in range({count}):\n" for level in range(depth)) \
            + "    " * depth + "relu()\n"

    emitted = []
    with pytest.raises(HNDLError, match="E_RESOURCE") as rejected:
        capture_config(nest(8, 20), registry=_counting_registry(emitted), **TOKENS)
    assert emitted == [] and rejected.value.line == 1
    assert "more than 10**15" not in rejected.value.message
    assert "1280000000 nodes" in rejected.value.message
    with pytest.raises(HNDLError, match="E_RESOURCE") as rejected:
        capture_config(nest(9, 1), **TOKENS)
    assert "nests loops more than 8 levels deep" in rejected.value.message
    assert len(capture_config(nest(8, 1), **TOKENS).nodes) == 1
    huge = "9" * 1000
    with pytest.raises(HNDLError, match="E_RESOURCE") as rejected:
        capture_config(nest(8, huge), **TOKENS)
    assert rejected.value.message.startswith("range(more than 10**15) repeats a body of more than 10**15 nodes")


def test_deep_loop_nesting_is_screened_on_a_small_thread_stack():
    sources = ["".join(" " * i + "for _ in range(1):\n" for i in range(99)) + " " * 99 + "relu()\n",
               "out = [relu() " + "for _ in x " * 5000 + "]",
               "linear(1" + " in 1" * 10000 + ")"]
    script = textwrap.dedent('''
        import json, sys, threading
        from hndl import HNDLError
        from hndl.config import _parse
        outcomes = []
        def run():
            for source in json.load(sys.stdin):
                try:
                    _parse(source, ("linear", "relu"))
                    outcomes.append("parsed")
                except HNDLError as exc:
                    outcomes.append(exc.code)
        threading.stack_size(256 * 1024)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        print(json.dumps(outcomes))
    ''')
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps(sources), env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["E_RESOURCE", "E_SYNTAX", "E_SYNTAX"]


# Names and diagnostics -----------------------------------------------------

def test_suffixed_names_that_collide_explain_the_suffix():
    with pytest.raises(HNDLError, match="E_NAME") as rejected:
        capture_config("relu(name='block2')\nfor _ in range(4):\n    relu(name='block')", **TOKENS)
    assert rejected.value.message == (
        "Duplicate node name 'block2': name='block' in a loop gets the iteration suffix '2', "
        "but an earlier node already has that ID (an explicit name='block2')")
    assert (rejected.value.line, rejected.value.column, rejected.value.iterations) == (3, 5, ((2, 4),))
    with pytest.raises(HNDLError, match="E_NAME") as rejected:
        capture_config("for _ in range(4):\n    relu(name='block')\nrelu(name='block1')", **TOKENS)
    assert rejected.value.message == (
        "Duplicate node name 'block1': an earlier loop already gave that name to name='block' "
        "with iteration suffix '1'")
    with pytest.raises(HNDLError, match="E_NAME") as rejected:
        capture_config("for _ in range(2):\n    relu(name='a')\nfor _ in range(2):\n    relu(name='a')",
                       **TOKENS)
    assert "(name='a' with iteration suffix '0')" in rejected.value.message


def test_names_outside_loops_are_unchanged():
    graph = capture_config("relu(name='block')\nfor _ in range(2):\n    relu()\nrelu(name='tail')", **TOKENS)
    assert [node.id for node in graph.nodes] == ["block", "n1", "n2", "tail"]
    assert "iterations" not in graph.nodes[0].source


def test_node_sources_record_the_statement_and_every_iteration():
    graph = capture_config("for _ in range(2):\n    for _ in range(3):\n        relu()", **TOKENS)
    assert [(node.source["line"], node.source["column"], node.source["iterations"]) for node in graph.nodes] == [
        (3, 9, ((i, 2), (j, 3))) for i in range(2) for j in range(3)]


def test_resolve_errors_name_the_iteration():
    source = textwrap.dedent("""
        linear(16)
        for _ in range(8):
            for _ in range(2):
                relu()
            linear(16, name="block")
        linear(4)
    """)
    with pytest.raises(HNDLError, match="E_CONSTRAINT") as rejected:
        resolve(source, **TOKENS)
    error = rejected.value
    assert (error.node, error.line, error.column, error.iterations) == ("n25", 7, 1, None)
    source = source.replace("        relu()", "        relu()\n        linear(8, name='narrow')")
    with pytest.raises(HNDLError) as rejected:
        resolve(source.replace("linear(16, name=\"block\")", "linear(16, name='block', in_features=16)"),
                **TOKENS)
    error = rejected.value
    assert (error.node, error.line, error.iterations) == ("block0", 7, ((0, 8),))
    assert str(error).startswith("E_CONSTRAINT (node block0; line 7, column 5, iteration 0 of 8): ")


def test_an_undefined_name_in_a_later_iteration_reports_it():
    with pytest.raises(HNDLError, match="E_NAME") as rejected:
        capture_config("for _ in range(3):\n    for _ in range(4):\n        linear(h, 16)\n        h = relu()",
                       **TOKENS)
    assert str(rejected.value) == ("E_NAME (line 3, column 16, iteration 0 of 3, 0 of 4): "
                                   "Undefined tensor variable 'h'")


def _third_fails():
    registry = Registry.builtins()
    calls = []

    @registry.operator("fragile", identity="tests.fragile", summary="The third call fails.",
                       shape="x[B, T, F] -> out[B, T, F]")
    class Fragile(nn.Module):
        def forward(self, x):
            calls.append(self)
            if len(calls) == 3:
                raise RuntimeError("the third instance broke")
            return x

    return registry


def test_runtime_errors_name_the_iteration():
    model = network("linear(16)\nfor _ in range(4):\n    fragile(name='stage')", device="cpu",
                    registry=_third_fails(), **TOKENS)
    with pytest.raises(RuntimeError, match="the third instance broke") as caught:
        model(torch.randn(2, 6, 16))
    notes = [note for note in getattr(caught.value, "__notes__", ()) if note.startswith("HNDL:")]
    assert notes[0].splitlines()[0] == (
        "HNDL: raised inside node 'stage2' (fragile, line 3, column 5, iteration 2 of 4)")


def test_runtime_port_errors_name_the_iteration():
    registry = Registry.builtins()

    @registry.operator("narrows", identity="tests.narrows", summary="Drop a feature after one call.",
                       shape="x[B, T, F] -> out[B, T, F]")
    class Narrows(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x if self.calls == 1 else x[..., :-1]

    model = network("for _ in range(2):\n    for _ in range(2):\n        narrows(name='n')\n        linear(16)",
                    device="cpu", registry=registry, **TOKENS)
    model(torch.randn(2, 6, 16))
    with pytest.raises(RuntimeError) as caught:
        model(torch.randn(2, 6, 16))
    notes = [note for note in getattr(caught.value, "__notes__", ()) if note.startswith("HNDL:")]
    assert notes[0].splitlines()[0] == (
        "HNDL: raised inside node 'n1' (linear, line 4, column 9, iteration 0 of 2, 0 of 2)")


def test_saved_plans_keep_the_iterations():
    plan = resolve("for _ in range(2):\n    linear(16)", **TOKENS)
    restored = ResolvedPlan.from_json(plan.to_json())
    assert restored.nodes[1].source["iterations"] == ((1, 2),)
    assert json.loads(plan.to_json())["nodes"][1]["source"]["iterations"] == [[1, 2]]
