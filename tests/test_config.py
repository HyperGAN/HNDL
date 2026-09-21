import json
from pathlib import Path
import subprocess
import sys

import pytest

from hndl import HNDLError, Registry, ResolvedPlan, ops, resolve, resolve_callable
from hndl.config import MAX_SOURCE_BYTES, _parse, capture_config, resolve_file
import hndl.config as config


def capture(source, **kwargs):
    return capture_config(source, input_shape=("B", 128), output_shape=("B", 10), **kwargs)


def test_minimal_config_and_original_dedented_locations():
    graph = capture('\n    linear(64)\n    relu()\n    linear()\n')
    assert [node.id for node in graph.nodes] == ["n0", "n1", "n2"]
    assert graph.nodes[0].source["line"] == 2
    assert graph.nodes[0].source["column"] == 5
    assert graph.output_ref == "node:n2/out"
    assert capture("# identity\n").output_ref == "input:x"


def test_comments_quotes_and_renamed_locals_do_not_change_node_names():
    graph = capture('''
    # Native Python comments; calls may span lines.
    saved = linear(
        x, 64, name="hidden"
    )
    saved = relu(saved)
    out = linear(saved)
    ''')
    assert [node.id for node in graph.nodes] == ["hidden", "n1", "n2"]
    assert graph.nodes[1].inputs["x"] == "node:hidden/out"
    assert graph.nodes[2].inputs["x"] == "node:n1/out"


def test_rebinding_does_not_move_current_and_explicit_out_overrides():
    graph = capture('saved = x\nlinear(64)\nx = saved\nrelu()\nout = x')
    assert graph.nodes[1].inputs["x"] == "node:n0/out"
    assert graph.output_ref == "input:x"  # Core, not capture, rejects dead nodes.


def test_split_tuple_storage_unpacking_and_explicit_branches():
    graph = capture('''
    parts = split(64)
    a, b = parts
    left = linear(a, 10)
    right = linear(10, x=b)
    add(a=left, b=right)
    ''')
    assert graph.nodes[1].inputs["x"] == "node:n0/first"
    assert graph.nodes[2].inputs["x"] == "node:n0/rest"
    assert graph.output_ref == "node:n3/out"
    assert capture('a, b = split(64)\nout = b').output_ref == "node:n0/rest"


def test_nested_calls_have_python_order_and_update_current():
    graph = capture('add(linear(10), linear(10))')
    assert [node.inputs.get("x") for node in graph.nodes[:2]] == ["input:x", "node:n0/out"]
    assert dict(graph.nodes[-1].inputs) == {"a": "node:n0/out", "b": "node:n1/out"}


@pytest.mark.parametrize("source,code", [
    ('a, b = split(64)\nlinear()', "E_CURRENT"),
    ('split(64)', "E_CURRENT"),
    ('out = split(64)\nlinear(x,10)', "E_OUTPUT"),
    ('a,b,c = split(64)', "E_OUTPUT_ARITY"),
    ('linear(missing)', "E_NAME"),
    ('linear(True)', "E_ARGUMENT"),
    ('linear(64, out_features=32)', "E_ARGUMENT"),
    ('concat(x, x, input_count=2.0)', "E_BINDING"),
])
def test_graph_binding_and_argument_errors(source, code):
    with pytest.raises(HNDLError, match=code):
        capture(source)


@pytest.mark.parametrize("source", [
    'import os', 'from os import system', 'print("oops")',
    'x.__class__', 'out = x[0]', 'linear(**{})', 'linear(*[64])',
    'linear(out_features=(lambda: 64)())', 'linear([relu()])',
    'out = [relu() for _ in x]', 'linear(f"{x}")',
    'out = (x if x else x)', 'out = (y := x)',
    'while True:\n    relu()', 'def fn():\n    return x',
    '"docstring"', 'x', '42', 'out = 42',
    'a = b = x', 'x += 1', 'x: int = 1', 'a, *b = split(64)',
    'linear = x', 'out = __secret', 'a, a = split(64)',
    'linear(2 + 2)', 'linear(2**100)', 'linear(1j)',
    'linear(1e999)', 'linear({"x": 1, "x": 2})',
    'linear({**{}})', 'linear(set())',
])
def test_entire_ast_rejected_before_any_registered_lookup(source):
    registry = Registry.builtins()
    calls = []

    def unexpected(*args):
        calls.append(args)
        raise AssertionError("Registered lookup occurred before full AST validation")

    registry.get = unexpected
    with pytest.raises(HNDLError):
        capture("linear(64)\n" + source, registry=registry)
    assert calls == []


def test_config_never_executes_source_or_falls_back(tmp_path):
    marker = tmp_path / "executed"
    source = f'__import__("pathlib").Path({str(marker)!r}).touch()'
    with pytest.raises(HNDLError):
        capture(source)
    assert not marker.exists()
    with pytest.raises(TypeError):
        capture(lambda x: x)


def test_limits_fail_before_parent_allocation_or_unbounded_loading(tmp_path, monkeypatch):
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("#" * (MAX_SOURCE_BYTES + 1))
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("\n" * 4096)
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("linear(" + "9" * 100 + ")")
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("linear(" + "[" * 100 + "1" + "]" * 100 + ")")
    large = tmp_path / "large.hndl"
    large.write_bytes(b"#" * (MAX_SOURCE_BYTES + 1))
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        resolve_file(large, input_shape=("B", 128), output_shape=("B", 10))
    invalid = tmp_path / "invalid.hndl"
    invalid.write_bytes(b"\xff")
    with pytest.raises(HNDLError, match="E_SYNTAX"):
        resolve_file(invalid, input_shape=("B", 128), output_shape=("B", 10))


def test_linux_isolation_and_no_unbounded_fallback(monkeypatch):
    monkeypatch.setattr(config.sys, "platform", "darwin")
    with pytest.raises(HNDLError, match="Linux"):
        capture("relu()")


def test_worker_invocation_is_isolated_for_strings_and_output_is_checked(monkeypatch):
    original = subprocess.run
    commands = []

    def observed(command, **kwargs):
        commands.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(config.subprocess, "run", observed)
    capture("relu()")
    assert commands[0][:3] == [sys.executable, "-I", "-S"]
    assert Path(commands[0][3]).name == "_parser_worker.py"

    def malformed(command, **kwargs):
        kwargs["stdout"].write(json.dumps({"ok": True, "program": {"bad": 1}}).encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(config.subprocess, "run", malformed)
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("relu()")


def test_timeout_and_worker_failure_are_explicit(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(config.subprocess, "run", timeout)
    with pytest.raises(HNDLError, match="wall-time"):
        capture("relu()")
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], -9))
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("relu()")


def test_invalid_ast_and_argument_locations_use_original_source_columns():
    with pytest.raises(HNDLError) as rejected:
        capture("\n    linear(64)\n    x[0]\n")
    assert (rejected.value.line, rejected.value.column) == (3, 5)
    with pytest.raises(HNDLError) as rejected:
        capture("\n    linear(True)\n")
    assert (rejected.value.line, rejected.value.column) == (2, 5)


def test_python_comments_respect_quoted_literal_hashes():
    program = _parse('relu(name="# not a comment") # actual comment', Registry.builtins().aliases)
    value = program["statements"][0]["value"]["kwargs"][0][1]
    assert value["value"] == "# not a comment"


def test_public_frontends_agree_and_restore_without_author_reexecution(tmp_path):
    source = "linear(64)\nrelu()\nlinear()"
    calls = []

    def author(x):
        calls.append(1)
        ops.linear(64)
        ops.relu()
        ops.linear()

    kwargs = {"input_shape": ("B", 128), "output_shape": ("B", 10)}
    path = tmp_path / "network.hndl"
    path.write_text(source, encoding="utf-8")
    text_plan = resolve(source, **kwargs)
    file_plan = resolve_file(path, **kwargs)
    native_plan = resolve_callable(author, **kwargs)
    assert text_plan.semantic_digest == file_plan.semantic_digest == native_plan.semantic_digest
    restored = ResolvedPlan.from_json(native_plan.to_json())
    assert calls == [1]
    assert restored.semantic_digest == text_plan.semantic_digest


def test_public_split_infers_omitted_size_from_selected_remainder():
    plan = resolve("a, b = split()\nout = b", input_shape=("B", 128), output_shape=("B", 32))
    assert plan.nodes[0].args["size"] == 96
    with pytest.raises(HNDLError, match="E_BINDING"):
        resolve("linear(64)\nout = x", input_shape=("B", 128), output_shape=("B", 128))


def test_import_and_resolution_do_not_load_torch():
    script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("pure frontend imported torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from hndl import resolve, resolve_callable, ops
resolve("linear(4)", input_shape=("B", 8), output_shape=("B", 4))
resolve_callable(lambda x: ops.linear(4), input_shape=("B", 8), output_shape=("B", 4))
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=15)
