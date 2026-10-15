import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hndl import HNDLError, Registry, ResolvedPlan, ops, resolve, resolve_callable
from hndl.config import MAX_SOURCE_BYTES, _parse, capture_config, resolve_file


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


def test_limits_fail_before_parent_allocation_or_unbounded_loading(tmp_path):
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        capture("#" * (MAX_SOURCE_BYTES + 1))
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


def test_parsing_runs_in_process_on_any_platform(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Parsing must not start a process")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert len(capture("relu()").nodes) == 1


_CAP = MAX_SOURCE_BYTES - 64
# Unscreened, most of these make ast.parse on CPython 3.11-3.13 overflow a
# 512 KiB thread stack and kill the process with SIGSEGV.
PATHOLOGICAL = [
    "linear(" + "-" * _CAP + "1)",
    "linear(" + "~" * _CAP + "1)",
    "linear(" + "not " * (_CAP // 4) + "1)",
    "linear(" + "(" * (_CAP // 2) + "1" + ")" * (_CAP // 2) + ")",
    "relu" + "()" * (_CAP // 2),
    '"a"' + "()" * (_CAP // 2),
    "x" + ".a" * (_CAP // 2),
    "x" + "[0]" * (_CAP // 3),
    "linear(1" + "+1" * (_CAP // 2) + ")",
    "linear(2" + "**2" * (_CAP // 3) + ")",
    "linear(" + "1 if 1 else " * (_CAP // 12) + "1)",
    "linear(" + "lambda:" * (_CAP // 7) + "1)",
    "linear(" + "lambda a, b=" * (_CAP // 12) + "1)",
    "linear(" + "[" * 49 + "-" * 32 + "1" + "]" * 49 + ")",
    "linear(" + "[" * 49 + "lambda:" * 32 + "1" + "]" * 49 + ")",
    "".join(" " * i + "if 1:\n" for i in range(99)) + " " * 99 + "relu()\n",
]


@pytest.mark.parametrize("source", PATHOLOGICAL)
def test_pathological_nesting_at_the_size_cap_fails_cleanly(source):
    assert len(source.encode()) <= MAX_SOURCE_BYTES
    with pytest.raises(HNDLError, match="E_SYNTAX|E_RESOURCE"):
        capture(source)


def test_pathological_nesting_fails_cleanly_on_a_small_thread_stack():
    # A crash here would take the test process down, so run it in a child.
    script = '''
import json, sys, threading
from hndl import HNDLError
from hndl.config import _parse
aliases, outcomes = ("linear", "relu"), []
def run():
    for source in json.load(sys.stdin):
        try:
            _parse(source, aliases)
            outcomes.append("parsed")
        except HNDLError as exc:
            outcomes.append(exc.code)
threading.stack_size(256 * 1024)
thread = threading.Thread(target=run)
thread.start()
thread.join()
print(json.dumps(outcomes))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps(PATHOLOGICAL), env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) <= {"E_SYNTAX", "E_RESOURCE"}


def test_valid_configs_are_not_limited_by_the_nesting_screen():
    many = "\n".join(f"h{i} = linear(64, name='l{i}', init={{'weight': normal(std=0.02), 'bias': -0.5}})"
                     for i in range(300))
    assert len(_parse(many, Registry.builtins().aliases)["statements"]) == 300
    negatives = "reshape(" + repr(tuple([-1] * 2000)) + ")"
    assert len(_parse(negatives, Registry.builtins().aliases)["statements"]) == 1
    nested = "linear(" + "[" * 40 + "1" + "]" * 40 + ")"
    assert len(_parse(nested, Registry.builtins().aliases)["statements"]) == 1


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


def test_line_break_normalization_cr_and_crlf(tmp_path):
    # H-01: Bare \r and CRLF \r\n line breaks normalize cleanly
    kwargs = {"input_shape": ("B", 128), "output_shape": ("B", 10)}
    baseline_plan = resolve("linear(64)\nrelu()\nlinear()", **kwargs)

    # Test bare \r
    cr_plan = resolve("linear(64)\rrelu()\rlinear()", **kwargs)
    assert [n.id for n in cr_plan.nodes] == [n.id for n in baseline_plan.nodes]
    assert cr_plan.semantic_digest == baseline_plan.semantic_digest

    # Test Windows CRLF \r\n
    crlf_plan = resolve("linear(64)\r\nrelu()\r\nlinear()", **kwargs)
    assert crlf_plan.semantic_digest == baseline_plan.semantic_digest

    # Test syntax error line reporting with \r
    with pytest.raises(HNDLError) as exc:
        resolve("linear(64)\rrelu(invalid\rlinear()", **kwargs)
    assert exc.value.line == 2
    assert exc.value.code == "E_SYNTAX"

    # Test resolve_file with \r and \r\n
    cr_file = tmp_path / "cr_model.hndl"
    cr_file.write_bytes(b"linear(64)\rrelu()\rlinear()")
    assert resolve_file(cr_file, **kwargs).semantic_digest == baseline_plan.semantic_digest

    crlf_file = tmp_path / "crlf_model.hndl"
    crlf_file.write_bytes(b"linear(64)\r\nrelu()\r\nlinear()")
    assert resolve_file(crlf_file, **kwargs).semantic_digest == baseline_plan.semantic_digest


def test_validator_location_bounds_guard():
    # H-01: Validator.location never raises IndexError on out-of-bounds lineno
    import ast
    from hndl._parser import Validator
    validator = Validator(["linear"], [0], ["linear(64)"])
    fake_node = ast.Name(id="x", ctx=ast.Load())
    fake_node.lineno = 999
    fake_node.col_offset = 10
    loc = validator.location(fake_node)
    assert loc == {"line": 999, "column": 11}


def test_screener_resets_across_newlines_for_unpacking():
    # H-02: 35+ consecutive tuple and list unpacking statements
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    lines = ["h = x"]
    for _ in range(35):
        lines.append("(a, b) = split(h, 32)")
        lines.append("h = concat(a, b)")
    lines.append("out = h")
    tuple_cfg = "\n".join(lines)
    plan = resolve(tuple_cfg, **kwargs)
    assert plan is not None

    lines = ["h = x"]
    for _ in range(35):
        lines.append("[a, b] = split(h, 32)")
        lines.append("h = concat(a, b)")
    lines.append("out = h")
    list_cfg = "\n".join(lines)
    plan_list = resolve(list_cfg, **kwargs)
    assert plan_list is not None

    # Multiline call followed by unpacking
    multiline_cfg = "linear(\n 64\n)\n(a, b) = split(32)\nout = concat(a, b)\n"
    assert resolve(multiline_cfg, **kwargs) is not None

    # Chained calls on the same line still rejected
    with pytest.raises(HNDLError, match="E_SYNTAX"):
        resolve("linear(64)(x)\n", **kwargs)


def test_empty_tuple_target_rejected():
    # M-03: Empty tuple target p = () is rejected with E_SYNTAX
    with pytest.raises(HNDLError) as exc:
        capture("p = ()\nout = x")
    assert exc.value.code == "E_SYNTAX"
    assert "Assignments must bind tensor expressions, not literal values" in exc.value.message
    assert (exc.value.line, exc.value.column) == (1, 5)


def test_tensor_tuple_source_attribution_on_arity_error():
    # M-03: tensor_tuple reports source line and column on E_OUTPUT_ARITY
    with pytest.raises(HNDLError) as exc:
        capture("parts = split(4)\nb = (parts, x)\nout = x")
    assert exc.value.code == "E_OUTPUT_ARITY"
    assert (exc.value.line, exc.value.column) == (2, 5)
    assert "A tensor tuple must contain only tensor symbols" in exc.value.message


@pytest.mark.parametrize("fstring", [
    'linear(f"{x}")',
    'linear(F"{x}")',
    'linear(fr"{x}")',
    'linear(rf"{x}")',
    'linear(f"""{x}""")',
])
def test_screener_rejects_fstrings(fstring):
    # M-04: f-strings rejected in _screen before ast.parse
    with pytest.raises(HNDLError) as exc:
        capture(fstring)
    assert exc.value.code == "E_SYNTAX"
    assert "f-strings are not permitted in declarative configurations" in exc.value.message

