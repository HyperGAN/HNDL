import random
from unittest.mock import patch

import pytest

from hndl import HNDLError, Registry, resolve
from hndl.config import capture_config


def _capture(source, **kwargs):
    return capture_config(source, input_shape=("B", 64), output_shape=("B", 64), **kwargs)


# ==============================================================================
# Suite 1 (H-01): Line ending combinations & whitespace fuzzing
# ==============================================================================

def test_fuzz_line_endings_and_whitespace():
    """Fuzz 1000 configurations with random mixtures of \\r, \\r\\n, \\n, and whitespace.

    Ensures no unhandled IndexError or internal crash ever escapes HNDLError.
    """
    rng = random.Random(42)
    line_endings = ["\n", "\r", "\r\n", "\r\r", "\r\n\r", "\r\r\n"]
    whitespaces = ["", " ", "   ", "\t", "  \t  "]
    snippets = [
        "linear(64)",
        "relu()",
        "h = x",
        "out = x",
        "(a, b) = split(x, 32)",
        "[a, b] = split(x, 32)",
        "out = concat(a, b)",
        "# comment at start of line",
        "linear(64)  # inline comment",
        "for _ in range(2):",
        "    relu()",
        "    h = linear(64)",
        "linear(\n    64\n)",
        "invalid syntax (((",
        "linear(unclosed",
        "x =",
        "    indent_without_block()",
        "",
    ]

    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}

    for _ in range(1000):
        num_lines = rng.randint(1, 15)
        lines = []
        for _ in range(num_lines):
            ws_before = rng.choice(whitespaces)
            snip = rng.choice(snippets)
            ws_after = rng.choice(whitespaces)
            ending = rng.choice(line_endings)
            lines.append(f"{ws_before}{snip}{ws_after}{ending}")

        cfg = "".join(lines)
        try:
            resolve(cfg, **kwargs)
        except HNDLError as exc:
            # Must have valid code and integer or None line/col
            assert exc.code.startswith("E_")
            if exc.line is not None:
                assert isinstance(exc.line, int)
            if exc.column is not None:
                assert isinstance(exc.column, int)
        except Exception as unexpected:
            pytest.fail(f"Unhandled non-HNDLError raised: {type(unexpected).__name__}: {unexpected}\nSource:\n{cfg!r}")


def test_cr_edge_cases():
    """Targeted boundary tests for CR, CRLF, and trailing whitespace."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}

    # Empty and whitespace-only configurations with various line endings
    for empty in ["\r", "\r\r", "\r\n\r", "   \r\t  \r\n \n"]:
        res = resolve(empty, **kwargs)
        assert res.output_ref == "input:x"

    # Trailing bare \r without newline
    assert resolve("out = linear(64)\r", **kwargs) is not None

    # Error on line with bare \r reports correct 1-indexed line
    with pytest.raises(HNDLError) as exc:
        resolve("linear(64)\rrelu(\rout = x", **kwargs)
    assert exc.value.code == "E_SYNTAX"
    assert exc.value.line == 2


# ==============================================================================
# Suite 2 (H-02): Multi-output unpacking chains stress
# ==============================================================================

@pytest.mark.parametrize("statement_count", [10, 20, 32, 33, 50, 100])
def test_unpacking_long_chains_stress(statement_count):
    """Verify up to 100 consecutive unpacking statements do not trigger E_SYNTAX operator count limit."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}

    # Tuple unpacking
    lines = ["h = x"]
    for _ in range(statement_count):
        lines.append("(a, b) = split(h, 32)")
        lines.append("h = concat(a, b)")
    lines.append("out = h")
    plan = resolve("\n".join(lines), **kwargs)
    assert len(plan.nodes) == statement_count * 2

    # List unpacking
    lines = ["h = x"]
    for _ in range(statement_count):
        lines.append("[a, b] = split(h, 32)")
        lines.append("h = concat(a, b)")
    lines.append("out = h")
    plan_list = resolve("\n".join(lines), **kwargs)
    assert len(plan_list.nodes) == statement_count * 2


def test_unpacking_interleaved_complex_stress():
    """Verify 100 unpacking statements interleaved with comments, empty lines, and multiline calls."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    lines = ["h = x"]
    for i in range(100):
        lines.append("# Comment preceding unpacking")
        lines.append("   ")
        if i % 3 == 0:
            lines.append("(\n  a,\n  b\n) = split(\n  h,\n  32\n)")
        elif i % 3 == 1:
            lines.append("(a, b) = split(h, 32)  # inline comment")
        else:
            lines.append("[a, b] = split(h, 32)")
        lines.append("h = concat(a, b)")
    lines.append("out = h")

    plan = resolve("\n".join(lines), **kwargs)
    assert len(plan.nodes) == 200


def test_operator_limit_still_enforced_for_genuine_operators():
    """Verify genuine chained operators (> 32) are still strictly rejected with E_SYNTAX."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    lines = ["h = x"]
    for _ in range(33):
        lines.append("linear(64)(h)")
    with pytest.raises(HNDLError) as exc:
        resolve("\n".join(lines), **kwargs)
    assert exc.value.code == "E_SYNTAX"
    assert "Source uses more than 32 Python operators" in exc.value.message


# ==============================================================================
# Suite 3 (M-03): Adversarial empty tuples and containers
# ==============================================================================

@pytest.mark.parametrize("bad_assignment", [
    "p = ()",
    "() = x",
    "p = []",
    "[] = x",
    "(p,) = ()",
    "(a, b) = ()",
    "p = (())",
    "p = ((),)",
    "p = (x, ())",
    "p = ((), x)",
    "p = (x, 1)",
    "p = (x, 'lit')",
])
def test_adversarial_empty_tuples_rejected(bad_assignment):
    """All invalid/empty tuple assignments must be rejected with E_SYNTAX."""
    with pytest.raises(HNDLError) as exc:
        _capture(f"{bad_assignment}\nout = x")
    assert exc.value.code == "E_SYNTAX"


def test_empty_tuple_target_exact_location():
    """Verify exact line and column attribution for p = ()."""
    with pytest.raises(HNDLError) as exc:
        _capture("p = ()\nout = x")
    assert exc.value.code == "E_SYNTAX"
    assert "Assignments must bind tensor expressions, not literal values" in exc.value.message
    assert (exc.value.line, exc.value.column) == (1, 5)


def test_empty_tuple_in_loop():
    """Verify empty tuple assignment in loop is rejected."""
    with pytest.raises(HNDLError) as exc:
        _capture("for _ in range(2):\n    p = ()\nout = x")
    assert exc.value.code == "E_SYNTAX"
    assert "Assignments must bind tensor expressions, not literal values" in exc.value.message


def test_empty_tuple_in_call_expressions():
    """Verify passing empty tuples into operator calls raises clean HNDLError, not crash."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    for expr, expected_code in [
        ("linear(())", "E_ARGUMENT"),
        ("relu(())", "E_ARGUMENT"),
        ("concat(())", "E_BINDING"),
        ("split(x, ())", "E_ARGUMENT"),
    ]:
        with pytest.raises(HNDLError) as exc:
            resolve(expr, **kwargs)
        assert exc.value.code == expected_code


def test_tensor_tuple_arity_attribution_and_iterations():
    """Verify tensor_tuple reports accurate source location and loop iterations on E_OUTPUT_ARITY."""
    # Top-level attribution
    with pytest.raises(HNDLError) as exc:
        _capture("parts = split(4)\nb = (parts, x)\nout = x")
    assert exc.value.code == "E_OUTPUT_ARITY"
    assert (exc.value.line, exc.value.column) == (2, 5)
    assert "A tensor tuple must contain only tensor symbols" in exc.value.message

    # Loop iterations attribution
    with pytest.raises(HNDLError) as exc:
        _capture("for _ in range(2):\n    parts = split(4)\n    b = (parts, x)\nout = x")
    assert exc.value.code == "E_OUTPUT_ARITY"
    assert (exc.value.line, exc.value.column) == (3, 9)
    assert exc.value.iterations == ((0, 2),)


# ==============================================================================
# Suite 4 (M-04): F-string rejection before ast.parse
# ==============================================================================

@pytest.mark.parametrize("fstring", [
    'linear(f"{x}")',
    'linear(F"{x}")',
    'linear(fr"{x}")',
    'linear(rf"{x}")',
    'linear(f"""{x}""")',
    'linear(F"""{x}""")',
    'linear(fr"""{x}""")',
    'linear(rf"""{x}""")',
    "linear(f'{x}')",
    "linear(F'{x}')",
    "linear(fr'{x}')",
    "linear(rf'{x}')",
    """linear(f'''{x}''')""",
    """linear(F'''{x}''')""",
    """linear(fr'''{x}''')""",
    """linear(rf'''{x}''')""",
    'linear(f"")',
    'linear(F"")',
    'linear(fr"")',
    'linear(rf"")',
    'linear(f"plain text")',
    'linear(f"""\n{x}\n""")',
])
def test_all_fstring_variants_rejected(fstring):
    """Verify every f-string permutation is rejected with E_SYNTAX."""
    with pytest.raises(HNDLError) as exc:
        _capture(fstring)
    assert exc.value.code == "E_SYNTAX"
    assert "f-strings are not permitted in declarative configurations" in exc.value.message


def test_fstrings_rejected_before_ast_parse():
    """Verify ast.parse is never invoked when an f-string is present in the source."""
    Registry.builtins()
    with patch("ast.parse", side_effect=AssertionError("ast.parse should NOT be reached")):
        with pytest.raises(HNDLError) as exc:
            _capture('linear(f"{x}")')
        assert exc.value.code == "E_SYNTAX"
        assert "f-strings are not permitted in declarative configurations" in exc.value.message


def test_non_fstrings_with_f_character_are_not_rejected():
    """Verify non-f-strings containing 'f' are valid and never falsely flagged."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    valid_configs = [
        'out = linear(64, name="final_layer")',
        'out = linear(64, name="f")',
        'out = linear(64, name=r"foo")',
        '# This comment has f"{x}" inside\nout = linear(64)',
        'out = linear(64, name="f_feature_layer")',
    ]
    for cfg in valid_configs:
        plan = resolve(cfg, **kwargs)
        assert plan is not None


def test_py311_prefix_extraction_logic():
    """Directly test the Python 3.11 token string prefix extraction logic."""
    fstring_tokens = [
        'f"{x}"', 'F"{x}"', 'fr"{x}"', 'rf"{x}"',
        'f"""{x}"""', 'F"""{x}"""', "f'{x}'", "F'{x}'",
        'f""', 'F""', 'fr""', 'rf""',
    ]
    for s in fstring_tokens:
        prefix = s.split("'", 1)[0].split('"', 1)[0].lower()
        assert "f" in prefix, f"Expected 'f' in prefix for {s}"

    safe_tokens = [
        '"f"', '"foo"', "'f'", "'foo'",
        '"hello f world"', "'hello f world'",
        'r"rawf"', '"""multiline f"""', "'''multiline f'''",
    ]
    for s in safe_tokens:
        prefix = s.split("'", 1)[0].split('"', 1)[0].lower()
        assert "f" not in prefix, f"Unexpected 'f' in prefix for {s}"


# ==============================================================================
# Suite 5 (L-01): Loop collision diagnostics and attribution
# ==============================================================================

def test_collision_with_auto_generated_id():
    """Collision with an auto-generated node ID reports 'an auto-generated ID'."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    with pytest.raises(HNDLError) as exc:
        resolve('relu()\nfor _ in range(2):\n    relu(name="n")\nout = x', **kwargs)
    assert exc.value.code == "E_NAME"
    assert "an auto-generated ID 'n0'" in exc.value.message
    assert "an explicit name='n0'" not in exc.value.message


def test_collision_with_explicit_name():
    """Collision with an explicit node name reports 'an explicit name='."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    with pytest.raises(HNDLError) as exc:
        resolve('relu(name="n0")\nfor _ in range(2):\n    relu(name="n")\nout = x', **kwargs)
    assert exc.value.code == "E_NAME"
    assert "an explicit name='n0'" in exc.value.message
    assert "an auto-generated ID 'n0'" not in exc.value.message


def test_collision_between_loop_iterations():
    """Collision where an earlier loop iteration created the ID reports iteration suffix."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    with pytest.raises(HNDLError) as exc:
        resolve('for _ in range(2):\n    relu(name="b")\nfor _ in range(2):\n    relu(name="b")\nout = x', **kwargs)
    assert exc.value.code == "E_NAME"
    assert "name='b' with iteration suffix '0'" in exc.value.message


def test_collision_explicit_after_loop():
    """Collision where an explicit name follows a loop that generated the same ID."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    with pytest.raises(HNDLError) as exc:
        resolve('for _ in range(2):\n    relu(name="b")\nrelu(name="b0")\nout = x', **kwargs)
    assert exc.value.code == "E_NAME"
    assert "an earlier loop already gave that name to name='b' with iteration suffix '0'" in exc.value.message


def test_nested_loops_collision_with_explicit_name():
    """Nested loop iteration (e.g. 0_1) colliding with explicit name."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    with pytest.raises(HNDLError) as exc:
        resolve('relu(name="b0_1")\nfor _ in range(2):\n    for _ in range(2):\n        relu(name="b")\nout = x', **kwargs)
    assert exc.value.code == "E_NAME"
    assert "an explicit name='b0_1'" in exc.value.message
    assert "an auto-generated ID 'b0_1'" not in exc.value.message


def test_nested_loops_collision_with_earlier_nested_loop():
    """Nested loop iteration colliding with earlier nested loop iteration."""
    kwargs = {"input_shape": ("B", 64), "output_shape": ("B", 64)}
    with pytest.raises(HNDLError) as exc:
        resolve(
            'for _ in range(2):\n    for _ in range(2):\n        relu(name="b")\n'
            'for _ in range(2):\n    for _ in range(2):\n        relu(name="b")\nout = x',
            **kwargs,
        )
    assert exc.value.code == "E_NAME"
    assert "name='b' with iteration suffix '0_0'" in exc.value.message
