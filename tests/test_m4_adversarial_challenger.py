"""Adversarial challenge and stress-test suite for Milestone 4.

Covers:
1. Constants: strict E_BINDING, diverse positions, types, lowercase vs uppercase, unpacking defect.
2. Macros: deep nesting, scoping uniqueness, recursion rejection, multi-IO, parameter shadowing.
3. Tuple loops: nested loops, heterogeneous dimensions, scope cleanup, invalid iterators.
4. Python scoping & subgraphs: deep nesting, invalid prefix validation, multi-IO subgraphs,
   exact autograd gradient flow parity vs unrolled baseline, and codegen limitation.
"""

from __future__ import annotations

import pytest
import torch

import hndl
from hndl import HNDLError, Registry, ops, resolve
from hndl.torch import build


# ==============================================================================
# 1. Constants Stress Tests
# ==============================================================================


def test_constant_reassignment_same_value_raises_e_binding() -> None:
    source = """
D = 64
D = 64
linear(D)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 16), output_shape=("B", 64))
    assert exc_info.value.code == "E_BINDING"
    assert "Constant 'D' cannot be rebound" in str(exc_info.value)


def test_constant_reassignment_different_value_raises_e_binding() -> None:
    source = """
D = 64
D = 128
linear(D)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 16), output_shape=("B", 128))
    assert exc_info.value.code == "E_BINDING"
    assert "Constant 'D' cannot be rebound" in str(exc_info.value)


def test_constant_reassignment_tensor_to_constant_raises_e_binding() -> None:
    source = """
D = 64
D = linear(32)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 16), output_shape=("B", 32))
    assert exc_info.value.code == "E_BINDING"
    assert "Constant 'D' cannot be rebound" in str(exc_info.value)


def test_constant_unpacking_reassignment_raises_e_binding() -> None:
    # Demonstrates that unpacking a constant raises E_BINDING.
    # Note: parser reports non_discard[0] in error text even if a later element is the constant.
    source = """
D = 64
a, D = split(2)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 8), output_shape={"a": ("B", 4), "D": ("B", 4)})
    assert exc_info.value.code == "E_BINDING"
    assert "cannot be rebound" in str(exc_info.value)


def test_constant_in_various_positions() -> None:
    # 1. Tensor shape args (reshape)
    # 2. Operator dimension args (linear)
    # 3. Loop bounds (range(DEPTH))
    # 4. Operator kwargs (bias)
    source = """
DEPTH = 2
D = 64
BIAS = False

for _ in range(DEPTH):
    x = linear(x, D, bias=BIAS)
    x = relu(x)

out = reshape(x, D, 1)
"""
    plan = resolve(source, input_shape={"x": ("B", 64)}, output_shape={"out": ("B", 64, 1)})
    assert len(plan.nodes) == 5  # (linear + relu) * 2 + reshape
    assert plan.nodes[0].args["out_features"] == 64
    assert plan.nodes[0].args["bias"] is False
    assert plan.nodes[4].args["shape"] == (64, 1)


def test_constant_scalar_types() -> None:
    source = """
INT_VAL = 32
FLOAT_VAL = 2.5
BOOL_VAL = False

h = linear(x, INT_VAL, bias=BOOL_VAL)
out = moe(h, 4, INT_VAL, routed_scaling_factor=FLOAT_VAL)
"""
    plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
    assert plan.nodes[0].args["out_features"] == 32
    assert plan.nodes[0].args["bias"] is False
    assert plan.nodes[1].args["routed_scaling_factor"] == 2.5


def test_constant_invalid_types_and_bounds_rejected() -> None:
    # Tuple constant rejected
    with pytest.raises(HNDLError) as exc_info:
        resolve("T = (1, 2)\nlinear(32)\n", input_shape=("B", 8), output_shape=("B", 32))
    assert exc_info.value.code == "E_SYNTAX"

    # Arithmetic expression rejected
    with pytest.raises(HNDLError) as exc_info:
        resolve("D = 32 + 32\nlinear(D)\n", input_shape=("B", 8), output_shape=("B", 64))
    assert exc_info.value.code == "E_SYNTAX"

    # Non-positive range count constant rejected
    with pytest.raises(HNDLError) as exc_info:
        resolve("DEPTH = 0\nfor _ in range(DEPTH):\n    x = linear(x, 16)\nout = x\n",
                input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_SYNTAX"


def test_lowercase_assignment_vs_uppercase_constant() -> None:
    # Lowercase scalar assignment must be rejected as an invalid literal assignment
    with pytest.raises(HNDLError) as exc_info:
        resolve("d = 64\nlinear(d)\n", input_shape=("B", 16), output_shape=("B", 64))
    assert exc_info.value.code == "E_SYNTAX"
    assert "Assignments must bind tensor expressions, not literal values" in str(exc_info.value)

    # Uppercase tensor node assignment is permitted
    source = """
x = linear(16)
OUT = linear(x, 32)
"""
    plan = resolve(source, input_shape=("B", 8), output_shape=("B", 32))
    assert len(plan.nodes) == 2


# ==============================================================================
# 2. Macro Stress Tests
# ==============================================================================


def test_macro_deep_nesting() -> None:
    source = """
def l1(x, d):
    return linear(x, d)

def l2(x, d):
    return relu(l1(x, d))

def l3(x, d):
    return l2(x, d)

out = l3(x, 32)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    assert len(plan.nodes) == 2
    assert plan.nodes[0].id == "l3_0_l2_0_l1_0_n0"
    assert plan.nodes[1].id == "l3_0_l2_0_n1"


def test_macro_multiple_invocations_scoping_uniqueness() -> None:
    source = """
def mlp(x, d):
    h = linear(x, d)
    return relu(h)

h1 = mlp(x, 32)
h2 = mlp(h1, 64)
h3 = mlp(h2, 16)
out = h3
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert len(plan.nodes) == 6
    node_ids = [n.id for n in plan.nodes]
    assert len(set(node_ids)) == 6
    assert node_ids[0] == "mlp_0_n0"
    assert node_ids[2] == "mlp_1_n2"
    assert node_ids[4] == "mlp_2_n4"


def test_macro_explicit_names_scoping() -> None:
    source = """
def block(x, d):
    h = linear(x, d, name="proj")
    return relu(h, name="act")

h = block(x, 32)
out = block(h, 32)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    node_ids = [n.id for n in plan.nodes]
    assert node_ids == ["block_0_proj", "block_0_act", "block_1_proj", "block_1_act"]


def test_macro_direct_recursion_rejected() -> None:
    source = """
def foo(x):
    return relu(foo(x))

out = foo(x)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_SYNTAX"
    assert "Recursion is forbidden" in str(exc_info.value)


def test_macro_mutual_recursion_rejected() -> None:
    source = """
def foo(x):
    return bar(x)

def bar(x):
    return foo(x)

out = foo(x)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_SYNTAX"
    assert "Calls must name a registered operator directly" in str(exc_info.value)


def test_macro_redefinition_and_operator_alias_rejected() -> None:
    # Duplicate macro name
    with pytest.raises(HNDLError) as exc_info:
        resolve("def f(x):\n    return relu(x)\ndef f(x):\n    return relu(x)\n",
                input_shape=("B", 16), output_shape=("B", 16))
    assert exc_info.value.code == "E_NAME"
    assert "Duplicate macro name 'f'" in str(exc_info.value)

    # Rebinding registered operator alias
    with pytest.raises(HNDLError) as exc_info:
        resolve("def linear(x):\n    return relu(x)\n",
                input_shape=("B", 16), output_shape=("B", 16))
    assert exc_info.value.code == "E_NAME"
    assert "Registered operator aliases cannot be rebound" in str(exc_info.value)


def test_macro_argument_validation() -> None:
    # Too many positional args
    src_too_many = """
def f(x, y):
    return relu(linear(x, 16))
out = f(x, y, 10)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(src_too_many, input_shape={"x": ("B", 16), "y": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_ARGUMENT"

    # Missing required argument
    src_missing = """
def f(x, y):
    return relu(linear(x, 16))
out = f(x)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(src_missing, input_shape={"x": ("B", 16), "y": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_ARGUMENT"

    # Unknown keyword argument
    src_unknown = """
def f(x, y):
    return relu(linear(x, 16))
out = f(x, y=y, invalid=123)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(src_unknown, input_shape={"x": ("B", 16), "y": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_ARGUMENT"


def test_macro_multi_io_and_residuals() -> None:
    source = """
def residual_split_block(x, y):
    h1 = linear(x, 16)
    h2 = linear(y, 16)
    out1 = add(x, h1)
    out2 = add(y, h2)
    return out1, out2

a, b = residual_split_block(x, y)
"""
    plan = resolve(source, input_shape={"x": ("B", 16), "y": ("B", 16)},
                   output_shape={"a": ("B", 16), "b": ("B", 16)})
    assert len(plan.nodes) == 4
    assert plan.outputs["a"]["ref"] == "node:residual_split_block_0_n2/out"
    assert plan.outputs["b"]["ref"] == "node:residual_split_block_0_n3/out"


# ==============================================================================
# 3. Tuple Loops Stress Tests
# ==============================================================================


def test_nested_tuple_loops() -> None:
    source = """
for stage in (1, 2):
    for width in (32, 64):
        x = linear(x, width)
        x = relu(x)
out = x
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert len(plan.nodes) == 8
    linear_widths = [n.args["out_features"] for n in plan.nodes if "out_features" in n.args]
    assert linear_widths == [32, 64, 32, 64]


def test_tuple_loop_scoping_cleanup() -> None:
    source = """
for width in (32, 64):
    x = linear(x, width)
x = linear(x, width)
out = x
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert exc_info.value.code == "E_NAME"
    assert "Undefined tensor variable 'width'" in str(exc_info.value)


def test_tuple_loop_heterogeneous_multi_stage_net() -> None:
    source = """
for hidden in (64, 128, 256, 32):
    x = linear(x, hidden)
    x = gelu(x)
out = x
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    model = build(plan, device="cpu")
    x = torch.randn(4, 16)
    out = model(x)
    assert out.shape == (4, 32)


def test_tuple_loop_invalid_iterators_rejected() -> None:
    # List in loop tuple
    with pytest.raises(HNDLError) as exc_info:
        resolve("for c in ([1, 2], [3, 4]):\n    x = linear(x, 16)\nout = x\n",
                input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_SYNTAX"

    # Outer loop variable in inner loop tuple elements
    with pytest.raises(HNDLError) as exc_info:
        resolve("for outer in (32, 64):\n    for inner in (16, outer):\n        x = linear(x, inner)\nout = x\n",
                input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert exc_info.value.code == "E_SYNTAX"


# ==============================================================================
# 4. Python Scoping & Subgraphs Stress Tests
# ==============================================================================


def test_ops_scope_deep_nesting() -> None:
    def net(x):
        with ops.scope("outer"):
            with ops.scope("mid"):
                with ops.scope("inner"):
                    with ops.scope("sub"):
                        h = ops.linear(x, 32, name="fc")
                        out = ops.relu(h)
        return out

    plan = hndl.resolve_callable(net, input_shape=("B", 16), output_shape=("B", 32))
    assert plan.nodes[0].id == "outer_mid_inner_sub_fc"
    assert plan.nodes[1].id == "outer_mid_inner_sub_n1"


@pytest.mark.parametrize("invalid_prefix", ["", "123", "Scope", "scope-name", "scope.name", 42, None])
def test_ops_scope_invalid_prefixes(invalid_prefix) -> None:
    with pytest.raises(HNDLError) as exc_info:
        with ops.scope(invalid_prefix):  # type: ignore[arg-type]
            pass
    assert exc_info.value.code == "E_NAME"
    assert "Scope prefix must match [a-z][a-z0-9_]*" in str(exc_info.value)


def test_registry_subgraph_multi_io() -> None:
    reg = Registry.builtins()
    sub_source = """
x_out = linear(x, 16)
y_out = linear(y, 32)
"""
    sub_plan = resolve(sub_source, input_shape={"x": ("B", 8), "y": ("B", 8)},
                       output_shape={"x_out": ("B", 16), "y_out": ("B", 32)}, registry=reg)
    reg.subgraph("multi_sub", sub_plan)

    main_source = """
xo, yo = multi_sub(a, b)
res = linear(yo, 8)
"""
    main_plan = resolve(main_source, input_shape={"a": ("B", 8), "b": ("B", 8)},
                        output_shape={"res": ("B", 8), "xo": ("B", 16)}, registry=reg)

    model = build(main_plan, device="cpu", registry=reg)
    a = torch.randn(2, 8, requires_grad=True)
    b = torch.randn(2, 8, requires_grad=True)
    out = model(a=a, b=b)
    assert "res" in out and "xo" in out
    loss = out["res"].sum() + out["xo"].sum()
    loss.backward()
    assert a.grad is not None and b.grad is not None


def test_registry_subgraph_nested_autograd_parity() -> None:
    reg = Registry.builtins()
    plan_a = resolve("""
h = linear(x, 32)
a = gelu(h)
out = linear(a, 16)
""", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)}, registry=reg)
    reg.subgraph("mlp_block", plan_a)

    plan_b = resolve("""
h = mlp_block(x)
out = add(x, h)
""", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)}, registry=reg)
    reg.subgraph("res_block", plan_b)

    main_plan = resolve("""
h1 = res_block(x)
h2 = res_block(h1)
out = linear(h2, 8)
""", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 8)}, registry=reg)

    unrolled_plan = resolve("""
h1_m_h = linear(x, 32)
h1_m_a = gelu(h1_m_h)
h1_m_o = linear(h1_m_a, 16)
h1 = add(x, h1_m_o)

h2_m_h = linear(h1, 32)
h2_m_a = gelu(h2_m_h)
h2_m_o = linear(h2_m_a, 16)
h2 = add(h1, h2_m_o)

out = linear(h2, 8)
""", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 8)})

    model_sub = build(main_plan, device="cpu", registry=reg)
    model_unrolled = build(unrolled_plan, device="cpu")

    sub_params = dict(model_sub.named_parameters())
    unrolled_params = dict(model_unrolled.named_parameters())

    with torch.no_grad():
        sub_params["nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n0.weight"].copy_(unrolled_params["nodes.n_n0.weight"])
        sub_params["nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n0.bias"].copy_(unrolled_params["nodes.n_n0.bias"])
        sub_params["nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n2.weight"].copy_(unrolled_params["nodes.n_n2.weight"])
        sub_params["nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n2.bias"].copy_(unrolled_params["nodes.n_n2.bias"])

        sub_params["nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n0.weight"].copy_(unrolled_params["nodes.n_n4.weight"])
        sub_params["nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n0.bias"].copy_(unrolled_params["nodes.n_n4.bias"])
        sub_params["nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n2.weight"].copy_(unrolled_params["nodes.n_n6.weight"])
        sub_params["nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n2.bias"].copy_(unrolled_params["nodes.n_n6.bias"])

        sub_params["nodes.n_n2.weight"].copy_(unrolled_params["nodes.n_n8.weight"])
        sub_params["nodes.n_n2.bias"].copy_(unrolled_params["nodes.n_n8.bias"])

    x_sub = torch.randn(4, 16, requires_grad=True)
    x_unrolled = x_sub.clone().detach().requires_grad_(True)

    out_sub = model_sub(x_sub)
    out_unrolled = model_unrolled(x_unrolled)

    assert torch.allclose(out_sub, out_unrolled, atol=1e-6)

    loss_sub = (out_sub ** 2).sum()
    loss_unrolled = (out_unrolled ** 2).sum()

    loss_sub.backward()
    loss_unrolled.backward()

    assert torch.allclose(x_sub.grad, x_unrolled.grad, atol=1e-6)

    for sub_name, unrolled_name in [
        ("nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n0.weight", "nodes.n_n0.weight"),
        ("nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n0.bias", "nodes.n_n0.bias"),
        ("nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n2.weight", "nodes.n_n2.weight"),
        ("nodes.n_n0.submodel.nodes.n_n0.submodel.nodes.n_n2.bias", "nodes.n_n2.bias"),
        ("nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n0.weight", "nodes.n_n4.weight"),
        ("nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n0.bias", "nodes.n_n4.bias"),
        ("nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n2.weight", "nodes.n_n6.weight"),
        ("nodes.n_n1.submodel.nodes.n_n0.submodel.nodes.n_n2.bias", "nodes.n_n6.bias"),
        ("nodes.n_n2.weight", "nodes.n_n8.weight"),
        ("nodes.n_n2.bias", "nodes.n_n8.bias"),
    ]:
        p1 = sub_params[sub_name]
        p2 = unrolled_params[unrolled_name]
        diff = (p1.grad - p2.grad).abs().max().item()
        assert diff < 1e-6, f"Gradient mismatch on {sub_name}: diff {diff}"


# ==============================================================================
# 5. Challenger Findings: Edge Cases & Defects
# ==============================================================================


def test_finding_1_unpacking_rebind_constant_error_reports_wrong_name() -> None:
    # Finding 1: In _parser.py lines 238-239, when checking unpacked assignment targets:
    # `if any(name in self.constants for name in non_discard):`
    # `    self.reject(f"Constant '{non_discard[0]}' cannot be rebound", node, code="E_BINDING")`
    # Notice it references non_discard[0] ('x') instead of the matching constant name ('D').
    source = """
D = 64
x, y, D = split(3)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 9), output_shape={"x": ("B", 3), "y": ("B", 3), "D": ("B", 3)})
    assert exc_info.value.code == "E_BINDING"
    # Demonstrates that it incorrectly reports 'x' rather than 'D'
    assert "Constant 'x' cannot be rebound" in str(exc_info.value)


def test_finding_2_constant_shadows_macro_parameter() -> None:
    # Finding 2: When a macro defines a parameter with the same name as an existing constant,
    # the constant AST folding / config evaluation takes precedence over the call argument.
    source = """
D = 64
def foo(x, D):
    return linear(x, D)
out = foo(x, 128)
"""
    # Calling foo(x, 128) should produce out_features=128, but because D=64 is folded,
    # it produces out_features=64, triggering E_CONSTRAINT against output_shape=("B", 128).
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 128)})
    assert exc_info.value.code == "E_CONSTRAINT"
    assert "Shared dimension D_out requires incompatible values 64 and 128" in str(exc_info.value)


def test_finding_3_subgraph_to_python_source_indentation_error() -> None:
    # Finding 3: In codegen.py lines 117-123, custom classes nested in methods (like _SubgraphModule)
    # are extracted via inspect.getsource without dedenting, emitting indented class blocks that
    # cause IndentationError when the generated script is executed.
    reg = Registry.builtins()
    sub_plan = resolve("linear(16)\n", input_shape=("B", 8), output_shape=("B", 16))
    reg.subgraph("my_sub", sub_plan)

    main_plan = resolve("out = my_sub(x)\n", input_shape={"x": ("B", 8)}, output_shape={"out": ("B", 16)}, registry=reg)
    code = main_plan.to_python_source()
    assert "        class _SubgraphModule(nn.Module):" in code
    with pytest.raises(IndentationError):
        exec(code, {})

