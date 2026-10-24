"""Adversarial challenge and stress-test suite for Milestone 4.

Covers:
1. Constants: strict E_BINDING, diverse positions, types, lowercase vs uppercase, unpacking defect.
2. Macros: deep nesting, scoping uniqueness, recursion rejection, multi-IO, parameter shadowing.
3. Tuple loops: nested loops, heterogeneous dimensions, scope cleanup, invalid iterators.
4. Python scoping & subgraphs: deep nesting, invalid prefix validation, multi-IO subgraphs,
   exact autograd gradient flow parity vs unrolled baseline, and codegen limitation.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import hndl
from hndl import HNDLError, Registry, ops, resolve
from hndl.testing import available_devices
from hndl.torch import build, load_weights, network

DEVICES = available_devices()


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
# 5. Challenger Findings: Remediated Edge Cases & Defects
# ==============================================================================


def test_finding_1_unpacking_rebind_constant_error_reports_wrong_name() -> None:
    # Finding 1 (Remediated): In _parser.py lines 238-239, unpacked assignment targets
    # report the actual matching constant name in the diagnostic, not non_discard[0].
    source = """
D = 64
x, y, D = split(3)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 9), output_shape={"x": ("B", 3), "y": ("B", 3), "D": ("B", 3)})
    assert exc_info.value.code == "E_BINDING"
    assert "Constant 'D' cannot be rebound" in str(exc_info.value)


def test_finding_2_constant_shadows_macro_parameter() -> None:
    # Finding 2 (Remediated): Macro parameter collision with defined constant is rejected with E_BINDING.
    source = """
D = 64
def foo(x, D):
    return linear(x, D)
out = foo(x, 128)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 128)})
    assert exc_info.value.code == "E_BINDING"
    assert "Macro parameter 'D' collides with defined constant" in str(exc_info.value)


def test_finding_3_subgraph_to_python_source_indentation_error() -> None:
    # Finding 3 (Remediated): Subgraph modules registered via Registry.subgraph cleanly emit
    # the standalone Python class definition Submodel_<alias> with zero dependency on closures
    # and zero indentation errors.
    reg = Registry.builtins()
    sub_plan = resolve("linear(16)\n", input_shape=("B", 8), output_shape=("B", 16))
    reg.subgraph("my_sub", sub_plan)

    main_plan = resolve("out = my_sub(x)\n", input_shape={"x": ("B", 8)}, output_shape={"out": ("B", 16)}, registry=reg)
    code = main_plan.to_python_source()
    assert "class Submodel_my_sub(torch.nn.Module):" in code
    assert "class Model(torch.nn.Module):" in code
    assert "self.nodes['n_n0'] = Submodel_my_sub()" in code
    ns: dict = {}
    exec(code, ns)
    model = ns["Model"]()
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 16)


def test_macro_formal_parameters_take_precedence() -> None:
    # Formal parameters in macro take precedence and are not shadowed
    source = """
def foo(x, width):
    return linear(x, width)
D = 64
out = foo(x, 128)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 128)})
    assert plan.nodes[0].args["out_features"] == 128


# ==============================================================================
# 6. Weight Tying, State Mutation, Codegen & Checkpoint Loading Stress Tests
# ==============================================================================


def test_declarative_weight_tying_data_ptr_sharing() -> None:
    """Verify exact parameter instance sharing via data_ptr equality."""
    source = """
src = linear(32, name='src')
tgt = linear(32, name='tgt', tie={'weight': 'src.weight'})
"""
    model = network(source, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    src_weight = model["src"].weight
    tgt_weight = model["tgt"].weight
    assert src_weight.data_ptr() == tgt_weight.data_ptr()
    assert src_weight is tgt_weight


def test_declarative_weight_tying_gradient_accumulation_mathematical_parity() -> None:
    """Verify gradient accumulation: in a model with tied weights W,
    grad_W = grad_{W_source} + grad_{W_target}.
    """
    src_tied = """
src = linear(32, name='src')
tgt = linear(32, name='tgt', tie={'weight': 'src.weight'})
"""
    src_untied = """
src = linear(32, name='src')
tgt = linear(32, name='tgt')
"""
    torch.manual_seed(42)
    m_tied = network(src_tied, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    m_untied = network(src_untied, input_shape=("B", 32), output_shape=("B", 32), device="cpu")

    # Initialize untied and tied models with identical weights
    with torch.no_grad():
        w = torch.randn(32, 32)
        b1 = torch.randn(32)
        b2 = torch.randn(32)
        m_tied["src"].weight.copy_(w)
        m_tied["src"].bias.copy_(b1)
        m_tied["tgt"].bias.copy_(b2)

        m_untied["src"].weight.copy_(w)
        m_untied["src"].bias.copy_(b1)
        m_untied["tgt"].weight.copy_(w)
        m_untied["tgt"].bias.copy_(b2)

    x = torch.randn(4, 32)
    out_tied = m_tied(x)
    out_untied = m_untied(x)
    assert torch.allclose(out_tied, out_untied, atol=1e-7)

    loss_tied = (out_tied**2).sum()
    loss_untied = (out_untied**2).sum()

    loss_tied.backward()
    loss_untied.backward()

    grad_tied = m_tied["src"].weight.grad
    grad_src_untied = m_untied["src"].weight.grad
    grad_tgt_untied = m_untied["tgt"].weight.grad

    expected_grad = grad_src_untied + grad_tgt_untied
    assert grad_tied is not None
    assert torch.allclose(grad_tied, expected_grad, atol=1e-6)


def test_transitive_weight_tying() -> None:
    """Verify multi-target transitive weight tying (A -> B -> C)."""
    source = """
a = linear(32, name='a')
b = linear(32, name='b', tie={'weight': 'a.weight'})
c = linear(32, name='c', tie={'weight': 'b.weight'})
"""
    model = network(source, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    p_a = model["a"].weight.data_ptr()
    p_b = model["b"].weight.data_ptr()
    p_c = model["c"].weight.data_ptr()
    assert p_a == p_b == p_c


def test_adversarial_weight_tying_shape_mismatch() -> None:
    """Attempt invalid weight ties with shape mismatch and verify rejection."""
    source = """
src = linear(64, name='src')
tgt = linear(32, name='tgt', tie={'weight': 'src.weight'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(source, input_shape=("B", 64), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code in ("E_SCHEMA", "E_BINDING")
    assert "Tied parameter shape mismatch" in str(exc_info.value)


def test_adversarial_weight_tying_dependency_cycle() -> None:
    """Attempt backward dependency cycle via weight tying and verify rejection."""
    source = """
src = linear(32, name='src', tie={'weight': 'tgt.weight'})
tgt = linear(32, name='tgt')
"""
    with pytest.raises(HNDLError) as exc_info:
        network(source, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code == "E_BINDING"
    assert "Graph contains a cycle" in str(exc_info.value)


def test_adversarial_weight_tying_nonexistent_source_node() -> None:
    """Attempt weight tie to non-existent source node and verify rejection."""
    source = """
tgt = linear(32, name='tgt', tie={'weight': 'ghost_node.weight'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(source, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code == "E_BINDING"
    assert "Tied source node 'ghost_node' not found" in str(exc_info.value)


def test_adversarial_weight_tying_nonexistent_parameters() -> None:
    """Attempt weight tie with non-existent source/target parameter names."""
    src_bad_source_p = """
src = linear(32, name='src')
tgt = linear(32, name='tgt', tie={'weight': 'src.ghost_param'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(src_bad_source_p, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code == "E_SCHEMA"
    assert "does not exist" in str(exc_info.value)

    src_bad_target_p = """
src = linear(32, name='src')
tgt = linear(32, name='tgt', tie={'ghost_param': 'src.weight'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(src_bad_target_p, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code == "E_SCHEMA"
    assert "does not exist" in str(exc_info.value)


def test_state_mutation_safety_checks_outside_context_manager() -> None:
    """Verify that mutations outside allow_state_mutation trigger safety checks."""
    source = "linear(32, name='lin')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    x = torch.randn(2, 16)
    _ = model(x)

    model["lin"].lora_param = nn.Parameter(torch.randn(4, 16))
    with pytest.raises(HNDLError) as exc_info:
        model(x)
    assert exc_info.value.code == "E_RUNTIME"
    assert "registered state of linear 'lin' no longer matches the build" in str(exc_info.value)
    assert "parameter 'nodes.n_lin.lora_param' was added" in str(exc_info.value)


def test_state_mutation_deletion_outside_context_manager() -> None:
    """Verify that parameter deletion outside context manager triggers E_RUNTIME."""
    source = "linear(32, name='lin')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    x = torch.randn(2, 16)
    _ = model(x)

    del model["lin"].bias
    with pytest.raises(HNDLError) as exc_info:
        model(x)
    assert exc_info.value.code == "E_RUNTIME"
    assert "registered state of linear 'lin' no longer matches the build" in str(exc_info.value)
    assert "parameter 'nodes.n_lin.bias' was removed" in str(exc_info.value)


def test_lora_adaptation_inside_allow_state_mutation() -> None:
    """Verify that injecting LoRA parameters inside allow_state_mutation and calling
    rebind_state allows execution of both forward and autograd backward passes.
    """
    source = "linear(32, name='lin')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    x = torch.randn(2, 16, requires_grad=True)
    _ = model(x)

    orig_weight = model["lin"].weight
    r = 4
    lora_A = nn.Parameter(torch.randn(r, 16) * 0.01)
    lora_B = nn.Parameter(torch.randn(32, r) * 0.01)

    with model.allow_state_mutation():
        orig_weight.requires_grad = False
        adapted_weight = nn.Parameter(orig_weight.data + (lora_B @ lora_A))
        model["lin"].weight = adapted_weight
        model.rebind_state()

    out = model(x)
    assert out.shape == (2, 32)
    loss = out.sum()
    loss.backward()
    assert adapted_weight.grad is not None
    assert x.grad is not None


@pytest.mark.parametrize("device", DEVICES)
def test_codegen_multi_layer_parity_isolated_env(device: str) -> None:
    """Verify standalone codegen for multi-layer model runs in isolated namespace
    with ONLY torch and achieves < 1e-6 numerical parity on CPU and CUDA.
    """
    source = """
h1 = relu(linear(x, 64))
h2 = gelu(linear(h1, 128))
out = linear(h2, 32)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    code = plan.to_python_source("MLPModule")
    assert "hndl" not in code

    ns: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code, ns)
    native_model = ns["MLPModule"]().to(device)

    hndl_model = build(plan, device=device)
    native_model.load_state_dict(hndl_model.state_dict())

    x = torch.randn(4, 16, device=device)
    with torch.no_grad():
        out_hndl = hndl_model(x)
        out_native = native_model(x)

    diff = (out_hndl - out_native).abs().max().item()
    assert diff < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_codegen_multi_io_and_residuals(device: str) -> None:
    """Verify standalone codegen for model with multiple inputs, outputs, and skip connections."""
    source = """
h1 = linear(x1, 32)
h2 = linear(x2, 32)
out1 = relu(add(h1, h2))
out2 = linear(out1, 16)
"""
    plan = resolve(
        source,
        input_shape={"x1": ("B", 32), "x2": ("B", 32)},
        output_shape={"out1": ("B", 32), "out2": ("B", 16)},
    )
    code = plan.to_python_source("MultiIOModule")
    assert "hndl" not in code

    ns: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code, ns)
    native_model = ns["MultiIOModule"]().to(device)

    hndl_model = build(plan, device=device)
    native_model.load_state_dict(hndl_model.state_dict())

    x1 = torch.randn(4, 32, device=device)
    x2 = torch.randn(4, 32, device=device)
    with torch.no_grad():
        out_hndl = hndl_model(x1=x1, x2=x2)
        out_native = native_model(x1, x2)

    assert (out_hndl["out1"] - out_native["out1"]).abs().max().item() < 1e-6
    assert (out_hndl["out2"] - out_native["out2"]).abs().max().item() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_codegen_tied_weights_instance_sharing_and_parity(device: str) -> None:
    """Verify standalone codegen preserves declarative weight tying and data_ptr sharing."""
    source = """
fc1 = linear(32, name='fc1')
h = relu(fc1)
fc2 = linear(32, name='fc2', tie={'weight': 'fc1.weight'})
"""
    plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"fc2": ("B", 32)})
    code = plan.to_python_source("TiedLinearModule")
    assert "hndl" not in code

    ns: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code, ns)
    native_model = ns["TiedLinearModule"]().to(device)

    assert (
        native_model.nodes["n_fc1"].weight.data_ptr()
        == native_model.nodes["n_fc2"].weight.data_ptr()
    )

    hndl_model = build(plan, device=device)
    native_model.load_state_dict(hndl_model.state_dict())

    x = torch.randn(4, 32, device=device)
    with torch.no_grad():
        out_hndl = hndl_model(x)
        out_native = native_model(x)

    assert (out_hndl - out_native).abs().max().item() < 1e-6


def test_adversarial_codegen_embedding_and_norm_remediated() -> None:
    """Verify that standalone codegen for embedding, layer_norm, and rms_norm instantiates and runs."""
    plan_emb = resolve(
        "embedding(100, 32)\nlinear(10)\n",
        input_shape={"x": ("B", 8)},
        output_shape={"out": ("B", 8, 10)},
        input_dtype={"x": "int64"},
    )
    code_emb = plan_emb.to_python_source("EmbModel")
    assert "nn.Embedding(100, 32)" in code_emb
    ns_emb: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code_emb, ns_emb)
    emb_model = ns_emb["EmbModel"]()
    assert emb_model is not None
    x_emb = torch.randint(0, 100, (2, 8), dtype=torch.int64)
    out_emb = emb_model(x_emb)
    assert out_emb.shape == (2, 8, 10)

    plan_ln = resolve("layer_norm()\n", input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
    code_ln = plan_ln.to_python_source("LNModel")
    assert "nn.LayerNorm(32, eps=1e-05, elementwise_affine=True, bias=True)" in code_ln
    ns_ln: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code_ln, ns_ln)
    ln_model = ns_ln["LNModel"]()
    assert ln_model is not None
    out_ln = ln_model(torch.randn(2, 32))
    assert out_ln.shape == (2, 32)

    plan_rms = resolve("rms_norm()\n", input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
    code_rms = plan_rms.to_python_source("RMSModel")
    assert "nn.RMSNorm(32, eps=1e-06, elementwise_affine=True)" in code_rms
    ns_rms: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code_rms, ns_rms)
    rms_model = ns_rms["RMSModel"]()
    assert rms_model is not None
    out_rms = rms_model(torch.randn(2, 32))
    assert out_rms.shape == (2, 32)


def test_load_weights_strict_true_vs_false_missing_keys() -> None:
    """Verify strict=True vs strict=False behavior on missing keys."""
    source = """
fc1 = linear(32, name='fc1')
fc2 = linear(16, name='fc2')
"""
    model = network(source, input_shape=("B", 64), output_shape=("B", 16), device="cpu")
    orig_fc2_w = model["fc2"].weight.clone()
    new_fc1_w = torch.randn(32, 64)

    with pytest.raises(RuntimeError) as exc_info:
        load_weights(model, {"nodes.n_fc1.weight": new_fc1_w}, strict=True)
    assert "Missing key(s) in state_dict" in str(exc_info.value)

    load_weights(model, {"nodes.n_fc1.weight": new_fc1_w}, strict=False)
    assert torch.equal(model["fc1"].weight, new_fc1_w)
    assert torch.equal(model["fc2"].weight, orig_fc2_w)


def test_load_weights_strict_true_vs_false_unexpected_keys() -> None:
    """Verify strict=True vs strict=False behavior on unexpected keys."""
    source = "linear(32, name='fc1')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    state = dict(model.state_dict())
    state["unexpected_param"] = torch.randn(10)

    with pytest.raises(RuntimeError) as exc_info:
        load_weights(model, state, strict=True)
    assert "Unexpected key(s) in state_dict" in str(exc_info.value)

    load_weights(model, state, strict=False)


def test_load_weights_prefix_dict_and_callable_mapping() -> None:
    """Verify key mapping via prefix dicts and callable functions."""
    source = """
fc1 = linear(32, name='fc1')
fc2 = linear(16, name='fc2')
"""
    model = network(source, input_shape=("B", 64), output_shape=("B", 16), device="cpu")

    ext_dict_prefix = {
        "external_model.fc1.weight": torch.randn(32, 64),
        "external_model.fc1.bias": torch.randn(32),
        "external_model.fc2.weight": torch.randn(16, 32),
        "external_model.fc2.bias": torch.randn(16),
    }
    load_weights(model, ext_dict_prefix, key_map={"external_model.": ""}, strict=True)
    assert torch.equal(model["fc1"].weight, ext_dict_prefix["external_model.fc1.weight"])
    assert torch.equal(model["fc2"].weight, ext_dict_prefix["external_model.fc2.weight"])

    ext_dict_callable = {
        "legacy_stem_fc1_weight": torch.randn(32, 64),
        "legacy_stem_fc1_bias": torch.randn(32),
        "legacy_stem_fc2_weight": torch.randn(16, 32),
        "legacy_stem_fc2_bias": torch.randn(16),
    }

    def remap_fn(k: str) -> str:
        return k.removeprefix("legacy_stem_").replace("_", ".")

    load_weights(model, ext_dict_callable, key_map=remap_fn, strict=True)
    assert torch.equal(model["fc1"].weight, ext_dict_callable["legacy_stem_fc1_weight"])
    assert torch.equal(model["fc2"].weight, ext_dict_callable["legacy_stem_fc2_weight"])


def test_load_weights_pt_file_on_disk() -> None:
    """Verify loading from .pt checkpoint file on disk."""
    source = "linear(32, name='fc1')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    saved_state = {
        "fc1.weight": torch.randn(32, 16),
        "fc1.bias": torch.randn(32),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(saved_state, tmp_path)
        load_weights(model, tmp_path, strict=True)
        assert torch.equal(model["fc1"].weight, saved_state["fc1.weight"])
        assert torch.equal(model["fc1"].bias, saved_state["fc1.bias"])
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_weights_tied_model_preserves_sharing() -> None:
    """Verify loading weights into a model with tied parameters preserves data_ptr sharing."""
    source = """
tok_emb = embedding(100, 32, name='tok_emb')
out = linear(100, name='out', tie={'weight': 'tok_emb.weight'})
"""
    model = network(
        source,
        input_shape={"x": ("B", 8)},
        output_shape={"out": ("B", 8, 100)},
        input_dtype={"x": "int64"},
        device="cpu",
    )
    assert model["tok_emb"].weight.data_ptr() == model["out"].weight.data_ptr()

    new_weight = torch.randn(100, 32)
    new_bias = torch.randn(100)
    state = {
        "tok_emb.weight": new_weight,
        "out.weight": new_weight,
        "out.bias": new_bias,
    }
    load_weights(model, state, strict=True)
    assert model["tok_emb"].weight.data_ptr() == model["out"].weight.data_ptr()
    assert torch.equal(model["tok_emb"].weight, new_weight)
    assert torch.equal(model["out"].weight, new_weight)


# ==============================================================================
# 7. Gate 8 Remediation Adversarial Stress Tests (v10_m4_r2)
# ==============================================================================


@pytest.mark.parametrize(
    ("source", "blamed", "non_blamed"),
    [
        # Constant in first position
        (
            """
D = 64
D, a, b = split(3)
""",
            "D",
            ["a", "b"],
        ),
        # Constant in middle position
        (
            """
D = 64
x, D, y = split(3)
""",
            "D",
            ["x", "y"],
        ),
        # Constant in last position
        (
            """
D = 64
a, b, c, D = split(4)
""",
            "D",
            ["a", "b", "c"],
        ),
        # Constant with interspersed discards
        (
            """
D = 64
_, _, D, _ = split(4)
""",
            "D",
            ["_"],
        ),
        # List unpacking syntax with constant in middle
        (
            """
D = 64
[x, D, y] = split(3)
""",
            "D",
            ["x", "y"],
        ),
        # Multiple constants: collision diagnostic reports the first colliding constant in unpack order
        (
            """
D = 64
E = 32
a, D, b, E = split(4)
""",
            "D",
            ["a", "b", "E"],
        ),
        (
            """
D = 64
E = 32
a, E, b, D = split(4)
""",
            "E",
            ["a", "b", "D"],
        ),
    ],
)
def test_constant_unpacking_rebind_multiple_positions_blames_exact_constant(
    source: str, blamed: str, non_blamed: list[str]
) -> None:
    """Adversarially verify that unpacking with constants at any position (first, middle, last,
    with discards, list syntax) strictly blames the exact colliding constant name with E_BINDING,
    and never misattributes the error to non-constant targets.
    """
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 12), output_shape=None)
    assert exc_info.value.code == "E_BINDING"
    err_msg = str(exc_info.value)
    assert f"Constant '{blamed}' cannot be rebound" in err_msg
    for target in non_blamed:
        assert f"Constant '{target}'" not in err_msg, (
            f"Diagnostic improperly blamed non-colliding target '{target}': {err_msg}"
        )


def test_for_loop_tuple_target_constant_rebind_raises_e_binding() -> None:
    """Adversarially verify that attempting to rebind a constant as a tuple loop target raises E_BINDING."""
    source = """
D = 64
for D in (1, 2, 3):
    x = linear(x, 16)
out = linear(x, 8)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 8)})
    assert exc_info.value.code == "E_BINDING"
    assert "Constant 'D' cannot be rebound" in str(exc_info.value)


def test_macro_def_parameter_collides_with_constant_raises_e_binding() -> None:
    """Adversarially verify that defining a macro parameter that collides with an existing constant
    is strictly rejected with E_BINDING at definition time.
    """
    source = """
D = 64
def block(x, D):
    return linear(x, D)
out = block(x, 128)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 128)})
    assert exc_info.value.code == "E_BINDING"
    assert "Macro parameter 'D' collides with defined constant" in str(exc_info.value)


def test_macro_def_multiple_parameter_collisions_blames_first() -> None:
    """Adversarially verify that multiple colliding parameters blame the first colliding parameter in order."""
    source = """
D = 64
WIDTH = 128
def block(x, D, WIDTH):
    return linear(x, D)
out = block(x, 64, 128)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert exc_info.value.code == "E_BINDING"
    assert "Macro parameter 'D' collides with defined constant" in str(exc_info.value)


def test_macro_formal_parameters_not_constants_strictly_respected() -> None:
    """Adversarially verify that formal parameters that are not defined as constants strictly
    respect arguments passed at call time across multiple distinct invocations, and do not
    interfere with global constants.
    """
    source = """
D = 64
def block(x, width):
    h = linear(x, width)
    return linear(h, D)

h1 = block(x, 128)
out = block(h1, 256)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert len(plan.nodes) == 4
    # Invocation 1: width=128, D=64
    assert plan.nodes[0].args["out_features"] == 128
    assert plan.nodes[1].args["out_features"] == 64
    # Invocation 2: width=256, D=64
    assert plan.nodes[2].args["out_features"] == 256
    assert plan.nodes[3].args["out_features"] == 64

    model = build(plan, device="cpu")
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 64)


def test_nested_macro_calls_with_parameters_and_constants() -> None:
    """Verify deep 3-level nested macro calls correctly propagate parameters and resolve constants."""
    source = """
D = 64
def inner3(x, dim):
    return linear(x, dim)

def inner2(x, dim):
    h = inner3(x, dim)
    return relu(h)

def inner1(x, dim):
    h = inner2(x, dim)
    return linear(h, D)

out = inner1(x, 32)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert len(plan.nodes) == 3
    assert plan.nodes[0].args["out_features"] == 32
    assert plan.nodes[2].args["out_features"] == 64

    model = build(plan, device="cpu")
    x = torch.randn(2, 16, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 64)
    loss = (out**2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_nested_macro_lexical_scoping_parameter_shadowing_isolation() -> None:
    """Adversarially verify that nested macros using the same formal parameter name 'p'
    maintain complete frame isolation via stack push/pop.
    """
    source = """
def inner(x, p):
    return linear(x, p)

def middle(x, p):
    h = inner(x, 16)
    return linear(h, p)

def outer(x, p):
    h = middle(x, 32)
    return linear(h, p)

out = outer(x, 64)
"""
    plan = resolve(source, input_shape={"x": ("B", 8)}, output_shape={"out": ("B", 64)})
    resolved_dims = [n.args["out_features"] for n in plan.nodes]
    # inner got p=16, middle restored p=32, outer restored p=64
    assert resolved_dims == [16, 32, 64]


def test_nested_macro_definition_strictly_forbidden() -> None:
    """Adversarially verify that attempting to define a macro inside a macro is rejected with E_SYNTAX."""
    source = """
def outer(x):
    def inner(y):
        return linear(y, 16)
    return inner(x)
out = outer(x)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 8)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code == "E_SYNTAX"
    assert "Nested macro definitions are not permitted" in str(exc_info.value)


def test_macro_backward_autograd_mathematical_parity_vs_pytorch() -> None:
    """Adversarially verify that macro execution in HNDL has exact numerical parity
    with a manually written unrolled PyTorch model in both forward output and autograd gradients.
    """
    source = """
def mlp_block(x, hidden_dim, out_dim):
    h = linear(x, hidden_dim)
    a = relu(h)
    return linear(a, out_dim)

h = mlp_block(x, 32, 16)
out = mlp_block(h, 24, 8)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 8)})
    model = build(plan, device="cpu")

    class ManualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 24)
            self.fc4 = nn.Linear(24, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x))
            h = self.fc2(h)
            h2 = torch.relu(self.fc3(h))
            return self.fc4(h2)

    manual = ManualMLP()

    # Synchronize initial parameter weights
    with torch.no_grad():
        manual.fc1.weight.copy_(model.nodes["n_mlp_block_0_n0"].weight)
        manual.fc1.bias.copy_(model.nodes["n_mlp_block_0_n0"].bias)
        manual.fc2.weight.copy_(model.nodes["n_mlp_block_0_n2"].weight)
        manual.fc2.bias.copy_(model.nodes["n_mlp_block_0_n2"].bias)
        manual.fc3.weight.copy_(model.nodes["n_mlp_block_1_n3"].weight)
        manual.fc3.bias.copy_(model.nodes["n_mlp_block_1_n3"].bias)
        manual.fc4.weight.copy_(model.nodes["n_mlp_block_1_n5"].weight)
        manual.fc4.bias.copy_(model.nodes["n_mlp_block_1_n5"].bias)

    x_hndl = torch.randn(4, 16, requires_grad=True)
    x_manual = x_hndl.clone().detach().requires_grad_(True)

    out_hndl = model(x_hndl)
    out_manual = manual(x_manual)
    assert torch.allclose(out_hndl, out_manual, atol=1e-6)

    loss_hndl = (out_hndl**2).sum()
    loss_manual = (out_manual**2).sum()

    loss_hndl.backward()
    loss_manual.backward()

    # Check input gradients match
    assert torch.allclose(x_hndl.grad, x_manual.grad, atol=1e-6)

    # Check parameter gradients match
    assert torch.allclose(model.nodes["n_mlp_block_0_n0"].weight.grad, manual.fc1.weight.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_0_n0"].bias.grad, manual.fc1.bias.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_0_n2"].weight.grad, manual.fc2.weight.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_0_n2"].bias.grad, manual.fc2.bias.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_1_n3"].weight.grad, manual.fc3.weight.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_1_n3"].bias.grad, manual.fc3.bias.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_1_n5"].weight.grad, manual.fc4.weight.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_mlp_block_1_n5"].bias.grad, manual.fc4.bias.grad, atol=1e-6)


def test_loop_and_macro_backward_autograd_mathematical_parity() -> None:
    """Adversarially verify that loops containing macros have exact autograd gradient flow
    matching unrolled manual PyTorch models.
    """
    source = """
def res_block(x, hidden_dim):
    h = linear(x, hidden_dim)
    h = gelu(h)
    h = linear(h, 16)
    return add(x, h)

for _ in range(3):
    x = res_block(x, 32)

out = linear(x, 8)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 8)})
    model = build(plan, device="cpu")

    class ManualLoopMacro(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.ModuleDict({"fc1": nn.Linear(16, 32), "fc2": nn.Linear(32, 16)})
                for _ in range(3)
            ])
            self.out = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                h = torch.nn.functional.gelu(block["fc1"](x))
                h = block["fc2"](h)
                x = x + h
            return self.out(x)

    manual = ManualLoopMacro()

    with torch.no_grad():
        for i in range(3):
            fc1 = model.nodes[f"n_res_block_{i}_n{i * 4}"]
            fc2 = model.nodes[f"n_res_block_{i}_n{i * 4 + 2}"]
            manual.blocks[i]["fc1"].weight.copy_(fc1.weight)
            manual.blocks[i]["fc1"].bias.copy_(fc1.bias)
            manual.blocks[i]["fc2"].weight.copy_(fc2.weight)
            manual.blocks[i]["fc2"].bias.copy_(fc2.bias)
        manual.out.weight.copy_(model.nodes["n_n12"].weight)
        manual.out.bias.copy_(model.nodes["n_n12"].bias)

    x_h = torch.randn(4, 16, requires_grad=True)
    x_m = x_h.clone().detach().requires_grad_(True)

    out_h = model(x_h)
    out_m = manual(x_m)
    assert torch.allclose(out_h, out_m, atol=1e-6)

    loss_h = (out_h**2).sum()
    loss_m = (out_m**2).sum()

    loss_h.backward()
    loss_m.backward()

    assert torch.allclose(x_h.grad, x_m.grad, atol=1e-6)
    for i in range(3):
        fc1 = model.nodes[f"n_res_block_{i}_n{i * 4}"]
        fc2 = model.nodes[f"n_res_block_{i}_n{i * 4 + 2}"]
        assert torch.allclose(fc1.weight.grad, manual.blocks[i]["fc1"].weight.grad, atol=1e-6)
        assert torch.allclose(fc1.bias.grad, manual.blocks[i]["fc1"].bias.grad, atol=1e-6)
        assert torch.allclose(fc2.weight.grad, manual.blocks[i]["fc2"].weight.grad, atol=1e-6)
        assert torch.allclose(fc2.bias.grad, manual.blocks[i]["fc2"].bias.grad, atol=1e-6)
    assert torch.allclose(model.nodes["n_n12"].weight.grad, manual.out.weight.grad, atol=1e-6)


def test_multi_io_macro_backward_autograd_gradient_flow() -> None:
    """Verify that a macro with multiple outputs supports full autograd backward flow."""
    source = """
def dual_head(x, d1, d2):
    h1 = linear(x, d1)
    h2 = linear(x, d2)
    return h1, h2

y1, y2 = dual_head(x, 16, 16)
y = add(y1, y2)
z1, z2 = dual_head(y, 8, 8)
m = mul(z1, z2)
out = linear(m, 4)
"""
    plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 4)})
    model = build(plan, device="cpu")

    x = torch.randn(3, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (3, 4)

    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all() and (x.grad.abs().sum() > 0)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} grad is None"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
        assert p.grad.abs().sum() > 0, f"{name} grad is zero"


def test_nested_macro_tuple_loop_autograd_analytical_gradient_parity() -> None:
    """Verify that nested macros called within tuple loops produce analytical autograd gradients."""
    source = """
def scale_block(x, factor):
    return scale(x, factor)

def residual_block(x, factor):
    h = scale_block(x, factor)
    return add(x, h)

for f in (0.5, 0.25, 0.125):
    x = residual_block(x, f)

out = linear(x, 8)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 8)})
    model = build(plan, device="cpu")

    x = torch.randn(2, 16, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Theoretical multiplier: (1 + 0.5) * (1 + 0.25) * (1 + 0.125) = 2.109375
    multiplier = 1.5 * 1.25 * 1.125
    expected_x_grad = multiplier * model.nodes["n_n6"].weight.sum(dim=0).unsqueeze(0).expand(2, -1)
    diff = (x.grad - expected_x_grad).abs().max().item()
    assert diff < 1e-6, f"Empirical grad differs from analytical by {diff}"


def test_deep_macro_loop_composition_codegen_autograd() -> None:
    """Verify that a deep composition of macros, range loops, and constants compiles
    via to_python_source into a standalone model that runs both forward and backward passes.
    """
    source = """
DEPTH = 2
INNER_DIM = 16

def mlp(x, hidden):
    h = linear(x, hidden)
    h = gelu(h)
    return linear(h, hidden)

def block(x, hidden):
    res = x
    for _ in range(DEPTH):
        x = mlp(x, hidden)
    return add(res, x)

for _ in range(2):
    x = block(x, INNER_DIM)

out = linear(x, 4)
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 4)})
    py_code = plan.to_python_source("DeepCompositionModel")

    ns: dict = {}
    exec(py_code, ns)
    model = ns["DeepCompositionModel"]()

    x = torch.randn(4, 16, requires_grad=True)
    out = model(x)
    assert out.shape == (4, 4)

    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all() and (x.grad.abs().sum() > 0)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} grad is None"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
        assert p.grad.abs().sum() > 0, f"{name} grad is zero"


# ==============================================================================
# 8. Gate 8 Standalone Codegen Remediations Adversarial Stress Tests (Challenger 2)
# ==============================================================================


def test_gate8_adversarial_codegen_embedding_and_norm_parameter_lookups() -> None:
    """Adversarially challenge and stress-test codegen parameter lookups for
    embedding, layer_norm, and rms_norm across diverse configurations:
    1. Verify constructors emit concrete numbers (e.g. nn.Embedding(100, 32),
       nn.LayerNorm(32, eps=1e-05, elementwise_affine=True, bias=True),
       nn.RMSNorm(32, eps=1e-06, elementwise_affine=True)).
    2. Verify zero occurrences of None as positional arguments in emitted code
       via AST node traversal.
    """
    reg = Registry.builtins()

    configs = [
        # Standard configuration with defaults
        (
            "tok = embedding(100, 32)\nln = layer_norm()\nrms = rms_norm()\n",
            {"x": ("B", 4)},
            {"rms": ("B", 4, 32)},
            "int64",
            [
                "nn.Embedding(100, 32)",
                "nn.LayerNorm(32, eps=1e-05, elementwise_affine=True, bias=True)",
                "nn.RMSNorm(32, eps=1e-06, elementwise_affine=True)",
            ],
        ),
        # Keyword arguments, non-default eps, bias=False
        (
            "tok = embedding(num_embeddings=200, dim=64)\n"
            "ln = layer_norm(eps=1e-4, affine=True, bias=False)\n"
            "rms = rms_norm(eps=1e-5, affine=False)\n",
            {"x": ("B", 6)},
            {"rms": ("B", 6, 64)},
            "int64",
            [
                "nn.Embedding(200, 64)",
                "nn.LayerNorm(64, eps=0.0001, elementwise_affine=True, bias=False)",
                "nn.RMSNorm(64, eps=1e-05, elementwise_affine=False)",
            ],
        ),
        # Affine=False on layer_norm and custom eps
        (
            "tok = embedding(50, 16)\n"
            "ln = layer_norm(eps=1e-3, affine=False, bias=False)\n"
            "rms = rms_norm()\n",
            {"x": ("B", 10)},
            {"rms": ("B", 10, 16)},
            "int64",
            [
                "nn.Embedding(50, 16)",
                "nn.LayerNorm(16, eps=0.001, elementwise_affine=False, bias=False)",
                "nn.RMSNorm(16, eps=1e-06, elementwise_affine=True)",
            ],
        ),
        # Inferred embedding dimension resolved from output contract
        (
            "tok = embedding(128)\nln = layer_norm()\n",
            {"x": ("B", 8)},
            {"ln": ("B", 8, 48)},
            "int64",
            [
                "nn.Embedding(128, 48)",
                "nn.LayerNorm(48, eps=1e-05, elementwise_affine=True, bias=True)",
            ],
        ),
    ]

    for src, in_shape, out_shape, in_dtype, expected_constructors in configs:
        plan = resolve(
            src,
            input_shape=in_shape,
            output_shape=out_shape,
            input_dtype=in_dtype,
            registry=reg,
        )
        code = plan.to_python_source("ParamLookupModel")

        # 1. Concrete constructor verification
        for expected in expected_constructors:
            assert expected in code, f"Expected constructor {expected!r} not found in emitted source:\n{code}"

        # 2. AST traversal: verify zero occurrences of None as positional arguments
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for i, arg in enumerate(node.args):
                    assert not (isinstance(arg, ast.Constant) and arg.value is None), (
                        f"Found None as positional argument {i} in call {ast.unparse(node)!r} in emitted code:\n{code}"
                    )


def test_gate8_adversarial_codegen_subgraphs_cleanliness_and_no_closures() -> None:
    """Adversarially challenge codegen with composite subgraphs (single-output, multi-output,
    multi-input, and nested):
    1. Verify zero IndentationError or syntax errors.
    2. Verify zero closures over plan or build.
    3. Verify zero imports or mentions of hndl.
    4. Verify dedenting: custom class definitions start at column 0.
    """
    reg = Registry.builtins()

    # Subgraph 1: Single output
    sub1 = resolve("linear(16)\nlayer_norm()\n", input_shape=("B", 8, 32), output_shape=("B", 8, 16))
    reg.subgraph("s1", sub1)

    # Subgraph 2: Multi output
    sub2 = resolve(
        """
o1 = linear(24)
o2 = linear(32)
""",
        input_shape={"x": ("B", 8, 16)},
        output_shape={"o1": ("B", 8, 24), "o2": ("B", 8, 32)},
    )
    reg.subgraph("s2", sub2)

    # Subgraph 3 & 4: Nested subgraphs
    s_inner = resolve("linear(16)\n", input_shape=("B", 8, 16), output_shape=("B", 8, 16))
    reg.subgraph("s_inner", s_inner)

    s_outer = resolve("s_inner()\nlinear(32)\n", input_shape=("B", 8, 16), output_shape=("B", 8, 32), registry=reg)
    reg.subgraph("s_outer", s_outer)

    # Composite top-level model exercising all subgraphs
    full_src = """
tok = embedding(100, 32)
ln = layer_norm()
h1 = s1(ln)
y1, y2 = s2(h1)
ho = s_outer(h1)
p1 = linear(y1, 10)
p2 = linear(y2, 10)
po = linear(ho, 10)
s12 = add(p1, p2)
out = add(s12, po)
"""
    plan = resolve(
        full_src,
        input_shape={"x": ("B", 8)},
        output_shape={"out": ("B", 8, 10)},
        input_dtype={"x": "int64"},
        registry=reg,
    )
    code = plan.to_python_source("FullModel")

    # 1. Zero IndentationError / valid AST parse
    tree = ast.parse(code)
    assert tree is not None

    # 2. Zero closures or calls to plan/build, zero hndl imports, column 0 class defs
    for line in code.splitlines():
        assert "import hndl" not in line, f"Illegal hndl import in emitted code line: {line}"
        assert "from hndl" not in line, f"Illegal from hndl in emitted code line: {line}"
        assert "build(" not in line, f"Closure over build found in emitted code line: {line}"
        assert "_SubgraphModule" not in line, f"Reference to _SubgraphModule found in line: {line}"
        if line.startswith("class "):
            assert not line.startswith(" "), f"Class definition not dedented to column 0: {line}"
            assert not line.startswith("\t"), f"Class definition indented with tab: {line}"


@pytest.mark.parametrize("device", DEVICES)
def test_gate8_adversarial_standalone_execution_parity_and_backprop(device: str) -> None:
    """Adversarially challenge standalone execution and numerical parity (< 1e-6)
    across CPU and CUDA in an isolated Python namespace (only torch imported):
    1. Composite model with embedding, layer_norm, rms_norm, single-output subgraph,
       multi-output subgraph, nested subgraphs, and tied weights.
    2. State dict loading with .submodel. key mapping.
    3. Exact numerical parity (< 1e-6) between HNDL build() and standalone model.
    4. Full autograd backpropagation verifying gradients on inputs, parameters, and
       tied weight sharing data pointer equality.
    """
    reg = Registry.builtins()

    sub1 = resolve("linear(16)\nlayer_norm()\n", input_shape=("B", 8, 32), output_shape=("B", 8, 16))
    reg.subgraph("s1", sub1)

    sub2 = resolve(
        """
o1 = linear(24)
o2 = linear(32)
""",
        input_shape={"x": ("B", 8, 16)},
        output_shape={"o1": ("B", 8, 24), "o2": ("B", 8, 32)},
    )
    reg.subgraph("s2", sub2)

    s_inner = resolve("linear(16)\n", input_shape=("B", 8, 16), output_shape=("B", 8, 16))
    reg.subgraph("s_inner", s_inner)

    s_outer = resolve("s_inner()\nlinear(32)\n", input_shape=("B", 8, 16), output_shape=("B", 8, 32), registry=reg)
    reg.subgraph("s_outer", s_outer)

    full_src = """
tok = embedding(100, 32, name='tok')
ln = layer_norm()
rms = rms_norm()
h1 = s1(rms)
y1, y2 = s2(h1)
ho = s_outer(h1)
p1 = linear(y1, 32)
p2 = linear(y2, 32)
po = linear(ho, 32)
s12 = add(p1, p2)
tot = add(s12, po)
out = linear(tot, 100, name='head', tie={'weight': 'tok.weight'})
"""
    plan = resolve(
        full_src,
        input_shape={"x": ("B", 8)},
        output_shape={"out": ("B", 8, 100)},
        input_dtype={"x": "int64"},
        registry=reg,
    )
    code = plan.to_python_source("AdversarialFullModel")

    # Execute in isolated namespace containing ONLY __builtins__ and torch
    isolated_ns: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code, isolated_ns)
    model_cls = isolated_ns["AdversarialFullModel"]
    native_model = model_cls().to(device)

    # Verify tied weight parameter sharing prior to state loading
    assert native_model.nodes["n_head"].weight.data_ptr() == native_model.nodes["n_tok"].weight.data_ptr()

    # Build HNDL model and map state dict
    hndl_model = build(plan, device=device)
    mapped_sd = {k.replace(".submodel.", "."): v for k, v in hndl_model.state_dict().items()}
    native_model.load_state_dict(mapped_sd)

    # Verify tied weight sharing is preserved after load_state_dict
    assert native_model.nodes["n_head"].weight.data_ptr() == native_model.nodes["n_tok"].weight.data_ptr()

    # Forward pass: numerical parity check (< 1e-6)
    x = torch.randint(0, 100, (4, 8), dtype=torch.int64, device=device)
    with torch.no_grad():
        out_hndl = hndl_model(x)
        out_native = native_model(x)

    max_diff = (out_hndl - out_native).abs().max().item()
    assert max_diff < 1e-6, f"Parity violation on device {device}: max diff {max_diff}"

    # Autograd backpropagation
    native_model.train()
    out_train = native_model(x)
    loss = out_train.sum()
    loss.backward()

    # Verify non-None, non-NaN gradients across all model parameters
    for name, param in native_model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has None grad"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"

    # Verify tied parameter gradients share storage
    assert (
        native_model.nodes["n_tok"].weight.grad.data_ptr()
        == native_model.nodes["n_head"].weight.grad.data_ptr()
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gate8_adversarial_standalone_multi_output_subgraph_numerical_parity(device: str) -> None:
    """Verify numerical parity (< 1e-6) and backward autograd on a top-level model
    returning dictionary outputs with multi-output subgraphs across CPU and CUDA.
    """
    reg = Registry.builtins()

    sub = resolve(
        """
o1 = linear(16)
o2 = linear(32)
""",
        input_shape={"x": ("B", 8)},
        output_shape={"o1": ("B", 16), "o2": ("B", 32)},
    )
    reg.subgraph("multi_branch", sub)

    main_plan = resolve(
        """
h1, h2 = multi_branch(x)
y1 = layer_norm(h1)
y2 = rms_norm(h2)
""",
        input_shape={"x": ("B", 8)},
        output_shape={"y1": ("B", 16), "y2": ("B", 32)},
        registry=reg,
    )
    code = main_plan.to_python_source("MultiOutputTopModel")

    isolated_ns: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code, isolated_ns)
    model = isolated_ns["MultiOutputTopModel"]().to(device)

    hndl_mod = build(main_plan, device=device)
    mapped_sd = {k.replace(".submodel.", "."): v for k, v in hndl_mod.state_dict().items()}
    model.load_state_dict(mapped_sd)

    x_test = torch.randn(4, 8, device=device)
    with torch.no_grad():
        out_h = hndl_mod(x_test)
        out_s = model(x_test)

    assert isinstance(out_s, dict)
    assert "y1" in out_s and "y2" in out_s
    diff_y1 = (out_h["y1"] - out_s["y1"]).abs().max().item()
    diff_y2 = (out_h["y2"] - out_s["y2"]).abs().max().item()
    assert diff_y1 < 1e-6, f"Device {device} y1 diff: {diff_y1}"
    assert diff_y2 < 1e-6, f"Device {device} y2 diff: {diff_y2}"

    # Autograd backprop through dictionary outputs
    x_train = torch.randn(4, 8, device=device, requires_grad=True)
    out_train = model(x_train)
    loss = out_train["y1"].sum() + out_train["y2"].sum()
    loss.backward()
    assert x_train.grad is not None and not torch.isnan(x_train.grad).any()
    for name, p in model.named_parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any(), f"{name} grad missing or NaN"


