"""Comprehensive test suite for Milestone 4 (Extensibility & Ecosystem Interoperability)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import hndl
from hndl import HNDLError, Registry, ops, resolve
from hndl.operator import make_operator
from hndl.torch import build, network


# ==============================================================================
# 1. .hndl Constants
# ==============================================================================


def test_scalar_constant_binding_and_usage() -> None:
    source = """
D = 64
H = 128
linear(D)
relu()
linear(H)
"""
    plan = resolve(source, input_shape=("B", 32), output_shape=("B", 128))
    assert plan.nodes[0].args["out_features"] == 64
    assert plan.nodes[2].args["out_features"] == 128


def test_constant_rebinding_raises_e_binding() -> None:
    source = """
D = 64
D = 128
linear(D)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape=("B", 32), output_shape=("B", 128))
    assert exc_info.value.code == "E_BINDING"
    assert "Constant 'D' cannot be rebound" in str(exc_info.value)


def test_constant_used_in_loop_count() -> None:
    source = """
DEPTH = 3
for _ in range(DEPTH):
    x = linear(x, 32)
out = x
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    assert len(plan.nodes) == 3


# ==============================================================================
# 2. .hndl AST Def Macros
# ==============================================================================


def test_def_macro_expansion() -> None:
    source = """
def mlp_block(x, hidden):
    h = linear(x, hidden)
    a = relu(h)
    return linear(a, hidden)

out = mlp_block(x, 64)
"""
    plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 64)})
    assert len(plan.nodes) == 3
    # Check that scoped names were generated for macro nodes
    assert any("mlp_block" in node.id for node in plan.nodes)


def test_def_macro_multiple_calls_have_distinct_scopes() -> None:
    source = """
def mlp_step(x, dim):
    return relu(linear(x, dim))

h1 = mlp_step(x, 32)
h2 = mlp_step(h1, 64)
out = h2
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert len(plan.nodes) == 4
    node_ids = [n.id for n in plan.nodes]
    assert len(set(node_ids)) == 4


def test_def_macro_recursion_rejected() -> None:
    source = """
def recursive(x, depth):
    return recursive(x, depth)

out = recursive(x, 2)
"""
    with pytest.raises(HNDLError) as exc_info:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    assert exc_info.value.code in ("E_SYNTAX", "E_OPERATOR")


# ==============================================================================
# 3. .hndl Tuple Loops
# ==============================================================================


def test_tuple_loop_unrolling() -> None:
    source = """
for width in (32, 64, 128):
    x = linear(x, width)
    x = relu(x)
out = x
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 128)})
    assert len(plan.nodes) == 6
    assert plan.nodes[0].args["out_features"] == 32
    assert plan.nodes[2].args["out_features"] == 64
    assert plan.nodes[4].args["out_features"] == 128


def test_macro_called_inside_tuple_loop() -> None:
    source = """
def block(x, w):
    return relu(linear(x, w))

for c in (32, 64):
    x = block(x, c)
out = x
"""
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
    assert len(plan.nodes) == 4


# ==============================================================================
# 4. Scopes (ops.scope)
# ==============================================================================


def test_ops_scope_context_manager() -> None:
    def network_fn(x):
        with ops.scope("encoder"):
            h = ops.relu(ops.linear(x, 32))
        with ops.scope("decoder"):
            out = ops.linear(h, 16)
        return out

    plan = hndl.resolve_callable(network_fn, input_shape=("B", 16), output_shape=("B", 16))
    node_ids = [n.id for n in plan.nodes]
    assert any(nid.startswith("encoder_") for nid in node_ids)
    assert any(nid.startswith("decoder_") for nid in node_ids)


def test_nested_ops_scope() -> None:
    def network_fn(x):
        with ops.scope("stage1"):
            with ops.scope("block_a"):
                out = ops.linear(x, 32)
        return out

    plan = hndl.resolve_callable(network_fn, input_shape=("B", 16), output_shape=("B", 32))
    assert plan.nodes[0].id.startswith("stage1_block_a_")


# ==============================================================================
# 5. Composite Subgraphs
# ==============================================================================


def test_registry_subgraph_registration_and_execution() -> None:
    reg = Registry.builtins()
    sub_plan = resolve("linear(32)\nrelu()\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    reg.subgraph("my_subgraph", sub_plan)
    assert "my_subgraph" in reg._aliases

    main_source = "out = my_subgraph(x)\n"
    main_plan = resolve(main_source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)}, registry=reg)
    assert len(main_plan.nodes) == 1

    model = build(main_plan, device="cpu", registry=reg)
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 32)


def test_subgraph_gradient_backprop() -> None:
    reg = Registry.builtins()
    sub_plan = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    reg.subgraph("linear_sub", sub_plan)

    main_plan = resolve("out = linear_sub(x)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)}, registry=reg)
    model = build(main_plan, device="cpu", registry=reg)
    x = torch.randn(2, 16, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 16)


# ==============================================================================
# 6. Declarative Weight Tying
# ==============================================================================


def test_declarative_weight_tying_storage_and_gradient() -> None:
    source = """
tok_emb = embedding(100, 32)
out = linear(tok_emb, 100, tie={'weight': 'tok_emb.weight'})
"""
    model = network(
        source,
        input_shape={"x": ("B", 8)},
        output_shape={"out": ("B", 8, 100)},
        input_dtype={"x": "int64"},
        device="cpu",
    )
    # Check data pointer equality
    emb_w = model["tok_emb"].weight
    lin_w = model["out"].weight
    assert emb_w.data_ptr() == lin_w.data_ptr()

    # Forward & backward to verify shared gradient accumulation
    tokens = torch.randint(0, 100, (2, 8))
    out = model(tokens)
    loss = out.sum()
    loss.backward()
    assert emb_w.grad is not None


def test_weight_tying_shape_mismatch_rejected() -> None:
    source = """
tok_emb = embedding(100, 32)
out = linear(tok_emb, 64, tie={'weight': 'tok_emb.weight'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(
            source,
            input_shape={"x": ("B", 8)},
            output_shape={"out": ("B", 8, 64)},
            input_dtype={"x": "int64"},
            device="cpu",
        )
    assert exc_info.value.code in ("E_SCHEMA", "E_BINDING")


# ==============================================================================
# 7. Post-Build State Mutation API
# ==============================================================================


def test_allow_state_mutation_and_rebind() -> None:
    source = "linear(32, name='lin')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")

    orig_weight = model["lin"].weight
    with model.allow_state_mutation():
        # Replace parameter as in LoRA adapter injection or quantization
        new_weight = nn.Parameter(orig_weight.data * 3.0)
        model["lin"].weight = new_weight
        model.rebind_state()

    # Model executes forward pass smoothly without E_RUNTIME
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 32)
    assert torch.equal(model["lin"].weight, new_weight)


# ==============================================================================
# 8. Standalone PyTorch Codegen
# ==============================================================================


def test_plan_to_python_source_roundtrip() -> None:
    plan = resolve(
        "h = relu(linear(x, 64))\nout = linear(h, 32)\n",
        input_shape={"x": ("B", 16)},
        output_shape={"out": ("B", 32)},
    )
    src = plan.to_python_source(class_name="MyNetwork")
    assert "class MyNetwork(torch.nn.Module):" in src
    assert "def forward(self, x):" in src

    # Execute standalone code in isolated namespace
    ns: dict[str, Any] = {}
    exec(src, ns)
    cls = ns["MyNetwork"]
    native_model = cls()

    # Load weights from HNDL build
    hndl_model = build(plan, device="cpu")
    native_model.load_state_dict(hndl_model.state_dict())

    x = torch.randn(3, 16)
    with torch.no_grad():
        out_h = hndl_model(x)
        out_n = native_model(x)
    assert torch.allclose(out_h, out_n, atol=1e-6)


# ==============================================================================
# 9. Checkpoint Loading with Key Mapping
# ==============================================================================


def test_load_weights_dict_and_file() -> None:
    import hndl.torch as htorch

    source = "linear(32, name='head')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")

    # In-memory dict with prefix remapping
    external_state = {
        "classifier.head.weight": torch.randn(32, 16),
        "classifier.head.bias": torch.randn(32),
    }
    htorch.load_weights(model, external_state, key_map={"classifier.head.": "head."})
    assert torch.equal(model["head"].weight, external_state["classifier.head.weight"])
    assert torch.equal(model["head"].bias, external_state["classifier.head.bias"])

    # From .pt file on disk
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(external_state, tmp_path)
        htorch.load_weights(model, tmp_path, key_map={"classifier.head.": "head."})
        assert torch.equal(model["head"].weight, external_state["classifier.head.weight"])
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ==============================================================================
# 10. Plugin Discovery
# ==============================================================================


def test_discover_plugins_with_mocked_entry_points() -> None:
    reg = Registry()

    # Mock an EntryPoint
    mock_ep = MagicMock()
    mock_ep.name = "dummy_plugin"

    class DummyPlugin(nn.Module):
        def forward(self, x):
            return x

    dummy_op = make_operator(DummyPlugin, "dummy_plugin", shape="x[B, ...] -> out[B, ...]", summary="Dummy plugin op")
    mock_ep.load.return_value = dummy_op

    with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
        reg.discover_plugins(group="hndl.operators")

    assert "dummy_plugin" in reg._aliases
    assert reg.get("dummy_plugin") is dummy_op
