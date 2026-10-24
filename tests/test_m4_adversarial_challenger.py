"""Adversarial stress-testing suite for Milestone 4 (Extensibility & Interoperability).

Empirical verification suite by challenger_v10_m4_2 targeting:
1. Declarative Weight Tying
2. Post-Build State Mutation API
3. Standalone PyTorch Codegen
4. Checkpoint Loading & Key Mapping
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from hndl import HNDLError, resolve
from hndl.testing import available_devices
from hndl.torch import build, load_weights, network

DEVICES = available_devices()


# ==============================================================================
# TARGET 1: Declarative Weight Tying
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
    # src feeds tgt, but src ties its weight to tgt.weight
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
    # Bad source parameter
    src_bad_source_p = """
src = linear(32, name='src')
tgt = linear(32, name='tgt', tie={'weight': 'src.ghost_param'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(src_bad_source_p, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code == "E_SCHEMA"
    assert "does not exist" in str(exc_info.value)

    # Bad target parameter
    src_bad_target_p = """
src = linear(32, name='src')
tgt = linear(32, name='tgt', tie={'ghost_param': 'src.weight'})
"""
    with pytest.raises(HNDLError) as exc_info:
        network(src_bad_target_p, input_shape=("B", 32), output_shape=("B", 32), device="cpu")
    assert exc_info.value.code == "E_SCHEMA"
    assert "does not exist" in str(exc_info.value)


# ==============================================================================
# TARGET 2: State Mutation API
# ==============================================================================


def test_state_mutation_safety_checks_outside_context_manager() -> None:
    """Verify that mutations outside allow_state_mutation trigger safety checks."""
    source = "linear(32, name='lin')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    x = torch.randn(2, 16)
    _ = model(x)  # warmup & validate

    # 1. Adding parameter outside context manager
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


# ==============================================================================
# TARGET 3: Standalone PyTorch Codegen
# ==============================================================================


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

    # Verify data_ptr equality in standalone module
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


def test_adversarial_codegen_embedding_and_norm_bugs() -> None:
    """Demonstrate codegen bugs for embedding, layer_norm, and rms_norm.
    _format_module_constructor queries kwargs.get('embedding_dim') and
    kwargs.get('normalized_shape') which are None, causing instantiation failure.
    """
    # 1. embedding: kwargs has 'dim', but codegen expects 'embedding_dim'
    plan_emb = resolve(
        "embedding(100, 32)\nlinear(10)\n",
        input_shape={"x": ("B", 8)},
        output_shape={"out": ("B", 8, 10)},
        input_dtype={"x": "int64"},
    )
    code_emb = plan_emb.to_python_source("EmbModel")
    assert "nn.Embedding(100, None)" in code_emb
    ns_emb: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code_emb, ns_emb)
    with pytest.raises(TypeError) as exc_info:
        ns_emb["EmbModel"]()
    assert "NoneType" in str(exc_info.value)

    # 2. layer_norm: kwargs has 'D', but codegen expects 'normalized_shape'
    plan_ln = resolve("layer_norm()\n", input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
    code_ln = plan_ln.to_python_source("LNModel")
    assert "nn.LayerNorm(None" in code_ln
    ns_ln: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code_ln, ns_ln)
    with pytest.raises(TypeError) as exc_info:
        ns_ln["LNModel"]()
    assert "'NoneType' object is not iterable" in str(exc_info.value)

    # 3. rms_norm: kwargs has 'D', but codegen expects 'normalized_shape'
    plan_rms = resolve("rms_norm()\n", input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
    code_rms = plan_rms.to_python_source("RMSModel")
    assert "nn.RMSNorm(None" in code_rms
    ns_rms: dict[str, Any] = {"__builtins__": __builtins__, "torch": torch}
    exec(code_rms, ns_rms)
    with pytest.raises(TypeError) as exc_info:
        ns_rms["RMSModel"]()
    assert "'NoneType' object is not iterable" in str(exc_info.value)


# ==============================================================================
# TARGET 4: Checkpoint Loading & Key Mapping
# ==============================================================================


def test_load_weights_strict_true_vs_false_missing_keys() -> None:
    """Verify strict=True vs strict=False behavior on missing keys."""
    source = """
fc1 = linear(32, name='fc1')
fc2 = linear(16, name='fc2')
"""
    model = network(source, input_shape=("B", 64), output_shape=("B", 16), device="cpu")
    orig_fc2_w = model["fc2"].weight.clone()
    new_fc1_w = torch.randn(32, 64)

    # Missing keys with strict=True raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        load_weights(model, {"nodes.n_fc1.weight": new_fc1_w}, strict=True)
    assert "Missing key(s) in state_dict" in str(exc_info.value)

    # Missing keys with strict=False loads existing and preserves omitted
    load_weights(model, {"nodes.n_fc1.weight": new_fc1_w}, strict=False)
    assert torch.equal(model["fc1"].weight, new_fc1_w)
    assert torch.equal(model["fc2"].weight, orig_fc2_w)


def test_load_weights_strict_true_vs_false_unexpected_keys() -> None:
    """Verify strict=True vs strict=False behavior on unexpected keys."""
    source = "linear(32, name='fc1')\n"
    model = network(source, input_shape=("B", 16), output_shape=("B", 32), device="cpu")
    state = dict(model.state_dict())
    state["unexpected_param"] = torch.randn(10)

    # Unexpected key with strict=True raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        load_weights(model, state, strict=True)
    assert "Unexpected key(s) in state_dict" in str(exc_info.value)

    # Unexpected key with strict=False ignores the unexpected key
    load_weights(model, state, strict=False)


def test_load_weights_prefix_dict_and_callable_mapping() -> None:
    """Verify key mapping via prefix dicts and callable functions."""
    source = """
fc1 = linear(32, name='fc1')
fc2 = linear(16, name='fc2')
"""
    model = network(source, input_shape=("B", 64), output_shape=("B", 16), device="cpu")

    # 1. Prefix dict remapping
    ext_dict_prefix = {
        "external_model.fc1.weight": torch.randn(32, 64),
        "external_model.fc1.bias": torch.randn(32),
        "external_model.fc2.weight": torch.randn(16, 32),
        "external_model.fc2.bias": torch.randn(16),
    }
    load_weights(model, ext_dict_prefix, key_map={"external_model.": ""}, strict=True)
    assert torch.equal(model["fc1"].weight, ext_dict_prefix["external_model.fc1.weight"])
    assert torch.equal(model["fc2"].weight, ext_dict_prefix["external_model.fc2.weight"])

    # 2. Callable function remapping
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
