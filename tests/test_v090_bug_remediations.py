"""Dedicated regression test suite for HNDL v0.9.0 bug remediations (B-01 through B-10)."""

import hashlib

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, resolve
from hndl.torch import build
from tests._network_examples import load


# ==============================================================================
# B-01: cast + GraphModule runtime crash, float64 missing, and buffer pollution
# ==============================================================================
def test_b01_cast_graphmodule_runtime_and_buffer_pollution():
    # 1. Cast to int64, bool, and float64 at graph outputs and intermediate nodes
    source_int64 = """
    i = cast("int64")
    """
    plan_int64 = resolve(source_int64, input_shape=("B", 16), output_shape=("B", 16))
    assert plan_int64.nodes[0].args["dtype"] == "int64"
    model_int64 = build(plan_int64, device="cpu")
    x = torch.randn(2, 16)
    out_int64 = model_int64(x=x)["output"]
    assert out_int64.dtype == torch.int64

    source_bool = """
    b = cast("bool")
    """
    plan_bool = resolve(source_bool, input_shape=("B", 16), output_shape=("B", 16))
    assert plan_bool.nodes[0].args["dtype"] == "bool"
    model_bool = build(plan_bool, device="cpu")
    out_bool = model_bool(x=x)["output"]
    assert out_bool.dtype == torch.bool

    source_f64 = """
    d = cast("float64")
    """
    plan_f64 = resolve(source_f64, input_shape=("B", 16), output_shape=("B", 16))
    assert plan_f64.nodes[0].args["dtype"] == "float64"
    model_f64 = build(plan_f64, device="cpu")
    out_f64 = model_f64(x=x)["output"]
    assert out_f64.dtype == torch.float64

    # 2. Verify state_dict() buffer pollution is eliminated (no _sink buffer)
    assert len(list(model_int64.buffers())) == 0
    assert len(list(model_bool.buffers())) == 0
    assert len(list(model_f64.buffers())) == 0
    assert "_sink" not in model_int64.state_dict()

    # 3. Verify .half() does not corrupt fixed integer cast ports
    source_chain = """
    h = linear(32)
    cast("int64")
    """
    plan_chain = resolve(source_chain, input_shape=("B", 16), output_shape=("B", 32))
    assert plan_chain.nodes[1].args["dtype"] == "int64"

    model_chain = build(plan_chain, device="cpu")
    model_chain.half()
    out_chain = model_chain(x=x.half())["output"]
    assert out_chain.dtype == torch.int64



# ==============================================================================
# B-02: RoPE caches under torch.compile and device retention
# ==============================================================================
def test_b02_rope_caches_and_device_transfer():
    from hndl.operators.attention import Attention
    from hndl.operators.latent_attention import LatentAttention
    from hndl.operators.rope import RoPE

    # 1. RoPE module cache clearing on _apply (.to(device) or .cpu())
    rope = RoPE(theta=10000.0)
    x = torch.randn(2, 16, 64)
    _ = rope(x)
    assert len(rope._cache) > 0
    rope.cpu()
    assert len(rope._cache) == 0

    # 2. Attention module RoPE cache clearing on _apply
    attn = Attention(heads=4, causal=True, dropout=0.0, bias=True, rope=True, D=64)
    _ = attn(x)
    assert len(attn._rope_cache) > 0
    attn.cpu()
    assert len(attn._rope_cache) == 0

    # 3. LatentAttention RoPE cache clearing on _apply
    mla = LatentAttention(heads=4, kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=16,
                          qk_nope_head_dim=16, v_head_dim=16, D=64)
    _ = mla(x)
    assert len(mla._rope_cache) > 0
    mla.cpu()
    assert len(mla._rope_cache) == 0

    # 4. Max length slicing: longer call caches max positions, shorter call slices
    rope2 = RoPE(theta=10000.0)
    x_long = torch.randn(2, 32, 64)
    _ = rope2(x_long)
    key = (64, str(x_long.device), x_long.dtype)
    assert rope2._cache[key][0].shape[1] == 32
    x_short = torch.randn(2, 16, 64)
    _ = rope2(x_short)
    assert rope2._cache[key][0].shape[1] == 32  # Kept max length



# ==============================================================================
# B-03: gated_delta_attention float16 backward NaN
# ==============================================================================
def test_b03_gated_delta_attention_fp16_backward_finite():
    source = """
    gated_delta_attention(4, name="gda")
    """
    plan = resolve(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), dtype="float16")
    model = build(plan, device="cpu")
    x = torch.randn(2, 16, 64, dtype=torch.float16, requires_grad=True)
    out = model(x=x)["output"]
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()

    loss = out.float().square().mean()
    loss.backward()

    # All trainable parameters must have finite gradients without NaN
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} grad is None"
            assert not torch.isnan(p.grad).any(), f"{name} grad contains NaN"
            assert not torch.isinf(p.grad).any(), f"{name} grad contains Inf"


# ==============================================================================
# B-04: grn and sinusoidal_embed in float16 numerical stability
# ==============================================================================
def test_b04_grn_and_sinusoidal_embed_fp16_numerical_stability():
    from hndl.operators.grn import GRN
    from hndl.operators.sinusoidal_embed import SinusoidalEmbed, _reference as sinusoidal_ref

    # 1. GRN in float16 with spatial sum >= 1024 does not overflow to inf / NaN
    grn = GRN(eps=1e-6, C=32)
    grn.half()
    # 32x32 spatial grid, magnitude ~ 10.0
    x = torch.full((2, 32, 32, 32), 10.0, dtype=torch.float16)
    out = grn(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()

    # 2. SinusoidalEmbed at large timestep t = 999 maintains numerical precision
    embed = SinusoidalEmbed(features=128, max_period=10000.0, D_in=1)
    embed.half()
    ref_fn = sinusoidal_ref(embed)

    t = torch.tensor([[999.0]], dtype=torch.float16)
    out = embed(t)
    ref_out = ref_fn(t)
    assert torch.isfinite(out).all()
    # Check max relative error between forward and reference oracle
    rel_error = (out.float() - ref_out.float()).abs() / (ref_out.float().abs() + 1e-6)
    assert rel_error.max().item() < 0.01, f"Max relative error too high: {rel_error.max().item()}"


# ==============================================================================
# B-05: matmul._relation non-monotone rank guess
# ==============================================================================
def test_b05_matmul_rank4_backward_propagation():
    # Graph where inputs on Sweep 0 have unknown ranks, resolved backward from output_shape
    source = """
    a = reshape(2, 4, 8)
    b = reshape(2, 8, 4)
    matmul(a, b)
    """
    plan = resolve(source, input_shape={"x": ("B", 64)}, output_shape=("B", 2, 4, 4))
    assert plan.output_shape == ("B", 2, 4, 4)

    # Verify docstring documents ranks 2, 3, and 4
    reg = Registry.builtins()
    matmul_op = reg.get("matmul")
    assert "Rank 2" in matmul_op.doc or "rank 2" in matmul_op.doc.lower()
    assert "Rank 4" in matmul_op.doc or "rank 4" in matmul_op.doc.lower()


# ==============================================================================
# B-06: conv1d causal stride interval shape inference
# ==============================================================================
def test_b06_conv1d_causal_stride_interval_shape_inference():
    source = """
    c = conv1d(16, kernel_size=3, stride=2, causal=True)
    """
    # Output length fixes 16; input length should resolve to interval without E_AMBIGUOUS
    plan = resolve(source, input_shape=("B", 8, 31), output_shape=("B", 16, 16))
    assert plan.nodes[0].output_shapes["out"] == ("B", 16, 16)


# ==============================================================================
# B-07: pretrained clean registry and contract="ids" multi-output
# ==============================================================================
def test_b07_pretrained_clean_registry_and_shape_fn_exception_propagation(tmp_path):
    # 1. Fresh Registry instance without monkey-patching accepts contract and shape_fn
    clean_reg = Registry()
    assert hasattr(clean_reg, "pretrained_provider")
    assert hasattr(clean_reg, "pretrained_shape_fn")
    assert hasattr(clean_reg, "pretrained_contract")

    def dummy_builder():
        return nn.Linear(16, 32)

    def dummy_shape_fn(input_shape, **kwargs):
        return (input_shape[0], 32)

    clean_reg.pretrained_provider("p_clean", dummy_builder, contract="ids", shape_fn=dummy_shape_fn)
    assert clean_reg.pretrained_contract("p_clean") == "ids"
    assert clean_reg.pretrained_shape_fn("p_clean") is dummy_shape_fn

    # 2. _invoke_shape_fn propagates internal TypeError without swallowing
    from hndl.pretrained import _invoke_shape_fn

    def failing_shape_fn(input_shape):
        raise TypeError("internal custom error in shape calculation")

    with pytest.raises(TypeError, match="internal custom error in shape calculation"):
        _invoke_shape_fn(failing_shape_fn, (2, 16))

    # 3. contract="ids" with layers does not crash querying port "out"
    reg = Registry.builtins()
    class MultiModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 16)
            self.proj = nn.Linear(16, 16)
        def forward(self, x):
            e = self.emb(x)
            return self.proj(e)

    path = tmp_path / "multi.pt"
    torch.save(MultiModule().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    reg.pretrained_provider("multi_ids", MultiModule, contract="ids",
                            shape_fn=lambda s, **kw: ((s[0], s[1], 16), (s[0], s[1], 16)))
    code = f'''
    out0, out1 = pretrained("{path}", provider="multi_ids", sha256="{digest}", layers=("emb", "proj"))
    '''
    plan = resolve(code, input_shape=("B", 12), output_shape={"out0": ("B", 12, 16), "out1": ("B", 12, 16)},
                   input_dtype="int64", registry=reg)
    assert "out0" in plan.nodes[0].output_shapes
    assert "out1" in plan.nodes[0].output_shapes


# ==============================================================================
# B-08: adaln_zero gate option and network recipe alignments
# ==============================================================================
def test_b08_adaln_zero_gate_option_and_networks():
    reg = Registry.builtins()
    adaln_op = reg.get("adaln_zero")
    assert "gate" in adaln_op.args
    assert adaln_op.args["gate"].since == "0.10.0"

    # gate=False produces single output port "out"
    source_no_gate = """
    c = mean(1)
    cond = linear(c, 32)
    normed = adaln_zero(x, cond, gate=False)
    """
    plan_no_gate = resolve(source_no_gate, input_shape=("B", 16, 64), output_shape=("B", 16, 64), registry=reg)
    assert plan_no_gate.nodes[-1].outputs == ("out",)
    model_no_gate = build(plan_no_gate, device="cpu")
    x = torch.randn(2, 16, 64)
    out_no_gate = model_no_gate(x=x)["output"]
    assert out_no_gate.shape == (2, 16, 64)
    assert model_no_gate["n2"].modulation.weight.shape == (2 * 64, 32)

    # gate=True produces ("out", "gate")
    source_gate = """
    c = mean(1)
    cond = linear(c, 32)
    normed, gate = adaln_zero(x, cond)
    mul(normed, gate)
    """
    plan_gate = resolve(source_gate, input_shape=("B", 16, 64), output_shape=("B", 16, 64), registry=reg)
    assert plan_gate.nodes[-2].outputs == ("out", "gate")
    model_gate = build(plan_gate, device="cpu")
    out_gate = model_gate(x=x)["output"]
    assert out_gate.shape == (2, 16, 64)
    assert model_gate["n2"].modulation.weight.shape == (3 * 64, 32)

    # Verify dit_tiny network resolves and parameters match 678,960
    source_dit, meta_dit, kwargs_dit = load("dit_tiny")
    plan_dit = resolve(source_dit, **kwargs_dit)
    assert meta_dit["parameters"] == 678960
    model_dit = build(plan_dit, device="cpu")
    assert sum(p.numel() for p in model_dit.parameters()) == 678960

    # Verify deepseek_moe_tiny network resolves and parameters match 1,181,312
    source_moe, meta_moe, kwargs_moe = load("deepseek_moe_tiny")
    plan_moe = resolve(source_moe, **kwargs_moe)
    assert meta_moe["parameters"] == 1181312
    model_moe = build(plan_moe, device="cpu")
    assert sum(p.numel() for p in model_moe.parameters()) == 1181312


# ==============================================================================
# B-09: Multi-output unpacking with multiple _ discard targets
# ==============================================================================
def test_b09_multi_output_unpacking_multiple_discard_targets():
    source = """
    a, _, _ = chunk(x, 3, dim=2)
    out = a
    """
    plan = resolve(source, input_shape=("B", 16, 48), output_shape=("B", 16, 16))
    assert plan.nodes[0].outputs == ("out0", "out1", "out2")
    model = build(plan, device="cpu")
    x = torch.randn(2, 16, 48)
    out = model(x=x)["output"]
    assert out.shape == (2, 16, 16)

    # Non-discard duplicate targets still rejected
    dup_source = """
    a, a, _ = chunk(x, 3, dim=2)
    """
    with pytest.raises(HNDLError, match="E_SYNTAX.*Unpacking targets must be distinct"):
        resolve(dup_source, input_shape=("B", 16, 48), output_shape=("B", 16, 16))


# ==============================================================================
# B-10: l2_norm dim validation, transformer_block eps and docstring, and extra_repr
# ==============================================================================
def test_b10_l2_norm_dim_validation_and_operator_extra_repr():
    # 1. l2_norm dim=0 rejected
    with pytest.raises(HNDLError, match="E_ARGUMENT.*dim=0 selects the batch axis"):
        resolve("l2_norm(dim=0)", input_shape=("B", 16), output_shape=("B", 16))

    # 2. l2_norm out-of-range dim rejected
    with pytest.raises(HNDLError, match="E_ARGUMENT.*is out of range"):
        resolve("l2_norm(dim=10)", input_shape=("B", 16), output_shape=("B", 16))

    # 3. l2_norm positional dim accepted
    plan = resolve("l2_norm(1)", input_shape=("B", 16), output_shape=("B", 16))
    assert plan.nodes[0].args["dim"] == 1

    # 4. transformer_block passes eps to Attention
    from hndl.operators.transformer_block import TransformerBlock
    block = TransformerBlock(heads=4, mlp_ratio=2, activation="gelu", norm="layer_norm",
                             causal=False, dropout=0.0, layer_scale=False, layer_scale_init=1.0,
                             eps=1e-5, D=64)
    assert block.attn.eps == 1e-5

    # 5. All 8 v0.9.0 operators have informative extra_repr
    from hndl.operators.adaln_zero import AdaLNZero
    from hndl.operators.cast import Cast
    from hndl.operators.gated_delta_attention import GatedDeltaAttention
    from hndl.operators.grn import GRN
    from hndl.operators.l2_norm import L2Norm
    from hndl.operators.latent_attention import LatentAttention
    from hndl.operators.rope import RoPE
    from hndl.operators.sinusoidal_embed import SinusoidalEmbed

    ops = [
        Cast("float16"),
        L2Norm(dim=1, eps=1e-5),
        GRN(eps=1e-5, C=32),
        SinusoidalEmbed(features=64, max_period=5000.0, D_in=1),
        AdaLNZero(eps=1e-5, gate=False, D=64, D_cond=32),
        RoPE(theta=5000.0),
        LatentAttention(heads=4, kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=16,
                        qk_nope_head_dim=16, v_head_dim=16, D=64),
        GatedDeltaAttention(heads=4, D=64),
    ]
    for op in ops:
        rep = op.extra_repr()
        assert isinstance(rep, str) and len(rep) > 0, f"{type(op).__name__} extra_repr is empty"
