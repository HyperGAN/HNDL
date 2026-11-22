"""Regression and optimization sanity tests for v0.13 runtime-fast track."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import threading
import weakref

import pytest
import torch
from torch import nn
import torch._inductor as inductor
from torch.nn import functional as F
from torch.nn.modules import module as _torch_module

import hndl.deployment as deployment
from hndl import Registry, resolve
from hndl.deployment import DeploymentModel, _evict, _tensor_digest, export
from hndl.errors import HNDLError
from hndl.inference import InferenceCache
from hndl.operators._packing import (
    expert_pair_parameters,
    guard_expert_pair_input,
    prepare_packed,
)
from hndl.operators.joint_attention import JointAttention
from hndl.operators.latent_attention import LatentAttention
from hndl.operators.moe import MixtureOfExperts
from hndl.torch import build, network


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# 1. PreparedNetwork state-guard parity, hooks, and thread safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutation", ["bias_none", "del_weight", "direct_param", "direct_buffer", "direct_module"])
def test_prepared_generated_detects_hookless_state_mutations(mutation):
    model = network("linear(16); relu(); linear(4)", input_shape=("B", 8),
                    output_shape=("B", 4), device="cpu").eval()
    prepared = model.prepare(backend="generated")
    x = torch.randn(2, 8)
    with torch.no_grad():
        torch.testing.assert_close(prepared(x), model(x))
        # Second call hits fast path.
        torch.testing.assert_close(prepared(x), model(x))
        if mutation == "bias_none":
            model[0].bias = None
        elif mutation == "del_weight":
            del model[0].weight
        elif mutation == "direct_param":
            model[0]._parameters["extra"] = nn.Parameter(torch.zeros(1))
        elif mutation == "direct_buffer":
            model[0]._buffers["extra"] = torch.zeros(1)
        elif mutation == "direct_module":
            model[0]._modules["extra"] = nn.Identity()
        with pytest.raises(HNDLError) as exc:
            prepared(x)
        assert exc.value.code == "E_RUNTIME"


def test_prepared_generated_detects_mid_forward_state_mutation():
    registry = Registry.builtins()

    @registry.operator("leaky_reg", identity="test.v013.leaky_reg",
                       shape="x[B, D] -> out[B, D]", summary="Registers buffer on later call.")
    class LeakyReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls >= 2:
                self.register_buffer("leaked", torch.zeros(1, device=x.device))
            return x * 2

    model = network("leaky_reg()", input_shape=("B", 4), output_shape=("B", 4),
                    registry=registry, device="cpu").eval()
    prepared = model.prepare(backend="generated")
    x = torch.randn(2, 4)
    with torch.no_grad():
        torch.testing.assert_close(prepared(x), x * 2)
        # Second call hits the generated fast path; mid-forward register_buffer must raise E_RUNTIME.
        with pytest.raises(HNDLError) as exc:
            prepared(x)
        assert exc.value.code == "E_RUNTIME"


def test_prepared_generated_respects_post_warmup_local_and_global_hooks():
    model = network("linear(8); relu(); linear(4)", input_shape=("B", 4),
                    output_shape=("B", 4), device="cpu").eval()
    prepared = model.prepare(backend="generated")
    x = torch.randn(2, 4)
    with torch.no_grad():
        baseline = prepared(x).clone()
        # Warm up fast path
        torch.testing.assert_close(prepared(x), baseline)
        # Register local forward hook AFTER warmup
        seen = []
        h_local = model[0].register_forward_hook(lambda m, inp, out: (seen.append("local"), out * 2)[1])
        try:
            with_local = prepared(x)
            assert seen == ["local"]
            torch.testing.assert_close(with_local, model(x))
        finally:
            h_local.remove()
        # After removal, fast path returns to baseline
        torch.testing.assert_close(prepared(x), baseline)
        # Register global forward hook AFTER warmup
        g_seen = []
        h_global = _torch_module.register_module_forward_hook(
            lambda m, inp, out: g_seen.append(m) or None
        )
        try:
            prepared(x)
            assert model[0] in g_seen and model[2] in g_seen
        finally:
            h_global.remove()


def test_prepared_generated_validated_cache_is_thread_safe():
    model = network("linear(16); relu(); linear(4)", input_shape=("B", 8),
                    output_shape=("B", 4), device="cpu").eval()
    prepared = model.prepare(backend="generated")
    errors = []

    def worker(tid):
        try:
            with torch.no_grad():
                for step in range(40):
                    batch = 1 + (tid * 7 + step) % 80
                    x = torch.randn(batch, 8)
                    actual = prepared(x)
                    expected = model[2](F.relu(model[0](x)))
                    torch.testing.assert_close(actual, expected)
                    if step % 15 == 0:
                        prepared.clear_cache()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert all(not t.is_alive() for t in threads)
    assert errors == []


def test_prepared_fast_path_matches_sequential_exact():
    model = network("linear(128); relu(); linear(128); relu(); linear(10)",
                    input_shape=("B", 64), output_shape=("B", 10), device="cpu").eval()
    prep = model.prepare(backend="generated")
    seq = nn.Sequential(model[0], model[1], model[2], model[3], model[4]).eval()
    x = torch.randn(1, 64)
    with torch.no_grad():
        for _ in range(3):
            torch.testing.assert_close(prep(x), seq(x), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. Packing: stale .data reassignment, Joint/Latent pre-packing, expert pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutation", ["rebind_clone", "transpose_view"])
def test_trusted_packing_detects_stale_weight_data_without_rebind(mutation):
    model = network("swiglu(8, bias=True)", input_shape=("B", 8),
                    output_shape=("B", 8), device="cpu").eval()
    prepared = model.prepare(pack=True)
    layer = model[0]
    x = torch.randn(2, 8)
    with torch.no_grad():
        torch.testing.assert_close(prepared(x), layer(x))
        if mutation == "rebind_clone":
            layer.gate.weight.data = layer.gate.weight.detach().clone() + 0.5
        else:
            layer.gate.weight.data = layer.gate.weight.data.t()
        expected = layer.down(F.silu(layer.gate(x)) * layer.up(x))
        torch.testing.assert_close(prepared(x), expected)
        torch.testing.assert_close(layer(x), expected)


def test_prepare_packed_covers_joint_attention_and_latent_attention():
    ja = JointAttention(heads=2, head_dim=8, D_x=16, D_c=12, bias=True)
    prepare_packed(ja)
    layouts = ja._hndl_packed_layouts
    assert (("x_q_proj", "x_k_proj", "x_v_proj"), "weight") in layouts
    assert (("x_q_proj", "x_k_proj", "x_v_proj"), "bias") in layouts
    assert (("c_q_proj", "c_k_proj", "c_v_proj"), "weight") in layouts
    assert (("c_q_proj", "c_k_proj", "c_v_proj"), "bias") in layouts
    assert ja.x_q_proj.weight.is_contiguous() and ja.c_v_proj.weight.is_contiguous()

    la = LatentAttention(heads=2, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, D=16, bias=False)
    prepare_packed(la)
    assert (("k_up", "v_up"), "weight") in la._hndl_packed_layouts
    assert la.k_up.weight.is_contiguous() and la.v_up.weight.is_contiguous()

    # Absorbed LatentAttention must NOT pack (k_up, v_up)
    la_abs = LatentAttention(heads=2, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, D=16, bias=False)
    la_abs.absorb_weights()
    prepare_packed(la_abs)
    assert (("k_up", "v_up"), "weight") not in la_abs._hndl_packed_layouts


def test_expert_pair_parameters_contiguity_storage_and_gradcheck():
    moe = MixtureOfExperts(3, 4, 2, "silu", D=2, expert_kind="swiglu", bias=True).double()
    prepare_packed(moe)
    packed_w = expert_pair_parameters(moe, ("gate", "up"), "weight")
    packed_b = expert_pair_parameters(moe, ("gate", "up"), "bias")
    assert packed_w.shape == (3, 8, 2)
    assert packed_b.shape == (3, 8)
    for e, expert in enumerate(moe.experts):
        assert expert.gate.weight.is_contiguous()
        assert expert.up.weight.is_contiguous()
        assert expert.gate.bias.is_contiguous()
        assert expert.up.bias.is_contiguous()
        torch.testing.assert_close(packed_w[e, :4], expert.gate.weight)
        torch.testing.assert_close(packed_w[e, 4:], expert.up.weight)
        torch.testing.assert_close(packed_b[e, :4], expert.gate.bias)
        torch.testing.assert_close(packed_b[e, 4:], expert.up.bias)

    # In-place update reflects immediately in packed view
    with torch.no_grad():
        moe.experts[1].up.weight.add_(0.25)
    packed_w2 = expert_pair_parameters(moe, ("gate", "up"), "weight")
    torch.testing.assert_close(packed_w2[1, 4:], moe.experts[1].up.weight)

    # Stale .data reassignment without rebind is caught
    with torch.no_grad():
        old_val = moe.experts[0].gate.weight.detach().clone() + 1.5
        moe.experts[0].gate.weight.data = old_val
    packed_w3 = expert_pair_parameters(moe, ("gate", "up"), "weight")
    torch.testing.assert_close(packed_w3[0, :4], old_val)

    # Autograd first- and second-order gradients
    moe_g = MixtureOfExperts(2, 3, 1, "silu", D=2, expert_kind="swiglu", bias=True).double()
    prepare_packed(moe_g)
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    def fn(inp):
        w = expert_pair_parameters(moe_g, ("gate", "up"), "weight")
        b = expert_pair_parameters(moe_g, ("gate", "up"), "bias")
        guarded = guard_expert_pair_input(moe_g, ("gate", "up"), inp, w)
        return (torch.einsum("bd,ehd->beh", guarded, w) + b.unsqueeze(0)).sum()

    assert torch.autograd.gradcheck(fn, (x,))
    assert torch.autograd.gradgradcheck(fn, (x,))


# ---------------------------------------------------------------------------
# 3. Streaming _TOKENWISE additions & fast InferenceCache guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_expr", [
    "mish()",
    "softplus()",
    "clamp(-1.0, 1.0)",
])
def test_streaming_new_tokenwise_unary_ops_match_full_sequence(op_expr):
    model = network(f"attention(4, causal=True); {op_expr}; linear(16)",
                    input_shape=("B", 8, 16), output_shape=("B", 8, 16), device="cpu").eval()
    x = torch.randn(2, 8, 16)
    cache = InferenceCache(16)
    with torch.no_grad():
        expected = model(x)
        actual = torch.cat([model(chunk, cache=cache) for chunk in x.split((3, 1, 2, 2), dim=1)], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_streaming_mul_and_adaln_zero_match_full_sequence():
    source = (
        "c = linear(cond, 16)\n"
        "h, gate = adaln_zero(x, c)\n"
        "a = attention(h, 4, causal=True)\n"
        "gated = mul(a, gate)\n"
        "out = add(x, gated)\n"
    )
    model = network(source, input_shape={"x": ("B", 8, 16), "cond": ("B", 16)},
                    output_shape={"out": ("B", 8, 16)}, device="cpu").eval()
    with torch.no_grad():
        nn.init.normal_(model["n1"].modulation.weight, std=0.2)
        nn.init.normal_(model["n1"].modulation.bias, std=0.2)
    x = torch.randn(2, 8, 16)
    cond = torch.randn(2, 16)
    cache = InferenceCache(16)
    with torch.no_grad():
        expected = model(x=x, cond=cond)
        parts = [
            model(x=xc, cond=cond, cache=cache)
            for xc in x.split((3, 1, 4), dim=1)
        ]
        torch.testing.assert_close(torch.cat(parts, dim=1), expected, rtol=1e-4, atol=1e-5)


def test_inference_cache_does_not_hold_strong_model_reference():
    model = network("attention(4, causal=True)", input_shape=("B", 8, 16),
                    output_shape=("B", 8, 16), device="cpu").eval()
    ref = weakref.ref(model)
    cache = InferenceCache(16)
    with torch.no_grad():
        model(torch.randn(1, 2, 16), cache=cache)
        model(torch.randn(1, 1, 16), cache=cache)
    del model
    gc.collect()
    assert ref() is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_inference_cache_node_reuses_active_cuda_stream_across_layers(monkeypatch):
    model = network(
        "transformer_block(4, causal=True)\ntransformer_block(4, causal=True)\ntransformer_block(4, causal=True)",
        input_shape=("B", 8, 16), output_shape=("B", 8, 16), device="cuda:0",
    ).eval()
    cache = InferenceCache(16)
    tok = torch.randn(1, 1, 16, device="cuda:0")
    with torch.no_grad():
        model(tok, cache=cache)
        calls = []
        orig = torch.cuda.current_stream

        def counting_stream(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "current_stream", counting_stream)
        model(tok, cache=cache)
        # Only _begin_call queries current_stream once (not 1 + 3 layers = 4).
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 4. init_policy="depth_scaled" fixes
# ---------------------------------------------------------------------------


def test_depth_scaled_init_scales_transformer_blocks_without_top_level_add():
    source = "\n".join("transformer_block(4, causal=True)" for _ in range(4))
    plan = resolve(source, input_shape=("B", 8, 16), output_shape=("B", 8, 16))
    m_def = build(plan, init_policy="default", initialization_seed=11, device="cpu")
    m_scl = build(plan, init_policy="depth_scaled", initialization_seed=11, device="cpu")
    factor = 1.0 / math.sqrt(2.0 * 4)
    for i in range(4):
        torch.testing.assert_close(m_scl[i].attn.o_proj.weight, m_def[i].attn.o_proj.weight * factor)
        torch.testing.assert_close(m_scl[i].ffn.down.weight, m_def[i].ffn.down.weight * factor)
        # Non-exit projections stay unscaled
        torch.testing.assert_close(m_scl[i].attn.q_proj.weight, m_def[i].attn.q_proj.weight)
        torch.testing.assert_close(m_scl[i].ffn.up.weight, m_def[i].ffn.up.weight)


@pytest.mark.parametrize("kind,exit_attr", [("basic", "conv2"), ("bottleneck", "conv3")])
def test_depth_scaled_init_scales_resblocks_without_top_level_add(kind, exit_attr):
    source = f"resblock(16, kind='{kind}')\nresblock(16, kind='{kind}')\n"
    plan = resolve(source, input_shape=("B", 16, 8, 8), output_shape=("B", 16, 8, 8))
    m_def = build(plan, init_policy="default", initialization_seed=19, device="cpu")
    m_scl = build(plan, init_policy="depth_scaled", initialization_seed=19, device="cpu")
    factor = 1.0 / math.sqrt(2.0 * 2)
    for i in range(2):
        torch.testing.assert_close(
            getattr(m_scl[i], exit_attr).weight,
            getattr(m_def[i], exit_attr).weight * factor,
        )
        torch.testing.assert_close(m_scl[i].conv1.weight, m_def[i].conv1.weight)


def test_depth_scaled_init_scales_all_moe_experts():
    source = "h = moe(x, 4, 32, expert_kind='swiglu')\nout = add(x, h)\n"
    plan = resolve(source, input_shape={"x": ("B", 8, 16)}, output_shape={"out": ("B", 8, 16)})
    m_def = build(plan, init_policy="default", initialization_seed=23, device="cpu")
    m_scl = build(plan, init_policy="depth_scaled", initialization_seed=23, device="cpu")
    factor = 1.0 / math.sqrt(2.0 * 1)
    for e in range(4):
        torch.testing.assert_close(
            m_scl["n0"].experts[e].down.weight,
            m_def["n0"].experts[e].down.weight * factor,
        )
        torch.testing.assert_close(
            m_scl["n0"].experts[e].up.weight,
            m_def["n0"].experts[e].up.weight,
        )


def test_depth_scaled_init_deduplicates_tied_residual_exit_parameters():
    source = (
        "a = linear(x, 16)\n"
        "x1 = add(x, a)\n"
        "b = linear(x1, 16, tie={'weight': 'n0.weight'})\n"
        "out = add(x1, b)\n"
    )
    plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    m_def = build(plan, init_policy="default", initialization_seed=31, device="cpu")
    m_scl = build(plan, init_policy="depth_scaled", initialization_seed=31, device="cpu")
    assert m_scl["n0"].weight is m_scl["n2"].weight
    factor = 1.0 / math.sqrt(2.0 * 2)
    # Scaled ONCE (by factor), NOT twice (factor ** 2)
    torch.testing.assert_close(m_scl["n0"].weight, m_def["n0"].weight * factor)


# ---------------------------------------------------------------------------
# 5. Deployment: fast _tensor_digest, single-pass export guard, _evict, __call__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_tensor_digest_dtypes_views_and_sensitivity(dtype):
    if dtype is torch.bool:
        t = torch.tensor([[True, False], [False, True]], dtype=dtype)
        t_mod = t.clone()
        t_mod[0, 1] = True
    elif dtype.is_floating_point:
        t = torch.randn(4, 6, dtype=dtype)
        t_mod = t.clone()
        t_mod[2, 3] = t_mod[2, 3] + 1
    else:
        t = torch.arange(24, dtype=dtype).reshape(4, 6)
        t_mod = t.clone()
        t_mod[2, 3] += 1
    assert _tensor_digest(t) == _tensor_digest(t.clone())
    assert _tensor_digest(t) != _tensor_digest(t_mod)
    # Non-contiguous transpose view matches its contiguous equivalent
    assert _tensor_digest(t.t()) == _tensor_digest(t.t().contiguous())
    # Contiguous slice view ignores unrelated storage outside the slice
    parent = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    view_a = parent[1:3]
    parent_copy = parent.clone()
    parent_copy[0].fill_(999.0)
    parent_copy[3].fill_(-999.0)
    assert _tensor_digest(view_a) == _tensor_digest(parent_copy[1:3])
    # Scalar and empty tensors
    assert _tensor_digest(torch.tensor(3.5)) == _tensor_digest(torch.tensor(3.5))
    assert _tensor_digest(torch.empty(0, 4)) == _tensor_digest(torch.empty(0, 4))
    assert _tensor_digest(torch.empty(0, 4)) != _tensor_digest(torch.empty(0, 5))


def test_export_hashes_model_state_once_and_catches_mid_compile_mutation(tmp_path, monkeypatch):
    model = network("linear(8); relu(); linear(2)", input_shape=("B", 4),
                    output_shape=("B", 2), initialization_seed=5, device="cpu").eval()
    x = torch.randn(2, 4)
    state_calls = []
    orig_state = deployment._state

    def counting_state(m):
        state_calls.append(1)
        return orig_state(m)

    monkeypatch.setattr(deployment, "_state", counting_state)
    monkeypatch.setattr(
        inductor,
        "aoti_compile_and_package",
        lambda program, *, package_path, inductor_configs: Path(package_path).write_bytes(b"pkg"),
    )
    export(model, (x,), tmp_path)
    assert len(state_calls) == 1

    # Now mutate weights mid-compile and confirm E_DEPLOYMENT is raised
    def mutating_compile(program, *, package_path, inductor_configs):
        with torch.no_grad():
            model[0].weight.add_(0.5)
        Path(package_path).write_bytes(b"pkg")

    monkeypatch.setattr(inductor, "aoti_compile_and_package", mutating_compile)
    with pytest.raises(HNDLError) as exc:
        export(model, (x,), tmp_path, cache=False)
    assert exc.value.code == "E_DEPLOYMENT"


def test_evict_linear_removes_oldest_and_preserves_keep(tmp_path):
    keep = None
    for i in range(12):
        entry = tmp_path / f"{i:064x}"
        entry.mkdir()
        (entry / "model.pt2").write_bytes(b"x" * 100)
        manifest = entry / "model.json"
        manifest.write_text("{}")
        ts = 1_000_000 + i * 10
        import os
        os.utime(manifest, (ts, ts))
        if i == 0:
            keep = entry
    _evict(tmp_path, keep, max_entries=4, max_bytes=10_000)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert len(remaining) == 4
    assert keep.name in remaining
    # Newest 3 entries plus keep remain
    assert f"{11:064x}" in remaining
    assert f"{10:064x}" in remaining
    assert f"{9:064x}" in remaining


def test_deployment_model_fast_positional_and_error_codes():
    manifest = {
        "specification": {
            "inputs": [
                {"name": "x", "shape": [2, 4], "stride": [4, 1], "dtype": "torch.float32", "device": "cpu"},
                {"name": "y", "shape": [2, 4], "stride": [4, 1], "dtype": "torch.float32", "device": "cpu"},
            ]
        }
    }
    grad_states = []

    def runner(a, b):
        grad_states.append(torch.is_grad_enabled())
        return a + b

    dm = DeploymentModel(runner, manifest)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    # Positional with grad enabled -> runner runs under no_grad
    torch.testing.assert_close(dm(x, y), x + y)
    assert grad_states[-1] is False
    # Positional already under no_grad -> fast path
    with torch.no_grad():
        torch.testing.assert_close(dm(x, y), x + y)
        assert grad_states[-1] is False
    # Named call
    torch.testing.assert_close(dm(y=y, x=x), x + y)
    # Error codes
    with pytest.raises(HNDLError) as exc:
        dm(x)
    assert exc.value.code == "E_BINDING"
    with pytest.raises(HNDLError) as exc:
        dm(x, y, x=x)
    assert exc.value.code == "E_BINDING"
    with pytest.raises(HNDLError) as exc:
        dm(x[:1], y)
    assert exc.value.code == "E_RUNTIME"
    with pytest.raises(HNDLError) as exc:
        dm("not-a-tensor", y)
    assert exc.value.code == "E_RUNTIME"
    with pytest.raises(HNDLError) as exc:
        dm(x.clone().requires_grad_(), y)
    assert exc.value.code == "E_UNSUPPORTED"
