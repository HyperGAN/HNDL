"""Regression, cross-check, and performance-sanity tests for v0.13 Track 4:
static profiler (`profile.py`), plan diff/mermaid/serialization (`types.py`),
and resolver (`resolver.py`).
"""

from __future__ import annotations

import json
import pathlib
import time
from hashlib import sha256

import pytest

from hndl import HNDLError, ResolvedPlan, resolve
from hndl.__main__ import _load_plan
from hndl.profile import _compute_node_parameters
from hndl.registry import Registry
from hndl.resolver import validate_concrete_plan
from hndl.torch import build

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "networks"


# ==============================================================================
# 1. Static Profiler: Parameter Count Cross-Checks Against Built Torch Modules
# ==============================================================================


class TestProfilerParameterCounts:
    def test_all_builtin_operator_examples_match_torch_params(self):
        """Every built-in operator's documented examples match actual built parameter counts."""
        reg = Registry.builtins()
        checked = 0
        for op in reg.operators:
            if op.alias == "pretrained":
                continue
            for ex in op.examples:
                plan = resolve(
                    ex.source,
                    input_shape=ex.input_shape,
                    output_shape=ex.output_shape,
                    input_dtype=ex.input_dtype,
                )
                prof = plan.profile()
                model = build(plan, device="cpu")
                actual = sum(p.numel() for p in model.parameters())
                assert prof.total_parameters == actual, (
                    f"{op.alias} example {ex.source!r}: profile={prof.total_parameters}, actual={actual}"
                )
                checked += 1
        assert checked >= 200

    def test_all_example_networks_match_torch_params(self):
        """All 25 network recipes in examples/networks/ match built parameter counts."""
        recipes = sorted(EXAMPLES_DIR.glob("*.hndl"))
        assert len(recipes) >= 25
        for path in recipes:
            plan = _load_plan(path)
            prof = plan.profile()
            model = build(plan, device="cpu")
            actual = sum(p.numel() for p in model.parameters())
            assert prof.total_parameters == actual, (
                f"{path.name}: profile={prof.total_parameters}, actual={actual}"
            )

    @pytest.mark.parametrize(
        ("source", "input_shape", "output_shape", "extra"),
        [
            # resblock: norm, bias=norm=="none", bottleneck_ratio, small mid_ch clamp
            ("resblock(16)", ("B", 16, 8, 8), ("B", 16, 8, 8), {}),
            ("resblock(32, stride=2)", ("B", 16, 8, 8), ("B", 32, 4, 4), {}),
            ('resblock(16, norm="group_norm", groups=4)', ("B", 16, 8, 8), ("B", 16, 8, 8), {}),
            ('resblock(16, norm="none")', ("B", 16, 8, 8), ("B", 16, 8, 8), {}),
            ('resblock(32, norm="none")', ("B", 16, 8, 8), ("B", 32, 8, 8), {}),
            ('resblock(32, kind="bottleneck")', ("B", 16, 8, 8), ("B", 32, 8, 8), {}),
            ('resblock(32, kind="bottleneck", bottleneck_ratio=2)', ("B", 16, 8, 8), ("B", 32, 8, 8), {}),
            ('resblock(32, kind="bottleneck", bottleneck_ratio=2, norm="none")', ("B", 16, 8, 8), ("B", 32, 8, 8), {}),
            ('resblock(3, kind="bottleneck", bottleneck_ratio=4, norm="none")', ("B", 3, 8, 8), ("B", 3, 8, 8), {}),
            # cross_attention: qk_norm, kv_heads, head_dim, bias variants
            (
                "cross_attention(x, context, 4)",
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                ("B", 6, 16),
                {},
            ),
            (
                'cross_attention(x, context, 4, qk_norm="rms_norm")',
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                ("B", 6, 16),
                {},
            ),
            (
                'cross_attention(x, context, 4, qk_norm="l2_norm", kv_heads=2, head_dim=8, bias=False)',
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                ("B", 6, 16),
                {},
            ),
            (
                "cross_attention(x, context, 4, qkv_bias=False, out_bias=True)",
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                ("B", 6, 16),
                {},
            ),
            # joint_attention: qk_norm, head_dim, bias
            (
                "out, ctx_out = joint_attention(x, context, 4)",
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                {"out": ("B", 6, 16), "ctx_out": ("B", 10, 24)},
                {},
            ),
            (
                'out, ctx_out = joint_attention(x, context, 4, head_dim=8, qk_norm="rms_norm")',
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                {"out": ("B", 6, 16), "ctx_out": ("B", 10, 24)},
                {},
            ),
            (
                'out, ctx_out = joint_attention(x, context, 4, qk_norm="l2_norm", bias=False)',
                {"x": ("B", 6, 16), "context": ("B", 10, 24)},
                {"out": ("B", 6, 16), "ctx_out": ("B", 10, 24)},
                {},
            ),
            # dy_tanh: affine True & False
            ("dy_tanh()", ("B", 8, 16), ("B", 8, 16), {}),
            ("dy_tanh(affine=False)", ("B", 8, 16), ("B", 8, 16), {}),
            # snake: explicit channels & dim
            ("snake(16)", ("B", 16, 8), ("B", 16, 8), {}),
            ("snake(dim=16)", ("B", 8, 16), ("B", 8, 16), {}),
            # attention: qk_norm, rpb, gqa, bias variants, equalized
            ('attention(4, qk_norm="rms_norm", kv_heads=2, head_dim=8)', ("B", 8, 16), ("B", 8, 16), {}),
            ("attention(4, relative_position_bias=True, spatial_shape=(2, 4))", ("B", 8, 16), ("B", 8, 16), {}),
            ("attention(4, qkv_bias=False, out_bias=False, equalized=True)", ("B", 8, 16), ("B", 8, 16), {}),
            # transformer_block: all variants
            (
                'transformer_block(4, norm="rms_norm", ffn_kind="swiglu", bias=False, '
                'layer_scale=True, kv_heads=2, head_dim=8, qk_norm="rms_norm", hidden=48)',
                ("B", 8, 16),
                ("B", 8, 16),
                {},
            ),
            ("transformer_block(4, mlp_ratio=2, qkv_bias=False, out_bias=False)", ("B", 8, 16), ("B", 8, 16), {}),
            # latent_attention, gated_delta_attention, mamba2, moe, adaln_zero
            (
                "latent_attention(4, 8, q_lora_rank=6, qk_nope_head_dim=6, qk_rope_head_dim=4, v_head_dim=8, bias=True)",
                ("B", 8, 16),
                ("B", 8, 16),
                {},
            ),
            ('gated_delta_attention(4, decay_kind="head", bias=True)', ("B", 8, 16), ("B", 8, 16), {}),
            ("gated_delta_attention(4, head_dim=8, conv_size=0, gate_rank=6)", ("B", 8, 16), ("B", 8, 16), {}),
            (
                "mamba2(d_model=16, d_state=8, heads=4, head_dim=8, bias=True, conv_bias=False)",
                ("B", 8, 16),
                ("B", 8, 16),
                {},
            ),
            ('moe(4, 32, top_k=2, expert_kind="swiglu", bias=False)', ("B", 8, 16), ("B", 8, 16), {}),
            (
                "out, gate = adaln_zero(x, cond)",
                {"x": ("B", 8, 16), "cond": ("B", 12)},
                {"out": ("B", 8, 16), "gate": ("B", 8, 16)},
                {},
            ),
            (
                "adaln_zero(x, cond, gate=False)",
                {"x": ("B", 8, 16), "cond": ("B", 12)},
                ("B", 8, 16),
                {},
            ),
            # conv/deconv/norms/misc non-defaults
            ("conv(16, kernel_size=(3, 5), padding=(1, 2), groups=2, bias=False)", ("B", 8, 8, 8), ("B", 16, 8, 8), {}),
            ("conv1d(16, 3, padding=1, groups=2, bias=False, channels_last=True)", ("B", 10, 8), ("B", 10, 16), {}),
            ("deconv(16, kernel_size=(4, 2), stride=2, padding=(1, 0), groups=2, bias=False)", ("B", 8, 4, 4), ("B", 16, 8, 8), {}),
            ("deconv1d(16, 4, stride=2, padding=1, groups=2, bias=False, channels_last=True)", ("B", 4, 8), ("B", 8, 16), {}),
            ("batch_norm(affine=False)", ("B", 8, 4, 4), ("B", 8, 4, 4), {}),
            ("group_norm(4, affine=False)", ("B", 8, 4, 4), ("B", 8, 4, 4), {}),
            ("instance_norm(affine=True)", ("B", 8, 4, 4), ("B", 8, 4, 4), {}),
            ("layer_norm(bias=False)", ("B", 8, 16), ("B", 8, 16), {}),
            ("rms_norm(affine=False)", ("B", 8, 16), ("B", 8, 16), {}),
            ("rms_norm(zero_centered=True)", ("B", 8, 16), ("B", 8, 16), {}),
            ("linear(32, bias=False, spectral_norm=True)", ("B", 16), ("B", 32), {}),
            ("swiglu(48, bias=True)", ("B", 8, 16), ("B", 8, 16), {}),
            ("feed_forward(48, bias=False, equalized=True)", ("B", 8, 16), ("B", 8, 16), {}),
            ("patch_embed(32, 4)", ("B", 3, 16, 16), ("B", 16, 32), {}),
            ("hopfield(12, steps=3, normalize=False)", ("B", 8, 16), ("B", 8, 16), {}),
            ("spatial_attention(4)", ("B", 16, 4, 4), ("B", 16, 4, 4), {}),
            ("embedding(100, 16)", ("B", 8), ("B", 8, 16), {"input_dtype": "int64"}),
        ],
    )
    def test_parametric_variants_match_torch(self, source, input_shape, output_shape, extra):
        plan = resolve(source, input_shape=input_shape, output_shape=output_shape, **extra)
        prof = plan.profile()
        model = build(plan, device="cpu")
        actual = sum(p.numel() for p in model.parameters())
        assert prof.total_parameters == actual

    def test_snake_zero_channels_rank_inference(self):
        """When channels==0 and dim==0, snake uses axis 1 for rank 4 and axis -1 for rank 2/3."""
        assert _compute_node_parameters("snake", {"channels": 0, "dim": 0}, {"x": (2, 16, 4, 4)}, {"out": (2, 16, 4, 4)}) == (16, 16)
        assert _compute_node_parameters("snake", {"channels": 0, "dim": 0}, {"x": (2, 8, 32)}, {"out": (2, 8, 32)}) == (32, 32)
        assert _compute_node_parameters("snake", {"channels": 0, "dim": 0}, {"x": (2, 24)}, {"out": (2, 24)}) == (24, 24)
        assert _compute_node_parameters("snake", {"channels": 12, "dim": 0}, {"x": (2, 12, 8)}, {"out": (2, 12, 8)}) == (12, 12)


# ==============================================================================
# 2. Static Profiler: Analytical FLOP Formulas
# ==============================================================================


class TestProfilerFlops:
    def test_swiglu_and_feed_forward_flops(self):
        b, t, d, hidden = 2, 8, 32, 96
        tokens = b * t

        swiglu_plan = resolve(f"swiglu({hidden})", input_shape=("B", t, d), output_shape=("B", t, d))
        assert swiglu_plan.profile(batch_size=b).forward_flops == 3 * 2 * tokens * d * hidden

        ffn_plan = resolve(f"feed_forward({hidden})", input_shape=("B", t, d), output_shape=("B", t, d))
        assert ffn_plan.profile(batch_size=b).forward_flops == 2 * tokens * (d * hidden + hidden * d)

    def test_transformer_block_flops_match_components_and_respect_args(self):
        b, t, d = 2, 16, 64
        # Standard MHA + MLP with mlp_ratio=2
        tb_plan = resolve("transformer_block(4, mlp_ratio=2)", input_shape=("B", t, d), output_shape=("B", t, d))
        composed = resolve(
            "layer_norm()\nattention(4)\nlayer_norm()\nfeed_forward(128)",
            input_shape=("B", t, d),
            output_shape=("B", t, d),
        )
        assert tb_plan.profile(batch_size=b).forward_flops == composed.profile(batch_size=b).forward_flops

        # GQA + RMSNorm + SwiGLU + explicit hidden
        tb_gqa = resolve(
            'transformer_block(8, kv_heads=2, norm="rms_norm", ffn_kind="swiglu", hidden=192)',
            input_shape=("B", t, d),
            output_shape=("B", t, d),
        )
        composed_gqa = resolve(
            "rms_norm()\nattention(8, kv_heads=2)\nrms_norm()\nswiglu(192)",
            input_shape=("B", t, d),
            output_shape=("B", t, d),
        )
        assert tb_gqa.profile(batch_size=b).forward_flops == composed_gqa.profile(batch_size=b).forward_flops

        # Larger hidden / mlp_ratio must strictly increase FLOPs (regression for ignored mlp_ratio/hidden)
        tb_r4 = resolve("transformer_block(4, mlp_ratio=4)", input_shape=("B", t, d), output_shape=("B", t, d))
        assert tb_r4.profile(batch_size=b).forward_flops > tb_plan.profile(batch_size=b).forward_flops

    def test_deconv_and_deconv1d_use_input_spatial_extents(self):
        # 2D deconv stride=2: input 4x4 -> output 8x8; FLOPs scale with 4*4, NOT 8*8
        p2d = resolve(
            "deconv(16, kernel_size=4, stride=2, padding=1)",
            input_shape=("B", 8, 4, 4),
            output_shape=("B", 16, 8, 8),
        )
        assert p2d.profile(batch_size=2).forward_flops == 2 * 2 * 4 * 4 * 8 * 16 * 4 * 4

        # 1D deconv stride=2: input L=4 -> output L=8; FLOPs scale with 4, NOT 8
        p1d = resolve(
            "deconv1d(16, 4, stride=2, padding=1)",
            input_shape=("B", 8, 4),
            output_shape=("B", 16, 8),
        )
        assert p1d.profile(batch_size=2).forward_flops == 2 * 2 * 4 * 8 * 16 * 4

        # channels_last=True
        p1d_cl = resolve(
            "deconv1d(16, 4, stride=2, padding=1, channels_last=True)",
            input_shape=("B", 4, 8),
            output_shape=("B", 8, 16),
        )
        assert p1d_cl.profile(batch_size=2).forward_flops == p1d.profile(batch_size=2).forward_flops

    def test_resblock_and_patch_embed_flops(self):
        rb_basic = resolve("resblock(32, stride=2)", input_shape=("B", 16, 8, 8), output_shape=("B", 32, 4, 4))
        # conv1 (3x3, 16->32 on 4x4) + conv2 (3x3, 32->32 on 4x4) + shortcut (1x1, 16->32 on 4x4) + 3*BN + add
        conv_flops = 2 * 1 * 4 * 4 * (32 * 16 * 9 + 32 * 32 * 9 + 32 * 16)
        norm_flops = 3 * 8 * 1 * 32 * 4 * 4
        add_flops = 1 * 32 * 4 * 4
        assert rb_basic.profile(batch_size=1).forward_flops == conv_flops + norm_flops + add_flops

        rb_bn = resolve(
            'resblock(32, kind="bottleneck", bottleneck_ratio=2, norm="none")',
            input_shape=("B", 16, 8, 8),
            output_shape=("B", 32, 8, 8),
        )
        # mid=16: conv1 1x1(16->16) + conv2 3x3(16->16) + conv3 1x1(16->32) + shortcut 1x1(16->32) + add
        expected_bn = 2 * 1 * 8 * 8 * (16 * 16 + 16 * 16 * 9 + 32 * 16 + 32 * 16) + 1 * 32 * 8 * 8
        assert rb_bn.profile(batch_size=1).forward_flops == expected_bn

        pe = resolve("patch_embed(32, 4)", input_shape=("B", 3, 16, 16), output_shape=("B", 16, 32))
        assert pe.profile(batch_size=2).forward_flops == 2 * 2 * 16 * 32 * 3 * 4 * 4

    def test_sota_sequence_and_spatial_operator_flops(self):
        # joint_attention
        ja = resolve(
            "out, ctx_out = joint_attention(x, context, 4, head_dim=8)",
            input_shape={"x": ("B", 6, 16), "context": ("B", 10, 24)},
            output_shape={"out": ("B", 6, 16), "ctx_out": ("B", 10, 24)},
        )
        t_tot = 16
        expected_ja = (
            4 * 2 * 1 * 6 * 16 * 32
            + 4 * 2 * 1 * 10 * 24 * 32
            + 2 * 1 * 4 * t_tot * t_tot * 8
            + 3 * 1 * 4 * t_tot * t_tot
            + 2 * 1 * 4 * t_tot * t_tot * 8
        )
        assert ja.profile(batch_size=1).forward_flops == expected_ja

        # latent_attention
        mla_no_q = resolve("latent_attention(4, 16, qk_rope_head_dim=8)", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        mla_q = resolve(
            "latent_attention(4, 16, q_lora_rank=12, qk_rope_head_dim=8)",
            input_shape=("B", 8, 32),
            output_shape=("B", 8, 32),
        )
        assert mla_no_q.profile().forward_flops > 10 * 8 * 32
        assert mla_q.profile().forward_flops != mla_no_q.profile().forward_flops

        # gated_delta_attention: linear scaling with T
        gda8 = resolve("gated_delta_attention(4)", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        gda16 = resolve("gated_delta_attention(4)", input_shape=("B", 16, 32), output_shape=("B", 16, 32))
        assert gda16.profile().forward_flops == 2 * gda8.profile().forward_flops

        # mamba2: linear scaling with T
        m8 = resolve("mamba2(d_model=16, d_state=8, heads=4, head_dim=8)", input_shape=("B", 8, 16), output_shape=("B", 8, 16))
        m16 = resolve("mamba2(d_model=16, d_state=8, heads=4, head_dim=8)", input_shape=("B", 16, 16), output_shape=("B", 16, 16))
        assert m16.profile().forward_flops == 2 * m8.profile().forward_flops

        # adaln_zero: gated vs ungated
        ada_g = resolve(
            "out, gate = adaln_zero(x, cond)",
            input_shape={"x": ("B", 8, 32), "cond": ("B", 16)},
            output_shape={"out": ("B", 8, 32), "gate": ("B", 8, 32)},
        )
        ada_ng = resolve(
            "adaln_zero(x, cond, gate=False)",
            input_shape={"x": ("B", 8, 32), "cond": ("B", 16)},
            output_shape=("B", 8, 32),
        )
        assert ada_g.profile().forward_flops > ada_ng.profile().forward_flops

        # hopfield: scales with steps and patterns
        hop1 = resolve("hopfield(16, steps=1, normalize=False)", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        hop3 = resolve("hopfield(16, steps=3, normalize=False)", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        assert hop3.profile().forward_flops == 3 * hop1.profile().forward_flops

        # spatial_attention: scales with H*W
        sa4 = resolve("spatial_attention(4)", input_shape=("B", 16, 4, 4), output_shape=("B", 16, 4, 4))
        sa8 = resolve("spatial_attention(4)", input_shape=("B", 16, 8, 8), output_shape=("B", 16, 8, 8))
        assert sa8.profile().forward_flops > 4 * sa4.profile().forward_flops


# ==============================================================================
# 3. types.py: diff_plans(), to_mermaid(), and to_json()/from_json()
# ==============================================================================


class TestTypesFixes:
    def test_diff_plans_detects_inputs_initialization_and_trainability(self):
        # Input rewiring
        p_in1 = resolve(
            'a = linear(x, 16, name="a")\nb = linear(y, 16, name="b")\nout = add(a, b, name="out")',
            input_shape={"x": ("B", 16), "y": ("B", 16)},
            output_shape=("B", 16),
        )
        p_in2 = resolve(
            'a = linear(y, 16, name="a")\nb = linear(x, 16, name="b")\nout = add(a, b, name="out")',
            input_shape={"x": ("B", 16), "y": ("B", 16)},
            output_shape=("B", 16),
        )
        diff_in = p_in1.diff(p_in2)
        assert not diff_in.same_digest
        assert diff_in.modified_nodes["a"]["inputs"] == ({"x": "input:x"}, {"x": "input:y"})
        assert diff_in.modified_nodes["b"]["inputs"] == ({"x": "input:y"}, {"x": "input:x"})

        # Initialization change
        p_init1 = resolve('linear(16, name="fc", init={"weight": 0.0})', input_shape=("B", 16), output_shape=("B", 16))
        p_init2 = resolve('linear(16, name="fc", init={"weight": 1.0})', input_shape=("B", 16), output_shape=("B", 16))
        diff_init = p_init1.diff(p_init2)
        assert not diff_init.same_digest
        assert "initialization" in diff_init.modified_nodes["fc"]

        # Trainability change
        p_tr1 = resolve('linear(16, name="fc", trainable=True)', input_shape=("B", 16), output_shape=("B", 16))
        p_tr2 = resolve('linear(16, name="fc", trainable=False)', input_shape=("B", 16), output_shape=("B", 16))
        diff_tr = p_tr1.diff(p_tr2)
        assert not diff_tr.same_digest
        assert "trainability" in diff_tr.modified_nodes["fc"]

    def test_diff_plans_shape_deltas_multi_output_and_single_output(self):
        # Single-output keeps (out1, out2) shape tuple pair
        p1 = resolve('linear(16, name="fc")', input_shape=("B", 8), output_shape=("B", 16))
        p2 = resolve('linear(32, name="fc")', input_shape=("B", 8), output_shape=("B", 32))
        d_single = p1.diff(p2)
        assert d_single.shape_deltas["fc"] == (("B", 16), ("B", 32))

        # Multi-output where FIRST output stays identical and SECOND output changes
        m1 = resolve(
            'out, ctx_out = joint_attention(x, context, 4, name="ja")',
            input_shape={"x": ("B", 6, 16), "context": ("B", 10, 16)},
            output_shape={"out": ("B", 6, 16), "ctx_out": ("B", 10, 16)},
        )
        m2 = resolve(
            'out, ctx_out = joint_attention(x, context, 4, name="ja")',
            input_shape={"x": ("B", 6, 16), "context": ("B", 12, 24)},
            output_shape={"out": ("B", 6, 16), "ctx_out": ("B", 12, 24)},
        )
        d_multi = m1.diff(m2)
        assert "ja" in d_multi.shape_deltas
        assert d_multi.shape_deltas["ja"] == (
            (("B", 6, 16), ("B", 10, 16)),
            (("B", 6, 16), ("B", 12, 24)),
        )

    def test_to_mermaid_passthrough_input_edges(self):
        # Direct pass-through alongside a compute branch
        plan = resolve(
            'h = linear(x, 16, name="fc")\npass_out = x\n',
            input_shape={"x": ("B", 8)},
            output_shape={"h": ("B", 16), "pass_out": ("B", 8)},
        )
        mermaid = plan.to_mermaid()
        assert "    in_x --> out_pass_out" in mermaid
        assert "    n_fc --> out_h" in mermaid
        assert "n_input" not in mermaid

    def test_to_json_byte_identical_canonical_format(self):
        plans = [
            resolve("linear(16)\nrelu()\nlinear(8)", input_shape=("B", 4), output_shape=("B", 8)),
            resolve(
                'a = linear(x, 8, name="a")\nb = linear(y, 8, name="b", tie={"weight": "a.weight"})\nout = add(a, b)\n',
                input_shape={"x": ("B", 8), "y": ("B", 8)},
                output_shape={"out": ("B", 8), "a": ("B", 8)},
            ),
        ]
        for plan in plans:
            raw = plan.to_json()
            # Compare against two-pass reference canonical serialization
            data = plan._data()
            data["semantic_digest"] = plan.semantic_digest
            body = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            data["artifact_digest"] = sha256(body.encode("utf-8")).hexdigest()
            expected = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            assert raw == expected
            restored = ResolvedPlan.from_json(raw)
            assert restored.to_json() == raw
            assert restored.semantic_digest == plan.semantic_digest


# ==============================================================================
# 4. resolver.py: O(1) Tie Lookup & validate_concrete_plan Fast Path
# ==============================================================================


class TestResolverFixes:
    def test_many_tied_nodes_by_id_and_binding(self):
        lines = ['base = linear(x, 8, name="anchor")']
        prev = "base"
        for i in range(150):
            target = "base.weight" if i % 2 == 0 else "anchor.weight"
            lines.append(f'h{i} = linear({prev}, 8, name="l{i}", tie={{"weight": "{target}"}})')
            prev = f"h{i}"
        lines.append(f"out = {prev}")
        source = "\n".join(lines)

        t0 = time.perf_counter()
        plan = resolve(source, input_shape={"x": ("B", 8)}, output_shape={"out": ("B", 8)})
        elapsed = time.perf_counter() - t0
        assert len(plan.nodes) == 151
        for node in plan.nodes[1:]:
            assert node.parameter_ties == {"weight": "anchor.weight"}
        assert elapsed < 5.0

        # Missing tie target still raises E_BINDING
        with pytest.raises(HNDLError, match="not found"):
            resolve(
                'linear(8, tie={"weight": "nonexistent.weight"})',
                input_shape=("B", 8),
                output_shape=("B", 8),
            )

    def test_validate_concrete_plan_reuses_plan_when_registry_matches(self):
        plan = resolve("linear(16)\nrelu()\nlinear(8)", input_shape=("B", 4), output_shape=("B", 8))
        # Clear stamp to force full validation path
        object.__setattr__(plan, "_validated_stamp", None)
        validated = validate_concrete_plan(plan, registry=plan.registry)
        assert validated is plan
        assert getattr(validated, "_validated_stamp", None) is not None

        # Passing a different registry does not mutate original plan's registry
        other_reg = Registry.builtins()
        custom_reg = Registry()
        for op in other_reg.operators:
            custom_reg._add(op)
        validated_other = validate_concrete_plan(plan, registry=custom_reg)
        assert validated_other is not plan
        assert validated_other.registry is custom_reg
        assert plan.registry is not custom_reg

        # from_json returns a stamped plan with a populated builtins registry
        restored = ResolvedPlan.from_json(plan.to_json())
        assert isinstance(restored.registry, Registry)
        assert getattr(restored, "_validated_stamp", None) is not None
        assert validate_concrete_plan(restored) is restored
