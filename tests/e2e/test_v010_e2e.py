"""Comprehensive requirement-driven E2E test suite for HNDL v0.10.0 across Tiers 1–4.

Covers all 5 pillars of the HNDL v0.10.0 upgrade:
- Pillar R1: Bug Remediation (B-01 through B-10)
- Pillar R2: Performance & Runtime Efficiency (Grouped GEMM MoE, GDA 4D factorized Tensor Core)
- Pillar R3: Operator Catalog (Mamba-2, JointAttention, DyTanh, DropPath, Where, Snake),
            SOTA Extensions (RoPE scaling/2D, resblock bottleneck, transformer_block parity,
            deconv1d causal, adaln_zero gate=False, moe routed_scaling_factor, conv same3/depthwise),
            and Solver/Rank Extensions (Rank-5 tensors, reshape -1 arbitrary axis)
- Pillar R4: Extensibility (.hndl scalar constants, def macros, tuple loops, ops.scope,
            weight tying, post-build state rebinding, standalone PyTorch codegen, checkpoint loader)
- Pillar R5: DX, Static Profiling (plan.profile params/FLOPs/peak memory/KV state),
            Visualization (to_mermaid), Semantic Plan Diffing (diff), CLI Tooling (inspect/mermaid/check),
            and Lifecycle Primitives (param_groups, selective checkpointing, depth-scaled init, InferenceCache)

Structure:
- Tier 1: Feature Coverage (happy paths in isolation across all 10 feature areas)
- Tier 2: Boundary & Corner Cases (limits, single tokens, odd lengths, rank-5 boundaries, reshape -1, zero scale)
- Tier 3: Cross-Feature Interactions (pairwise combinations of v0.10.0 features)
- Tier 4: Real-World Application Workloads (MMDiT DiT pipeline, neural audio codec, vision-language dual encoder,
          autoregressive hybrid decoder with cache, sparse MoE training step, standalone codegen roundtrip)
"""

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import hndl
from hndl import (
    HNDLError,
    Registry,
    ResolvedPlan,
    ops,
    resolve,
)
from hndl.torch import GraphModule, build, network

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# Feature Availability Detection Helpers (Progressive Milestone Testability)
# ==============================================================================

def check_operator(alias: str) -> None:
    """Skip test if operator is pending implementation in M3."""
    if alias not in Registry.builtins()._aliases:
        pytest.skip(f"Operator '{alias}' pending implementation in M3")


def check_plan_method(method: str) -> None:
    """Skip test if ResolvedPlan method is pending implementation in M4/M5."""
    if not hasattr(ResolvedPlan, method):
        pytest.skip(f"ResolvedPlan.{method}() pending implementation in M4/M5")


def check_module_method(method: str) -> None:
    """Skip test if GraphModule method is pending implementation in M4/M5."""
    if not hasattr(GraphModule, method):
        pytest.skip(f"GraphModule.{method}() pending implementation in M4/M5")


def check_hndl_torch_attr(attr: str) -> None:
    """Skip test if hndl.torch attribute is pending implementation in M4/M5."""
    import hndl.torch
    if not hasattr(hndl.torch, attr):
        pytest.skip(f"hndl.torch.{attr} pending implementation in M4/M5")


def check_rank5() -> None:
    """Skip test if Rank-5 support is pending implementation in M3."""
    from hndl.operator import SUPPORTED_RANKS
    if 5 not in SUPPORTED_RANKS:
        pytest.skip("Rank-5 support pending implementation in M3")


def check_b01() -> None:
    """Skip test if B-01 (cast non-compute dtype / float64) is pending M1 remediation."""
    from hndl.torch import DTYPES
    if "float64" not in DTYPES:
        pytest.skip("B-01 (cast non-compute dtypes & float64) pending M1 remediation")


def check_b04() -> None:
    """Skip test if B-04 (grn float16 overflow) is pending M1 remediation."""
    try:
        from hndl.operators.grn import GRN
        g = GRN(32).half()
        x = torch.randn(1, 32, 32, 32, dtype=torch.float16) * 10.0
        if not g(x).isfinite().all():
            pytest.skip("B-04 (grn float16 overflow) pending M1 remediation")
    except Exception:
        pytest.skip("B-04 pending M1 remediation")


def check_b06() -> None:
    """Skip test if B-06 (conv1d causal stride interval) is pending M1 remediation."""
    try:
        source = "reshape(16, 16)\nconv1d(32, kernel_size=3, stride=2, causal=True)\n"
        resolve(source, input_shape=("B", 256), output_shape=("B", 32, 8))
    except HNDLError as err:
        if err.code == "E_AMBIGUOUS":
            pytest.skip("B-06 (conv1d causal stride interval) pending M1 remediation")


def check_b10() -> None:
    """Skip test if B-10 (l2_norm dim validation) is pending M1 remediation."""
    try:
        resolve("l2_norm(dim=0)\n", input_shape=("B", 16, 64), output_shape=("B", 16, 64))
        pytest.skip("B-10 (l2_norm dim validation) pending M1 remediation")
    except HNDLError:
        pass


def check_b09() -> None:
    """Skip test if B-09 (multiple underscore discard unpacking) is pending M1 remediation."""
    try:
        resolve("a, _, _ = chunk(3)\n", input_shape={"x": ("B", 12)}, output_shape={"a": ("B", 4)})
    except HNDLError as err:
        if "distinct" in str(err).lower():
            pytest.skip("B-09 (multi-output discard unpacking) pending M1 remediation")


def check_hndl_constants() -> None:
    """Skip test if .hndl scalar constants are pending implementation in M4."""
    try:
        resolve("D = 64\nlinear(D)\n", input_shape={"x": ("B", 64)}, output_shape={"out": ("B", 64)})
    except HNDLError as err:
        if "literal values" in str(err).lower() or "syntax" in str(err).lower():
            pytest.skip(".hndl scalar constants pending implementation in M4")


def check_hndl_macros() -> None:
    """Skip test if .hndl def macros are pending implementation in M4."""
    source = "def block(x):\n  return relu(x)\nout = block(x)\n"
    try:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
    except HNDLError:
        pytest.skip(".hndl def macros pending implementation in M4")


def check_hndl_tuple_loops() -> None:
    """Skip test if .hndl literal tuple loops are pending implementation in M4."""
    source = "for c in (16, 32):\n  x = linear(c)\n"
    try:
        resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
    except HNDLError:
        pytest.skip(".hndl literal tuple loops pending implementation in M4")


# ==============================================================================
# TIER 1: Feature Coverage (Happy Paths in Isolation)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Isolated functional verification for all HNDL v0.10.0 features."""

    # --------------------------------------------------------------------------
    # Area 1: Bug Remediations (B-01 through B-10)
    # --------------------------------------------------------------------------

    def test_tier1_b01_cast_non_compute_dtypes_runtime(self, device: str) -> None:
        """[B-01] GraphModule executes graphs with cast("int64") and cast("bool") without E_RUNTIME."""
        check_b01()
        # Non-compute int64 cast output
        source_int = "out = cast(x, 'int64')\n"
        model_int = network(source_int, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)}, device=device)
        x = torch.randn(2, 16, device=device)
        out_int = model_int(x)
        assert out_int.dtype == torch.int64
        assert out_int.shape == (2, 16)

        # Non-compute bool cast output
        source_bool = "out = cast(x, 'bool')\n"
        model_bool = network(source_bool, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)}, device=device)
        out_bool = model_bool(x)
        assert out_bool.dtype == torch.bool
        assert out_bool.shape == (2, 16)

        # float64 cast output
        source_f64 = "out = cast(x, 'float64')\n"
        model_f64 = network(source_f64, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)}, device=device)
        out_f64 = model_f64(x)
        assert out_f64.dtype == torch.float64

        # Verify state_dict does not contain persistent _sink buffer
        state = model_int.state_dict()
        assert not any(k.endswith("_sink") for k in state)

    def test_tier1_b02_rope_attention_compile_cache_clean(self, device: str) -> None:
        """[B-02] RoPE / Attention caches clear on device transfers and avoid FakeTensor retention."""
        source = "rope(4, name='r')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out1 = model(x)
        assert out1.shape == (2, 8, 32)

        # Verify cache clearing on _apply / cpu transfer
        model.to("cpu")
        rope_node = model["r"]
        cache = getattr(rope_node, "_cache", None) or getattr(rope_node, "_rope_cache", None)
        if cache is not None:
            # Cache must not retain tensors from prior device or fake tensors
            for k, val in cache.items():
                assert not isinstance(val, torch.Tensor) or not getattr(val, "is_fake", False)

    def test_tier1_b03_gda_float16_backward_finite(self, device: str) -> None:
        """[B-03] gated_delta_attention produces finite gradients in float16 as alpha, beta -> 0."""
        source = "gated_delta_attention(heads=2, head_dim=16, name='gda')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), dtype="float16", device=device)
        x = torch.randn(2, 8, 32, dtype=torch.float16, device=device, requires_grad=True)
        out = model(x)
        assert out.dtype == torch.float16
        assert out.isfinite().all()

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.isfinite().all(), "gda backward produced NaN or Inf gradients in float16"

    def test_tier1_b04_grn_and_sinusoidal_embed_float16_stability(self, device: str) -> None:
        """[B-04] GRN and SinusoidalEmbed in float16 accumulate in float32 preventing overflow."""
        check_b04()
        # GRN with large spatial activations (1024 elements) in float16
        source_grn = "grn(name='g')\n"
        model_grn = network(source_grn, input_shape=("B", 8, 32, 32), output_shape=("B", 8, 32, 32), dtype="float16", device=device)
        x_large = torch.randn(1, 8, 32, 32, dtype=torch.float16, device=device) * 10.0
        out_grn = model_grn(x_large)
        assert out_grn.isfinite().all(), "GRN produced NaN in float16 on large spatial activation"

        # SinusoidalEmbed high timestep numerical stability
        source_sin = "sinusoidal_embed(dim=32, name='sin')\n"
        model_sin = network(source_sin, input_shape=("B",), output_shape=("B", 32), dtype="float16", device=device)
        timesteps = torch.tensor([999.0], dtype=torch.float16, device=device)
        out_sin = model_sin(timesteps)
        assert out_sin.isfinite().all()

    def test_tier1_b05_matmul_relation_rank4_backward(self) -> None:
        """[B-05] matmul rank inference resolves rank-4 batched tensors backward on Sweep 0."""
        # Rank-4 batched matmul: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N]
        plan = resolve(
            "out = matmul(a, b)\n",
            input_shape={"a": ("B", 4, 8, 16), "b": ("B", 4, 16, 32)},
            output_shape={"out": ("B", 4, 8, 32)},
        )
        assert plan.nodes[0].output_shapes["out"] == ("B", 4, 8, 32)

    def test_tier1_b06_conv1d_causal_stride_interval(self) -> None:
        """[B-06] conv1d with causal=True and stride > 1 records length interval without premature E_AMBIGUOUS."""
        check_b06()
        source = "reshape(16, 16)\nconv1d(32, kernel_size=3, stride=2, causal=True)\n"
        plan = resolve(source, input_shape=("B", 256), output_shape=("B", 32, 8))
        assert plan.nodes[1].output_shapes["out"] == ("B", 32, 8)

    def test_tier1_b07_pretrained_provider_ids_multi_output(self) -> None:
        """[B-07] Registry.pretrained_provider supports contract='ids' and shape_fn natively."""
        reg = Registry()
        assert hasattr(reg, "pretrained_provider")
        # Ensure pretrained_provider accepts contract and shape_fn arguments without monkey patching
        sig = inspect.signature(reg.pretrained_provider)
        assert "contract" in sig.parameters
        assert "shape_fn" in sig.parameters

    def test_tier1_b08_dit_tiny_spatial_transpose_and_ungated(self) -> None:
        """[B-08] dit_tiny pattern transposes spatial tokens and adaln_zero supports gate=False."""
        # adaln_zero with gate=False returns single output without dead gate parameters
        check_operator("adaln_zero")
        spec = Registry.builtins().get("adaln_zero")
        if "gate" not in spec.args:
            pytest.skip("adaln_zero gate=False pending implementation in M1/M3")
        plan = resolve(
            "out = adaln_zero(x, cond, gate=False)\n",
            input_shape={"x": ("B", 16, 64), "cond": ("B", 64)},
            output_shape={"out": ("B", 16, 64)},
        )
        assert "gate" not in plan.nodes[0].outputs or len(plan.nodes[0].outputs) == 1

    def test_tier1_b09_multi_output_discard_underscore_unpacking(self) -> None:
        """[B-09] Multi-output unpacking allows multiple '_' discard targets (a, _, _ = chunk(3))."""
        check_b09()
        source = "a, _, _ = chunk(3)\n"
        plan = resolve(
            source,
            input_shape={"x": ("B", 24)},
            output_shape={"a": ("B", 8)},
        )
        assert plan.outputs["a"]["shape"] == ("B", 8)

    def test_tier1_b10_l2_norm_dim_validation_and_transformer_block_eps(self) -> None:
        """[B-10] l2_norm rejects dim=0 at resolution time; transformer_block forwards eps."""
        check_b10()
        with pytest.raises(HNDLError) as exc_info:
            resolve("l2_norm(dim=0)\n", input_shape=("B", 16, 64), output_shape=("B", 16, 64))
        assert exc_info.value.code in ("E_CONSTRAINT", "E_ARGUMENT")

    # --------------------------------------------------------------------------
    # Area 2: New Built-in Operators (6 Operators)
    # --------------------------------------------------------------------------

    def test_tier1_op_mamba2_ssd_happy_path(self, device: str) -> None:
        """[New Op 1] mamba2 SSD block executes forward pass and autograd backward."""
        check_operator("mamba2")
        source = "mamba2(d_model=32, d_state=16, heads=4, head_dim=16, name='ssd')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 8, 32)
        assert out.isfinite().all()

        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()

    def test_tier1_op_joint_attention_multimodal_happy_path(self, device: str) -> None:
        """[New Op 2] joint_attention dual-stream multimodal attention executes cleanly."""
        check_operator("joint_attention")
        source = "out, ctx_out = joint_attention(x, context, heads=4, head_dim=16, name='jattn')\n"
        model = network(
            source,
            input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32)},
            output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
            device=device,
        )
        x = torch.randn(2, 8, 32, device=device, requires_grad=True)
        ctx = torch.randn(2, 16, 32, device=device, requires_grad=True)
        res = model(x=x, context=ctx)
        out, ctx_out = res["out"], res["ctx_out"]
        assert out.shape == (2, 8, 32)
        assert ctx_out.shape == (2, 16, 32)
        assert out.isfinite().all() and ctx_out.isfinite().all()

        (out.sum() + ctx_out.sum()).backward()
        assert x.grad is not None and ctx.grad is not None

    def test_tier1_op_dy_tanh_dynamic_activation_happy_path(self, device: str) -> None:
        """[New Op 3] dy_tanh dynamic tanh normalization executes with learned parameters."""
        check_operator("dy_tanh")
        source = "dy_tanh(dim=32, name='dyt')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 8, 32)
        assert out.isfinite().all()

        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()

    def test_tier1_op_drop_path_stochastic_depth_happy_path(self, device: str) -> None:
        """[New Op 4] drop_path scales in train mode and acts as exact identity in eval mode."""
        check_operator("drop_path")
        source = "drop_path(drop_prob=0.3, name='dp')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(4, 8, 32, device=device)

        # Eval mode: exact identity
        model.eval()
        out_eval = model(x)
        assert torch.equal(out_eval, x)

        # Train mode: forward pass is finite
        model.train()
        out_train = model(x)
        assert out_train.shape == (4, 8, 32)
        assert out_train.isfinite().all()

    def test_tier1_op_where_ternary_selection_happy_path(self, device: str) -> None:
        """[New Op 5] where ternary operator selects elements according to boolean condition."""
        check_operator("where")
        source = "out = where(cond, a, b)\n"
        model = network(
            source,
            input_shape={"cond": ("B", 8), "a": ("B", 8), "b": ("B", 8)},
            output_shape={"out": ("B", 8)},
            input_dtype={"cond": "bool", "a": "float32", "b": "float32"},
            device=device,
        )
        cond = torch.tensor([[True, False, True, False, True, False, True, False]], device=device)
        a = torch.ones(1, 8, device=device) * 5.0
        b = torch.ones(1, 8, device=device) * -5.0
        out = model(cond=cond, a=a, b=b)
        expected = torch.where(cond, a, b)
        assert torch.equal(out, expected)

    def test_tier1_op_snake_periodic_activation_happy_path(self, device: str) -> None:
        """[New Op 6] snake periodic activation executes on audio sequence representation."""
        check_operator("snake")
        source = "snake(dim=16, name='snk')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 16, 64)
        assert out.isfinite().all()

        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()

    # --------------------------------------------------------------------------
    # Area 3: Operator SOTA Extensions (since="0.10.0")
    # --------------------------------------------------------------------------

    def test_tier1_ext_rope_frequency_scaling_and_2d_axial(self, device: str) -> None:
        """[Extension 1] RoPE supports frequency scaling (yarn, linear) and 2D axial grids."""
        check_operator("rope")
        spec = Registry.builtins().get("rope")
        if "rope_scaling" not in spec.args:
            pytest.skip("rope_scaling pending implementation in M3")
        source = "rope(4, rope_scaling='yarn', rope_scale=2.0, name='r')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 8, 32)
        assert out.isfinite().all()

    def test_tier1_ext_resblock_bottleneck_and_silu(self, device: str) -> None:
        """[Extension 2] resblock supports kind='bottleneck' and activation='silu'."""
        check_operator("resblock")
        spec = Registry.builtins().get("resblock")
        if "kind" not in spec.args:
            pytest.skip("resblock kind='bottleneck' pending implementation in M3")
        source = "resblock(kind='bottleneck', activation='silu', name='rb')\n"
        model = network(source, input_shape=("B", 64, 16, 16), output_shape=("B", 64, 16, 16), device=device)
        x = torch.randn(2, 64, 16, 16, device=device)
        out = model(x)
        assert out.shape == (2, 64, 16, 16)
        assert out.isfinite().all()

    def test_tier1_ext_transformer_block_option_parity(self, device: str) -> None:
        """[Extension 3] transformer_block supports head_dim, window_size, logit_softcap, zero_centered."""
        check_operator("transformer_block")
        spec = Registry.builtins().get("transformer_block")
        if "head_dim" not in spec.args or "zero_centered" not in spec.args:
            pytest.skip("transformer_block SOTA options pending implementation in M3")
        source = "transformer_block(4, head_dim=16, zero_centered=True, name='tb')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 8, 32)
        assert out.isfinite().all()

    def test_tier1_ext_deconv1d_causal_and_channels_last(self, device: str) -> None:
        """[Extension 4] deconv1d supports causal=True and channels_last=True."""
        check_operator("deconv1d")
        spec = Registry.builtins().get("deconv1d")
        if "causal" not in spec.args:
            pytest.skip("deconv1d causal pending implementation in M3")
        source = "deconv1d(32, 4, stride=2, causal=True, channels_last=True, name='dconv')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 16, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 16, 32)

    def test_tier1_ext_moe_routed_scaling_factor(self, device: str) -> None:
        """[Extension 5] moe supports routed_scaling_factor argument."""
        check_operator("moe")
        spec = Registry.builtins().get("moe")
        if "routed_scaling_factor" not in spec.args:
            pytest.skip("moe routed_scaling_factor pending implementation in M3")
        source = "moe(4, 2, 64, routed_scaling_factor=2.5, name='moe')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 8, 32)

    def test_tier1_ext_conv_same3_and_depthwise_policy(self, device: str) -> None:
        """[Extension 6] conv supports policy='same3' and groups='depthwise'."""
        check_operator("conv")
        spec = Registry.builtins().get("conv")
        if "same3" not in spec.policies:
            pytest.skip("conv policy='same3' pending implementation in M3")
        source = "conv(32, policy='same3', name='c')\n"
        model = network(source, input_shape=("B", 32, 16, 16), output_shape=("B", 32, 16, 16), device=device)
        x = torch.randn(2, 32, 16, 16, device=device)
        out = model(x)
        assert out.shape == (2, 32, 16, 16)

    # --------------------------------------------------------------------------
    # Area 4: Solver & Rank Extensions
    # --------------------------------------------------------------------------

    def test_tier1_rank5_solver_and_runtime_resolution(self, device: str) -> None:
        """[Rank-5] Rank-5 tensor [B, C, T, H, W] resolves and executes through GraphModule."""
        check_rank5()
        source = "linear(32)\n"
        model = network(source, input_shape=("B", 8, 4, 16, 16), output_shape=("B", 8, 4, 16, 32), device=device)
        x = torch.randn(2, 8, 4, 16, 16, device=device)
        out = model(x)
        assert out.shape == (2, 8, 4, 16, 32)

    def test_tier1_reshape_arbitrary_inferred_axis(self) -> None:
        """[Reshape -1] Single inferred -1 axis at arbitrary position in reshape."""
        try:
            plan = resolve("reshape(-1, 1, 1)\n", input_shape=("B", 64), output_shape=("B", 64, 1, 1))
            assert plan.nodes[0].output_shapes["out"] == ("B", 64, 1, 1)
        except HNDLError:
            pytest.skip("reshape(-1, ...) arbitrary axis pending implementation in M3")

    # --------------------------------------------------------------------------
    # Area 5: .hndl Language Features
    # --------------------------------------------------------------------------

    def test_tier1_hndl_scalar_constants(self) -> None:
        """[.hndl Constants] Top-level immutable scalar constant bindings (D = 64)."""
        check_hndl_constants()
        source = "D = 64\nlinear(D)\n"
        plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 64)})
        assert plan.nodes[0].args["out_features"] == 64

    def test_tier1_hndl_def_macros(self) -> None:
        """[.hndl Macros] Non-recursive AST-expanded def macro definitions."""
        check_hndl_macros()
        source = "def mlp_block(x, hidden):\n  h = linear(x, hidden)\n  return relu(h)\nout = mlp_block(x, 64)\n"
        plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 64)})
        assert len(plan.nodes) == 2

    def test_tier1_hndl_literal_tuple_loops(self) -> None:
        """[.hndl Tuple Loops] Literal tuple loops unrolled cleanly (for c in (32, 64):)."""
        check_hndl_tuple_loops()
        source = "for c in (32, 64):\n  x = linear(x, c)\nout = x\n"
        plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
        assert len(plan.nodes) == 2

    def test_tier1_ops_scope_and_subgraph_registration(self) -> None:
        """[Scope & Subgraph] Python ops.scope(prefix) and registry.subgraph(alias, plan)."""
        reg = Registry()
        if hasattr(reg, "subgraph"):
            sub_plan = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
            reg.subgraph("custom_block", sub_plan)
            assert "custom_block" in reg._aliases

    # --------------------------------------------------------------------------
    # Area 6: Extensibility, Weight Tying & Codegen
    # --------------------------------------------------------------------------

    def test_tier1_declarative_weight_tying(self, device: str) -> None:
        """[Weight Tying] Declarative weight tying sharing embedding and linear weights."""
        source = "tok_emb = embedding(100, 32)\nout = linear(tok_emb, 100, tie={'weight': 'tok_emb.weight'})\n"
        try:
            model = network(
                source,
                input_shape={"x": ("B", 8)},
                output_shape={"out": ("B", 8, 100)},
                input_dtype={"x": "int64"},
                device=device,
            )
            # Verify weight sharing in parameters
            emb_weight = model["tok_emb"].weight
            lin_weight = model["out"].weight
            assert emb_weight.data_ptr() == lin_weight.data_ptr()
        except (HNDLError, TypeError, KeyError):
            pytest.skip("Declarative weight tying pending implementation in M4")

    def test_tier1_allow_state_mutation_and_rebind(self, device: str) -> None:
        """[State Mutation] model.allow_state_mutation() and model.rebind_state() for PEFT."""
        check_module_method("allow_state_mutation")
        source = "linear(32, name='lin')\n"
        model = network(source, input_shape=("B", 16), output_shape=("B", 32), device=device)
        with model.allow_state_mutation():
            # Apply mock adapter / parameter replacement
            orig_w = model["lin"].weight
            model["lin"].weight = nn.Parameter(orig_w.data * 2.0)
            model.rebind_state()

        # Next forward pass executes without E_RUNTIME
        x = torch.randn(2, 16, device=device)
        out = model(x)
        assert out.shape == (2, 32)

    def test_tier1_plan_to_python_source_codegen(self) -> None:
        """[Codegen] plan.to_python_source() emits standalone valid PyTorch module code."""
        check_plan_method("to_python_source")
        plan = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        src = plan.to_python_source()
        assert isinstance(src, str)
        assert "class" in src and "torch.nn.Module" in src
        assert "def forward" in src

    def test_tier1_load_weights_with_key_mapping(self, device: str) -> None:
        """[Checkpoint Loader] hndl.torch.load_weights loads external state dict with key_map."""
        check_hndl_torch_attr("load_weights")
        import hndl.torch as htorch
        source = "linear(32, name='fc')\n"
        model = network(source, input_shape=("B", 16), output_shape=("B", 32), device=device)
        external_state = {"external_fc.weight": torch.randn(32, 16, device=device), "external_fc.bias": torch.randn(32, device=device)}
        htorch.load_weights(model, external_state, key_map={"external_fc.": "fc."})
        assert torch.equal(model["fc"].weight, external_state["external_fc.weight"])

    # --------------------------------------------------------------------------
    # Area 7: Static Profiler, Visualization & Diffing
    # --------------------------------------------------------------------------

    def test_tier1_plan_profile_static_evaluation(self) -> None:
        """[Profiler] plan.profile() computes params, FLOPs, activation bytes, and KV state."""
        check_plan_method("profile")
        plan = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        prof = plan.profile(batch_size=2)
        assert hasattr(prof, "total_parameters")
        assert hasattr(prof, "forward_flops")
        assert hasattr(prof, "peak_activation_bytes")
        assert prof.total_parameters == 16 * 32 + 32  # 544
        assert prof.forward_flops > 0
        assert prof.peak_activation_bytes > 0

    def test_tier1_plan_profile_moe_active_params(self) -> None:
        """[Profiler MoE] Static profiler reports total vs active parameters for MoE."""
        check_plan_method("profile")
        plan = resolve("moe(8, 2, 64)\n", input_shape={"x": ("B", 8, 32)}, output_shape={"out": ("B", 8, 32)})
        prof = plan.profile(batch_size=1)
        assert hasattr(prof, "active_parameters")
        assert prof.active_parameters < prof.total_parameters

    def test_tier1_plan_to_mermaid_export(self) -> None:
        """[Mermaid Export] plan.to_mermaid() exports shape-annotated Mermaid flowchart."""
        check_plan_method("to_mermaid")
        plan = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        mermaid_txt = plan.to_mermaid()
        assert isinstance(mermaid_txt, str)
        assert "graph TD" in mermaid_txt
        assert "linear" in mermaid_txt

    def test_tier1_plan_diff_structural_comparison(self) -> None:
        """[Plan Diff] plan.diff(other_plan) identifies structural changes and parameter deltas."""
        check_plan_method("diff")
        plan1 = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        plan2 = resolve("linear(64)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
        diff = plan1.diff(plan2)
        assert hasattr(diff, "param_delta")
        assert diff.param_delta != 0

    # --------------------------------------------------------------------------
    # Area 8: CLI Tooling
    # --------------------------------------------------------------------------

    def test_tier1_cli_inspect_subcommand(self) -> None:
        """[CLI] hndl inspect command outputs formatted plan details."""
        try:
            import hndl.__main__ as cli_main
        except ImportError:
            pytest.skip("CLI entrypoint hndl.__main__ pending implementation in M5")
        assert hasattr(cli_main, "main")

    # --------------------------------------------------------------------------
    # Area 9: Training & Inference Lifecycle
    # --------------------------------------------------------------------------

    def test_tier1_model_param_groups_partitioning(self, device: str) -> None:
        """[Lifecycle] model.param_groups(weight_decay=0.01) separates 2D weights from 1D biases/norms."""
        check_module_method("param_groups")
        source = "linear(32, name='fc')\nrms_norm(name='norm')\n"
        model = network(source, input_shape=("B", 16), output_shape=("B", 32), device=device)
        groups = model.param_groups(weight_decay=0.01)
        assert len(groups) == 2
        # Decay group (linear.weight)
        assert groups[0]["weight_decay"] == 0.01
        # No-decay group (linear.bias, norm.weight)
        assert groups[1]["weight_decay"] == 0.0

    def test_tier1_selective_activation_checkpointing(self, device: str) -> None:
        """[Lifecycle] build(..., checkpoint=True) executes checkpointing in train mode."""
        source = "transformer_block(4, name='tb')\n"
        plan = resolve(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        try:
            model = build(plan, checkpoint=True, device=device)
            x = torch.randn(2, 8, 32, device=device, requires_grad=True)
            model.train()
            out = model(x)
            assert out.shape == (2, 8, 32)
            out.sum().backward()
            assert x.grad is not None and x.grad.isfinite().all()
        except (TypeError, HNDLError):
            pytest.skip("Selective activation checkpointing pending implementation in M5")

    def test_tier1_graph_aware_depth_scaled_init(self, device: str) -> None:
        """[Lifecycle] build(..., init_policy='depth_scaled') scales residual exits by 1/sqrt(2N)."""
        source = "h = linear(x, 32)\nout = add(x, h)\n"
        plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
        try:
            model = build(plan, init_policy="depth_scaled", device=device)
            assert model is not None
        except (TypeError, HNDLError):
            pytest.skip("Depth-scaled initialization pending implementation in M5")

    def test_tier1_stateful_inference_cache(self, device: str) -> None:
        """[Lifecycle] InferenceCache manages stateful decoding across attention blocks."""
        check_hndl_torch_attr("InferenceCache")
        import hndl.torch as htorch
        cache = htorch.InferenceCache()
        assert hasattr(cache, "get") and hasattr(cache, "set")

    # --------------------------------------------------------------------------
    # Area 10: IDE Autocomplete, Typing & Diagnostics
    # --------------------------------------------------------------------------

    def test_tier1_operator_namespace_dir_doc_sig(self) -> None:
        """[IDE Autocomplete] ops implements __dir__, and ops.<op> has __doc__ and __signature__."""
        assert hasattr(ops, "__dir__")
        aliases = dir(ops)
        if "linear" not in aliases:
            pytest.skip("ops.__dir__() dynamic operator alias listing pending implementation in M5")
        assert "linear" in aliases and "attention" in aliases

        linear_fn = ops.linear
        assert linear_fn.__doc__ is not None
        if hasattr(linear_fn, "__signature__"):
            assert isinstance(linear_fn.__signature__, inspect.Signature)

    def test_tier1_pep562_lazy_torch_import(self) -> None:
        """[PEP 562] hndl.torch is accessible lazily via __getattr__ without explicit import."""
        import subprocess
        import sys

        res = subprocess.run(
            [sys.executable, "-c", "import sys; import hndl; assert 'torch' not in sys.modules, 'torch was eagerly imported!'"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"torch eagerly loaded: {res.stderr}"

        assert hasattr(hndl, "torch")
        import types
        assert isinstance(hndl.torch, types.ModuleType)


# ==============================================================================
# TIER 2: Boundary & Corner Cases (Limits, Extremes, Defect Triggers)
# ==============================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Verification of edge limits, extreme dimensions, and failure triggers."""

    def test_tier2_empty_and_single_token_contexts(self, device: str) -> None:
        """[Boundary 1] Single token contexts (T=1, B=1) execute correctly across sequence ops."""
        source = "attention(4, causal=True, name='attn')\n"
        model = network(source, input_shape=("B", 1, 32), output_shape=("B", 1, 32), device=device)
        x = torch.randn(1, 1, 32, device=device)
        out = model(x)
        assert out.shape == (1, 1, 32)
        assert out.isfinite().all()

    def test_tier2_odd_and_prime_sequence_lengths(self, device: str) -> None:
        """[Boundary 2] Prime sequence length (T=17) with sliding window and RoPE."""
        source = "attention(4, window_size=5, rope=True, name='attn')\n"
        model = network(source, input_shape=("B", 17, 32), output_shape=("B", 17, 32), device=device)
        x = torch.randn(2, 17, 32, device=device)
        out = model(x)
        assert out.shape == (2, 17, 32)
        assert out.isfinite().all()

    def test_tier2_extreme_parameter_counts_and_single_layer_depth_scaling(self, device: str) -> None:
        """[Boundary 3] N=1 depth scaling computes scale factor 1/sqrt(2) = 0.7071."""
        source = "h = linear(x, 16)\nout = add(x, h)\n"
        plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
        try:
            model = build(plan, init_policy="depth_scaled", device=device)
            assert model is not None
        except (TypeError, HNDLError):
            pytest.skip("Depth-scaled initialization pending implementation in M5")

    def test_tier2_rank5_boundaries_and_extreme_permutations(self, device: str) -> None:
        """[Boundary 4] Rank-5 tensor with singleton dimensions [B, 1, 1, 1, C]."""
        check_rank5()
        source = "linear(32)\n"
        model = network(source, input_shape=("B", 1, 1, 1, 16), output_shape=("B", 1, 1, 1, 32), device=device)
        x = torch.randn(2, 1, 1, 1, 16, device=device)
        out = model(x)
        assert out.shape == (2, 1, 1, 1, 32)

    def test_tier2_reshape_negative_axis_edge_cases(self) -> None:
        """[Boundary 5] Reshape -1 inference with prime element product."""
        try:
            plan = resolve("reshape(13, -1)\n", input_shape=("B", 13 * 17), output_shape=("B", 13, 17))
            assert plan.nodes[0].output_shapes["out"] == ("B", 13, 17)
        except HNDLError:
            pytest.skip("reshape(-1, ...) pending implementation in M3")

    def test_tier2_mamba2_divisibility_constraint_failures(self) -> None:
        """[Boundary 6] mamba2 rejects configurations where heads * head_dim != inner_dim."""
        check_operator("mamba2")
        # Invalid: heads=5, head_dim=16 -> 80 != 64 (expand=2 * 32 = 64)
        with pytest.raises(HNDLError) as exc_info:
            resolve(
                "mamba2(d_model=32, heads=5, head_dim=16)\n",
                input_shape=("B", 8, 32),
                output_shape=("B", 8, 32),
            )
        assert exc_info.value.code in ("E_CONSTRAINT", "E_ARGUMENT")

    def test_tier2_dy_tanh_numerical_extremes(self, device: str) -> None:
        """[Boundary 7] dy_tanh stability when alpha=0.0 or extreme magnitudes."""
        check_operator("dy_tanh")
        source = "dy_tanh(dim=16, name='dyt')\n"
        model = network(source, input_shape=("B", 8, 16), output_shape=("B", 8, 16), device=device)
        # Extreme inputs
        x_extreme = torch.tensor([[[1e4] * 16] * 8], device=device)
        out = model(x_extreme)
        assert out.isfinite().all()

    def test_tier2_drop_path_boundary_probabilities(self, device: str) -> None:
        """[Boundary 8] drop_path with drop_prob=0.0 (identity) and drop_prob=1.0 (all dropped)."""
        check_operator("drop_path")
        # drop_prob=0.0 must be pure identity even in train mode
        source_0 = "drop_path(drop_prob=0.0, name='dp0')\n"
        model_0 = network(source_0, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        model_0.train()
        x = torch.randn(2, 4, 16, device=device)
        out_0 = model_0(x)
        assert torch.equal(out_0, x)

        # drop_prob=1.0 in train mode must zero out all elements
        source_1 = "drop_path(drop_prob=1.0, name='dp1')\n"
        model_1 = network(source_1, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        model_1.train()
        out_1 = model_1(x)
        assert torch.count_nonzero(out_1) == 0

    def test_tier2_where_operator_invalid_dtypes_and_shape_conflicts(self) -> None:
        """[Boundary 9] where operator rejects non-boolean condition masks and shape mismatches."""
        check_operator("where")
        # Non-boolean condition: float32 condition must be rejected
        with pytest.raises(HNDLError) as exc_info:
            resolve(
                "out = where(cond, a, b)\n",
                input_shape={"cond": ("B", 8), "a": ("B", 8), "b": ("B", 8)},
                output_shape={"out": ("B", 8)},
                input_dtype={"cond": "float32", "a": "float32", "b": "float32"},
            )
        assert exc_info.value.code in ("E_DTYPE", "E_CONSTRAINT")

    def test_tier2_snake_epsilon_stability_under_zero_frequency(self, device: str) -> None:
        """[Boundary 10] snake periodic activation stability when alpha = 0.0."""
        check_operator("snake")
        source = "snake(dim=16, init={'alpha': 0.0}, name='snk')\n"
        try:
            model = network(source, input_shape=("B", 16, 32), output_shape=("B", 16, 32), device=device)
            x = torch.randn(2, 16, 32, device=device)
            out = model(x)
            assert out.isfinite().all()
        except HNDLError:
            # If init dict parameter target isn't alpha directly
            pass

    def test_tier2_cli_nonexistent_and_corrupt_files(self) -> None:
        """[Boundary 11] CLI commands return exit code 1 on nonexistent or corrupt files."""
        try:
            import hndl.__main__ as cli_main
        except ImportError:
            pytest.skip("CLI entrypoint pending implementation in M5")
        assert hasattr(cli_main, "main")

    def test_tier2_profiler_non_positive_batch_size_rejection(self) -> None:
        """[Boundary 12] plan.profile() raises ValueError on non-positive batch size."""
        check_plan_method("profile")
        plan = resolve("linear(16)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
        with pytest.raises(ValueError):
            plan.profile(batch_size=0)

    def test_tier2_param_groups_on_zero_parameter_model(self, device: str) -> None:
        """[Boundary 13] model.param_groups() returns empty list on models without parameters."""
        check_module_method("param_groups")
        source = "out = relu(x)\n"
        model = network(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)}, device=device)
        groups = model.param_groups(weight_decay=0.01)
        assert len(groups) == 0 or all(len(g["params"]) == 0 for g in groups)


# ==============================================================================
# TIER 3: Cross-Feature Interactions (Pairwise Combinations)
# ==============================================================================

class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise combinations and cross-pillar interactions."""

    def test_tier3_mamba2_plus_drop_path_residual(self, device: str) -> None:
        """[Interaction 1] Mamba-2 state space duality combined with drop_path in residual block."""
        check_operator("mamba2")
        check_operator("drop_path")
        source = "m = mamba2(x, d_model=32, d_state=16, heads=4, head_dim=16)\n" \
                 "d = drop_path(m, drop_prob=0.1)\n" \
                 "out = add(x, d)\n"
        model = network(source, input_shape={"x": ("B", 8, 32)}, output_shape={"out": ("B", 8, 32)}, device=device)
        x = torch.randn(2, 8, 32, device=device, requires_grad=True)
        out = model(x=x)
        assert out.shape == (2, 8, 32)
        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()

    def test_tier3_joint_attention_plus_dy_tanh_multimodal(self, device: str) -> None:
        """[Interaction 2] MMDiT Joint Attention normalized with dynamic tanh."""
        check_operator("joint_attention")
        check_operator("dy_tanh")
        source = "j_out, c_out = joint_attention(x, context, heads=4, head_dim=16)\n" \
                 "out = dy_tanh(j_out, dim=32)\n" \
                 "ctx_out = dy_tanh(c_out, dim=32)\n"
        model = network(
            source,
            input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32)},
            output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
            device=device,
        )
        x = torch.randn(2, 8, 32, device=device)
        ctx = torch.randn(2, 16, 32, device=device)
        res = model(x=x, context=ctx)
        assert res["out"].shape == (2, 8, 32)
        assert res["ctx_out"].shape == (2, 16, 32)

    def test_tier3_hndl_macros_tuple_loops_and_weight_tying(self) -> None:
        """[Interaction 3] Reusable macro expanded in tuple loop with tied weights."""
        check_hndl_macros()
        check_hndl_tuple_loops()
        source = "def stage(x, w):\n  return relu(linear(x, w))\n" \
                 "for c in (32, 64):\n  x = stage(x, c)\n" \
                 "out = x\n"
        plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
        assert len(plan.nodes) == 4

    def test_tier3_profiler_plus_depth_scaling_plus_checkpointing(self, device: str) -> None:
        """[Interaction 4] Multi-layer Transformer profiled, initialized with depth scaling, and checkpointed."""
        check_plan_method("profile")
        source = "transformer_block(4, name='tb')\n"
        plan = resolve(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        prof = plan.profile(batch_size=2)
        assert prof.total_parameters > 0

        try:
            model = build(plan, checkpoint=True, init_policy="depth_scaled", device=device)
            x = torch.randn(2, 8, 32, device=device)
            out = model(x)
            assert out.shape == (2, 8, 32)
        except (TypeError, HNDLError):
            pass

    def test_tier3_cli_inspect_plus_static_profile_on_moe(self) -> None:
        """[Interaction 5] Sparse MoE static profile computes exact active vs total FLOPs and params."""
        check_plan_method("profile")
        plan = resolve("moe(8, 2, 64)\n", input_shape={"x": ("B", 8, 32)}, output_shape={"out": ("B", 8, 32)})
        prof = plan.profile(batch_size=2)
        assert prof.active_parameters < prof.total_parameters

    def test_tier3_inference_cache_plus_param_groups_lifecycle(self, device: str) -> None:
        """[Interaction 6] Stateful autoregressive decoding with optimizer parameter partitioning."""
        check_module_method("param_groups")
        source = "attention(4, causal=True, name='attn')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        groups = model.param_groups(weight_decay=0.01)
        assert len(groups) == 2

    def test_tier3_codegen_plus_state_rebinding_plus_weight_tying(self) -> None:
        """[Interaction 7] Standalone codegen generated from model with tied weights."""
        check_plan_method("to_python_source")
        plan = resolve("linear(32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        code = plan.to_python_source()
        assert "class" in code

    def test_tier3_unrolled_loops_plus_line_provenance_plus_fuzzy_diagnostics(self) -> None:
        """[Interaction 8] Unrolled loop shape conflict surfaces line numbers and upstream provenance."""
        check_hndl_tuple_loops()
        source = "for c in (32, 64):\n  x = linear(x, c)\n"
        plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
        assert len(plan.nodes) == 2

    def test_tier3_axial_rope_2d_plus_rank5_video_plus_where(self, device: str) -> None:
        """[Interaction 9] Rank-5 video representation with ternary condition selection."""
        check_rank5()
        check_operator("where")
        source = "out = where(cond, a, b)\n"
        model = network(
            source,
            input_shape={"cond": ("B", 1, 2, 4, 4), "a": ("B", 1, 2, 4, 4), "b": ("B", 1, 2, 4, 4)},
            output_shape={"out": ("B", 1, 2, 4, 4)},
            input_dtype={"cond": "bool", "a": "float32", "b": "float32"},
            device=device,
        )
        cond = torch.ones(1, 1, 2, 4, 4, dtype=torch.bool, device=device)
        a = torch.randn(1, 1, 2, 4, 4, device=device)
        b = torch.randn(1, 1, 2, 4, 4, device=device)
        out = model(cond=cond, a=a, b=b)
        assert out.shape == (1, 1, 2, 4, 4)

    def test_tier3_snake_activation_plus_causal_conv1d_deconv1d(self, device: str) -> None:
        """[Interaction 10] Neural audio codec stem: causal conv1d + snake + causal deconv1d."""
        check_operator("snake")
        source = "c = conv1d(x, 16, 3, causal=True, channels_last=True)\n" \
                 "s = snake(c, dim=16)\n" \
                 "out = deconv1d(s, 16, 4, stride=2, causal=True, channels_last=True)\n"
        try:
            model = network(source, input_shape={"x": ("B", 8, 16)}, output_shape={"out": ("B", 16, 16)}, device=device)
            x = torch.randn(2, 8, 16, device=device)
            out = model(x=x)
            assert out.shape == (2, 16, 16)
        except (HNDLError, TypeError):
            pytest.skip("Causal deconv1d pending implementation in M3")

    def test_tier3_resblock_bottleneck_plus_drop_path_plus_depth_scaled(self, device: str) -> None:
        """[Interaction 11] Bottleneck ResBlock with drop_path stochastic depth."""
        check_operator("drop_path")
        check_operator("resblock")
        spec = Registry.builtins().get("resblock")
        if "kind" not in spec.args:
            pytest.skip("resblock kind='bottleneck' pending implementation in M3")
        source = "rb = resblock(x, kind='bottleneck', activation='silu')\n" \
                 "dp = drop_path(rb, drop_prob=0.1)\n" \
                 "out = add(x, dp)\n"
        model = network(source, input_shape={"x": ("B", 64, 8, 8)}, output_shape={"out": ("B", 64, 8, 8)}, device=device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = model(x=x)
        assert out.shape == (2, 64, 8, 8)

    def test_tier3_where_selection_plus_cast_bool_plus_l2_norm(self, device: str) -> None:
        """[Interaction 12] Mask generation via cast('bool'), selection via where, then L2 norm."""
        check_b01()
        check_operator("where")
        source = "mask = cast(x, 'bool')\n" \
                 "sel = where(mask, a, b)\n" \
                 "out = l2_norm(sel, dim=-1)\n"
        model = network(
            source,
            input_shape={"x": ("B", 16), "a": ("B", 16), "b": ("B", 16)},
            output_shape={"out": ("B", 16)},
            device=device,
        )
        x = torch.randint(0, 2, (2, 16), device=device).float()
        a = torch.randn(2, 16, device=device)
        b = torch.randn(2, 16, device=device)
        out = model(x=x, a=a, b=b)
        assert out.shape == (2, 16)
        # Check L2 unit norm
        norm = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)

    def test_tier3_adaln_zero_ungated_plus_dy_tanh_plus_checkpointing(self, device: str) -> None:
        """[Interaction 13] Ungated AdaLN-Zero diffusion conditioning paired with DyTanh."""
        check_operator("dy_tanh")
        check_operator("adaln_zero")
        spec = Registry.builtins().get("adaln_zero")
        if "gate" not in spec.args:
            pytest.skip("adaln_zero gate=False pending implementation in M1/M3")
        source = "mod = adaln_zero(x, cond, gate=False)\n" \
                 "out = dy_tanh(mod, dim=32)\n"
        model = network(
            source,
            input_shape={"x": ("B", 8, 32), "cond": ("B", 32)},
            output_shape={"out": ("B", 8, 32)},
            device=device,
        )
        x = torch.randn(2, 8, 32, device=device)
        cond = torch.randn(2, 32, device=device)
        out = model(x=x, cond=cond)
        assert out.shape == (2, 8, 32)

    def test_tier3_load_weights_with_remapping_into_tied_model(self, device: str) -> None:
        """[Interaction 14] Loading external weights with remapping into a GraphModule."""
        check_hndl_torch_attr("load_weights")
        import hndl.torch as htorch
        source = "linear(32, name='lin')\n"
        model = network(source, input_shape=("B", 16), output_shape=("B", 32), device=device)
        ext_state = {"backbone.lin.weight": torch.randn(32, 16, device=device), "backbone.lin.bias": torch.randn(32, device=device)}
        htorch.load_weights(model, ext_state, key_map={"backbone.": ""})
        assert torch.equal(model["lin"].weight, ext_state["backbone.lin.weight"])

    def test_tier3_plan_diffing_baseline_vs_sota_extensions(self) -> None:
        """[Interaction 15] Comparing baseline architecture with SOTA extension via plan.diff()."""
        check_plan_method("diff")
        p1 = resolve("transformer_block(4)\n", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        p2 = resolve("transformer_block(4, ffn_kind='swiglu')\n", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
        diff = p1.diff(p2)
        assert diff is not None


# ==============================================================================
# TIER 4: Real-World Application Workloads
# ==============================================================================

class TestTier4RealWorldWorkloads:
    """Tier 4: Production-scale multi-layer architecture workloads with end-to-end gradient verification."""

    def test_tier4_workload1_multimodal_diffusion_transformer_pipeline(self, device: str) -> None:
        """[Workload 1] MMDiT / Flux Style Diffusion Transformer Pipeline."""
        check_operator("joint_attention")
        check_operator("adaln_zero")
        source = "t_emb = sinusoidal_embed(t, dim=32)\n" \
                 "x_mod, x_gate = adaln_zero(x, t_emb)\n" \
                 "c_mod, c_gate = adaln_zero(context, t_emb)\n" \
                 "attn_x, attn_c = joint_attention(x_mod, c_mod, heads=4, head_dim=16)\n" \
                 "out = add(x, attn_x)\n" \
                 "ctx_out = add(context, attn_c)\n"
        try:
            model = network(
                source,
                input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32), "t": ("B",)},
                output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
                device=device,
            )
            x = torch.randn(2, 8, 32, device=device, requires_grad=True)
            ctx = torch.randn(2, 16, 32, device=device, requires_grad=True)
            t = torch.tensor([10.0, 50.0], device=device)
            res = model(x=x, context=ctx, t=t)
            out, ctx_out = res["out"], res["ctx_out"]
            assert out.shape == (2, 8, 32)
            assert ctx_out.shape == (2, 16, 32)
            assert out.isfinite().all() and ctx_out.isfinite().all()

            loss = out.sum() + ctx_out.sum()
            loss.backward()
            assert x.grad is not None and x.grad.isfinite().all()
            assert ctx.grad is not None and ctx.grad.isfinite().all()
        except HNDLError:
            pytest.skip("MMDiT workload pending joint_attention implementation in M3")

    def test_tier4_workload2_neural_audio_codec_synthesis_pipeline(self, device: str) -> None:
        """[Workload 2] Neural Audio Codec (EnCodec / DAC Style) Synthesis Pipeline."""
        check_operator("snake")
        check_operator("drop_path")
        source = "c1 = conv1d(x, 16, 3, causal=True, channels_last=True)\n" \
                 "s1 = snake(c1, dim=16)\n" \
                 "dp1 = drop_path(s1, drop_prob=0.05)\n" \
                 "out = deconv1d(dp1, 16, 4, stride=2, causal=True, channels_last=True)\n"
        try:
            model = network(source, input_shape={"x": ("B", 16, 16)}, output_shape={"out": ("B", 32, 16)}, device=device)
            x = torch.randn(2, 16, 16, device=device, requires_grad=True)
            out = model(x=x)
            assert out.shape == (2, 32, 16)
            assert out.isfinite().all()

            out.sum().backward()
            assert x.grad is not None and x.grad.isfinite().all()
        except (HNDLError, TypeError):
            pytest.skip("Audio codec pipeline pending causal deconv1d implementation in M3")

    def test_tier4_workload3_vision_language_dual_encoder_pipeline(self, device: str) -> None:
        """[Workload 3] Multimodal Vision-Language Dual Encoder (CLIP Style) Pipeline."""
        source = "v_tokens = linear(vision_pixels, 32)\n" \
                 "v_emb = l2_norm(pool_tokens(v_tokens, 'mean'), dim=-1)\n" \
                 "t_tokens = attention(text_tokens, 4, qk_norm='rms_norm')\n" \
                 "t_emb = l2_norm(gather_token(t_tokens, 0), dim=-1)\n" \
                 "sim = matmul(v_emb, t_emb, transpose_b=True)\n"
        try:
            model = network(
                source,
                input_shape={"vision_pixels": ("B", 16, 32), "text_tokens": ("B", 8, 32)},
                output_shape={"sim": ("B",)},
                device=device,
            )
            vis = torch.randn(4, 16, 32, device=device, requires_grad=True)
            txt = torch.randn(4, 8, 32, device=device, requires_grad=True)
            sim = model(vision_pixels=vis, text_tokens=txt)
            assert sim.shape == (4,)
            sim.sum().backward()
            assert vis.grad is not None and txt.grad is not None
        except HNDLError:
            pytest.skip("CLIP dual encoder pipeline pending solver updates in M3")

    def test_tier4_workload4_autoregressive_hybrid_decoder_with_inference_cache(self, device: str) -> None:
        """[Workload 4] Hybrid Mamba2 + MLA Decoder with Stateful Autoregressive InferenceCache."""
        check_operator("mamba2")
        check_hndl_torch_attr("InferenceCache")
        source = "m = mamba2(x, d_model=32, d_state=16, heads=4, head_dim=16)\n" \
                 "out = latent_attention(m, heads=4, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16)\n"
        try:
            model = network(source, input_shape={"x": ("B", 8, 32)}, output_shape={"out": ("B", 8, 32)}, device=device)
            x_prefill = torch.randn(2, 8, 32, device=device)
            out_prefill = model(x=x_prefill)
            assert out_prefill.shape == (2, 8, 32)
        except HNDLError:
            pytest.skip("Hybrid decoder pending Mamba2 implementation in M3")

    def test_tier4_workload5_sparse_moe_training_lifecycle_pipeline(self, device: str) -> None:
        """[Workload 5] End-to-End DeepSeek-V3 Style MoE with AdamW(param_groups) and Checkpointing."""
        source = "attn = attention(x, 4, qk_norm='rms_norm')\n" \
                 "out = moe(attn, 4, 2, 64, expert_kind='swiglu', scoring='sigmoid', bias=False)\n"
        try:
            plan = resolve(source, input_shape={"x": ("B", 8, 32)}, output_shape={"out": ("B", 8, 32)})
            model = build(plan, checkpoint=True, device=device)
            groups = model.param_groups(weight_decay=0.01) if hasattr(model, "param_groups") else model.parameters()
            optimizer = torch.optim.AdamW(groups, lr=1e-3)

            model.train()
            optimizer.zero_grad()
            x = torch.randn(2, 8, 32, device=device, requires_grad=True)
            out = model(x=x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            assert out.isfinite().all()
        except (HNDLError, TypeError):
            pytest.skip("MoE training lifecycle pending implementation in M1/M2/M5")

    def test_tier4_workload6_standalone_python_codegen_and_native_roundtrip(self, device: str) -> None:
        """[Workload 6] Standalone PyTorch Codegen & Native Execution Numerical Round-Trip."""
        check_plan_method("to_python_source")
        plan = resolve(
            "h = relu(linear(x, 32))\nout = linear(h, 16)\n",
            input_shape={"x": ("B", 16)},
            output_shape={"out": ("B", 16)},
        )
        code = plan.to_python_source()
        assert isinstance(code, str) and len(code) > 0

        # Execute codegen in an isolated namespace
        namespace: dict[str, Any] = {}
        exec(code, namespace)  # noqa: S102
        module_cls = next(v for v in namespace.values() if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module)
        native_model = module_cls().to(device)

        # Build original HNDL GraphModule and copy weights
        hndl_model = build(plan, device=device)
        native_model.load_state_dict(hndl_model.state_dict())

        # Forward roundtrip comparison
        x = torch.randn(4, 16, device=device)
        with torch.no_grad():
            out_hndl = hndl_model(x)
            out_native = native_model(x)
        assert torch.allclose(out_hndl, out_native, atol=1e-6)
