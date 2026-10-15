"""Empirical Stress-Test and Verification Harness for Milestone 3 (Operators, Numerics & References).

Validates:
- M-12: Dense MoE reference oracle parity against MixtureOfExperts.forward across various
  batch sizes, token counts, number of experts (2, 4, 8, 16), top-k (1, 2, 4), hidden dimensions,
  activations, dtypes, and autograd gradient parity.
- M-13: Mathematical reference oracle parity (reference=) across all 19 updated operators:
  adaptive_norm, add, chunk, cls_token, concat, conv, coordinate_grid, deconv, flatten,
  fourier_features, grid_sample, group_norm, leaky_relu, linear, pos_embed, relu, reshape,
  split, tanh; plus full reference coverage check across all 68 non-pretrained operators.
- L-07: GridSample positional vs keyword argument rejection in declarative plan resolution,
  positional=False argument metadata, and Chunk cleanup.
- L-08: ResBlock norm="group_norm" divisibility validation (out_channels % groups == 0),
  shortcut handling, and extra_repr() validation across CoordinateGrid, FourierFeatures,
  GridSample, EqualLinear, Attention, and ResBlock.
"""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.operators._equalized import EqualLinear
from hndl.operators.adaptive_norm import AdaptiveNorm
from hndl.operators.add import Add
from hndl.operators.attention import Attention
from hndl.operators.chunk import Chunk
from hndl.operators.cls_token import ClsToken
from hndl.operators.concat import Concat
from hndl.operators.conv import Conv2d
from hndl.operators.coordinate_grid import CoordinateGrid
from hndl.operators.deconv import ConvTranspose2d
from hndl.operators.flatten import Flatten
from hndl.operators.fourier_features import FourierFeatures
from hndl.operators.grid_sample import GridSample
from hndl.operators.group_norm import GroupNorm
from hndl.operators.leaky_relu import LeakyReLU
from hndl.operators.linear import Linear
from hndl.operators.moe import MixtureOfExperts, _reference as moe_reference
from hndl.operators.pos_embed import PosEmbed
from hndl.operators.relu import ReLU
from hndl.operators.resblock import ResBlock
from hndl.operators.reshape import Reshape
from hndl.operators.split import Split
from hndl.operators.tanh import Tanh
from hndl.torch import build


# ==============================================================================
# 1. Dense MoE Reference Oracle Parity (M-12)
# ==============================================================================

class TestMoEOracleParity:
    """Empirical verification of MixtureOfExperts vs dense reference oracle."""

    @pytest.mark.parametrize("experts,top_k", [
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (8, 4),
        (16, 1),
        (16, 2),
        (16, 4),
    ])
    @pytest.mark.parametrize("batch,tokens,d_model,hidden", [
        (1, 1, 16, 32),
        (2, 5, 24, 48),
        (4, 17, 32, 64),
    ])
    @pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "relu", "silu"])
    def test_moe_dense_oracle_forward_parity(self, experts, top_k, batch, tokens, d_model, hidden, activation):
        """MoE forward matches dense reference oracle across diverse configurations."""
        torch.manual_seed(42 + experts + top_k)
        moe = MixtureOfExperts(experts=experts, hidden=hidden, top_k=top_k, activation=activation, D=d_model)
        oracle = moe_reference(moe)

        x = torch.randn(batch, tokens, d_model)
        out_actual = moe(x)
        out_oracle = oracle(x)

        assert out_actual.shape == (batch, tokens, d_model)
        assert out_oracle.shape == (batch, tokens, d_model)
        torch.testing.assert_close(out_actual, out_oracle, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("experts,top_k", [(2, 1), (4, 2), (8, 4)])
    def test_moe_2d_input_forward_parity(self, experts, top_k):
        """MoE forward works identically on 2D [B, D] inputs."""
        moe = MixtureOfExperts(experts=experts, hidden=32, top_k=top_k, activation="gelu", D=16)
        oracle = moe_reference(moe)

        x = torch.randn(7, 16)
        out_actual = moe(x)
        out_oracle = oracle(x)

        assert out_actual.shape == (7, 16)
        torch.testing.assert_close(out_actual, out_oracle, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_moe_precision_and_gradcheck(self, dtype):
        """Dense oracle and module preserve precision and produce matching gradients."""
        torch.manual_seed(101)
        tol = 1e-5 if dtype == torch.float32 else 1e-8
        moe = MixtureOfExperts(experts=4, hidden=16, top_k=2, activation="silu", D=8).to(dtype)
        oracle = moe_reference(moe)

        x1 = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        out1 = moe(x1)
        out2 = oracle(x2)

        torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)

        grad_target = torch.randn_like(out1)
        out1.backward(grad_target)
        out2.backward(grad_target)

        torch.testing.assert_close(x1.grad, x2.grad, atol=tol, rtol=tol)

    def test_moe_invalid_top_k_rejected(self):
        """top_k > experts is rejected with E_ARGUMENT."""
        reg = Registry.builtins()
        spec = reg.get("moe")
        with pytest.raises(HNDLError) as exc_info:
            spec.validate({"experts": 4, "top_k": 5, "hidden": 16, "activation": "gelu"})
        assert exc_info.value.code == "E_ARGUMENT"

    def test_moe_4d_tensor_rejected_in_relation(self):
        """MoE rejects 4D image tensors [B, C, H, W] with E_CONSTRAINT."""
        with pytest.raises(HNDLError) as exc_info:
            resolve("moe(4, 32)", input_shape=("B", 16, 4, 4), output_shape=("B", 16, 4, 4))
        assert exc_info.value.code == "E_CONSTRAINT"

    def test_moe_extra_repr(self):
        """extra_repr documents experts count, hidden width, top_k, and activation."""
        moe = MixtureOfExperts(experts=4, hidden=32, top_k=2, activation="silu", D=16)
        rep = moe.extra_repr()
        assert "experts=4" in rep
        assert "hidden=32" in rep
        assert "top_k=2" in rep
        assert "activation='silu'" in rep


# ==============================================================================
# 2. Operator Forward vs Reference Parity across all 19 Updated Operators (M-13)
# ==============================================================================

class TestAll19UpdatedOperatorsParity:
    """Empirical verification that all 19 updated operators match reference= functions."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        self.reg = Registry.builtins()

    def test_all_68_non_pretrained_operators_have_reference(self):
        """100% of non-pretrained operators must have a reference implementation."""
        missing = [
            name for name in self.reg.aliases
            if name != "pretrained" and self.reg.get(name).reference is None
        ]
        assert not missing, f"Operators missing reference=: {missing}"

    def test_parity_01_adaptive_norm(self):
        spec = self.reg.get("adaptive_norm")
        assert spec.reference is not None
        for channels in (4, 8):
            module = AdaptiveNorm(eps=1e-5)
            oracle = spec.reference(module)
            x = torch.randn(2, channels, 6, 6, requires_grad=True)
            p = torch.randn(2, 2 * channels, requires_grad=True)

            out_mod = module(x, p)
            out_ref = oracle(x, p)
            torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)

            # Gradient parity
            g = torch.randn_like(out_mod)
            gx_m, gp_m = torch.autograd.grad(out_mod, (x, p), g, retain_graph=True)
            gx_r, gp_r = torch.autograd.grad(out_ref, (x, p), g)
            torch.testing.assert_close(gx_m, gx_r, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(gp_m, gp_r, atol=1e-5, rtol=1e-5)

    def test_parity_02_add(self):
        spec = self.reg.get("add")
        assert spec.reference is not None
        module = Add()
        oracle = spec.reference(module)
        for shape in [(3, 8), (2, 4, 6), (2, 3, 4, 5)]:
            a = torch.randn(*shape, requires_grad=True)
            b = torch.randn(*shape, requires_grad=True)
            out_mod = module(a, b)
            out_ref = oracle(a, b)
            torch.testing.assert_close(out_mod, out_ref)

    def test_parity_03_chunk(self):
        spec = self.reg.get("chunk")
        assert spec.reference is not None
        for chunks in (2, 3):
            for dim in (0, 1, 2):
                module = Chunk(chunks=chunks, dim=dim)
                oracle = spec.reference(module)
                shape = [6, 12, 18]
                x = torch.randn(*shape, requires_grad=True)
                out_mod = module(x)
                out_ref = oracle(x)
                assert len(out_mod) == chunks
                assert len(out_ref) == chunks
                for m_sec, r_sec in zip(out_mod, out_ref, strict=True):
                    torch.testing.assert_close(m_sec, r_sec)

    def test_parity_04_cls_token(self):
        spec = self.reg.get("cls_token")
        assert spec.reference is not None
        module = ClsToken(D=16)
        oracle = spec.reference(module)
        x = torch.randn(3, 7, 16, requires_grad=True)
        out_mod = module(x)
        out_ref = oracle(x)
        assert out_mod.shape == (3, 8, 16)
        torch.testing.assert_close(out_mod, out_ref)

    def test_parity_05_concat(self):
        spec = self.reg.get("concat")
        assert spec.reference is not None
        for axis in (0, 1, 2):
            module = Concat(input_count=3, axis=axis)
            oracle = spec.reference(module)
            shapes = [[2, 4, 6] for _ in range(3)]
            xs = [torch.randn(*s, requires_grad=True) for s in shapes]
            out_mod = module(*xs)
            out_ref = oracle(*xs)
            torch.testing.assert_close(out_mod, out_ref)

    def test_parity_06_conv(self):
        spec = self.reg.get("conv")
        assert spec.reference is not None
        # Normal conv and spectral norm conv
        for sn in (False, True):
            module = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1,
                            padding=1, dilation=1, groups=1, bias=True, spectral_norm=sn)
            oracle = spec.reference(module)
            x = torch.randn(2, 3, 8, 8)
            out_mod = module(x)
            out_ref = oracle(x)
            torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)

    def test_parity_07_coordinate_grid(self):
        spec = self.reg.get("coordinate_grid")
        assert spec.reference is not None
        module = CoordinateGrid(height=6, width=8)
        oracle = spec.reference(module)
        x = torch.randn(4, 1, 1, 1)
        out_mod = module(x)
        out_ref = oracle(x)
        assert out_mod.shape == (4, 6, 8, 2)
        torch.testing.assert_close(out_mod, out_ref)

    def test_parity_08_deconv(self):
        spec = self.reg.get("deconv")
        assert spec.reference is not None
        for sn in (False, True):
            module = ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, stride=2,
                                     padding=1, dilation=1, output_padding=1, groups=1, bias=True, spectral_norm=sn)
            oracle = spec.reference(module)
            x = torch.randn(2, 6, 4, 4)
            out_mod = module(x)
            out_ref = oracle(x)
            torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)

    def test_parity_09_flatten(self):
        spec = self.reg.get("flatten")
        assert spec.reference is not None
        module = Flatten()
        oracle = spec.reference(module)
        for shape in [(2, 3, 4), (2, 3, 4, 5), (1, 2, 3, 4, 5)]:
            x = torch.randn(*shape)
            out_mod = module(x)
            out_ref = oracle(x)
            torch.testing.assert_close(out_mod, out_ref)

    def test_parity_10_fourier_features(self):
        spec = self.reg.get("fourier_features")
        assert spec.reference is not None
        module = FourierFeatures(num_frequencies=16, scale=4.0, D=2)
        oracle = spec.reference(module)
        x = torch.randn(3, 5, 2)
        out_mod = module(x)
        out_ref = oracle(x)
        assert out_mod.shape == (3, 5, 32)
        torch.testing.assert_close(out_mod, out_ref)

    def test_parity_11_grid_sample(self):
        spec = self.reg.get("grid_sample")
        assert spec.reference is not None
        for mode in ("bilinear", "nearest"):
            for padding_mode in ("zeros", "border"):
                for align_corners in (False, True):
                    module = GridSample(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
                    oracle = spec.reference(module)
                    x = torch.randn(2, 3, 6, 6)
                    grid = torch.clamp(torch.randn(2, 4, 5, 2), -1.0, 1.0)
                    out_mod = module(x, grid)
                    out_ref = oracle(x, grid)
                    torch.testing.assert_close(out_mod, out_ref)

    def test_parity_12_group_norm(self):
        spec = self.reg.get("group_norm")
        assert spec.reference is not None
        for affine in (True, False):
            module = GroupNorm(num_groups=4, num_channels=16, eps=1e-5, affine=affine)
            oracle = spec.reference(module)
            x = torch.randn(2, 16, 8, 8)
            out_mod = module(x)
            out_ref = oracle(x)
            torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)

    def test_parity_13_leaky_relu(self):
        spec = self.reg.get("leaky_relu")
        assert spec.reference is not None
        for slope in (0.01, 0.2, -0.5):
            module = LeakyReLU(negative_slope=slope)
            oracle = spec.reference(module)
            x = torch.randn(4, 16)
            out_mod = module(x)
            out_ref = oracle(x)
            torch.testing.assert_close(out_mod, out_ref)

    def test_parity_14_linear(self):
        spec = self.reg.get("linear")
        assert spec.reference is not None
        # standard, equalized, spectral_norm
        for eq, sn in [(False, False), (True, False), (False, True)]:
            module = Linear(in_features=16, out_features=8, bias=True, spectral_norm=sn, equalized=eq)
            oracle = spec.reference(module)
            x = torch.randn(3, 16)
            out_mod = module(x)
            out_ref = oracle(x)
            torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)

    def test_parity_15_pos_embed(self):
        spec = self.reg.get("pos_embed")
        assert spec.reference is not None
        module = PosEmbed(max_len=32, D=16)
        oracle = spec.reference(module)
        x = torch.randn(2, 10, 16)
        out_mod = module(x)
        out_ref = oracle(x)
        torch.testing.assert_close(out_mod, out_ref)

        # Boundary check: T > max_len raises E_CONSTRAINT
        x_overflow = torch.randn(2, 33, 16)
        with pytest.raises(HNDLError) as exc_info:
            module(x_overflow)
        assert exc_info.value.code == "E_CONSTRAINT"

    def test_parity_16_relu(self):
        spec = self.reg.get("relu")
        assert spec.reference is not None
        module = ReLU()
        oracle = spec.reference(module)
        x = torch.randn(3, 12)
        torch.testing.assert_close(module(x), oracle(x))

    def test_parity_17_reshape(self):
        spec = self.reg.get("reshape")
        assert spec.reference is not None
        module = Reshape(shape=(4, 4))
        oracle = spec.reference(module)
        x = torch.randn(2, 16)
        torch.testing.assert_close(module(x), oracle(x))

    def test_parity_18_split(self):
        spec = self.reg.get("split")
        assert spec.reference is not None
        module = Split(size=4, dim=1, input_shapes={"x": (2, 10)})
        oracle = spec.reference(module)
        x = torch.randn(2, 10)
        out_mod = module(x)
        out_ref = oracle(x)
        assert len(out_mod) == 2
        assert len(out_ref) == 2
        torch.testing.assert_close(out_mod[0], out_ref[0])
        torch.testing.assert_close(out_mod[1], out_ref[1])

    def test_parity_19_tanh(self):
        spec = self.reg.get("tanh")
        assert spec.reference is not None
        module = Tanh()
        oracle = spec.reference(module)
        x = torch.randn(3, 8)
        torch.testing.assert_close(module(x), oracle(x))


# ==============================================================================
# 3. ResBlock Validation and Representations (L-08)
# ==============================================================================

class TestResBlockValidationAndRepr:
    """Empirical verification of ResBlock group_norm divisibility and representations."""

    @pytest.mark.parametrize("out_channels,groups", [
        (15, 8),
        (10, 4),
        (7, 2),
        (14, 4),
        (25, 10),
    ])
    def test_resblock_direct_init_group_norm_indivisible_rejected(self, out_channels, groups):
        """Direct ResBlock init raises E_CONSTRAINT when out_channels % groups != 0."""
        with pytest.raises(HNDLError) as exc_info:
            ResBlock(out_channels=out_channels, in_channels=16, stride=1,
                     norm="group_norm", groups=groups, C_in=16, C_out=out_channels)
        assert exc_info.value.code == "E_CONSTRAINT"
        assert f"groups={groups} must divide out_channels={out_channels}" in str(exc_info.value)

    @pytest.mark.parametrize("out_channels,groups", [
        (16, 8),
        (32, 8),
        (12, 4),
        (8, 8),
        (10, 5),
    ])
    def test_resblock_direct_init_group_norm_divisible_succeeds(self, out_channels, groups):
        """Direct ResBlock init succeeds when out_channels % groups == 0."""
        block = ResBlock(out_channels=out_channels, in_channels=16, stride=1,
                         norm="group_norm", groups=groups, C_in=16, C_out=out_channels)
        assert block.norm == "group_norm"
        assert block.groups == groups

    @pytest.mark.parametrize("norm", ["batch_norm", "none"])
    def test_resblock_non_group_norm_ignores_indivisible_groups(self, norm):
        """norm='batch_norm' or 'none' does not fail even if out_channels % groups != 0."""
        block = ResBlock(out_channels=15, in_channels=16, stride=1,
                         norm=norm, groups=8, C_in=16, C_out=15)
        assert block.norm == norm

    def test_resblock_resolve_group_norm_indivisible_rejected(self):
        """Plan resolution rejects indivisible group_norm with E_CONSTRAINT."""
        with pytest.raises(HNDLError) as exc_info:
            resolve("resblock(15, norm='group_norm', groups=8)",
                    input_shape=("B", 16, 8, 8), output_shape=("B", 15, 8, 8))
        assert exc_info.value.code == "E_CONSTRAINT"

    def test_resblock_resolve_group_norm_divisible_succeeds(self):
        """Plan resolution and build succeed when group_norm divides evenly."""
        plan = resolve("resblock(16, norm='group_norm', groups=8)",
                       input_shape=("B", 16, 8, 8), output_shape=("B", 16, 8, 8))
        model = build(plan, device="cpu")
        x = torch.randn(2, 16, 8, 8)
        out = model(x=x)["output"]
        assert out.shape == (2, 16, 8, 8)

    def test_resblock_extra_repr_formatting(self):
        """extra_repr correctly includes groups only for group_norm."""
        rb_gn = ResBlock(out_channels=16, in_channels=16, stride=2, norm="group_norm", groups=8)
        assert rb_gn.extra_repr() == "stride=2, norm='group_norm', groups=8"

        rb_bn = ResBlock(out_channels=16, in_channels=16, stride=1, norm="batch_norm", groups=8)
        assert rb_bn.extra_repr() == "stride=1, norm='batch_norm'"

        rb_none = ResBlock(out_channels=16, in_channels=16, stride=1, norm="none", groups=8)
        assert rb_none.extra_repr() == "stride=1, norm='none'"

    def test_resblock_reference_parity_across_configurations(self):
        """ResBlock forward matches reference= across norm options and strides."""
        reg = Registry.builtins()
        spec = reg.get("resblock")
        for norm in ("batch_norm", "group_norm", "none"):
            for stride in (1, 2):
                block = ResBlock(out_channels=16, in_channels=8, stride=stride, norm=norm, groups=4)
                block.eval()
                oracle = spec.reference(block)
                x = torch.randn(2, 8, 8, 8)
                out_mod = block(x)
                out_ref = oracle(x)
                torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)


# ==============================================================================
# 4. GridSample Validation and Layer Representations (L-07, L-08)
# ==============================================================================

class TestGridSampleValidationAndLayerRepr:
    """Empirical verification of GridSample argument gating and spatial representations."""

    def test_grid_sample_arg_positional_is_false(self):
        """GridSample optional arguments must be keyword-only in Arg spec."""
        reg = Registry.builtins()
        op = reg.get("grid_sample")
        for opt in ("mode", "padding_mode", "align_corners"):
            assert op.args[opt].positional is False, f"{opt} must have positional=False"

    @pytest.mark.parametrize("positional_call", [
        "grid_sample(x, g, 'bilinear')",
        "grid_sample(x, g, 'nearest')",
        "grid_sample(x, g, 'nearest', 'border')",
        "grid_sample(x, g, 'nearest', 'border', True)",
    ])
    def test_grid_sample_positional_arguments_rejected_in_plan(self, positional_call):
        """Plan resolution rejects positional optional arguments with E_ARGUMENT."""
        source = f"g = coordinate_grid(x, 8, 8)\n{positional_call}"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 3, 4, 4), output_shape=("B", 3, 8, 8))
        assert exc_info.value.code == "E_ARGUMENT"
        assert "grid_sample accepts at most 0 author positional arguments" in str(exc_info.value)

    @pytest.mark.parametrize("kw_args", [
        "mode='bilinear'",
        "mode='nearest', padding_mode='border'",
        "mode='nearest', padding_mode='reflection', align_corners=True",
    ])
    def test_grid_sample_keyword_arguments_succeed_in_plan(self, kw_args):
        """Plan resolution succeeds with keyword optional arguments."""
        source = f"g = coordinate_grid(x, 8, 8)\ngrid_sample(x, g, {kw_args})"
        plan = resolve(source, input_shape=("B", 3, 4, 4), output_shape=("B", 3, 8, 8))
        model = build(plan, device="cpu")
        x = torch.randn(2, 3, 4, 4)
        out = model(x=x)["output"]
        assert out.shape == (2, 3, 8, 8)

    def test_spatial_and_layer_extra_repr_present_and_valid(self):
        """All spatial operators and equalized layers provide accurate extra_repr()."""
        # CoordinateGrid
        cg = CoordinateGrid(height=8, width=6)
        assert cg.extra_repr() == "height=8, width=6"
        cg_spatial = CoordinateGrid(spatial_shape=(12, 10))
        assert cg_spatial.extra_repr() == "height=12, width=10"

        # FourierFeatures
        ff = FourierFeatures(num_frequencies=16, scale=8.0, D=2)
        assert ff.extra_repr() == "num_frequencies=16, scale=8.0"

        # GridSample
        gs = GridSample(mode="nearest", padding_mode="border", align_corners=True)
        assert gs.extra_repr() == "mode='nearest', padding_mode='border', align_corners=True"

        # EqualLinear
        lin = EqualLinear(16, 8, equalized=True)
        assert "equalized=True" in lin.extra_repr()

        # Attention
        att = Attention(heads=2, causal=False, dropout=0.0, bias=True, equalized=True,
                        qkv_bias=True, out_bias=True, rope=False, relative_position_bias=False,
                        spatial_shape=(0, 0), D=16)
        assert "equalized=True" in att.extra_repr()
