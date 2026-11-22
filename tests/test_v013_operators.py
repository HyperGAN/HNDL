"""Comprehensive regression, autocast, gradient, and parity tests for v0.13 operator fixes and optimizations."""

import copy

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.deployment import _implementation
from hndl.operators._equalized import EqualLinear
from hndl.operators._packing import prepare_packed
from hndl.operators.deconv1d import ConvTranspose1d
from hndl.operators.drop_path import DropPath
from hndl.operators.grn import GRN, _reference as grn_reference
from hndl.operators.hopfield import Hopfield, _reference as hopfield_reference
from hndl.operators.moe import MixtureOfExperts, _RoutedLinear, _reference as moe_reference
from hndl.operators.rms_norm import RMSNorm, _reference as rms_norm_reference
from hndl.operators.sinusoidal_embed import SinusoidalEmbed, _reference as sinusoidal_reference
from hndl.operators.snake import Snake, _reference as snake_reference
from hndl.operators.spatial_attention import SpatialAttention, _reference as spatial_attn_reference
from hndl.torch import build, network

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


# ==============================================================================
# 1. MoE: autocast correctness, dispatch parity, hooks, scaling, _RoutedLinear
# ==============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA autocast tests")
@pytest.mark.parametrize("dispatch", ["auto", "padded", "sparse"])
@pytest.mark.parametrize("expert_kind", ["mlp", "swiglu"])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
def test_moe_cuda_autocast_bf16_forward_and_backward(dispatch, expert_kind, input_dtype, weight_dtype, packed):
    torch.manual_seed(0)
    module = MixtureOfExperts(
        experts=4, hidden=32, top_k=2, activation="silu", D=16,
        expert_kind=expert_kind, bias=True,
    ).to(device="cuda:0", dtype=weight_dtype)
    module.configure_dispatch(dispatch)
    if packed:
        prepare_packed(module)

    x = torch.randn(2, 6, 16, device="cuda:0", dtype=input_dtype, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(x)
    assert out.shape == (2, 6, 16)
    assert out.dtype == torch.bfloat16
    assert out.isfinite().all()

    out.float().square().mean().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert module.router.weight.grad is not None and module.router.weight.grad.isfinite().all()
    for p in module.parameters():
        if p.grad is not None:
            assert p.grad.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA autocast tests")
@pytest.mark.parametrize("dispatch", ["auto", "padded", "sparse"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_moe_cuda_autocast_fp16_forward_and_backward(dispatch, weight_dtype):
    torch.manual_seed(1)
    module = MixtureOfExperts(
        experts=4, hidden=32, top_k=2, activation="gelu", D=16,
        expert_kind="swiglu", bias=False,
    ).to(device="cuda:0", dtype=weight_dtype)
    module.configure_dispatch(dispatch)
    prepare_packed(module)

    x = torch.randn(2, 5, 16, device="cuda:0", dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = module(x)
    assert out.shape == (2, 5, 16)
    assert out.dtype == torch.float16
    assert out.isfinite().all()
    out.float().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("expert_kind", ["mlp", "swiglu"])
def test_moe_cuda_autocast_dispatch_parity(expert_kind):
    torch.manual_seed(7)
    base = MixtureOfExperts(4, 32, 2, "silu", D=16, expert_kind=expert_kind, bias=True).cuda()
    x = torch.randn(3, 8, 16, device="cuda", dtype=torch.float32)

    outputs = {}
    for dispatch in ("auto", "padded", "sparse"):
        mod = prepare_packed(copy.deepcopy(base)).configure_dispatch(dispatch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs[dispatch] = mod(x)
    torch.testing.assert_close(outputs["auto"], outputs["padded"], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(outputs["sparse"], outputs["padded"], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dispatch", ["auto", "padded", "sparse"])
def test_moe_cpu_autocast_bf16_forward_and_backward(dispatch):
    torch.manual_seed(2)
    module = prepare_packed(
        MixtureOfExperts(4, 32, 2, "silu", D=16, expert_kind="swiglu", bias=True)
    ).configure_dispatch(dispatch)
    x = torch.randn(2, 5, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(x)
    assert out.shape == (2, 5, 16)
    assert out.dtype == torch.bfloat16
    assert out.isfinite().all()
    out.float().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_routed_linear_autocast_and_large_tile():
    torch.manual_seed(3)
    inputs = torch.randn(8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    weights = torch.randn(4, 32, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(4, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    ids = torch.tensor([0, 2, 1, 3, 2, 0, 1, 3], device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = _RoutedLinear.apply(inputs, weights, ids, bias)
    assert out.dtype == torch.bfloat16 and out.isfinite().all()
    out.float().sum().backward()
    assert inputs.grad is not None and inputs.grad.isfinite().all()
    assert weights.grad is not None and weights.grad.isfinite().all()
    assert bias.grad is not None and bias.grad.isfinite().all()

    # Large matrix (1024 x 2048 bf16 = 4MB) should not collapse tile to 1.
    big_x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    big_w = torch.randn(2, 2048, 1024, device="cuda", dtype=torch.bfloat16)
    big_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device="cuda")
    big_out = _RoutedLinear.apply(big_x, big_w, big_ids, None)
    assert big_out.shape == (8, 2048) and big_out.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dispatch", ["auto", "padded", "sparse"])
@pytest.mark.parametrize("expert_kind", ["mlp", "swiglu"])
def test_moe_empty_experts_zero_gradients_with_nonzero_bias(device, dispatch, expert_kind):
    torch.manual_seed(11)
    module = MixtureOfExperts(4, 16, 1, "gelu", D=8, expert_kind=expert_kind, bias=True).to(device)
    with torch.no_grad():
        module.router.weight.zero_()
        module.router.weight[1].fill_(10.0)
        for expert in module.experts:
            for p in expert.parameters():
                p.normal_(std=0.2)
    ref = copy.deepcopy(module)
    prepare_packed(module).configure_dispatch(dispatch)

    x = torch.rand(6, 8, device=device).add_(1.0).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    actual = module(x)
    expected = moe_reference(ref)(x_ref)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=2e-4, atol=2e-4)
    for idx, (e_mod, e_ref) in enumerate(zip(module.experts, ref.experts, strict=True)):
        for p, q in zip(e_mod.parameters(), e_ref.parameters(), strict=True):
            if idx != 1:
                assert p.grad is None or p.grad.abs().max().item() == 0.0
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_moe_fallback_skips_empty_unhooked_but_runs_hooked_experts(device):
    module = MixtureOfExperts(4, 16, 1, "silu", D=8, routed_scaling_factor=1.5).to(device)
    with torch.no_grad():
        module.router.weight.zero_()
        module.router.weight[0].fill_(10.0)
    hook_calls = []
    # Attach hook to expert 2 (which receives 0 routes) — forces fallback path and must still invoke hook.
    module.experts[2].register_forward_hook(lambda m, args, out: hook_calls.append(args[0].shape[0]))
    x = torch.ones(4, 8, device=device)
    actual = module(x)
    expected = moe_reference(module)(x)
    assert hook_calls == [0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dispatch", ["auto", "padded", "sparse"])
def test_moe_packed_version_guard_detects_inplace_mutation(dispatch):
    module = prepare_packed(
        MixtureOfExperts(3, 16, 2, "silu", D=16, expert_kind="swiglu", bias=False)
    ).configure_dispatch(dispatch)
    x = torch.randn(4, 16, requires_grad=True)
    loss = module(x).sum()
    with torch.no_grad():
        module.experts[0].gate.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("device", DEVICES)
def test_moe_sparse_torch_compile_fullgraph(device):
    module = prepare_packed(
        MixtureOfExperts(4, 16, 2, "silu", D=8, expert_kind="swiglu").to(device)
    ).configure_dispatch("sparse")
    x = torch.randn(3, 8, device=device)
    eager = module(x)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), eager, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_empty_batch_under_autocast():
    module = MixtureOfExperts(4, 32, 2, "silu", D=16).cuda()
    x = torch.empty(0, 4, 16, device="cuda", dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(x)
    assert out.shape == (0, 4, 16) and out.dtype == torch.bfloat16


# ==============================================================================
# 2. EqualLinear: operand-scaling parity across both branches & dtypes
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [2, 64])  # batch=2 scales x; batch=64 scales weight
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_equal_linear_both_branches_forward_and_backward_parity(device, batch, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 linear on CPU is slow/unsupported")
    torch.manual_seed(5)
    layer = EqualLinear(16, 24, bias=True, equalized=True).to(device=device, dtype=dtype)
    x = torch.randn(batch, 16, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = layer.weight.detach().clone().requires_grad_()
    b_ref = layer.bias.detach().clone().requires_grad_()

    actual = layer(x)
    expected = F.linear(x_ref, w_ref * layer._scale, b_ref)
    tol = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(actual, expected, **tol)

    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(layer.weight.grad, w_ref.grad, **tol)
    torch.testing.assert_close(layer.bias.grad, b_ref.grad, **tol)


# ==============================================================================
# 3. RMSNorm zero_centered=True fused weight parity
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rms_norm_zero_centered_parity(device, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 rms_norm qualified on CUDA")
    torch.manual_seed(6)
    norm = RMSNorm(1e-6, True, D=32, zero_centered=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_(std=0.25)
    ref = rms_norm_reference(copy.deepcopy(norm))
    x = torch.randn(4, 12, 32, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    actual = norm(x)
    expected = ref(x_ref)
    tol = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(actual, expected, **tol)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)


# ==============================================================================
# 4. GRN addcmul/square parity
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_grn_optimized_forward_and_backward_parity(device, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 grn qualified on CUDA")
    torch.manual_seed(8)
    mod = GRN(eps=1e-5, C=16).to(device=device, dtype=dtype)
    with torch.no_grad():
        mod.gamma.normal_(std=0.3)
        mod.beta.normal_(std=0.3)
    ref_mod = copy.deepcopy(mod)
    ref = grn_reference(ref_mod)
    x = torch.randn(3, 16, 8, 8, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    actual = mod(x)
    expected = ref(x_ref)
    tol = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(actual, expected, **tol)

    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(mod.gamma.grad, ref_mod.gamma.grad, **tol)
    torch.testing.assert_close(mod.beta.grad, ref_mod.beta.grad, **tol)


# ==============================================================================
# 5. Hopfield SDPA optimization parity (CPU + CUDA, rank-2 & rank-3, autocast)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("patterns", [16, 128, 256])
@pytest.mark.parametrize("shape", [(4, 32), (2, 10, 32)])
@pytest.mark.parametrize("steps", [1, 2])
def test_hopfield_sdpa_forward_and_backward_parity(device, patterns, shape, steps):
    torch.manual_seed(9)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    mod = Hopfield(patterns, beta=1.5, steps=steps, normalize=True, D=32).to(device=device, dtype=dtype)
    ref_mod = copy.deepcopy(mod)
    ref = hopfield_reference(ref_mod)
    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    actual = mod(x)
    expected = ref(x_ref)
    tol = {"rtol": 1e-4, "atol": 1e-4} if dtype == torch.float32 else {"rtol": 3e-2, "atol": 3e-2}
    torch.testing.assert_close(actual, expected, **tol)

    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(mod.stored.grad, ref_mod.stored.grad, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hopfield_cuda_autocast_large_patterns():
    torch.manual_seed(10)
    mod = Hopfield(256, beta=2.0, steps=2, normalize=True, D=64).cuda()
    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mod(x)
    assert out.dtype == torch.bfloat16 and out.isfinite().all()
    out.float().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert mod.stored.grad is not None and mod.stored.grad.isfinite().all()


# ==============================================================================
# 6. SpatialAttention SDPA optimization parity
# ==============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels,reduction,extent", [(64, 8, 16), (256, 8, 16), (64, 1, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_spatial_attention_sdpa_parity_on_cuda(channels, reduction, extent, dtype):
    torch.manual_seed(12)
    mod = SpatialAttention(reduction=reduction, C=channels).to(device="cuda:0", dtype=dtype)
    with torch.no_grad():
        mod.gamma.fill_(0.7)
    ref_mod = copy.deepcopy(mod)
    ref = spatial_attn_reference(ref_mod)
    x = torch.randn(2, channels, extent, extent, device="cuda:0", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    actual = mod(x)
    expected = ref(x_ref)
    tol = {"rtol": 1e-3, "atol": 1e-4} if dtype == torch.float32 else {"rtol": 3e-2, "atol": 3e-2}
    torch.testing.assert_close(actual, expected, **tol)

    w = torch.randn_like(actual)
    (actual * w).float().mean().backward()
    (expected * w).float().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    for (n1, p1), (_, p2) in zip(mod.named_parameters(), ref_mod.named_parameters(), strict=True):
        torch.testing.assert_close(p1.grad, p2.grad, **tol, msg=f"grad mismatch for {n1}")


# ==============================================================================
# 7. SinusoidalEmbed frequency cache & deployment stability
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
def test_sinusoidal_embed_caches_freqs_and_clears_on_apply(device, monkeypatch):
    dtype = torch.bfloat16
    mod = SinusoidalEmbed(64).to(device=device, dtype=dtype)
    x = torch.randn(8, 1, device=device, dtype=dtype)

    # First call populates cache; second call must not call torch.exp again.
    out1 = mod(x)
    key = (x.device, torch.float32)
    assert key in mod._freq_cache
    cached_tensor = mod._freq_cache[key]

    orig_exp = torch.exp

    def fail_exp(*args, **kwargs):
        raise AssertionError("SinusoidalEmbed recomputed exp() instead of using _freq_cache")

    monkeypatch.setattr(torch, "exp", fail_exp)
    out2 = mod(x)
    assert mod._freq_cache[key] is cached_tensor
    torch.testing.assert_close(out1, out2)
    monkeypatch.setattr(torch, "exp", orig_exp)

    # Parity vs reference
    torch.testing.assert_close(out1, sinusoidal_reference(mod)(x))

    # _apply clears the cache (e.g. .float() or .to(cpu))
    mod.float()
    assert mod._freq_cache == {}


def test_sinusoidal_embed_cache_deployment_fingerprint_stable():
    model = network("sinusoidal_embed(32)", input_shape=("B", 1), output_shape=("B", 32),
                    dtype="bfloat16", device="cpu").eval()
    x = torch.randn(4, 1, dtype=torch.bfloat16)
    with torch.no_grad():
        model(x)
    fp1 = _implementation(model)
    with torch.no_grad():
        model(x)
    fp2 = _implementation(model)
    assert fp1 == fp2


# ==============================================================================
# 8. Operator correctness fixes: dy_tanh, drop_path, snake, deconv1d, resblock, matmul
# ==============================================================================


def test_dy_tanh_no_rank3_guess_in_multihop_rank2_and_rank4_graphs():
    # Rank-2 graph where dy_tanh sits after reshape (unknown rank on first pass if guessed)
    plan2 = resolve(
        "h = reshape(x, 16)\ny = dy_tanh(h)\nlinear(y, 8)",
        input_shape=("B", 4, 4),
        output_shape=("B", 8),
    )
    assert plan2.nodes[1].output_shapes["out"] == ("B", 16)
    assert plan2.nodes[1].args["dim"] == 16

    # Rank-4 graph where dy_tanh sits in the middle
    plan4 = resolve(
        "conv(8, kernel_size=1)\ndy_tanh()\nconv(4, kernel_size=1)",
        input_shape=("B", 3, 6, 6),
        output_shape=("B", 4, 6, 6),
    )
    assert plan4.nodes[1].output_shapes["out"] == ("B", 8, 6, 6)
    assert plan4.nodes[1].args["dim"] == 6


def test_drop_path_no_rank3_guess_and_small_prob_in_bf16():
    plan4 = resolve(
        "conv(8, kernel_size=1)\ndrop_path(0.1)\nconv(4, kernel_size=1)",
        input_shape=("B", 3, 6, 6),
        output_shape=("B", 4, 6, 6),
    )
    assert plan4.nodes[1].output_shapes["out"] == ("B", 8, 6, 6)

    # Small drop_prob=0.005 in bfloat16: 1.0 - 0.005 = 0.995 rounds to 1.0 in bf16,
    # so sampling in float32 is required to drop ~0.5% of samples instead of 0%.
    torch.manual_seed(42)
    dp = DropPath(drop_prob=0.005, scale_by_keep=True).train()
    x = torch.ones(20_000, 4, dtype=torch.bfloat16)
    out = dp(x)
    dropped_frac = (out[:, 0] == 0).float().mean().item()
    assert 0.002 < dropped_frac < 0.01, f"Expected ~0.005 drop rate in bf16, got {dropped_frac}"


def test_snake_channel_inference_multihop_and_axis_disambiguation():
    # Rank-4 channels=0 infers shape[1]
    plan4 = resolve("conv(12, kernel_size=1)\nsnake()\nconv(6, kernel_size=1)",
                    input_shape=("B", 3, 8, 8), output_shape=("B", 6, 8, 8))
    assert plan4.nodes[1].args["channels"] == 12
    m4 = build(plan4, device="cpu")
    assert m4["n1"].alpha.shape == (12,)

    # Rank-3 channels=0 infers shape[-1]
    plan3 = resolve("snake()", input_shape=("B", 8, 16), output_shape=("B", 8, 16))
    assert plan3.nodes[0].args["channels"] == 16
    m3 = build(plan3, device="cpu")
    assert m3["n0"].alpha.shape == (16,)

    # Rank-2 channels=0 infers shape[-1]
    plan2 = resolve("snake()", input_shape=("B", 10), output_shape=("B", 10))
    assert plan2.nodes[0].args["channels"] == 10

    # Rank-3 T == D disambiguation: must apply alpha along last axis (-1)
    s3 = Snake(channels=4)
    with torch.no_grad():
        s3.alpha.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x3 = torch.ones(2, 4, 4)
    y3 = s3(x3)
    # Variation should be along dim -1 (columns), identical across dim 1 (rows)
    torch.testing.assert_close(y3[:, 0, :], y3[:, 1, :])
    assert not torch.allclose(y3[:, :, 0], y3[:, :, 1])
    torch.testing.assert_close(y3, snake_reference(s3)(x3))

    # Rank-4 C == W disambiguation: must apply alpha along axis 1 (C)
    s4 = Snake(channels=4)
    with torch.no_grad():
        s4.alpha.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x4 = torch.ones(2, 4, 3, 4)
    y4 = s4(x4)
    torch.testing.assert_close(y4[:, :, 0, 0], y4[:, :, 0, 1])
    assert not torch.allclose(y4[:, 0, :, :], y4[:, 1, :, :])
    torch.testing.assert_close(y4, snake_reference(s4)(x4))


@pytest.mark.parametrize("device", DEVICES)
def test_snake_float16_alpha_zero_no_nan(device):
    s = Snake(channels=8, alpha=0.0, eps=1e-9).to(device=device, dtype=torch.float16)
    x = torch.randn(2, 6, 8, device=device, dtype=torch.float16, requires_grad=True)
    out = s(x)
    assert out.dtype == torch.float16
    assert out.isfinite().all()
    torch.testing.assert_close(out, x)
    out.float().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert s.alpha.grad is not None and s.alpha.grad.isfinite().all()


def test_deconv1d_causal_validates_kernel_stride_dilation_output_padding():
    # Invalid: dilation*(k-1) + output_padding + 1 = 1*(1-1) + 0 + 1 = 1 < stride=2
    with pytest.raises(HNDLError, match="E_ARGUMENT.*causal"):
        resolve("deconv1d(8, kernel_size=1, stride=2, causal=True)",
                input_shape=("B", 4, 8), output_shape=("B", 8, 16))
    with pytest.raises(HNDLError, match="E_ARGUMENT.*causal"):
        ConvTranspose1d(4, 8, kernel_size=1, stride=2, padding=0, dilation=1,
                        output_padding=0, groups=1, bias=True, spectral_norm=False, causal=True)

    # Invalid: 1*(2-1) + 0 + 1 = 2 < stride=4
    with pytest.raises(HNDLError, match="E_ARGUMENT.*causal"):
        resolve("deconv1d(8, kernel_size=2, stride=4, causal=True)",
                input_shape=("B", 4, 8), output_shape=("B", 8, 32))

    # Valid boundary: 1*(2-1) + 0 + 1 = 2 >= stride=2
    plan = resolve("deconv1d(8, kernel_size=2, stride=2, causal=True)",
                   input_shape=("B", 4, 8), output_shape=("B", 8, 16))
    model = build(plan, device="cpu")
    out = model(x=torch.randn(2, 4, 8))["output"]
    assert out.shape == (2, 8, 16)

    # Valid with dilation: 3*(2-1) + 0 + 1 = 4 >= stride=4
    plan_dil = resolve("deconv1d(8, kernel_size=2, stride=4, dilation=3, causal=True)",
                       input_shape=("B", 4, 5), output_shape=("B", 8, 20))
    model_dil = build(plan_dil, device="cpu")
    out_dil = model_dil(x=torch.randn(2, 4, 5))["output"]
    assert out_dil.shape == (2, 8, 20)


def test_resblock_bottleneck_group_norm_rejects_indivisible_mid_channels_at_resolve_time():
    # out_channels=16 is divisible by groups=8, but mid_channels=16//4=4 is NOT divisible by 8.
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*mid_channels"):
        resolve('resblock(16, norm="group_norm", groups=8, kind="bottleneck", bottleneck_ratio=4)',
                input_shape=("B", 8, 8, 8), output_shape=("B", 16, 8, 8))

    # Valid: out_channels=32, mid_channels=8, groups=8
    plan = resolve('resblock(32, norm="group_norm", groups=8, kind="bottleneck", bottleneck_ratio=4)',
                   input_shape=("B", 8, 8, 8), output_shape=("B", 32, 8, 8))
    model = build(plan, device="cpu")
    assert model(x=torch.randn(2, 8, 8, 8))["output"].shape == (2, 32, 8, 8)


def test_matmul_multihop_rank4_graph_does_not_guess_rank3():
    # Rank-4 flows from output backward through relu -> matmul -> softmax -> matmul -> transpose -> reshape
    source = (
        "a = reshape(x, 2, 4)\n"
        "t = transpose(a, 2, 3)\n"
        "e = matmul(a, t)\n"
        "w = softmax(e, -1)\n"
        "o = matmul(w, a)\n"
        "relu(o)"
    )
    plan = resolve(source, input_shape=("B", 48), output_shape=("B", 2, 4, 6))
    assert plan.nodes[2].output_shapes["out"] == ("B", 2, 4, 4)
    assert plan.nodes[4].output_shapes["out"] == ("B", 2, 4, 6)
    model = build(plan, device="cpu")
    out = model(x=torch.randn(3, 48))["output"]
    assert out.shape == (3, 2, 4, 6)
