"""Adversarial stress-testing suite for Milestone 3 Gate Iteration 2 Challenger 2.

Empirical verification suite by challenger_v10_m3_r2_2 targeting:
1. Standalone RoPE (rank 3, rank 4, invalid ranks, rotational invariance property, gradcheck/backward)
2. Snake operator (channel dimension constraint validation, mathematical output, autograd)
3. Operator canonical identities (Registry.by_identity across 6 new operators, invalid versions)
4. Network recipes (convnext_v2_tiny, encodec_autoencoder_tiny, clip_dual_encoder_tiny, nerf_coord_field
   with varying batch sizes, PyTorch export, forward/backward)
"""

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, resolve
from hndl.operators.rope import RoPE, _reference as rope_ref
from hndl.testing import available_devices
from hndl.torch import GraphModule, build
from tests._network_examples import load, random_inputs

DEVICES = available_devices()


# ==============================================================================
# TARGET 1: Standalone RoPE
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
@pytest.mark.parametrize("shape", [(1, 16, 64), (4, 32, 128), (2, 7, 32)])
def test_rope_rank3_resolve_and_build(device, heads, shape):
    """Stress test rank-3 [B, T, D] with hndl.resolve and hndl.build."""
    B, T, D = shape
    if D % heads != 0 or (D // heads) % 2 != 0:
        return
    plan = resolve(f"rope(heads={heads})", input_shape=("B", T, D), output_shape=("B", T, D))
    model = build(plan, device=device)
    assert isinstance(model, GraphModule)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    out = model(x=x)["output"]
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()

    ref = rope_ref(model[plan.nodes[0].id])
    expected = ref(x)
    torch.testing.assert_close(out, expected)

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
@pytest.mark.parametrize("shape", [(1, 4, 16, 32), (3, 8, 32, 64), (2, 1, 7, 16)])
def test_rope_rank4_resolve_and_build(device, heads, shape):
    """Stress test rank-4 [B, H, T, D_h] with hndl.resolve and hndl.build."""
    B, H, T, Dh = shape
    # heads argument in rope: if specified > 1, shape[1] must match heads
    plan = resolve(f"rope(heads={H})", input_shape=("B", H, T, Dh), output_shape=("B", H, T, Dh))
    model = build(plan, device=device)
    assert isinstance(model, GraphModule)
    x = torch.randn(B, H, T, Dh, device=device, requires_grad=True)
    out = model(x=x)["output"]
    assert out.shape == (B, H, T, Dh)
    assert torch.isfinite(out).all()

    ref = rope_ref(model[plan.nodes[0].id])
    expected = ref(x)
    torch.testing.assert_close(out, expected)

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("invalid_shape,expected_code", [
    (("B",), "E_SCHEMA"),                # rank 1: rejected by global supported ranks schema (must be 2..5)
    (("B", 16), "E_CONSTRAINT"),         # rank 2: rejected by rope relation (requires rank 3 or 4)
    (("B", 2, 4, 8, 8), "E_CONSTRAINT"), # rank 5: rejected by rope relation (requires rank 3 or 4)
    (("B", 2, 2, 4, 8, 8), "E_SCHEMA"),  # rank 6: rejected by global supported ranks schema (must be 2..5)
])
def test_rope_rejects_invalid_ranks(invalid_shape, expected_code):
    """Adversarially verify invalid ranks raise appropriate error code during resolve."""
    with pytest.raises(HNDLError) as exc_info:
        resolve("rope()", input_shape=invalid_shape, output_shape=invalid_shape)
    assert exc_info.value.code == expected_code


@pytest.mark.parametrize("shape", [
    (1, 8, 32, 64),
    (4, 12, 128, 32),
    (2, 4, 17, 48),
    (1, 1, 5, 16),
])
def test_rope_rotational_property_rank4(shape):
    """Verify rotational property: q and k rotated by pos have inner product dependent only on (pos_q - pos_k).

    For RoPE, <R_m(q), R_n(k)> = <R_{m+delta}(q), R_{n+delta}(k)> for any relative shift delta.
    """
    B, H, T, Dh = shape
    module = RoPE(heads=H).eval()

    torch.manual_seed(12345)
    # Generate two random query and key vectors
    q_vec = torch.randn(B, H, 1, Dh, dtype=torch.float64)
    k_vec = torch.randn(B, H, 1, Dh, dtype=torch.float64)

    # Pick positions m and n such that m + delta < T and n + delta < T
    m, n = 0, 1
    delta = max(1, (T - 2) // 2)
    assert m + delta < T and n + delta < T

    # Embed q_vec at position m and m+delta, k_vec at position n and n+delta
    q_seq = torch.zeros(B, H, T, Dh, dtype=torch.float64)
    q_seq[:, :, m:m+1, :] = q_vec
    q_seq[:, :, m+delta:m+delta+1, :] = q_vec

    k_seq = torch.zeros(B, H, T, Dh, dtype=torch.float64)
    k_seq[:, :, n:n+1, :] = k_vec
    k_seq[:, :, n+delta:n+delta+1, :] = k_vec

    q_rot = module(q_seq)
    k_rot = module(k_seq)

    # Inner product at (m, n)
    q_m = q_rot[:, :, m, :]
    k_n = k_rot[:, :, n, :]
    inner_prod_mn = (q_m * k_n).sum(dim=-1)

    # Inner product at (m+delta, n+delta)
    q_m_delta = q_rot[:, :, m+delta, :]
    k_n_delta = k_rot[:, :, n+delta, :]
    inner_prod_delta = (q_m_delta * k_n_delta).sum(dim=-1)

    # Invariance check: inner products must match within double numerical precision
    torch.testing.assert_close(inner_prod_mn, inner_prod_delta, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [
    (1, 2, 8, 16),
    (2, 4, 6, 8),
])
def test_rope_rank4_backward_gradcheck(device, shape):
    """Verify backward pass on rank 4 computes correct analytical gradients via gradcheck."""
    B, H, T, Dh = shape
    module = RoPE(heads=H).to(device)
    x = torch.randn(B, H, T, Dh, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


# ==============================================================================
# TARGET 2: Snake Operator
# ==============================================================================


@pytest.mark.parametrize("shape,channels,is_cl", [
    # channels_last=False (dim 1 is checked)
    (("B", 32, 64), 16, False),
    (("B", 8, 64, 64), 16, False),
    # channels_last=True (dim -1 is checked)
    (("B", 64, 32), 16, True),
    (("B", 64, 64, 8), 16, True),
    # rank 2
    (("B", 32), 16, False),
])
def test_snake_mismatched_channels_raises_e_constraint(shape, channels, is_cl):
    """Adversarially verify mismatched channels dimension raises PlanBuildError with code E_CONSTRAINT."""
    with pytest.raises(HNDLError) as exc_info:
        resolve(f"snake(channels={channels})", input_shape=shape, output_shape=shape)
    assert exc_info.value.code == "E_CONSTRAINT"


@pytest.mark.parametrize("shape,channels", [
    (("B", 16, 64), 16),      # channels-first
    (("B", 64, 16), 16),      # channels-last
    (("B", 16), 16),          # rank 2
    (("B", 16, 8, 8), 16),    # rank 4 channels-first
    (("B", 8, 8, 16), 16),    # rank 4 channels-last
])
@pytest.mark.parametrize("device", DEVICES)
def test_snake_valid_channels_math_output(device, shape, channels):
    """Verify valid channels resolve and execute with zero errors and correct mathematical output: x + (1/alpha) * sin^2(alpha * x)."""
    plan = resolve(f"snake(channels={channels}, alpha=1.5)", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)

    # Determine concrete shape with B=2
    concrete_shape = tuple(2 if d == "B" else d for d in shape)
    x = torch.randn(*concrete_shape, device=device, requires_grad=True)
    out = model(x=x)["output"]

    assert out.shape == concrete_shape
    assert torch.isfinite(out).all()

    # Pure mathematical verification:
    # y = x + (1 / (alpha + eps)) * sin^2(alpha * x)
    alpha = 1.5
    eps = 1e-9
    expected = x + (1.0 / (alpha + eps)) * (torch.sin(alpha * x) ** 2)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    mod = model[plan.nodes[0].id]
    assert mod.alpha.grad is not None and torch.isfinite(mod.alpha.grad).all()


# ==============================================================================
# TARGET 3: Operator Canonical Identities
# ==============================================================================


@pytest.mark.parametrize("identity,expected_alias", [
    ("ssd@1", "mamba2"),
    ("mmdit_attention@1", "joint_attention"),
    ("dynamic_tanh@1", "dy_tanh"),
    ("stochastic_depth@1", "drop_path"),
    ("where@1", "where"),
    ("snake@1", "snake"),
])
def test_operator_canonical_identities(identity, expected_alias):
    """Test Registry.by_identity across all 6 new operators."""
    registry = Registry.builtins()
    op = registry.by_identity(identity)
    assert op.alias == expected_alias
    assert op.key == identity


@pytest.mark.parametrize("invalid_identity", [
    "ssd@2",
    "ssd@0",
    "ssd@99",
    "mmdit_attention@2",
    "mmdit_attention@v1",
    "dynamic_tanh@2",
    "stochastic_depth@0",
    "where@99",
    "snake@2",
    "nonexistent_operator@1",
])
def test_invalid_identity_versions_raise_e_state_version(invalid_identity):
    """Verify that invalid identity versions raise appropriate HNDLError with E_STATE_VERSION."""
    registry = Registry.builtins()
    with pytest.raises(HNDLError) as exc_info:
        registry.by_identity(invalid_identity)
    assert exc_info.value.code == "E_STATE_VERSION"


# ==============================================================================
# TARGET 4: Network Recipes Stress Testing
# ==============================================================================

M3_RECIPES = [
    "convnext_v2_tiny",
    "encodec_autoencoder_tiny",
    "clip_dual_encoder_tiny",
    "nerf_coord_field",
]


@pytest.mark.parametrize("name", M3_RECIPES)
@pytest.mark.parametrize("batch_size", [1, 2, 5, 8])
@pytest.mark.parametrize("device", DEVICES)
def test_network_recipes_varying_batch_sizes(name, batch_size, device):
    """Stress test all 4 new recipes with varying batch sizes, export to PyTorch, and run forward/backward pass."""
    source, meta, kwargs = load(name)
    plan = resolve(source, **kwargs)

    # Build PyTorch model
    model = build(plan, device=device)
    assert isinstance(model, GraphModule)
    assert isinstance(model, nn.Module)

    # Generate random inputs with specified batch_size
    inputs = random_inputs(plan, device=device, batch_size=batch_size, requires_grad=True)
    outputs = model(**inputs)

    assert set(outputs.keys()) == set(plan.outputs.keys())
    for port_name, entry in plan.outputs.items():
        out_tensor = outputs[port_name]
        expected_shape = (batch_size, *entry["shape"][1:])
        assert tuple(out_tensor.shape) == expected_shape, f"{name} port {port_name} shape mismatch"
        assert torch.isfinite(out_tensor).all(), f"{name} port {port_name} contains non-finite values"

    # Verify backward pass produces finite gradients
    loss = sum(out.float().square().mean() for out in outputs.values())
    loss.backward()

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, f"{name} has no trainable parameters"
    for p in trainable:
        assert p.grad is not None, f"{name} parameter gradient is None"
        assert torch.isfinite(p.grad).all(), f"{name} parameter gradient has NaN/Inf"
