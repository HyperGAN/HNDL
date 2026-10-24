"""Numerics, inference and error codes for the ``joint_attention`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.operators.joint_attention import _reference as joint_attn_ref
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "joint_attention")


def test_joint_attention_forward_and_backward(device):
    plan = resolve(
        "out, ctx_out = joint_attention(x, context, heads=4, head_dim=16)",
        input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32)},
        output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
    )
    model = build(plan, device=device)
    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    ctx = torch.randn(2, 16, 32, device=device, requires_grad=True)
    res = model(x=x, context=ctx)
    out, ctx_out = res["out"], res["ctx_out"]
    assert out.shape == (2, 8, 32)
    assert ctx_out.shape == (2, 16, 32)
    assert out.isfinite().all() and ctx_out.isfinite().all()

    (out.sum() + ctx_out.sum()).backward()
    assert x.grad is not None and ctx.grad is not None


def test_joint_attention_shape_inference():
    plan = resolve(
        "out, ctx_out = joint_attention(x, context, heads=4, head_dim=16)",
        input_shape={"x": ("B", 6, 64), "context": ("B", 12, 64)},
        output_shape={"out": ("B", 6, 64), "ctx_out": ("B", 12, 64)},
    )
    assert plan.nodes[0].output_shapes["out"] == ("B", 6, 64)
    assert plan.nodes[0].output_shapes["ctx_out"] == ("B", 12, 64)


def test_joint_attention_rejects_non_divisible_heads():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(
            "out, ctx_out = joint_attention(x, context, heads=5)",
            input_shape={"x": ("B", 8, 32), "context": ("B", 8, 32)},
            output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 8, 32)},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        "out, ctx_out = joint_attention(x, context, heads=4, head_dim=16)",
        input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32)},
        output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 8, 32, device="cuda:0", dtype=torch_dtype)
    ctx = torch.randn(2, 16, 32, device="cuda:0", dtype=torch_dtype)
    res = model(x=x, context=ctx)
    assert res["out"].dtype == torch_dtype
    assert res["ctx_out"].dtype == torch_dtype


def test_joint_attention_qk_norm_l2_norm(device):
    plan = resolve(
        "out, ctx_out = joint_attention(x, context, heads=4, head_dim=16, qk_norm='l2_norm')",
        input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32)},
        output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
    )
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    ctx = torch.randn(2, 16, 32, device=device, requires_grad=True)
    res = model(x=x, context=ctx)
    out, ctx_out = res["out"], res["ctx_out"]
    assert out.shape == (2, 8, 32)
    assert ctx_out.shape == (2, 16, 32)
    assert out.isfinite().all() and ctx_out.isfinite().all()

    ref = joint_attn_ref(module)
    out_ref, ctx_ref = ref(x, ctx)
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(ctx_out, ctx_ref)

    (out.sum() + ctx_out.sum()).backward()
    assert x.grad is not None and ctx.grad is not None
    assert x.grad.isfinite().all() and ctx.grad.isfinite().all()
