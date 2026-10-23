"""Numerics, inference and error codes for the ``mamba2`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "mamba2")


def test_mamba2_forward_and_backward(device):
    plan = resolve(
        "mamba2(d_model=32, d_state=16, heads=4, head_dim=16)",
        input_shape=("B", 8, 32),
        output_shape=("B", 8, 32),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 32)
    assert out.isfinite().all()

    out.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()


def test_mamba2_infers_d_model_bidirectionally():
    forward = resolve(
        "mamba2(heads=4, head_dim=16)\nlinear(10)",
        input_shape=("B", 8, 32),
        output_shape=("B", 8, 10),
    )
    assert forward.nodes[0].args["d_model"] == 32
    assert forward.nodes[0].output_shapes["out"] == ("B", 8, 32)

    backward = resolve(
        "linear()\nmamba2(heads=4, head_dim=16)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 32),
    )
    assert backward.nodes[0].args["out_features"] == 32
    assert backward.nodes[1].args["d_model"] == 32


def test_mamba2_rejects_divisibility_constraint():
    # heads * head_dim != inner_dim (expand * d_model = 2 * 32 = 64)
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(
            "mamba2(d_model=32, heads=5, head_dim=16)",
            input_shape=("B", 8, 32),
            output_shape=("B", 8, 32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        "mamba2(d_model=32, d_state=16, heads=4, head_dim=16)",
        input_shape=("B", 8, 32),
        output_shape=("B", 8, 32),
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 8, 32, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
