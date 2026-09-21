"""Rank-3 sequences, compute dtypes, and integer input ports."""

import json

import pytest
import torch
from torch import nn

from hndl import Arg, HNDLError, Registry, ResolvedPlan, resolve, resolve_callable, ops
from hndl.torch import build, network


def test_sequences_resolve_through_linear_and_activations():
    plan = resolve("linear(64)\nrelu()\nlinear()", input_shape=("B", 16, 32), output_shape=("B", 16, 8))
    assert [node.output_shapes["out"] for node in plan.nodes] == [("B", 16, 64), ("B", 16, 64), ("B", 16, 8)]
    assert plan.nodes[0].args["in_features"] == 32 and plan.nodes[2].args["out_features"] == 8
    native = resolve_callable(lambda x: (ops.linear(64), ops.relu(), ops.linear())[-1],
                              input_shape=("B", 16, 32), output_shape=("B", 16, 8))
    assert native.semantic_digest == plan.semantic_digest
    model = build(plan, device="cpu")
    x = torch.randn(3, 16, 32)
    assert model(x=x)["output"].shape == (3, 16, 8)
    assert "[B, 16, 64]" in repr(model)


def test_linear_rejects_images_and_reshape_prefix_rank_comes_from_neighbours():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*flatten"):
        resolve("linear(4)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 4, 4))
    plan = resolve("reshape(4, 8)", input_shape=("B", 64), output_shape=("B", 4, 8, 2))
    assert plan.nodes[0].args["shape"] == (4, 8, 2)
    plan = resolve("reshape(4, 8)", input_shape=("B", 32), output_shape=("B", 4, 8))
    assert plan.nodes[0].args["shape"] == (4, 8)
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("reshape(4, 8)\nrelu()\nflatten()", input_shape=("B", 64), output_shape=("B", 64))
    bare = resolve("reshape()\nlinear()", input_shape=("B", 2, 4, 4), output_shape=("B", 10))
    assert bare.nodes[0].args["shape"] == (32,)
    tokens = resolve("reshape(16)\nlinear(8)", input_shape=("B", 64), output_shape=("B", 16, 8))
    assert tokens.nodes[0].args["shape"] == (16, 4)


def test_split_and_concat_work_on_sequence_axes():
    plan = resolve("a, b = split(4, dim=1)\nconcat(b, a, axis=1)", input_shape=("B", 10, 3), output_shape=("B", 10, 3))
    assert plan.nodes[0].output_shapes["rest"] == ("B", 6, 3)
    model = build(plan, device="cpu")
    x = torch.randn(2, 10, 3)
    torch.testing.assert_close(model(x=x)["output"], torch.cat((x[:, 4:], x[:, :4]), dim=1))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_plans_round_trip_and_build_on_cpu_or_cuda(dtype):
    plan = resolve("linear(4, init={'weight': 0.1})\ntanh()", input_shape=("B", 3), output_shape=("B", 4), dtype=dtype)
    assert plan.dtype == dtype and plan.input_dtype == dtype
    data = json.loads(plan.to_json())
    assert data["dtype"] == dtype and data["input_dtype"] == dtype
    restored = ResolvedPlan.from_json(plan.to_json())
    assert restored.semantic_digest == plan.semantic_digest
    assert plan.semantic_digest != resolve("linear(4, init={'weight': 0.1})\ntanh()", input_shape=("B", 3),
                                           output_shape=("B", 4)).semantic_digest
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build(plan, device=device)
    torch_dtype = getattr(torch, dtype)
    assert all(p.dtype == torch_dtype for p in model.parameters())
    assert torch.equal(model["n0"].weight, torch.full_like(model["n0"].weight, 0.1))
    x = torch.randn(2, 3, device=device, dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (2, 4)
    with pytest.raises(HNDLError, match="E_RUNTIME.*dtype"):
        model(x=x.float())
    assert model.build_receipt["dtype"] == dtype
    assert model.build_receipt["state_bytes"] == (3 * 4 + 4) * 2


@pytest.mark.parametrize("kwargs,code", [
    ({"dtype": "float64"}, "E_SCHEMA"),
    ({"dtype": "int64"}, "E_SCHEMA"),
    ({"input_dtype": "float16"}, "E_SCHEMA"),
    ({"input_dtype": "uint8"}, "E_SCHEMA"),
    ({"input_dtype": "int64"}, "E_DTYPE"),
])
def test_invalid_dtype_combinations_fail_before_construction(kwargs, code):
    with pytest.raises(HNDLError, match=code):
        resolve("linear(4)", input_shape=("B", 3), output_shape=("B", 4), **kwargs)


def lookup_registry():
    registry = Registry.builtins()

    @registry.operator("lookup", identity="tests.lookup", summary="Embed integer ids.",
                       shape="ids[B, T]:int64 -> out[B, T, D]",
                       args={"dim": Arg(int, inferable=True, dim="D", min=1, help="Embedding width."),
                             "count": Arg(int, 16, min=1, positional=False, help="Vocabulary size.")})
    class Lookup(nn.Module):
        def __init__(self, dim, count):
            super().__init__()
            self.table = nn.Embedding(count, dim)

        def forward(self, ids):
            return self.table(ids)

    return registry


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_integer_input_ports_flow_into_floating_compute(dtype):
    registry = lookup_registry()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype == "bfloat16" and device == "cpu":
        pytest.skip("bfloat16 embedding backward is CUDA-only in this suite")
    model = network("lookup(8)\nlinear(5)", input_shape=("B", 7), output_shape=("B", 7, 5), input_dtype="int64",
                    dtype=dtype, registry=registry, device=device)
    assert model.plan.nodes[0].args["dim"] == 8 and model.plan.nodes[1].args["in_features"] == 8
    assert model.plan.input_dtype == "int64" and "input_dtype=int64" in repr(model)
    ids = torch.randint(0, 16, (2, 7), device=device)
    output = model(ids)
    assert output.shape == (2, 7, 5) and output.dtype == getattr(torch, dtype)
    output.float().square().mean().backward()
    assert model["n0"].table.weight.grad is not None
    with pytest.raises(HNDLError, match="E_RUNTIME.*dtype"):
        model(ids.float() if dtype == "float32" else ids.to(getattr(torch, dtype)))
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("relu()\nlookup()", input_shape=("B", 7), output_shape=("B", 7, 5), input_dtype="int64", registry=registry)
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("lookup()", input_shape=("B", 7), output_shape=("B", 7, 5), registry=registry)
    restored = ResolvedPlan.from_json(model.plan.to_json(), registry=registry)
    assert restored.input_dtype == "int64"


def test_omitted_lookup_width_is_solved_or_reported():
    registry = lookup_registry()
    plan = resolve("lookup()\nrelu()", input_shape=("B", 7), output_shape=("B", 7, 5), input_dtype="int64",
                   registry=registry)
    assert plan.nodes[0].args["dim"] == 5
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("lookup()\nlinear(3)", input_shape=("B", 7), output_shape=("B", 7, 3), input_dtype="int64",
                registry=registry)
