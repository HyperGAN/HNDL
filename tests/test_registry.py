import pytest
from torch import nn

from hndl import Arg, Registry, operator
from hndl.errors import HNDLError
from hndl.registry import normalize_arguments


def silu(registry, alias="silu", identity="example.silu"):
    @registry.operator(alias, identity=identity, summary="SiLU activation.", shape="x[B, ...] -> out[B, ...]")
    class SiLU(nn.SiLU):
        pass
    return SiLU


def test_registries_are_independent_and_duplicates_fail():
    a, b = Registry.builtins(), Registry.builtins()
    cls = silu(a)
    entry = cls.__hndl_operator__
    assert a.get("silu") is a.by_identity("example.silu@1") is entry
    assert entry.module is cls
    assert "silu" not in b.aliases
    with pytest.raises(HNDLError, match="Duplicate"):
        silu(a)
    with pytest.raises(HNDLError, match="Duplicate"):
        silu(a, alias="other")
    b.add(cls)
    assert b.get("silu") is entry
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        Registry.builtins().add(nn.SiLU)


def test_builtins_are_discovered_once_and_carry_declarations():
    registry = Registry.builtins()
    assert set(registry.aliases) >= {"linear", "conv", "deconv", "reshape", "flatten", "relu", "leaky_relu",
                                     "tanh", "group_norm", "add", "split", "concat", "adaptive_norm"}
    assert registry.get("conv").identity == "conv2d"
    assert registry.get("deconv").key == "conv_transpose2d@1"
    for spec in registry.operators:
        assert spec.summary and spec.doc and spec.examples, spec.alias
        assert all(arg.help for arg in spec.args.values()), spec.alias
    assert Registry.builtins().get("linear") is registry.get("linear")


@pytest.mark.parametrize("alias", ["x", "out", "import", "__import__", "bad-name", "_private"])
def test_reserved_aliases_fail(alias):
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        silu(Registry.builtins(), alias=alias, identity="example.custom")


def test_up2_has_exact_requirements_and_no_width_default():
    op = Registry.builtins().get("deconv")
    args = normalize_arguments(op, (), {}, policy="up2")
    assert args["kernel_size"] == (4, 4)
    assert args["stride"] == (2, 2)
    assert args["padding"] == (1, 1)
    assert "out_channels" not in args
    with pytest.raises(HNDLError, match="E_POLICY_CONFLICT"):
        normalize_arguments(op, (64,), {"stride": 1}, policy="up2")
    with pytest.raises(HNDLError, match="E_POLICY_CONFLICT"):
        normalize_arguments(Registry.builtins().get("conv"), (), {"kernel_size": 3}, policy="up2")


@pytest.mark.parametrize("args", [({"out_features": True}), ({"out_features": None}),
                                   ({"out_features": -1}), ({"bias": 1}), ({"unknown": 3})])
def test_literal_schema_rejects_non_dimensions_and_unknown_fields(args):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        normalize_arguments(Registry.builtins().get("linear"), (), args)


def test_required_kernel_and_duplicate_arguments_fail():
    r = Registry.builtins()
    with pytest.raises(HNDLError, match="requires kernel_size"):
        normalize_arguments(r.get("conv"), (3,), {})
    with pytest.raises(HNDLError, match="positionally"):
        normalize_arguments(r.get("linear"), (64,), {"out_features": 32})
    with pytest.raises(HNDLError, match="at most"):
        normalize_arguments(r.get("linear"), (64, 32), {})


def test_declarations_validate_signatures_and_arguments():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    with pytest.raises(HNDLError, match="does not accept"):
        operator("gain", summary="Gain.", shape="x[B, ...] -> out[B, ...]",
                 args={"gain": Arg(float, 1.0, help="Multiplier.")})(Plain)

    class Missing(nn.Module):
        def __init__(self, gain, other):
            super().__init__()

        def forward(self, x):
            return x

    with pytest.raises(HNDLError, match="neither a declared argument"):
        operator("gain", summary="Gain.", shape="x[B, ...] -> out[B, ...]",
                 args={"gain": Arg(float, 1.0, help="Multiplier.")})(Missing)
    with pytest.raises(HNDLError, match="help"):
        Arg(float, 1.0)
    with pytest.raises(HNDLError, match="summary"):
        operator("plain", summary="", shape="x[B, ...] -> out[B, ...]")(nn.Identity)
    with pytest.raises(HNDLError, match="forward"):
        operator("plain", summary="Not a module.", shape="x[B, ...] -> out[B, ...]")(object)
