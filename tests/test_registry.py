import pytest

from hndl.errors import HNDLError
from hndl.registry import Registry, normalize_arguments, preserves_shape


def test_registries_are_independent_and_custom_identity_is_explicit():
    a, b = Registry.builtins(), Registry.builtins()
    entry = a.register("silu", identity="example.silu", version=1,
                       shape=preserves_shape, max_state_bytes=0)
    assert a.get("silu") is a.by_identity("example.silu@1") is entry
    assert "silu" not in b.aliases
    assert a.backends is not b.backends
    with pytest.raises(HNDLError, match="Duplicate"):
        a.register("silu", identity="example.silu", version=1, shape=preserves_shape, max_state_bytes=0)


@pytest.mark.parametrize("alias", ["x", "out", "import", "__import__", "bad-name"])
def test_reserved_aliases_fail(alias):
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        Registry.builtins().register(alias, identity="example.custom", version=1,
                                     shape=preserves_shape, max_state_bytes=0)


def test_up2_has_exact_requirements_and_no_width_default():
    op = Registry.builtins().get("deconv")
    args = normalize_arguments(op, (), {}, policy="up2")
    assert args["kernel_size"] == (4, 4)
    assert args["stride"] == (2, 2)
    assert args["padding"] == (1, 1)
    assert "out_channels" not in args
    with pytest.raises(HNDLError, match="E_POLICY_CONFLICT"):
        normalize_arguments(op, (64,), {"stride": 1}, policy="up2")


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

