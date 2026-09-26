"""The extension API as a package that declares its own operators sees it.

Everything here imports only public names and registers on
``Registry.builtins()``, the way code outside HNDL does: the global ``ops``
namespace inside a capture of a custom registry, the ``hndl.relations``
helpers in a custom ``relation=``, and the ``hndl.testing`` harness.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import hndl
from hndl import Arg, Example, HNDLError, Registry, ops, resolve, resolve_callable, testing
from hndl.relations import (conv_axis, conv_input_range, conv_output, conv_transpose_input,
                            conv_transpose_output, spatial)
from hndl.torch import network_from_callable


def _decimate_relation(s):
    s.rank("x", 3)
    s.rank("out", 3)
    channels = s.shape("x")[1] or s.shape("out")[1]
    s.axis("x", 1, channels)
    s.axis("out", 1, channels)
    # Keeping positions offset, offset + factor, ... is a kernel offset + 1,
    # stride factor window with no padding.
    conv_axis(s, 2, kernel=s.args["offset"] + 1, stride=s.args["factor"])


def host_registry():
    registry = Registry.builtins()

    @registry.operator(
        "decimate",
        identity="host.decimate",
        summary="Keep every factor-th position of a [B, C, L] signal.",
        shape="x -> out",
        relation=_decimate_relation,
        shape_text="L_out = floor((L_in - offset - 1) / factor) + 1, channels unchanged",
        reference=lambda module: lambda x: x[:, :, module.offset::module.factor],
        args={
            "factor": Arg(int, min=1, help="Keep one position in every factor."),
            "offset": Arg(int, 0, min=0, positional=False, since="1.1.0",
                          help="Index of the first kept position."),
        },
        examples=[
            Example("conv1d(8, kernel_size=3, padding=1)\ndecimate(2)", ("B", 4, 16), ("B", 8, 8),
                    "Halve the length after a convolution."),
            Example("decimate(4, offset=1)\nconv1d(6, kernel_size=1)", ("B", 3, 16), ("B", 6, 4),
                    "Keep positions 1, 5, 9 and 13."),
        ],
    )
    class Decimate(nn.Module):
        """``out = x[:, :, offset::factor]``."""

        def __init__(self, factor, offset):
            super().__init__()
            self.factor, self.offset = factor, offset

        def forward(self, x):
            return x[:, :, self.offset::self.factor]

    return registry


REGISTRY = host_registry()
CONTRACT = dict(input_shape=("B", 4, 16), output_shape=("B", 8, 8))


def conv_then_decimate(namespace):
    def author(x):
        namespace.conv1d(8, kernel_size=3, padding=1)
        return namespace.decimate(2)
    return author


# -- the global ops namespace follows the capture's registry -------------------------------


def test_global_ops_resolves_aliases_against_the_capture_registry():
    config = resolve("conv1d(8, kernel_size=3, padding=1)\ndecimate(2)", registry=REGISTRY, **CONTRACT)
    native = resolve_callable(conv_then_decimate(ops), registry=REGISTRY, **CONTRACT)
    bound = resolve_callable(conv_then_decimate(REGISTRY.ops), registry=REGISTRY, **CONTRACT)
    assert native.semantic_digest == bound.semantic_digest == config.semantic_digest
    assert native.nodes[1].op == "host.decimate@1"
    model = network_from_callable(conv_then_decimate(ops), registry=REGISTRY, device="cpu", **CONTRACT)
    assert tuple(model(torch.randn(2, 4, 16)).shape) == (2, 8, 8)


def test_global_and_registry_ops_mix_in_one_capture():
    def author(x):
        ops.conv1d(8, kernel_size=3, padding=1)
        REGISTRY.ops.relu()
        h = ops.decimate(2)
        return REGISTRY.ops.decimate(h, 1)

    plan = resolve_callable(author, registry=REGISTRY, input_shape=("B", 4, 16), output_shape=("B", 8, 8))
    assert [node.op for node in plan.nodes] == ["conv1d@1", "relu@1", "host.decimate@1", "host.decimate@1"]


def test_global_ops_outside_a_capture_checks_the_builtin_catalog():
    with pytest.raises(HNDLError, match="E_OPERATOR"):
        ops.decimate
    linear = ops.linear
    with pytest.raises(HNDLError, match="E_CAPTURE"):
        linear(10)
    # A factory read outside a capture resolves its alias when it is called.
    plan = resolve_callable(lambda x: linear(10), registry=REGISTRY, input_shape=("B", 4), output_shape=("B", 10))
    assert plan.nodes[0].op == "linear@1"


def test_global_ops_has_only_the_capture_registry_aliases():
    registry = Registry()

    @registry.operator("twice", identity="host.twice", summary="Double the input.",
                       shape="x[B, ...] -> out[B, ...]")
    class Twice(nn.Module):
        """``out = 2 * x``."""

        def forward(self, x):
            return 2 * x

    plan = resolve_callable(lambda x: ops.twice(), registry=registry,
                            input_shape=("B", 3), output_shape=("B", 3))
    assert plan.nodes[0].op == "host.twice@1"
    with pytest.raises(HNDLError, match="E_OPERATOR: Unknown operator alias 'linear'"):
        resolve_callable(lambda x: ops.linear(3), registry=registry, input_shape=("B", 3), output_shape=("B", 3))


def test_global_ops_follows_nested_captures():
    def inner(x):
        with pytest.raises(HNDLError, match="E_OPERATOR"):
            ops.decimate(2)
        return ops.relu()

    def outer(x):
        resolve_callable(inner, input_shape=("B", 4, 16), output_shape=("B", 4, 16))
        ops.conv1d(8, kernel_size=3, padding=1)
        return ops.decimate(2)

    plan = resolve_callable(outer, registry=REGISTRY, **CONTRACT)
    assert plan.nodes[1].op == "host.decimate@1"


def test_another_registrys_ops_fails_unless_it_binds_the_same_declaration():
    other = host_registry()
    with pytest.raises(HNDLError, match="E_CAPTURE: decimate was taken from another registry's ops .* "
                                        "declares that identity with a different implementation"):
        resolve_callable(conv_then_decimate(other.ops), registry=REGISTRY, **CONTRACT)
    with pytest.raises(HNDLError, match="E_CAPTURE: decimate was taken from another registry's ops .* "
                                        "does not hold it; pass the registry that declares it"):
        resolve_callable(conv_then_decimate(REGISTRY.ops), **CONTRACT)
    # Built-in declarations are shared by every Registry.builtins(), so a
    # built-in factory from another registry binds exactly what this one does.
    plan = resolve_callable(lambda x: other.ops.linear(10), registry=REGISTRY,
                            input_shape=("B", 4), output_shape=("B", 10))
    assert plan.nodes[0].op == "linear@1"


# -- public relation helpers ---------------------------------------------------------------


@pytest.mark.parametrize("kernel,stride,padding,dilation", [(1, 1, 0, 1), (3, 1, 1, 1), (4, 2, 1, 1),
                                                            (3, 3, 0, 2), (5, 2, 2, 1)])
def test_convolution_arithmetic_matches_torch_and_inverts(kernel, stride, padding, dilation):
    conv = nn.Conv1d(1, 1, kernel, stride=stride, padding=padding, dilation=dilation)
    for padding_out in range(min(stride, dilation) if stride > 1 or dilation > 1 else 1):
        deconv = nn.ConvTranspose1d(1, 1, kernel, stride=stride, padding=padding, dilation=dilation,
                                    output_padding=padding_out)
        for extent in range(1, 24):
            out = deconv(torch.zeros(1, 1, extent)).shape[-1]
            assert conv_transpose_output(extent, kernel, stride, padding, dilation, padding_out) == out
            assert conv_transpose_input(out, kernel, stride, padding, dilation, padding_out) == extent
    for extent in range(1, 24):
        expected = conv_output(extent, kernel, stride, padding, dilation)
        if expected < 1:
            continue
        assert conv(torch.zeros(1, 1, extent)).shape[-1] == expected
        lower, upper = conv_input_range(expected, kernel, stride, padding, dilation)
        assert lower <= extent <= upper
        assert all(conv_output(i, kernel, stride, padding, dilation) == expected for i in range(lower, upper + 1))


def test_relation_helpers_infer_both_directions_through_a_host_operator():
    plan = resolve("decimate(2)\nconv1d(8, kernel_size=1)", registry=REGISTRY,
                   input_shape=("B", 4, 16), output_shape=("B", 8, 8))
    assert plan.nodes[0].output_shapes["out"] == ("B", 4, 8)
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("decimate(2)", registry=REGISTRY, input_shape=("B", 4, 16), output_shape=("B", 4, 9))
    # Backward: the output length fixes the linear width through the reshape
    # when the window is one position, and stays an unresolved interval
    # (15 or 16 positions) when it is two.
    contract = dict(input_shape=("B", 10), output_shape=("B", 4, 8))
    plan = resolve("linear()\nreshape(4)\ndecimate(1)", registry=REGISTRY, **contract)
    assert plan.nodes[0].args["out_features"] == 32
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("linear()\nreshape(4)\ndecimate(2)", registry=REGISTRY, **contract)
    plan = resolve("linear(64)\nreshape(4)\ndecimate(2)", registry=REGISTRY, **contract)
    assert plan.nodes[2].input_shapes["x"] == ("B", 4, 16)


def test_relations_module_is_what_builtins_use_and_the_old_path_still_imports():
    from hndl.operators import _relations
    assert _relations.spatial is spatial and _relations.conv_output is conv_output
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        spatial("conv3d")


# -- public test harness -------------------------------------------------------------------


def test_host_operator_passes_the_public_harness():
    testing.check_operator(REGISTRY, "decimate")


@pytest.mark.parametrize("spec,example", testing.example_params(REGISTRY, ["decimate"]))
@pytest.mark.parametrize("device", testing.available_devices())
def test_harness_checks_run_one_example_at_a_time(spec, example, device):
    testing.check_declaration(REGISTRY, spec)
    plan = testing.check_round_trip(REGISTRY, spec, example)
    assert any(node.op == "host.decimate@1" for node in plan.nodes)
    testing.check_build_and_run(REGISTRY, spec, example, device=device)
    assert testing.check_reference(REGISTRY, spec, example, device=device)


def test_example_params_ids_and_filtering():
    params = testing.example_params(REGISTRY, ["decimate"])
    assert [param.id for param in params] == ["decimate-0", "decimate-1"]
    assert [param.id for param in testing.operator_params(REGISTRY, ["decimate", "relu"])] == ["decimate", "relu"]
    cases = testing.example_cases(REGISTRY)
    assert len(cases) == sum(len(spec.examples) for spec in REGISTRY.operators)
    assert all(not example.network for _, example in testing.example_cases(REGISTRY, network=False))


def broken_registry(**overrides):
    registry = Registry.builtins()
    declaration = dict(identity="host.shift", summary="Add one.", shape="x[B, ...] -> out[B, ...]",
                       reference=lambda module: lambda x: x + 1,
                       examples=[Example("linear(8)\nshift()", ("B", 4), ("B", 8))])
    declaration.update(overrides)

    @registry.operator("shift", **declaration)
    class Shift(nn.Module):
        """``out = x + 1``."""

        def forward(self, x):
            return x + 1

    return registry


@pytest.mark.parametrize("overrides,message", [
    ({"reference": lambda module: lambda x: x + 2}, "shift example 0 on cpu, reference: .*differs from the reference"),
    ({"examples": [Example("linear(8)", ("B", 4), ("B", 8))]}, "example does not use shift"),
    ({"examples": []}, "shift needs at least one Example"),
])
def test_harness_reports_a_broken_operator(overrides, message):
    with pytest.raises(AssertionError, match=message):
        testing.check_operator(broken_registry(**overrides), "shift", devices=["cpu"])


def test_harness_rejects_a_declaration_from_another_registry():
    with pytest.raises(AssertionError, match="different declaration"):
        testing.check_round_trip(REGISTRY, host_registry().get("decimate"), REGISTRY.get("decimate").examples[0])


def test_importing_hndl_and_its_harness_does_not_import_pytest():
    src = str(Path(hndl.__file__).resolve().parents[1])
    code = ("import sys, hndl, hndl.testing, hndl.relations\n"
            "assert 'pytest' not in sys.modules and '_pytest' not in sys.modules")
    env = dict(os.environ, PYTHONPATH=src)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
