"""The extension API as a package that declares its own operators sees it.

Everything here imports only public names and registers on
``Registry.builtins()``, the way code outside HNDL does: the global ``ops``
namespace inside a capture of a custom registry, the ``hndl.relations``
helpers in a custom ``relation=``, and the ``hndl.testing`` harness.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import hndl
from hndl import Arg, Example, HNDLError, Registry, ops, resolve, resolve_callable, testing
from hndl.relations import (conv_axis, conv_input_range, conv_output, conv_transpose_axis, conv_transpose_input,
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


class Shift(nn.Module):
    """``out = x + 1``."""

    def forward(self, x):
        return x + 1


def _allocating():
    # parameter_counts and the build receipt's first estimate construct on meta.
    return torch.empty(0).device.type != "meta"


class WrongWidth(Shift):
    """Drops half the features."""

    def forward(self, x):
        return x[:, :4]


class NotFinite(Shift):
    """Returns NaN."""

    def forward(self, x):
        return x * float("nan")


class NaNGradient(Shift):
    """A finite output whose input gradient is 0 * inf."""

    def forward(self, x):
        return x + 1 + 0 * (x - x.detach()).sqrt()


class Noisy(Shift):
    """Draws fresh noise in every forward."""

    def forward(self, x):
        return x + torch.rand_like(x)


class HiddenParameter(Shift):
    """Allocates a parameter only when it is not being measured."""

    def __init__(self):
        super().__init__()
        if _allocating():
            self.weight = nn.Parameter(torch.zeros(3))


class HiddenBuffer(Shift):
    """Allocates a buffer only when it is not being measured."""

    def __init__(self):
        super().__init__()
        if _allocating():
            self.register_buffer("state", torch.zeros(3))


class IntegerParameter(Shift):
    """Holds a parameter the plan's dtype cannot cast."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(3, dtype=torch.int64), requires_grad=False)


class Undocumented(nn.Module):
    def forward(self, x):
        return x + 1


def _rank_two(s):
    s.rank("x", 2)
    s.rank("out", 2)
    s.equal("x", "out")


def broken_registry(module=Shift, **overrides):
    registry = Registry.builtins()
    declaration = dict(identity="host.shift", summary="Add one.", shape="x[B, ...] -> out[B, ...]",
                       reference=lambda module: lambda x: x + 1,
                       examples=[Example("linear(8)\nshift()", ("B", 4), ("B", 8))])
    declaration.update(overrides)
    # A fresh subclass per registry, keeping only the class's own docstring.
    cls = type(module.__name__, (module,), {"__doc__": module.__dict__.get("__doc__")})
    registry.operator("shift", **declaration)(cls)
    return registry


@pytest.mark.parametrize("module,overrides,message", [
    (Shift, {"reference": lambda module: lambda x: x + 2},
     "shift example 0 on cpu, reference: n1: output differs from the reference"),
    (Shift, {"reference": lambda module: lambda x: (x + 1, x + 1)},
     "reference: n1: the module returns 1 tensors, the reference 2"),
    (Shift, {"reference": lambda module: lambda x: 2 * x.detach() - x + 1},
     "reference: n1: input gradient differs from the reference"),
    (Shift, {"reference": lambda module: lambda x: x.detach() + 1},
     "reference: n1: the module carries a gradient the reference does not"),
    (Shift, {"examples": [Example("linear(8)", ("B", 4), ("B", 8))]}, "example does not use shift"),
    (Shift, {"examples": []}, "shift needs at least one Example"),
    (Shift, {"summary": "Add one:"}, "shift needs a one-sentence summary"),
    (Undocumented, {}, "shift needs a class docstring"),
    (Shift, {"shape": "x -> out", "relation": _rank_two}, "shift uses a relation function and needs shape_text"),
    (NotFinite, {}, "shift example 0 on cpu in float32: shift: output is not finite"),
    (NaNGradient, {}, "in float32: shift: the input gradient is missing or not finite"),
    (Noisy, {}, "in float32: shift: a rebuild from the same seed differs"),
    (HiddenParameter, {}, r"in float32: shift: parameter_counts \{'n0': 40, 'n1': 0\} disagrees with the built "
                          r"modules \{'n0': 40, 'n1': 3\}"),
    (HiddenBuffer, {}, "in float32: shift: the build receipt reports 160 state bytes, the module holds 172"),
    (IntegerParameter, {}, "in float32: shift: float32 plan built parameters of dtype torch.int64"),
])
def test_harness_reports_a_broken_operator(module, overrides, message):
    with pytest.raises(AssertionError, match=message):
        testing.check_operator(broken_registry(module, **overrides), "shift", devices=["cpu"])


def test_harness_passes_on_the_error_hndl_raises_for_a_wrong_output_shape():
    with pytest.raises(HNDLError, match=r"E_RUNTIME.*shift 'n1' output 'out': expected shape \[B=2, 8\], got \[2, 4\]"):
        testing.check_operator(broken_registry(WrongWidth), "shift", devices=["cpu"])


def test_harness_checks_the_plan_and_the_model_hndl_produce(monkeypatch):
    """The checks that guard HNDL itself rather than the operator, failed on purpose."""
    import hndl.resolver
    from hndl.torch import GraphModule

    spec, example = REGISTRY.get("decimate"), REGISTRY.get("decimate").examples[0]
    registry = broken_registry()
    shift = registry.get("shift")
    original_repr = GraphModule.__repr__
    with monkeypatch.context() as patch:
        patch.setattr(GraphModule, "__repr__", lambda self: (torch.rand(1), original_repr(self))[1])
        with pytest.raises(AssertionError, match=r"shift: repr\(model\) consumed random numbers"):
            testing.check_build_and_run(registry, shift, shift.examples[0])
    with monkeypatch.context() as patch:
        patch.setattr(GraphModule, "__repr__", lambda self: "GraphModule()")
        with pytest.raises(AssertionError, match=r"shift: repr\(model\) does not name the operator"):
            testing.check_build_and_run(registry, shift, shift.examples[0])
    with monkeypatch.context() as patch:
        patch.setattr(testing, "_replay", lambda graph, registry: lambda x: (registry.ops.linear(8, bias=False),
                                                                           registry.ops.shift())[1])
        with pytest.raises(AssertionError, match="shift: native Python replay resolves to a different plan"):
            testing.check_round_trip(registry, shift, shift.examples[0])
    with monkeypatch.context() as patch:
        other = resolve("linear(8)", registry=registry, input_shape=("B", 4), output_shape=("B", 8))
        patch.setattr(testing, "ResolvedPlan", type("Stale", (), {"from_json": staticmethod(lambda *a, **k: other)}))
        with pytest.raises(AssertionError, match="shift: the plan changes when saved and restored"):
            testing.check_round_trip(registry, shift, shift.examples[0])
    with monkeypatch.context() as patch:
        # A resolver that stops leaving Arg(since=...) defaults out of the plan.
        patch.setattr(hndl.resolver, "_canonical_arguments", lambda spec, args, source: (args, source))
        with pytest.raises(AssertionError, match="n1.offset keeps its post-release default in the plan"):
            testing.check_round_trip(REGISTRY, spec, example)


def test_harness_rejects_a_declaration_from_another_registry():
    with pytest.raises(AssertionError, match="different declaration"):
        testing.check_round_trip(REGISTRY, host_registry().get("decimate"), REGISTRY.get("decimate").examples[0])
    with pytest.raises(AssertionError, match="host.decimate@1 is not registered in this registry"):
        testing.check_operator(Registry.builtins(), REGISTRY.get("decimate"))


def test_harness_takes_one_device_or_dtype_name():
    testing.check_operator(REGISTRY, "decimate", devices="cpu", dtypes="float32")


def test_readme_registration_example_passes_the_harness():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = next(block for block in re.findall(r"```python\n(.*?)\n```", readme, re.S) if '"my_silu"' in block)
    namespace = {}
    exec(block, namespace)
    testing.check_operator(namespace["registry"], "my_silu", devices="cpu")


def test_capture_hint_does_not_suggest_an_alias_bound_to_another_operator():
    registry = Registry()

    @registry.operator("linear", identity="host.mylinear", summary="Not the built-in linear.",
                       shape="x[B, ...] -> out[B, ...]")
    class MyLinear(Shift):
        """``out = x + 1``."""

    with pytest.raises(HNDLError, match=r"binds linear@1, .* \(ops.linear in this capture binds "
                                        r"host.mylinear@1, a different operator\)") as caught:
        resolve_callable(lambda x: Registry.builtins().ops.linear(3), registry=registry,
                         input_shape=("B", 3), output_shape=("B", 3))
    assert "call ops.linear" not in str(caught.value)


@pytest.mark.parametrize("helper", [conv_axis, conv_transpose_axis])
def test_conv_axis_helpers_reject_a_port_without_the_axis(helper):
    registry = Registry()

    @registry.operator("bad", identity="host.bad", summary="Strides an axis the port lacks.", shape="x -> out",
                       relation=lambda s: helper(s, 2, kernel=1, stride=2), shape_text="axis 2 strided by 2")
    class Bad(Shift):
        """``out = x + 1``."""

    with pytest.raises(HNDLError, match="E_CONSTRAINT.*Port x has rank 2, which has no axis 2"):
        resolve("bad()", registry=registry, input_shape=("B", 4), output_shape=("B", 2))


def test_importing_hndl_and_its_harness_does_not_import_pytest():
    src = str(Path(hndl.__file__).resolve().parents[1])
    code = ("import sys, hndl, hndl.testing, hndl.relations\n"
            "assert 'pytest' not in sys.modules and '_pytest' not in sys.modules")
    env = dict(os.environ, PYTHONPATH=src)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
