"""Joining and splitting the batch axis, and the ``k*B`` entries it produces.

``concat(..., axis=0)`` stacks examples so one shared network sees both
branches in a single forward pass, and ``chunk(..., dim=0)`` takes the result
apart. The batch stays symbolic throughout: the plan records ``2*B``, and the
runtime scales it by whatever batch the call supplies.
"""

import json

import pytest
import torch

from hndl import HNDLError, Registry, ResolvedPlan, resolve, resolve_callable
from hndl.torch import build
from hndl.types import (ResolvedNode, batch_dimension, batch_extent, batch_multiple, batch_symbol,
                        batch_units)

# The downstream shape: two [B, 3, H, W] branches share one frozen trunk.
PAIRED = """
candidate = conv(x, 4, kernel_size=3, padding=1)
context = conv(y, 4, kernel_size=3, padding=1)
combined = concat(candidate, context, axis=0)
normalized = group_norm(combined, 2)
features = conv(normalized, 6, kernel_size=3, padding=1)
candidate_features, context_features = chunk(features, 2, dim=0)
out = concat(candidate_features, context_features, axis=1)
"""
CONTRACT = {"input_shape": {"x": ("B", 3, 8, 8), "y": ("B", 3, 8, 8)},
            "output_shape": ("B", 12, 8, 8)}


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


@pytest.fixture
def paired():
    return resolve(PAIRED, **CONTRACT)


def shapes(plan, node_id):
    node = next(node for node in plan.nodes if node.id == node_id)
    return dict(node.input_shapes), dict(node.output_shapes)


def test_the_plan_tracks_two_batches_separately_from_one(paired):
    assert shapes(paired, "n2")[1] == {"out": ("2*B", 4, 8, 8)}
    assert shapes(paired, "n4")[1] == {"out": ("2*B", 6, 8, 8)}
    inputs, outputs = shapes(paired, "n5")
    assert inputs == {"x": ("2*B", 6, 8, 8)}
    assert outputs == {"out0": ("B", 6, 8, 8), "out1": ("B", 6, 8, 8)}
    assert paired.output_shape == ("B", 12, 8, 8)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_one_built_module_runs_at_any_batch(paired, batch, device):
    model = build(paired, device=device, initialization_seed=5)
    candidate = torch.randn(batch, 3, 8, 8, device=device)
    context = torch.randn(batch, 3, 8, 8, device=device)
    out = model(x=candidate, y=context)["output"]
    assert tuple(out.shape) == (batch, 12, 8, 8)
    assert out.isfinite().all()
    # The shared trunk sees both branches at once, which is the point: running
    # each branch separately through the same modules gives the same values.
    trunk = lambda value: model["n4"](model["n3"](value))  # noqa: E731
    separate = torch.cat((trunk(model["n0"](candidate)), trunk(model["n1"](context))), dim=1)
    torch.testing.assert_close(out, separate)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_first_derivatives_reach_both_inputs(paired, batch, device):
    model = build(paired, device=device, initialization_seed=5).double()
    candidate = torch.randn(batch, 3, 8, 8, device=device, dtype=torch.double, requires_grad=True)
    context = torch.randn(batch, 3, 8, 8, device=device, dtype=torch.double, requires_grad=True)
    out = model(x=candidate, y=context)["output"]
    # Weighting the halves differently keeps the two branches distinguishable.
    loss = (out[:, :6].square().sum() + 3 * out[:, 6:].square().sum())
    first = torch.autograd.grad(loss, (candidate, context), create_graph=True)
    assert all(grad.isfinite().all() and grad.abs().sum() > 0 for grad in first)
    second = torch.autograd.grad(sum(grad.square().sum() for grad in first), (candidate, context))
    assert all(grad.isfinite().all() and grad.abs().sum() > 0 for grad in second)


def test_gradients_match_running_the_branches_separately(device):
    plan = resolve(PAIRED, **CONTRACT)
    model = build(plan, device=device, initialization_seed=5).double()
    candidate = torch.randn(3, 3, 8, 8, device=device, dtype=torch.double, requires_grad=True)
    context = torch.randn(3, 3, 8, 8, device=device, dtype=torch.double, requires_grad=True)
    mirrors = [value.detach().clone().requires_grad_() for value in (candidate, context)]
    model(x=candidate, y=context)["output"].square().sum().backward()
    trunk = lambda value: model["n4"](model["n3"](value))  # noqa: E731
    torch.cat((trunk(model["n0"](mirrors[0])), trunk(model["n1"](mirrors[1]))), dim=1).square().sum().backward()
    torch.testing.assert_close(candidate.grad, mirrors[0].grad)
    torch.testing.assert_close(context.grad, mirrors[1].grad)


def test_a_batch_join_of_a_join_counts_three(paired):
    plan = resolve("p = concat(x, x, axis=0)\nq = concat(p, x, axis=0)\na, b, c = chunk(q, 3, dim=0)\n"
                   "out = concat(a, b, c, axis=1)", input_shape=("B", 4), output_shape=("B", 12))
    assert shapes(plan, "n0")[1] == {"out": ("2*B", 4)}
    assert shapes(plan, "n1")[1] == {"out": ("3*B", 4)}
    assert shapes(plan, "n2")[1] == {f"out{i}": ("B", 4) for i in range(3)}


def test_an_unknown_input_batch_is_solved_from_the_joined_one():
    plan = resolve("p = concat(x, x, axis=0)\nq = concat(p, x, axis=0)\na, b, c = chunk(q, 3, dim=0)\n"
                   "out = concat(a, b, c, axis=1)", input_shape=("B", 4), output_shape=("B", 12))
    assert shapes(plan, "n1")[0] == {"x0": ("2*B", 4), "x1": ("B", 4)}


def test_the_plan_round_trips_with_its_batch_multiples(paired):
    registry = Registry.builtins()
    plan = resolve(PAIRED, registry=registry, **CONTRACT)
    encoded = plan.to_json()
    assert '"2*B"' in encoded
    restored = ResolvedPlan.from_json(encoded, registry=registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.to_json() == encoded
    assert shapes(restored, "n2")[1] == {"out": ("2*B", 4, 8, 8)}


def reseal(data):
    """Re-encode an edited plan so its digests match and the schema is what fails."""
    fields = {key: value for key, value in data.items()
              if key not in ("semantic_digest", "artifact_digest", "nodes", "inputs", "outputs")}
    if "inputs" in data:
        fields["named_inputs"] = {item["name"]: {"shape": item["shape"], "dtype": item["dtype"]}
                                  for item in data["inputs"]}
    if "outputs" in data:
        fields["named_outputs"] = {item["name"]: {"ref": item["ref"], "shape": item["shape"]}
                                   for item in data["outputs"]}
    return ResolvedPlan(nodes=tuple(ResolvedNode(**item) for item in data["nodes"]), **fields).to_json()


@pytest.mark.parametrize("entry", ["0*B", "B*2", "2*C", "1*B", "02*B", "2048*B", "-2*B", "2 * B", 0])
def test_a_malformed_batch_entry_in_a_saved_plan_is_rejected(entry, paired):
    tampered = json.loads(paired.to_json())
    for node in tampered["nodes"]:
        for group in ("input_shapes", "output_shapes"):
            for port, shape in node[group].items():
                if shape[0] == "2*B":
                    node[group][port] = [entry, *shape[1:]]
    with pytest.raises(HNDLError, match="E_SCHEMA"):
        ResolvedPlan.from_json(reseal(tampered))


def test_an_edited_but_well_formed_batch_multiple_fails_the_digest(paired):
    """Rewriting a consistent 2*B to 3*B keeps the schema valid, so the
    re-resolution behind the load is what rejects it."""
    tampered = json.loads(paired.to_json())
    for node in tampered["nodes"]:
        for group in ("input_shapes", "output_shapes"):
            for port, shape in node[group].items():
                if shape[0] == "2*B":
                    node[group][port] = ["3*B", *shape[1:]]
    with pytest.raises(HNDLError, match="E_CONSTRAINT|E_INTEGRITY"):
        ResolvedPlan.from_json(reseal(tampered))


def test_external_contracts_stay_one_plan_batch():
    with pytest.raises(HNDLError, match="E_SCHEMA"):
        resolve("concat(x, x, axis=0)", input_shape=("B", 4), output_shape=("2*B", 4))


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    # A plain B cannot be halved: nothing says the runtime batch is even.
    ("a, b = chunk(x, 2, dim=0)\nout = concat(a, b, axis=1)", ("B", 4), ("B", 8), "E_CONSTRAINT"),
    # 2*B into three sections is uneven.
    ("p = concat(x, x, axis=0)\na, b, c = chunk(p, 3, dim=0)\nout = concat(a, b, c, axis=1)",
     ("B", 4), ("B", 12), "E_CONSTRAINT"),
    # Axis 0 still requires agreement on every other axis.
    ("a = linear(x, 4)\nb = linear(x, 6)\np = concat(a, b, axis=0)\nc, d = chunk(p, 2, dim=0)\n"
     "out = concat(c, d, axis=1)", ("B", 8), ("B", 8), "E_CONSTRAINT"),
    # A batch-joined tensor cannot be published as a B output.
    ("out = concat(x, x, axis=0)", ("B", 4), ("B", 4), "E_CONSTRAINT"),
])
def test_invalid_batch_graphs_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_a_runtime_batch_that_does_not_divide_is_rejected(paired, device):
    """The graph checks contracts before each node, so an indivisible batch can
    only reach ``chunk`` when its module is called on its own."""
    model = build(paired, device=device)
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        model["n5"](torch.randn(5, 6, 8, 8, device=device))


def test_a_mismatched_second_input_batch_is_rejected(paired, device):
    model = build(paired, device=device)
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        model(x=torch.randn(3, 3, 8, 8, device=device), y=torch.randn(2, 3, 8, 8, device=device))


def test_both_frontends_agree_on_a_batch_join():
    registry = Registry.builtins()
    kwargs = {"input_shape": ("B", 4), "output_shape": ("B", 8), "registry": registry}
    plan = resolve("p = concat(x, x, axis=0)\na, b = chunk(p, 2, dim=0)\nout = concat(a, b, axis=1)", **kwargs)

    def author(x):
        pair = registry.ops.concat(x, x, axis=0, name="n0")
        a, b = registry.ops.chunk(pair, 2, dim=0, name="n1")
        return registry.ops.concat(a, b, axis=1, name="n2")

    assert resolve_callable(author, **kwargs).semantic_digest == plan.semantic_digest


def test_plans_that_never_touch_the_batch_axis_keep_their_encoding():
    plan = resolve("linear(64)\nrelu()\nlinear()", input_shape=("B", 32), output_shape=("B", 10))
    assert "*B" not in plan.to_json()
    assert all(shape[0] == "B" for node in plan.nodes
               for shape in (*node.input_shapes.values(), *node.output_shapes.values()))


@pytest.mark.parametrize("value,multiple", [("B", 1), ("2*B", 2), ("1024*B", 1024), ("1*B", None),
                                            ("0*B", None), ("B*2", None), ("2*C", None), ("02*B", None),
                                            ("1025*B", None), (4, None), (None, None)])
def test_batch_entries_have_exactly_one_spelling(value, multiple):
    assert batch_multiple(value) == multiple


def test_batch_helpers_round_trip():
    assert batch_symbol(1) == "B" and batch_symbol(3) == "3*B"
    assert batch_symbol(0) is None and batch_symbol(1025) is None
    assert batch_units("3*B") == 3 and batch_units(7) == 7 and batch_units(True) is None
    assert batch_dimension(2, True) == "2*B" and batch_dimension(2, False) == 2
    assert batch_extent("3*B", 5) == 15 and batch_extent(6, 5) == 6
