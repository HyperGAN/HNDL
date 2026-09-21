"""Generic checks every built-in operator must pass through its declared examples.

Operator-specific numerics and error codes live in ``tests/operators/test_<alias>.py``.
"""

import pytest
import torch

from hndl import Registry, ResolvedPlan, resolve, resolve_callable
from hndl.config import capture_config
from hndl.torch import DTYPES, build, parameter_counts

from .conftest import all_operators, contract, example_input, operator_examples


def port_input(spec, port, shape, compute, device):
    """A random tensor for one input port, honouring the port's declared dtype."""
    declared = spec.port_dtype(port, compute)
    # A port that accepts any dtype is fed the plan's compute dtype.
    dtype = DTYPES[compute if declared == "any" else declared]
    if not dtype.is_floating_point:
        return torch.randint(0, 8, (2, *shape[1:]), device=device).to(dtype)
    return torch.randn(2, *shape[1:], device=device, dtype=dtype, requires_grad=True)


def replay(graph, registry):
    """Rebuild a captured configuration through the native ``ops`` frontend."""
    def author(x):
        symbols = {"input:x": x}
        for node in graph.nodes:
            spec = registry.by_identity(node.op)
            ports = spec.input_ports_for(node.args)
            tensors = [symbols[node.inputs[port]] for port in ports]
            origins = node.source["argument_origins"]
            kwargs = {key: value for key, value in node.args.items() if origins.get(key) == "explicit"}
            if spec.variadic is not None:
                kwargs.pop("input_count", None)
            if node.source.get("policy"):
                kwargs["policy"] = next(alias for alias, policy in spec.policies.items()
                                        if policy.identity == node.source["policy"])
            init = dict(node.initialization["overrides"])
            if init:
                kwargs["init"] = init
            trainability = node.trainability
            if trainability["default"] is not None:
                kwargs["trainable"] = trainability["default"]
            elif trainability["overrides"]:
                kwargs["trainable"] = dict(trainability["overrides"])
            result = getattr(registry.ops, spec.alias)(*tensors, name=node.id, **kwargs)
            outputs = (result,) if len(node.outputs) == 1 else result
            for port, symbol in zip(node.outputs, outputs):
                symbols[f"node:{node.id}/{port}"] = symbol
        return symbols[graph.output_ref]
    return author


@pytest.mark.parametrize("spec", all_operators())
def test_declaration_is_complete(spec):
    assert spec.summary.strip() and not spec.summary.strip().endswith(":")
    assert spec.doc.strip(), f"{spec.alias} needs a class docstring"
    assert spec.examples, f"{spec.alias} needs at least one example"
    assert spec.category
    for name, arg in spec.args.items():
        assert arg.help.strip(), f"{spec.alias}.{name} needs help text"
    if spec.relation is not None and not any(port.pattern for port in (*spec.inputs, *spec.outputs)):
        assert spec.relation_text, f"{spec.alias} uses a relation function and needs shape_text"


@pytest.mark.parametrize("spec,example", operator_examples())
def test_example_resolves_identically_in_both_frontends_and_round_trips(spec, example):
    registry = Registry.builtins()
    kwargs = contract(example, registry=registry)
    plan = resolve(example.source, **kwargs)
    assert any(node.op == spec.key for node in plan.nodes), f"example does not use {spec.alias}"
    graph = capture_config(example.source, **kwargs)
    native = resolve_callable(replay(graph, registry), **kwargs)
    assert native.semantic_digest == plan.semantic_digest
    restored = ResolvedPlan.from_json(plan.to_json(), registry=registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.to_json() == plan.to_json()
    for node in plan.nodes:
        for name, arg in registry.by_identity(node.op).args.items():
            assert name in node.args, f"{node.id}.{name} missing from the concrete plan"


@pytest.mark.parametrize("spec,example", operator_examples())
def test_example_builds_runs_and_is_deterministic(spec, example, device):
    plan = resolve(example.source, **contract(example))
    model = build(plan, device=device, initialization_seed=3)
    before = torch.get_rng_state().clone()
    text = repr(model)
    assert torch.equal(before, torch.get_rng_state())
    assert spec.alias in text
    counts = parameter_counts(plan)
    real = {node.id: sum(p.numel() for p in model[node.id].parameters()) for node in plan.nodes}
    assert counts == real
    assert model.build_receipt["state_bytes"] == sum(
        t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
    x = example_input(example, 2, device, DTYPES[plan.dtype])
    output = model(x=x)["output"]
    assert tuple(output.shape) == (2, *example.output_shape[1:])
    assert output.isfinite().all()
    if any(p.requires_grad for p in model.parameters()) or x.requires_grad:
        output.float().square().mean().backward()
    if x.requires_grad:
        assert x.grad is not None and x.grad.isfinite().all()
    again = build(plan, device=device, initialization_seed=3)
    torch.testing.assert_close(again(x=x.detach())["output"], output.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("spec,example", operator_examples())
def test_example_runs_in_reduced_precision_on_cuda(spec, example, dtype):
    plan = resolve(example.source, **contract(example, dtype=dtype))
    model = build(plan, device="cuda:0", initialization_seed=3)
    torch_dtype = DTYPES[dtype]
    assert all(p.dtype == torch_dtype for p in model.parameters())
    x = example_input(example, 2, "cuda:0", torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and tuple(output.shape) == (2, *example.output_shape[1:])
    assert output.isfinite().all()
    if x.requires_grad:
        output.float().square().mean().backward()
        assert x.grad.isfinite().all()


@pytest.mark.parametrize("spec,example", operator_examples())
def test_reference_implementation_matches(spec, example, device):
    if spec.reference is None:
        pytest.skip(f"{spec.alias} declares no reference implementation")
    plan = resolve(example.source, **contract(example))
    model = build(plan, device=device, initialization_seed=5)
    for node in plan.nodes:
        if node.op != spec.key:
            continue
        module = model[node.id]
        reference = spec.reference(module)
        ports = spec.input_ports_for(node.args)
        inputs = [port_input(spec, port, node.input_shapes[port], plan.dtype, device) for port in ports]
        mirrors = [tensor.detach().clone().requires_grad_() if tensor.requires_grad else tensor.clone()
                   for tensor in inputs]
        actual, expected = module(*inputs), reference(*mirrors)
        actual = actual if isinstance(actual, (tuple, list)) else (actual,)
        expected = expected if isinstance(expected, (tuple, list)) else (expected,)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
        if not any(tensor.requires_grad for tensor in actual):
            # A source such as constant() detaches its output; there is no
            # gradient to compare, and both sides must agree on that.
            assert not any(tensor.requires_grad for tensor in expected)
            continue
        sum(a.square().mean() for a in actual).backward()
        sum(e.square().mean() for e in expected).backward()
        for a, e in zip(inputs, mirrors):
            if a.grad is not None or e.grad is not None:
                torch.testing.assert_close(a.grad, e.grad)
