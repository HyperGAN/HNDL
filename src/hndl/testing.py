"""The operator test harness, public so packages that declare operators can run it.

Every built-in operator passes these checks through its declared
``Example``\\ s, and ``tests/operators/test_all_operators.py`` is written
with nothing but this module. An operator declared outside HNDL gets the
same coverage by pointing the harness at the registry that holds it:

.. code-block:: python

    import pytest
    from hndl import Registry, testing

    registry = Registry.builtins()
    # ... register the package's operators on it ...

    def test_my_operators():
        for alias in ("my_blur", "my_pool"):
            testing.check_operator(registry, alias)

    # or one pytest case per example and device:
    @pytest.mark.parametrize("spec,example", testing.example_params(registry, ["my_blur"]))
    @pytest.mark.parametrize("device", testing.available_devices())
    def test_example_builds_and_runs(spec, example, device):
        testing.check_build_and_run(registry, spec, example, device=device)

The ``check_*`` functions raise ``AssertionError`` on the first failure, so
they work under pytest, unittest or a plain script. A failure HNDL itself
detects while resolving, building or running an example, such as a module
whose output shape differs from the declared one, surfaces as the
``HNDLError`` HNDL raises for it. Importing this module
imports torch but not pytest; only :func:`example_params` and
:func:`operator_params`, which build ``pytest.param`` objects, need pytest.

The checks, for every example of an operator:

- :func:`check_declaration`: summary, a docstring on the class itself,
  argument help text, at least one example, and ``shape_text`` for a relation the shape string
  does not show.
- :func:`check_round_trip`: the example resolves, uses the operator, gives
  the same semantic digest when replayed through ``registry.ops`` in native
  Python, survives ``to_json``/``from_json`` unchanged, and records every
  argument in the plan except an ``Arg(since=...)`` left at its default.
- :func:`check_build_and_run`: the plan builds on a device, ``repr`` draws no
  randomness, ``parameter_counts`` and the build receipt agree with the
  module, a batch runs to a finite output of the declared shape with finite
  input gradients, and a rebuild from the same seed reproduces the output.
- :func:`check_reference`: when the operator declares ``reference=``, every
  node of it matches that reference in outputs and input gradients.
"""

from __future__ import annotations

import torch

from .config import capture_config, resolve
from .capture import resolve_callable
from .operator import Example, Operator
from .registry import Registry
from .types import ResolvedPlan

__all__ = ["REDUCED_PRECISION", "available_devices", "check_build_and_run", "check_declaration",
           "check_operator", "check_reference", "check_round_trip", "example_cases", "example_contract",
           "example_input", "example_params", "operator_params"]

#: Reduced-precision dtypes :func:`check_operator` runs on CUDA devices by default.
REDUCED_PRECISION = ("float16", "bfloat16")


def _fail(message):
    raise AssertionError(message)


def _spec(registry, operator):
    """The registered declaration for an alias, or ``operator`` itself once it is known to be registered."""
    if not isinstance(registry, Registry):
        raise TypeError("the harness needs the Registry that holds the operator")
    if isinstance(operator, str):
        return registry.get(operator)
    if type(operator) is not Operator:
        raise TypeError("expected an operator alias or a declaration from registry.operators")
    registered = registry._identities.get(operator.key)
    if registered is None:
        _fail(f"{operator.key} is not registered in this registry")
    if registered is not operator:
        _fail(f"{operator.key} is registered in this registry with a different declaration")
    return operator


def _names(value):
    """One device or dtype name, or an iterable of them, as a tuple."""
    return (value,) if isinstance(value, (str, torch.device)) else tuple(value)


def _torch_dtype(name):
    from .torch import DTYPES
    return DTYPES[name]


def available_devices():
    """``["cpu"]``, plus ``"cuda:0"`` when CUDA is available."""
    return ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def example_cases(registry, aliases=None, *, network=True):
    """``(declaration, example)`` pairs for the registry's operators, in registry order.

    ``aliases`` limits the result to those operators, in the order given; a
    package usually passes its own aliases so it does not rerun the
    built-in catalog. ``network=False`` drops examples that download a
    checkpoint (``Example(network=True)``).
    """
    return [(spec, example) for spec, _, example in _indexed(registry, aliases, network)]


def _indexed(registry, aliases, network=True):
    specs = registry.operators if aliases is None else [registry.get(alias) for alias in aliases]
    return [(spec, index, example) for spec in specs for index, example in enumerate(spec.examples)
            if network or not example.network]


def example_params(registry, aliases=None):
    """One ``pytest.param(spec, example)`` per example, with ids ``<alias>-<index>``.

    Examples declared with ``network=True`` carry the ``network`` mark, so
    ``-m "not network"`` deselects them; register the mark in the package's
    pytest configuration. Requires pytest.
    """
    import pytest

    params = []
    for spec, index, example in _indexed(registry, aliases):
        marks = [pytest.mark.network] if example.network else []
        params.append(pytest.param(spec, example, id=f"{spec.alias}-{index}", marks=marks))
    return params


def operator_params(registry, aliases=None):
    """One ``pytest.param(spec)`` per operator, with the alias as its id. Requires pytest."""
    import pytest

    specs = registry.operators if aliases is None else [registry.get(alias) for alias in aliases]
    return [pytest.param(spec, id=spec.alias) for spec in specs]


def example_contract(example, **extra):
    """The ``input_shape``/``output_shape``/``input_dtype`` keywords that resolve ``example``."""
    kwargs = dict(input_shape=example.input_shape, output_shape=example.output_shape)
    if example.input_dtype is not None:
        kwargs["input_dtype"] = example.input_dtype
    kwargs.update(extra)
    return kwargs


def example_input(example, batch, device, dtype):
    """A random input tensor for ``example`` with batch ``batch``.

    Index inputs are integers in ``[0, 8)`` and boolean inputs are random
    masks; floating inputs are standard normal in ``dtype`` and require
    gradients.
    """
    shape = (batch, *example.input_shape[1:])
    if example.input_dtype in ("int64", "int32"):
        return torch.randint(0, 8, shape, device=device, dtype=getattr(torch, example.input_dtype))
    if example.input_dtype == "bool":
        return torch.rand(shape, device=device) > 0.5
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=True)


def _port_input(spec, port, shape, compute, device):
    """A random tensor for one input port, honouring the port's declared dtype."""
    declared = spec.port_dtype(port, compute)
    # A port that accepts any dtype is fed the plan's compute dtype.
    dtype = _torch_dtype(compute if declared == "any" else declared)
    if not dtype.is_floating_point:
        return torch.randint(0, 8, (2, *shape[1:]), device=device).to(dtype)
    return torch.randn(2, *shape[1:], device=device, dtype=dtype, requires_grad=True)


def _replay(graph, registry):
    """Rebuild a captured configuration through the native ``registry.ops`` frontend."""
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
            # A variadic output port returns a tuple even for a single output.
            outputs = result if spec.returns_tuple(node.args) else (result,)
            for port, symbol in zip(node.outputs, outputs):
                symbols[f"node:{node.id}/{port}"] = symbol
        return symbols[graph.output_ref]
    return author


def check_declaration(registry, operator):
    """The declaration documents itself: summary, docstring, help text, examples."""
    spec = _spec(registry, operator)
    summary = spec.summary.strip()
    if not summary or summary.endswith(":"):
        _fail(f"{spec.alias} needs a one-sentence summary")
    # inspect.getdoc falls back to a base class's docstring, such as
    # nn.Module's; the operator's own class has to document it.
    if not spec.doc.strip() or not (spec.module.__dict__.get("__doc__") or "").strip():
        _fail(f"{spec.alias} needs a class docstring")
    if not spec.examples:
        _fail(f"{spec.alias} needs at least one Example")
    if not spec.category:
        _fail(f"{spec.alias} needs a category")
    for name, arg in spec.args.items():
        if not arg.help.strip():
            _fail(f"{spec.alias}.{name} needs help text")
    if spec.relation is not None and not any(port.pattern for port in (*spec.inputs, *spec.outputs)):
        if not spec.relation_text:
            _fail(f"{spec.alias} uses a relation function and needs shape_text")


def check_round_trip(registry, operator, example):
    """Both frontends agree on ``example`` and its plan survives JSON unchanged."""
    spec = _spec(registry, operator)
    if not isinstance(example, Example):
        raise TypeError("expected one of the operator's Example declarations")
    kwargs = example_contract(example, registry=registry)
    plan = resolve(example.source, **kwargs)
    if not any(node.op == spec.key for node in plan.nodes):
        _fail(f"example does not use {spec.alias}: {example.source!r}")
    graph = capture_config(example.source, **kwargs)
    native = resolve_callable(_replay(graph, registry), **kwargs)
    if native.semantic_digest != plan.semantic_digest:
        _fail(f"{spec.alias}: native Python replay resolves to a different plan than the configuration")
    restored = ResolvedPlan.from_json(plan.to_json(), registry=registry)
    if restored.semantic_digest != plan.semantic_digest or restored.to_json() != plan.to_json():
        _fail(f"{spec.alias}: the plan changes when saved and restored")
    for node in plan.nodes:
        for name, arg in registry.by_identity(node.op).args.items():
            # Only an argument added after release may be left out, and only at its default.
            if name not in node.args and arg.since is None:
                _fail(f"{node.id}.{name} missing from the concrete plan")
            if name in node.args and arg.since is not None and node.args[name] == arg.default:
                _fail(f"{node.id}.{name} keeps its post-release default in the plan")
    return plan


def check_build_and_run(registry, operator, example, *, device="cpu", dtype="float32", batch=2):
    """``example`` builds on ``device`` in ``dtype``, runs forward and backward, and is reproducible."""
    from .torch import build, parameter_counts

    spec = _spec(registry, operator)
    plan = resolve(example.source, **example_contract(example, registry=registry, dtype=dtype))
    model = build(plan, device=device, initialization_seed=3)
    torch_dtype = _torch_dtype(plan.dtype)
    wrong = sorted({str(p.dtype) for p in model.parameters() if p.dtype != torch_dtype})
    if wrong:
        _fail(f"{spec.alias}: {dtype} plan built parameters of dtype {', '.join(wrong)}")
    before = torch.get_rng_state().clone()
    text = repr(model)
    if not torch.equal(before, torch.get_rng_state()):
        _fail(f"{spec.alias}: repr(model) consumed random numbers")
    if spec.alias not in text:
        _fail(f"{spec.alias}: repr(model) does not name the operator")
    counts = parameter_counts(plan)
    real = {node.id: sum(p.numel() for p in model[node.id].parameters()) for node in plan.nodes}
    if counts != real:
        _fail(f"{spec.alias}: parameter_counts {counts} disagrees with the built modules {real}")
    stored = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
    if model.build_receipt["state_bytes"] != stored:
        _fail(f"{spec.alias}: the build receipt reports {model.build_receipt['state_bytes']} state bytes, "
              f"the module holds {stored}")
    x = example_input(example, batch, device, torch_dtype)
    output = model(x=x)["output"]
    expected = (batch, *example.output_shape[1:])
    if tuple(output.shape) != expected:
        _fail(f"{spec.alias}: output shape {tuple(output.shape)}, declared {expected}")
    if output.is_floating_point() and output.dtype != torch_dtype:
        _fail(f"{spec.alias}: {dtype} plan produced {output.dtype}")
    if not output.isfinite().all():
        _fail(f"{spec.alias}: output is not finite")
    if any(p.requires_grad for p in model.parameters()) or x.requires_grad:
        output.float().square().mean().backward()
    if x.requires_grad and (x.grad is None or not x.grad.isfinite().all()):
        _fail(f"{spec.alias}: the input gradient is missing or not finite")
    again = build(plan, device=device, initialization_seed=3)
    torch.testing.assert_close(again(x=x.detach())["output"], output.detach(),
                               msg=lambda text: f"{spec.alias}: a rebuild from the same seed differs\n{text}")
    return model


def check_reference(registry, operator, example, *, device="cpu"):
    """Every node of the operator in ``example`` matches its declared ``reference``.

    Outputs and the gradients of every floating input are compared with
    ``torch.testing.assert_close``. Returns False, checking nothing, when the
    operator declares no reference, and True otherwise.
    """
    from .torch import build

    spec = _spec(registry, operator)
    if spec.reference is None:
        return False
    plan = resolve(example.source, **example_contract(example, registry=registry))
    model = build(plan, device=device, initialization_seed=5)
    for node in plan.nodes:
        if node.op != spec.key:
            continue
        module = model[node.id]
        reference = spec.reference(module)
        ports = spec.input_ports_for(node.args)
        inputs = [_port_input(spec, port, node.input_shapes[port], plan.dtype, device) for port in ports]
        mirrors = [tensor.detach().clone().requires_grad_() if tensor.requires_grad else tensor.clone()
                   for tensor in inputs]
        actual, expected = module(*inputs), reference(*mirrors)
        actual = actual if isinstance(actual, (tuple, list)) else (actual,)
        expected = expected if isinstance(expected, (tuple, list)) else (expected,)
        if len(actual) != len(expected):
            _fail(f"{node.id}: the module returns {len(actual)} tensors, the reference {len(expected)}")
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, msg=lambda text: f"{node.id}: output differs from the reference\n{text}")
        if not any(tensor.requires_grad for tensor in actual):
            # A source such as constant() detaches its output; there is no
            # gradient to compare, and both sides must agree on that.
            if any(tensor.requires_grad for tensor in expected):
                _fail(f"{node.id}: the reference carries a gradient the module does not")
            continue
        if not any(tensor.requires_grad for tensor in expected):
            _fail(f"{node.id}: the module carries a gradient the reference does not")
        sum(a.square().mean() for a in actual).backward()
        sum(e.square().mean() for e in expected).backward()
        for a, e in zip(inputs, mirrors):
            if a.grad is not None or e.grad is not None:
                if a.grad is None or e.grad is None:
                    _fail(f"{node.id}: only one of the module and the reference produces an input gradient")
                torch.testing.assert_close(a.grad, e.grad,
                                           msg=lambda text: f"{node.id}: input gradient differs from the "
                                                            f"reference\n{text}")
    return True


def check_operator(registry, alias, *, devices=None, dtypes=None, network=False):
    """Run every check on every example of one operator.

    ``devices`` defaults to :func:`available_devices`. ``dtypes`` defaults to
    float32 on every device plus :data:`REDUCED_PRECISION` on CUDA devices,
    which is what the built-in suite runs; an explicit list applies to every
    device. Either takes one name (``devices="cpu"``) or a list of them. Reference comparisons run in float32. Examples marked
    ``network=True`` are skipped unless ``network=True``. Raises
    ``AssertionError`` on the first failure, naming the alias, example and
    device.
    """
    spec = _spec(registry, alias)
    check_declaration(registry, spec)
    devices = available_devices() if devices is None else _names(devices)
    for index, example in enumerate(spec.examples):
        if example.network and not network:
            continue
        label = f"{spec.alias} example {index}"
        try:
            check_round_trip(registry, spec, example)
        except AssertionError as exc:
            raise AssertionError(f"{label}: {exc}") from exc
        for device in devices:
            if dtypes is not None:
                selected = _names(dtypes)
            else:
                selected = ("float32",) + (REDUCED_PRECISION if str(device).startswith("cuda") else ())
            for dtype in selected:
                try:
                    check_build_and_run(registry, spec, example, device=device, dtype=dtype)
                except AssertionError as exc:
                    raise AssertionError(f"{label} on {device} in {dtype}: {exc}") from exc
            try:
                check_reference(registry, spec, example, device=device)
            except AssertionError as exc:
                raise AssertionError(f"{label} on {device}, reference: {exc}") from exc
