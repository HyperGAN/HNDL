"""Operator registries: exact alias/identity bindings and argument normalization."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import importlib.metadata
from types import MappingProxyType
from typing import TYPE_CHECKING

from torch import nn

from .errors import HNDLError
from .operator import Operator, make_operator
from .settings import SCHEME_NAMES

if TYPE_CHECKING:
    from .types import ResolvedPlan

_ACTIVE = ContextVar("hndl_active_registry", default=None)
_BUILTINS_TEMPLATE: Registry | None = None
MAX_PROVIDER_NAME = 64


class Registry:
    """An explicit set of operators. ``Registry.builtins()`` holds the catalog."""

    def __init__(self):
        self._aliases = {}
        self._identities = {}
        self._pretrained_providers = {}
        self._pretrained_readouts = {}
        self._pretrained_shape_fns = {}
        self._pretrained_contracts = {}

    @property
    def aliases(self):
        return tuple(self._aliases)

    @property
    def operators(self):
        return tuple(self._aliases.values())

    @property
    def ops(self):
        from .capture import OperatorNamespace
        return OperatorNamespace(self)

    @classmethod
    def builtins(cls):
        global _BUILTINS_TEMPLATE
        if _BUILTINS_TEMPLATE is None:
            from .operators import discover
            template = cls()
            for spec in discover():
                template._add(spec)
            _BUILTINS_TEMPLATE = template
        registry = cls()
        registry._aliases = dict(_BUILTINS_TEMPLATE._aliases)
        registry._identities = dict(_BUILTINS_TEMPLATE._identities)
        registry._pretrained_providers = dict(_BUILTINS_TEMPLATE._pretrained_providers)
        registry._pretrained_readouts = dict(_BUILTINS_TEMPLATE._pretrained_readouts)
        registry._pretrained_shape_fns = dict(_BUILTINS_TEMPLATE._pretrained_shape_fns)
        registry._pretrained_contracts = dict(_BUILTINS_TEMPLATE._pretrained_contracts)
        return registry

    def _add(self, spec):
        if type(spec) is not Operator:
            raise HNDLError("E_REGISTRY", "Registries hold @operator declarations")
        if spec.alias in SCHEME_NAMES:
            raise HNDLError("E_REGISTRY", f"Operator alias {spec.alias} conflicts with the reserved "
                                          "initializer scheme of the same name")
        if spec.alias in self._aliases or spec.key in self._identities:
            raise HNDLError("E_REGISTRY", f"Duplicate operator alias or identity: {spec.alias} / {spec.key}")
        self._aliases[spec.alias] = spec
        self._identities[spec.key] = spec
        return spec

    def add(self, cls):
        """Register a class already declared with ``@operator``."""
        spec = getattr(cls, "__hndl_operator__", None)
        if type(spec) is not Operator or spec.module is not cls:
            raise HNDLError("E_REGISTRY", "add() expects a class decorated with @operator")
        return self._add(spec)

    def operator(self, alias, **declaration):
        """Declare and register an operator in one step."""
        def decorate(cls):
            cls.__hndl_operator__ = self._add(make_operator(cls, alias, **declaration))
            return cls
        return decorate

    def get(self, alias):
        try:
            return self._aliases[alias]
        except (KeyError, TypeError):
            raise HNDLError("E_OPERATOR", f"Unknown operator alias {alias!r}; register it explicitly") from None

    def by_identity(self, identity):
        try:
            return self._identities[identity]
        except (KeyError, TypeError):
            raise HNDLError("E_STATE_VERSION", f"Operator {identity!r} is unavailable; supply its exact compatible registration") from None

    def subgraph(self, alias: str, plan: ResolvedPlan) -> Operator:
        """Register a resolved plan as a composite subgraph operator."""
        inputs_str = ", ".join(f"{name}[{', '.join(str(d) for d in entry['shape'])}]" for name, entry in plan.inputs.items())
        outputs_str = ", ".join(f"{name}[{', '.join(str(d) for d in entry['shape'])}]" for name, entry in plan.outputs.items())
        shape_str = f"{inputs_str} -> {outputs_str}"

        class _SubgraphModule(nn.Module):
            def __init__(self):
                super().__init__()
                from .torch import build
                self.submodel = build(plan, device="cpu")

            def forward(self, *args, **kwargs):
                if args and not kwargs and len(plan.inputs) == 1:
                    input_name = next(iter(plan.inputs))
                    kwargs = {input_name: args[0]}
                    args = ()
                out = self.submodel(*args, **kwargs)
                if len(plan.outputs) == 1 and isinstance(out, Mapping):
                    return next(iter(out.values()))
                return out

        spec = make_operator(
            _SubgraphModule,
            alias,
            shape=shape_str,
            summary=f"Composite subgraph operator for {alias}",
            category="composite",
        )
        return self._add(spec)

    def discover_plugins(self, group: str = "hndl.operators") -> None:
        """Load third-party operators registered via pyproject.toml entry-points."""
        for ep in importlib.metadata.entry_points(group=group):
            obj = ep.load()
            if isinstance(obj, type) and hasattr(obj, "__hndl_operator__"):
                self.add(obj)
            elif isinstance(obj, Operator):
                self._add(obj)

    # -- pretrained architecture providers -------------------------------------------------
    #
    # A provider is trusted host Python, never something configuration selects:
    # the host registers a zero-argument callable that returns the ``nn.Module``
    # a local ``.pth`` checkpoint belongs to, and configuration may only name an
    # already registered provider. Providers live on this instance, so a
    # provider added to one ``Registry.builtins()`` is invisible to the next.
    # A readout is the same kind of trusted Python: a named ``callable(model, x)``
    # the host binds to a provider, which configuration may only name.

    @property
    def pretrained_providers(self):
        return tuple(self._pretrained_providers)

    def pretrained_provider(self, name, build, readouts=None, *args, contract="tensor", shape_fn=None, **kwargs):
        """Register a trusted architecture builder for ``pretrained(..., provider=name)``.

        ``readouts`` optionally maps a name to a ``callable(model, x)`` returning
        one tensor, selected in configuration with ``readout="<name>"`` when the
        model's own ``forward`` is not what the network wants.

        ``contract`` declares what inputs the model consumes: ``"tensor"`` (default,
        floating-point), ``"ids"`` (rank 2 [B, T] int64 token IDs), or ``"pixels"``.

        ``shape_fn`` optionally provides an explicit output shape calculation
        ``callable(input_shape, output, **kwargs) -> tuple[int, ...]``, bypassing
        PyTorch meta-device tracing during graph resolution.
        """
        if len(args) == 1:
            contract = args[0]
        elif len(args) >= 2:
            contract = args[0]
            shape_fn = args[1]
        if "contract" in kwargs:
            contract = kwargs["contract"]
        if "shape_fn" in kwargs:
            shape_fn = kwargs["shape_fn"]

        if type(name) is not str or not name or len(name) > MAX_PROVIDER_NAME:
            raise HNDLError("E_REGISTRY", f"A pretrained provider name must be a string of 1-{MAX_PROVIDER_NAME} characters")
        if not callable(build):
            raise HNDLError("E_REGISTRY", f"Pretrained provider {name!r} must be a callable returning an nn.Module")
        if name in self._pretrained_providers:
            raise HNDLError("E_REGISTRY", f"Duplicate pretrained provider {name!r}")
        if readouts is not None and not isinstance(readouts, Mapping):
            raise HNDLError("E_REGISTRY", f"Pretrained provider {name!r} readouts must be a mapping of name to callable(model, x)")
        if shape_fn is not None and not callable(shape_fn):
            raise HNDLError("E_REGISTRY", f"Pretrained provider {name!r} shape_fn must be a callable")

        if hasattr(contract, "kind"):
            contract_str = contract.kind
        elif isinstance(contract, str):
            contract_str = contract
        else:
            raise HNDLError("E_REGISTRY", f"Pretrained provider {name!r} contract must be a string or Contract, got {type(contract).__name__}")
        if contract_str not in ("tensor", "ids", "pixels"):
            raise HNDLError("E_REGISTRY", f"Pretrained provider {name!r} contract must be 'tensor', 'ids', or 'pixels', got {contract_str!r}")

        self._pretrained_providers[name] = build
        self._pretrained_readouts[name] = {}
        if shape_fn is not None:
            self._pretrained_shape_fns[name] = shape_fn
        self._pretrained_contracts[name] = contract_str
        for readout_name, readout in (readouts or {}).items():
            self.pretrained_readout(name, readout_name, readout)
        return build

    def pretrained_readout(self, provider, name, readout):
        """Add one named readout to an already registered provider."""
        if provider not in self._pretrained_providers:
            raise HNDLError("E_REGISTRY", f"No pretrained provider named {provider!r}; "
                                          "register it with registry.pretrained_provider(name, build) first")
        if type(name) is not str or not name or len(name) > MAX_PROVIDER_NAME:
            raise HNDLError("E_REGISTRY", f"A pretrained readout name must be a string of 1-{MAX_PROVIDER_NAME} characters")
        if not callable(readout):
            raise HNDLError("E_REGISTRY", f"Pretrained readout {name!r} of provider {provider!r} must be a "
                                          "callable(model, x) returning one tensor")
        if name in self._pretrained_readouts[provider]:
            raise HNDLError("E_REGISTRY", f"Duplicate pretrained readout {name!r} for provider {provider!r}")
        self._pretrained_readouts[provider][name] = readout
        return readout

    def pretrained_builder(self, name):
        """The registered builder for ``name``, or None."""
        return self._pretrained_providers.get(name)

    def pretrained_shape_fn(self, name):
        """The registered shape_fn for ``name``, or None."""
        return self._pretrained_shape_fns.get(name)

    def pretrained_contract(self, name):
        """The registered contract kind for ``name``, defaulting to 'tensor'."""
        return self._pretrained_contracts.get(name, "tensor")

    def pretrained_readouts(self, provider):
        """The readouts registered for ``provider``, as a name to callable mapping."""
        return MappingProxyType(self._pretrained_readouts.get(provider, {}))

    @contextmanager
    def activated(self):
        """Make this registry the one provider lookups see on this thread/task."""
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    @classmethod
    def active(cls):
        """The registry a resolution or build is currently running under, or None."""
        return _ACTIVE.get()


def normalize_arguments(op, positional, kwargs, policy=None):
    """Validate author scalar arguments; symbolic tensor binding is frontend-owned."""
    if not isinstance(kwargs, Mapping):
        raise HNDLError("E_ARGUMENT", "Operator keyword arguments must be a mapping")
    args = dict(kwargs)
    if "policy" in args:
        if policy is not None:
            raise HNDLError("E_ARGUMENT", "Policy supplied twice")
        policy = args.pop("policy")
    if "name" in args:
        raise HNDLError("E_ARGUMENT", "name is frontend metadata, not an operator argument")
    positional = tuple(positional)
    if op.positional_rest is not None:
        rest = op.positional_rest
        if positional:
            if rest in args:
                raise HNDLError("E_ARGUMENT", f"{rest} supplied twice")
            if len(positional) == 1 and isinstance(positional[0], (tuple, list)):
                args[rest] = tuple(positional[0])
            else:
                args[rest] = positional
        args.setdefault(rest, ())
    else:
        names = op.positional_names
        if op.alias == "moe" and len(positional) == 2:
            names = ("experts", "hidden")
        elif op.alias == "moe" and len(positional) == 3:
            names = ("experts", "top_k", "hidden")
        if len(positional) > len(names):
            raise HNDLError("E_ARGUMENT", f"{op.alias} accepts at most {len(names)} author positional arguments")
        for name, value in zip(names, positional):
            if name in args:
                raise HNDLError("E_ARGUMENT", f"{name} supplied positionally and by keyword")
            args[name] = value
    unknown = set(args) - set(op.args)
    if unknown:
        raise HNDLError("E_ARGUMENT", f"Unknown arguments for {op.alias}: {', '.join(sorted(unknown))}")
    if policy is not None:
        profile = op.policies.get(policy) if type(policy) is str else None
        if profile is None:
            raise HNDLError("E_POLICY_CONFLICT", f"Policy {policy!r} does not apply to {op.key}")
        for name, value in profile.requires.items():
            expected = op.args[name].validate(value, name)
            if name in args:
                actual = op.args[name].validate(args[name], name)
                if actual != expected:
                    raise HNDLError("E_POLICY_CONFLICT", f"Explicit {name}={args[name]!r} conflicts with {policy} requirement {value}")
            args[name] = expected
    normalized = {}
    for name, value in args.items():
        normalized[name] = op.args[name].validate(value, name)
    for name, arg in op.args.items():
        if name not in normalized:
            if arg.has_default:
                normalized[name] = arg.default
            elif arg.required:
                raise HNDLError("E_ARGUMENT", f"{op.alias} requires {name}; supply it or select an applicable policy")
    if op.validate is not None:
        op.validate(normalized)
    return normalized
