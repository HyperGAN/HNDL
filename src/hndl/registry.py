"""Operator registries: exact alias/identity bindings and argument normalization."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from .errors import HNDLError
from .operator import Operator, make_operator

_ACTIVE = ContextVar("hndl_active_registry", default=None)
MAX_PROVIDER_NAME = 64


class Registry:
    """An explicit set of operators. ``Registry.builtins()`` holds the catalog."""

    def __init__(self):
        self._aliases = {}
        self._identities = {}
        self._pretrained_providers = {}

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
        from .operators import discover
        registry = cls()
        for spec in discover():
            registry._add(spec)
        return registry

    def _add(self, spec):
        if type(spec) is not Operator:
            raise HNDLError("E_REGISTRY", "Registries hold @operator declarations")
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

    # -- pretrained architecture providers -------------------------------------------------
    #
    # A provider is trusted host Python, never something configuration selects:
    # the host registers a zero-argument callable that returns the ``nn.Module``
    # a local ``.pth`` checkpoint belongs to, and configuration may only name an
    # already registered provider. Providers live on this instance, so a
    # provider added to one ``Registry.builtins()`` is invisible to the next.

    @property
    def pretrained_providers(self):
        return tuple(self._pretrained_providers)

    def pretrained_provider(self, name, build):
        """Register a trusted architecture builder for ``pretrained(..., provider=name)``."""
        if type(name) is not str or not name or len(name) > MAX_PROVIDER_NAME:
            raise HNDLError("E_REGISTRY", f"A pretrained provider name must be a string of 1-{MAX_PROVIDER_NAME} characters")
        if not callable(build):
            raise HNDLError("E_REGISTRY", f"Pretrained provider {name!r} must be a callable returning an nn.Module")
        if name in self._pretrained_providers:
            raise HNDLError("E_REGISTRY", f"Duplicate pretrained provider {name!r}")
        self._pretrained_providers[name] = build
        return build

    def pretrained_builder(self, name):
        """The registered builder for ``name``, or None."""
        return self._pretrained_providers.get(name)

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
    for name, arg in op.args.items():
        if name not in args and arg.has_default:
            args[name] = arg.default
    for name in op.required:
        if name not in args:
            raise HNDLError("E_ARGUMENT", f"{op.alias} requires {name}; supply it or select an applicable policy")
    normalized = {name: op.args[name].validate(value, name) for name, value in args.items()}
    if op.validate is not None:
        op.validate(normalized)
    return normalized
