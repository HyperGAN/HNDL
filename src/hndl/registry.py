"""Explicit operator bindings, pure argument schemas, and backend separation."""

from dataclasses import dataclass, field
from collections.abc import Mapping
import math
import re

from .errors import HNDLError
from .types import freeze


def preserves_shape(*args, **kwargs):
    """Marker for bidirectional equality of the sole input and output shape."""
    return "preserves_shape"


@dataclass(frozen=True)
class OpSpec:
    alias: str
    identity: str
    version: int = 1
    input_ports: tuple = ("x",)
    output_ports: tuple = ("out",)
    argument_names: tuple = ()
    defaults: Mapping = field(default_factory=dict)
    shape: object = None
    max_state_bytes: object = 0
    state_version: int = 1
    variadic_inputs: bool = False
    required: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "defaults", freeze(self.defaults))

    @property
    def key(self):
        return f"{self.identity}@{self.version}"


class Registry:
    def __init__(self):
        self._aliases = {}
        self._identities = {}
        self.backends = {}

    @property
    def aliases(self):
        return tuple(self._aliases)

    @property
    def ops(self):
        from .capture import OperatorNamespace
        return OperatorNamespace(self)

    @classmethod
    def builtins(cls):
        registry = cls()
        def add(alias, identity=None, **kwargs):
            spec = OpSpec(alias, identity or alias, shape=identity or alias, **kwargs)
            registry._add(spec)
        add("linear", argument_names=("out_features",), defaults={"bias": True})
        add("reshape", argument_names=("shape",))
        add("flatten")
        add("relu")
        add("leaky_relu", argument_names=("negative_slope",), defaults={"negative_slope": 0.01})
        add("tanh")
        conv_defaults = dict(stride=1, padding=0, dilation=1, groups=1, bias=True)
        add("conv", "conv2d", argument_names=("out_channels",), defaults=conv_defaults, required=("kernel_size",))
        add("deconv", "conv_transpose2d", argument_names=("out_channels",),
            defaults={**conv_defaults, "output_padding": 0}, required=("kernel_size",))
        add("group_norm", argument_names=("num_groups",), defaults={"eps": 1e-5, "affine": True}, required=("num_groups",))
        add("add", input_ports=("a", "b"))
        add("split", output_ports=("first", "rest"), argument_names=("size",), defaults={"dim": 1})
        add("concat", input_ports=(), variadic_inputs=True, defaults={"axis": 1}, required=("input_count",))
        return registry

    def _add(self, spec):
        if (not isinstance(spec.alias, str) or not spec.alias.isidentifier()
                or spec.alias in ("x", "out") or spec.alias.startswith("__")):
            raise HNDLError("E_REGISTRY", "Operator alias must be an identifier other than x/out or a dunder name")
        import keyword
        if keyword.iskeyword(spec.alias):
            raise HNDLError("E_REGISTRY", f"Keyword cannot be an operator alias: {spec.alias}")
        if spec.alias in self._aliases or spec.key in self._identities:
            raise HNDLError("E_REGISTRY", f"Duplicate operator alias or identity: {spec.alias} / {spec.key}")
        self._aliases[spec.alias] = spec
        self._identities[spec.key] = spec
        return spec

    def register(self, alias, *, identity, version, shape, max_state_bytes, state_version=1):
        if not isinstance(identity, str) or not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", identity):
            raise HNDLError("E_REGISTRY", "Custom identities must be namespaced, for example example.silu")
        if type(version) is not int or version < 1 or type(state_version) is not int or state_version < 1:
            raise HNDLError("E_REGISTRY", "Operator and state versions must be positive integers")
        if shape is not preserves_shape:
            raise HNDLError("E_UNRESOLVED", "This initial core admits custom unary preserves_shape providers only")
        if type(max_state_bytes) is not int or max_state_bytes < 0:
            raise HNDLError("E_REGISTRY", "max_state_bytes must be an explicit nonnegative integer")
        return self._add(OpSpec(alias, identity, version, shape=shape,
                                max_state_bytes=max_state_bytes, state_version=state_version))

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


def _integer(value, field_name, minimum=1):
    if type(value) is not int or value < minimum:
        raise HNDLError("E_ARGUMENT", f"{field_name} must be an integer >= {minimum}, got {value!r}")
    if value > 2**31 - 1:
        raise HNDLError("E_RESOURCE", f"{field_name} exceeds the supported integer bound")
    return value


def _pair(value, field_name, minimum=1):
    if type(value) is int:
        return (_integer(value, field_name, minimum),) * 2
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(_integer(part, field_name, minimum) for part in value)
    raise HNDLError("E_ARGUMENT", f"{field_name} must be an integer or pair of integers")


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
    if op.identity == "reshape":
        if positional:
            if "shape" in args:
                raise HNDLError("E_ARGUMENT", "reshape shape supplied twice")
            if len(positional) == 1 and isinstance(positional[0], (tuple, list)):
                args["shape"] = tuple(positional[0])
            else:
                args["shape"] = positional
        args.setdefault("shape", ())
    else:
        if len(positional) > len(op.argument_names):
            raise HNDLError("E_ARGUMENT", f"{op.alias} accepts at most {len(op.argument_names)} author positional arguments")
        for name, value in zip(op.argument_names, positional):
            if name in args:
                raise HNDLError("E_ARGUMENT", f"{name} supplied positionally and by keyword")
            args[name] = value
    allowed = set(op.argument_names) | set(op.defaults) | set(op.required)
    if op.identity == "linear":
        allowed.add("in_features")
    if op.identity in ("conv2d", "conv_transpose2d"):
        allowed.add("in_channels")
    if op.identity == "group_norm":
        allowed.add("num_channels")
    if op.identity == "reshape":
        allowed.add("shape")
    unknown = set(args) - allowed
    if unknown:
        raise HNDLError("E_ARGUMENT", f"Unknown arguments for {op.alias}: {', '.join(sorted(unknown))}")
    if policy is not None:
        if policy not in ("up2", "spatial.up2_transpose@1") or op.identity != "conv_transpose2d":
            raise HNDLError("E_POLICY_CONFLICT", f"Policy {policy!r} does not apply to {op.key}")
        profile = dict(kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0, groups=1)
        for name, value in profile.items():
            if name in args:
                actual = args[name] if name == "groups" else _pair(args[name], name, 0 if name in ("padding", "output_padding") else 1)
                expected = value if name == "groups" else (value, value)
                if actual != expected:
                    raise HNDLError("E_POLICY_CONFLICT", f"Explicit {name}={args[name]!r} conflicts with up2 requirement {value}")
            args[name] = value
    for name, value in op.defaults.items():
        args.setdefault(name, value)
    for name in op.required:
        if name not in args:
            raise HNDLError("E_ARGUMENT", f"{op.alias} requires {name}; supply it or select an applicable policy")
    integer_fields = {"out_features", "in_features", "in_channels", "out_channels", "groups", "num_groups", "num_channels", "size", "dim", "input_count"}
    for name in integer_fields & args.keys():
        args[name] = _integer(args[name], name)
    if "input_count" in args and args["input_count"] < 2:
        raise HNDLError("E_ARGUMENT", "concat requires at least two inputs")
    if "axis" in args:
        args["axis"] = _integer(args["axis"], "axis")
    for name in ("kernel_size", "stride", "padding", "dilation", "output_padding"):
        if name in args:
            args[name] = _pair(args[name], name, 0 if name in ("padding", "output_padding") else 1)
    for name in ("bias", "affine"):
        if name in args and type(args[name]) is not bool:
            raise HNDLError("E_ARGUMENT", f"{name} must be a boolean")
    for name in ("eps", "negative_slope"):
        if name in args:
            value = args[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise HNDLError("E_ARGUMENT", f"{name} must be a finite number")
            if name == "eps" and value <= 0:
                raise HNDLError("E_ARGUMENT", "eps must be positive")
            args[name] = float(value)
    if "shape" in args:
        if not isinstance(args["shape"], (tuple, list)) or len(args["shape"]) > 3:
            raise HNDLError("E_ARGUMENT", "reshape shape must be a prefix of at most three dimensions")
        args["shape"] = tuple(_integer(item, "shape") for item in args["shape"])
    if "output_padding" in args:
        if any(o >= s and o >= d for o, s, d in zip(args["output_padding"], args["stride"], args["dilation"])):
            raise HNDLError("E_ARGUMENT", "output_padding must be smaller than stride or dilation on each axis")
    return args

