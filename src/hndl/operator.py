"""The ``@operator`` declaration that every HNDL operation carries.

An operator is an ``nn.Module`` subclass decorated with its alias, a one-line
summary, a shape relation, scalar arguments with help text, and runnable
examples. The declaration is data: the resolver reads it without constructing
modules, the backend constructs the class from concrete arguments, and the
documentation generator renders it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, KW_ONLY
import inspect
import keyword
import math
import re
from types import MappingProxyType

from .errors import HNDLError

REQUIRED = object()
ELLIPSIS = "..."
MAX_PORTS = 32
MAX_ARGUMENTS = 64
MAX_SYMBOLS = 64
MAX_NAME_LENGTH = 64
MAX_DIMENSION_LITERAL = 2**31 - 1
MAX_INTEGER_ARGUMENT = 2**63 - 1
MAX_STRING_BYTES = 16_384
SUPPORTED_RANKS = (2, 4)
RESERVED = frozenset({"name", "policy", "init", "trainable"})
PAIR = "pair"
INTS = "ints"
_TYPES = {int: "int", float: "float", bool: "bool", str: "str", PAIR: "pair", INTS: "ints"}
_IDENT = re.compile(r"[a-z][a-z0-9_]*\Z")
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_PORT_SPEC = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)(\*?)\s*(?:\[([^\]]*)\])?\s*(?::\s*([A-Za-z0-9_]+))?\s*\Z")
_DTYPES = ("compute", "float32", "float16", "bfloat16", "int64", "int32", "bool")


def _registry_error(message):
    raise HNDLError("E_REGISTRY", message)


def _integer_in_range(value, name, minimum, maximum):
    if type(value) is not int:
        raise HNDLError("E_ARGUMENT", f"{name} must have type int")
    if abs(value) > MAX_INTEGER_ARGUMENT:
        raise HNDLError("E_RESOURCE", f"{name} exceeds the supported integer argument bound")
    if minimum is not None and value < minimum:
        raise HNDLError("E_ARGUMENT", f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise HNDLError("E_ARGUMENT", f"{name} must be <= {maximum}")
    return value


@dataclass(frozen=True)
class Arg:
    """One scalar operator argument.

    ``Arg(float, 1e-5, min=0, exclusive_min=True, help="...")`` describes a
    positive epsilon. Omit ``default`` to require the value; set
    ``inferable=True`` for a dimension the resolver may solve when omitted, and
    ``dim="D"`` to tie it to a shape symbol so it flows in both directions.
    Types are ``int``, ``float``, ``bool``, ``str``, ``"pair"`` (an int or two
    ints, normalized to a pair) or ``"ints"`` (a tuple of ints).
    """

    type: object
    default: object = field(default=REQUIRED, repr=False)
    _: KW_ONLY
    min: object = None
    max: object = None
    exclusive_min: bool = False
    exclusive_max: bool = False
    help: str = ""
    inferable: bool = False
    dim: object = None
    positional: bool = True
    choices: object = None

    def __post_init__(self):
        if self.type not in _TYPES:
            _registry_error("Arg type must be int, float, bool, str, 'pair', or 'ints'")
        if type(self.help) is not str or not self.help.strip():
            _registry_error("Every Arg needs a non-empty help string")
        if type(self.inferable) is not bool or type(self.positional) is not bool:
            _registry_error("Arg flags must be booleans")
        if type(self.exclusive_min) is not bool or type(self.exclusive_max) is not bool:
            _registry_error("Arg exclusive-bound flags must be booleans")
        if self.inferable and self.has_default:
            _registry_error("An inferable argument cannot also declare a default")
        if self.dim is not None and (type(self.dim) is not str or not _SYMBOL.fullmatch(self.dim)):
            _registry_error("Arg dim must name a shape symbol")
        if self.dim is not None and self.type is not int:
            _registry_error("Only int arguments can bind to a shape symbol")
        numeric = self.type in (int, float, PAIR, INTS)
        for label, value, exclusive in (("min", self.min, self.exclusive_min), ("max", self.max, self.exclusive_max)):
            if exclusive and value is None:
                _registry_error(f"exclusive_{label} requires {label}")
            if value is not None:
                if not numeric:
                    _registry_error("Numeric bounds apply only to numeric arguments")
                if type(value) not in (int, float) or not math.isfinite(value):
                    _registry_error(f"{label} must be a finite numeric bound")
                if type(value) is int and abs(value) > MAX_INTEGER_ARGUMENT:
                    _registry_error(f"{label} exceeds the supported integer bound")
        if self.min is not None and self.max is not None:
            if self.min > self.max or (self.min == self.max and (self.exclusive_min or self.exclusive_max)):
                _registry_error("Arg bounds describe an empty interval")
        if self.type in (int, PAIR, INTS):
            lower = -MAX_INTEGER_ARGUMENT if self.min is None else math.ceil(self.min)
            upper = MAX_INTEGER_ARGUMENT if self.max is None else math.floor(self.max)
            if self.exclusive_min and lower == self.min:
                lower += 1
            if self.exclusive_max and upper == self.max:
                upper -= 1
            if lower > upper:
                _registry_error("Arg bounds admit no supported integer")
        if self.choices is not None:
            if self.type is not str or not isinstance(self.choices, (tuple, list)) or not self.choices:
                _registry_error("choices apply to str arguments and must be a non-empty sequence")
            if any(type(choice) is not str for choice in self.choices):
                _registry_error("choices must be strings")
            object.__setattr__(self, "choices", tuple(self.choices))
        if self.has_default:
            try:
                object.__setattr__(self, "default", self.validate(self.default, "default"))
            except HNDLError as exc:
                _registry_error(f"Invalid argument default: {exc.message}")

    @property
    def has_default(self):
        return self.default is not REQUIRED

    @property
    def required(self):
        return not self.has_default and not self.inferable

    @property
    def type_name(self):
        return _TYPES[self.type]

    def _bounds(self, value, name):
        if self.min is not None and (value < self.min or self.exclusive_min and value == self.min):
            raise HNDLError("E_ARGUMENT", f"{name} must be {'>' if self.exclusive_min else '>='} {self.min}")
        if self.max is not None and (value > self.max or self.exclusive_max and value == self.max):
            raise HNDLError("E_ARGUMENT", f"{name} must be {'<' if self.exclusive_max else '<='} {self.max}")

    def _element(self, value, name):
        if type(value) is not int:
            raise HNDLError("E_ARGUMENT", f"{name} must be an integer or a sequence of integers")
        if abs(value) > MAX_INTEGER_ARGUMENT:
            raise HNDLError("E_RESOURCE", f"{name} exceeds the supported integer argument bound")
        self._bounds(value, name)
        return value

    def validate(self, value, name):
        if self.type is PAIR:
            if type(value) is int:
                return (self._element(value, name),) * 2
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return tuple(self._element(item, name) for item in value)
            raise HNDLError("E_ARGUMENT", f"{name} must be an integer or pair of integers")
        if self.type is INTS:
            if not isinstance(value, (list, tuple)):
                raise HNDLError("E_ARGUMENT", f"{name} must be a sequence of integers")
            if len(value) > MAX_PORTS:
                raise HNDLError("E_RESOURCE", f"{name} has too many entries")
            return tuple(self._element(item, name) for item in value)
        if self.type is float:
            if type(value) not in (float, int):
                raise HNDLError("E_ARGUMENT", f"{name} must be a finite float, not {type(value).__name__}")
            try:
                value = float(value)
            except (OverflowError, ValueError):
                raise HNDLError("E_ARGUMENT", f"{name} must be finite") from None
            if not math.isfinite(value):
                raise HNDLError("E_ARGUMENT", f"{name} must be finite")
        elif type(value) is not self.type:
            raise HNDLError("E_ARGUMENT", f"{name} must have type {self.type.__name__}")
        if self.type is int and abs(value) > MAX_INTEGER_ARGUMENT:
            raise HNDLError("E_RESOURCE", f"{name} exceeds the supported integer argument bound")
        if self.type is str:
            if len(value) > MAX_STRING_BYTES:
                raise HNDLError("E_RESOURCE", f"{name} exceeds {MAX_STRING_BYTES} UTF-8 bytes")
            try:
                length = len(value.encode("utf-8"))
            except UnicodeEncodeError:
                raise HNDLError("E_ARGUMENT", f"{name} must be valid UTF-8 text") from None
            if length > MAX_STRING_BYTES:
                raise HNDLError("E_RESOURCE", f"{name} exceeds {MAX_STRING_BYTES} UTF-8 bytes")
            if self.choices is not None and value not in self.choices:
                raise HNDLError("E_ARGUMENT", f"{name} must be one of {', '.join(self.choices)}")
        if self.type in (int, float):
            self._bounds(value, name)
        return value


@dataclass(frozen=True)
class Sym:
    """A positive node-local dimension symbol multiplied by a fixed integer."""

    name: str
    scale: int = 1


@dataclass(frozen=True)
class Port:
    name: str
    pattern: object = None   # tuple of "B" | Sym | int | ELLIPSIS, or None when unconstrained
    dtype: str = "compute"
    variadic: bool = False


@dataclass(frozen=True)
class Policy:
    """A named construction profile: hard requirements that fill omitted fields."""

    identity: str
    requires: Mapping

    def __post_init__(self):
        if type(self.identity) is not str or not self.identity:
            _registry_error("Policy identity must be a non-empty string")
        if not isinstance(self.requires, Mapping) or not self.requires:
            _registry_error("Policy requires must be a non-empty mapping")
        object.__setattr__(self, "requires", MappingProxyType(dict(self.requires)))


@dataclass(frozen=True)
class Example:
    """A runnable configuration snippet rendered into the operator's docs."""

    source: str
    input_shape: tuple
    output_shape: tuple
    note: str = ""

    def __post_init__(self):
        if type(self.source) is not str or not self.source.strip():
            _registry_error("Example source must be a non-empty configuration string")
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "output_shape", tuple(self.output_shape))


def _validate_name(name, kind):
    if type(name) is not str or len(name) > MAX_NAME_LENGTH or not _IDENT.fullmatch(name) or keyword.iskeyword(name):
        _registry_error(f"{kind} names must be non-keyword lowercase identifiers of at most {MAX_NAME_LENGTH} characters")
    if name in RESERVED:
        _registry_error(f"{kind} {name!r} conflicts with reserved frontend metadata")
    return name


def _parse_pattern(text, port):
    entries = [entry.strip() for entry in text.split(",")] if text.strip() else []
    if not entries:
        _registry_error(f"Shape pattern for {port} must list at least the batch axis")
    if len(entries) > 5:
        _registry_error(f"Shape pattern for {port} exceeds the supported rank")
    dims = []
    seen_ellipsis = False
    for axis, entry in enumerate(entries):
        if entry == ELLIPSIS:
            if axis == 0 or seen_ellipsis:
                _registry_error(f"Shape pattern for {port} may use ... once, after the batch axis")
            seen_ellipsis = True
            dims.append(ELLIPSIS)
            continue
        if entry == "B":
            if axis != 0:
                _registry_error("The batch symbol B is only valid at axis 0")
            dims.append("B")
            continue
        if axis == 0:
            _registry_error(f"Shape pattern for {port} must start with B")
        scale, name = 1, entry
        if "*" in entry:
            left, _, right = entry.partition("*")
            left, right = left.strip(), right.strip()
            if not left.isdigit():
                _registry_error(f"Scaled dimension must be written as k*Name, got {entry!r}")
            scale, name = int(left), right
            if not 1 <= scale <= MAX_DIMENSION_LITERAL:
                _registry_error("Dimension scale must be a positive bounded integer")
        if name.isdigit():
            if scale != 1:
                _registry_error("Literal dimensions cannot be scaled")
            value = int(name)
            if not 1 <= value <= MAX_DIMENSION_LITERAL:
                _registry_error("Dimension literals must be positive bounded integers")
            dims.append(value)
            continue
        if name == "B" or not _SYMBOL.fullmatch(name) or len(name) > MAX_NAME_LENGTH:
            _registry_error(f"Dimension {name!r} must be an identifier other than B")
        dims.append(Sym(name, scale))
    if not seen_ellipsis and len(dims) not in SUPPORTED_RANKS:
        ranks = ", ".join(map(str, SUPPORTED_RANKS))
        _registry_error(f"Shape pattern for {port} must have rank {ranks} including batch, or use ...")
    return tuple(dims)


def parse_shape(text):
    """Parse ``"x[B, D_in], y[B, D_in] -> out[B, D_out]"`` into ports."""
    if type(text) is not str or "->" not in text:
        _registry_error("shape must be a string of the form 'inputs -> outputs'")
    left, arrow, right = text.partition("->")
    if "->" in right:
        _registry_error("shape must contain exactly one ->")
    sides = []
    for side, is_input in ((left, True), (right, False)):
        ports = []
        # Split on commas outside brackets.
        depth, start, pieces = 0, 0, []
        for index, char in enumerate(side):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            elif char == "," and depth == 0:
                pieces.append(side[start:index])
                start = index + 1
        pieces.append(side[start:])
        for piece in pieces:
            if not piece.strip():
                if len(pieces) == 1:
                    continue
                _registry_error("Empty port entry in shape")
            match = _PORT_SPEC.fullmatch(piece)
            if match is None:
                _registry_error(f"Invalid port specification {piece.strip()!r}")
            name, star, pattern, dtype = match.groups()
            _validate_name(name, "Port")
            if star and not is_input:
                _registry_error("Only input ports can be variadic")
            if dtype is not None and dtype not in _DTYPES:
                _registry_error(f"Unsupported port dtype {dtype!r}")
            ports.append(Port(name, None if pattern is None else _parse_pattern(pattern, name),
                              dtype or "compute", bool(star)))
        if not 1 <= len(ports) <= MAX_PORTS:
            _registry_error(f"Declare between 1 and {MAX_PORTS} {'input' if is_input else 'output'} ports")
        names = [port.name for port in ports]
        if len(set(names)) != len(names):
            _registry_error("Port names must be unique within their input/output namespace")
        sides.append(tuple(ports))
    inputs, outputs = sides
    if sum(port.variadic for port in inputs) > 1 or any(port.variadic for port in inputs) and len(inputs) != 1:
        _registry_error("A variadic input must be the sole declared input port")
    symbols = {dim.name for port in (*inputs, *outputs) if port.pattern
               for dim in port.pattern if isinstance(dim, Sym)}
    if len(symbols) > MAX_SYMBOLS:
        _registry_error(f"shape exceeds {MAX_SYMBOLS} distinct dimension symbols")
    return inputs, outputs, frozenset(symbols)


@dataclass(frozen=True)
class Operator:
    """The registered declaration of one operation and its module class."""

    alias: str
    identity: str
    version: int
    summary: str
    doc: str
    category: str
    inputs: tuple
    outputs: tuple
    args: Mapping
    symbols: frozenset
    shape_text: str
    relation: object = None
    relation_text: str = ""
    examples: tuple = ()
    module: object = None
    positional_rest: object = None
    policies: Mapping = field(default_factory=dict)
    validate: object = None
    finalize: object = None
    expand: object = None
    reference: object = None
    init_symbols: tuple = ()
    init_shapes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))
        object.__setattr__(self, "policies", MappingProxyType(dict(self.policies)))

    @property
    def key(self):
        return f"{self.identity}@{self.version}"

    @property
    def input_ports(self):
        return tuple(port.name for port in self.inputs)

    @property
    def output_ports(self):
        return tuple(port.name for port in self.outputs)

    @property
    def variadic(self):
        return self.inputs[0].name if self.inputs and self.inputs[0].variadic else None

    def input_ports_for(self, args):
        if self.variadic is None:
            return self.input_ports
        return tuple(f"{self.variadic}{i}" for i in range(args["input_count"]))

    @property
    def positional_names(self):
        return tuple(name for name, arg in self.args.items() if arg.positional)

    @property
    def defaults(self):
        return {name: arg.default for name, arg in self.args.items() if arg.has_default}

    @property
    def required(self):
        return tuple(name for name, arg in self.args.items() if arg.required)

    @property
    def is_macro(self):
        return self.expand is not None


def _check_init_signature(cls, args, symbols):
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return (), ()
    parameters = list(signature.parameters.values())[1:]
    accepts_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    names = {p.name for p in parameters if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    if not accepts_var_keyword:
        missing = [name for name in args if name not in names]
        if missing:
            _registry_error(f"{cls.__name__}.__init__ does not accept declared arguments: {', '.join(missing)}")
    requested_symbols, requested_shapes = [], []
    for parameter in parameters:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.name in args:
            continue
        if parameter.name in ("input_shapes", "output_shapes"):
            requested_shapes.append(parameter.name)
        elif parameter.name in symbols:
            requested_symbols.append(parameter.name)
        elif parameter.default is inspect.Parameter.empty:
            _registry_error(f"{cls.__name__}.__init__ parameter {parameter.name!r} is neither a declared argument, "
                            "a shape symbol, nor input_shapes/output_shapes")
    return tuple(requested_symbols), tuple(requested_shapes)


def make_operator(cls, alias, *, identity=None, version=1, summary, shape, args=None, examples=(),
                  category="other", relation=None, shape_text=None, positional_rest=None, policies=None,
                  validate=None, finalize=None, expand=None, reference=None):
    if not isinstance(alias, str) or not alias.isidentifier() or alias in ("x", "out") or alias.startswith("_"):
        _registry_error("Operator alias must be an identifier other than x/out without a leading underscore")
    if keyword.iskeyword(alias):
        _registry_error(f"Keyword cannot be an operator alias: {alias}")
    identity = alias if identity is None else identity
    if type(identity) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", identity):
        _registry_error("Operator identity must be lowercase dotted identifiers, for example example.silu")
    if type(version) is not int or isinstance(version, bool) or version < 1:
        _registry_error("Operator version must be a positive integer")
    if type(summary) is not str or not summary.strip():
        _registry_error(f"Operator {alias!r} needs a one-line summary")
    if type(category) is not str or not _IDENT.fullmatch(category.replace("-", "_")):
        _registry_error("Operator category must be a short lowercase identifier")
    inputs, outputs, symbols = parse_shape(shape)
    if relation is not None and not callable(relation):
        _registry_error("relation must be a callable receiving the node view")
    if shape_text is not None and type(shape_text) is not str:
        _registry_error("shape_text must be a string describing the relation")
    for hook, label in ((validate, "validate"), (finalize, "finalize"), (expand, "expand"), (reference, "reference")):
        if hook is not None and not callable(hook):
            _registry_error(f"{label} must be callable")
    args = {} if args is None else args
    if not isinstance(args, Mapping) or len(args) > MAX_ARGUMENTS:
        _registry_error(f"args must be a mapping with at most {MAX_ARGUMENTS} fields")
    port_names = {port.name for port in (*inputs, *outputs)}
    checked = {}
    for name, arg in args.items():
        _validate_name(name, "Argument")
        if name in port_names:
            _registry_error(f"Argument {name!r} conflicts with a port")
        if type(arg) is not Arg:
            _registry_error(f"Argument {name!r} must be an Arg declaration")
        if arg.dim is not None and arg.dim not in symbols:
            _registry_error(f"Argument {name!r} binds to unknown shape symbol {arg.dim!r}")
        checked[name] = arg
    bound = [name for name, arg in checked.items() if arg.dim is not None]
    if len({checked[name].dim for name in bound}) != len(bound):
        _registry_error("Two arguments cannot bind to the same shape symbol")
    if positional_rest is not None:
        if positional_rest not in checked or checked[positional_rest].type is not INTS:
            _registry_error("positional_rest must name an 'ints' argument")
    variadic = inputs[0].variadic if inputs else False
    if variadic and ("input_count" not in checked or checked["input_count"].type is not int):
        _registry_error("A variadic operator must declare an int argument named input_count")
    policies = {} if policies is None else policies
    if not isinstance(policies, Mapping):
        _registry_error("policies must be a mapping of alias to Policy")
    for policy_alias, policy in policies.items():
        _validate_name(policy_alias, "Policy")
        if type(policy) is not Policy:
            _registry_error("policies values must be Policy declarations")
        unknown = set(policy.requires) - set(checked)
        if unknown:
            _registry_error(f"Policy {policy_alias!r} requires unknown arguments: {', '.join(sorted(unknown))}")
    normalized_examples = []
    for example in examples:
        if type(example) is Example:
            normalized_examples.append(example)
        elif isinstance(example, (tuple, list)) and len(example) in (3, 4):
            normalized_examples.append(Example(*example))
        else:
            _registry_error("examples must be Example(source, input_shape, output_shape) entries")
    if expand is None:
        if not (inspect.isclass(cls) and hasattr(cls, "forward")):
            _registry_error("@operator decorates an nn.Module subclass with a forward method")
        init_symbols, init_shapes = _check_init_signature(cls, checked, symbols)
    else:
        init_symbols, init_shapes = (), ()
    doc = inspect.getdoc(cls) or ""
    return Operator(
        alias=alias, identity=identity, version=version, summary=summary.strip(), doc=doc, category=category,
        inputs=inputs, outputs=outputs, args=checked, symbols=symbols, shape_text=shape.strip(),
        relation=relation, relation_text=(shape_text or "").strip(), examples=tuple(normalized_examples),
        module=cls, positional_rest=positional_rest,
        policies=policies, validate=validate, finalize=finalize, expand=expand, reference=reference,
        init_symbols=init_symbols, init_shapes=init_shapes,
    )


def operator(alias, **declaration):
    """Declare an operation on the ``nn.Module`` class that implements it.

    The decorated class gains an ``__hndl_operator__`` attribute. Classes in
    ``hndl.operators`` are discovered as built-ins; other classes are added to
    a registry with ``registry.add(cls)`` or declared with
    ``@registry.operator(...)``.
    """

    def decorate(cls):
        cls.__hndl_operator__ = make_operator(cls, alias, **declaration)
        return cls

    return decorate


class NodeView:
    """The bounded solver facade a shape relation sees for one node."""

    def __init__(self, solver, node, spec):
        self._solver = solver
        self._node = node
        self.spec = spec
        self.args = solver.args[node.id]
        refs = dict(node.inputs)
        refs.update({port: f"node:{node.id}/{port}" for port in node.outputs})
        self._refs = refs
        self.inputs = tuple(node.inputs)
        self.outputs = tuple(node.outputs)

    def _ref(self, port):
        try:
            return self._refs[port]
        except KeyError:
            raise HNDLError("E_REGISTRY", f"Shape relation referenced unknown port {port!r}") from None

    def shape(self, port):
        """The known dimensions of a port (None entries unknown), or None."""
        shape = self._solver.shapes.get(self._ref(port))
        return None if shape is None else tuple(shape)

    def rank(self, port, rank):
        return tuple(self._solver.rank(self._ref(port), rank))

    def axis(self, port, axis, value, code="E_CONSTRAINT"):
        self._solver.axis(self._ref(port), axis, value, code)

    def equal(self, left, right):
        self._solver.equal(self._ref(left), self._ref(right))

    def arg(self, name, value):
        self._solver.argument(self.args, name, value)

    def product(self, left, right):
        self._solver.product(self._ref(left), self._ref(right))

    def interval(self, port, axis, lower, upper):
        self._solver.interval(self._ref(port), axis, lower, upper)

    def error(self, code, message):
        self._solver.error(code, message)


__all__ = ["Arg", "Example", "NodeView", "Operator", "Policy", "Port", "Sym", "operator", "parse_shape",
           "make_operator", "REQUIRED", "PAIR", "INTS", "ELLIPSIS", "SUPPORTED_RANKS"]
