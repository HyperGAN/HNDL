"""Bounded, declarative custom-operator schemas; no executable shape callbacks."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import keyword
import math
import re
from types import MappingProxyType

from .errors import HNDLError


MAX_PORTS = 32
MAX_ARGUMENTS = 64
MAX_SYMBOLS = 64
MAX_NAME_LENGTH = 64
MAX_DIMENSION_LITERAL = 2**31 - 1
MAX_INTEGER_ARGUMENT = 2**63 - 1
MAX_STRING_BYTES = 16_384
_MISSING = object()
_PORT = re.compile(r"[a-z][a-z0-9_]*\Z")
_DIM = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _schema_error(message):
    raise HNDLError("E_REGISTRY", message)


def validate_ports(ports, *, inputs):
    """Validate explicit port ordering without consuming arbitrary iterators."""
    if not isinstance(ports, (tuple, list)) or not 1 <= len(ports) <= MAX_PORTS:
        _schema_error(f"Declare between 1 and {MAX_PORTS} {'input' if inputs else 'output'} ports")
    result = tuple(ports)
    for name in result:
        if (type(name) is not str or len(name) > MAX_NAME_LENGTH or not _PORT.fullmatch(name)
                or keyword.iskeyword(name)):
            _schema_error("Port names must be non-keyword lowercase identifiers of at most 64 characters")
        if inputs and name in ("name", "policy"):
            _schema_error(f"Input port {name!r} conflicts with reserved frontend metadata")
    if len(set(result)) != len(result):
        _schema_error("Port names must be unique within their input/output namespace")
    return result


@dataclass(frozen=True)
class Argument:
    """One scalar argument; absent default means required unless contradicted.

    ``Argument(float, default=1e-5, minimum=0, exclusive_minimum=True)``
    describes a positive finite epsilon. Numeric bounds are inclusive unless
    their corresponding exclusive flag is true. Integer values never include
    booleans. Optional arguments must have an explicit, schema-valid default.
    """

    type: type
    default: object = field(default=_MISSING, repr=False)
    required: object = None
    minimum: object = None
    maximum: object = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False

    def __post_init__(self):
        if not any(self.type is candidate for candidate in (int, float, bool, str)):
            _schema_error("Argument type must be int, float, bool, or str")
        if self.required is not None and type(self.required) is not bool:
            _schema_error("Argument required must be a boolean when supplied")
        if type(self.exclusive_minimum) is not bool or type(self.exclusive_maximum) is not bool:
            _schema_error("Argument exclusive-bound flags must be booleans")
        required = not self.has_default if self.required is None else self.required
        if required and self.has_default:
            _schema_error("A required argument cannot also declare a default")
        if not required and not self.has_default:
            _schema_error("An optional argument must declare an explicit default")
        object.__setattr__(self, "required", required)
        for label, value, exclusive in (
            ("minimum", self.minimum, self.exclusive_minimum),
            ("maximum", self.maximum, self.exclusive_maximum),
        ):
            if exclusive and value is None:
                _schema_error(f"exclusive_{label} requires {label}")
            if value is not None:
                if self.type not in (int, float):
                    _schema_error("Numeric bounds apply only to int/float arguments")
                if type(value) not in (int, float):
                    _schema_error(f"{label} must be a finite numeric bound")
                if type(value) is int and abs(value) > MAX_INTEGER_ARGUMENT:
                    _schema_error(f"{label} exceeds the supported integer bound")
                if not math.isfinite(value):
                    _schema_error(f"{label} must be finite")
        if self.minimum is not None and self.maximum is not None:
            if (self.minimum > self.maximum
                    or self.minimum == self.maximum and (self.exclusive_minimum or self.exclusive_maximum)):
                _schema_error("Argument bounds describe an empty interval")
        if self.type is int:
            lower = -MAX_INTEGER_ARGUMENT if self.minimum is None else math.ceil(self.minimum)
            upper = MAX_INTEGER_ARGUMENT if self.maximum is None else math.floor(self.maximum)
            if self.exclusive_minimum and lower == self.minimum:
                lower += 1
            if self.exclusive_maximum and upper == self.maximum:
                upper -= 1
            if max(lower, -MAX_INTEGER_ARGUMENT) > min(upper, MAX_INTEGER_ARGUMENT):
                _schema_error("Argument bounds admit no supported integer")
        if self.has_default:
            try:
                object.__setattr__(self, "default", self.validate(self.default, "default"))
            except HNDLError as exc:
                _schema_error(f"Invalid argument default: {exc.message}")

    @property
    def has_default(self):
        return self.default is not _MISSING

    def validate(self, value, name):
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
        if self.minimum is not None and (value < self.minimum or self.exclusive_minimum and value == self.minimum):
            sign = ">" if self.exclusive_minimum else ">="
            raise HNDLError("E_ARGUMENT", f"{name} must be {sign} {self.minimum}")
        if self.maximum is not None and (value > self.maximum or self.exclusive_maximum and value == self.maximum):
            sign = "<" if self.exclusive_maximum else "<="
            raise HNDLError("E_ARGUMENT", f"{name} must be {sign} {self.maximum}")
        return value


@dataclass(frozen=True)
class Dim:
    """A positive node-local dimension variable multiplied by a fixed integer."""

    name: str
    scale: int = 1

    def __post_init__(self):
        if type(self.name) is not str or len(self.name) > MAX_NAME_LENGTH or not _DIM.fullmatch(self.name):
            _schema_error("Dimension names must be identifiers of at most 64 characters")
        if type(self.scale) is not int or not 1 <= self.scale <= MAX_DIMENSION_LITERAL:
            _schema_error("Dimension scale must be a positive bounded integer")
        if self.name == "B" and self.scale != 1:
            _schema_error("Batch symbol B cannot be scaled")


def _patterns(value, *, inputs):
    if not isinstance(value, Mapping):
        _schema_error("ShapeRule inputs/outputs must be ordered port-to-pattern mappings")
    if not 1 <= len(value) <= MAX_PORTS:
        _schema_error(f"ShapeRule must declare between 1 and {MAX_PORTS} ports per side")
    validate_ports(tuple(value), inputs=inputs)
    result = {}
    for port, shape in value.items():
        if not isinstance(shape, (tuple, list)) or len(shape) not in (2, 4):
            _schema_error(f"Shape pattern for {port} must have rank 2 (BF) or 4 (NCHW), including batch")
        dims = []
        for axis, value in enumerate(shape):
            if type(value) is str:
                value = Dim(value)
            if type(value) is Dim:
                if (axis == 0) != (value.name == "B"):
                    _schema_error("Each pattern must use B at batch only; feature/spatial axes cannot reuse B")
            elif type(value) is int:
                if axis == 0:
                    _schema_error("Batch pattern must be B, not a fixed literal")
                if not 1 <= value <= MAX_DIMENSION_LITERAL:
                    _schema_error("Dimension literals must be positive bounded integers")
            else:
                _schema_error("Pattern dimensions must be strings, Dim instances, or positive integer literals")
            dims.append(value)
        result[port] = tuple(dims)
    return MappingProxyType(result)


@dataclass(frozen=True)
class ShapeRule:
    """Port shape patterns imposing literal and shared/scaled-dimension equality.

    A rule is data, never a callback. Dimension variables are scoped to each
    operator node. Repeated names relate axes across all input/output patterns.
    ``B`` preserves the graph's batch dimension; rank fixes BF or NCHW layout.
    """

    inputs: Mapping
    outputs: Mapping

    def __post_init__(self):
        inputs, outputs = _patterns(self.inputs, inputs=True), _patterns(self.outputs, inputs=False)
        names = {dim.name for shape in (*inputs.values(), *outputs.values()) for dim in shape if isinstance(dim, Dim)}
        if len(names) > MAX_SYMBOLS:
            _schema_error(f"ShapeRule exceeds {MAX_SYMBOLS} distinct dimension symbols")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)


def argument_schema(arguments, *, input_ports, output_ports):
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping) or len(arguments) > MAX_ARGUMENTS:
        _schema_error(f"Arguments must be a mapping with at most {MAX_ARGUMENTS} fields")
    result = {}
    for name, schema in arguments.items():
        if (type(name) is not str or len(name) > MAX_NAME_LENGTH or not _PORT.fullmatch(name)
                or keyword.iskeyword(name)):
            _schema_error("Argument names must be non-keyword lowercase identifiers of at most 64 characters")
        if name in (*input_ports, *output_ports, "name", "policy"):
            _schema_error(f"Argument {name!r} conflicts with a port or reserved frontend metadata")
        if type(schema) is not Argument:
            _schema_error(f"Argument {name!r} must use an Argument schema, not a callback or raw default")
        result[name] = schema
    return MappingProxyType(result)
