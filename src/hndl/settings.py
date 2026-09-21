"""Pure normalization of persisted construction metadata.

Author metadata uses ``init={path: constant|scheme}`` and
``trainable=bool|mapping``. Plans store explicit canonical records. These
functions never import torch or inspect parameters; parameter existence and
alias checks belong to the backend.
"""

from collections.abc import Mapping
import math
import re
import struct

from .errors import HNDLError


UNSET = object()
MAX_OVERRIDES = 256
MAX_PARAMETER_PATH = 256
INITIALIZER = "torch_default@1"
INITIALIZER_SCHEMES = "torch_default@2"
INITIALIZER_VERSIONS = (INITIALIZER, INITIALIZER_SCHEMES)
_SEGMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)\Z")

# Random initializer schemes, one per ``torch.nn.init`` function, with torch's
# own keyword names and defaults. Float fields round to float32 exactly like
# constant overrides do; the two string fields take a fixed choice set.
MODES = ("fan_in", "fan_out")
NONLINEARITIES = ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
                  "conv_transpose2d", "conv_transpose3d", "sigmoid", "tanh",
                  "relu", "leaky_relu", "selu")
_CHOICES = {"mode": MODES, "nonlinearity": NONLINEARITIES}
SCHEMES = {
    "xavier_uniform": (("gain", 1.0),),
    "xavier_normal": (("gain", 1.0),),
    "kaiming_uniform": (("a", 0.0), ("mode", "fan_in"), ("nonlinearity", "leaky_relu")),
    "kaiming_normal": (("a", 0.0), ("mode", "fan_in"), ("nonlinearity", "leaky_relu")),
    "truncated_normal": (("mean", 0.0), ("std", 1.0), ("a", -2.0), ("b", 2.0)),
    "normal": (("mean", 0.0), ("std", 1.0)),
    "uniform": (("a", 0.0), ("b", 1.0)),
    "orthogonal": (("gain", 1.0),),
}
# Schemes that ``torch.nn.init`` can only apply to a parameter of rank >= 2.
MATRIX_SCHEMES = frozenset({"xavier_uniform", "xavier_normal", "kaiming_uniform",
                            "kaiming_normal", "orthogonal"})
SCHEME_NAMES = frozenset(SCHEMES)


def _mapping(value, *, code, label):
    if not isinstance(value, Mapping):
        raise HNDLError(code, f"{label} must be a mapping of exact parameter paths")
    if len(value) > MAX_OVERRIDES:
        raise HNDLError("E_RESOURCE", f"{label} exceeds {MAX_OVERRIDES} overrides")
    result = {}
    for path, item in value.items():
        if (type(path) is not str or not 1 <= len(path) <= MAX_PARAMETER_PATH
                or any(not _SEGMENT.fullmatch(segment) for segment in path.split("."))):
            raise HNDLError(code, f"{label} keys must be exact dotted parameter paths of at most {MAX_PARAMETER_PATH} characters")
        result[path] = item
    return result


def _float32(value, *, path):
    if type(value) not in (int, float):
        raise HNDLError("E_INITIALIZATION", f"Initializer for {path!r} must be a finite numeric constant, not a boolean")
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite")
        rounded = struct.unpack("!f", struct.pack("!f", number))[0]
    except (OverflowError, ValueError, struct.error):
        raise HNDLError("E_INITIALIZATION", f"Initializer for {path!r} must be finite after conversion to float32") from None
    if not math.isfinite(rounded):
        raise HNDLError("E_INITIALIZATION", f"Initializer for {path!r} must be finite after conversion to float32")
    # Match ordinary float32 rounding, including subnormal underflow and the
    # sign of zero. JSON stores this exactly representable value as a float.
    return rounded


def initializer(kind, arguments=None, *, path=None):
    """Build one canonical scheme record from author keyword arguments.

    Both frontends funnel through this single builder: ``hndl.initializers``
    for trusted Python and the declarative parser's ``init_call`` nodes.
    """
    label = "an initializer" if path is None else f"Initializer for {path!r}"
    if type(kind) is not str or kind not in SCHEMES:
        raise HNDLError("E_INITIALIZATION",
                        f"Unknown initializer {kind!r}; choose one of {', '.join(sorted(SCHEMES))}")
    arguments = {} if arguments is None else arguments
    if not isinstance(arguments, Mapping):
        raise HNDLError("E_INITIALIZATION", f"{label}: {kind} arguments must be a mapping")
    fields = dict(SCHEMES[kind])
    unknown = sorted(name for name in arguments if name not in fields)
    if unknown:
        raise HNDLError("E_INITIALIZATION",
                        f"{label}: {kind} does not accept {', '.join(unknown)}; "
                        f"it takes {', '.join(fields)}")
    record = {"kind": kind}
    for name, default in SCHEMES[kind]:
        value = arguments.get(name, default)
        if name in _CHOICES:
            if type(value) is not str or value not in _CHOICES[name]:
                raise HNDLError("E_INITIALIZATION",
                                f"{label}: {kind} {name} must be one of {', '.join(_CHOICES[name])}")
            record[name] = value
            continue
        record[name] = _float32(value, path=f"{kind} {name}" if path is None else f"{path} ({kind} {name})")
    return record


def _validate_scheme(value, *, path):
    """Validate a canonical persisted scheme record without rewriting it."""
    kind = value.get("kind") if isinstance(value, Mapping) else None
    if type(kind) is not str or kind not in SCHEMES:
        raise HNDLError("E_INITIALIZATION",
                        f"Saved initializer for {path!r} must name one of {', '.join(sorted(SCHEMES))}")
    fields = dict(SCHEMES[kind])
    if set(value) != {"kind", *fields}:
        raise HNDLError("E_INITIALIZATION",
                        f"Saved {kind} initializer for {path!r} requires exactly kind, {', '.join(fields)}")
    for name in fields:
        item = value[name]
        if name in _CHOICES:
            if type(item) is not str or item not in _CHOICES[name]:
                raise HNDLError("E_INITIALIZATION",
                                f"Saved {kind} initializer for {path!r} has an unsupported {name}")
            continue
        rounded = _float32(item, path=f"{path} ({kind} {name})")
        if type(item) is not float or rounded != item:
            raise HNDLError("E_INITIALIZATION",
                            f"Saved {kind} initializer for {path!r} must store a canonical float32 {name}; "
                            "normalize author metadata before constructing nodes")
    return {"kind": kind, **{name: value[name] for name in fields}}


def normalize_initialization(init=UNSET):
    """Normalize raw author metadata; omission and an empty map are equal."""
    overrides = {} if init is UNSET else _mapping(init, code="E_INITIALIZATION", label="init")
    normalized, schemes = {}, False
    for path, value in overrides.items():
        if isinstance(value, Mapping):
            normalized[path] = initializer(value.get("kind"),
                                           {name: item for name, item in value.items() if name != "kind"},
                                           path=path)
            schemes = True
        else:
            normalized[path] = _float32(value, path=path)
    # Constant-only plans keep emitting torch_default@1 byte for byte.
    return {"kind": INITIALIZER_SCHEMES if schemes else INITIALIZER, "overrides": normalized}


def normalize_trainability(trainable=UNSET):
    """None in the canonical default preserves constructor requires_grad."""
    if trainable is UNSET:
        return {"default": None, "overrides": {}}
    if type(trainable) is bool:
        return {"default": trainable, "overrides": {}}
    overrides = _mapping(trainable, code="E_TRAINABILITY", label="trainable")
    for path, value in overrides.items():
        if type(value) is not bool:
            raise HNDLError("E_TRAINABILITY", f"Trainability override for {path!r} must be a boolean")
    return {"default": None, "overrides": overrides}


def normalize_settings(*, init=UNSET, trainable=UNSET):
    """Return the canonical initialization/trainability pair for a new node."""
    return normalize_initialization(init), normalize_trainability(trainable)


def validate_initialization(value):
    """Validate a canonical persisted record without rounding or filling it.

    Both versions are accepted forever: ``torch_default@1`` carries constants
    only, ``torch_default@2`` adds scheme records and requires at least one.
    """
    if not isinstance(value, Mapping) or len(value) != 2 or set(value) != {"kind", "overrides"}:
        raise HNDLError("E_INITIALIZATION", "Canonical initialization requires exactly kind and overrides")
    kind = value["kind"]
    if type(kind) is not str or kind not in INITIALIZER_VERSIONS:
        raise HNDLError("E_STATE_VERSION",
                        f"Unsupported initializer version; expected {' or '.join(INITIALIZER_VERSIONS)}")
    overrides = _mapping(value["overrides"], code="E_INITIALIZATION", label="Initialization overrides")
    result, schemes = {}, False
    for path, constant in overrides.items():
        if isinstance(constant, Mapping):
            if kind == INITIALIZER:
                raise HNDLError("E_INITIALIZATION",
                                f"Saved initializer for {path!r} selects a scheme, which requires {INITIALIZER_SCHEMES}")
            result[path] = _validate_scheme(constant, path=path)
            schemes = True
            continue
        rounded = _float32(constant, path=path)
        if type(constant) is not float or rounded != constant:
            raise HNDLError("E_INITIALIZATION", f"Saved initializer for {path!r} must be a canonical float32 value; normalize author metadata before constructing nodes")
        result[path] = constant
    if kind == INITIALIZER_SCHEMES and not schemes:
        raise HNDLError("E_INITIALIZATION",
                        f"{INITIALIZER_SCHEMES} requires at least one scheme override; "
                        f"constant-only initialization is {INITIALIZER}")
    return {"kind": kind, "overrides": result}


def validate_trainability(value):
    """Validate a canonical mask without changing its default or overrides."""
    if not isinstance(value, Mapping) or len(value) != 2 or set(value) != {"default", "overrides"}:
        raise HNDLError("E_TRAINABILITY", "Canonical trainability requires exactly default and overrides")
    default = value["default"]
    if default is not None and type(default) is not bool:
        raise HNDLError("E_TRAINABILITY", "Canonical trainability default must be None or a boolean")
    overrides = _mapping(value["overrides"], code="E_TRAINABILITY", label="Trainability overrides")
    for path, trainable in overrides.items():
        if type(trainable) is not bool:
            raise HNDLError("E_TRAINABILITY", f"Trainability override for {path!r} must be a boolean")
    return {"default": default, "overrides": overrides}
