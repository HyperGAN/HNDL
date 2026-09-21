"""Pure normalization of persisted construction metadata.

Author metadata uses ``init={path: constant}`` and ``trainable=bool|mapping``.
Plans store explicit canonical records. These functions never import torch or
inspect parameters; parameter existence and alias checks belong to the backend.
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
_SEGMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)\Z")


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


def normalize_initialization(init=UNSET):
    """Normalize raw author metadata; omission and an empty map are equal."""
    overrides = {} if init is UNSET else _mapping(init, code="E_INITIALIZATION", label="init")
    return {"kind": INITIALIZER,
            "overrides": {path: _float32(value, path=path) for path, value in overrides.items()}}


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
    """Validate a canonical persisted record without rounding or filling it."""
    if not isinstance(value, Mapping) or len(value) != 2 or set(value) != {"kind", "overrides"}:
        raise HNDLError("E_INITIALIZATION", "Canonical initialization requires exactly kind and overrides")
    if type(value["kind"]) is not str or value["kind"] != INITIALIZER:
        raise HNDLError("E_STATE_VERSION", f"Unsupported initializer version; expected {INITIALIZER}")
    overrides = _mapping(value["overrides"], code="E_INITIALIZATION", label="Initialization overrides")
    for path, constant in overrides.items():
        rounded = _float32(constant, path=path)
        if type(constant) is not float or rounded != constant:
            raise HNDLError("E_INITIALIZATION", f"Saved initializer for {path!r} must be a canonical float32 value; normalize author metadata before constructing nodes")
    return {"kind": INITIALIZER, "overrides": overrides}


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
