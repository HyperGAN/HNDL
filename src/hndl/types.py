"""Immutable, torch-free graph records and portable resolved plans."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from collections.abc import Mapping

from .errors import HNDLError
from .settings import (normalize_initialization, normalize_trainability,
                       validate_initialization, validate_trainability)


EXTERNAL_INPUT = "x"
EXTERNAL_OUTPUT = "output"
MAX_EXTERNAL_PORTS = 32
BATCH = "B"
MAX_BATCH_MULTIPLE = 1024
_PORT_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_BATCH_MULTIPLE = re.compile(r"([1-9][0-9]{0,6})\*B\Z")


_CORRUPTED = ("Saved plan is corrupted or was edited by hand: its {field} does not match its contents. "
              "Restore the file, or re-resolve the plan from its source; to change a plan, edit it in "
              "Python and save it with to_json().")


def batch_multiple(value):
    """How many plan batches a symbolic batch entry stands for, or ``None``.

    Batch is the one dimension that stays symbolic after resolution. ``"B"`` is
    one plan batch; ``"k*B"`` is ``k`` of them, which is what joining tensors
    along axis 0 produces and what splitting them consumes. Each multiple has
    exactly one spelling --- ``"1*B"``, ``"0*B"``, ``"B*2"`` and a leading zero
    are all rejected --- so a plan that never touches the batch axis keeps the
    encoding, and therefore the digests, it had before multiples existed.
    """
    if value == BATCH:
        return 1
    if type(value) is not str:
        return None
    match = _BATCH_MULTIPLE.fullmatch(value)
    if match is None:
        return None
    multiple = int(match.group(1))
    return multiple if 2 <= multiple <= MAX_BATCH_MULTIPLE else None


def batch_symbol(multiple):
    """The canonical spelling of ``multiple`` plan batches, or ``None``."""
    if type(multiple) is not int or not 1 <= multiple <= MAX_BATCH_MULTIPLE:
        return None
    return BATCH if multiple == 1 else f"{multiple}*B"


def batch_units(value):
    """The batch count a batch entry carries: ``k`` for ``"k*B"``, the literal
    itself for a fixed integer batch, or ``None`` when it is neither."""
    if type(value) is int:
        return value if value > 0 else None
    return batch_multiple(value)


def batch_dimension(units, symbolic):
    """Rebuild a batch entry from a unit count, symbolic or fixed."""
    if type(units) is not int or units < 1:
        return None
    return batch_symbol(units) if symbolic else units


def batch_extent(value, batch):
    """The concrete size a batch entry requires of a tensor at runtime."""
    multiple = batch_multiple(value)
    return value if multiple is None else multiple * batch


class Immutable:
    """A record that never changes once constructed, so copies can share it.

    These records hold frozen mappings, which cannot be pickled, and they are
    compared by value, so copying one would only cost memory. Returning
    ``self`` keeps ``copy.deepcopy`` of anything that references a plan --- a
    built network, for one --- cheap and correct.
    """

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise HNDLError("E_SCHEMA", f"Unsupported or non-finite data value: {type(value).__name__}")


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def canonical(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def named_contracts(value, label, default_name):
    """Accept one shape tuple or a mapping of named shapes; keep author order.

    Returns the ordered mapping and whether the author named the ports.
    """
    if isinstance(value, Mapping):
        if not value or len(value) > MAX_EXTERNAL_PORTS:
            raise HNDLError("E_SCHEMA", f"{label} must name between 1 and {MAX_EXTERNAL_PORTS} tensors")
        names = tuple(value)
        for name in names:
            if type(name) is not str or _PORT_NAME.fullmatch(name) is None:
                raise HNDLError("E_SCHEMA", f"{label} names must match [a-z][a-z0-9_]*")
        return {name: _shape(value[name], f"{label}[{name}]") for name in names}, True
    return {default_name: _shape(value, label)}, False


def named_dtypes(value, names, label="input_dtype"):
    """Accept one dtype for every input or a mapping naming some of them."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise HNDLError("E_SCHEMA", f"{label} names undeclared inputs: {', '.join(unknown)}")
        return {name: value.get(name) for name in names}
    return {name: value for name in names}


def _shape(value, label):
    if not isinstance(value, (tuple, list)):
        raise HNDLError("E_SCHEMA", f"{label} must be a shape tuple including batch, or a mapping of named shapes")
    return tuple(value)


def _external_ports(record):
    """Derive the canonical port mappings; a single x -> output graph keeps its
    original fields, so ``named_inputs``/``named_outputs`` stay ``None`` and the
    saved encoding is unchanged."""
    if record.input_dtype is None:
        object.__setattr__(record, "input_dtype", record.dtype)
    if record.named_inputs is None:
        inputs = {EXTERNAL_INPUT: {"shape": _shape(record.input_shape, "input_shape"),
                                   "dtype": record.input_dtype}}
    else:
        inputs = {}
        for name, entry in _entries(record.named_inputs, "inputs", ("shape",), ("dtype",)):
            inputs[name] = {"shape": _shape(entry["shape"], f"input_shape[{name}]"),
                            "dtype": entry.get("dtype") or record.dtype}
    leading = next(iter(inputs.values()))
    object.__setattr__(record, "input_shape", leading["shape"])
    object.__setattr__(record, "input_dtype", leading["dtype"])
    frozen = _frozen_ports(inputs)
    object.__setattr__(record, "named_inputs", None if tuple(inputs) == (EXTERNAL_INPUT,) else frozen)
    object.__setattr__(record, "_inputs", frozen)
    if record.named_outputs is None:
        outputs = {EXTERNAL_OUTPUT: {"ref": record.output_ref,
                                     "shape": _shape(record.output_shape, "output_shape")}}
    else:
        outputs = {}
        for name, entry in _entries(record.named_outputs, "outputs", ("ref", "shape"), ()):
            if not isinstance(entry["ref"], str):
                raise HNDLError("E_SCHEMA", f"Output {name} must name one tensor reference")
            outputs[name] = {"ref": entry["ref"], "shape": _shape(entry["shape"], f"output_shape[{name}]")}
    leading = next(iter(outputs.values()))
    object.__setattr__(record, "output_ref", leading["ref"])
    object.__setattr__(record, "output_shape", leading["shape"])
    frozen = _frozen_ports(outputs)
    object.__setattr__(record, "named_outputs", None if tuple(outputs) == (EXTERNAL_OUTPUT,) else frozen)
    object.__setattr__(record, "_outputs", frozen)


def _entries(mapping, label, required, optional):
    if not isinstance(mapping, Mapping) or not mapping or len(mapping) > MAX_EXTERNAL_PORTS:
        raise HNDLError("E_SCHEMA", f"Graph {label} must be a mapping of 1 to {MAX_EXTERNAL_PORTS} named contracts")
    for name, entry in mapping.items():
        if type(name) is not str or _PORT_NAME.fullmatch(name) is None:
            raise HNDLError("E_SCHEMA", f"External {label} names must match [a-z][a-z0-9_]*")
        if (not isinstance(entry, Mapping) or set(required) - set(entry)
                or set(entry) - set(required) - set(optional)):
            raise HNDLError("E_SCHEMA", f"Each external {label[:-1]} declares {', '.join(required)}")
        yield name, entry


def _saved_ports(items, label, fields):
    """Restore the ordered named-port arrays a multi-port plan stores."""
    if type(items) is not list or not items or len(items) > MAX_EXTERNAL_PORTS:
        raise HNDLError("E_SCHEMA", f"Saved {label} must be an ordered array of named contracts")
    restored = {}
    for item in items:
        if type(item) is not dict or set(item) != set(fields):
            raise HNDLError("E_SCHEMA", f"Saved {label} require the fields {', '.join(fields)}")
        entry = dict(item)
        name = entry.pop("name")
        if type(name) is not str or name in restored:
            raise HNDLError("E_SCHEMA", f"Saved {label} names must be unique strings")
        restored[name] = entry
    return restored


def _frozen_ports(mapping):
    return MappingProxyType({name: MappingProxyType(dict(entry)) for name, entry in mapping.items()})


def contract_header(plan, label):
    """The one-line contract header shared by plans and built modules."""
    def shape(value):
        return "[" + ", ".join(str(part) for part in value) + "]"

    def side(ports, default):
        if tuple(ports) == (default,):
            return shape(ports[default]["shape"])
        return ", ".join(f"{name}={shape(entry['shape'])}" for name, entry in ports.items())

    header = (f"{label}: {side(plan.inputs, EXTERNAL_INPUT)} -> "
              f"{side(plan.outputs, EXTERNAL_OUTPUT)}  dtype={plan.dtype}")
    differing = {name: entry["dtype"] for name, entry in plan.inputs.items() if entry["dtype"] != plan.dtype}
    if differing:
        if tuple(plan.inputs) == (EXTERNAL_INPUT,):
            header += f"  input_dtype={plan.input_dtype}"
        else:
            header += "  input_dtype=" + ", ".join(f"{name}={value}" for name, value in differing.items())
    return header


@dataclass(frozen=True)
class Node(Immutable):
    id: str
    op: str
    args: Mapping = field(default_factory=dict)
    inputs: Mapping = field(default_factory=dict)
    outputs: tuple = ("out",)
    source: object = None
    initialization: Mapping = field(default_factory=normalize_initialization, kw_only=True)
    trainability: Mapping = field(default_factory=normalize_trainability, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "args", freeze(self.args))
        object.__setattr__(self, "inputs", freeze(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "source", freeze(self.source))
        object.__setattr__(self, "initialization", freeze(validate_initialization(self.initialization)))
        object.__setattr__(self, "trainability", freeze(validate_trainability(self.trainability)))


@dataclass(frozen=True)
class Graph(Immutable):
    nodes: tuple
    input_shape: tuple
    output_shape: tuple
    output_ref: str
    dtype: str = "float32"
    frontend: str = "python_config@1"
    input_dtype: object = None
    named_inputs: object = None
    named_outputs: object = None

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        _external_ports(self)

    inputs = property(lambda self: self._inputs)
    outputs = property(lambda self: self._outputs)


@dataclass(frozen=True)
class ResolvedNode(Node):
    input_shapes: Mapping = field(default_factory=dict)
    output_shapes: Mapping = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "input_shapes", freeze(self.input_shapes))
        object.__setattr__(self, "output_shapes", freeze(self.output_shapes))
        object.__setattr__(self, "provenance", freeze(self.provenance))

    def to_dict(self, *, semantic=False):
        result = {key: plain(getattr(self, key)) for key in (
            "id", "op", "args", "inputs", "outputs", "input_shapes", "output_shapes",
            "initialization", "trainability"
        )}
        if not semantic:
            result.update(source=plain(self.source), provenance=plain(self.provenance))
        return result


@dataclass(frozen=True)
class ResolvedPlan(Immutable):
    nodes: tuple
    input_shape: tuple
    output_shape: tuple
    output_ref: str
    dtype: str = "float32"
    frontend: str = "python_config@1"
    registry: object = field(default=None, repr=False, compare=False)
    schema_version: int = 1
    resolution_version: int = 1
    input_dtype: object = None
    named_inputs: object = None
    named_outputs: object = None

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        _external_ports(self)

    inputs = property(lambda self: self._inputs)
    outputs = property(lambda self: self._outputs)

    def _data(self, *, semantic=False):
        data = {
            "schema_version": self.schema_version,
            "resolution_version": self.resolution_version,
            "nodes": [node.to_dict(semantic=semantic) for node in self.nodes],
            "input_shape": list(self.input_shape), "output_shape": list(self.output_shape),
            "output_ref": self.output_ref, "dtype": self.dtype, "input_dtype": self.input_dtype,
        }
        # Named ports are extra fields, so a single x -> output plan keeps the
        # original encoding and digests byte for byte.
        if self.named_inputs is not None:
            data["inputs"] = [{"name": name, "shape": list(entry["shape"]), "dtype": entry["dtype"]}
                              for name, entry in self.inputs.items()]
        if self.named_outputs is not None:
            data["outputs"] = [{"name": name, "ref": entry["ref"], "shape": list(entry["shape"])}
                               for name, entry in self.outputs.items()]
        if not semantic:
            data["frontend"] = self.frontend
        return data

    @property
    def semantic_digest(self):
        return digest(self._data(semantic=True))

    def to_json(self):
        data = self._data()
        data["semantic_digest"] = self.semantic_digest
        data["artifact_digest"] = digest(data)
        return canonical(data)

    @classmethod
    def from_json(cls, source, *, registry=None, limits=None):
        if not isinstance(source, str):
            raise HNDLError("E_SCHEMA", "Saved plan must be a JSON string")
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise HNDLError("E_SCHEMA", f"Duplicate JSON field {key!r}")
                result[key] = value
            return result
        try:
            data = json.loads(source, object_pairs_hook=unique_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(data, dict):
                raise ValueError("plan must be an object")
            expected = {"schema_version", "resolution_version", "nodes", "input_shape", "output_shape",
                        "output_ref", "dtype", "input_dtype", "frontend", "semantic_digest", "artifact_digest"}
            if set(data) - {"inputs", "outputs"} != expected:
                raise ValueError("unexpected or missing plan fields")
            if (type(data["schema_version"]) is not int or data["schema_version"] != 1
                    or type(data["resolution_version"]) is not int or data["resolution_version"] != 1):
                raise HNDLError("E_STATE_VERSION", "Expected plan schema 1 and resolution version 1")
            # The digests detect corruption and hand edits; they are not a
            # signature. Consistency with the operators is checked below.
            artifact = data.pop("artifact_digest")
            if artifact != digest(data):
                raise HNDLError("E_INTEGRITY", _CORRUPTED.format(field="artifact_digest"))
            semantic = data.pop("semantic_digest")
            serialized_nodes = data.pop("nodes")
            if not isinstance(serialized_nodes, list):
                raise HNDLError("E_SCHEMA", "Saved nodes must be a list")
            required_node_fields = {"id", "op", "args", "inputs", "outputs", "input_shapes", "output_shapes",
                                    "initialization", "trainability", "source", "provenance"}
            if any(type(item) is not dict or set(item) != required_node_fields for item in serialized_nodes):
                raise HNDLError("E_SCHEMA", "Saved schema 1 nodes require all canonical fields, including initialization and trainability")
            nodes = tuple(ResolvedNode(**item) for item in serialized_nodes)
            for key, fields in (("inputs", ("name", "shape", "dtype")), ("outputs", ("name", "ref", "shape"))):
                if key in data:
                    data[f"named_{key}"] = _saved_ports(data.pop(key), key, fields)
            plan = cls(nodes=nodes, registry=registry, **data)
            if semantic != plan.semantic_digest:
                raise HNDLError("E_INTEGRITY", _CORRUPTED.format(field="semantic_digest"))
            from .resolver import validate_concrete_plan
            return validate_concrete_plan(plan, registry=registry, limits=limits)
        except HNDLError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError) as error:
            raise HNDLError("E_SCHEMA", f"Invalid saved plan: {error}") from error

    def _operation_name(self, op):
        if self.registry is not None:
            try:
                return self.registry.by_identity(op).alias
            except HNDLError:
                pass
        return op

    def __repr__(self):
        def shape(value):
            return "[" + ", ".join(str(part) for part in value) + "]"
        lines = [contract_header(self, "Network")]
        rows = [("index", "name", "operation", "input shapes", "output shapes")]
        for index, node in enumerate(self.nodes):
            inputs = ", ".join(f"{key}={shape(value)}" for key, value in node.input_shapes.items())
            outputs = ", ".join(f"{key}={shape(value)}" for key, value in node.output_shapes.items())
            rows.append((str(index), node.id, self._operation_name(node.op), inputs, outputs))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)
        return "\n".join(lines)

    def describe(self):
        lines = [repr(self), f"Semantic digest: {self.semantic_digest}"]
        for node in self.nodes:
            lines.append(f"{node.id}: " + ", ".join(f"{key}={plain(value)}" for key, value in node.provenance.items()))
            lines.append(f"  initialization={plain(node.initialization)}, trainability={plain(node.trainability)}")
        return "\n".join(lines)
