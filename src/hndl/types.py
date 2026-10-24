"""Immutable, torch-free graph records and portable resolved plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import keyword
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .errors import HNDLError
from .settings import (DEFAULT_INITIALIZATION, DEFAULT_TRAINABILITY,
                       validate_initialization, validate_trainability)

if TYPE_CHECKING:
    from .registry import Registry


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


def batch_multiple(value: Any) -> int | None:
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


def batch_symbol(multiple: Any) -> str | None:
    """The canonical spelling of ``multiple`` plan batches, or ``None``."""
    if type(multiple) is not int or not 1 <= multiple <= MAX_BATCH_MULTIPLE:
        return None
    return BATCH if multiple == 1 else f"{multiple}*B"


def batch_units(value: Any) -> int | None:
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


def format_shape(shape: Any) -> str:
    """Format a shape sequence into the canonical bracketed string '[B, 64]'."""
    return "[" + ", ".join(str(part) for part in shape) + "]"


def _validate_port_name(name: str, label: str) -> None:
    if type(name) is not str or _PORT_NAME.fullmatch(name) is None:
        raise HNDLError("E_SCHEMA", f"{label} names must match [a-z][a-z0-9_]*")
    if keyword.iskeyword(name):
        raise HNDLError("E_SCHEMA", f"{label} names must not be Python keywords")


def named_contracts(value, label, default_name):
    """Accept one shape tuple or a mapping of named shapes; keep author order.

    Returns the ordered mapping and whether the author named the ports.
    """
    if isinstance(value, Mapping):
        if not value or len(value) > MAX_EXTERNAL_PORTS:
            raise HNDLError("E_SCHEMA", f"{label} must name between 1 and {MAX_EXTERNAL_PORTS} tensors")
        names = tuple(value)
        for name in names:
            _validate_port_name(name, label)
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
                            "dtype": record.dtype if entry.get("dtype") is None else entry.get("dtype")}
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
        _validate_port_name(name, f"External {label}")
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


def _is_chain(plan: ResolvedPlan | Graph) -> bool:
    """A chain has one external input feeding one unary node after another."""
    if len(plan.inputs) != 1 or len(plan.outputs) != 1:
        return False
    registry = getattr(plan, "registry", None)
    previous = f"input:{next(iter(plan.inputs))}"
    for node in plan.nodes:
        variadic = False
        if registry is not None:
            try:
                variadic = tuple(node.outputs) != registry.by_identity(node.op).output_ports
            except HNDLError:
                pass
        if (len(node.inputs) != 1 or tuple(node.inputs.values()) != (previous,)
                or len(node.outputs) != 1
                or variadic
                or (node.outputs[0].startswith("out") and node.outputs[0][3:].isdigit())):
            return False
        previous = f"node:{node.id}/{node.outputs[0]}"
    return plan.output_ref == previous


def contract_header(plan: ResolvedPlan | Graph, label: str) -> str:
    """The one-line contract header shared by plans and built modules."""
    def side(ports: Mapping[str, Any], default: str) -> str:
        if tuple(ports) == (default,):
            return format_shape(ports[default]["shape"])
        return ", ".join(f"{name}={format_shape(entry['shape'])}" for name, entry in ports.items())

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
    args: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: tuple[str, ...] = ("out",)
    source: Mapping[str, Any] | None = None
    initialization: Mapping[str, Any] = field(default_factory=lambda: DEFAULT_INITIALIZATION, kw_only=True)
    trainability: Mapping[str, Any] = field(default_factory=lambda: DEFAULT_TRAINABILITY, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.args, Mapping):
            raise TypeError(f"Node args must be a Mapping, got {type(self.args).__name__}")
        if not isinstance(self.inputs, Mapping):
            raise TypeError(f"Node inputs must be a Mapping, got {type(self.inputs).__name__}")
        if isinstance(self.outputs, str) or not isinstance(self.outputs, (tuple, list)):
            raise TypeError(f"Node outputs must be a sequence of strings, got {type(self.outputs).__name__}")
        if not all(isinstance(port, str) for port in self.outputs):
            raise TypeError("All output ports must be strings")
        object.__setattr__(self, "args", freeze(self.args))
        object.__setattr__(self, "inputs", freeze(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "source", freeze(self.source))
        if self.initialization is DEFAULT_INITIALIZATION or self.initialization == DEFAULT_INITIALIZATION:
            object.__setattr__(self, "initialization", DEFAULT_INITIALIZATION)
        else:
            object.__setattr__(self, "initialization", freeze(validate_initialization(self.initialization)))
        if self.trainability is DEFAULT_TRAINABILITY or self.trainability == DEFAULT_TRAINABILITY:
            object.__setattr__(self, "trainability", DEFAULT_TRAINABILITY)
        else:
            object.__setattr__(self, "trainability", freeze(validate_trainability(self.trainability)))


@dataclass(frozen=True)
class Graph(Immutable):
    nodes: tuple[Node, ...]
    input_shape: tuple[int | str, ...]
    output_shape: tuple[int | str, ...]
    output_ref: str
    dtype: str = "float32"
    frontend: str = "python_config@1"
    input_dtype: str | Mapping[str, str] | None = None
    named_inputs: Mapping[str, Any] | None = None
    named_outputs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        _external_ports(self)

    inputs = property(lambda self: self._inputs)
    outputs = property(lambda self: self._outputs)


@dataclass(frozen=True)
class ResolvedNode(Node):
    input_shapes: Mapping[str, tuple[int | str, ...]] = field(default_factory=dict)
    output_shapes: Mapping[str, tuple[int | str, ...]] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "input_shapes", freeze(self.input_shapes))
        object.__setattr__(self, "output_shapes", freeze(self.output_shapes))
        object.__setattr__(self, "provenance", freeze(self.provenance))

    def to_dict(self, *, semantic: bool = False) -> dict[str, Any]:
        result = {key: plain(getattr(self, key)) for key in (
            "id", "op", "args", "inputs", "outputs", "input_shapes", "output_shapes",
            "initialization", "trainability"
        )}
        if not semantic:
            result.update(source=plain(self.source), provenance=plain(self.provenance))
        return result


@dataclass(frozen=True)
class ResolvedPlan(Immutable):
    nodes: tuple[ResolvedNode, ...]
    input_shape: tuple[int | str, ...]
    output_shape: tuple[int | str, ...]
    output_ref: str
    dtype: str = "float32"
    frontend: str = "python_config@1"
    registry: Registry | None = field(default=None, repr=False, compare=False)
    schema_version: int = 1
    resolution_version: int = 1
    input_dtype: str | Mapping[str, str] | None = None
    named_inputs: Mapping[str, Any] | None = None
    named_outputs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        _external_ports(self)
        object.__setattr__(self, "_semantic_digest", digest(self._data(semantic=True)))
        object.__setattr__(self, "_validated_stamp", None)

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
    def _chain(self) -> bool:
        return _is_chain(self)

    @property
    def semantic_digest(self) -> str:
        return self._semantic_digest

    def to_json(self) -> str:
        data = self._data()
        data["semantic_digest"] = self._semantic_digest
        data_str = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        data["artifact_digest"] = sha256(data_str.encode("utf-8")).hexdigest()
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(
        cls,
        source: str,
        *,
        registry: Registry | None = None,
        limits: Mapping[str, int] | None = None,
    ) -> ResolvedPlan:
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

    def _operation_name(self, op: str) -> str:
        if self.registry is not None:
            try:
                return self.registry.by_identity(op).alias
            except HNDLError:
                pass
        return op.split("@")[0] if "@" in op else op

    def __repr__(self) -> str:
        return format_plan_table(self, "Network")

    def describe(self) -> str:
        lines = [repr(self), f"Semantic digest: {self.semantic_digest}"]
        for node in self.nodes:
            lines.append(f"{node.id}: " + ", ".join(f"{key}={plain(value)}" for key, value in node.provenance.items()))
            lines.append(f"  initialization={plain(node.initialization)}, trainability={plain(node.trainability)}")
        return "\n".join(lines)

    def to_python_source(self, class_name: str = "Model", *args: Any, **kwargs: Any) -> str:
        """Emit a standalone, self-contained PyTorch module from this plan."""
        from .codegen import to_python_source
        return to_python_source(self, class_name=class_name, *args, **kwargs)


def format_plan_table(plan: ResolvedPlan, label: str) -> str:
    """Render a formatted ASCII table representing a resolved plan."""
    lines = [contract_header(plan, label)]
    chain = _is_chain(plan)
    rows = [("index", "name", "operation", "input shape", "output shape")
            if chain else ("name", "operation", "input shapes", "output shapes")]
    for index, node in enumerate(plan.nodes):
        op_name = plan._operation_name(node.op) if hasattr(plan, "_operation_name") else (
            node.op.split("@")[0] if "@" in node.op else node.op
        )
        if chain:
            rows.append((str(index), node.id, op_name,
                         format_shape(next(iter(node.input_shapes.values()))),
                         format_shape(next(iter(node.output_shapes.values())))))
        else:
            inputs = ", ".join(f"{p}={node.inputs[p]}:{format_shape(s)}" for p, s in node.input_shapes.items())
            outputs = ", ".join(f"{p}={format_shape(s)}" for p, s in node.output_shapes.items())
            rows.append((node.id, op_name, inputs, outputs))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)
    return "\n".join(lines)
