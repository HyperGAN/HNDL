"""Immutable, torch-free graph records and portable resolved plans."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from collections.abc import Mapping

from .errors import HNDLError
from .settings import (normalize_initialization, normalize_trainability,
                       validate_initialization, validate_trainability)


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


@dataclass(frozen=True)
class Node:
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
class Graph:
    nodes: tuple
    input_shape: tuple
    output_shape: tuple
    output_ref: str
    dtype: str = "float32"
    frontend: str = "python_config@1"

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "output_shape", tuple(self.output_shape))


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
class ResolvedPlan:
    nodes: tuple
    input_shape: tuple
    output_shape: tuple
    output_ref: str
    dtype: str = "float32"
    frontend: str = "python_config@1"
    registry: object = field(default=None, repr=False, compare=False)
    schema_version: int = 1
    resolution_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "output_shape", tuple(self.output_shape))

    def _data(self, *, semantic=False):
        data = {
            "schema_version": self.schema_version,
            "resolution_version": self.resolution_version,
            "nodes": [node.to_dict(semantic=semantic) for node in self.nodes],
            "input_shape": list(self.input_shape), "output_shape": list(self.output_shape),
            "output_ref": self.output_ref, "dtype": self.dtype,
        }
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
        if not isinstance(source, str) or len(source.encode("utf-8")) > 16 * 1024 * 1024:
            raise HNDLError("E_RESOURCE", "Saved plan must be a JSON string of at most 16 MiB")
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
                        "output_ref", "dtype", "frontend", "semantic_digest", "artifact_digest"}
            if set(data) != expected:
                raise ValueError("unexpected or missing plan fields")
            if (type(data["schema_version"]) is not int or data["schema_version"] != 1
                    or type(data["resolution_version"]) is not int or data["resolution_version"] != 1):
                raise HNDLError("E_STATE_VERSION", "Expected plan schema 1 and resolution version 1")
            artifact = data.pop("artifact_digest")
            if artifact != digest(data):
                raise HNDLError("E_INTEGRITY", "Saved plan artifact digest does not match its contents")
            semantic = data.pop("semantic_digest")
            from .resolver import _limits
            serialized_nodes = data.pop("nodes")
            if not isinstance(serialized_nodes, list):
                raise HNDLError("E_SCHEMA", "Saved nodes must be a list")
            if len(serialized_nodes) > _limits(limits)["max_nodes"]:
                raise HNDLError("E_RESOURCE", "Saved plan exceeds max_nodes")
            required_node_fields = {"id", "op", "args", "inputs", "outputs", "input_shapes", "output_shapes",
                                    "initialization", "trainability", "source", "provenance"}
            if any(type(item) is not dict or set(item) != required_node_fields for item in serialized_nodes):
                raise HNDLError("E_SCHEMA", "Saved schema 1 nodes require all canonical fields, including initialization and trainability")
            nodes = tuple(ResolvedNode(**item) for item in serialized_nodes)
            plan = cls(nodes=nodes, registry=registry, **data)
            if semantic != plan.semantic_digest:
                raise HNDLError("E_INTEGRITY", "Saved plan semantic digest does not match its contents")
            from .resolver import validate_concrete_plan
            return validate_concrete_plan(plan, registry=registry, limits=limits)
        except HNDLError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError) as error:
            raise HNDLError("E_SCHEMA", f"Invalid saved plan: {error}") from error

    def __repr__(self):
        def shape(value):
            return "[" + ", ".join(str(part) for part in value) + "]"
        lines = [f"Network: {shape(self.input_shape)} -> {shape(self.output_shape)}  dtype={self.dtype}",
                 "index  name  operation  input shapes  output shapes"]
        for index, node in enumerate(self.nodes):
            inputs = ", ".join(f"{key}={shape(value)}" for key, value in node.input_shapes.items())
            outputs = ", ".join(f"{key}={shape(value)}" for key, value in node.output_shapes.items())
            lines.append(f"{index}  {node.id}  {node.op}  {inputs}  {outputs}")
        return "\n".join(lines)

    def describe(self):
        lines = [repr(self), f"Semantic digest: {self.semantic_digest}"]
        for node in self.nodes:
            lines.append(f"{node.id}: " + ", ".join(f"{key}={plain(value)}" for key, value in node.provenance.items()))
            lines.append(f"  initialization={plain(node.initialization)}, trainability={plain(node.trainability)}")
        return "\n".join(lines)
