"""Stable, source-aware errors shared by the pure core and backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def describe_iterations(iterations: Sequence[Sequence[int]] | None) -> str:
    """``iteration 3 of 8``, or ``iteration 1 of 2, 3 of 4`` outermost first.

    Indices count from 0, like ``range`` and like the suffix a loop appends to
    a node name, so ``block3`` is ``iteration 3 of 8``.
    """
    if not iterations:
        return ""
    return "iteration " + ", ".join(f"{index} of {count}" for index, count in iterations)


def source_location(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``line``, ``column`` and ``iterations`` a frontend recorded, as HNDLError keywords."""
    if not source:
        return {}
    return {key: source[key] for key in ("line", "column", "iterations") if source.get(key) is not None}


class HNDLError(ValueError):
    code: str
    message: str
    node: str | None
    line: int | None
    column: int | None
    iterations: tuple[tuple[int, int], ...] | None

    def __init__(
        self,
        code: str,
        message: str,
        *,
        node: str | None = None,
        line: int | None = None,
        column: int | None = None,
        iterations: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.node = node
        self.line = line
        self.column = column
        # One (index, count) pair per enclosing config loop, outermost first.
        self.iterations = tuple(tuple(level) for level in iterations) if iterations else None
        location = []
        if node is not None:
            location.append(f"node {node}")
        if line is not None:
            location.append(f"line {line}" + (f", column {column}" if column is not None else "")
                            + (f", {describe_iterations(self.iterations)}" if self.iterations else ""))
        suffix = f" ({'; '.join(location)})" if location else ""
        super().__init__(f"{code}{suffix}: {message}")
