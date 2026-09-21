"""Stable, source-aware errors shared by the pure core and backends."""


class HNDLError(ValueError):
    def __init__(self, code, message, *, node=None, line=None, column=None):
        self.code = code
        self.message = message
        self.node = node
        self.line = line
        self.column = column
        location = []
        if node is not None:
            location.append(f"node {node}")
        if line is not None:
            location.append(f"line {line}" + (f", column {column}" if column is not None else ""))
        suffix = f" ({'; '.join(location)})" if location else ""
        super().__init__(f"{code}{suffix}: {message}")

