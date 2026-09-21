"""Generate operator documentation from ``@operator`` declarations.

``python -m hndl.docs`` writes ``docs/operators/<alias>.md`` and an index;
``python -m hndl.docs --check`` fails when the committed pages are stale.
"""

import argparse
from pathlib import Path
import sys

from .operator import ELLIPSIS, REQUIRED, Sym
from .registry import Registry


def _pattern_text(port):
    if port.pattern is None:
        return port.name + ("*" if port.variadic else "")
    dims = []
    for dim in port.pattern:
        if dim == ELLIPSIS:
            dims.append("...")
        elif isinstance(dim, Sym):
            dims.append(dim.name if dim.scale == 1 else f"{dim.scale}*{dim.name}")
        else:
            dims.append(str(dim))
    suffix = "" if port.dtype == "compute" else f":{port.dtype}"
    return f"{port.name}[{', '.join(dims)}]{suffix}"


def _default_text(arg):
    if arg.inferable:
        return "inferred"
    if arg.default is REQUIRED:
        return "required"
    value = arg.default
    if isinstance(value, str):
        return f'`"{value}"`'
    return f"`{value!r}`"


def _constraints(arg):
    parts = []
    if arg.min is not None:
        parts.append(f"{'>' if arg.exclusive_min else '>='} {arg.min}")
    if arg.max is not None:
        parts.append(f"{'<' if arg.exclusive_max else '<='} {arg.max}")
    if arg.choices:
        parts.append("one of " + ", ".join(f"`{c}`" for c in arg.choices))
    if arg.dim:
        parts.append(f"binds `{arg.dim}`")
    return "; ".join(parts) or "—"


def _shape_table(plan):
    return "\n".join(line for line in repr(plan).splitlines())


def render_operator(spec, registry):
    from .torch import parameter_counts
    from .config import resolve

    lines = [f"# `{spec.alias}`", "", spec.summary, "",
             f"**Category:** {spec.category} · **Identity:** `{spec.key}`", "", "## Shape", "",
             "```text", spec.shape_text, "```"]
    if spec.relation_text:
        lines += ["", "Relation: `" + spec.relation_text + "`"]
    ports = [(port, "input") for port in spec.inputs] + [(port, "output") for port in spec.outputs]
    lines += ["", "| Port | Direction | Pattern | dtype |", "| --- | --- | --- | --- |"]
    for port, direction in ports:
        pattern = _pattern_text(port)
        lines.append(f"| `{port.name}` | {direction} | `{pattern}` | {port.dtype} |")
    if spec.policies:
        lines += ["", "**Policies:** " + ", ".join(
            f"`policy=\"{alias}\"` ({policy.identity}: " + ", ".join(f"{k}={v}" for k, v in policy.requires.items()) + ")"
            for alias, policy in spec.policies.items())]
    lines += ["", "## Arguments", ""]
    if spec.args:
        lines += ["| Name | Type | Default | Constraints | Description |", "| --- | --- | --- | --- | --- |"]
        for name, arg in spec.args.items():
            positional = "" if not arg.positional else " (positional)"
            lines.append(f"| `{name}`{positional} | {arg.type_name} | {_default_text(arg)} | {_constraints(arg)} | {arg.help} |")
        if spec.positional_rest:
            lines += ["", f"Positional values fill `{spec.positional_rest}`."]
    else:
        lines.append("This operator takes no scalar arguments.")
    lines += ["", "## Description", "", spec.doc.strip(), ""]
    lines += ["## Examples", ""]
    for index, example in enumerate(spec.examples, 1):
        lines.append(f"### Example {index}")
        lines.append("")
        if example.note:
            lines += [example.note, ""]
        lines += ["```python", example.source.strip("\n"), "```", ""]
        lines.append(f"Input `{list(example.input_shape)}` → output `{list(example.output_shape)}`.")
        lines.append("")
        plan = resolve(example.source, input_shape=example.input_shape, output_shape=example.output_shape,
                       registry=registry)
        counts = parameter_counts(plan, registry=registry)
        total = sum(count for count in counts.values() if count is not None)
        lines += ["```text", _shape_table(plan), "```", "", f"Parameters: {total:,}", ""]
    return "\n".join(lines).rstrip() + "\n"


def render_index(specs):
    lines = ["# Operators", "", "Every operator below is declared with `@operator` on its module class; these pages",
             "are generated from those declarations by `python -m hndl.docs`.", ""]
    by_category = {}
    for spec in specs:
        by_category.setdefault(spec.category, []).append(spec)
    for category in sorted(by_category):
        lines += [f"## {category}", "", "| Operator | Summary |", "| --- | --- |"]
        for spec in sorted(by_category[category], key=lambda item: item.alias):
            lines.append(f"| [`{spec.alias}`]({spec.alias}.md) | {spec.summary} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate(registry=None):
    registry = registry or Registry.builtins()
    pages = {f"{spec.alias}.md": render_operator(spec, registry) for spec in registry.operators}
    pages["index.md"] = render_index(registry.operators)
    return pages


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate operator documentation from declarations.")
    parser.add_argument("--out", default="docs/operators", help="Output directory")
    parser.add_argument("--check", action="store_true", help="Fail if the committed pages are stale")
    args = parser.parse_args(argv)
    out = Path(args.out)
    pages = generate()
    if args.check:
        stale = [name for name, text in pages.items()
                 if not (out / name).exists() or (out / name).read_text(encoding="utf-8") != text]
        extra = [path.name for path in out.glob("*.md") if path.name not in pages] if out.exists() else []
        if stale or extra:
            print("Stale operator docs: " + ", ".join(stale + [f"{name} (unexpected)" for name in extra]), file=sys.stderr)
            print("Run `python -m hndl.docs` and commit the result.", file=sys.stderr)
            return 1
        print(f"{len(pages)} operator pages are current.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (out / name).write_text(text, encoding="utf-8")
    for path in out.glob("*.md"):
        if path.name not in pages:
            path.unlink()
    print(f"Wrote {len(pages)} pages to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
