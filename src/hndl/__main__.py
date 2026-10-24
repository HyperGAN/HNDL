"""HNDL command-line interface entrypoint: inspect, mermaid, and check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from hndl import HNDLError, resolve
from hndl.docs import contract_value
from hndl.types import ResolvedPlan


def _load_plan(hndl_path: Path, sidecar_path: Path | None = None) -> ResolvedPlan:
    """Load and resolve a .hndl file with its optional JSON sidecar metadata."""
    if not hndl_path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {hndl_path}")

    source = hndl_path.read_text(encoding="utf-8")
    meta_path = sidecar_path
    if meta_path is None:
        candidate = hndl_path.with_suffix(".json")
        if candidate.is_file():
            meta_path = candidate

    kwargs: dict[str, Any] = {}
    if meta_path is not None:
        if not meta_path.is_file():
            raise FileNotFoundError(f"Sidecar file not found: {meta_path}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Corrupt JSON sidecar {meta_path}: {err}") from err
        if "input_shape" in meta:
            kwargs["input_shape"] = contract_value(meta["input_shape"])
        if "output_shape" in meta:
            kwargs["output_shape"] = contract_value(meta["output_shape"])
        if "input_dtype" in meta:
            kwargs["input_dtype"] = meta["input_dtype"]
        if "dtype" in meta:
            kwargs["dtype"] = meta["dtype"]

    return resolve(source, **kwargs)


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect resolved contracts, nodes, and static profiling metrics."""
    hndl_path = Path(args.file)
    sidecar_path = Path(args.sidecar) if args.sidecar else None
    try:
        plan = _load_plan(hndl_path, sidecar_path)
    except Exception as err:
        sys.stderr.write(f"Error resolving plan {hndl_path}: {err}\n")
        return 1

    sys.stdout.write(f"Plan: {hndl_path.name}\n")
    sys.stdout.write(f"Semantic digest: {plan.semantic_digest}\n\n")
    sys.stdout.write(f"{plan}\n\n")

    if args.profile:
        try:
            prof = plan.profile(batch_size=args.batch_size)
            sys.stdout.write(f"{prof.summary()}\n")
        except Exception as err:
            sys.stderr.write(f"Profiling error: {err}\n")
            return 1

    return 0


def _cmd_mermaid(args: argparse.Namespace) -> int:
    """Export shape-annotated Mermaid flowchart diagram."""
    hndl_path = Path(args.file)
    sidecar_path = Path(args.sidecar) if args.sidecar else None
    try:
        plan = _load_plan(hndl_path, sidecar_path)
    except Exception as err:
        sys.stderr.write(f"Error resolving plan {hndl_path}: {err}\n")
        return 1

    mermaid_text = plan.to_mermaid()
    if args.out:
        out_path = Path(args.out)
        try:
            out_path.write_text(mermaid_text, encoding="utf-8")
            sys.stdout.write(f"Wrote Mermaid diagram to {out_path}\n")
        except Exception as err:
            sys.stderr.write(f"Failed to write output file {out_path}: {err}\n")
            return 1
    else:
        sys.stdout.write(mermaid_text)

    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Validate recipe parsing, sidecar consistency, and contract resolution."""
    target_path = Path(args.path)
    if not target_path.exists():
        sys.stderr.write(f"Path not found: {target_path}\n")
        return 1

    if target_path.is_file():
        hndl_files = [target_path] if target_path.suffix == ".hndl" else []
        if not hndl_files:
            sys.stderr.write(f"Target file is not a .hndl file: {target_path}\n")
            return 1
    else:
        hndl_files = sorted(target_path.rglob("*.hndl"))
        if not hndl_files:
            sys.stderr.write(f"No .hndl files found in directory: {target_path}\n")
            return 1

    failed = 0
    passed = 0
    for hndl_file in hndl_files:
        try:
            plan = _load_plan(hndl_file)
            prof = plan.profile(batch_size=1)
            passed += 1
            sys.stdout.write(f"✓ {hndl_file} (nodes={len(plan.nodes)}, params={prof.total_parameters:,d})\n")
        except Exception as err:
            failed += 1
            sys.stderr.write(f"✗ {hndl_file}: {err}\n")

    sys.stdout.write(f"\nCheck completed: {passed} passed, {failed} failed.\n")
    return 1 if failed > 0 else 0


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hndl",
        description="HNDL CLI: inspect architectures, export diagrams, and validate recipes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect resolved network architecture.")
    inspect_parser.add_argument("file", help="Path to .hndl recipe file.")
    inspect_parser.add_argument("--sidecar", help="Path to optional .json sidecar metadata.")
    inspect_parser.add_argument("--batch-size", type=int, default=1, help="Batch size for static profiling.")
    inspect_parser.add_argument("--profile", action="store_true", help="Compute zero-compute static profile.")
    inspect_parser.add_argument("--device", default="cpu", help="Target device.")

    # mermaid
    mermaid_parser = subparsers.add_parser("mermaid", help="Export Mermaid flowchart diagram.")
    mermaid_parser.add_argument("file", help="Path to .hndl recipe file.")
    mermaid_parser.add_argument("--sidecar", help="Path to optional .json sidecar metadata.")
    mermaid_parser.add_argument("--out", help="Output file path (default: stdout).")

    # check
    check_parser = subparsers.add_parser("check", help="Validate recipes and sidecars.")
    check_parser.add_argument("path", help="Path to .hndl file or directory of recipes.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entrypoint returning integer exit code."""
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        if args.command == "inspect":
            if args.batch_size <= 0:
                sys.stderr.write(f"batch-size must be positive, got {args.batch_size}\n")
                return 1
            return _cmd_inspect(args)
        elif args.command == "mermaid":
            return _cmd_mermaid(args)
        elif args.command == "check":
            return _cmd_check(args)
        else:
            sys.stderr.write(f"Unknown command {args.command}\n")
            return 1
    except HNDLError as err:
        sys.stderr.write(f"HNDL error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected error: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
