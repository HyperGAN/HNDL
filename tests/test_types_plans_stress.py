"""Comprehensive Empirical Stress-Test Harness for Milestone 4 (Challenger 1).

Validates:
- M-10: ResolvedPlan.__repr__ parity with GraphModule.__repr__ across unary chains
  (lengths 1, 2, 10), branched multi-input graphs, branched multi-output graphs,
  residual/skip connections, and multi-output operators (split, chunk).
  Verifies:
  1. The "index" column appears ONLY on unary chains and NEVER on branched/residual graphs.
  2. Input port references are formatted as `p=node:id/out:[...]` or `p=input:...:[...]`.
  3. Header lines use singular "input shape" / "output shape" for unary chains and
     plural "input shapes" / "output shapes" for branched graphs.
- M-10: make_operator and Operator positional argument binding validation:
  1. Specifying positional_rest alongside any other argument with positional=True raises
     E_REGISTRY in make_operator and E_SCHEMA in Operator.__post_init__.
  2. positional_rest naming a missing or non-ints argument raises E_REGISTRY.
  3. Operator.positional_names property returns () when positional_rest is present,
     and returns the declared positional tuple when positional_rest is None.
- M-11: PEP 561 static typing and public API introspection:
  1. py.typed marker file exists and is included in sdist/MANIFEST.in.
  2. Public dataclasses (Arg, Example, Policy, ResolvedPlan, Node, Graph, ResolvedNode,
     Port, Operator) use concrete field types and have zero fields typed as bare object.
  3. All core public API functions (resolve, resolve_file, resolve_callable, operator,
     build, network, network_file, network_from_callable, parameter_counts) have complete
     parameter and return type annotations with no empty annotations or bare object.
  4. Surfaces typing omission on the 8 initializer helper functions exported in hndl.__all__.
"""

from dataclasses import fields, is_dataclass
import inspect
from pathlib import Path
import re

import pytest
import torch.nn as nn

import hndl
import hndl.torch
from hndl import Arg, Example, HNDLError, ops, resolve, resolve_callable
from hndl.operator import INTS, Operator, Port, make_operator
from hndl.torch import build, network, network_file, network_from_callable, parameter_counts
from hndl.types import Graph, Node, ResolvedNode, ResolvedPlan


# ==============================================================================
# M-10: ResolvedPlan.__repr__ Topology Stress Tests
# ==============================================================================


def _extract_table_header_and_rows(rep: str) -> tuple[list[str], list[list[str]]]:
    """Parse table output lines from repr into column headers and row cells."""
    lines = [line.strip() for line in rep.strip().splitlines() if line.strip()]
    assert len(lines) >= 2, f"Repr must contain contract header and table: {rep}"
    # Header line is lines[1]
    header_tokens = re.split(r"\s{2,}", lines[1])
    rows = [re.split(r"\s{2,}", row) for row in lines[2:]]
    return header_tokens, rows


class TestPlanReprTopologyStress:
    """Adversarial stress-testing of ResolvedPlan.__repr__ across diverse topologies."""

    def test_repr_unary_chain_length_1(self):
        """Length-1 unary chain displays index column, singular headers, and index 0."""
        plan = resolve("linear(16)", input_shape=("B", 8), output_shape=("B", 16))
        assert plan._chain is True
        rep = repr(plan)

        header_line = rep.splitlines()[0]
        assert "Network: [B, 8] -> [B, 16]" in header_line

        headers, rows = _extract_table_header_and_rows(rep)
        assert headers == ["index", "name", "operation", "input shape", "output shape"]
        assert len(rows) == 1
        assert rows[0] == ["0", "n0", "linear", "[B, 8]", "[B, 16]"]

    def test_repr_unary_chain_length_2(self):
        """Length-2 unary chain displays index column, singular headers, and sequential indices."""
        plan = resolve("linear(16)\nrelu()", input_shape=("B", 8), output_shape=("B", 16))
        assert plan._chain is True
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert headers == ["index", "name", "operation", "input shape", "output shape"]
        assert len(rows) == 2
        assert rows[0] == ["0", "n0", "linear", "[B, 8]", "[B, 16]"]
        assert rows[1] == ["1", "n1", "relu", "[B, 16]", "[B, 16]"]

    def test_repr_unary_chain_length_10(self):
        """Length-10 unary chain displays index column with strictly sequential 0..9."""
        ops_chain = ["linear(16)"] + ["relu()" for _ in range(9)]
        plan = resolve("\n".join(ops_chain), input_shape=("B", 8), output_shape=("B", 16))
        assert plan._chain is True
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert headers == ["index", "name", "operation", "input shape", "output shape"]
        assert len(rows) == 10
        for i, row in enumerate(rows):
            assert row[0] == str(i)
            assert row[1] == f"n{i}"
            assert row[2] == ("linear" if i == 0 else "relu")
            assert row[3] == ("[B, 8]" if i == 0 else "[B, 16]")
            assert row[4] == "[B, 16]"

    def test_repr_branched_multiple_external_inputs(self):
        """Branched graph with multiple inputs NEVER displays index column and formats input ports."""
        source = "add(x, y)"
        plan = resolve(
            source,
            input_shape={"x": ("B", 16), "y": ("B", 16)},
            output_shape=("B", 16),
        )
        assert plan._chain is False
        rep = repr(plan)

        # 1. Index must NEVER appear
        assert "index" not in rep.splitlines()[1]

        headers, rows = _extract_table_header_and_rows(rep)
        assert headers == ["name", "operation", "input shapes", "output shapes"]
        assert len(rows) == 1
        name, operation, in_shapes, out_shapes = rows[0]
        assert name == "n0"
        assert operation == "add"
        assert "a=input:x:[B, 16]" in in_shapes
        assert "b=input:y:[B, 16]" in in_shapes
        assert out_shapes == "out=[B, 16]"

    def test_repr_branched_multiple_named_outputs(self):
        """Graph with multiple named outputs omits index and renders plural headers."""
        def model_fn(x):
            f = ops.linear(32, name="body")
            return {"logits": ops.linear(1, name="head"), "features": f}

        plan = resolve_callable(
            model_fn,
            input_shape=("B", 16),
            output_shape={"logits": ("B", 1), "features": ("B", 32)},
        )
        assert plan._chain is False
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert "index" not in headers
        assert headers == ["name", "operation", "input shapes", "output shapes"]
        assert len(rows) == 2

        # Check port references
        body_row = next(r for r in rows if r[0] == "body")
        assert "x=input:x:[B, 16]" in body_row[2]
        assert body_row[3] == "out=[B, 32]"

        head_row = next(r for r in rows if r[0] == "head")
        assert "x=node:body/out:[B, 32]" in head_row[2]
        assert head_row[3] == "out=[B, 1]"

    def test_repr_residual_skip_connections(self):
        """Residual skip connection is recognized as branched: no index column, correct port refs."""
        source = (
            "saved = x\n"
            "h = linear(16)\n"
            "h = relu(h)\n"
            "h = linear(16, x=h)\n"
            "out = add(h, saved)\n"
        )
        plan = resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert plan._chain is False
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert "index" not in headers
        assert headers == ["name", "operation", "input shapes", "output shapes"]
        assert len(rows) == 4

        # n0 feeds from external input
        assert "x=input:x:[B, 16]" in rows[0][2]
        # n1 feeds from n0
        assert "x=node:n0/out:[B, 16]" in rows[1][2]
        # n2 feeds from n1
        assert "x=node:n1/out:[B, 16]" in rows[2][2]
        # n3 (add) feeds from n2 and external input
        assert "a=node:n2/out:[B, 16]" in rows[3][2]
        assert "b=input:x:[B, 16]" in rows[3][2]

    def test_repr_multi_output_split(self):
        """Multi-output split operator is branched, omitting index and printing named port targets."""
        source = (
            "linear(32)\n"
            "(a, b) = split(16)\n"
            "out = add(a, b)\n"
        )
        plan = resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert plan._chain is False
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert "index" not in headers
        assert headers == ["name", "operation", "input shapes", "output shapes"]
        assert len(rows) == 3

        # split outputs
        assert rows[1][1] == "split"
        assert "first=[B, 16]" in rows[1][3]
        assert "rest=[B, 16]" in rows[1][3]

        # add inputs from split ports
        assert "a=node:n1/first:[B, 16]" in rows[2][2]
        assert "b=node:n1/rest:[B, 16]" in rows[2][2]

    def test_repr_multi_output_chunk(self):
        """Multi-output chunk operator formats out0/out1 ports and omits index."""
        source = (
            "linear(32)\n"
            "(c1, c2) = chunk(2, dim=1)\n"
            "out = add(c1, c2)\n"
        )
        plan = resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert plan._chain is False
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert "index" not in headers
        assert len(rows) == 3

        # chunk outputs out0 and out1
        assert "out0=[B, 16]" in rows[1][3]
        assert "out1=[B, 16]" in rows[1][3]

        # add inputs
        assert "a=node:n1/out0:[B, 16]" in rows[2][2]
        assert "b=node:n1/out1:[B, 16]" in rows[2][2]

    def test_repr_single_output_chunk_is_not_chain(self):
        """H-03: chunk(1) with 1 output produces out0, so it is NOT classified as a chain."""
        source = (
            "linear(16)\n"
            "(c,) = chunk(1, dim=1)\n"
            "out = c\n"
        )
        plan = resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert plan._chain is False
        rep = repr(plan)

        headers, rows = _extract_table_header_and_rows(rep)
        assert "index" not in headers
        assert headers == ["name", "operation", "input shapes", "output shapes"]
        assert rows[1][1] == "chunk"
        assert "out0=[B, 16]" in rows[1][3]

    def test_repr_parity_between_plan_and_graph_module(self):
        """ResolvedPlan.__repr__ and GraphModule.__repr__ match table headers and rows exactly."""
        # 1. Unary chain parity
        chain_src = "linear(16)\nrelu()"
        chain_plan = resolve(chain_src, input_shape=("B", 8), output_shape=("B", 16))
        chain_mod = network(chain_src, input_shape=("B", 8), output_shape=("B", 16), device="cpu")

        plan_headers, plan_rows = _extract_table_header_and_rows(repr(chain_plan))
        mod_headers, mod_rows = _extract_table_header_and_rows(repr(chain_mod))
        assert plan_headers == mod_headers
        assert plan_rows == mod_rows

        # 2. Branched parity
        branch_src = "a = linear(16)\nout = add(a, x)"
        branch_plan = resolve(branch_src, input_shape=("B", 16), output_shape=("B", 16))
        branch_mod = network(branch_src, input_shape=("B", 16), output_shape=("B", 16), device="cpu")

        plan_headers, plan_rows = _extract_table_header_and_rows(repr(branch_plan))
        mod_headers, mod_rows = _extract_table_header_and_rows(repr(branch_mod))
        assert plan_headers == mod_headers
        assert plan_rows == mod_rows


# ==============================================================================
# M-10: Positional Argument Binding & Validation Stress Tests
# ==============================================================================


class _DummyModule(nn.Module):
    def forward(self, *args, **kwargs):
        return args[0] if args else kwargs.get("x")


class TestPositionalBindingStress:
    """Stress-testing make_operator and Operator positional argument binding constraints."""

    def test_make_operator_positional_rest_conflicts_with_positional_arg_raises_e_registry(self):
        """make_operator raises E_REGISTRY when positional_rest is used and any other arg is positional."""
        with pytest.raises(HNDLError) as exc:
            make_operator(
                _DummyModule,
                "dummy_op",
                identity="stress.dummy_op",
                summary="A dummy op for stress testing.",
                shape="x[B, F] -> out[B, F]",
                args={
                    "dims": Arg(INTS, help="Dimensions."),
                    "extra": Arg(int, positional=True, help="Extra int."),
                },
                positional_rest="dims",
            )
        assert exc.value.code == "E_REGISTRY"
        assert "Argument 'extra' cannot be positional when positional_rest ('dims') is used" in exc.value.message

    def test_make_operator_positional_rest_multiple_positional_args_raises_e_registry(self):
        """make_operator raises E_REGISTRY when multiple args are positional alongside positional_rest."""
        with pytest.raises(HNDLError) as exc:
            make_operator(
                _DummyModule,
                "multi_pos",
                identity="stress.multi_pos",
                summary="Multi positional op.",
                shape="x[B, F] -> out[B, F]",
                args={
                    "dims": Arg(INTS, help="Dimensions."),
                    "p1": Arg(int, positional=True, help="First."),
                    "p2": Arg(int, positional=True, help="Second."),
                },
                positional_rest="dims",
            )
        assert exc.value.code == "E_REGISTRY"
        assert "cannot be positional when positional_rest" in exc.value.message

    def test_make_operator_positional_rest_not_in_args_raises_e_registry(self):
        """make_operator raises E_REGISTRY when positional_rest names an argument not in args."""
        with pytest.raises(HNDLError) as exc:
            make_operator(
                _DummyModule,
                "missing_rest",
                identity="stress.missing_rest",
                summary="Missing rest arg op.",
                shape="x[B, F] -> out[B, F]",
                args={"a": Arg(int, help="Arg a.")},
                positional_rest="nonexistent",
            )
        assert exc.value.code == "E_REGISTRY"
        assert "positional_rest must name an 'ints' argument" in exc.value.message

    @pytest.mark.parametrize("bad_type", [int, str, float, bool])
    def test_make_operator_positional_rest_non_ints_type_raises_e_registry(self, bad_type):
        """make_operator raises E_REGISTRY when positional_rest names an argument whose type is not INTS."""
        with pytest.raises(HNDLError) as exc:
            make_operator(
                _DummyModule,
                "bad_rest_type",
                identity="stress.bad_rest_type",
                summary="Bad rest type op.",
                shape="x[B, F] -> out[B, F]",
                args={"bad_rest": Arg(bad_type, help="Not an ints arg.")},
                positional_rest="bad_rest",
            )
        assert exc.value.code == "E_REGISTRY"
        assert "positional_rest must name an 'ints' argument" in exc.value.message

    def test_operator_direct_instantiation_positional_conflict_raises_e_schema(self):
        """Direct Operator construction raises E_SCHEMA when positional_rest combines with positional=True."""
        with pytest.raises(HNDLError) as exc:
            Operator(
                alias="direct_op",
                identity="stress.direct_op",
                version=1,
                summary="summary",
                doc="doc",
                category="other",
                inputs=(Port("x"),),
                outputs=(Port("out"),),
                args={
                    "dims": Arg(INTS, help="Dimensions."),
                    "bad_pos": Arg(int, positional=True, help="Positional arg."),
                },
                symbols=frozenset(),
                shape_text="x -> out",
                positional_rest="dims",
            )
        assert exc.value.code == "E_SCHEMA"
        assert "Argument 'bad_pos' cannot be positional when positional_rest ('dims') is used" in exc.value.message

    def test_operator_positional_names_returns_empty_tuple_with_positional_rest(self):
        """Operator.positional_names returns () when positional_rest is defined."""
        op = make_operator(
            _DummyModule,
            "valid_rest",
            identity="stress.valid_rest",
            summary="Valid rest operator.",
            shape="x[B, F] -> out[B, F]",
            args={"dims": Arg(INTS, help="Dims.")},
            positional_rest="dims",
        )
        assert op.positional_rest == "dims"
        assert op.positional_names == ()

    def test_operator_positional_names_returns_declared_names_without_positional_rest(self):
        """Operator.positional_names returns tuple of positional args in declaration order when positional_rest is None."""
        op = make_operator(
            _DummyModule,
            "ordered_pos",
            identity="stress.ordered_pos",
            summary="Ordered positional operator.",
            shape="x[B, F] -> out[B, F]",
            args={
                "first": Arg(int, positional=True, help="First."),
                "middle_kw": Arg(int, positional=False, help="Keyword only."),
                "second": Arg(str, positional=True, help="Second."),
            },
        )
        assert op.positional_rest is None
        assert op.positional_names == ("first", "second")


# ==============================================================================
# M-11: PEP 561 Static Typing & Public API Introspection Stress Tests
# ==============================================================================


class TestPEP561TypingStress:
    """Stress-testing PEP 561 compliance, dataclass type hygiene, and public API type annotations."""

    def test_pep561_py_typed_marker_shipped(self):
        """py.typed marker file exists in src/hndl/ and is specified in MANIFEST.in."""
        marker = Path(__file__).resolve().parent.parent / "src" / "hndl" / "py.typed"
        assert marker.is_file(), f"PEP 561 marker {marker} must exist"

        manifest = Path(__file__).resolve().parent.parent / "MANIFEST.in"
        assert manifest.is_file()
        content = manifest.read_text(encoding="utf-8")
        assert "src/hndl/py.typed" in content

    def test_pep561_public_dataclasses_contain_no_bare_object_fields(self):
        """Exported dataclasses must use concrete field types and have zero fields typed as bare 'object'."""
        dataclass_types = [
            Arg, Example, hndl.Policy, ResolvedPlan,
            Node, Graph, ResolvedNode, Port, Operator,
        ]
        for cls in dataclass_types:
            assert is_dataclass(cls), f"{cls.__name__} must be a dataclass"
            for f in fields(cls):
                assert f.type is not object and f.type != "object", (
                    f"Dataclass {cls.__name__} field '{f.name}' has bare 'object' type annotation"
                )

    def test_pep561_core_public_api_functions_type_annotations(self):
        """All core public API functions must have explicit, concrete parameter and return annotations."""
        core_functions = [
            resolve,
            hndl.resolve_file,
            resolve_callable,
            hndl.operator,
            build,
            network,
            network_file,
            network_from_callable,
            parameter_counts,
        ]
        missing_report = {}
        for fn in core_functions:
            sig = inspect.signature(fn)
            issues = []
            for name, param in sig.parameters.items():
                if param.annotation is inspect.Parameter.empty:
                    issues.append(f"param '{name}' is unannotated")
                elif param.annotation is object or param.annotation == "object":
                    issues.append(f"param '{name}' is bare object")

            if sig.return_annotation is inspect.Signature.empty:
                issues.append("return is unannotated")
            elif sig.return_annotation is object or sig.return_annotation == "object":
                issues.append("return is bare object")

            if issues:
                missing_report[fn.__name__] = issues

        assert not missing_report, f"Core public API functions missing type annotations: {missing_report}"

    @pytest.mark.xfail(
        reason="M-11 edge case: 8 random initializer helpers exported in hndl.__all__ (src/hndl/initializers.py) "
               "lack parameter and return type annotations.",
        strict=False,
    )
    def test_pep561_public_initializer_functions_type_annotations(self):
        """Empirically audit the 8 random initializer functions in hndl.__all__ for type annotations."""
        initializers = [
            hndl.kaiming_normal,
            hndl.kaiming_uniform,
            hndl.normal,
            hndl.orthogonal,
            hndl.truncated_normal,
            hndl.uniform,
            hndl.xavier_normal,
            hndl.xavier_uniform,
        ]
        unannotated = {}
        for fn in initializers:
            sig = inspect.signature(fn)
            missing = [p for p, param in sig.parameters.items() if param.annotation is inspect.Parameter.empty]
            if sig.return_annotation is inspect.Signature.empty:
                missing.append("<return>")
            if missing:
                unannotated[fn.__name__] = missing

        assert not unannotated, (
            f"Public initializer helpers lack type annotations on parameters or return: {unannotated}"
        )
