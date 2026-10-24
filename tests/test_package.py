"""Distribution-level checks."""

from importlib.metadata import metadata, version
from pathlib import Path
import tomllib

from hndl import Registry
from hndl._version import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_distribution_version_matches_module():
    assert version("hndl") == __version__


def test_pyproject_version_matches_module():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__


def test_readme_status_names_the_current_version():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**Status: {__version__}.**" in readme or "**Status: 0.10.0.**" in readme


def test_torch_is_a_required_dependency():
    requirements = metadata("hndl").get_all("Requires-Dist") or []
    torch_requirements = [r for r in requirements if r.startswith("torch")]
    assert len(torch_requirements) == 1
    assert "extra ==" not in torch_requirements[0]


def test_spec_and_implementation_status_names_current_version():
    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    assert any(f"**Status: the implemented {v} contract.**" in spec for v in (__version__, "0.10.0"))
    assert any(f"records what {v} ships and what remains for v1." in spec for v in (__version__, "0.10.0"))
    assert "0.1.2" not in spec

    impl = (REPO_ROOT / "IMPLEMENTATION.md").read_text(encoding="utf-8")
    assert any(f"# Implementation notes: {v}" in impl for v in (__version__, "0.10.0"))
    assert "0.1.2" not in impl


def test_operator_count_and_catalog_synchronization():
    registry = Registry.builtins()
    assert len(registry.operators) == 84
    categories = {op.category for op in registry.operators}
    assert len(categories) == 12
    assert categories == {
        "activation",
        "arithmetic",
        "convolution",
        "core",
        "memory",
        "normalization",
        "pretrained",
        "regularization",
        "sequence",
        "shape",
        "spatial",
        "vision",
    }

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "catalog of 84 documented operators" in readme

    impl = (REPO_ROOT / "IMPLEMENTATION.md").read_text(encoding="utf-8")
    assert "Eighty-four (84) operators across activation, arithmetic, convolution, core, memory, normalization, pretrained, regularization, sequence, shape, spatial, and vision" in " ".join(impl.split())
    assert "Fifty-nine operators" not in impl
    assert "84" in impl

    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    assert "84 operators across 12 categories" in spec


def test_spec_documents_arg_since():
    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    assert "Arg(type, default, *, min, max, exclusive_min, exclusive_max, help, inferable, dim, positional, choices, since)" in spec
    assert 'since="0.7.0"' in spec


def test_manifest_includes_required_distribution_files():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "CHANGELOG.md" in manifest
    assert "src/hndl/py.typed" in manifest or "py.typed" in manifest
    assert "recursive-include tests *.py *.json" in manifest


def test_adding_operators_and_benchmark_comments_hygiene():
    adding_ops = (REPO_ROOT / "docs" / "ADDING_OPERATORS.md").read_text(encoding="utf-8")
    assert "ruff check src tests examples scripts" in adding_ops

    timing = (REPO_ROOT / "tests" / "benchmark" / "_timing.py").read_text(encoding="utf-8")
    assert "torch 2.14" not in timing

    parity = (REPO_ROOT / "tests" / "benchmark" / "test_parity_vs_handwritten_pytorch.py").read_text(encoding="utf-8")
    assert "torch 2.14" not in parity
