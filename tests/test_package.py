"""Distribution-level checks."""

from importlib.metadata import metadata, version
from pathlib import Path

from hndl._version import __version__


def test_distribution_version_matches_module():
    assert version("hndl") == __version__


def test_readme_status_names_the_current_version():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert f"**Status: {__version__}.**" in readme


def test_torch_is_a_required_dependency():
    requirements = metadata("hndl").get_all("Requires-Dist") or []
    torch_requirements = [r for r in requirements if r.startswith("torch")]
    assert len(torch_requirements) == 1
    assert "extra ==" not in torch_requirements[0]
