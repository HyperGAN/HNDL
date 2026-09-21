"""Distribution-level checks."""

from importlib.metadata import metadata, version

from hndl._version import __version__


def test_distribution_version_matches_module():
    assert version("hndl") == __version__


def test_torch_is_a_required_dependency():
    requirements = metadata("hndl").get_all("Requires-Dist") or []
    torch_requirements = [r for r in requirements if r.startswith("torch")]
    assert len(torch_requirements) == 1
    assert "extra ==" not in torch_requirements[0]
