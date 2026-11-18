import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mirror_github_release.py"
_SPEC = importlib.util.spec_from_file_location("hndl_mirror_github_release", _SCRIPT)
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)
VERSION = "0.12.0"
SOURCE = "a" * 40


@pytest.fixture
def distributions(tmp_path):
    (tmp_path / f"hndl-{VERSION}-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / f"hndl-{VERSION}.tar.gz").write_bytes(b"sdist")
    hashes = release.dist_hashes(VERSION, tmp_path)
    manifest = "".join(f"{digest}  {name}\n" for name, digest in hashes.items()).encode()
    return tmp_path, hashes, {**hashes, release.CHECKSUMS: hashlib.sha256(manifest).hexdigest()}


def gh_record(hashes, *, draft=True, source=SOURCE):
    return {
        "id": 42, "tag_name": f"v{VERSION}", "draft": draft, "prerelease": False,
        "target_commitish": source, "body": f"<!-- hndl-release-source: {source} -->",
        "html_url": f"https://github.com/HyperGAN/HNDL/releases/tag/v{VERSION}",
        "assets": [{"name": name, "digest": f"sha256:{sha}", "state": "uploaded"} for name, sha in hashes.items()],
    }


def test_conflicting_pypi_digest_stops_before_github(distributions):
    directory, pypi, _ = distributions
    name = next(iter(pypi))
    with (
        patch.object(release, "pypi_hashes", return_value={name: "0" * 64}),
        patch.object(release, "tag_commit") as tag,
        patch.object(release, "gh") as gh,
    ):
        with pytest.raises(RuntimeError, match="PyPI files.*differ"):
            release.mirror(VERSION, SOURCE, directory)
    tag.assert_not_called()
    gh.assert_not_called()


def test_waits_for_registry_to_show_the_complete_upload(distributions):
    _, pypi, _ = distributions
    with (
        patch.object(release, "pypi_hashes", side_effect=[{}, pypi]),
        patch.object(release.time, "sleep") as sleep,
    ):
        release.verify_pypi(VERSION, pypi)
    sleep.assert_called_once()


def test_mirror_creates_draft_and_checks_files_before_publishing(distributions):
    directory, pypi, complete = distributions
    records = [None, gh_record({}), gh_record(complete), gh_record(complete, draft=False)]
    with (
        patch.object(release, "pypi_hashes", return_value=pypi),
        patch.object(release, "tag_commit", side_effect=[None, None, SOURCE]),
        patch.object(release, "release_record", side_effect=records),
        patch.object(release, "gh") as gh,
    ):
        assert release.mirror(VERSION, SOURCE, directory).endswith(f"/v{VERSION}")
    create, upload, publish = (call.args for call in gh.call_args_list)
    assert "--draft" in create and create[create.index("--target") + 1] == SOURCE
    assert upload[:3] == ("release", "upload", f"v{VERSION}")
    assert set(upload[5:]) == {str(directory / name) for name in complete}
    assert publish[:3] == ("api", "--method", "PATCH") and "draft=false" in publish


def test_exact_draft_retry_uploads_only_missing_assets(distributions):
    directory, pypi, complete = distributions
    name = next(iter(pypi))
    records = [gh_record({name: pypi[name]}), gh_record(complete), gh_record(complete, draft=False)]
    with (
        patch.object(release, "pypi_hashes", return_value=pypi),
        patch.object(release, "tag_commit", side_effect=[None, None, SOURCE]),
        patch.object(release, "release_record", side_effect=records),
        patch.object(release, "gh") as gh,
    ):
        assert release.mirror(VERSION, SOURCE, directory).endswith(f"/v{VERSION}")
    upload, publish = (call.args for call in gh.call_args_list)
    assert set(upload[5:]) == {str(directory / file) for file in complete if file != name}
    assert publish[:3] == ("api", "--method", "PATCH")


def test_published_match_is_a_noop_and_existing_tag_is_pinned(distributions):
    directory, pypi, complete = distributions
    with (
        patch.object(release, "pypi_hashes", return_value=pypi),
        patch.object(release, "release_record", return_value=gh_record(complete, draft=False)),
        patch.object(release, "tag_commit", side_effect=[SOURCE, "b" * 40]),
        patch.object(release, "gh") as gh,
    ):
        assert release.mirror(VERSION, SOURCE, directory) == "unchanged"
        with pytest.raises(RuntimeError, match="already points"):
            release.mirror(VERSION, SOURCE, directory)
    gh.assert_not_called()
