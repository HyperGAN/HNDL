"""Mirror one tested HNDL release to GitHub after PyPI confirms its digests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


REPOSITORY = "HyperGAN/HNDL"
CHECKSUMS = "SHA256SUMS"


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def release_record(tag):
    result = gh("api", "--paginate", f"repos/{REPOSITORY}/releases?per_page=100",
                "--jq", f"[.[] | select(.tag_name == {json.dumps(tag)})]")
    records = [record for line in result.splitlines() for record in json.loads(line)]
    if len(records) > 1:
        raise RuntimeError(f"Multiple GitHub releases claim {tag}")
    return next(iter(records), None)


def dist_hashes(version, dist):
    files = [dist / f"hndl-{version}-py3-none-any.whl", dist / f"hndl-{version}.tar.gz"]
    return {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def pypi_hashes(version):
    with urlopen(f"https://pypi.org/pypi/hndl/{quote(version, safe='')}/json", timeout=20) as response:
        published = json.load(response)
    if published["info"]["name"].lower() != "hndl" or published["info"]["version"] != version:
        raise RuntimeError("PyPI returned a different package or release")
    return {item["filename"]: item["digests"]["sha256"] for item in published["urls"] if not item["yanked"]}


def verify_pypi(version, expected):
    # Allow the index to catch up with a successful upload. Conflicts fail immediately.
    for attempt in range(5):
        try:
            observed = pypi_hashes(version)
        except HTTPError as error:
            if error.code != 404:
                raise
            observed = {}
        if observed == expected:
            return
        if not observed.items() < expected.items():
            break
        if attempt < 4:
            time.sleep(3)
    raise RuntimeError(f"PyPI files for {version} differ from this run's tested distributions")


def tag_commit(tag):
    ref = f"refs/tags/{tag}"
    result = subprocess.check_output([
        "git", "ls-remote", f"https://github.com/{REPOSITORY}.git", ref, f"{ref}^{{}}",
    ], text=True)
    refs = {name: oid for oid, name in (line.split("\t") for line in result.splitlines())}
    return refs.get(f"{ref}^{{}}", refs.get(ref))


def asset_names(record, expected):
    observed = set()
    for asset in record["assets"]:
        name = asset["name"]
        if name not in expected or asset["state"] != "uploaded" or asset["digest"] != f"sha256:{expected[name]}":
            raise RuntimeError(f"Unexpected file in {record['tag_name']}: {name}")
        observed.add(name)
    return observed


def mirror(version, source, dist):
    tag = f"v{version}"
    expected = dist_hashes(version, dist)
    verify_pypi(version, expected)

    contents = "".join(f"{digest}  {name}\n" for name, digest in expected.items()).encode()
    (dist / CHECKSUMS).write_bytes(contents)
    expected[CHECKSUMS] = hashlib.sha256(contents).hexdigest()
    tagged = tag_commit(tag)
    if tagged not in (None, source):
        raise RuntimeError(f"{tag} already points to {tagged}, expected {source}")

    prerelease = bool(re.search(r"\d(?:a\d|b\d|rc\d|\.dev\d)", version, re.IGNORECASE))
    marker = f"<!-- hndl-release-source: {source} -->"
    record = release_record(tag)
    if record and not record["draft"]:
        if tagged != source or record["prerelease"] != prerelease or asset_names(record, expected) != expected.keys():
            raise RuntimeError(f"The published GitHub release {tag} requires review")
        return "unchanged"

    if record is None:
        flags = ["--prerelease"] if prerelease else []
        gh("release", "create", tag, "--repo", REPOSITORY, "--draft", "--target", source,
           "--title", f"HNDL {version}", "--generate-notes",
           "--notes", f"Files verified against PyPI.\n\n{marker}", *flags)
        record = release_record(tag)
    if (not record or not record["draft"] or record["prerelease"] != prerelease
            or record["target_commitish"] != source or marker not in (record["body"] or "")):
        raise RuntimeError(f"Review the draft for {tag} before uploading files")
    observed = asset_names(record, expected)
    missing = [str(dist / name) for name in expected.keys() - observed]
    if missing:
        gh("release", "upload", tag, "--repo", REPOSITORY, *sorted(missing))
    record = release_record(tag)
    if not record["draft"] or asset_names(record, expected) != expected.keys():
        raise RuntimeError(f"GitHub did not receive the complete tested draft for {tag}")
    if tag_commit(tag) not in (None, source):
        raise RuntimeError(f"{tag} changed while staging release files")

    latest = [] if prerelease else ["-f", "make_latest=legacy"]
    gh("api", "--method", "PATCH", f"repos/{REPOSITORY}/releases/{record['id']}", "-F", "draft=false", *latest)
    record = release_record(tag)
    if record["draft"] or tag_commit(tag) != source or asset_names(record, expected) != expected.keys():
        raise RuntimeError(f"GitHub did not publish the tested revision and files for {tag}")
    return record["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "mirror"])
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "hndl" or not re.fullmatch(r"[0-9][0-9a-zA-Z.!+]{0,64}", version):
        parser.error("Expected an HNDL package version")
    if args.action == "inspect":
        record = release_record(f"v{version}")
        if record:
            print(f"GitHub v{version}: {'draft' if record['draft'] else 'published'}")
        else:
            print(f"GitHub v{version}: available")
        return

    source = os.environ.get("GITHUB_SHA", "")
    if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY or os.environ.get("GITHUB_REF") != "refs/heads/master"
            or os.environ.get("GITHUB_EVENT_NAME") != "push" or not re.fullmatch(r"[0-9a-f]{40}", source)):
        parser.error("Mirroring requires a push to HyperGAN/HNDL master")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != source:
        raise RuntimeError("The checkout differs from the tested GitHub revision")
    subprocess.run(["git", "fetch", "--no-tags", "origin", "master"], check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "FETCH_HEAD"], check=True)
    print(mirror(version, source, args.dist))


if __name__ == "__main__":
    main()
