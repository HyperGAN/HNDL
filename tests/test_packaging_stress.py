"""Empirical stress test suite for packaging, sdist isolation, doc link rewriting,

and repository hygiene (H-07, M-14, L-10).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
from types import ModuleType
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Sdist Build & Isolation Fixtures & Tests (H-07, M-14)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def built_sdist_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the source distribution archive once per test session."""
    pytest.importorskip("build")
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("Not running inside a repository checkout")
    outdir = tmp_path_factory.mktemp("sdist_build")
    cmd = [
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--outdir",
        str(outdir),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"Failed to build sdist (exit {proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    tarballs = list(outdir.glob("*.tar.gz"))
    assert len(tarballs) == 1, f"Expected 1 tarball in {outdir}, found {tarballs}"
    return tarballs[0]


def test_sdist_archive_contains_all_required_files(built_sdist_archive: Path) -> None:
    """Verify that the sdist tarball packages CHANGELOG, py.typed, and plan fixtures."""
    with tarfile.open(built_sdist_archive, "r:gz") as tar:
        names = tar.getnames()

    # Identify top-level folder name (e.g., 'hndl-0.8.0')
    prefix = names[0].split("/")[0]
    assert prefix.startswith("hndl-"), f"Unexpected archive root folder: {prefix}"

    # 1. CHANGELOG.md exists in root
    assert f"{prefix}/CHANGELOG.md" in names, "CHANGELOG.md missing from sdist root"

    # 2. py.typed exists in package
    assert f"{prefix}/src/hndl/py.typed" in names, "src/hndl/py.typed missing from sdist"

    # 3. All 5 plan fixtures exist
    expected_fixtures = [
        "attention.json",
        "feed_forward.json",
        "gpt_tiny.json",
        "linear_mlp.json",
        "linear_options.json",
    ]
    for fix in expected_fixtures:
        fixture_path = f"{prefix}/tests/fixtures/plans_0_6_0/{fix}"
        assert fixture_path in names, f"Fixture {fix} missing from sdist: {fixture_path}"

    # 4. generate.py fixture generator exists
    assert f"{prefix}/tests/fixtures/plans_0_6_0/generate.py" in names

    # 5. Core metadata and root docs exist
    for root_doc in ["README.md", "SPEC.md", "IMPLEMENTATION.md", "CONTRIBUTING.md", "LICENSE", "MANIFEST.in", "pyproject.toml"]:
        assert f"{prefix}/{root_doc}" in names, f"{root_doc} missing from sdist"

    # 6. Sdist hygiene: forbid leaking VCS, caches, or worktrees into source distribution
    forbidden_substrings = [
        "/.git/",
        "/.github/",
        "/.claude/",
        "/.agents/",
        "/.venv/",
        "/__pycache__/",
        "/.pytest_cache/",
    ]
    for name in names:
        for forbidden in forbidden_substrings:
            assert forbidden not in name, f"Sdist tarball leaks forbidden directory {forbidden}: {name}"


def test_sdist_unpack_and_run_saved_plan_compat(built_sdist_archive: Path, tmp_path: Path) -> None:
    """Unpack sdist into a clean isolated directory outside repo checkout and run saved plan tests."""
    unpack_dir = tmp_path / "clean_unpack"
    unpack_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(built_sdist_archive, "r:gz") as tar:
        tar.extractall(path=unpack_dir)

    extracted_roots = list(unpack_dir.glob("hndl-*"))
    assert len(extracted_roots) == 1, f"Expected 1 extracted root, found {extracted_roots}"
    sdist_dir = extracted_roots[0]

    # Verify key files exist on disk
    changelog = sdist_dir / "CHANGELOG.md"
    assert changelog.is_file(), f"CHANGELOG.md not found at {changelog}"
    assert changelog.stat().st_size > 1000, "CHANGELOG.md appears empty or truncated"

    py_typed = sdist_dir / "src" / "hndl" / "py.typed"
    assert py_typed.is_file(), f"py.typed not found at {py_typed}"

    fixtures_dir = sdist_dir / "tests" / "fixtures" / "plans_0_6_0"
    assert fixtures_dir.is_dir(), f"Fixtures directory not found at {fixtures_dir}"

    expected_fixtures = [
        "attention.json",
        "feed_forward.json",
        "gpt_tiny.json",
        "linear_mlp.json",
        "linear_options.json",
    ]
    for fix_name in expected_fixtures:
        fix_path = fixtures_dir / fix_name
        assert fix_path.is_file(), f"Plan fixture {fix_name} missing from unpacked sdist"
        data = json.loads(fix_path.read_text(encoding="utf-8"))
        assert "plan" in data, f"Fixture {fix_name} missing 'plan' key"
        assert "source" in data, f"Fixture {fix_name} missing 'source' key"
        assert "input_shape" in data, f"Fixture {fix_name} missing 'input_shape' key"
        assert "output_shape" in data, f"Fixture {fix_name} missing 'output_shape' key"

    # Run pytest tests/test_saved_plan_compat.py inside the unpacked directory
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_saved_plan_compat.py", "-v"],
        cwd=sdist_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"pytest tests/test_saved_plan_compat.py failed in unpacked sdist:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "13 passed" in proc.stdout, f"Expected 13 tests to pass, got output:\n{proc.stdout}"


def test_sdist_saved_plan_compat_mutation_oracle(built_sdist_archive: Path, tmp_path: Path) -> None:
    """Verify that corrupting a fixture in unpacked sdist causes test_saved_plan_compat.py to fail."""
    unpack_dir = tmp_path / "corrupted_unpack"
    unpack_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(built_sdist_archive, "r:gz") as tar:
        tar.extractall(path=unpack_dir)

    sdist_dir = list(unpack_dir.glob("hndl-*"))[0]
    target_fixture = sdist_dir / "tests" / "fixtures" / "plans_0_6_0" / "attention.json"
    assert target_fixture.is_file()

    # Corrupt the fixture's semantic digest
    content = json.loads(target_fixture.read_text(encoding="utf-8"))
    plan_dict = json.loads(content["plan"])
    plan_dict["semantic_digest"] = "corrupted_digest_0000000000000000"
    content["plan"] = json.dumps(plan_dict)
    target_fixture.write_text(json.dumps(content), encoding="utf-8")

    # Run pytest - must fail due to corrupted digest
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_saved_plan_compat.py"],
        cwd=sdist_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode != 0, "Corrupted fixture unexpectedly passed test suite!"
    assert "FAILED" in proc.stdout or "FAILED" in proc.stderr


# ---------------------------------------------------------------------------
# Doc Link Rewriting Stress Tests (M-14)
# ---------------------------------------------------------------------------

def _load_gen_doc_pages() -> ModuleType:
    """Load scripts/gen_doc_pages.py with mock mkdocs dependencies if uninstalled."""
    if not (REPO_ROOT / "scripts" / "gen_doc_pages.py").is_file():
        pytest.skip("scripts/gen_doc_pages.py not present")
    module_name = "scripts.gen_doc_pages"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for mod in ["mkdocs", "mkdocs.structure", "mkdocs.structure.files", "mkdocs_gen_files"]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    import scripts.gen_doc_pages as gdp
    return gdp


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Base site URL and anchors
        ("https://hypergan.github.io/HNDL/", "index.md"),
        ("https://hypergan.github.io/HNDL/#quick-start", "index.md#quick-start"),
        ("http://hypergan.github.io/HNDL/", "index.md"),
        ("http://hypergan.github.io/HNDL/#overview", "index.md#overview"),
        # README aliases
        ("https://hypergan.github.io/HNDL/README", "index.md"),
        ("https://hypergan.github.io/HNDL/README/", "index.md"),
        ("https://hypergan.github.io/HNDL/README.md", "index.md"),
        ("https://hypergan.github.io/HNDL/README#installation", "index.md#installation"),
        ("https://hypergan.github.io/HNDL/README.md#features", "index.md#features"),
        # Operator catalog root
        ("https://hypergan.github.io/HNDL/operators", "operators/index.md"),
        ("https://hypergan.github.io/HNDL/operators/", "operators/index.md"),
        ("https://hypergan.github.io/HNDL/operators#catalog", "operators/index.md#catalog"),
        ("https://hypergan.github.io/HNDL/operators/#catalog", "operators/index.md#catalog"),
        ("https://hypergan.github.io/HNDL/operators/index", "operators/index.md"),
        ("https://hypergan.github.io/HNDL/operators/index/", "operators/index.md"),
        ("https://hypergan.github.io/HNDL/operators/index.md", "operators/index.md"),
        ("https://hypergan.github.io/HNDL/operators/index.md#summary", "operators/index.md#summary"),
        # Operator subpages
        ("https://hypergan.github.io/HNDL/operators/linear", "operators/linear.md"),
        ("https://hypergan.github.io/HNDL/operators/linear/", "operators/linear.md"),
        ("https://hypergan.github.io/HNDL/operators/linear.md", "operators/linear.md"),
        ("https://hypergan.github.io/HNDL/operators/attention#parameters", "operators/attention.md#parameters"),
        ("https://hypergan.github.io/HNDL/operators/attention/#parameters", "operators/attention.md#parameters"),
        ("https://hypergan.github.io/HNDL/operators/conv2d#stride", "operators/conv2d.md#stride"),
        ("https://hypergan.github.io/HNDL/operators/conv2d/#stride", "operators/conv2d.md#stride"),
        # Documentation subpages
        ("https://hypergan.github.io/HNDL/networks", "networks.md"),
        ("https://hypergan.github.io/HNDL/networks/", "networks.md"),
        ("https://hypergan.github.io/HNDL/networks.md", "networks.md"),
        ("https://hypergan.github.io/HNDL/networks#autoencoder", "networks.md#autoencoder"),
        ("https://hypergan.github.io/HNDL/pretrained", "pretrained.md"),
        ("https://hypergan.github.io/HNDL/pretrained/", "pretrained.md"),
        ("https://hypergan.github.io/HNDL/pretrained#capturing-intermediate-activations", "pretrained.md#capturing-intermediate-activations"),
        ("https://hypergan.github.io/HNDL/SPEC", "SPEC.md"),
        ("https://hypergan.github.io/HNDL/SPEC/", "SPEC.md"),
        ("https://hypergan.github.io/HNDL/SPEC.md", "SPEC.md"),
        ("https://hypergan.github.io/HNDL/SPEC#loading-limits-and-trust-boundaries", "SPEC.md#loading-limits-and-trust-boundaries"),
        ("https://hypergan.github.io/HNDL/SPEC/#loading-limits-and-trust-boundaries", "SPEC.md#loading-limits-and-trust-boundaries"),
        ("https://hypergan.github.io/HNDL/IMPLEMENTATION", "IMPLEMENTATION.md"),
        ("https://hypergan.github.io/HNDL/IMPLEMENTATION/", "IMPLEMENTATION.md"),
        ("https://hypergan.github.io/HNDL/CHANGELOG", "CHANGELOG.md"),
        ("https://hypergan.github.io/HNDL/CHANGELOG/", "CHANGELOG.md"),
        ("https://hypergan.github.io/HNDL/CONTRIBUTING", "CONTRIBUTING.md"),
        ("https://hypergan.github.io/HNDL/CONTRIBUTING/", "CONTRIBUTING.md"),
        ("https://hypergan.github.io/HNDL/ADDING_OPERATORS", "ADDING_OPERATORS.md"),
        ("https://hypergan.github.io/HNDL/ADDING_OPERATORS/", "ADDING_OPERATORS.md"),
        # External URLs - must remain unchanged
        ("https://github.com/HyperGAN/HNDL", "https://github.com/HyperGAN/HNDL"),
        ("https://github.com/HyperGAN/HNDL/issues", "https://github.com/HyperGAN/HNDL/issues"),
        ("https://pypi.org/project/hndl/", "https://pypi.org/project/hndl/"),
        ("https://docs.pytorch.org/docs/stable/index.html", "https://docs.pytorch.org/docs/stable/index.html"),
        ("mailto:dev@hypergan.org", "mailto:dev@hypergan.org"),
        ("#section-anchor", "#section-anchor"),
        # Repository-relative targets
        ("README.md", "index.md"),
        ("SPEC.md", "SPEC.md"),
        ("IMPLEMENTATION.md", "IMPLEMENTATION.md"),
        ("CHANGELOG.md", "CHANGELOG.md"),
        ("CONTRIBUTING.md", "CONTRIBUTING.md"),
        ("docs/pretrained.md", "pretrained.md"),
        ("docs/operators/linear.md", "operators/linear.md"),
        ("examples/networks/autoencoder.py", "https://github.com/HyperGAN/HNDL/blob/master/examples/networks/autoencoder.py"),
        ("LICENSE", "https://github.com/HyperGAN/HNDL/blob/master/LICENSE"),
    ],
)
def test_doc_pages_rewrite_target_stress_matrix(url: str, expected: str) -> None:
    """Stress-test rewrite_target across docs URLs, anchors, subpages, and repo links."""
    gdp = _load_gen_doc_pages()
    actual = gdp.rewrite_target(url)
    assert actual == expected, f"rewrite_target({url!r}) produced {actual!r}, expected {expected!r}"


def test_rewrite_links_fenced_code_block_protection() -> None:
    """Verify that links inside backtick or tilde code fences are preserved verbatim."""
    gdp = _load_gen_doc_pages()

    markdown_input = (
        "Outside link: [catalog](https://hypergan.github.io/HNDL/operators/)\n"
        "\n"
        "```python\n"
        "# Inside backtick code fence: [catalog](https://hypergan.github.io/HNDL/operators/)\n"
        "url = 'https://hypergan.github.io/HNDL/operators/'\n"
        "```\n"
        "\n"
        "Middle link: [networks](https://hypergan.github.io/HNDL/networks/)\n"
        "\n"
        "~~~markdown\n"
        "# Inside tilde code fence: [networks](https://hypergan.github.io/HNDL/networks/)\n"
        "~~~\n"
        "\n"
        "Final link: [`attention`](https://hypergan.github.io/HNDL/operators/attention/)\n"
    )

    rewritten = gdp.rewrite_links(markdown_input)

    # Outside links must be rewritten
    assert "[catalog](operators/index.md)" in rewritten
    assert "[networks](networks.md)" in rewritten
    assert "[`attention`](operators/attention.md)" in rewritten

    # Inside code blocks must remain untouched
    assert "# Inside backtick code fence: [catalog](https://hypergan.github.io/HNDL/operators/)" in rewritten
    assert "url = 'https://hypergan.github.io/HNDL/operators/'" in rewritten
    assert "# Inside tilde code fence: [networks](https://hypergan.github.io/HNDL/networks/)" in rewritten


def test_all_repository_markdown_links_rewrite_to_valid_files() -> None:
    """Scan all markdown files across repository and verify all rewritten docs links target real files."""
    gdp = _load_gen_doc_pages()

    link_pattern = re.compile(r"\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")
    all_md_files = list(REPO_ROOT.glob("*.md")) + list((REPO_ROOT / "docs").rglob("*.md"))
    assert len(all_md_files) >= 70, f"Expected at least 70 markdown files, found {len(all_md_files)}"

    docs_root = REPO_ROOT / "docs"

    for md_path in all_md_files:
        content = md_path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(content):
            target = match.group("target")
            if any(target.startswith(prefix) for prefix in gdp.DOCS_PREFIXES):
                rewritten = gdp.rewrite_target(target)
                # Ensure no external loopback remains
                assert not rewritten.startswith(gdp.DOCS_PREFIXES), (
                    f"In {md_path.relative_to(REPO_ROOT)}, link {target} was not rewritten to an internal path: {rewritten}"
                )
                clean_target, _, _ = rewritten.partition("#")
                # Clean target must correspond to an existing site doc
                candidate_paths = [
                    docs_root / clean_target,
                    REPO_ROOT / clean_target,
                ]
                if clean_target == "index.md":
                    candidate_paths.append(REPO_ROOT / "README.md")

                exists = any(p.exists() for p in candidate_paths)
                assert exists, (
                    f"In {md_path.relative_to(REPO_ROOT)}, link {target} rewrote to {rewritten}, "
                    f"but target file {clean_target} does not exist in site docs or repo root"
                )


# ---------------------------------------------------------------------------
# Repository & Git Hygiene Stress Tests (L-10)
# ---------------------------------------------------------------------------

def test_git_worktrees_zero_stale_entries() -> None:
    """Verify git worktree list contains 0 stale/linked worktrees (only main repository)."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("Not running inside a git repository checkout")
    proc = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    worktree_lines = [line for line in proc.stdout.splitlines() if line.startswith("worktree ")]
    assert len(worktree_lines) == 1, (
        f"Expected exactly 1 active worktree (main checkout), found {len(worktree_lines)}:\n"
        f"{proc.stdout}"
    )


def test_claude_worktrees_directory_is_strictly_empty() -> None:
    """Verify that .claude/worktrees/ directory has 0 stale nested checkouts."""
    claude_worktrees = REPO_ROOT / ".claude" / "worktrees"
    if claude_worktrees.exists():
        entries = list(claude_worktrees.iterdir())
        assert len(entries) == 0, f"Found {len(entries)} stale items in {claude_worktrees}: {entries}"


def test_git_merged_branches_are_pruned() -> None:
    """Verify that all branches fully merged into develop have been pruned."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("Not running inside a git repository checkout")
    has_develop = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/heads/develop"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if has_develop.returncode != 0:
        pytest.skip("Local develop branch ref not present (e.g. shallow CI checkout)")
    proc = subprocess.run(
        ["git", "branch", "--merged", "develop"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    merged_branches = [
        line.strip().lstrip("* ").strip()
        for line in proc.stdout.splitlines()
        if line.strip()
    ]
    # develop and master are the only persistent branches allowed to be merged
    for branch in merged_branches:
        assert branch in ("develop", "master") or branch.startswith("("), (
            f"Stale merged branch '{branch}' found in local git branches! "
            f"Expected only develop and master to be merged into develop."
        )


# ---------------------------------------------------------------------------
# Pytest Norecursedirs Stress Tests (L-10)
# ---------------------------------------------------------------------------

def test_pyproject_contains_required_norecursedirs() -> None:
    """Verify pyproject.toml explicitly configures norecursedirs with .claude and .agents."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    ini_options = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
    norecursedirs = ini_options.get("norecursedirs", [])

    assert ".*" in norecursedirs, "norecursedirs missing '.*'"
    assert "build" in norecursedirs, "norecursedirs missing 'build'"
    assert "dist" in norecursedirs, "norecursedirs missing 'dist'"
    assert ".claude" in norecursedirs, "norecursedirs missing '.claude'"
    assert ".agents" in norecursedirs, "norecursedirs missing '.agents'"


def test_norecursedirs_prevents_scanning_claude_and_agents(tmp_path: Path) -> None:
    """Empirically test that pytest collection ignores test files in .claude and .agents."""
    test_root = tmp_path / "norecurse_harness"
    test_root.mkdir(parents=True, exist_ok=True)

    # Replicate pytest.ini_options from pyproject.toml
    pyproject_content = (
        "[tool.pytest.ini_options]\n"
        "testpaths = ['tests']\n"
        "norecursedirs = ['.*', 'build', 'dist', '.claude', '.agents']\n"
    )
    (test_root / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    # Create legitimate tests
    (test_root / "tests").mkdir()
    (test_root / "tests" / "test_legitimate.py").write_text("def test_ok(): assert True\n", encoding="utf-8")

    # Create test files in forbidden directories that would fail if executed
    (test_root / ".claude").mkdir()
    (test_root / ".claude" / "test_claude_trap.py").write_text("def test_claude_trap(): assert False\n", encoding="utf-8")
    (test_root / ".claude" / "worktrees" / "nested_wt").mkdir(parents=True)
    (test_root / ".claude" / "worktrees" / "nested_wt" / "test_nested_trap.py").write_text("def test_nested_trap(): assert False\n", encoding="utf-8")

    (test_root / ".agents").mkdir()
    (test_root / ".agents" / "test_agents_trap.py").write_text("def test_agents_trap(): assert False\n", encoding="utf-8")

    # Run pytest in collection mode
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=test_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"pytest collection failed:\n{proc.stderr}"

    output_lines = proc.stdout.splitlines()
    assert any("test_legitimate.py::test_ok" in line for line in output_lines), "Legitimate test was not collected"
    assert not any("test_claude_trap" in line for line in output_lines), ".claude test was unexpectedly collected"
    assert not any("test_nested_trap" in line for line in output_lines), "Nested .claude worktree test was collected"
    assert not any("test_agents_trap" in line for line in output_lines), ".agents test was unexpectedly collected"

    # Also run the tests directly to ensure none of the trap tests run and fail
    run_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        cwd=test_root,
        capture_output=True,
        text=True,
    )
    assert run_proc.returncode == 0, f"pytest execution failed, traps were run:\n{run_proc.stdout}"
    assert "1 passed" in run_proc.stdout
