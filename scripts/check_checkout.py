"""Optional release-maintainer checkout checks. Run with pytest on this file."""

from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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
