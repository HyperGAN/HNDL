"""README.md is the PyPI long description, so its links must be absolute.

A repository-relative target such as ``docs/operators/index.md`` resolves to
``https://pypi.org/project/hndl/docs/operators/index.md`` on the project page
and 404s. Only absolute URLs and same-page anchors survive the trip.
"""

from pathlib import Path
import re

README = Path(__file__).resolve().parents[1] / "README.md"

# ``[text](target)``, ``[text](target "title")``, ``![alt](target)``.
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
# ``[label]: target`` at the start of a line.
REFERENCE_LINK = re.compile(r"^\[[^\]]+\]:\s*(?P<target>\S+)", re.MULTILINE)
# ``<https://example.com>`` and, the failure we guard against, ``<SPEC.md>``.
AUTOLINK = re.compile(r"<(?P<target>[A-Za-z0-9][^<>\s]*\.[^<>\s]*)>")

ABSOLUTE = ("http://", "https://", "#")


def strip_code(text):
    """Drop fenced code blocks and inline code, which may contain brackets."""
    text = re.sub(r"^(```|~~~).*?^\1", "", text, flags=re.MULTILINE | re.DOTALL)
    return re.sub(r"`[^`]*`", "", text)


def readme_targets():
    text = strip_code(README.read_text(encoding="utf-8"))
    patterns = (INLINE_LINK, REFERENCE_LINK, AUTOLINK)
    return [match["target"] for pattern in patterns for match in pattern.finditer(text)]


def test_readme_has_links():
    assert len(readme_targets()) > 5


def test_readme_links_are_absolute():
    relative = sorted({t for t in readme_targets() if not t.startswith(ABSOLUTE)})
    assert not relative, (
        "README.md is the PyPI long description; these targets must be absolute "
        f"URLs (docs site or github.com/HyperGAN/HNDL/blob/master/...): {relative}"
    )
