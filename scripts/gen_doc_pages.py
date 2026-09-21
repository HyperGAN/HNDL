"""Stage the repository's Markdown into the documentation site.

Run by the ``mkdocs-gen-files`` plugin during ``mkdocs build``. The guide lives
in Markdown files that must stay readable on GitHub, so nothing here rewrites
prose: the root pages (README, SPEC, ...) are copied into the site with their
repository-relative links repointed at the equivalent site pages, and the
navigation is written as a ``SUMMARY.md`` for ``mkdocs-literate-nav``.

The operator section of the navigation is derived from
``docs/operators/index.md``, which ``python -m hndl.docs`` generates, so adding
an operator never requires editing ``mkdocs.yml``.
"""

from pathlib import Path
import re

from mkdocs.structure.files import InclusionLevel
import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
BLOB = "https://github.com/HyperGAN/HNDL/blob/master/"

# Root Markdown file -> its page in the site. ``docs/`` is the site root, so a
# link to ``docs/pretrained.md`` becomes ``pretrained.md``.
ROOT_PAGES = {
    "README.md": "index.md",
    "SPEC.md": "SPEC.md",
    "IMPLEMENTATION.md": "IMPLEMENTATION.md",
    "CHANGELOG.md": "CHANGELOG.md",
    "CONTRIBUTING.md": "CONTRIBUTING.md",
}

LINK = re.compile(r"\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")
CATEGORY = re.compile(r"^## +(?P<name>.+?)\s*$")
OPERATOR = re.compile(r"^\| \[`(?P<alias>[^`]+)`\]\((?P<page>[^)]+)\)")


def rewrite_target(target):
    """Point a repository-relative link at the matching page of the site."""
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return target
    path, _, anchor = target.partition("#")
    if path in ROOT_PAGES:
        path = ROOT_PAGES[path]
    elif path.startswith("docs/"):
        path = path[len("docs/"):]
    else:
        # Not part of the site (examples, LICENSE): keep it on GitHub.
        return BLOB + target
    return f"{path}#{anchor}" if anchor else path


def rewrite_links(text):
    """Rewrite every Markdown link outside of fenced code blocks."""
    out = []
    fence = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
        elif stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
        else:
            line = LINK.sub(
                lambda match: "](" + rewrite_target(match["target"]) + (match["title"] or "") + ")",
                line,
            )
        out.append(line)
    return "".join(out)


def operator_nav():
    """Navigation entries for docs/operators, grouped by the catalog's categories."""
    index = (ROOT / "docs" / "operators" / "index.md").read_text(encoding="utf-8")
    entries = []
    for line in index.splitlines():
        category = CATEGORY.match(line)
        if category:
            entries.append(f"    * {category['name'].replace('_', ' ').capitalize()}")
            continue
        operator = OPERATOR.match(line)
        if operator:
            entries.append(f"        * [{operator['alias']}](operators/{operator['page']})")
    if not entries:
        raise SystemExit("No operators found in docs/operators/index.md")
    return entries


for source, page in ROOT_PAGES.items():
    with mkdocs_gen_files.open(page, "w") as handle:
        handle.write(rewrite_links((ROOT / source).read_text(encoding="utf-8")))
    mkdocs_gen_files.set_edit_path(page, f"../{source}")

nav = [
    "* [Home](index.md)",
    "* Guide",
    "    * [Pretrained networks](pretrained.md)",
    "    * [Adding an operator](ADDING_OPERATORS.md)",
    "* Operators",
    "    * [Catalog](operators/index.md)",
    *operator_nav(),
    "* [Networks](networks.md)",
    "* [Specification](SPEC.md)",
    "* [Implementation notes](IMPLEMENTATION.md)",
    "* [Changelog](CHANGELOG.md)",
    "* [Contributing](CONTRIBUTING.md)",
]

with mkdocs_gen_files.open("SUMMARY.md", "w") as handle:
    handle.write("\n".join(nav) + "\n")

# literate-nav reads SUMMARY.md out of the file collection, so the site does not
# need a page rendering it. Excluding it is best effort: publishing the extra
# page is better than failing the build if MkDocs renames this API.
try:
    summary = mkdocs_gen_files.FilesEditor.current().files.get_file_from_path("SUMMARY.md")
    summary.inclusion = InclusionLevel.EXCLUDED
except Exception as error:  # pragma: no cover - depends on the MkDocs version
    print(f"gen_doc_pages: leaving SUMMARY.md in the site ({error})")
