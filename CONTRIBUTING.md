# Developing HNDL

Use Python 3.11–3.14 on Linux. Create an environment and install the editable
package with its development tools:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check src tests examples scripts
python -m hndl.docs --check
```

PyTorch is a dependency. CUDA tests run when a CUDA device is available; CPU
CI does not establish GPU correctness. Operator documentation under
`docs/operators` is generated from the `@operator` declarations; regenerate it
with `python -m hndl.docs` after changing one. See
[docs/ADDING_OPERATORS.md](docs/ADDING_OPERATORS.md) to add an operator.

Changes go through pull requests. CI tests the CPU backend on Python 3.11 and
3.14, testing PyTorch 2.6 on Python 3.11 and the latest supported 2.x version
on Python 3.14. Keep the explicit activation and implicit
current-tensor behavior consistent between declarative configurations and
trusted Python callables. Configuration input must never be executed as Python.

Build release candidates with:

```sh
python -m build
python -m twine check --strict dist/*
```

The distribution version lives in `pyproject.toml` and
`src/hndl/_version.py`; update both together, and record the release in
`CHANGELOG.md`. Building a distribution does not publish it to PyPI.

## Documentation site

<https://hypergan.github.io/HNDL/> is built with MkDocs and the Material theme
from the Markdown already in the repository. Build it locally with:

```sh
python -m pip install -e '.[docs]'
mkdocs serve
```

`mkdocs.yml` configures the site and `scripts/gen_doc_pages.py` stages the root
Markdown files into it, rewriting their repository-relative links and
generating the navigation from `docs/operators/index.md` — a new operator
appears on the site as soon as `python -m hndl.docs` regenerates that page, with
no edit to `mkdocs.yml`. Pull requests run `mkdocs build --strict`, which fails
on a broken internal link; pushes to `master` deploy the site.

## Publishing

Releases are published by `.github/workflows/publish.yml` when a GitHub
release is published. The release tag must be `v<version>`, matching
`pyproject.toml`. The workflow checks the tag, builds the distributions, installs
the wheel in a clean environment with the CPU backend, runs the test suite and
the documentation check against it, and only then uploads. A failure at any
step prevents publishing. The wheel and source archive are retained as workflow
artifacts either way.

The upload authenticates in one of two ways; the first one that applies wins.

1. **API token secret.** Create a PyPI API token and store it as the
   repository secret `PYPI_API_TOKEN` (or as a secret on the GitHub `pypi`
   environment). Until the project exists on PyPI the token must be
   account-scoped; after the first upload it can be replaced with a token
   scoped to `hndl`.
2. **Trusted Publishing.** With no secret set, the workflow uses PyPI's
   short-lived OIDC credentials. Register a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   for project `hndl`, owner `HyperGAN`, repository `HNDL`, workflow
   `publish.yml`, environment `pypi`. This is PyPI account configuration, not
   a repository secret.

The `pypi` GitHub environment is created on first use; add release reviewers
to it if publishing should require approval.

To rehearse without publishing, run the workflow manually from the Actions tab
(`workflow_dispatch`). The manual run builds and tests the distributions and
uploads them as artifacts; the publish job is skipped.

To release: update the version in `pyproject.toml` and `src/hndl/_version.py`,
record the release in `CHANGELOG.md`, merge, then create a GitHub release with
tag `v<version>` on `master`. Publishing the release triggers the workflow.
