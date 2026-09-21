# Developing HNDL

Use Python 3.11–3.14 on Linux. Create an environment and install the editable
package with its development tools:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check src tests examples
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
`src/hndl/_version.py`; update both together. The initial version is an alpha,
`0.1.0a1`. Building a distribution does not publish it to PyPI.

## Publishing

An owner must first configure a [PyPI pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for project `hndl`, GitHub owner `HyperGAN`, repository `HNDL`, workflow
`publish.yml`, and environment `pypi`. Set any desired release reviewers on the
GitHub `pypi` environment. This is account configuration, not a repository secret.

Once configured, publish a GitHub release tagged `v0.1.0a1` (or the matching
future package version). The publishing workflow checks the tag, builds the
distributions, tests the installed wheel with the CPU backend, and uploads
using PyPI's short-lived OIDC credentials. A failed test prevents publishing.
The wheel and source archive are also retained as workflow artifacts.
