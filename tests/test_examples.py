"""Cross-component acceptance examples, independent of internal solver rules."""

import ast
import re
from pathlib import Path
import subprocess
import sys

import pytest

from hndl import HNDLError, ops, resolve, resolve_callable, resolve_file


def generator_source():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    for block in re.findall(r"```python\n(.*?)\n```", readme.read_text(), re.S):
        for statement in ast.parse(block).body:
            if (isinstance(statement, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "generator_config"
                            for t in statement.targets)):
                assert isinstance(statement.value, ast.Constant)
                return statement.value.value
    raise AssertionError("README generator example disappeared")


@pytest.mark.parametrize("target,seed,width", [
    ((32, 32), (4, 4), 8192),
    ((64, 64), (8, 8), 32768),
    ((32, 64), (4, 8), 16384),
])
def test_readme_generator_targets(target, seed, width):
    plan = resolve(generator_source(), input_shape=("B", 128),
                   output_shape=("B", 3, *target))
    assert plan.nodes[0].output_shapes["out"] == ("B", width)
    assert plan.nodes[2].output_shapes["out"] == ("B", 512, *seed)
    assert len(plan.nodes) == 11


@pytest.mark.parametrize("source,target", [
    (None, (30, 30)),
    ("literal", (32, 32)),
])
def test_readme_impossible_generator_has_no_backend_allocation(source, target):
    config = generator_source()
    if source == "literal":
        config = config.replace("linear()", "linear(128)", 1)
    with pytest.raises(HNDLError) as exc:
        resolve(config, input_shape=("B", 128), output_shape=("B", 3, *target))
    assert exc.value.code in {"E_CONSTRAINT", "E_RESHAPE"}


def test_frontends_files_and_saved_plan_have_same_numerical_identity(tmp_path):
    source = "linear(64)\nrelu()\nlinear()\n"
    path = tmp_path / "classifier.hndl"
    path.write_text(source)
    calls = []

    def classifier(x):
        calls.append(x)
        ops.linear(64)
        ops.relu()
        ops.linear()

    contracts = dict(input_shape=("B", 128), output_shape=("B", 10))
    config_plan = resolve(source, **contracts)
    file_plan = resolve_file(path, **contracts)
    native_plan = resolve_callable(classifier, **contracts)
    restored = type(config_plan).from_json(config_plan.to_json())
    assert len(calls) == 1
    assert len({plan.semantic_digest for plan in
                (config_plan, file_plan, native_plan, restored)}) == 1
    assert "[B, 10]" in str(config_plan)


def test_pure_resolve_and_restore_never_import_torch():
    # A fresh process catches accidental transitive backend imports even when
    # other tests in this process have already imported torch.
    script = '''
import sys
import importlib.abc
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("pure HNDL imported torch")
sys.meta_path.insert(0, RejectTorch())
from hndl import resolve
plan = resolve("linear(64); relu(); linear()",
               input_shape=("B", 128), output_shape=("B", 10))
restored = type(plan).from_json(plan.to_json())
assert restored.semantic_digest == plan.semantic_digest
assert "torch" not in sys.modules
print("pure core passed")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pure core passed"
