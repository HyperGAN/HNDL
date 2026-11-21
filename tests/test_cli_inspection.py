from pathlib import Path
import subprocess
import sys

from hndl import Registry, resolve
from hndl.__main__ import _load_plan, main
from hndl._inspection import InspectionRegistry
from tests._network_examples import load, names


RECIPES = Path(__file__).resolve().parents[1] / "examples" / "networks"


def test_static_catalog_matches_runtime():
    from hndl.operators._activations import ACTIVATION_NAMES
    from hndl.operators.feed_forward import ACTIVATIONS

    assert ACTIVATION_NAMES == tuple(ACTIVATIONS)
    registry = InspectionRegistry()
    runtime = Registry.builtins()
    assert set(registry.aliases) == set(runtime.aliases)
    for alias in runtime.aliases:
        schema, op = registry.get(alias), runtime.get(alias)
        for attr in ("key", "args", "inputs", "outputs", "init_symbols", "init_shapes", "examples"):
            assert getattr(schema, attr) == getattr(op, attr), (alias, attr)
    for name in names():
        source, _, kwargs = load(name)
        plan = _load_plan(RECIPES / f"{name}.hndl")
        eager = resolve(source, **kwargs)
        assert plan.semantic_digest == eager.semantic_digest
        assert plan.profile() == eager.profile()


def test_static_profile_constructs_real_parameters_for_ties():
    registry = InspectionRegistry()
    plan = resolve('linear(8, name="a"); linear(8, tie={"weight":"a.weight"})', input_shape=("B", 8),
                   output_shape=("B", 8), registry=registry)
    assert plan.profile().total_parameters == 80


def test_cold_cli_skips_torch_and_device_has_a_diagnostic(capsys):
    subprocess.run([sys.executable, "-c",
                    "from hndl.__main__ import main; import sys;"
                    "assert main(['check', sys.argv[1]]) == 0; assert 'torch' not in sys.modules", str(RECIPES)],
                   check=True, stdout=subprocess.DEVNULL)
    args = ["inspect", str(RECIPES / "mlp.hndl"), "--profile", "--device"]
    assert main([*args, "cpu"]) == 0
    assert "Deprecated" in capsys.readouterr().err
    assert main([*args, "nonsense"]) != 0
    assert "unknown device" in capsys.readouterr().err
