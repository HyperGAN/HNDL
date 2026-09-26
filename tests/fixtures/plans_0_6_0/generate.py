"""Regenerate the saved-plan fixtures with the released 0.6.0 code, never HEAD.

These plans were written by HNDL 0.6.0 (commit f7712af, the PyPI release) and
guard that later releases still load them and resolve the same sources to the
same semantic digests. From the repository root:

    S=$(mktemp -d); git archive f7712af src | tar -x -C "$S"
    PYTHONPATH="$S/src" python tests/fixtures/plans_0_6_0/generate.py

Each ``<case>.json`` holds the resolve inputs next to the exact ``to_json()``
text 0.6.0 produced for them.
"""

import json
from pathlib import Path

import hndl

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2] / "examples" / "networks"

CASES = {
    "linear_mlp": ("linear(8)\nrelu()\nlinear()", ["B", 4], ["B", 2], {}),
    "linear_options": ("linear(16, bias=False)\nrelu()\nlinear(spectral_norm=True)",
                       ["B", 5, 12], ["B", 5, 6], {}),
    "attention": ("attention(4)\nattention(2, causal=True, rope=True, bias=False)",
                  ["B", 8, 32], ["B", 8, 32], {}),
    "feed_forward": ('feed_forward(64)\nfeed_forward(48, activation="silu", bias=False)',
                     ["B", 6, 16], ["B", 6, 16], {}),
    "gpt_tiny": ((EXAMPLES / "gpt_tiny.hndl").read_text(encoding="utf-8"),
                 ["B", 64], ["B", 64, 256], {"input_dtype": "int64"}),
}


def main():
    assert hndl.__version__ == "0.6.0", f"generate with HNDL 0.6.0, not {hndl.__version__}"
    for name, (source, input_shape, output_shape, extra) in CASES.items():
        plan = hndl.resolve(source, input_shape=tuple(input_shape), output_shape=tuple(output_shape), **extra)
        record = {"hndl_version": hndl.__version__, "source": source, "input_shape": input_shape,
                  "output_shape": output_shape, **extra, "plan": plan.to_json()}
        (HERE / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
