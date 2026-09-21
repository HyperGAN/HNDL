"""Shared loader for the authored networks in examples/networks."""

import json
from pathlib import Path

import torch

from hndl.docs import contract_value

NETWORKS = Path(__file__).resolve().parents[1] / "examples" / "networks"
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def load(name):
    meta = json.loads((NETWORKS / f"{name}.json").read_text(encoding="utf-8"))
    source = (NETWORKS / f"{name}.hndl").read_text(encoding="utf-8")
    kwargs = {"input_shape": contract_value(meta["input_shape"]),
              "output_shape": contract_value(meta["output_shape"])}
    for key in ("input_dtype", "dtype"):
        if key in meta:
            kwargs[key] = meta[key]
    return source, meta, kwargs


def names():
    return sorted(path.stem for path in NETWORKS.glob("*.hndl"))
