"""Shared loader for the authored networks in examples/networks."""

import json
from pathlib import Path

from typing import Any

import torch

from hndl.docs import contract_value
from hndl.torch import DTYPES

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


def random_inputs(
    plan: Any,
    device: str | torch.device = "cpu",
    batch_size: int = 2,
    generator: torch.Generator | None = None,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate random input tensors matching ``plan.inputs`` for testing and benchmarking.

    Args:
        plan: ResolvedPlan instance describing inputs and their shapes/dtypes.
        device: PyTorch device (str or torch.device).
        batch_size: Batch size to replace the leading batch dimension (default: 2).
        generator: Optional torch.Generator for reproducible random generation.
        requires_grad: Whether generated float tensors should require gradients.

    Returns:
        Dict[str, torch.Tensor] mapping input port names to randomly initialized tensors.
    """
    if isinstance(device, torch.Generator):
        generator = device
        device = "cpu"

    dtype_map = {
        **DTYPES,
        "float64": torch.float64,
    }

    gen_kw = {"generator": generator} if generator is not None else {}
    inputs: dict[str, torch.Tensor] = {}
    for name, entry in plan.inputs.items():
        shape = (batch_size, *entry["shape"][1:])
        entry_dtype = entry.get("dtype")
        plan_dtype = getattr(plan, "dtype", "float32")
        dtype_str = entry_dtype or plan_dtype
        torch_dtype = dtype_map.get(dtype_str)
        if torch_dtype is None:
            torch_dtype = getattr(torch, dtype_str, torch.float32)

        if torch_dtype in (torch.int64, torch.int32):
            inputs[name] = torch.randint(0, 8, shape, device=device, dtype=torch_dtype, **gen_kw)
        elif torch_dtype == torch.bool:
            inputs[name] = torch.randint(0, 2, shape, device=device, dtype=torch.bool, **gen_kw)
        else:
            inputs[name] = torch.randn(
                shape, device=device, dtype=torch_dtype, requires_grad=requires_grad, **gen_kw
            )
    return inputs

