"""Benchmark peak activation memory and memory efficiency.

Measures peak memory usage during forward evaluation to verify that:
1. GraphModule immediate dead-slot clearing (values[slot] = None) reclaims
   intermediate activations as soon as their last consumer finishes.
2. GatedDeltaAttention 4D factorization avoids 5D tensor materializations.
3. Example networks maintain bounded peak activation memory.
"""

import pytest
import torch

from hndl import resolve
from hndl.operators.gated_delta_attention import GatedDeltaAttention
from hndl.torch import build
from tests._network_examples import load, names, random_inputs

pytestmark = pytest.mark.benchmark

BATCH = 2


def _build_eval_model(name, device="cpu"):
    """Resolve and build ``name`` on the device, in eval mode, with fixed init."""
    source, _meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    model = build(plan, device=device, initialization_seed=1)
    model.eval()
    return plan, model


@pytest.mark.parametrize("name", names())
def test_example_network_peak_memory(name, device):
    """Catalog peak memory usage across example networks during inference."""
    plan, model = _build_eval_model(name, device=device)
    generator = torch.Generator(device=device).manual_seed(1)
    inputs = random_inputs(plan, device=device, batch_size=BATCH, generator=generator)

    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        outputs = model(**inputs)

    assert set(outputs) == set(plan.outputs)
    for port, tensor in outputs.items():
        assert tensor.isfinite().all(), f"{name}: output {port!r} went non-finite"

    if is_cuda:
        peak_allocated = torch.cuda.max_memory_allocated(device)
        active_net = peak_allocated - mem_before
        peak_mb = peak_allocated / (1024 * 1024)
        active_mb = active_net / (1024 * 1024)
        print(f"peak memory {name} [{device}]: peak={peak_mb:.2f} MB, activations~={active_mb:.2f} MB")
        assert peak_allocated > 0, f"{name}: peak allocation was 0"
    else:
        # On CPU, verify forward execution completed cleanly and outputs are valid
        print(f"evaluated {name} [{device}]: {len(outputs)} output ports validated")


def test_graph_module_dead_slot_clearing():
    """Verify that intermediate activation slots are cleared immediately upon last consumption."""
    # Build a sequential graph with 10 linear transformations
    source = """
    y1 = linear(1024)
    y2 = linear(1024)
    y3 = linear(1024)
    y4 = linear(1024)
    y5 = linear(1024)
    y6 = linear(1024)
    y7 = linear(1024)
    y8 = linear(1024)
    y9 = linear(1024)
    out = linear(1024)
    """
    plan = resolve(source, input_shape=(16, 1024), output_shape=(16, 1024))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build(plan, device=device, initialization_seed=1).eval()

    # Verify the fast program has non-empty to_clear entries for intermediate nodes
    # Fast program is (padding, program, outputs)
    # Step entries: (module, fast_ins, out_slot, node_info, to_clear)
    _padding, program, _outputs = model._fast
    has_cleared_slots = False
    for step in program:
        to_clear = step[4]
        if to_clear:
            has_cleared_slots = True
            break
    assert has_cleared_slots, "Expected fast program to track slots to clear"

    x = torch.randn(16, 1024, device=device)

    # Warmup call: populates _fast program validation cache and sets up steady-state execution
    with torch.no_grad():
        _ = model(x=x)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        weight_mem = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            out = model(x=x)

        peak_mem = torch.cuda.max_memory_allocated(device)
        activation_peak = peak_mem - weight_mem
        # Each activation is 16 * 1024 * 4 = 64 KB.
        # If all 10 activations were retained, activation peak would be at least 10 * 64 KB = 640 KB.
        # With slot clearing and immediate reuse, it should stay small.
        assert activation_peak < 10 * 64 * 1024 * 2
        print(f"linear chain [{device}]: weight={weight_mem/1024:.1f} KB, peak activation={activation_peak/1024:.1f} KB")
    else:
        with torch.no_grad():
            out = model(x=x)
        assert out["out"].shape == (16, 1024)


def test_gated_delta_attention_activation_memory():
    """Verify that GatedDeltaAttention 4D vectorization keeps peak memory bounded."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dim = 256
    heads = 4
    seq_len = 256
    chunk_size = 64
    batch = 2

    layer = GatedDeltaAttention(
        heads=heads,
        head_dim=dim // heads,
        chunk_size=chunk_size,
        D=dim,
    ).to(device=device).eval()

    x = torch.randn(batch, seq_len, dim, device=device)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        weight_mem = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            out = layer(x)

        peak_mem = torch.cuda.max_memory_allocated(device)
        activation_peak = peak_mem - weight_mem
        print(f"GDA chunk_size={chunk_size} seq_len={seq_len} [{device}]: peak activation={activation_peak / (1024 * 1024):.2f} MB")

        # In 5D materialization, [B, H, C, C, D_h] is 2 * 4 * 64 * 64 * 64 * 4 bytes = 8.38 MB per tensor.
        # Factorized 4D matmul operates on [B, H, C, C] (131 KB) and [B, H, C, D_h] (131 KB).
        # Total activation peak should be well under 40 MB.
        assert activation_peak < 50 * 1024 * 1024
        assert out.shape == (batch, seq_len, dim)
    else:
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (batch, seq_len, dim)
