"""Forward-pass latency parity: hndl's built graphs vs. hand-written PyTorch.

Correctness is covered elsewhere (``tests/operators`` compares each operator
against a reference implementation). This module asks the other question the
declarative front end has to answer: does running a resolved hndl network cost
about the same as running the ``nn.Module`` a developer would have typed by
hand?

Only wall-clock time is compared, so the two sides deliberately do *not* share
weights — each is randomly initialized and the outputs differ. Three
architecture families stand in for the shapes hndl is used for: an MLP, a
convolutional discriminator trunk, and a transformer block.

The comparison is not free of overhead on purpose: ``GraphModule._execute``
validates every node's shape, dtype and device on the first call for each
input signature (see ``_run_checked`` in ``src/hndl/torch.py``), which a
hand-written module never does, and on every later call compares that
signature, and a record of every module's ``_parameters``/``_buffers``/
``_modules`` (which names, which are ``None``, which submodules), before
running the modules from a slot-indexed program. The timing loop repeats one
signature, so it measures the second, cheap path. :data:`TOLERANCE` is the
multiple of hand-written time that overhead is allowed to cost. These
assertions are meant to fail loudly rather than skip if that overhead ever
grows unreasonable.

That overhead is a roughly **constant** cost per forward call — about 2 us per
call plus well under 1 us per node on this machine, so ~7-10 us for the
five-node MLP, independent of batch size — which means the ratio a case
reports depends on how much arithmetic the batch gives it to amortize against.
The MLP is timed at batch 256, where the fixed cost was once large; at batch
32 the same network now measures about 1.15-1.25x. (Before validation moved to
once per signature, every call checked every port and replayed the
registered-state walk: ~5 us per node after the resolved-shape and
baked-program caches, ~13 us before them, and batch 32 measured ~1.7x and
~2.5x respectively.) The registered-state record costs about 0.7 us per call
for the MLP and 1.3 us for the transformer block.

Each side is timed in :data:`ROUNDS` interleaved rounds and represented by its
fastest round (see :func:`_compare`). Measured ratios, 12 runs of
``python -m pytest -m benchmark tests/benchmark -k parity`` on a 24-core CPU
machine shared with other work (load average 3-7), torch 2.14, Python 3.14,
2026-09-26:

===========================  =============
case                         ratio range
===========================  =============
mlp (batch 256)              1.02x - 1.10x
conv_stack (batch 16)        0.60x - 1.58x
transformer_block (batch 8)  1.06x - 1.22x
===========================  =============

The MLP and the transformer block are where per-call overhead shows, and
:data:`TOLERANCE` holds them to 1.5x. That is not much headroom on a busy
machine. With 20 extra busy-looping processes on the same machine (load
average 6-17), five runs measured the MLP at 0.95x-1.06x and the transformer
block at 1.16x-1.38x, and a trial with one intra-op thread put the transformer
block at 1.49x. When each side was timed once, one after the other, as before
the rounds were interleaved, the MLP reached 1.51x at load 3-25. A failure on
a loaded machine is worth re-running before it is believed.

The convolution case is almost all arithmetic: hndl's ~10 us per call is about
1% of it or less, so it cannot show overhead at any tolerance. Its ratio swings
from one process to the next on a shared machine (above, and up to 2.12x with
sequential timing), even single-threaded, and it keeps the looser
:data:`ARITHMETIC_TOLERANCE` as a check against gross regressions only.

Run with ``-s`` to see each case's two timings and their ratio.
"""

import pytest
import torch
from torch import nn

import hndl.torch

from ._timing import format_measurement, measure

pytestmark = pytest.mark.benchmark

#: hndl's best median forward time may be at most this multiple of hand-written
#: PyTorch's (see the module docstring for measured ratios).
TOLERANCE = 1.5

#: The same bound for a case whose time is almost all arithmetic, which cannot
#: show per-call overhead and whose ratio is dominated by machine noise.
ARITHMETIC_TOLERANCE = 2.0

DEVICE = "cpu"


#: How many interleaved rounds each side is timed in; see :func:`_compare`.
ROUNDS = 10


def _compare(name, hndl_module, handwritten, inputs, min_run_time, tolerance=TOLERANCE):
    """Time both modules on ``inputs``, print the pair, and assert the ratio.

    The two sides are timed in :data:`ROUNDS` interleaved rounds, alternating
    which goes first, each round spending ``min_run_time / ROUNDS`` per side,
    and each side is represented by its fastest round's median. Load from
    other work only ever slows a round down, so the fastest rounds are the
    ones it touched least, and interleaving keeps a burst of load from landing
    on one side's whole measurement.
    """
    hndl_module.eval()
    handwritten.eval()
    assert hndl_module is not handwritten
    sides = {"hndl": hndl_module, "handwritten": handwritten}
    rounds = {side: [] for side in sides}
    with torch.no_grad():
        for number in range(ROUNDS):
            for side in (sides if number % 2 == 0 else reversed(sides)):
                module = sides[side]
                rounds[side].append(measure(lambda module=module: module(*inputs),
                                            min_run_time=min_run_time / ROUNDS))
    hndl_measurement, handwritten_measurement = (min(rounds[side], key=lambda m: m.median) for side in sides)
    ratio = hndl_measurement.median / handwritten_measurement.median
    print()
    print(format_measurement(f"{name} hndl", hndl_measurement))
    print(format_measurement(f"{name} handwritten", handwritten_measurement))
    print(f"{name} ratio: {ratio:.2f}x (tolerance {tolerance:.2f}x)")
    assert ratio <= tolerance, (
        f"{name}: hndl is {ratio:.2f}x hand-written PyTorch, over the {tolerance:.2f}x tolerance "
        f"({format_measurement('hndl', hndl_measurement)}; "
        f"{format_measurement('handwritten', handwritten_measurement)})"
    )


MLP_SOURCE = """
linear(128)
relu()
linear(128)
relu()
linear(10)
"""


def test_mlp_matches_handwritten_sequential(benchmark_min_time):
    """Three linear layers with ReLUs, the plainest thing hndl can build."""
    built = hndl.torch.network(MLP_SOURCE, input_shape=("B", 64), output_shape=("B", 10), device=DEVICE)
    handwritten = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(DEVICE)
    # Batch 256: large enough that the fixed per-call validation cost is
    # amortized rather than being the whole measurement (see the module docstring).
    x = torch.randn(256, 64, device=DEVICE)
    _compare("mlp", built, handwritten, (x,), benchmark_min_time)


CONV_SOURCE = """
conv(32, kernel_size=3, padding=1)
relu()
conv(64, kernel_size=3, padding=1, stride=2)
relu()
conv(64, kernel_size=3, padding=1, stride=2)
relu()
global_avg_pool()
linear(10)
"""


class HandwrittenConvStack(nn.Module):
    """The convolutional trunk above, typed out directly.

    ``global_avg_pool`` reduces ``[B, C, H, W]`` to ``[B, C]`` — it flattens as
    it pools — so the mean over the spatial axes feeds ``nn.Linear`` with no
    separate flatten step.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def test_conv_stack_matches_handwritten_module(benchmark_min_time):
    """A GAN-discriminator-shaped trunk: strided convolutions into a pooled head."""
    built = hndl.torch.network(CONV_SOURCE, input_shape=("B", 3, 32, 32),
                               output_shape=("B", 10), device=DEVICE)
    handwritten = HandwrittenConvStack().to(DEVICE)
    x = torch.randn(16, 3, 32, 32, device=DEVICE)
    _compare("conv_stack", built, handwritten, (x,), benchmark_min_time, tolerance=ARITHMETIC_TOLERANCE)


TRANSFORMER_SOURCE = "transformer_block(4)"

TRANSFORMER_WIDTH = 32
TRANSFORMER_HEADS = 4
#: ``transformer_block``'s defaults, confirmed against the operator's arg table.
TRANSFORMER_MLP_RATIO = 4
TRANSFORMER_EPS = 1e-5


class HandwrittenTransformerBlock(nn.Module):
    """A pre-norm encoder block written the obvious PyTorch way.

    ``x = x + attn(norm1(x))`` then ``out = x + ffn(norm2(x))``, matching the
    structure ``transformer_block``'s ``_reference`` spells out. The attention
    sub-layer is ``nn.MultiheadAttention``, which ``tests/operators/test_attention.py``
    already establishes is numerically equivalent to hndl's ``attention`` when
    weights are copied; nothing is copied here, only the work is the same.
    """

    def __init__(self, width=TRANSFORMER_WIDTH, heads=TRANSFORMER_HEADS,
                 mlp_ratio=TRANSFORMER_MLP_RATIO, eps=TRANSFORMER_EPS):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=eps)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(width, mlp_ratio * width),
            nn.GELU(),
            nn.Linear(mlp_ratio * width, width),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


def test_transformer_block_matches_handwritten_block(benchmark_min_time):
    """One pre-norm block: multi-head attention and a 4x feed-forward, both residual."""
    shape = ("B", 16, TRANSFORMER_WIDTH)
    built = hndl.torch.network(TRANSFORMER_SOURCE, input_shape=shape, output_shape=shape, device=DEVICE)
    handwritten = HandwrittenTransformerBlock().to(DEVICE)
    x = torch.randn(8, 16, TRANSFORMER_WIDTH, device=DEVICE)
    _compare("transformer_block", built, handwritten, (x,), benchmark_min_time)
