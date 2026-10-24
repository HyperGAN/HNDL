from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

from .operator import Operator
from .registry import Registry
from .types import Graph, Node, ResolvedPlan

class Symbol:
    ref: str
    def __init__(self, owner: Capture, ref: str) -> None: ...
    def __repr__(self) -> str: ...

class Capture:
    registry: Registry
    input_shape: tuple[int | str, ...]
    output_shape: tuple[int | str, ...]
    dtype: str
    nodes: list[Node]
    inputs: dict[str, Symbol]
    input: Symbol
    current: Symbol | None
    def __init__(
        self,
        *,
        input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
        output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
        dtype: str = ...,
        registry: Registry | None = ...,
        frontend: str = ...,
        limits: Mapping[str, int] | None = ...,
        input_dtype: str | Mapping[str, str] | None = ...,
    ) -> None: ...
    def __enter__(self) -> Capture: ...
    def __exit__(self, *exc: Any) -> None: ...
    def emit(
        self,
        op: Operator,
        positional: Sequence[Any],
        keyword: Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = ...,
    ) -> Symbol | tuple[Symbol, ...]: ...
    def finish(self, selected: Any = ...) -> Graph: ...

class OperatorNamespace:
    def __init__(self, registry: Registry | None = ...) -> None: ...
    def __repr__(self) -> str: ...
    def __dir__(self) -> list[str]: ...
    @staticmethod
    def scope(prefix: str): ...
    def adaln_zero(self, x: Symbol, cond: Symbol, *, eps: float = 1e-06, gate: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.

        Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.

Maps a conditioning vector ``cond[B, D_cond]`` through a zero-initialized
linear layer to produce per-token shift, scale, and optional dimension-wise gate
parameters:
``shift, scale, gate = Linear(SiLU(cond)).chunk(3, dim=-1)`` (when ``gate=True``)
``shift, scale = Linear(SiLU(cond)).chunk(2, dim=-1)`` (when ``gate=False``)
``normed = LayerNorm(x, affine=False)``
``out = (1 + scale) * normed + shift``
``gate_out = gate.expand_as(x)``.
        """
        ...
    @overload
    def adaptive_avg_pool(self, x: Symbol, output_size: int | tuple[int, int] | None = None, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Average-pool [B, C, H, W] images to a fixed output height and width.

        Averages each ``[B, C, H_in, W_in]`` feature map over a grid of
``output_size = (H_out, W_out)`` windows, producing
``[B, C, H_out, W_out]``. Window ``i`` along an axis of extent ``n`` with
``k`` outputs covers

```text
[floor(i * n / k), ceil((i + 1) * n / k))
```

so the windows tile the axis, overlap when ``k`` does not divide ``n``,
and repeat a single element when ``k > n`` (an upsampling by replication,
which torch permits). The operation has no parameters and behaves
identically in train and eval mode.

### Determinism and higher-order gradients

Every path produces the values of `torch.nn.functional.adaptive_avg_pool2d`
within floating-point tolerance, but which path runs decides whether the
backward pass is deterministic on CUDA:

| Case | Implementation | Deterministic on CUDA |
| --- | --- | --- |
| `H_in % H_out == 0` and `W_in % W_out == 0` | reshape to `[B, C, H_out, H_in//H_out, W_out, W_in//W_out]`, then `mean` over the two window axes | yes |
| ragged extents, `H_out * W_out <= 64` | the windows written out with slicing and `stack` | yes |
| ragged extents, `H_out * W_out > 64` | `F.adaptive_avg_pool2d` | **no** |

The first two paths are built from `reshape`, `mean`, slicing and `stack`,
none of which accumulate with atomics, so they do not raise under
`torch.use_deterministic_algorithms(True)`, they repeat bit-identical
gradients run to run, and they differentiate to arbitrary order. That
makes the common critic pooling — a power-of-two map pooled to 4x4 —
usable with a gradient penalty, which takes a second derivative through
the pool.

The third path falls back to torch's own kernel, whose CUDA backward
accumulates with atomic adds: it raises ``adaptive_avg_pool2d_backward_cuda
does not have a deterministic implementation`` under
`torch.use_deterministic_algorithms(True)`, and its gradients are only
reproducible run to run on the CPU. It is reached only by a ragged pool to
more than 64 windows; pool to a divisor of the input extents, or to a
smaller grid, to stay on a deterministic path.

Shape inference runs forward only for the spatial axes: ``H_out`` and
``W_out`` come from ``output_size``, but ``H_in`` and ``W_in`` are *not*
inferable backward, because every input extent maps to the requested
output. Give the input extents from the network input or the preceding
operation. The channel count ``C`` is preserved and flows both ways, and
``output_size`` itself can be inferred backward from a known output shape.

The average is accumulated in the input dtype; in ``float16`` a very large
window loses precision, so pool in two stages if that matters.
        """
        ...
    @overload
    def adaptive_avg_pool(self, output_size: int | tuple[int, int] | None = None, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Average-pool [B, C, H, W] images to a fixed output height and width.

        Averages each ``[B, C, H_in, W_in]`` feature map over a grid of
``output_size = (H_out, W_out)`` windows, producing
``[B, C, H_out, W_out]``. Window ``i`` along an axis of extent ``n`` with
``k`` outputs covers

```text
[floor(i * n / k), ceil((i + 1) * n / k))
```

so the windows tile the axis, overlap when ``k`` does not divide ``n``,
and repeat a single element when ``k > n`` (an upsampling by replication,
which torch permits). The operation has no parameters and behaves
identically in train and eval mode.

### Determinism and higher-order gradients

Every path produces the values of `torch.nn.functional.adaptive_avg_pool2d`
within floating-point tolerance, but which path runs decides whether the
backward pass is deterministic on CUDA:

| Case | Implementation | Deterministic on CUDA |
| --- | --- | --- |
| `H_in % H_out == 0` and `W_in % W_out == 0` | reshape to `[B, C, H_out, H_in//H_out, W_out, W_in//W_out]`, then `mean` over the two window axes | yes |
| ragged extents, `H_out * W_out <= 64` | the windows written out with slicing and `stack` | yes |
| ragged extents, `H_out * W_out > 64` | `F.adaptive_avg_pool2d` | **no** |

The first two paths are built from `reshape`, `mean`, slicing and `stack`,
none of which accumulate with atomics, so they do not raise under
`torch.use_deterministic_algorithms(True)`, they repeat bit-identical
gradients run to run, and they differentiate to arbitrary order. That
makes the common critic pooling — a power-of-two map pooled to 4x4 —
usable with a gradient penalty, which takes a second derivative through
the pool.

The third path falls back to torch's own kernel, whose CUDA backward
accumulates with atomic adds: it raises ``adaptive_avg_pool2d_backward_cuda
does not have a deterministic implementation`` under
`torch.use_deterministic_algorithms(True)`, and its gradients are only
reproducible run to run on the CPU. It is reached only by a ragged pool to
more than 64 windows; pool to a divisor of the input extents, or to a
smaller grid, to stay on a deterministic path.

Shape inference runs forward only for the spatial axes: ``H_out`` and
``W_out`` come from ``output_size``, but ``H_in`` and ``W_in`` are *not*
inferable backward, because every input extent maps to the requested
output. Give the input extents from the network input or the preceding
operation. The channel count ``C`` is preserved and flows both ways, and
``output_size`` itself can be inferred backward from a known output shape.

The average is accumulated in the input dtype; in ``float16`` a very large
window loses precision, so pool in two stages if that matters.
        """
        ...
    def adaptive_norm(self, x: Symbol, params: Symbol, *, eps: float = 1e-05, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Instance-normalize features, then apply a per-example learned scale and bias.

        Population instance normalization followed by a learned affine that the
``params`` tensor supplies per example: ``params = [delta_gamma, beta]``
with one value of each per channel.

```text
normalized = (x - mean(x, HW)) / sqrt(var(x, HW) + eps)      # rank 4
normalized = (x - mean(x, C)) / sqrt(var(x, C) + eps)        # rank 3
out = (1 + delta_gamma) * normalized + beta
```

A zero ``params`` vector yields the normalized features. There are no
running statistics and no parameters; train and eval behave identically.
        """
        ...
    def add(self, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Elementwise sum of two tensors with identical shapes.

        ``out = a + b`` with no broadcasting; shapes, dtype, and batch must
match exactly. Both tensor inputs must be supplied explicitly.
        """
        ...
    @overload
    def attention(self, x: Symbol, heads: int, *, causal: bool = False, dropout: float = 0.0, bias: bool = True, equalized: bool = False, qkv_bias: bool = True, out_bias: bool = True, rope: bool = False, relative_position_bias: bool = False, spatial_shape: int | tuple[int, int] = (0, 0), kv_heads: int = 0, head_dim: int = 0, qk_norm: str = "none", rope_theta: float = 10000.0, rope_scaling: str = "none", rope_scale: float = 1.0, window_size: int = 0, logit_softcap: float = 0.0, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multi-head self-attention over a [B, T, D] sequence.

        Multi-head self-attention on a ``[B, T, D]`` sequence: ``B`` examples of
``T`` positions with ``D`` features each. Queries, keys and values all come
from the same tensor.

```text
Q, K, V = x @ Wq.T + bq, x @ Wk.T + bk, x @ Wv.T + bv    # each [B, T, D]
Q, K, V -> [B, heads, T, head_dim]                       # head_dim = D / heads
A       = softmax(Q @ K.T / sqrt(head_dim) + bias + mask) # [B, heads, T, T]
out     = (A @ V) -> [B, T, D] -> out @ Wo.T + bo
```

``heads`` must divide ``D``; the resolver reports ``E_CONSTRAINT`` when it
does not. Submodules are ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj``,
each an ``nn.Linear(D, D)``, so their parameters are ``q_proj.weight``,
``q_proj.bias`` and so on. The attention itself is computed by
``torch.nn.functional.scaled_dot_product_attention``, which picks a fused
kernel when one is available.

With ``equalized=True`` all four projections initialize raw weights N(0,1)
and any biases at zero, applying ``weight / sqrt(D)`` at every forward.
Gain and learning-rate multiplier are one. Relative-position tables and
the attention-logit head scaling are unchanged. ``init=`` and checkpoints
target raw projection weights, not their runtime-scaled values.

**Biases.** ``bias`` is the one flag for all four projections. ``qkv_bias``
and ``out_bias`` narrow it per projection: ``q_proj``, ``k_proj`` and
``v_proj`` carry a bias when ``bias and qkv_bias``, and ``o_proj`` when
``bias and out_bias``. Both narrow flags default to ``True``, so ``bias=``
on its own behaves exactly as it always has, and
``attention(4, qkv_bias=False, out_bias=True)`` gives the unbiased
projections with a biased output that ViT-style blocks use. A projection
built without a bias has no ``bias`` parameter at all.

**Masking.** With ``causal=True`` position ``t`` attends only to positions
``<= t``: the strictly upper triangle of the score matrix is set to
``-inf`` before the softmax. With ``causal=False`` every position attends
to every other; there is no padding mask, so pad positions are attended to
like any other position.

**Dropout.** ``dropout`` is applied to the attention weights and only in
training mode (``module.train()``); in eval mode the layer is
deterministic. At ``dropout=0`` the forward pass draws no random numbers in
either mode.

**Relative position bias.** With ``relative_position_bias=True`` the
sequence is read as a row-major ``H x W`` grid given by
``spatial_shape=(H, W)``, so token ``t`` sits at ``row = t // W``,
``col = t % W`` and ``H*W`` must equal ``T`` — the resolver reports
``E_CONSTRAINT`` when it does not, and infers ``T`` from the grid when the
sequence length is still open. Each head then learns one bias per possible
query-minus-key offset:

```text
dy = row(q) - row(k) + (H - 1)            # 0 .. 2H-2
dx = col(q) - col(k) + (W - 1)            # 0 .. 2W-2
row_of_table = dy * (2W - 1) + dx         # 0 .. (2H-1)*(2W-1) - 1
bias[head, q, k] = table[row_of_table, head]
```

``relative_position_bias_table`` is the learned parameter, of shape
``[(2H-1)*(2W-1), heads]``, and ``relative_position_index`` is the
precomputed ``[T, T]`` gather, a non-persistent int64 buffer rebuilt from
``spatial_shape`` on every construction rather than stored in a checkpoint.
The table is constructed at ``0.0``, so the layer starts as ordinary
attention and learns the bias from there; construction overrides target it
by name, as ``attention(4, spatial_shape=(8, 8), relative_position_bias=True,
init={"relative_position_bias_table": 0.02})`` or
``trainable={"relative_position_bias_table": False}``.

The resulting ``[heads, T, T]`` bias is added to the scaled logits before
the softmax, as an additive ``attn_mask`` so the fused kernels still apply.
With ``causal=True`` as well, the causal ``-inf`` mask and the bias are
summed into one additive mask rather than handed to the kernel separately.
The operator carries no windowing or shifting of its own: partition a
feature map into windows with `chunk` and reassemble it with `concat`, and
give this node the ``(H, W)`` of one window.

**Rotary embeddings.** With ``rope=True`` each query and key head is
rotated by its absolute position before the scores are formed, which makes
the scores depend on relative distance. The rotate-half convention over the
full head dimension is used, with base 10000:

```text
theta_i   = 10000 ** (-2i / head_dim)     # i = 0 .. head_dim/2 - 1
angle     = t * theta_i                   # at position t
rotate_half([a, b]) = [-b, a]             # a, b the two halves of the head
q_rotated = q * cos(angle) + rotate_half(q) * sin(angle)
```

The tables are recomputed from ``T`` on every forward pass, so there are no
position buffers and no maximum sequence length. The angles are built in
float32 and cast to the compute dtype, because ``10000 ** (-2i/head_dim)``
underflows in float16. ``rope`` therefore needs an even ``head_dim``. At
``T == 1`` the only angle is zero, so the layer matches ``rope=False``.
``rope`` and ``relative_position_bias`` are independent and compose.

Everything else stays in the plan compute dtype.
        """
        ...
    @overload
    def attention(self, heads: int, *, causal: bool = False, dropout: float = 0.0, bias: bool = True, equalized: bool = False, qkv_bias: bool = True, out_bias: bool = True, rope: bool = False, relative_position_bias: bool = False, spatial_shape: int | tuple[int, int] = (0, 0), kv_heads: int = 0, head_dim: int = 0, qk_norm: str = "none", rope_theta: float = 10000.0, rope_scaling: str = "none", rope_scale: float = 1.0, window_size: int = 0, logit_softcap: float = 0.0, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multi-head self-attention over a [B, T, D] sequence.

        Multi-head self-attention on a ``[B, T, D]`` sequence: ``B`` examples of
``T`` positions with ``D`` features each. Queries, keys and values all come
from the same tensor.

```text
Q, K, V = x @ Wq.T + bq, x @ Wk.T + bk, x @ Wv.T + bv    # each [B, T, D]
Q, K, V -> [B, heads, T, head_dim]                       # head_dim = D / heads
A       = softmax(Q @ K.T / sqrt(head_dim) + bias + mask) # [B, heads, T, T]
out     = (A @ V) -> [B, T, D] -> out @ Wo.T + bo
```

``heads`` must divide ``D``; the resolver reports ``E_CONSTRAINT`` when it
does not. Submodules are ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj``,
each an ``nn.Linear(D, D)``, so their parameters are ``q_proj.weight``,
``q_proj.bias`` and so on. The attention itself is computed by
``torch.nn.functional.scaled_dot_product_attention``, which picks a fused
kernel when one is available.

With ``equalized=True`` all four projections initialize raw weights N(0,1)
and any biases at zero, applying ``weight / sqrt(D)`` at every forward.
Gain and learning-rate multiplier are one. Relative-position tables and
the attention-logit head scaling are unchanged. ``init=`` and checkpoints
target raw projection weights, not their runtime-scaled values.

**Biases.** ``bias`` is the one flag for all four projections. ``qkv_bias``
and ``out_bias`` narrow it per projection: ``q_proj``, ``k_proj`` and
``v_proj`` carry a bias when ``bias and qkv_bias``, and ``o_proj`` when
``bias and out_bias``. Both narrow flags default to ``True``, so ``bias=``
on its own behaves exactly as it always has, and
``attention(4, qkv_bias=False, out_bias=True)`` gives the unbiased
projections with a biased output that ViT-style blocks use. A projection
built without a bias has no ``bias`` parameter at all.

**Masking.** With ``causal=True`` position ``t`` attends only to positions
``<= t``: the strictly upper triangle of the score matrix is set to
``-inf`` before the softmax. With ``causal=False`` every position attends
to every other; there is no padding mask, so pad positions are attended to
like any other position.

**Dropout.** ``dropout`` is applied to the attention weights and only in
training mode (``module.train()``); in eval mode the layer is
deterministic. At ``dropout=0`` the forward pass draws no random numbers in
either mode.

**Relative position bias.** With ``relative_position_bias=True`` the
sequence is read as a row-major ``H x W`` grid given by
``spatial_shape=(H, W)``, so token ``t`` sits at ``row = t // W``,
``col = t % W`` and ``H*W`` must equal ``T`` — the resolver reports
``E_CONSTRAINT`` when it does not, and infers ``T`` from the grid when the
sequence length is still open. Each head then learns one bias per possible
query-minus-key offset:

```text
dy = row(q) - row(k) + (H - 1)            # 0 .. 2H-2
dx = col(q) - col(k) + (W - 1)            # 0 .. 2W-2
row_of_table = dy * (2W - 1) + dx         # 0 .. (2H-1)*(2W-1) - 1
bias[head, q, k] = table[row_of_table, head]
```

``relative_position_bias_table`` is the learned parameter, of shape
``[(2H-1)*(2W-1), heads]``, and ``relative_position_index`` is the
precomputed ``[T, T]`` gather, a non-persistent int64 buffer rebuilt from
``spatial_shape`` on every construction rather than stored in a checkpoint.
The table is constructed at ``0.0``, so the layer starts as ordinary
attention and learns the bias from there; construction overrides target it
by name, as ``attention(4, spatial_shape=(8, 8), relative_position_bias=True,
init={"relative_position_bias_table": 0.02})`` or
``trainable={"relative_position_bias_table": False}``.

The resulting ``[heads, T, T]`` bias is added to the scaled logits before
the softmax, as an additive ``attn_mask`` so the fused kernels still apply.
With ``causal=True`` as well, the causal ``-inf`` mask and the bias are
summed into one additive mask rather than handed to the kernel separately.
The operator carries no windowing or shifting of its own: partition a
feature map into windows with `chunk` and reassemble it with `concat`, and
give this node the ``(H, W)`` of one window.

**Rotary embeddings.** With ``rope=True`` each query and key head is
rotated by its absolute position before the scores are formed, which makes
the scores depend on relative distance. The rotate-half convention over the
full head dimension is used, with base 10000:

```text
theta_i   = 10000 ** (-2i / head_dim)     # i = 0 .. head_dim/2 - 1
angle     = t * theta_i                   # at position t
rotate_half([a, b]) = [-b, a]             # a, b the two halves of the head
q_rotated = q * cos(angle) + rotate_half(q) * sin(angle)
```

The tables are recomputed from ``T`` on every forward pass, so there are no
position buffers and no maximum sequence length. The angles are built in
float32 and cast to the compute dtype, because ``10000 ** (-2i/head_dim)``
underflows in float16. ``rope`` therefore needs an even ``head_dim``. At
``T == 1`` the only angle is zero, so the layer matches ``rope=False``.
``rope`` and ``relative_position_bias`` are independent and compose.

Everything else stays in the plan compute dtype.
        """
        ...
    @overload
    def avg_pool(self, x: Symbol, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] = (0, 0), padding: int | tuple[int, int] = (0, 0), count_include_pad: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Two-dimensional average pooling over [B, C, H, W] images.

        Averages each ``kernel_size`` window of the spatial axes of a
``[B, C, H, W]`` image, stepping by ``stride``. Batch and channels pass
through unchanged and

``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

with the same rule on width; ``ceil_mode`` is false, so a trailing
partial window is dropped. ``stride=0`` is the default and means "the
same as ``kernel_size``", which is PyTorch's own default of
non-overlapping windows; any other value is used as written. ``padding``
adds zeros and may not exceed ``kernel_size//2`` on an axis: with
``count_include_pad`` true, the default, those zeros count in the
denominator, and with it false each window divides by the number of real
input elements it covers. There are no parameters and no train/eval
difference; the gradient is spread equally over every counted position.
The sum accumulates in the input dtype, so a large window in float16
loses precision.
        """
        ...
    @overload
    def avg_pool(self, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] = (0, 0), padding: int | tuple[int, int] = (0, 0), count_include_pad: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Two-dimensional average pooling over [B, C, H, W] images.

        Averages each ``kernel_size`` window of the spatial axes of a
``[B, C, H, W]`` image, stepping by ``stride``. Batch and channels pass
through unchanged and

``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

with the same rule on width; ``ceil_mode`` is false, so a trailing
partial window is dropped. ``stride=0`` is the default and means "the
same as ``kernel_size``", which is PyTorch's own default of
non-overlapping windows; any other value is used as written. ``padding``
adds zeros and may not exceed ``kernel_size//2`` on an axis: with
``count_include_pad`` true, the default, those zeros count in the
denominator, and with it false each window divides by the number of real
input elements it covers. There are no parameters and no train/eval
difference; the gradient is spread equally over every counted position.
The sum accumulates in the input dtype, so a large window in float16
loses precision.
        """
        ...
    @overload
    def batch_norm(self, x: Symbol, *, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, num_features: int | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize each channel over the batch and spatial axes, tracking running statistics.

        Normalizes each channel of axis 1 across the batch and every remaining
axis:

```text
out = weight * (x - mean) / sqrt(var + eps) + bias
```

Supported ranks are 2 `[B, C]`, 3 `[B, C, L]` and 4 `[B, C, H, W]`; the
rank is fixed at build time from the resolved input shape, so the module
is exactly `nn.BatchNorm1d` for ranks 2 and 3 and `nn.BatchNorm2d` for
rank 4, including its state-dict layout.

In train mode the statistics come from the current batch, and the
`running_mean` / `running_var` buffers are updated once per forward as
`running = (1 - momentum) * running + momentum * batch` (the variance uses
the unbiased batch estimate, the normalization the biased one);
`num_batches_tracked` counts the updates. In eval mode the buffers are
used instead, so the layer becomes a fixed affine map. With
`track_running_stats=False` there are no buffers and batch statistics are
used in both modes.

Parameters are `weight` and `bias` when `affine` is true, one value per
channel. Statistics are accumulated in the plan's compute dtype.
        """
        ...
    @overload
    def batch_norm(self, *, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, num_features: int | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize each channel over the batch and spatial axes, tracking running statistics.

        Normalizes each channel of axis 1 across the batch and every remaining
axis:

```text
out = weight * (x - mean) / sqrt(var + eps) + bias
```

Supported ranks are 2 `[B, C]`, 3 `[B, C, L]` and 4 `[B, C, H, W]`; the
rank is fixed at build time from the resolved input shape, so the module
is exactly `nn.BatchNorm1d` for ranks 2 and 3 and `nn.BatchNorm2d` for
rank 4, including its state-dict layout.

In train mode the statistics come from the current batch, and the
`running_mean` / `running_var` buffers are updated once per forward as
`running = (1 - momentum) * running + momentum * batch` (the variance uses
the unbiased batch estimate, the normalization the biased one);
`num_batches_tracked` counts the updates. In eval mode the buffers are
used instead, so the layer becomes a fixed affine map. With
`track_running_stats=False` there are no buffers and batch statistics are
used in both modes.

Parameters are `weight` and `bias` when `affine` is true, one value per
channel. Statistics are accumulated in the plan's compute dtype.
        """
        ...
    def broadcast_add(self, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Elementwise sum of two tensors of equal rank, broadcasting size-1 axes.

        ``out = a + b`` with numpy-style broadcasting restricted to operands of
**equal rank**. Use this when one operand is a per-sample, per-channel
bias such as ``[B, C, H, W] + [B, C, 1, 1]``; use `add` when the shapes
match exactly, which is the stricter and cheaper contract.

Broadcasting is symmetric: for every non-batch axis the two extents must
either be equal or one of them must be 1, and the output takes the larger
extent. Either operand may hold the size-1 axis, so ``[B, 1, H, W] +
[B, C, 1, 1]`` is accepted and produces ``[B, C, H, W]``. The batch axis
is never broadcast — both operands carry the full batch — and rank is
never padded, so a ``[B, C, H, W]`` map and a ``[B, C]`` vector are a rank
mismatch (``E_CONSTRAINT``); `reshape` the vector to ``[B, C, 1, 1]``
first. Incompatible extents report ``E_CONSTRAINT`` naming the axis.

Shape inference runs in both directions wherever the answer is unique. A
known pair of inputs always fixes the output. Going backward, a known
output plus one known operand fixes the other operand on every axis where
the known operand has extent 1 (the other must supply the full extent) and
on every axis where the output extent is 1 (both operands must be 1). On
an axis where the known operand already matches a larger output extent the
other operand could be either that extent or 1, so nothing is inferred
there and the extent has to come from the operand's own producer.

### Determinism

The backward pass reduces each broadcast axis with a plain ``sum``
(torch's ``sum_to_size``), which is a deterministic tree reduction with no
atomic accumulation — unlike ``index_add``/``scatter_add``, which are the
usual source of run-to-run drift on CUDA. Both the forward and the
backward run cleanly under
``torch.use_deterministic_algorithms(True)``, and repeated backward passes
over identical inputs on CUDA give bit-identical gradients. The operator
has no parameters and behaves identically in train and eval mode; the sum
is computed in the plan compute dtype.
        """
        ...
    def broadcast_mul(self, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Elementwise product of equal-rank tensors, broadcasting size-1 axes.

        Compute ``a * b`` with equal-rank, non-batch broadcasting.

For example, ``[B,T,D] * [B,1,D]`` applies one channel gate per image.
Both operands must have the same batch and rank; reshape style vectors
explicitly. Gradients sum over broadcast axes, including for higher
derivatives. No parameters or buffers; train and eval are identical.
Use ``mul`` when broadcasting should be forbidden.
        """
        ...
    @overload
    def cast(self, x: Symbol, dtype: str = "compute", *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Explicit dtype conversion across graph edges.

        Cast a tensor to a target data type across graph edges.

Converts an input tensor along the edge to the specified target data type.
When ``dtype`` is set to ``"compute"``, the tensor is cast to the active
plan compute dtype (e.g. ``float32``, ``float16``, or ``bfloat16``).
        """
        ...
    @overload
    def cast(self, dtype: str = "compute", *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Explicit dtype conversion across graph edges.

        Cast a tensor to a target data type across graph edges.

Converts an input tensor along the edge to the specified target data type.
When ``dtype`` is set to ``"compute"``, the tensor is cast to the active
plan compute dtype (e.g. ``float32``, ``float16``, or ``bfloat16``).
        """
        ...
    @overload
    def chunk(self, x: Symbol, chunks: int, *, dim: int = 1, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Cut one axis into a fixed number of equal sections.

        Returns ``chunks`` equal sections of ``dim`` as views sharing autograd
with the input, in order, from ``torch.split`` with one exact size.

``chunk`` is the equal-split counterpart of ``split``, which cuts one
tensor into a first section and the remainder. The axis must divide
exactly: an extent of 9 into 2 sections fails with ``E_CONSTRAINT`` at
resolution, not at runtime, and no section may be empty.

``dim=0`` divides the batch. Batch is symbolic, so the division is done on
batch entries: a ``2*B`` input --- what ``concat(..., axis=0)`` of two
``B`` tensors produces --- chunks into two ``B`` outputs, and ``3*B`` into
three. A plain ``B`` input cannot be chunked, because nothing says the
runtime batch is even; that fails at resolution with ``E_CONSTRAINT``. The
module still checks divisibility in ``forward`` (``E_RUNTIME``) for a
submodule called directly, outside the graph that checked its contract::

    combined = concat(candidate, context, axis=0)   # [2*B, C, H, W]
    features = pretrained(combined, ...)            # one shared pass
    a, b = chunk(features, 2, dim=0)                # [B, ...] and [B, ...]

Like every multi-output call, ``chunk`` returns a tuple and clears the
current tensor, so the next operation must name its input. ``chunks=1`` is
allowed and returns a one-tuple containing the whole tensor, which also
clears current. Both first and second derivatives flow through every
section: the sections are views, not copies.
        """
        ...
    @overload
    def chunk(self, chunks: int, *, dim: int = 1, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Cut one axis into a fixed number of equal sections.

        Returns ``chunks`` equal sections of ``dim`` as views sharing autograd
with the input, in order, from ``torch.split`` with one exact size.

``chunk`` is the equal-split counterpart of ``split``, which cuts one
tensor into a first section and the remainder. The axis must divide
exactly: an extent of 9 into 2 sections fails with ``E_CONSTRAINT`` at
resolution, not at runtime, and no section may be empty.

``dim=0`` divides the batch. Batch is symbolic, so the division is done on
batch entries: a ``2*B`` input --- what ``concat(..., axis=0)`` of two
``B`` tensors produces --- chunks into two ``B`` outputs, and ``3*B`` into
three. A plain ``B`` input cannot be chunked, because nothing says the
runtime batch is even; that fails at resolution with ``E_CONSTRAINT``. The
module still checks divisibility in ``forward`` (``E_RUNTIME``) for a
submodule called directly, outside the graph that checked its contract::

    combined = concat(candidate, context, axis=0)   # [2*B, C, H, W]
    features = pretrained(combined, ...)            # one shared pass
    a, b = chunk(features, 2, dim=0)                # [B, ...] and [B, ...]

Like every multi-output call, ``chunk`` returns a tuple and clears the
current tensor, so the next operation must name its input. ``chunks=1`` is
allowed and returns a one-tuple containing the whole tensor, which also
clears current. Both first and second derivatives flow through every
section: the sections are views, not copies.
        """
        ...
    @overload
    def clamp(self, x: Symbol, min: float, max: float, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Saturate every element into the closed interval [min, max].

        Elementwise ``out = min(max(x, min), max)``, computed as
``torch.clamp(x, min=min, max=max)`` out of place.

The operation is shape preserving: every supported rank (``[B, F]``,
``[B, T, D]``, ``[B, C, H, W]``) passes through unchanged, and no axis has
a special meaning. Both bounds are required floats and ``min <= max`` is
checked when arguments are normalized (``E_ARGUMENT`` otherwise).

The gradient is one strictly inside the interval and zero outside it, so a
saturated element stops contributing to its input's gradient; elements
exactly on a bound keep a gradient of one, matching ``torch.clamp``.
Behavior is identical in train and eval mode and there are no parameters.
Computation stays in the input dtype, so ``float16`` and ``bfloat16``
activations are clamped without an upcast.
        """
        ...
    @overload
    def clamp(self, min: float, max: float, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Saturate every element into the closed interval [min, max].

        Elementwise ``out = min(max(x, min), max)``, computed as
``torch.clamp(x, min=min, max=max)`` out of place.

The operation is shape preserving: every supported rank (``[B, F]``,
``[B, T, D]``, ``[B, C, H, W]``) passes through unchanged, and no axis has
a special meaning. Both bounds are required floats and ``min <= max`` is
checked when arguments are normalized (``E_ARGUMENT`` otherwise).

The gradient is one strictly inside the interval and zero outside it, so a
saturated element stops contributing to its input's gradient; elements
exactly on a bound keep a gradient of one, matching ``torch.clamp``.
Behavior is identical in train and eval mode and there are no parameters.
Computation stays in the input dtype, so ``float16`` and ``bfloat16``
activations are clamped without an upcast.
        """
        ...
    @overload
    def cls_token(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Prepend one learned classification token to a sequence.

        Prepends one learned vector to the position axis of a sequence:

```text
out = cat([token.expand(B, 1, D), x], dim=1)
out[:, 0, :] = token        # the class position
out[:, 1:, :] = x           # the original sequence
```

``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features
and the result has ``T + 1`` positions, so every downstream operator that
counts positions — a position table, a pooling step, an output contract —
sees the extra slot. The relation is bidirectional: a known input length
fixes the output length and a known output length fixes the input length.

The single parameter is ``token`` of shape ``[1, 1, D]``, initialized from
a normal distribution with standard deviation 0.02 and broadcast across
the batch, so every example starts from the same learned summary vector.
It is a plain concatenation with no projection and no normalization;
train and eval behave identically.
        """
        ...
    @overload
    def cls_token(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Prepend one learned classification token to a sequence.

        Prepends one learned vector to the position axis of a sequence:

```text
out = cat([token.expand(B, 1, D), x], dim=1)
out[:, 0, :] = token        # the class position
out[:, 1:, :] = x           # the original sequence
```

``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features
and the result has ``T + 1`` positions, so every downstream operator that
counts positions — a position table, a pooling step, an output contract —
sees the extra slot. The relation is bidirectional: a known input length
fixes the output length and a known output length fixes the input length.

The single parameter is ``token`` of shape ``[1, 1, D]``, initialized from
a normal distribution with standard deviation 0.02 and broadcast across
the batch, so every example starts from the same learned summary vector.
It is a plain concatenation with no projection and no normalization;
train and eval behave identically.
        """
        ...
    def concat(self, *x: Symbol, input_count: int = ..., axis: int = 1, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Join two or more tensors along one axis.

        ``torch.cat(inputs, dim=axis)``. Every input is explicit; one missing
extent along ``axis`` can be inferred from the output contract.

``axis=0`` joins along batch: the inputs must agree on every other axis and
the result carries their batches added together, written ``2*B`` for two
equal batches. That is how one shared --- often frozen --- network runs over
two branches in a single forward pass; ``chunk(..., dim=0)`` takes the
result apart again.
        """
        ...
    @overload
    def constant(self, x: Symbol, shape: int | Sequence[int] = (), *, value: float = 0.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Emit a tensor of a fixed shape filled with one constant value.

        ``out = full([B, *shape], value)``: a source of fixed values with no
parameters and no buffers.

```text
out[b, ...] = value
```

Give the non-batch dimensions positionally, ``constant(3, 8, 8)``, or as
``shape=(3, 8, 8)``; omit them entirely and the plan reads them from the
output contract, the way ``reshape()`` does. ``value`` defaults to ``0.0``,
so ``constant(16)`` is a zero vector — a null conditioning input — and
``constant(16, value=1.0)`` is a vector of ones.

The incoming tensor ``x`` supplies only the runtime batch extent and the
device; its own shape, rank, and values are ignored, so the input port
accepts any dtype including an integer graph input such as token ids. The
result always carries the plan's compute dtype and it is created fresh on
every call, detached from the autograd graph: it requires no gradient and
no gradient reaches ``x``. The operation is deterministic and identical in
train and eval mode.

Because the constant tells the solver nothing about ``x``, it does not
propagate shapes backward: the operation before it must have its shape
fixed from the input side or by its own arguments.
        """
        ...
    @overload
    def constant(self, shape: int | Sequence[int] = (), *, value: float = 0.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Emit a tensor of a fixed shape filled with one constant value.

        ``out = full([B, *shape], value)``: a source of fixed values with no
parameters and no buffers.

```text
out[b, ...] = value
```

Give the non-batch dimensions positionally, ``constant(3, 8, 8)``, or as
``shape=(3, 8, 8)``; omit them entirely and the plan reads them from the
output contract, the way ``reshape()`` does. ``value`` defaults to ``0.0``,
so ``constant(16)`` is a zero vector — a null conditioning input — and
``constant(16, value=1.0)`` is a vector of ones.

The incoming tensor ``x`` supplies only the runtime batch extent and the
device; its own shape, rank, and values are ignored, so the input port
accepts any dtype including an integer graph input such as token ids. The
result always carries the plan's compute dtype and it is created fresh on
every call, detached from the autograd graph: it requires no gradient and
no gradient reaches ``x``. The operation is deterministic and identical in
train and eval mode.

Because the constant tells the solver nothing about ``x``, it does not
propagate shapes backward: the operation before it must have its shape
fixed from the input side or by its own arguments.
        """
        ...
    @overload
    def conv(self, x: Symbol, out_channels: int | None = None, *, in_channels: int | None = None, kernel_size: int | tuple[int, int] = ..., stride: int | tuple[int, int] = (1, 1), padding: int | tuple[int, int] = (0, 0), dilation: int | tuple[int, int] = (1, 1), groups: int = 1, bias: bool = True, spectral_norm: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Two-dimensional convolution over [B, C, H, W] images.

        Cross-correlation of the input with ``out_channels`` learned kernels of
shape ``[in_channels / groups, kH, kW]``. The input is ``[B, C_in, H_in,
W_in]`` and the output ``[B, C_out, H_out, W_out]``, with

```text
H_out = floor((H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
```

and the same formula on the width axis. Parameters are ``weight`` with
shape ``[out_channels, in_channels / groups, kH, kW]`` and, when
``bias=True``, ``bias`` with shape ``[out_channels]``. Behavior is
identical in training and evaluation unless ``spectral_norm`` is enabled.

Inverse inference from a known output extent may leave an interval of
valid input sizes; that ambiguity is reported rather than resolved
arbitrarily. Select ``policy="down2"`` for the standard downsampling
block — kernel 4, stride 2, padding 1, dilation 1, groups 1 — which also
*asserts* exact halving: input extents must be even, ``H_out = H_in / 2``
and ``W_out = W_in / 2``. That removes the usual off-by-one interval, so a
stack of ``down2`` convolutions resolves backward from the output contract
alone. An explicit argument contradicting the policy fails with
``E_POLICY_CONFLICT``; an odd input extent under the policy fails with
``E_CONSTRAINT``.

## Spectral normalization

With ``spectral_norm=True`` the weight is reparametrized as ``weight /
sigma(weight)``, where ``sigma`` is the largest singular value of the
weight viewed as an ``[out_channels, -1]`` matrix, estimated by one power
iteration per forward pass
(``torch.nn.utils.parametrizations.spectral_norm``). This is the
discriminator constraint from SN-GAN; pair it with ``policy="down2"`` for
the usual downsampling critic.

The parametrization renames the registered state: the learned tensor
becomes ``parametrizations.weight.original`` and ``weight`` turns into a
computed attribute, with persistent buffers
``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
holding the power-iteration vectors. ``init`` and ``trainable``
overrides must therefore target ``parametrizations.weight.original``
instead of ``weight``; ``bias`` is unaffected. The power iteration
refreshes the buffers in training mode only, so evaluation is a pure
function of the stored state — the one case where this operator's
behavior differs between training and evaluation.
        """
        ...
    @overload
    def conv(self, out_channels: int | None = None, *, in_channels: int | None = None, kernel_size: int | tuple[int, int] = ..., stride: int | tuple[int, int] = (1, 1), padding: int | tuple[int, int] = (0, 0), dilation: int | tuple[int, int] = (1, 1), groups: int = 1, bias: bool = True, spectral_norm: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Two-dimensional convolution over [B, C, H, W] images.

        Cross-correlation of the input with ``out_channels`` learned kernels of
shape ``[in_channels / groups, kH, kW]``. The input is ``[B, C_in, H_in,
W_in]`` and the output ``[B, C_out, H_out, W_out]``, with

```text
H_out = floor((H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
```

and the same formula on the width axis. Parameters are ``weight`` with
shape ``[out_channels, in_channels / groups, kH, kW]`` and, when
``bias=True``, ``bias`` with shape ``[out_channels]``. Behavior is
identical in training and evaluation unless ``spectral_norm`` is enabled.

Inverse inference from a known output extent may leave an interval of
valid input sizes; that ambiguity is reported rather than resolved
arbitrarily. Select ``policy="down2"`` for the standard downsampling
block — kernel 4, stride 2, padding 1, dilation 1, groups 1 — which also
*asserts* exact halving: input extents must be even, ``H_out = H_in / 2``
and ``W_out = W_in / 2``. That removes the usual off-by-one interval, so a
stack of ``down2`` convolutions resolves backward from the output contract
alone. An explicit argument contradicting the policy fails with
``E_POLICY_CONFLICT``; an odd input extent under the policy fails with
``E_CONSTRAINT``.

## Spectral normalization

With ``spectral_norm=True`` the weight is reparametrized as ``weight /
sigma(weight)``, where ``sigma`` is the largest singular value of the
weight viewed as an ``[out_channels, -1]`` matrix, estimated by one power
iteration per forward pass
(``torch.nn.utils.parametrizations.spectral_norm``). This is the
discriminator constraint from SN-GAN; pair it with ``policy="down2"`` for
the usual downsampling critic.

The parametrization renames the registered state: the learned tensor
becomes ``parametrizations.weight.original`` and ``weight`` turns into a
computed attribute, with persistent buffers
``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
holding the power-iteration vectors. ``init`` and ``trainable``
overrides must therefore target ``parametrizations.weight.original``
instead of ``weight``; ``bias`` is unaffected. The power iteration
refreshes the buffers in training mode only, so evaluation is a pure
function of the stored state — the one case where this operator's
behavior differs between training and evaluation.
        """
        ...
    @overload
    def conv1d(self, x: Symbol, out_channels: int | None = None, kernel_size: int = ..., *, in_channels: int | None = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, spectral_norm: bool = False, causal: bool = False, channels_last: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """One-dimensional convolution over [B, C, L] or [B, T, C] signals.

        Cross-correlation of the input with `out_channels` learned kernels of
shape `[in_channels / groups, kernel_size]`:

```text
out[b, co, l] = bias[co] + sum_{ci, k} weight[co, ci, k] * x[b, ci, l*stride - padding + k*dilation]
```

## Axis convention

The default input is `[B, C, L]`: channels at axis 1, positions at axis 2.
With `channels_last=True`, the input layout is `[B, T, C]`.
With `causal=True`, left-padding of `dilation * (kernel_size - 1)` is applied
so each position attends only to past and present positions.
When `causal=True` and `stride=1`, `L_out == L_in` forms a bijective relation.
        """
        ...
    @overload
    def conv1d(self, out_channels: int | None = None, kernel_size: int = ..., *, in_channels: int | None = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, spectral_norm: bool = False, causal: bool = False, channels_last: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """One-dimensional convolution over [B, C, L] or [B, T, C] signals.

        Cross-correlation of the input with `out_channels` learned kernels of
shape `[in_channels / groups, kernel_size]`:

```text
out[b, co, l] = bias[co] + sum_{ci, k} weight[co, ci, k] * x[b, ci, l*stride - padding + k*dilation]
```

## Axis convention

The default input is `[B, C, L]`: channels at axis 1, positions at axis 2.
With `channels_last=True`, the input layout is `[B, T, C]`.
With `causal=True`, left-padding of `dilation * (kernel_size - 1)` is applied
so each position attends only to past and present positions.
When `causal=True` and `stride=1`, `L_out == L_in` forms a bijective relation.
        """
        ...
    @overload
    def coordinate_grid(self, x: Symbol, height: int, width: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Emit a normalized 2D coordinate grid in (x, y) order.

        Emit ``[B,H,W,2]`` coordinates spanning [-1,1] including both endpoints.

The final axis is (x, y); x varies along columns, y along rows. Input
values are ignored: only batch size is used. The fixed ``grid`` buffer
follows model device/dtype and is saved in the state dict. No random
draws or trainable parameters. There is no gradient to the input.
This is the endpoint convention of ``linspace(-1,1,extent)``, regardless
of the ``align_corners`` choice in a downstream sampler.
        """
        ...
    @overload
    def coordinate_grid(self, height: int, width: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Emit a normalized 2D coordinate grid in (x, y) order.

        Emit ``[B,H,W,2]`` coordinates spanning [-1,1] including both endpoints.

The final axis is (x, y); x varies along columns, y along rows. Input
values are ignored: only batch size is used. The fixed ``grid`` buffer
follows model device/dtype and is saved in the state dict. No random
draws or trainable parameters. There is no gradient to the input.
This is the endpoint convention of ``linspace(-1,1,extent)``, regardless
of the ``align_corners`` choice in a downstream sampler.
        """
        ...
    def cross_attention(self, x: Symbol, context: Symbol, heads: int, *, bias: bool = True, dropout: float = 0.0, qkv_bias: bool = True, out_bias: bool = True, kv_heads: int = 0, head_dim: int = 0, qk_norm: str = "none", logit_softcap: float = 0.0, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multi-head attention with queries from one sequence and keys and values from another.

        Attends from a query sequence `x` of shape `[B, T, D]` to a context
sequence of shape `[B, S, D_ctx]`. Positions are axis 1 and features are
the last axis; the two sequences may differ in both length and width, but
share the batch.

```text
q = x @ Wq.T + bq                      # [B, T, D]
k = context @ Wk.T + bk                # [B, S, D]
v = context @ Wv.T + bv                # [B, S, D]
per head: a = softmax(q_h @ k_h.T / sqrt(D / heads)) @ v_h
out = concat(a_0 .. a_{heads-1}) @ Wo.T + bo
```

`heads` must divide `D`; each head works on `D / heads` features. There is
no mask: every query position attends to every context position, which is
the usual encoder-decoder cross-attention.

Submodules are `q_proj` (`D -> D`), `k_proj` and `v_proj` (`D_ctx -> D`),
and `o_proj` (`D -> D`), each an `nn.Linear` carrying `weight` and, when
`bias` is true, `bias`.

In train mode `dropout` is applied to the attention weights; in eval mode
the layer is deterministic. The default `dropout=0` behaves identically in
both modes. Activations stay in the plan compute dtype; PyTorch's attention
kernel may accumulate the softmax in float32 for stability in float16 and
bfloat16, so reduced-precision results can be slightly more accurate than
the same arithmetic done by hand.
        """
        ...
    @overload
    def deconv(self, x: Symbol, out_channels: int | None = None, *, in_channels: int | None = None, kernel_size: int | tuple[int, int] = ..., stride: int | tuple[int, int] = (1, 1), padding: int | tuple[int, int] = (0, 0), dilation: int | tuple[int, int] = (1, 1), output_padding: int | tuple[int, int] = (0, 0), groups: int = 1, bias: bool = True, spectral_norm: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Transposed two-dimensional convolution, typically for upsampling.

        The gradient of ``conv`` with respect to its input, used as a learned
upsampling layer. Select ``policy="up2"`` to guarantee that height and
width double; explicit arguments that contradict the policy fail.
Parameters are ``weight`` with shape ``[in_channels, out_channels /
groups, kH, kW]`` and, when ``bias=True``, ``bias`` with shape
``[out_channels]``.

## Spectral normalization

With ``spectral_norm=True`` the weight is reparametrized as ``weight /
sigma(weight)``, where ``sigma`` is the largest singular value estimated
by one power iteration per forward pass
(``torch.nn.utils.parametrizations.spectral_norm``). A transposed
convolution stores its weight input-channel first, so the matrix view is
taken over axis 1: the output-channel axis moves to the front and the
rest flattens, exactly as PyTorch does for ``ConvTranspose2d``.

The parametrization renames the registered state: the learned tensor
becomes ``parametrizations.weight.original`` and ``weight`` turns into a
computed attribute, with persistent buffers
``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
holding the power-iteration vectors. ``init`` and ``trainable``
overrides must therefore target ``parametrizations.weight.original``
instead of ``weight``; ``bias`` is unaffected. The power iteration
refreshes the buffers in training mode only, so evaluation is a pure
function of the stored state.
        """
        ...
    @overload
    def deconv(self, out_channels: int | None = None, *, in_channels: int | None = None, kernel_size: int | tuple[int, int] = ..., stride: int | tuple[int, int] = (1, 1), padding: int | tuple[int, int] = (0, 0), dilation: int | tuple[int, int] = (1, 1), output_padding: int | tuple[int, int] = (0, 0), groups: int = 1, bias: bool = True, spectral_norm: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Transposed two-dimensional convolution, typically for upsampling.

        The gradient of ``conv`` with respect to its input, used as a learned
upsampling layer. Select ``policy="up2"`` to guarantee that height and
width double; explicit arguments that contradict the policy fail.
Parameters are ``weight`` with shape ``[in_channels, out_channels /
groups, kH, kW]`` and, when ``bias=True``, ``bias`` with shape
``[out_channels]``.

## Spectral normalization

With ``spectral_norm=True`` the weight is reparametrized as ``weight /
sigma(weight)``, where ``sigma`` is the largest singular value estimated
by one power iteration per forward pass
(``torch.nn.utils.parametrizations.spectral_norm``). A transposed
convolution stores its weight input-channel first, so the matrix view is
taken over axis 1: the output-channel axis moves to the front and the
rest flattens, exactly as PyTorch does for ``ConvTranspose2d``.

The parametrization renames the registered state: the learned tensor
becomes ``parametrizations.weight.original`` and ``weight`` turns into a
computed attribute, with persistent buffers
``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
holding the power-iteration vectors. ``init`` and ``trainable``
overrides must therefore target ``parametrizations.weight.original``
instead of ``weight``; ``bias`` is unaffected. The power iteration
refreshes the buffers in training mode only, so evaluation is a pure
function of the stored state.
        """
        ...
    @overload
    def deconv1d(self, x: Symbol, out_channels: int | None = None, kernel_size: int = ..., *, in_channels: int | None = None, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, groups: int = 1, bias: bool = True, spectral_norm: bool = False, causal: bool = False, channels_last: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Transposed one-dimensional convolution for waveform synthesis and upsampling.

        The gradient of ``conv1d`` with respect to its input, used as a learned
upsampling layer along the length axis of ``[B, C, L]`` signals.

Select ``policy="up2"`` to guarantee that length doubles exactly.
Parameters are ``weight`` of shape ``[in_channels, out_channels / groups, kernel_size]``
and, when ``bias=True``, ``bias`` of shape ``[out_channels]``.
        """
        ...
    @overload
    def deconv1d(self, out_channels: int | None = None, kernel_size: int = ..., *, in_channels: int | None = None, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, groups: int = 1, bias: bool = True, spectral_norm: bool = False, causal: bool = False, channels_last: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Transposed one-dimensional convolution for waveform synthesis and upsampling.

        The gradient of ``conv1d`` with respect to its input, used as a learned
upsampling layer along the length axis of ``[B, C, L]`` signals.

Select ``policy="up2"`` to guarantee that length doubles exactly.
Parameters are ``weight`` of shape ``[in_channels, out_channels / groups, kernel_size]``
and, when ``bias=True``, ``bias`` of shape ``[out_channels]``.
        """
        ...
    @overload
    def drop_path(self, x: Symbol, drop_prob: float = 0.0, *, scale_by_keep: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Stochastic depth (DropPath) residual regularization.

        Stochastic depth layer (DropPath) per sample across batches.
        """
        ...
    @overload
    def drop_path(self, drop_prob: float = 0.0, *, scale_by_keep: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Stochastic depth (DropPath) residual regularization.

        Stochastic depth layer (DropPath) per sample across batches.
        """
        ...
    @overload
    def dropout(self, x: Symbol, p: float = 0.1, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Randomly zero elements during training and rescale the rest.

        Elementwise dropout, computed out of place so shared branches are never
mutated.

In **training mode** each element of the input is independently zeroed with
probability ``p``, and the surviving elements are scaled by ``1 / (1 - p)``
so the expected value of the activation is unchanged:

```text
m ~ Bernoulli(1 - p)          (drawn per element)
out = x * m / (1 - p)
```

In **eval mode** (``model.eval()``) dropout is the identity: ``out = x``,
with no mask and no rescaling. ``p = 0.0`` is the identity in both modes.

The mask is elementwise and shape agnostic: the operator accepts ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` and drops every element of the tensor
independently — it does not drop whole channels or whole positions (that
would be a separate ``dropout2d``-style operator). The output shape always
equals the input shape, and the operator holds no parameters or buffers.

### Randomness

Dropout is the one deliberate exception to HNDL's "no hidden forward
randomness" rule: in training mode ``forward`` draws from torch's **global**
RNG, so two calls on the same input return different results unless the RNG
is reseeded between them, and a plan containing ``dropout(p > 0)`` is only
reproducible if you control ``torch.manual_seed`` yourself. Everything the
resolver sees stays deterministic — the drawn mask never affects shapes,
arguments, or the plan digest.

Because of that, the declared examples all use ``p=0.0``: the generic
example harness builds in training mode and compares two forward passes
without reseeding, which any ``p > 0`` would fail by construction. The
operator's own tests cover the statistics and scaling for ``p > 0``.

The mask is drawn and applied in the plan's compute dtype (float32,
float16, or bfloat16); nothing is upcast.
        """
        ...
    @overload
    def dropout(self, p: float = 0.1, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Randomly zero elements during training and rescale the rest.

        Elementwise dropout, computed out of place so shared branches are never
mutated.

In **training mode** each element of the input is independently zeroed with
probability ``p``, and the surviving elements are scaled by ``1 / (1 - p)``
so the expected value of the activation is unchanged:

```text
m ~ Bernoulli(1 - p)          (drawn per element)
out = x * m / (1 - p)
```

In **eval mode** (``model.eval()``) dropout is the identity: ``out = x``,
with no mask and no rescaling. ``p = 0.0`` is the identity in both modes.

The mask is elementwise and shape agnostic: the operator accepts ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` and drops every element of the tensor
independently — it does not drop whole channels or whole positions (that
would be a separate ``dropout2d``-style operator). The output shape always
equals the input shape, and the operator holds no parameters or buffers.

### Randomness

Dropout is the one deliberate exception to HNDL's "no hidden forward
randomness" rule: in training mode ``forward`` draws from torch's **global**
RNG, so two calls on the same input return different results unless the RNG
is reseeded between them, and a plan containing ``dropout(p > 0)`` is only
reproducible if you control ``torch.manual_seed`` yourself. Everything the
resolver sees stays deterministic — the drawn mask never affects shapes,
arguments, or the plan digest.

Because of that, the declared examples all use ``p=0.0``: the generic
example harness builds in training mode and compares two forward passes
without reseeding, which any ``p > 0`` would fail by construction. The
operator's own tests cover the statistics and scaling for ``p > 0``.

The mask is drawn and applied in the plan's compute dtype (float32,
float16, or bfloat16); nothing is upcast.
        """
        ...
    @overload
    def dy_tanh(self, x: Symbol, dim: int | None = None, *, init_alpha: float = 1.0, affine: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Dynamic Tanh activation with learned scalar alpha and affine scaling.

        Dynamic Tanh activation layer: y = gamma * tanh(alpha * x) + beta.
        """
        ...
    @overload
    def dy_tanh(self, dim: int | None = None, *, init_alpha: float = 1.0, affine: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Dynamic Tanh activation with learned scalar alpha and affine scaling.

        Dynamic Tanh activation layer: y = gamma * tanh(alpha * x) + beta.
        """
        ...
    @overload
    def elu(self, x: Symbol, alpha: float = 1.0, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Exponential linear unit: identity above zero, saturating below.

        Elementwise exponential linear unit:

```
out = x                          if x > 0
out = alpha * (exp(x) - 1)       if x <= 0
```

The function is continuous at the origin with value 0, and for negative
inputs it saturates smoothly at `-alpha` instead of clamping to zero, so
unlike ReLU it keeps a nonzero gradient there. `alpha = 1` gives the
standard form with a continuous derivative at 0.

The operation is computed out of place in the activation dtype
(float32, float16, or bfloat16), preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — has no
parameters, and behaves identically in train and eval mode.
        """
        ...
    @overload
    def elu(self, alpha: float = 1.0, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Exponential linear unit: identity above zero, saturating below.

        Elementwise exponential linear unit:

```
out = x                          if x > 0
out = alpha * (exp(x) - 1)       if x <= 0
```

The function is continuous at the origin with value 0, and for negative
inputs it saturates smoothly at `-alpha` instead of clamping to zero, so
unlike ReLU it keeps a nonzero gradient there. `alpha = 1` gives the
standard form with a continuous derivative at 0.

The operation is computed out of place in the activation dtype
(float32, float16, or bfloat16), preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — has no
parameters, and behaves identically in train and eval mode.
        """
        ...
    def embedding(self, ids: Symbol, num_embeddings: int, dim: int | None = None, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Look up a learned vector for every integer token id.

        A learned lookup table: row ``i`` of ``weight`` is the vector for token
id ``i``.

```text
out[b, t, :] = weight[ids[b, t], :]
```

The input port ``ids`` carries **int64** values, not activations, so the
graph input must be declared with ``input_dtype="int64"`` when
``embedding`` is the first operation:

```python
network("embedding(1000, 32)\nlinear(8)", input_shape=("B", 16),
        output_shape=("B", 16, 8), input_dtype="int64", device="cpu")
```

Ids are positions on the ``T`` axis of a ``[B, T]`` tensor and the result
is the ``[B, T, D]`` sequence those tokens embed to. Every id must satisfy
``0 <= id < num_embeddings``; an out-of-range id is a CUDA-side or
CPU-side indexing fault, not a shape error, because ids are data.

The single parameter is ``weight`` of shape ``[num_embeddings, dim]``,
initialized from the standard normal distribution. It is dense: the whole
table receives a gradient contribution on every step. There is no padding
index and no norm clipping; train and eval behave identically.
        """
        ...
    @overload
    def feed_forward(self, x: Symbol, hidden: int, *, activation: str = "gelu", dropout: float = 0.0, bias: bool = True, equalized: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Transformer feed-forward block: widen, activate, project back.

        The position-wise feed-forward network of a transformer block: a widening
projection, a nonlinearity, optional dropout, and a projection back to the
input width.

```text
h   = activation(x @ up.weight.T + up.bias)      # [B, ..., hidden]
out = dropout(h) @ down.weight.T + down.bias     # [B, ..., D]
```

The last axis carries the features `D`; every leading axis is a batch or
position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
applied independently at each position. Image tensors `[B, C, H, W]` are
rejected with `E_CONSTRAINT`; flatten or reshape them first.

Submodules are `up` (`Linear(D, hidden)`), `dropout` and `down`
(`Linear(hidden, D)`), so the parameters are `up.weight`, `up.bias`,
`down.weight` and `down.bias`. With biases the block holds
`2 * D * hidden + hidden + D` parameters, and `2 * D * hidden` without.

``equalized=True`` initializes both raw projection weights N(0,1), zeros
their biases, and scales each weight by the inverse square root of its own
fan-in (D for up, hidden for down) on every forward. Gain and learning-rate
multiplier are one. Activations and dropout are unchanged. ``init=`` and
checkpoints contain raw weights; parameter names and shapes stay the same.

`activation` selects one of:

| Name | Formula |
| --- | --- |
| `gelu` | `h * Phi(h)` with the exact Gaussian CDF |
| `gelu_tanh` | the `tanh` approximation of GELU |
| `relu` | `max(h, 0)` |
| `silu` | `h * sigmoid(h)` |
| `quick_gelu` | `h * sigmoid(1.702 * h)` |

Dropout is active in train mode only; in eval mode, and whenever `dropout`
is 0, the block is deterministic. Everything is computed in the incoming
dtype, with no upcasting.
        """
        ...
    @overload
    def feed_forward(self, hidden: int, *, activation: str = "gelu", dropout: float = 0.0, bias: bool = True, equalized: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Transformer feed-forward block: widen, activate, project back.

        The position-wise feed-forward network of a transformer block: a widening
projection, a nonlinearity, optional dropout, and a projection back to the
input width.

```text
h   = activation(x @ up.weight.T + up.bias)      # [B, ..., hidden]
out = dropout(h) @ down.weight.T + down.bias     # [B, ..., D]
```

The last axis carries the features `D`; every leading axis is a batch or
position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
applied independently at each position. Image tensors `[B, C, H, W]` are
rejected with `E_CONSTRAINT`; flatten or reshape them first.

Submodules are `up` (`Linear(D, hidden)`), `dropout` and `down`
(`Linear(hidden, D)`), so the parameters are `up.weight`, `up.bias`,
`down.weight` and `down.bias`. With biases the block holds
`2 * D * hidden + hidden + D` parameters, and `2 * D * hidden` without.

``equalized=True`` initializes both raw projection weights N(0,1), zeros
their biases, and scales each weight by the inverse square root of its own
fan-in (D for up, hidden for down) on every forward. Gain and learning-rate
multiplier are one. Activations and dropout are unchanged. ``init=`` and
checkpoints contain raw weights; parameter names and shapes stay the same.

`activation` selects one of:

| Name | Formula |
| --- | --- |
| `gelu` | `h * Phi(h)` with the exact Gaussian CDF |
| `gelu_tanh` | the `tanh` approximation of GELU |
| `relu` | `max(h, 0)` |
| `silu` | `h * sigmoid(h)` |
| `quick_gelu` | `h * sigmoid(1.702 * h)` |

Dropout is active in train mode only; in eval mode, and whenever `dropout`
is 0, the block is deterministic. Everything is computed in the incoming
dtype, with no upcasting.
        """
        ...
    @overload
    def flatten(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Collapse every non-batch axis into one feature axis.

        Equivalent to ``x.reshape(batch, -1)``. The input rank can be inferred
backward only through the element count when the other axes are known.
        """
        ...
    @overload
    def flatten(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Collapse every non-batch axis into one feature axis.

        Equivalent to ``x.reshape(batch, -1)``. The input rank can be inferred
backward only through the element count when the other axes are known.
        """
        ...
    @overload
    def fourier_features(self, x: Symbol, num_frequencies: int, *, scale: float = 8.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Encode coordinates with a fixed random Fourier frequency table.

        With fixed ``frequencies ~ Normal(0, scale)``, compute
``p = x @ frequencies.T`` and ``concat(sin(p), cos(p), dim=-1)``.
No extra 2*pi factor is applied. Frequencies have shape [F,D] and are
a persistent buffer, not trainable parameters. Initialization uses the
construction RNG; forward draws nothing. To share one table between
two sets of coordinates, concatenate them before this operator and
split their features afterwards. Gradients flow to the coordinates.
        """
        ...
    @overload
    def fourier_features(self, num_frequencies: int, *, scale: float = 8.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Encode coordinates with a fixed random Fourier frequency table.

        With fixed ``frequencies ~ Normal(0, scale)``, compute
``p = x @ frequencies.T`` and ``concat(sin(p), cos(p), dim=-1)``.
No extra 2*pi factor is applied. Frequencies have shape [F,D] and are
a persistent buffer, not trainable parameters. Initialization uses the
construction RNG; forward draws nothing. To share one table between
two sets of coordinates, concatenate them before this operator and
split their features afterwards. Gradients flow to the coordinates.
        """
        ...
    @overload
    def gated_delta_attention(self, x: Symbol, heads: int, *, head_dim: int = 0, decay_kind: str = "channel", gate_rank: int = 0, conv_size: int = 4, chunk_size: int = 64, bias: bool = False, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Kimi Delta Attention (KDA) and Gated DeltaNet with chunkwise parallel WY evaluation.

        Kimi Delta Attention (KDA) and Gated DeltaNet associative memory attention.

Evaluates associative memory updates with error-correction delta rule:
residual error ``e_t = v_t - S_{t-1}^T k_t`` is written into memory state ``S_t``.
In ``forward()``, native PyTorch Tensor Core chunkwise parallel WY representation
evaluates intra-chunk dependencies via ``torch.linalg.solve_triangular``, while
inter-chunk states are propagated across chunks.
        """
        ...
    @overload
    def gated_delta_attention(self, heads: int, *, head_dim: int = 0, decay_kind: str = "channel", gate_rank: int = 0, conv_size: int = 4, chunk_size: int = 64, bias: bool = False, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Kimi Delta Attention (KDA) and Gated DeltaNet with chunkwise parallel WY evaluation.

        Kimi Delta Attention (KDA) and Gated DeltaNet associative memory attention.

Evaluates associative memory updates with error-correction delta rule:
residual error ``e_t = v_t - S_{t-1}^T k_t`` is written into memory state ``S_t``.
In ``forward()``, native PyTorch Tensor Core chunkwise parallel WY representation
evaluates intra-chunk dependencies via ``torch.linalg.solve_triangular``, while
inter-chunk states are propagated across chunks.
        """
        ...
    def gather_token(self, x: Symbol, ids: Symbol, *, mode: str = "argmax", name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Select the token of a sequence at the position holding the largest id.

        Picks one position out of a ``[B, T, D]`` sequence per example and
returns the ``[B, D]`` vector found there. Which position is chosen is
decided by a second input, ``ids``, an int64 tensor of shape ``[B, T]``:

```text
p[b]      = argmax_t ids[b, t]
out[b, :] = x[b, p[b], :]
```

This is CLIP's end-of-text pooling. A tokenizer writes the end-of-text
marker as the highest id in the vocabulary, so the argmax of the id row
lands on the final real token of that example and the padding after it is
ignored. Unlike ``pool_tokens("last")``, which always takes position
``T - 1``, the selected position varies per example, which is what a
padded batch needs.

``ids`` is data, not activations. It is typically the graph input itself,
declared with ``input_dtype="int64"`` and fanned out to both
``embedding`` and this operator:

```python
network('tokens = embedding(x, 32, 8)\n'
        'features = linear(tokens, 8)\n'
        'gather_token(features, x)',
        input_shape=("B", 6), output_shape=("B", 8),
        input_dtype="int64", device="cpu")
```

Both inputs are explicit: neither port is filled by the implicit current
tensor, so the configuration names the sequence and the ids. The two must
agree on the batch and on the token count ``T``; a mismatch is a shape
contradiction, reported as ``E_CONSTRAINT`` at resolution.

Ties follow ``torch.argmax``: when a row holds its maximum id more than
once, the first such position wins. An all-equal row therefore selects
position 0. Ids are never bounds-checked against the vocabulary here,
because only their ordering matters.

``mode`` exists so that other selection rules can be added later; today
``"argmax"`` is the only choice.

There are no parameters and no state, so train and eval behave
identically. The gather is exact in every compute dtype: values are moved,
never combined. Gradients flow to ``x`` only, as a scatter that leaves
every unselected position at zero; ``ids`` is integer and carries none.
        """
        ...
    @overload
    def gelu(self, x: Symbol, approximate: str = "none", *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Gaussian error linear unit, x * Phi(x).

        Elementwise ``out = x * Phi(x)``, where ``Phi`` is the standard normal
cumulative distribution function.

With ``approximate="none"`` (the default) the exact form is computed:

```text
out = 0.5 * x * (1 + erf(x / sqrt(2)))
```

With ``approximate="tanh"`` the cheaper tanh approximation is used, which
is what the original BERT and GPT-2 implementations shipped:

```text
out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

The two forms agree to roughly 1e-3 in absolute value but are not
interchangeable when loading weights trained against a specific one, so
the choice is recorded in the plan.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.
        """
        ...
    @overload
    def gelu(self, approximate: str = "none", *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Gaussian error linear unit, x * Phi(x).

        Elementwise ``out = x * Phi(x)``, where ``Phi`` is the standard normal
cumulative distribution function.

With ``approximate="none"`` (the default) the exact form is computed:

```text
out = 0.5 * x * (1 + erf(x / sqrt(2)))
```

With ``approximate="tanh"`` the cheaper tanh approximation is used, which
is what the original BERT and GPT-2 implementations shipped:

```text
out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

The two forms agree to roughly 1e-3 in absolute value but are not
interchangeable when loading weights trained against a specific one, so
the choice is recorded in the plan.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.
        """
        ...
    @overload
    def global_avg_pool(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Average each channel over height and width, producing one value per channel.

        Computes ``out[b, c] = mean(x[b, c, :, :])`` on a ``[B, C, H, W]``
input, returning ``[B, C]``: the spatial axes are reduced away rather
than kept as size-1 axes, so the result feeds `linear` directly.

This is `adaptive_avg_pool(1)` followed by `flatten`, written as one
operation because the pattern is the standard head of a convolutional
classifier. There are no parameters and train and eval behave
identically.

Only rank-4 inputs are accepted, which keeps the axis convention
unambiguous; reduce a ``[B, C, L]`` sequence with a `reshape` to
``[B, C, L, 1]`` first, or use `adaptive_avg_pool`.

Shape inference: ``C`` flows in both directions, while ``H`` and ``W``
are *not* inferable backward — every spatial extent produces the same
output — so they must come from the network input or the preceding
operation. The mean is accumulated in the input dtype.
        """
        ...
    @overload
    def global_avg_pool(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Average each channel over height and width, producing one value per channel.

        Computes ``out[b, c] = mean(x[b, c, :, :])`` on a ``[B, C, H, W]``
input, returning ``[B, C]``: the spatial axes are reduced away rather
than kept as size-1 axes, so the result feeds `linear` directly.

This is `adaptive_avg_pool(1)` followed by `flatten`, written as one
operation because the pattern is the standard head of a convolutional
classifier. There are no parameters and train and eval behave
identically.

Only rank-4 inputs are accepted, which keeps the axis convention
unambiguous; reduce a ``[B, C, L]`` sequence with a `reshape` to
``[B, C, L, 1]`` first, or use `adaptive_avg_pool`.

Shape inference: ``C`` flows in both directions, while ``H`` and ``W``
are *not* inferable backward — every spatial extent produces the same
output — so they must come from the network input or the preceding
operation. The mean is accumulated in the input dtype.
        """
        ...
    def grid_sample(self, x: Symbol, grid: Symbol, *, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Sample an image using a per-example normalized coordinate grid.

        A direct ``torch.nn.functional.grid_sample`` with an explicit grid.

The grid's last axis is (x,y) in normalized [-1,1] coordinates. Image
and grid batches must match. Defaults are bilinear interpolation, zero
padding, and align_corners=False. No parameters or random draws;
train and eval are identical. Gradients flow to image and coordinates
as supported by PyTorch. CUDA backward can be nondeterministic, and
PyTorch's sampler does not support double backward.
        """
        ...
    @overload
    def grn(self, x: Symbol, *, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Global Response Normalization (GRN) for ConvNeXt-V2.

        Global Response Normalization (GRN) for ConvNeXt-V2.

Computes spatial L2 norm per channel, normalizes across the channel axis,
and recalibrates features via zero-initialized learned scale and bias:
``Gx = sqrt(sum(x^2, dim=(2, 3)) + eps)``
``Nx = Gx / (mean(Gx, dim=1) + eps)``
``out = gamma * (x * Nx) + beta + x``.
        """
        ...
    @overload
    def grn(self, *, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Global Response Normalization (GRN) for ConvNeXt-V2.

        Global Response Normalization (GRN) for ConvNeXt-V2.

Computes spatial L2 norm per channel, normalizes across the channel axis,
and recalibrates features via zero-initialized learned scale and bias:
``Gx = sqrt(sum(x^2, dim=(2, 3)) + eps)``
``Nx = Gx / (mean(Gx, dim=1) + eps)``
``out = gamma * (x * Nx) + beta + x``.
        """
        ...
    @overload
    def group_norm(self, x: Symbol, num_groups: int, *, num_channels: int | None = None, eps: float = 1e-05, affine: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize channel groups per example, with learned per-channel affine.

        Splits the channel axis into ``num_groups`` groups and normalizes each
group over its channels and spatial positions using population
statistics. Behavior is identical in train and eval mode. Parameters are
``weight`` and ``bias`` when ``affine`` is true.
        """
        ...
    @overload
    def group_norm(self, num_groups: int, *, num_channels: int | None = None, eps: float = 1e-05, affine: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize channel groups per example, with learned per-channel affine.

        Splits the channel axis into ``num_groups`` groups and normalizes each
group over its channels and spatial positions using population
statistics. Behavior is identical in train and eval mode. Parameters are
``weight`` and ``bias`` when ``affine`` is true.
        """
        ...
    @overload
    def hardswish(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Piecewise-linear approximation of swish, x * relu6(x + 3) / 6.

        Elementwise piecewise-linear approximation of `swish = x * sigmoid(x)`:

```
out = 0                    if x <= -3
out = x                    if x >= +3
out = x * (x + 3) / 6      otherwise
```

Equivalently `out = x * relu6(x + 3) / 6`. The hard sigmoid gate needs
only clamping and a multiply, which makes this noticeably cheaper than
`swish` on hardware without a fast `exp`, while tracking it closely. The
function is continuous, its derivative is discontinuous at `x = -3` and
`x = +3`, and negative inputs below -3 are zeroed exactly.

The operation is computed out of place in the activation dtype
(float32, float16, or bfloat16) and preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`. It has no
parameters and behaves identically in train and eval mode.
        """
        ...
    @overload
    def hardswish(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Piecewise-linear approximation of swish, x * relu6(x + 3) / 6.

        Elementwise piecewise-linear approximation of `swish = x * sigmoid(x)`:

```
out = 0                    if x <= -3
out = x                    if x >= +3
out = x * (x + 3) / 6      otherwise
```

Equivalently `out = x * relu6(x + 3) / 6`. The hard sigmoid gate needs
only clamping and a multiply, which makes this noticeably cheaper than
`swish` on hardware without a fast `exp`, while tracking it closely. The
function is continuous, its derivative is discontinuous at `x = -3` and
`x = +3`, and negative inputs below -3 are zeroed exactly.

The operation is computed out of place in the activation dtype
(float32, float16, or bfloat16) and preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`. It has no
parameters and behaves identically in train and eval mode.
        """
        ...
    @overload
    def hopfield(self, x: Symbol, patterns: int, *, beta: float = 1.0, steps: int = 1, normalize: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Retrieve learned patterns by iterating the modern Hopfield update on the last axis.

        A modern (continuous) Hopfield layer, following Ramsauer et al.,
*Hopfield Networks is All You Need* (2020).

The layer owns a learned memory matrix ``stored`` of shape
``[patterns, D]`` whose rows are the stored patterns ``X``. For a query
``xi`` of width ``D`` it minimizes the energy

```text
E(xi) = -lse(beta, X xi) + 0.5 * xi.T @ xi + log(patterns)/beta + 0.5 * M**2
```

where ``lse`` is the log-sum-exp over the stored patterns and ``M`` is the
largest pattern norm. The concave-convex procedure gives the update that
this layer applies, repeated ``steps`` times:

```text
xi <- X.T @ softmax(beta * X @ xi)
```

Written as a batch of rows, ``xi <- softmax(beta * xi @ stored.T) @
stored``. A single update is exactly dot-product attention with the query
``xi``, the stored patterns as both keys and values, and ``beta`` in place
of the ``1/sqrt(d)`` scale: this layer is attention whose keys and values
are parameters rather than activations. Iterating the update is the
associative-memory reading of the same equation, and the fixed points are
the stored patterns (or, at small ``beta``, their metastable averages).

Retrieval behavior follows ``beta``. With ``beta`` large and the patterns
well separated, a query near a stored pattern makes one softmax weight
dominate, so the output is that pattern and further steps do not move it.
With ``beta`` small the softmax is flat and the output tends toward the
mean of the stored patterns; intermediate values retrieve averages of the
patterns a query is close to. Because the energy decreases monotonically
the iteration converges, in practice within a handful of updates, so
``steps > 1`` mainly sharpens retrieval rather than changing its nature.

Axis convention: the layer acts on the last axis of ``[B, D]`` or
``[B, T, D]`` inputs, so every position of a sequence queries the same
memory independently. Image tensors are rejected; flatten or reshape
first. The output has the shape of the input.

When ``normalize`` is true the query is layer-normalized over the last
axis, without a learned affine and with ``eps = 1e-5``, before the first
update only; later updates consume the previous retrieval unchanged. The
normalization keeps ``beta * X @ xi`` in a usable range when the incoming
activations have drifted in scale.

The only parameter is ``stored``, initialized from ``normal(0, 1/sqrt(D))``
so that initial scores have unit scale. Behavior is identical in train and
eval mode; there are no buffers and no running statistics. Everything is
computed in the input dtype, except that the layer normalization follows
``torch.nn.functional.layer_norm``, which accumulates its statistics in
float32 for float16 and bfloat16 inputs.
        """
        ...
    @overload
    def hopfield(self, patterns: int, *, beta: float = 1.0, steps: int = 1, normalize: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Retrieve learned patterns by iterating the modern Hopfield update on the last axis.

        A modern (continuous) Hopfield layer, following Ramsauer et al.,
*Hopfield Networks is All You Need* (2020).

The layer owns a learned memory matrix ``stored`` of shape
``[patterns, D]`` whose rows are the stored patterns ``X``. For a query
``xi`` of width ``D`` it minimizes the energy

```text
E(xi) = -lse(beta, X xi) + 0.5 * xi.T @ xi + log(patterns)/beta + 0.5 * M**2
```

where ``lse`` is the log-sum-exp over the stored patterns and ``M`` is the
largest pattern norm. The concave-convex procedure gives the update that
this layer applies, repeated ``steps`` times:

```text
xi <- X.T @ softmax(beta * X @ xi)
```

Written as a batch of rows, ``xi <- softmax(beta * xi @ stored.T) @
stored``. A single update is exactly dot-product attention with the query
``xi``, the stored patterns as both keys and values, and ``beta`` in place
of the ``1/sqrt(d)`` scale: this layer is attention whose keys and values
are parameters rather than activations. Iterating the update is the
associative-memory reading of the same equation, and the fixed points are
the stored patterns (or, at small ``beta``, their metastable averages).

Retrieval behavior follows ``beta``. With ``beta`` large and the patterns
well separated, a query near a stored pattern makes one softmax weight
dominate, so the output is that pattern and further steps do not move it.
With ``beta`` small the softmax is flat and the output tends toward the
mean of the stored patterns; intermediate values retrieve averages of the
patterns a query is close to. Because the energy decreases monotonically
the iteration converges, in practice within a handful of updates, so
``steps > 1`` mainly sharpens retrieval rather than changing its nature.

Axis convention: the layer acts on the last axis of ``[B, D]`` or
``[B, T, D]`` inputs, so every position of a sequence queries the same
memory independently. Image tensors are rejected; flatten or reshape
first. The output has the shape of the input.

When ``normalize`` is true the query is layer-normalized over the last
axis, without a learned affine and with ``eps = 1e-5``, before the first
update only; later updates consume the previous retrieval unchanged. The
normalization keeps ``beta * X @ xi`` in a usable range when the incoming
activations have drifted in scale.

The only parameter is ``stored``, initialized from ``normal(0, 1/sqrt(D))``
so that initial scores have unit scale. Behavior is identical in train and
eval mode; there are no buffers and no running statistics. Everything is
computed in the input dtype, except that the layer normalization follows
``torch.nn.functional.layer_norm``, which accumulates its statistics in
float32 for float16 and bfloat16 inputs.
        """
        ...
    @overload
    def identity(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Pass the tensor through unchanged, as a named node.

        Returns `x` unchanged: `out = x`, the same tensor object, with no copy,
no parameters, and no effect on the autograd graph beyond passing the
gradient straight through.

It exists so a configuration can name a point in the graph — `h =
identity(name="hidden")` gives a branch or a later `add` something to refer
to — and so a slot in a chain can be filled without changing the
computation, for instance when a normalization or activation is being
ablated. Shape, dtype, and device are preserved exactly for any supported
rank: rank 2 `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`.
Because the shape relation is the identity, it is fully transparent to
resolution in both directions. Behavior is identical in train and eval
mode.
        """
        ...
    @overload
    def identity(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Pass the tensor through unchanged, as a named node.

        Returns `x` unchanged: `out = x`, the same tensor object, with no copy,
no parameters, and no effect on the autograd graph beyond passing the
gradient straight through.

It exists so a configuration can name a point in the graph — `h =
identity(name="hidden")` gives a branch or a later `add` something to refer
to — and so a slot in a chain can be filled without changing the
computation, for instance when a normalization or activation is being
ablated. Shape, dtype, and device are preserved exactly for any supported
rank: rank 2 `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`.
Because the shape relation is the identity, it is fully transparent to
resolution in both directions. Behavior is identical in train and eval
mode.
        """
        ...
    @overload
    def instance_norm(self, x: Symbol, *, eps: float = 1e-05, affine: bool = False, num_features: int | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize every channel of every example over its own spatial positions.

        Normalizes each channel of each example independently over its spatial
positions, with no interaction between examples:

```text
out[b, c] = weight[c] * (x[b, c] - mean(x[b, c])) / sqrt(var(x[b, c]) + eps) + bias[c]
```

Supported ranks are 3 `[B, C, L]` and 4 `[B, C, H, W]`; the rank is fixed
at build time from the resolved input shape, so the module is exactly
`nn.InstanceNorm1d` for rank 3 and `nn.InstanceNorm2d` for rank 4. A
rank-2 `[B, C]` input is rejected during resolution, because a single
value per channel has no variance to normalize.

There are no running statistics (`track_running_stats` is always false),
so train and eval mode behave identically and the layer is deterministic
per example. Parameters are `weight` and `bias` when `affine` is true, one
value per channel; with the default `affine=False` the layer has no
parameters and no buffers at all.

Statistics are computed in the plan's compute dtype. A constant channel
normalizes to zero, up to `eps`.
        """
        ...
    @overload
    def instance_norm(self, *, eps: float = 1e-05, affine: bool = False, num_features: int | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize every channel of every example over its own spatial positions.

        Normalizes each channel of each example independently over its spatial
positions, with no interaction between examples:

```text
out[b, c] = weight[c] * (x[b, c] - mean(x[b, c])) / sqrt(var(x[b, c]) + eps) + bias[c]
```

Supported ranks are 3 `[B, C, L]` and 4 `[B, C, H, W]`; the rank is fixed
at build time from the resolved input shape, so the module is exactly
`nn.InstanceNorm1d` for rank 3 and `nn.InstanceNorm2d` for rank 4. A
rank-2 `[B, C]` input is rejected during resolution, because a single
value per channel has no variance to normalize.

There are no running statistics (`track_running_stats` is always false),
so train and eval mode behave identically and the layer is deterministic
per example. Parameters are `weight` and `bias` when `affine` is true, one
value per channel; with the default `affine=False` the layer has no
parameters and no buffers at all.

Statistics are computed in the plan's compute dtype. A constant channel
normalizes to zero, up to `eps`.
        """
        ...
    def joint_attention(self, x: Symbol, context: Symbol, heads: int, *, head_dim: int = 0, qk_norm: str = "none", bias: bool = True, dropout: float = 0.0, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Dual-stream MMDiT joint self-attention across image and context streams.

        Dual-stream Joint Attention (MMDiT / Flux) operator.
        """
        ...
    @overload
    def l2_norm(self, x: Symbol, dim: int = -1, *, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Parameter-free L2 normalization along a specified axis.

        Normalize vectors along a specified dimension to unit L2 norm.

Applies parameter-free L2 normalization:
``out = x / max(norm(x, 2, dim), eps)``.
        """
        ...
    @overload
    def l2_norm(self, dim: int = -1, *, eps: float = 1e-06, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Parameter-free L2 normalization along a specified axis.

        Normalize vectors along a specified dimension to unit L2 norm.

Applies parameter-free L2 normalization:
``out = x / max(norm(x, 2, dim), eps)``.
        """
        ...
    @overload
    def latent_attention(self, x: Symbol, heads: int, kv_lora_rank: int, *, q_lora_rank: int = 0, qk_nope_head_dim: int = 0, qk_rope_head_dim: int = 0, v_head_dim: int = 0, causal: bool = True, rope_theta: float = 10000.0, rope_scaling: str = "none", rope_scale: float = 1.0, spatial_shape: int | tuple[int, int] = (0, 0), bias: bool = False, softmax_scale: float = 0.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multi-Head Latent Attention (MLA) with compressed KV cache and decoupled RoPE.

        Multi-Head Latent Attention (MLA) with low-rank KV compression and decoupled RoPE.

Introduced in DeepSeek-V2/V3 and Kimi Linear, MLA down-projects keys and
values into a shared latent vector ``c_kv = RMSNorm(W_DKV x)`` of rank
``kv_lora_rank``, which is up-projected into non-positional content keys
and values. A separate decoupled RoPE key vector ``k_rope`` is shared
across all query heads when ``qk_rope_head_dim > 0``, while setting
``qk_rope_head_dim = 0`` activates position-free NoPE mode.
        """
        ...
    @overload
    def latent_attention(self, heads: int, kv_lora_rank: int, *, q_lora_rank: int = 0, qk_nope_head_dim: int = 0, qk_rope_head_dim: int = 0, v_head_dim: int = 0, causal: bool = True, rope_theta: float = 10000.0, rope_scaling: str = "none", rope_scale: float = 1.0, spatial_shape: int | tuple[int, int] = (0, 0), bias: bool = False, softmax_scale: float = 0.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multi-Head Latent Attention (MLA) with compressed KV cache and decoupled RoPE.

        Multi-Head Latent Attention (MLA) with low-rank KV compression and decoupled RoPE.

Introduced in DeepSeek-V2/V3 and Kimi Linear, MLA down-projects keys and
values into a shared latent vector ``c_kv = RMSNorm(W_DKV x)`` of rank
``kv_lora_rank``, which is up-projected into non-positional content keys
and values. A separate decoupled RoPE key vector ``k_rope`` is shared
across all query heads when ``qk_rope_head_dim > 0``, while setting
``qk_rope_head_dim = 0`` activates position-free NoPE mode.
        """
        ...
    @overload
    def layer_norm(self, x: Symbol, *, eps: float = 1e-05, affine: bool = True, bias: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize the last axis of every position with a learned scale and bias.

        Normalizes each row of the last axis ``D`` to zero mean and unit
variance, then applies a learned per-feature affine:

```text
mean = mean(x, dim=-1)
var  = mean((x - mean)^2, dim=-1)            # biased: divided by D
out  = (x - mean) / sqrt(var + eps) * weight + bias
```

The statistics are taken over the last axis only, so a ``[B, D]`` input
normalizes each example and a ``[B, T, D]`` sequence normalizes each
position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
relation but would be normalized over ``W`` alone, which is almost never
what is wanted; reach for `group_norm` on images. The examples below
therefore cover only ranks 2 and 3.

Parameters are ``weight`` (initialized to ones) and, when ``bias`` is
true, ``bias`` (initialized to zeros), both of shape ``[D]``, and exist
only when ``affine`` is true. There are no running statistics, so train
and eval behave identically. The computation runs in the plan's compute
dtype.
        """
        ...
    @overload
    def layer_norm(self, *, eps: float = 1e-05, affine: bool = True, bias: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize the last axis of every position with a learned scale and bias.

        Normalizes each row of the last axis ``D`` to zero mean and unit
variance, then applies a learned per-feature affine:

```text
mean = mean(x, dim=-1)
var  = mean((x - mean)^2, dim=-1)            # biased: divided by D
out  = (x - mean) / sqrt(var + eps) * weight + bias
```

The statistics are taken over the last axis only, so a ``[B, D]`` input
normalizes each example and a ``[B, T, D]`` sequence normalizes each
position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
relation but would be normalized over ``W`` alone, which is almost never
what is wanted; reach for `group_norm` on images. The examples below
therefore cover only ranks 2 and 3.

Parameters are ``weight`` (initialized to ones) and, when ``bias`` is
true, ``bias`` (initialized to zeros), both of shape ``[D]``, and exist
only when ``affine`` is true. There are no running statistics, so train
and eval behave identically. The computation runs in the plan's compute
dtype.
        """
        ...
    @overload
    def leaky_relu(self, x: Symbol, negative_slope: float = 0.01, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """ReLU with a small slope for negative inputs.

        ``out = x if x >= 0 else negative_slope * x``, computed out of place.
        """
        ...
    @overload
    def leaky_relu(self, negative_slope: float = 0.01, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """ReLU with a small slope for negative inputs.

        ``out = x if x >= 0 else negative_slope * x``, computed out of place.
        """
        ...
    @overload
    def learned_scale(self, x: Symbol, *, init_value: float = 0.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multiply a tensor by one learned scalar.

        ``out = gamma * x`` where ``gamma`` is a single learned scalar shared by
every element of the tensor:

```text
out[i, ...] = gamma * x[i, ...]
```

The one parameter is named ``gamma`` and has shape ``[1]``, so it is the
target of construction overrides by that exact name:
``learned_scale(init={"gamma": 1.0})`` sets it and
``learned_scale(trainable={"gamma": False})`` freezes it. ``init_value``
gives the value the constructor starts from when no initialization
override applies; it is a float and does not take part in shape inference.

The default ``init_value=0.0`` is the Self-Attention GAN convention (Zhang
et al. 2018): the layer starts as a zero map, so a residual
``add(learned_scale(branch), saved)`` begins exactly as the identity and
the network learns how much of the branch to admit. Note that a gamma of
zero also zeroes the gradient flowing back through ``x``; the branch
parameters start learning through ``gamma`` itself, whose gradient is the
inner product of the branch output with the incoming gradient.

Rank, shape, and dtype are preserved, there are no buffers, and train and
eval behave identically. Use ``scale`` instead for a fixed constant that is
not learned.
        """
        ...
    @overload
    def learned_scale(self, *, init_value: float = 0.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multiply a tensor by one learned scalar.

        ``out = gamma * x`` where ``gamma`` is a single learned scalar shared by
every element of the tensor:

```text
out[i, ...] = gamma * x[i, ...]
```

The one parameter is named ``gamma`` and has shape ``[1]``, so it is the
target of construction overrides by that exact name:
``learned_scale(init={"gamma": 1.0})`` sets it and
``learned_scale(trainable={"gamma": False})`` freezes it. ``init_value``
gives the value the constructor starts from when no initialization
override applies; it is a float and does not take part in shape inference.

The default ``init_value=0.0`` is the Self-Attention GAN convention (Zhang
et al. 2018): the layer starts as a zero map, so a residual
``add(learned_scale(branch), saved)`` begins exactly as the identity and
the network learns how much of the branch to admit. Note that a gamma of
zero also zeroes the gradient flowing back through ``x``; the branch
parameters start learning through ``gamma`` itself, whose gradient is the
inner product of the branch output with the incoming gradient.

Rank, shape, and dtype are preserved, there are no buffers, and train and
eval behave identically. Use ``scale`` instead for a fixed constant that is
not learned.
        """
        ...
    @overload
    def linear(self, x: Symbol, out_features: int | None = None, *, in_features: int | None = None, bias: bool = True, spectral_norm: bool = False, equalized: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Fully connected layer: a learned affine map on the last axis.

        Computes ``out = x @ weight.T + bias`` with ``weight`` of shape
``[out_features, in_features]``, applied to the last axis of ``[B, D]``
or ``[B, T, D]`` inputs. No activation is applied; add one explicitly.
Parameters are ``weight`` and, when enabled, ``bias``.

## Equalized learning rate

``equalized=True`` initializes raw weights from N(0,1) and biases at zero.
Every forward uses ``weight / sqrt(in_features)``; bias remains unscaled.
Gain and learning-rate multiplier are both one; activations remain separate.
Parameter names and shapes are unchanged, but checkpoints and ``init=``
contain raw weights. Loading ordinary linear weights directly therefore
changes their effective scale. This option cannot combine with spectral norm.

## Spectral normalization

With ``spectral_norm=True`` the weight is reparametrized as ``weight /
sigma(weight)``, where ``sigma`` is the largest singular value estimated
by one power iteration per forward pass
(``torch.nn.utils.parametrizations.spectral_norm``). Each layer is then
1-Lipschitz, which is the standard constraint for a GAN discriminator.

The parametrization renames the registered state: the learned tensor
becomes ``parametrizations.weight.original`` and ``weight`` turns into a
computed attribute, with persistent buffers
``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
holding the power-iteration vectors. ``init`` and ``trainable``
overrides must therefore target ``parametrizations.weight.original``
instead of ``weight``; ``bias`` is unaffected. The power iteration
refreshes the buffers in training mode only, so evaluation is a pure
function of the stored state.
        """
        ...
    @overload
    def linear(self, out_features: int | None = None, *, in_features: int | None = None, bias: bool = True, spectral_norm: bool = False, equalized: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Fully connected layer: a learned affine map on the last axis.

        Computes ``out = x @ weight.T + bias`` with ``weight`` of shape
``[out_features, in_features]``, applied to the last axis of ``[B, D]``
or ``[B, T, D]`` inputs. No activation is applied; add one explicitly.
Parameters are ``weight`` and, when enabled, ``bias``.

## Equalized learning rate

``equalized=True`` initializes raw weights from N(0,1) and biases at zero.
Every forward uses ``weight / sqrt(in_features)``; bias remains unscaled.
Gain and learning-rate multiplier are both one; activations remain separate.
Parameter names and shapes are unchanged, but checkpoints and ``init=``
contain raw weights. Loading ordinary linear weights directly therefore
changes their effective scale. This option cannot combine with spectral norm.

## Spectral normalization

With ``spectral_norm=True`` the weight is reparametrized as ``weight /
sigma(weight)``, where ``sigma`` is the largest singular value estimated
by one power iteration per forward pass
(``torch.nn.utils.parametrizations.spectral_norm``). Each layer is then
1-Lipschitz, which is the standard constraint for a GAN discriminator.

The parametrization renames the registered state: the learned tensor
becomes ``parametrizations.weight.original`` and ``weight`` turns into a
computed attribute, with persistent buffers
``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
holding the power-iteration vectors. ``init`` and ``trainable``
overrides must therefore target ``parametrizations.weight.original``
instead of ``weight``; ``bias`` is unaffected. The power iteration
refreshes the buffers in training mode only, so evaluation is a pure
function of the stored state.
        """
        ...
    @overload
    def mamba2(self, x: Symbol, d_model: int | None = None, *, d_state: int = 64, d_conv: int = 4, expand: int = 2, heads: int = 8, head_dim: int = 64, chunk_size: int = 64, bias: bool = False, conv_bias: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """State Space Duality (SSD) block from Mamba-2.

        Mamba-2 State Space Duality (SSD) layer.
        """
        ...
    @overload
    def mamba2(self, d_model: int | None = None, *, d_state: int = 64, d_conv: int = 4, expand: int = 2, heads: int = 8, head_dim: int = 64, chunk_size: int = 64, bias: bool = False, conv_bias: bool = True, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """State Space Duality (SSD) block from Mamba-2.

        Mamba-2 State Space Duality (SSD) layer.
        """
        ...
    def matmul(self, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Batched matrix product of tensors with arbitrary leading batch dimensions.

        ``out[i] = a[i] @ b[i]`` for every batch element ``i``: the operation
``torch.bmm`` performs (``torch.matmul`` agrees at rank 3).

```text
a[B, N, K] @ b[B, K, M] -> out[B, N, M]
out[i, n, m] = sum_k a[i, n, k] * b[i, k, m]
```

Both tensor inputs must be supplied explicitly, as with ``mul`` and
``add``; there is no implicit current-tensor operand for the second port.
The inner dimension ``K`` is a shared symbol, so a known width on either
side fixes the other and a mismatch is reported as ``E_CONSTRAINT`` while
the plan resolves. The batch is shared: there is no broadcasting of the
batch axis or of a missing axis.

Batched matrix multiplication supports rank 2, rank 3, and rank 4 tensors with
matching leading batch dimensions [B, ..., N, K] @ [B, ..., K, M] -> [B, ..., N, M].
Both operands must share the same rank and leading dimensions.

There are no parameters and no buffers, train and eval behave identically,
and the product runs in the plan compute dtype. Reduced-precision matrix
multiplies accumulate in float32 on CUDA tensor cores, so float16 and
bfloat16 results are close to, but not bitwise equal to, the same product
done in full precision.
        """
        ...
    @overload
    def max_pool(self, x: Symbol, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] = (0, 0), padding: int | tuple[int, int] = (0, 0), name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Two-dimensional max pooling over [B, C, H, W] images.

        Takes the maximum over each ``kernel_size`` window of the spatial axes
of a ``[B, C, H, W]`` image, stepping by ``stride`` and treating the
``padding`` border as negative infinity so padded positions never win.
Batch and channels pass through unchanged and

``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

with the same rule on width; dilation is 1 and ``ceil_mode`` is false, so
a trailing partial window is dropped. ``stride=0`` is the default and
means "the same as ``kernel_size``", which is PyTorch's own default of
non-overlapping windows; any other value is used as written. ``padding``
may not exceed ``kernel_size//2`` on an axis. There are no parameters and
no train/eval difference. The gradient routes to the argmax of each
window, so ties send it to a single position.
        """
        ...
    @overload
    def max_pool(self, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] = (0, 0), padding: int | tuple[int, int] = (0, 0), name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Two-dimensional max pooling over [B, C, H, W] images.

        Takes the maximum over each ``kernel_size`` window of the spatial axes
of a ``[B, C, H, W]`` image, stepping by ``stride`` and treating the
``padding`` border as negative infinity so padded positions never win.
Batch and channels pass through unchanged and

``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

with the same rule on width; dilation is 1 and ``ceil_mode`` is false, so
a trailing partial window is dropped. ``stride=0`` is the default and
means "the same as ``kernel_size``", which is PyTorch's own default of
non-overlapping windows; any other value is used as written. ``padding``
may not exceed ``kernel_size//2`` on an axis. There are no parameters and
no train/eval difference. The gradient routes to the argmax of each
window, so ties send it to a single position.
        """
        ...
    @overload
    def mean(self, x: Symbol, dim: int, *, keepdim: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Average one non-batch axis away.

        Arithmetic mean over exactly one non-batch axis:
``out = x.mean(dim=dim, keepdim=keepdim)``.

With ``keepdim=False`` (the default) the output rank is one lower than the
input rank and the axes after ``dim`` shift down by one. With
``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
``[B, T, D]`` (``mean(1)`` pools over positions, ``mean(2)`` over
features), rank 4 is ``[B, C, H, W]`` (``mean(1)`` pools over channels,
``mean(2)`` over height, ``mean(3)`` over width), and a 1-D convolution
layout ``[B, C, L]`` reduces its length with ``mean(2)``.

Because HNDL tensors always keep a batch axis and at least one further
axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
``E_CONSTRAINT``; use ``keepdim=True`` instead.

The layer has no parameters and behaves identically in train and eval
mode. The reduction runs in the incoming dtype — under ``float16`` a long
axis accumulates rounding error, which ``sum`` shares.

Shape inference is bidirectional: a known output fixes every non-reduced
input axis, but the extent of the reduced axis itself carries no
information backward and must come from the input side.
        """
        ...
    @overload
    def mean(self, dim: int, *, keepdim: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Average one non-batch axis away.

        Arithmetic mean over exactly one non-batch axis:
``out = x.mean(dim=dim, keepdim=keepdim)``.

With ``keepdim=False`` (the default) the output rank is one lower than the
input rank and the axes after ``dim`` shift down by one. With
``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
``[B, T, D]`` (``mean(1)`` pools over positions, ``mean(2)`` over
features), rank 4 is ``[B, C, H, W]`` (``mean(1)`` pools over channels,
``mean(2)`` over height, ``mean(3)`` over width), and a 1-D convolution
layout ``[B, C, L]`` reduces its length with ``mean(2)``.

Because HNDL tensors always keep a batch axis and at least one further
axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
``E_CONSTRAINT``; use ``keepdim=True`` instead.

The layer has no parameters and behaves identically in train and eval
mode. The reduction runs in the incoming dtype — under ``float16`` a long
axis accumulates rounding error, which ``sum`` shares.

Shape inference is bidirectional: a known output fixes every non-reduced
input axis, but the extent of the reduced axis itself carries no
information backward and must come from the input side.
        """
        ...
    @overload
    def mish(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Self-gated smooth activation, x * tanh(softplus(x)).

        Elementwise self-gated activation:

```
out = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
```

The gate `tanh(softplus(x))` rises smoothly from 0 to 1, so `mish`
approaches the identity for large positive inputs and decays towards 0 for
large negative ones, with a small negative dip near `x = -1`. It is smooth
everywhere (unlike ReLU) and unbounded above, which keeps gradients alive
on the negative side.

`softplus` inside the gate uses PyTorch's stable formulation, so the
activation is safe in float16 and bfloat16 as well as float32; it is
computed in the activation dtype without upcasting. It is elementwise, so
it preserves any supported shape — rank 2 `[B, F]`, rank 3 `[B, T, D]`, or
rank 4 `[B, C, H, W]` — has no parameters, and behaves identically in
train and eval mode.
        """
        ...
    @overload
    def mish(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Self-gated smooth activation, x * tanh(softplus(x)).

        Elementwise self-gated activation:

```
out = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
```

The gate `tanh(softplus(x))` rises smoothly from 0 to 1, so `mish`
approaches the identity for large positive inputs and decays towards 0 for
large negative ones, with a small negative dip near `x = -1`. It is smooth
everywhere (unlike ReLU) and unbounded above, which keeps gradients alive
on the negative side.

`softplus` inside the gate uses PyTorch's stable formulation, so the
activation is safe in float16 and bfloat16 as well as float32; it is
computed in the activation dtype without upcasting. It is elementwise, so
it preserves any supported shape — rank 2 `[B, F]`, rank 3 `[B, T, D]`, or
rank 4 `[B, C, H, W]` — has no parameters, and behaves identically in
train and eval mode.
        """
        ...
    @overload
    def moe(self, x: Symbol, experts: int, hidden: int, top_k: int = 2, *, activation: str = "gelu", expert_kind: str = "mlp", bias: bool = True, scoring: str = "softmax", routed_scaling_factor: float = 1.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Sparse mixture of experts: route every token to its top-k feed-forward experts.

        A sparsely gated mixture of experts over the last axis of ``[B, D]`` or
``[B, T, D]`` inputs. Every row of the flattened tensor (one example, or
one position of one example) is a *token* and is routed independently.

```text
logits  = router(x)                        # [N, experts], bias-free linear
v, idx  = topk(logits, top_k)              # the top_k experts per token
w       = softmax(v) or sigmoid(v)/sum     # renormalized over the selection
out     = sum_k w[:, k] * expert[idx[:, k]](x)
```

Each expert is either ``down(activation(up(x)))`` (for ``expert_kind="mlp"``)
or ``down(silu(gate(x)) * up(x))`` (for ``expert_kind="swiglu"``).
The experts live in an ``nn.ModuleList`` named ``experts``; the router is
``router.weight``.
        """
        ...
    @overload
    def moe(self, experts: int, hidden: int, top_k: int = 2, *, activation: str = "gelu", expert_kind: str = "mlp", bias: bool = True, scoring: str = "softmax", routed_scaling_factor: float = 1.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Sparse mixture of experts: route every token to its top-k feed-forward experts.

        A sparsely gated mixture of experts over the last axis of ``[B, D]`` or
``[B, T, D]`` inputs. Every row of the flattened tensor (one example, or
one position of one example) is a *token* and is routed independently.

```text
logits  = router(x)                        # [N, experts], bias-free linear
v, idx  = topk(logits, top_k)              # the top_k experts per token
w       = softmax(v) or sigmoid(v)/sum     # renormalized over the selection
out     = sum_k w[:, k] * expert[idx[:, k]](x)
```

Each expert is either ``down(activation(up(x)))`` (for ``expert_kind="mlp"``)
or ``down(silu(gate(x)) * up(x))`` (for ``expert_kind="swiglu"``).
The experts live in an ``nn.ModuleList`` named ``experts``; the router is
``router.weight``.
        """
        ...
    def mul(self, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Elementwise product of two tensors with identical shapes.

        ``out = a * b`` (the Hadamard product) with no broadcasting; the two
operands must agree on rank, every axis extent, and dtype, and the product
is computed in the plan compute dtype. Both tensor inputs must be supplied
explicitly. Use ``scale`` instead to multiply by a constant.
        """
        ...
    @overload
    def pad(self, x: Symbol, padding: int | Sequence[int] = (), *, value: float = 0.0, mode: str = "constant", name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Enlarge trailing axes with constant, reflected, or replicated borders.

        ``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

``padding`` lists ``(left, right)`` pairs **starting from the last axis**,
exactly as ``F.pad`` orders them:

- rank 2 ``[B, F]``: ``pad(left, right)`` widens the feature axis ``F``.
- rank 3 ``[B, C, L]`` (equivalently ``[B, T, D]``): ``pad(left, right)``
  widens the last axis ``L``.
- rank 4 ``[B, C, H, W]``: ``pad(left, right, top, bottom)`` widens ``W``
  first and then ``H``.

The batch axis is never padded, so ``padding`` holds at most
``rank - 1`` pairs; anything longer, or an odd number of values, is
``E_ARGUMENT``. Each padded axis grows by ``left + right`` and every other
axis is unchanged, and the relation runs in both directions: a known input
extent fixes the output, and a known output extent fixes the input.

``mode='constant'`` fills the new positions with ``value`` and works for
every supported rank. ``mode='reflect'`` and ``mode='replicate'`` follow
the PyTorch restriction to spatial padding: a rank-3 tensor padded on its
last axis, or a rank-4 tensor padded on ``H`` and ``W``; other
rank/padding combinations are rejected with ``E_ARGUMENT``. ``reflect``
additionally requires each pad width to be smaller than the extent it
mirrors (``E_CONSTRAINT``), because the border itself is not repeated.
``value`` must stay at its default for the non-constant modes, which
PyTorch does not accept.

There are no parameters, behavior is identical in train and eval mode, the
output keeps the input dtype, and the gradient of the padded positions is
discarded (``constant``) or accumulated back onto the mirrored or repeated
source positions (``reflect``, ``replicate``).
        """
        ...
    @overload
    def pad(self, padding: int | Sequence[int] = (), *, value: float = 0.0, mode: str = "constant", name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Enlarge trailing axes with constant, reflected, or replicated borders.

        ``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

``padding`` lists ``(left, right)`` pairs **starting from the last axis**,
exactly as ``F.pad`` orders them:

- rank 2 ``[B, F]``: ``pad(left, right)`` widens the feature axis ``F``.
- rank 3 ``[B, C, L]`` (equivalently ``[B, T, D]``): ``pad(left, right)``
  widens the last axis ``L``.
- rank 4 ``[B, C, H, W]``: ``pad(left, right, top, bottom)`` widens ``W``
  first and then ``H``.

The batch axis is never padded, so ``padding`` holds at most
``rank - 1`` pairs; anything longer, or an odd number of values, is
``E_ARGUMENT``. Each padded axis grows by ``left + right`` and every other
axis is unchanged, and the relation runs in both directions: a known input
extent fixes the output, and a known output extent fixes the input.

``mode='constant'`` fills the new positions with ``value`` and works for
every supported rank. ``mode='reflect'`` and ``mode='replicate'`` follow
the PyTorch restriction to spatial padding: a rank-3 tensor padded on its
last axis, or a rank-4 tensor padded on ``H`` and ``W``; other
rank/padding combinations are rejected with ``E_ARGUMENT``. ``reflect``
additionally requires each pad width to be smaller than the extent it
mirrors (``E_CONSTRAINT``), because the border itself is not repeated.
``value`` must stay at its default for the non-constant modes, which
PyTorch does not accept.

There are no parameters, behavior is identical in train and eval mode, the
output keeps the input dtype, and the gradient of the padded positions is
discarded (``constant``) or accumulated back onto the mirrored or repeated
source positions (``reflect``, ``replicate``).
        """
        ...
    @overload
    def patch_embed(self, x: Symbol, dim: int | None = None, patch_size: int = ..., *, in_channels: int | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Cut an image into non-overlapping patches and embed each one as a token.

        The patch-embedding stem of a vision transformer.

The input image ``[B, C, H, W]`` is cut into non-overlapping ``patch_size``
x ``patch_size`` tiles and each tile is projected to ``dim`` features:

```
out = proj(x).flatten(2).transpose(1, 2)
```

where ``proj`` is an ``nn.Conv2d(in_channels, dim, kernel_size=patch_size,
stride=patch_size)``. Because the kernel and the stride are equal, the
convolution reads every pixel exactly once, so the layer is a per-patch
affine map on the flattened ``[C, patch_size, patch_size]`` window.

Axis conventions: the input is an image ``[B, C, H, W]`` and the output is
a sequence ``[B, N, D]`` with ``N = (H / patch_size) * (W / patch_size)``
tokens in row-major order (all patches of the first patch row first) and
``D = dim`` features on the last axis, which is what the sequence
operations such as ``linear`` act on. ``patch_size`` must divide both ``H``
and ``W``; a remainder is reported as ``E_CONSTRAINT`` rather than being
cropped away. Inference runs in both directions: ``dim`` and
``in_channels`` follow the neighbouring contracts, and a known token count
fixes the missing spatial extent when the other one is known. A token
count alone does not determine ``H`` and ``W`` and is reported as
ambiguous instead of guessed.

No positional embedding, class token, or normalization is added; compose
those explicitly. Behavior is identical in train and eval mode. The single
submodule is ``proj``, whose parameters are ``proj.weight`` of shape
``[dim, in_channels, patch_size, patch_size]`` and ``proj.bias``. The
projection runs entirely in the plan compute dtype.
        """
        ...
    @overload
    def patch_embed(self, dim: int | None = None, patch_size: int = ..., *, in_channels: int | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Cut an image into non-overlapping patches and embed each one as a token.

        The patch-embedding stem of a vision transformer.

The input image ``[B, C, H, W]`` is cut into non-overlapping ``patch_size``
x ``patch_size`` tiles and each tile is projected to ``dim`` features:

```
out = proj(x).flatten(2).transpose(1, 2)
```

where ``proj`` is an ``nn.Conv2d(in_channels, dim, kernel_size=patch_size,
stride=patch_size)``. Because the kernel and the stride are equal, the
convolution reads every pixel exactly once, so the layer is a per-patch
affine map on the flattened ``[C, patch_size, patch_size]`` window.

Axis conventions: the input is an image ``[B, C, H, W]`` and the output is
a sequence ``[B, N, D]`` with ``N = (H / patch_size) * (W / patch_size)``
tokens in row-major order (all patches of the first patch row first) and
``D = dim`` features on the last axis, which is what the sequence
operations such as ``linear`` act on. ``patch_size`` must divide both ``H``
and ``W``; a remainder is reported as ``E_CONSTRAINT`` rather than being
cropped away. Inference runs in both directions: ``dim`` and
``in_channels`` follow the neighbouring contracts, and a known token count
fixes the missing spatial extent when the other one is known. A token
count alone does not determine ``H`` and ``W`` and is reported as
ambiguous instead of guessed.

No positional embedding, class token, or normalization is added; compose
those explicitly. Behavior is identical in train and eval mode. The single
submodule is ``proj``, whose parameters are ``proj.weight`` of shape
``[dim, in_channels, patch_size, patch_size]`` and ``proj.bias``. The
projection runs entirely in the plan compute dtype.
        """
        ...
    @overload
    def permute(self, x: Symbol, dims: int | Sequence[int] = (), *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Reorder the non-batch axes, keeping the batch axis first.

        Returns ``x.permute(0, *dims)``. ``dims`` lists the non-batch axes of
the input in the order they take in the output, so ``out`` axis ``i``
carries ``x`` axis ``dims[i - 1]``: ``permute(3, 1, 2)`` turns
``[B, C, H, W]`` into ``[B, W, C, H]``. The batch axis is never listed and
always stays first.

``dims`` must be a permutation of ``1..rank-1``, which means its length
alone fixes the rank of both ports: ``permute(2, 1)`` is rank 3 and
``permute(2, 3, 1)`` is rank 4. A list that is not a permutation of that
range, one that includes the batch axis 0, or one whose length disagrees
with a rank the neighbouring operations already fixed, is reported as
``E_ARGUMENT``. Rank 4 is ``[B, C, H, W]``; rank 3 is ``[B, T, D]`` for
the sequence operations and ``[B, C, L]`` for 1-D convolution, and
``permute(2, 1)`` bridges those two readings.

The result is a view sharing storage and autograd history with the input,
generally non-contiguous; ``reshape`` and ``flatten`` copy when they must.
There are no parameters and no train/eval difference, and the stride
permutation is exact in every compute dtype.
        """
        ...
    @overload
    def permute(self, dims: int | Sequence[int] = (), *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Reorder the non-batch axes, keeping the batch axis first.

        Returns ``x.permute(0, *dims)``. ``dims`` lists the non-batch axes of
the input in the order they take in the output, so ``out`` axis ``i``
carries ``x`` axis ``dims[i - 1]``: ``permute(3, 1, 2)`` turns
``[B, C, H, W]`` into ``[B, W, C, H]``. The batch axis is never listed and
always stays first.

``dims`` must be a permutation of ``1..rank-1``, which means its length
alone fixes the rank of both ports: ``permute(2, 1)`` is rank 3 and
``permute(2, 3, 1)`` is rank 4. A list that is not a permutation of that
range, one that includes the batch axis 0, or one whose length disagrees
with a rank the neighbouring operations already fixed, is reported as
``E_ARGUMENT``. Rank 4 is ``[B, C, H, W]``; rank 3 is ``[B, T, D]`` for
the sequence operations and ``[B, C, L]`` for 1-D convolution, and
``permute(2, 1)`` bridges those two readings.

The result is a view sharing storage and autograd history with the input,
generally non-contiguous; ``reshape`` and ``flatten`` copy when they must.
There are no parameters and no train/eval difference, and the stride
permutation is exact in every compute dtype.
        """
        ...
    @overload
    def pixel_shuffle(self, x: Symbol, upscale_factor: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Trade channels for resolution: rearrange C*r^2 channels into an r-times larger image.

        Rearranges a ``[B, C*r^2, H, W]`` tensor into ``[B, C, H*r, W*r]``,
where ``r`` is ``upscale_factor``. Channel ``c*r^2 + i*r + j`` of the
input supplies output channel ``c`` at the sub-pixel offset ``(i, j)``
inside each ``r`` x ``r`` output block:

```text
out[b, c, h*r + i, w*r + j] == x[b, c*r^2 + i*r + j, h, w]
```

This is the standard sub-pixel convolution upsampler: a preceding
``conv`` produces ``r^2`` values per output pixel and the shuffle lays
them out spatially, which avoids the checkerboard artifacts of a strided
transposed convolution. It is a pure permutation and reshape, with no
parameters, no arithmetic, and identical train and eval behavior; the
output dtype is the input dtype.

Input channels must be divisible by ``upscale_factor**2`` and a known
output extent must be divisible by ``upscale_factor``; otherwise
``E_CONSTRAINT`` is reported. Resolution runs in both directions, so the
channel width of the producing layer can be left to the solver.
        """
        ...
    @overload
    def pixel_shuffle(self, upscale_factor: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Trade channels for resolution: rearrange C*r^2 channels into an r-times larger image.

        Rearranges a ``[B, C*r^2, H, W]`` tensor into ``[B, C, H*r, W*r]``,
where ``r`` is ``upscale_factor``. Channel ``c*r^2 + i*r + j`` of the
input supplies output channel ``c`` at the sub-pixel offset ``(i, j)``
inside each ``r`` x ``r`` output block:

```text
out[b, c, h*r + i, w*r + j] == x[b, c*r^2 + i*r + j, h, w]
```

This is the standard sub-pixel convolution upsampler: a preceding
``conv`` produces ``r^2`` values per output pixel and the shuffle lays
them out spatially, which avoids the checkerboard artifacts of a strided
transposed convolution. It is a pure permutation and reshape, with no
parameters, no arithmetic, and identical train and eval behavior; the
output dtype is the input dtype.

Input channels must be divisible by ``upscale_factor**2`` and a known
output extent must be divisible by ``upscale_factor``; otherwise
``E_CONSTRAINT`` is reported. Resolution runs in both directions, so the
channel width of the producing layer can be left to the solver.
        """
        ...
    @overload
    def pool_tokens(self, x: Symbol, mode: str = "mean", *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Reduce a [B, T, D] sequence to one [B, D] vector per example.

        Collapses the token axis of a rank-3 sequence ``[B, T, D]`` into a
single vector per example, giving ``[B, D]``. ``T`` is the position axis
and ``D`` the feature axis; the feature axis is carried through unchanged,
so the pooled width always equals the incoming width.

```text
mean   out[b, d] = (1 / T) * sum_t x[b, t, d]
first  out[b, d] = x[b, 0, d]
last   out[b, d] = x[b, T - 1, d]
max    out[b, d] = max_t x[b, t, d]
```

``mean`` averages every position, which is the default and the usual
choice for bidirectional encoders. ``first`` picks the leading token, the
``[CLS]`` convention. ``last`` picks the final token, the convention for
causal models whose sequences all end at the same position; it is the
wrong choice when the useful position varies per example. ``max`` takes
the elementwise maximum, resolving ties to the same value rather than to
a position.

There are no parameters and no state, so train and eval behave
identically. The reduction runs in the incoming dtype: with ``mean`` in
float16 a long sequence accumulates rounding error in the sum, which is
inherent to computing in the plan's compute dtype.
        """
        ...
    @overload
    def pool_tokens(self, mode: str = "mean", *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Reduce a [B, T, D] sequence to one [B, D] vector per example.

        Collapses the token axis of a rank-3 sequence ``[B, T, D]`` into a
single vector per example, giving ``[B, D]``. ``T`` is the position axis
and ``D`` the feature axis; the feature axis is carried through unchanged,
so the pooled width always equals the incoming width.

```text
mean   out[b, d] = (1 / T) * sum_t x[b, t, d]
first  out[b, d] = x[b, 0, d]
last   out[b, d] = x[b, T - 1, d]
max    out[b, d] = max_t x[b, t, d]
```

``mean`` averages every position, which is the default and the usual
choice for bidirectional encoders. ``first`` picks the leading token, the
``[CLS]`` convention. ``last`` picks the final token, the convention for
causal models whose sequences all end at the same position; it is the
wrong choice when the useful position varies per example. ``max`` takes
the elementwise maximum, resolving ties to the same value rather than to
a position.

There are no parameters and no state, so train and eval behave
identically. The reduction runs in the incoming dtype: with ``mean`` in
float16 a long sequence accumulates rounding error in the sum, which is
inherent to computing in the plan's compute dtype.
        """
        ...
    @overload
    def pos_embed(self, x: Symbol, max_len: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Add a learned position vector to every position of a sequence.

        A learned absolute position encoding, added to the sequence:

```text
out[b, t, :] = x[b, t, :] + weight[t, :]
```

``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features;
the same position vector is added to every example in the batch. The
parameter ``weight`` has shape ``[max_len, D]`` and is initialized from a
normal distribution with standard deviation 0.02, the convention for
transformer position tables. Only the first ``T`` rows take part in the
forward pass, so only those rows receive a gradient; the remaining rows
are kept so that the same module can encode longer sequences after a
re-resolve.

``max_len`` is a capacity, not a shape: the resolver only requires
``T <= max_len`` and reports ``E_CONSTRAINT`` when a resolved sequence is
longer than the table. Train and eval behave identically, and the addition
is performed in the input dtype.
        """
        ...
    @overload
    def pos_embed(self, max_len: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Add a learned position vector to every position of a sequence.

        A learned absolute position encoding, added to the sequence:

```text
out[b, t, :] = x[b, t, :] + weight[t, :]
```

``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features;
the same position vector is added to every example in the batch. The
parameter ``weight`` has shape ``[max_len, D]`` and is initialized from a
normal distribution with standard deviation 0.02, the convention for
transformer position tables. Only the first ``T`` rows take part in the
forward pass, so only those rows receive a gradient; the remaining rows
are kept so that the same module can encode longer sequences after a
re-resolve.

``max_len`` is a capacity, not a shape: the resolver only requires
``T <= max_len`` and reports ``E_CONSTRAINT`` when a resolved sequence is
longer than the table. Train and eval behave identically, and the addition
is performed in the input dtype.
        """
        ...
    @overload
    def pretrained(self, x: Symbol, source: str, *, output: str = "features", component: str = "", config: str = "", provider: str = "", sha256: str = "", layer: str = "", layers: str | Sequence[str] = (), readout: str = "", revision: str | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Load a pretrained network from disk or the Hugging Face Hub as one frozen node.

        Wraps a checkpoint whose architecture is defined by its own configuration.

The source's ``config.json`` selects the implementation: a ``model_type``
is built by ``transformers`` (``AutoModel``, the ``*ForCausalLM`` /
``*ForImageClassification`` heads for ``output="logits"``, or the
``<Model>VisionModel`` / ``<Model>TextModel`` tower named by
``component``), and a timm ``architecture`` is built by ``timm``. HNDL
does not describe these layouts; it traces the model on PyTorch's meta
device to learn the output shape, then loads the weights at build time.

Text models consume token ids: declare the graph input as ``[B, T]`` with
``input_dtype="int64"``. Vision models consume ``[B, C, H, W]`` at the
checkpoint's native resolution, which the relation fixes. The wrapped
model is frozen and kept in eval mode by default; pass ``trainable=True``
to fine-tune it, which also lets ``model.train()`` reach it. Weights are
loaded in float32 and cast to the plan dtype. Requires the optional
``hndl[pretrained]`` extra.

The plan records the resolved ``revision`` so a restored plan fails
(``E_CONSTRAINT``) if the source now points at a different checkpoint.

A local ``.pth`` state dict describes no architecture, so it names one the
host registered as trusted Python:
``registry.pretrained_provider("vgg16", build)`` binds a zero-argument
callable returning the ``nn.Module``, and configuration may only name an
already registered provider. Such a source is written
``pretrained("/path/weights.pth", provider="vgg16", sha256="<64 hex>",
layer="features.16")``. The ``sha256`` is required, is verified against the
file at resolution and again before loading, and the weights are read with
``torch.load(..., weights_only=True)``, which unpickles no objects; keys
must match exactly (``strict=True``). ``layer`` names a submodule by its
dotted ``named_modules()`` path and returns that submodule's output through
a forward hook, stopping the pass there; omitted, the node returns the
model's own output. A registered provider defaults to consuming floating-point
tensors, but may declare ``contract="ids"`` (rank 2 ``[B, T]`` ``int64`` token IDs)
to consume integer sequences, and may register an optional ``shape_fn`` callback
to resolve output shapes directly without meta-device tracing.

``layers=("layer1", "layer2", "layer3")`` returns several intermediates
from *one* forward pass. The node gains one output port per entry ---
``out0``, ``out1``, ``out2`` in the order written --- each carrying that
submodule's own output at its native shape, with no pooling and no
concatenation::

    f1, f2, f3 = pretrained("/path/weights.pth", provider="resnet18",
                            sha256="<64 hex>",
                            layers=("layer1", "layer2", "layer3"))

A hook on every requested submodule captures its output and the pass
unwinds once the last of them has run, so the unused tail of the network
never executes. Each captured tensor is cloned: an ``nn.ReLU(inplace=True)``
or a residual ``+=`` further along the pass would otherwise overwrite the
values the hook saw --- silently, or, where something saved that tensor for
backward, as a version-counter failure. The clone is differentiable, so
every output carries gradients to the graph input and supports second
derivatives. A submodule that runs more than once before the pass stops
fails with ``E_PRETRAINED`` naming it, rather than returning one of its
calls at random; name the block that contains it instead. A one-entry
``layers=("layer1",)`` is allowed and returns a one-tuple, which --- like
any multi-output call --- clears the current tensor, so the next operation
must name its input. Duplicate entries, an empty entry, and an unknown
submodule each fail with ``E_PRETRAINED``; ``layers=``, ``layer=`` and
``readout=`` are mutually exclusive, and like them ``layers=`` applies to
provider checkpoints only, never to transformers or timm sources.

``readout=`` names host code instead of a submodule, for checkpoints whose
useful tensor comes from a method rather than ``forward``. The host binds
named ``callable(model, x)`` readouts to the provider, and configuration may
only name one of them::

    registry.pretrained_provider("dinov2_vits14", build_dinov2, readouts={
        "patch_tokens": lambda m, x: m.forward_features(x)["x_norm_patchtokens"],
        "layers_2_5_8_11": lambda m, x: torch.cat(
            m.get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True), dim=1),
    })

which a network then selects with
``pretrained("/path/dinov2_vits14.pth", provider="dinov2_vits14",
sha256="<64 hex>", readout="patch_tokens")``. Readouts can also be added to
an existing provider with
``registry.pretrained_readout("dinov2_vits14", "cls_token", fn)``. A readout
must return exactly one tensor --- concatenate or stack several inside the
readout, or register one readout per tensor --- and it runs during
resolution on PyTorch's meta device, so it must be a pure function of
``(model, x)`` that touches no real data. ``readout=`` and ``layer=`` are
mutually exclusive, an unknown readout name fails with ``E_PRETRAINED``
listing the ones registered for that provider, and neither applies to
transformers or timm checkpoints, which select ``output=``.
        """
        ...
    @overload
    def pretrained(self, source: str, *, output: str = "features", component: str = "", config: str = "", provider: str = "", sha256: str = "", layer: str = "", layers: str | Sequence[str] = (), readout: str = "", revision: str | None = None, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Load a pretrained network from disk or the Hugging Face Hub as one frozen node.

        Wraps a checkpoint whose architecture is defined by its own configuration.

The source's ``config.json`` selects the implementation: a ``model_type``
is built by ``transformers`` (``AutoModel``, the ``*ForCausalLM`` /
``*ForImageClassification`` heads for ``output="logits"``, or the
``<Model>VisionModel`` / ``<Model>TextModel`` tower named by
``component``), and a timm ``architecture`` is built by ``timm``. HNDL
does not describe these layouts; it traces the model on PyTorch's meta
device to learn the output shape, then loads the weights at build time.

Text models consume token ids: declare the graph input as ``[B, T]`` with
``input_dtype="int64"``. Vision models consume ``[B, C, H, W]`` at the
checkpoint's native resolution, which the relation fixes. The wrapped
model is frozen and kept in eval mode by default; pass ``trainable=True``
to fine-tune it, which also lets ``model.train()`` reach it. Weights are
loaded in float32 and cast to the plan dtype. Requires the optional
``hndl[pretrained]`` extra.

The plan records the resolved ``revision`` so a restored plan fails
(``E_CONSTRAINT``) if the source now points at a different checkpoint.

A local ``.pth`` state dict describes no architecture, so it names one the
host registered as trusted Python:
``registry.pretrained_provider("vgg16", build)`` binds a zero-argument
callable returning the ``nn.Module``, and configuration may only name an
already registered provider. Such a source is written
``pretrained("/path/weights.pth", provider="vgg16", sha256="<64 hex>",
layer="features.16")``. The ``sha256`` is required, is verified against the
file at resolution and again before loading, and the weights are read with
``torch.load(..., weights_only=True)``, which unpickles no objects; keys
must match exactly (``strict=True``). ``layer`` names a submodule by its
dotted ``named_modules()`` path and returns that submodule's output through
a forward hook, stopping the pass there; omitted, the node returns the
model's own output. A registered provider defaults to consuming floating-point
tensors, but may declare ``contract="ids"`` (rank 2 ``[B, T]`` ``int64`` token IDs)
to consume integer sequences, and may register an optional ``shape_fn`` callback
to resolve output shapes directly without meta-device tracing.

``layers=("layer1", "layer2", "layer3")`` returns several intermediates
from *one* forward pass. The node gains one output port per entry ---
``out0``, ``out1``, ``out2`` in the order written --- each carrying that
submodule's own output at its native shape, with no pooling and no
concatenation::

    f1, f2, f3 = pretrained("/path/weights.pth", provider="resnet18",
                            sha256="<64 hex>",
                            layers=("layer1", "layer2", "layer3"))

A hook on every requested submodule captures its output and the pass
unwinds once the last of them has run, so the unused tail of the network
never executes. Each captured tensor is cloned: an ``nn.ReLU(inplace=True)``
or a residual ``+=`` further along the pass would otherwise overwrite the
values the hook saw --- silently, or, where something saved that tensor for
backward, as a version-counter failure. The clone is differentiable, so
every output carries gradients to the graph input and supports second
derivatives. A submodule that runs more than once before the pass stops
fails with ``E_PRETRAINED`` naming it, rather than returning one of its
calls at random; name the block that contains it instead. A one-entry
``layers=("layer1",)`` is allowed and returns a one-tuple, which --- like
any multi-output call --- clears the current tensor, so the next operation
must name its input. Duplicate entries, an empty entry, and an unknown
submodule each fail with ``E_PRETRAINED``; ``layers=``, ``layer=`` and
``readout=`` are mutually exclusive, and like them ``layers=`` applies to
provider checkpoints only, never to transformers or timm sources.

``readout=`` names host code instead of a submodule, for checkpoints whose
useful tensor comes from a method rather than ``forward``. The host binds
named ``callable(model, x)`` readouts to the provider, and configuration may
only name one of them::

    registry.pretrained_provider("dinov2_vits14", build_dinov2, readouts={
        "patch_tokens": lambda m, x: m.forward_features(x)["x_norm_patchtokens"],
        "layers_2_5_8_11": lambda m, x: torch.cat(
            m.get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True), dim=1),
    })

which a network then selects with
``pretrained("/path/dinov2_vits14.pth", provider="dinov2_vits14",
sha256="<64 hex>", readout="patch_tokens")``. Readouts can also be added to
an existing provider with
``registry.pretrained_readout("dinov2_vits14", "cls_token", fn)``. A readout
must return exactly one tensor --- concatenate or stack several inside the
readout, or register one readout per tensor --- and it runs during
resolution on PyTorch's meta device, so it must be a pure function of
``(model, x)`` that touches no real data. ``readout=`` and ``layer=`` are
mutually exclusive, an unknown readout name fails with ``E_PRETRAINED``
listing the ones registered for that provider, and neither applies to
transformers or timm checkpoints, which select ``output=``.
        """
        ...
    @overload
    def quick_gelu(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """CLIP's fast GELU approximation, x * sigmoid(1.702 * x).

        Elementwise ``out = x * sigmoid(1.702 * x)``.

This is the "quick GELU" used by OpenAI's CLIP and by the models derived
from it. The logistic curve approximates the Gaussian cumulative
distribution function, so the result tracks `gelu` within about 1e-2 in
absolute value while costing one sigmoid instead of an erf. A model
trained with this activation must keep it: it is close to, but not
numerically interchangeable with, `gelu`.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters and behaves
identically in train and eval mode. It is computed out of place in the
input dtype; the 1.702 coefficient is applied as a plain multiply, so
float16 and bfloat16 activations are never silently upcast.
        """
        ...
    @overload
    def quick_gelu(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """CLIP's fast GELU approximation, x * sigmoid(1.702 * x).

        Elementwise ``out = x * sigmoid(1.702 * x)``.

This is the "quick GELU" used by OpenAI's CLIP and by the models derived
from it. The logistic curve approximates the Gaussian cumulative
distribution function, so the result tracks `gelu` within about 1e-2 in
absolute value while costing one sigmoid instead of an erf. A model
trained with this activation must keep it: it is close to, but not
numerically interchangeable with, `gelu`.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters and behaves
identically in train and eval mode. It is computed out of place in the
input dtype; the 1.702 coefficient is applied as a plain multiply, so
float16 and bfloat16 activations are never silently upcast.
        """
        ...
    @overload
    def relu(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Rectified linear unit, max(x, 0).

        Out-of-place ReLU; shared branches are never mutated.
        """
        ...
    @overload
    def relu(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Rectified linear unit, max(x, 0).

        Out-of-place ReLU; shared branches are never mutated.
        """
        ...
    @overload
    def resblock(self, x: Symbol, out_channels: int | None = None, *, in_channels: int | None = None, stride: int = 1, norm: str = "batch_norm", groups: int = 8, kind: str = "basic", bottleneck_ratio: int = 4, activation: str = "relu", name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Residual basic block: two 3x3 convolutions with a normalized shortcut.

        The ResNet *basic block* on ``[B, C, H, W]`` images:

```text
y = norm2(conv2(relu(norm1(conv1(x)))))
out = relu(y + shortcut(x))
```

Both convolutions use a 3x3 kernel with padding 1; only ``conv1`` carries
the stride, so height and width map as
``H_out = floor((H_in + 2 - 3) / stride) + 1``, which equals
``ceil(H_in / stride)`` — stride 1 preserves the spatial extent and
stride 2 halves it (rounding up).

The shortcut is an ``nn.Identity`` when ``stride == 1`` and the channel
count is unchanged. Otherwise it is an ``nn.Sequential`` of a 1x1
convolution with the same stride followed by its own normalization, so
the residual sum is well defined. Submodules are named ``conv1``,
``norm1``, ``conv2``, ``norm2`` and ``shortcut``; parameter targets for
``init`` and ``trainable`` use those paths, for example
``init={"norm2.weight": 0}`` for a zero-initialized residual branch.

``norm="batch_norm"`` keeps running mean and variance buffers, so train
and eval mode differ: training normalizes with the statistics of the
current batch and updates the buffers, while evaluation uses the stored
running statistics. ``norm="group_norm"`` normalizes ``groups`` channel
groups per example and behaves identically in both modes; ``groups`` must
divide the channel count. ``norm="none"`` gives a plain residual
convolution block — no normalization at all — and in that case the
convolutions carry a learned bias, which the normalized variants omit
because the following normalization would cancel it.

Convolutions and normalizations run in the plan's compute dtype; batch
normalization accumulates its batch statistics in that dtype as PyTorch
does, so ``float16`` plans inherit the usual reduced-precision behavior.
        """
        ...
    @overload
    def resblock(self, out_channels: int | None = None, *, in_channels: int | None = None, stride: int = 1, norm: str = "batch_norm", groups: int = 8, kind: str = "basic", bottleneck_ratio: int = 4, activation: str = "relu", name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Residual basic block: two 3x3 convolutions with a normalized shortcut.

        The ResNet *basic block* on ``[B, C, H, W]`` images:

```text
y = norm2(conv2(relu(norm1(conv1(x)))))
out = relu(y + shortcut(x))
```

Both convolutions use a 3x3 kernel with padding 1; only ``conv1`` carries
the stride, so height and width map as
``H_out = floor((H_in + 2 - 3) / stride) + 1``, which equals
``ceil(H_in / stride)`` — stride 1 preserves the spatial extent and
stride 2 halves it (rounding up).

The shortcut is an ``nn.Identity`` when ``stride == 1`` and the channel
count is unchanged. Otherwise it is an ``nn.Sequential`` of a 1x1
convolution with the same stride followed by its own normalization, so
the residual sum is well defined. Submodules are named ``conv1``,
``norm1``, ``conv2``, ``norm2`` and ``shortcut``; parameter targets for
``init`` and ``trainable`` use those paths, for example
``init={"norm2.weight": 0}`` for a zero-initialized residual branch.

``norm="batch_norm"`` keeps running mean and variance buffers, so train
and eval mode differ: training normalizes with the statistics of the
current batch and updates the buffers, while evaluation uses the stored
running statistics. ``norm="group_norm"`` normalizes ``groups`` channel
groups per example and behaves identically in both modes; ``groups`` must
divide the channel count. ``norm="none"`` gives a plain residual
convolution block — no normalization at all — and in that case the
convolutions carry a learned bias, which the normalized variants omit
because the following normalization would cancel it.

Convolutions and normalizations run in the plan's compute dtype; batch
normalization accumulates its batch statistics in that dtype as PyTorch
does, so ``float16`` plans inherit the usual reduced-precision behavior.
        """
        ...
    @overload
    def reshape(self, x: Symbol, shape: int | Sequence[int] = (), *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """View the tensor with new non-batch dimensions, preserving the element count.

        Returns ``x.reshape(batch, *shape)``. The batch axis is never reshaped.
Give the leading dimensions positionally, ``reshape(512, 4, 4)``, or as
``shape=(512, 4, 4)``; exactly one omitted factor can be solved from the
element count. Three leading dimensions fix an image ``[B, C, H, W]``;
shorter prefixes take their rank from the neighbouring operations. The
plan records the full resolved shape.
        """
        ...
    @overload
    def reshape(self, shape: int | Sequence[int] = (), *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """View the tensor with new non-batch dimensions, preserving the element count.

        Returns ``x.reshape(batch, *shape)``. The batch axis is never reshaped.
Give the leading dimensions positionally, ``reshape(512, 4, 4)``, or as
``shape=(512, 4, 4)``; exactly one omitted factor can be solved from the
element count. Three leading dimensions fix an image ``[B, C, H, W]``;
shorter prefixes take their rank from the neighbouring operations. The
plan records the full resolved shape.
        """
        ...
    @overload
    def rms_norm(self, x: Symbol, *, eps: float = 1e-06, affine: bool = True, zero_centered: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Scale the last axis by its root-mean-square, with a learned per-feature gain.

        Root-mean-square normalization of the last axis ``D``: the cheaper
half of `layer_norm` that rescales without re-centering.

```text
out = x / sqrt(mean(x^2, dim=-1) + eps) * weight
```

The statistics are taken over the last axis only, so a ``[B, D]`` input
normalizes each example and a ``[B, T, D]`` sequence normalizes each
position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
relation but would be normalized over ``W`` alone, which is almost never
what is wanted; reach for `group_norm` on images. The examples below
therefore cover only ranks 2 and 3.

The mean square and its reciprocal square root are accumulated in
``float32`` even when the plan's compute dtype is ``float16`` or
``bfloat16``, because squaring a half-precision activation overflows
around 256; the normalized tensor is cast back to the input dtype before
the gain is applied, so the layer's inputs, outputs and parameters stay
in the plan dtype.

The single parameter is ``weight`` of shape ``[D]``, initialized to ones
(or zeros when ``zero_centered=True``) and present only when ``affine`` is
true. There are no running statistics, so train and eval behave identically.
        """
        ...
    @overload
    def rms_norm(self, *, eps: float = 1e-06, affine: bool = True, zero_centered: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Scale the last axis by its root-mean-square, with a learned per-feature gain.

        Root-mean-square normalization of the last axis ``D``: the cheaper
half of `layer_norm` that rescales without re-centering.

```text
out = x / sqrt(mean(x^2, dim=-1) + eps) * weight
```

The statistics are taken over the last axis only, so a ``[B, D]`` input
normalizes each example and a ``[B, T, D]`` sequence normalizes each
position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
relation but would be normalized over ``W`` alone, which is almost never
what is wanted; reach for `group_norm` on images. The examples below
therefore cover only ranks 2 and 3.

The mean square and its reciprocal square root are accumulated in
``float32`` even when the plan's compute dtype is ``float16`` or
``bfloat16``, because squaring a half-precision activation overflows
around 256; the normalized tensor is cast back to the input dtype before
the gain is applied, so the layer's inputs, outputs and parameters stay
in the plan dtype.

The single parameter is ``weight`` of shape ``[D]``, initialized to ones
(or zeros when ``zero_centered=True``) and present only when ``affine`` is
true. There are no running statistics, so train and eval behave identically.
        """
        ...
    @overload
    def rope(self, x: Symbol, heads: int = 1, *, theta: float = 10000.0, rope_scaling: str = "none", rope_scale: float = 1.0, spatial_shape: int | tuple[int, int] = (0, 0), name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Standalone rotary position embedding over a [B, T, D] or [B, H, T, D_h] sequence.

        Standalone rotary position embedding (RoPE) for [B, T, D] or [B, H, T, D_h] sequences.

Applies rotary position embeddings independently to each head of dimension
``head_dim = D // heads`` along the sequence length axis:
``out = x * cos + rotate_half(x) * sin``.
        """
        ...
    @overload
    def rope(self, heads: int = 1, *, theta: float = 10000.0, rope_scaling: str = "none", rope_scale: float = 1.0, spatial_shape: int | tuple[int, int] = (0, 0), name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Standalone rotary position embedding over a [B, T, D] or [B, H, T, D_h] sequence.

        Standalone rotary position embedding (RoPE) for [B, T, D] or [B, H, T, D_h] sequences.

Applies rotary position embeddings independently to each head of dimension
``head_dim = D // heads`` along the sequence length axis:
``out = x * cos + rotate_half(x) * sin``.
        """
        ...
    @overload
    def scale(self, x: Symbol, factor: float, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multiply a tensor by a fixed scalar.

        ``out = factor * x``: a shape-, rank-, and dtype-preserving constant
gain with no parameters and no buffers. ``factor`` is a plan constant, not
a learned value, so the gradient is ``factor * grad_out`` and the behavior
is identical in train and eval mode. The multiplication happens in the
input dtype; a large ``factor`` can overflow float16, so keep the product
inside the representable range for the plan's compute dtype.
        """
        ...
    @overload
    def scale(self, factor: float, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Multiply a tensor by a fixed scalar.

        ``out = factor * x``: a shape-, rank-, and dtype-preserving constant
gain with no parameters and no buffers. ``factor`` is a plan constant, not
a learned value, so the gradient is ``factor * grad_out`` and the behavior
is identical in train and eval mode. The multiplication happens in the
input dtype; a large ``factor`` can overflow float16, so keep the product
inside the representable range for the plan's compute dtype.
        """
        ...
    @overload
    def sigmoid(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Logistic sigmoid, squashing values into (0, 1).

        Elementwise ``out = 1 / (1 + exp(-x))``.

The output lies strictly in ``(0, 1)``, which makes the operator a natural
final activation for probabilities and for images normalized to the unit
interval. Gradients vanish for inputs far from zero, so it is a poor
choice for hidden layers; prefer `silu` or `gelu` there. When the loss is
a binary cross entropy, keep the logits and use a fused loss rather than
stacking `sigmoid` in front of it.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.
        """
        ...
    @overload
    def sigmoid(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Logistic sigmoid, squashing values into (0, 1).

        Elementwise ``out = 1 / (1 + exp(-x))``.

The output lies strictly in ``(0, 1)``, which makes the operator a natural
final activation for probabilities and for images normalized to the unit
interval. Gradients vanish for inputs far from zero, so it is a poor
choice for hidden layers; prefer `silu` or `gelu` there. When the loss is
a binary cross entropy, keep the logits and use a fused loss rather than
stacking `sigmoid` in front of it.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.
        """
        ...
    @overload
    def silu(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Sigmoid linear unit (swish), x * sigmoid(x).

        Elementwise ``out = x * sigmoid(x)``, also known as swish.

Unlike `relu` the function is smooth everywhere and keeps a small negative
response, with a minimum of about ``-0.278`` near ``x = -1.278``; unlike
`sigmoid` it is unbounded above, so it does not saturate for large
positive inputs.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.
        """
        ...
    @overload
    def silu(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Sigmoid linear unit (swish), x * sigmoid(x).

        Elementwise ``out = x * sigmoid(x)``, also known as swish.

Unlike `relu` the function is smooth everywhere and keeps a small negative
response, with a minimum of about ``-0.278`` near ``x = -1.278``; unlike
`sigmoid` it is unbounded above, so it does not saturate for large
positive inputs.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.
        """
        ...
    @overload
    def sinusoidal_embed(self, x: Symbol, features: int | None = None, *, dim: int = 0, max_period: float = 10000.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Deterministic log-spaced sinusoidal timestep embeddings.

        Deterministic log-spaced sinusoidal timestep and coordinate embedding.

Computes frequency bands geometrically spaced from 1 down to 1 / max_period,
mapping inputs into interleaved or concatenated sine and cosine components:
``angles = x * freqs``
``out = [sin(angles), cos(angles)]``.
        """
        ...
    @overload
    def sinusoidal_embed(self, features: int | None = None, *, dim: int = 0, max_period: float = 10000.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Deterministic log-spaced sinusoidal timestep embeddings.

        Deterministic log-spaced sinusoidal timestep and coordinate embedding.

Computes frequency bands geometrically spaced from 1 down to 1 / max_period,
mapping inputs into interleaved or concatenated sine and cosine components:
``angles = x * freqs``
``out = [sin(angles), cos(angles)]``.
        """
        ...
    @overload
    def snake(self, x: Symbol, channels: int = 0, *, dim: int = 0, alpha: float = 1.0, eps: float = 1e-09, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Periodic activation: x + (1 / (alpha + eps)) * sin^2(alpha * x).

        Periodic activation function for neural audio and inductive bias.
        """
        ...
    @overload
    def snake(self, channels: int = 0, *, dim: int = 0, alpha: float = 1.0, eps: float = 1e-09, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Periodic activation: x + (1 / (alpha + eps)) * sin^2(alpha * x).

        Periodic activation function for neural audio and inductive bias.
        """
        ...
    @overload
    def softmax(self, x: Symbol, dim: int = -1, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize one non-batch axis into a probability distribution.

        Softmax along one axis:

```
out[..., i, ...] = exp(x[..., i, ...] - m) / sum_j exp(x[..., j, ...] - m)
```

where `m` is the maximum over the axis selected by `dim` and the sum runs
over that same axis (subtracting the maximum is what `torch.softmax` does
internally for numerical stability). Every slice along the axis sums to 1
and every element lies in (0, 1). The shape is preserved.

`dim` indexes the full tensor, batch axis included, following the usual
PyTorch convention: `-1` is the last axis, `1` is the channel axis `C` of a
`[B, C, H, W]` image or the position axis `T` of a `[B, T, D]` sequence,
and `2` is `H`. Normalizing across the batch axis would make examples in a
batch depend on each other, so a `dim` that resolves to axis 0, or that
lies outside the rank of the incoming tensor, is rejected with
`E_ARGUMENT` while the plan resolves.

The computation runs in the activation dtype, including float16 and
bfloat16. Behavior is identical in train and eval mode, and the operator
has no parameters.
        """
        ...
    @overload
    def softmax(self, dim: int = -1, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Normalize one non-batch axis into a probability distribution.

        Softmax along one axis:

```
out[..., i, ...] = exp(x[..., i, ...] - m) / sum_j exp(x[..., j, ...] - m)
```

where `m` is the maximum over the axis selected by `dim` and the sum runs
over that same axis (subtracting the maximum is what `torch.softmax` does
internally for numerical stability). Every slice along the axis sums to 1
and every element lies in (0, 1). The shape is preserved.

`dim` indexes the full tensor, batch axis included, following the usual
PyTorch convention: `-1` is the last axis, `1` is the channel axis `C` of a
`[B, C, H, W]` image or the position axis `T` of a `[B, T, D]` sequence,
and `2` is `H`. Normalizing across the batch axis would make examples in a
batch depend on each other, so a `dim` that resolves to axis 0, or that
lies outside the rank of the incoming tensor, is rejected with
`E_ARGUMENT` while the plan resolves.

The computation runs in the activation dtype, including float16 and
bfloat16. Behavior is identical in train and eval mode, and the operator
has no parameters.
        """
        ...
    @overload
    def softplus(self, x: Symbol, beta: float = 1.0, *, threshold: float = 20.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Smooth positive activation, log(1 + exp(beta*x)) / beta.

        Elementwise smooth approximation of ReLU:

```
out = log(1 + exp(beta * x)) / beta
```

The output is strictly positive and the function is differentiable
everywhere; its derivative is the logistic sigmoid `sigmoid(beta * x)`.
Larger `beta` sharpens the bend at the origin and the limit is ReLU.

For numerical stability the linear branch `out = x` is used wherever
`beta * x > threshold`, which is exact in floating point well before the
default threshold of 20. The rule applies unchanged in float16 and
bfloat16, where the identity branch avoids overflowing `exp`.

The operator is elementwise, so it preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — and it has no
parameters and no train/eval difference.
        """
        ...
    @overload
    def softplus(self, beta: float = 1.0, *, threshold: float = 20.0, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Smooth positive activation, log(1 + exp(beta*x)) / beta.

        Elementwise smooth approximation of ReLU:

```
out = log(1 + exp(beta * x)) / beta
```

The output is strictly positive and the function is differentiable
everywhere; its derivative is the logistic sigmoid `sigmoid(beta * x)`.
Larger `beta` sharpens the bend at the origin and the limit is ReLU.

For numerical stability the linear branch `out = x` is used wherever
`beta * x > threshold`, which is exact in floating point well before the
default threshold of 20. The rule applies unchanged in float16 and
bfloat16, where the identity branch avoids overflowing `exp`.

The operator is elementwise, so it preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — and it has no
parameters and no train/eval difference.
        """
        ...
    @overload
    def spatial_attention(self, x: Symbol, reduction: int = 8, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """SAGAN self-attention over every position of a [B, C, H, W] feature map.

        The Self-Attention GAN block on a ``[B, C, H, W]`` feature map: every one
of the ``N = H*W`` positions attends to every other, so the layer sees the
whole map where a 3x3 convolution sees a neighbourhood.

```text
f, g    = f_proj(x), g_proj(x)              # 1x1 convs, C -> C/reduction
h       = h_proj(x)                         # 1x1 conv,  C -> C
f, g, h -> [B, C', N], [B, C', N], [B, C, N]     # N = H*W
energy[b, i, j] = sum_c f[b, c, i] * g[b, c, j]  # [B, N, N]
beta    = softmax(energy, dim=-1)           # over the key positions j
o[b, c, i] = sum_j h[b, c, j] * beta[b, i, j]    # [B, C, N] -> [B, C, H, W]
out     = gamma * o + x
```

``reduction`` must divide ``C``; the resolver reports ``E_CONSTRAINT`` when
it does not. Submodules are ``f_proj`` (query), ``g_proj`` (key) and
``h_proj`` (value), each an ``nn.Conv2d(C, ·, kernel_size=1)`` with a bias,
so their parameters are ``f_proj.weight``, ``f_proj.bias`` and so on.
``f_proj`` and ``g_proj`` project to ``C // reduction`` channels while
``h_proj`` keeps all ``C``. Nothing else is learned: the softmax, the two
contractions and the reshapes carry no state and there are no buffers.

The remaining parameter is ``gamma``, a single learned scalar of shape
``[1]`` that gates the attention branch, exactly as `learned_scale` does.
It starts at ``0.0``, so the block begins as the identity — ``out == x`` on
the first step — and the network learns how much attention to admit. A
gamma of zero also zeroes the gradient reaching the three projections;
they start learning through ``gamma`` itself, whose gradient is the inner
product of the attention branch with the incoming gradient. Construction
overrides target it by name: ``spatial_attention(init={"gamma": 1.0})``
opens the gate and ``spatial_attention(trainable={"gamma": False})``
freezes it.

Shape, rank and dtype are preserved and train and eval behave identically.
Cost grows with ``N**2 = (H*W)**2``: the attention map is ``[B, N, N]``,
which is 256x256 per example on a 16x16 map and 4096x4096 on a 64x64 one,
so the block is normally placed at a middle resolution. ``H`` and ``W`` are
not inferable backward — every spatial extent is accepted — so they come
from the network input or the preceding operation, while ``C`` flows in
both directions.

Zhang, Goodfellow, Metaxas & Odena, "Self-Attention Generative Adversarial
Networks" (ICML 2019, arXiv 2018), section 3: f and g project to
``C/8`` channels, h keeps ``C``, the attention map is
``softmax(f(x)^T g(x))`` over the key positions, and the output is
``y = gamma * o + x`` with gamma initialized to 0.
        """
        ...
    @overload
    def spatial_attention(self, reduction: int = 8, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """SAGAN self-attention over every position of a [B, C, H, W] feature map.

        The Self-Attention GAN block on a ``[B, C, H, W]`` feature map: every one
of the ``N = H*W`` positions attends to every other, so the layer sees the
whole map where a 3x3 convolution sees a neighbourhood.

```text
f, g    = f_proj(x), g_proj(x)              # 1x1 convs, C -> C/reduction
h       = h_proj(x)                         # 1x1 conv,  C -> C
f, g, h -> [B, C', N], [B, C', N], [B, C, N]     # N = H*W
energy[b, i, j] = sum_c f[b, c, i] * g[b, c, j]  # [B, N, N]
beta    = softmax(energy, dim=-1)           # over the key positions j
o[b, c, i] = sum_j h[b, c, j] * beta[b, i, j]    # [B, C, N] -> [B, C, H, W]
out     = gamma * o + x
```

``reduction`` must divide ``C``; the resolver reports ``E_CONSTRAINT`` when
it does not. Submodules are ``f_proj`` (query), ``g_proj`` (key) and
``h_proj`` (value), each an ``nn.Conv2d(C, ·, kernel_size=1)`` with a bias,
so their parameters are ``f_proj.weight``, ``f_proj.bias`` and so on.
``f_proj`` and ``g_proj`` project to ``C // reduction`` channels while
``h_proj`` keeps all ``C``. Nothing else is learned: the softmax, the two
contractions and the reshapes carry no state and there are no buffers.

The remaining parameter is ``gamma``, a single learned scalar of shape
``[1]`` that gates the attention branch, exactly as `learned_scale` does.
It starts at ``0.0``, so the block begins as the identity — ``out == x`` on
the first step — and the network learns how much attention to admit. A
gamma of zero also zeroes the gradient reaching the three projections;
they start learning through ``gamma`` itself, whose gradient is the inner
product of the attention branch with the incoming gradient. Construction
overrides target it by name: ``spatial_attention(init={"gamma": 1.0})``
opens the gate and ``spatial_attention(trainable={"gamma": False})``
freezes it.

Shape, rank and dtype are preserved and train and eval behave identically.
Cost grows with ``N**2 = (H*W)**2``: the attention map is ``[B, N, N]``,
which is 256x256 per example on a 16x16 map and 4096x4096 on a 64x64 one,
so the block is normally placed at a middle resolution. ``H`` and ``W`` are
not inferable backward — every spatial extent is accepted — so they come
from the network input or the preceding operation, while ``C`` flows in
both directions.

Zhang, Goodfellow, Metaxas & Odena, "Self-Attention Generative Adversarial
Networks" (ICML 2019, arXiv 2018), section 3: f and g project to
``C/8`` channels, h keeps ``C``, the attention map is
``softmax(f(x)^T g(x))`` over the key positions, and the output is
``y = gamma * o + x`` with gamma initialized to 0.
        """
        ...
    @overload
    def split(self, x: Symbol, size: int | None = None, *, dim: int = 1, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Cut one axis into a first section and the remainder.

        Returns ``(x[:size], x[size:])`` along ``dim`` as views sharing autograd
with the input. Exactly two non-empty sections are produced; this is not
a repeated chunking. After ``split`` there is no single current tensor, so
the next operation must name its input.
        """
        ...
    @overload
    def split(self, size: int | None = None, *, dim: int = 1, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> tuple[Symbol, ...]:
        """Cut one axis into a first section and the remainder.

        Returns ``(x[:size], x[size:])`` along ``dim`` as views sharing autograd
with the input. Exactly two non-empty sections are produced; this is not
a repeated chunking. After ``split`` there is no single current tensor, so
the next operation must name its input.
        """
        ...
    def sub(self, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Elementwise difference of two tensors with identical shapes.

        ``out = a - b`` with no broadcasting; the two operands must agree on
rank, every axis extent, and dtype, and the difference is computed in the
plan compute dtype. Order matters, so both tensor inputs must be supplied
explicitly: ``sub(h, saved)`` is ``h - saved``.
        """
        ...
    @overload
    def sum(self, x: Symbol, dim: int, *, keepdim: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Add up one non-batch axis.

        Sum over exactly one non-batch axis:
``out = x.sum(dim=dim, keepdim=keepdim)``.

With ``keepdim=False`` (the default) the output rank is one lower than the
input rank and the axes after ``dim`` shift down by one. With
``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
``[B, T, D]`` (``sum(1)`` adds up positions, ``sum(2)`` adds up features),
rank 4 is ``[B, C, H, W]`` (``sum(1)`` over channels, ``sum(2)`` over
height, ``sum(3)`` over width), and a 1-D convolution layout ``[B, C, L]``
reduces its length with ``sum(2)``.

Because HNDL tensors always keep a batch axis and at least one further
axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
``E_CONSTRAINT``; use ``keepdim=True`` instead.

The layer has no parameters and behaves identically in train and eval
mode. The reduction accumulates in the incoming dtype and is not upcast,
so a long axis in ``float16`` can lose precision or overflow; scale the
inputs, use ``mean``, or run the plan in ``bfloat16``/``float32`` when
that matters.

Shape inference is bidirectional: a known output fixes every non-reduced
input axis, but the extent of the reduced axis itself carries no
information backward and must come from the input side.
        """
        ...
    @overload
    def sum(self, dim: int, *, keepdim: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Add up one non-batch axis.

        Sum over exactly one non-batch axis:
``out = x.sum(dim=dim, keepdim=keepdim)``.

With ``keepdim=False`` (the default) the output rank is one lower than the
input rank and the axes after ``dim`` shift down by one. With
``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
``[B, T, D]`` (``sum(1)`` adds up positions, ``sum(2)`` adds up features),
rank 4 is ``[B, C, H, W]`` (``sum(1)`` over channels, ``sum(2)`` over
height, ``sum(3)`` over width), and a 1-D convolution layout ``[B, C, L]``
reduces its length with ``sum(2)``.

Because HNDL tensors always keep a batch axis and at least one further
axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
``E_CONSTRAINT``; use ``keepdim=True`` instead.

The layer has no parameters and behaves identically in train and eval
mode. The reduction accumulates in the incoming dtype and is not upcast,
so a long axis in ``float16`` can lose precision or overflow; scale the
inputs, use ``mean``, or run the plan in ``bfloat16``/``float32`` when
that matters.

Shape inference is bidirectional: a known output fixes every non-reduced
input axis, but the extent of the reduced axis itself carries no
information backward and must come from the input side.
        """
        ...
    @overload
    def swiglu(self, x: Symbol, hidden: int, *, bias: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """SwiGLU feed-forward block: a SiLU-gated projection folded back to the input width.

        The gated feed-forward block used by PaLM- and LLaMA-style transformers.
Two projections read the same input: one is passed through SiLU and gates
the other elementwise, and a third projection folds the product back to the
input width.

```text
out = down(silu(gate(x)) * up(x))
silu(v) = v * sigmoid(v)
```

The last axis carries the features `D`; every leading axis is a batch or
position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
applied independently at each position. Image tensors `[B, C, H, W]` are
rejected with `E_CONSTRAINT`; flatten or reshape them first.

Submodules are `gate` and `up` (both `Linear(D, hidden)`) and `down`
(`Linear(hidden, D)`). Without biases the block holds `3 * D * hidden`
parameters; with `bias=True` it holds `3 * D * hidden + 2 * hidden + D`.
Because a gated block spends three matrices where a plain `feed_forward`
spends two, `hidden` is commonly set to about two thirds of the width a
plain block would use, for equal parameter count.

There is no dropout and no normalization here, and no running state, so
train and eval mode behave identically. Everything is computed in the
incoming dtype, with no upcasting.
        """
        ...
    @overload
    def swiglu(self, hidden: int, *, bias: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """SwiGLU feed-forward block: a SiLU-gated projection folded back to the input width.

        The gated feed-forward block used by PaLM- and LLaMA-style transformers.
Two projections read the same input: one is passed through SiLU and gates
the other elementwise, and a third projection folds the product back to the
input width.

```text
out = down(silu(gate(x)) * up(x))
silu(v) = v * sigmoid(v)
```

The last axis carries the features `D`; every leading axis is a batch or
position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
applied independently at each position. Image tensors `[B, C, H, W]` are
rejected with `E_CONSTRAINT`; flatten or reshape them first.

Submodules are `gate` and `up` (both `Linear(D, hidden)`) and `down`
(`Linear(hidden, D)`). Without biases the block holds `3 * D * hidden`
parameters; with `bias=True` it holds `3 * D * hidden + 2 * hidden + D`.
Because a gated block spends three matrices where a plain `feed_forward`
spends two, `hidden` is commonly set to about two thirds of the width a
plain block would use, for equal parameter count.

There is no dropout and no normalization here, and no running state, so
train and eval mode behave identically. Everything is computed in the
incoming dtype, with no upcasting.
        """
        ...
    @overload
    def tanh(self, x: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Hyperbolic tangent, squashing values into (-1, 1).

        Elementwise ``tanh(x)``.
        """
        ...
    @overload
    def tanh(self, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Hyperbolic tangent, squashing values into (-1, 1).

        Elementwise ``tanh(x)``.
        """
        ...
    @overload
    def transformer_block(self, x: Symbol, heads: int, *, mlp_ratio: int = 4, activation: str = "gelu", norm: str = "layer_norm", causal: bool = False, dropout: float = 0.0, layer_scale: bool = False, layer_scale_init: float = 1e-05, eps: float = 1e-05, ffn_kind: str = "mlp", hidden: int = 0, rope: bool = False, rope_theta: float = 10000.0, kv_heads: int = 0, qk_norm: str = "none", bias: bool = True, head_dim: int = 0, window_size: int = 0, logit_softcap: float = 0.0, qkv_bias: bool = True, out_bias: bool = True, zero_centered: bool = False, post_norm: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Pre-norm transformer block: residual self-attention followed by a residual feed-forward.

        One pre-norm transformer block on a ``[B, T, D]`` sequence: ``B``
examples of ``T`` positions with ``D`` features each. Both branches are
residual and both normalize their input rather than their output:

```text
x   = x + ls1 * attn(norm1(x))      # multi-head self-attention
out = x + ls2 * ffn(norm2(x))       # position-wise feed-forward
```

``ls1`` and ``ls2`` are the layer-scale gains, and are absent (a plain sum)
unless ``layer_scale`` is set. The block preserves its shape exactly, so
blocks stack by repetition and ``D`` flows through untouched in both
directions of the resolver.

**Submodules and parameter names.** The four children are ``norm1``,
``attn``, ``norm2`` and ``ffn``, so the state dict holds

| Path | Shape |
| --- | --- |
| `norm1.weight`, `norm1.bias` | `[D]` (no `bias` with `rms_norm`) |
| `attn.{q,k,v,o}_proj.weight` | `[D, D]` |
| `attn.{q,k,v,o}_proj.bias` | `[D]` |
| `norm2.weight`, `norm2.bias` | `[D]` (no `bias` with `rms_norm`) |
| `ffn.up.weight`, `ffn.up.bias` | `[hidden, D]`, `[hidden]` |
| `ffn.down.weight`, `ffn.down.bias` | `[D, hidden]`, `[D]` |
| `gamma1`, `gamma2` | `[D]`, only when `layer_scale` |

with ``hidden = mlp_ratio * D``. Those paths are what ``init`` and
``trainable`` target, for example ``init={"gamma1": 0.1}``.

**Attention.** `attention` with all four projections biased and no rotary
embeddings. ``heads`` must divide ``D``; the resolver reports
``E_CONSTRAINT`` when it does not. With ``causal=True`` position ``t``
attends only to positions ``<= t``, which makes a stack of blocks
autoregressive; with ``causal=False`` every position sees every other and
the block is a bidirectional encoder. There is no padding mask.

**Feed-forward.** `feed_forward` of inner width ``mlp_ratio * D``, so the
default ``mlp_ratio=4`` gives the classic ``4 * D`` transformer MLP.
``activation`` selects ``gelu``, ``gelu_tanh``, ``relu``, ``silu`` or
``quick_gelu``.

**Normalization.** ``norm="layer_norm"`` re-centers and rescales each
position; ``norm="rms_norm"`` only rescales and has no bias. Both use
``eps`` and both are affine. ``rms_norm`` accumulates its mean square in
float32 before casting back, as that operator documents.

**Layer scale.** With ``layer_scale=True`` each branch is multiplied by a
learned vector of width ``D`` — ``gamma1`` for attention and ``gamma2``
for the feed-forward — both filled with ``layer_scale_init``. A small
initial value (the default ``1e-5``) starts the block at almost exactly
the identity, which is what makes very deep stacks trainable.

**Correspondences.**

| Recipe | Arguments |
| --- | --- |
| GPT-2 | `transformer_block(heads, activation="gelu_tanh", causal=True)` |
| ViT / BERT-style encoder | `transformer_block(heads)` |
| LLaMA-style decoder | `transformer_block(heads, norm="rms_norm", causal=True)` |
| CaiT / DINOv2 | `transformer_block(heads, layer_scale=True)` |

(LLaMA-style architectures can use `ffn_kind="swiglu"` and `rope=True`
directly within `transformer_block`, or compose `attention(rope=True)` and
`swiglu` by hand.)

**Train and eval.** ``dropout`` is passed to both branches — the attention
weights and the feed-forward hidden activations — and is active in
training mode only. With ``dropout=0`` the block is deterministic in both
modes; there are no running statistics either way. Everything is computed
in the plan's compute dtype, apart from the float32 accumulation inside
``rms_norm``.
        """
        ...
    @overload
    def transformer_block(self, heads: int, *, mlp_ratio: int = 4, activation: str = "gelu", norm: str = "layer_norm", causal: bool = False, dropout: float = 0.0, layer_scale: bool = False, layer_scale_init: float = 1e-05, eps: float = 1e-05, ffn_kind: str = "mlp", hidden: int = 0, rope: bool = False, rope_theta: float = 10000.0, kv_heads: int = 0, qk_norm: str = "none", bias: bool = True, head_dim: int = 0, window_size: int = 0, logit_softcap: float = 0.0, qkv_bias: bool = True, out_bias: bool = True, zero_centered: bool = False, post_norm: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Pre-norm transformer block: residual self-attention followed by a residual feed-forward.

        One pre-norm transformer block on a ``[B, T, D]`` sequence: ``B``
examples of ``T`` positions with ``D`` features each. Both branches are
residual and both normalize their input rather than their output:

```text
x   = x + ls1 * attn(norm1(x))      # multi-head self-attention
out = x + ls2 * ffn(norm2(x))       # position-wise feed-forward
```

``ls1`` and ``ls2`` are the layer-scale gains, and are absent (a plain sum)
unless ``layer_scale`` is set. The block preserves its shape exactly, so
blocks stack by repetition and ``D`` flows through untouched in both
directions of the resolver.

**Submodules and parameter names.** The four children are ``norm1``,
``attn``, ``norm2`` and ``ffn``, so the state dict holds

| Path | Shape |
| --- | --- |
| `norm1.weight`, `norm1.bias` | `[D]` (no `bias` with `rms_norm`) |
| `attn.{q,k,v,o}_proj.weight` | `[D, D]` |
| `attn.{q,k,v,o}_proj.bias` | `[D]` |
| `norm2.weight`, `norm2.bias` | `[D]` (no `bias` with `rms_norm`) |
| `ffn.up.weight`, `ffn.up.bias` | `[hidden, D]`, `[hidden]` |
| `ffn.down.weight`, `ffn.down.bias` | `[D, hidden]`, `[D]` |
| `gamma1`, `gamma2` | `[D]`, only when `layer_scale` |

with ``hidden = mlp_ratio * D``. Those paths are what ``init`` and
``trainable`` target, for example ``init={"gamma1": 0.1}``.

**Attention.** `attention` with all four projections biased and no rotary
embeddings. ``heads`` must divide ``D``; the resolver reports
``E_CONSTRAINT`` when it does not. With ``causal=True`` position ``t``
attends only to positions ``<= t``, which makes a stack of blocks
autoregressive; with ``causal=False`` every position sees every other and
the block is a bidirectional encoder. There is no padding mask.

**Feed-forward.** `feed_forward` of inner width ``mlp_ratio * D``, so the
default ``mlp_ratio=4`` gives the classic ``4 * D`` transformer MLP.
``activation`` selects ``gelu``, ``gelu_tanh``, ``relu``, ``silu`` or
``quick_gelu``.

**Normalization.** ``norm="layer_norm"`` re-centers and rescales each
position; ``norm="rms_norm"`` only rescales and has no bias. Both use
``eps`` and both are affine. ``rms_norm`` accumulates its mean square in
float32 before casting back, as that operator documents.

**Layer scale.** With ``layer_scale=True`` each branch is multiplied by a
learned vector of width ``D`` — ``gamma1`` for attention and ``gamma2``
for the feed-forward — both filled with ``layer_scale_init``. A small
initial value (the default ``1e-5``) starts the block at almost exactly
the identity, which is what makes very deep stacks trainable.

**Correspondences.**

| Recipe | Arguments |
| --- | --- |
| GPT-2 | `transformer_block(heads, activation="gelu_tanh", causal=True)` |
| ViT / BERT-style encoder | `transformer_block(heads)` |
| LLaMA-style decoder | `transformer_block(heads, norm="rms_norm", causal=True)` |
| CaiT / DINOv2 | `transformer_block(heads, layer_scale=True)` |

(LLaMA-style architectures can use `ffn_kind="swiglu"` and `rope=True`
directly within `transformer_block`, or compose `attention(rope=True)` and
`swiglu` by hand.)

**Train and eval.** ``dropout`` is passed to both branches — the attention
weights and the feed-forward hidden activations — and is active in
training mode only. With ``dropout=0`` the block is deterministic in both
modes; there are no running statistics either way. Everything is computed
in the plan's compute dtype, apart from the float32 accumulation inside
``rms_norm``.
        """
        ...
    @overload
    def transpose(self, x: Symbol, dim0: int, dim1: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Swap two non-batch axes of the tensor.

        Returns ``x.transpose(dim0, dim1)``: the extents at ``dim0`` and
``dim1`` trade places and every other axis keeps its extent. The batch
axis 0 is never part of a swap, so both arguments are at least 1 and at
most 3 (the last axis of the deepest supported rank). Passing the same
axis twice is the identity.

Axis conventions. A rank-3 tensor is ``[B, T, D]`` for the sequence
operations — ``linear`` and the activations act on the last axis, ``D``,
across ``T`` positions — while 1-D convolution and pooling read rank 3 as
``[B, C, L]``, channels before length. ``transpose(1, 2)`` is the bridge
between the two readings, and a second ``transpose(1, 2)`` afterwards
returns to the sequence layout. At rank 4 the layout is ``[B, C, H, W]``,
so ``transpose(2, 3)`` swaps height and width and ``transpose(1, 3)``
exchanges channels with width.

The result is a view: it shares storage and autograd history with the
input and is generally not contiguous. Downstream operations handle
non-contiguous inputs; ``reshape`` and ``flatten`` copy when they must.
There are no parameters, no buffers, and no difference between train and
eval mode. The computation is a stride permutation, so it is exact in
every compute dtype.
        """
        ...
    @overload
    def transpose(self, dim0: int, dim1: int, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Swap two non-batch axes of the tensor.

        Returns ``x.transpose(dim0, dim1)``: the extents at ``dim0`` and
``dim1`` trade places and every other axis keeps its extent. The batch
axis 0 is never part of a swap, so both arguments are at least 1 and at
most 3 (the last axis of the deepest supported rank). Passing the same
axis twice is the identity.

Axis conventions. A rank-3 tensor is ``[B, T, D]`` for the sequence
operations — ``linear`` and the activations act on the last axis, ``D``,
across ``T`` positions — while 1-D convolution and pooling read rank 3 as
``[B, C, L]``, channels before length. ``transpose(1, 2)`` is the bridge
between the two readings, and a second ``transpose(1, 2)`` afterwards
returns to the sequence layout. At rank 4 the layout is ``[B, C, H, W]``,
so ``transpose(2, 3)`` swaps height and width and ``transpose(1, 3)``
exchanges channels with width.

The result is a view: it shares storage and autograd history with the
input and is generally not contiguous. Downstream operations handle
non-contiguous inputs; ``reshape`` and ``flatten`` copy when they must.
There are no parameters, no buffers, and no difference between train and
eval mode. The computation is a stride permutation, so it is exact in
every compute dtype.
        """
        ...
    @overload
    def upsample(self, x: Symbol, scale_factor: int = 2, *, mode: str = "nearest", align_corners: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Enlarge height and width by an integer factor with a fixed interpolation kernel.

        Resamples the spatial axes of a ``[B, C, H, W]`` image by an exact
integer factor: ``H_out = H * scale_factor`` and
``W_out = W * scale_factor``. The channel axis is untouched and there are
no parameters, so train and eval behave identically and the layer is a
pure function of its input.

Modes:

- ``nearest`` repeats each input pixel in a ``scale_factor`` x
  ``scale_factor`` block. It is the cheapest choice and is exactly
  equivalent to ``x.repeat_interleave(scale_factor, 2).repeat_interleave(scale_factor, 3)``.
- ``bilinear`` and ``bicubic`` interpolate between neighbouring pixels.
  ``align_corners`` selects the sampling grid convention: ``False`` (the
  default) treats pixels as areas, ``True`` pins the corner pixel centers
  of input and output together.

``align_corners`` is meaningful only for ``bilinear`` and ``bicubic``;
with ``nearest`` it must stay ``False``, and the underlying PyTorch call
receives ``None`` so no warning is emitted. Note that ``bicubic`` can
overshoot the input range; clamp afterwards if bounded output matters.

Resolution is bidirectional: a known input extent fixes the output, and a
known output extent fixes the input when it divides by ``scale_factor``,
otherwise ``E_CONSTRAINT`` is reported.
        """
        ...
    @overload
    def upsample(self, scale_factor: int = 2, *, mode: str = "nearest", align_corners: bool = False, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Enlarge height and width by an integer factor with a fixed interpolation kernel.

        Resamples the spatial axes of a ``[B, C, H, W]`` image by an exact
integer factor: ``H_out = H * scale_factor`` and
``W_out = W * scale_factor``. The channel axis is untouched and there are
no parameters, so train and eval behave identically and the layer is a
pure function of its input.

Modes:

- ``nearest`` repeats each input pixel in a ``scale_factor`` x
  ``scale_factor`` block. It is the cheapest choice and is exactly
  equivalent to ``x.repeat_interleave(scale_factor, 2).repeat_interleave(scale_factor, 3)``.
- ``bilinear`` and ``bicubic`` interpolate between neighbouring pixels.
  ``align_corners`` selects the sampling grid convention: ``False`` (the
  default) treats pixels as areas, ``True`` pins the corner pixel centers
  of input and output together.

``align_corners`` is meaningful only for ``bilinear`` and ``bicubic``;
with ``nearest`` it must stay ``False``, and the underlying PyTorch call
receives ``None`` so no warning is emitted. Note that ``bicubic`` can
overshoot the input range; clamp afterwards if bounded output matters.

Resolution is bidirectional: a known input extent fixes the output, and a
known output extent fixes the input when it divides by ``scale_factor``,
otherwise ``E_CONSTRAINT`` is reported.
        """
        ...
    def where(self, cond: Symbol, a: Symbol, b: Symbol, *, name: str | None = None, policy: str | None = None, init: Mapping[str, Any] | None = None, trainable: Mapping[str, bool] | bool | None = None, tie: Mapping[str, str] | None = None) -> Symbol:
        """Ternary conditional selection: out = where(cond, a, b).

        Ternary element-wise selection: out = torch.where(cond, a, b).
        """
        ...

ops: OperatorNamespace

def scope(prefix: str): ...

def capture_callable(
    fn: Callable[..., Any],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    dtype: str = 'float32',
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
) -> Graph: ...

def resolve_callable(
    fn: Callable[..., Any],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    dtype: str = 'float32',
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
) -> ResolvedPlan: ...
