# `coordinate_grid`

Emit a normalized 2D coordinate grid in (x, y) order.

**Category:** spatial · **Identity:** `coordinate_grid@1`

## Shape

```text
x[B, ...]:any -> out[B, H, W, 2]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]:any` | any |
| `out` | output | `out[B, H, W, 2]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `height` (positional) | int | required | >= 2; <= 2147483647; binds `H` | Grid rows, with centers spanning -1 to 1 inclusive. |
| `width` (positional) | int | required | >= 2; <= 2147483647; binds `W` | Grid columns, with centers spanning -1 to 1 inclusive. |

## Description

Emit ``[B,H,W,2]`` coordinates spanning [-1,1] including both endpoints.

The final axis is (x, y); x varies along columns, y along rows. Input
values are ignored: only batch size is used. The fixed ``grid`` buffer
follows model device/dtype and is saved in the state dict. No random
draws or trainable parameters. There is no gradient to the input.
This is the endpoint convention of ``linspace(-1,1,extent)``, regardless
of the ``align_corners`` choice in a downstream sampler.

## Examples

### Example 1

A fixed rectangular coordinate grid; only the input batch size is used.

```python
coordinate_grid(4, 6)
```

Input `['B', 8]` (`input_dtype="int64"`) → output `['B', 4, 6, 2]`.

```text
Network: [B, 8] -> [B, 4, 6, 2]  dtype=float32  input_dtype=int64
index  name  operation        input shapes  output shapes
0      n0    coordinate_grid  x=[B, 8]      out=[B, 4, 6, 2]
```

Parameters: 0
