# Operators

Every operator below is declared with `@operator` on its module class; these pages
are generated from those declarations by `python -m hndl.docs`.

## activation

| Operator | Summary |
| --- | --- |
| [`clamp`](clamp.md) | Saturate every element into the closed interval [min, max]. |
| [`dropout`](dropout.md) | Randomly zero elements during training and rescale the rest. |
| [`leaky_relu`](leaky_relu.md) | ReLU with a small slope for negative inputs. |
| [`relu`](relu.md) | Rectified linear unit, max(x, 0). |
| [`tanh`](tanh.md) | Hyperbolic tangent, squashing values into (-1, 1). |

## arithmetic

| Operator | Summary |
| --- | --- |
| [`mul`](mul.md) | Elementwise product of two tensors with identical shapes. |
| [`scale`](scale.md) | Multiply a tensor by a fixed scalar. |
| [`sub`](sub.md) | Elementwise difference of two tensors with identical shapes. |

## convolution

| Operator | Summary |
| --- | --- |
| [`conv`](conv.md) | Two-dimensional convolution over [B, C, H, W] images. |
| [`deconv`](deconv.md) | Transposed two-dimensional convolution, typically for upsampling. |

## core

| Operator | Summary |
| --- | --- |
| [`linear`](linear.md) | Fully connected layer: a learned affine map on the last axis. |

## join

| Operator | Summary |
| --- | --- |
| [`add`](add.md) | Elementwise sum of two tensors with identical shapes. |
| [`concat`](concat.md) | Join two or more tensors along one axis. |

## normalization

| Operator | Summary |
| --- | --- |
| [`adaptive_norm`](adaptive_norm.md) | Instance-normalize features, then apply a per-example learned scale and bias. |
| [`group_norm`](group_norm.md) | Normalize channel groups per example, with learned per-channel affine. |
| [`layer_norm`](layer_norm.md) | Normalize the last axis of every position with a learned scale and bias. |
| [`rms_norm`](rms_norm.md) | Scale the last axis by its root-mean-square, with a learned per-feature gain. |

## pretrained

| Operator | Summary |
| --- | --- |
| [`pretrained`](pretrained.md) | Load a pretrained network from disk or the Hugging Face Hub as one frozen node. |

## shape

| Operator | Summary |
| --- | --- |
| [`flatten`](flatten.md) | Collapse every non-batch axis into one feature axis. |
| [`mean`](mean.md) | Average one non-batch axis away. |
| [`pad`](pad.md) | Enlarge trailing axes with constant, reflected, or replicated borders. |
| [`reshape`](reshape.md) | View the tensor with new non-batch dimensions, preserving the element count. |
| [`split`](split.md) | Cut one axis into a first section and the remainder. |
| [`sum`](sum.md) | Add up one non-batch axis. |

## spatial

| Operator | Summary |
| --- | --- |
| [`adaptive_avg_pool`](adaptive_avg_pool.md) | Average-pool [B, C, H, W] images to a fixed output height and width. |
| [`global_avg_pool`](global_avg_pool.md) | Average each channel over height and width, producing one value per channel. |
