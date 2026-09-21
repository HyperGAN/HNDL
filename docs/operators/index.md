# Operators

Every operator below is declared with `@operator` on its module class; these pages
are generated from those declarations by `python -m hndl.docs`.

## activation

| Operator | Summary |
| --- | --- |
| [`clamp`](clamp.md) | Saturate every element into the closed interval [min, max]. |
| [`dropout`](dropout.md) | Randomly zero elements during training and rescale the rest. |
| [`elu`](elu.md) | Exponential linear unit: identity above zero, saturating below. |
| [`gelu`](gelu.md) | Gaussian error linear unit, x * Phi(x). |
| [`hardswish`](hardswish.md) | Piecewise-linear approximation of swish, x * relu6(x + 3) / 6. |
| [`identity`](identity.md) | Pass the tensor through unchanged, as a named node. |
| [`leaky_relu`](leaky_relu.md) | ReLU with a small slope for negative inputs. |
| [`mish`](mish.md) | Self-gated smooth activation, x * tanh(softplus(x)). |
| [`quick_gelu`](quick_gelu.md) | CLIP's fast GELU approximation, x * sigmoid(1.702 * x). |
| [`relu`](relu.md) | Rectified linear unit, max(x, 0). |
| [`sigmoid`](sigmoid.md) | Logistic sigmoid, squashing values into (0, 1). |
| [`silu`](silu.md) | Sigmoid linear unit (swish), x * sigmoid(x). |
| [`softmax`](softmax.md) | Normalize one non-batch axis into a probability distribution. |
| [`softplus`](softplus.md) | Smooth positive activation, log(1 + exp(beta*x)) / beta. |
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
| [`conv1d`](conv1d.md) | One-dimensional convolution over [B, C, L] signals. |
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

## memory

| Operator | Summary |
| --- | --- |
| [`hopfield`](hopfield.md) | Retrieve learned patterns by iterating the modern Hopfield update on the last axis. |
| [`moe`](moe.md) | Sparse mixture of experts: route every token to its top-k feed-forward experts. |

## normalization

| Operator | Summary |
| --- | --- |
| [`adaptive_norm`](adaptive_norm.md) | Instance-normalize features, then apply a per-example learned scale and bias. |
| [`batch_norm`](batch_norm.md) | Normalize each channel over the batch and spatial axes, tracking running statistics. |
| [`group_norm`](group_norm.md) | Normalize channel groups per example, with learned per-channel affine. |
| [`instance_norm`](instance_norm.md) | Normalize every channel of every example over its own spatial positions. |
| [`layer_norm`](layer_norm.md) | Normalize the last axis of every position with a learned scale and bias. |
| [`rms_norm`](rms_norm.md) | Scale the last axis by its root-mean-square, with a learned per-feature gain. |

## pretrained

| Operator | Summary |
| --- | --- |
| [`pretrained`](pretrained.md) | Load a pretrained network from disk or the Hugging Face Hub as one frozen node. |

## sequence

| Operator | Summary |
| --- | --- |
| [`attention`](attention.md) | Multi-head self-attention over a [B, T, D] sequence. |
| [`cls_token`](cls_token.md) | Prepend one learned classification token to a sequence. |
| [`cross_attention`](cross_attention.md) | Multi-head attention with queries from one sequence and keys and values from another. |
| [`embedding`](embedding.md) | Look up a learned vector for every integer token id. |
| [`feed_forward`](feed_forward.md) | Transformer feed-forward block: widen, activate, project back. |
| [`pool_tokens`](pool_tokens.md) | Reduce a [B, T, D] sequence to one [B, D] vector per example. |
| [`pos_embed`](pos_embed.md) | Add a learned position vector to every position of a sequence. |
| [`swiglu`](swiglu.md) | SwiGLU feed-forward block: a SiLU-gated projection folded back to the input width. |

## shape

| Operator | Summary |
| --- | --- |
| [`flatten`](flatten.md) | Collapse every non-batch axis into one feature axis. |
| [`mean`](mean.md) | Average one non-batch axis away. |
| [`pad`](pad.md) | Enlarge trailing axes with constant, reflected, or replicated borders. |
| [`permute`](permute.md) | Reorder the non-batch axes, keeping the batch axis first. |
| [`reshape`](reshape.md) | View the tensor with new non-batch dimensions, preserving the element count. |
| [`split`](split.md) | Cut one axis into a first section and the remainder. |
| [`sum`](sum.md) | Add up one non-batch axis. |
| [`transpose`](transpose.md) | Swap two non-batch axes of the tensor. |

## spatial

| Operator | Summary |
| --- | --- |
| [`adaptive_avg_pool`](adaptive_avg_pool.md) | Average-pool [B, C, H, W] images to a fixed output height and width. |
| [`avg_pool`](avg_pool.md) | Two-dimensional average pooling over [B, C, H, W] images. |
| [`global_avg_pool`](global_avg_pool.md) | Average each channel over height and width, producing one value per channel. |
| [`max_pool`](max_pool.md) | Two-dimensional max pooling over [B, C, H, W] images. |
| [`pixel_shuffle`](pixel_shuffle.md) | Trade channels for resolution: rearrange C*r^2 channels into an r-times larger image. |
| [`upsample`](upsample.md) | Enlarge height and width by an integer factor with a fixed interpolation kernel. |
