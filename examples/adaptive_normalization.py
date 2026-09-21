"""Register learned affine instance normalization and build two style graphs.

Run from an installed checkout: ``python examples/adaptive_normalization.py --device cpu``.
CUDA is also supported when available; device selection never falls back.
This fixture defines its own affine convention, not a full StyleGAN model.
"""

import argparse
import math

import torch
from torch import nn

from hndl import Argument, Dim, Registry, ShapeRule
from hndl.torch import network, register_torch


class AdaptiveNorm(nn.Module):
    """Population instance normalization followed by learned scale and bias.

    ``params`` holds [delta_gamma, beta], one of each per feature channel.
    A zero style vector gives normalized features, not the raw input.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        self.eps = float(eps)

    def forward(self, x, params):
        if x.ndim != 4 or params.ndim != 2 or params.shape != (x.shape[0], 2 * x.shape[1]):
            raise ValueError("Expected features [B,C,H,W] and style parameters [B,2*C]")
        mean = x.mean(dim=(2, 3), keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=(2, 3), keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        delta_gamma, beta = params.split(x.shape[1], dim=1)
        return (1 + delta_gamma[:, :, None, None]) * normalized + beta[:, :, None, None]


def make_registry():
    """Explicitly register pure shape relations and their trusted backend."""
    registry = Registry.builtins()
    registry.register(
        "adaptive_norm",
        identity="example.adaptive_norm",
        version=1,
        shape=ShapeRule(
            inputs={"x": ("B", "C", "H", "W"), "params": ("B", Dim("C", scale=2))},
            outputs={"out": ("B", "C", "H", "W")},
        ),
        arguments={"eps": Argument(float, default=1e-5, minimum=0.0, exclusive_minimum=True)},
        max_state_bytes=0,
    )
    register_torch(registry, "adaptive_norm", module=AdaptiveNorm, state_version=1)
    return registry


MAPPING_CONFIG = '''
linear(256, name="mapping")
w = relu(name="w")
linear(w, name="project")
features = reshape(64, 4, 4, name="seed")
style = linear(w, name="style")
adaptive_norm(features, style, name="norm")
'''

SPLIT_CONFIG = '''
z1, z2 = split(64, name="partition")
linear(z1, name="project")
features = reshape(32, 4, 4, name="seed")
adaptive_norm(features, z2, name="norm")
'''


def mapping_model(*, device="cpu", initialization_seed=7):
    """Fan one mapping tensor into feature projection and a style affine."""
    model = network(MAPPING_CONFIG, input_shape=("B", 128), output_shape=("B", 64, 4, 4),
                    registry=make_registry(), device=device, initialization_seed=initialization_seed)
    # Initialization is an explicit host action, saved in the model state.
    nn.init.zeros_(model["style"].weight)
    nn.init.zeros_(model["style"].bias)
    return model


def split_model(*, device="cpu", initialization_seed=7):
    """Use the first 64 features for content and the remainder for style."""
    return network(SPLIT_CONFIG, input_shape=("B", 128), output_shape=("B", 32, 4, 4),
                   registry=make_registry(), device=device, initialization_seed=initialization_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    for name, make_model in (("Mapping fan-out", mapping_model), ("Split and remainder", split_model)):
        model = make_model(device=args.device)
        x = torch.randn(2, 128, device=args.device, requires_grad=True)
        output = model(x)
        output.square().mean().backward()
        print(f"\n{name}\n{model}")
        print(f"Output: {tuple(output.shape)}; finite input gradient: {bool(x.grad.isfinite().all())}")


if __name__ == "__main__":
    main()
