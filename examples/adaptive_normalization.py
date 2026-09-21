"""Build two style-conditioned graphs with the built-in adaptive normalization.

Run from an installed checkout: ``python examples/adaptive_normalization.py --device cuda:0``.
Device selection never falls back. This fixture defines its own affine
convention, not a full StyleGAN model.
"""

import argparse

import torch

from hndl.torch import network


MAPPING_CONFIG = '''
linear(256, name="mapping")
w = relu(name="w")
linear(w, name="project")
features = reshape(64, 4, 4, name="seed")
style = linear(w, name="style", init={"weight": 0, "bias": 0})
adaptive_norm(features, style, name="norm")
'''

SPLIT_CONFIG = '''
z1, z2 = split(64, name="partition")
linear(z1, name="project")
features = reshape(32, 4, 4, name="seed")
adaptive_norm(features, z2, name="norm")
'''


def mapping_model(*, device="cpu", initialization_seed=7):
    """Fan mapping features into projection and a declaratively zeroed affine."""
    return network(MAPPING_CONFIG, input_shape=("B", 128), output_shape=("B", 64, 4, 4),
                   device=device, initialization_seed=initialization_seed)


def split_model(*, device="cpu", initialization_seed=7):
    """Use the first 64 features for content and the remainder for style."""
    return network(SPLIT_CONFIG, input_shape=("B", 128), output_shape=("B", 32, 4, 4),
                   device=device, initialization_seed=initialization_seed)


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
