"""Pinned shape tables and arguments for fixture plans.

These guard resolution behavior across refactors of the operator format. Update
them deliberately when a relation legitimately changes.
"""

import pytest

from hndl import resolve

GENERATOR = '''
linear()
relu()
reshape(512)
deconv(256, policy="up2")
relu()
deconv(128, policy="up2")
relu()
deconv(64, policy="up2")
relu()
conv(3, kernel_size=3, stride=1, padding=1)
tanh()
'''

BRANCHES = '''
z1, z2 = split(64, name="partition")
linear(z1, 128, name="content")
features = relu(name="features")
style = linear(z2, 128, name="style")
add(features, style, name="combined")
'''

ADAPTIVE = '''
linear(256, name="mapping")
w = relu(name="w")
linear(w, name="project")
features = reshape(64, 4, 4, name="seed")
style = linear(w, name="style", init={"weight": 0, "bias": 0})
adaptive_norm(features, style, name="norm")
'''

CASES = {
    "mlp": ("linear(64); relu(); linear()", ("B", 128), ("B", 10), [
        ("n0", "linear@1", {"out_features": 64, "in_features": 128, "bias": True}, {"x": ("B", 128)}, {"out": ("B", 64)}),
        ("n1", "relu@1", {}, {"x": ("B", 64)}, {"out": ("B", 64)}),
        ("n2", "linear@1", {"out_features": 10, "in_features": 64, "bias": True}, {"x": ("B", 64)}, {"out": ("B", 10)}),
    ]),
    "generator": (GENERATOR, ("B", 128), ("B", 3, 32, 32), [
        ("n0", "linear@1", {"out_features": 8192, "in_features": 128, "bias": True}, {"x": ("B", 128)}, {"out": ("B", 8192)}),
        ("n1", "relu@1", {}, {"x": ("B", 8192)}, {"out": ("B", 8192)}),
        ("n2", "reshape@1", {"shape": (512, 4, 4)}, {"x": ("B", 8192)}, {"out": ("B", 512, 4, 4)}),
        ("n3", "conv_transpose2d@1", {"out_channels": 256, "in_channels": 512, "kernel_size": (4, 4), "stride": (2, 2),
                                      "padding": (1, 1), "dilation": (1, 1), "output_padding": (0, 0), "groups": 1,
                                      "bias": True}, {"x": ("B", 512, 4, 4)}, {"out": ("B", 256, 8, 8)}),
        ("n4", "relu@1", {}, {"x": ("B", 256, 8, 8)}, {"out": ("B", 256, 8, 8)}),
        ("n5", "conv_transpose2d@1", {"out_channels": 128, "in_channels": 256, "kernel_size": (4, 4), "stride": (2, 2),
                                      "padding": (1, 1), "dilation": (1, 1), "output_padding": (0, 0), "groups": 1,
                                      "bias": True}, {"x": ("B", 256, 8, 8)}, {"out": ("B", 128, 16, 16)}),
        ("n6", "relu@1", {}, {"x": ("B", 128, 16, 16)}, {"out": ("B", 128, 16, 16)}),
        ("n7", "conv_transpose2d@1", {"out_channels": 64, "in_channels": 128, "kernel_size": (4, 4), "stride": (2, 2),
                                      "padding": (1, 1), "dilation": (1, 1), "output_padding": (0, 0), "groups": 1,
                                      "bias": True}, {"x": ("B", 128, 16, 16)}, {"out": ("B", 64, 32, 32)}),
        ("n8", "relu@1", {}, {"x": ("B", 64, 32, 32)}, {"out": ("B", 64, 32, 32)}),
        ("n9", "conv2d@1", {"out_channels": 3, "in_channels": 64, "kernel_size": (3, 3), "stride": (1, 1),
                            "padding": (1, 1), "dilation": (1, 1), "groups": 1, "bias": True},
         {"x": ("B", 64, 32, 32)}, {"out": ("B", 3, 32, 32)}),
        ("n10", "tanh@1", {}, {"x": ("B", 3, 32, 32)}, {"out": ("B", 3, 32, 32)}),
    ]),
    "branches": (BRANCHES, ("B", 128), ("B", 128), [
        ("partition", "split@1", {"size": 64, "dim": 1}, {"x": ("B", 128)}, {"first": ("B", 64), "rest": ("B", 64)}),
        ("content", "linear@1", {"out_features": 128, "in_features": 64, "bias": True}, {"x": ("B", 64)}, {"out": ("B", 128)}),
        ("features", "relu@1", {}, {"x": ("B", 128)}, {"out": ("B", 128)}),
        ("style", "linear@1", {"out_features": 128, "in_features": 64, "bias": True}, {"x": ("B", 64)}, {"out": ("B", 128)}),
        ("combined", "add@1", {}, {"a": ("B", 128), "b": ("B", 128)}, {"out": ("B", 128)}),
    ]),
    "adaptive": (ADAPTIVE, ("B", 128), ("B", 64, 4, 4), [
        ("mapping", "linear@1", {"out_features": 256, "in_features": 128, "bias": True}, {"x": ("B", 128)}, {"out": ("B", 256)}),
        ("w", "relu@1", {}, {"x": ("B", 256)}, {"out": ("B", 256)}),
        ("project", "linear@1", {"out_features": 1024, "in_features": 256, "bias": True}, {"x": ("B", 256)}, {"out": ("B", 1024)}),
        ("seed", "reshape@1", {"shape": (64, 4, 4)}, {"x": ("B", 1024)}, {"out": ("B", 64, 4, 4)}),
        ("style", "linear@1", {"out_features": 128, "in_features": 256, "bias": True}, {"x": ("B", 256)}, {"out": ("B", 128)}),
        ("norm", "adaptive_norm@1", {"eps": 1e-5}, {"x": ("B", 64, 4, 4), "params": ("B", 128)}, {"out": ("B", 64, 4, 4)}),
    ]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_plan(case):
    source, input_shape, output_shape, expected = CASES[case]
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    actual = [(node.id, node.op, dict(node.args), dict(node.input_shapes), dict(node.output_shapes))
              for node in plan.nodes]
    assert actual == expected
    assert plan.semantic_digest == type(plan).from_json(plan.to_json()).semantic_digest
