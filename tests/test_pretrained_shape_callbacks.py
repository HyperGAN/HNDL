"""Shape callback dispatch does not require checkpoint-provider dependencies."""

import pytest

from hndl.pretrained import _invoke_shape_fn


def test_layers_keyword_is_not_bound_as_the_output_position():
    def callback(shape, layers=(), **kwargs):
        return shape, layers, kwargs["output"]

    assert _invoke_shape_fn(callback, ("B", 16), layers=("a", "b")) == (
        ("B", 16), ("a", "b"), "features")


def test_named_layer_and_output_are_dispatched_independently():
    def callback(shape, layer, *, output):
        return shape, layer, output

    assert _invoke_shape_fn(callback, ("B", 16), output="logits", layer="head") == (
        ("B", 16), "head", "logits")


def test_legacy_second_positional_output_and_single_argument():
    assert _invoke_shape_fn(lambda shape, out: (shape, out), ("B", 16), output="logits") == (
        ("B", 16), "logits")
    assert _invoke_shape_fn(lambda shape: shape, ("B", 16)) == ("B", 16)


def test_positional_only_semantic_arguments_are_not_duplicated():
    def callback(shape, output, layers, /):
        return shape, output, layers

    assert _invoke_shape_fn(callback, ("B", 16), output="logits", layers=("a",)) == (
        ("B", 16), "logits", ("a",))


def test_callback_error_is_not_retried_or_swallowed():
    calls = []

    def callback(shape, layers=()):
        calls.append(layers)
        raise TypeError("callback failure")

    with pytest.raises(TypeError, match="callback failure"):
        _invoke_shape_fn(callback, ("B", 16), layers=("a",))
    assert calls == [("a",)]
