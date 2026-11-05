"""Cache sessions guard frozen state and reject overlapping graph calls."""

import threading

import pytest
import torch
from torch import nn

import hndl.inference as inference
from hndl.errors import HNDLError
from hndl.inference import InferenceCache
from hndl.torch import network


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _buffer_weight_model(device):
    model = network("attention(4,causal=True)", input_shape=("B", 2, 16),
                    output_shape=("B", 2, 16), device=device).eval()
    with model.allow_state_mutation():
        projection = model[0].k_proj
        weight = projection.weight.detach().clone()
        del projection.weight
        projection.register_buffer("weight", weight)
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_graph_session_rejects_buffer_weight_drift_before_state_advances(device):
    model = _buffer_weight_model(device)
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16, device=device)
    with torch.no_grad():
        model(x, cache=cache)
        model[0].k_proj.weight.add_(1)
        with pytest.raises(HNDLError, match="buffers"):
            model(x, cache=cache)
        assert cache.position("n0") == 1
        cache.reset()
        actual = model(x, cache=cache)
        torch.testing.assert_close(actual, model[0](x))


@pytest.mark.parametrize("device", DEVICES)
def test_standalone_layer_rejects_buffer_weight_drift(device):
    layer = _buffer_weight_model(device)[0]
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16, device=device)
    with torch.no_grad():
        layer(x, cache=cache, node_id="attention")
        layer.k_proj.weight.add_(1)
        with pytest.raises(HNDLError, match="buffers"):
            layer(x, cache=cache, node_id="attention")
        assert cache.position("attention") == 1
        cache.reset("attention")
        torch.testing.assert_close(layer(x, cache=cache, node_id="attention"), layer(x))


def test_graph_session_lock_covers_begin_validation_and_releases_after_success(monkeypatch):
    model = _buffer_weight_model("cpu")
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16)
    entered = threading.Event()
    release = threading.Event()
    errors = []
    original = inference._parameter_signature

    def delayed_signature(module):
        result = original(module)
        if threading.current_thread().name == "cache-owner" and module is model:
            entered.set()
            if not release.wait(10):
                raise AssertionError("test did not release the cache owner")
        return result

    monkeypatch.setattr(inference, "_parameter_signature", delayed_signature)

    def run_owner():
        try:
            with torch.no_grad():
                cache._begin_call(model, (x,))
                cache._end_call(True)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_owner, name="cache-owner")
    thread.start()
    try:
        assert entered.wait(10)
        with torch.no_grad(), pytest.raises(HNDLError, match="concurrently"):
            cache._begin_call(model, (x,))
        with pytest.raises(HNDLError, match="active"):
            cache.reset()
        with torch.no_grad(), pytest.raises(HNDLError, match="concurrently"):
            model[0](x, cache=cache, node_id="attention")
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    with torch.no_grad():
        cache._begin_call(model, (x,))
        cache._end_call(True)
    cache.reset()


@pytest.mark.parametrize("node_id", [None, "attention"])
def test_reset_cannot_modify_active_session(node_id):
    model = _buffer_weight_model("cpu")
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16)
    with torch.no_grad():
        cache._begin_call(model, (x,))
        model[0](x, cache=cache, node_id="attention")
        with pytest.raises(HNDLError, match="active"):
            cache.reset(node_id)
        assert cache.position("attention") == 1
        cache._end_call(True)
    cache.reset(node_id)
    assert cache.position("attention") == 0


def test_failed_begin_and_finish_both_release_the_session_lock():
    model = _buffer_weight_model("cpu")
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16)
    with pytest.raises(HNDLError, match="no_grad"):
        cache._begin_call(model, (x,))
    cache.reset()
    with torch.no_grad():
        cache._begin_call(model, (x,))
        cache._end_call(False)
        with pytest.raises(HNDLError, match="previous cached call failed"):
            cache._begin_call(model, (x,))
        cache.reset()
        cache._begin_call(model, (x,))
        cache._end_call(True)


def test_position_validation_failure_unlocks_for_reset():
    model = _buffer_weight_model("cpu")
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16)
    with torch.no_grad():
        cache._begin_call(model, (x,))
        model[0](x, cache=cache, node_id="first")
        model[0](x, cache=cache, node_id="second")
        cache["second"].position = 0
        with pytest.raises(HNDLError, match="different positions"):
            cache._end_call(True)
        cache.reset()
        cache._begin_call(model, (x,))
        cache._end_call(True)


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_unversioned_inference_state_is_rejected_clearly_and_unlocks(kind):
    model = nn.Identity().eval()
    with torch.inference_mode():
        value = torch.ones(1)
        if kind == "parameter":
            model.register_parameter("state", nn.Parameter(value, requires_grad=False))
        else:
            model.register_buffer("state", value)
    cache = InferenceCache()
    x = torch.randn(2, 1, 16)
    with torch.no_grad(), pytest.raises(HNDLError, match="mutation versions.*inference_mode"):
        cache._begin_call(model, (x,))
    cache.reset()
    with torch.no_grad():
        model.state = (nn.Parameter(model.state.clone(), requires_grad=False)
                       if kind == "parameter" else model.state.clone())
        cache._begin_call(model, (x,))
        cache._end_call(True)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prepared", [False, True])
def test_graph_session_rejects_cuda_stream_changes_before_queued_state_writes(prepared):
    model = _buffer_weight_model("cuda:0")
    executor = model.prepare() if prepared else model
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16, device="cuda:0")
    first = torch.cuda.Stream()
    second = torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    try:
        with torch.no_grad(), torch.cuda.stream(first):
            # Queue a delay before the state writes; host completion does not
            # imply that the queued KV/recurrent updates have finished.
            torch.cuda._sleep(50_000_000)
            initial = executor(x, cache=cache)
        with torch.no_grad(), torch.cuda.stream(second):
            with pytest.raises(HNDLError, match="CUDA stream changed"):
                executor(x, cache=cache)
            assert cache.position("n0") == 1
        # Rejection leaves the original session usable on its stream, even
        # while the work queued above has not been explicitly synchronized.
        with torch.no_grad(), torch.cuda.stream(first):
            following = executor(x, cache=cache)
            expected = model(torch.cat((x, x), dim=1))
            torch.testing.assert_close(torch.cat((initial, following), dim=1), expected)
        first.synchronize()
        cache.reset()
        with torch.no_grad(), torch.cuda.stream(second):
            actual = executor(x, cache=cache)
            torch.testing.assert_close(actual, model[0](x))
    finally:
        first.synchronize()
        second.synchronize()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_standalone_session_rejects_cuda_stream_changes_and_partial_reset_rebinds():
    layer = _buffer_weight_model("cuda:0")[0]
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16, device="cuda:0")
    first = torch.cuda.Stream()
    second = torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    try:
        with torch.no_grad(), torch.cuda.stream(first):
            torch.cuda._sleep(50_000_000)
            initial = layer(x, cache=cache, node_id="attention")
        with torch.no_grad(), torch.cuda.stream(second):
            with pytest.raises(HNDLError, match="CUDA stream changed"):
                layer(x, cache=cache, node_id="attention")
            # A new node ID must not bypass the cache-wide stream binding.
            with pytest.raises(HNDLError, match="CUDA stream changed"):
                layer(x, cache=cache, node_id="other")
            assert cache.position("attention") == 1 and "other" not in cache
        with torch.no_grad(), torch.cuda.stream(first):
            following = layer(x, cache=cache, node_id="attention")
            expected = layer(torch.cat((x, x), dim=1))
            torch.testing.assert_close(torch.cat((initial, following), dim=1), expected)
        first.synchronize()
        cache.reset("attention")
        with torch.no_grad(), torch.cuda.stream(second):
            torch.testing.assert_close(layer(x, cache=cache, node_id="attention"), layer(x))
    finally:
        first.synchronize()
        second.synchronize()


def test_cpu_sessions_do_not_query_cuda_streams(monkeypatch):
    model = _buffer_weight_model("cpu")
    cache = InferenceCache(8)
    x = torch.randn(2, 1, 16)
    def unexpected_stream_query(*args, **kwargs):
        raise AssertionError("CPU decoding queried a CUDA stream")
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_stream_query)
    with torch.no_grad():
        cache._begin_call(model, (x,))
        actual = model[0](x, cache=cache, node_id="attention")
        cache._end_call(True)
        torch.testing.assert_close(actual, model[0](x))
        cache.reset()
        torch.testing.assert_close(model[0](x, cache=cache, node_id="attention"), model[0](x))
