"""Comprehensive Empirical Stress-Test Harness for Milestone 2 Challenger 1.

Covers:
- H-03: _is_chain classification (unary operators, multi-output operators, single-output
  variadic operators, custom output names, branched topologies) and GraphModule slice step
  validation (step == 1, step != 1, out-of-bounds, empty slices, negative slices).
- M-08: torch.compile pre-forward state checks vs during-forward state changes (mutating
  parameters, buffers, submodules prior to forward vs during forward).
- M-09: Shallow copy (copy.copy(model)) and submodule in-place casts (model[0].double(),
  .half(), .to(device)) fast-path cache invalidation and store mutation tracking.
- L-06: _state_bytes exact memory accounting: storage deduplication for meta devices,
  multiple views, slices, transposes, compute_dtype scaling, mixed int/float buffers,
  and cross-device parity (meta vs cpu vs cuda).
"""

import copy
import pytest
import torch
from torch import nn

from hndl import Arg, HNDLError, Registry
from hndl.torch import (
    _compiling,
    _is_chain,
    _state_bytes,
    network,
)


# ==============================================================================
# H-03: _is_chain Topology & Operator Stress Tests
# ==============================================================================

@pytest.mark.parametrize("unary_op", [
    "linear(8)",
    "relu()",
    "gelu()",
    "silu()",
    "sigmoid()",
    "tanh()",
    "mish()",
    "softplus()",
    "elu()",
    "quick_gelu()",
    "hardswish()",
    "identity()",
    "flatten()",
    "rms_norm()",
])
def test_is_chain_standard_single_unary_operator(unary_op):
    """Every standard unary operator with single input and output 'out' is a chain."""
    m = network(unary_op, input_shape=("B", 8), output_shape=("B", 8), device="cpu")
    assert m._chain is True
    assert _is_chain(m.plan) is True
    assert len(m) == 1
    assert len(list(m)) == 1
    assert "index" in repr(m)


@pytest.mark.parametrize("chain_len", [2, 3, 5, 8])
def test_is_chain_consecutive_unary_chains_of_various_lengths(chain_len):
    """Linear chains of varying lengths are correctly classified as chains."""
    ops = ["linear(8)" if i == 0 else "relu()" if i % 2 == 1 else "linear(8)" for i in range(chain_len)]
    source = "\n".join(ops)
    m = network(source, input_shape=("B", 8), output_shape=("B", 8), device="cpu")
    assert m._chain is True
    assert _is_chain(m.plan) is True
    assert len(m) == chain_len
    assert len(list(m)) == chain_len
    # Verify slicing is enabled
    sliced = m[:chain_len]
    assert isinstance(sliced, nn.Sequential)
    assert len(sliced) == chain_len


@pytest.mark.parametrize("multi_op,source", [
    ("split_2", "a, b = split(4)\nconcat(a, b)"),
    ("chunk_2", "a, b = chunk(2)\nconcat(a, b)"),
    ("chunk_4", "a, b, c, d = chunk(4)\nconcat(a, b, c, d)"),
])
def test_is_chain_multi_output_operators_are_not_chains(multi_op, source):
    """Operators producing multiple outputs are branched and not chains."""
    m = network(source, input_shape=("B", 8), output_shape=("B", 8), device="cpu")
    assert m._chain is False
    assert _is_chain(m.plan) is False

    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = len(m)
    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = list(m)
    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = m[:2]
    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = m[0]

    # Node-name indexing works
    assert m["n0"] is not None


def test_is_chain_chunk_1_variadic_single_output_is_not_chain():
    """H-03: chunk(1) outputs ('out0',) which returns a 1-tuple, so it must not be a chain."""
    m = network("a, = chunk(1)\nlinear(a, 10)", input_shape=("B", 8), output_shape=("B", 10), device="cpu")
    assert m._chain is False
    assert _is_chain(m.plan) is False

    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = m[:2]
    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = m[0]
    assert m["n0"] is not None


@pytest.mark.parametrize("topology,source", [
    ("diamond", "a = linear(8)\nb = relu(a)\nc = linear(8)\nadd(b, c)"),
    ("skip_connection", "a = linear(8)\nb = relu(a)\nadd(a, b)"),
    ("residual", "a = linear(8)\nadd(x, a)"),
    ("multi_input", "add(x, y)"),
])
def test_is_chain_branched_topologies_are_not_chains(topology, source):
    """Graphs with skip-connections, residuals, or multiple inputs are not chains."""
    if topology == "multi_input":
        m = network(source, input_shape={"x": ("B", 8), "y": ("B", 8)}, output_shape=("B", 8), device="cpu")
    else:
        m = network(source, input_shape=("B", 8), output_shape=("B", 8), device="cpu")
    assert m._chain is False
    assert _is_chain(m.plan) is False
    with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
        _ = m[:2]


def test_is_chain_custom_unary_semantic_output_name_shift():
    """Custom unary operator with single semantic output port (e.g. 'shifted') is a chain."""
    registry = Registry.builtins()

    def _rel_shift(s):
        s.equal("x", "shifted")

    @registry.operator(
        "shift",
        summary="Shift input by 1",
        shape="x -> shifted",
        relation=_rel_shift,
        category="elementwise",
    )
    class Shift(nn.Module):
        def forward(self, x):
            return x + 1.0

    m = network("shift()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)
    assert m._chain is True
    assert _is_chain(m.plan) is True

    # Slicing returns nn.Sequential and forward works seamlessly
    sliced = m[:2]
    assert isinstance(sliced, nn.Sequential)
    x = torch.randn(2, 4)
    out = sliced(x)
    assert out.shape == (2, 4)


@pytest.mark.xfail(
    reason="H-03 edge case: _is_chain checks (node.outputs[0].startswith('out') and node.outputs[0][3:].isdigit()) "
           "which misses variadic operators whose prefix is not 'out' (e.g. 'part0').",
    strict=False,
)
def test_is_chain_custom_variadic_non_out_prefix_fails_or_crashes():
    """Demonstrate vulnerability: custom variadic operator with prefix 'part*' returns a 1-tuple.

    If _is_chain only checks for 'out<digits>', it treats 'part0' as a chain,
    causing m[:2] to return an nn.Sequential that crashes on forward!
    """
    registry = Registry.builtins()

    def _rel_part(s):
        rank = len(s.shape("x"))
        for port in s.outputs:
            s.rank(port, rank)
            s.share_batch("x", port)
            for i in range(1, rank):
                s.axis(port, i, s.shape("x")[i])

    @registry.operator(
        "custom_chunk",
        summary="Custom chunk with part* prefix",
        shape="x -> part*",
        outputs_from="parts",
        relation=_rel_part,
        args={"parts": Arg(int, min=1, max=32, help="parts count")},
        category="shape",
    )
    class CustomChunk(nn.Module):
        def __init__(self, parts):
            super().__init__()
            self.parts = parts

        def forward(self, x):
            return tuple(torch.chunk(x, self.parts, dim=-1))

    m = network(
        "p, = custom_chunk(1)\nlinear(p, 4)",
        input_shape=("B", 4),
        output_shape=("B", 4),
        device="cpu",
        registry=registry,
    )
    # The operator returns a 1-tuple, so it should NOT be classified as a unary chain!
    assert m._chain is False, "Custom variadic operator returning 1-tuple was misclassified as chain"


# ==============================================================================
# H-03: GraphModule Slice Step & Bounds Validation Stress Tests
# ==============================================================================

@pytest.mark.parametrize("invalid_step", [2, 3, 5, -1, -2, -3, 0])
def test_slice_step_not_one_strictly_rejected(invalid_step):
    """H-03: Any slice step other than None or 1 must raise ValueError."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    assert m._chain is True

    with pytest.raises(ValueError, match=f"GraphModule slice step must be 1; got {invalid_step}"):
        if invalid_step == 0:
            _ = m[::0]
        else:
            _ = m[::invalid_step]

    if invalid_step != 0:
        with pytest.raises(ValueError, match=f"GraphModule slice step must be 1; got {invalid_step}"):
            _ = m[0:2:invalid_step]
    else:
        with pytest.raises(ValueError, match="GraphModule slice step must be 1; got 0"):
            _ = m[0:2:0]


@pytest.mark.parametrize("valid_slice", [
    slice(None, None, None),
    slice(None, None, 1),
    slice(0, 2, None),
    slice(0, 2, 1),
    slice(1, 3, 1),
    slice(1, None, None),
    slice(None, 2, None),
])
def test_slice_valid_step_one_accepted(valid_slice):
    """Valid slices with step in (None, 1) return nn.Sequential."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    seq = m[valid_slice]
    assert isinstance(seq, nn.Sequential)


@pytest.mark.parametrize("bounds_slice,expected_count", [
    (slice(10, 20), 0),   # completely past the end
    (slice(3, 1), 0),     # inverted bounds
    (slice(-1, 0), 0),    # inverted negative bounds
    (slice(-2, None), 2), # last two nodes
    (slice(-1, None), 1), # last node
    (slice(None, -1), 2), # all except last node
    (slice(-3, -1), 2),   # middle nodes with negative indexing
])
def test_slice_bounds_out_of_range_and_negative(bounds_slice, expected_count):
    """Slices with out-of-bounds or negative indices behave like tuple slices."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    seq = m[bounds_slice]
    assert isinstance(seq, nn.Sequential)
    assert len(seq) == expected_count

    # If empty, calling it acts as identity
    if expected_count == 0:
        x = torch.randn(2, 4)
        torch.testing.assert_close(seq(x), x)


def test_slice_composition_numerical_equivalence():
    """Executing sliced sequential segments in series matches full model execution."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(3, 4)
    expected = m(x)

    # Slice into [0:1], [1:2], [2:3]
    seg0 = m[:1]
    seg1 = m[1:2]
    seg2 = m[2:]

    h0 = seg0(x)
    h1 = seg1(h0)
    out = seg2(h1)

    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("invalid_key", [1.5, [0, 1], (0, 1), object()])
def test_slice_invalid_key_types_raise_type_error(invalid_key):
    """Non-integer, non-slice, non-string keys raise TypeError."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    with pytest.raises(TypeError, match="Layer index must be an integer, slice, or node name"):
        _ = m[invalid_key]


def test_key_error_on_missing_node_name():
    """String lookup on non-existent node raises KeyError."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    with pytest.raises(KeyError, match="nonexistent"):
        _ = m["nonexistent"]


# ==============================================================================
# M-08: torch.compile Pre-Forward vs During-Forward State Verification
# ==============================================================================

@pytest.mark.parametrize("mutation_kind", ["param_add", "buffer_add", "submodule_add", "param_del"])
def test_compile_pre_forward_state_changes_blamed_on_build(mutation_kind):
    """M-08: State mutated prior to forward under compilation must raise 'no longer matches the build'."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    node0 = m.nodes["n_n0"]

    if mutation_kind == "param_add":
        node0.register_parameter("extra_p", nn.Parameter(torch.zeros(1)))
    elif mutation_kind == "buffer_add":
        node0.register_buffer("extra_b", torch.zeros(1))
    elif mutation_kind == "submodule_add":
        node0.extra_mod = nn.Identity()
    elif mutation_kind == "param_del":
        del node0.bias

    with _compiling():
        with pytest.raises(HNDLError) as exc:
            m(torch.randn(2, 4))
        assert exc.value.code == "E_RUNTIME"
        assert "no longer matches the build" in exc.value.message
        assert "changed during forward" not in exc.value.message


def test_compile_pre_forward_in_place_weight_version_modification():
    """M-08: In-place parameter mutation before forward is detected via _version under compilation."""
    m = network("linear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)

    # First call establishes baseline
    m(x)

    with _compiling():
        # Mutate parameter in-place
        p = next(m.parameters())
        with torch.no_grad():
            p.add_(1.0)

        with pytest.raises(HNDLError) as exc:
            m(x)
        assert exc.value.code == "E_RUNTIME"
        assert "no longer matches the build" in exc.value.message
        assert "was modified" in exc.value.message
        assert "changed during forward" not in exc.value.message


@pytest.mark.parametrize("mutate_target", ["parameter", "buffer", "submodule", "in_place_version"])
def test_compile_during_forward_state_changes_blamed_on_forward(mutate_target):
    """M-08: State mutated during forward is blamed on forward, and subsequent call on build."""
    registry = Registry.builtins()

    @registry.operator(
        f"mut_{mutate_target}",
        summary=f"Mutates {mutate_target} during forward",
        shape="x -> out",
    )
    class Mutator(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(4))
            self.b = None

        def forward(self, x):
            if mutate_target == "parameter":
                self.extra_p = nn.Parameter(torch.zeros(1))
            elif mutate_target == "buffer":
                self.register_buffer("extra_b", torch.zeros(1))
            elif mutate_target == "submodule":
                self.sub = nn.Identity()
            elif mutate_target == "in_place_version":
                with torch.no_grad():
                    self.w.add_(1.0)
            return x

    m = network(f"mut_{mutate_target}()", input_shape=("B", 4), output_shape=("B", 4), device="cpu", registry=registry)
    x = torch.randn(2, 4)

    with _compiling():
        # Call 1: Mutated during forward -> must report "changed during forward"
        with pytest.raises(HNDLError) as exc1:
            m(x)
        assert exc1.value.code == "E_RUNTIME"
        assert "changed during forward" in exc1.value.message
        assert "no longer matches the build" not in exc1.value.message

        # Call 2: Pre-forward check now catches the residual state -> must report "no longer matches the build"
        with pytest.raises(HNDLError) as exc2:
            m(x)
        assert exc2.value.code == "E_RUNTIME"
        assert "no longer matches the build" in exc2.value.message
        assert "changed during forward" not in exc2.value.message


# ==============================================================================
# M-09: Shallow Copy & Submodule In-Place Cast Cache Invalidation Stress Tests
# ==============================================================================

def test_shallow_copy_double_cast_clears_shared_cache():
    """M-09: Calling .double() on shallow copy clears validated cache and tracks dtype change."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)
    m(x)
    assert len(m._validated) == 1

    shallow = copy.copy(m)
    # Shallow copy initially shares the same _validated dict
    shallow.double()
    # Casting shallow copy must clear the shared dictionary in place
    assert len(m._validated) == 0

    # Shallow copy executes cleanly with float64
    x64 = torch.randn(2, 4, dtype=torch.float64)
    out64 = shallow(x64)
    assert out64.dtype == torch.float64

    # Original model detects parameter dtype change via _stores and avoids fast path
    try:
        m(x)
        pytest.fail("Original model should have rejected call after parameter dtype changed")
    except RuntimeError as exc:
        notes = getattr(exc, "__notes__", [])
        assert any("node 'n0'" in n for n in notes)


@pytest.mark.parametrize("cast_fn", ["half", "double"])
def test_shallow_copy_various_dtype_casts_invalidate_cache(cast_fn):
    """M-09: .half() and .double() on shallow copy clear validated cache."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)
    m(x)
    assert len(m._validated) == 1

    shallow = copy.copy(m)
    getattr(shallow, cast_fn)()
    assert len(m._validated) == 0


def test_submodule_direct_cast_leaf_node_invalidates_fast_path():
    """M-09: Casting child module directly (m[0].double()) triggers revalidation via _stores."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)
    m(x)
    assert len(m._validated) == 1

    # In-place cast child module
    m[0].double()

    # Next call with original input triggers re-validation and fails inside node with note
    with pytest.raises(RuntimeError) as exc:
        m(x)
    notes = getattr(exc.value, "__notes__", [])
    assert any("node 'n0'" in n for n in notes)


def test_submodule_direct_cast_intermediate_node_invalidates_fast_path():
    """M-09: Casting an intermediate node (m[2].double()) triggers revalidation and flags node 2."""
    m = network("linear(8)\nrelu()\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)
    m(x)
    assert len(m._validated) == 1

    # In-place cast last node
    m[2].double()

    # Node 0 produces float32, which reaches node 2 (now double) and fails
    with pytest.raises(RuntimeError) as exc:
        m(x)
    notes = getattr(exc.value, "__notes__", [])
    assert any("node 'n2'" in n for n in notes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_submodule_device_mutation_triggers_revalidation_and_fails_safely():
    """M-09: Moving a submodule to CUDA in-place triggers re-validation via _stores."""
    m = network("linear(8)\nlinear(4)", input_shape=("B", 4), output_shape=("B", 4), device="cpu")
    x = torch.randn(2, 4)
    m(x)
    assert len(m._validated) == 1

    # Move first module to CUDA
    m[0].to("cuda:0")

    # Next call with CPU input detects device mutation and fails with note
    with pytest.raises(RuntimeError) as exc:
        m(x)
    notes = getattr(exc.value, "__notes__", [])
    assert any("node 'n0'" in n for n in notes)


# ==============================================================================
# L-06: _state_bytes Storage Deduplication & Parity Stress Tests
# ==============================================================================

class MultiStorageModule(nn.Module):
    """Test module with shared storages, slices, strided views, and buffers."""

    def __init__(self, device):
        super().__init__()
        base = torch.empty(120, dtype=torch.float32, device=device)
        self.p1 = nn.Parameter(base[:60])
        self.p2 = nn.Parameter(base[60:].view(6, 10))
        self.p3 = nn.Parameter(base[::2])
        self.register_buffer("int_buf", torch.empty(20, dtype=torch.int64, device=device))
        self.register_buffer("bool_buf", torch.empty(10, dtype=torch.bool, device=device))


def test_state_bytes_meta_views_exact_deduplication():
    """L-06: Multiple parameters and views sharing one storage on meta device count storage once."""
    mod = MultiStorageModule("meta")
    # Base float32 storage: 120 elements * 4 bytes = 480 bytes
    # int_buf: 20 int64 elements * 8 bytes = 160 bytes
    # bool_buf: 10 bool elements * 1 byte = 10 bytes
    # Total = 480 + 160 + 10 = 650 bytes
    assert _state_bytes(mod) == 650


@pytest.mark.parametrize("compute_dtype,expected_float_scale", [
    (None, 1.0),
    (torch.float64, 2.0),   # 8 / 4 = 2.0
    (torch.float32, 1.0),   # 4 / 4 = 1.0
    (torch.float16, 0.5),   # 2 / 4 = 0.5
    (torch.bfloat16, 0.5),  # 2 / 4 = 0.5
])
def test_state_bytes_compute_dtype_scaling_exact(compute_dtype, expected_float_scale):
    """L-06: compute_dtype scales floating point storages while preserving non-float buffers."""
    mod = MultiStorageModule("meta")
    base_float_bytes = 480
    non_float_bytes = 170  # 160 + 10
    expected = int(base_float_bytes * expected_float_scale) + non_float_bytes
    assert _state_bytes(mod, compute_dtype=compute_dtype) == expected


def test_state_bytes_cross_device_meta_cpu_parity():
    """L-06: Meta and CPU modules with identical topologies produce identical byte counts."""
    mod_meta = MultiStorageModule("meta")
    mod_cpu = MultiStorageModule("cpu")

    for dt in (None, torch.float64, torch.float32, torch.float16, torch.bfloat16):
        meta_bytes = _state_bytes(mod_meta, compute_dtype=dt)
        cpu_bytes = _state_bytes(mod_cpu, compute_dtype=dt)
        assert meta_bytes == cpu_bytes, f"Mismatch at compute_dtype={dt}: meta={meta_bytes}, cpu={cpu_bytes}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_state_bytes_cross_device_meta_cuda_parity():
    """L-06: Meta and CUDA modules with identical topologies produce identical byte counts."""
    mod_meta = MultiStorageModule("meta")
    mod_cuda = MultiStorageModule("cuda:0")

    for dt in (None, torch.float64, torch.float32, torch.float16, torch.bfloat16):
        meta_bytes = _state_bytes(mod_meta, compute_dtype=dt)
        cuda_bytes = _state_bytes(mod_cuda, compute_dtype=dt)
        assert meta_bytes == cuda_bytes, f"Mismatch at compute_dtype={dt}: meta={meta_bytes}, cuda={cuda_bytes}"


def test_state_bytes_zero_element_and_transposed_tensors():
    """L-06: Zero-element tensors and transposed views are handled cleanly."""
    class EdgeCaseModule(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.empty_p = nn.Parameter(torch.empty(0, dtype=torch.float32, device=device))
            mat = torch.empty(10, 10, dtype=torch.float32, device=device)
            self.mat = nn.Parameter(mat)
            self.mat_t = nn.Parameter(mat.t())  # non-contiguous view

    mod_meta = EdgeCaseModule("meta")
    # 0 bytes + 100 float32 (400 bytes, deduplicated) = 400 bytes
    assert _state_bytes(mod_meta) == 400
    assert _state_bytes(mod_meta, compute_dtype=torch.float64) == 800
    assert _state_bytes(mod_meta, compute_dtype=torch.float16) == 200

    mod_cpu = EdgeCaseModule("cpu")
    assert _state_bytes(mod_cpu) == 400
    assert _state_bytes(mod_cpu, compute_dtype=torch.float64) == 800
    assert _state_bytes(mod_cpu, compute_dtype=torch.float16) == 200
