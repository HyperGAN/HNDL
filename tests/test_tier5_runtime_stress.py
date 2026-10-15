"""Adversarial stress test suite for Tier 5 runtime and operators hardening.

Validates:
1. PyTorch GraphModule execution, compilation caching, signature bounding, and
   shallow-copy submodule state mutation tracking (M-08, M-09, H-03).
2. Meta-device tensor allocation, storage deduplication, and parameter counting (L-06).
3. Checkpoint hashing across corrupted or missing safetensors and config files (M-01, M-02).
4. Numerical edge cases: RMSNorm float64 autograd gradcheck, RoPE extreme position
   indices and isometry, and MoE sparse gating boundary top-k combinations (H-06, M-12, M-13).
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, resolve
from hndl.operators.attention import Attention, _apply_rope, _rope_tables
from hndl.operators.moe import MixtureOfExperts, _reference as moe_ref
from hndl.operators.pos_embed import PosEmbed, _reference as pos_ref
from hndl.operators.rms_norm import RMSNorm, _reference as rms_ref
from hndl.pretrained import (
    LocalProvider,
    TransformersProvider,
    file_sha256,
    resolve_source,
)
from hndl.torch import (
    _MAX_SIGNATURES,
    _compiling,
    _is_chain,
    _state_bytes,
    build,
    network,
    parameter_counts,
)

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


# ==============================================================================
# 1. PyTorch GraphModule Execution & Compilation Caching
# ==============================================================================


class TestGraphModuleExecutionAndCompilationCaching:
    """Stress tests for GraphModule signature caching, compilation, and error handling."""

    def test_signature_cache_bounding_exceeds_max_signatures(self):
        """Executing with more than _MAX_SIGNATURES distinct batch sizes clears and bounds cache."""
        m = network("linear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        for b in range(1, 71):
            x = torch.randn(b, 8)
            out = m(x=x)
            assert out.shape == (b, 8)

        # After 64 entries, cache clears; with 70 total calls, 6 remain.
        assert len(m._validated) <= _MAX_SIGNATURES
        assert len(m._validated) == 6

    def test_runtime_dtype_cast_invalidates_compiled_shapes(self):
        """Casting a model with .double() updates _runtime_dtype and rejects old dtypes."""
        m = network("linear(8)\nrelu()", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        x_float = torch.randn(2, 8, dtype=torch.float32)
        out1 = m(x=x_float)
        assert out1.dtype == torch.float32

        # Cast to float64
        m.double()
        assert m._runtime_dtype == torch.float64
        x_double = torch.randn(2, 8, dtype=torch.float64)
        out2 = m(x=x_double)
        assert out2.dtype == torch.float64

        # Passing old float32 now fails validation
        with pytest.raises(HNDLError, match="expected dtype float64, got float32"):
            m(x=x_float)

    def test_runtime_device_tracking_parameterless_module(self):
        """A parameterless network tracks device moves via probe tensor."""
        m = network("relu()\nsigmoid()", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        assert m._runtime_device == torch.device("cpu")
        # Moving to cpu explicitly should remain cpu and clear _validated
        m.to("cpu")
        assert m._runtime_device == torch.device("cpu")
        assert len(m._validated) == 0

    def test_fast_program_multi_output_split_handling(self):
        """Networks with intermediate branches and multi-output operators run fast program cleanly."""
        src = "a, b = split(4)\nc = add(a, b)"
        m = network(src, input_shape=("B", 8), output_shape=("B", 4), device="cpu")
        x = torch.randn(3, 8)
        # Call multiple times to execute fast program
        for _ in range(5):
            out = m(x=x)
            assert out.shape == (3, 4)
            expected = x[:, :4] + x[:, 4:]
            torch.testing.assert_close(out, expected)

    def test_annotate_failure_preserves_exception_type_and_attaches_note(self):
        """Exceptions in a node propagate with their original type and an HNDL diagnostic note."""
        reg = Registry.builtins()

        @reg.operator("adversarial_fault", summary="Adversarial fault generator", shape="x[B, D] -> out[B, D]")
        class AdversarialFault(nn.Module):
            def __init__(self, *, D):
                super().__init__()

            def forward(self, x):
                raise ValueError("Injected runtime failure inside adversarial node")

        plan = resolve("adversarial_fault()", input_shape=("B", 8), output_shape=("B", 8), registry=reg)
        m = build(plan, device="cpu", registry=reg)
        with pytest.raises(ValueError, match="Injected runtime failure inside adversarial node") as exc_info:
            m(x=torch.randn(2, 8))

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("HNDL: raised inside node 'n0'" in note for note in notes)
        assert any("adversarial_fault" in note for note in notes)

    def test_simulated_compiling_catches_inline_parameter_mutation(self):
        """In-place modification of parameter version counter under simulated compiling triggers E_RUNTIME."""
        m = network("linear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        x = torch.randn(2, 8)
        m(x=x)

        with _compiling():
            with torch.no_grad():
                m.nodes.n_n0.weight.add_(1.0)
            with pytest.raises(HNDLError, match="was modified"):
                m(x=x)

    def test_port_error_diagnostics_for_scalar_and_negative_batch(self):
        """0-dimensional tensors and empty batch inputs produce informative E_RUNTIME messages."""
        m = network("linear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        with pytest.raises(HNDLError, match="got a 0-dimensional tensor"):
            m(x=torch.tensor(42.0))

        with pytest.raises(HNDLError, match=r"batch size must be positive, got shape \[0, 8\]"):
            m(x=torch.empty(0, 8))

    def test_binding_errors_extra_missing_duplicate_inputs(self):
        """Calling with extra, missing, or duplicate positional/keyword arguments raises E_BINDING."""
        m = network("linear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        x = torch.randn(2, 8)

        # Missing input
        with pytest.raises(HNDLError) as exc1:
            m()
        assert exc1.value.code == "E_BINDING"

        # Extra input
        with pytest.raises(HNDLError) as exc2:
            m(x=x, unexpected=torch.randn(2, 8))
        assert exc2.value.code == "E_BINDING"

        # Input supplied twice
        with pytest.raises(HNDLError, match="Input 'x' was supplied twice") as exc3:
            m(x, x=x)
        assert exc3.value.code == "E_BINDING"

    def test_single_output_tuple_and_variadic_chain_classification(self):
        """Multi-output operators and variadic operators are not classified as linear chains."""
        src = "a, b = chunk(2)\nconcat(a, b)"
        m = network(src, input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        assert not m._chain
        assert not _is_chain(m.plan)

        with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
            _ = m[0]
        with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
            _ = m[:1]
        with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
            _ = len(m)
        with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
            _ = iter(m)


# ==============================================================================
# 2. State Mutation & Shallow Copy Integrity
# ==============================================================================


class TestStateMutationAndShallowCopyIntegrity:
    """Stress tests for shallow copy state isolation and mutation detection."""

    def test_shallow_copy_submodule_parameter_deletion_detected(self):
        """Deleting a parameter on a shallow copy triggers E_RUNTIME on the original."""
        x = torch.randn(2, 8)
        m = network("linear(8)\nlinear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        m(x=x)

        m_copy = copy.copy(m)
        del m_copy.nodes.n_n1.bias
        with pytest.raises(HNDLError, match="parameter 'nodes.n_n1.bias' was removed"):
            m(x=x)

    def test_shallow_copy_submodule_parameter_set_to_none_detected(self):
        """Assigning None to a parameter on a shallow copy triggers E_RUNTIME."""
        x = torch.randn(2, 8)
        m = network("linear(8)\nlinear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        m(x=x)

        m_copy = copy.copy(m)
        m_copy.nodes.n_n0.bias = None
        with pytest.raises(HNDLError, match="parameter 'nodes.n_n0.bias' was set to None"):
            m(x=x)

    def test_shallow_copy_submodule_parameter_added_detected(self):
        """Adding a new parameter to a submodule on a shallow copy triggers E_RUNTIME."""
        x = torch.randn(2, 8)
        m = network("linear(8)\nlinear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        m(x=x)

        m_copy = copy.copy(m)
        m_copy.nodes.n_n0.extra_param = nn.Parameter(torch.randn(8))
        with pytest.raises(HNDLError, match="parameter 'nodes.n_n0.extra_param' was added"):
            m(x=x)

    def test_shallow_copy_submodule_buffer_added_detected(self):
        """Registering a new buffer on a submodule triggers E_RUNTIME."""
        x = torch.randn(2, 8)
        m = network("linear(8)\nlinear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        m(x=x)

        m_copy = copy.copy(m)
        m_copy.nodes.n_n0.register_buffer("running_stat", torch.zeros(8))
        with pytest.raises(HNDLError, match="buffer 'nodes.n_n0.running_stat' was added"):
            m(x=x)

    def test_shallow_copy_submodule_replacement_detected(self):
        """Directly replacing a submodule in _modules triggers E_RUNTIME."""
        x = torch.randn(2, 8)
        m = network("linear(8)\nlinear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        m(x=x)

        m_copy = copy.copy(m)
        m_copy.nodes._modules["n_n0"] = nn.Linear(8, 8)
        with pytest.raises(HNDLError, match=r"submodule 'nodes\.n_n0' was replaced"):
            m(x=x)

    def test_deepcopy_isolation(self):
        """A deepcopy creates an independent instance with fresh validation and stores."""
        x = torch.randn(2, 8)
        m = network("linear(8)\nlinear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        m(x=x)
        assert len(m._validated) == 1

        m_deep = copy.deepcopy(m)
        assert len(m_deep._validated) == 0
        assert m_deep.nodes.n_n0.weight is not m.nodes.n_n0.weight

        # Mutating deepcopy parameter values does not alter the original
        with torch.no_grad():
            m_deep.nodes.n_n0.weight.fill_(99.0)
        assert not torch.allclose(m.nodes.n_n0.weight, m_deep.nodes.n_n0.weight)

    def test_immutable_nodes_and_plan_attributes(self):
        """Assigning or deleting nodes, plan, or submodules in _NodeModules raises TypeError."""
        m = network("linear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        with pytest.raises(TypeError, match="HNDL architecture is fixed"):
            m.nodes = nn.ModuleDict()

        with pytest.raises(TypeError, match="HNDL architecture is fixed"):
            m.plan = None

        with pytest.raises(TypeError, match="HNDL architecture is fixed"):
            del m.nodes

        with pytest.raises(TypeError, match="HNDL architecture is fixed"):
            m.nodes.add_module("n_extra", nn.Identity())

        with pytest.raises(TypeError, match="HNDL architecture is fixed"):
            del m.nodes["n_n0"]

    def test_graphmodule_pickle_rejection(self):
        """GraphModule rejects pickle.dumps with instructions to use to_json and state_dict."""
        m = network("linear(8)", input_shape=("B", 8), output_shape=("B", 8), device="cpu")
        with pytest.raises(TypeError, match="cannot be pickled because its plan holds frozen mappings"):
            pickle.dumps(m)


# ==============================================================================
# 3. Meta-Device Tensor Allocation & Storage Deduplication
# ==============================================================================


class TestMetaDeviceAllocationAndStorageDeduplication:
    """Stress tests for parameter_counts and _state_bytes deduplication on meta device."""

    def test_parameter_counts_meta_allocation_diverse_ops(self):
        """parameter_counts matches instantiated parameters across varied operators."""
        src = "linear(16)\nrelu()\nrms_norm()\nflatten()"
        plan = resolve(src, input_shape=("B", 8), output_shape=("B", 16))
        counts = parameter_counts(plan)
        assert counts == {"n0": 144, "n1": 0, "n2": 16, "n3": 0}

        # Compare directly with built model
        m = network(src, input_shape=("B", 8), output_shape=("B", 16), device="cpu")
        for node_id, count in counts.items():
            actual = sum(p.numel() for p in m.nodes[f"n_{node_id}"].parameters())
            assert count == actual

    def test_state_bytes_meta_storage_deduplication(self):
        """Untyped storage with multiple transposed and sliced views is counted only once."""
        class SharedViewModule(nn.Module):
            def __init__(self):
                super().__init__()
                base = torch.empty(10, 10, device="meta", dtype=torch.float32)
                self.p1 = nn.Parameter(base)
                self.p2 = nn.Parameter(base.view(-1))
                self.p3 = nn.Parameter(base.t())

        mod = SharedViewModule()
        bytes_used = _state_bytes(mod)
        expected = 100 * 4  # 100 float32 elements = 400 bytes
        assert bytes_used == expected

    def test_state_bytes_compute_dtype_scaling_meta(self):
        """Floating parameters scale according to compute_dtype element size."""
        class LinearMeta(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(32, 32, device="meta", dtype=torch.float32))

        mod = LinearMeta()
        num_elements = 32 * 32
        bytes_f32 = _state_bytes(mod, compute_dtype=torch.float32)
        bytes_f16 = _state_bytes(mod, compute_dtype=torch.float16)
        bytes_f64 = _state_bytes(mod, compute_dtype=torch.float64)

        assert bytes_f32 == num_elements * 4
        assert bytes_f16 == num_elements * 2
        assert bytes_f64 == num_elements * 8
        assert bytes_f64 == bytes_f16 * 4

    def test_max_state_bytes_limit_exceeded_before_allocation(self):
        """Allocation-free probe rejects oversized plans with E_RESOURCE prior to allocation."""
        plan = resolve("linear(1000)\nlinear(1000)", input_shape=("B", 1000), output_shape=("B", 1000))
        with pytest.raises(HNDLError, match="Plan exceeds max_state_bytes before allocation") as exc_info:
            build(plan, device="cpu", limits={"max_state_bytes": 1024})
        assert exc_info.value.code == "E_RESOURCE"

    def test_parameter_settings_invalid_targets(self):
        """Initialization or trainability overrides targeting missing parameters or buffers raise errors."""
        reg = Registry.builtins()

        @reg.operator("buffer_op", summary="Buffer test operator", shape="x[B, D] -> out[B, D]")
        class BufferOp(nn.Module):
            def __init__(self, *, D):
                super().__init__()
                self.register_buffer("buf", torch.zeros(D))
                self.w = nn.Parameter(torch.ones(D))

            def forward(self, x):
                return x * self.w + self.buf

        # Target that does not exist
        plan1 = resolve('linear(8, init={"missing_param": 0.0})', input_shape=("B", 8), output_shape=("B", 8))
        with pytest.raises(HNDLError, match="is not a registered parameter") as exc1:
            build(plan1, device="cpu")
        assert exc1.value.code == "E_INITIALIZATION"

        # Target that is a buffer
        plan2 = resolve('buffer_op(init={"buf": 0.0})', input_shape=("B", 8), output_shape=("B", 8), registry=reg)
        with pytest.raises(HNDLError, match="is a buffer, not a parameter") as exc2:
            build(plan2, device="cpu", registry=reg)
        assert exc2.value.code == "E_INITIALIZATION"

    def test_matrix_scheme_on_vector_parameter_rejected(self):
        """Matrix initialization schemes like xavier_uniform on 1D biases raise E_INITIALIZATION."""
        plan = resolve('linear(8, init={"bias": xavier_uniform()})', input_shape=("B", 8), output_shape=("B", 8))
        with pytest.raises(HNDLError, match="requires a parameter of at least two dimensions") as exc:
            build(plan, device="cpu")
        assert exc.value.code == "E_INITIALIZATION"

    def test_trainable_non_floating_point_parameter_rejected(self):
        """Setting trainable=True on an integer parameter raises E_TRAINABILITY."""
        reg = Registry.builtins()

        @reg.operator("int_param_op", summary="Int param operator", shape="x[B, D] -> out[B, D]")
        class IntParamOp(nn.Module):
            def __init__(self, *, D):
                super().__init__()
                self.int_param = nn.Parameter(torch.zeros(D, dtype=torch.int64), requires_grad=False)

            def forward(self, x):
                return x + self.int_param.float()

        plan = resolve('int_param_op(trainable={"int_param": True})', input_shape=("B", 8), output_shape=("B", 8), registry=reg)
        with pytest.raises(HNDLError, match="cannot require gradients") as exc:
            build(plan, device="cpu", registry=reg)
        assert exc.value.code == "E_TRAINABILITY"


# ==============================================================================
# 4. Checkpoint Hashing Across Corrupted or Missing Files
# ==============================================================================


class TestCheckpointHashingCorruptedSafetensors:
    """Stress tests for checkpoint digesting, missing files, and corruption handling."""

    def test_file_sha256_nonexistent_file_raises_pretrained_error(self, tmp_path):
        """Missing checkpoint files wrap stat errors in E_PRETRAINED."""
        with pytest.raises(HNDLError, match="Cannot stat checkpoint file") as exc:
            file_sha256(tmp_path / "nonexistent.safetensors")
        assert exc.value.code == "E_PRETRAINED"

    def test_resolve_source_corrupted_safetensors_file(self, tmp_path):
        """Corrupted safetensors bytes raise E_PRETRAINED during weight loading."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"model_type": "gpt2"}')
        corrupt_st = tmp_path / "corrupt.safetensors"
        corrupt_st.write_bytes(b"CORRUPTED_NOT_A_VALID_SAFETENSORS_HEADER")

        source = resolve_source(str(corrupt_st), config_path=str(cfg_file))
        assert source.kind == "file"
        provider = TransformersProvider(source, "logits", "")
        with pytest.raises(HNDLError, match="Cannot load weights from") as exc:
            provider.instantiate(weights=True)
        assert exc.value.code == "E_PRETRAINED"

    def test_resolve_source_corrupted_config_json(self, tmp_path):
        """Truncated or invalid JSON in config.json raises E_PRETRAINED."""
        bad_cfg = tmp_path / "config.json"
        bad_cfg.write_text('{"model_type": "gpt2", unclosed')
        with pytest.raises(HNDLError, match="Cannot read config") as exc:
            resolve_source(str(tmp_path))
        assert exc.value.code == "E_PRETRAINED"

    def test_resolve_source_bare_safetensors_missing_config_arg(self, tmp_path):
        """A bare .safetensors file without config= raises E_PRETRAINED."""
        sf = tmp_path / "model.safetensors"
        sf.write_bytes(b"empty")
        with pytest.raises(HNDLError, match=r"needs config=<path to config\.json>") as exc:
            resolve_source(str(sf))
        assert exc.value.code == "E_PRETRAINED"

    def test_state_dict_sha256_digest_verification_failure(self, tmp_path):
        """State dict loading with mismatched sha256 checksum raises E_PRETRAINED."""
        pth_file = tmp_path / "weights.pth"
        torch.save({"w": torch.randn(4)}, pth_file)
        fake_sha = "0" * 64

        reg = Registry.builtins()
        reg.pretrained_provider("custom_arch", lambda: nn.Linear(4, 4))
        with reg.activated():
            with pytest.raises(HNDLError, match="the checkpoint file changed") as exc:
                resolve_source(str(pth_file), provider="custom_arch", checksum=fake_sha)
            assert exc.value.code == "E_PRETRAINED"

    def test_state_dict_corrupted_weights_raises_pretrained_error(self, tmp_path):
        """Corrupted .pth file with matching sha256 raises E_PRETRAINED on instantiate."""
        corrupt_pth = tmp_path / "corrupt.pth"
        content = b"INVALID_PICKLE_STREAM"
        corrupt_pth.write_bytes(content)
        actual_sha = hashlib.sha256(content).hexdigest()

        reg = Registry.builtins()
        reg.pretrained_provider("custom_arch2", lambda: nn.Linear(4, 4))
        with reg.activated():
            source = resolve_source(str(corrupt_pth), provider="custom_arch2", checksum=actual_sha)
            loc_prov = LocalProvider(source, "features", "", "")
            with pytest.raises(HNDLError, match="Cannot load weights from") as exc:
                loc_prov.instantiate(weights=True)
            assert exc.value.code == "E_PRETRAINED"


# ==============================================================================
# 5. Numerical Edge Cases & Autograd Precision
# ==============================================================================


class TestNumericalEdgeCasesAndAutograd:
    """Stress tests for float64 autograd, extreme RoPE positions, and MoE boundary combinations."""

    def test_rmsnorm_float64_autograd_gradcheck(self):
        """torch.autograd.gradcheck verifies analytical float64 gradients for RMSNorm."""
        norm_affine = RMSNorm(eps=1e-5, affine=True, D=8).double()
        x1 = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(norm_affine, (x1,), eps=1e-6, atol=1e-4, rtol=1e-3)

        norm_no_affine = RMSNorm(eps=1e-5, affine=False, D=8).double()
        x2 = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(norm_no_affine, (x2,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_rmsnorm_float64_extreme_magnitudes_precision(self):
        """RMSNorm maintains double precision and numerical stability across extreme dynamic ranges."""
        norm = RMSNorm(eps=1e-6, affine=True, D=4).double()
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64))

        ref_fn = rms_ref(norm)
        # Extreme scales from 1e-12 to 1e12
        for scale in [1e-12, 1e-6, 1.0, 1e6, 1e12]:
            x = (torch.randn(3, 4, dtype=torch.float64) + 1.0) * scale
            out = norm(x)
            ref = ref_fn(x)
            assert out.dtype == torch.float64
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
            torch.testing.assert_close(out, ref, rtol=1e-13, atol=1e-13)

    def test_rope_extreme_sequence_length_isometry(self):
        """RoPE rotary embeddings preserve exact isometric norm at extreme sequence lengths."""
        head_dim = 16
        for T in [1, 1024, 65536, 131072]:
            cos, sin = _rope_tables(T, head_dim, "cpu", torch.float64)
            assert cos.shape == (1, 1, T, head_dim)
            assert sin.shape == (1, 1, T, head_dim)
            assert cos.dtype == torch.float64
            assert sin.dtype == torch.float64

            # Verify trigonometric identity: cos^2 + sin^2 == 1
            half = head_dim // 2
            trig_sum = cos[..., :half] ** 2 + sin[..., :half] ** 2
            torch.testing.assert_close(trig_sum, torch.ones_like(trig_sum), atol=1e-14, rtol=1e-14)

            # Isometry test: unitary rotation preserves L2 norm
            q = torch.randn(2, 2, T, head_dim, dtype=torch.float64)
            q_rot = _apply_rope(q, cos, sin)
            norm_orig = torch.linalg.norm(q, dim=-1)
            norm_rot = torch.linalg.norm(q_rot, dim=-1)
            torch.testing.assert_close(norm_orig, norm_rot, atol=1e-13, rtol=1e-13)

    def test_rope_head_dim_boundary_validation(self):
        """Odd head_dim with rope=True raises E_CONSTRAINT."""
        # D=10, heads=4 -> head_dim = 10 // 4 = 2 (valid)
        att = Attention(heads=4, causal=False, dropout=0.0, bias=True, rope=True, D=8)
        assert att.head_dim == 2

        # D=9, heads=3 -> head_dim = 3 (odd, invalid)
        with pytest.raises(HNDLError, match="rope needs an even head dimension") as exc:
            Attention(heads=3, causal=False, dropout=0.0, bias=True, rope=True, D=9)
        assert exc.value.code == "E_CONSTRAINT"

    @pytest.mark.parametrize("experts,top_k", [
        (1, 1),  # minimal single expert
        (4, 1),  # hard routing
        (4, 4),  # dense mixture
        (6, 2),  # typical sparse gating
        (8, 3),  # multi-expert selection
    ])
    def test_moe_sparse_gating_boundary_topk_combinations(self, experts, top_k):
        """MoE outputs match dense reference oracle across boundary top-k combinations."""
        moe = MixtureOfExperts(experts=experts, hidden=16, top_k=top_k, activation="gelu", D=8)
        ref_fn = moe_ref(moe)

        x = torch.randn(4, 8)
        out = moe(x)
        ref = ref_fn(x)

        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_moe_zero_token_expert_execution(self):
        """When an expert receives zero routed tokens, execution succeeds without indexing errors."""
        moe = MixtureOfExperts(experts=4, hidden=16, top_k=1, activation="gelu", D=8)
        # Heavily bias router to expert 0 so experts 1, 2, 3 receive 0 tokens
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0, :] = 100.0
            moe.router.weight[1:, :] = -100.0

        x = torch.randn(5, 8)
        out = moe(x)
        assert out.shape == (5, 8)
        assert not torch.isnan(out).any()

    def test_moe_autograd_gradient_flow(self):
        """MoE backward pass computes valid non-zero gradients for router and expert parameters."""
        moe = MixtureOfExperts(experts=3, hidden=16, top_k=2, activation="silu", D=8)
        x = torch.randn(4, 8, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert moe.router.weight.grad is not None
        for expert in moe.experts:
            assert expert.up.weight.grad is not None
            assert expert.up.bias.grad is not None
            assert expert.down.weight.grad is not None
            assert expert.down.bias.grad is not None

    def test_posembed_sequence_length_boundary_enforcement(self):
        """PosEmbed validates T <= max_len and matches reference at boundary lengths."""
        pos = PosEmbed(max_len=8, D=16)
        ref_fn = pos_ref(pos)

        # T == 1 (minimal length)
        x1 = torch.randn(2, 1, 16)
        torch.testing.assert_close(pos(x1), ref_fn(x1))

        # T == max_len (boundary length)
        x8 = torch.randn(2, 8, 16)
        torch.testing.assert_close(pos(x8), ref_fn(x8))

        # T == max_len + 1 (exceeded) raises E_CONSTRAINT
        x9 = torch.randn(2, 9, 16)
        with pytest.raises(HNDLError, match="Sequence length T=9 exceeds max_len=8") as exc:
            pos(x9)
        assert exc.value.code == "E_CONSTRAINT"
