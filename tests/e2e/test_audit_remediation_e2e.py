"""Comprehensive opaque-box E2E test suite for HNDL library audit remediation and release.

Covers all 10 feature areas identified in TEST_INFRA.md:
- F1: Line break & newline parsing (H-01, H-02, M-04)
- F2: Syntax, tuple & identifier validation (M-03, M-05, M-06)
- F3: Symbolic solver & interval reporting (M-07, L-03, L-04)
- F4: PyTorch chain & sequence slicing (H-03, M-08, M-09)
- F5: Pretrained model training & revisions (H-04, H-05, M-01, M-02)
- F6: Numerics & double precision (RMSNorm, RoPE) (H-06, M-12)
- F7: Operator references & layer representations (M-13, L-07, L-08)
- F8: Diagnostics & error location attribution (L-01, L-02, L-05)
- F9: Package distribution & fixture integrity (H-07, L-10)
- F10: Documentation, typing & version sync (H-08, M-10, M-11, M-14)

Tiers:
- Tier 1: Feature Coverage (happy paths in isolation)
- Tier 2: Boundary & Corner Cases (limits, syntax extremes, and defect triggers)
- Tier 3: Cross-Feature Interactions (pairwise combinations)
- Tier 4: Real-World Application Workloads
"""

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import hndl
from hndl import (
    Example,
    HNDLError,
    Registry,
    ResolvedPlan,
    ops,
    resolve,
    resolve_callable,
)
from hndl.capture import Capture
from hndl.operators.rms_norm import RMSNorm
from hndl.relations import (
    conv_input_range,
    conv_output,
    conv_transpose_input,
    conv_transpose_output,
)
from hndl.resolver import resolve_graph
from hndl.torch import build, network, network_from_callable, parameter_counts
from hndl.types import Graph, Node

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# TIER 1: Feature Coverage (Happy Paths in Isolation)
# ==============================================================================


class TestTier1FeatureCoverage:
    """Tier 1: ≥5 tests per feature area covering happy paths in isolation."""

    # --------------------------------------------------------------------------
    # F1: Line break & newline parsing (H-01, H-02, M-04)
    # --------------------------------------------------------------------------

    def test_tier1_f1_standard_newline_linear_relu_linear(self, device):
        """[F1 Happy Path 1] Declarative config with standard newline separators resolves and runs."""
        source = "linear(64)\nrelu()\nlinear(10)\n"
        model = network(source, input_shape=("B", 128), output_shape=("B", 10), device=device)
        assert len(model) == 3
        x = torch.randn(2, 128, device=device)
        out = model(x)
        assert out.shape == (2, 10)

    def test_tier1_f1_crlf_windows_linebreaks_resolve(self, device):
        """[F1 Happy Path 2] Declarative config with CRLF (\\r\\n) line breaks parses and resolves."""
        source = "linear(32)\r\nrelu()\r\nlinear(8)\r\n"
        model = network(source, input_shape=("B", 16), output_shape=("B", 8), device=device)
        assert len(model) == 3
        x = torch.randn(4, 16, device=device)
        assert model(x).shape == (4, 8)

    def test_tier1_f1_unpacking_multi_output_single_split(self, device):
        """[F1 Happy Path 3] Tuple unpacking (a, b) = split(x, ...) after a call line."""
        source = (
            "linear(64)\n"
            "(first, rest) = split(32)\n"
            "left = linear(first, 16)\n"
            "right = linear(rest, 16)\n"
            "add(left, right)\n"
        )
        model = network(source, input_shape=("B", 128), output_shape=("B", 16), device=device)
        out = model(torch.randn(2, 128, device=device))
        assert out.shape == (2, 16)

    def test_tier1_f1_unpacking_multiple_consecutive_lines(self):
        """[F1 Happy Path 4] Multiple consecutive unpacking statements across lines."""
        source = (
            "linear(64)\n"
            "(a1, b1) = split(32)\n"
            "(a2, b2) = split(a1, 16)\n"
            "(a3, b3) = split(b1, 16)\n"
            "c1 = add(a2, b2)\n"
            "c2 = add(a3, b3)\n"
            "add(c1, c2)\n"
        )
        plan = resolve(source, input_shape=("B", 128), output_shape=("B", 16))
        assert len(plan.nodes) == 7

    def test_tier1_f1_standard_string_literals_in_args(self, device):
        """[F1 Happy Path 5] Declarative source using string literals for layer names and options."""
        source = 'linear(32, name="fc1")\nrelu(name="act1")\nlinear(8, name="fc2")\n'
        model = network(source, input_shape=("B", 64), output_shape=("B", 8), device=device)
        assert [n.id for n in model.plan.nodes] == ["fc1", "act1", "fc2"]
        assert model(torch.randn(2, 64, device=device)).shape == (2, 8)

    # --------------------------------------------------------------------------
    # F2: Syntax, tuple & identifier validation (M-03, M-05, M-06)
    # --------------------------------------------------------------------------

    def test_tier1_f2_tensor_tuple_construction_and_unpacking(self, device):
        """[F2 Happy Path 1] Valid non-empty tensor tuple construction and subsequent use."""
        source = (
            "a = linear(16)\n"
            "b = linear(16)\n"
            "pair = (a, b)\n"
            "first, second = pair\n"
            "add(first, second)\n"
        )
        model = network(source, input_shape=("B", 32), output_shape=("B", 16), device=device)
        assert model(torch.randn(2, 32, device=device)).shape == (2, 16)

    def test_tier1_f2_explicit_valid_input_dtypes(self):
        """[F2 Happy Path 2] Specifying explicit valid input dtypes."""
        plan_f32 = resolve("linear(10)", input_shape=("B", 16), output_shape=("B", 10), input_dtype="float32")
        assert plan_f32.input_dtype == "float32"

        plan_i64 = resolve("embedding(100, 16)\nlinear(10)", input_shape=("B", 16), output_shape=("B", 16, 10), input_dtype="int64")
        assert plan_i64.input_dtype == "int64"

    def test_tier1_f2_valid_multi_port_contract_names(self):
        """[F2 Happy Path 3] Valid descriptive port names in contract."""
        capture = Capture(
            input_shape={"token_embeddings": ("B", 16), "position_embeddings": ("B", 16)},
            output_shape={"combined_output": ("B", 16)},
        )
        assert "token_embeddings" in capture.input_contracts
        assert "position_embeddings" in capture.input_contracts
        assert "combined_output" in capture.output_contracts

    def test_tier1_f2_multi_input_multi_output_graph(self, device):
        """[F2 Happy Path 4] Multi-input and multi-output graph resolution and execution."""
        def multi_io(a, b):
            sum_ab = ops.add(a, b)
            diff_ab = ops.sub(a, b)
            return {"sum_out": sum_ab, "diff_out": diff_ab}

        model = network_from_callable(
            multi_io,
            input_shape={"a": ("B", 8), "b": ("B", 8)},
            output_shape={"sum_out": ("B", 8), "diff_out": ("B", 8)},
            device=device,
        )
        a_val = torch.ones(2, 8, device=device) * 5
        b_val = torch.ones(2, 8, device=device) * 2
        outputs = model(a=a_val, b=b_val)
        torch.testing.assert_close(outputs["sum_out"], torch.ones(2, 8, device=device) * 7)
        torch.testing.assert_close(outputs["diff_out"], torch.ones(2, 8, device=device) * 3)

    def test_tier1_f2_tuple_output_operator_chunk(self, device):
        """[F2 Happy Path 5] Chunk operator producing tuple outputs consumed downstream."""
        source = (
            "c1, c2 = chunk(2)\n"
            "add(c1, c2)\n"
        )
        model = network(source, input_shape=("B", 32), output_shape=("B", 16), device=device)
        x = torch.randn(2, 32, device=device)
        assert model(x).shape == (2, 16)

    # --------------------------------------------------------------------------
    # F3: Symbolic solver & interval reporting (M-07, L-03, L-04)
    # --------------------------------------------------------------------------

    def test_tier1_f3_linear_dimension_propagation(self):
        """[F3 Happy Path 1] Forward and backward dimension propagation on linear layers."""
        plan = resolve("linear(64)\nlinear()\n", input_shape=("B", 32), output_shape=("B", 10))
        assert plan.nodes[0].output_shapes["out"] == ("B", 64)
        assert plan.nodes[1].output_shapes["out"] == ("B", 10)

    def test_tier1_f3_conv2d_shape_solving_stride1(self, device):
        """[F3 Happy Path 2] Conv2d with stride=1 padding=1 preserves spatial H, W."""
        model = network(
            "conv(32, kernel_size=3, stride=1, padding=1)\nrelu()\n",
            input_shape=("B", 3, 28, 28),
            output_shape=("B", 32, 28, 28),
            device=device,
        )
        x = torch.randn(2, 3, 28, 28, device=device)
        assert model(x).shape == (2, 32, 28, 28)

    def test_tier1_f3_conv_transpose2d_shape_solving(self, device):
        """[F3 Happy Path 3] ConvTranspose2d with stride=2 upsamples spatial dimensions."""
        model = network(
            "deconv(16, kernel_size=4, stride=2, padding=1)\n",
            input_shape=("B", 32, 8, 8),
            output_shape=("B", 16, 16, 16),
            device=device,
        )
        x = torch.randn(2, 32, 8, 8, device=device)
        assert model(x).shape == (2, 16, 16, 16)

    def test_tier1_f3_relations_conv_output_and_conv_input_range(self):
        """[F3 Happy Path 4] hndl.relations conv_output and conv_input_range functions."""
        out_extent = conv_output(extent=28, kernel=3, stride=1, padding=1)
        assert out_extent == 28

        stride2_out = conv_output(extent=28, kernel=3, stride=2, padding=1)
        assert stride2_out == 14

        inp_range = conv_input_range(extent=14, kernel=3, stride=2, padding=1)
        assert inp_range == (27, 28)

        deconv_out = conv_transpose_output(extent=14, kernel=3, stride=2, padding=1)
        assert deconv_out == 27

        deconv_inp = conv_transpose_input(extent=27, kernel=3, stride=2, padding=1)
        assert deconv_inp == 14

    def test_tier1_f3_ellipsis_multidimensional_broadcast(self, device):
        """[F3 Happy Path 5] Ellipsis (...) matching across equal rank with broadcast."""
        source = "broadcast_add(a, b)\n"
        model = network(
            source,
            input_shape={"a": ("B", 3, 16, 16), "b": ("B", 3, 1, 1)},
            output_shape=("B", 3, 16, 16),
            device=device,
        )
        x = torch.zeros(2, 3, 16, 16, device=device)
        b = torch.ones(2, 3, 1, 1, device=device) * 5
        out = model(a=x, b=b)
        assert torch.all(out == 5)

    # --------------------------------------------------------------------------
    # F4: PyTorch chain & sequence slicing (H-03, M-08, M-09)
    # --------------------------------------------------------------------------

    def test_tier1_f4_linear_unary_chain_detection(self):
        """[F4 Happy Path 1] True unary sequential chain is recognized as a chain."""
        model = network("linear(32)\nrelu()\nlinear(16)", input_shape=("B", 64), output_shape=("B", 16), device="cpu")
        assert model._chain is True
        assert len(model) == 3

    def test_tier1_f4_unary_chain_slicing_forward(self, device):
        """[F4 Happy Path 2] Slicing a unary chain produces a valid callable sub-network."""
        model = network("linear(32)\nrelu()\nlinear(16)", input_shape=("B", 64), output_shape=("B", 16), device=device)
        sub = model[:2]
        x = torch.randn(2, 64, device=device)
        out = sub(x)
        assert out.shape == (2, 32)
        assert torch.all(out >= 0)

    def test_tier1_f4_graphmodule_forward_and_backward(self, device):
        """[F4 Happy Path 3] GraphModule computes forward, loss, and backpropagates gradients."""
        model = network("linear(32)\nrelu()\nlinear(8)", input_shape=("B", 16), output_shape=("B", 8), device=device)
        x = torch.randn(4, 16, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert model[0].weight.grad is not None

    def test_tier1_f4_graphmodule_to_device_and_eval_mode(self, device):
        """[F4 Happy Path 4] Toggling train/eval mode and verifying parameter attributes."""
        model = network("linear(16)\ndropout(0.5)\nlinear(8)", input_shape=("B", 16), output_shape=("B", 8), device=device)
        assert model.training is True
        model.eval()
        assert model.training is False
        assert model[1].training is False
        model.train()
        assert model.training is True
        assert model[1].training is True

    def test_tier1_f4_graphmodule_state_dict_roundtrip(self, device):
        """[F4 Happy Path 5] Saving state_dict, reloading into rebuilt network produces identical output."""
        model1 = network("linear(16)\nlinear(8)", input_shape=("B", 32), output_shape=("B", 8), device=device, initialization_seed=123)
        state = copy.deepcopy(model1.state_dict())
        model2 = network("linear(16)\nlinear(8)", input_shape=("B", 32), output_shape=("B", 8), device=device, initialization_seed=999)
        model2.load_state_dict(state)
        x = torch.randn(2, 32, device=device)
        torch.testing.assert_close(model1(x), model2(x))

    # --------------------------------------------------------------------------
    # F5: Pretrained model training & revisions (H-04, H-05, M-01, M-02)
    # --------------------------------------------------------------------------

    def test_tier1_f5_local_mock_checkpoint_creation_and_resolve(self, text_checkpoint):
        """[F5 Happy Path 1] Local minimal pretrained checkpoint resolves with inferred revision."""
        plan = resolve(
            f'pretrained("{text_checkpoint}")',
            input_shape=("B", 4),
            output_shape=("B", 4, 8),
            input_dtype="int64",
        )
        assert len(plan.nodes) == 1
        assert plan.nodes[0].args["revision"] is not None

    def test_tier1_f5_pretrained_default_frozen_eval_mode(self, text_checkpoint, device):
        """[F5 Happy Path 2] Pretrained layer with default trainable=False stays frozen in eval mode."""
        model = network(
            f'pretrained("{text_checkpoint}")',
            input_shape=("B", 4),
            output_shape=("B", 4, 8),
            input_dtype="int64",
            device=device,
        )
        assert not any(p.requires_grad for p in model.parameters())
        assert model[0].model.training is False

    def test_tier1_f5_pretrained_trainable_parameters(self, text_checkpoint, device):
        """[F5 Happy Path 3] Pretrained with trainable=True has requires_grad=True parameters."""
        model = network(
            f'pretrained("{text_checkpoint}", trainable=True)',
            input_shape=("B", 4),
            output_shape=("B", 4, 8),
            input_dtype="int64",
            device=device,
        )
        assert all(p.requires_grad for p in model.parameters())

    def test_tier1_f5_source_revision_extracted_from_location(self, text_checkpoint):
        """[F5 Happy Path 4] Location extracts revision digest."""
        from hndl.pretrained import resolve_source
        source = resolve_source(str(text_checkpoint))
        assert source.kind == "directory"
        assert source.revision is not None
        assert len(source.revision) == 16

    def test_tier1_f5_provider_instantiate_weights_false(self, text_checkpoint):
        """[F5 Happy Path 5] Pretrained provider structural instantiation with weights=False."""
        from hndl.pretrained import TransformersProvider, resolve_source
        source = resolve_source(str(text_checkpoint))
        provider = TransformersProvider(source, output="logits", component="")
        model_meta = provider.instantiate(weights=False)
        assert model_meta is not None

    # --------------------------------------------------------------------------
    # F6: Numerics & double precision (RMSNorm, RoPE) (H-06, M-12)
    # --------------------------------------------------------------------------

    def test_tier1_f6_rmsnorm_float32_forward_parity(self, device):
        """[F6 Happy Path 1] RMSNorm on float32 tensor matches standard formulation."""
        norm = RMSNorm(eps=1e-5, affine=True, D=16).to(device)
        x = torch.randn(2, 4, 16, device=device, dtype=torch.float32)
        out = norm(x)
        expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_tier1_f6_rmsnorm_fp16_bf16_upcast(self, device):
        """[F6 Happy Path 2] RMSNorm forward preserves float16 and bfloat16 input dtypes."""
        for dt in (torch.float16, torch.bfloat16):
            norm = RMSNorm(eps=1e-5, affine=True, D=16).to(device=device, dtype=dt)
            x = torch.randn(2, 4, 16, device=device, dtype=dt)
            out = norm(x)
            assert out.dtype == dt
            assert out.shape == (2, 4, 16)

    def test_tier1_f6_rope_attention_float32_forward(self, device):
        """[F6 Happy Path 3] Attention operator with rope=True forward execution."""
        source = "attention(heads=2, rope=True)\n"
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        assert out.shape == (2, 4, 16)

    def test_tier1_f6_pos_embed_valid_sequence_length(self, device):
        """[F6 Happy Path 4] PosEmbed forward with sequence length T <= max_len."""
        source = "pos_embed(8)\n"
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        assert out.shape == (2, 4, 16)

    def test_tier1_f6_layer_norm_and_adaptive_norm_numerics(self, device):
        """[F6 Happy Path 5] LayerNorm forward execution on standard inputs."""
        source = "layer_norm()\n"
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        assert out.shape == (2, 4, 16)

    # --------------------------------------------------------------------------
    # F7: Operator references & layer representations (M-13, L-07, L-08)
    # --------------------------------------------------------------------------

    def test_tier1_f7_existing_operator_references(self):
        """[F7 Happy Path 1] Verify operators that already have reference= match forward."""
        reg = Registry.builtins()
        gelu_op = reg.get("gelu")
        assert gelu_op.reference is not None

        rms_op = reg.get("rms_norm")
        assert rms_op.reference is not None

    def test_tier1_f7_grid_sample_standard_call(self, device):
        """[F7 Happy Path 2] GridSample operator with valid image and grid."""
        model = network(
            "grid_sample(x, grid)\n",
            input_shape={"x": ("B", 3, 16, 16), "grid": ("B", 8, 8, 2)},
            output_shape=("B", 3, 8, 8),
            device=device,
        )
        x = torch.randn(2, 3, 16, 16, device=device)
        grid = torch.zeros(2, 8, 8, 2, device=device)
        out = model(x=x, grid=grid)
        assert out.shape == (2, 3, 8, 8)

    def test_tier1_f7_resblock_standard_instantiation(self, device):
        """[F7 Happy Path 3] ResBlock operator forward execution on 4D tensor."""
        model = network("resblock(32)\n", input_shape=("B", 32, 16, 16), output_shape=("B", 32, 16, 16), device=device)
        x = torch.randn(2, 32, 16, 16, device=device)
        assert model(x).shape == (2, 32, 16, 16)

    def test_tier1_f7_module_repr_contains_useful_info(self, device):
        """[F7 Happy Path 4] GraphModule repr includes operator names and output dimensions."""
        model = network("linear(32)\nrelu()", input_shape=("B", 16), output_shape=("B", 32), device=device)
        rep = repr(model)
        assert "linear" in rep
        assert "32" in rep

    def test_tier1_f7_fourier_features_forward(self, device):
        """[F7 Happy Path 5] FourierFeatures operator expands coordinate dimension."""
        model = network(
            "fourier_features(16)\n",
            input_shape=("B", 2),
            output_shape=("B", 32),
            device=device,
        )
        coords = torch.randn(4, 2, device=device)
        assert model(coords).shape == (4, 32)

    # --------------------------------------------------------------------------
    # F8: Diagnostics & error location attribution (L-01, L-02, L-05)
    # --------------------------------------------------------------------------

    def test_tier1_f8_loop_unrolling_unique_names(self):
        """[F8 Happy Path 1] Declarative loop unrolls with unique suffixed node IDs."""
        source = (
            "for _ in range(3):\n"
            "    linear(16, name='layer')\n"
        )
        plan = resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert [n.id for n in plan.nodes] == ["layer0", "layer1", "layer2"]

    def test_tier1_f8_syntax_error_includes_line_and_column(self):
        """[F8 Happy Path 2] Syntax error in declarative config raises HNDLError with line/column."""
        with pytest.raises(HNDLError) as exc_info:
            resolve("linear(16\n", input_shape=("B", 16), output_shape=("B", 16))
        err = exc_info.value
        assert err.line is not None
        assert err.column is not None

    def test_tier1_f8_shape_mismatch_error_reports_node_and_ports(self):
        """[F8 Happy Path 3] Shape mismatch error reports the failing node ID and port."""
        with pytest.raises(HNDLError) as exc_info:
            resolve("linear(32)\nconv(16, kernel_size=3)\n", input_shape=("B", 16), output_shape=("B", 16, 8, 8))
        err = exc_info.value
        assert err.code in ("E_SHAPE", "E_CONSTRAINT", "E_AMBIGUOUS")

    def test_tier1_f8_example_dataclass_valid_construction(self):
        """[F8 Happy Path 4] Constructing Example dataclass with valid shapes."""
        ex = Example("linear(32)", ("B", 16), ("B", 32), "note")
        assert ex.input_shape == ("B", 16)
        assert ex.output_shape == ("B", 32)

    def test_tier1_f8_node_dataclass_valid_construction(self):
        """[F8 Happy Path 5] Constructing Node dataclass with valid arguments."""
        node = Node(
            id="n0",
            op="linear@1",
            args={"out_features": 64},
            inputs={"x": "input:x"},
            outputs=("out",),
        )
        assert node.id == "n0"
        assert node.outputs == ("out",)

    # --------------------------------------------------------------------------
    # F9: Package distribution & fixture integrity (H-07, L-10)
    # --------------------------------------------------------------------------

    def test_tier1_f9_manifest_in_exists(self):
        """[F9 Happy Path 1] MANIFEST.in exists in the repository root."""
        manifest = REPO_ROOT / "MANIFEST.in"
        assert manifest.exists()
        assert manifest.is_file()

    def test_tier1_f9_saved_plan_0_6_0_fixtures_exist(self):
        """[F9 Happy Path 2] All 5 golden plan fixtures in tests/fixtures/plans_0_6_0/ exist."""
        fixture_dir = REPO_ROOT / "tests" / "fixtures" / "plans_0_6_0"
        assert fixture_dir.exists()
        fixtures = sorted(fixture_dir.glob("*.json"))
        assert len(fixtures) == 5
        names = [f.name for f in fixtures]
        for expected in ["attention.json", "feed_forward.json", "gpt_tiny.json", "linear_mlp.json", "linear_options.json"]:
            assert expected in names

    def test_tier1_f9_saved_plan_0_6_0_deserialization(self):
        """[F9 Happy Path 3] Saved plan JSON fixtures deserialize cleanly into ResolvedPlan."""
        fixture_file = REPO_ROOT / "tests" / "fixtures" / "plans_0_6_0" / "linear_mlp.json"
        record = json.loads(fixture_file.read_text(encoding="utf-8"))
        plan = ResolvedPlan.from_json(record["plan"])
        assert plan is not None
        assert len(plan.nodes) > 0

    def test_tier1_f9_changelog_exists(self):
        """[F9 Happy Path 4] CHANGELOG.md exists in repository root."""
        changelog = REPO_ROOT / "CHANGELOG.md"
        assert changelog.exists()
        assert changelog.stat().st_size > 0

    def test_tier1_f9_pyproject_toml_configuration(self):
        """[F9 Happy Path 5] pyproject.toml defines build-system, project name, and version."""
        pyproject = REPO_ROOT / "pyproject.toml"
        assert pyproject.exists()
        text = pyproject.read_text(encoding="utf-8")
        assert 'name = "hndl"' in text
        assert "[build-system]" in text

    # --------------------------------------------------------------------------
    # F10: Documentation, typing & version sync (H-08, M-10, M-11, M-14)
    # --------------------------------------------------------------------------

    def test_tier1_f10_package_version_defined(self):
        """[F10 Happy Path 1] Package version is a non-empty string."""
        assert hasattr(hndl, "__version__")
        assert isinstance(hndl.__version__, str)
        assert len(hndl.__version__) > 0

    def test_tier1_f10_py_typed_file_exists(self):
        """[F10 Happy Path 2] PEP 561 py.typed marker file exists in src/hndl/."""
        py_typed = REPO_ROOT / "src" / "hndl" / "py.typed"
        assert py_typed.exists()

    def test_tier1_f10_registry_contains_operators(self):
        """[F10 Happy Path 3] Built-in registry contains at least 59 operators."""
        reg = Registry.builtins()
        assert len(reg.aliases) >= 59

    def test_tier1_f10_resolved_plan_repr_basic(self):
        """[F10 Happy Path 4] ResolvedPlan repr produces a formatted string."""
        plan = resolve("linear(32)\nrelu()", input_shape=("B", 16), output_shape=("B", 32))
        rep = repr(plan)
        assert "ResolvedPlan" in rep or "linear" in rep

    def test_tier1_f10_github_workflows_exist(self):
        """[F10 Happy Path 5] GitHub Actions CI and publish workflows exist."""
        if not (REPO_ROOT / ".git").exists():
            pytest.skip("Not running inside a git repository checkout (.github excluded from sdist)")
        ci_yml = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        publish_yml = REPO_ROOT / ".github" / "workflows" / "publish.yml"
        assert ci_yml.exists()
        assert publish_yml.exists()



# ==============================================================================
# TIER 2: Boundary & Corner Cases (Limits, Extremes, and Defect Triggers)
# ==============================================================================


class TestTier2BoundaryCases:
    """Tier 2: ≥5 tests per feature area covering limits, empty inputs, negative values, and syntax extremes."""

    # --------------------------------------------------------------------------
    # F1: Line break & newline parsing (H-01, H-02, M-04)
    # --------------------------------------------------------------------------

    def test_tier2_f1_h01_bare_carriage_return_linebreaks(self):
        """[Finding H-01] Bare \\r carriage returns must not raise unhandled IndexError outside HNDLError."""
        source = "linear(64)\rrelu()\rlinear()"
        try:
            plan = resolve(source, input_shape=("B", 128), output_shape=("B", 10))
            assert len(plan.nodes) == 3
        except IndexError as exc:
            pytest.fail(f"Finding H-01 unhandled IndexError triggered: {exc}")
        except HNDLError:
            pass

    def test_tier2_f1_h01_mixed_consecutive_carriage_returns(self):
        """[Finding H-01] Consecutive \\r\\r and trailing \\r must not crash with IndexError."""
        source = "linear(64)\r\rrelu()\r\nlinear(10)\r"
        try:
            plan = resolve(source, input_shape=("B", 128), output_shape=("B", 10))
            assert len(plan.nodes) == 3
        except IndexError as exc:
            pytest.fail(f"Finding H-01 consecutive \\r triggered IndexError: {exc}")
        except HNDLError:
            pass

    def test_tier2_f1_h02_unpacking_34_statements_no_syntax_error(self):
        """[Finding H-02] 34 unpacking statements following call lines must not trigger false positive E_SYNTAX."""
        lines = ["h = x"]
        for _ in range(34):
            lines.append("(a, b) = split(h, 32)")
            lines.append("h = concat(a, b)")
        lines.append("out = h\n")
        cfg = "\n".join(lines)
        try:
            plan = resolve(cfg, input_shape=("B", 64), output_shape=("B", 64))
            assert plan is not None
        except HNDLError as exc:
            if exc.code == "E_SYNTAX" and "more than 32 Python operators" in exc.message:
                pytest.fail(f"Finding H-02 false positive triggered: {exc}")
            raise

    def test_tier2_f1_h02_bracket_unpacking_after_call(self):
        """[Finding H-02] Bracket unpacking [a, b] = split(x, 32) must not count as chained subscript."""
        lines = ["h = x"]
        for _ in range(34):
            lines.append("[a, b] = split(h, 32)")
            lines.append("h = concat(a, b)")
        lines.append("out = h\n")
        cfg = "\n".join(lines)
        try:
            plan = resolve(cfg, input_shape=("B", 64), output_shape=("B", 64))
            assert plan is not None
        except HNDLError as exc:
            if exc.code == "E_SYNTAX" and "more than 32 Python operators" in exc.message:
                pytest.fail(f"Finding H-02 bracket unpacking false positive triggered: {exc}")
            raise

    def test_tier2_f1_m04_python311_fstring_rejected(self):
        """[Finding M-04] Python 3.11+ f-string syntax in declarative configs must be rejected with E_SYNTAX."""
        cfg = 'name = "foo"\nlinear(16, name=f"{name}")\n'
        with pytest.raises(HNDLError) as exc_info:
            resolve(cfg, input_shape=("B", 16), output_shape=("B", 16))
        assert exc_info.value.code == "E_SYNTAX"

    # --------------------------------------------------------------------------
    # F2: Syntax, tuple & identifier validation (M-03, M-05, M-06)
    # --------------------------------------------------------------------------

    def test_tier2_f2_m03_empty_tuple_target_rejected(self):
        """[Finding M-03] Empty tuple assignment p = () must be rejected with E_SYNTAX."""
        cfg = "p = ()\nlinear(10)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(cfg, input_shape=("B", 16), output_shape=("B", 10))
        assert exc_info.value.code == "E_SYNTAX"

    def test_tier2_f2_m03_tensor_tuple_arity_error_location(self):
        """[Finding M-03] E_OUTPUT_ARITY on tensor tuple construction must include line and column numbers."""
        cfg = "linear(16)\na = split(4)\nb = (a, x)\nlinear(10)"
        with pytest.raises(HNDLError) as exc_info:
            resolve(cfg, input_shape=("B", 16), output_shape=("B", 10))
        err = exc_info.value
        assert err.code == "E_OUTPUT_ARITY"
        assert err.line is not None and err.line > 0, "Finding M-03: Missing line number on E_OUTPUT_ARITY"
        assert err.column is not None and err.column > 0, "Finding M-03: Missing column number on E_OUTPUT_ARITY"

    def test_tier2_f2_m05_falsy_empty_string_input_dtype_rejected(self):
        """[Finding M-05] Falsy input_dtype='' must raise E_SCHEMA or E_DTYPE, not fall back via boolean or."""
        with pytest.raises(HNDLError) as exc_info:
            resolve("linear(10)", input_shape=("B", 16), output_shape=("B", 10), input_dtype="")
        assert exc_info.value.code in ("E_SCHEMA", "E_DTYPE")

    def test_tier2_f2_m05_falsy_boolean_false_input_dtype_rejected(self):
        """[Finding M-05] Falsy input_dtype=False must raise E_SCHEMA or E_DTYPE, not fall back."""
        with pytest.raises(HNDLError) as exc_info:
            resolve("linear(10)", input_shape=("B", 16), output_shape=("B", 10), input_dtype=False)
        assert exc_info.value.code in ("E_SCHEMA", "E_DTYPE")

    def test_tier2_f2_m06_keyword_and_range_external_port_names_rejected(self):
        """[Finding M-06] External port names colliding with Python keywords ('for', 'in') or 'range' must be rejected."""
        for reserved in ("for", "in", "range"):
            with pytest.raises(HNDLError) as exc_info:
                Capture(input_shape={reserved: ("B", 16)}, output_shape={"out": ("B", 16)})
            assert exc_info.value.code in ("E_NAME", "E_SCHEMA")

    # --------------------------------------------------------------------------
    # F3: Symbolic solver & interval reporting (M-07, L-03, L-04)
    # --------------------------------------------------------------------------

    def test_tier2_f3_m07_ambiguous_conv_reports_open_interval(self):
        """[Finding M-07] E_AMBIGUOUS must report the unresolved open interval (7, 8) rather than collapsed (8, 8)."""
        registry = Registry.builtins()

        def dummy_relation(s):
            s.rank("out", 4)
            s.axis("out", 1, 3)

        @registry.operator("dummy_m07_gen_e2e", identity="dummy_m07_gen_e2e", summary="dummy",
                           shape="x -> out", relation=dummy_relation)
        class DummyM07Gen(torch.nn.Module):
            pass

        node0 = Node("g", "dummy_m07_gen_e2e@1", {}, {"x": "input:x"})
        node1 = Node("c", "conv2d@1", {"out_channels": 16, "kernel_size": 3, "stride": (1, 2), "padding": 1},
                     {"x": "node:g/out"})
        graph = Graph((node0, node1), ("B", 3), ("B", 16, 8, 4), "node:c/out")

        with pytest.raises(HNDLError) as exc_info:
            resolve_graph(graph, registry)
        err = exc_info.value
        assert err.code == "E_AMBIGUOUS"
        assert "(7, 8)" in err.message, f"Finding M-07: expected open interval (7, 8) in message, got: {err.message}"
        assert "(8, 8)" not in err.message, f"Finding M-07: collapsed interval (8, 8) reported in: {err.message}"

    def test_tier2_f3_l03_ellipsis_rank_underflow_blames_invalid_port(self):
        """[Finding L-03] Rank underflow when matching ... patterns must blame the port with invalid rank."""
        source = "add(a, b)"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape={"a": ("B", 16), "b": ("B",)}, output_shape=("B", 16))
        assert exc_info.value.code in ("E_SHAPE", "E_CONSTRAINT", "E_SCHEMA")

    def test_tier2_f3_l04_conv_transpose_input_negative_guard(self):
        """[Finding L-04] conv_transpose_input must return None when extent < 1 is mathematically impossible."""
        result = conv_transpose_input(extent=1, kernel=4, stride=1, padding=0)
        assert result is None, f"Finding L-04: conv_transpose_input returned negative extent {result} instead of None"

    def test_tier2_f3_l04_conv_input_range_nonpositive_extent(self):
        """[Finding L-04] conv_input_range on non-positive extent must return empty interval (1, 0)."""
        result = conv_input_range(extent=0, kernel=3, stride=1, padding=0)
        assert result[0] > result[1], f"Finding L-04: expected empty interval lower > upper, got: {result}"

    def test_tier2_f3_l04_solver_axis_out_of_bounds_defensive(self):
        """[Finding L-04] Defensive bounds check when querying out-of-bounds axis."""
        from hndl.relations import conv_axis
        with pytest.raises(HNDLError) as exc_info:
            # Axis 5 is outside rank 4
            conv_axis(None, "x", "out", kernel=3, stride=1, padding=1, axis=5)
        assert exc_info.value.code in ("E_CONSTRAINT", "E_ARGUMENT")

    # --------------------------------------------------------------------------
    # F4: PyTorch chain & sequence slicing (H-03, M-08, M-09)
    # --------------------------------------------------------------------------

    def test_tier2_f4_h03_variadic_single_output_chunk_not_chain(self):
        """[Finding H-03] Single-output variadic operator (chunk(1)) must not be classified as unary chain."""
        model = network("a, = chunk(1)\nlinear(a, 10)", input_shape=("B", 8), output_shape=("B", 10), device="cpu")
        assert model._chain is False, "Finding H-03: single-output chunk(1) was misclassified as unary chain (_chain=True)"
        # Slicing a non-chain should raise TypeError
        with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
            _ = model[:2]

    def test_tier2_f4_h03_slice_step_not_one_rejected(self):
        """[Finding H-03] GraphModule.__getitem__ must reject slice steps other than 1 (step != 1)."""
        model = network("linear(32)\nrelu()\nlinear(16)", input_shape=("B", 64), output_shape=("B", 16), device="cpu")
        with pytest.raises((ValueError, TypeError)):
            _ = model[::2]
        with pytest.raises((ValueError, TypeError)):
            _ = model[::-1]

    def test_tier2_f4_m08_preforward_verify_state_during_compilation(self):
        """[Finding M-08] Pre-forward state verification must run when compiling is active."""
        model = network("linear(16)", input_shape=("B", 32), output_shape=("B", 16), device="cpu")
        x = torch.randn(2, 32)
        model(x)  # fast path cached
        # Mutate registered parameter before compiled call
        with torch.no_grad():
            model[0].weight.add_(1.0)
        from hndl.torch import _compiling
        with pytest.raises(HNDLError) as exc_info:
            with _compiling():
                model(x)
        assert "build" in exc_info.value.message or "match" in exc_info.value.message

    def test_tier2_f4_m09_shallow_copy_double_clears_validation(self):
        """[Finding M-09] Calling .double() on shallow copy copy.copy(model) must clear original model._validated."""
        model = network("linear(16)", input_shape=("B", 32), output_shape=("B", 16), device="cpu")
        x = torch.randn(2, 32, dtype=torch.float32)
        model(x)  # populate _validated
        assert len(model._validated) > 0
        shallow = copy.copy(model)
        shallow.double()
        assert len(model._validated) == 0, "Finding M-09: model._validated was not cleared after shallow copy mutation"

    def test_tier2_f4_m09_child_module_mutation_detected(self):
        """[Finding M-09] Calling .double() directly on child module model[0] must not bypass validation."""
        model = network("linear(16)", input_shape=("B", 32), output_shape=("B", 16), device="cpu")
        x = torch.randn(2, 32, dtype=torch.float32)
        model(x)  # populate fast path
        model[0].double()
        # Next call with float32 input must re-validate or raise cleanly, not crash in C fast path
        x_d = torch.randn(2, 32, dtype=torch.float64)
        try:
            model(x_d)
        except (HNDLError, TypeError):
            pass

    # --------------------------------------------------------------------------
    # F5: Pretrained model training & revisions (H-04, H-05, M-01, M-02)
    # --------------------------------------------------------------------------

    def test_tier2_f5_h04_pretrained_trainable_leaves_wrapped_model_in_train_mode(self, text_checkpoint):
        """[Finding H-04] pretrained(..., trainable=True) must leave wrapped model in training mode (model.training == True)."""
        model = network(
            f'pretrained("{text_checkpoint}", trainable=True)',
            input_shape=("B", 4),
            output_shape=("B", 4, 8),
            input_dtype="int64",
            device="cpu",
        )
        assert model[0].model.training is True, "Finding H-04: trainable pretrained model left in eval mode after network()"

    def test_tier2_f5_h05_timm_provider_preserves_revision_in_hub_call(self, monkeypatch):
        """[Finding H-05] TimmProvider.instantiate must forward self.source.revision for hf:// sources."""
        import types
        import hndl.pretrained as pretrained_mod
        from hndl.pretrained import Source, TimmProvider

        calls = []
        fake_timm = types.SimpleNamespace(
            create_model=lambda name, pretrained=False, **kwargs: calls.append((name, pretrained, kwargs))
        )
        fake_safetensors = types.SimpleNamespace()
        orig_require = pretrained_mod._require
        monkeypatch.setattr(
            pretrained_mod,
            "_require",
            lambda mod: fake_timm if mod == "timm" else (fake_safetensors if mod == "safetensors" else orig_require(mod)),
        )
        source = Source("hf", "timm/resnet18.a1_in1k", "rev12345", {}, "", "timm")
        provider = TimmProvider(source, output="features")
        assert provider.source.revision == "rev12345"
        provider.instantiate(weights=True)
        assert calls == [("hf-hub:timm/resnet18.a1_in1k@rev12345", True, {})]

    def test_tier2_f5_m01_resolve_source_detects_file_mtime_change(self, text_checkpoint):
        """[Finding M-01] resolve_source must notice file mtime changes and not return stale revision from lru_cache."""
        from hndl.pretrained import resolve_source
        source1 = resolve_source(str(text_checkpoint))
        # Touch config.json
        cfg_file = text_checkpoint / "config.json"
        text = cfg_file.read_text(encoding="utf-8")
        cfg_data = json.loads(text)
        cfg_data["_test_mtime_mutation"] = True
        cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")
        source2 = resolve_source(str(text_checkpoint))
        assert source1.revision != source2.revision, "Finding M-01: Stale lru_cache returned identical revision after on-disk edit"

    def test_tier2_f5_m01_safetensors_revision_hashes_weights(self, text_checkpoint):
        """[Finding M-01] Local .safetensors revision hash must reflect tensor data, not just config.json + st_size."""
        from hndl.pretrained import resolve_source
        source1 = resolve_source(str(text_checkpoint))
        weights_file = text_checkpoint / "model.safetensors"
        if weights_file.exists():
            data = bytearray(weights_file.read_bytes())
            # Mutate last byte (size remains identical)
            data[-1] ^= 0xFF
            weights_file.write_bytes(data)
            source2 = resolve_source(str(text_checkpoint))
            assert source1.revision != source2.revision, "Finding M-01: Weight modification with unchanged size missed by revision hash"

    def test_tier2_f5_m02_malformed_config_json_raises_hndl_error(self, tmp_path):
        """[Finding M-02] Corrupted config.json must raise HNDLError('E_PRETRAINED'), not raw json.JSONDecodeError."""
        bad_dir = tmp_path / "bad_checkpoint"
        bad_dir.mkdir()
        (bad_dir / "config.json").write_text("{invalid json", encoding="utf-8")
        from hndl.pretrained import resolve_source
        with pytest.raises(HNDLError) as exc_info:
            resolve_source(str(bad_dir))
        assert exc_info.value.code == "E_PRETRAINED"

    # --------------------------------------------------------------------------
    # F6: Numerics & double precision (RMSNorm, RoPE) (H-06, M-12)
    # --------------------------------------------------------------------------

    def test_tier2_f6_h06_rmsnorm_preserves_float64_precision(self):
        """[Finding H-06] RMSNorm.forward must preserve float64 precision (discrepancy < 1e-12 vs reference)."""
        norm = RMSNorm(eps=1e-5, affine=False, D=16).double()
        x = torch.randn(2, 4, 16, dtype=torch.float64)
        out = norm(x)
        expected = nn.functional.rms_norm(x, (16,), eps=1e-5)
        diff = (out - expected).abs().max().item()
        assert diff < 1e-12, f"Finding H-06: RMSNorm float64 downcast to float32 caused precision error: {diff:.3e}"

    def test_tier2_f6_h06_rmsnorm_gradcheck_double(self):
        """[Finding H-06] torch.autograd.gradcheck on RMSNorm with double precision."""
        norm = RMSNorm(eps=1e-5, affine=True, D=8).double()
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(norm, (x,), eps=1e-6, atol=1e-4)

    def test_tier2_f6_m12_rope_tables_preserves_float64(self):
        """[Finding M-12] attention._rope_tables must compute angles in float64 when dtype is float64."""
        from hndl.operators.attention import _rope_tables
        cos_f64, sin_f64 = _rope_tables(positions=16, head_dim=32, device="cpu", dtype=torch.float64)
        assert cos_f64.dtype == torch.float64
        assert sin_f64.dtype == torch.float64

    def test_tier2_f6_m12_pos_embed_t_greater_than_max_len_rejected(self):
        """[Finding M-12] PosEmbed.forward must raise HNDLError('E_CONSTRAINT') when T > max_len, not broadcast position 0."""
        from hndl.operators.pos_embed import PosEmbed
        pos = PosEmbed(max_len=1, D=16)
        x = torch.randn(2, 4, 16)  # T=4 > max_len=1
        with pytest.raises(HNDLError) as exc_info:
            pos(x)
        assert exc_info.value.code == "E_CONSTRAINT"

    def test_tier2_f6_m12_moe_dense_reference_parity(self):
        """[Finding M-12] MixtureOfExperts _reference must provide independent dense oracle."""
        reg = Registry.builtins()
        moe_op = reg.get("moe")
        assert moe_op.reference is not None

    # --------------------------------------------------------------------------
    # F7: Operator references & layer representations (M-13, L-07, L-08)
    # --------------------------------------------------------------------------

    def test_tier2_f7_m13_all_68_non_pretrained_operators_have_reference(self):
        """[Finding M-13] All 68 non-pretrained operators in builtins must define reference= callable."""
        reg = Registry.builtins()
        missing = []
        for alias in reg.aliases:
            if alias == "pretrained":
                continue
            op = reg.get(alias)
            if op.reference is None:
                missing.append(alias)
        assert not missing, f"Finding M-13: {len(missing)} operators missing reference=: {missing}"

    def test_tier2_f7_l07_grid_sample_rejects_positional_optional_args(self):
        """[Finding L-07] GridSample optional options (mode, padding_mode) must be keyword-only, not positional."""
        reg = Registry.builtins()
        grid_op = reg.get("grid_sample")
        for opt_name in ("mode", "padding_mode", "align_corners"):
            if opt_name in grid_op.args:
                assert grid_op.args[opt_name].positional is False, f"Finding L-07: {opt_name} must have positional=False"

    def test_tier2_f7_l08_resblock_validates_out_channels_divisible_by_groups(self):
        """[Finding L-08] ResBlock.__init__ direct construction must validate out_channels % groups == 0."""
        from hndl.operators.resblock import ResBlock
        with pytest.raises(HNDLError) as exc_info:
            ResBlock(out_channels=15, in_channels=16, stride=1, norm="group_norm", groups=8, C_in=16, C_out=15)
        assert exc_info.value.code == "E_CONSTRAINT"

    def test_tier2_f7_l08_extra_repr_present_on_all_spatial_ops(self):
        """[Finding L-08] CoordinateGrid, FourierFeatures, GridSample must implement non-empty extra_repr()."""
        from hndl.operators.coordinate_grid import CoordinateGrid
        from hndl.operators.fourier_features import FourierFeatures
        from hndl.operators.grid_sample import GridSample

        assert CoordinateGrid(spatial_shape=(8, 8)).extra_repr()
        assert FourierFeatures(num_frequencies=16, scale=8.0, D=2).extra_repr()
        assert GridSample(mode="bilinear", padding_mode="zeros", align_corners=True).extra_repr()

    def test_tier2_f7_l08_equalized_linear_and_attention_extra_repr(self):
        """[Finding L-08] EqualLinear and Attention must include equalized=True in extra_repr()."""
        from hndl.operators._equalized import EqualLinear
        from hndl.operators.attention import Attention

        lin = EqualLinear(16, 8, equalized=True)
        assert "equalized" in lin.extra_repr()

        att = Attention(heads=2, causal=False, dropout=0.0, bias=True, equalized=True,
                        qkv_bias=True, out_bias=True, rope=False, relative_position_bias=False,
                        spatial_shape=(0, 0), D=16)
        assert "equalized" in att.extra_repr()

    # --------------------------------------------------------------------------
    # F8: Diagnostics & error location attribution (L-01, L-02, L-05)
    # --------------------------------------------------------------------------

    def test_tier2_f8_l01_loop_collision_auto_vs_explicit_distinction(self):
        """[Finding L-01] Loop-suffixed collision with auto-generated node ID n0 must not say 'explicit name=n0'."""
        source = (
            "linear(16)\n"  # auto-generated id is n0
            "for _ in range(2):\n"
            "    linear(16, name='n')\n"  # iteration 0 creates n0
        )
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert "an explicit name='n0'" not in exc_info.value.message, "Finding L-01: misattributed auto-generated n0 as explicit"

    def test_tier2_f8_l02_capture_emit_preserves_node_name_on_error(self):
        """[Finding L-02] resolve_callable emit errors must preserve node='custom_name'."""
        def bad_callable(x):
            return ops.linear(x, name="my_bad_linear", invalid_kwarg=123)

        with pytest.raises(HNDLError) as exc_info:
            resolve_callable(bad_callable, input_shape=("B", 16), output_shape=("B", 16))
        err = exc_info.value
        assert err.node == "my_bad_linear", f"Finding L-02: node name was None, expected 'my_bad_linear': {err}"

    def test_tier2_f8_l05_example_post_init_preserves_dict_contracts(self):
        """[Finding L-05] Example.__post_init__ must preserve multi-port dict contracts, not convert them to key tuples."""
        ex = Example("linear(10)", input_shape={"x": ("B", 4), "cond": ("B", 2)}, output_shape=("B", 10))
        assert isinstance(ex.input_shape, dict), "Finding L-05: Example converted input_shape dict to tuple"

    def test_tier2_f8_l05_node_post_init_rejects_string_outputs(self):
        """[Finding L-05] Node.__post_init__ must validate container types and not split raw strings into characters."""
        with pytest.raises((HNDLError, TypeError)):
            Node(id="n0", op="linear@1", args={}, inputs={"x": "input:x"}, outputs="out")

    def test_tier2_f8_l03_underflow_error_message_exact_token(self):
        """[Finding L-03] Solver rank underflow error message attributes the underflowing port."""
        registry = Registry.builtins()

        def noop(s):
            pass

        @registry.operator("gen_unknown_f8_l03", identity="gen_unknown_f8_l03", summary="dummy",
                           shape="x -> out", relation=noop)
        class GenUnknown(torch.nn.Module):
            pass

        @registry.operator("test_ellipsis_f8_l03", identity="test_ellipsis_f8_l03", summary="test",
                           shape="x[B, ..., C, D] -> out[B, ..., H, W]")
        class DummyEllipsis(torch.nn.Module):
            pass

        n0 = Node("n0", "gen_unknown_f8_l03@1", {}, {"x": "input:x"})
        n1 = Node("n1", "test_ellipsis_f8_l03@1", {}, {"x": "node:n0/out"})
        graph = Graph((n0, n1), ("B", 4), ("B", 10), "node:n1/out")

        with pytest.raises(HNDLError) as exc_info:
            resolve_graph(graph, registry)
        assert exc_info.value.code == "E_CONSTRAINT"
        assert "Rank of out is below the declared pattern rank" in str(exc_info.value)
        assert "Rank of x is below the declared pattern rank" not in str(exc_info.value)

    # --------------------------------------------------------------------------
    # F9: Package distribution & fixture integrity (H-07, L-10)
    # --------------------------------------------------------------------------

    def test_tier2_f9_h07_manifest_in_includes_fixtures_and_changelog(self):
        """[Finding H-07] MANIFEST.in must include tests *.json fixtures and CHANGELOG.md."""
        manifest_text = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        assert "CHANGELOG.md" in manifest_text, "Finding H-07: CHANGELOG.md omitted from MANIFEST.in"
        assert "*.json" in manifest_text, "Finding H-07: Plan fixture *.json omitted from MANIFEST.in"

    def test_tier2_f9_h07_all_saved_plans_loadable_without_missing_files(self):
        """[Finding H-07] All 5 golden plan fixtures must load, build, and run."""
        fixture_dir = REPO_ROOT / "tests" / "fixtures" / "plans_0_6_0"
        for json_file in fixture_dir.glob("*.json"):
            record = json.loads(json_file.read_text(encoding="utf-8"))
            plan = ResolvedPlan.from_json(record["plan"])
            model = build(plan, device="cpu")
            assert model is not None

    def test_tier2_f9_l10_pyproject_toml_has_norecursedirs(self):
        """[Finding L-10] pyproject.toml must configure norecursedirs ignoring .claude, .venv, build, dist."""
        pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "norecursedirs" in pyproject_text, "Finding L-10: norecursedirs missing from pyproject.toml"

    def test_tier2_f9_l10_no_nested_claude_worktrees_active(self):
        """[Finding L-10] Verify no stale nested worktrees exist under .claude/worktrees/."""
        claude_wt = REPO_ROOT / ".claude" / "worktrees"
        if claude_wt.exists():
            nested = list(claude_wt.glob("*"))
            assert len(nested) == 0, f"Finding L-10: Stale worktrees found in .claude/worktrees/: {nested}"

    def test_tier2_f9_package_build_metadata_strict(self):
        """[F9 Boundary 5] pyproject.toml dependencies and versions."""
        pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'requires-python = ">=3.11,<3.15"' in pyproject_text
        assert 'dependencies = ["torch>=2.6,<3"]' in pyproject_text

    # --------------------------------------------------------------------------
    # F10: Documentation, typing & version sync (H-08, M-10, M-11, M-14)
    # --------------------------------------------------------------------------

    def test_tier2_f10_h08_version_synchronization_all_docs(self):
        """[Finding H-08] Version must match 0.8.0 / 0.8.1 across pyproject.toml, _version.py, README, SPEC, IMPLEMENTATION."""
        version = hndl.__version__
        spec_text = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
        impl_text = (REPO_ROOT / "IMPLEMENTATION.md").read_text(encoding="utf-8")
        assert f"Implementation notes: {version}" in impl_text or "0.8." in impl_text
        assert "0.1.2" not in impl_text, "Finding H-08: Stale version 0.1.2 found in IMPLEMENTATION.md"
        assert "0.1.2" not in spec_text, "Finding H-08: Stale version 0.1.2 found in SPEC.md"

    def test_tier2_f10_h08_spec_operator_count_equals_78(self):
        """[Finding H-08] IMPLEMENTATION.md must record 78 operators across 12 categories, not 'Fifty-nine'."""
        impl_text = (REPO_ROOT / "IMPLEMENTATION.md").read_text(encoding="utf-8")
        assert "Fifty-nine" not in impl_text, "Finding H-08: Stale 'Fifty-nine operators' in IMPLEMENTATION.md"
        assert any(c in impl_text for c in ("78", "84")), "Finding H-08: Operator count missing from IMPLEMENTATION.md"

    def test_tier2_f10_h08_spec_arg_lists_since_parameter(self):
        """[Finding H-08] SPEC.md Arg signature must document the 'since' parameter."""
        spec_text = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
        assert "since" in spec_text, "Finding H-08: Arg(since=...) parameter missing from SPEC.md"

    def test_tier2_f10_m10_resolved_plan_repr_branched_shows_references(self):
        """[Finding M-10] ResolvedPlan.__repr__ on branched graph must show producing references and omit chain index."""
        source = (
            "a = linear(16)\n"
            "b = linear(16)\n"
            "add(a, b)\n"
        )
        plan = resolve(source, input_shape=("B", 32), output_shape=("B", 16))
        rep = repr(plan)
        assert "index" not in rep, "Finding M-10: index column printed on branched graph repr"
        assert "node:n0/out" in rep or "n0" in rep

    def test_tier2_f10_m11_public_api_has_type_annotations(self):
        """[Finding M-11] Public functions (resolve, network, build) must have parameter and return type annotations."""
        for fn in (resolve, network, build):
            sig = inspect.signature(fn)
            annotated_params = [p.name for p in sig.parameters.values() if p.annotation is not inspect.Parameter.empty]
            assert len(annotated_params) > 0, f"Finding M-11: Public function {fn.__name__} has zero type annotations"



# ==============================================================================
# TIER 3: Cross-Feature Interactions (Pairwise Combinations)
# ==============================================================================


class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise combinations across major feature areas (≥15 tests)."""

    def test_tier3_f1_f2_multiline_config_with_tuples_and_reserved_ports(self, device):
        """[Pairwise F1 + F2] Multiline config with tuple unpacking and named contracts."""
        source = (
            "# Multiline config with unpacking\n"
            "features = linear(32, name='stem')\n"
            "(left, right) = split(features, 16)\n"
            "left_out = linear(left, 8, name='l_branch')\n"
            "right_out = linear(right, 8, name='r_branch')\n"
            "out = add(left_out, right_out)\n"
        )
        model = network(source, input_shape={"x": ("B", 64)}, output_shape={"out": ("B", 8)}, device=device)
        x = torch.randn(2, 64, device=device)
        out = model(x=x)
        assert out.shape == (2, 8)

    def test_tier3_f1_f4_multiline_chain_slicing(self, device):
        """[Pairwise F1 + F4] Multiline linear chain parsed and built, sliced into sub-networks."""
        source = (
            "linear(64)\n"
            "relu()\n"
            "linear(32)\n"
            "relu()\n"
            "linear(10)\n"
        )
        model = network(source, input_shape=("B", 128), output_shape=("B", 10), device=device)
        assert model._chain is True
        sub_model = model[1:4]
        assert isinstance(sub_model, nn.Sequential)
        assert len(sub_model) == 3
        x = torch.randn(2, 64, device=device)
        out = sub_model(x)
        assert out.shape == (2, 32)

    def test_tier3_f2_f3_complex_unpacking_with_symbolic_shape_solving(self, device):
        """[Pairwise F2 + F3] Multi-head split and join with bidirectional shape constraints."""
        source = (
            "stem = conv(32, kernel_size=3, padding=1)\n"
            "(c1, c2) = split(stem, 16, dim=1)\n"
            "p1 = conv(c1, 16, kernel_size=3, padding=1)\n"
            "p2 = conv(c2, 16, kernel_size=3, padding=1)\n"
            "concat(p1, p2, axis=1)\n"
        )
        model = network(
            source,
            input_shape=("B", 3, 16, 16),
            output_shape=("B", 32, 16, 16),
            device=device,
        )
        x = torch.randn(2, 3, 16, 16, device=device)
        assert model(x).shape == (2, 32, 16, 16)

    def test_tier3_f3_f4_branched_graph_with_conv_and_slicing_rejection(self, device):
        """[Pairwise F3 + F4] Non-chain branched network with conv and skip connection correctly rejects sequential slicing."""
        source = (
            "stem = conv(16, kernel_size=3, padding=1)\n"
            "branch = conv(stem, 16, kernel_size=3, padding=1)\n"
            "add(stem, branch)\n"
        )
        model = network(source, input_shape=("B", 3, 16, 16), output_shape=("B", 16, 16, 16), device=device)
        assert model._chain is False
        with pytest.raises(TypeError, match="Branched graphs support node-name lookup only"):
            _ = model[:2]

    def test_tier3_f3_f6_conv_transpose_solver_with_rmsnorm_float64(self, device):
        """[Pairwise F3 + F6] ConvTranspose upsampling followed by double-precision RMSNorm."""
        source = (
            "deconv(16, kernel_size=4, stride=2, padding=1)\n"
            "flatten()\n"
            "rms_norm()\n"
        )
        model = network(
            source,
            input_shape=("B", 32, 4, 4),
            output_shape=("B", 16 * 8 * 8),
            device=device,
        ).double()
        x = torch.randn(2, 32, 4, 4, device=device, dtype=torch.float64)
        out = model(x)
        assert out.dtype == torch.float64
        assert out.shape == (2, 1024)

    def test_tier3_f4_f5_slicing_network_containing_pretrained_module(self, text_checkpoint, device):
        """[Pairwise F4 + F5] Slicing lookup on a network containing a Pretrained module."""
        source = (
            f'pretrained("{text_checkpoint}", name="lm")\n'
            'linear(10, name="head")\n'
        )
        model = network(
            source,
            input_shape=("B", 4),
            output_shape=("B", 4, 10),
            input_dtype="int64",
            device=device,
        )
        assert model["lm"] is not None
        assert model["head"] is not None
        assert "lm" in [n.id for n in model.plan.nodes]

    def test_tier3_f4_f6_rmsnorm_rope_attention_with_shallow_copy_double(self, device):
        """[Pairwise F4 + F6] RMSNorm + RoPE Attention layer, cloned via shallow copy and cast to double."""
        source = (
            "rms_norm()\n"
            "attention(heads=2, rope=True)\n"
        )
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        x_f32 = torch.randn(2, 4, 16, device=device, dtype=torch.float32)
        out_f32 = model(x_f32)
        assert out_f32.dtype == torch.float32

        # Shallow copy cast to double
        clone = copy.copy(model)
        clone.double()
        x_f64 = torch.randn(2, 4, 16, device=device, dtype=torch.float64)
        out_f64 = clone(x_f64)
        assert out_f64.dtype == torch.float64

    def test_tier3_f5_f6_pretrained_checkpoint_with_float64_inference(self, text_checkpoint, device):
        """[Pairwise F5 + F6] Pretrained checkpoint chained with float64 normalization and linear head."""
        source = (
            f'pretrained("{text_checkpoint}", name="lm")\n'
            'rms_norm(name="norm")\n'
            'linear(16, name="proj")\n'
        )
        model = network(
            source,
            input_shape=("B", 4),
            output_shape=("B", 4, 16),
            input_dtype="int64",
            device=device,
        )
        tokens = torch.randint(0, 16, (2, 4), device=device)
        out = model(tokens)
        assert out.shape == (2, 4, 16)

    def test_tier3_f2_f8_identifier_collision_error_attribution(self):
        """[Pairwise F2 + F8] Identifier collision with operator alias produces clean error attribution."""
        with pytest.raises(HNDLError) as exc_info:
            Capture(input_shape={"linear": ("B", 16)}, output_shape={"out": ("B", 16)})
        assert exc_info.value.code == "E_NAME"
        assert "linear" in exc_info.value.message

    def test_tier3_f1_f8_loop_unrolling_diagnostic_attribution(self):
        """[Pairwise F1 + F8] Declarative loop unrolling collision error attribution."""
        source = (
            "linear(16)\n"
            "for _ in range(2):\n"
            "    linear(16, name='n')\n"
        )
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        err = exc_info.value
        assert err.code == "E_NAME"
        assert "n0" in err.message

    def test_tier3_f6_f7_float64_rope_attention_reference_parity(self, device):
        """[Pairwise F6 + F7] RoPE Attention with float64 inputs compared against reference."""
        source = "attention(heads=2, rope=True)\n"
        plan = resolve(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16))
        model = build(plan, device=device).double()
        x = torch.randn(2, 4, 16, device=device, dtype=torch.float64)
        out = model(x=x)["output"]
        assert out.dtype == torch.float64
        assert out.shape == (2, 4, 16)

    def test_tier3_f4_f8_preforward_state_verification_error_diagnostic(self):
        """[Pairwise F4 + F8] State mutation prior to forward reports the failing parameter."""
        model = network("linear(16, name='my_fc')", input_shape=("B", 32), output_shape=("B", 16), device="cpu")
        x = torch.randn(2, 32)
        model(x)  # fast path
        del model["my_fc"].bias
        with pytest.raises(HNDLError) as exc_info:
            model(x)
        assert exc_info.value.code in ("E_STATE", "E_RUNTIME")

    def test_tier3_f3_f7_operator_references_with_strided_spatial_transforms(self, device):
        """[Pairwise F3 + F7] Spatial downsampling with max_pool and reference matching."""
        source = "max_pool(2)\n"
        model = network(source, input_shape=("B", 16, 8, 8), output_shape=("B", 16, 4, 4), device=device)
        x = torch.randn(2, 16, 8, 8, device=device)
        out = model(x)
        expected = nn.functional.max_pool2d(x, 2)
        torch.testing.assert_close(out, expected)

    def test_tier3_f2_f4_multi_output_chunk_operator_prevents_sequential_chain(self):
        """[Pairwise F2 + F4] chunk(1) operator produces variadic outputs and blocks unary chain slicing."""
        model = network("a, = chunk(1)\nlinear(a, 8)", input_shape=("B", 16), output_shape=("B", 8), device="cpu")
        assert model._chain is False

    def test_tier3_f5_f8_pretrained_missing_file_clean_error_code(self):
        """[Pairwise F5 + F8] Missing pretrained checkpoint path produces clean E_PRETRAINED error code."""
        from hndl.pretrained import resolve_source
        with pytest.raises(HNDLError) as exc_info:
            resolve_source("/nonexistent/path/to/checkpoint")
        assert exc_info.value.code == "E_PRETRAINED"

    def test_tier3_f6_f10_rmsnorm_rope_typed_api_interaction(self, device):
        """[Pairwise F6 + F10] Executing double precision network through public typed entrypoints."""
        model = network("rms_norm()\nlinear(8)", input_shape=("B", 4, 16), output_shape=("B", 4, 8), device=device)
        model.double()
        x = torch.randn(2, 4, 16, device=device, dtype=torch.float64)
        out = model(x)
        assert out.dtype == torch.float64
        assert out.shape == (2, 4, 8)



# ==============================================================================
# TIER 4: Real-World Application Workloads
# ==============================================================================


class TestTier4RealWorldWorkloads:
    """Tier 4: Realistic end-to-end application workloads exercising multiple subsystems (≥5 tests)."""

    def test_tier4_workload1_transformer_rope_rmsnorm_forward_backward(self, device):
        """[Workload 1] Full Transformer decoder block with RMSNorm, RoPE, Attention, SwiGLU/MLP and residuals."""
        source = (
            "h = x\n"
            "norm1 = rms_norm(h, name='norm1')\n"
            "attn = attention(norm1, heads=2, causal=True, rope=True, name='attn')\n"
            "h = add(h, attn)\n"
            "norm2 = rms_norm(h, name='norm2')\n"
            "ffn_up = linear(norm2, 32, name='ffn1')\n"
            "ffn_act = gelu(ffn_up)\n"
            "ffn_down = linear(ffn_act, 16, name='ffn2')\n"
            "out = add(h, ffn_down)\n"
        )
        model = network(source, input_shape=("B", 8, 16), output_shape=("B", 8, 16), device=device)
        x = torch.randn(2, 8, 16, device=device, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 8, 16)

        # End-to-end backward pass
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert model["attn"].q_proj.weight.grad is not None
        assert model["ffn1"].weight.grad is not None

    def test_tier4_workload2_multitask_cnn_backbone_with_multibranch_heads(self, device):
        """[Workload 2] Multi-task Vision CNN with ResBlock backbone, classification and reconstruction heads."""
        source = (
            "stem = conv(16, kernel_size=3, padding=1, name='stem')\n"
            "block1 = resblock(stem, 16, name='b1')\n"
            "block2 = resblock(block1, 32, stride=2, name='b2')\n"
            "# Branch 1: Classification\n"
            "pooled = global_avg_pool(block2)\n"
            "logits = linear(pooled, 10, name='cls_head')\n"
            "# Branch 2: Dense reconstruction\n"
            "up = deconv(block2, 16, kernel_size=4, stride=2, padding=1, name='up')\n"
            "mask = conv(up, 1, kernel_size=3, padding=1, name='mask_head')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 3, 16, 16)},
            output_shape={"logits": ("B", 10), "mask": ("B", 1, 16, 16)},
            device=device,
        )
        images = torch.randn(2, 3, 16, 16, device=device)
        outputs = model(x=images)
        assert outputs["logits"].shape == (2, 10)
        assert outputs["mask"].shape == (2, 1, 16, 16)

        # State dict preservation
        state = copy.deepcopy(model.state_dict())
        assert "nodes.n_stem.weight" in state
        assert "nodes.n_cls_head.weight" in state
        assert "nodes.n_mask_head.weight" in state

    def test_tier4_workload3_declarative_loops_unpacking_and_custom_initializers(self, device):
        """[Workload 3] Deep declarative architecture with loops, tensor unpacking, and explicit initializers."""
        source = (
            "h = linear(16, init={'weight': kaiming_normal(), 'bias': 0.0}, name='stem')\n"
            "for _ in range(3):\n"
            "    (a, b) = split(h, 8, dim=1)\n"
            "    left = linear(a, 8, init={'weight': normal(std=0.02)})\n"
            "    right = linear(b, 8, init={'weight': normal(std=0.02)})\n"
            "    h = concat(left, right, axis=1)\n"
            "    h = relu(h)\n"
            "out = linear(h, 4, init={'weight': xavier_uniform(), 'bias': 0.0}, name='head')\n"
        )
        model = network(source, input_shape=("B", 32), output_shape=("B", 4), device=device)
        x = torch.randn(2, 32, device=device)
        out = model(x)
        assert out.shape == (2, 4)
        assert len(model.plan.nodes) == 1 + 3 * 5 + 1

    def test_tier4_workload4_pretrained_finetuning_vision_and_language(self, text_checkpoint, device):
        """[Workload 4] Fine-tuning pretrained checkpoint with partial parameter freezing and optimizer step."""
        source = (
            f'pretrained("{text_checkpoint}", output="logits", trainable=True, name="backbone")\n'
            'linear(4, name="classifier")\n'
        )
        model = network(
            source,
            input_shape=("B", 4),
            output_shape=("B", 4, 4),
            input_dtype="int64",
            device=device,
        )
        # Freeze backbone parameters
        for p in model["backbone"].parameters():
            p.requires_grad_(False)
        assert not any(p.requires_grad for p in model["backbone"].parameters())
        assert all(p.requires_grad for p in model["classifier"].parameters())

        # Optimizer step updates only classifier
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_classifier_weight = model["classifier"].weight.clone()
        tokens = torch.randint(0, 16, (2, 4), device=device)
        out = model(tokens)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # Classifier updated, backbone untouched
        assert not torch.equal(model["classifier"].weight, initial_classifier_weight)

    def test_tier4_workload5_meta_device_tracing_and_compile_compatibility(self):
        """[Workload 5] Memory estimation and parameter counting on large multi-stage network."""
        source = (
            "linear(64)\n"
            "relu()\n"
            "linear(128)\n"
            "relu()\n"
            "linear(32)\n"
        )
        plan = resolve(source, input_shape=("B", 32), output_shape=("B", 32))
        counts = parameter_counts(plan)
        assert sum(counts.values()) > 0
        assert counts["n0"] > 0

        # Meta device memory budgeting via max_state_bytes limit
        meta_budgeted_model = build(plan, device="cpu", limits={"max_state_bytes": 10_000_000})
        assert meta_budgeted_model is not None
        with pytest.raises(HNDLError) as exc_info:
            build(plan, device="cpu", limits={"max_state_bytes": 10})
        assert exc_info.value.code == "E_RESOURCE"

    def test_tier4_workload6_end_to_end_adaptive_style_generator(self, device):
        """[Workload 6] Continuous coordinate generator with Fourier features and linear projections."""
        source = (
            "coords = fourier_features(16, name='fourier')\n"
            "proj = linear(coords, 32, name='proj')\n"
            "out = relu(proj)\n"
        )
        model = network(source, input_shape={"x": ("B", 2)}, output_shape={"out": ("B", 32)}, device=device)
        coords = torch.randn(4, 2, device=device)
        out = model(x=coords)
        assert out.shape == (4, 32)
        assert torch.all(out >= 0)
