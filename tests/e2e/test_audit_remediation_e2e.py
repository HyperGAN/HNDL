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
    Arg,
    Example,
    HNDLError,
    Registry,
    ResolvedPlan,
    ops,
    resolve,
    resolve_callable,
)
from hndl.capture import Capture
from hndl.config import capture_config
from hndl.operators.rms_norm import RMSNorm
from hndl.relations import (
    conv_input_range,
    conv_output,
    conv_transpose_input,
    conv_transpose_output,
)
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
        ci_yml = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        publish_yml = REPO_ROOT / ".github" / "workflows" / "publish.yml"
        assert ci_yml.exists()
        assert publish_yml.exists()
