"""Comprehensive unit tests for Milestone 5: DX, Tooling, Profiling & Lifecycle (Features 38-43)."""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

import hndl
from hndl import ops, resolve, resolve_callable
from hndl.__main__ import main
from hndl.errors import HNDLError
from hndl.registry import Registry
from hndl.torch import InferenceCache, build

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "networks"


# ==============================================================================
# Feature 38: IDE Typing & Autocompletion
# ==============================================================================


class TestFeature38IDETyping:
    def test_operator_namespace_dir_and_attributes(self):
        dir_ops = dir(ops)
        assert "linear" in dir_ops
        assert "conv" in dir_ops
        assert "mamba2" in dir_ops
        assert "attention" in dir_ops
        assert dir_ops == sorted(dir_ops)

        custom_reg = Registry()

        @custom_reg.operator("custom_op", identity="custom.op", summary="A custom op.", shape="x[B, ...] -> out[B, ...]")
        class CustomOp(torch.nn.Identity):
            pass

        dir_custom = dir(custom_reg.ops)
        assert "custom_op" in dir_custom
        assert "linear" not in dir_custom

    def test_operator_docstring_and_signature(self):
        linear_op = ops.linear
        assert linear_op.__name__ == "linear"
        assert linear_op.__doc__ is not None
        assert len(linear_op.__doc__) > 0

        import inspect

        sig = inspect.signature(linear_op)
        assert "out_features" in sig.parameters
        assert "bias" in sig.parameters

    def test_export_manifests(self):
        assert "Symbol" in hndl.__all__
        assert "NodeView" in hndl.__all__
        assert "PlanProfile" in hndl.__all__
        assert "PlanDiff" in hndl.__all__
        assert hasattr(hndl, "Symbol")
        assert hasattr(hndl, "NodeView")
        assert hasattr(hndl, "PlanProfile")
        assert hasattr(hndl, "PlanDiff")

    def test_lazy_pep562_torch_import(self):
        import subprocess
        import sys

        cmd = (
            "import sys\n"
            "assert 'torch' not in sys.modules\n"
            "import hndl\n"
            "assert 'torch' not in sys.modules, 'torch was eagerly imported!'\n"
            "_ = hndl.torch\n"
            "assert 'torch' in sys.modules, 'torch was not lazily loaded on attribute access!'\n"
        )
        res = subprocess.run([sys.executable, "-c", cmd], capture_output=True, text=True)
        assert res.returncode == 0, f"PEP 562 lazy torch import failed:\n{res.stderr}"

        torch_mod = getattr(hndl, "torch")
        assert hasattr(torch_mod, "build")
        assert hasattr(torch_mod, "network")
        assert hasattr(torch_mod, "InferenceCache")


# ==============================================================================
# Feature 39: Diagnostics & Provenance Chains
# ==============================================================================


class TestFeature39DiagnosticsAndProvenance:
    def test_fuzzy_operator_suggestions_in_parser(self):
        with pytest.raises(HNDLError, match=r"unknown operator 'linearr'\. Did you mean 'linear'\?"):
            resolve("out = linearr(x, 16)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})

    def test_ops_alias_diagnostic_in_hndl_source(self):
        with pytest.raises(HNDLError, match=r"without 'ops\.' prefix"):
            resolve("out = ops.linear(x, 16)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})

    def test_fuzzy_argument_suggestions_in_registry(self):
        with pytest.raises(HNDLError, match=r"Unknown arguments for linear: 'biass' \(did you mean 'bias'\?\)"):
            resolve("out = linear(x, 16, biass=True)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})

    def test_fuzzy_policy_suggestions_in_registry(self):
        with pytest.raises(HNDLError, match=r"Policy 'same' does not apply to conv2d@1\. Did you mean policy='same3'\?"):
            resolve(
                "out = conv(x, 16, kernel_size=3, policy='same')\n",
                input_shape={"x": ("B", 16, 8, 8)},
                output_shape={"out": ("B", 16, 8, 8)},
            )

    def test_fuzzy_variable_suggestions_in_config(self):
        source = "hidden = linear(x, 32)\nout = relu(hiddn)\n"
        with pytest.raises(HNDLError, match=r"Undefined tensor variable 'hiddn'\. Did you mean 'hidden'\?"):
            resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})

    def test_fuzzy_parameter_target_suggestions_in_torch(self):
        plan = resolve(
            "out = linear(x, 16, init={'weigh': 0.0})\n",
            input_shape={"x": ("B", 16)},
            output_shape={"out": ("B", 16)},
        )
        with pytest.raises(HNDLError, match=r"Parameter target 'weigh' is not a registered parameter\. Did you mean 'weight'\?"):
            hndl.torch.build(plan, device="cpu")

    def test_fuzzy_node_key_suggestions_in_graph_module(self):
        plan = resolve("out = linear(x, 16)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
        model = hndl.torch.build(plan, device="cpu")
        with pytest.raises(KeyError, match=r"n00.*Did you mean 'n0'\?"):
            _ = model["n00"]

    def test_solver_origin_tracking_in_conflict(self):
        source = "h = linear(x, 32)\nout = conv(h, 16, kernel_size=3)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16, 8, 8)})
        err_msg = str(exc_info.value)
        assert "established by" in err_msg
        assert "required by" in err_msg

    def test_solver_pattern_shared_dim_origin_tracking(self):
        source = "a = linear(x, 64)\nb = linear(x, 128)\nout = add(a, b)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 64)})
        err_msg = str(exc_info.value)
        assert "Shared dimension" in err_msg
        assert "established by node 'n0' (linear)" in err_msg
        assert "required by node 'n1' (linear)" in err_msg

    def test_loop_digit_ending_name_separation(self):
        source = "for c in (16, 32):\n  x = linear(x, c, name='pwconv1')\n"
        plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        assert plan.nodes[0].id == "pwconv1_0"
        assert plan.nodes[1].id == "pwconv1_1"

    def test_caller_line_capture_in_callable(self):
        def my_net(x):
            h = ops.linear(x, 32)
            return ops.relu(h)

        plan = resolve_callable(my_net, input_shape=("B", 16), output_shape=("B", 32))
        assert "line" in plan.nodes[0].source
        assert isinstance(plan.nodes[0].source["line"], int)


# ==============================================================================
# Feature 40: Static Zero-Compute Profiler
# ==============================================================================


class TestFeature40StaticProfiler:
    def test_profiler_rejects_non_positive_batch(self):
        plan = resolve("out = linear(x, 32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        with pytest.raises(ValueError, match="positive"):
            plan.profile(batch_size=0)
        with pytest.raises(ValueError, match="positive"):
            plan.profile(batch_size=-4)

    def test_linear_flops_and_params(self):
        plan = resolve("out = linear(x, 32, bias=True)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        prof = plan.profile(batch_size=4)
        assert prof.total_parameters == 544
        assert prof.active_parameters == 544
        assert prof.flops == 2 * 4 * 16 * 32
        assert prof.forward_flops == prof.flops

    def test_conv_flops_and_params(self):
        plan = resolve(
            "out = conv(x, 16, kernel_size=3, padding=1)\n",
            input_shape={"x": ("B", 8, 14, 14)},
            output_shape={"out": ("B", 16, 14, 14)},
        )
        prof = plan.profile(batch_size=2)
        assert prof.total_parameters == 1168
        assert prof.flops == 2 * 2 * 14 * 14 * 16 * (8 * 3 * 3)

    def test_moe_active_vs_total_parameters(self):
        plan = resolve("out = moe(x, 8, 2, 64)\n", input_shape={"x": ("B", 4, 32)}, output_shape={"out": ("B", 4, 32)})
        prof = plan.profile(batch_size=2)
        assert prof.total_parameters > prof.active_parameters
        assert prof.active_parameters > 0
        assert prof.flops > 0

    def test_peak_activation_memory_slot_recycling(self):
        source = "h1 = linear(x, 32)\nh2 = relu(h1)\nh3 = linear(h2, 32)\nout = relu(h3)\n"
        plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})
        prof = plan.profile(batch_size=4)
        total_act = sum(np.activation_bytes for np in prof.nodes)
        assert prof.peak_activation_bytes < total_act
        assert prof.peak_activation_bytes > 0

    def test_kv_state_bytes_for_attention(self):
        plan = resolve("out = attention(x, 4)\n", input_shape={"x": ("B", 16, 32)}, output_shape={"out": ("B", 16, 32)})
        prof = plan.profile(batch_size=2)
        assert prof.state_bytes == 2 * 2 * 16 * 32 * 4
        assert prof.kv_cache_bytes == prof.state_bytes

    def test_exact_param_match_with_torch(self):
        from hndl.__main__ import _load_plan

        for recipe in ["mlp", "gpt_tiny", "lenet5"]:
            plan = _load_plan(EXAMPLES_DIR / f"{recipe}.hndl")
            prof = plan.profile(batch_size=1)
            model_counts = hndl.torch.parameter_counts(plan)
            total_torch = sum(model_counts.values())
            assert prof.total_parameters == total_torch, f"Mismatch on {recipe}: {prof.total_parameters} vs {total_torch}"

    def test_plan_profile_summary_formatting(self):
        plan = resolve("out = linear(x, 32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        prof = plan.profile(batch_size=2)
        summary = prof.summary()
        assert "HNDL Static Plan Profile" in summary
        assert "Total Parameters:" in summary
        assert "Forward FLOPs:" in summary
        assert "Peak Activation" in summary
        assert str(prof) == summary


# ==============================================================================
# Feature 41: Visualization & Diffing
# ==============================================================================


class TestFeature41VisualizationAndDiff:
    def test_mermaid_export_syntax(self):
        source = "h = relu(linear(x, 32))\nout = linear(h, 16)\n"
        plan = resolve(source, input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
        mermaid = plan.to_mermaid()
        assert mermaid.startswith("graph TD")
        assert "subgraph Inputs" in mermaid
        assert "subgraph Outputs" in mermaid
        assert "linear" in mermaid
        assert "relu" in mermaid

    def test_plan_diff_identical_plans(self):
        plan1 = resolve("out = linear(x, 32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        plan2 = resolve("out = linear(x, 32)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 32)})
        diff = plan1.diff(plan2)
        assert diff.identical is True
        assert diff.same_digest is True
        assert len(diff.added_nodes) == 0
        assert len(diff.removed_nodes) == 0
        assert len(diff.modified_nodes) == 0
        assert diff.param_delta == 0
        assert diff.parameter_delta == 0
        assert diff.flop_delta == 0

    def test_plan_diff_structural_changes(self):
        p_base = resolve(
            "h = linear(x, 32)\nout = linear(h, 16)\n",
            input_shape={"x": ("B", 16)},
            output_shape={"out": ("B", 16)},
        )
        p_mod = resolve(
            "h = relu(linear(x, 64))\nout = linear(h, 16)\n",
            input_shape={"x": ("B", 16)},
            output_shape={"out": ("B", 16)},
        )
        diff = p_base.diff(p_mod)
        assert diff.identical is False
        assert diff.same_digest is False
        assert diff.parameter_delta != 0
        summary = str(diff)
        assert "Plan Diff" in summary
        assert "Parameter Delta:" in summary


# ==============================================================================
# Feature 42: CLI Tooling
# ==============================================================================


class TestFeature42CLITooling:
    def test_cli_inspect_basic(self, capsys):
        exit_code = main(["inspect", str(EXAMPLES_DIR / "mlp.hndl")])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Network" in captured or "linear" in captured

    def test_cli_inspect_with_profile(self, capsys):
        exit_code = main(["inspect", str(EXAMPLES_DIR / "mlp.hndl"), "--profile", "--batch-size", "4"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "HNDL Static Plan Profile" in captured
        assert "Total Parameters:" in captured

    def test_cli_mermaid(self, capsys):
        exit_code = main(["mermaid", str(EXAMPLES_DIR / "mlp.hndl")])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "graph TD" in captured

    def test_cli_mermaid_file_output(self, tmp_path):
        out_file = tmp_path / "diagram.mmd"
        exit_code = main(["mermaid", str(EXAMPLES_DIR / "mlp.hndl"), "--out", str(out_file)])
        assert exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "graph TD" in content

    def test_cli_check_single_file(self, capsys):
        exit_code = main(["check", str(EXAMPLES_DIR / "gpt_tiny.hndl")])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "gpt_tiny.hndl" in captured
        assert "passed" in captured.lower()

    def test_cli_check_directory(self, capsys):
        exit_code = main(["check", str(EXAMPLES_DIR)])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "25 passed, 0 failed" in captured

    def test_cli_error_on_nonexistent_file(self):
        exit_code = main(["inspect", "nonexistent_file.hndl"])
        assert exit_code == 1
        exit_code = main(["check", "nonexistent_file.hndl"])
        assert exit_code == 1

    def test_cli_no_args_exits_code_2(self):
        exit_code = main([])
        assert exit_code == 2


# ==============================================================================
# Feature 43: Training & Inference Lifecycle
# ==============================================================================


class TestFeature43Lifecycle:
    def test_param_groups_partitioning(self):
        source = (
            "emb = embedding(idx, 64, 32, name='emb')\n"
            "ln = layer_norm(emb, name='ln')\n"
            "fc = linear(ln, 16, bias=True, name='fc')\n"
            "out = relu(fc, name='relu')\n"
        )
        plan = resolve(
            source,
            input_shape={"idx": ("B", 8)},
            output_shape={"out": ("B", 8, 16)},
            input_dtype={"idx": "int64"},
        )
        model = build(plan, device="cpu")
        groups = model.param_groups(weight_decay=0.05)
        assert len(groups) == 2
        decay_group = groups[0]
        no_decay_group = groups[1]
        assert decay_group["weight_decay"] == 0.05
        assert no_decay_group["weight_decay"] == 0.0

        assert any(p is model["fc"].weight for p in decay_group["params"])
        assert any(p is model["fc"].bias for p in no_decay_group["params"])
        assert any(p is model["ln"].weight for p in no_decay_group["params"])
        assert any(p is model["emb"].weight for p in no_decay_group["params"])

    def test_param_groups_zero_parameter_model(self):
        plan = resolve("out = relu(x)\n", input_shape={"x": ("B", 16)}, output_shape={"out": ("B", 16)})
        model = build(plan, device="cpu")
        groups = model.param_groups(weight_decay=0.01)
        assert len(groups) == 0

    def test_selective_activation_checkpointing(self):
        source = (
            "tb1 = transformer_block(x, 4, name='tb1')\n"
            "tb2 = transformer_block(tb1, 4, name='tb2')\n"
        )
        plan = resolve(source, input_shape={"x": ("B", 8, 32)}, output_shape={"out": ("B", 8, 32)})

        model = build(plan, checkpoint=["tb1"], device="cpu")
        assert model.checkpoint_nodes == frozenset({"tb1"})

        x = torch.randn(2, 8, 32, requires_grad=True)
        model.train()
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and x.grad.isfinite().all()

    def test_graph_aware_depth_scaled_init(self):
        source = "h = linear(x, 32)\nout = add(x, h)\n"
        plan = resolve(source, input_shape={"x": ("B", 32)}, output_shape={"out": ("B", 32)})

        model_default = build(plan, init_policy="default", initialization_seed=42, device="cpu")
        model_scaled = build(plan, init_policy="depth_scaled", initialization_seed=42, device="cpu")

        expected_scale = 1.0 / math.sqrt(2.0)
        std_default = model_default.nodes["n_n0"].weight.std().item()
        std_scaled = model_scaled.nodes["n_n0"].weight.std().item()

        assert math.isclose(std_scaled / std_default, expected_scale, rel_tol=1e-4)

    def test_stateful_inference_cache_container(self):
        cache = InferenceCache()
        assert len(cache) == 0
        assert "n0" not in cache

        t1 = torch.zeros(2, 4, 8)
        cache["n0"] = t1
        assert "n0" in cache
        assert len(cache) == 1
        assert cache["n0"] is t1
        assert cache.get("n0") is t1

        cache.reset("n0")
        assert cache.get("n0") is None

        cache["n1"] = t1
        cache.reset()
        assert len(cache) == 0

    def test_stateful_inference_cache_methods(self):
        cache = InferenceCache()
        t = torch.randn(2, 4, 8)
        cache.set("n0", t)
        assert cache.get("n0") is t
        assert list(cache.keys()) == ["n0"]
        assert list(cache.values()) == [t]
        assert "InferenceCache" in repr(cache)
