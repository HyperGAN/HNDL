"""The generic ``pretrained`` operator: sources, contracts, freezing, dtypes, restore checks."""

import hashlib
import json
import shutil

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, ResolvedPlan, resolve
from hndl import pretrained as sources
from hndl.torch import build, network, parameter_counts

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def text_checkpoint(tmp_path):
    config = transformers.GPT2Config(n_layer=1, n_embd=8, n_head=2, vocab_size=16, n_positions=16)
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config)
    path = tmp_path / "tiny-gpt2"
    model.save_pretrained(path, safe_serialization=True)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return path


@pytest.fixture
def vision_checkpoint(tmp_path):
    config = transformers.ViTConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                                    image_size=8, patch_size=4, num_labels=3)
    torch.manual_seed(0)
    model = transformers.ViTForImageClassification(config)
    path = tmp_path / "tiny-vit"
    model.save_pretrained(path, safe_serialization=True)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return path


def test_text_checkpoint_resolves_freezes_and_matches_transformers(text_checkpoint):
    source = f'pretrained("{text_checkpoint}", output="logits", name="lm")'
    plan = resolve(source, input_shape=("B", 5), output_shape=("B", 5, 16), input_dtype="int64")
    node = plan.nodes[0]
    assert node.output_shapes["out"] == ("B", 5, 16)
    assert node.args["revision"] and len(node.args["revision"]) == 16
    assert node.provenance["revision"] == "inferred"
    restored = ResolvedPlan.from_json(plan.to_json())
    assert restored.semantic_digest == plan.semantic_digest
    features = resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    assert features.nodes[0].args["output"] == "features"

    model = build(plan, device=DEVICE)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not model["lm"].model.training
    model.train()
    assert not model["lm"].model.training, "frozen checkpoints stay in eval mode"
    ids = torch.randint(0, 16, (2, 5), device=DEVICE)
    reference = transformers.GPT2LMHeadModel.from_pretrained(text_checkpoint).to(DEVICE).eval()
    with torch.no_grad():
        expected = reference(input_ids=ids).logits
    torch.testing.assert_close(model(x=ids)["output"], expected)
    assert parameter_counts(plan)["lm"] == sum(p.numel() for p in reference.parameters())
    assert "source=" in repr(model["lm"]) and "input_dtype=int64" in repr(model)


def test_trainable_unfreezes_and_a_new_head_trains(text_checkpoint):
    source = f'pretrained("{text_checkpoint}", name="lm")\nlinear(2, name="head")'
    model = network(source, input_shape=("B", 5), output_shape=("B", 5, 2), input_dtype="int64", device=DEVICE)
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen and all(name.startswith("nodes.n_lm.") for name in frozen)
    assert model["head"].weight.requires_grad
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    ids = torch.randint(0, 16, (2, 5), device=DEVICE)
    model(ids).square().mean().backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]) == name.startswith("nodes.n_lm.")

    tuned = network(f'pretrained("{text_checkpoint}", trainable=True)', input_shape=("B", 5), output_shape=("B", 5, 8),
                    input_dtype="int64", device=DEVICE)
    assert all(parameter.requires_grad for parameter in tuned.parameters())
    tuned.train()
    assert tuned["n0"].model.training
    tuned.eval()
    assert not tuned["n0"].model.training


def test_vision_checkpoint_fixes_the_input_contract_and_selects_outputs(vision_checkpoint):
    plan = resolve(f'pretrained("{vision_checkpoint}", output="logits")', input_shape=("B", 3, 8, 8), output_shape=("B", 3))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3)
    pooled = resolve(f'pretrained("{vision_checkpoint}", output="pooled")', input_shape=("B", 3, 8, 8), output_shape=("B", 8))
    assert pooled.nodes[0].output_shapes["out"] == ("B", 8)
    hidden = resolve(f'pretrained("{vision_checkpoint}", output="last_hidden_state")', input_shape=("B", 3, 8, 8),
                     output_shape=("B", 5, 8))
    assert hidden.nodes[0].output_shapes["out"] == ("B", 5, 8)
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(f'pretrained("{vision_checkpoint}", output="logits")', input_shape=("B", 3, 16, 16), output_shape=("B", 3))
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve(f'pretrained("{vision_checkpoint}", output="logits")', input_shape=("B", 3, 8, 8), output_shape=("B", 3),
                input_dtype="int64")
    with pytest.raises(HNDLError, match="E_PRETRAINED"):
        resolve(f'pretrained("{vision_checkpoint}", output="nonsense")', input_shape=("B", 3, 8, 8), output_shape=("B", 3))
    model = build(plan, device=DEVICE)
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    reference = transformers.ViTForImageClassification.from_pretrained(vision_checkpoint).to(DEVICE).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x=pixels)["output"], reference(pixel_values=pixels).logits)


def test_text_checkpoint_requires_integer_ids(text_checkpoint):
    with pytest.raises(HNDLError, match="E_DTYPE.*int64"):
        resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 5), output_shape=("B", 5, 8))
    with pytest.raises(HNDLError, match="E_CONSTRAINT|E_SCHEMA"):
        resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 3, 5), output_shape=("B", 5, 8), input_dtype="int64")


def test_bare_safetensors_needs_config_and_invalid_sources_fail(text_checkpoint, tmp_path):
    weights = text_checkpoint / "model.safetensors"
    config = text_checkpoint / "config.json"
    plan = resolve(f'pretrained("{weights}", config="{config}", output="logits")', input_shape=("B", 5),
                   output_shape=("B", 5, 16), input_dtype="int64")
    model = build(plan, device=DEVICE)
    whole = build(resolve(f'pretrained("{text_checkpoint}", output="logits")', input_shape=("B", 5),
                          output_shape=("B", 5, 16), input_dtype="int64"), device=DEVICE)
    ids = torch.randint(0, 16, (2, 5), device=DEVICE)
    torch.testing.assert_close(model(x=ids)["output"], whole(x=ids)["output"])
    for source, kwargs in [
        (str(weights), {}),
        (str(tmp_path / "missing"), {}),
        ("hf://nope", {}),
        (str(text_checkpoint), {"config": str(config)}),
    ]:
        arguments = ", ".join(f'{key}="{value}"' for key, value in kwargs.items())
        with pytest.raises(HNDLError, match="E_PRETRAINED"):
            resolve(f'pretrained("{source}"{", " + arguments if arguments else ""})', input_shape=("B", 5),
                    output_shape=("B", 5, 8), input_dtype="int64")


def test_restore_rejects_a_source_whose_checkpoint_changed(text_checkpoint):
    plan = resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    encoded = plan.to_json()
    assert ResolvedPlan.from_json(encoded).nodes[0].args["revision"] == plan.nodes[0].args["revision"]
    config = json.loads((text_checkpoint / "config.json").read_text())
    config["hndl_marker"] = 1
    (text_checkpoint / "config.json").write_text(json.dumps(config))
    sources.resolve_source.cache_clear()
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*revision"):
        ResolvedPlan.from_json(encoded)


def test_directory_copies_share_a_revision_and_differ_by_path(text_checkpoint, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(text_checkpoint, copy)
    first = resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    second = resolve(f'pretrained("{copy}")', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    assert first.nodes[0].args["revision"] == second.nodes[0].args["revision"]
    assert first.semantic_digest != second.semantic_digest


def test_resolve_source_detects_inplace_config_mutation_without_cache_clear(text_checkpoint):
    first = sources.resolve_source(str(text_checkpoint))
    config_file = text_checkpoint / "config.json"
    config = json.loads(config_file.read_text())
    config["dummy_marker"] = 42
    config_file.write_text(json.dumps(config))
    second = sources.resolve_source(str(text_checkpoint))
    assert first.revision != second.revision


def test_resolve_source_detects_inplace_weight_mutation_without_cache_clear(text_checkpoint):
    first = sources.resolve_source(str(text_checkpoint))
    weights_file = text_checkpoint / "model.safetensors"
    data = bytearray(weights_file.read_bytes())
    data[-1] ^= 0xFF
    weights_file.write_bytes(data)
    second = sources.resolve_source(str(text_checkpoint))
    assert first.revision != second.revision


def test_bare_safetensors_revision_detects_weight_mutation(text_checkpoint):
    weights = text_checkpoint / "model.safetensors"
    config = text_checkpoint / "config.json"
    first = sources.resolve_source(str(weights), str(config))
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(data)
    second = sources.resolve_source(str(weights), str(config))
    assert first.revision != second.revision


def test_restore_rejects_inplace_weight_mutation_without_cache_clear(text_checkpoint):
    plan = resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    encoded = plan.to_json()
    weights_file = text_checkpoint / "model.safetensors"
    data = bytearray(weights_file.read_bytes())
    data[-1] ^= 0xFF
    weights_file.write_bytes(data)
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*revision"):
        ResolvedPlan.from_json(encoded)


def test_pretrained_trainable_mode_syncs_wrapped_model(text_checkpoint):
    plan_frozen = resolve(f'pretrained("{text_checkpoint}")', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    model_frozen = build(plan_frozen, device=DEVICE)
    assert not model_frozen["n0"].model.training
    plan_trainable = resolve(f'pretrained("{text_checkpoint}", trainable=True)', input_shape=("B", 5), output_shape=("B", 5, 8), input_dtype="int64")
    model_trainable = build(plan_trainable, device=DEVICE)
    assert model_trainable["n0"].model.training


def test_malformed_config_json_raises_e_pretrained(tmp_path):
    bad_dir = tmp_path / "bad_checkpoint"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("{malformed: json", encoding="utf-8")
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(bad_dir))
    assert exc_info.value.code == "E_PRETRAINED"


def test_invalid_utf8_config_json_raises_e_pretrained(tmp_path):
    bad_dir = tmp_path / "bad_checkpoint_utf8"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_bytes(b"\x80\x81\x82")
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(bad_dir))
    assert exc_info.value.code == "E_PRETRAINED"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
def test_reduced_precision_build_casts_the_checkpoint(text_checkpoint):
    model = network(f'pretrained("{text_checkpoint}", output="logits")', input_shape=("B", 5), output_shape=("B", 5, 16),
                    input_dtype="int64", dtype="bfloat16", device="cuda:0")
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    output = model(torch.randint(0, 16, (2, 5), device="cuda:0"))
    assert output.dtype == torch.bfloat16 and output.shape == (2, 5, 16)


@pytest.mark.network
class TestHubCheckpoints:
    def test_gpt2_logits_match_transformers_and_run_in_bfloat16(self):
        ids = torch.randint(0, 50257, (2, 8), device=DEVICE)
        model = network('pretrained("hf://openai-community/gpt2", output="logits")', input_shape=("B", 8),
                        output_shape=("B", 8, 50257), input_dtype="int64", device=DEVICE)
        reference = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEVICE).eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids), reference(input_ids=ids).logits)
        assert sum(p.numel() for p in model.parameters()) == 124_439_808
        if torch.cuda.is_available():
            half = network('pretrained("hf://openai-community/gpt2", output="features")\nlinear(2)', input_shape=("B", 8),
                           output_shape=("B", 8, 2), input_dtype="int64", dtype="bfloat16", device="cuda:0")
            output = half(ids)
            assert output.dtype == torch.bfloat16 and output.shape == (2, 8, 2)
            output.float().square().mean().backward()
            assert half["n1"].weight.grad is not None

    def test_timm_resnet18_matches_timm(self):
        timm = pytest.importorskip("timm")
        pixels = torch.randn(2, 3, 224, 224, device=DEVICE)
        model = network('pretrained("hf://timm/resnet18.a1_in1k", output="logits")', input_shape=("B", 3, 224, 224),
                        output_shape=("B", 1000), device=DEVICE)
        reference = timm.create_model("hf-hub:timm/resnet18.a1_in1k", pretrained=True).to(DEVICE).eval()
        with torch.no_grad():
            torch.testing.assert_close(model(pixels), reference(pixels))
        assert sum(p.numel() for p in model.parameters()) == 11_689_512
        pooled = resolve('pretrained("hf://timm/resnet18.a1_in1k", output="pooled")', input_shape=("B", 3, 224, 224),
                         output_shape=("B", 512))
        assert pooled.nodes[0].output_shapes["out"] == ("B", 512)

    def test_vit_clip_and_dinov2_contracts(self):
        vit = resolve('pretrained("hf://google/vit-base-patch16-224", output="logits")', input_shape=("B", 3, 224, 224),
                      output_shape=("B", 1000))
        assert vit.nodes[0].output_shapes["out"] == ("B", 1000)
        clip_vision = network('pretrained("hf://openai/clip-vit-base-patch32", component="vision", output="embeds")',
                              input_shape=("B", 3, 224, 224), output_shape=("B", 512), device=DEVICE)
        clip_text = network('pretrained("hf://openai/clip-vit-base-patch32", component="text", output="embeds")',
                            input_shape=("B", 7), output_shape=("B", 512), input_dtype="int64", device=DEVICE)
        pixels = torch.randn(1, 3, 224, 224, device=DEVICE)
        ids = torch.randint(0, 49408, (1, 7), device=DEVICE)
        vision_reference = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        text_reference = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        with torch.no_grad():
            torch.testing.assert_close(clip_vision(pixels), vision_reference(pixel_values=pixels).image_embeds)
            torch.testing.assert_close(clip_text(ids), text_reference(input_ids=ids).text_embeds)
        assert clip_vision(pixels).shape == (1, 512)
        with pytest.raises(HNDLError, match="E_PRETRAINED.*component"):
            resolve('pretrained("hf://openai/clip-vit-base-patch32")', input_shape=("B", 3, 224, 224), output_shape=("B", 512))
        dino = resolve('pretrained("hf://facebook/dinov2-small", output="pooled")', input_shape=("B", 3, 518, 518),
                       output_shape=("B", 384))
        assert dino.nodes[0].output_shapes["out"] == ("B", 384)


class MetaCrashingModule(nn.Module):
    """Simulates a custom Triton/CUDA kernel layer that cannot run on PyTorch meta device."""

    def __init__(self):
        super().__init__()
        if torch.empty(0).device.type == "meta":
            raise RuntimeError("Custom Triton/CUDA kernel does not support meta device")
        self.linear = nn.Linear(16, 32)
        self.layer1 = nn.Linear(16, 24)
        self.layer2 = nn.Linear(24, 32)

    def forward(self, x):
        if x.device.type == "meta":
            raise RuntimeError("Custom Triton/CUDA kernel forward does not support meta device")
        return self.layer2(self.layer1(x))


class TokenEmbeddingProvider(nn.Module):
    """Local provider that consumes int64 token IDs and outputs embeddings."""

    def __init__(self, vocab_size=64, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, 48)

    def forward(self, x):
        if x.dtype not in (torch.int64, torch.long):
            raise TypeError(f"Expected int64 tensor, got {x.dtype}")
        if x.ndim != 2:
            raise ValueError(f"Expected rank 2 [B, T], got rank {x.ndim}")
        return self.proj(self.embedding(x))


def test_provider_with_shape_fn_bypasses_meta_device(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "meta_crash.pth"
    torch.save(MetaCrashingModule().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    # 1. Register with shape_fn: resolution and build MUST succeed without meta crash
    def shape_fn(input_shape, **kwargs):
        return (input_shape[0], 32)

    reg.pretrained_provider("meta_kernel", MetaCrashingModule, shape_fn=shape_fn)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()

    plan = resolve(f'pretrained("{path}", provider="meta_kernel", sha256="{digest}")',
                   input_shape=("B", 16), output_shape=("B", 32), registry=reg)
    assert plan.nodes[0].output_shapes["out"] == ("B", 32)

    # Build and verify forward pass on CPU
    model = build(plan, device="cpu")
    x = torch.randn(3, 16)
    out = model(x=x)["output"]
    assert out.shape == (3, 32)

    # 2. Register without shape_fn: resolution MUST fail because meta device crashes
    reg_fail = Registry.builtins()
    reg_fail.pretrained_provider("meta_kernel_fail", MetaCrashingModule)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*Custom Triton/CUDA kernel"):
        resolve(f'pretrained("{path}", provider="meta_kernel_fail", sha256="{digest}")',
                input_shape=("B", 16), output_shape=("B", 32), registry=reg_fail)


def test_provider_with_shape_fn_multi_layer_outputs(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "multi_layer.pth"
    torch.save(MetaCrashingModule().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def shape_fn(input_shape, layers=(), **kwargs):
        layer_dims = {"layer1": 24, "layer2": 32}
        return tuple((input_shape[0], layer_dims[name]) for name in layers)

    reg.pretrained_provider("multi_layer_provider", MetaCrashingModule, shape_fn=shape_fn)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()

    plan = resolve(f'f1, f2 = pretrained("{path}", provider="multi_layer_provider", sha256="{digest}", '
                   f'layers=("layer1", "layer2"))',
                   input_shape=("B", 16), output_shape={"f1": ("B", 24), "f2": ("B", 32)}, registry=reg)
    assert plan.nodes[0].output_shapes["out0"] == ("B", 24)
    assert plan.nodes[0].output_shapes["out1"] == ("B", 32)


def test_provider_shape_fn_signature_flexibility(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "signatures.pth"
    torch.save(MetaCrashingModule().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    # Signature with only input_shape
    reg.pretrained_provider("p_single_arg", MetaCrashingModule, shape_fn=lambda s: (s[0], 32))
    plan1 = resolve(f'pretrained("{path}", provider="p_single_arg", sha256="{digest}")',
                    input_shape=("B", 16), output_shape=("B", 32), registry=reg)
    assert plan1.nodes[0].output_shapes["out"] == ("B", 32)

    # Signature with input_shape and output
    reg.pretrained_provider("p_two_args", MetaCrashingModule, shape_fn=lambda s, out: (s[0], 32))
    plan2 = resolve(f'pretrained("{path}", provider="p_two_args", sha256="{digest}")',
                    input_shape=("B", 16), output_shape=("B", 32), registry=reg)
    assert plan2.nodes[0].output_shapes["out"] == ("B", 32)


def test_provider_shape_fn_error_handling(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "err.pth"
    torch.save(MetaCrashingModule().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def bad_shape_fn(input_shape, **kwargs):
        raise RuntimeError("Something went wrong inside shape calculation")

    reg.pretrained_provider("p_err", MetaCrashingModule, shape_fn=bad_shape_fn)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*shape_fn failed"):
        resolve(f'pretrained("{path}", provider="p_err", sha256="{digest}")',
                input_shape=("B", 16), output_shape=("B", 32), registry=reg)


def test_provider_contract_ids_accepts_int64_tokens(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "tokens.pth"
    torch.save(TokenEmbeddingProvider().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    reg.pretrained_provider("token_provider", TokenEmbeddingProvider, contract="ids")
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()

    # 1. Resolves and builds with int64 inputs and rank 2 [B, T]
    plan = resolve(f'pretrained("{path}", provider="token_provider", sha256="{digest}")',
                   input_shape=("B", 12), output_shape=("B", 12, 48), input_dtype="int64", registry=reg)
    assert plan.nodes[0].output_shapes["out"] == ("B", 12, 48)

    model = build(plan, device="cpu")
    ids = torch.randint(0, 64, (2, 12), dtype=torch.int64)
    out = model(x=ids)["output"]
    assert out.shape == (2, 12, 48)
    assert out.dtype == torch.float32

    # 2. Rejects floating-point input with E_DTYPE
    with pytest.raises(HNDLError, match="E_DTYPE.*declare input_dtype=\"int64\""):
        resolve(f'pretrained("{path}", provider="token_provider", sha256="{digest}")',
                input_shape=("B", 12), output_shape=("B", 12, 48), input_dtype="float32", registry=reg)

    # 3. Rejects rank != 2
    with pytest.raises(HNDLError, match="E_CONSTRAINT|E_SCHEMA"):
        resolve(f'pretrained("{path}", provider="token_provider", sha256="{digest}")',
                input_shape=("B", 3, 12), output_shape=("B", 12, 48), input_dtype="int64", registry=reg)


def test_provider_contract_ids_combined_with_shape_fn(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "tokens_shape_fn.pth"
    torch.save(TokenEmbeddingProvider().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def token_shape_fn(input_shape, **kwargs):
        return (input_shape[0], input_shape[1], 48)

    reg.pretrained_provider("token_fast", TokenEmbeddingProvider, contract="ids", shape_fn=token_shape_fn)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()

    plan = resolve(f'pretrained("{path}", provider="token_fast", sha256="{digest}")',
                   input_shape=("B", 20), output_shape=("B", 20, 48), input_dtype="int64", registry=reg)
    assert plan.nodes[0].output_shapes["out"] == ("B", 20, 48)

    model = build(plan, device="cpu")
    ids = torch.randint(0, 64, (3, 20), dtype=torch.int64)
    out = model(x=ids)["output"]
    assert out.shape == (3, 20, 48)


def test_provider_contract_backward_compatibility(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "compat.pth"
    torch.save(nn.Linear(8, 16).state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    # Omitted contract and shape_fn defaults to "tensor" and None
    reg.pretrained_provider("compat_linear", lambda: nn.Linear(8, 16))
    assert reg.pretrained_contract("compat_linear") == "tensor"
    assert reg.pretrained_shape_fn("compat_linear") is None

    # Resolves with floating point
    plan = resolve(f'pretrained("{path}", provider="compat_linear", sha256="{digest}")',
                   input_shape=("B", 8), output_shape=("B", 16), registry=reg)
    assert plan.nodes[0].output_shapes["out"] == ("B", 16)

    # Rejects int64 on contract="tensor" with E_DTYPE
    with pytest.raises(HNDLError, match="E_DTYPE.*floating-point"):
        resolve(f'pretrained("{path}", provider="compat_linear", sha256="{digest}")',
                input_shape=("B", 8), output_shape=("B", 16), input_dtype="int64", registry=reg)

    # Invalid contract raises E_REGISTRY
    with pytest.raises(HNDLError, match="E_REGISTRY.*contract must be 'tensor', 'ids', or 'pixels'"):
        reg.pretrained_provider("bad_contract", lambda: nn.Linear(8, 16), contract="invalid_contract")

    # Non-callable shape_fn raises E_REGISTRY
    with pytest.raises(HNDLError, match="E_REGISTRY.*shape_fn must be a callable"):
        reg.pretrained_provider("bad_shape_fn", lambda: nn.Linear(8, 16), shape_fn="not_a_callable")

