"""The generic ``pretrained`` operator: sources, contracts, freezing, dtypes, restore checks."""

import json
import shutil

import pytest
import torch

from hndl import HNDLError, ResolvedPlan, resolve
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
