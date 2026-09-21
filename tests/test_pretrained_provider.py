"""Local ``.pth`` checkpoints loaded through host-registered architecture providers."""

import copy
import hashlib

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, ResolvedPlan, resolve
from hndl import pretrained as sources
from hndl.torch import build, network, parameter_counts

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TinyExtractor(nn.Module):
    """A feature extractor whose intermediate layers have dotted names."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(4, 5, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(5, 2)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


class OtherExtractor(TinyExtractor):
    """A second architecture, used to check that a mismatched state dict fails."""

    def __init__(self):
        super().__init__()
        self.extra = nn.Linear(2, 2)


class TinyViT(nn.Module):
    """A patch-token model shaped like DINOv2: forward_features + get_intermediate_layers."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 6, 2, stride=2)
        self.blocks = nn.ModuleList(nn.Linear(6, 6) for _ in range(4))
        self.norm = nn.LayerNorm(6)

    def _tokens(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        outputs = []
        for block in self.blocks:
            tokens = block(tokens)
            outputs.append(tokens)
        return outputs

    def forward_features(self, x):
        tokens = self.norm(self._tokens(x)[-1])
        return {"x_norm_clstoken": tokens[:, 0], "x_norm_patchtokens": tokens}

    def get_intermediate_layers(self, x, n=(1, 3), reshape=False, norm=True):
        outputs = self._tokens(x)
        picked = tuple(self.norm(outputs[index]) if norm else outputs[index] for index in n)
        if reshape:
            side = int(round(picked[0].shape[1] ** 0.5))
            picked = tuple(tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side) for tokens in picked)
        return picked

    def forward(self, x):
        return self.forward_features(x)["x_norm_clstoken"]


PATCH_TOKENS = lambda model, x: model.forward_features(x)["x_norm_patchtokens"]  # noqa: E731
LAYERS_1_3 = lambda model, x: torch.cat(  # noqa: E731
    model.get_intermediate_layers(x, n=(1, 3), reshape=True, norm=True), dim=1)
CLS_TOKEN = lambda model, x: model.forward_features(x)["x_norm_clstoken"]  # noqa: E731
EVERY_LAYER = lambda model, x: model.get_intermediate_layers(x, n=(1, 3))  # noqa: E731


def digest_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(path, digest, **arguments):
    rendered = "".join(f', {name}="{value}"' for name, value in arguments.items())
    return f'pretrained("{path}", provider="tiny", sha256="{digest}"{rendered})'


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "tiny.pth"
    torch.save(TinyExtractor().state_dict(), path)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return path


@pytest.fixture
def registry():
    registry = Registry.builtins()
    registry.pretrained_provider("tiny", TinyExtractor)
    return registry


@pytest.fixture
def vit_checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "vit.pth"
    torch.save(TinyViT().state_dict(), path)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return path


@pytest.fixture
def vit_registry():
    registry = Registry.builtins()
    registry.pretrained_provider("vit", TinyViT, readouts={"patch_tokens": PATCH_TOKENS, "layers_1_3": LAYERS_1_3})
    return registry


@pytest.fixture
def vit_reference(vit_checkpoint):
    model = TinyViT()
    model.load_state_dict(torch.load(vit_checkpoint, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


def vit_call(path, digest, **arguments):
    rendered = "".join(f', {name}="{value}"' for name, value in arguments.items())
    return f'pretrained("{path}", provider="vit", sha256="{digest}"{rendered})'


@pytest.fixture
def reference(checkpoint):
    model = TinyExtractor()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


def test_intermediate_layer_matches_a_manual_hook_and_stays_frozen(checkpoint, registry, reference):
    digest = digest_of(checkpoint)
    plan = resolve(call(checkpoint, digest, layer="features.2"), input_shape=("B", 3, 8, 8),
                   output_shape=("B", 5, 8, 8), registry=registry)
    node = plan.nodes[0]
    assert node.output_shapes["out"] == ("B", 5, 8, 8)
    assert node.args["revision"] == digest[:16]
    assert node.provenance["revision"] == "inferred"

    model = build(plan, device=DEVICE, registry=registry)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not model["n0"].model.training
    model.train()
    assert not model["n0"].model.training, "frozen provider checkpoints stay in eval mode"
    assert "provider='tiny'" in repr(model["n0"]) and "layer='features.2'" in repr(model["n0"])

    captured = []
    handle = reference.features[2].register_forward_hook(lambda module, inputs, output: captured.append(output))
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    with torch.no_grad():
        reference(pixels)
    handle.remove()
    torch.testing.assert_close(model(x=pixels)["output"], captured[0])


def test_the_whole_model_output_is_the_default_and_gradients_reach_a_new_head(checkpoint, registry, reference):
    digest = digest_of(checkpoint)
    model = network(call(checkpoint, digest), input_shape=("B", 3, 8, 8), output_shape=("B", 2),
                    registry=registry, device=DEVICE)
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(model(pixels), reference(pixels))

    perceptual = network(f'{call(checkpoint, digest, layer="features.2")}\nconv(2, kernel_size=1)',
                         input_shape=("B", 3, 8, 8), output_shape=("B", 2, 8, 8), registry=registry, device=DEVICE)
    perceptual(pixels).square().mean().backward()
    assert perceptual["n1"].weight.grad is not None
    assert all(parameter.grad is None for parameter in perceptual["n0"].parameters())


def test_a_pinned_digest_is_required_and_verified(checkpoint, registry, tmp_path):
    digest = digest_of(checkpoint)
    shapes = {"input_shape": ("B", 3, 8, 8), "output_shape": ("B", 2), "registry": registry}
    with pytest.raises(HNDLError, match="E_PRETRAINED.*sha256"):
        resolve(f'pretrained("{checkpoint}", provider="tiny")', **shapes)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*sha256"):
        resolve(call(checkpoint, "0" * 64), **shapes)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*64 lowercase hex"):
        resolve(call(checkpoint, digest.upper()), **shapes)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*describes no architecture"):
        resolve(f'pretrained("{checkpoint}")', **shapes)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*does not exist"):
        resolve(call(tmp_path / "absent.pth", digest), **shapes)

    plain = Registry.builtins()
    with pytest.raises(HNDLError, match="E_PRETRAINED.*No pretrained provider"):
        resolve(call(checkpoint, digest), input_shape=("B", 3, 8, 8), output_shape=("B", 2), registry=plain)


def test_a_wrong_layer_lists_the_available_submodules(checkpoint, registry):
    with pytest.raises(HNDLError, match=r"E_PRETRAINED.*no submodule 'features\.9'.*features\.2.*head"):
        resolve(call(checkpoint, digest_of(checkpoint), layer="features.9"), input_shape=("B", 3, 8, 8),
                output_shape=("B", 5, 8, 8), registry=registry)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*layer="):
        resolve(call(checkpoint, digest_of(checkpoint), output="pooled"), input_shape=("B", 3, 8, 8),
                output_shape=("B", 2), registry=registry)


def test_a_state_dict_that_does_not_fit_the_provider_fails_at_build(checkpoint, registry):
    registry.pretrained_provider("other", OtherExtractor)
    source = call(checkpoint, digest_of(checkpoint)).replace('provider="tiny"', 'provider="other"')
    plan = resolve(source, input_shape=("B", 3, 8, 8), output_shape=("B", 2), registry=registry)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*does not fit provider"):
        build(plan, device=DEVICE, registry=registry)


def test_the_plan_round_trips_and_a_changed_file_fails_restore(checkpoint, registry):
    plan = resolve(call(checkpoint, digest_of(checkpoint), layer="features.2"), input_shape=("B", 3, 8, 8),
                   output_shape=("B", 5, 8, 8), registry=registry)
    encoded = plan.to_json()
    restored = ResolvedPlan.from_json(encoded, registry=registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.nodes[0].args["sha256"] == plan.nodes[0].args["sha256"]

    with checkpoint.open("ab") as handle:
        handle.write(b"\0")
    # The load path verifies the file even when resolution answered from its cache.
    with pytest.raises(HNDLError, match="E_PRETRAINED.*sha256"):
        build(plan, device=DEVICE, registry=registry)
    sources.resolve_source.cache_clear()
    with pytest.raises(HNDLError, match="E_PRETRAINED.*checkpoint file changed"):
        ResolvedPlan.from_json(encoded, registry=registry)


def test_parameter_counts_never_reads_the_checkpoint(checkpoint, registry):
    plan = resolve(call(checkpoint, digest_of(checkpoint), layer="features.2"), input_shape=("B", 3, 8, 8),
                   output_shape=("B", 5, 8, 8), registry=registry)
    expected = sum(parameter.numel() for parameter in TinyExtractor().parameters())
    checkpoint.unlink()
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    assert parameter_counts(plan, registry=registry) == {"n0": expected}
    with pytest.raises(HNDLError, match="E_PRETRAINED"):
        build(plan, device=DEVICE, registry=registry)


def test_providers_belong_to_one_registry(registry):
    assert registry.pretrained_providers == ("tiny",)
    assert registry.pretrained_builder("tiny") is TinyExtractor
    assert Registry.builtins().pretrained_providers == ()
    assert Registry.builtins().pretrained_builder("tiny") is None
    assert Registry.active() is None
    with pytest.raises(HNDLError, match="E_REGISTRY.*Duplicate"):
        registry.pretrained_provider("tiny", TinyExtractor)
    with pytest.raises(HNDLError, match="E_REGISTRY.*callable"):
        registry.pretrained_provider("broken", "TinyExtractor")
    with pytest.raises(HNDLError, match="E_REGISTRY.*name"):
        registry.pretrained_provider("", TinyExtractor)


def test_a_provider_returning_something_other_than_a_module_fails(checkpoint, registry):
    registry.pretrained_provider("bogus", lambda: "not a module")
    source = call(checkpoint, digest_of(checkpoint)).replace('provider="tiny"', 'provider="bogus"')
    with pytest.raises(HNDLError, match="E_PRETRAINED.*not an nn.Module"):
        resolve(source, input_shape=("B", 3, 8, 8), output_shape=("B", 2), registry=registry)


def test_a_readout_returns_the_tensor_its_registered_callable_selects(vit_checkpoint, vit_registry, vit_reference):
    plan = resolve(vit_call(vit_checkpoint, digest_of(vit_checkpoint), readout="patch_tokens"),
                   input_shape=("B", 3, 8, 8), output_shape=("B", 16, 6), registry=vit_registry)
    # 8x8 pixels -> 4x4 patches -> 16 tokens of width 6, traced on the meta device.
    assert plan.nodes[0].output_shapes["out"] == ("B", 16, 6)
    assert plan.nodes[0].args["readout"] == "patch_tokens" and plan.nodes[0].args["layer"] == ""

    model = build(plan, device=DEVICE, registry=vit_registry)
    assert "readout='patch_tokens'" in repr(model["n0"]) and "layer=" not in repr(model["n0"])
    assert not any(parameter.requires_grad for parameter in model.parameters())
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    with torch.no_grad():
        expected = vit_reference.forward_features(pixels)["x_norm_patchtokens"]
    torch.testing.assert_close(model(x=pixels)["output"], expected)


def test_a_readout_may_concatenate_the_intermediate_layers_it_asks_for(vit_checkpoint, vit_registry, vit_reference):
    model = network(vit_call(vit_checkpoint, digest_of(vit_checkpoint), readout="layers_1_3"),
                    input_shape=("B", 3, 8, 8), output_shape=("B", 12, 4, 4), registry=vit_registry, device=DEVICE)
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    with torch.no_grad():
        expected = torch.cat(vit_reference.get_intermediate_layers(pixels, n=(1, 3), reshape=True, norm=True), dim=1)
        torch.testing.assert_close(model(pixels), expected)

    # A copy of the built network keeps the readout and answers identically.
    duplicate = copy.deepcopy(model)
    with torch.no_grad():
        torch.testing.assert_close(duplicate(pixels), expected)


def test_a_readout_can_be_added_to_a_provider_that_is_already_registered(vit_checkpoint, vit_registry, vit_reference):
    vit_registry.pretrained_readout("vit", "cls_token", CLS_TOKEN)
    assert sorted(vit_registry.pretrained_readouts("vit")) == ["cls_token", "layers_1_3", "patch_tokens"]
    assert vit_registry.pretrained_readouts("absent") == {}
    model = network(vit_call(vit_checkpoint, digest_of(vit_checkpoint), readout="cls_token"),
                    input_shape=("B", 3, 8, 8), output_shape=("B", 6), registry=vit_registry, device=DEVICE)
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(model(pixels), vit_reference.forward_features(pixels)["x_norm_clstoken"])

    with pytest.raises(HNDLError, match="E_REGISTRY.*Duplicate pretrained readout"):
        vit_registry.pretrained_readout("vit", "cls_token", CLS_TOKEN)
    with pytest.raises(HNDLError, match="E_REGISTRY.*No pretrained provider"):
        vit_registry.pretrained_readout("absent", "cls_token", CLS_TOKEN)
    with pytest.raises(HNDLError, match="E_REGISTRY.*callable"):
        vit_registry.pretrained_readout("vit", "broken", "not callable")
    with pytest.raises(HNDLError, match="E_REGISTRY.*readout name"):
        vit_registry.pretrained_readout("vit", "", CLS_TOKEN)
    with pytest.raises(HNDLError, match="E_REGISTRY.*mapping"):
        Registry.builtins().pretrained_provider("vit", TinyViT, readouts=[("cls_token", CLS_TOKEN)])


def test_an_unknown_or_conflicting_readout_fails_with_the_registered_names(vit_checkpoint, vit_registry):
    digest = digest_of(vit_checkpoint)
    shapes = {"input_shape": ("B", 3, 8, 8), "output_shape": ("B", 16, 6), "registry": vit_registry}
    with pytest.raises(HNDLError, match="E_PRETRAINED.*no readout named 'patchtokens'.*layers_1_3, patch_tokens"):
        resolve(vit_call(vit_checkpoint, digest, readout="patchtokens"), **shapes)
    bare = Registry.builtins()
    bare.pretrained_provider("vit", TinyViT)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*no readout named 'patch_tokens'.*registered: \\(none\\)"):
        resolve(vit_call(vit_checkpoint, digest, readout="patch_tokens"),
                input_shape=("B", 3, 8, 8), output_shape=("B", 16, 6), registry=bare)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*both say what the node returns"):
        resolve(vit_call(vit_checkpoint, digest, layer="blocks.1", readout="patch_tokens"), **shapes)
    with pytest.raises(HNDLError, match='E_PRETRAINED.*readout="<registered readout>"'):
        resolve(vit_call(vit_checkpoint, digest, readout="patch_tokens", output="pooled"), **shapes)

    # transformers and timm checkpoints have no readouts; the rejection needs no library installed.
    timm_source = sources.Source("directory", "/checkpoints/resnet18", "r0", {"architecture": "resnet18"},
                                 "/checkpoints/resnet18/config.json", "timm")
    with pytest.raises(HNDLError, match="E_PRETRAINED.*readout=.*select output="):
        sources.provider_for(timm_source, "features", "", "", "patch_tokens")


def test_a_readout_that_returns_more_than_one_tensor_fails_clearly(vit_checkpoint, vit_registry):
    vit_registry.pretrained_readout("vit", "every_layer", EVERY_LAYER)
    with pytest.raises(HNDLError, match="E_PRETRAINED.*returned tuple, not a tensor.*torch.cat"):
        resolve(vit_call(vit_checkpoint, digest_of(vit_checkpoint), readout="every_layer"),
                input_shape=("B", 3, 8, 8), output_shape=("B", 16, 6), registry=vit_registry)


def test_a_readout_plan_round_trips_and_a_config_without_one_is_unchanged(vit_checkpoint, vit_registry):
    checksum = digest_of(vit_checkpoint)
    plan = resolve(vit_call(vit_checkpoint, checksum, readout="patch_tokens"), input_shape=("B", 3, 8, 8),
                   output_shape=("B", 16, 6), registry=vit_registry)
    encoded = plan.to_json()
    assert '"readout":"patch_tokens"' in encoded
    restored = ResolvedPlan.from_json(encoded, registry=vit_registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.nodes[0].args["readout"] == "patch_tokens"
    assert restored.nodes[0].output_shapes["out"] == ("B", 16, 6)
    build(restored, device=DEVICE, registry=vit_registry)

    # A config written before readouts existed resolves as it always did: the new
    # argument only takes its default, which reads the same as writing it out.
    shapes = {"input_shape": ("B", 3, 8, 8), "output_shape": ("B", 6), "registry": vit_registry}
    plain = resolve(vit_call(vit_checkpoint, checksum), **shapes)
    explicit = resolve(vit_call(vit_checkpoint, checksum, readout=""), **shapes)
    assert plain.nodes[0].args["readout"] == ""
    assert explicit.semantic_digest == plain.semantic_digest
    assert plain.semantic_digest != plan.semantic_digest
    assert ResolvedPlan.from_json(plain.to_json(), registry=vit_registry).nodes[0].args["readout"] == ""
