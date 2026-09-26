"""Local ``.pth`` checkpoints loaded through host-registered architecture providers."""

import copy
import hashlib
import json

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, ResolvedPlan, ops, resolve, resolve_callable
from hndl import pretrained as sources
from hndl.torch import build, network, parameter_counts
from hndl.types import ResolvedNode

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


class Tripwire(nn.Module):
    """An identity that records whether the tail of the network ran."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x


class TinyTrunk(nn.Module):
    """Three stages whose in-place ReLUs overwrite their convolution outputs."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(4, 6, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(6, 8, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.tail = Tripwire()
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        features = self.tail(self.layer3(self.layer2(self.layer1(x))))
        return self.head(features.mean(dim=(2, 3)))


class TwiceCalled(nn.Module):
    """A model that runs one submodule twice in a single forward pass."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(3, 3, 3, padding=1)
        self.late = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return self.late(self.shared(self.shared(x)))


TRUNK_LAYERS = ("layer1.0", "layer2.0", "layer3.0")
TRUNK_OUTPUTS = {"f1": ("B", 4, 8, 8), "f2": ("B", 6, 4, 4), "f3": ("B", 8, 2, 2)}
TRUNK_INPUT = ("B", 3, 8, 8)


def trunk_call(path, digest, layers=TRUNK_LAYERS, provider="trunk", **arguments):
    rendered = "".join(f', {name}="{value}"' for name, value in arguments.items())
    entries = "".join(f'"{name}", ' for name in layers)
    return (f'pretrained("{path}", provider="{provider}", sha256="{digest}"{rendered}, '
            f"layers=({entries}))")


def trunk_source(path, digest, **arguments):
    return f"f1, f2, f3 = {trunk_call(path, digest, **arguments)}"


def reference_captures(model, pixels, names=TRUNK_LAYERS):
    """What a hand-written hook sees for each name, cloned before the ReLUs run."""
    values = {}

    def record(name):
        return lambda module, inputs, output: values.__setitem__(name, output.clone())

    handles = [model.get_submodule(name).register_forward_hook(record(name)) for name in names]
    with torch.no_grad():
        model(pixels)
    for handle in handles:
        handle.remove()
    return tuple(values[name] for name in names)


@pytest.fixture
def trunk_checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "trunk.pth"
    torch.save(TinyTrunk().state_dict(), path)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return path


@pytest.fixture
def trunk_registry():
    registry = Registry.builtins()
    registry.pretrained_provider("trunk", TinyTrunk, readouts={"logits": lambda model, x: model(x)})
    return registry


@pytest.fixture
def trunk_reference(trunk_checkpoint):
    model = TinyTrunk()
    model.load_state_dict(torch.load(trunk_checkpoint, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


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


def test_layers_capture_every_requested_stage_in_one_pass(trunk_checkpoint, trunk_registry, trunk_reference):
    digest = digest_of(trunk_checkpoint)
    plan = resolve(trunk_source(trunk_checkpoint, digest), input_shape=TRUNK_INPUT,
                   output_shape=TRUNK_OUTPUTS, registry=trunk_registry)
    node = plan.nodes[0]
    assert node.outputs == ("out0", "out1", "out2")
    assert node.args["layers"] == TRUNK_LAYERS and node.args["layer"] == "" and node.args["readout"] == ""
    # Each output keeps its own native shape: no pooling, no concatenation.
    assert dict(node.output_shapes) == {"out0": ("B", 4, 8, 8), "out1": ("B", 6, 4, 4), "out2": ("B", 8, 2, 2)}

    model = build(plan, device=DEVICE, registry=trunk_registry)
    assert "layers=('layer1.0', 'layer2.0', 'layer3.0')" in repr(model["n0"])
    assert not any(parameter.requires_grad for parameter in model.parameters())
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE)
    with torch.no_grad():
        outputs = model(x=pixels)
    for port, expected in zip(("f1", "f2", "f3"), reference_captures(trunk_reference, pixels)):
        torch.testing.assert_close(outputs[port], expected)
    # The pass stops after the last requested layer, so the tail never runs.
    assert model["n0"].model.tail.calls == 0


def test_layer_outputs_carry_input_gradients_and_second_derivatives(trunk_checkpoint, trunk_registry):
    model = network(trunk_source(trunk_checkpoint, digest_of(trunk_checkpoint)), input_shape=TRUNK_INPUT,
                    output_shape=TRUNK_OUTPUTS, registry=trunk_registry, device=DEVICE)
    pixels = torch.randn(2, 3, 8, 8, device=DEVICE, requires_grad=True)
    outputs = model(pixels)
    # Captured before the in-place ReLU that follows each convolution, so the
    # negative activations the hook saw are still there.
    assert all((value < 0).any() for value in outputs.values())
    for value in outputs.values():
        gradient, = torch.autograd.grad(value.square().mean(), pixels, create_graph=True)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        second, = torch.autograd.grad(gradient.square().sum(), pixels, retain_graph=True)
        assert torch.isfinite(second).all() and second.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_a_batch_joined_input_reports_batch_joined_outputs(trunk_checkpoint, trunk_registry):
    """One frozen trunk over two branches: the batch entry passes straight
    through the meta-device trace, with ``layers=`` and without it."""
    digest = digest_of(trunk_checkpoint)
    paired = {"input_shape": {"x": TRUNK_INPUT, "y": TRUNK_INPUT},
              "output_shape": ("B", 8, 8, 8), "registry": trunk_registry}
    entries = ", ".join(f'"{name}"' for name in TRUNK_LAYERS)
    stack = (f'pretrained(pair, "{trunk_checkpoint}", provider="trunk", sha256="{digest}", '
             f"layers=({entries}))")
    source = (f"pair = concat(x, y, axis=0)\n"
              f"f1, f2, f3 = {stack}\n"
              "a, b = chunk(f1, 2, dim=0)\n"
              "out = concat(a, b, axis=1)")
    plan = resolve(source, **paired)
    node = plan.nodes[1]
    assert dict(node.input_shapes) == {"x": ("2*B", 3, 8, 8)}
    assert dict(node.output_shapes) == {"out0": ("2*B", 4, 8, 8), "out1": ("2*B", 6, 4, 4),
                                        "out2": ("2*B", 8, 2, 2)}
    assert plan.nodes[2].output_shapes == {"out0": ("B", 4, 8, 8), "out1": ("B", 4, 8, 8)}

    single = resolve(f"pair = concat(x, y, axis=0)\n"
                     f'only = pretrained(pair, "{trunk_checkpoint}", provider="trunk", sha256="{digest}", '
                     'layer="layer1.0")\n'
                     "a, b = chunk(only, 2, dim=0)\n"
                     "out = concat(a, b, axis=1)", **paired)
    assert single.nodes[1].output_shapes == {"out": ("2*B", 4, 8, 8)}

    model = build(plan, device=DEVICE, registry=trunk_registry)
    pixels = [torch.randn(3, 3, 8, 8, device=DEVICE) for _ in range(2)]
    with torch.no_grad():
        joined = model(x=pixels[0], y=pixels[1])["output"]
    assert tuple(joined.shape) == (3, 8, 8, 8)
    separate = [model["n1"](value)[0] for value in pixels]
    with torch.no_grad():
        torch.testing.assert_close(joined, torch.cat(separate, dim=1))


def test_layers_unpack_in_both_frontends_and_the_count_must_match(trunk_checkpoint, trunk_registry):
    digest = digest_of(trunk_checkpoint)
    shapes = {"input_shape": TRUNK_INPUT, "output_shape": TRUNK_OUTPUTS, "registry": trunk_registry}
    plan = resolve(trunk_source(trunk_checkpoint, digest), **shapes)

    def author(x):
        f1, f2, f3 = ops.pretrained(str(trunk_checkpoint), provider="trunk", sha256=digest, layers=TRUNK_LAYERS)
        return {"f1": f1, "f2": f2, "f3": f3}

    native = resolve_callable(author, **shapes)
    assert native.semantic_digest == plan.semantic_digest
    assert native.nodes[0].outputs == ("out0", "out1", "out2")

    with pytest.raises(HNDLError, match="E_OUTPUT_ARITY"):
        resolve(f"f1, f2 = {trunk_call(trunk_checkpoint, digest)}", **shapes)

    def mismatched(x):
        f1, f2 = ops.pretrained(str(trunk_checkpoint), provider="trunk", sha256=digest, layers=TRUNK_LAYERS)
        return {"f1": f1, "f2": f2, "f3": f2}

    with pytest.raises(ValueError):
        resolve_callable(mismatched, **shapes)


def test_one_entry_layers_returns_a_tuple_and_clears_current(trunk_checkpoint, trunk_registry, trunk_reference):
    digest = digest_of(trunk_checkpoint)
    call = trunk_call(trunk_checkpoint, digest, layers=("layer1.0",))
    shapes = {"input_shape": TRUNK_INPUT, "output_shape": ("B", 2, 8, 8), "registry": trunk_registry}
    plan = resolve(f"only, = {call}\nconv(only, 2, kernel_size=1)", **shapes)
    assert plan.nodes[0].outputs == ("out0",)
    assert plan.nodes[1].inputs["x"] == "node:n0/out0"
    with pytest.raises(HNDLError, match="E_CURRENT"):
        resolve(f"{call}\nconv(2, kernel_size=1)", **shapes)


def test_layers_conflicts_duplicates_and_unknown_names_fail(trunk_checkpoint, trunk_registry):
    digest = digest_of(trunk_checkpoint)

    def failing(source, output_shape=TRUNK_OUTPUTS):
        return resolve(source, input_shape=TRUNK_INPUT, output_shape=output_shape, registry=trunk_registry)

    with pytest.raises(HNDLError, match="E_PRETRAINED.*both say what the node returns"):
        failing(trunk_source(trunk_checkpoint, digest, layer="layer1.0"))
    with pytest.raises(HNDLError, match="E_PRETRAINED.*both say what the node returns"):
        failing(trunk_source(trunk_checkpoint, digest, readout="logits"))
    with pytest.raises(HNDLError, match="E_PRETRAINED.*all say what the node returns"):
        failing(trunk_source(trunk_checkpoint, digest, layer="layer1.0", readout="logits"))
    with pytest.raises(HNDLError, match='E_PRETRAINED.*layers=\\("<path>", "<path>"\\)'):
        failing(trunk_source(trunk_checkpoint, digest, output="pooled"))

    duplicated = trunk_call(trunk_checkpoint, digest, layers=("layer1.0", "layer1.0"))
    with pytest.raises(HNDLError, match="E_PRETRAINED.*'layer1.0' more than once"):
        failing(f"a, b = {duplicated}\nout = a", ("B", 4, 8, 8))
    unknown = trunk_call(trunk_checkpoint, digest, layers=("layer1.0", "layer9"))
    with pytest.raises(HNDLError, match="E_PRETRAINED.*no submodule 'layer9'.*layer1"):
        failing(f"a, b = {unknown}\nout = a", ("B", 4, 8, 8))
    empty = trunk_call(trunk_checkpoint, digest, layers=("layer1.0", ""))
    with pytest.raises(HNDLError, match="E_PRETRAINED.*empty entry names nothing"):
        failing(f"a, b = {empty}\nout = a", ("B", 4, 8, 8))

    # transformers and timm checkpoints have no submodule selection; no library is needed to say so.
    timm_source = sources.Source("directory", "/checkpoints/resnet18", "r0", {"architecture": "resnet18"},
                                 "/checkpoints/resnet18/config.json", "timm")
    with pytest.raises(HNDLError, match="E_PRETRAINED.*layers=.*select output="):
        sources.provider_for(timm_source, "features", "", "", "", ("layer1",))


def test_a_submodule_that_runs_twice_in_one_pass_fails_clearly(tmp_path, trunk_registry):
    torch.manual_seed(0)
    path = tmp_path / "twice.pth"
    torch.save(TwiceCalled().state_dict(), path)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    trunk_registry.pretrained_provider("twice", TwiceCalled)
    call = trunk_call(path, digest_of(path), layers=("shared", "late"), provider="twice")
    with pytest.raises(HNDLError, match="E_PRETRAINED.*'shared' runs more than once"):
        resolve(f"a, b = {call}\nout = b", input_shape=TRUNK_INPUT, output_shape=("B", 4, 8, 8),
                registry=trunk_registry)


def test_a_layers_plan_round_trips_and_a_config_without_one_is_unchanged(trunk_checkpoint, trunk_registry):
    digest = digest_of(trunk_checkpoint)
    plan = resolve(trunk_source(trunk_checkpoint, digest), input_shape=TRUNK_INPUT,
                   output_shape=TRUNK_OUTPUTS, registry=trunk_registry)
    encoded = plan.to_json()
    assert '"layers":["layer1.0","layer2.0","layer3.0"]' in encoded
    assert '"outputs":["out0","out1","out2"]' in encoded
    restored = ResolvedPlan.from_json(encoded, registry=trunk_registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.nodes[0].args["layers"] == TRUNK_LAYERS
    assert dict(restored.nodes[0].output_shapes) == dict(plan.nodes[0].output_shapes)
    build(restored, device=DEVICE, registry=trunk_registry)

    # A config written before layers= existed keeps its single out port, and
    # writing the new argument out explicitly reads the same.
    shapes = {"input_shape": TRUNK_INPUT, "output_shape": ("B", 4, 8, 8), "registry": trunk_registry}
    single = f'pretrained("{trunk_checkpoint}", provider="trunk", sha256="{digest}", layer="layer1.0")'
    plain = resolve(single, **shapes)
    explicit = resolve(single[:-1] + ", layers=())", **shapes)
    assert plain.nodes[0].outputs == ("out",) and plain.nodes[0].args["layers"] == ()
    assert explicit.semantic_digest == plain.semantic_digest
    assert plain.semantic_digest != plan.semantic_digest


def test_a_plan_saved_before_layers_existed_loads_completed(trunk_checkpoint, trunk_registry):
    """layers= was added to `pretrained` without since=, so a node saved without
    it no longer matches its own digest. Its default is the released behavior,
    so restoring fills it in, warns, and loads rather than refusing."""
    digest = digest_of(trunk_checkpoint)
    plan = resolve(f'pretrained("{trunk_checkpoint}", provider="trunk", sha256="{digest}", layer="layer1.0")',
                   input_shape=TRUNK_INPUT, output_shape=("B", 4, 8, 8), registry=trunk_registry)
    data = json.loads(plan.to_json())
    for node in data["nodes"]:
        del node["args"]["layers"]
        del node["provenance"]["layers"]
        del node["source"]["argument_origins"]["layers"]
    older = ResolvedPlan(nodes=tuple(ResolvedNode(**node) for node in data["nodes"]),
                         input_shape=data["input_shape"], output_shape=data["output_shape"],
                         output_ref=data["output_ref"], dtype=data["dtype"], frontend=data["frontend"],
                         input_dtype=data["input_dtype"]).to_json()
    with pytest.warns(UserWarning, match=r"filled layers=\(\) \(operator default\)"):
        restored = ResolvedPlan.from_json(older, registry=trunk_registry)
    assert restored.to_json() == plan.to_json()
    build(restored, device=DEVICE, registry=trunk_registry)
