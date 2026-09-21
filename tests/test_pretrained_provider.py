"""Local ``.pth`` checkpoints loaded through host-registered architecture providers."""

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
