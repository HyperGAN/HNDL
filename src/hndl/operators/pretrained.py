import torch
from torch import nn

from ..operator import Arg, Example, INDEX_DTYPES, operator
from .. import pretrained as sources


def _relation(s):
    args = s.args
    source = sources.resolve_source(args["source"], args["config"], args["provider"], args["sha256"])
    s.arg("revision", source.revision)
    provider = sources.provider_for(source, args["output"], args["component"], args["layer"], args["readout"])
    contract = provider.contract()
    if contract.kind == "tensor":
        # A provider checkpoint declares no contract of its own; only the dtype is fixed.
        dtype = s.dtype("x")
        if dtype is not None and dtype in INDEX_DTYPES:
            s.error("E_DTYPE", f"{source.location} consumes floating-point tensors (got {dtype})")
    elif contract.kind == "ids":
        s.rank("x", 2)
        dtype = s.dtype("x")
        if dtype is not None and dtype not in INDEX_DTYPES:
            s.error("E_DTYPE", f"{source.location} consumes token ids; declare input_dtype=\"int64\" (got {dtype})")
        out = s.shape("out")
        if out is not None and len(out) == 3 and args["output"] in ("features", "logits"):
            s.axis("x", 1, out[1])
    else:
        s.rank("x", 4)
        dtype = s.dtype("x")
        if dtype is not None and dtype in INDEX_DTYPES:
            s.error("E_DTYPE", f"{source.location} consumes images; the graph input must be floating point (got {dtype})")
        s.axis("x", 1, contract.channels)
        if contract.image_size is not None:
            s.axis("x", 2, contract.image_size[0])
            s.axis("x", 3, contract.image_size[1])
    x = s.shape("x")
    if x is None or any(value is None for value in x[1:]):
        return
    shape = sources.output_shape(args["source"], args["config"], args["output"], args["component"], tuple(x),
                                 args["provider"], args["sha256"], args["layer"], args["readout"])
    s.rank("out", len(shape))
    for axis, value in enumerate(shape):
        if axis:
            s.axis("out", axis, value)


@operator(
    "pretrained",
    summary="Load a pretrained network from disk or the Hugging Face Hub as one frozen node.",
    shape="x:any -> out",
    relation=_relation,
    shape_text="input contract and output shape come from the checkpoint's configuration (meta-device trace)",
    args={
        "source": Arg(str, help="Checkpoint location: a directory with config.json and safetensors weights, "
                                "a .safetensors file (with config=), hf://owner/repo[@revision], or a local "
                                ".pth state dict (with provider= and sha256=)."),
        "output": Arg(str, "features", positional=False,
                      help='Which tensor to return: "features" (last hidden state), "pooled", "logits", '
                           '"embeds" (projected CLIP-style embeddings), or a raw output attribute name.'),
        "component": Arg(str, "", positional=False,
                         help='Tower of a multi-modal checkpoint such as CLIP: "vision" or "text".'),
        "config": Arg(str, "", positional=False, help="Path to config.json when source is a bare .safetensors file."),
        "provider": Arg(str, "", positional=False,
                        help="Name of an architecture builder the host registered with "
                             "registry.pretrained_provider(name, build); required for a local .pth state dict."),
        "sha256": Arg(str, "", positional=False,
                      help="The 64 hex character digest of a local .pth file, verified before it is loaded."),
        "layer": Arg(str, "", positional=False,
                     help='Dotted named_modules() path of the provider submodule whose output the node returns, '
                          'such as "features.16"; empty returns the model\'s own output.'),
        "readout": Arg(str, "", positional=False,
                       help='Name of a readout the host registered with the provider, such as "patch_tokens"; the '
                            'node returns that callable\'s tensor instead of the model\'s own output. Mutually '
                            'exclusive with layer=.'),
        "revision": Arg(str, inferable=True, positional=False,
                        help="Resolved commit hash or content digest. Filled in at resolution and checked on restore."),
    },
    examples=[
        Example('pretrained("hf://openai-community/gpt2", output="features", name="gpt2")\nlinear(2)',
                ("B", 16), ("B", 16, 2), "GPT-2 hidden states feeding a new trainable head.", input_dtype="int64",
                network=True),
        Example('pretrained("hf://timm/resnet18.a1_in1k", output="logits")', ("B", 3, 224, 224), ("B", 1000),
                "ImageNet logits from a timm ResNet-18.", network=True),
        Example('pretrained("hf://openai/clip-vit-base-patch32", component="vision", output="embeds")',
                ("B", 3, 224, 224), ("B", 512), "The projected image embedding of CLIP's vision tower.", network=True),
    ],
    category="pretrained",
)
class Pretrained(nn.Module):
    """Wraps a checkpoint whose architecture is defined by its own configuration.

    The source's ``config.json`` selects the implementation: a ``model_type``
    is built by ``transformers`` (``AutoModel``, the ``*ForCausalLM`` /
    ``*ForImageClassification`` heads for ``output="logits"``, or the
    ``<Model>VisionModel`` / ``<Model>TextModel`` tower named by
    ``component``), and a timm ``architecture`` is built by ``timm``. HNDL
    does not describe these layouts; it traces the model on PyTorch's meta
    device to learn the output shape, then loads the weights at build time.

    Text models consume token ids: declare the graph input as ``[B, T]`` with
    ``input_dtype="int64"``. Vision models consume ``[B, C, H, W]`` at the
    checkpoint's native resolution, which the relation fixes. The wrapped
    model is frozen and kept in eval mode by default; pass ``trainable=True``
    to fine-tune it, which also lets ``model.train()`` reach it. Weights are
    loaded in float32 and cast to the plan dtype. Requires the optional
    ``hndl[pretrained]`` extra.

    The plan records the resolved ``revision`` so a restored plan fails
    (``E_CONSTRAINT``) if the source now points at a different checkpoint.

    A local ``.pth`` state dict describes no architecture, so it names one the
    host registered as trusted Python:
    ``registry.pretrained_provider("vgg16", build)`` binds a zero-argument
    callable returning the ``nn.Module``, and configuration may only name an
    already registered provider. Such a source is written
    ``pretrained("/path/weights.pth", provider="vgg16", sha256="<64 hex>",
    layer="features.16")``. The ``sha256`` is required, is verified against the
    file at resolution and again before loading, and the weights are read with
    ``torch.load(..., weights_only=True)``, which unpickles no objects; keys
    must match exactly (``strict=True``). ``layer`` names a submodule by its
    dotted ``named_modules()`` path and returns that submodule's output through
    a forward hook, stopping the pass there; omitted, the node returns the
    model's own output. Provider checkpoints declare no input contract, so the
    graph input shape is whatever the module accepts (floating point); the
    meta-device trace checks it.

    ``readout=`` names host code instead of a submodule, for checkpoints whose
    useful tensor comes from a method rather than ``forward``. The host binds
    named ``callable(model, x)`` readouts to the provider, and configuration may
    only name one of them::

        registry.pretrained_provider("dinov2_vits14", build_dinov2, readouts={
            "patch_tokens": lambda m, x: m.forward_features(x)["x_norm_patchtokens"],
            "layers_2_5_8_11": lambda m, x: torch.cat(
                m.get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True), dim=1),
        })

    which a network then selects with
    ``pretrained("/path/dinov2_vits14.pth", provider="dinov2_vits14",
    sha256="<64 hex>", readout="patch_tokens")``. Readouts can also be added to
    an existing provider with
    ``registry.pretrained_readout("dinov2_vits14", "cls_token", fn)``. A readout
    must return exactly one tensor --- concatenate or stack several inside the
    readout, or register one readout per tensor --- and it runs during
    resolution on PyTorch's meta device, so it must be a pure function of
    ``(model, x)`` that touches no real data. ``readout=`` and ``layer=`` are
    mutually exclusive, an unknown readout name fails with ``E_PRETRAINED``
    listing the ones registered for that provider, and neither applies to
    transformers or timm checkpoints, which select ``output=``.
    """

    def __init__(self, source, output, component, config, revision, provider, sha256, layer, readout, *,
                 input_shapes):
        super().__init__()
        materialize = torch.empty(0).device.type != "meta"
        if provider and not materialize:
            self.source = sources.unverified_state_dict_source(source, provider, sha256, revision)
        else:
            self.source = sources.resolve_source(source, config, provider, sha256)
        self.output = output
        self.component = component
        self.layer = layer
        self.readout = readout
        self.provider = sources.provider_for(self.source, output, component, layer, readout)
        self.kind = self.provider.contract().kind
        self.model = self.provider.instantiate(weights=materialize)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, x):
        return self.provider.select(self.provider.call(self.model, x, self.kind))

    def extra_repr(self):
        component = f", component={self.component!r}" if self.component else ""
        if self.source.kind == "state_dict":
            selected = f", layer={self.layer!r}" if self.layer else ""
            selected += f", readout={self.readout!r}" if self.readout else ""
            return f"source={self.source.spec!r}, provider={self.source.config['provider']!r}{selected}"
        return f"source={self.source.spec!r}, output={self.output!r}{component}"
