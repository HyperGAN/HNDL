import torch
from torch import nn

from ..operator import Arg, Example, INDEX_DTYPES, operator
from .. import pretrained as sources


def _relation(s):
    args = s.args
    source = sources.resolve_source(args["source"], args["config"])
    s.arg("revision", source.revision)
    provider = sources.provider_for(source, args["output"], args["component"])
    contract = provider.contract()
    if contract.kind == "ids":
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
    shape = sources.output_shape(args["source"], args["config"], args["output"], args["component"], tuple(x))
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
                                "a .safetensors file (with config=), or hf://owner/repo[@revision]."),
        "output": Arg(str, "features", positional=False,
                      help='Which tensor to return: "features" (last hidden state), "pooled", "logits", '
                           '"embeds" (projected CLIP-style embeddings), or a raw output attribute name.'),
        "component": Arg(str, "", positional=False,
                         help='Tower of a multi-modal checkpoint such as CLIP: "vision" or "text".'),
        "config": Arg(str, "", positional=False, help="Path to config.json when source is a bare .safetensors file."),
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
    """

    def __init__(self, source, output, component, config, revision, *, input_shapes):
        super().__init__()
        self.source = sources.resolve_source(source, config)
        self.output = output
        self.component = component
        self.provider = sources.provider_for(self.source, output, component)
        self.kind = self.provider.contract().kind
        materialize = torch.empty(0).device.type != "meta"
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
        return f"source={self.source.spec!r}, output={self.output!r}{component}"
