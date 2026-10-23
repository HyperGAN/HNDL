"""Multi-Head Latent Attention (MLA) with compressed KV cache and decoupled RoPE."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _rotate_half(tensor):
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def _relation(s):
    s.rank("x", 3)
    s.rank("out", 3)
    heads = s.args["heads"]
    qk_nope_head_dim = s.args["qk_nope_head_dim"]
    qk_rope_head_dim = s.args["qk_rope_head_dim"]

    a, b = s.shape("x"), s.shape("out")
    if a is not None:
        if a[1] is not None:
            s.axis("out", 1, a[1])
        if a[2] is not None:
            s.axis("out", 2, a[2])
    if b is not None:
        if b[1] is not None:
            s.axis("x", 1, b[1])
        if b[2] is not None:
            s.axis("x", 2, b[2])

    shape = s.shape("x") or s.shape("out")
    if shape is not None and shape[2] is not None:
        d = shape[2]
        if qk_nope_head_dim == 0 and d % heads != 0:
            s.error(
                "E_CONSTRAINT",
                f"latent_attention heads={heads} must divide width D={d} when qk_nope_head_dim=0",
            )
    if qk_rope_head_dim > 0 and qk_rope_head_dim % 2 != 0:
        s.error("E_CONSTRAINT", f"latent_attention qk_rope_head_dim must be even (got {qk_rope_head_dim})")


def _reference(module):
    heads = module.heads
    d_nope = module.qk_nope_head_dim
    d_rope = module.qk_rope_head_dim
    d_v = module.v_head_dim
    scale = module.scale
    causal = module.causal

    def run(x):
        b, t, _ = x.shape
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        c_kv = F.linear(x, module.kv_down.weight, module.kv_down.bias)
        c_kv = F.rms_norm(c_kv, (module.kv_lora_rank,), module.kv_norm.weight, module.kv_norm.eps)
        kc = F.linear(c_kv, module.k_up.weight, module.k_up.bias).view(b, t, heads, d_nope).transpose(1, 2)
        v = F.linear(c_kv, module.v_up.weight, module.v_up.bias).view(b, t, heads, d_v).transpose(1, 2)

        if d_rope > 0:
            kr_raw = F.linear(x, module.k_rope_proj.weight, module.k_rope_proj.bias)
            exponent = torch.arange(0, d_rope, 2, device=x.device, dtype=calc_dtype) / d_rope
            inv_freq = module.rope_theta ** (-exponent)
            angles = torch.arange(t, device=x.device, dtype=calc_dtype)[:, None] * inv_freq[None, :]
            dup = torch.cat((angles, angles), dim=-1)
            cos = dup.cos().to(x.dtype)
            sin = dup.sin().to(x.dtype)
            kr = kr_raw * cos.unsqueeze(0) + _rotate_half(kr_raw) * sin.unsqueeze(0)
            kr = kr.unsqueeze(1).expand(b, heads, t, d_rope)
            k = torch.cat([kc, kr], dim=-1)
        else:
            k = kc

        if module.q_lora_rank > 0:
            cq = F.linear(x, module.q_down.weight, module.q_down.bias)
            cq = F.rms_norm(cq, (module.q_lora_rank,), module.q_norm.weight, module.q_norm.eps)
            q_all = F.linear(cq, module.q_up.weight, module.q_up.bias).view(b, t, heads, -1).transpose(1, 2)
        else:
            q_all = F.linear(x, module.q_up.weight, module.q_up.bias).view(b, t, heads, -1).transpose(1, 2)

        if d_rope > 0:
            qc = q_all[..., :d_nope]
            qr_raw = q_all[..., d_nope:]
            qr = qr_raw * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(qr_raw) * sin.unsqueeze(0).unsqueeze(0)
            q = torch.cat([qc, qr], dim=-1)
        else:
            q = q_all

        scores = (q @ k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = weights @ v

        out = F.linear(
            attended.transpose(1, 2).contiguous().view(b, t, heads * d_v),
            module.o_proj.weight,
            module.o_proj.bias,
        )
        return out

    return run


@operator(
    "latent_attention",
    summary="Multi-Head Latent Attention (MLA) with compressed KV cache and decoupled RoPE.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    args={
        "heads": Arg(int, min=1, help="Number of query attention heads."),
        "kv_lora_rank": Arg(
            int, min=1, help="Compressed KV latent dimension d_c (e.g. 512 or D // 4)."
        ),
        "q_lora_rank": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="Low-rank Q compression rank; 0 disables Q compression.",
        ),
        "qk_nope_head_dim": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="NoPE query/key head dimension; 0 defaults to D // heads.",
        ),
        "qk_rope_head_dim": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="Decoupled RoPE head dimension; 0 enables NoPE mode (Kimi Linear), >0 must be even.",
        ),
        "v_head_dim": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="Value head dimension; 0 defaults to qk_nope_head_dim.",
        ),
        "causal": Arg(bool, True, positional=False, help="Apply causal autoregressive attention mask."),
        "rope_theta": Arg(
            float,
            10000.0,
            min=1.0,
            exclusive_min=True,
            positional=False,
            help="RoPE base frequency theta.",
        ),
        "bias": Arg(bool, False, positional=False, help="Add learned bias to all projections."),
        "softmax_scale": Arg(
            float,
            0.0,
            min=0.0,
            positional=False,
            help="Softmax scale factor; 0 defaults to 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim).",
        ),
    },
    examples=[
        Example(
            "latent_attention(4, 32, qk_nope_head_dim=16, qk_rope_head_dim=16, causal=True)",
            ("B", 16, 64),
            ("B", 16, 64),
            "MLA with 4 heads, KV LoRA rank 32, and decoupled RoPE.",
        ),
        Example(
            "latent_attention(4, 32, qk_rope_head_dim=0, causal=True)",
            ("B", 16, 64),
            ("B", 16, 64),
            "NoPE MLA without positional embeddings for hybrid linear-attention models.",
        ),
        Example(
            "latent_attention(4, 32, q_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16)",
            ("B", 16, 64),
            ("B", 16, 64),
            "Full low-rank Q and KV compression MLA.",
        ),
    ],
    reference=_reference,
    category="sequence",
)
class LatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) with low-rank KV compression and decoupled RoPE.

    Introduced in DeepSeek-V2/V3 and Kimi Linear, MLA down-projects keys and
    values into a shared latent vector ``c_kv = RMSNorm(W_DKV x)`` of rank
    ``kv_lora_rank``, which is up-projected into non-positional content keys
    and values. A separate decoupled RoPE key vector ``k_rope`` is shared
    across all query heads when ``qk_rope_head_dim > 0``, while setting
    ``qk_rope_head_dim = 0`` activates position-free NoPE mode.
    """

    def __init__(
        self,
        heads,
        kv_lora_rank,
        q_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
        causal=True,
        rope_theta=10000.0,
        bias=False,
        softmax_scale=0.0,
        *,
        D,
    ):
        super().__init__()
        self.heads = int(heads)
        self.kv_lora_rank = int(kv_lora_rank)
        self.q_lora_rank = int(q_lora_rank)
        self.D = int(D)
        self.qk_nope_head_dim = int(qk_nope_head_dim) if qk_nope_head_dim > 0 else self.D // self.heads
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim) if v_head_dim > 0 else self.qk_nope_head_dim
        self.causal = bool(causal)
        self.rope_theta = float(rope_theta)
        self.bias = bool(bias)
        self.softmax_scale = float(softmax_scale)

        total_qk_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = (
            self.softmax_scale if self.softmax_scale > 0.0 else 1.0 / math.sqrt(total_qk_dim)
        )

        self.kv_down = nn.Linear(self.D, self.kv_lora_rank, bias=self.bias)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.k_up = nn.Linear(self.kv_lora_rank, self.heads * self.qk_nope_head_dim, bias=self.bias)
        self.v_up = nn.Linear(self.kv_lora_rank, self.heads * self.v_head_dim, bias=self.bias)

        if self.qk_rope_head_dim > 0:
            self.k_rope_proj = nn.Linear(self.D, self.qk_rope_head_dim, bias=self.bias)
            self._cache = {}
        else:
            self.k_rope_proj = None

        if self.q_lora_rank > 0:
            self.q_down = nn.Linear(self.D, self.q_lora_rank, bias=self.bias)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_up = nn.Linear(self.q_lora_rank, self.heads * total_qk_dim, bias=self.bias)
        else:
            self.q_down = None
            self.q_norm = None
            self.q_up = nn.Linear(self.D, self.heads * total_qk_dim, bias=self.bias)

        self.o_proj = nn.Linear(self.heads * self.v_head_dim, self.D, bias=self.bias)

    @property
    def _rope_cache(self):
        return getattr(self, "_cache", {})

    def _apply(self, fn, recurse=True):
        res = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_cache"):
            self._cache.clear()
        return res


    def _tables(self, positions, dim, device, dtype):
        if torch.compiler.is_compiling():
            calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            exponent = torch.arange(0, dim, 2, device=device, dtype=calc_dtype) / dim
            inverse_frequency = self.rope_theta ** (-exponent)
            angles = torch.arange(positions, device=device, dtype=calc_dtype)[:, None] * inverse_frequency[None, :]
            duplicated = torch.cat((angles, angles), dim=-1)
            cos = duplicated.cos().to(dtype)
            sin = duplicated.sin().to(dtype)
            return cos, sin
        key = (dim, str(device), dtype)
        hit = self._cache.get(key)
        if hit is not None and hit[0].shape[0] >= positions:
            return hit[0][:positions], hit[1][:positions]
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        exponent = torch.arange(0, dim, 2, device=device, dtype=calc_dtype) / dim
        inverse_frequency = self.rope_theta ** (-exponent)
        angles = torch.arange(positions, device=device, dtype=calc_dtype)[:, None] * inverse_frequency[None, :]
        duplicated = torch.cat((angles, angles), dim=-1)
        cos = duplicated.cos().to(dtype)
        sin = duplicated.sin().to(dtype)
        self._cache[key] = (cos, sin)
        return cos, sin

    def absorb_weights(self):
        """Absorb k_up into q_proj and v_up into o_proj for low-memory inference.

        Eliminates uncompressed keys and values from memory by evaluating attention
        directly over compressed latent tokens c_kv.
        """
        W_uk = self.k_up.weight.view(self.heads, self.qk_nope_head_dim, self.kv_lora_rank)
        W_uv = self.v_up.weight.view(self.heads, self.v_head_dim, self.kv_lora_rank)
        W_o = self.o_proj.weight.view(self.D, self.heads, self.v_head_dim)
        W_o_abs = torch.einsum("mhk,hkc->mhc", W_o, W_uv)

        self.register_buffer("_absorbed_W_uk", W_uk.contiguous(), persistent=False)
        self.register_buffer("_absorbed_W_o", W_o_abs.contiguous(), persistent=False)
        self._weight_absorption = True
        return self

    absorb_for_inference = absorb_weights

    @property
    def is_weight_absorbed(self):
        return getattr(self, "_weight_absorption", False)

    def forward(self, x):
        b, t, _ = x.shape
        c_kv = self.kv_norm(self.kv_down(x))

        if getattr(self, "_weight_absorption", False):
            if self.q_lora_rank > 0:
                cq = self.q_norm(self.q_down(x))
                q_all = self.q_up(cq).view(b, t, self.heads, -1).transpose(1, 2)
            else:
                q_all = self.q_up(x).view(b, t, self.heads, -1).transpose(1, 2)

            qc = q_all[..., : self.qk_nope_head_dim]
            q_abs = torch.einsum("bhtd,hdc->bhtc", qc, self._absorbed_W_uk)
            scores_c = torch.einsum("bhtc,bsc->bhts", q_abs, c_kv)

            if self.qk_rope_head_dim > 0:
                qr_raw = q_all[..., self.qk_nope_head_dim :]
                cos, sin = self._tables(t, self.qk_rope_head_dim, x.device, x.dtype)
                cos_u = cos.unsqueeze(0).unsqueeze(0)
                sin_u = sin.unsqueeze(0).unsqueeze(0)
                qr = qr_raw * cos_u + _rotate_half(qr_raw) * sin_u
                kr_raw = self.k_rope_proj(x)
                kr = kr_raw * cos.unsqueeze(0) + _rotate_half(kr_raw) * sin.unsqueeze(0)
                scores_r = torch.einsum("bhtd,bsd->bhts", qr, kr)
                scores = (scores_c + scores_r) * self.scale
            else:
                scores = scores_c * self.scale

            if self.causal:
                causal_mask = torch.full((t, t), float("-inf"), device=x.device, dtype=scores.dtype).triu(1)
                scores = scores + causal_mask

            attn_weights = scores.softmax(dim=-1)
            attn_c = torch.einsum("bhts,bsc->bhtc", attn_weights, c_kv)
            out = torch.einsum("bhtc,mhc->btm", attn_c, self._absorbed_W_o)
            if self.o_proj.bias is not None:
                out = out + self.o_proj.bias
            return out

        kc = self.k_up(c_kv).view(b, t, self.heads, self.qk_nope_head_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(b, t, self.heads, self.v_head_dim).transpose(1, 2)

        if self.qk_rope_head_dim > 0:
            kr_raw = self.k_rope_proj(x)
            cos, sin = self._tables(t, self.qk_rope_head_dim, x.device, x.dtype)
            kr = kr_raw * cos.unsqueeze(0) + _rotate_half(kr_raw) * sin.unsqueeze(0)
            kr = kr.unsqueeze(1).expand(b, self.heads, t, self.qk_rope_head_dim)
            k = torch.empty(b, self.heads, t, self.qk_nope_head_dim + self.qk_rope_head_dim,
                            device=x.device, dtype=x.dtype)
            k[..., : self.qk_nope_head_dim] = kc
            k[..., self.qk_nope_head_dim :] = kr
        else:
            k = kc

        if self.q_lora_rank > 0:
            cq = self.q_norm(self.q_down(x))
            q_all = self.q_up(cq).view(b, t, self.heads, -1).transpose(1, 2)
        else:
            q_all = self.q_up(x).view(b, t, self.heads, -1).transpose(1, 2)

        if self.qk_rope_head_dim > 0:
            qr_raw = q_all[..., self.qk_nope_head_dim :]
            cos_u = cos.unsqueeze(0).unsqueeze(0)
            sin_u = sin.unsqueeze(0).unsqueeze(0)
            q = torch.empty_like(q_all)
            q[..., : self.qk_nope_head_dim] = q_all[..., : self.qk_nope_head_dim]
            q[..., self.qk_nope_head_dim :] = qr_raw * cos_u + _rotate_half(qr_raw) * sin_u
        else:
            q = q_all

        out_attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=self.causal)
        out = self.o_proj(out_attn.transpose(1, 2).contiguous().view(b, t, self.heads * self.v_head_dim))
        return out

    def extra_repr(self):
        return (f"heads={self.heads}, kv_lora_rank={self.kv_lora_rank}, "
                f"qk_nope_head_dim={self.qk_nope_head_dim}, qk_rope_head_dim={self.qk_rope_head_dim}, "
                f"v_head_dim={self.v_head_dim}, causal={self.causal}")
