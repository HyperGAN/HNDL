"""Kimi Delta Attention (KDA) and Gated DeltaNet with chunkwise parallel WY evaluation."""

import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _relation(s):
    s.rank("x", 3)
    s.rank("out", 3)
    heads = s.args["heads"]
    head_dim = s.args["head_dim"]

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
        if head_dim == 0 and d % heads != 0:
            s.error(
                "E_CONSTRAINT",
                f"gated_delta_attention heads={heads} must divide width D={d} when head_dim=0",
            )


def _reference(module):
    heads = module.heads
    head_dim = module.head_dim
    inner_dim = heads * head_dim
    conv_size = module.conv_size
    decay_kind = module.decay_kind
    eps = module.eps

    def run(x):
        b, t, _ = x.shape
        q = F.linear(x, module.q_proj.weight, module.q_proj.bias)
        k = F.linear(x, module.k_proj.weight, module.k_proj.bias)
        v = F.linear(x, module.v_proj.weight, module.v_proj.bias)

        if conv_size > 0:
            pad = (conv_size - 1, 0)
            q = F.conv1d(
                F.pad(q.transpose(1, 2), pad),
                module.conv_q.weight,
                module.conv_q.bias,
                groups=inner_dim,
            )[..., :t].transpose(1, 2)
            k = F.conv1d(
                F.pad(k.transpose(1, 2), pad),
                module.conv_k.weight,
                module.conv_k.bias,
                groups=inner_dim,
            )[..., :t].transpose(1, 2)
            v = F.conv1d(
                F.pad(v.transpose(1, 2), pad),
                module.conv_v.weight,
                module.conv_v.bias,
                groups=inner_dim,
            )[..., :t].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(b, t, heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(b, t, heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(b, t, heads, head_dim).permute(0, 2, 1, 3)

        q = F.normalize(q, p=2.0, dim=-1, eps=eps)
        k = F.normalize(k, p=2.0, dim=-1, eps=eps)

        beta = torch.sigmoid(F.linear(x, module.b_proj.weight, module.b_proj.bias)).permute(0, 2, 1)

        if decay_kind == "channel":
            a_down = F.linear(x, module.alpha_down.weight, module.alpha_down.bias)
            a_up = F.linear(F.silu(a_down), module.alpha_up.weight, module.alpha_up.bias)
            a_logits = a_up.view(b, t, heads, head_dim).permute(0, 2, 1, 3)
            rate = torch.exp(module.A_log).view(1, heads, 1, head_dim)
            alpha = torch.exp(-rate * F.softplus(a_logits))
        else:
            a_logits = (
                F.linear(x, module.alpha_proj.weight, module.alpha_proj.bias)
                .permute(0, 2, 1)
                .unsqueeze(-1)
            )
            rate = torch.exp(module.A_log).view(1, heads, 1, 1)
            alpha = torch.exp(-rate * F.softplus(a_logits))

        g_down = F.linear(x, module.gate_down.weight, module.gate_down.bias)
        g_up = F.linear(F.silu(g_down), module.gate_up.weight, module.gate_up.bias)
        g = torch.sigmoid(g_up).view(b, t, heads, head_dim)

        out = []
        S = torch.zeros(b, heads, head_dim, head_dim, device=x.device, dtype=x.dtype)
        for step in range(t):
            a_t = alpha[:, :, step, :, None]
            S_decay = a_t * S
            v_pred = (S_decay.transpose(-2, -1) @ k[:, :, step, :, None]).squeeze(-1)
            err = v[:, :, step] - v_pred
            S = S_decay + beta[:, :, step, None, None] * (k[:, :, step, :, None] @ err[:, :, None, :])
            o_step = (S.transpose(-2, -1) @ q[:, :, step, :, None]).squeeze(-1)
            out.append(o_step)

        out = torch.stack(out, dim=2)
        out = out.permute(0, 2, 1, 3)
        out = F.rms_norm(out, (head_dim,), eps=eps) * module.rms_norm_weight
        out = out * g
        out = out.reshape(b, t, inner_dim)
        return F.linear(out, module.o_proj.weight, module.o_proj.bias)

    return run


@operator(
    "gated_delta_attention",
    summary="Kimi Delta Attention (KDA) and Gated DeltaNet with chunkwise parallel WY evaluation.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    args={
        "heads": Arg(
            int, min=1, help="Number of attention heads; must divide width D when head_dim=0."
        ),
        "head_dim": Arg(
            int, 0, min=0, positional=False, help="Dimension per head; 0 defaults to D // heads."
        ),
        "decay_kind": Arg(
            str,
            "channel",
            choices=("channel", "head"),
            positional=False,
            help='Decay granularity: "channel" for Kimi KDA, "head" for Gated DeltaNet.',
        ),
        "gate_rank": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="Low-rank bottleneck rank for decay and output gates; 0 defaults to max(head_dim, D // 16).",
        ),
        "conv_size": Arg(
            int,
            4,
            min=0,
            max=16,
            positional=False,
            help="Causal depthwise 1D conv kernel size on Q/K/V; 0 disables convolution.",
        ),
        "chunk_size": Arg(
            int,
            64,
            min=1,
            max=512,
            positional=False,
            help="Chunk size for parallel WY/DPLR Tensor Core evaluation in forward().",
        ),
        "bias": Arg(bool, False, positional=False, help="Add learned bias to input and output projections."),
        "eps": Arg(
            float,
            1e-6,
            min=0,
            exclusive_min=True,
            positional=False,
            help="Epsilon for L2 norm and RMSNorm.",
        ),
    },
    examples=[
        Example(
            'gated_delta_attention(4, decay_kind="channel", chunk_size=32)',
            ("B", 16, 64),
            ("B", 16, 64),
            "Kimi Delta Attention with channel-wise decay and chunk size 32.",
        ),
        Example(
            'gated_delta_attention(4, decay_kind="head", chunk_size=32)',
            ("B", 16, 64),
            ("B", 16, 64),
            "Gated DeltaNet with scalar head decay and chunk size 32.",
        ),
        Example(
            "gated_delta_attention(2, head_dim=16, conv_size=0, chunk_size=16)",
            ("B", 12, 32),
            ("B", 12, 32),
            "Gated delta attention with explicit head_dim and convolution disabled.",
        ),
    ],
    reference=_reference,
    category="sequence",
)
class GatedDeltaAttention(nn.Module):
    """Kimi Delta Attention (KDA) and Gated DeltaNet associative memory attention.

    Evaluates associative memory updates with error-correction delta rule:
    residual error ``e_t = v_t - S_{t-1}^T k_t`` is written into memory state ``S_t``.
    In ``forward()``, native PyTorch Tensor Core chunkwise parallel WY representation
    evaluates intra-chunk dependencies via ``torch.linalg.solve_triangular``, while
    inter-chunk states are propagated across chunks.
    """

    def __init__(
        self,
        heads,
        head_dim=0,
        decay_kind="channel",
        gate_rank=0,
        conv_size=4,
        chunk_size=64,
        bias=False,
        eps=1e-6,
        *,
        D,
    ):
        super().__init__()
        self.heads = int(heads)
        self.D = int(D)
        self.head_dim = int(head_dim) if head_dim > 0 else self.D // self.heads
        self.decay_kind = str(decay_kind)
        self.gate_rank = (
            int(gate_rank) if gate_rank > 0 else max(self.head_dim, max(1, self.D // 16))
        )
        self.conv_size = int(conv_size)
        self.chunk_size = int(chunk_size)
        self.bias = bool(bias)
        self.eps = float(eps)

        inner_dim = self.heads * self.head_dim
        self.q_proj = nn.Linear(self.D, inner_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.D, inner_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.D, inner_dim, bias=self.bias)

        if self.conv_size > 0:
            self.conv_q = nn.Conv1d(
                inner_dim, inner_dim, kernel_size=self.conv_size, groups=inner_dim, bias=True
            )
            self.conv_k = nn.Conv1d(
                inner_dim, inner_dim, kernel_size=self.conv_size, groups=inner_dim, bias=True
            )
            self.conv_v = nn.Conv1d(
                inner_dim, inner_dim, kernel_size=self.conv_size, groups=inner_dim, bias=True
            )
        else:
            self.conv_q = None
            self.conv_k = None
            self.conv_v = None

        self.b_proj = nn.Linear(self.D, self.heads, bias=self.bias)

        if self.decay_kind == "channel":
            self.alpha_down = nn.Linear(self.D, self.gate_rank, bias=False)
            self.alpha_up = nn.Linear(self.gate_rank, inner_dim, bias=True)
            self.A_log = nn.Parameter(torch.empty(self.heads, self.head_dim))
        else:
            self.alpha_proj = nn.Linear(self.D, self.heads, bias=self.bias)
            self.A_log = nn.Parameter(torch.empty(self.heads))
        nn.init.uniform_(self.A_log, -2.5, -0.5)

        self.gate_down = nn.Linear(self.D, self.gate_rank, bias=False)
        self.gate_up = nn.Linear(self.gate_rank, inner_dim, bias=False)
        self.rms_norm_weight = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.o_proj = nn.Linear(inner_dim, self.D, bias=self.bias)

    def _prepare_inputs(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.conv_size > 0:
            pad = (self.conv_size - 1, 0)
            q = F.silu(self.conv_q(F.pad(q.transpose(1, 2), pad))[..., :t].transpose(1, 2))
            k = F.silu(self.conv_k(F.pad(k.transpose(1, 2), pad))[..., :t].transpose(1, 2))
            v = F.silu(self.conv_v(F.pad(v.transpose(1, 2), pad))[..., :t].transpose(1, 2))

        q = q.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q = F.normalize(q, p=2.0, dim=-1, eps=self.eps)
        k = F.normalize(k, p=2.0, dim=-1, eps=self.eps)

        beta = torch.sigmoid(self.b_proj(x)).permute(0, 2, 1)

        if self.decay_kind == "channel":
            a_logits = (
                self.alpha_up(F.silu(self.alpha_down(x)))
                .view(b, t, self.heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            rate = torch.exp(self.A_log).view(1, self.heads, 1, self.head_dim)
            alpha = torch.exp(-rate * F.softplus(a_logits))
        else:
            a_logits = self.alpha_proj(x).permute(0, 2, 1).unsqueeze(-1)
            rate = torch.exp(self.A_log).view(1, self.heads, 1, 1)
            alpha = torch.exp(-rate * F.softplus(a_logits))

        g = torch.sigmoid(self.gate_up(F.silu(self.gate_down(x)))).view(
            b, t, self.heads, self.head_dim
        )
        return q, k, v, beta, alpha, g

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v, beta, alpha, g = self._prepare_inputs(x)
        out = torch.zeros_like(q)
        S = torch.zeros(b, self.heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        for start in range(0, t, self.chunk_size):
            end = min(start + self.chunk_size, t)
            c_len = end - start

            qc = q[:, :, start:end]
            kc = k[:, :, start:end]
            vc = v[:, :, start:end]
            betac = beta[:, :, start:end]
            alphac = alpha[:, :, start:end]

            log_alpha = torch.log(alphac)
            cum_log_alpha = torch.cumsum(log_alpha, dim=2)
            lambda_init = torch.exp(cum_log_alpha)

            k_decay_init = lambda_init * kc
            v_pred_inter = torch.matmul(k_decay_init, S)
            v_prime = vc - v_pred_inter

            diff = cum_log_alpha.unsqueeze(3) - cum_log_alpha.unsqueeze(2)
            lambda_mat = torch.exp(diff)

            ki = kc.unsqueeze(3)
            kj = kc.unsqueeze(2)
            bj = betac.unsqueeze(2)

            inner_k = (ki * lambda_mat * kj).sum(-1) * bj
            strict_lower = torch.tril(
                torch.ones(c_len, c_len, dtype=torch.bool, device=x.device), diagonal=-1
            )
            A = torch.where(strict_lower, inner_k, torch.zeros_like(inner_k))
            A = A + torch.eye(c_len, dtype=x.dtype, device=x.device)

            if A.dtype in (torch.float16, torch.bfloat16):
                E = torch.linalg.solve_triangular(
                    A.float(), v_prime.float(), upper=False, unitriangular=True
                ).to(x.dtype)
            else:
                E = torch.linalg.solve_triangular(A, v_prime, upper=False, unitriangular=True)

            qi = qc.unsqueeze(3)
            inner_q = (qi * lambda_mat * kj).sum(-1) * bj
            lower = torch.tril(
                torch.ones(c_len, c_len, dtype=torch.bool, device=x.device), diagonal=0
            )
            B_mat = torch.where(lower, inner_q, torch.zeros_like(inner_q))

            o_intra = B_mat @ E
            q_decay_init = lambda_init * qc
            o_inter = torch.matmul(q_decay_init, S)
            out[:, :, start:end] = o_intra + o_inter

            lambda_end = torch.exp(cum_log_alpha[:, :, -1])
            S_init_decayed = lambda_end.unsqueeze(-1) * S

            lambda_to_end = torch.exp(cum_log_alpha[:, :, -1:] - cum_log_alpha)
            W = betac.unsqueeze(-1) * (lambda_to_end * kc)
            S_update = W.transpose(-2, -1) @ E
            S = S_init_decayed + S_update

        out = out.permute(0, 2, 1, 3)
        out = F.rms_norm(out, (self.head_dim,), eps=self.eps) * self.rms_norm_weight
        out = out * g
        out = out.reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(out)
