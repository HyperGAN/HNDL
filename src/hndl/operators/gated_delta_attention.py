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

        calc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        if decay_kind == "channel":
            a_down = F.linear(x, module.alpha_down.weight, module.alpha_down.bias)
            a_up = F.linear(F.silu(a_down), module.alpha_up.weight, module.alpha_up.bias)
            a_logits = a_up.view(b, t, heads, head_dim).permute(0, 2, 1, 3)
            rate = torch.exp(module.A_log.to(calc_dtype)).view(1, heads, 1, head_dim)
            log_alpha = -rate * F.softplus(a_logits.to(calc_dtype))
        else:
            a_logits = (
                F.linear(x, module.alpha_proj.weight, module.alpha_proj.bias)
                .permute(0, 2, 1)
                .unsqueeze(-1)
            )
            rate = torch.exp(module.A_log.to(calc_dtype)).view(1, heads, 1, 1)
            log_alpha = -rate * F.softplus(a_logits.to(calc_dtype))
        alpha = torch.exp(log_alpha)

        g_down = F.linear(x, module.gate_down.weight, module.gate_down.bias)
        g_up = F.linear(F.silu(g_down), module.gate_up.weight, module.gate_up.bias)
        g = torch.sigmoid(g_up).view(b, t, heads, head_dim)

        out = []
        S = torch.zeros(b, heads, head_dim, head_dim, device=x.device, dtype=calc_dtype)
        q_c = q.to(calc_dtype)
        k_c = k.to(calc_dtype)
        v_c = v.to(calc_dtype)
        beta_c = beta.to(calc_dtype)
        alpha_c = alpha.to(calc_dtype)
        for step in range(t):
            a_t = alpha_c[:, :, step, :, None]
            S_decay = a_t * S
            v_pred = (S_decay.transpose(-2, -1) @ k_c[:, :, step, :, None]).squeeze(-1)
            err = v_c[:, :, step] - v_pred
            S = S_decay + beta_c[:, :, step, None, None] * (k_c[:, :, step, :, None] @ err[:, :, None, :])
            o_step = (S.transpose(-2, -1) @ q_c[:, :, step, :, None]).squeeze(-1)
            out.append(o_step)

        out = torch.stack(out, dim=2)
        out = out.permute(0, 2, 1, 3)
        out = F.rms_norm(out, (head_dim,), eps=eps) * module.rms_norm_weight.to(calc_dtype)
        out = (out * g.to(calc_dtype)).to(x.dtype)
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
            inner_dim = self.heads * self.head_dim
            qkv = torch.cat([q, k, v], dim=-1)
            qkv_pad = F.pad(qkv.transpose(1, 2), pad)
            conv_w = torch.cat([self.conv_q.weight, self.conv_k.weight, self.conv_v.weight], dim=0)
            conv_b = (
                torch.cat([self.conv_q.bias, self.conv_k.bias, self.conv_v.bias], dim=0)
                if self.conv_q.bias is not None
                else None
            )
            qkv_conv = F.silu(
                F.conv1d(qkv_pad, conv_w, conv_b, groups=3 * inner_dim)[..., :t].transpose(1, 2)
            )
            q, k, v = qkv_conv.chunk(3, dim=-1)

        q = q.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q = F.normalize(q, p=2.0, dim=-1, eps=self.eps)
        k = F.normalize(k, p=2.0, dim=-1, eps=self.eps)

        beta = torch.sigmoid(self.b_proj(x)).permute(0, 2, 1)

        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if self.decay_kind == "channel":
            a_logits = (
                self.alpha_up(F.silu(self.alpha_down(x)))
                .view(b, t, self.heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            rate = torch.exp(self.A_log.to(calc_dtype)).view(1, self.heads, 1, self.head_dim)
            log_alpha = -rate * F.softplus(a_logits.to(calc_dtype))
        else:
            a_logits = self.alpha_proj(x).permute(0, 2, 1).unsqueeze(-1)
            rate = torch.exp(self.A_log.to(calc_dtype)).view(1, self.heads, 1, 1)
            log_alpha = -rate * F.softplus(a_logits.to(calc_dtype))

        g = torch.sigmoid(self.gate_up(F.silu(self.gate_down(x)))).view(
            b, t, self.heads, self.head_dim
        )
        return q, k, v, beta, log_alpha, g

    def forward(self, x):
        b, t, _ = x.shape
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        q, k, v, beta, log_alpha, g = self._prepare_inputs(x)
        q_f, k_f, v_f, beta_f = q.to(calc_dtype), k.to(calc_dtype), v.to(calc_dtype), beta.to(calc_dtype)
        out = torch.zeros(b, self.heads, t, self.head_dim, device=x.device, dtype=calc_dtype)
        S = torch.zeros(b, self.heads, self.head_dim, self.head_dim, device=x.device, dtype=calc_dtype)

        C = self.chunk_size
        N_full = t // C

        if N_full > 0:
            full_len = N_full * C
            q_full = q_f[:, :, :full_len].view(b, self.heads, N_full, C, self.head_dim)
            k_full = k_f[:, :, :full_len].view(b, self.heads, N_full, C, self.head_dim)
            v_full = v_f[:, :, :full_len].view(b, self.heads, N_full, C, self.head_dim)
            beta_full = beta_f[:, :, :full_len].view(b, self.heads, N_full, C)
            log_a_full = log_alpha[:, :, :full_len].view(
                b, self.heads, N_full, C, log_alpha.shape[-1]
            )

            cum_log_a = torch.cumsum(log_a_full, dim=3)
            lambda_init = torch.exp(cum_log_a)

            if self.decay_kind == "head":
                diff = (cum_log_a - cum_log_a.transpose(-2, -1)).clamp(max=0.0)
                decay = torch.exp(diff) * beta_full.unsqueeze(-2)
                inner_k = torch.matmul(k_full, k_full.transpose(-2, -1)) * decay
                inner_q = torch.matmul(q_full, k_full.transpose(-2, -1)) * decay
                A = torch.tril(inner_k, diagonal=-1)
                B_mat = torch.tril(inner_q, diagonal=0)
            else:
                A_list = []
                B_list = []
                for c in range(N_full):
                    diff_c = (cum_log_a[:, :, c].unsqueeze(3) - cum_log_a[:, :, c].unsqueeze(2)).clamp(max=0.0)
                    decay_c = torch.exp(diff_c)
                    beta_c = beta_full[:, :, c].unsqueeze(-2)
                    ik_c = (k_full[:, :, c].unsqueeze(3) * decay_c * k_full[:, :, c].unsqueeze(2)).sum(-1) * beta_c
                    iq_c = (q_full[:, :, c].unsqueeze(3) * decay_c * k_full[:, :, c].unsqueeze(2)).sum(-1) * beta_c
                    A_list.append(torch.tril(ik_c, diagonal=-1))
                    B_list.append(torch.tril(iq_c, diagonal=0))
                A = torch.stack(A_list, dim=2)
                B_mat = torch.stack(B_list, dim=2)

            E_intra = torch.linalg.solve_triangular(A, v_full, upper=False, unitriangular=True)
            k_decay_init = lambda_init * k_full
            K_solve = torch.linalg.solve_triangular(A, k_decay_init, upper=False, unitriangular=True)

            lambda_end = torch.exp(cum_log_a[:, :, :, -1])
            lambda_to_end = torch.exp((cum_log_a[:, :, :, -1:] - cum_log_a).clamp(max=0.0))
            W = beta_full.unsqueeze(-1) * (lambda_to_end * k_full)
            Q_decay = q_full * lambda_init

            for c in range(N_full):
                E_c = E_intra[:, :, c] - torch.matmul(K_solve[:, :, c], S)
                o_intra = B_mat[:, :, c] @ E_c
                o_inter = torch.matmul(Q_decay[:, :, c], S)
                out[:, :, c * C : (c + 1) * C] = o_intra + o_inter
                S = lambda_end[:, :, c].unsqueeze(-1) * S + W[:, :, c].transpose(-2, -1) @ E_c

        rem_start = N_full * C
        if rem_start < t:
            qc = q_f[:, :, rem_start:t]
            kc = k_f[:, :, rem_start:t]
            vc = v_f[:, :, rem_start:t]
            betac = beta_f[:, :, rem_start:t]

            log_alphac = log_alpha[:, :, rem_start:t]
            cum_log_alpha = torch.cumsum(log_alphac, dim=2)
            lambda_init = torch.exp(cum_log_alpha)

            if self.decay_kind == "head":
                diff = (cum_log_alpha - cum_log_alpha.transpose(-2, -1)).clamp(max=0.0)
                decay = torch.exp(diff) * betac.unsqueeze(-2)
                inner_k = torch.matmul(kc, kc.transpose(-2, -1)) * decay
                inner_q = torch.matmul(qc, kc.transpose(-2, -1)) * decay
            else:
                diff = (cum_log_alpha.unsqueeze(3) - cum_log_alpha.unsqueeze(2)).clamp(max=0.0)
                decay = torch.exp(diff)
                inner_k = (kc.unsqueeze(3) * decay * kc.unsqueeze(2)).sum(-1) * betac.unsqueeze(-2)
                inner_q = (qc.unsqueeze(3) * decay * kc.unsqueeze(2)).sum(-1) * betac.unsqueeze(-2)

            A = torch.tril(inner_k, diagonal=-1)
            B_mat = torch.tril(inner_q, diagonal=0)

            k_decay_init = lambda_init * kc
            v_pred_inter = torch.matmul(k_decay_init, S)
            v_prime = vc - v_pred_inter

            E = torch.linalg.solve_triangular(A, v_prime, upper=False, unitriangular=True)

            o_intra = B_mat @ E
            q_decay_init = lambda_init * qc
            o_inter = torch.matmul(q_decay_init, S)
            out[:, :, rem_start:t] = o_intra + o_inter

            lambda_end = torch.exp(cum_log_alpha[:, :, -1])
            S_init_decayed = lambda_end.unsqueeze(-1) * S

            lambda_to_end = torch.exp((cum_log_alpha[:, :, -1:] - cum_log_alpha).clamp(max=0.0))
            W = betac.unsqueeze(-1) * (lambda_to_end * kc)
            S_update = W.transpose(-2, -1) @ E
            S = S_init_decayed + S_update

        out = out.permute(0, 2, 1, 3)
        out = F.rms_norm(out, (self.head_dim,), eps=self.eps) * self.rms_norm_weight.to(calc_dtype)
        out = (out * g.to(calc_dtype)).to(x.dtype)
        out = out.reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(out)

    def extra_repr(self):
        return (f"heads={self.heads}, head_dim={self.head_dim}, "
                f"decay_kind={self.decay_kind!r}, chunk_size={self.chunk_size}, "
                f"conv_size={self.conv_size}")
