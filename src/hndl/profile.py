"""Static, allocation-free, zero-compute profiler for HNDL ResolvedPlans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

from .types import batch_multiple

if TYPE_CHECKING:
    from .types import ResolvedPlan

DTYPE_SIZES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float64": 8,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


@dataclass(frozen=True)
class NodeProfile:
    """Zero-compute profile metrics for a single resolved graph node."""

    id: str
    op: str
    total_parameters: int
    active_parameters: int
    forward_flops: int
    activation_bytes: int
    kv_cache_bytes: int

    @property
    def flops(self) -> int:
        return self.forward_flops

    @property
    def state_bytes(self) -> int:
        return self.kv_cache_bytes


@dataclass(frozen=True)
class PlanProfile:
    """Zero-compute static profile for an entire ResolvedPlan."""

    total_parameters: int
    active_parameters: int
    forward_flops: int
    peak_activation_bytes: int
    kv_cache_bytes: int
    nodes: tuple[NodeProfile, ...] = ()
    _by_id: dict[str, NodeProfile] = field(default_factory=dict, repr=False)

    def __getitem__(self, node_id: str) -> NodeProfile:
        return self._by_id[node_id]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._by_id

    @property
    def flops(self) -> int:
        return self.forward_flops

    @property
    def state_bytes(self) -> int:
        return self.kv_cache_bytes

    @property
    def training_flops(self) -> int:
        """Analytical training FLOPs (standard 3x forward pass with backward autograd)."""
        return 3 * self.forward_flops

    def summary(self) -> str:
        """Human-readable formatted summary of static plan profile."""
        lines = [
            "============================================================",
            "                   HNDL Static Plan Profile                 ",
            "============================================================",
            f"  Total Parameters:       {self.total_parameters:>14,d}",
            f"  Active Parameters:      {self.active_parameters:>14,d}",
            f"  Forward FLOPs:          {self.forward_flops:>14,d}",
            f"  Training FLOPs (est):   {self.training_flops:>14,d}",
            f"  Peak Activation Memory: {self._format_bytes(self.peak_activation_bytes):>14}",
            f"  KV / Recurrent State:   {self._format_bytes(self.kv_cache_bytes):>14}",
            "============================================================",
        ]
        return "\n".join(lines)

    __str__ = summary

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(num_bytes) < 1024.0 or unit == "TB":
                return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{num_bytes} B"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} B"


def _resolve_dim(dim: Any, batch_size: int, seq_lens: Mapping[str, int]) -> int:
    if isinstance(dim, int):
        return dim
    if dim == "B":
        return batch_size
    multiple = batch_multiple(dim)
    if multiple is not None:
        return multiple * batch_size
    if isinstance(dim, str) and dim in seq_lens:
        return seq_lens[dim]
    return 1


def _resolve_shape(shape: tuple[Any, ...], batch_size: int, seq_lens: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(_resolve_dim(d, batch_size, seq_lens) for d in shape)


def _compute_node_parameters(
    alias: str,
    args: dict[str, Any],
    in_shapes: dict[str, tuple[int, ...]],
    out_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, int]:
    """Calculate (total_parameters, active_parameters) analytically without PyTorch."""
    if alias == "linear":
        din = in_shapes["x"][-1]
        dout = args.get("out_features") or out_shapes["out"][-1]
        bias = args.get("bias", True)
        params = din * dout + (dout if bias else 0)
        return params, params

    if alias in ("conv", "conv2d"):
        cin = in_shapes["x"][1]
        cout = args.get("out_channels", out_shapes["out"][1])
        k = args.get("kernel_size", 1)
        kh, kw = (k, k) if isinstance(k, int) else (int(k[0]), int(k[1]))
        groups = args.get("groups", 1)
        bias = args.get("bias", True)
        params = cout * (cin // groups) * kh * kw + (cout if bias else 0)
        return params, params

    if alias == "conv1d":
        channels_last = args.get("channels_last", False)
        cin = in_shapes["x"][-1] if channels_last else in_shapes["x"][1]
        cout = args.get("out_channels", out_shapes["out"][-1] if channels_last else out_shapes["out"][1])
        k = args.get("kernel_size", 1)
        groups = args.get("groups", 1)
        bias = args.get("bias", True)
        params = cout * (cin // groups) * k + (cout if bias else 0)
        return params, params

    if alias in ("deconv", "deconv2d", "conv_transpose2d"):
        cin = in_shapes["x"][1]
        cout = args.get("out_channels", out_shapes["out"][1])
        k = args.get("kernel_size", 1)
        kh, kw = (k, k) if isinstance(k, int) else (int(k[0]), int(k[1]))
        groups = args.get("groups", 1)
        bias = args.get("bias", True)
        params = cin * (cout // groups) * kh * kw + (cout if bias else 0)
        return params, params

    if alias in ("deconv1d", "conv_transpose1d"):
        channels_last = args.get("channels_last", False)
        cin = in_shapes["x"][-1] if channels_last else in_shapes["x"][1]
        cout = args.get("out_channels", out_shapes["out"][-1] if channels_last else out_shapes["out"][1])
        k = args.get("kernel_size", 1)
        groups = args.get("groups", 1)
        bias = args.get("bias", True)
        params = cin * (cout // groups) * k + (cout if bias else 0)
        return params, params

    if alias == "embedding":
        num_emb = args["num_embeddings"]
        emb_dim = args.get("dim") or out_shapes["out"][-1]
        params = num_emb * emb_dim
        return params, params

    if alias == "cls_token":
        dim = in_shapes["x"][-1]
        return dim, dim

    if alias == "pos_embed":
        max_len = args.get("max_len") or in_shapes["x"][1]
        dim = in_shapes["x"][-1]
        params = max_len * dim
        return params, params

    if alias == "rms_norm":
        dim = in_shapes["x"][-1]
        affine = args.get("affine", True)
        params = dim if affine else 0
        return params, params

    if alias == "layer_norm":
        dim = in_shapes["x"][-1]
        affine = args.get("elementwise_affine", True)
        bias = args.get("bias", True)
        params = (dim + (dim if bias else 0)) if affine else 0
        return params, params

    if alias in ("group_norm", "batch_norm"):
        c = in_shapes["x"][1]
        affine = args.get("affine", True)
        params = (2 * c) if affine else 0
        return params, params

    if alias == "instance_norm":
        c = in_shapes["x"][1]
        affine = args.get("affine", False)
        params = (2 * c) if affine else 0
        return params, params

    if alias == "adaln_zero":
        d = in_shapes["x"][-1]
        d_cond = in_shapes["cond"][-1]
        gate = args.get("gate", True)
        dim_mult = 3 if gate else 2
        params = d_cond * (dim_mult * d) + (dim_mult * d)
        return params, params

    if alias == "grn":
        c = in_shapes["x"][1]
        params = 2 * c
        return params, params

    if alias == "dy_tanh":
        dim = args.get("dim") or in_shapes["x"][-1]
        params = 1 + 2 * dim
        return params, params

    if alias == "snake":
        dim = args.get("dim") or args.get("channels") or in_shapes["x"][-1]
        return dim, dim

    if alias == "learned_scale":
        return 1, 1

    if alias == "hopfield":
        patterns = int(args.get("patterns", 1))
        d = in_shapes["x"][-1]
        params = patterns * d
        return params, params

    if alias == "spatial_attention":
        c = in_shapes["x"][1]
        reduction = int(args.get("reduction", 8))
        query_ch = c // reduction
        params = 2 * (c * query_ch + query_ch) + (c * c + c) + 1
        return params, params

    if alias == "feed_forward":
        din = in_shapes["x"][-1]
        hidden = args.get("hidden") or args.get("hidden_features") or (din * args.get("expansion", 4))
        dout = args.get("out_features") or din
        bias = args.get("bias", True)
        fc1 = din * hidden + (hidden if bias else 0)
        fc2 = hidden * dout + (dout if bias else 0)
        params = fc1 + fc2
        return params, params

    if alias == "swiglu":
        din = in_shapes["x"][-1]
        hidden = args.get("hidden") or args.get("hidden_features") or (din * args.get("expansion", 4))
        dout = args.get("out_features") or din
        bias = args.get("bias", False)
        w_gate = din * hidden + (hidden if bias else 0)
        w_up = din * hidden + (hidden if bias else 0)
        w_down = hidden * dout + (dout if bias else 0)
        params = w_gate + w_up + w_down
        return params, params

    if alias == "attention":
        d = in_shapes["x"][-1]
        heads = args["heads"]
        head_dim = args.get("head_dim", 0) or (d // heads)
        kv_heads = args.get("kv_heads", 0) or heads
        bias = args.get("bias", True)
        qkv_bias = bias and args.get("qkv_bias", True)
        out_bias = bias and args.get("out_bias", True)
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        q_proj = d * q_dim + (q_dim if qkv_bias else 0)
        k_proj = d * kv_dim + (kv_dim if qkv_bias else 0)
        v_proj = d * kv_dim + (kv_dim if qkv_bias else 0)
        o_proj = q_dim * d + (d if out_bias else 0)
        qk_norm = args.get("qk_norm", "none")
        qk_params = (2 * head_dim) if qk_norm == "rms_norm" else 0
        rpb = args.get("relative_position_bias", False)
        rpb_params = 0
        if rpb:
            sh = args.get("spatial_shape", (0, 0))
            h, w = (sh, sh) if isinstance(sh, int) else (sh[0], sh[1])
            rpb_params = (2 * h - 1) * (2 * w - 1) * heads
        total = q_proj + k_proj + v_proj + o_proj + qk_params + rpb_params
        return total, total

    if alias == "cross_attention":
        d = in_shapes["x"][-1]
        d_cross = in_shapes["context"][-1]
        heads = args["heads"]
        head_dim = args.get("head_dim", 0) or (d // heads)
        kv_heads = args.get("kv_heads", 0) or heads
        bias = args.get("bias", True)
        qkv_bias = bias and args.get("qkv_bias", True)
        out_bias = bias and args.get("out_bias", True)
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        q_proj = d * q_dim + (q_dim if qkv_bias else 0)
        k_proj = d_cross * kv_dim + (kv_dim if qkv_bias else 0)
        v_proj = d_cross * kv_dim + (kv_dim if qkv_bias else 0)
        o_proj = q_dim * d + (d if out_bias else 0)
        total = q_proj + k_proj + v_proj + o_proj
        return total, total

    if alias == "joint_attention":
        d_x = in_shapes["x"][-1]
        d_ctx = in_shapes["context"][-1]
        heads = args["heads"]
        head_dim = args.get("head_dim", 0) or (d_x // heads)
        bias = args.get("bias", True)
        qkv_bias = bias and args.get("qkv_bias", True)
        out_bias = bias and args.get("out_bias", True)
        inner = heads * head_dim
        qkv_x = 3 * (d_x * inner + (inner if qkv_bias else 0))
        qkv_c = 3 * (d_ctx * inner + (inner if qkv_bias else 0))
        o_x = inner * d_x + (d_x if out_bias else 0)
        o_c = inner * d_ctx + (d_ctx if out_bias else 0)
        total = qkv_x + qkv_c + o_x + o_c
        return total, total

    if alias == "latent_attention":
        d = in_shapes["x"][-1]
        heads = int(args["heads"])
        kv_lora_rank = int(args["kv_lora_rank"])
        q_lora_rank = int(args.get("q_lora_rank", 0))
        qk_nope_head_dim = int(args.get("qk_nope_head_dim", 0) or (d // heads))
        qk_rope_head_dim = int(args.get("qk_rope_head_dim", 0))
        v_head_dim = int(args.get("v_head_dim", 0) or qk_nope_head_dim)
        bias = bool(args.get("bias", False))
        total_qk_dim = qk_nope_head_dim + qk_rope_head_dim

        kv_down = d * kv_lora_rank + (kv_lora_rank if bias else 0)
        kv_norm = kv_lora_rank
        k_up = kv_lora_rank * (heads * qk_nope_head_dim) + ((heads * qk_nope_head_dim) if bias else 0)
        v_up = kv_lora_rank * (heads * v_head_dim) + ((heads * v_head_dim) if bias else 0)
        k_rope = (d * qk_rope_head_dim + (qk_rope_head_dim if bias else 0)) if qk_rope_head_dim > 0 else 0

        if q_lora_rank > 0:
            q_down = d * q_lora_rank + (q_lora_rank if bias else 0)
            q_norm = q_lora_rank
            q_up = q_lora_rank * (heads * total_qk_dim) + ((heads * total_qk_dim) if bias else 0)
        else:
            q_down = 0
            q_norm = 0
            q_up = d * (heads * total_qk_dim) + ((heads * total_qk_dim) if bias else 0)

        o_proj = (heads * v_head_dim) * d + (d if bias else 0)
        total = kv_down + kv_norm + k_up + v_up + k_rope + q_down + q_norm + q_up + o_proj
        return total, total

    if alias == "gated_delta_attention":
        d = in_shapes["x"][-1]
        heads = int(args["heads"])
        head_dim = int(args.get("head_dim", 0) or (d // heads))
        inner_dim = heads * head_dim
        conv_size = int(args.get("conv_size", 4))
        bias = bool(args.get("bias", False))
        decay_kind = str(args.get("decay_kind", "channel"))
        gate_rank = int(args.get("gate_rank", 0) or max(head_dim, max(1, d // 16)))

        q_proj = d * inner_dim + (inner_dim if bias else 0)
        k_proj = d * inner_dim + (inner_dim if bias else 0)
        v_proj = d * inner_dim + (inner_dim if bias else 0)
        conv_params = 3 * (conv_size * inner_dim + inner_dim) if conv_size > 0 else 0
        b_proj = d * heads + (heads if bias else 0)

        if decay_kind == "channel":
            alpha_down = d * gate_rank
            alpha_up = gate_rank * inner_dim + inner_dim
            a_log = heads * head_dim
        else:
            alpha_down = 0
            alpha_up = d * heads + (heads if bias else 0)
            a_log = heads

        gate_down = d * gate_rank
        gate_up = gate_rank * inner_dim
        rms_norm_w = heads * head_dim
        o_proj = inner_dim * d + (d if bias else 0)

        total = q_proj + k_proj + v_proj + conv_params + b_proj + alpha_down + alpha_up + a_log + gate_down + gate_up + rms_norm_w + o_proj
        return total, total

    if alias in ("mamba2", "ssd"):
        d_model = int(args.get("d_model", 0) or in_shapes["x"][-1])
        d_state = int(args.get("d_state", 64))
        d_conv = int(args.get("d_conv", 4))
        heads = int(args.get("heads", 8))
        head_dim = int(args.get("head_dim", 64))
        bias = bool(args.get("bias", False))
        conv_bias = bool(args.get("conv_bias", True))

        inner_dim = heads * head_dim
        bc_dim = heads * d_state
        in_proj_dim = 2 * inner_dim + 2 * bc_dim + heads
        in_proj = d_model * in_proj_dim + (in_proj_dim if bias else 0)

        conv_dim = inner_dim + 2 * bc_dim
        conv_params = (d_conv * conv_dim + (conv_dim if conv_bias else 0)) if d_conv > 0 else 0
        a_log = heads
        dt_bias = heads
        d_param = heads
        norm = inner_dim
        out_proj = inner_dim * d_model + (d_model if bias else 0)

        total = in_proj + conv_params + a_log + dt_bias + d_param + norm + out_proj
        return total, total

    if alias == "moe":
        d = in_shapes["x"][-1]
        experts = int(args["experts"])
        hidden = int(args["hidden"])
        top_k = int(args["top_k"])
        expert_kind = str(args.get("expert_kind", "mlp"))
        bias = bool(args.get("bias", True))

        router = d * experts
        if expert_kind == "swiglu":
            per_expert = 3 * d * hidden + ((2 * hidden + d) if bias else 0)
        else:
            per_expert = 2 * d * hidden + ((hidden + d) if bias else 0)

        total = router + experts * per_expert
        active = router + top_k * per_expert
        return total, active

    if alias == "transformer_block":
        d = in_shapes["x"][-1]
        heads = int(args["heads"])
        head_dim = int(args.get("head_dim", 0) or (d // heads))
        kv_heads = int(args.get("kv_heads", 0) or heads)
        bias = bool(args.get("bias", True))
        qkv_bias = bias and bool(args.get("qkv_bias", True))
        out_bias = bias and bool(args.get("out_bias", True))
        norm = str(args.get("norm") or args.get("norm_kind", "layer_norm"))
        ffn_kind = str(args.get("ffn_kind", "mlp"))
        mlp_ratio = int(args.get("mlp_ratio", 4))
        hidden_dim = int(args.get("hidden") or args.get("hidden_dim", 0) or (d * mlp_ratio))
        layer_scale = bool(args.get("layer_scale", False))

        norm_params = d if norm == "rms_norm" else (d + (d if bias else 0))
        norm1 = norm_params
        norm2 = norm_params

        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        attn = (d * q_dim + (q_dim if qkv_bias else 0) +
                2 * (d * kv_dim + (kv_dim if qkv_bias else 0)) +
                q_dim * d + (d if out_bias else 0))
        if args.get("qk_norm", "none") == "rms_norm":
            attn += 2 * head_dim

        if ffn_kind == "swiglu":
            ffn = 3 * d * hidden_dim + ((2 * hidden_dim + d) if bias else 0)
        else:
            ffn = d * hidden_dim + (hidden_dim if bias else 0) + hidden_dim * d + (d if bias else 0)

        gamma = (2 * d) if layer_scale else 0
        total = norm1 + attn + norm2 + ffn + gamma
        return total, total

    if alias == "resblock":
        kind = str(args.get("kind", "basic"))
        in_ch = in_shapes["x"][1]
        out_ch = out_shapes["out"][1]
        stride = int(args.get("stride", 1))
        bias = bool(args.get("bias", False))
        norm_kind = str(args.get("norm_kind", "batch_norm"))
        norm_p = (2 if norm_kind in ("batch_norm", "group_norm") else 1)

        total = 0
        if kind == "bottleneck":
            mid_ch = out_ch // 4
            # conv1: 1x1 in_ch -> mid_ch
            total += in_ch * mid_ch * 1 * 1 + (mid_ch if bias else 0) + norm_p * mid_ch
            # conv2: 3x3 mid_ch -> mid_ch
            total += mid_ch * mid_ch * 3 * 3 + (mid_ch if bias else 0) + norm_p * mid_ch
            # conv3: 1x1 mid_ch -> out_ch
            total += mid_ch * out_ch * 1 * 1 + (out_ch if bias else 0) + norm_p * out_ch
        else:
            # basic: two 3x3 convs
            total += in_ch * out_ch * 3 * 3 + (out_ch if bias else 0) + norm_p * out_ch
            total += out_ch * out_ch * 3 * 3 + (out_ch if bias else 0) + norm_p * out_ch

        # shortcut
        if stride != 1 or in_ch != out_ch:
            total += in_ch * out_ch * 1 * 1 + (out_ch if bias else 0) + norm_p * out_ch

        return total, total

    if alias == "patch_embed":
        cin = in_shapes["x"][1]
        cout = out_shapes["out"][-1]
        p = args.get("patch_size", 16)
        ph, pw = (p, p) if isinstance(p, int) else (p[0], p[1])
        bias = args.get("bias", True)
        norm = args.get("norm", False)
        params = cout * cin * ph * pw + (cout if bias else 0)
        if norm:
            params += 2 * cout
        return params, params

    return 0, 0


def _compute_node_flops(
    alias: str,
    args: dict[str, Any],
    in_shapes: dict[str, tuple[int, ...]],
    out_shapes: dict[str, tuple[int, ...]],
) -> int:
    """Calculate forward FLOPs analytically (standard: 1 MAC = 2 FLOPs)."""
    out_s = next(iter(out_shapes.values()))
    out_elements = math.prod(out_s)

    if alias == "linear":
        din = in_shapes["x"][-1]
        dout = out_s[-1]
        num_tokens = math.prod(out_s[:-1])
        return 2 * num_tokens * din * dout

    if alias == "conv1d":
        channels_last = args.get("channels_last", False)
        b = in_shapes["x"][0]
        lout = out_s[1] if channels_last else out_s[-1]
        cin = in_shapes["x"][-1] if channels_last else in_shapes["x"][1]
        cout = out_s[-1] if channels_last else out_s[1]
        k = args.get("kernel_size", 1)
        groups = args.get("groups", 1)
        return 2 * b * lout * cout * (cin // groups) * k

    if alias in ("conv", "conv2d"):
        b = in_shapes["x"][0]
        cin = in_shapes["x"][1]
        cout = out_s[1]
        hout, wout = out_s[2], out_s[3]
        k = args.get("kernel_size", 1)
        kh, kw = (k, k) if isinstance(k, int) else (int(k[0]), int(k[1]))
        groups = args.get("groups", 1)
        return 2 * b * hout * wout * cout * (cin // groups) * kh * kw

    if alias in ("deconv", "deconv2d", "conv_transpose2d"):
        b = in_shapes["x"][0]
        cin = in_shapes["x"][1]
        cout = out_s[1]
        hout, wout = out_s[2], out_s[3]
        k = args.get("kernel_size", 1)
        kh, kw = (k, k) if isinstance(k, int) else (int(k[0]), int(k[1]))
        groups = args.get("groups", 1)
        return 2 * b * hout * wout * cout * (cin // groups) * kh * kw

    if alias in ("deconv1d", "conv_transpose1d"):
        channels_last = args.get("channels_last", False)
        b = in_shapes["x"][0]
        lout = out_s[1] if channels_last else out_s[-1]
        cin = in_shapes["x"][-1] if channels_last else in_shapes["x"][1]
        cout = out_s[-1] if channels_last else out_s[1]
        k = args.get("kernel_size", 1)
        groups = args.get("groups", 1)
        return 2 * b * lout * cout * (cin // groups) * k

    if alias == "matmul":
        sa = in_shapes["a"]
        m = sa[-2] if not args.get("transpose_a") else sa[-1]
        k_dim = sa[-1] if not args.get("transpose_a") else sa[-2]
        n_dim = out_s[-1]
        batch_elems = math.prod(out_s[:-2])
        return 2 * batch_elems * m * k_dim * n_dim

    if alias in ("attention", "cross_attention"):
        b = in_shapes["x"][0]
        t = in_shapes["x"][1]
        d = in_shapes["x"][2]
        heads = args["heads"]
        head_dim = args.get("head_dim", 0) or (d // heads)
        kv_heads = args.get("kv_heads", 0) or heads
        ctx_shape = in_shapes.get("context", in_shapes["x"])
        t_kv = ctx_shape[1]
        d_cross = ctx_shape[2]

        qkv_projs = 2 * b * t * d * (heads * head_dim) + 2 * 2 * b * t_kv * d_cross * (kv_heads * head_dim)
        qk_matmul = 2 * b * heads * t * t_kv * head_dim
        softmax_flops = 3 * b * heads * t * t_kv
        attn_v_matmul = 2 * b * heads * t * t_kv * head_dim
        out_proj = 2 * b * t * (heads * head_dim) * d
        return qkv_projs + qk_matmul + softmax_flops + attn_v_matmul + out_proj

    if alias == "moe":
        d = in_shapes["x"][-1]
        experts = int(args["experts"])
        hidden = int(args["hidden"])
        top_k = int(args["top_k"])
        expert_kind = str(args.get("expert_kind", "mlp"))
        tokens = math.prod(in_shapes["x"][:-1])

        router_flops = 2 * tokens * d * experts
        mult = 3 if expert_kind == "swiglu" else 2
        expert_flops = top_k * (mult * 2 * tokens * d * hidden)
        return router_flops + expert_flops

    if alias == "transformer_block":
        d = in_shapes["x"][-1]
        heads = int(args["heads"])
        head_dim = int(args.get("head_dim", 0) or (d // heads))
        expansion = int(args.get("expansion", 4))
        hidden_dim = int(args.get("hidden_dim", 0) or (d * expansion))
        ffn_kind = str(args.get("ffn_kind", "mlp"))
        b, t = in_shapes["x"][0], in_shapes["x"][1]

        qkv_projs = 4 * 2 * b * t * d * (heads * head_dim)
        scores = 2 * b * heads * t * t * head_dim
        softmax = 3 * b * heads * t * t
        attn_v = 2 * b * heads * t * t * head_dim
        attn_out = 2 * b * t * (heads * head_dim) * d
        attn_total = qkv_projs + scores + softmax + attn_v + attn_out

        norms = 2 * 8 * b * t * d
        mult = 3 if ffn_kind == "swiglu" else 2
        ffn = mult * 2 * b * t * d * hidden_dim
        return attn_total + norms + ffn

    if alias == "rms_norm":
        return 5 * out_elements

    if alias in ("layer_norm", "group_norm", "batch_norm", "instance_norm", "adaptive_norm"):
        return 8 * out_elements

    if alias in ("gelu", "silu", "mish", "swiglu", "dy_tanh"):
        return 8 * out_elements

    if alias in ("relu", "clamp", "where"):
        return 1 * out_elements

    if alias in ("add", "sub", "mul", "broadcast_add", "broadcast_mul"):
        return 1 * out_elements

    return 2 * out_elements


def _compute_kv_cache_bytes(
    alias: str,
    args: dict[str, Any],
    in_shapes: dict[str, tuple[int, ...]],
    dtype_size: int,
) -> int:
    """Calculate KV-cache or recurrent state bytes for stateful sequence decoders."""
    if alias in ("attention", "cross_attention"):
        b = in_shapes["x"][0]
        t = in_shapes.get("context", in_shapes["x"])[1]
        d = in_shapes["x"][2]
        heads = args["heads"]
        head_dim = args.get("head_dim", 0) or (d // heads)
        kv_heads = args.get("kv_heads", 0) or heads
        return 2 * b * kv_heads * head_dim * t * dtype_size

    if alias == "latent_attention":
        b = in_shapes["x"][0]
        t = in_shapes["x"][1]
        kv_lora_rank = int(args["kv_lora_rank"])
        qk_rope_head_dim = int(args.get("qk_rope_head_dim", 0))
        return b * t * (kv_lora_rank + qk_rope_head_dim) * dtype_size

    if alias == "gated_delta_attention":
        b = in_shapes["x"][0]
        d = in_shapes["x"][2]
        heads = int(args["heads"])
        head_dim = int(args.get("head_dim", 0) or (d // heads))
        conv_size = int(args.get("conv_size", 4))
        inner_dim = heads * head_dim
        recurrent_state = b * heads * (head_dim ** 2) * dtype_size
        conv_buf = b * inner_dim * max(0, conv_size - 1) * dtype_size
        return recurrent_state + conv_buf

    if alias in ("mamba2", "ssd"):
        b = in_shapes["x"][0]
        heads = int(args.get("heads", 8))
        head_dim = int(args.get("head_dim", 64))
        d_state = int(args.get("d_state", 64))
        d_conv = int(args.get("d_conv", 4))
        inner_dim = heads * head_dim
        state_bytes = b * heads * head_dim * d_state * dtype_size
        conv_bytes = b * inner_dim * max(0, d_conv - 1) * dtype_size
        return state_bytes + conv_bytes

    if alias == "transformer_block":
        b = in_shapes["x"][0]
        t = in_shapes["x"][1]
        d = in_shapes["x"][2]
        heads = int(args["heads"])
        head_dim = int(args.get("head_dim", 0) or (d // heads))
        kv_heads = int(args.get("kv_heads", 0) or heads)
        return 2 * b * kv_heads * head_dim * t * dtype_size

    return 0


def profile_plan(
    plan: ResolvedPlan,
    batch_size: int = 1,
    sequence_lengths: Mapping[str, int] | None = None,
) -> PlanProfile:
    """Perform zero-compute static profiling on a ResolvedPlan."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    seq_lens = sequence_lengths or {}
    dtype_size = DTYPE_SIZES.get(plan.dtype, 4)

    # 1. Resolve concrete shapes for external contracts and all nodes
    concrete_ext_inputs: dict[str, tuple[int, ...]] = {
        name: _resolve_shape(port["shape"] if isinstance(port, Mapping) else getattr(port, "shape", port), batch_size, seq_lens)
        for name, port in plan.inputs.items()
    }

    node_in_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    node_out_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    ref_shapes: dict[str, tuple[int, ...]] = {}

    for name, s in concrete_ext_inputs.items():
        ref_shapes[name] = s
        ref_shapes[f"input:{name}"] = s

    for node in plan.nodes:
        in_s = {
            port: _resolve_shape(s, batch_size, seq_lens)
            for port, s in node.input_shapes.items()
        }
        out_s = {
            port: _resolve_shape(s, batch_size, seq_lens)
            for port, s in node.output_shapes.items()
        }
        node_in_shapes[node.id] = in_s
        node_out_shapes[node.id] = out_s

        # Register output refs
        for port, s in out_s.items():
            ref_shapes[f"node:{node.id}/{port}"] = s
            ref_shapes[f"{node.id}:{port}"] = s
            ref_shapes[f"{node.id}/{port}"] = s
            ref_shapes[node.id] = s

    # 2. Compute per-node profiles
    node_profiles: list[NodeProfile] = []
    total_params = 0
    active_params = 0
    total_flops = 0
    total_kv_bytes = 0

    for node in plan.nodes:
        alias = plan._operation_name(node.op) if hasattr(plan, "_operation_name") else (
            node.op.split("@")[0] if "@" in node.op else node.op
        )
        in_s = node_in_shapes[node.id]
        out_s = node_out_shapes[node.id]
        tot_p, act_p = _compute_node_parameters(alias, node.args, in_s, out_s)
        flops = _compute_node_flops(alias, node.args, in_s, out_s)
        act_bytes = sum(math.prod(s) * dtype_size for s in out_s.values())
        kv_bytes = _compute_kv_cache_bytes(alias, node.args, in_s, dtype_size)

        np = NodeProfile(
            id=node.id,
            op=alias,
            total_parameters=tot_p,
            active_parameters=act_p,
            forward_flops=flops,
            activation_bytes=act_bytes,
            kv_cache_bytes=kv_bytes,
        )
        node_profiles.append(np)
        total_params += tot_p
        active_params += act_p
        total_flops += flops
        total_kv_bytes += kv_bytes

    # 3. Peak Live Activation Memory Simulation (Slot-Recycled Liveness)
    graph_output_refs = set()
    for port in plan.outputs.values():
        ref = port["ref"] if isinstance(port, Mapping) else getattr(port, "ref", str(port))
        graph_output_refs.add(ref)

    last_consumer_step: dict[str, int] = {}
    for i, node in enumerate(plan.nodes):
        for ref in node.inputs.values():
            last_consumer_step[ref] = i

    current_live_bytes = sum(
        math.prod(s) * dtype_size for s in concrete_ext_inputs.values()
    )
    peak_bytes = current_live_bytes

    for i, node in enumerate(plan.nodes):
        # Outputs produced by node
        out_s = node_out_shapes[node.id]
        for s in out_s.values():
            current_live_bytes += math.prod(s) * dtype_size
        peak_bytes = max(peak_bytes, current_live_bytes)

        # Free inputs whose last consumer is step i
        for ref in set(node.inputs.values()):
            if last_consumer_step.get(ref) == i and ref not in graph_output_refs:
                s = ref_shapes.get(ref)
                if s is not None:
                    current_live_bytes -= math.prod(s) * dtype_size

    nodes_tuple = tuple(node_profiles)
    by_id = {np.id: np for np in node_profiles}

    return PlanProfile(
        total_parameters=total_params,
        active_parameters=active_params,
        forward_flops=total_flops,
        peak_activation_bytes=max(peak_bytes, 1),
        kv_cache_bytes=total_kv_bytes,
        nodes=nodes_tuple,
        _by_id=by_id,
    )
