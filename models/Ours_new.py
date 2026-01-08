import math
import copy
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Sin/Cos timestep embedding (DDPM 스타일).
    t: (B,)
    return: (B, dim)
    """
    if t.dim() != 1:
        raise ValueError(f"t must be 1D (B,), got {tuple(t.shape)}")

    half = dim // 2
    device = t.device
    dtype = torch.float32  # trig는 fp32가 안정적
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device, dtype=dtype) / half
    )  # (half,)
    args = t.to(dtype)[:, None] * freqs[None]  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """
    DiT-style timestep embedder:
      sinusoidal -> MLP -> (B, hidden_dim)
    """
    def __init__(self, hidden_dim: int, freq_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_dim = freq_dim or hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.freq_dim)
        return self.mlp(emb)


class LearnedPositionalEmbedding1D(nn.Module):
    """
    Learnable 1D positional embedding for sequences.
    """
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.shape[1]
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.max_len}")
        return self.pos[:, :L, :]


class SelfAttentionSDPA(nn.Module):
    """
    torch.scaled_dot_product_attention 기반 MHA.
    (GPU에서 조건 맞으면 FlashAttention으로 메모리/속도 이점 큼)
    """
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = attn_dropout
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, L, Hd)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False
        )  # (B, H, L, Hd)

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(attn)
        out = self.proj_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock1D(nn.Module):
    """
    DiT-style Transformer block with AdaLN-Zero conditioning.
    x: (B, L, D)
    cond: (B, D)  (여기서는 t-embedding)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttentionSDPA(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        # (shift, scale, gate) for attention and MLP => 6*D
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # AdaLN-Zero: 마지막 linear를 0 init -> 초기엔 거의 identity
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h = self.attn(h)
        x = x + gate_msa[:, None, :] * h

        h = self.norm2(x)
        h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = self.mlp(h)
        x = x + gate_mlp[:, None, :] * h
        return x


class ZeroLinear(nn.Linear):
    """
    ControlNet-style 'zero conv'의 token 버전: 초기 출력이 0이 되도록 0 init.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# @dataclass
# class DiTControlNet1DConfig:
#     seq_len: int = 96
#     in_dim: int = 7
#     hidden_dim: int = 256
#     depth: int = 8
#     num_heads: int = 8
#     mlp_ratio: float = 4.0
#     dropout: float = 0.0
#     attn_dropout: float = 0.0
#     out_dim: int = 7  # 보통 epsilon 예측이면 in_dim과 동일

@dataclass
class DiTControlNet1DConfig:
    seq_len: int = 96
    in_dim: int = 7
    hidden_dim: int = 64
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    out_dim: int = 7  # 보통 epsilon 예측이면 in_dim과 동일

class DiT(nn.Module):
    """
    Timeseries DiT + ControlNet (1D tokens)

    Inputs:
      x: (B, L, C)  noisy sample at diffusion step t
      y: (B, L, C)  control conditioning sequence
      t: (B,)       diffusion timestep (int or float)

    Output:
      eps_hat: (B, L, C) predicted noise (혹은 velocity 목적이면 그에 맞게 사용)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # channel_independent: 0 = channel-mixing, 1 = channel-independent
        self.channel_independent = getattr(args, 'channel_independent', 0)
        self.feature_dim = args.feature_dim

        # Channel-independent면 항상 in_dim/out_dim=1로 처리
        embed_dim = 1 if self.channel_independent else args.feature_dim

        self.cfg = DiTControlNet1DConfig(
            in_dim=embed_dim,
            out_dim=embed_dim,
            seq_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            depth=args.num_dit_block,
            num_heads=args.num_heads,
            mlp_ratio=getattr(args, 'mlp_ratio', 4.0),
        )
        cfg = self.cfg
        D = cfg.hidden_dim

        # Token embed
        self.x_embed = nn.Linear(cfg.in_dim, D)
        self.y_embed = nn.Linear(cfg.in_dim, D)

        # Pos/time embed
        self.pos_embed = LearnedPositionalEmbedding1D(cfg.seq_len, D)
        self.time_embed = TimestepEmbedder(D)

        # Main DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock1D(
                dim=D,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                attn_dropout=cfg.attn_dropout,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])

        # Control DiT blocks (same config)
        self.control_blocks = nn.ModuleList([
            DiTBlock1D(
                dim=D,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                attn_dropout=cfg.attn_dropout,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])

        # Zero projections for injection (input + each block)
        self.control_zero_in = ZeroLinear(D, D)
        self.control_zero_out = nn.ModuleList([ZeroLinear(D, D) for _ in range(cfg.depth)])

        # Output head
        self.final_norm = nn.LayerNorm(D, elementwise_affine=True, eps=1e-6)
        self.out_proj = nn.Linear(D, cfg.out_dim)

    @torch.no_grad()
    def copy_main_weights_to_control(self):
        """
        (선택) main DiT를 미리 학습(pretrain)해둔 경우,
        ControlNet branch를 main과 같은 weight로 시작시키고 싶을 때 호출.
        """
        for cb, mb in zip(self.control_blocks, self.blocks):
            cb.load_state_dict(copy.deepcopy(mb.state_dict()))

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, control_scale: Union[float, List[float]] = 1.0) -> torch.Tensor:
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError(f"x and y must be (B, L, C). got x={tuple(x.shape)}, y={tuple(y.shape)}")
        if t.dim() != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must be (B,). got t={tuple(t.shape)}, B={x.shape[0]}")

        B, L_x, F = x.shape
        L_y = y.shape[1]

        # Channel-independent: (B, L, F) → (B*F, L, 1)
        if self.channel_independent and F > 1:
            x = x.permute(0, 2, 1).reshape(B * F, L_x, 1)
            y = y.permute(0, 2, 1).reshape(B * F, L_y, 1)
            t = t.repeat_interleave(F, dim=0)  # (B,) → (B*F,)

        B_eff, L, _ = x.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"L={L} exceeds configured seq_len={self.cfg.seq_len}")

        # timestep conditioning
        cond = self.time_embed(t)  # (B, D)

        # tokens + pos
        x_tok = self.x_embed(x)  # (B, L, D)
        y_tok = self.y_embed(y)  # (B, L, D)
        pos = self.pos_embed(x_tok)  # (1, L, D)
        x_tok = x_tok + pos
        y_tok = y_tok + pos

        # ---- Control branch: residuals 만들기 ----
        ctrl_residuals: List[torch.Tensor] = []

        # input injection residual
        ctrl_residuals.append(self.control_zero_in(y_tok))  # (B, L, D)

        # per-block residual
        for i, blk in enumerate(self.control_blocks):
            y_tok = blk(y_tok, cond)
            ctrl_residuals.append(self.control_zero_out[i](y_tok))  # (B, L, D)

        # Optional: control strength 스케일 (ControlNet의 conditioning scale 느낌)
        if isinstance(control_scale, (float, int)):
            scales = [float(control_scale)] * (self.cfg.depth + 1)
        else:
            if len(control_scale) != self.cfg.depth + 1:
                raise ValueError(
                    f"control_scale list must have length depth+1={self.cfg.depth + 1}, got {len(control_scale)}"
                )
            scales = [float(s) for s in control_scale]

        ctrl_residuals = [r * s for r, s in zip(ctrl_residuals, scales)]

        # ---- Main branch: 매 블록마다 residual 주입 ----
        x_tok = x_tok + ctrl_residuals[0]
        for i, blk in enumerate(self.blocks):
            x_tok = blk(x_tok, cond)
            x_tok = x_tok + ctrl_residuals[i + 1]

        x_tok = self.final_norm(x_tok)
        eps_hat = self.out_proj(x_tok)  # (B_eff, L, out_dim)

        # Channel-independent: (B*F, L, 1) → (B, L, F)
        if self.channel_independent and F > 1:
            eps_hat = eps_hat.reshape(B, F, L_x).permute(0, 2, 1)

        return eps_hat
