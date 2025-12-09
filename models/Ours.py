import torch
import torch.nn as nn
import numpy as np
import math


# ============================
# RoPE Utilities
# ============================
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def get_rope_cos_sin(seq_len, head_dim, device, dtype, base: float = 10000.0):
    """
    Return cos, sin shape = (1, 1, seq_len, head_dim)
    """
    assert head_dim % 2 == 0
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (idx / head_dim))  # (head_dim/2,)

    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, head_dim/2)

    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# ============================
# Helpers
# ============================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================
# RoPE Self-Attention
# ============================
class Attention(nn.Module):
    """
    Multi-head self-attention with RoPE.
    x: (B, L, C)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)  # (B, L, 3C)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)
        dtype = x.dtype
        device = x.device

        # === RoPE ===
        cos, sin = get_rope_cos_sin(L, self.head_dim, device, dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ============================
# Cross-Attention with RoPE
# ============================
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, return_attn_weights=False):
        """
        x:       (B, N, C)
        context: (B, M, C)
        """
        B, N, C = x.shape
        M = context.shape[1]
        dtype = x.dtype
        device = x.device

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # === RoPE ===
        cos_q, sin_q = get_rope_cos_sin(N, self.head_dim, device, dtype)
        cos_k, sin_k = get_rope_cos_sin(M, self.head_dim, device, dtype)

        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return (out, attn) if return_attn_weights else out


# ============================
# Timestep Embedding
# ============================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ============================
# DiT Block
# ============================
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(hidden_dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.cross_attn = CrossAttention(hidden_dim, num_heads)

        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(self, x, y, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + self.cross_attn(self.norm2(x), y)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


# ============================
# Final Layer
# ============================
class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ============================
# DiT (Full Model)
# ============================
class DiT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.x_embedder = nn.Linear(args.feature_dim, args.hidden_dim)
        self.y_embedder = nn.Linear(args.feature_dim, args.hidden_dim)
        self.t_embedder = TimestepEmbedder(args.hidden_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(args.hidden_dim, args.num_heads, mlp_ratio=args.mlp_ratio)
            for _ in range(args.num_dit_block)
        ])
        self.final_layer = FinalLayer(args.hidden_dim, args.feature_dim)

    def forward(self, x, y, t):
        x = self.x_embedder(x)
        y = self.y_embedder(y)
        t = self.t_embedder(t)

        for blk in self.blocks:
            x = blk(x, y, t)

        return self.final_layer(x, t)
