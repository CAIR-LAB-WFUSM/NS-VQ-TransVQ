from __future__ import annotations
from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import rearrange, pack, unpack

from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

# class

class SimVQ(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_transform: Module | None = None,
        init_fn: Callable = identity,
        channel_first = False,
        rotation_trick = True,  # works even better with rotation trick turned on, with no straight through and the commit loss from input to quantize
        input_to_quantize_commit_loss_weight = 0.25,
        commitment_weight = 1.,
        frozen_codebook_dim = None # frozen codebook dim could have different dimensions than projection
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.channel_first = channel_first

        frozen_codebook_dim = default(frozen_codebook_dim, dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (frozen_codebook_dim ** -0.5)
        codebook = init_fn(codebook)

        # the codebook is actually implicit from a linear layer from frozen gaussian or uniform


        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, dim, bias = False)

        self.code_transform = codebook_transform

        self.register_buffer('frozen_codebook', codebook)


        # whether to use rotation trick from Fifty et al. 
        # https://arxiv.org/abs/2410.06424

        self.rotation_trick = rotation_trick

        # commit loss weighting - weighing input to quantize a bit less is crucial for it to work

        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight

        # total commitment loss weight

        self.commitment_weight = commitment_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(
        self,
        indices
    ):
        implicit_codebook = self.codebook

        frozen_codes = get_at('[c] d, b ... -> b ... d', self.frozen_codebook, indices)
        quantized = self.code_transform(frozen_codes)

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')

        return quantized

    def forward(
        self,
        x
    ):
        if self.channel_first:
            x = rearrange(x, 'b d ... -> b ... d')

        x, inverse_pack = pack_one(x, 'b * d')

        implicit_codebook = self.codebook

        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim = -1)

        # select codes

        quantized = get_at('[c] d, b n -> b n d', implicit_codebook, indices)

        # commit loss and straight through, as was done in the paper

        commit_loss = (
            F.mse_loss(x.detach(), quantized) +
            F.mse_loss(x, quantized.detach()) * self.input_to_quantize_commit_loss_weight
        )

        if self.rotation_trick:
            # rotation trick from @cfifty
            quantized = rotate_to(x, quantized)
        else:
            quantized = (quantized - x).detach() + x

        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d-> b d ...')

        return quantized, indices, commit_loss * self.commitment_weight

import torch
from torch import nn
import torch.nn.functional as F

class _FFN(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = _FFN(d_model, mlp_ratio, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))
        return x

class LowRankToTokens(nn.Module):
    """
    低秩分解:  Linear(D_in -> r) [可选非线性] -> Linear(r -> K*d_model)
    若 nonlinear=False，则严格对应 W ≈ U @ V 的低秩近似。
    """
    def __init__(
        self,
        d_in: int,
        k_tokens: int,
        d_model: int,
        rank: int = 64,
        nonlinear: bool = False,   # 置 True 可提升表达，但不再是严格线性低秩
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.rank = rank
        self.proj_u = nn.Linear(d_in, rank, bias=bias)                    # U: D_in -> r
        self.act = nn.GELU() if nonlinear else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.proj_v = nn.Linear(rank, k_tokens * d_model, bias=bias)      # V: r -> K*d_model
        self.k_tokens = k_tokens
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D_in) -> (B, K*d_model)
        h = self.proj_u(x)            # (B, r)
        h = self.act(h)
        h = self.drop(h)
        y = self.proj_v(h)            # (B, K*d_model)
        return y
    
class CodeAsBatchTokenTransformer(nn.Module):
    """
    输入:  (C, D_in)  — 每个 code 一行
    输出:  (C, out_dim)

    做法: 每个 code -> K 个token(共享权重生成) + 1个CLS -> Transformer(L层) -> 取CLS -> out_proj
    计算随 K 和 d_model，而非 codebook_size C。
    """
    def __init__(
        self,
        d_in: int,              # D_in = frozen_codebook_dim
        out_dim: int,           # 输出维度 = SimVQ 的 dim
        K_tokens: int = 8,      # 每个 code 的 token 数(不含CLS)
        d_model: int = 256,     # Transformer 宽度
        depth: int = 2,         # Transformer 层数
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_pos_embed: bool = False
    ):
        super().__init__()
        self.K = K_tokens
        self.d_model = d_model

        # 将单个 code 向量映射为 K 个 token： (B, D_in) -> (B, K, d_model)
        # 用一个两层 MLP 后 reshape，为了表达力更强
        self.to_tokens = LowRankToTokens(
                                d_in=d_in,
                                k_tokens=K_tokens,
                                d_model=d_model,
                                rank=64,          # 推荐: min(128, max(32, (min(d_in, K_tokens*d_model)//4)))
                                nonlinear=True,  # 严格低秩时用 False；想要更强表达可设 True
                                dropout=0.0,
                                bias=True
                            )

        # 可选的 K 长度的位置编码（不依赖 C）
        self.pos = nn.Parameter(torch.zeros(1, K_tokens, d_model)) if use_pos_embed else None

        # 共享的 CLS token
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer 编码器
        self.blocks = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 输出投影到 out_dim
        self.out_proj = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, codebook: torch.Tensor):
        """
        codebook: (C, D_in)
        return:   (C, out_dim)
        """
        B = codebook.shape[0]                  # 这里 B = C
        x = self.to_tokens(codebook)           # (B, K*d_model)
        x = x.view(B, self.K, self.d_model)    # (B, K, d_model)

        if self.pos is not None:
            x = x + self.pos                   # (B, K, d_model)

        # prepend CLS
        cls = self.cls.expand(B, -1, -1)       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)         # (B, K+1, d_model)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                      # (B, d_model)
        y = self.out_proj(cls_out)             # (B, out_dim)
        return y                               # (C, out_dim)

# main

if __name__ == '__main__':

    x = torch.randn(1, 512, 32, 32)

    sim_vq = SimVQ(
        dim = 512,
        codebook_transform = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ),
        codebook_size = 1024,
        channel_first = True
    )

    quantized, indices, commit_loss = sim_vq(x)

    assert x.shape == quantized.shape

import math
import torch
from torch import nn
import torch.nn.functional as F

# ----------------------------
# Linear Attention (batch_first)
# ----------------------------
class LinearAttention(nn.Module):
    """
    Performer-style linear attention with a positive feature map:
      phi(x) = elu(x) + 1

    Shapes (batch_first):
      x: (B, L, D)
    Internally we use heads with D = n_heads * d_head.

    Output: (B, L, D)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.eps     = eps

        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.to_out = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        # positive feature map for linear attention
        return F.elu(x, inplace=False) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # projections
        q = self.to_q(x)  # (B, L, D)
        k = self.to_k(x)
        v = self.to_v(x)

        # split heads
        q = q.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        v = v.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)

        # feature map
        q_phi = self._phi(q)                     # (B, H, L, Dh)
        k_phi = self._phi(k)                     # (B, H, L, Dh)

        # Precompute KV = (sum over time of k_phi^T @ v)
        # k_phi^T v: (B,H,Dh,L) @ (B,H,L,Dh) -> (B,H,Dh,Dh)
        KV = torch.matmul(k_phi.transpose(-2, -1), v)  # (B, H, Dh, Dh)

        # Normalizer: z = q_phi · (sum_t k_phi_t)
        k_sum = k_phi.sum(dim=2)                        # (B, H, Dh)
        # (B,H,L,Dh) * (B,H,1,Dh) -> sum over Dh -> (B,H,L,1)
        z = torch.sum(q_phi * k_sum.unsqueeze(2), dim=-1, keepdim=True)  # (B,H,L,1)

        # Numerator: q_phi @ KV -> (B,H,L,Dh)
        num = torch.matmul(q_phi, KV)                  # (B, H, L, Dh)

        # Normalize
        out = num / (z + self.eps)                     # (B, H, L, Dh)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        out = self.drop(self.to_out(out))                      # final projection + dropout
        return out


class TransformerFFN(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN block with LinearAttention instead of standard MHA.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = LinearAttention(d_model, n_heads, dropout=dropout, bias=False)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = TransformerFFN(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D)
        x_res = x
        x = self.norm1(x)
        attn_out = self.attn(x)                 # linear attention
        x = x_res + self.drop1(attn_out)        # residual 1

        x_res = x
        x = self.norm2(x)
        x = x_res + self.drop2(self.ffn(x))     # residual 2
        return x


class CodebookTransformer(nn.Module):
    """
    Same as before, but blocks now use LinearAttention internally.
    """
    def __init__(
        self,
        frozen_codebook_dim: int,
        out_dim: int,
        codebook_size: int,
        depth: int = 2,
        model_dim: int = 256,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_learned_pos: bool = True
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.in_proj  = nn.Linear(frozen_codebook_dim, model_dim, bias=False)
        self.pos_emb  = nn.Embedding(codebook_size, model_dim) if use_learned_pos else None
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(model_dim, n_heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.norm   = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, out_dim, bias=False)

        if not use_learned_pos:
            self.register_buffer("sinpos", self._build_sinusoidal_positions(codebook_size, model_dim), persistent=False)

    @staticmethod
    def _build_sinusoidal_positions(n: int, d: int):
        pos = torch.arange(n).float().unsqueeze(1)     # (n,1)
        i   = torch.arange(d).float().unsqueeze(0)     # (1,d)
        angles = pos / (10000 ** ((2 * (i // 2)) / d))
        emb = torch.zeros(n, d)
        emb[:, 0::2] = torch.sin(angles[:, 0::2])
        emb[:, 1::2] = torch.cos(angles[:, 1::2])
        return emb

    def forward(self, frozen_codebook: torch.Tensor) -> torch.Tensor:
        """
        frozen_codebook: (C, D_frozen)
        returns:         (C, out_dim)
        """
        C = frozen_codebook.shape[0]
        x = self.in_proj(frozen_codebook)           # (C, model_dim)
        x = x.unsqueeze(0)                          # (1, C, model_dim)

        if self.pos_emb is not None:
            idx = torch.arange(C, device=x.device)
            x = x + self.pos_emb(idx)[None, :, :]
        else:
            x = x + self.sinpos[:C, :][None, :, :]

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)                              # (1, C, model_dim)

        x = self.norm(x)
        x = self.out_proj(x)                        # (1, C, out_dim)
        return x.squeeze(0)                         # (C, out_dim)
