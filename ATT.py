import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce, repeat
from torch.nn import Softmax
from torch import einsum
from typing import Optional

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class ECA1D(nn.Module):
    def __init__(self, dim, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(dim, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x):
        # x shape: (B, C, L)
        y = self.avg(x)
        y = y.squeeze(-1)
        y = y.unsqueeze(1)
        y = self.conv(y)
        y = y.squeeze(1).unsqueeze(-1)
        return x * y.sigmoid()
        
class SE1D(nn.Module):
    def __init__(self, channel, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class AdaptiveFrequencyEnhancer(nn.Module):
    """自适应频域增强模块"""
    def __init__(self, dim, num_heads=4, dropout=0.1, freq_ratio=0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        ...
        self.output_fusion = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, L = x.shape

        x_proj = self.input_proj(x)
        x_fft = torch.fft.rfft(x_proj, dim=-1, norm='ortho')  # [B, C, F]
        freq_len = x_fft.shape[-1]
        ...
        output = self.output_fusion(torch.cat([x, enhanced_time], dim=1))
        return x + self.gate * output
