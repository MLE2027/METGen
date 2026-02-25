import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path
from random import random
from functools import partial
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
from collections import namedtuple
from multiprocessing import cpu_count
from torch import einsum
from ATT import ECA1D, SE1D, LinearAttention, Attention, AdaptiveFrequencyEnhancer

# ---------------- 工具箱 ---------------- #
def exists(x): return x is not None
def default(val, d): return val if exists(val) else (d() if callable(d) else d)

# ---------------- 小模块 ---------------- #
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
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn, self.norm = fn, LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=[1,2], keepdim=True)
        var  = weight.var(dim=[1,2], unbiased=False, keepdim=True)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# ---------------- 位置编码 ---------------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)
    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# ---------------- 基础卷积块 ---------------- #
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x); x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = torch.cat(tuple(filter(exists, (time_emb, class_emb))), dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        WeightStandardizedConv1d(dim, default(dim_out, dim), 3, padding=1)
    )
def Downsample(dim, dim_out=None):
    return WeightStandardizedConv1d(dim, default(dim_out, dim), 4, 2, 1)

# ---------------- 主要模块 ---------------- #
class PolyTrend(nn.Module):
    def __init__(self, dim, degree=3):
        super().__init__()
        self.degree = degree
        self.weight = nn.Parameter(torch.randn(dim, degree + 1) * 0.01)
        ...
    def forward(self, x):
        B, C, L = x.shape
        t = self.t[:, :, :L]
        ...
        trend = torch.einsum('cd,dol->ocl', self.weight, basis)
        return trend.expand(B, -1, -1)            # [B,C,L]


class FFTPeak(nn.Module):
    def __init__(self, dim, K=5, freq_threshold=3):
        super().__init__()
        ...
    def forward(self, x):
        B, C, L = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        ...
        peak = torch.fft.irfft(peak_freq, n=L, dim=-1)
        return peak


class TPEBlock1D(nn.Module):
    """Trend + Peak + Error 分解"""
    def __init__(self, dim, kernel_size=3, poly_degree=3, fft_K=5, freq_th=3):
        super().__init__()
        self.trend_extractor = PolyTrend(dim, poly_degree)
        ...
        self.afe = AdaptiveFrequencyEnhancer(dim)

    def forward(self, x):
        # x: [B, dim, L]
        trend = self.trend_extractor(x)
        peak = self.peak_pool(x)
        ... 
        out = self.fuse(out)
        return x + out * self.gate(out)   # 门控残差


class DCN1D(nn.Module):
    """Dilated Convolutional Networks"""
    def __init__(self, dim, dilations=[1, 2, 4]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                WeightStandardizedConv1d(in_channels=dim, out_channels=dim, kernel_size=3,
                                         dilation=d, padding=d, groups=dim),
                nn.BatchNorm1d(dim), nn.GELU()
            ) for d in dilations
        ])
        ...
        self.scale_weight = nn.Parameter(torch.ones(len(dilations), 1, 1), requires_grad=True)
    def forward(self, x):
        self.scale_weight = self.scale_weight.to(x.device)
        ...
        fused = self.fuse(fused)
        return fused + x

# ---------------- 模型骨架 ---------------- #
class MET_Block(nn.Module):
    """Multi-scale Efficient Temporal Block"""
    def __init__(self, dim, fpn_dilations=[1, 2, 4], se_ratio=16, tpe_ks=7,
                 poly_degree=3, fft_K=5, freq_th=3):
        super().__init__()
        self.fpn = DCN1D(dim, dilations=fpn_dilations)
        ...
    def forward(self, x):
        x = self.fpn(x)
        ...
        x = self.tpe(x)
        return x


class METGen(nn.Module):
    def __init__(
        self,
        dim,
        cond_drop_prob=0.5,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        arch='att',
        use_block=True,
        fpn_dilations=[1, 2, 4],
        se_ratio=16,
        tpe_ks=7,
        poly_degree=3,
        fft_K=6,
        freq_th=3
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.arch, self.use_block = arch, use_block
        self.channels = channels
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        classes_dim = dim * 4
        self.classes_emb = nn.Linear(1, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim), nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # Encoder
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            down_modules = [
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
            ]
            if use_block:
                down_modules.append(
                    MET_Block(
                        dim=dim_in, fpn_dilations=fpn_dilations, se_ratio=se_ratio,
                        tpe_ks=tpe_ks, poly_degree=poly_degree, fft_K=fft_K, freq_th=freq_th
                    )
                )
            down_modules.append(
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            )
            self.downs.append(nn.ModuleList(down_modules))

        # Mid
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_combined = MET_Block(
            dim=mid_dim, fpn_dilations=fpn_dilations, se_ratio=se_ratio,
            tpe_ks=tpe_ks, poly_degree=poly_degree, fft_K=fft_K, freq_th=freq_th) if use_block else nn.Identity()
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            up_modules = [
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
            ]
            if use_block:
                up_modules.append(
                    MET_Block(
                        dim=dim_out, fpn_dilations=fpn_dilations, se_ratio=se_ratio,
                        tpe_ks=tpe_ks, poly_degree=poly_degree, fft_K=fft_K, freq_th=freq_th
                    )
                )
            up_modules.append(
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            )
            self.ups.append(nn.ModuleList(up_modules))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, classes, cond_drop_prob=None):
        b, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        classes_emb = self.classes_emb(classes.float())
        if cond_drop_prob > 0:
            keep_mask = torch.zeros(b, device=device).float().uniform_(0, 1) < (1 - cond_drop_prob)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=b)
            classes_emb = torch.where(keep_mask.unsqueeze(1), classes_emb, null_classes_emb)
        c = self.classes_mlp(classes_emb)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        # Encoder
        for ind, (block1, block2, *rest_modules) in enumerate(self.downs):
            downsample = rest_modules[-1]
            if self.use_block:
                combined_block = rest_modules[0]
            else:
                combined_block = None

            x = block1(x, t, c)
            h.append(x)
            x = block2(x, t, c)
            h.append(x)
            if self.use_block:
                x = combined_block(x)
            x = downsample(x)

        # Mid
        x = self.mid_block1(x, t, c)
        if self.arch == 'original':
            x = x
        elif self.arch == 'att':
            x = self.mid_attn(x)
        if self.use_block:
            x = self.mid_combined(x)
        x = self.mid_block2(x, t, c)

        # Decoder
        for ind, (block1, block2, *rest_modules) in enumerate(self.ups):
            upsample = rest_modules[-1]
            if self.use_block:
                combined_block = rest_modules[0]
            else:
                combined_block = None

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            if self.use_block:
                x = combined_block(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
