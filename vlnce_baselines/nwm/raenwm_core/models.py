# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed
from .RAE.src.stage2.models.model_utils import RMSNorm, NormAttention, SwiGLUFFN, VisionRotaryEmbeddingFast


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pack_cls_patch(cls: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    """Pack one CLS token and a spatial patch map into a token sequence."""

    if cls.dim() != 2:
        raise ValueError(f"cls must have shape [B, C], got {tuple(cls.shape)}")
    if patch.dim() != 4:
        raise ValueError(
            f"patch must have shape [B, C, H, W], got {tuple(patch.shape)}"
        )
    batch, channels, _height, _width = patch.shape
    if tuple(cls.shape) != (batch, channels):
        raise ValueError(
            f"cls shape {tuple(cls.shape)} must match patch batch/channels "
            f"{(batch, channels)}"
        )
    patch_tokens = patch.flatten(2).transpose(1, 2)
    return torch.cat([cls.unsqueeze(1), patch_tokens], dim=1)


def unpack_cls_patch(
    sequence: torch.Tensor,
    latent_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a native CLS+patch sequence back into CLS and patch tensors."""

    if sequence.dim() != 3:
        raise ValueError(
            "sequence must have shape [B, 1+H*W, C], got "
            f"{tuple(sequence.shape)}"
        )
    latent_size = int(latent_size)
    expected_tokens = latent_size * latent_size + 1
    if int(sequence.shape[1]) != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} tokens for latent_size={latent_size}, "
            f"got {int(sequence.shape[1])}"
        )
    cls = sequence[:, 0]
    patch = sequence[:, 1:].transpose(1, 2).contiguous()
    patch = patch.reshape(
        sequence.shape[0], sequence.shape[2], latent_size, latent_size
    )
    return cls, patch


def latent_to_patch_map(latent: torch.Tensor, latent_size: int) -> torch.Tensor:
    if latent.dim() == 4:
        return latent
    if latent.dim() == 3:
        return unpack_cls_patch(latent, latent_size)[1]
    raise ValueError(
        f"latent must be 3D sequence or 4D patch map, got {tuple(latent.shape)}"
    )


def make_latent_noise(
    batch_size: int,
    latent_dim: int,
    latent_size: int,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    predict_cls_token: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    kwargs = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if generator is not None:
        kwargs["generator"] = generator
    if bool(predict_cls_token):
        return torch.randn(
            int(batch_size),
            int(latent_size) * int(latent_size) + 1,
            int(latent_dim),
            **kwargs,
        )
    return torch.randn(
        int(batch_size),
        int(latent_dim),
        int(latent_size),
        int(latent_size),
        **kwargs,
    )


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    #调制是元素乘法
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    #gate调制方法
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        gate = gate.repeat_interleave(repeat, dim=1)
    return x * gate     #逐元素乘法


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class GaussianFourierEmbedding(nn.Module):
    #时间步编码器
    """
    Gaussian Fourier Embedding for timesteps, suitable for inputs in [0, 1] or [-0.5, 0.5].
    """
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)  #量输入 t 做 Gaussian Fourier features 映射，也就是把一个低维标量时间值变成一组高频正弦/余弦特征
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W  # stop gradient manually
        # t: (B, 1) or (B,) -> (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_proj = t * W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class ActionEmbedder(nn.Module):
    #动作编码器
    """
    Unified ActionEmbedder using GaussianFourierEmbedding for all continuous components.
    """
    def __init__(self, hidden_size, embedding_size=256):
        super().__init__()
        #把总维度分成三分
        hsize = hidden_size // 3

        self.x_emb = GaussianFourierEmbedding(hsize, embedding_size, scale=1.0) #一份x动作编码
        self.y_emb = GaussianFourierEmbedding(hsize, embedding_size, scale=1.0) #一份y动作编码

        self.angle_emb = GaussianFourierEmbedding(hidden_size - 2 * hsize, embedding_size, scale=1.0)   #一份角度动作编码

    def forward(self, xya):
        # xya: [Batch, 3] -> (x, y, angle)
        x = self.x_emb(xya[..., 0])
        y = self.y_emb(xya[..., 1])
        a = self.angle_emb(xya[..., 2])
        return torch.cat([x, y, a], dim=-1) #拼接成一个向量返回


#################################################################################
#                                 Core CDiT Model                               #
#################################################################################

class CDiTBlock(nn.Module):
    #原本的CDiT的标准实现
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_low_rank_adaln=False, use_qknorm: bool = False, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=bool(use_qknorm),
            use_rmsnorm=True,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.norm_cond = RMSNorm(hidden_size)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        if use_low_rank_adaln:
            rank_dim = min(hidden_size // 3, 512)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, rank_dim, bias=True),
                nn.SiLU(),
                nn.Linear(rank_dim, 11 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 11 * hidden_size, bias=True),
            )
        self.norm3 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.feat_rope = None

    def forward(self, x, c, x_cond, rope=None):
        #产生调制信息
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa) #进行一步调制
        x = x + gate_msa.unsqueeze(1) * self.attn(attn_in, rope=rope)   #进行gate调制

        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)  #对上下文信息进行调制
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(     #当前要预测的特征 x 去“看”上下文特征 x_cond，把有用的上下文信息吸收进来
            query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
            key=x_cond_norm,
            value=x_cond_norm,
            need_weights=False  #表示这次不需要返回注意力权重图
        )[0]    #当前 x 去从 x_cond 里找相关信息，就是只拿输出特征，再乘一个门控系数
        #经过mlp，再进行残差链接
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

        #，mlp_ratio = 4.0，use_qknorm = False
class DDTHeadBlock(nn.Module):
    def __init__(self, hidden_size,     #hidden_size = 2048
                 num_heads,             #head_num_heads = 16
                 mlp_ratio=4.0,         #mlp_ratio=4.0      FFN 隐层放大倍数 会先把特征维度扩到更高，再压回原维度
                 use_qknorm: bool = False,  #不做额外归一化。
                 **block_kwargs):

        super().__init__()
        self.norm1 = RMSNorm(hidden_size)   #定义两个 RMSNorm 归一化层。RMSNorm 通常只按均方根来缩放，不减均值。看当前向量整体有多大，再缩放到到更稳定的尺寸
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(  #自注意力层的封装，带 norm 选项的 attention
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,      #qkv的线性投影带bias
            qk_norm=bool(use_qknorm),
            use_rmsnorm=True,   #内部用 RMSNorm 变体
            **block_kwargs,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        #MLP 一样，两条分支，一条过silu，与另外一条逐元素相乘，太投影回原维度。SwiGLU 版前馈网络
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))

        #调制层
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  #平滑、非线性强、梯度更顺，一种激活函数
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope=None) -> torch.Tensor:
        if c.dim() == 2:
            c = c.unsqueeze(1)
        #调制出来6个条件向量
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        #先用内层的均值和方差调制，再用外层的gate调制
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=rope), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)   #gate调制就是逐个元素乘法调制
        return x


class DDTFinalLayer(nn.Module):
    #最终输出头层
    #hidden_size: 2048.patch_size=1,out_channels = 768
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        #elementwise_affine归一化之后，要不要再给每个特征维度乘一个可学习的缩放参数、再加一个可学习的偏置参数，当前只做 只做标准归一化
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        #从hiddensize投影成outchannels。
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        #adaLN类似于LN调制，但是他的两个参数是根据条件动作动态生成的。同一个网络，在不同条件下，用不同的归一化调制参数
        #“调制”本质上就是：让条件去控制主特征的缩放和偏移。调制的是 归一化后的 token 特征 x
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if c.dim() == 2:
            c = c.unsqueeze(1)
        #获取调制参数
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)   #进行调制，这个函数的作用就是通过shift和scale调制x
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    #depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=1,
        in_channels=768,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        head_width=None,
        head_depth=2,
        head_num_heads=16,
        use_low_rank_adaln_head=False,
        use_qknorm: bool = False,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels

        # DiT-DH head width: default to token channels (in_channels) to match RAE latent
        self.head_width = head_width if head_width is not None else in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.head_depth = head_depth
        self.head_num_heads = head_num_heads
        self.use_low_rank_adaln_head = bool(use_low_rank_adaln_head)

        #核心模块都在这里初始化
        self._init_core_components(
            input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio,
            use_qknorm=bool(use_qknorm)
        )

        #初始化核心数据
        self.initialize_weights()

    def _init_core_components(self, input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, use_qknorm: bool = False):

        # -------------------------
        # Encoder (base DiT): token dim = hidden_size
        # -------------------------
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)   #整理patch级别的输入
        # Use GaussianFourierEmbedding for flow matching time t in [0, 1]
        self.t_embedder = GaussianFourierEmbedding(hidden_size) #一个t的时间投影，两层简单的mlp，扩散/flow matching 里的时间步 t
        self.y_embedder = ActionEmbedder(hidden_size)
        # Use GaussianFourierEmbedding for relative time rel_t in [-0.5, 0.5]
        self.time_embedder = GaussianFourierEmbedding(hidden_size)  #轨迹里的相对时间 rel_t

        self.dyn_fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dyn_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        num_patches = self.x_embedder.num_patches
        #位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, num_patches, hidden_size),
            requires_grad=False
        )

        #CDiTblock
        self.blocks = nn.ModuleList([
            CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_low_rank_adaln=False, use_qknorm=use_qknorm)
            for _ in range(depth)
        ])

        half_head = hidden_size // num_heads // 2
        grid = int(self.x_embedder.num_patches ** 0.5)  #网格数量
        #VisionRotaryEmbeddingFast 给 attention 的 q/k 注入二维空间位置信息
        #给视觉 Transformer 的自注意力准备二维旋转位置编码，让模型知道每个 token 在图像网格中的空间位置
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head, pt_seq_len=grid)

        # -------------------------
        # Head (DDT / DiT-DH style):
        #   - NEW: head re-embeds raw x_t into head_width (query stream)
        #   - z_t from encoder (token-wise) projected to head_width as conditioning (key/value stream)
        # -------------------------
        self.x_embedder_head = PatchEmbed(input_size, patch_size, in_channels, self.head_width, bias=True)

        # Project encoder token features to head_width (token-wise condition)
        self.head_projector = nn.Linear(hidden_size, self.head_width, bias=True)

        # Keep these projections (not strictly needed, but keep for compatibility / cleanliness)
        #暂时没有使用。
        self.head_cond_projector = nn.Linear(hidden_size, self.head_width, bias=True)
        self.final_cond_proj = nn.Linear(hidden_size, self.head_width, bias=True)

        #两个block，对于输入，先归一化，再进行均值偏差调制，再经过自注意力，在gate调制，在经过SwiGLUFFN层（均值方差调制、gate调制）
        self.head_blocks = nn.ModuleList([
            DDTHeadBlock(self.head_width, self.head_num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm)
            for _ in range(self.head_depth)
        ])

        half_head2 = self.head_width // self.head_num_heads // 2
        grid2 = int(self.x_embedder.num_patches ** 0.5) #单边grid的数量
        self.head_feat_rope = VisionRotaryEmbeddingFast(dim=half_head2, pt_seq_len=grid2)   #head里面的位置编码

        self.final_layer = DDTFinalLayer(self.head_width, patch_size, self.out_channels)    #最终的输出头，一个投影层，一层条件调制

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):   #如果是线性层
                torch.nn.init.xavier_uniform_(module.weight)    #用 Xavier uniform initialization 的方式初始化，把权重填成某个均匀分布里的随机数，而不是全 0 或完全随便乱设
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)   #偏置初始化为0

        #apply(fn)是父函数方法，递归的访问每一层，对每一层调用指定的函数
        self.apply(_basic_init) #遍历整个模型，遇到 nn.Linear 就用 Xavier uniform 初始化权重，bias置0


        # Initialize (and freeze) pos_embed by sin-cos embedding:
        grid_size = int(self.x_embedder.num_patches ** 0.5) #网格尺寸
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)    #一个给每个空间位置分配的固定向量表，码的是每个 patch 在二维网格里的坐标 (h, w)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)    #把 numpy 数组转成 torch tensor，最前面再加一个维度
        pos_embed = pos_embed.repeat(self.context_size + 1, 1, 1)       #沿着0维度复制，四个上下文帧和一个当前帧
        self.pos_embed.data.copy_(pos_embed)                            #拷贝到模型参数中

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data    #取出 patch embedding 投影层的权重
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))   #按线性层视角做 Xavier 初始化
        nn.init.constant_(self.x_embedder.proj.bias, 0) #把偏置置零

        # Initialize head patch_embed too:初始化 head 里的 patch embedding
        w2 = self.x_embedder_head.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder_head.proj.bias, 0)

        # Initialize action embedding:  #初始化成均值为0,标准差为0.02的标准高斯分布
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.head_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:   #调制层和最终层初始化为0
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, num_patches, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        num_patches = x.shape[1]
        h = w = int(num_patches ** 0.5)
        assert h * w == num_patches, f"num_patches {num_patches} is not a square"
        # derive patch size dynamically from the last dimension
        patch_area_times_c = x.shape[2]
        if patch_area_times_c % c != 0:
            # handle swapped token layout: (N, Ctok, num_patches)
            if x.shape[2] == num_patches and x.shape[1] != num_patches:
                x = x.transpose(1, 2)
                patch_area_times_c = x.shape[2]
            # if still not divisible, raise
        # decide effective channel count
        if patch_area_times_c % c == 0:
            c_eff = c
        elif patch_area_times_c % (2 * c) == 0:
            c_eff = 2 * c
        else:
            raise AssertionError(f"channel mismatch: {patch_area_times_c} not divisible by {c} or {2*c}")
        p = int(((patch_area_times_c // c_eff)) ** 0.5)
        assert p * p * c_eff == patch_area_times_c, f"cannot factor last dim {patch_area_times_c} into p^2*c_eff with c_eff={c_eff}"

        x = x.reshape(x.shape[0], h, w, p, p, c_eff)
        x = x.permute(0, 5, 3, 4, 1, 2)  # (N, C_eff, p, p, H, W)
        imgs = x.reshape(x.shape[0], c_eff, h * p, w * p)
        if c_eff == 2 * c:
            imgs_mu, imgs_sigma = imgs.chunk(2, dim=1)
            return imgs_mu
        return imgs

    def _is_sequence_latent(self, value: torch.Tensor) -> bool:
        return value.dim() == 3

    def _embed_latent(
        self,
        value: torch.Tensor,
        embedder: PatchEmbed,
    ) -> torch.Tensor:
        if value.dim() == 4:
            return embedder(value)
        if value.dim() != 3:
            raise ValueError(
                "latent must be 3D sequence or 4D patch map, got "
                f"{tuple(value.shape)}"
            )
        if self.patch_size != 1:
            raise ValueError("3D CLS+patch latent sequence requires patch_size=1")
        expected_tokens = int(self.x_embedder.num_patches) + 1
        if int(value.shape[1]) != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} tokens, got {int(value.shape[1])}"
            )
        if int(value.shape[2]) != int(self.in_channels):
            raise ValueError(
                f"expected token dim {self.in_channels}, got {int(value.shape[2])}"
            )
        weight = embedder.proj.weight.flatten(1)
        return F.linear(value, weight, embedder.proj.bias)

    @staticmethod
    def _pos_embed_for(
        pos_embed: torch.Tensor,
        include_cls: bool,
    ) -> torch.Tensor:
        if not include_cls:
            return pos_embed
        cls_pos = torch.zeros(
            pos_embed.shape[0],
            1,
            pos_embed.shape[2],
            device=pos_embed.device,
            dtype=pos_embed.dtype,
        )
        return torch.cat([cls_pos, pos_embed], dim=1)

    def _apply_rope_with_optional_cls(self, value: torch.Tensor, rope):
        patch_tokens = int(self.x_embedder.num_patches)
        token_count = int(value.shape[-2])
        if token_count == patch_tokens:
            return rope(value)
        if token_count == patch_tokens + 1:
            cls = value[:, :, :1, :]
            patch = rope(value[:, :, 1:, :])
            return torch.cat([cls, patch], dim=-2)
        raise ValueError(
            f"RoPE expected {patch_tokens} or {patch_tokens + 1} tokens, "
            f"got {token_count}"
        )

    def _rope_for_tokens(self, rope, token_count: int):
        patch_tokens = int(self.x_embedder.num_patches)
        if int(token_count) not in (patch_tokens, patch_tokens + 1):
            raise ValueError(
                f"expected {patch_tokens} or {patch_tokens + 1} tokens, "
                f"got {int(token_count)}"
            )

        def wrapped(value: torch.Tensor) -> torch.Tensor:
            return self._apply_rope_with_optional_cls(value, rope)

        return wrapped

    def forward(self,
                x,
                t,
                y,
                x_cond,
                rel_t
                ):
        sequence_output = self._is_sequence_latent(x)
        # Keep raw x_t for DDT head (RAE-style: head re-embeds raw x_t)
        x_raw = x   #混合过噪声的图像数据

        # Prepare encoder token inputs and global condition
        #把所有的输入信息都整理好：#最终输出的是带噪声的x、上下文序列、条件向量、时间编码
        x_tok, x_cond_tok, c, t_emb = self._prepare_inputs(x, t, y, x_cond, rel_t)

        # -------------------------
        # Encoder (base DiT): identical to your original design
        # -------------------------
        #得到dit的原始输出zi
        z = self._process_blocks(x_tok, c, x_cond_tok)

        # RAE-style: fuse time embedding into token features before head (token-wise)
        # z_t := SiLU(z + t_emb)
        z = F.silu(z + t_emb[:, None, :])   #给每一个维度加上时间编码，再经过一个激活函数

        # Project token-wise z to head width as conditioning stream
        z_head = self.head_projector(z)  # (B, L, head_width)

        # -------------------------
        # DDT Head (DiT-DH style):
        #   Query stream: re-embed raw x_t -> x_head
        #   Key/Value stream: z_head (token-wise condition)
        # -------------------------
        x_head = self._embed_latent(x_raw, self.x_embedder_head)  # (B, L, head_width)  (no APE in head; RoPE inside attention)
        head_rope = self._rope_for_tokens(self.head_feat_rope, x_head.shape[1])

        for blk in self.head_blocks:    #经过头
            x_head = blk(x_head, z_head, rope=head_rope)

        x_out = self.final_layer(x_head, z_head)    #最终层mlp
        if sequence_output:
            return x_out

        num_patches = self.x_embedder.num_patches
        N = x_out.shape[0]
        if x_out.dim() != 3:
            x_out = x_out.reshape(N, num_patches, -1)
        x_out = self.unpatchify(x_out)
        return x_out    #把返回的token序列重新拼回二维特征图，[N, 768, 16, 16]

    def _prepare_inputs(self, x, t, y, x_cond, rel_t):
        is_sequence = self._is_sequence_latent(x)
        if is_sequence:
            if x_cond.dim() != 4:
                raise ValueError(
                    "sequence latent x_cond must have shape [B,T,L,C], got "
                    f"{tuple(x_cond.shape)}"
                )
        elif x_cond.dim() != 5:
            raise ValueError(
                "patch latent x_cond must have shape [B,T,C,H,W], got "
                f"{tuple(x_cond.shape)}"
            )

        #x_cond是多帧上下文信息
        #[N, 768, 16, 16] -> [N, 256, 768]再加上位置编码，x_embedder其实就是一个卷积和一个展平
        x_pos = self._pos_embed_for(
            self.pos_embed[self.context_size:], is_sequence
        )
        x = self._embed_latent(x, self.x_embedder) + x_pos

        #4 张上下文 latent 图”变成“4 组上下文 token”，并标明它们各自是第几帧
        cond_pos = self._pos_embed_for(
            self.pos_embed[:self.context_size], is_sequence
        )
        x_cond = self._embed_latent(
            x_cond.flatten(0, 1), self.x_embedder
        ).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + cond_pos
        x_cond = x_cond.flatten(1, 2)

        t_emb = self.t_embedder(t[..., None])   #把时间步进行编码，经过正余弦编码之后，再通过一个mlp
        y_emb = self.y_embedder(y)              #将动作进行编码，分别编码x方向，y方向和角度变化
        span_emb = self.time_embedder(rel_t[..., None]) #轨迹里的相对时间进行编码

        c_dyn = self.dyn_fuse(torch.cat([y_emb, span_emb], dim=-1))     #把时间和动作条件拼接成一个大向量，再经过一个小的mlp投影到hiddensize
        gate = torch.sigmoid(self.dyn_gate(t_emb))
        c = t_emb + gate * c_dyn        #这一段就是论文中的调制信息生成方式

        return x, x_cond, c, t_emb  #最终输出的是带噪声的x、上下文序列、条件向量、时间编码

    def _process_blocks(self, x, c, x_cond):
        rope = self._rope_for_tokens(self.feat_rope, x.shape[1])
        for block in self.blocks:
            x = block(x, c, x_cond, rope=rope)    #rope=self.feat_rope是二维旋转位置编码
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first #展开成二维网格
    grid = np.stack(grid, axis=0)   #横坐标网格和纵坐标网格堆成一个三维张量

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2,
    'CDiT-L/2':  CDiT_L_2,
    'CDiT-B/2':  CDiT_B_2,
    'CDiT-S/2':  CDiT_S_2
}
