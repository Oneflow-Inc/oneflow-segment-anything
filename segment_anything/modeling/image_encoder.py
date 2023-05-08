# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
# 这个代码定义了 SAM 的图像编码器 ImageEncoderViT。它包含以下主要部分:
# 1. patch_embed: 这是 ViT 的 patch embedding 层,用于将输入图像划分为 patch,并获得 patch 的 embedding。
# 2. pos_embed: 这是 ViT的绝对位置 embedding,用于为每个patch提供位置信息。
# 3. blocks: 这是 ViT 的 transformer encoder 块的列表,每个块包含多头自注意力层和前馈神经网络。
# 4. neck: 这是图像编码器的“颈部”,包含几个卷积层和 LayerNorm 层,用于从 transformer encoder 块的输出中提取特征。
# 5. forward(): 这是图像编码器的前向传播过程。首先通过 patch_embed 层获得 patch embedding, 然后加上 pos_embed。
# 接着,patch embedding通过transformer encoder块。最后, neck 层从 transformer encoder 块的输出中提取特征。
# 所以,这个 ImageEncoderViT 类定义了 SAM 的图像编码器,它基于 ViT,包含 patch embedding、位置 embedding、
# transformer encoder块以及 neck, 可以从输入图像中提取特征。

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

# 这个 Block 类实现了 transformer block, 可以选择使用全局注意力或局部窗口注意力,同时包含残差连接。它包含:
# __init__方法:
# 1. 输入参数:
#     - dim: 输入通道数
#     - num_heads: 注意力头数
#     - mlp_ratio: mlp 隐藏层与输入 embedding 维度的比例
#     - qkv_bias: 是否为 query、key、value 添加偏置
#     - norm_layer: 归一化层
#     - act_layer: 激活层
#     - use_rel_pos: 是否使用相对位置 embedding
#     - rel_pos_zero_init: 是否将相对位置 embedding 初始化为 0
#     - window_size: 窗口注意力的窗口大小,如果为 0 则使用全局注意力
#     - input_size: 计算相对位置 embedding 大小所需的输入分辨率
# 2. 实例化第 1 次和第 2 次归一化层 norm1 和 norm2。
# 3. 实例化 Attention 层和 MLPBlock 层。Attention 层的输入大小根据是否使用窗口注意力进行了调整。
# 4. 记录窗口注意力的窗口大小 window_size。
# forward方法:
# 1. 提取 shortcut 并对 x 进行第 1 次归一化。 
# 2. 如果使用窗口注意力, 则调用 window_partition 对 x 进行窗口划分。
# 3. 将 x 输入 Attention 层。
# 4. 如果使用窗口注意力,则调用 window_unpartition 对 x 进行窗口反划分。
# 5. x = shortcut + x,实现第 1 次残差连接。
# 6. x = x + mlp(norm2(x)),实现第 2 次残差连接和 MLPBlock。
# 7. 返回最终的 x。
# 所以,这个 Block 类实现了带有可选的窗口注意力和双残差连接的transformer block。
# 窗口注意力可以更好地建模局部结构,双残差连接可以提高梯度流动,都是transformer结构的重要改进。
# 这个 Block 类实现了 transformer 的关键组成部分,同时提供了窗口注意力和残差连接等重要变体,可以显著提高其表现力和泛化能力。
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

# 这个Attention类实现了多头注意力机制,可以加入相对位置 embedding。它包含:
# __init__方法:
# 1. 输入参数:
#     - dim: 输入通道数
#     - num_heads: 注意力头数
#     - qkv_bias: 是否为查询、键、值添加偏置
#     - use_rel_pos: 是否使用相对位置 embedding
#     - rel_pos_zero_init: 是否将相对位置 embedding 初始化为0
#     - input_size: 计算相对位置 embedding 大小所需的输入分辨率
# 2. 计算每个注意力头的维度 head_dim。
# 3. 实例化 self.qkv和 输出投影 self.proj。
# 4. 如果使用相对位置 embedding, 则初始化 rel_pos_h 和 rel_pos_w。
# forward方法:
# 1. 从输入 x 中提取批次大小 B、高度 H、宽度 W 和通道数 C。
# 2. 计算 qkv,形状为 (3, B, nHead, H * W, C), 包含 query、key 和 value。
# 3. 提取 q、 k 和 v, 形状为 (B * nHead, H * W, C)。
# 4. 计算注意力图 attn,形状为 (B * nHead, H * W, H * W)。
# 5. 如果使用相对位置 embedding, 则调用 add_decomposed_rel_pos 函数将其加入 attn。
# 6. 对 attn 进行 softmax 归一化。
# 7. 计算输出 x , (attn @ v), 形状为 (B, nHead, H, W, C), 然后合并注意力头, 形状为(B, H, W, C)。
# 8. 对 x 进行投影, 返回最终的输出。
# 所以,这个 Attention 类实现了带有相对位置 embedding 的多头注意力机制。
# 它可以高效地建模图像和视频等二维结构数据,是 transformer 在这些领域得到广泛应用的关键。
# 这个 Attention 类提供了相对位置 embedding 和多头注意力机制的实现,
# 是理解 transformer 在图像和视频建模中的重要组成部分。
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


# 这个 window_partition 函数的作用是将输入张量划分为非重叠的窗口。它包含:
# 1. 输入参数:
    # - x: 输入的张量,形状为 [B, H, W, C]
    # - window_size: 窗口大小
# 2. 首先计算输入需要 padding 的高度和宽度,将x进行padding。
# 3. 然后将 x 的形状变化为 [B, Hp//window_size, window_size, Wp//window_size, window_size, C],
# 表示将图像划分为 Hp//window_size * Wp//window_size 个 window_size * window_size 的 patch。
# 4. 最后,通过 permute 和 view 操作,得到 windows 的形状为 [B * num_windows, window_size, window_size, C],
# 表示将所有 patch 打平, num_windows 是 patch 的总数
# 5. 返回windows和原来的高度和宽度(包含padding)Hp和Wp。
# 所以,这个 window_partition 函数的作用是,将输入的图像划分为 window_size * window_size 的 patch,
# 并将所有的 patch 打平, 输出可以输入到 transformer encoder 中的 token 序列。
# 这个函数实现了将二维图像转化为一维 token 序列的过程,是 transformer 用于处理图像的一个关键步骤。
# 通过这个函数,图像可以被 transformer encoder 所处理,就像处理文本序列一样。
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

# 这个 window_unpartition 函数的作用是将 window_partition 函数的输出进行反划分, 恢复成原始的图像形状。它包含:
# 1. 输入参数:
    # - windows: window_partition的输出,形状为 [B * num_windows, window_size, window_size, C]
    # - window_size: 窗口大小
    # - pad_hw: padding后的高度和宽度 (Hp, Wp)
    # - hw: padding前的原始高度和宽度 (H, W)
# 2. 首先根据窗口大小和 padding 后的 hw 计算原始的 batch_size B。
# 3. 然后将 windows 的形状变回 [B, Hp//window_size, Wp//window_size, window_size, window_size, C], 表示每个patch的位置。
# 4. 接着通过permute和view操作,得到x的形状为 [B, Hp, Wp, C], 恢复成图像的形状。
# 5. 最后,如果进行了padding,则截取x到原始的高度H和宽度W。
# 6. 返回恢复后的图像x。
# 所以,这个 window_unpartition 函数的作用是将通过 window_partition 函数得到的 patch 序列恢复成原始的图像。
# 它实现了从一维 patch token 序列到二维图像的反过程。
# 这个函数与 window_partition 函数相反,使得 transformer 能够最终从 patch token 序列恢复成图像,完成对图像的建模。
# 总的来说,这个 window_unpartition 函数实现了从 patch token 序列恢复成原始图像的过程,与 window_partition 函数相对应,
# 是使得 transformer 可以处理图像的另一个关键步骤
def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

# 这个 get_rel_pos 函数的作用是根据 query 和 key 的相对位置获取相对位置 embedding。它包含:
# 1. 输入参数:
#     - q_size: query 的大小
#     - k_size: key 的大小
#     - rel_pos: 相对位置 embedding, 形状为[L, C]
# 2. 首先计算最大的相对距离 max_rel_dist, 它等于 query 和 key 大小的 2 倍减 1。
# 3. 如果相对位置 embedding 的长度小于 max_rel_dist, 则通过线性插值将其调整到 max_rel_dist 的长度。
# 4. 如果 q_size 和 k_size 不同, 则将 q_size 和 k_size 的坐标按比例缩放,使它们之间的相对距离保持不变。
# 5. 根据调整后的 q_size 和 k_size 坐标计算相对坐标 relative_coords。
# 6. 根据 relative_coords 从 rel_pos_resized 中提取相对位置 embedding。
# 7. 返回提取出的相对位置 embedding。
# 所以,这个 get_rel_pos 函数的主要作用是,当 query 和 key 的大小不同时,根据它们的相对位置关系提取相应的相对位置 embedding。
# 它实现了相对位置 embedding 的可变长度和可缩放性。
# 这个函数使得相对位置 embedding 可以用于 query 和 key 大小不同的 attention 中,是相对位置表示的一个关键步骤。
# 总的来说,这个 get_rel_pos 函数实现了根据 query 和 key 的相对位置关系提取相应相对位置 embedding 的过程。
# 它提供了相对位置 embedding 的可变长度和可缩放性,使其可以支持不同的 query 和 key 大小,从而应用到更加灵活的 attention 机制中。
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

# 这个 add_decomposed_rel_pos 函数的作用是根据 query q 和 key k 的空间尺寸, 添加分解的相对位置 embedding 到注意力图 attn 中。它包含:
# 1. 输入参数:
#     - attn: 注意力图,形状为 [B, q_h * q_w, k_h * k_w]
#     - q: 查询 q,形状为 [B, q_h * q_w, C]
#     - rel_pos_h: 高度轴的相对位置 embedding, 形状为[Lh, C]
#     - rel_pos_w: 宽度轴的相对位置 embedding, 形状为[Lw, C]
#     - q_size: 查询 q的空间尺寸 (q_h, q_w)
#     - k_size: 键 k的空间尺寸 (k_h, k_w)
# 2. 从 q_size 和 k_size 中提取高度 q_h、宽度 q_w 以及高度 k_h、宽度 k_w。
# 3. 调用 get_rel_pos 函数获取高度轴 Rh 和宽度轴 Rw 的相对位置 embedding。
# 4. 重塑 q 为 [B, q_h, q_w, C]。
# 5. 计算高度轴 rel_h 和宽度轴 rel_w 的相对位置图, 形状为 [B, q_h, q_w, k_h] 和 [B, q_h, q_w, k_w]。
# 6. 将 attn 的形状变为 [B, q_h, q_w, k_h, k_w], 并加上 rel_h 和 rel_w。
# 7. 将 attn 的形状变回 [B, q_h * q_w, k_h * k_w]。
# 8. 返回加了相对位置 embedding 的 attn。
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


# 这个 PatchEmbed 类定义了 ViT 的 patch embedding 层。它包含:
# 1. __init__: 初始化,设置卷积层的 kernel size、stride、padding以 及输入通道数和 embedding 维度。
# 2. proj: 这是一个卷积层,用于将输入图像划分为 patch, 并获得每个 patch 的 embedding。
# 3. forward: 前向传播过程。首先通过 proj 卷积层获得 patch embedding ,然后将维度从 [B, C, H, W] 转置成 [B, H, W, C]。

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
