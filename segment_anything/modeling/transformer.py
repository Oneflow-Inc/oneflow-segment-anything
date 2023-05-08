# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock

# 这个 TwoWayTransformer 类实现了双向transformer解码器。
class TwoWayTransformer(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - depth: transformer 的层数
    #     - embedding_dim: 输入 embedding 的通道维度
    #     - num_heads: 多头注意力的头数
    #     - mlp_dim: MLP 块内部的通道维度
    #     - activation: MLP 块使用的激活函数
    #     - attention_downsample_rate: 注意力下采样率
    # 2. 记录 depth、embedding_dim、num_heads 和 mlp_dim。
    # 3. 定义 layers 为 nn.ModuleList, 包含 depth 个 TwoWayAttentionBlock。
    # 4. 定义 final_attn_token_to_image 为从点到图像的注意力层。
    # 5. 定义 norm_final_attn 为 final_attn_token_to_image 的 LayerNorm。

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    # forward 方法:
    # 1. 输入参数:
    #     - image_embedding: 要处理的图像,形状为 B x embedding_dim x h x w
    #     - image_pe: 与 image_embedding 形状相同的位置编码
    #     - point_embedding: 要添加到查询点的 embedding ,形状为 B x N_points x embedding_dim
    # 2. 将 image_embedding 变形为 B x HW x C, image_pe 相应变形。
    # 3. 将 queries 初始化为 point_embedding, keys 初始化为 image_embedding。
    # 4. 对 queries 和 keys 重复使用 layers 中的 TwoWayAttentionBlock。
    # 5. 应用 final_attn_token_to_image 从 points 到 image 的注意力。
    # 6. 使用 norm_final_attn 规范化 queries。
    # 7. 返回 queries 和 keys。

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

# 这个 TwoWayAttentionBlock 类实现了 transformer 块,包含四个层:
# 1. 稀疏输入的自注意力
# 2. 稀疏输入到密集输入的交叉注意力
# 3. 稀疏输入的 MLP 块
# 4. 密集输入到稀疏输入的交叉注意力
class TwoWayAttentionBlock(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - embedding_dim: embedding 的通道维度
    #     - num_heads: 注意力层中的头数
    #     - mlp_dim: MLP 块的隐藏维度
    #     - activation: MLP 块的激活函数
    #     - attention_downsample_rate: 注意力下采样率
    #     - skip_first_layer_pe: 是否跳过第一层的位置编码
    # 2. 定义 self_attn 为自注意力层。
    # 3. 定义 norm1 为 self_attn 的 LayerNorm。
    # 4. 定义 cross_attn_token_to_image 为从 token 到 image 的交叉注意力层。
    # 5. 定义 norm2 为 cross_attn_token_to_image 的 LayerNorm。
    # 6. 定义 mlp 为 MLP 块。
    # 7. 定义 norm3 为 mlp 的 LayerNorm。 
    # 8. 定义 norm4 为 LayerNorm。
    # 9. 定义 cross_attn_image_to_token 为从 image 到 token 的交叉注意力层。
    # 10. 记录 skip_first_layer_pe。
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    # forward 方法:
    # 1. 输入:
    #     - queries: 稀疏输入,即点输入
    #     - query_pe: query 的位置编码
    #     - keys: 密集输入,即图像输入
    #     - key_pe: key 的位置编码
    # 2. 如果 skip_first_layer_pe 为 True，则 qkv 都来自 queries
    # 3. 使用 self_attn 计算 queries 的自注意力。
    # 4. 通过 norm1 规范化 queries。
    # 5. 使用 cross_attn_token_to_image 计算 queries 到 keys 的注意力。
    # 6. 通过 norm2 规范化 queries。 
    # 7. 使用 mlp 更新 queries。
    # 8. 通过 norm3 规范化 queries。
    # 9. 使用 cross_attn_image_to_token 计算 keys 到 queries 的注意力。
    # 10. 通过 norm4 规范化 queries。
    # 11. 返回 queries 和 keys。
    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

# 这个Attention类实现了带下采样的注意力机制。
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """
    # __init__ 方法:
    # 1. 输入参数:
    #     - embedding_dim: embedding 的维度
    #     - num_heads: 多头注意力的头数
    #     - downsample_rate: 下采样率
    # 2. 计算 internal_dim 为 embedding_dim 除以 downsample_rate。
    # 3. 确保 internal_dim 可以被 num_heads 整除。
    # 4. 定义 q_proj、k_proj 和 v_proj为 输入的投影层,将 embedding_dim 映射到 internal_dim。
    # 5. 定义 out_proj 为输出的投影层,将 internal_dim 映射回 embedding_dim。
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    # _separate_heads方法:
    # 1. 将 x 分离为 num_heads 个头, x 的形状变为 b x n x num_heads x c // num_heads。
    # 2. 交换第二和第三维, x 的形状变为 b x num_heads x n x c // num_heads。
    # 3. 返回x。
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    # _recombine_heads 方法:
    # 1. x 的形状为 b x num_heads x n x c // num_heads。
    # 2. 交换第二和第三维,x的形状变为 b x n x num_heads x c // num_heads。 
    # 3. 将 x 变形为 b x n x num_heads * c // num_heads。
    # 4. 返回 x。
    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C
    
    # forward方法:
    # 1. 对 q、k 和 v 使用 q_proj、k_proj 和 v_proj 进行投影,将 embedding_dim 映射到 internal_dim。
    # 2. 使用 _separate_heads 将 q、k 和 v 分离为 num_heads 个头。
    # 3. 计算 attn 为 q 和 k 的点积,除以 c_per_head 开根号,再使用 softmax 归一化。
    # 4. 使用 attn 和 v 计算 out。
    # 5. 使用 _recombine_heads 重新组合出 num_heads 个头。
    # 6. 使用 out_proj 将 out 投影回 embedding_dim。
    # 7. 返回 out。
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
