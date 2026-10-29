# 代码实现参考模板，参考规则优先级如下
# 1. 代码简洁
# 2. 与pytorch官方实现尽量靠近

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(size):
    """
    生成自回归掩码，屏蔽当前位置之后的未来信息。

    这个掩码常用于解码器自注意力。对于位置 i，只允许它看到
    自己以及它之前的位置，不能看到未来位置。

    Args:
        size: 目标序列长度 T。

    Returns:
        布尔掩码，形状为 (1, T, T)。
        值为 True 表示该位置可见，False 表示该位置被屏蔽。
    """
    future_mask = torch.triu(
        torch.ones(1, size, size, dtype=torch.bool),
        diagonal=1,
    )
    return ~future_mask


class LayerNorm(nn.Module):
    """
    层归一化模块，对最后一个维度做标准化。

    Args:
        features: 最后一个维度的大小 D。
        eps: 数值稳定性常数。
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (..., D)。

        Returns:
            归一化后的张量，形状与 x 相同。
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    """
    正弦位置编码。

    Args:
        d_model: 位置编码维度 D。
        dropout: dropout 比例。
        max_len: 支持的最大序列长度。
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: 输入嵌入，形状为 (B, L, D)。

        Returns:
            加入位置编码后的结果，形状为 (B, L, D)。
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    """
    词嵌入层，并按论文实现乘以 sqrt(d_model)。

    Args:
        d_model: 嵌入维度 D。
        vocab: 词表大小 V。
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: token 索引张量，形状为 (B, L)。

        Returns:
            嵌入结果，形状为 (B, L, D)。
        """
        token_embeddings = self.lut(x)
        scale = math.sqrt(self.d_model)
        return token_embeddings * scale


class Generator(nn.Module):
	"""解码器输出投影层。"""

	def __init__(self, d_model: int, vocab_size: int):
		super().__init__()
		self.proj = nn.Linear(d_model, vocab_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return F.log_softmax(self.proj(x), dim=-1)


def attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力。

    Args:
        query: 查询张量，形状为 (B, H, Lq, Dh)。
        key: 键张量，形状为 (B, H, Lk, Dh)。
        value: 值张量，形状为 (B, H, Lk, Dh)。
        mask: 可选掩码。
            编码器自注意力中形状通常为 (B, 1, 1, S)。
            解码器源目标注意力中形状通常为 (B, 1, 1, S)。
            解码器自注意力中形状通常为 (B, 1, T, T)。
        dropout: 可选 dropout 模块。

    Returns:
        output: 注意力输出，形状为 (B, H, Lq, Dh)。
        p_attn: 注意力权重，形状为 (B, H, Lq, Lk)。
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # (B, H, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    out = torch.matmul(p_attn, value) # (B, H, Lq, Dh)
    return out, p_attn



class MultiHeadAttention(nn.Module):
    """
    多头注意力模块。
    Args:
        h: 注意力头数 H。
        d_model: 模型隐藏维度 D。
        dropout: dropout 比例。
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: 查询张量，形状为 (B, Lq, D)。
            key: 键张量，形状为 (B, Lk, D)。
            value: 值张量，形状为 (B, Lk, D)。
            mask: 可选掩码。
                编码器自注意力中通常为 (B, 1, S)。
                解码器源目标注意力中通常为 (B, 1, S)。
                解码器自注意力中通常为 (B, T, T)。

        Returns:
            多头拼接并线性映射后的结果，形状为 (B, Lq, D)。
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # (B, Lq, D) -> (B, Lq, H, d_k) -> (B, H, Lq, d_k)
        key = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # (B, Lk, D) -> (B, Lk, H, d_k) -> (B, H, Lk, d_k)
        value = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # (B, Lk, D) -> (B, Lk, H, d_k) -> (B, H, Lk, d_k)

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        ) # (B, H, Lq, d_k), (B, H, Lq, Lk)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)


class PositionwiseFeedForward(nn.Module):
    """
    位置逐位前馈网络。

    Args:
        d_model: 输入和输出维度 D。
        d_ff: 中间隐藏层维度 Dff。
        dropout: dropout 比例。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (B, L, D)。

        Returns:
            前馈网络输出，形状为 (B, L, D)。
        """
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderLayer(nn.Module):
    """
    单个编码器层，由自注意力子层和前馈网络子层组成。

    为了更直观地展示数据流，这里直接在层内展开
    "先归一化 -> 子层 -> dropout -> 残差相加"。

    Args:
        size: 隐藏维度 D。
        self_attn: 自注意力模块，输入输出形状为 (B, S, D)。
        feed_forward: 前馈网络模块，输入输出形状为 (B, S, D)。
        dropout: dropout 比例。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask):
        """
        Args:
            x: 编码器输入，形状为 (B, S, D)。
            mask: 源序列掩码，形状为 (B, 1, S)。

        Returns:
            编码器层输出，形状为 (B, S, D)。
        """
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, mask)) # 第一个子层：自注意力
        x = x + self.dropout2(self.feed_forward(self.norm2(x))) # 第二个子层：前馈网络
        return x


class Encoder(nn.Module):
    """
    由 N 个编码器层堆叠而成的编码器。

    Args:
        layer: 单个编码器层，输入输出形状均为 (B, S, D)。
        N: 编码器层数。
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Args:
            x: 输入表示，形状为 (B, S, D)。
            mask: 源序列掩码，形状为 (B, 1, S)。

        Returns:
            编码器输出，形状为 (B, S, D)。
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    单个解码器层，由目标端自注意力、源目标注意力和前馈网络组成。

    Args:
        size: 隐藏维度 D。
        self_attn: 目标端自注意力模块，输入输出形状为 (B, T, D)。
        src_attn: 源目标注意力模块，输入为 (B, T, D) 与 (B, S, D)，输出为 (B, T, D)。
        feed_forward: 前馈网络模块，输入输出形状为 (B, T, D)。
        dropout: dropout 比例。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: 解码器输入，形状为 (B, T, D)。
            memory: 编码器输出，形状为 (B, S, D)。
            src_mask: 源序列掩码，形状为 (B, 1, S)。
            tgt_mask: 目标序列掩码，形状为 (B, T, T)。

        Returns:
            解码器层输出，形状为 (B, T, D)。
        """
        m = memory
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, tgt_mask)) # 自注意力机制
        norm_x = self.norm2(x)
        x = x + self.dropout2(self.src_attn(norm_x, m, m, src_mask)) # 在编码器输出上做注意力
        x = x + self.dropout3(self.feed_forward(self.norm3(x))) # 前馈网络
        return x


class Decoder(nn.Module):
    """
    由 N 个解码器层堆叠而成的解码器。

    Args:
        layer: 单个解码器层，输入输出形状为 (B, T, D)。
        N: 解码器层数。
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: 目标序列表示，形状为 (B, T, D)。
            memory: 编码器输出，形状为 (B, S, D)。
            src_mask: 源序列掩码，形状为 (B, 1, S)。
            tgt_mask: 目标序列掩码，形状为 (B, T, T)。

        Returns:
            解码器输出，形状为 (B, T, D)。
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器架构。

    Args:
        encoder: 编码器模块，输入形状为 (B, S, D)，输出形状为 (B, S, D)。
        decoder: 解码器模块，输入目标序列形状为 (B, T, D)，编码器记忆形状为 (B, S, D)。
        src_embed: 源序列嵌入模块，输入 token 形状为 (B, S)，输出形状为 (B, S, D)。
        tgt_embed: 目标序列嵌入模块，输入 token 形状为 (B, T)，输出形状为 (B, T, D)。
        generator: 生成器模块，输入形状为 (B, T, D)，输出词表对数概率形状为 (B, T, V)。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播，先编码源序列，再解码目标序列。
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

