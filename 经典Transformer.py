"""
经典 transformer 模型

参数
vocab_len 词汇数目，例如英文词汇 5000
feature_size 词嵌入维度
hidden_size 前馈全连接层中，两个全连接层中间的维度


方法
subseq_mask(size)
attention(Q,K,V) 
module_repeat(M,N) 深度复制 M 模型N 词


refer https://www.bilibili.com/video/BV17y4y1m737

2021年12月12日 注释完毕
"""

from typing import Callable, Generator, Iterable, Iterator, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
from sys import exit



class Utils:
    @staticmethod
    def subseq_mask(size):
        """
        形成向后遮掩的掩码张量

        本 py. 中调用此函数 size = 9
        """
        attn_shape = (1, size, size)
        # 生成一个 1 方阵（不用管最前面的 1 维）
        # 然后转为上三角矩阵，k=1 表示取 0 的分界线上移，最终效果如下
        # [[[0 1 1 1]
        #   [0 0 1 1]
        #   [0 0 0 1]
        #   [0 0 0 0]]]
        mask = np.triu(np.ones(attn_shape, dtype=np.int8), k=1)
        # 反转 01，然后转为 tensor
        # 即下三角矩阵
        # [[[1 0 0 0]
        #   [1 1 0 0]
        #   [1 1 1 0]
        #   [1 1 1 1]]]
        return torch.from_numpy(1 - mask)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: torch.Tensor = None, dropout: Callable = None):
        """
        注意力计算

        scores = drop(~(Q * K.T))
        attention = scores * value

        :param query (batch_size, head_size, sentence_len, feature_size/haad_size)
        :param key (batch_size, head_size, sentence_len, feature_size/haad_size)
        :param value (batch_size, head_size, sentence_len, feature_size/haad_size)
        :param mask like Q*K.T

        key 和 value 的 sentence_len 长度必须相同，但可以和 query 不同

        :return attention, scores
        # 总之 attention.size = query.size

        实例一（自注意力模式）：
        query = key = value [batch_size = 30, head = 8, words_len = 10, part_size = 512/8] # 注意值各不相同，因为走了一个线性层
        mask = [30, 1, 1, 10] 全 True
        返回的 attention 形状和 query key value 相同，即 [30, 8, 10, 64] 和
        返回的 scores 形状 [30, 8, 10, 10]

        实例二（解码端的自注意模式）
        query = key = value = [30, 9, 512]，表示标准答案 10 个的前 9 个
        mask = 对角线为 True 的下三角矩阵

        实例三（解码端输入注意模式）
        query = [30, 8, 9, 64] key = value = [30, 8, 10, 64]，注意此时 -2 维度不同！
        mask = [30, 1, 1, 10] 全 True
        """
        # 提取文本最后一维大小，就是 feature_size
        # 补充，实际是 feature_size/head 因为多头注意力机制
        hidden_size = query.size(-1)

        # query 和 key 最后一维相同，所以后者转置，然后矩阵乘法 Q*K.T/k
        # 如果 query 和 key 为 [30, 8, 10, 64]，则 scores = [30, 8, 10, 10]
        # 如果 query = [30, 8, 9, 64] key = [30, 8, 10, 64]，则 scores = [30, 8, 9, 10]，也就是说，query 和 key 的句子长度可以不同，但是 value 和 key 长度必须相同
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(hidden_size)
        # print(f"query = {query.size()}")
        # print(f"key = {key.size()}")
        # print(f"value = {value.size()}")
        # print(f"scores = {scores.size()}")
        # print(f"mask = {mask.size()}")
        # exit(1)

        # 掩码 0 处设为 -1e9
        if mask is not None:
            # 如果 mask 对应位置为 0，则填充为 -1e-9
            # 自注意力时，mask 全为 true，所以 scores 不变
            # 解码端的自注意模式 mask = 对角线为 True 的下三角矩阵，掩盖了未来的答案，不能注意到
            scores = scores.masked_fill(mask == 0, -1e9)

        # 最后一层过 softmax
        scores = F.softmax(scores, dim=-1)

        # dropout
        if dropout is not None:
            scores = dropout(scores)

        # scores = [30, 8, 10, 10]
        # value = [30, 8, 10, 64]
        # 矩阵乘法后为，[30, 8, 10, 64]
        # ---
        # 若 scores = [30, 8, 9, 10]，返回 scores = [30, 8, 9, 64]
        return torch.matmul(scores, value), scores

    @staticmethod
    def module_repeat(modele: nn.Module, number: int) -> nn.Module:
        # 深度复制模型，形成一个级联模型
        # ModuleList 是 List 的子类，可以迭代，索引，之后也是这样用的
        return nn.ModuleList([copy.deepcopy(modele) for _ in range(number)])


class Embedding(nn.Module):
    """
    做两件事
    1. embedding
    2. embedding 结果乘上 sqrt(feature_size)

    输入 (*, n) 输出 (*, n, feature_size)
    """

    def __init__(self, vocab_len, feature_size) -> None:
        super().__init__()
        self.vocab_len = vocab_len
        self.feature_size = feature_size
        self.sqrt_feature_size = math.sqrt(feature_size)
        self.embedding = nn.Embedding(vocab_len, feature_size)

    def forward(self, x: torch.Tensor):
        """
        输入 (*, n)
        输出 (*, n, feature_size)
        """
        # 转为词向量后，再乘上 sqrt_feature_size
        return self.embedding(x) * self.sqrt_feature_size


class PositionalEncoding(nn.Module):
    """
    做了两件事
    1. 让所有输入加上位置编码矩阵
    2. dropout
    """

    def __init__(self, feature_size, dropout_ratio, sentence_max_len=5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_ratio)

        # 位置编码矩阵，形状 (sentence_max_len,feature_size) 不需要优化
        pe = torch.zeros(sentence_max_len, feature_size, requires_grad=False)

        # 位置矩阵 0,1,2,3... 形状 (sentence_max_len,1) 的序列
        position = torch.arange(0, sentence_max_len).unsqueeze(1)
        # term = exp(-9*2*i/f) 是一个递减序列，值域 [1, -e-4)，下降速度越来越慢。形状 (feature_size/2)
        term = torch.exp(
            torch.arange(0, feature_size, 2) *
            (-math.log(10000.) / feature_size)
        )
        # 广播乘法 (sentence_max_len,1) * (feature_size/2) = (sentence_max_len,feature_size/2)
        res = position * term
        # 最终的位置矩阵，很神奇，它整体是一个不同相位、频率的正弦，这样 feature
        pe[:, 0::2] = torch.sin(res)
        pe[:, 1::2] = torch.cos(res)

        # 升维 (1,sentence_max_len,feature_size)
        pe = pe.unsqueeze(0)

        # 注册这个位置编码矩阵，是一个不被优化的大参数
        self.position_encode = pe
        self.register_buffer('positon_encoding_tensor', pe)

    def forward(self, x: torch.Tensor):
        this_pe = self.position_encode[:, :x.size(1)]
        x = x + this_pe
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    # 多头注意力机制，就是把 feature_size 切成及部分，每部分由不同的"头"注意，注意词汇的不同部分
    # 总结：多头注意力，返回形状和 query 相同
    def __init__(self, head: int, feature_size: int, dropout=0.1) -> None:
        super().__init__()
        assert feature_size % head == 0

        self.head = head
        self.part_size = feature_size // head

        # feature_size 的等尺寸变换
        self.linears = Utils.module_repeat(
            nn.Linear(feature_size, feature_size), 4)
        
        # self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None):
        """
        自注意力模式： query = key = value = x 是 embedding 后的句子
        自注意力模式： x = [batch_size = 30, words_len = 10, feature_size = 512]
        自注意力模式： mask = [30, 1, 10] 全 True，表示不进行掩盖
        自注意力模式： 返回 [30, 10, 512]，和 x 相同，可以看作 embedding 后的句子每个单词间互相作用了一下

        解码端的自注意模式：query = key = value = x = [30, 9, 512]，表示标准答案 10 个的前 9 个
        解码端的自注意模式：mask = 对角线为 True 的下三角矩阵
        解码端的自注意模式：返回必然还是 [30, 9, 512]

        解码端输入注意模式：query = [30, 9, 512]
        解码端输入注意模式：key = value = [30, 10, 512]
        解码端输入注意模式：mask = [30, 1, 10] 全 True，表示不进行掩盖
        解码端输入注意模式：返回必然还是 [30, 9, 512]

        总结：多头注意力，返回形状和 query 相同
        """
        if mask is not None:
            # 加一个加到多头的头上面 (?, head=1, ?, ?)
            # 自注意力模式： mask from [30, 1, 10] to [30, 1, 1, 10]
            mask = mask.unsqueeze(1)

        # e.g. 30
        batch_size = query.size(0)

        # print(query.shape) # torch.Size([30, 10, 512])
        # print(key.shape) # torch.Size([30, 10, 512])
        # print(value.shape) # torch.Size([30, 10, 512])

        # 自注意力模式： (query, key, value) 通过一个等尺寸线性层，还是 [batch_size = 30, words_len = 10, feature_size = 512]
        # 自注意力模式： 按照 head 数目拆开，变成 [batch_size = 30, words_len = 10, head = 8, part_size = 512/8]
        # 自注意力模式： transpose 转置后为 [batch_size = 30, head = 8, words_len = 10, part_size = 512/8]，头提前
        # ---
        # 解码端输入注意模式：输入 query = [30, 9, 512] key = value = [30, 10, 512]
        # 解码端输入注意模式：过后 query = [30, 8, 9, 64] key = value = [30, 8, 10, 64]
        query, key, value = \
            [m(x).view(batch_size, -1, self.head, self.part_size).transpose(1, 2)
             for m, x in zip(self.linears, (query, key, value))]  # x 依次是 QKV
        
        # print(query.shape) # torch.Size([30, 8, 10, 64])
        # print(key.shape) # torch.Size([30, 8, 10, 64])
        # print(value.shape) # torch.Size([30, 8, 10, 64])
        # exit(1)

        # 自注意力模式： 返回 x 和 qkv 形状相同，为 [30, 8, 10, 64]
        # 自注意力模式： 返回 _ 值为 [30, 8, 10, 10]
        # ---
        # 解码端输入注意模式：返回 x 和 q 形状相同，为 [30, 8, 9, 64]
        x, _ = Utils.attention(query, key, value, mask, self.dropout)

        # self.head * self.part_size = feature_size
        # x = [30, 8, 10, 64] 转置为 [30, 10, 8, 64]
        # 然后变为 [30, 10, 512] 变回去了
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.head * self.part_size)

        # 自注意力模式： x 等尺寸线性变换，还是 [30, 10, 512]
        # 解码端输入注意模式：[30, 9, 512]
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    # 前馈全连接层（两个线性层）
    def __init__(self, featrure_size, hidden_size, dropout=0.1) -> None:
        """
        两个全连接层，输入长度 featrure_size
        转为 hidden_size
        最后变回 featrure_size
        """
        super().__init__()
        self.fc1 = nn.Linear(featrure_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, featrure_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):
    """
    规范化层，输入输出形状都是 (*,feature_size)
    对于每个 feature_size（512个），做了去偏（-mean），然后标准差置为 1（/std）
    """
    def __init__(self, feature_size, eps=1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(feature_size))
        self.bias = nn.Parameter(torch.zeros(feature_size))
        # 防止标准差为 0
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdim=True)
        return self.w * (x-mean) / (std + self.eps) + self.bias


class SubLayerConnection(nn.Module):
    # 子层连接，即 X -- 规范化 -- 子层 -- drop -- 残差
    def __init__(self, feature_size, dropout=0.1) -> None:
        super().__init__()
        self.norm = LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]):
        n = self.norm(x)
        t = sublayer(n)
        d = self.dropout(t)
        # 残差
        return x + d


class EncoderLayer(nn.Module):
    # 编码器层 = 多头自注意 + 前馈全连接 + 层层衔接（规范化 -- 子层 -- drop -- 残差）
    def __init__(self, feature_size,
                 # self_attn = (Q,K,V,mask)->X
                 self_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float) -> None:
        super().__init__()
        # treat as callable
        self.self_attn: Callable[[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor], torch.Tensor] = self_attn
        self.feed_forward: Callable[[torch.Tensor],
                                    torch.Tensor] = feed_forward
        self.feature_size = feature_size
        self.sublayers = Utils.module_repeat(
            SubLayerConnection(feature_size, dropout), 2)

    def forward(self, x, mask):
        """
        (多头)自注意层
        常见输入： x 是 embedding 后的句子
        x = [batch_size = 30, words_len = 10, feature_size = 512]
        mask = [30, 1, 10] 全 True
        """
        def attn(x): return self.self_attn(x, x, x, mask)
        x = self.sublayers[0](x, attn)
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    总结：输入输出形状不变
    n 个编码器层叠加，最后过一个 norm
    """
    def __init__(self, encoder_layer: EncoderLayer, number: int) -> None:
        super().__init__()
        self.layers = Utils.module_repeat(encoder_layer, number)
        self.norm = LayerNorm(encoder_layer.feature_size)

    def forward(self, x, mask):
        """
        常见输入:
        x = [batch_size = 30, words_len = 10, feature_size = 512] 输入的句子
        mask = [30, 1, 10] 全 True，表示不经行遮盖
        返回：形状不变
        """
        for layer in self.layers:
            # 作用完后，x 应该大小不变？回答：对的，还是 [30, 10, 512]
            # 总之，自注意力模式下，mask 不起作用，相对于 x 做了多次 attention
            x = layer(x, mask)
        # 规范化 对于每个 feature_size（512个），做了去偏（-mean），然后标准差置为 1（/std）
        return self.norm(x)


class DecoderLayer(nn.Module):
    # 解码层 = masked 多头自注意 + 层层衔接 + 多头自注意 + 层层衔接 + 前馈全连接 + 层层衔接
    # 层层衔接 = （规范化 -- 子层 -- drop -- 残差）
    def __init__(self, feature_size: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.self_attn: Callable[[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor], torch.Tensor] = self_attn
        self.src_attn: Callable[[torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor], torch.Tensor] = src_attn
        self.feed_forward: Callable[[torch.Tensor],
                                    torch.Tensor] = feed_forward
        self.subLayerConns = Utils.module_repeat(
            SubLayerConnection(feature_size, dropout), 3)

    def forward(self, x, memory, src_mask, target_mask):
        """
        @params: x [30, 9] = trg[:, :-1]，是标准答案的前 9 项（共 10 项）
        @params: memory 即 encoder 的输出 [30, 10, 512]
        @params: src_mask = torch.Size([30, 1, 10])，全 true
        @params: target_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵

        输出形状和 x 相同，可以看作一种复杂的等尺寸变换
        """

        # 下面两个函数，返回值形状和入参相同，可以看作一种复杂的等尺寸变换
        def self_attn_f(x): return self.self_attn(x, x, x, target_mask)
        def src_attn_f(x): return self.src_attn(x, memory, memory, src_mask)

        # print(x.shape)
        # print(memory.shape)
        # print(src_mask.shape)
        # print(target_mask.shape)
        # exit(1)

        x = self.subLayerConns[0](x, self_attn_f)
        x = self.subLayerConns[1](x, src_attn_f)
        x = self.subLayerConns[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    n 个解码器层叠加，然后最后 norm

    """
    def __init__(self, decoder_layer: DecoderLayer, number: int) -> None:
        super().__init__()
        self.layers = Utils.module_repeat(decoder_layer, number)
        self.norm = LayerNorm(decoder_layer.feature_size)

    def forward(self, x, memory, src_mask, target_mask):
        """
        @params: x [30, 9] = trg[:, :-1]，是标准答案的前 9 项（共 10 项）
        @params: memory 即 encoder 的输出 [30, 10, 512]
        @params: src_mask = torch.Size([30, 1, 10])，全 true
        @params: target_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵

        总结：输出同输入的 x，看作复杂的等尺寸变换
        """
        for layer in self.layers:
            # 输出输出同，看作复杂的等尺寸变换
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class OutputLayer(nn.Module):
    # 输出层，线性变化将 feature_size 映射为 vocab_size
    # 然后将值 log_softmax 缩放到 0-1 之间，和为 0，用于多分类
    def __init__(self, feature_size, vocab_size) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.project = nn.Linear(feature_size, vocab_size)

    def forward(self, x):
        """
        模型计算输出，实例 [30, 9, 512]

        """
        # project from [30, 9, feature_size=512] to [30, 9, vocab_size=11]
        x = self.project(x)
        # 过一个 softmax
        x = F.log_softmax(x, dim=-1)
        return x


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: Embedding, target_embed: Embedding, output_layer: OutputLayer) -> None:
        super().__init__()

        self.feature_size = output_layer.feature_size

        # forward(self, x, mask)
        self.encoder = encoder

        # forward(self, x, memory, src_mask, target_mask)
        self.decoder = decoder

        # 输入 (*, n) 输出 (*, n, feature_size)
        # 注意：这里还经过了 pos 位置编码
        self.src_embed = src_embed
        self.target_embed = target_embed


        self.output_layer = output_layer

    def forward(self, src, target, src_mask, target_mask) -> torch.Tensor:
        """
        src = torch.Size([batch=30, word_len=10])
        target = trg[:, :-1] = torch.Size([30, 9])
        src_mask = torch.Size([30, 1, 10])，全 true
        target_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵

        输出：形状同 target 的编码长度 [30, 9, 512]
        """
        # encoding = [30, 10, 512]
        encoding = self.encode(src, src_mask)

        # decoding = [30, 9, 512]，根据 target 长度确定
        decoding = self.decode(encoding, src_mask, target, target_mask)
        return decoding

    def encode(self, src, src_mask) -> torch.Tensor:
        """
        src = torch.Size([30, 10])
        src_mask = torch.Size([30, 1, 10])，全 true
        返回 [30, 10, 512]
        """
        # 输入 (*, n) 输出 (*, n, feature_size)
        # 具体来说 src = [30, 10]，src_em = [30, 10, 512]
        src_em = self.src_embed(src)
        # 输出输出形状不变，因此还是 [30, 10, 512]，encoder 内部实现了自注意
        encoding = self.encoder(src_em, src_mask)
        return encoding

    def decode(self, memory, src_mask, target, target_mask) -> torch.Tensor:
        """
        memory 即 encoder 的输出 [30, 10, 512]
        src_mask = torch.Size([30, 1, 10])，全 true
        target = trg[:, :-1] = torch.Size([30, 9])
        target_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵

        返回值：形状 [30, 9, 512]
        """
        # [30, 9] embed 后变成 [30, 9, 512]
        target_em = self.target_embed(target)

        # 输出同输入的 target_em，看作复杂的等尺寸变换        
        decoding = self.decoder(target_em, memory, src_mask, target_mask)
        return decoding


def create_model(src_vocab_len, target_vocab_len, layer_number=6,
                 feature_size=512, hidden_size=2048, head=8, dropout=0.1) -> Model:

    c = copy.deepcopy
    fs = feature_size
    d = dropout

    attn = MultiHeadAttention(head, fs)

    fc = PositionwiseFeedForward(fs, hidden_size, d)

    pos = PositionalEncoding(fs, d)

    src_em = Embedding(src_vocab_len, fs)
    target_em = Embedding(target_vocab_len, fs)



    encoder_layer = EncoderLayer(fs, c(attn), c(fc), d)
    decoder_layer = DecoderLayer(fs, c(attn), c(attn), c(fc), d)

    encoder = Encoder(encoder_layer, layer_number)
    decoder = Decoder(decoder_layer, layer_number)

    output_layer = OutputLayer(feature_size, target_vocab_len)

    model = Model(
        encoder, 
        decoder, 
        nn.Sequential(src_em, c(pos)),
        nn.Sequential(target_em,c(pos)),
        output_layer)
    
    for params in model.parameters():
        if params.dim() > 1:
            # xavier_uniform 是一种初始化模型参数的方法，使得每层输出的方差尽可能相同
            nn.init.xavier_uniform_(params)
    
    return model

class Batch:
    # "Object for holding a batch of data with mask during training."
    # batch object that holds the src and target sentences for training
    # as well as constructing the masks.
    # 入参 src、trg 形状为 (batch=30, 10)，取值 [1, vovab_len=11)
    # self.src = torch.Size([30, 10])
    # self.src_mask = torch.Size([30, 1, 10])，全 true
    # self.trg = trg[:, :-1] = torch.Size([30, 9])
    # self.trg_y = trg[:, 1:] = torch.Size([30, 9])
    # self.trg_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵
    # self.ntokens = tensor(270)
    def __init__(self, src:torch.Tensor, trg:Optional[torch.Tensor]=None, pad=0):
        # 形状 torch.Size([30, 10])
        self.src:torch.Tensor = src
        # src_mask 形状 torch.Size([30, 1, 10])，全 true
        self.src_mask:torch.Tensor = (src != pad).unsqueeze(-2)

        if trg is not None:
            # self.trg self.trg_y = torch.Size([30, 9])
            self.trg:torch.Tensor = trg[:, :-1]
            self.trg_y:torch.Tensor = trg[:, 1:]

            # trg_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵
            self.trg_mask:torch.Tensor = \
                self.make_std_mask(self.trg, pad)
            
            # 是 tensor(270)
            self.ntokens:torch.Tensor = (self.trg_y != pad).data.sum()

        # print(self.src.shape)
        # print(self.src_mask.shape)
        # print(self.trg.shape)
        # print(self.trg_y.shape)
        # print(self.trg_mask.shape)
        # print(self.ntokens)
        # torch.Size([30, 10])
        # torch.Size([30, 1, 10])
        # torch.Size([30, 9])
        # torch.Size([30, 9])
        # torch.Size([30, 9, 9])
        # tensor(270)
        # exit(1)
    
    @staticmethod
    def make_std_mask(tgt:torch.Tensor, pad:torch.Tensor):
        """
        Create a mask to hide padding and future words.
        tgt = torch.Size([batch = 3, 9]) = trg[:, :-1]
        pad = 0
        """
        # torch.Size([batch = 3, 1, 9])
        tgt_mask = (tgt != pad).unsqueeze(-2)

        # tgt.size(-1) = 9
        # 得到对角线为 1 的下三角方阵
        # [[[1 0 0 0]
        #   [1 1 0 0]
        #   [1 1 1 0]
        #   [1 1 1 1]]]
        # tgt_mask = torch.Size([batch = 3, 9, 9])
        tgt_mask = tgt_mask & Utils.subseq_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

def run_epoch(data_iter:Iterator[Batch], model:Model, loss_compute:'SimpleLossCompute'):
    """
    Standard Training and Logging Function
    @param data_iter 遍历 Batch
    @param model (vocab_size = 11, layer = 2) feature_size=512, hidden_size=2048, head=8
    @param loss_compute
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # self.src = torch.Size([30, 10])
        # self.trg = trg[:, :-1] = torch.Size([30, 9])
        # self.src_mask = torch.Size([30, 1, 10])，全 true
        # self.trg_mask = torch.Size([batch = 30, 9, 9])，对角线为 True 的下三角方阵
        # 返回 out = 形状同 target 的编码长度 [30, 9, 512]
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)

        # out = 形状同 target 的编码长度 [30, 9, 512]
        # batch.trg_y = trg[:, 1:] 这就是答案 —— 明白了，通透了
        # batch.ntokens = 30*9 = 270
        # 返回 loss，并且做优化
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# optimizer
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, feature_size:int, factor:int, warmup:int, optimizer:torch.optim.Adam):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.feature_size = feature_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.feature_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    @staticmethod
    def get_std_opt(model:Model):
        return NoamOpt(model.feature_size, 2, 4000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def data_gen(vocab_len, batch, nbatches):
    """
    Generate random data for a src-tgt copy task.
    被调用时参数示例：11, 30, 20
    这是一个生成器（调用此函数生成一个生成器），长度 nbatches
    """
    for i in range(nbatches):
        # randint 的范围是 [1, vocab_len)，data 形状为 (batch, 10)
        data:torch.IntTensor = torch.randint(1, vocab_len, size = (batch, 10))
        # batch 每个元素的第一个值设为 0，意义未知，形状不变
        data[:, 0] = 1
        # src tgt形状为 (batch, 10)
        src:torch.LongTensor = data.clone().long()
        tgt:torch.LongTensor = data.clone().long()
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    """
    A simple loss compute and train function.
    最终调用它计算 loss
    """
    def __init__(self, generator:OutputLayer, criterion:'LabelSmoothing', opt:Optional[NoamOpt]=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x:torch.Tensor, y:torch.Tensor, norm:torch.Tensor):
        """
        x = 模型计算输出，实例 [30, 9, 512]
        y = 标准答案 [30, 9]
        norm = batch.ntokens = 270
        """
        # 线性变化将 feature_size 映射为 vocab_size
        # 即 [30, 9, 512] 到 [30, 9, 11]
        x:torch.Tensor = self.generator(x)

        # 输入 x.contiguous().view(-1, x.size(-1)) = [30*9, 11]
        # 输入 y.contiguous().view(-1)) = [30*9]
        loss:torch.Tensor = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    "https://zhuanlan.zhihu.com/p/116466239"
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        """
        size 就是词汇长度 vocab_len
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing # 1.0
        self.smoothing = smoothing # 0.0
        self.size = size # vocab_len = 11
        self.true_dist:torch.Tensor = None
        
    def forward(self, x:torch.Tensor, target:torch.Tensor):
        """
        # 输入 x = [30*9, 11]
        # 输入 target = [30*9]
        """
        assert x.size(1) == self.size

        # print(x.shape)
        # print(target.shape)
        # exit(1)

        true_dist = x.clone()
        # [30*9, 11]
        true_dist.fill_(self.smoothing / (self.size - 2)) # fill(0.0)
        # scatter() 一般可以用来对标签进行 one-hot 编码
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence) # 1.0
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

def greedy_decode(model:Model, src:torch.LongTensor, src_mask, max_len, start_symbol=1):
    """
    src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) shape is [1,10]
    src_mask = [1,1,10] 全 1
    """
    # 直接编码，得到 memory
    memory = model.encode(src, src_mask)
    # [[1]]
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        # 循环解码
        out = model.decode(memory, src_mask, 
                           ys, 
                           Utils.subseq_mask(ys.size(1)).type_as(src.data))
        prob = model.output_layer(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

if __name__ == '__main__':
    V = 11
    criterion:LabelSmoothing = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model:Model = create_model(V, V, layer_number = 2)
    model_opt:NoamOpt = NoamOpt(model.feature_size, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, 
                SimpleLossCompute(model.output_layer, criterion, model_opt))

    model.eval()
    src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
    src_mask = torch.ones(1, 1, 10)
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    # 测试位置编码，绘制 pe 热图
    if not True:
        feature_size = 4096
        sentence_max_len = 5000
        pe = torch.zeros(sentence_max_len, feature_size)

        position = torch.arange(0, sentence_max_len).unsqueeze(1)
        term = torch.exp(
            torch.arange(0, feature_size, 2) * (-math.log(10000.) / feature_size))
        res = position * term
        pe[:, 0::2] = torch.sin(res)
        pe[:, 1::2] = torch.cos(res)
        fig, ax = plt.subplots(figsize=(10, 18))
        im = ax.imshow(pe.numpy())
        fig.colorbar(im, pad=0.03)  # 设置颜色条
        plt.show()

    # 测试位置编码，绘制单个 feature[i] 的加权
    if not True:
        feature_size = 4096
        sentence_max_len = 5000
        pe = torch.zeros(sentence_max_len, feature_size)

        position = torch.arange(0, sentence_max_len).unsqueeze(1)
        term = torch.exp(
            torch.arange(0, feature_size, 2) * (-math.log(10000.) / feature_size))
        res = position * term
        pe[:, 0::2] = torch.sin(res)
        pe[:, 1::2] = torch.cos(res)

        if not True:
            plt.plot(pe[:, 3000], 'r-')
            plt.plot(pe[:, 3001], 'r--')
            plt.plot(pe[:, 3002], 'k-')
            plt.plot(pe[:, 3003], 'k--')
            plt.legend(['3000', '3001', '3002', '3003'])

        if not True:
            plt.plot(pe[:, 3000], 'r-')
            plt.plot(pe[:, 3002], 'r--')
            plt.plot(pe[:, 3004], 'k-')
            plt.plot(pe[:, 3006], 'k--')
            plt.legend(['3000', '3002', '3004', '3006'])

        if True:
            plt.plot(pe[:, 3000], 'r-')
            plt.plot(pe[:, 3100], 'r--')
            plt.plot(pe[:, 3200], 'k-')
            plt.plot(pe[:, 3300], 'k--')
            plt.legend(['3000', '3100', '3200', '3300'])

        plt.show()

    # 测试位置编码，绘制某位置对 feature 的加权
    if not True:
        feature_size = 4096
        sentence_max_len = 5000
        pe = torch.zeros(sentence_max_len, feature_size)

        position = torch.arange(0, sentence_max_len).unsqueeze(1)
        term = torch.exp(
            torch.arange(0, feature_size, 2) * (-math.log(10000.) / feature_size))
        res = position * term
        pe[:, 0::2] = torch.sin(res)
        pe[:, 1::2] = torch.cos(res)

        if not True:
            plt.plot(pe[0, :], 'r-')

        if not True:
            plt.plot(pe[1000, :], 'r-')

        if True:
            plt.plot(pe[4999, :], 'r-')
            plt.plot(pe[4998, :], 'k-')

        plt.show()

    # 测试 EncoderLayer
    if not True:
        vocab_size = 5000
        feature_size = 512
        head = 8
        hidden_size = 64
        dropout = 0.2
        self_attn = MultiHeadAttention(head, feature_size)
        fc = PositionwiseFeedForward(feature_size, hidden_size, dropout)

        em = Embedding(vocab_size, feature_size)
        pe = PositionalEncoding(feature_size, dropout)
        el = EncoderLayer(feature_size, self_attn, fc, dropout)

        x = torch.LongTensor([[1, 2, 3]])
        mask = torch.zeros(x.size(0), x.size(-1), x.size(-1))
        print(f"x size = {x.size()}")

        em_x = em(x)
        print(f"em_x size = {em_x.size()}")

        pe_x = pe(em_x)
        print(f"pe_x size = {pe_x.size()}")

        el_x = el(pe_x, mask)
        print(f"el_x size = {el_x.size()}")

    # 测试 Encoder
    if not True:
        vocab_size = 5000
        feature_size = 512
        head = 8
        hidden_size = 64
        dropout = 0.2
        self_attn = MultiHeadAttention(head, feature_size)
        fc = PositionwiseFeedForward(feature_size, hidden_size, dropout)

        em = Embedding(vocab_size, feature_size)
        pe = PositionalEncoding(feature_size, dropout)
        el = EncoderLayer(feature_size, self_attn, fc, dropout)

        encoder = Encoder(el, number=8)

        for i in range(10):
            _en = Encoder(el, i)
            print(f"num = {i} params = {len(list(_en.parameters()))}")

        x = torch.LongTensor([[1, 2, 3, 4], [1001, 1003, 1004, 1014]])
        mask = torch.zeros(x.size(0), x.size(-1), x.size(-1))
        print(f"x size = {x.size()}")

        em_x = em(x)
        print(f"em_x size = {em_x.size()}")

        pe_x = pe(em_x)
        print(f"pe_x size = {pe_x.size()}")

        encoder_x = encoder(pe_x, mask)
        print(f"encoder_x size = {encoder_x.size()}")

    # DecoderLayer
    if not True:
        vocab_size = 5000
        feature_size = 512
        head = 8
        hidden_size = 64
        dropout = 0.2
        self_attn = src_attn = MultiHeadAttention(head, feature_size)
        fc = PositionwiseFeedForward(feature_size, hidden_size, dropout)

        em = Embedding(vocab_size, feature_size)
        pe = PositionalEncoding(feature_size, dropout)
        el = EncoderLayer(feature_size, self_attn, fc, dropout)

        encoder = Encoder(el, number=8)

        x = torch.LongTensor([[1, 2, 3, 4], [1001, 1003, 1004, 1014]])
        mask = torch.zeros(x.size(0), x.size(-1), x.size(-1))
        print(f"x size = {x.size()}")  # [2, 4]

        em_x = em(x)
        print(f"em_x size = {em_x.size()}")  # [2, 4, 512]

        pe_x = pe(em_x)
        print(f"pe_x size = {pe_x.size()}")  # [2, 4, 512]

        encoder_x = encoder(pe_x, mask)
        print(f"encoder_x size = {encoder_x.size()}")

        memery = encoder_x

        d_in_x = pe_x

        src_mask = target_mask = mask

        dl = DecoderLayer(feature_size, self_attn, src_attn, fc, dropout)

        dl_x = dl(d_in_x, memery, src_mask, target_mask)

        print(f"d_in_x size = {d_in_x.size()}")
        print(f"dl_x size = {dl_x.size()}")

    # create_model
    if not True:
        model = create_model(src_vocab_len=11,target_vocab_len=11)
        print(model)