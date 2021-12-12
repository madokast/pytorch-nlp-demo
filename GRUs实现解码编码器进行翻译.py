"""
refer https://www.bilibili.com/video/BV17y4y1m737
"""

import io
from typing import Callable, NamedTuple, Optional, Sized
import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------- 数据预处理 -----------------------


class Lang:
    # 构建某语言的 单词2序号 字典
    # 起始标志、结束标志
    SOS_TOKEN, EOS_TOKEN = 0, 1

    def __init__(self, languageName: str) -> None:
        self.languageName = languageName
        self.word2Index = {}
        self.index2word = {Lang.SOS_TOKEN: 'SOS', Lang.EOS_TOKEN: 'EOS'}
        self.currentIndex = 2  # 因为 0，1 已经被使用

    def addSentence(self, sentence: str):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word: str):
        if word not in self.word2Index:
            self.word2Index[word] = self.currentIndex
            self.index2word[self.currentIndex] = word
            self.currentIndex += 1

    def sentence2tensor(self, sentence: str) -> torch.Tensor:
        # 将句子 I am student 转为 [12 3 5 1] 这样的 seq，注意末尾加了结束标记，seq 形状为 (n,1)
        indexes = [self.word2Index[word] for word in sentence.split()]
        # 句子加上结束标记
        indexes.append(Lang.EOS_TOKEN)
        # 形状变为 n,1
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def __str__(self) -> str:
        return f"Dict {self.languageName} scala is {self.currentIndex}"

    def __len__(self) -> int:
        return self.currentIndex

    @staticmethod
    def unicode2ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    @staticmethod
    def normalizeStr(s: str):
        s = Lang.unicode2ascii(s.lower().strip())
        # re.sub 是正则替换
        # 在 .!? 前加入空格
        s = re.sub(r"([.!?])", r" \1", s)
        # 不是字母和 .!? 都换成空格
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s


def pair2tensor(dict1: Lang, dict2: Lang, pair: tuple[str]) -> tuple[torch.Tensor]:
    # 将 pair(原句, 目标句) 转为 (seq, seq)，seq 形状为 (n,1)
    # sentence2tensor 将句子 I am student 转为 [12 3 5 1] 这样的 seq，注意末尾加了结束标记
    return (dict1.sentence2tensor(pair[0]), dict2.sentence2tensor(pair[1]))


class EncoderRNN(nn.Module):
    # input 为一个单词 id，遍历句子依次输入，hidden 作为中间传递
    # input_size 词汇表大小
    # hidden_size 词嵌入维度
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # embedding 向量的长度即 hidden_size
        self.embedding: Callable[[torch.Tensor], torch.Tensor] = nn.Embedding(
            input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    # 一次只输入一个词，循环调用
    def forward(self, input: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor]:
        if hidden is None:
            hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        # 拓展成三维，因为 gru 的输入必须是三维
        # 补充：input 实际上是一个单词 id，维度为 (1)，经过 embedding 后变为 (hidden_size)，view 后为 (1,1,hidden_size)
        output: torch.Tensor = self.embedding(input).view(1, 1, -1)
        # 经过 gru，input(1,1,hidden_size) 映射为 output(1,1,hidden_size)，hidden 形状也是 (1,1,hidden_size)
        output, hidden = self.gru(output, hidden)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length) -> None:
        # output_size 整个解码器的输出（目标语言词汇数目）
        # hidden_size 解码器中输出尺寸
        # dropout_p dropout 层置零比例
        # max_length 句子最大长度
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 将单词（目标语言）编码为 hidden_size 长度的向量
        self.embedding: Callable[[torch.Tensor], torch.Tensor] = nn.Embedding(
            output_size, hidden_size)

        self.attn = nn.Linear(hidden_size*2, max_length)
        self.attn_comb: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(
            hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs: torch.Tensor):
        """
        input 即一个单词 id，形状 (1,1)，更详细来说，这个第一次输入为 SOS，其后输入为上一步输出拿到的单词 id
        hidden 形状 (1,1,hidden_size) 第一次输入为 encoder 最后得到的 hidden，其后为上一步拿到的 hidden
        encoder_outputs 形状为 (seq_max_len, hidden_size)，是 encoder 每一步的 hidden 集中起来，不足填 0
        """
        if hidden is None:
            hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        # 单词 embedding 转 (1,1,hidden_size)
        embedded = self.embedding(input).view(1, 1, -1)
        # 将 embedded 中的数一部分随机置零，形状不变 (1,1,hidden_size)
        embedded = self.dropout(embedded)

        # 1. torch.cat((embedded[0], hidden[0]), 1) 这里 dim=1，即指定维度做加法 (1,h)+(1,h)=(1,2h)
        # 2. 过 attn 全连接层，形状变为 (1,max_length)
        # 3. 过 softmax，并指定 dim=1，因为 softmax 计算涉及一个组
        # 最后 attn_weights 形状为 (1,max_length)
        attn_weights = F.softmax(self.attn(
            torch.cat((embedded[0], hidden[0]), 1)
        ), dim=1)

        # attn_weights 形状为 (1,seq_max_len)
        # encoder_outputs 形状为 (seq_max_len, hidden_size)
        # 升维，bmm  后结果为 (1,1,hidden_size)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # (1,hidden_size) + (1,hidden_size) = (1,2*hidden_size)
        output = torch.cat((embedded[0], attn_applied[0]), dim=1)

        # attn_comb 线性层，将 (1,2*hidden_size) 变为 (1,hidden_size)，生维 (1,1,hidden_size)
        output = self.attn_comb(output).unsqueeze(0)

        # 非线性
        output = F.relu(output)

        # 这里四个张量都是 (1,1,hidden_size)
        output, hidden = self.gru(output, hidden)

        # 降维过 out 从 (1,hidden_size) 变成 (1,target_vocab_len)
        output = F.log_softmax(self.out(output[0]), dim=1)

        # 返回形状 (1,target_vocab_len) (1,1,hidden_size) (1,max_length)
        return output, hidden, attn_weights


def train(input: torch.Tensor, target: torch.Tensor,
          encoder: torch.nn.Module, decoder: torch.nn.Module,
          encoder_optimizer: torch.optim.Optimizer, decoder_optimizer: torch.optim.Optimizer,
          criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          sentence_max_length: int, teacher_forcing_ratio: float):
    """
    input target，原句和目标句构成的 seq，形状为 (n,1)
    encoder encoderRNN
    decoder attnDecoderRNN
    encoder_optimizer decoder_optimizer 相应的优化器
    criterion NLLLoss 损失函数，输入 input(batch_size, feature_size) 和 targrt(batch_size) 用于分类问题，不用将 targrt 转为 one-hot
    sentence_max_length 句子最大长度
    teacher_forcing_ratio 进行老师指导的概率
    return loss
    """
    encoder_hid = None

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # tensor.size(0) 即 dim=0 维度的长度，也就是句子长度（包括结束标记）
    input_len = input.size(0)
    target_len = target.size(0)

    # 形状 (seq_max_len, hidden_size)
    encoder_outs = torch.zeros(
        sentence_max_length, encoder.hidden_size, device=device)

    loss = 0.

    for ei in range(input_len):
        # encoder 传入 input 形状为 (1)，输出 encoder_out 和 encoder_hid 形状为 (1,1,hidden_size)
        encoder_out, encoder_hid = encoder(input[ei], encoder_hid)
        # t[0,0] 就是 t[0][0]，那么 encoder_out[0, 0] 形状为 (hidden_size)
        # encoder_outs 形状 (seq_max_len, hidden_size)，也就是把 seq 每个输出都保存
        encoder_outs[ei] = encoder_out[0, 0]

    # 输入就是一个起始标记，注意形状 (1,1)
    decoder_input = torch.tensor([[Lang.SOS_TOKEN]], device=device)

    # 解码器的 hid 即编码器的 hid (1,1,hidden_size)
    decoder_hid = encoder_hid

    if random.random() < teacher_forcing_ratio:
        # 老师指导
        for di in range(target_len):
            # 形状依次是 (1,target_vocab_len) (1,1,hidden_size) (1,max_length)
            decoder_out, decoder_hid, decoder_attn = decoder(
                decoder_input, decoder_hid, encoder_outs)

            loss += criterion(decoder_out, target[di])
            decoder_input = target[di]  # 强制下一次输入为正确答案

    else:
        # 不指导
        for di in range(target_len):
            decoder_out, decoder_hid, decoder_attn = decoder(
                decoder_input, decoder_hid, encoder_outs
            )
            topv, topi = decoder_out.topk(1)
            loss += criterion(decoder_out, target[di])
            if topi.squeeze().item() == Lang.EOS_TOKEN:
                break
            decoder_input = topi.squeeze().detach()

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_len


def trainIters(encoder: nn.Module, decoder: nn.Module,
               n_iters: int, dict1: Lang, dict2: Lang,
               pairs: tuple[str], sentence_max_len: int,
               teacher_forcing_ratio: float,
               print_every: int = 100, lr: float = 0.001):
    """
    encoder encoderRNN
    decoder attnDecoderRNN
    n_iters 训练次数
    dict1 原语言词典
    dict2 目标语言词典
    pairs 内含两个 str，[0] 为原语言句子，[1] 为翻译后的目标句子
    sentence_max_len 句子最大长度
    teacher_forcing_ratio 进行老师指导的概率
    print_every 每多少个 loss 取平均，打印出来
    lr 学习率
    """
    plot_loss = []
    print_loss_total = 0

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)

    # 输入 input(batch_size, feature_size) 和 targrt(batch_size) 用于分类问题，不用将 targrt 转为 one-hot
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        # 将 pair(原句, 目标句) 转为 (seq, seq)，seq 形状为 (n,1)
        pair = pair2tensor(dict1, dict2, random.choice(pairs))
        input = pair[0]  # 形状为 (n,1)
        target = pair[1]  # 形状为 (n,1)

        loss = train(input, target, encoder, decoder,
                     encoder_optimizer, decoder_optimizer,
                     criterion, sentence_max_len, teacher_forcing_ratio)

        print_loss_total += loss

        if iter % print_every == 0:
            avg_loss = print_loss_total / print_every
            print(f"iter = {iter} avg_loss = {avg_loss}")
            plot_loss.append(avg_loss)
            print_loss_total = 0

    plt.plot(plot_loss)
    plt.show()


def test(object: str):
    print('test ' + object)
    if object == 'device':
        print(f"device = {device}")
    elif object == 'class Lang':
        eng = Lang('eng')
        eng.addSentence('hello I am Jay')
        print(f"word2Index = {eng.word2Index}")
        print(f"index2word = {eng.index2word}")
    elif object == 'normalizeStr':
        s = "Are you kidding me?"
        print(Lang.normalizeStr(s))
    elif object == "class EncoderRNN":
        eng = Lang('eng')
        eng.addSentence('hello I am Jay')
        m = EncoderRNN(len(eng), 8)
        encoder_output, hidden = m(eng.sentence2tensor('hello I am Jay')[0][0])
        print(f"encoder_output = {encoder_output}")


if __name__ == '__main__':
    test("device")
    test("class Lang")
    test("normalizeStr")
    test("class EncoderRNN")

    # 读数据
    data_path = './data/eng-fra.txt'
    lines = io.open(data_path, encoding='utf-8').readlines()
    pairs = [[Lang.normalizeStr(s) for s in line.split('\t')]
             for line in lines]
    print(f"read pairs len = {len(pairs)}, {pairs[:3]}")

    engDict, fraDict = Lang('eng'), Lang('fra')

    # 只处理短语言对，且必须是 sb is 结构
    sentence_max_len = 10
    eng_prefix = (
        "i am", "i m", "he is", "he s", "you are", "you re", "we are", "we re", "they are", "they re"
    )
    fliterPairs = []
    for pair in pairs:
        eng = pair[0]
        fra = pair[1]
        if (len(eng.split()) < sentence_max_len
            and len(fra.split()) < sentence_max_len
                and eng.startswith(eng_prefix)):
            fliterPairs.append(pair)
            # 同时放入字典
            engDict.addSentence(eng)
            fraDict.addSentence(fra)
    print(f"fliterPairs len = {len(fliterPairs)}, {fliterPairs[:3]}")
    print(f"engDict = {engDict}, fraDict = {fraDict}")
    print(f"pair2tensor = {pair2tensor(engDict,fraDict,fliterPairs[500])}")

    # 启动纠错比例
    teacher_forcing_ratio = 0.5
    hidden_size = 256
    encoder = EncoderRNN(len(engDict), hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, len(fraDict),
                                  dropout_p=0.1, max_length=sentence_max_len).to(device)
    n_iters = 75000
    print_every = 100
    lr = 0.001

    trainIters(encoder, attn_decoder,
               n_iters, engDict, fraDict, fliterPairs,
               sentence_max_len, teacher_forcing_ratio, print_every, lr)
