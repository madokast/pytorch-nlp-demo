"""
refer https://www.bilibili.com/video/BV17y4y1m737
"""

import io
import glob
import os
import string
from typing import NamedTuple
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------- 数据处理 --------------------

# 常用字符（大小写、标点）
all_letters = string.ascii_letters + " .,;'"
# 字符数量
n_letters = len(all_letters)
print(f"all_letters = {all_letters}")
print(f"n_letters = {n_letters}")

# uft 转 ascii，去除重音标记


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters)


print(f"unicode2ascii('abc') = {unicode2ascii('abc')}")
print(f"unicode2ascii('à') = {unicode2ascii('à')}")

# 读名字的函数


def readNames(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]


chinese_name_file = './data/names/Chinese.txt'
print(readNames(chinese_name_file)[:3])

# 构建 dict: cpuntry2names
all_countries = []
countryNames = {}
for file in glob.glob('./data/names/' + '*.txt'):
    country = os.path.splitext(os.path.basename(file))[0]
    all_countries.append(country)
    names = readNames(file)
    print(f'{country} 存在名字 {names[:3]}... 共 {len(names)} 个')
    countryNames[country] = names

# 人名转为 one-hot 张量


def name2tensor(name):
    a = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        a[i][0][all_letters.find(letter)] = 1
    return a


print(f"name2tensor('abc') = {name2tensor('abc').shape}")

# -------------------- 构建模型 --------------------


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initHidden()
        # input 形状为 1, n_letters 扩充一维
        input = input.unsqueeze(0)
        rr, hn = self.rnn(input, hidden)
        return self.softmax(self.fc(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden_and_cell: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_and_cell is None:
            hidden_and_cell = (self.initHiddenCell(), self.initHiddenCell())
        # input 形状为 1, n_letters 扩充一维
        input = input.unsqueeze(0)
        rr, (hn, c) = self.lstm(input, hidden_and_cell)
        return self.softmax(self.fc(rr)), (hn, c)

    def initHiddenCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initHidden()
        # input 形状为 1, n_letters 扩充一维
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.fc(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 模型测试
input = name2tensor('B').squeeze(0)
rnn = RNN(n_letters, 256, len(all_countries))
lstm = LSTM(n_letters, 256, len(all_countries))
gru = GRU(n_letters, 256, len(all_countries))
print(f"rnn(input) = {rnn(input)[0].shape} {rnn(input)[1].shape}")
print(
    f"lstm(input) = {lstm(input)[0].shape} {lstm(input)[1][0].shape} {lstm(input)[1][1].shape}")
print(f"gru(input) = {gru(input)[0].shape} {gru(input)[1].shape}")


# -------------------- 实用函数 --------------------

# 输出中拿到国家
def countryFromOutput(output: torch.Tensor) -> NamedTuple("country_id", [("country", str), ("id", int)]):
    _, i = output.topk(1)
    i = i[0].item()
    return all_countries[i], i


print(f"countryFromOutput = {countryFromOutput(rnn(input)[0])}")

# 随机生成训练数据


def randTrainData():
    country = random.choice(all_countries)
    name = random.choice(countryNames[country])
    countryTensor = torch.tensor(
        [all_countries.index(country)], dtype=torch.long)
    nameTensor = name2tensor(name)
    return country, name, countryTensor, nameTensor


print(f"randTrainData = {randTrainData()[:3]} {randTrainData()[3].shape}")


def getOptimizer(model: nn.Module, lr):
    return torch.optim.SGD(model.parameters(), lr)

def timeSince(start):
    now = time.time()
    p = now - start
    m = math.floor(p/60)
    p -= m*60
    return f"{m} m {int(p)} s"


# -------------------- 训练准备 --------------------

# 配合 log softmax，这个函数输入为 m,n 和 m，即 batch_size=m，分类 n 个。第一个输入为模型结果，第二个输入为每个的编号（不用转为one-hot）
criterion = nn.NLLLoss()  
lr = 0.005

# 训练一个数据
def trainOnce(model: nn.Module, optimizer: torch.optim.Optimizer, countryTensor, nameTensor):
    optimizer.zero_grad()
    out, hi = None, None
    for i in range(nameTensor.size()[0]):
        out, hi = model(nameTensor[i], hi)
    # 只要最后的 out
    loss: torch.Tensor = criterion(out.squeeze(0), countryTensor)
    loss.backward()
    optimizer.step()
    return out, loss.item()
print(f"train = {trainOnce(rnn, getOptimizer(rnn, lr), *(randTrainData()[2:]))}")
print(f"train = {trainOnce(lstm, getOptimizer(lstm, lr), *(randTrainData()[2:]))}")
print(f"train = {trainOnce(gru, getOptimizer(gru, lr), *(randTrainData()[2:]))}")


epochs = 1000
print_every = 50

def train(model: nn.Module, optimizer: torch.optim.Optimizer):
    losses = 0.
    avg_losses = []
    start = time.time()
    for epoch in range(epochs):
        country, name, countyTensor, nameTensor = randTrainData()
        out, loss = trainOnce(model, optimizer, countyTensor, nameTensor)
        losses += loss
        if (epoch+1) % print_every == 0:
            print(f"epoch = {epoch} time = {timeSince(start)} loss = {losses/print_every}")
            avg_losses.append(losses/print_every)
            losses = 0.
            start = time.time()
    return avg_losses

rnn_losses = train(rnn, getOptimizer(rnn,lr))
lstm_losses = train(lstm, getOptimizer(lstm,lr))
gru_losses = train(gru, getOptimizer(gru,lr))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(rnn_losses, color='r', label='RNN')
ax.plot(lstm_losses, color='b', label='LSTM')
ax.plot(gru_losses, color='k', label='GRU')
ax.legend()
plt.show()