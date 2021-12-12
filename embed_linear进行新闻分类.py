"""
refer https://www.bilibili.com/video/BV17y4y1m737
"""

import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import time
from torch.utils.data.dataset import random_split



# 下载数据
train_dataset, test_dataset = torchtext.datasets.DATASETS['AG_NEWS']("./data")

BATCH_SIZE = 16
# 单词总类数目
VOCAB_SIZE = 10000
# 词嵌入向量长度
EMBED_DIM = 32
# 分类标签数目
NUM_CLASS = 4 #len(train_dataset.get_labels())


device = torch.device('cpu')

# 带有 embedding 层的文本分类模型
class TextClassify(torch.nn.Module):
    # 词汇总数目、embedding 向量维度、分类任务类别总数
    def __init__(self, vocab_size, embed_dim, num_classes) -> None:
        super().__init__()
        # 实例化 embedding 层，sparse true 表示每次只更新部分权重
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        # 线性层
        self.fc = nn.Linear(embed_dim, num_classes)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 均匀分布的权重
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        # 偏置初值为 0
        with torch.no_grad():
            self.fc.bias.zero_()
    
    # text 数值化映射后的文本
    def forward(self, text):
        # embedded 大小为 m*32，m 是 batch_size 大小的数据的总词汇数目。（32 词嵌入维度）
        embedded = self.embedding(text)
        # 将 embedded 转为 (batch_size, 32)
        # 首先 m 整除 batch_size
        c = embedded.size(0) // BATCH_SIZE # 整除
        embedded:torch.Tensor = embedded[:c*BATCH_SIZE] # 截取 c*batch_size
        embedded:torch.Tensor = embedded.transpose(0,1).unsqueeze(0) # 转置，再新增 0 维度，变成 3 维度
        embedded:torch.Tensor = F.avg_pool1d(embedded, kernel_size=c) # 一维池化，作用域三维，作用在最后一维度，这样就完成了 m*32 到 batch_size*32
        return self.fc(embedded[0].transpose(0,1)) # 去除新增的 0 维，然后转置回去


model = TextClassify(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

print(model)


# 数据 batch 生成
def generate_batch(data):
    # data = [(sample, label), (sample, label), ...]
    # 生成的 text = [*sample, *sample] 被展开了
    # 生成的 label = [label, label, label]
    label = torch.tensor([each[1] for each in data])
    text = [each[0] for each in data]
    text = torch.cat(text)
    return text, label

# 构建训练与验证的函数
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
criterion = torch.nn.CrossEntropyLoss().to(device)
# 学习率调整
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


def train(train_data):
    train_loss = 0.0
    train_acc = 0.0

    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    for text, cls in data:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    
    scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)

def valid(valid_data):
    loss = 0.0
    acc = 0.0
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    for text, cls in data:
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()
    
    return loss / len(valid_data), acc / len(valid_data)

# 输出损失验证
min_valid_loss = float('inf')

# 95% 训练
train_len = int(len(train_dataset)*0.95)

# 划分训练集 验证集
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(2):
    start_time = time.time()
    
    # 训练验证
    train_loss, train_acc = train(sub_train_)
    valid_loss, valid_acc = valid(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    
    print(f"epoch {epoch} tiem {mins} m {secs} s")
    print(f"train_loss = {train_loss}, train_acc = {train_acc}")
    print(f"valid_loss = {valid_loss}, valid_acc = {valid_acc}")

