#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file check_pytorch_thread_number.py
# @brief
# @author kaiwu
# @date 2023-11-28


import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 创建一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# 创建一个简单的数据集
data = torch.randn(1000, 10)
labels = torch.randint(0, 2, (1000,))
dataset = TensorDataset(data, labels)

# 定义训练函数
def train_model(dataloader, model, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()


# 性能测试
num_threads_list = list(range(1, 17))
batch_size = 32
num_workers = 4  # DataLoader 的 num_workers

loop_count = 5
cost_times = {num_threads: 0 for num_threads in num_threads_list}

for i in range(loop_count):
    for num_threads in num_threads_list:
        torch.set_num_threads(num_threads)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        start_time = time.time()
        train_model(dataloader, model, criterion, optimizer, epochs=5)
        end_time = time.time()

        print(f"num_threads={num_threads}, Training time: {end_time - start_time:.2f} seconds")

        cost_times[num_threads] += end_time - start_time

for key, value in cost_times.items():
    print(f"key is {key}, value is {value/loop_count}")
