#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """DQN使用的Q网络（MLP）。"""

    def __init__(self, state_dim, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """基于deque的经验回放池。"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states/next_states是连续向量，直接拼成二维Tensor: [B, state_dim]
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)

        # actions用于gather，需为Long类型并扩展到[B, 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)

        # rewards和dones在计算TD目标时与Q值对齐为[B, 1]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones


class Algorithm:
    def __init__(
        self,
        state_dim,
        action_size,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=100000,
        target_update_freq=500,
    ):
        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_net = QNetwork(state_dim, action_size).to(self.device)
        self.target_net = QNetwork(state_dim, action_size).to(self.device)

        # 初始化时保证两个网络参数完全一致
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.SmoothL1Loss()

        self.learn_step = 0

    def learn(self, list_sample_data):
        # 先把框架送来的样本写入回放池
        for sample in list_sample_data:
            self.replay_buffer.push(
                sample.state,
                sample.action,
                sample.reward,
                sample.next_state,
                sample.done,
            )

        # 样本不足一个batch时不进行训练
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        # 当前Q值: Q(s,a)，使用gather从所有动作Q中取出执行动作对应值，形状[B, 1]
        current_q = self.eval_net(states).gather(1, actions)

        # 目标Q值: r + gamma * max_a' Q_target(s',a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0].detach()
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1

        # 固定频率硬更新Target网络
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        return float(loss.item())
