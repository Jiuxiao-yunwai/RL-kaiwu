#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized: 精简Dueling DQN - 去掉残差连接，减少参数量，加速收敛
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    精简版 Dueling DQN:
    - 保留Dueling分支(核心收益)
    - 去掉残差连接(减少参数量/计算量)
    - 减小FC宽度(更快收敛)
    - 保留LayerNorm(训练稳定性)
    
    目标: 在13000轮训练预算内快速收敛
    """
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        
        # CNN: 保持原始padding=2以维持和原代码一致的输出维度
        # 但减少通道数以加速计算
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        max_pool = [nn.MaxPool2d(kernel_size=(2, 2))]
        self.cnn_layer = cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        # 共享特征层: 更紧凑
        self.shared_fc = nn.Sequential(
            nn.Linear(np.prod(state_shape), 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        
        # Dueling分支
        self.action_dim = np.prod(action_shape) if action_shape else 0
        
        # V(s) - 状态价值
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        # A(s,a) - 优势函数
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.action_dim),
        )
        
        self.softmax_layer = nn.Softmax(dim=-1) if softmax else None
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)
        feature_maps = feature_maps.view(feature_maps.shape[0], -1)
        
        concat_feature = torch.cat([feature_vec, feature_maps], dim=1)
        
        shared = self.shared_fc(concat_feature)
        
        # Dueling: Q = V + (A - mean(A))
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        if self.softmax_layer:
            q_values = self.softmax_layer(q_values)
        
        return q_values, state
