#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized: Dueling DQN 网络架构，增强特征提取能力
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Dueling DQN 架构:
    - 共享CNN层提取地图特征
    - 共享FC层融合向量+地图特征
    - 分支为 V(s) 状态价值流 和 A(s,a) 优势流
    - Q(s,a) = V(s) + A(s,a) - mean(A)
    
    相比普通DQN，Dueling能更好地评估状态价值，
    加速有效动作的学习，减少对稀疏奖励环境的样本需求。
    """
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        
        # ========================================
        # CNN特征提取器 - 增加通道数和残差连接
        # ========================================
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 残差连接的降采样适配层
        self.res_adapt = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )
        
        # ========================================
        # 共享特征层
        # ========================================
        # CNN输出维度计算: 51x51 -> pool -> 25x25 -> pool -> 12x12 -> pool -> 6x6
        # 64 * 6 * 6 = 2304 (这里需要根据实际计算)
        self.shared_fc = nn.Sequential(
            nn.Linear(np.prod(state_shape), 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        
        # ========================================
        # Dueling架构: 价值流 + 优势流
        # ========================================
        self.action_dim = np.prod(action_shape) if action_shape else 0
        
        # 状态价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        
        # 优势流 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.action_dim),
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

    # Forward inference
    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        
        # CNN前向 with 残差连接
        x1 = self.cnn_layer1(feature_maps)
        x1_pooled = self.pool(x1)
        
        x2 = self.cnn_layer2(x1_pooled)
        x2_pooled = self.pool(x2)
        
        # 残差连接: 将layer1的输出经过适配后和layer3输入相加
        x2_res = self.res_adapt(x1_pooled)
        # 需要对x2_res进行池化以匹配x2_pooled的尺寸
        x2_res_pooled = self.pool(x2_res)
        
        x3 = self.cnn_layer3(x2_pooled + x2_res_pooled)
        x3_pooled = self.pool(x3)
        
        feature_maps_flat = x3_pooled.view(x3_pooled.shape[0], -1)
        
        # 拼接向量特征和CNN特征
        concat_feature = torch.cat([feature_vec, feature_maps_flat], dim=1)
        
        # 共享层
        shared = self.shared_fc(concat_feature)
        
        # Dueling: Q = V + (A - mean(A))
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        if self.softmax_layer:
            q_values = self.softmax_layer(q_values)
        
        return q_values, state
