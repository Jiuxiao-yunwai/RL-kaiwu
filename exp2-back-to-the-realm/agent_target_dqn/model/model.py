#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False, use_dueling=True):
        super().__init__()
        self.use_dueling = use_dueling
        self.action_dim = int(np.prod(action_shape)) if action_shape else 0

        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ]
        max_pool = [nn.MaxPool2d(kernel_size=(2, 2))]
        self.cnn_layer = cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        self.backbone = nn.Sequential(
            nn.Linear(np.prod(state_shape), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        if self.use_dueling and self.action_dim:
            # Dueling head: Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
            # Dueling头: Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
            self.value_head = nn.Linear(128, 1)
            self.adv_head = nn.Linear(128, self.action_dim)
            self.softmax = nn.Softmax(dim=-1) if softmax else None
        else:
            layers = []
            if self.action_dim:
                layers.append(nn.Linear(128, self.action_dim))
            if softmax:
                layers.append(nn.Softmax(dim=-1))
            self.model = nn.Sequential(*layers)

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
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        hidden = self.backbone(concat_feature)
        if self.use_dueling and self.action_dim:
            value = self.value_head(hidden)
            advantage = self.adv_head(hidden)
            logits = value + advantage - advantage.mean(dim=1, keepdim=True)
            if self.softmax is not None:
                logits = self.softmax(logits)
        else:
            logits = self.model(hidden)
        return logits, state
