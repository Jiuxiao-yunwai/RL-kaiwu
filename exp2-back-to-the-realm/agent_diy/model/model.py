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


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
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
        self.cnn_model = nn.Sequential(*(cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool))

        hidden_dim = 256
        self.encoder = nn.Sequential(
            nn.Linear(int(np.prod(state_shape)), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, int(np.prod(action_shape))),
        )

        self.use_softmax = softmax
        self.softmax = nn.Softmax(dim=-1)
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
        x = torch.concat([feature_vec, feature_maps], dim=1)
        x = self.encoder(x)

        value = self.value_head(x)
        advantage = self.adv_head(x)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)

        if self.use_softmax:
            logits = self.softmax(logits)
        return logits, state
