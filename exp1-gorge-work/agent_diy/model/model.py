#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from torch import nn


class Model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        state_dim = int(state_shape)
        action_dim = int(action_shape)

        # 用户自定义网络（MLP）
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)
