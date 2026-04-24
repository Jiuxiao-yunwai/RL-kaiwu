#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

DP算法不需要神经网络模型，此文件保留为空壳以满足框架要求。
"""


import torch
import numpy as np
from torch import nn


class Model(nn.Module):
    """
    Placeholder model for DP algorithm
    DP算法的占位模型 - 不实际使用

    保留此类以满足框架的import要求，
    但DP算法完全不使用神经网络。
    """
    def __init__(self, state_shape=0, action_shape=0, softmax=False):
        super().__init__()
        # 最小化的占位参数，防止框架报错
        self.dummy = nn.Linear(1, 1)

    def forward(self, s, state=None, info=None):
        return torch.zeros(1), state
