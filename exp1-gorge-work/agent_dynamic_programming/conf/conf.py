#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration of dimensions
# 关于维度的配置
class Config:

    MAP_SIZE = 64 * 64
    TREASURE_NUM = 10
    TREASURE_MASK_SIZE = 2 ** TREASURE_NUM
    STATE_SIZE = MAP_SIZE * TREASURE_MASK_SIZE
    ACTION_SIZE = 4
    GAMMA = 0.99
    THETA = 1e-3
    EPISODES = 100

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 214

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 250
