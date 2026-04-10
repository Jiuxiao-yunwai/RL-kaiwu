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

    ACTION_SIZE = 4
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    EPSILON = 0.1
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.9995
    EPISODES = 10000
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 100000
    TARGET_UPDATE_FREQ = 500

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 5

    # Dimension of observation vector from preprocessor
    # 预处理器输出的一维观测向量维度
    OBSERVATION_SHAPE = 250

    # DQN网络输入维度
    STATE_DIM = OBSERVATION_SHAPE
