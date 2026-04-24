#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # For example, the dimension of dqn in the sample code is 21624, and the dimension of target_dqn is 21624
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中dqn的维度是21624, target_dqn的维度是21624
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 21624

    # Size of observation. After users design their own features, they should set the correct dimensions
    # observation的维度，用户设计了自己的特征之后应该设置正确的维度
    VIEW_SIZE = 25
    DIM_OF_OBSERVATION = 4096 + 404

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # Feature split:
    # vector part: 404, map part: 4 * 51 * 51
    DESC_OBS_SPLIT = [404, (4, VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1)]

    # Dueling Double DQN hyper-parameters
    TARGET_UPDATE_FREQ = 800
    TARGET_SOFT_TAU = 0.003
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY_STEPS = 180000
    EPSILON_WARMUP_STEPS = 15000
    EPSILON = EPSILON_START
    GAMMA = 0.995
    N_STEP = 3
    GRAD_CLIP_NORM = 1.0
    START_LR = 6e-5

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
