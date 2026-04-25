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

    # field of view radius
    # 视野域半径
    VIEW_SIZE = 25

    # Size of observation. Note that in our sample code the original feature dimension is 10808.
    # Here is the dimension after CNN processing and the original vector feature concatenation.
    # observation的维度，注意在我们的示例代码中原特征维度是10808，这里是经过CNN处理之后的维度与原始向量特征拼接后的维度
    DIM_OF_OBSERVATION = 4096 + 404

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # norm_pos + one_hot_pos + target_pos_features + treasure_pos_features + [buff_availability, talent_availability]
    #  + obstacle_map + treasure_map + end_map + location_memory
    # 2 + 128*2 + 9 + 9*15 + 2 + 4*51*51 = 10808
    DESC_OBS_SPLIT = [404, (4, VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1)]

    # Update frequency of hard target sync
    # target网络硬同步频率
    TARGET_UPDATE_FREQ = 160

    # Soft update ratio for target network
    # target网络软更新系数
    TARGET_SOFT_TAU = 0.01

    # Exploration schedule
    # 探索率调度
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 1600000
    EPSILON_WARMUP_STEPS = 20000

    # Keep compatibility with the original code path
    # 兼容旧代码路径
    EPSILON = EPSILON_START
    EPSILON_GREEDY_PROBABILITY = EPSILON_DECAY_STEPS

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.99

    # N-step target horizon
    # n-step目标回报长度
    N_STEP = 6

    # Gradient clipping norm
    # 梯度裁剪阈值
    GRAD_CLIP_NORM = 5.0

    # Initial learning rate
    # 初始的学习率
    START_LR = 2e-4

    # Reward clipping range
    # 奖励裁剪范围
    REWARD_CLIP = 18.0

    # Anti-stuck action bias
    # 防卡死动作偏置参数
    GUIDE_ACTION_BONUS = 0.4
    BACKTRACK_ACTION_PENALTY = 1.4
    NO_PROGRESS_ACTION_PENALTY = 0.9
    STUCK_VISIT_COUNT = 2

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
