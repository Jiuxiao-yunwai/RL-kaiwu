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
    # For example, the dimension of this target_dqn is 21624 using the environment's 2-dimension legal action mask
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 当前target_dqn使用环境2维动作掩码后维度是21624
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

    # norm_pos + one_hot_pos + end_pos_features + treasure_poss_features + [buff_availability, talent_availability]
    #  + obstacle_map + treasure_map + end_map + location_memory
    # 2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808
    # # Describe how to perform feature segmentation. The features in the sample code are processed into vectors + feature maps.
    # The following configuration describes the dimensions of the two.
    # 描述如何进行特征分割，示例代码中的特征处理成向量+特征图，以下配置描述了两者的维度
    DESC_OBS_SPLIT = [404, (4, VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1)]  # sum = 10808

    # Update frequency of target network
    # target网络的更新频率
    TARGET_UPDATE_FREQ = 200

    # Exploration factor, see the calculation of epsilon in the function in the above comment
    # 探索因子, epsilon的计算见上面注释中的函数
    EPSILON_GREEDY_PROBABILITY = 60000

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.99

    # epsilon
    EPSILON = 0.25

    # Minimum epsilon after linear decay
    # 线性衰减后的最小探索率
    MIN_EPSILON = 0.01

    # Initial learning rate
    # 初始的学习率
    START_LR = 3e-4

    # Optimizer and target-guided exploration settings
    # 优化器与目标引导探索配置
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 10.0
    TRAIN_TARGET_BIAS = 0.75
    TRAIN_TALENT_BIAS = 0.35
    EXPLOIT_TARGET_BIAS = 1.40
    EXPLOIT_TALENT_BIAS = 0.85
    TALENT_MIN_TARGET_DISTANCE = 12.0
    NEAR_TARGET_TALENT_PENALTY = 0.25
    GUIDED_EXPLORATION_PROBABILITY = 0.97
    GUIDED_EXPLORATION_NOISE = 0.03

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
