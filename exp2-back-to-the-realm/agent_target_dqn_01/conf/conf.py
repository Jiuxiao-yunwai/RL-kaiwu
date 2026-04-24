#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized: 超参数调优 - 更适合13宝箱全收集任务
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
    DIM_OF_OBSERVATION = 2304 + 404

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

    # ========================================
    # 优化后的超参数
    # ========================================

    # Target网络软更新频率 (每N步做一次Polyak average)
    # 原值500，配合soft update使用
    TARGET_UPDATE_FREQ = 200

    # 初始探索率 - 从高探索开始
    # 原值0.1
    EPSILON = 1.0
    
    # 最小探索率 - 保持少量随机探索
    EPSILON_MIN = 0.02
    
    # 探索率衰减系数 (每次predict调用)
    # 使用指数衰减: epsilon = max(min, epsilon * decay)
    EPSILON_DECAY = 0.9999
    
    # 保留旧参数以兼容
    EPSILON_GREEDY_PROBABILITY = 300000

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    # 原值0.9，对于需要长期规划的任务(1000步)太低
    # 0.99意味着100步后的奖励还能保留约37%的权重
    GAMMA = 0.99

    # Initial learning rate
    # 初始的学习率
    # 原值1e-4，稍微提高以加速初期训练
    START_LR = 3e-4

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
