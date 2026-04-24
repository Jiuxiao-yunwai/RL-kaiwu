#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized: 13000轮极限效率超参数
"""


class Config:

    # SAMPLE_DIM 必须与 NumpyData2SampleData 对齐
    SAMPLE_DIM = 21624

    # 视野域半径
    VIEW_SIZE = 25

    # CNN输出4096(padding=2) + 向量特征404 = 4500
    DIM_OF_OBSERVATION = 4096 + 404

    # 动作维度
    DIM_OF_ACTION_DIRECTION = 8
    DIM_OF_TALENT = 8

    # 特征分割: 向量404 + 地图4*51*51
    DESC_OBS_SPLIT = [404, (4, VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1)]

    # ========================================
    # 针对13000轮预算优化的超参数
    # ========================================

    # Target网络每步都做软更新(Polyak)，不需要频率了
    # 但为了效率，每100步做一次
    TARGET_UPDATE_FREQ = 100

    # Epsilon: 从0.5开始(不需要从1.0全随机，浪费前期轮次)
    # 衰减到0.03，保持少量探索
    EPSILON = 0.5
    EPSILON_MIN = 0.03
    # 每次predict衰减: 0.5 * 0.9998^N
    # ~3500步到0.25, ~8000步到0.1, ~17000步到0.03
    # 在13000轮(每轮~600-1000步)下大约第2-3轮就开始利用
    EPSILON_DECAY = 0.9998

    # 保留旧参数以兼容
    EPSILON_GREEDY_PROBABILITY = 300000

    # GAMMA: 0.97比0.99收敛更快
    # 0.97: 30步后的奖励权重~40%, 100步~5%
    # 0.99: 100步后的奖励权重~37%
    # 在有密集引导奖励的情况下，0.97足够看到关键奖励
    GAMMA = 0.97

    # 学习率: 稍高加速初期
    START_LR = 5e-4

    # 平台配置 (不改动)
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
