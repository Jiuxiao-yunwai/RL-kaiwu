#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None, done=None)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
    _ = frame_no
    reward = -0.02

    # 分离事件奖励，避免直接使用累计积分造成震荡
    score_delta = score - prev_score
    if score_delta > 0:
        if terminated:
            reward += 20.0 + 0.05 * score_delta
        else:
            reward += 8.0 + 0.08 * score_delta

    # 鼓励靠近终点，惩罚远离终点
    end_dist, next_end_dist = obs[0], _obs[0]
    if next_end_dist < end_dist:
        reward += 0.25
    elif next_end_dist > end_dist:
        reward -= 0.2

    # 鼓励靠近已生成宝箱（999代表该宝箱未生成）
    treasure_dist = np.array(obs[1:11], dtype=np.float32)
    next_treasure_dist = np.array(_obs[1:11], dtype=np.float32)
    valid_mask = np.logical_and(treasure_dist < 999, next_treasure_dist < 999)
    if np.any(valid_mask):
        nearest_prev = float(np.min(treasure_dist[valid_mask]))
        nearest_next = float(np.min(next_treasure_dist[valid_mask]))
        if nearest_next < nearest_prev:
            reward += 0.35
        elif nearest_next > nearest_prev:
            reward -= 0.1

    # 显式处理撞墙/无效移动：动作后观测基本不变且无得分增量，额外惩罚。
    # 在该环境中，这类情况通常由撞到障碍物或边界导致。
    if (not terminated) and (not truncated):
        if score_delta <= 0 and np.allclose(np.asarray(obs), np.asarray(_obs), atol=1e-6):
            reward -= 0.3

    # 超时或异常终止惩罚
    if truncated and not terminated:
        reward -= 15.0

    return float(reward)
