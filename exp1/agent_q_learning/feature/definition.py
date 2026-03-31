#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached


# Old code (kept for comparison):
# SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None)
# New code: add done field so algorithm.learn can stop bootstrapping on terminal transitions.
SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None, done=None)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs):
    reward = 0

    # The reward for winning
    # 奖励1. 获胜的奖励
    if terminated:
        reward += score

    # The reward for obtaining a treasure chest
    # 奖励3. 获得宝箱的奖励
    if score > 0 and not terminated:
        reward += score

    # Small step penalty to encourage shorter paths
    # 步骤惩罚：鼓励智能体以更少的步数到达终点
    reward -= 0.01

    return reward
