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


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
    reward = -0.02

    # Use score increment as event reward to separate chest gain from terminal gain.
    score_delta = score - prev_score
    if score_delta > 0:
        if terminated:
            reward += 20.0 + 0.05 * score_delta
        else:
            reward += 8.0 + 0.08 * score_delta

    # Encourage moving toward the endpoint; punish moving away.
    end_dist, next_end_dist = obs[0], _obs[0]
    if next_end_dist < end_dist:
        reward += 0.25
    elif next_end_dist > end_dist:
        reward -= 0.2

    # Encourage approaching existing treasures (distance 999 means treasure not generated).
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

    # Strong penalty for timeout or abnormal truncation.
    if truncated and not terminated:
        reward -= 15.0

    return float(reward)
