#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_target_dqn.conf.conf import Config


def bump(a1, b1, a2, b2):
    """
    This function is used to determine whether the game hits a wall.
        - There will be no bump in the first frame.
        - Starting from the second frame, if the moving distance is less than 500, it will be considered as hitting a wall.

    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False

    dist = ((a1 - a2) ** 2 + (b1 - b2) ** 2) ** (0.5)

    return dist <= 500


# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def convert_pos_to_grid_pos(x, z):
    """
    Convert the position 'pos' into grid-based coordinates
    将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: tuple
    """
    x = (x + 2250) // 500
    z = (z + 5250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是为了与json文件的顺序保持一致
    x, z = z, x

    return x, z


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    remain_info,
    _remain_info,
    obs_data,
    _obs_data,
):
    def rel_active(rel_pos):
        return rel_pos is not None and getattr(rel_pos, "direction", 0) != 0

    def rel_distance(rel_pos):
        if not rel_active(rel_pos):
            return None
        if rel_pos.grid_distance >= 0:
            return rel_pos.grid_distance
        return rel_pos.l2_distance + 128

    def nearest_active_index(rel_pos_list):
        candidates = []
        for idx, rel_pos in enumerate(rel_pos_list):
            dist = rel_distance(rel_pos)
            if dist is not None:
                candidates.append((dist, idx))
        if not candidates:
            return None
        return min(candidates)[1]

    def progress_reward(prev_dist, curr_dist):
        if prev_dist is None or curr_dist is None:
            return 0
        delta = prev_dist - curr_dist
        if delta > 0:
            return min(1.0, 0.2 + delta / 8.0)
        if delta == 0:
            return -0.1
        return max(-1.0, delta / 8.0)

    curr_pos = _obs_data.frame_state.heroes[0].pos
    prev_pos = obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = curr_pos.x, curr_pos.z
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z
    move_dist = ((curr_pos_x - prev_pos_x) ** 2 + (curr_pos_z - prev_pos_z) ** 2) ** 0.5

    treasure_count = max(1, _remain_info.get("treasure_count") or remain_info.get("treasure_count") or 13)
    treasure_collected_count = _remain_info.get("treasure_collected_count", 0)
    prev_treasure_collected_count = remain_info.get("treasure_collected_count", 0)
    missing_treasures = max(0, treasure_count - treasure_collected_count)
    is_treasures_remain = missing_treasures > 0

    curr_treasure_pos = _remain_info.get("treasure_pos")
    prev_treasure_pos = remain_info.get("treasure_pos")

    reward_end_dist = 0
    reward_treasure_dist = 0
    if is_treasures_remain:
        target_idx = nearest_active_index(curr_treasure_pos)
        if target_idx is not None:
            reward_treasure_dist = progress_reward(
                rel_distance(prev_treasure_pos[target_idx]),
                rel_distance(curr_treasure_pos[target_idx]),
            )
    else:
        reward_end_dist = progress_reward(
            rel_distance(remain_info.get("end_pos")),
            rel_distance(_remain_info.get("end_pos")),
        )

    collected_delta = max(0, treasure_collected_count - prev_treasure_collected_count)
    reward_treasure = 6.0 * collected_delta

    prev_buff_count = getattr(obs_data.game_info, "buff_count", 0)
    curr_buff_count = getattr(_obs_data.game_info, "buff_count", 0)
    reward_buff = 0.8 if curr_buff_count > prev_buff_count else 0
    reward_buff_dist = 0

    prev_talent_count = getattr(obs_data.game_info, "talent_count", 0)
    curr_talent_count = getattr(_obs_data.game_info, "talent_count", 0)
    reward_flicker = 0
    if curr_talent_count > prev_talent_count:
        reward_flicker = 1.0 if move_dist >= 1500 else -1.0

    reward_win = 0
    if terminated:
        completion_ratio = treasure_collected_count / treasure_count
        reward_win = 10.0 * completion_ratio
        if missing_treasures == 0:
            reward_win += 30.0
        else:
            reward_win -= 3.0 * missing_treasures
    elif truncated:
        reward_win = -15.0 - missing_treasures

    reward_step = 1
    reward_bump = 0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    if is_bump:
        reward_bump = 1

    reward_stay = 1 if move_dist <= 120 else 0

    memory_map = _remain_info.get("memory_map")
    reward_memory = memory_map[len(memory_map) // 2] if memory_map is not None and len(memory_map) > 0 else 0

    reward_exploration = 0
    recent_position_map = remain_info.get("recent_position_map") or {}
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)
    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.25 * pass_times, -2)

    reward_weight = {
        "reward_end_dist": 0.8,
        "reward_win": 1.0,
        "reward_buff_dist": 0,
        "reward_buff": 0.5,
        "reward_treasure_dists": 0.7,
        "reward_treasure": 1.0,
        "reward_flicker": 0.4,
        "reward_step": -0.01,
        "reward_bump": -1.5,
        "reward_memory": -0.03,
        "reward_exploration": 0.08,
        "reward_stay": -1.0,
    }

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_stay * reward_weight["reward_stay"],
    ]

    return (
        sum(reward),
        is_bump,
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
    )


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = (2 * (Config.VIEW_SIZE) + 1) ** 2 * 4 + 404
    legal_action_size = Config.LEGAL_ACTION_SHAPE
    obs_legal_begin = 2 * obs_data_size
    obs_legal_end = obs_legal_begin + legal_action_size
    next_obs_legal_end = obs_legal_end + legal_action_size
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[obs_legal_begin:obs_legal_end],
        _obs_legal=s_data[obs_legal_end:next_obs_legal_end],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
