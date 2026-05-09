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


def is_valid_relative_pos(rel_pos):
    return rel_pos is not None and getattr(rel_pos, "direction", 0) != 0


def relative_progress_distance(rel_pos):
    if not is_valid_relative_pos(rel_pos):
        return float("inf")

    grid_distance = getattr(rel_pos, "grid_distance", -1)
    if grid_distance is not None and grid_distance >= 0:
        return float(grid_distance)

    l2_distance = getattr(rel_pos, "l2_distance", float("inf"))
    try:
        return float(l2_distance)
    except (TypeError, ValueError):
        return float("inf")


def progress_delta(prev_rel_pos, curr_rel_pos, scale=3.0):
    prev_dist = relative_progress_distance(prev_rel_pos)
    curr_dist = relative_progress_distance(curr_rel_pos)
    if not np.isfinite(prev_dist) or not np.isfinite(curr_dist):
        return 0.0
    return float(np.clip((prev_dist - curr_dist) / scale, -1.0, 1.0))


def select_nearest_treasure(treasure_pos_list):
    candidates = [
        (idx, pos)
        for idx, pos in enumerate(treasure_pos_list or [])
        if is_valid_relative_pos(pos)
    ]
    if not candidates:
        return -1, None
    return min(candidates, key=lambda item: relative_progress_distance(item[1]))


def organ_status(frame_obs, sub_type):
    for organ in frame_obs.frame_state.organs:
        if organ.sub_type == sub_type:
            return organ.status
    return 0


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
    # Get the current and previous position coordinates of the agent
    # 获取当前与上一帧智能体的位置坐标
    pos = _obs_data.frame_state.heroes[0].pos
    prev_pos = obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z
    move_dist = ((curr_pos_x - prev_pos_x) ** 2 + (curr_pos_z - prev_pos_z) ** 2) ** 0.5
    curr_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    treasure_count = int(_remain_info.get("treasure_count") or 0)
    treasure_collected_count = int(_remain_info.get("treasure_collected_count") or 0)
    prev_treasure_collected_count = int(remain_info.get("treasure_collected_count") or 0)
    treasure_total = max(treasure_count, 1)
    missing_treasure = max(treasure_count - treasure_collected_count, 0)
    missing_ratio = missing_treasure / float(treasure_total)

    # Are there any remaining treasure chests
    # 是否有剩余宝箱
    is_treasures_remain = missing_treasure > 0

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    end_progress = progress_delta(remain_info.get("end_pos"), _remain_info.get("end_pos"), scale=4.0)
    reward_end_dist = end_progress if not is_treasures_remain else 0.0

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    progress_time = 0.0
    if frame_no is not None:
        progress_time = max(0.0, 1.0 - min(float(frame_no), 2000.0) / 2000.0)
    reward_win = 1.0 if terminated and missing_treasure == 0 else 0.0
    reward_fast_finish = progress_time if reward_win else 0.0
    reward_early_finish = 1.0 + missing_ratio if terminated and missing_treasure > 0 else 0.0
    reward_timeout = 1.0 if truncated else 0.0
    reward_timeout_missing = missing_ratio if truncated else 0.0
    reward_timeout_after_all = 1.0 if truncated and missing_treasure == 0 else 0.0

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0.0
    reward_treasure_visible = 0.0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)
    if is_treasures_remain:
        treasure_idx, treasure_pos = select_nearest_treasure(_remain_info.get("treasure_pos"))
        if treasure_idx >= 0:
            prev_treasure_pos = remain_info.get("treasure_pos")[treasure_idx]
            reward_treasure_dist = progress_delta(prev_treasure_pos, treasure_pos)
            if not is_valid_relative_pos(prev_treasure_pos):
                reward_treasure_visible = 1.0

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    treasure_delta = max(treasure_collected_count - prev_treasure_collected_count, 0)
    reward_treasure = treasure_delta * (1.0 + 0.05 * treasure_collected_count)
    reward_all_treasure = (
        1.0
        if treasure_count > 0
        and prev_treasure_collected_count < treasure_count
        and treasure_collected_count == treasure_count
        else 0.0
    )
    reward_fast_all_treasure = progress_time if reward_all_treasure else 0.0

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = progress_delta(remain_info.get("buff_pos"), _remain_info.get("buff_pos"))

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励
    reward_buff = 0.0
    if organ_status(obs_data, 2) == 1 and organ_status(_obs_data, 2) == 0:
        reward_buff = 1.0

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    is_flicker = move_dist >= 3000
    target_progress = reward_treasure_dist if is_treasures_remain else reward_end_dist
    reward_flicker = 0.0
    if is_flicker and is_bump:
        reward_flicker = -1.0
    elif is_flicker and (target_progress > 0.4 or treasure_delta > 0):
        reward_flicker = 1.0
    elif is_flicker:
        reward_flicker = 0.2

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 8.0 if not is_treasures_remain else 1.0
    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # Give stronger step pressure after collecting all treasure chests.
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # 收集完宝箱后加大步数惩罚，鼓励尽快到达终点

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    memory_map = remain_info.get("memory_map")
    reward_memory = memory_map[len(memory_map) // 2]

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 1.0 if is_bump else 0.0
    reward_stall = 1.0 if move_dist < 150 else 0.0

    # Exploration Reward
    # 探索奖励
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = curr_grid_pos

    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1.0
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.35 * pass_times, -6.0)
    if not is_treasures_remain:
        reward_exploration = min(reward_exploration, 0.0)
    revisit_times = recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 0)
    reward_revisit = min(max(revisit_times - 1, 0), 12)

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    reward_weight = {
        "reward_end_dist": 4.0,
        "reward_win": 36.0,
        "reward_fast_finish": 24.0,
        "reward_early_finish": -20.0,
        "reward_timeout": -10.0,
        "reward_timeout_missing": -14.0,
        "reward_timeout_after_all": -28.0,
        "reward_buff_dist": 0.1,
        "reward_buff": 0.5,
        "reward_treasure_visible": 0.3,
        "reward_treasure_dists": 1.0,
        "reward_treasure": 4.0,
        "reward_all_treasure": 6.0,
        "reward_fast_all_treasure": 2.0,
        "reward_flicker": 0.6,
        "reward_step": -0.006,
        "reward_bump": -1.5,
        "reward_memory": -0.02,
        "reward_exploration": 0.025,
        "reward_stall": -0.6,
        "reward_revisit": -0.08,
    }

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_fast_finish * reward_weight["reward_fast_finish"],
        reward_early_finish * reward_weight["reward_early_finish"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_timeout_missing * reward_weight["reward_timeout_missing"],
        reward_timeout_after_all * reward_weight["reward_timeout_after_all"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_visible * reward_weight["reward_treasure_visible"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_all_treasure * reward_weight["reward_all_treasure"],
        reward_fast_all_treasure * reward_weight["reward_fast_all_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_stall * reward_weight["reward_stall"],
        reward_revisit * reward_weight["reward_revisit"],
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
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
