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
    move_mask=None,
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


def clipped_progress_reward(prev_dist, curr_dist, clip_value=1.0):
    """
    Convert a distance change into a bounded dense reward.
    将距离变化转换成有界的稠密奖励。
    """
    if prev_dist is None or curr_dist is None:
        return 0.0
    if prev_dist < 0 or curr_dist < 0:
        return 0.0

    return float(np.clip(prev_dist - curr_dist, -clip_value, clip_value))


def _min_positive(values):
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return None
    return min(positive)


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
    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    # Observation total score aligned with final task objective.
    # observation total_score 与最终任务目标保持一致。
    prev_total_score = obs_data.game_info.total_score
    curr_total_score = _obs_data.game_info.total_score
    reward_score = (curr_total_score - prev_total_score) / 100.0

    # Distances in current frame
    # 当前帧距离信息
    end_dist = _remain_info.get("end_pos").l2_distance
    treasure_dists = [item.grid_distance if item.grid_distance > 0 else -1 for item in _remain_info.get("treasure_pos")]
    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count")

    # Previous frame positions / distances
    # 上一帧位置信息与距离信息
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z
    prev_end_dist = remain_info.get("end_pos").l2_distance
    prev_treasure_dists = [
        item.grid_distance if item.grid_distance > 0 else -1 for item in remain_info.get("treasure_pos")
    ]
    prev_treasure_collected_count = remain_info.get("treasure_collected_count")

    is_treasures_remain = treasure_collected_count < treasure_count

    # Dense progress reward towards nearest objective (end or visible treasure).
    # 朝最近目标（终点或可见宝箱）的稠密距离奖励。
    curr_goal_dist = _min_positive([end_dist] + treasure_dists)
    prev_goal_dist = _min_positive([prev_end_dist] + prev_treasure_dists)
    reward_end_dist = clipped_progress_reward(
        prev_goal_dist,
        curr_goal_dist,
        clip_value=Config.PROGRESS_REWARD_CLIP,
    )

    reward_success = 1.0 if terminated else 0.0
    reward_timeout = 1.0 if truncated else 0.0

    reward_treasure_dist = 0.0
    has_visible_treasure = False
    if is_treasures_remain:
        visible_indices = [idx for idx, dist in enumerate(treasure_dists) if dist > 0]
        has_visible_treasure = len(visible_indices) > 0
        visible_dists = [treasure_dists[idx] for idx in visible_indices]
        if visible_dists:
            min_dist = min(visible_dists)
            min_index = treasure_dists.index(min_dist)
            prev_min_dist = prev_treasure_dists[min_index]
            reward_treasure_dist = clipped_progress_reward(prev_min_dist, min_dist)

    reward_treasure = max(0, treasure_collected_count - prev_treasure_collected_count)

    # Reserved reward terms to preserve output interface.
    # 预留奖励项，保持返回接口不变。
    reward_buff_dist = 0
    reward_buff = 0
    reward_flicker = 0
    reward_memory = 0

    reward_step = Config.STEP_PENALTY

    reward_bump = 0.0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    if is_bump:
        reward_bump = 1.0

    # Count-based exploration bonus.
    # 基于访问计数的探索奖励。
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)
    arrival_position_map = _remain_info.get("arrival_position_map")
    recent_position_map = remain_info.get("recent_position_map")

    pass_times = 1
    if arrival_position_map is not None:
        pass_times = max(1, arrival_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 1))
    elif recent_position_map is not None:
        pass_times = max(1, recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 1))

    reward_exploration = Config.EXPLORATION_BONUS_SCALE * (1.0 / np.sqrt(pass_times) - 0.3)
    reward_exploration = max(reward_exploration, Config.EXPLORATION_BONUS_MIN)
    if has_visible_treasure:
        reward_exploration *= 0.25
    elif not is_treasures_remain:
        reward_exploration *= 0.1

    end_progress_weight = 0.8 if not is_treasures_remain else 0.55
    treasure_progress_weight = 0.3 if has_visible_treasure else 0.1
    exploration_weight = 1.0

    reward_weight = {
        "reward_score": 1.0,
        "reward_end_dist": end_progress_weight,
        "reward_success": 2.0,
        "reward_timeout": -6.0,
        "reward_buff_dist": 0,
        "reward_buff": 0,
        "reward_treasure_dists": treasure_progress_weight,
        "reward_treasure": 0.4,
        "reward_flicker": 0,
        "reward_step": 1.0,
        "reward_bump": -1.0,
        "reward_memory": 0,
        "reward_exploration": exploration_weight,
    }

    reward = [
        reward_score * reward_weight["reward_score"],
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_success * reward_weight["reward_success"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
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
