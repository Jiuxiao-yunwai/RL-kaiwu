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
    action_mask=None,
    target_direction=None,
    target_distance=None,
    target_is_end=None,
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


def select_progress_target(treasure_pos_list, end_pos, treasure_collected_count, treasure_count):
    if treasure_count > 0 and treasure_collected_count < treasure_count:
        candidates = [
            (idx, pos)
            for idx, pos in enumerate(treasure_pos_list)
            if is_valid_relative_pos(pos)
        ]
        if candidates:
            target_idx, target_pos = min(candidates, key=lambda item: relative_progress_distance(item[1]))
            return "treasure", target_idx, target_pos

    return "end", -1, end_pos


def progress_delta(prev_rel_pos, curr_rel_pos, scale=3.0):
    prev_dist = relative_progress_distance(prev_rel_pos)
    curr_dist = relative_progress_distance(curr_rel_pos)
    if not np.isfinite(prev_dist) or not np.isfinite(curr_dist):
        return 0
    return float(np.clip((prev_dist - curr_dist) / scale, -1.0, 1.0))


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

    def _get_buff_status(frame_obs):
        for organ in frame_obs.frame_state.organs:
            if organ.sub_type == 2:
                return organ.status
        return 0

    treasure_count = int(_remain_info.get("treasure_count") or 0)
    treasure_collected_count = int(_remain_info.get("treasure_collected_count") or 0)
    prev_treasure_collected_count = int(remain_info.get("treasure_collected_count") or 0)
    treasure_total = max(treasure_count, 1)
    missing_treasure = max(treasure_count - treasure_collected_count, 0)
    missing_ratio = missing_treasure / float(treasure_total)
    is_treasures_remain = missing_treasure > 0

    prev_buff_dist = remain_info.get("buff_pos").grid_distance
    curr_buff_dist = _remain_info.get("buff_pos").grid_distance
    prev_buff_status = _get_buff_status(obs_data)
    curr_buff_status = _get_buff_status(_obs_data)

    target_kind, target_idx, target_pos = select_progress_target(
        _remain_info.get("treasure_pos"),
        _remain_info.get("end_pos"),
        treasure_collected_count,
        treasure_count,
    )
    if target_kind == "treasure" and target_idx >= 0:
        prev_target_pos = remain_info.get("treasure_pos")[target_idx]
    else:
        prev_target_pos = remain_info.get("end_pos")

    target_progress = progress_delta(prev_target_pos, target_pos)

    """
    Reward 1. Reward related to the end point.
    End progress is active only after all treasures are collected.
    奖励1. 与终点相关。只有收齐宝箱后才奖励靠近终点。
    """
    reward_end_dist = target_progress if (not is_treasures_remain and target_kind == "end") else 0

    reward_win = 1 if terminated and missing_treasure == 0 else 0
    reward_early_finish = 1 if terminated and missing_treasure > 0 else 0
    reward_timeout = 1 if truncated else 0
    reward_timeout_missing = missing_ratio if truncated else 0

    """
    Reward 2. Rewards related to the treasure chest.
    Use the same progress target as observation_process.
    奖励2. 与宝箱相关，和 observation_process 使用同一个目标选择逻辑。
    """
    reward_treasure_dist = target_progress if (is_treasures_remain and target_kind == "treasure") else 0
    treasure_delta = max(treasure_collected_count - prev_treasure_collected_count, 0)
    reward_treasure = treasure_delta
    reward_all_treasure = (
        1
        if treasure_count > 0
        and prev_treasure_collected_count < treasure_count
        and treasure_collected_count == treasure_count
        else 0
    )

    """
    Reward 3. Rewards related to the buff.
    奖励3. 与 buff 相关。
    """
    reward_buff_dist = 0
    if prev_buff_dist >= 0 or curr_buff_dist >= 0:
        if prev_buff_dist < 0 and curr_buff_dist >= 0:
            reward_buff_dist = 0.3
        elif prev_buff_dist >= 0 and curr_buff_dist >= 0:
            reward_buff_dist = float(np.clip((prev_buff_dist - curr_buff_dist) / 3.0, -0.5, 1.0))

    reward_buff = 1 if prev_buff_status == 1 and curr_buff_status == 0 else 0

    """
    Reward 4. Rewards related to flicker.
    奖励4. 与闪现相关。
    """
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    is_flicker = move_dist >= 3000
    reward_flicker_bump_penalty = -1 if is_flicker and is_bump else 0
    reward_flicker_normal = 0.1 if is_flicker and not is_bump else 0
    reward_flicker_super = 1 if is_flicker and (target_progress > 0.6 or treasure_delta > 0) else 0
    reward_flicker = reward_flicker_bump_penalty + reward_flicker_normal + reward_flicker_super

    """
    Reward 5. Rewards for quick clearance and anti-stuck behavior.
    奖励5. 快速通关与防卡住。
    """
    reward_step = 1

    reward_perfect_clear = 1 if terminated and treasure_count > 0 and missing_treasure == 0 else 0
    reward_missing_treasure_on_finish = missing_ratio if terminated and missing_treasure > 0 else 0

    memory_map = remain_info.get("memory_map")
    reward_memory = memory_map[len(memory_map) // 2]

    reward_bump = 1 if is_bump else 0

    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = curr_grid_pos
    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.5 * pass_times, -6)

    reward_stall = 1 if move_dist < 150 else 0
    revisit_times = recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 0)
    reward_revisit = min(max(revisit_times - 1, 0), 12)

    reward_weight = {
        "reward_end_dist": 1.0,
        "reward_win": 8.0,
        "reward_early_finish": -10.0,
        "reward_timeout": -5.0,
        "reward_timeout_missing": -7.0,
        "reward_buff_dist": 0.15,
        "reward_buff": 0.6,
        "reward_treasure_dists": 1.0,
        "reward_treasure": 3.0,
        "reward_all_treasure": 8.0,
        "reward_flicker": 0.35,
        "reward_step": -0.003,
        "reward_bump": -1.2,
        "reward_memory": -0.01,
        "reward_exploration": 0.04,
        "reward_perfect_clear": 16.0,
        "reward_missing_treasure_on_finish": -8.0,
        "reward_stall": -0.25,
        "reward_revisit": -0.05,
    }

    reward_terms = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_early_finish * reward_weight["reward_early_finish"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_timeout_missing * reward_weight["reward_timeout_missing"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_all_treasure * reward_weight["reward_all_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_perfect_clear * reward_weight["reward_perfect_clear"],
        reward_missing_treasure_on_finish * reward_weight["reward_missing_treasure_on_finish"],
        reward_stall * reward_weight["reward_stall"],
        reward_revisit * reward_weight["reward_revisit"],
    ]

    return (
        sum(reward_terms),
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
