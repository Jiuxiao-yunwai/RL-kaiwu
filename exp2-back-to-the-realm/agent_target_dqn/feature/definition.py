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
    def _safe_dist(dist):
        return None if dist is None or dist < 0 else float(dist)

    def _delta_reward(prev_dist, curr_dist, scale):
        if prev_dist is None or curr_dist is None:
            return 0.0
        return float(np.clip((prev_dist - curr_dist) / scale, -1.0, 1.0))

    # Get current and previous hero position.
    # 获取当前与上一帧英雄位置。
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Distances and treasure progress info.
    # 距离与宝箱进度信息。
    end_dist = _safe_dist(_remain_info.get("end_pos").l2_distance)
    prev_end_dist = _safe_dist(remain_info.get("end_pos").l2_distance)
    buff_dist = _safe_dist(_remain_info.get("buff_pos").grid_distance)
    prev_buff_dist = _safe_dist(remain_info.get("buff_pos").grid_distance)
    treasure_dists = [p.grid_distance for p in _remain_info.get("treasure_pos")]
    prev_treasure_dists = [p.grid_distance for p in remain_info.get("treasure_pos")]

    treasure_count = max(int(_remain_info.get("treasure_count")), 1)
    treasure_collected_count = int(_remain_info.get("treasure_collected_count"))
    prev_treasure_collected_count = int(remain_info.get("treasure_collected_count"))
    is_treasures_remain = treasure_collected_count < treasure_count

    reward_end_dist = 0.0
    reward_win = 0.0
    reward_treasure_dist = 0.0
    reward_treasure = 0.0
    reward_buff_dist = 0.0
    reward_buff = 0.0
    reward_flicker = 0.0
    reward_step = 1.0
    reward_memory = 0.0
    reward_loop = 0.0
    reward_timeout = 0.0
    reward_stagnation = 0.0

    # End-point progress is emphasized after collecting all treasures.
    # 全收宝箱后，重点鼓励向终点推进。
    if not is_treasures_remain:
        reward_end_dist = _delta_reward(prev_end_dist, end_dist, scale=1200.0)

    if terminated:
        collect_ratio = treasure_collected_count / float(treasure_count)
        reward_win = 3.0 + 4.0 * collect_ratio
    elif truncated:
        missing = max(treasure_count - treasure_collected_count, 0)
        reward_timeout = 1.5 + 0.1 * missing

    # Treasure progress: use currently nearest visible treasure.
    # 宝箱推进：使用当前可见的最近宝箱。
    if is_treasures_remain:
        visible_idx = [idx for idx, d in enumerate(treasure_dists) if d >= 0]
        if visible_idx:
            nearest_idx = min(visible_idx, key=lambda idx: treasure_dists[idx])
            curr_nearest = _safe_dist(treasure_dists[nearest_idx])
            prev_nearest = _safe_dist(prev_treasure_dists[nearest_idx])
            reward_treasure_dist = _delta_reward(prev_nearest, curr_nearest, scale=3.0)

    collected_delta = treasure_collected_count - prev_treasure_collected_count
    if collected_delta > 0:
        reward_treasure = float(collected_delta)

    # Buff rewards.
    # Buff奖励。
    reward_buff_dist = _delta_reward(prev_buff_dist, buff_dist, scale=2.0)
    curr_buff_status = 0
    prev_buff_status = 0
    for organ in _obs_data.frame_state.organs:
        if organ.sub_type == 2:
            curr_buff_status = organ.status
            break
    for organ in obs_data.frame_state.organs:
        if organ.sub_type == 2:
            prev_buff_status = organ.status
            break
    if prev_buff_status == 1 and curr_buff_status == 0:
        reward_buff = 1.0

    # Wall bump penalty.
    # 撞墙惩罚。
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    # Exploration and anti-loop signals.
    # 探索和反绕圈信号。
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)
    recent_position_map = _remain_info.get("recent_position_map")
    pass_times = recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 1)
    if pass_times <= 1:
        reward_exploration = 1.0
    else:
        reward_exploration = -min(0.5 * (pass_times - 1), 8.0)
    if pass_times > 3:
        reward_loop = min(0.25 * (pass_times - 3), 2.0)

    # Penalize staying in frequently visited cells.
    # 惩罚在高频访问区域停留。
    memory_map = _remain_info.get("memory_map")
    reward_memory = float(memory_map[len(memory_map) // 2])

    if is_bump:
        reward_bump = 1.0 + min(0.15 * max(pass_times - 1, 0), 1.0)
    else:
        reward_bump = 0.0

    # Flicker/talent usage shaping.
    # 闪现技能奖励塑形。
    prev_talent_status = obs_data.frame_state.heroes[0].talent.status
    curr_talent_status = _obs_data.frame_state.heroes[0].talent.status
    used_flicker = prev_talent_status == 1 and curr_talent_status == 0
    progress_signal = reward_treasure_dist if is_treasures_remain else reward_end_dist
    if used_flicker:
        if is_bump:
            reward_flicker = -1.0
        elif progress_signal > 0.05:
            reward_flicker = 0.6
        else:
            reward_flicker = -0.2

    # Stagnation penalty: moving without meaningful progress for repeated passes.
    # 停滞惩罚：反复经过且无推进。
    if (not is_bump) and pass_times >= 3 and abs(progress_signal) < 0.03:
        reward_stagnation = min(0.2 * (pass_times - 2), 1.2)

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    reward_weight = {
        "reward_end_dist": 1.2,
        "reward_win": 1.0,
        "reward_buff_dist": 0.15,
        "reward_buff": 0.4,
        "reward_treasure_dists": 1.4,
        "reward_treasure": 2.2,
        "reward_flicker": 0.4,
        "reward_step": -0.0015,
        "reward_bump": -1.2,
        "reward_memory": -0.01,
        "reward_exploration": 0.08,
        "reward_loop": -0.35,
        "reward_timeout": -1.0,
        "reward_stagnation": -0.25,
    }

    reward_terms = [
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
        reward_loop * reward_weight["reward_loop"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_stagnation * reward_weight["reward_stagnation"],
    ]
    total_reward = float(np.clip(sum(reward_terms), -3.0, 3.0))

    return (
        total_reward,
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
