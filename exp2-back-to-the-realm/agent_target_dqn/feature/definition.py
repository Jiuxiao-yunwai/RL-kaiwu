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
    """
    x = (x + 2250) // 500
    z = (z + 5250) // 500
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
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    max_step = 2000.0
    step_ratio = min(max(float(frame_no or 0) / 3.0 / max_step, 0.0), 1.0)

    end_dist = _remain_info.get("end_pos").grid_distance
    prev_end_dist = remain_info.get("end_pos").grid_distance
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    prev_treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")]
    treasure_count = int(_remain_info.get("treasure_count"))
    treasure_collected_count = int(_remain_info.get("treasure_collected_count"))
    prev_treasure_collected_count = int(remain_info.get("treasure_collected_count"))
    current_grid = _remain_info.get("grid_pos")
    prev_grid = remain_info.get("grid_pos")
    prev_prev_grid = remain_info.get("prev_grid_pos")
    current_visit_count = _remain_info.get("recent_position_map", {}).get(current_grid, 0) if current_grid else 0
    is_treasures_remain = treasure_collected_count < treasure_count

    def _clip_delta(prev_dist, curr_dist, scale):
        if prev_dist is None or curr_dist is None or prev_dist < 0 or curr_dist < 0:
            return 0.0
        delta = (prev_dist - curr_dist) / float(scale)
        return float(np.clip(delta, -1.0, 1.0))

    reward_end_dist = 0.0
    if not is_treasures_remain:
        reward_end_dist = _clip_delta(prev_end_dist, end_dist, 3.0)

    reward_treasure_dist = 0.0
    if is_treasures_remain:
        visible_dists = [d for d in treasure_dists if d > 0]
        prev_visible_dists = [d for d in prev_treasure_dists if d > 0]
        if visible_dists and prev_visible_dists:
            reward_treasure_dist = _clip_delta(min(prev_visible_dists), min(visible_dists), 2.0)

    reward_treasure = 0.0
    if treasure_collected_count > prev_treasure_collected_count:
        progress = treasure_collected_count / float(max(treasure_count, 1))
        reward_treasure = 2.5 + 3.0 * progress

    reward_all_treasure = 0.0
    if treasure_count > 0 and treasure_collected_count == treasure_count and prev_treasure_collected_count < treasure_count:
        reward_all_treasure = 4.0

    reward_finish = 0.0
    if terminated:
        if treasure_count > 0 and treasure_collected_count == treasure_count:
            reward_finish = 12.0
        else:
            reward_finish = -8.0

    reward_timeout = 0.0
    if truncated:
        missing_ratio = 0.0 if treasure_count <= 0 else (treasure_count - treasure_collected_count) / float(treasure_count)
        reward_timeout = -(4.0 + 8.0 * missing_ratio)

    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    reward_bump = -1.8 if is_bump else 0.0

    reward_step = -0.001 - 0.002 * step_ratio
    if not is_treasures_remain:
        reward_step -= 0.004

    reward_memory = 0.0
    memory_map = remain_info.get("memory_map")
    if memory_map:
        reward_memory = -0.03 * float(memory_map[len(memory_map) // 2])

    reward_exploration = 0.0
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 0.28
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = -min(0.09 * pass_times, 0.7)

    reward_backtrack = 0.0
    if current_grid and prev_prev_grid and current_grid == prev_prev_grid and current_grid != prev_grid:
        reward_backtrack = -0.9
    if current_visit_count >= 3:
        reward_backtrack -= min(0.18 * (current_visit_count - 2), 0.9)

    reward = [
        reward_end_dist * 2.4,
        reward_treasure_dist * 2.2,
        reward_treasure,
        reward_all_treasure,
        reward_finish,
        reward_timeout,
        reward_step,
        reward_bump,
        reward_memory,
        reward_backtrack,
        reward_exploration,
    ]

    return (
        sum(reward),
        is_bump,
        reward_end_dist * 2.4,
        reward_exploration,
        reward_treasure_dist * 2.2,
        reward_treasure,
    )


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


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
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
