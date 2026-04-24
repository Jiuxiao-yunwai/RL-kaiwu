#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.back_to_the_realm.target_dqn.feature_process import bump
from agent_diy.conf.conf import Config

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
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


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, state_env_info, _state_env_info):
    est_step = max(float(frame_no) / 3.0, 0.0) if frame_no is not None else 0.0
    max_step_cfg = float(getattr(_state_env_info.game_info, "max_step", 2000))
    time_pressure = min(est_step / max(max_step_cfg, 1.0), 1.0)

    pos = _state_env_info.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    def _delta_distance(prev_dist, curr_dist):
        if prev_dist < 0 or curr_dist < 0:
            return None
        delta = prev_dist - curr_dist
        if max(prev_dist, curr_dist) <= 2.0:
            return delta * 256.0
        return delta

    end_dist = _remain_info.get("end_pos").grid_distance
    prev_end_dist = remain_info.get("end_pos").grid_distance
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    prev_treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")]
    treasure_count = int(_remain_info.get("treasure_count"))
    treasure_collected_count = int(_remain_info.get("treasure_collected_count"))
    prev_treasure_collected_count = int(remain_info.get("treasure_collected_count"))
    is_treasures_remain = treasure_collected_count < treasure_count

    reward_end_dist = 0
    delta_end_grid = _delta_distance(prev_end_dist, end_dist)
    if delta_end_grid is not None:
        if abs(delta_end_grid) <= 0.1:
            reward_end_dist = 0
        else:
            reward_end_dist = max(min(delta_end_grid / 2.0, 1.0), -1.0)

    reward_treasure_dist = 0
    visible_dists = [d for d in treasure_dists if d > 0]
    prev_visible_dists = [d for d in prev_treasure_dists if d > 0]
    if is_treasures_remain and visible_dists and prev_visible_dists:
        curr_min_dist = min(visible_dists)
        prev_min_dist = min(prev_visible_dists)
        delta_treasure = _delta_distance(prev_min_dist, curr_min_dist)
        if delta_treasure is not None:
            if abs(delta_treasure) <= 0.1:
                reward_treasure_dist = 0
            else:
                reward_treasure_dist = max(min(delta_treasure / 2.0, 1.0), -1.0)

    reward_treasure = 1 if treasure_collected_count > prev_treasure_collected_count else 0
    reward_all_treasure = 1 if treasure_count > 0 and treasure_collected_count == treasure_count and prev_treasure_collected_count < treasure_count else 0
    reward_missing_treasure_on_finish = 0
    if terminated and treasure_count > 0 and treasure_collected_count < treasure_count:
        reward_missing_treasure_on_finish = (treasure_count - treasure_collected_count) / float(treasure_count)

    reward_win = 0
    if terminated:
        if treasure_count > 0 and treasure_collected_count == treasure_count:
            reward_win = 3.0
        else:
            reward_win = -1.0

    reward_timeout = 1 if truncated else 0

    prev_pos = state_env_info.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    reward_bump = 1 if is_bump else 0

    reward_step = 1.0

    reward_exploration = 0
    recent_position_map = remain_info.get("recent_position_map")
    grid_x = (curr_pos_x + 2250) // 500
    grid_z = (curr_pos_z + 5250) // 500
    hero_grid_pos = (grid_z, grid_x)
    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.5 * pass_times, -10)

    reward_weight = {
        "reward_end_dist": 1.0,
        "reward_win": 5.0,
        "reward_timeout": -6.0,
        "reward_treasure_dist": 0.9,
        "reward_treasure": 1.8,
        "reward_all_treasure": 3.2,
        "reward_missing_treasure_on_finish": -5.0,
        "reward_step": -0.001,
        "reward_bump": -1.2,
        "reward_exploration": 0.03,
    }

    if is_treasures_remain:
        phase_fade = min(max((time_pressure - 0.6) / 0.3, 0.0), 1.0)
        reward_weight["reward_end_dist"] = 0.1 + 0.6 * phase_fade
        reward_weight["reward_treasure_dist"] = 1.6 - 1.2 * phase_fade
        reward_weight["reward_treasure"] = 2.8 - 2.0 * phase_fade
    else:
        reward_weight["reward_end_dist"] = 1.8
        reward_weight["reward_treasure_dist"] = 0.1
        reward_weight["reward_treasure"] = 0.2

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_treasure_dist * reward_weight["reward_treasure_dist"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_all_treasure * reward_weight["reward_all_treasure"],
        reward_missing_treasure_on_finish * reward_weight["reward_missing_treasure_on_finish"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_exploration * reward_weight["reward_exploration"],
    ]

    return (
        sum(reward),
        is_bump,
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_treasure_dist * reward_weight["reward_treasure_dist"],
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
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
